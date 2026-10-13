"""Non-circular source shifts and a carry-aware last-stride construction."""
from contextlib import contextmanager
import copy
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from . import design as d
from experiments.cvs_source_pseudotarget_health.frozen import preserve_rng, snapshot, assert_unchanged

DELTAS = (-2, -1, 1, 2)
# Stride, input-coordinate center offset, convolution/pooling receptive width.
# GroupNorm uses all time positions: excluding convolution boundaries does not
# remove its global statistic dependence, so native errors remain observational.
GEOMETRY = {'time_fuse': (1, 0., 79), 'time_down': (2, .5, 80),
            't1.act': (2, .5, 88), 't1.pool': (4, 1.5, 90),
            't2.act': (4, 1.5, 106), 't2.pool': (8, 3.5, 110),
            't3.act': (8, 3.5, 126)}


def shift(x, delta):
    """Right positive integer translation, with zeros and no cyclic wrap."""
    if not torch.is_tensor(x) or x.ndim < 1 or not isinstance(delta, int) or isinstance(delta, bool):
        raise ValueError('Tensor and integer non-circular shift required')
    if abs(delta) >= x.shape[-1]:
        raise ValueError('Shift must retain a nonempty common input interval')
    y = torch.zeros_like(x)
    if delta > 0:
        y[..., delta:] = x[..., :-delta]
    elif delta < 0:
        y[..., :delta] = x[..., -delta:]
    else:
        y = x.clone()
    return y


def s2(x):
    if x.ndim != 3 or x.shape[-1] % 2:
        raise ValueError('S2 requires even [B,C,T] sequence')
    return torch.stack((x[..., ::2], x[..., 1::2]), 2)


def inverse_s2(z):
    if z.ndim != 4 or z.shape[2] != 2:
        raise ValueError('Two ordered polyphases required')
    return z.permute(0, 1, 3, 2).reshape(z.shape[0], z.shape[1], 2 * z.shape[-1])


def carry_shift(z, delta):
    """Exact phase exchange and coarse-index carry for a fine-grid shift."""
    if z.ndim != 4 or z.shape[2] != 2 or not isinstance(delta, int) or isinstance(delta, bool):
        raise ValueError('Two polyphases and integer shift required')
    if abs(delta) >= 2 * z.shape[-1]:
        raise ValueError('Carry shift must retain common input')
    k, r = divmod(delta, 2); n = torch.arange(z.shape[-1], device=z.device)
    outputs = []
    for phase in range(2):
        raw_phase = phase - r
        source_phase = raw_phase % 2
        source_index = n - k + raw_phase // 2
        valid = (source_index >= 0) & (source_index < len(n))
        output = z[:, :, source_phase, source_index.clamp(0, len(n) - 1)]
        outputs.append(output * valid.to(z.dtype))
    return torch.stack(outputs, 2)


class S2Downsample(nn.Module):
    """Reindex rather than discard either last-stride phase."""
    def forward(self, x):
        return s2(x)


class PointwiseChannelGroupNorm(nn.Module):
    """Group normalization across channels at each individual time coordinate."""
    def __init__(self, channels=96, groups=16, eps=1e-5):
        super().__init__()
        if channels % groups:
            raise ValueError('Channel groups must divide channels')
        self.channels = channels; self.groups = groups; self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels)); self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError('Pointwise channel GN expects [B,C,T]')
        v = x.transpose(1, 2).reshape(len(x), x.shape[-1], self.groups, self.channels // self.groups)
        mean = v.mean(-1, keepdim=True); var = v.var(-1, unbiased=False, keepdim=True)
        v = ((v - mean) / (var + self.eps).sqrt()).reshape(len(x), x.shape[-1], self.channels).transpose(1, 2)
        return v * self.weight[None, :, None] + self.bias[None, :, None]


class CarryAwareT3(nn.Module):
    """Shared fine-grid DW/PW, pointwise GN, and interleaved global-pool input.

    Future scratch use replaces t2.pool with S2Downsample and t3 with this class.
    Both phases survive: the output length is twice that of the old t3. No claim
    is made that this preserves the former activation or convolution cost.
    """
    def __init__(self, channels=96, groups=16, dropout=0.):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm = PointwiseChannelGroupNorm(channels, groups)
        self.act = nn.ReLU()
        self.drop = nn.Dropout1d(dropout) if dropout else nn.Identity()

    def forward(self, z):
        if z.ndim != 4 or z.shape[2] != 2 or z.shape[1] != self.dw.in_channels:
            raise ValueError('Carry-aware t3 expects the two preserved input phases')
        terms = [carry_shift(z, 1 - k) * self.dw.weight[:, 0, k][None, :, None, None] for k in range(3)]
        v = inverse_s2(sum(terms))
        return self.drop(self.act(self.norm(self.pw(v))))


class PointT3(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.dw = copy.deepcopy(native.dw); self.pw = copy.deepcopy(native.pw)
        self.norm = PointwiseChannelGroupNorm(native.pw.out_channels, native.norm.num_groups, native.norm.eps)
        with torch.no_grad():
            self.norm.weight.copy_(native.norm.weight); self.norm.bias.copy_(native.norm.bias)
        self.act = nn.ReLU()
        self.drop = copy.deepcopy(getattr(native, 'drop', nn.Identity()))

    def forward(self, x):
        return self.drop(self.act(self.norm(self.pw(self.dw(x)))))


class LowpassDownsample(nn.Module):
    def forward(self, x):
        weight = x.new_tensor([.25, .5, .25]).reshape(1, 1, 3).expand(x.shape[1], 1, 3)
        return F.conv1d(x, weight, padding=1, groups=x.shape[1])[..., ::2]


def replacements(native, device):
    dropout = getattr(getattr(native, 'drop', None), 'p', 0.)
    carry = CarryAwareT3(native.pw.out_channels, native.norm.num_groups, dropout).to(device)
    with torch.no_grad():
        carry.dw.weight.copy_(native.dw.weight); carry.pw.weight.copy_(native.pw.weight)
        carry.norm.weight.copy_(native.norm.weight); carry.norm.bias.copy_(native.norm.bias)
    return {'ordinary_global_GN': nn.Sequential(copy.deepcopy(nn.MaxPool1d(2)), copy.deepcopy(native)),
            'ordinary_point_GN': nn.Sequential(nn.MaxPool1d(2), PointT3(native)),
            'lowpass_point_GN': nn.Sequential(LowpassDownsample(), PointT3(native)),
            'carry_point_GN': nn.Sequential(S2Downsample(), carry)}


@contextmanager
def capture_levels(model):
    from experiments.cvs_e20_cross_style.style import _paths
    modules, paths = _paths(model.base); prefix = paths['time_down'].rsplit('.', 1)[0]
    native = modules[prefix]
    if native.sinc.kernel_size != 79:
        raise ValueError('Registered input-coordinate geometry requires actual sinc79')
    maps = {}; handles = []
    try:
        for level in GEOMETRY:
            path = prefix + '.' + level
            if path not in modules:
                raise ValueError('Actual sampling layer missing')
            def hook(module, inputs, output, level=level):
                if level in maps or output.ndim != 3:
                    raise ValueError('One actual native sequence per sampling layer required')
                maps[level] = output.detach().clone()
            handles.append(modules[path].register_forward_hook(hook))
        yield maps, native
        if tuple(maps) != tuple(GEOMETRY):
            raise ValueError('Actual sampling layer order differs')
    finally:
        for handle in handles:
            handle.remove()


def paired_sums(reference, moved, delta, stride, center=0., receptive_width=1, input_length=None):
    if reference.shape != moved.shape or reference.ndim != 3:
        raise ValueError('Matched native [B,C,T] maps required')
    input_length = reference.shape[-1] * stride if input_length is None else input_length
    exact = delta % stride == 0
    shifts = [delta // stride] if exact else sorted({math.floor(delta / stride), math.ceil(delta / stride)})
    result = dict(cumulative_stride=stride, input_delta=delta, exact_alignment=exact,
                  alignment='integer input-coordinate alignment' if exact else 'phase bracket only; no rounded equivariance claim',
                  receptive_width=receptive_width, center=center, alternatives=[])
    for coarse_delta in shifts:
        j = torch.arange(reference.shape[-1], device=reference.device)
        source = j - coarse_delta
        radius = (receptive_width - 1) / 2
        moved_coordinate = j * stride + center
        reference_coordinate = source * stride + center
        lower = max(0, delta); upper = min(input_length - 1, input_length - 1 + delta)
        valid = ((source >= 0) & (source < len(j)) & (moved_coordinate - radius >= lower)
                 & (moved_coordinate + radius <= upper) & (reference_coordinate - radius >= 0)
                 & (reference_coordinate + radius < input_length))
        if not valid.any():
            raise ValueError('No common convolution-valid coordinate interval')
        a = reference[..., source[valid]]; b = moved[..., j[valid]]
        squared_reference = a.square().sum((1, 2)); squared_moved = b.square().sum((1, 2))
        squared_difference = (a - b).square().sum((1, 2)); dot = (a * b).sum((1, 2))
        result['alternatives'].append(dict(coarse_delta=coarse_delta, common_positions=int(valid.sum()),
            first_moved_index=int(j[valid][0]), last_moved_index=int(j[valid][-1]),
            squared_reference=squared_reference.cpu(), squared_moved=squared_moved.cpu(),
            squared_difference=squared_difference.cpu(), dot=dot.cpu(),
            relative_L2=(squared_difference / squared_reference.clamp_min(1e-12)).sqrt().cpu(),
            cosine_distance=(1 - dot / (squared_reference * squared_moved).clamp_min(1e-24).sqrt()).cpu(),
            descriptor_difference=(a.mean(-1) - b.mean(-1)).square().sum(1).sqrt().cpu()))
    return result


def _append(collector, key, value):
    collector.setdefault(key, []).append(value.detach().cpu())


def _margin(logits, labels):
    correct = logits[torch.arange(len(logits), device=logits.device), labels]
    other = logits.clone(); other[torch.arange(len(logits), device=logits.device), labels] = -torch.inf
    return correct - other.max(1).values


def _summary(records):
    return {str(delta): {level: dict(exact_alignment=value['exact_alignment'],
                cumulative_stride=value['cumulative_stride'],
                alternatives=[dict(coarse_delta=a['coarse_delta'], common_positions=a['common_positions'],
                    relative_L2_mean=float(a['relative_L2'].mean()),
                    cosine_distance_mean=float(a['cosine_distance'].mean()),
                    descriptor_difference_mean=float(a['descriptor_difference'].mean())) for a in value['alternatives']])
            for level, value in levels.items()} for delta, levels in records.items()}


def structure_checks(device='cpu'):
    generator = torch.Generator().manual_seed(9202)
    x = torch.randn(3, 8, 64, generator=generator).to(device)
    roundtrip = torch.equal(inverse_s2(s2(x)), x)
    carry_exact = all(torch.equal(carry_shift(s2(x), delta), s2(shift(x, delta))) for delta in DELTAS)
    with preserve_rng(device):
        module = CarryAwareT3(8, 2).to(device).eval().requires_grad_(False)
        original = module(s2(x)); errors = []
        for delta in DELTAS:
            moved = module(s2(shift(x, delta)))
            begin = max(0, delta) + 2; end = min(64, 64 + delta) - 2
            errors.append(float((moved[..., begin:end] - original[..., begin - delta:end - delta]).abs().max()))
    # Mixing concatenated phases with unrelated weights breaks the shift/carry representation.
    z = s2(x); arbitrary = z[:, :, 0] + 2 * z[:, :, 1]
    moved = carry_shift(z, 1); arbitrary_moved = moved[:, :, 0] + 2 * moved[:, :, 1]
    counterexample = float((arbitrary_moved[..., 2:-2] - arbitrary[..., 2:-2]).abs().max())
    return dict(S2_roundtrip_exact=roundtrip, phase_exchange_carry_exact=carry_exact,
                carry_common_maxabs=max(errors), arbitrary_phase_concat_counterexample_maxabs=counterexample,
                passed=roundtrip and carry_exact and max(errors) <= 1e-5 and counterexample >= 1e-3,
                exact_equivariance_scope='fine-grid shifts at this replacement input, common valid region, eval/zero dropout; earlier strides and native global GN are not repaired')


def _sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def _whole_model_measure(model, replacement, batch, device):
    """Measure an actual cloned identity path with only pool/t3 replaced."""
    from experiments.cvs_e20_cross_style.style import _paths

    clone = copy.deepcopy(model).to(device).eval()
    modules, paths = _paths(clone)
    prefix = paths['time_down'].rsplit('.', 1)[0]
    native = modules[prefix]
    if not hasattr(native, 't2') or not hasattr(native.t2, 'pool') or len(replacement) != 2:
        raise ValueError('Native identity path must expose t2.pool and t3 for the registered replacement')
    native.t2.pool = copy.deepcopy(replacement[0]).to(device)
    native.t3 = copy.deepcopy(replacement[1]).to(device)
    clone.eval().requires_grad_(False)
    params = sum(parameter.numel() for parameter in clone.parameters())
    with torch.no_grad():
        for _ in range(2):
            output = clone.forward_identity_only(batch)
        _sync(device)
        start = time.perf_counter()
        for _ in range(5):
            output = clone.forward_identity_only(batch)
        _sync(device)
    if output['id_feat_joint'].shape != (len(batch), 160) or output['tx_logits'].shape != (len(batch), 6):
        raise ValueError('Sampling replacement changed the registered joint160/cosine output interface')
    return dict(whole_model_parameters=params,
                whole_model_output_bytes_per_packet=output['id_feat_joint'][0].numel() * output['id_feat_joint'].element_size(),
                whole_model_batch_size=len(batch),
                whole_model_forward_seconds= (time.perf_counter() - start) / 5,
                whole_model_timing='forward_identity_only, evaluation mode, batch latency measured after two warmups')


def _cost(modules, device, model=None, input_batch=None):
    x = torch.zeros(8, 96, 64, device=device); result = {}
    with torch.no_grad():
        for name, module in modules.items():
            module.to(device).eval().requires_grad_(False)
            macs = [0]
            handles = []
            for child in module.modules():
                if isinstance(child, nn.Conv1d):
                    def count_conv(layer, inputs, output, macs=macs):
                        macs[0] += output.numel() * (layer.in_channels // layer.groups) * layer.kernel_size[0]
                    handles.append(child.register_forward_hook(count_conv))
            value = module(x)
            for handle in handles:
                handle.remove()
            for _ in range(2):
                value = module(x)
            if isinstance(module[0], LowpassDownsample):
                macs[0] += x.shape[0] * x.shape[1] * x.shape[-1] * 3
            if len(module) > 1 and isinstance(module[1], CarryAwareT3):
                macs[0] += value.numel() * module[1].dw.kernel_size[0]
            _sync(device)
            start = time.perf_counter()
            for _ in range(10):
                value = module(x)
            _sync(device)
            length = value.shape[-1]
            result[name] = dict(replacement_parameters=sum(p.numel() for p in module.parameters()),
                               output_shape=list(value.shape), output_bytes_per_packet=value[0].numel() * value.element_size(),
                               t3_MAC_per_packet=macs[0] // x.shape[0],
                               measured_batch8_seconds=(time.perf_counter() - start) / 10,
                               MAC_scope='Conv1d hooks plus the explicit depthwise carry and lowpass convolutions; excludes GN, activation, reindexing and memory traffic')
            if model is not None:
                if input_batch is None or len(input_batch) == 0:
                    raise ValueError('Whole-model timing requires a nonempty real source input batch')
                result[name].update(_whole_model_measure(model, module, input_batch, device))
    return result


@torch.no_grad()
def diagnose(ctx, output, device):
    output.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    identity_before = snapshot(ctx['model'])
    audit = dict(deltas=list(DELTAS), source_only=True, identity_updates=0, roles={})
    role_summaries = {}; local_records = {}; modules = None
    with preserve_rng(device):
        for role, cache in (('L_s', ctx['L']), ('V', ctx['V'])):
            tensors = {}; map_parts = {delta: {} for delta in DELTAS}; local_parts = {}
            for begin in range(0, len(cache['ids']), d.BATCH_SIZE):
                x = cache['x'][begin:begin + d.BATCH_SIZE].to(device)
                y = cache['y'][begin:begin + len(x)].to(device)
                with capture_levels(ctx['model']) as (reference, native):
                    values = ctx['model'].forward_identity_only(x)
                if modules is None:
                    modules = replacements(native.t3, device)
                    for module in modules.values():
                        module.to(device).eval().requires_grad_(False)
                logits = values['tx_logits']; _append(tensors, 'base_logits', logits)
                _append(tensors, 'base_margin', _margin(logits, y))
                _append(tensors, 'base_joint160', values['id_feat_joint'])
                local_mask = ctx['folds'][begin:begin + len(x)] == 2 if role == 'L_s' else None
                local_input = reference['t2.act'][local_mask.to(device)] if role == 'L_s' and local_mask.any() else None
                local_base = {name: module(local_input) for name, module in modules.items()} if local_input is not None else {}
                for delta in DELTAS:
                    with capture_levels(ctx['model']) as (moved, _):
                        changed = ctx['model'].forward_identity_only(shift(x, delta))
                    _append(tensors, f'logits_{delta}', changed['tx_logits'])
                    _append(tensors, f'margin_{delta}', _margin(changed['tx_logits'], y))
                    _append(tensors, f'joint160_{delta}', changed['id_feat_joint'])
                    for level, (stride, center, width) in GEOMETRY.items():
                        pair = paired_sums(reference[level], moved[level], delta, stride, center, width, 256)
                        bucket = map_parts[delta].setdefault(level, dict(**{k: v for k, v in pair.items() if k != 'alternatives'}, alternatives=[]))
                        for i, alternative in enumerate(pair['alternatives']):
                            if len(bucket['alternatives']) <= i:
                                bucket['alternatives'].append({k: v for k, v in alternative.items() if not torch.is_tensor(v)})
                            for key, value in alternative.items():
                                if torch.is_tensor(value):
                                    _append(bucket['alternatives'][i], key, value)
                    if local_input is not None:
                        for name, module in modules.items():
                            local_moved = module(shift(local_input, delta))
                            stride = 1 if name == 'carry_point_GN' else 2
                            center = 0. if name in ('carry_point_GN', 'lowpass_point_GN') else .5
                            width = 3 if name == 'carry_point_GN' else 7 if name == 'lowpass_point_GN' else 6
                            pair = paired_sums(local_base[name], local_moved, delta, stride, center, width, 64)
                            bucket = local_parts.setdefault((name, delta), dict(**{k: v for k, v in pair.items() if k != 'alternatives'}, alternatives=[]))
                            for i, alternative in enumerate(pair['alternatives']):
                                if len(bucket['alternatives']) <= i:
                                    bucket['alternatives'].append({k: v for k, v in alternative.items() if not torch.is_tensor(v)})
                                for key, value in alternative.items():
                                    if torch.is_tensor(value):
                                        _append(bucket['alternatives'][i], key, value)
            for levels in map_parts.values():
                for value in levels.values():
                    for alternative in value['alternatives']:
                        for key, parts in list(alternative.items()):
                            if isinstance(parts, list):
                                alternative[key] = torch.cat(parts)
            for value in local_parts.values():
                for alternative in value['alternatives']:
                    for key, parts in list(alternative.items()):
                        if isinstance(parts, list):
                            alternative[key] = torch.cat(parts)
            tensors = {key: torch.cat(parts) for key, parts in tensors.items()}
            audit['roles'][role] = dict(ids=list(cache['ids']), y=cache['y'], rx=cache['rx'], day=cache['day'],
                                        values=tensors, layer_pairs=map_parts)
            prediction = tensors['base_logits'].argmax(1)
            role_summaries[role] = dict(count=len(cache['ids']), base_accuracy=float((prediction == cache['y']).float().mean()),
                shifts={str(delta): dict(accuracy=float((tensors[f'logits_{delta}'].argmax(1) == cache['y']).float().mean()),
                    prediction_change_count=int((tensors[f'logits_{delta}'].argmax(1) != prediction).sum()),
                    mean_margin_change=float((tensors[f'margin_{delta}'] - tensors['base_margin']).mean()),
                    median_absolute_margin_change=float((tensors[f'margin_{delta}'] - tensors['base_margin']).abs().median()))
                    for delta in DELTAS}, layers=_summary(map_parts))
            if role == 'L_s':
                local_records = local_parts
                audit['local_fold2'] = dict(ids=[pid for i, pid in enumerate(cache['ids']) if ctx['folds'][i] == 2],
                                           records={f'{name}:delta{delta}': value for (name, delta), value in local_parts.items()})
            print(f'SAMPLING_SOURCE_ROLE_COMPLETE role={role} count={len(cache["ids"])}', flush=True)
        structural = structure_checks(device)
        input_batch = ctx['L']['x'][:min(8, len(ctx['L']['ids']))].to(device)
        costs = _cost(modules, device, ctx['model'], input_batch)
    identity_state = assert_unchanged(ctx['model'], identity_before)
    local = {f'{name}:delta{delta}': dict(exact_alignment=value['exact_alignment'],
                relative_L2_mean_by_bracket=[float(a['relative_L2'].mean()) for a in value['alternatives']])
             for (name, delta), value in local_records.items()}
    improvements = []; carry_errors = []
    for delta in (-1, 1):
        ordinary = local_records[('ordinary_point_GN', delta)]['alternatives']
        carry = local_records[('carry_point_GN', delta)]['alternatives'][0]['relative_L2']
        improvements.append(float(torch.stack([a['relative_L2'] for a in ordinary]).min(0).values.mean() - carry.mean()))
        carry_errors.append(float(carry.max()))
    values = audit['roles']['L_s']['values']; mask = ctx['folds'] == 2
    margin_change = torch.cat([(values[f'margin_{delta}'][mask] - values['base_margin'][mask]).abs() for delta in (-1, 1)])
    prediction_changes = torch.cat([values[f'logits_{delta}'][mask].argmax(1) != values['base_logits'][mask].argmax(1)
                                    for delta in (-1, 1)])
    sensitivity = dict(L_fold2_count=int(mask.sum()), unit_shift_prediction_change_count=int(prediction_changes.sum()),
                      unit_shift_comparisons=len(prediction_changes), unit_shift_prediction_change_rate=float(prediction_changes.float().mean()),
                      unit_shift_median_absolute_margin_change=float(margin_change.median()),
                      local_ordinary_minus_carry_relativeL2=improvements, local_carry_max_relativeL2=carry_errors)
    criteria = dict(structural=structural['passed'], source_phase_alias=min(improvements) >= .05,
                    source_carry_precision=max(carry_errors) <= 1e-5,
                    source_classifier_sensitive=sensitivity['unit_shift_prediction_change_rate'] >= .01 or
                                                sensitivity['unit_shift_median_absolute_margin_change'] >= .25)
    result = dict(direction='sampling', source_only=True, identity_updates=0, U_truth_read=False,
                  target_access=False, source_roles=role_summaries, structural=structural, local_fold2=local,
                  sensitivity=sensitivity, costs=costs, identity_state=identity_state, source_performance_gate=False,
                  applicability=dict(passed=all(criteria.values()), criteria=criteria,
                      reason='Fixed source phase-alias and margin sensitivity criteria; this supports a structural candidate, not measured unseen-RX identification improvement.'),
                  declared_limits=['Only exactly divisible cumulative-stride shifts count as aligned native equivariance comparisons.',
                    'Native temporal GroupNorm mixes the entire sequence; convolution-valid crops do not remove that dependence.',
                    'Last-stride S2 preserves both phases and doubles t3 sequence/MAC cost; it does not repair earlier strided layers.',
                    'V is source health only; there is zero identity fitting and no target selection.'],
                  source_forward_packets=5 * (len(ctx['L']['ids']) + len(ctx['V']['ids'])),
                  elapsed_seconds=time.perf_counter() - started)
    audit['identity_state'] = identity_state
    with (output / 'audit.pt').open('xb') as stream:
        torch.save(audit, stream)
    d.write(output / 'diagnostic.json', result)
    return result
