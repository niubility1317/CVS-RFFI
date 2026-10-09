"""Explicit increments and recipient-state conditional source action models.

Artificial paired L/T increments are not claimed to identify physical TX/RX
parameters. Auxiliary audit roles are disjoint physical packets; an optional TX
holdout concerns auxiliary learning only, never the identity training contract.
"""
import copy
import time
import torch
from torch import nn
from torch.nn import functional as F
from experiments.cvs_multi_disentangle.physics import apply_linear, apply_temporal
from experiments.cvs_multi_disentangle.model import intermediate, identity_from_intermediate, _encoder
from experiments.cvs_multi_action_audit.actions import fixed_identity
from experiments.cvs_multi_action_audit.action_checks import spectrum

KINDS = ('linear', 'temporal')
MODES = {
    'p_h_first_all': dict(use_state=False, second_order=False, exact_reference=False, estimator='true'),
    'p_hs_first_all': dict(use_state=True, second_order=False, exact_reference=False, estimator='true'),
    'p_hs_second_all': dict(use_state=True, second_order=True, exact_reference=False, estimator='true'),
    'p_hs_second_exact': dict(use_state=True, second_order=True, exact_reference=True, estimator='true'),
    'analytic_hs_second_exact': dict(use_state=True, second_order=True, exact_reference=True, estimator='analytic'),
    'neural_hs_second_exact': dict(use_state=True, second_order=True, exact_reference=True, estimator='neural'),
}


def apply_action(x, p, kind):
    return (apply_linear if kind == 'linear' else apply_temporal)(x, p)


def _svd_solve(a, b):
    """Batched double SVD least squares; no unstable normal equations."""
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    tol = s[:, :1] * max(a.shape[-2:]) * torch.finfo(s.dtype).eps
    valid = s > tol
    inv = torch.where(valid, s.clamp_min(torch.finfo(s.dtype).tiny).reciprocal(), 0.)
    answer = vh.mH @ (inv[..., None] * (u.mH @ b))
    cond = s[:, 0] / s[:, -1].clamp_min(torch.finfo(s.dtype).tiny)
    return answer, dict(condition=cond, rank=valid.sum(1), singular_values=s)


def estimate_pair(x0, x1, kind, window=(0, 256)):
    """Known digital pair LS; T weights are shared observed-packet power.

    Source noise rotates with the packet and is not independent pair noise.
    Diagnostics expose singular cases; the pseudoinverse does not imply that an
    unidentifiable increment has become identifiable. No labels enter estimation.
    """
    if kind not in KINDS or x0.shape != x1.shape or x0.shape[1:] != (2, 256):
        raise ValueError('Expected known kind and matched IQ [B,2,256]')
    z0 = torch.complex(x0[:, 0].double(), x0[:, 1].double())
    z1 = torch.complex(x1[:, 0].double(), x1[:, 1].double())
    lo, hi = window
    if not 0 <= lo < hi <= 256:
        raise ValueError('Invalid observation window')
    if kind == 'linear':
        columns = [F.pad(z0[:, :-lag], (lag, 0)) for lag in (1, 3)]
        a = .06 * torch.stack(columns, -1)[:, lo:hi]
        coefficient, info = _svd_solve(a, (z1 - z0)[:, lo:hi, None])
        c = coefficient[..., 0]
        result = torch.stack((c[:, 0].real, c[:, 0].imag, c[:, 1].real, c[:, 1].imag), 1)
    else:
        n = torch.linspace(-1., 1., 256, device=x0.device, dtype=torch.float64)[lo:hi]
        basis = torch.stack((torch.full_like(n, .08), .10 * n, .06 * n.square()), 1)
        weight = z0[:, lo:hi].abs()
        # atan2 receives neither added independent noise nor unwrapping assumptions.
        phase = torch.angle(z1[:, lo:hi] * z0[:, lo:hi].conj())
        result, info = _svd_solve(basis[None] * weight[..., None], (phase * weight)[..., None])
        result = result[..., 0]
    info['full_rank'] = info['rank'] == (2 if kind == 'linear' else 3)
    return result.to(x0), info


def signal_state(x):
    """40 recipient-only coordinates: 32 spectral bins and 8 moment/lag terms."""
    z = torch.complex(x[:, 0], x[:, 1])
    power = z.abs().square()
    scale = power.mean(1).clamp_min(1e-10)
    psd = torch.fft.fft(z).abs().square().reshape(len(x), 32, 8).mean(2)
    psd = torch.log1p(psd / (psd.mean(1, keepdim=True) + 1e-10))
    ratio = power / scale[:, None]
    values = [scale.log(), ratio.square().mean(1), ratio.pow(3).mean(1), ratio.max(1).values]
    for lag in (1, 20):
        cross = (z[:, lag:] * z[:, :-lag].conj()).mean(1) / scale
        values.extend((cross.real, cross.imag))
    return torch.cat((psd, 8 * torch.tanh(torch.stack(values, 1) / 8)), 1)


class NeuralPairEstimator(nn.Module):
    """Paired IQ estimation, exact zero for identical observations, not odd in p."""
    def __init__(self, kind, hidden=64):
        super().__init__()
        self.kind = kind
        self.parameter_dim = 4 if kind in ('linear', 'unified') else 3
        self.net = nn.Sequential(nn.Linear(1024, hidden), nn.SiLU(), nn.Linear(hidden, self.parameter_dim))

    def forward(self, x0, x1):
        scale = x0.square().mean((1, 2), keepdim=True).clamp_min(1e-10).sqrt()
        base, difference = (x0 / scale).flatten(1), ((x1 - x0) / scale).flatten(1)
        return self.net(torch.cat((base, difference), 1)) - self.net(torch.cat((base, torch.zeros_like(difference)), 1))


class StateAction(nn.Module):
    """State-conditioned Jp + Q vec_sym(pp^T), without a fixed global subspace.

    unified shares all learned weights and accepts a kind tag, enabling a fair
    one-network comparator. exact_reference uses the actual fixed frontend.
    """
    def __init__(self, kind, use_state=True, second_order=True, exact_reference=True, hidden=48):
        super().__init__()
        if kind not in (*KINDS, 'unified'):
            raise ValueError('Unknown action kind')
        self.kind, self.use_state = kind, use_state
        self.second_order, self.exact_reference = second_order, exact_reference
        self.parameter_dim = 4 if kind in ('linear', 'unified') else 3
        self.output_dim = 320 if exact_reference else 349
        self.nterms = self.parameter_dim + (self.parameter_dim * (self.parameter_dim + 1) // 2 if second_order else 0)
        self.h_norm = nn.LayerNorm(349)
        self.state_norm = nn.LayerNorm(40) if use_state else None
        size = 349 + (40 if use_state else 0) + (2 if kind == 'unified' else 0)
        self.trunk = nn.Sequential(nn.Linear(size, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.coefficients = nn.Linear(hidden, self.output_dim * self.nterms)
        nn.init.normal_(self.coefficients.weight, std=.001)
        nn.init.zeros_(self.coefficients.bias)

    def forward(self, h, x, p, reference_fn=None, kind=None):
        actual = kind or self.kind
        if actual not in KINDS:
            raise ValueError('unified action requires kind=linear/temporal')
        original = p[:, :4 if actual == 'linear' else 3]
        if self.kind == 'unified':
            p = F.pad(original, (0, 4 - original.shape[1]))
        elif p.shape[1] != self.parameter_dim:
            raise ValueError('Wrong increment dimension')
        terms = [p]
        if self.second_order:
            terms.append(torch.stack([p[:, i] * p[:, j] for i in range(p.shape[1]) for j in range(i, p.shape[1])], 1))
        basis = torch.cat(terms, 1)
        inputs = [self.h_norm(h)]
        if self.use_state:
            inputs.append(self.state_norm(signal_state(x)))
        if self.kind == 'unified':
            tag = h.new_zeros((len(h), 2)); tag[:, KINDS.index(actual)] = 1
            inputs.append(tag)
        coefficients = self.coefficients(self.trunk(torch.cat(inputs, 1))).reshape(len(h), self.nterms, self.output_dim)
        learned = (coefficients * basis[..., None]).sum(1)
        if not self.exact_reference:
            return learned
        if reference_fn is None:
            raise ValueError('Exact reference requires actual frozen reference frontend')
        exact = reference_fn(apply_action(x, original, actual)) - reference_fn(x)
        return torch.cat((learned, exact), 1)

    predict = forward


def vector_margin(logits, y):
    differences = logits.gather(1, y[:, None]) - logits
    keep = torch.arange(logits.shape[1], device=logits.device)[None] != y[:, None]
    return differences[keep].reshape(len(y), logits.shape[1] - 1)


def feature_scales(delta, relative_floor=.10, absolute_floor=1e-4):
    total = delta.square().mean().detach()
    floor = total.mul(relative_floor).clamp_min(absolute_floor)
    return torch.stack([delta[:, a:b].square().mean().detach().clamp_min(floor)
                        for a, b in ((0, 160), (160, 320), (320, 349))])


def action_loss(identity, h, target_delta, pred_delta, y, scales=None, lambda_z=1., lambda_margin=.1):
    """Frozen G/C weights retain derivatives to predicted features; all margins."""
    scales = feature_scales(target_delta) if scales is None else scales
    widths = (160 / 349, 160 / 349, 29 / 349)
    feature = sum(w * (pred_delta[:, a:b] - target_delta[:, a:b]).square().mean() / s
                  for w, (a, b), s in zip(widths, ((0, 160), (160, 320), (320, 349)), scales))
    target_z = identity_from_intermediate(identity, h + target_delta).detach()
    pred_z = identity_from_intermediate(identity, h + pred_delta)
    norm_z = (F.normalize(pred_z, dim=1) - F.normalize(target_z, dim=1)).square().mean()
    classifier = _encoder(identity).classify_features
    margins = (vector_margin(classifier(pred_z), y) - vector_margin(classifier(target_z).detach(), y)).square().mean()
    return dict(loss=feature + lambda_z * norm_z + lambda_margin * margins,
                feature=feature, normalized_z=norm_z, margins=margins)


def _batch(function, x, batch_size=32):
    return torch.cat([function(x[i:i + batch_size]).detach() for i in range(0, len(x), batch_size)])


def _values(data, key):
    value = data[key]
    return value.detach().cpu().tolist() if torch.is_tensor(value) else list(value)


def validate_roles(fit, audit, heldout_tx=None):
    for data in (fit, audit):
        for key in ('x', 'y', 'rx', 'day', 'ids', 'condition'):
            if len(data[key]) != len(data['x']):
                raise ValueError('Missing/inconsistent source metadata ' + key)
        keys = list(zip(map(str, data['ids']), map(str, _values(data, 'condition'))))
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate packet/view within auxiliary role')
    if set(map(str, fit['ids'])) & set(map(str, audit['ids'])):
        raise ValueError('Auxiliary fit/audit physical IDs overlap')
    if heldout_tx is not None and heldout_tx in set(_values(fit, 'y')):
        raise ValueError('Heldout auxiliary TX present in fit role')


def donor_indices(data):
    """Match RX/day/view then nearest observed quality; preserve missing matches."""
    ys, rx, day, condition = [_values(data, k) for k in ('y', 'rx', 'day', 'condition')]
    ids = list(map(str, data['ids']))
    if 'quality' in data:
        quality = _values(data, 'quality'); quality_kind = 'provided_receiver_quality'
    else:
        quality = data['x'].square().mean((1, 2)).clamp_min(1e-10).log().detach().cpu().tolist()
        quality_kind = 'log_IQ_power_proxy_not_calibrated_SNR'
    names = ('own', 'same_tx_other_packet', 'cross_tx_matched', 'cross_tx_unmatched')
    result = {key: [] for key in names}; available = {key: [] for key in names}
    for i in range(len(ys)):
        match = lambda j: rx[j] == rx[i] and day[j] == day[i] and condition[j] == condition[i]
        candidates = {
            'own': [i],
            'same_tx_other_packet': [j for j in range(len(ys)) if ys[j] == ys[i] and ids[j] != ids[i] and match(j)],
            'cross_tx_matched': [j for j in range(len(ys)) if ys[j] != ys[i] and match(j)],
            'cross_tx_unmatched': [j for j in range(len(ys)) if ys[j] != ys[i] and not match(j)],
        }
        for key, allowed in candidates.items():
            selected = min(allowed, key=lambda j: (abs(quality[j] - quality[i]), ids[j])) if allowed else i
            result[key].append(selected); available[key].append(bool(allowed))
    device = data['x'].device
    return ({k: torch.tensor(v, device=device) for k, v in result.items()},
            {k: torch.tensor(v, dtype=torch.bool, device=device) for k, v in available.items()}, quality_kind)


@torch.no_grad()
def _cache(identity, data, kind, generator, interventions, batch_size):
    x, y = data['x'], data['y'].long().to(data['x'].device)
    h = _batch(lambda v: intermediate(identity, v), x, batch_size)
    width = 4 if kind == 'linear' else 3
    # Several preregistered p per packet; each includes zero and both signs.
    unique_ids = {key: i for i, key in enumerate(sorted(set(map(str, data['ids']))))}
    physical_index = torch.tensor([unique_ids[str(key)] for key in data['ids']], device=x.device)
    physical_p = (2 * torch.rand(interventions, len(unique_ids), width, generator=generator) - 1).to(x)
    positive = physical_p[:, physical_index]  # Clean/LEO views share each physical packet's intervention.
    params = torch.cat((torch.zeros_like(positive[:1]), positive, -positive), 0)
    index = torch.arange(len(x), device=x.device).repeat(len(params))
    flat_p = params.flatten(0, 1)
    changed = apply_action(x[index], flat_p, kind)
    delta = _batch(lambda v: intermediate(identity, v), changed, batch_size) - h[index]
    donors, masks, quality_kind = donor_indices(data)
    return dict(data=data, h=h, x=x, y=y, p=flat_p, index=index, delta=delta,
                donors=donors, masks=masks, quality_kind=quality_kind,
                sign=torch.tensor([0] + [1] * interventions + [-1] * interventions, device=x.device).repeat_interleave(len(x)))


@torch.no_grad()
def _metrics(identity, h, delta, prediction, y):
    z0 = identity_from_intermediate(identity, h)
    z1 = identity_from_intermediate(identity, h + delta)
    zp = identity_from_intermediate(identity, h + prediction)
    classifier = _encoder(identity).classify_features
    m0, m1, mp = [vector_margin(classifier(z), y) for z in (z0, z1, zp)]
    def error(p, t):
        energy = float(t.square().mean()); mse = float((p - t).square().mean())
        return dict(mse=mse, zero_mse=energy, skill_vs_zero=1 - mse / energy if energy > 1e-12 else None)
    return dict(count=len(h), h=error(prediction, delta), z=error(zp - z0, z1 - z0),
                normalized_z=error(F.normalize(zp, dim=1) - F.normalize(z0, dim=1),
                                   F.normalize(z1, dim=1) - F.normalize(z0, dim=1)),
                margin=error(mp - m0, m1 - m0), margin_mae=float((mp - m1).abs().mean()),
                margin_zero_mae=float((m1 - m0).abs().mean()))


def _estimate(estimator, source, x0, x1, p, kind):
    if source == 'true':
        return p
    if source == 'analytic':
        return estimate_pair(x0, x1, kind)[0]
    return estimator(x0, x1)


@torch.no_grad()
def _audit(identity, model, estimator, source, cache, batch_size):
    reports = {}
    n = len(cache['index'])
    for donor_name in (*cache['donors'], 'shuffled_increment'):
        predictions, estimates = [], []
        for start in range(0, n, batch_size):
            sl = slice(start, start + batch_size); idx = cache['index'][sl]; p = cache['p'][sl]
            name = 'cross_tx_matched' if donor_name == 'shuffled_increment' else donor_name
            donor = cache['donors'][name][idx]
            # Permute intervention assignment, retaining donor RX/day/quality.
            observed_p = cache['p'].roll(len(cache['x']), 0)[sl] if donor_name == 'shuffled_increment' else p
            x0 = cache['x'][donor]; x1 = apply_action(x0, observed_p, model.kind)
            estimate = _estimate(estimator, source, x0, x1, observed_p, model.kind)
            predictions.append(model(cache['h'][idx], cache['x'][idx], estimate, _encoder(identity).response))
            estimates.append(estimate)
        prediction, estimates = torch.cat(predictions), torch.cat(estimates)
        name = 'cross_tx_matched' if donor_name == 'shuffled_increment' else donor_name
        valid = cache['masks'][name][cache['index']]
        result = dict(available_count=int(valid.sum()), missing_count=int((~valid).sum()), strata={})
        conditions = _values(cache['data'], 'condition')
        for condition in sorted(set(conditions), key=str):
            condition_mask = torch.tensor([c == condition for c in conditions], device=valid.device)[cache['index']]
            for sign, sign_name in ((0, 'zero'), (1, 'positive'), (-1, 'negative'), (None, 'all')):
                mask = valid & condition_mask
                if sign is not None:
                    mask = mask & (cache['sign'] == sign)
                key = str(condition) + '/' + sign_name
                if not bool(mask.any()):
                    result['strata'][key] = dict(status='N/A_NO_MATCH', count=0)
                    continue
                idx = cache['index'][mask]
                values = _metrics(identity, cache['h'][idx], cache['delta'][mask], prediction[mask], cache['y'][idx])
                values['parameter_mse'] = float((estimates[mask] - cache['p'][mask]).square().mean())
                result['strata'][key] = values
        reports[donor_name] = result
        packet_count = len(cache['x'])
        draws = (len(cache['p']) // packet_count - 1) // 2
        positive = slice(packet_count, (draws + 1) * packet_count)
        negative = slice((draws + 1) * packet_count, None)
        parity_valid = valid[positive] & valid[negative]
        if bool(parity_valid.any()):
            even_target = (cache['delta'][positive] + cache['delta'][negative]) / 2
            even_prediction = (prediction[positive] + prediction[negative]) / 2
            odd_target = (cache['delta'][positive] - cache['delta'][negative]) / 2
            odd_prediction = (prediction[positive] - prediction[negative]) / 2
            result['even_odd'] = dict(even_target_energy=float(even_target[parity_valid].square().mean()),
                even_prediction_mse=float((even_prediction[parity_valid] - even_target[parity_valid]).square().mean()),
                odd_target_energy=float(odd_target[parity_valid].square().mean()),
                odd_prediction_mse=float((odd_prediction[parity_valid] - odd_target[parity_valid]).square().mean()))
        result['tx_strata'] = {}
        for tx in sorted(set(cache['y'].tolist())):
            mask = valid & (cache['y'][cache['index']] == tx) & (cache['sign'] != 0)
            if bool(mask.any()):
                idx = cache['index'][mask]
                result['tx_strata'][str(tx)] = _metrics(identity, cache['h'][idx], cache['delta'][mask], prediction[mask], cache['y'][idx])
    return reports


def fit_and_audit(identity, fit, audit, generator, *, steps=200, batch_size=32, interventions=3,
                  heldout_tx=None, modes=None, kinds=KINDS, learning_rate=2e-4, on_step=None):
    """Equal exposure schedule and joint-condition model, never audit-driven stop.

    Return JSON-safe report/logs and CPU state dicts. Caller is responsible for
    source-only provenance and publication. Physical packet views retain roles.
    """
    validate_roles(fit, audit, heldout_tx)
    if generator is None or str(generator.device) != 'cpu' or not 1 <= batch_size <= 32 or steps < 1 or interventions < 1:
        raise ValueError('Private CPU generator, positive budget, and batch <=32 required')
    modes = tuple(MODES) if modes is None else tuple(modes)
    if not set(modes) <= set(MODES) or not set(kinds) <= set(KINDS):
        raise ValueError('Unregistered action mode')
    report = dict(identity_updated=False, target_access=False, fit_count=len(fit['x']), audit_count=len(audit['x']),
                  heldout_tx=heldout_tx, heldout_scope='auxiliary only; identity may have trained on all source TX',
                  jointly_fitted_conditions=sorted(set(_values(fit, 'condition')), key=str), steps=steps,
                  interventions=interventions, signs=[0, 1, -1], audit_feedback=False, kinds={})
    logs, states, configs = [], {}, {}
    with fixed_identity(identity):
        for kind in kinds:
            cf = _cache(identity, fit, kind, generator, interventions, batch_size)
            ca = _cache(identity, audit, kind, generator, interventions, batch_size)
            scales = feature_scales(cf['delta'])
            init_seed = int(torch.randint(2**31 - 1, (1,), generator=generator))
            schedule_seed = int(torch.randint(2**31 - 1, (1,), generator=generator))
            kr = dict(modes={}, scale_floor_rule='max(block_energy, 0.1*global_energy,1e-4); dimension weights',
                      feature_scales=scales.tolist(), quality_matching=ca['quality_kind'],
                      spectra={str(c): spectrum(ca['delta'][torch.tensor([v == c for v in _values(audit, 'condition')], device=ca['x'].device)[ca['index']]])
                               for c in sorted(set(_values(audit, 'condition')), key=str)})
            from .action_checks import physical_diagnostics
            kr['physical'] = physical_diagnostics(identity, audit, kind, ca, batch_size)
            for mode in modes:
                config = copy.deepcopy(MODES[mode]); source = config.pop('estimator')
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(init_seed)
                    model = StateAction(kind, **config).to(fit['x'].device)
                    estimator = NeuralPairEstimator(kind).to(fit['x'].device) if source == 'neural' else None
                parameters = list(model.parameters()) + (list(estimator.parameters()) if estimator is not None else [])
                optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
                schedule = torch.Generator().manual_seed(schedule_seed)
                mode_logs = []
                for step in range(steps):
                    tick = time.perf_counter()
                    row = torch.randperm(len(cf['index']), generator=schedule)[:batch_size].to(fit['x'].device)
                    idx, p = cf['index'][row], cf['p'][row]
                    donor = cf['donors']['cross_tx_matched'][idx]
                    valid = cf['masks']['cross_tx_matched'][idx]
                    # Own and cross both appear at a fixed total recipient budget.
                    cross = torch.arange(len(idx), device=idx.device) % 2 == step % 2
                    donor = torch.where(cross & valid, donor, idx)
                    x0 = cf['x'][donor]; x1 = apply_action(x0, p, kind)
                    estimate = _estimate(estimator, source, x0, x1, p, kind)
                    prediction = model(cf['h'][idx], cf['x'][idx], estimate, _encoder(identity).response)
                    losses = action_loss(identity, cf['h'][idx], cf['delta'][row], prediction, cf['y'][idx], scales)
                    parameter_loss = (estimate - p).square().mean()
                    loss = losses['loss'] + (parameter_loss if estimator is not None else 0.)
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 10.)
                    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):
                        raise FloatingPointError('Nonfinite source action optimization')
                    optimizer.step()
                    record = dict(kind=kind, mode=mode, step=step + 1, loss=float(loss.detach()),
                                  **{k: float(v.detach()) for k, v in losses.items() if k != 'loss'},
                                  parameter=float(parameter_loss.detach()), gradient_norm=float(norm),
                                  samples=len(idx), cross_samples=int((cross & valid).sum()),
                                  learning_rate=learning_rate, seconds=time.perf_counter() - tick)
                    logs.append(record); mode_logs.append(record)
                    if on_step: on_step(record)
                audit_result = _audit(identity, model, estimator, source, ca, batch_size)
                head = sum(v['loss'] for v in mode_logs[:min(20, steps)]) / min(20, steps)
                tail = sum(v['loss'] for v in mode_logs[-min(20, steps):]) / min(20, steps)
                kr['modes'][mode] = dict(audit=audit_result, trainable_parameters=sum(p.numel() for p in parameters),
                    optimizer_steps=steps, exposure=steps * min(batch_size, len(cf['index'])),
                    loss_first20=head, loss_last20=tail,
                    convergence_claim='budget-bounded fit; declining loss does not establish capacity ceiling',
                    sample_schedule_seed=schedule_seed, initializer_seed=init_seed)
                key = kind + '/' + mode
                states[key] = dict(action={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                   estimator={k: v.detach().cpu() for k, v in estimator.state_dict().items()} if estimator else None)
                configs[key] = dict(kind=kind, estimator=source, **config)
            report['kinds'][kind] = kr
    return dict(report=report, step_logs=logs, state_dicts=states, model_configs=configs)


@torch.no_grad()
def composition_audit(identity, x, generator, state_dicts, model_configs, batch_size=32):
    """No LT network: real four-corner residual versus learned main-effect error."""
    pl = (2 * torch.rand(len(x), 4, generator=generator) - 1).to(x)
    pt = (2 * torch.rand(len(x), 3, generator=generator) - 1).to(x)
    xl, xt = apply_linear(x, pl), apply_temporal(x, pt)
    xlt, xtl = apply_temporal(xl, pt), apply_linear(xt, pl)
    with fixed_identity(identity), torch.random.fork_rng(devices=[]):
        h, hl, ht, hlt, htl = [_batch(lambda v: intermediate(identity, v), a, batch_size) for a in (x, xl, xt, xlt, xtl)]
        report = dict(lt_network=False, four_corner_energy=float((hlt - hl - ht + h).square().mean()),
                      real_order_mse=float((hlt - htl).square().mean()),
                      temporal_changed_state_inverse_iq_mse=float((apply_temporal(xt, -pt) - x).square().mean()),
                      linear_negative_not_inverse_iq_mse=float((apply_linear(xl, -pl) - x).square().mean()),
                      interpretation='Finite main-effect residual includes curvature; not unique physical interaction.', modes={})
        for mode in MODES:
            if not all(k + '/' + mode in state_dicts for k in KINDS): continue
            models, deltas = {}, {}
            for kind, p in zip(KINDS, (pl, pt)):
                key = kind + '/' + mode; config = copy.deepcopy(model_configs[key]); source = config.pop('estimator')
                model = StateAction(**config).to(x.device); model.load_state_dict(state_dicts[key]['action'])
                estimator = None
                if source == 'neural':
                    estimator = NeuralPairEstimator(kind).to(x.device); estimator.load_state_dict(state_dicts[key]['estimator'])
                estimate = _estimate(estimator, source, x, apply_action(x, p, kind), p, kind)
                deltas[kind] = model(h, x, estimate, _encoder(identity).response)
                models[kind] = (model, estimator, source)
            model, estimator, source = models['temporal']
            estimate = _estimate(estimator, source, xl, xlt, pt, 'temporal')
            sequential = h + deltas['linear'] + model(h + deltas['linear'], xl, estimate, _encoder(identity).response)
            inverse_estimate = _estimate(estimator, source, xt, apply_temporal(xt, -pt), -pt, 'temporal')
            inverse_prediction = model(ht, xt, inverse_estimate, _encoder(identity).response)
            report['modes'][mode] = dict(additive_mse=float((h + deltas['linear'] + deltas['temporal'] - hlt).square().mean()),
                                       sequential_mse=float((sequential - hlt).square().mean()),
                                       temporal_changed_state_inverse_prediction_mse=float((ht + inverse_prediction - h).square().mean()))
    return report
