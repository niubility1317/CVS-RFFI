"""One actual-feature forward for complete temporary CE continuation states."""
from collections import Counter
from collections.abc import Mapping, Sequence
import copy

import torch
from torch import nn
from torch.nn import functional as F

from experiments.cvs_crossrx_sharpness20.source_features import geometry
from experiments.cvs_evidence_recomposition20.capture import forward as capture, HEAD_PATH, SEQUENCE_PATH
from experiments.cvs_identity_difference20.bank import SOURCE_RX, SOURCE_DAYS
from experiments.cvs_sigtime_heldcredit.held_credit import CACHE_ATTRIBUTES, physical_ids
from experiments.cvs_u_schedule50.counterfactual import _preserve_rng


ENCODER = 'id_backbone.encoder'
EPS = 1e-4
OPTIMIZER_PARAMETER_COUNT = 188


def _finite(value, shape, name, *, dtype=None, device=None):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.shape != shape
            or not value.is_floating_point() or (dtype is not None and value.dtype != dtype)
            or (device is not None and value.device != device) or not bool(torch.isfinite(value).all())):
        raise ValueError('Finite actual state/shape/dtype/device required: ' + name)


def _bank(bank, model):
    if not isinstance(bank, Mapping) or set(bank) != {'x', 'y', 'rx', 'day', 'side', 'ids'}:
        raise ValueError('The visible source held-L180 bank is required')
    x, y, rx, day, side = (bank[key] for key in ('x', 'y', 'rx', 'day', 'side'))
    parameter = next(model.parameters())
    _finite(x, (180, 2, 256), 'held L IQ', dtype=parameter.dtype, device=parameter.device)
    if x.requires_grad:
        raise ValueError('Held L IQ must be detached; no evaluation gradient')
    for name, value in (('y', y), ('rx', rx), ('day', day), ('side', side)):
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.shape != (180,)
                or value.dtype != torch.long or value.device != x.device):
            raise ValueError('Visible held-L integer vector differs: ' + name)
    if (not bool(((y >= 0) & (y < 6)).all()) or set(rx.tolist()) != set(SOURCE_RX)
            or set(day.tolist()) != set(SOURCE_DAYS) or set(side.tolist()) != {0, 1}):
        raise ValueError('Visible source TX/RX/day and two sides are required; no hidden U labels')
    counts = Counter(zip(y.tolist(), rx.tolist(), day.tolist(), side.tolist()))
    expected = {(label, receiver, source_day, arm): 1 for label in range(6)
                for receiver in SOURCE_RX for source_day in SOURCE_DAYS for arm in (0, 1)}
    if counts != expected:
        raise ValueError('Held L180 must have one packet per six-TX/five-RX/three-day/side cell')
    identities = physical_ids(bank['ids'], 180)
    if any(identity[1] != int(rx[index]) or identity[2] != int(day[index])
           for index, identity in enumerate(identities)):
        raise ValueError('Held L physical RX/day differs from bank metadata')
    return x, y, rx, day


def _readout(logits, labels, receivers):
    _finite(logits, (180, 6), 'default-head class logits')
    logits64 = logits.double()
    losses = F.cross_entropy(logits64, labels, reduction='none')
    truth_scores = logits64.gather(1, labels[:, None]).squeeze(1)
    other = logits64.clone()
    other.scatter_(1, labels[:, None], float('-inf'))
    margins = truth_scores - other.max(dim=1).values
    correct = logits64.argmax(1) == labels
    if not bool(torch.isfinite(losses).all()) or not bool(torch.isfinite(margins).all()):
        raise ValueError('Nonfinite source CE or actual class-logit margin')
    groups = {str(receiver): dict(count=int((receivers == receiver).sum()),
        CE=float(losses[receivers == receiver].mean()),
        class_logit_margin_mean=float(margins[receivers == receiver].mean()),
        accuracy=float(correct[receivers == receiver].double().mean())) for receiver in SOURCE_RX}
    per_rx = {receiver: row['CE'] for receiver, row in groups.items()}
    return dict(mean_CE=float(losses.mean()), by_RX=per_rx, RX=groups,
        min_RX_CE=min(per_rx.values()), max_RX_CE=max(per_rx.values()), worst_RX_CE=max(per_rx.values()),
        class_logit_margin_mean=float(margins.mean()),
        class_logit_margin_by_RX={receiver: row['class_logit_margin_mean'] for receiver, row in groups.items()},
        class_logit_margin_min=float(margins.min()), accuracy=float(correct.double().mean()),
        logits_dtype=str(logits.dtype), CE_dtype='float64')


def evaluate(model, names, after, bank):
    """Materialize all188 optimizer parameters, including after W/three gains.

    ACTUAL/PRE_W both use the identical actual raw joint and original default
    formula in FP64 at scale30/eps1e-4, with afterW/preW respectively. Native
    tx_logits from that same forward are checked with FP32 rounding tolerance.
    Fusion gains remain the after values in both readouts.
    Source ancestry/held ordinal selection are the bank caller's responsibility.
    """
    if not isinstance(model, nn.Module):
        raise ValueError('The existing actual native source model is required')
    x, labels, receivers, days = _bank(bank, model)
    named = dict(model.named_parameters())
    if (not isinstance(names, Sequence) or isinstance(names, (str, bytes))
            or len(names) != OPTIMIZER_PARAMETER_COUNT or any(not isinstance(name, str) for name in names)
            or len(set(names)) != len(names) or set(names) != set(named)
            or not isinstance(after, (tuple, list)) or len(after) != len(names)):
        raise ValueError('Complete188 actual optimizer names and same-length after parameters required')
    for name, value in zip(names, after):
        parameter = named[name]
        _finite(value, parameter.shape, name, dtype=parameter.dtype, device=parameter.device)
    try:
        head = model.get_submodule(HEAD_PATH)
        encoder = model.get_submodule(ENCODER)
        pool = model.get_submodule(SEQUENCE_PATH)
        gains = (head.gain, encoder.response_gain, pool.gain_raw)
        weight = head.weight
    except (AttributeError, ValueError) as error:
        raise ValueError('The actual default head and three deployed fusion gains are required') from error
    if float(head.scale) != 30. or getattr(pool, 'arm', None) != 'SIG_TIME':
        raise ValueError('Unchanged native scale30 and actual SIG_TIME deployment required')
    _finite(weight, (6, 160), 'pre-step W', dtype=x.dtype, device=x.device)
    for gain in gains:
        _finite(gain, (160,), 'pre-step fusion gain', dtype=x.dtype, device=x.device)
    weight_name = next((name for name, value in named.items() if value is weight), None)
    gain_names = [next((name for name, value in named.items() if value is gain), None) for gain in gains]
    if weight_name is None or any(name is None for name in gain_names) or len(set(gain_names)) != 3:
        raise ValueError('W/three gains must be genuine distinct optimizer parameters')
    parameters = [(parameter, parameter.detach().clone(), parameter._version, parameter.grad,
        None if parameter.grad is None else parameter.grad.detach().clone()) for parameter in model.parameters()]
    buffers = [(buffer, buffer.detach().clone(), buffer._version) for buffer in model.buffers()]
    flags = [(module, module.training) for module in model.modules()]
    caches = [(module, key, getattr(module, key)) for module in model.modules()
              for key in CACHE_ATTRIBUTES if hasattr(module, key)]
    hooks = [(module, dict(module._forward_hooks), dict(module._forward_pre_hooks)) for module in model.modules()]
    with _preserve_rng(), torch.no_grad():
        temporary = copy.deepcopy(model).eval()
        target = dict(temporary.named_parameters())
        for name, value in zip(names, after):
            target[name].copy_(value.detach())
        # No W/gain restoration: this is the real full continuation state.
        components = capture(temporary, x)
        z = components['joint'].detach().clone()
        unit = F.normalize(z.double(), dim=1, eps=EPS)
        actual_logits = 30. * F.linear(unit, F.normalize(target[weight_name].double(), dim=1, eps=EPS))
        native_logits = components['logits'].double()
        native_error = float((actual_logits - native_logits).abs().max())
        native_atol, native_rtol = 2e-5, 2e-6
        if not torch.allclose(actual_logits, native_logits, atol=native_atol, rtol=native_rtol):
            raise ValueError('Default FP64 afterW formula differs from native logits beyond FP32 rounding')
        actual = _readout(actual_logits, labels, receivers)
        pre_logits = 30. * F.linear(unit,
            F.normalize(weight.detach().double(), dim=1, eps=EPS))
        pre_w = _readout(pre_logits, labels, receivers)
        summary = dict(role='L_s', count=180, geometry=geometry(z, labels, receivers, days),
            ACTUAL=actual, PRE_W=pre_w,
            W_change_norm=float((target[weight_name].double() - weight.detach().double()).norm()),
            gain_delta_norms={name: float((target[name].double() - named[name].detach().double()).norm())
                              for name in gain_names},
            W_parameter_name=weight_name, gain_parameter_names=gain_names,
            optimizer_parameters_materialized=OPTIMIZER_PARAMETER_COUNT,
            same_features_both_readouts=True, PRE_W_replaces_only_W=True,
            after_fusion_gains_retained=True, feature='actual_raw_joint160',
            ACTUAL_readout='same original default formula; afterW/scale30/eps1e-4/FP64',
            PRE_W_readout='pre-step W on identical after joint; scale30/eps1e-4/FP64',
            native_default_logits_used_for_path=True,
            native_default_logits_match_FP32_tolerance=True,
            native_default_logits_max_abs_FP32_rounding_error=native_error,
            native_default_logits_FP32_atol=native_atol, native_default_logits_FP32_rtol=native_rtol,
            scale=30., eps=EPS, IQ_forward_count=1, network_IQ_forwards=180,
            new_head_fit_updates=0, head_fit_updates=0, actual_optimizer_updates=0,
            source_provenance='asserted_by_caller', target_access=False)
        del temporary
    if (any(parameter._version != version or not torch.equal(parameter.detach(), value)
            or parameter.grad is not grad or (grad is not None and not torch.equal(grad, grad_value))
            for parameter, value, version, grad, grad_value in parameters)
            or any(buffer._version != version or not torch.equal(buffer, value) for buffer, value, version in buffers)
            or any(module.training != flag for module, flag in flags)
            or any(getattr(module, key) is not value for module, key, value in caches)
            or any(dict(module._forward_hooks) != forward or dict(module._forward_pre_hooks) != before
                   for module, forward, before in hooks)):
        raise RuntimeError('Temporary continuation evaluation changed original state/flags/gradients/caches/hooks')
    summary['live_parameters_buffers_gradients_flags_caches_hooks_unchanged'] = True
    return z, summary
