"""Fixed-readout materialization and actual FP32 parameter-addition risks."""
import copy
import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from experiments.cvs_evidence_recomposition20.capture import forward as capture
from experiments.cvs_sigtime_heldcredit.held_credit import CACHE_ATTRIBUTES, fresh_parameter_caches
from experiments.cvs_u_schedule50.counterfactual import _preserve_rng

SOURCE_RX = (1, 3, 4, 6, 8)
ENCODER = 'id_backbone.encoder'
HEAD = ENCODER + '.core.id_backbone.cls_head'
EPS = 1e-4


def _finite(value, shape, name, *, dtype=None, device=None):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.shape != shape or not value.is_floating_point()
            or (dtype is not None and value.dtype != dtype)
            or (device is not None and value.device != device)
            or not bool(torch.isfinite(value).all())):
        raise ValueError('Finite floating shape/dtype/device mismatch: ' + name)


def protected_names(model):
    """The actual default W and PA/reference/SIG gains, in native traversal order."""
    if not isinstance(model, nn.Module):
        raise ValueError('Existing native identity model required')
    try:
        encoder = model.get_submodule(ENCODER)
        head = model.get_submodule(HEAD)
        objects = [head.weight, head.gain, encoder.response_gain, encoder.sequence_pool.gain_raw]
    except (AttributeError, ValueError) as error:
        raise ValueError('Original default head and three fusion gains required') from error
    for i, value in enumerate(objects):
        if not isinstance(value, nn.Parameter):
            raise ValueError('Protected readout must be an actual native parameter')
        _finite(value, (6, 160) if i == 0 else (160,), 'protected readout')
    names = [name for name, value in model.named_parameters() if any(value is p for p in objects)]
    if len(names) != 4 or len({id(p) for p in objects}) != 4:
        raise ValueError('Exactly four distinct native W/gain parameters required')
    return names


def fixed_logits(model, components):
    """SELF raw joint, attached features and detached original W/three gains."""
    protected_names(model)
    required = ('identity', 'physical', 'reference_raw', 'sequence_raw', 'joint')
    if not isinstance(components, Mapping) or any(k not in components for k in required):
        raise ValueError('Real same-call four-evidence components and deployed joint required')
    identity = components['identity']
    if not isinstance(identity, torch.Tensor) or identity.ndim != 2 or identity.shape[1] != 160:
        raise ValueError('Actual components[N,160] required')
    for key in required:
        _finite(components[key], identity.shape, key, dtype=identity.dtype, device=identity.device)
    encoder = model.get_submodule(ENCODER)
    head = model.get_submodule(HEAD)
    pool = encoder.sequence_pool
    if getattr(pool, 'arm', None) != 'SIG_TIME' or float(head.scale) != 30.:
        raise ValueError('Original SIG_TIME and native scale30 required')
    for value in (head.weight, head.gain, encoder.response_gain, pool.gain_raw):
        _finite(value, value.shape, 'readout', dtype=identity.dtype, device=identity.device)
    joint = ((identity + head.gain.detach().tanh() * components['physical'])
             + encoder.response_gain.detach().tanh() * components['reference_raw'])
    joint = joint + (.2 * pool.gain_raw.detach().tanh()) * components['sequence_raw']
    if not torch.equal(joint.detach(), components['joint'].detach()):
        raise ValueError('Fixed SELF does not exactly reconstruct the actual four-term joint')
    logits = 30. * F.linear(F.normalize(joint, dim=1, eps=EPS),
                            F.normalize(head.weight.detach(), dim=1, eps=EPS))
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('Nonfinite fixed native default-head readout')
    return logits


def _bank(bank, model, role):
    if role != 'L_s' or not isinstance(bank, (tuple, list)) or len(bank) != 3:
        raise ValueError('Only visible source L_s bank (IQ, labels, RX) is permitted')
    x, y, rx = bank
    if (not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1:] != (2, 256)
            or len(x) < 5 or not x.is_floating_point() or not bool(torch.isfinite(x).all())):
        raise ValueError('Finite source IQ[N,2,256] required')
    parameter = next(model.parameters())
    if x.device != parameter.device:
        raise ValueError('Source bank must use the existing model device')
    if (not isinstance(y, torch.Tensor) or not isinstance(rx, torch.Tensor)
            or y.shape != (len(x),) or rx.shape != y.shape or y.dtype != torch.long
            or rx.dtype != torch.long or y.device != x.device or rx.device != x.device
            or not bool(((y >= 0) & (y < 6)).all()) or set(rx.tolist()) != set(SOURCE_RX)):
        raise ValueError('Visible six-class source labels and all five native source RX required')
    return x, y, rx


def evaluate(model, names, after, bank, protected, *, role='L_s'):
    """Evaluate a temporary state, restoring pre-step W/gains in that copy only.

    z is the actual FP32 raw deployment joint, never unit-normalized or inferred
    by subtraction. CE is separately calculated in FP64 with pre-step W and the
    original scale30/eps1e-4. No temporary model is returned or retained.
    """
    expected_protected = protected_names(model)
    if not isinstance(protected, Sequence) or isinstance(protected, (str, bytes)) or list(protected) != expected_protected:
        raise ValueError('Exact original protected W/gain names in native order required')
    x, y, rx = _bank(bank, model, role)
    if (not isinstance(names, Sequence) or isinstance(names, (str, bytes))
            or not names or any(not isinstance(n, str) for n in names)
            or len(set(names)) != len(names) or not isinstance(after, (tuple, list))
            or len(after) != len(names)):
        raise ValueError('Unique optimizer parameter names and same-length after state required')
    named = dict(model.named_parameters())
    for name, value in zip(names, after):
        if name not in named:
            raise ValueError('Temporary optimizer parameter is outside this native model')
        p = named[name]
        _finite(value, p.shape, name, dtype=p.dtype, device=p.device)
    before = [(p, p.detach().clone(), p._version, p.grad,
               None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()]
    buffers = [(b, b.detach().clone(), b._version) for b in model.buffers()]
    flags = [(m, m.training) for m in model.modules()]
    caches = [(m, key, getattr(m, key)) for m in model.modules()
              for key in CACHE_ATTRIBUTES if hasattr(m, key)]
    with _preserve_rng(), torch.no_grad():
        temporary = copy.deepcopy(model).eval()
        target = dict(temporary.named_parameters())
        for name, value in zip(names, after):
            target[name].copy_(value.detach())
        for name in expected_protected:
            target[name].copy_(named[name].detach())
        with fresh_parameter_caches(temporary):
            components = capture(temporary, x)
            fixed_logits(temporary, components)  # Exact same-call SELF boundary.
        z = components['joint'].detach().clone()
        weight = model.get_submodule(HEAD).weight.detach().double()
        logits = 30. * F.linear(F.normalize(z.double(), dim=1, eps=EPS),
                                F.normalize(weight, dim=1, eps=EPS))
        ce = F.cross_entropy(logits, y, reduction='none')
        if not bool(torch.isfinite(ce).all()):
            raise ValueError('Nonfinite FP64 fixed-preW source CE')
        groups = {str(r): dict(count=int((rx == r).sum()), CE=float(ce[rx == r].mean()))
                  for r in SOURCE_RX}
        summary = dict(role=role, count=len(x), RX=groups, mean_CE=float(ce.mean()),
                       by_RX={r: g['CE'] for r, g in groups.items()},
                       min_RX_CE=min(g['CE'] for g in groups.values()),
                       max_RX_CE=max(g['CE'] for g in groups.values()),
                       protected_names=expected_protected, protected_pre_step_readout=True,
                       CE_dtype='float64', feature='actual_raw_joint160', scale=30., eps=EPS,
                       network_IQ_forwards=len(x), actual_optimizer_updates=0, head_fit_updates=0,
                       source_provenance='asserted_by_caller', target_access=False)
        del temporary
    if (any(p._version != version or not torch.equal(p.detach(), value) or p.grad is not grad
            or (grad is not None and not torch.equal(grad, grad_value))
            for p, value, version, grad, grad_value in before)
            or any(b._version != version or not torch.equal(b, value) for b, value, version in buffers)
            or any(m.training != flag for m, flag in flags)
            or any(getattr(m, key) is not value for m, key, value in caches)):
        raise RuntimeError('Temporary materialization altered the original model state')
    summary['live_parameters_buffers_gradients_flags_caches_unchanged'] = True
    return z, summary


def _sum(values, name):
    result = math.fsum(float(v.detach().sum()) for v in values)
    if not math.isfinite(result):
        raise ValueError('Nonfinite FP64 scalar arithmetic: ' + name)
    return result


def actual_displacement(before_F_tuple, ideal_FP64_delta, risk_gradients_F64_by_RX):
    """Measure real FP32 additions and each row's own conversion-error bound."""
    if (not isinstance(before_F_tuple, tuple) or not before_F_tuple
            or not isinstance(ideal_FP64_delta, tuple)
            or len(before_F_tuple) != len(ideal_FP64_delta)):
        raise ValueError('Existing FP32 feature parameters and same-length FP64 delta tuples required')
    for i, (before, delta) in enumerate(zip(before_F_tuple, ideal_FP64_delta)):
        if not isinstance(before, torch.Tensor):
            raise ValueError('Existing dense FP32 parameter tensor required')
        _finite(before, before.shape, 'before_F', dtype=torch.float32)
        _finite(delta, before.shape, 'ideal delta', dtype=torch.float64, device=before.device)
    if not isinstance(risk_gradients_F64_by_RX, Mapping):
        raise ValueError('All five source RX FP64 risk-gradient tuples required')
    gradients = {}
    for key, row in risk_gradients_F64_by_RX.items():
        if type(key) is int and key in SOURCE_RX:
            receiver = key
        elif type(key) is str and key in tuple(map(str, SOURCE_RX)):
            receiver = int(key)
        else:
            raise ValueError('Only native source RX1/3/4/6/8 risk keys permitted')
        if receiver in gradients or not isinstance(row, tuple) or len(row) != len(before_F_tuple):
            raise ValueError('Unique source RX and same-length risk tuple required')
        for before, value in zip(before_F_tuple, row):
            if value is not None:
                _finite(value, before.shape, 'risk gradient', dtype=torch.float64, device=before.device)
        gradients[receiver] = row
    if set(gradients) != set(SOURCE_RX):
        raise ValueError('Complete five-RX source risk rows required')
    before = tuple(v.detach().double() for v in before_F_tuple)
    ideal = tuple(v.detach() for v in ideal_FP64_delta)
    after = tuple((theta + delta).to(dtype=original.dtype).detach()
                  for theta, delta, original in zip(before, ideal, before_F_tuple))
    if any(not bool(torch.isfinite(v).all()) for v in after):
        raise ValueError('Materialized parameter addition overflowed the original FP32 dtype')
    actual = tuple(value.double() - theta for value, theta in zip(after, before))
    error = tuple(value - desired for value, desired in zip(actual, ideal))
    rows = {}
    for receiver, gradient in gradients.items():
        values = [(g.detach(), desired, realized, rounding)
                  for g, desired, realized, rounding in zip(gradient, ideal, actual, error) if g is not None]
        ideal_dot = _sum((g * desired for g, desired, realized, rounding in values), 'ideal risk')
        actual_dot = _sum((g * realized for g, desired, realized, rounding in values), 'actual risk')
        bound = _sum((g.abs() * rounding.abs() for g, desired, realized, rounding in values), 'row error bound')
        rounding_dot = _sum((g * rounding for g, desired, realized, rounding in values), 'rounding risk')
        rows[str(receiver)] = dict(ideal_directional_risk=ideal_dot,
            actual_directional_risk=actual_dot, measured_rounding_error_bound=bound,
            rounding_error_dot=rounding_dot,
            dot_difference=actual_dot - ideal_dot,
            dot_subtraction_residual_FP64=(actual_dot - ideal_dot) - rounding_dot,
            bound_formula='sum |g_RX|*|actual_delta-ideal_delta|; per-row parameter-addition error')
    summary = dict(parameter_tensors=len(before), original_dtype='float32', calculation_dtype='float64',
                   actual_parameter_addition_measured=True, by_RX=rows,
                   actual_positive_risk_RX=[str(r) for r in SOURCE_RX if rows[str(r)]['actual_directional_risk'] > 0],
                   swallowed_nonzero_ideal_coordinates=sum(int(((d != 0) & (a == 0)).sum()) for d, a in zip(ideal, actual)),
                   ideal_nonzero_coordinates=sum(int((v != 0).sum()) for v in ideal),
                   rule='after=(before.float64+ideal_FP64_delta).to(original_FP32_dtype); actual=after.float64-before.float64',
                   global_parameter_maxabs_threshold=False, actual_optimizer_updates=0)
    return dict(after_F=after, actual_delta_FP64=actual, summary=summary)
