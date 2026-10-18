"""Separate AdamW view moments inside one existing joint optimizer.

The native loss, backward, physical clip and Sinc coordinate policy are kept.
SPLIT changes the adaptive displacement to .8*u(rest)+.2*u(As), with separate
first/second moments and bias-correction counters. This does not preserve the
effective As gradient share. MEASURE leaves the native AdamW step untouched.
Only state_dict serialization is supported; no optimizer object is pickled.
"""
import math
import time

import torch
from torch.optim.adamw import AdamW as OriginalAdamW

from experiments.cvs_rdrop_sinc_units.optimizer import QNAMES

MODES = ('MEASURE', 'SPLIT')
WEIGHTS = (.8, .2)
_PARTS = ('rest', 'as')


def _mode(value):
    if value not in MODES:
        raise ValueError('Only fixed MEASURE/SPLIT mode is permitted')
    return value


def _finite_number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError(name + ' cannot be boolean')
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError('Invalid ' + name)
    return value


def _hyperparameters(group):
    if any(group.get(k, False) for k in ('amsgrad', 'maximize', 'capturable', 'differentiable', 'fused')):
        raise ValueError('Unsupported AdamW flag')
    if group.get('foreach') not in (None, False):
        raise ValueError('Explicit foreach AdamW is unsupported')
    betas = group['betas']
    if len(betas) != 2:
        raise ValueError('Two AdamW betas required')
    b1, b2 = (float(v) for v in betas)
    if not all(math.isfinite(v) and 0 <= v < 1 for v in (b1, b2)):
        raise ValueError('Invalid AdamW betas')
    lr = _finite_number(group['lr'], 'learning rate', positive=True)
    eps = _finite_number(group['eps'], 'epsilon', positive=True)
    wd = _finite_number(group['weight_decay'], 'weight decay', nonnegative=True)
    return b1, b2, lr, eps, wd


def _clip_factor(max_norm, total_norm):
    limit = _finite_number(max_norm, 'clip limit', positive=True)
    norm = _finite_number(total_norm, 'physical gradient norm', nonnegative=True)
    if limit != 5.:
        raise ValueError('Only the existing physical global clip5 is permitted')
    return min(1., limit / (norm + 1e-6))


def _rms_comparison(error_rms, reference_rms, expected_rms, scale=1.):
    """Same tolerances, applied to one parameter's RMS instead of each entry.

    GPU re-evaluation may change nearly zero entries even in a repeated total
    VJP. Unit conversion divides BOTH the discrepancy and absolute bound;
    Sinc therefore gets exactly the same stored-coordinate acceptance bound.
    """
    error = _finite_number(error_rms, 'component RMS error', nonnegative=True)
    reference = _finite_number(reference_rms, 'actual gradient RMS', nonnegative=True)
    expected = _finite_number(expected_rms, 'component sum RMS', nonnegative=True)
    scale = _finite_number(scale, 'gradient coordinate scale', positive=True)
    bound = 5e-7 + 5e-4 * max(reference, expected)
    return dict(error_rms=error, reference_rms=reference, expected_rms=expected,
                bound=bound, error_bound_ratio=error / bound,
                coordinate_scale=scale, physical_error_rms=error / scale,
                physical_bound=bound / scale, within_bound=error <= bound)


def initialize(model, mode):
    if hasattr(model, '_sa_mode'):
        raise ValueError('One split-moment initialization per model')
    model._sa_mode = _mode(mode)
    model._sa_pending = None
    model._sa_ready = None
    model._sa_steps = model._sa_clips = model._sa_vjp_calls = 0
    model._sa_records = []
    model._sa_step_record = None
    model._sa_optimizer = None
    model._sa_last_comparison = None
    return model


def _named_parameters(model, optimizer):
    named = tuple((n, p) for n, p in model.named_parameters() if p.requires_grad)
    flat = [p for group in optimizer.param_groups for p in group['params']]
    if not named or len({id(p) for p in flat}) != len(flat):
        raise ValueError('Nonempty deduplicated optimizer parameters required')
    if {id(p) for _, p in named} != {id(p) for p in flat}:
        raise ValueError('All and only original jointly trainable parameters required')
    for group in optimizer.param_groups:
        _hyperparameters(group)
    if any(p.is_complex() or p.dtype not in (torch.float32, torch.float64) for _, p in named):
        raise ValueError('Original real float32/float64 storage coordinates required')
    return named


def _gradient_ok(p, gradient):
    if gradient is not None and (gradient.is_sparse or gradient.shape != p.shape or
                                 gradient.device != p.device or gradient.dtype != p.dtype):
        raise FloatingPointError('Incompatible dense gradient')


def _finite_tensors(values, label):
    values = [v for v in values if v is not None]
    if values and not bool(torch.stack([torch.isfinite(v).all() for v in values]).all()):
        raise FloatingPointError('Nonfinite ' + label)


def _geometry(named, rest, satellite):
    device = named[0][1].device
    if any(p.device != device for _, p in named):
        raise ValueError('One native model device required')
    rr = torch.zeros((), dtype=torch.float64, device=device)
    aa = torch.zeros_like(rr)
    ra = torch.zeros_like(rr)
    for (name, _), r, a in zip(named, rest, satellite):
        scale = 25000. if name in QNAMES else 1.
        # Units below are diagnostic only. Updates retain original storage units.
        if r is not None:
            rr.add_(r.double().square().sum() / scale ** 2)
        if a is not None:
            aa.add_(a.double().square().sum() / scale ** 2)
        if r is not None and a is not None:
            ra.add_((r.double() * a.double()).sum() / scale ** 2)
    values = torch.stack((rr, aa, ra))
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError('Nonfinite component gradient geometry')
    rr, aa, ra = (float(v) for v in values)
    return dict(rest_physical_norm=math.sqrt(rr), as_physical_norm=math.sqrt(aa),
                physical_dot=ra, physical_cosine=ra / math.sqrt(rr * aa) if rr and aa else None)


def capture(model, total_loss, Asloss, optimizer, step):
    if model._sa_optimizer is not optimizer or optimizer._sa_model is not model:
        raise ValueError('Capture must use the bound native optimizer')
    if _mode(model._sa_mode) != optimizer._sa_mode:
        raise ValueError('Optimizer mode changed after association')
    if isinstance(step, bool) or not isinstance(step, int) or \
            model._sa_pending is not None or model._sa_ready is not None or \
            step != model._sa_steps + 1 or model._sa_clips != model._sa_steps:
        raise ValueError('One ordered capture before each real update required')
    for loss in (total_loss, Asloss):
        if loss.ndim != 0 or not loss.requires_grad or not bool(torch.isfinite(loss)):
            raise ValueError('Finite graph-connected scalar losses required')
    start = time.perf_counter()
    named = _named_parameters(model, optimizer)
    params = tuple(p for _, p in named)
    if any(p.grad is not None for p in params):
        raise ValueError('Capture must follow native zero_grad and precede backward')
    rest = torch.autograd.grad(total_loss - Asloss, params, retain_graph=True, allow_unused=True)
    satellite = torch.autograd.grad(Asloss, params, retain_graph=True, allow_unused=True)
    if any(p.grad is not None for p in params):
        raise ValueError('VJP changed native gradient accumulation')
    for (_, p), r, a in zip(named, rest, satellite):
        _gradient_ok(p, r)
        _gradient_ok(p, a)
    _finite_tensors((*rest, *satellite), 'component gradient')
    rest = tuple(None if g is None else g.detach() for g in rest)
    satellite = tuple(None if g is None else g.detach() for g in satellite)
    stats = _geometry(named, rest, satellite)
    model._sa_pending = (named, rest, satellite, stats, step, time.perf_counter() - start)
    model._sa_vjp_calls += 2


def after_clip(model, parameters, max_norm, total_norm):
    optimizer = model._sa_optimizer
    pending = model._sa_pending
    if optimizer is None or pending is None or model._sa_ready is not None or \
            model._sa_clips != model._sa_steps:
        raise ValueError('One native physical clip after capture required')
    named, rest, satellite, stats, step, elapsed = pending
    params = list(parameters)
    if len(params) != len({id(p) for p in params}) or \
            {id(p) for p in params} != {id(p) for _, p in named} or \
            tuple((n, id(p)) for n, p in _named_parameters(model, optimizer)) != \
            tuple((n, id(p)) for n, p in named) or step != model._sa_steps + 1:
        raise ValueError('Stale or different original parameter set')
    factor = _clip_factor(max_norm, total_norm)
    _finite_tensors((p.grad for _, p in named), 'actual clipped total gradient')
    clipped = []
    comparisons = []
    comparison_names = []
    none_consistent = []
    for (name, p), r, a in zip(named, rest, satellite):
        _gradient_ok(p, p.grad)
        cr = None if r is None else r.mul(factor)
        ca = None if a is None else a.mul(factor)
        if p.grad is None:
            # Preserve exact None-versus-zero semantics; RMS never relaxes it.
            none_consistent.extend((g == 0).all() for g in (cr, ca) if g is not None)
        else:
            if cr is None and ca is None:
                raise ValueError('Native active parameter absent from both VJPs')
            expected = torch.zeros_like(p)
            if cr is not None:
                expected.add_(cr)
            if ca is not None:
                expected.add_(ca)
            actual64 = p.grad.detach().double()
            expected64 = expected.double()
            error64 = expected64 - actual64
            # The old pointwise test is retained solely as telemetry. All
            # reductions below inspect gradients without changing either one.
            old_failures = (~torch.isclose(expected, p.grad, rtol=5e-4, atol=5e-7)).sum()
            comparisons.append(torch.stack((error64.abs().max(), error64.square().mean().sqrt(),
                actual64.square().mean().sqrt(), expected64.square().mean().sqrt(), old_failures.double())))
            comparison_names.append(name)
        clipped.append((cr, ca))
    if none_consistent and not bool(torch.stack(none_consistent).all()):
        raise ValueError('Native backward omitted a nonzero component/active mask')
    rows = []
    values = torch.stack(comparisons).cpu().tolist() if comparisons else []
    for name, (maximum, error, reference, expected, old_failures) in zip(comparison_names, values):
        comparison = _rms_comparison(error, reference, expected, 25000. if name in QNAMES else 1.)
        rows.append(dict(parameter=name, max_abs_error=maximum,
                         old_pointwise_failure_coordinates=int(old_failures), **comparison))
    worst = max(rows, key=lambda value: value['error_bound_ratio']) if rows else None
    model._sa_last_comparison = dict(rule='per_parameter_RMS', rtol=5e-4, atol_stored=5e-7,
        parameters=len(rows), old_pointwise_nonblocking=True,
        old_pointwise_failure_coordinates=sum(value['old_pointwise_failure_coordinates'] for value in rows),
        old_pointwise_failure_parameters=sum(value['old_pointwise_failure_coordinates'] > 0 for value in rows),
        max_error_rms=max((value['error_rms'] for value in rows), default=0.),
        max_bound=max((value['bound'] for value in rows), default=5e-7),
        max_error_bound_ratio=worst['error_bound_ratio'] if worst else 0., worst_parameter=worst,
        old_pointwise_failures=[value for value in rows if value['old_pointwise_failure_coordinates']])
    if any(not value['within_bound'] for value in rows):
        raise ValueError('Component sum differs from actual clipped total gradient (parameter RMS)')
    max_error = max((value['max_abs_error'] for value in rows), default=0.)
    selected = set(QNAMES) | {'id_backbone.encoder.core.id_backbone.cls_head.weight'}
    if not any(n in selected for n, _ in named):
        selected = {named[0][0]}
    witness = [dict(parameter=n, parameter_before_step=float(p.detach().reshape(-1)[0]))
               for n, p in named if n in selected]
    versions = tuple(None if p.grad is None else (p.grad, p.grad._version) for _, p in named)
    model._sa_ready = (named, tuple(clipped), stats, step, elapsed, factor, versions, witness, max_error)
    model._sa_pending = None
    model._sa_clips = step
    model._sa_last_norm = float(total_norm)


def _branch_direction(state, part, gradient, parameter, beta1, beta2, eps):
    if gradient is None:
        return None
    prefix = '_sa_' + part + '_'
    keys = (prefix + 'step', prefix + 'exp_avg', prefix + 'exp_avg_sq')
    present = [key in state for key in keys]
    if any(present) and not all(present):
        raise ValueError('Partial split AdamW branch state')
    if not any(present):
        state[keys[0]] = 0
        state[keys[1]] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        state[keys[2]] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
    count, first, second = (state[key] for key in keys)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError('Invalid independent AdamW branch counter')
    if first is None or second is None:
        raise ValueError('Missing independent AdamW moment tensor')
    _gradient_ok(parameter, first)
    _gradient_ok(parameter, second)
    count += 1
    state[keys[0]] = count
    first.lerp_(gradient, 1 - beta1)
    second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
    denominator = second.sqrt().div_(math.sqrt(1 - beta2 ** count)).add_(eps)
    direction = first / (1 - beta1 ** count) / denominator
    return direction


@torch.no_grad()
def _split_step(optimizer, closure=None):
    if closure is not None:
        raise ValueError('Native split step does not accept a closure')
    model = optimizer._sa_model
    named, components = model._sa_ready[:2]
    by_id = {id(p): pair for (_, p), pair in zip(named, components)}
    updates = []
    for group in optimizer.param_groups:
        beta1, beta2, lr, eps, wd = _hyperparameters(group)
        for p in group['params']:
            if p.grad is None:
                continue
            state = optimizer.state[p]
            # AdamW.__setstate__ expects this native key when loading a state_dict.
            if not state:
                state['step'] = torch.tensor(0., dtype=torch.float32, device='cpu')
                state['_sa_version'] = 1
            if state.get('_sa_version') != 1 or 'exp_avg' in state or 'exp_avg_sq' in state:
                raise ValueError('SPLIT requires its own reset/compatible moment state')
            native_step = state.get('step')
            if not torch.is_tensor(native_step) or native_step.numel() != 1:
                raise ValueError('Native-compatible scalar AdamW step state required')
            count = float(native_step)
            if not math.isfinite(count) or count < 0 or not count.is_integer():
                raise ValueError('Invalid native-compatible AdamW step counter')
            state['step'].add_(1)
            update = torch.zeros_like(p)
            for part, weight, gradient in zip(_PARTS, WEIGHTS, by_id[id(p)]):
                direction = _branch_direction(state, part, gradient, p, beta1, beta2, eps)
                if direction is not None:
                    update.add_(direction, alpha=weight)
            # Existing CENTER_UNITS pre-hook already applies physical Sinc decay.
            # Its actual group wd=0 prevents any second Sinc decay here.
            updates.append((p, update, lr, wd))
    _finite_tensors((update for p, update, lr, wd in updates), 'split AdamW direction/state')
    for p, update, lr, wd in updates:
        p.mul_(1 - lr * wd)
        p.add_(update, alpha=-lr)
    return None


def attach_optimizer(optimizer, model, *, amp=False):
    if amp is not False:
        raise ValueError('This fixed experiment requires native amp=False')
    if not isinstance(optimizer, OriginalAdamW) or not hasattr(model, '_sa_mode'):
        raise ValueError('Existing joint native AdamW and initialized model required')
    if getattr(optimizer, '_sa_model', None) is not None or model._sa_optimizer is not None:
        raise ValueError('One optimizer association per model')
    named = _named_parameters(model, optimizer)
    _mode(model._sa_mode)
    if model._sa_mode == 'SPLIT':
        for _, p in named:
            state = optimizer.state.get(p)
            if state and (state.get('_sa_version') != 1 or 'exp_avg' in state or 'exp_avg_sq' in state):
                raise ValueError('Cannot infer independent moments from shared AdamW state')
        # Keep this exact instance and all already registered Sinc hooks. The
        # standard PyTorch wrapper invokes pre-hooks, this body, then post-hooks.
        split_class = type('SplitViewAdamW', (type(optimizer),), {'step': _split_step})
        optimizer.__class__ = split_class
        optimizer._patch_step_function()
    elif any('_sa_version' in optimizer.state.get(p, {}) for _, p in named):
        raise ValueError('MEASURE requires original shared AdamW state')
    optimizer._sa_model = model
    optimizer._sa_mode = model._sa_mode
    model._sa_optimizer = optimizer

    def pre(opt, args, kwargs):
        if args != (opt,) or kwargs or model._sa_ready is None or model._sa_step_record is not None:
            raise ValueError('One native closure-free step after capture/clip required')
        if _mode(model._sa_mode) != opt._sa_mode or \
                getattr(opt, 'grad_scale', None) is not None or getattr(opt, 'found_inf', None) is not None:
            raise ValueError('Fixed mode and unscaled native amp=False step required')
        named, components, stats, step, elapsed, factor, versions, witness, error = model._sa_ready
        if step != model._sa_steps + 1 or model._sa_clips != step or \
                tuple((n, id(p)) for n, p in _named_parameters(model, opt)) != \
                tuple((n, id(p)) for n, p in named):
            raise ValueError('Stale optimizer lifecycle or parameter set')
        for (_, p), saved in zip(named, versions):
            if (saved is None and p.grad is not None) or (saved is not None and
                    (p.grad is not saved[0] or p.grad._version != saved[1])):
                raise ValueError('Native gradients changed after checked physical clip')
        names = {id(p): n for n, p in named}
        snapshots = {v['parameter']: v for v in witness}
        for group in opt.param_groups:
            b1, b2, lr, eps, wd = _hyperparameters(group)
            for p in group['params']:
                if names[id(p)] in snapshots:
                    snapshots[names[id(p)]].update(lr=lr, eps=eps, weight_decay=wd,
                                                  beta1=b1, beta2=b2, native_active=p.grad is not None)
        model._sa_step_record = dict(step=step, mode=model._sa_mode, extra_VJP_calls=2,
            clip_factor=factor, physical_global_norm=model._sa_last_norm,
            capture_host_enqueue_seconds=elapsed, component_sum_max_abs_error=error,
            component_sum_comparison=model._sa_last_comparison,
            rest_active_parameters=sum(r is not None for r, a in components),
            as_active_parameters=sum(a is not None for r, a in components),
            native_active_parameters=sum(p.grad is not None for _, p in named),
            adaptive_weights=list(WEIGHTS) if model._sa_mode == 'SPLIT' else None,
            weighted_component_geometry=stats, witness=witness)

    def post(opt, args, kwargs):
        record = model._sa_step_record
        if record is None or model._sa_ready is None:
            raise ValueError('Unexpected optimizer post-step')
        lookup = dict(model._sa_ready[0])
        for value in record['witness']:
            p = lookup[value['parameter']]
            state = opt.state.get(p, {})
            value['parameter_after_step'] = float(p.detach().reshape(-1)[0])
            value['optimizer_state_step'] = float(state['step']) if state else 0.
            if model._sa_mode == 'SPLIT':
                for part in _PARTS:
                    prefix = '_sa_' + part + '_'
                    value[part + '_step'] = state.get(prefix + 'step', 0)
                    value[part + '_first'] = float(state[prefix + 'exp_avg'].reshape(-1)[0]) if prefix + 'exp_avg' in state else None
                    value[part + '_second'] = float(state[prefix + 'exp_avg_sq'].reshape(-1)[0]) if prefix + 'exp_avg_sq' in state else None
            elif state:
                value['native_first'] = float(state['exp_avg'].reshape(-1)[0])
                value['native_second'] = float(state['exp_avg_sq'].reshape(-1)[0])
        model._sa_steps += 1
        model._sa_records.append(record)
        model._sa_ready = None
        model._sa_step_record = None
        model._sa_last_comparison = None

    optimizer.register_step_pre_hook(pre)
    optimizer.register_step_post_hook(post)
    return optimizer
