"""Actual-model, explicitly scoped higher-order source diagnostics.

``audit_core90_field`` includes the complete current StepContext objective.
The older ``audit_local_field`` remains a clearly named CE-only comparison.
GRL's custom backward is never used to construct either field's derivatives.
Implicit response is a conditional final-linear-layer approximation and an
explicit separate parameter correction; optimizer moments are not advanced.
"""
from copy import deepcopy
import math
import time
import types

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call
from torch.overrides import TorchFunctionMode

from .jacobian_audit import local_jacobian_audit
from .response_tracking import implicit_head_response
from .source_audit import isolated_rng


def _clone_tree(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        # torch.return_types values are iterable but not ordinary namedtuples.
        if hasattr(value, '_fields'):
            return type(value)(*(_clone_tree(v) for v in value))
        if type(value) is not tuple:
            try:
                return type(value)(tuple(_clone_tree(v) for v in value))
            except TypeError:
                pass
        return tuple(_clone_tree(v) for v in value)
    if isinstance(value, list):
        return [_clone_tree(v) for v in value]
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    return value


class _StopGradientTape(TorchFunctionMode):
    """Give stop-gradient targets a separate differentiable parameter source.

    For L(a, stopgrad(t(a))), record t(b) on a reference graph, then replay it
    into L(a, t(b)). Differentiate only with respect to a, and finally evaluate
    a=b=w. The resulting first field is unchanged, while its outer derivative
    includes the response dt(w)/dw that a naive detached Hessian would lose.

    Reference no_grad computations are also replayed as outer-graph targets.
    The mode is local/thread-scoped: no global loss/model function is patched.
    Discrete selections retain their ordinary piecewise autograd semantics.
    """
    def __init__(self, entries=None):
        super().__init__()
        self.record = entries is None
        self.entries = [] if entries is None else entries
        self.index = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        name = getattr(func, '__name__', str(func))
        if name in {'_set_grad_enabled', 'is_grad_enabled'}:
            return func(*args, **kwargs)
        if name in {'__float__', '__int__', 'item', 'tolist'} and args and torch.is_tensor(args[0]):
            # Python logging/selection values are nondifferentiable in the
            # original program too. Real detach here avoids spurious warnings.
            return func(args[0].detach(), *args[1:], **kwargs)
        if name == 'detach_':
            raise NotImplementedError('in-place detach has no supported dynamic-target replay')
        stopped = name == 'detach' or not torch.is_grad_enabled()
        if not stopped:
            return func(*args, **kwargs)
        if self.record:
            with torch.enable_grad():
                value = args[0] if name == 'detach' else func(*args, **kwargs)
            self.entries.append((func, value))
            return value
        if self.index >= len(self.entries) or self.entries[self.index][0] is not func:
            raise RuntimeError('stop-gradient tape structure changed between reference and active objective')
        value = self.entries[self.index][1]
        self.index += 1
        # Execute the original operation too: a no_grad random draw must consume
        # the same stream, and local in-place bookkeeping must still happen.
        actual = args[0] if name == 'detach' else func(*args, **kwargs)
        if torch.is_tensor(actual) and torch.is_tensor(value):
            if actual.shape != value.shape or actual.dtype != value.dtype or not torch.allclose(actual, value, rtol=1e-6, atol=1e-7, equal_nan=True):
                raise RuntimeError('stop-gradient target value changed under stochastic/reference replay')
        with torch.enable_grad():
            return _clone_tree(value)


def dynamic_stopgrad_gradient(objective, point, *, seed=0):
    """Differentiate an ordinary scalar while retaining dynamic target response.

    This helper is useful for independent verification of detached thresholds,
    virtual examples, group weights and FISHR targets. External frozen objects
    such as prototype memory or EMA are constants and stay constants.
    """
    reference = point.clone()
    active = point.clone()
    tape = _StopGradientTape()
    with isolated_rng(seed), tape:
        objective(reference)
    replay = _StopGradientTape(tape.entries)
    with isolated_rng(seed), replay:
        loss = objective(active)
    if replay.index != len(tape.entries):
        raise RuntimeError('unused stop-gradient entries: objective control flow changed')
    gradient = torch.autograd.grad(loss, active, create_graph=True, allow_unused=True)[0]
    return active * 0 if gradient is None else gradient


def _ordinary_model_copy(model):
    """Bypass legacy GRL on private bound forwards without a second head pass.

    Re-evaluating a dropout head after the normal model forward would consume a
    different mask and change the stochastic objective. A copied function's
    private globals substitute the identity forward instead; original model
    methods and their module globals remain untouched.
    """
    local = deepcopy(model)
    patched = 0
    for module in local.modules():
        forward = getattr(module.forward, '__func__', None)
        if forward is None or 'grad_reverse' not in forward.__code__.co_names:
            continue
        if 'grad_reverse' not in forward.__globals__:
            raise NotImplementedError('GRL method does not expose the supported legacy global')
        namespace = dict(forward.__globals__)
        namespace['grad_reverse'] = lambda x, lambd=1.: x
        ordinary = types.FunctionType(forward.__code__, namespace, forward.__name__,
                                      forward.__defaults__, forward.__closure__)
        ordinary.__kwdefaults__ = forward.__kwdefaults__
        module.forward = types.MethodType(ordinary, module)
        patched += 1
    if not patched:
        raise NotImplementedError('No supported legacy grad_reverse call was found; ordinary forward unverified')
    return local


def build_core90_field(model, context, args, proto_bank, *, seed=0, parameter_names=None):
    """Full CORE90 signed field restricted to a declared local parameter block.

    Includes every labeled_terms output at its current stage weight, clean and
    satellite joint-forward semantics, and fixed selected U_s CE/entropy. Model
    training/eval flags are preserved on an isolated copy; buffers and RNG are
    replayed for every evaluation. No optimizer, teacher or prototype advances.
    """
    from .legacy.objective import labeled_terms
    from .legacy.options import _stage_gate_scale
    from .step_context import slice_batch

    local, fixed = _ordinary_model_copy(model), deepcopy(context)
    fixed_args, frozen_proto = deepcopy(args), deepcopy(proto_bank)
    parameters = {name: parameter.detach().clone() for name, parameter in local.named_parameters()}
    buffers = {name: buffer.detach().clone() for name, buffer in local.named_buffers()}
    head_names = [name for name in parameters if name.startswith('adv_head.')]
    names = list(parameter_names) if parameter_names is not None else [
        'id_backbone.fuse.0.weight', 'id_backbone.fuse.0.bias', *head_names]
    if not names or len(set(names)) != len(names) or any(name not in parameters for name in names):
        raise ValueError('declared local CORE90 parameter block is missing or duplicated')
    if not head_names or not all(name in names for name in head_names):
        raise ValueError('local game field must include the complete adversarial head block')
    encoder_names = [name for name in names if name not in head_names]
    if not encoder_names:
        raise ValueError('local game field needs an encoder/shared parameter block')
    # Physical ownership, including aliases, must be unambiguous.
    aliases = dict(local.named_parameters(remove_duplicate=False))
    if {id(aliases[n]) for n in encoder_names} & {id(aliases[n]) for n in head_names}:
        raise ValueError('encoder/head ownership overlaps through shared parameter aliases')
    point = torch.cat([parameters[name].reshape(-1) for name in names])
    counters = dict(model_forwards=0, head_forwards=0, field_evaluations=0,
                    reference_objective_evaluations=0, stop_gradient_targets=0)
    metadata = dict(loss_terms=[], effective_weights={}, stage_epoch=int(fixed.epoch),
                    stop_gradient_semantics='two_variable_reference_target_response',
                    stochastic_semantics='fixed_context_replayed_rng_and_original_buffers',
                    model_mode='preserved_per_module',
                    derivative_scope='piecewise_local_autograd_with_dynamic_detached_targets')
    stage_scales = {}
    for term, prefix in (('open_world_feat', 'ow_feat'), ('zid_compact', 'zid_compact'),
                         ('proxy_unknown', 'proxy_unknown'), ('source_episode', 'source_episode')):
        stage_scales[term] = _stage_gate_scale(fixed.epoch,
            start_epoch=int(getattr(fixed_args, prefix+'_start_epoch', 1)),
            warmup_epochs=int(getattr(fixed_args, prefix+'_warmup_epochs', 0)))
    mix_start = int(fixed_args.soft_unknown_mixup_start_epoch)
    mix_warmup = int(fixed_args.soft_unknown_mixup_warmup_epochs)
    stage_scales['soft_unknown_mixup'] = _stage_gate_scale(fixed.epoch,
        start_epoch=mix_start if mix_start > 0 else fixed_args.proxy_unknown_start_epoch,
        warmup_epochs=mix_warmup if mix_warmup >= 0 else fixed_args.proxy_unknown_warmup_epochs)
    stage_scales['sat_cls'] = float(fixed.epoch >= fixed_args.sat_cons_start_epoch)
    metadata['internal_stage_scales'] = stage_scales
    metadata['prototype_snapshot_available'] = frozen_proto is not None
    metadata['requested_loss_terms'] = ['tx','dom','adv','orth','cons','group_ce','fishr','proto',
        'open_world_feat','zid_compact','proxy_unknown','soft_unknown_mixup','source_episode','sat_cls']
    if fixed.strong is not None:
        metadata['requested_loss_terms'] += ['unlabeled_ce','unlabeled_entropy']
    regions, region_calls = {}, {}
    origin_regions = None
    metadata['relu_region_changes_per_field'] = []

    def region_hook(name):
        def capture(module, inputs, output):
            if torch.is_tensor(output) and output.requires_grad:
                call = region_calls.get(name, 0)
                region_calls[name] = call + 1
                regions[(name, call)] = (output.detach() > 0).cpu().clone()
        return capture

    for name, module in local.named_modules():
        if isinstance(module, nn.ReLU):
            module.register_forward_hook(region_hook(name))

    def losses(flat, *, frozen_mask=True):
        regions.clear()
        region_calls.clear()
        selected = _unflatten(flat, names, parameters)
        call_parameters = {**parameters, **selected}
        call_buffers = {key: value.clone() for key, value in buffers.items()}
        proto = deepcopy(frozen_proto)
        n = len(fixed.y)
        combined = functional_call(local, (call_parameters, call_buffers),
            (torch.cat((fixed.x, fixed.satellite)),),
            dict(y_tx=torch.cat((fixed.y, fixed.y)), grl_lambda=1., return_aux=True,
                 domain_labels=torch.cat((fixed.domain, fixed.domain))))
        counters['model_forwards'] += 1
        out = slice_batch(combined, 0, n, 2*n)
        satellite = slice_batch(combined, n, 2*n, 2*n)
        # Private copied forwards use an ordinary identity at the GRL site.
        # Therefore the original single head call/mask is preserved exactly.
        counters['head_forwards'] += 1
        weights = dict(fixed.weights)
        encoder_scale = float(weights['adv'])
        if fixed_args.game_head_scale == 'separate_head_scale':
            weights['adv'] = 1. if encoder_scale > 0 else 0.
        positive, terms = labeled_terms(out, fixed.y, fixed.domain, fixed_args,
                                        fixed.epoch, fixed.batch_index, weights, proto)
        satellite_ce = F.cross_entropy(satellite['tx_logits'], fixed.y) if fixed.epoch >= fixed_args.sat_cons_start_epoch else out['tx_logits'].sum()*0.
        positive = positive + float(fixed.weights['sat_cls']) * satellite_ce
        terms['sat_cls'] = satellite_ce
        effective = {'tx': 1., **weights}
        if fixed.epoch < fixed_args.sat_cons_start_epoch:
            effective['sat_cls'] = 0.
        if fixed.strong is not None:
            strong = functional_call(local, (call_parameters, call_buffers), (fixed.strong,), dict(return_aux=True))
            counters['model_forwards'] += 1
            counters['head_forwards'] += 1
            if not frozen_mask:
                agreement = strong['tx_logits'].detach().argmax(1) == fixed.pseudo if fixed_args.pseudo_strong_agreement else torch.ones_like(fixed.base_mask)
                fixed.strong_mask = (fixed.base_mask & agreement).detach().clone()
            if fixed.strong_mask is None:
                raise ValueError('pseudo selection must be frozen at the original context')
            mask = fixed.strong_mask
            pseudo_ce = F.cross_entropy(strong['tx_logits'][mask], fixed.pseudo[mask]) if mask.any() else strong['tx_logits'].sum()*0.
            probability = strong['tx_logits'].softmax(1)
            entropy = -(probability * probability.clamp_min(1e-8).log()).sum(1).mean()
            positive = positive + fixed_args.lambda_u*pseudo_ce + fixed_args.lambda_ent*entropy
            terms.update(unlabeled_ce=pseudo_ce, unlabeled_entropy=entropy)
            effective.update(unlabeled_ce=float(fixed_args.lambda_u), unlabeled_entropy=float(fixed_args.lambda_ent))
        metadata['loss_terms'] = list(terms)
        metadata['effective_weights'] = {key: float(effective.get(key, 0.))*stage_scales.get(key,1.) for key in terms}
        metadata['active_weight_terms'] = [key for key, weight in metadata['effective_weights'].items() if weight != 0]
        metadata['encoder_adversarial_scale'] = encoder_scale
        metadata['head_adversarial_scale'] = float(weights['adv'])
        if any(not torch.isfinite(value).all() for value in terms.values()):
            bad = [key for key, value in terms.items() if not torch.isfinite(value).all()]
            raise ValueError('nonfinite CORE90 objective terms: ' + ','.join(bad))
        metadata['stage_scaled_term_values'] = {key: float(value.detach()) for key, value in terms.items()}
        metadata['weighted_contributions'] = {key: float(effective.get(key,0.))*float(value.detach()) for key,value in terms.items()}
        negative = positive - (float(weights['adv']) + encoder_scale) * terms['adv']
        return negative, positive, selected

    # Strong agreement belongs to the original problem, not each JVP probe.
    if fixed.strong is not None and fixed.strong_mask is None:
        with isolated_rng(seed), torch.no_grad():
            losses(point, frozen_mask=False)
        metadata['pseudo_mask_origin_initialized'] = True
    else:
        metadata['pseudo_mask_origin_initialized'] = False

    def field(flat):
        nonlocal origin_regions
        reference, active = flat.clone(), flat.clone()
        tape = _StopGradientTape()
        with isolated_rng(seed), tape:
            losses(reference)
        counters['reference_objective_evaluations'] += 1
        replay = _StopGradientTape(tape.entries)
        with isolated_rng(seed), replay:
            negative, positive, selected = losses(active)
        if replay.index != len(tape.entries):
            raise RuntimeError('CORE90 stop-gradient tape length changed')
        encoder = tuple(selected[name] for name in encoder_names)
        heads = tuple(selected[name] for name in head_names)
        encoder_grad = torch.autograd.grad(negative, encoder, create_graph=True, retain_graph=True, allow_unused=True)
        head_grad = torch.autograd.grad(positive, heads, create_graph=True, retain_graph=True, allow_unused=True)
        gradients = dict(zip(encoder_names + head_names, encoder_grad + head_grad))
        counters['field_evaluations'] += 1
        counters['stop_gradient_targets'] = len(tape.entries)
        if origin_regions is None:
            origin_regions = {key: value.clone() for key, value in regions.items()}
            metadata['origin_stage_scaled_term_values'] = dict(metadata['stage_scaled_term_values'])
            metadata['origin_weighted_contributions'] = dict(metadata['weighted_contributions'])
        if origin_regions.keys() != regions.keys():
            raise RuntimeError('differentiable ReLU region layout changed during local field audit')
        metadata['relu_region_changes_per_field'].append(sum(int((value != origin_regions[key]).sum()) for key,value in regions.items()))
        return torch.cat([(selected[name] * 0. if gradients[name] is None else gradients[name]).reshape(-1) for name in names])

    return field, point, names, counters, metadata


def audit_core90_field(model, context, args, proto_bank, *, seed=0, parameter_names=None,
                       finite_difference_step=1e-3, finite_difference_tolerance=.1,
                       finite_difference_retries=2):
    """Full active-objective R14 audit; unsupported derivatives stay invalid."""
    started = time.perf_counter()
    metrics = dict(valid=False, scope='selected_parameter_block_full_CORE90_current_objective',
                   field_kind='explicit_signed_no_GRL', direction_seed=seed,
                   loss_terms=[], excluded_loss_terms=[], model_forwards=0,
                   head_forwards=0, field_evaluations=0)
    counters, metadata = {}, {}
    try:
        with isolated_rng(seed):
            field, point, names, counters, metadata = build_core90_field(
                model, context, args, proto_bank, seed=seed, parameter_names=parameter_names)
            result = local_jacobian_audit(field, point, seed=seed,
                                          finite_difference_step=finite_difference_step)
            error = result.finite_difference_relative_error
            trials = [dict(step=finite_difference_step, relative_error=error,
                           relu_region_changes=sum(metadata['relu_region_changes_per_field'][-2:]))]
            def trial_valid(trial):
                value = trial['relative_error']
                return value is not None and math.isfinite(value) and value <= finite_difference_tolerance and trial['relu_region_changes'] == 0
            # A finite difference through a ReLU switch is not the same local
            # Jacobian. Use a bounded, declared shrinking sequence, logging all
            # failures and extra evaluations; never loosen the error tolerance.
            if not isinstance(finite_difference_retries, int) or not 0 <= finite_difference_retries <= 2:
                raise ValueError('finite difference refinement budget must be 0, 1 or 2')
            generator = torch.Generator(device=point.device).manual_seed(seed)
            direction = torch.randn(point.shape, dtype=point.dtype, device=point.device, generator=generator)
            direction = direction/direction.norm()
            origin = point.detach().requires_grad_(True)
            for attempt in range(finite_difference_retries):
                if trial_valid(trials[-1]):
                    break
                h = finite_difference_step * (.25 ** (attempt+1))
                plus, minus = field(origin+h*direction), field(origin-h*direction)
                fd = (plus-minus)/(2*h)
                error = float(((fd-result.jvp).norm()/(result.jvp.norm()+1e-12)).detach())
                trials.append(dict(step=h,relative_error=error,
                                   relu_region_changes=sum(metadata['relu_region_changes_per_field'][-2:])))
        finite_difference_ok = trial_valid(trials[-1])
        valid = bool(result.valid and finite_difference_ok)
        metrics.update(valid=valid, reason='' if valid else 'nonfinite_zero_or_finite_difference_mismatch',
                       rotation_ratio=result.rotation_ratio, jvp_norm=result.jvp_norm,
                       vjp_norm=result.vjp_norm, parameter_norm=result.parameter_norm,
                       parameter_names=names, parameter_count=point.numel(),
                       finite_difference_relative_error=error,
                       finite_difference_step=trials[-1]['step'],
                       finite_difference_trials=trials,
                       finite_difference_tolerance=finite_difference_tolerance)
    except (RuntimeError, ValueError, KeyError, NotImplementedError, TypeError) as error:
        metrics.update(reason='full_CORE90_higher_order_audit_failed',
                       error_type=type(error).__name__, error=str(error))
    metrics.update(metadata)
    if 'origin_stage_scaled_term_values' in metadata:
        metrics['stage_scaled_term_values'] = metadata['origin_stage_scaled_term_values']
        metrics['weighted_contributions'] = metadata['origin_weighted_contributions']
    metrics.update(counters)
    metrics['elapsed_seconds'] = time.perf_counter() - started
    return metrics


def fishr_proxy_higher_order(logits, y, d, *, min_domains=2):
    """Legacy logit-gradient variance proxy with a differentiable mean.

    The sum of centered residuals is zero, so removing mean.detach() preserves
    its scalar value and first derivative, while making higher derivatives
    match finite differences of that first-derivative field. Keep FP64 inputs
    in FP64 for derivative checks; lower-precision inputs use legacy FP32 math.
    """
    if d is None or logits.size(0) <= 1:
        return logits.sum() * 0.
    domains = d.reshape(-1).long()
    if len(domains) != len(logits) or len(y) != len(logits):
        raise ValueError('Fishr label/sample length mismatch')
    valid = domains >= 0
    probability = F.softmax(logits if logits.dtype == torch.float64 else logits.float(), dim=1)
    proxy = probability - F.one_hot(y.reshape(-1).long(), logits.size(1)).to(probability)
    variances = [proxy[valid & (domains == domain)].var(dim=0, unbiased=False)
                 for domain in domains[valid].unique()
                 if int((valid & (domains == domain)).sum()) > 1]
    if len(variances) < max(2, int(min_domains)):
        return logits.sum() * 0.
    values = torch.stack(variances)
    return (values - values.mean(0, keepdim=True)).square().mean()


def _unflatten(flat, names, parameters):
    result, offset = {}, 0
    for name in names:
        parameter = parameters[name]
        result[name] = flat[offset:offset + parameter.numel()].view_as(parameter)
        offset += parameter.numel()
    if offset != flat.numel():
        raise ValueError('selected parameter dimension mismatch')
    return result


def build_local_ce_field(model, x, y, domain, adv_weight):
    """Build a differentiable flat field on an isolated eval-mode model copy.

    Returns (callable, point, parameter_names, counters). Useful for analytic
    verification; callers should use audit_local_field for RNG isolation.
    """
    if not math.isfinite(adv_weight) or adv_weight < 0:
        raise ValueError('adversarial weight must be finite and nonnegative')
    local = deepcopy(model).eval()
    parameters = {name: parameter.detach().clone() for name, parameter in local.named_parameters()}
    buffers = {name: buffer.detach().clone() for name, buffer in local.named_buffers()}
    encoder_names = ['id_backbone.fuse.0.weight', 'id_backbone.fuse.0.bias']
    if not all(name in parameters for name in encoder_names):
        raise ValueError('required local identity fuse weight/bias not present')
    head_names = [name for name in parameters if name.startswith('adv_head.')]
    if not head_names:
        raise ValueError('adversarial head parameter block not present')
    names = encoder_names + head_names
    point = torch.cat([parameters[name].reshape(-1) for name in names])
    counters = dict(model_forwards=0, head_forwards=0, field_evaluations=0)

    def field(flat):
        selected = _unflatten(flat, names, parameters)
        call_parameters = {**parameters, **selected}
        out = functional_call(local, (call_parameters, {k: v.clone() for k, v in buffers.items()}),
                              (x,), {'return_aux': True})
        counters['model_forwards'] += 1
        if not isinstance(out, dict) or 'z_id' not in out or 'tx_logits' not in out:
            raise ValueError('model must expose tx_logits and z_id via return_aux=True')
        head_parameters = {name[len('adv_head.'):]: value for name, value in call_parameters.items()
                           if name.startswith('adv_head.')}
        head_buffers = {name[len('adv_head.'):]: value.clone() for name, value in buffers.items()
                        if name.startswith('adv_head.')}
        # The built-in adv_dom_logits / GRL output is deliberately unused.
        rx_logits = functional_call(local.adv_head, (head_parameters, head_buffers), (out['z_id'],))
        counters['head_forwards'] += 1
        tx_loss, rx_loss = F.cross_entropy(out['tx_logits'], y.long()), F.cross_entropy(rx_logits, domain.long())
        enc = tuple(selected[name] for name in encoder_names)
        heads = tuple(selected[name] for name in head_names)
        enc_grad = torch.autograd.grad(tx_loss - adv_weight * rx_loss, enc, create_graph=True,
                                       retain_graph=True, allow_unused=True)
        head_grad = torch.autograd.grad(adv_weight * rx_loss, heads, create_graph=True,
                                        retain_graph=True, allow_unused=True)
        counters['field_evaluations'] += 1
        return torch.cat([(g if g is not None else p * 0.).reshape(-1)
                          for g, p in zip(enc_grad + head_grad, enc + heads)])

    return field, point, names, counters


def audit_local_field(model, x, y, domain, adv_weight, seed=0):
    """Low-frequency actual-model JVP/VJP audit; unsupported ops stay invalid."""
    started = time.perf_counter()
    metrics = dict(valid=False, scope='selected_identity_fuse_and_adv_head_TX_CE_RX_CE_only',
                   loss_terms=['ordinary_TX_CE', 'ordinary_RX_CE'],
                   excludes='all_CORE90_auxiliary_losses_including_FISHR',
                   field_kind='explicit_signed_no_GRL', direction_seed=seed,
                   model_forwards=0, head_forwards=0, field_evaluations=0)
    counters = {}
    try:
        with isolated_rng(seed):
            field, point, names, counters = build_local_ce_field(model, x, y, domain, adv_weight)
            result = local_jacobian_audit(field, point, seed=seed)
        metrics.update(valid=result.valid, reason='' if result.valid else 'nonfinite_or_zero_local_derivative',
                       rotation_ratio=result.rotation_ratio, jvp_norm=result.jvp_norm,
                       vjp_norm=result.vjp_norm, parameter_norm=result.parameter_norm,
                       parameter_names=names, parameter_count=point.numel())
    except (RuntimeError, ValueError, KeyError, NotImplementedError, TypeError) as error:
        metrics.update(reason='local_higher_order_audit_failed', error_type=type(error).__name__, error=str(error))
    metrics.update(counters)
    metrics['elapsed_seconds'] = time.perf_counter() - started
    return metrics


def _final_linear(head):
    linear = [(name, module) for name, module in head.named_modules() if isinstance(module, nn.Linear)]
    if not linear:
        raise ValueError('adversarial head needs a final Linear layer')
    name, module = linear[-1]
    # Convexity requires that this layer itself supplies logits, with no
    # nonlinear post-transform. Check this against actual forward values below.
    return name, module


def apply_response_tracking(reference_model, current_model, x, domain, delta=None, *,
                            monitor_x, monitor_domain, damping=.1, max_iterations=10):
    """Compensate only the final RX-logit Linear layer after an actual mainstep.

    reference_model is the pre-step model. x/monitor_x must be legal source
    training fit/monitor data. Monitor is used only for candidate acceptance.
    `delta`, if supplied, must equal the observed parameter displacement.
    All solve/evaluation graphs belong to private copies; only an accepted
    final-layer candidate is copied to current_model, leaving optimizer state
    and every other parameter/buffer/train flag/gradient unchanged.
    """
    started = time.perf_counter()
    metrics = dict(valid=False, accepted=False, reason='', iterations=0, hvp_iterations=0,
                   model_forwards=0, head_forwards=0, damping=damping,
                   scope='final_linear_layer_conditional_response_approximation',
                   optimizer_moments_updated=False, separate_compensation_update=True)
    try:
        if damping <= 0 or not math.isfinite(damping):
            raise ValueError('strictly positive finite damping is required for convex-head SPD')
        with isolated_rng(0):
            reference, candidate = deepcopy(reference_model).eval(), deepcopy(current_model).eval()
            old = dict(reference.named_parameters())
            new = dict(candidate.named_parameters())
            if old.keys() != new.keys() or any(old[name].shape != new[name].shape for name in old):
                raise ValueError('reference/current parameter contract differs')
            head_name, final = _final_linear(reference.adv_head)
            current_name, candidate_final = _final_linear(candidate.adv_head)
            if head_name != current_name:
                raise ValueError('reference/current final layer differs')
            prefix = 'adv_head.' + (head_name + '.' if head_name else '')
            head_names = [prefix + name for name, _ in final.named_parameters(recurse=False)]
            encoder_names = [name for name in old if not name.startswith('adv_head.')]
            if delta is not None:
                if set(delta) - set(encoder_names):
                    raise ValueError('delta contains a non-encoder or unknown parameter')
                for name, displacement in delta.items():
                    actual = new[name].detach() - old[name].detach()
                    if displacement.shape != actual.shape or not torch.equal(displacement.to(actual), actual):
                        raise ValueError('supplied displacement differs from actual mainstep: ' + name)
            for name, parameter in old.items():
                parameter.requires_grad_(name in encoder_names or name in head_names)
            output = reference(x, return_aux=True)
            metrics['model_forwards'] += 1
            z_id = output['z_id']
            # Keep only actual z_id ancestors; dom/TX-only parameters do not
            # belong to the encoder response block, even if they took a step.
            dependencies = torch.autograd.grad(z_id.sum(), tuple(old[name] for name in encoder_names),
                                                retain_graph=True, allow_unused=True)
            encoder_names = [name for name, g in zip(encoder_names, dependencies) if g is not None]
            displacements = tuple((new[name] - old[name]).detach() for name in encoder_names)
            saved_outputs = []
            hook = final.register_forward_hook(lambda module, args, output: saved_outputs.append(output))
            try:
                logits = reference.adv_head(z_id)
                metrics['head_forwards'] += 1
            finally:
                hook.remove()
            if len(saved_outputs) != 1 or logits is not saved_outputs[0]:
                raise ValueError('final Linear must directly supply logits; convex conditional-head contract unverified')
            if not encoder_names:
                raise ValueError('no encoder parameters affect z_id')
            loss = F.cross_entropy(logits, domain.long())
            solved = implicit_head_response(loss, [old[name] for name in head_names],
                       [old[name] for name in encoder_names], displacements,
                       damping=damping, positive_definite=True, max_iterations=max_iterations)
            metrics.update(iterations=solved.iterations, hvp_iterations=solved.iterations,
                           residual_norm=solved.residual_norm, negative_curvature=solved.negative_curvature,
                           reason=solved.reason, head_parameter_names=head_names,
                           encoder_parameter_count=sum(old[name].numel() for name in encoder_names),
                           encoder_displacement_norm=float(torch.cat([d.reshape(-1) for d in displacements]).norm()))
            if solved.accepted and solved.delta is not None:
                with torch.no_grad():
                    monitor_z = candidate(monitor_x, return_aux=True)['z_id']
                    metrics['model_forwards'] += 1
                    before = F.cross_entropy(candidate.adv_head(monitor_z), monitor_domain.long())
                    metrics['head_forwards'] += 1
                    offset = 0
                    for name in head_names:
                        parameter = new[name]
                        parameter.add_(solved.delta[offset:offset + parameter.numel()].view_as(parameter))
                        offset += parameter.numel()
                    after = F.cross_entropy(candidate.adv_head(monitor_z), monitor_domain.long())
                    metrics['head_forwards'] += 1
                    finite = bool(torch.isfinite(before) and torch.isfinite(after) and torch.isfinite(solved.delta).all())
                    accept = finite and bool(after <= before)
                    metrics.update(valid=finite, accepted=accept, monitor_ce_before=float(before),
                                   monitor_ce_after=float(after), compensation_norm=float(solved.delta.norm()),
                                   reason='accepted_source_monitor_nonworsening' if accept else 'source_monitor_rejected')
                    if accept:
                        live = dict(current_model.named_parameters())
                        for name in head_names:
                            live[name].copy_(new[name])
    except (RuntimeError, ValueError, KeyError, NotImplementedError, TypeError) as error:
        metrics.update(valid=False, accepted=False, reason='response_tracking_failed',
                       error_type=type(error).__name__, error=str(error))
    metrics['elapsed_seconds'] = time.perf_counter() - started
    return metrics
