"""Read-only source counterfactuals for stochastic-forward confounding.

No optimizer, target loader, target prediction, or test-score input exists here.
The formal CLI fixes E200 provenance and reads only the original L_s cache and
frozen source actions. The small public audit_source API also accepts fixtures.
"""
import argparse
import json
import math
import random
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .common import identity_logits, margins, positive_huber


SCALE = 30.
TOLERANCE = .5
KINDS = ('linear', 'temporal')


class RNGState:
    def __init__(self):
        self.python = random.getstate()
        self.numpy = np.random.get_state()
        self.cpu = torch.get_rng_state().clone()
        self.cuda = [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []

    def restore(self):
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.cpu)
        if self.cuda:
            torch.cuda.set_rng_state_all(self.cuda)

    def matches(self):
        current = np.random.get_state()
        numpy_equal = (current[0] == self.numpy[0] and current[1].tobytes() == self.numpy[1].tobytes()
                       and current[2:] == self.numpy[2:])
        cuda = torch.cuda.get_rng_state_all() if self.cuda else []
        return (random.getstate() == self.python and numpy_equal and torch.equal(torch.get_rng_state(), self.cpu)
                and len(cuda) == len(self.cuda) and all(torch.equal(x, y) for x, y in zip(cuda, self.cuda)))


class ReadOnlyState:
    """Restore parameters, all buffers, module modes, flags, .grad and RNG."""
    def __init__(self, *models):
        self.parameters = [(p, p.detach().clone(), p.requires_grad,
                            None if p.grad is None else p.grad.detach().clone())
                           for model in models for p in model.parameters()]
        self.gradient_objects = {id(p): p.grad for model in models for p in model.parameters()}
        self.buffers = [(b, b.detach().clone()) for model in models for b in model.buffers()]
        self.parameter_versions = [(p, p._version) for model in models for p in model.parameters()]
        self.buffer_slots = [(module, name, value) for model in models for module in model.modules()
                             for name, value in module._buffers.items()]
        self.modes = [(m, m.training) for model in models for m in model.modules()]
        volatile = ('_filter_cache_key', '_filter_cache', '_norm_weight_cache_key', '_norm_weight_cache',
                    'last_mixstyle_calls', 'last_mixstyle_changed_samples')
        # These native forward caches/counters are replaced, not mutated in
        # place. Preserve their original objects, including cached Tensor graphs.
        self.forward_attributes = [(m, name, getattr(m, name)) for model in models for m in model.modules()
                                   for name in volatile if hasattr(m, name)]
        self.rng = RNGState()
        self.report = {}

    def reset_buffers(self):
        # Forward owns temporary buffer objects. Original buffers/version
        # counters and previous autograd graphs are never mutated by a new mode.
        clones = {id(buffer): value.clone() for buffer, value in self.buffers}
        for module, name, original in self.buffer_slots:
            module._buffers[name] = None if original is None else clones[id(original)]
        for module, name, original in self.forward_attributes:
            setattr(module, name, original)

    def __enter__(self):
        self.reset_buffers()
        return self

    def __exit__(self, kind, value, traceback):
        unchanged = all(torch.equal(p.detach(), saved) for p, saved, _, _ in self.parameters)
        with torch.no_grad():
            for p, saved, flag, grad in self.parameters:
                # Avoid incrementing an unchanged parameter's version counter:
                # native Sinc cache keys deliberately include those versions.
                if not torch.equal(p.detach(), saved):
                    p.copy_(saved)
                p.requires_grad_(flag)
                p.grad = self.gradient_objects[id(p)]
        for module, name, original in self.buffer_slots:
            module._buffers[name] = original
        for module, name, original in self.forward_attributes:
            setattr(module, name, original)
        for module, training in self.modes:
            module.training = training
        self.rng.restore()
        checks = dict(parameters_never_updated=unchanged,
                      parameters_restored=all(torch.equal(p.detach(), saved) for p, saved, _, _ in self.parameters),
                      buffers_restored=all(torch.equal(b, saved) for b, saved in self.buffers),
                      buffer_objects_restored=all(module._buffers[name] is original for module, name, original in self.buffer_slots),
                      parameter_versions_unchanged=all(p._version == version for p, version in self.parameter_versions),
                      forward_caches_and_counters_restored=all(getattr(m, name) is original for m, name, original in self.forward_attributes),
                      modes_restored=all(m.training == flag for m, flag in self.modes),
                      requires_grad_restored=all(p.requires_grad == flag for p, _, flag, _ in self.parameters),
                      gradients_restored=all((p.grad is None if g is None else p.grad is not None and torch.equal(p.grad, g))
                                             for p, _, _, g in self.parameters),
                      gradient_objects_restored=all(p.grad is self.gradient_objects[id(p)] for p, _, _, _ in self.parameters),
                      global_rng_restored=self.rng.matches())
        self.report.update(checks)
        if not all(checks.values()):
            raise RuntimeError('Read-only counterfactual state restoration failed: ' + repr(checks))
        return False


def iq_tensor(values, device):
    """Existing cache tensors or the established NumPy-independent buffer path."""
    if torch.is_tensor(values):
        return values.detach().to(device=device, dtype=torch.float32)
    if values.dtype.name != 'float32' or values.ndim != 3:
        raise ValueError('Expected float32 IQ array')
    return torch.frombuffer(bytearray(values.tobytes(order='C')), dtype=torch.float32).reshape(values.shape).to(device)


def margin_penalty(reference, changed, normalization='logit'):
    """Same physical tolerance; report both meanings of coordinate rescaling.

    unit_margin_fixed_huber divides margins and tolerance by30 but keeps the
    Huber breakpoint1 (thus changes loss geometry). unit_margin_scaled_huber
    also scales that breakpoint; its loss and gradient are exactly /30.
    """
    violation = reference.detach() - changed - TOLERANCE
    if normalization == 'logit':
        return positive_huber(violation)
    if normalization == 'unit_margin_fixed_huber':
        return positive_huber(violation / SCALE)
    if normalization == 'unit_margin_scaled_huber':
        return positive_huber(violation) / SCALE
    raise ValueError('Unknown margin normalization')


def _stats(values):
    v = values.detach().cpu().double().reshape(-1)
    if not len(v) or not bool(torch.isfinite(v).all()):
        raise ValueError('Empty or nonfinite counterfactual scalar observations')
    quantiles = torch.quantile(v, torch.tensor([0., .01, .05, .25, .5, .75, .95, .99, 1.], dtype=torch.double))
    return dict(count=len(v), mean=float(v.mean()), rms=float(v.square().mean().sqrt()),
                positive_fraction=float((v > 0).double().mean()),
                quantiles=dict(zip(('min', 'q01', 'q05', 'q25', 'q50', 'q75', 'q95', 'q99', 'max'), quantiles.tolist())))


def _stratified(values, y, rx):
    result = dict(all=_stats(values), by_tx={}, by_rx={}, by_tx_rx={})
    for label in sorted(set(y.tolist())):
        result['by_tx'][str(label)] = _stats(values[y == label])
    for receiver in sorted(set(rx.tolist())):
        result['by_rx'][str(receiver)] = _stats(values[rx == receiver])
    for label in sorted(set(y.tolist())):
        for receiver in sorted(set(rx.tolist())):
            selected = (y == label) & (rx == receiver)
            if bool(selected.any()):
                result['by_tx_rx'][f'{label}/{receiver}'] = _stats(values[selected])
    return result


def _gradient(loss, parameters):
    if not loss.requires_grad:
        raise ValueError('Counterfactual loss lost its identity autograd path')
    parts = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().reshape(-1)
                      for p, g in zip(parameters, parts)])


def _comparison(vector, reference):
    norm, base = float(vector.norm()), float(reference.norm())
    valid = norm > 1e-12 and base > 1e-12
    cosine = max(-1., min(1., float(torch.dot(vector, reference)) / (norm * base))) if valid else None
    return dict(norm_ratio=norm / base if base > 1e-12 else None, cosine=cosine,
                angle_degrees=math.degrees(math.acos(cosine)) if cosine is not None else None,
                zero_vector_angle_is_undefined=not valid)


def _audit_gradients(logits, y, parameters):
    u = {name: margins(value, y) for name, value in logits.items()}
    losses = dict(clean_train_ce=F.cross_entropy(logits['train_iq_1'], y),
                  clean_eval_ce=F.cross_entropy(logits['eval_iq_1'], y),
                  old_zero_action_drop=margin_penalty(u['train_iq_1'], u['train_iq_2']).mean(),
                  paired_zero_action_drop=margin_penalty(u['eval_iq_1'], u['eval_iq_2']).mean())
    comparisons = {'old_zero_action_drop': ('train_iq_1', 'train_iq_2'),
                   'paired_zero_action_drop': ('eval_iq_1', 'eval_iq_2')}
    for kind in KINDS:
        losses[kind + '_eval_ce'] = F.cross_entropy(logits['eval_' + kind], y)
        name = kind + '_eval_drop'
        comparisons[name] = ('eval_iq_1', 'eval_' + kind)
        losses[name] = margin_penalty(u['eval_iq_1'], u['eval_' + kind]).mean()
    for name, (reference, changed) in comparisons.items():
        for normalization in ('unit_margin_fixed_huber', 'unit_margin_scaled_huber'):
            losses[name + '__' + normalization] = margin_penalty(u[reference], u[changed], normalization).mean()
    vectors = {name: _gradient(loss, parameters) for name, loss in losses.items()}
    result = {}
    for name, vector in vectors.items():
        weight = .1 if 'drop' in name else .25 if name.startswith(KINDS) else 1.
        base_name = name.split('__')[0]
        result[name] = dict(loss=float(losses[name].detach()), raw_E_norm=float(vector.norm()),
                            fixed_external_weight=weight, weighted_E_norm=weight * float(vector.norm()),
                            versus_clean_train_ce=_comparison(vector, vectors['clean_train_ce']),
                            versus_clean_eval_ce=_comparison(vector, vectors['clean_eval_ce']),
                            weighted_versus_clean_eval_ce=_comparison(weight * vector, vectors['clean_eval_ce']),
                            versus_logit_coordinate_loss=_comparison(vector, vectors[base_name]))
    return result


def audit_source(model, cache, actions, seed, device, tau_y, encoder_parameters=None,
                 batch_size=128, gradient_batches=12, forward=identity_logits, progress=None):
    """Complete source-L pass; no model or global RNG mutation survives return."""
    count = len(cache['ids'])
    if (cache.get('role') != 'L_s' or count == 0 or len(set(cache['ids'])) != count
            or any(len(cache[k]) != count for k in ('x', 'y', 'rx', 'parent_margin', 'parent_correct'))):
        raise ValueError('Expected a complete unique physical source-L cache')
    if batch_size < 1 or not 0 <= gradient_batches <= 12:
        raise ValueError('Positive batch size and at most12 gradient batches required')
    labels, receivers = cache['y'].detach().cpu(), cache['rx'].detach().cpu()
    tau_y = torch.as_tensor(tau_y, dtype=torch.float32)
    if labels.min() < 0 or labels.max() >= len(tau_y):
        raise ValueError('Missing source class tolerance')
    batches = math.ceil(count / batch_size)
    k = min(gradient_batches, batches)
    gradient_indices = sorted({round(i * (batches - 1) / (k - 1)) for i in range(k)}) if k > 1 else ([0] if k else [])
    values, audits, action_bounds = {}, [], {kind: True for kind in KINDS}
    action_generators = {kind: torch.Generator().manual_seed(int(seed) + offset)
                         for kind, offset in (('linear', 701001), ('temporal', 701002))}
    models = [model] + ([actions.networks] if hasattr(actions, 'networks') else [])
    state = ReadOnlyState(*models)
    module_inventory = [dict(name=name, type=type(module).__name__, dropout_probability=getattr(module, 'p', None))
                        for name, module in model.named_modules()
                        if isinstance(module, (torch.nn.modules.dropout._DropoutNd, torch.nn.modules.batchnorm._BatchNorm))]
    started = time.perf_counter()
    with state:
        if encoder_parameters is None:
            from .common import parameter_groups
            groups, _ = parameter_groups(model)
            encoder_parameters = next(g['params'] for g in groups if g['name'] == 'encoder')
        else:
            encoder_parameters = list(encoder_parameters)
            selected = {id(p) for p in encoder_parameters}
            for p in model.parameters():
                p.requires_grad_(id(p) in selected)
        if not encoder_parameters or len({id(p) for p in encoder_parameters}) != len(encoder_parameters):
            raise ValueError('Empty or duplicate encoder parameter list')

        def remember(name, value):
            values.setdefault(name, []).append(value.detach().cpu().float().clone())

        for batch_index, start in enumerate(range(0, count, batch_size)):
            x = iq_tensor(cache['x'][start:start + batch_size], device)
            y = labels[start:start + batch_size].to(device)
            cached_margin = cache['parent_margin'][start:start + batch_size].to(device)
            parent_correct = cache['parent_correct'][start:start + batch_size].to(device)
            tau = tau_y[y.cpu()].to(device)
            augmented = {}
            for kind in KINDS:
                augmented[kind], stats = actions.transform(x, kind, action_generators[kind])
                action_bounds[kind] &= bool(stats['within_registered_bounds'] and stats['finite'])
            logits = {}
            wants_gradient = batch_index in gradient_indices
            with torch.set_grad_enabled(wants_gradient):
                for offset, mode in enumerate(('train_iq_1', 'train_iq_2', 'eval_iq_1', 'eval_iq_2', 'eval_linear', 'eval_temporal')):
                    state.reset_buffers()
                    model.train(mode.startswith('train_'))
                    # Independent private RNG seeds isolate the two stochastic
                    # forwards; eval repeats test the zero-intervention identity.
                    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(int(seed) + batch_index * 37 + 702000 + offset)
                        inputs = augmented[mode[5:]] if mode in ('eval_linear', 'eval_temporal') else x
                        logits[mode] = forward(model, inputs)
                    if not bool(torch.isfinite(logits[mode]).all()):
                        raise FloatingPointError('Nonfinite identity logits')
                    z = logits[mode]
                    remember(mode + '/margin', margins(z, y))
                    remember(mode + '/ce', F.cross_entropy(z, y, reduction='none'))
                    remember(mode + '/correct', z.argmax(1) == y)
                    remember(mode + '/true_logit', z.gather(1, y[:, None]).squeeze(1))
                    remember(mode + '/max_rival_logit', z.gather(1, y[:, None]).squeeze(1) - margins(z, y))
                    for cls in range(z.shape[1]):
                        remember(mode + '/logit_class_' + str(cls), z[:, cls])
                u = {mode: margins(z, y) for mode, z in logits.items()}
                pairs = {'old_zero_action': ('train_iq_1', 'train_iq_2'),
                         'paired_zero_action': ('eval_iq_1', 'eval_iq_2'),
                         'real_linear': ('eval_iq_1', 'eval_linear'),
                         'real_temporal': ('eval_iq_1', 'eval_temporal')}
                for name, (reference, changed) in pairs.items():
                    remember(name + '/margin_drop', u[reference] - u[changed])
                    remember(name + '/unit_margin_drop', (u[reference] - u[changed]) / SCALE)
                    remember(name + '/drop_loss', margin_penalty(u[reference], u[changed]))
                    remember(name + '/drop_loss_unit_margin_fixed_huber', margin_penalty(u[reference], u[changed], 'unit_margin_fixed_huber'))
                    remember(name + '/drop_loss_unit_margin_scaled_huber', margin_penalty(u[reference], u[changed], 'unit_margin_scaled_huber'))
                eligible = parent_correct & torch.isfinite(tau)
                keep_target = torch.minimum(cached_margin, tau).nan_to_num(0.)
                for mode in ('train_iq_1', 'eval_iq_1'):
                    remember('same_parent_keep_' + mode + '/cache_minus_margin', cached_margin - u[mode])
                    remember('same_parent_keep_' + mode + '/loss', positive_huber(keep_target - u[mode] - TOLERANCE) * eligible)
                if wants_gradient:
                    audits.append(dict(batch_index=batch_index, first_source_offset=start, count=len(y),
                                       losses=_audit_gradients(logits, y, encoder_parameters)))
            del logits, x, augmented
            if progress is not None and (batch_index == 0 or (batch_index + 1) % 10 == 0 or batch_index + 1 == batches):
                progress(dict(event='SOURCE_COUNTERFACTUAL_PROGRESS', seed=int(seed),
                              complete_samples=min(start + batch_size, count), count=count,
                              batches_done=batch_index + 1, batches=batches, target_access=False))
    # All model and RNG state is already restored before CPU aggregation.
    scalar_arrays = {name: torch.cat(parts) for name, parts in values.items()}
    if any(len(v) != count for v in scalar_arrays.values()):
        raise RuntimeError('Counterfactual modes did not cover the complete source role')
    return dict(schema='stage2_source_counterfactual_v1', status='SOURCE_COUNTERFACTUAL_AUDIT_COMPLETE',
                seed=int(seed), count=count, model_state_restored=all(state.report.values()),
                model_seed=int(seed), source_role='L_s', source_count=count, batch_size=batch_size,
                batches=batches, source_order='original prepared cache order; no replacement',
                target_access=False, optimizer_created=False, parameter_updates=0,
                action_samples_per_kind=count, action_bounds=action_bounds,
                action_random_seeds={kind: int(seed) + offset for kind, offset in (('linear', 701001), ('temporal', 701002))},
                forward_random_rule='seed + 702000 + batch_index*37 + mode_index; private fork_rng',
                logit_reference=dict(classifier_scale=SCALE, margin='true_logit - max_other_logit',
                                     tolerance_logit=TOLERANCE, normalized_tolerance=TOLERANCE / SCALE,
                                     huber_breakpoint_logit=1., physical_state='fixed E200 parent; buffers reset before every forward'),
                mode_contract=dict(train='same IQ twice, independent RNG; train-mode stochasticity and batch statistics',
                                   eval='autograd enabled at audit points; stochastic modules disabled; no no_grad shortcut for gradient points',
                                   keep='exact original min(cached parent margin,tau_y) floor and cached-correct mask; no teacher update'),
                normalized_loss_contract=dict(unit_margin_fixed_huber='Huber((margin_drop-.5)/30), breakpoint remains1; geometry changes',
                                              unit_margin_scaled_huber='Huber(margin_drop-.5)/30; threshold and breakpoint both converted; gradient scales1/30'),
                metrics={name: _stratified(v, labels, receivers) for name, v in scalar_arrays.items()},
                gradient_batch_indices=gradient_indices, gradient_audits=audits,
                gradient_scope='complete selected batch; identity encoder E only; cosine/angle undefined for zero norm; no optimizer',
                state_invariance=state.report, mode_dependent_module_inventory=module_inventory,
                elapsed_seconds=time.perf_counter() - started,
                limitations=['Train/eval differences include every mode-dependent module, not automatically dropout alone.',
                             'Old zero-action measures stochastic forward confounding without changing IQ.',
                             'Real L/T eval comparisons remove train-mode randomness but are bounded source actions, not target performance.',
                             'Full source scalar distributions are descriptive; repeated stochastic forwards are not independent model seeds.',
                             'Gradient diagnostics cover at most12 fixed batches, not the entire source gradient distribution.'])


def run_seed(seed, output, device='cuda:0'):
    from . import design as d
    from .common import verify_parent, build_model
    from .actions import load_actions
    from experiments.cvs_phase1_overlay.model import native_modules
    from experiments.cvs_phase1_stack.recover import verify_contract
    from experiments.cvs_equivariant_identity.precision import numerical_context, actual_flags
    if seed not in d.SEEDS:
        raise ValueError('Only the original three registered parent seeds are allowed')
    output = Path(output)
    if output.exists() or output.suffix != '.json':
        raise FileExistsError('Use a new .json diagnostic output; never overwrite evidence')
    c = d.config(next(r for r in d.rows() if r['arm'] == 'C0' and r['model_seed'] == seed))
    prepared = Path(c['preparation'])
    rng = RNGState()
    print(json.dumps(dict(event='SOURCE_COUNTERFACTUAL_START', seed=seed, count=6300,
                          checkpoint=c['parent_checkpoint'], target_access=False)), flush=True)
    try:
        native_modules()  # Before deserializing historical SatViewStage objects.
        provenance = verify_parent(c)
        done = d.read(prepared / 'preparation_complete.json')
        if (done['status'] != 'PREPARATION_COMPLETE' or done['seed'] != seed or done['target_access']
                or done['config'] != c or not done['strict_parent_loaded_before_auxiliary']):
            raise ValueError('Source preparation differs from the fixed parent seed')
        packed = torch.load(prepared / 'source_cache.pt', map_location='cpu', weights_only=False)
        verify_contract(packed['contract'], d.read(d.SOURCE))
        cache = packed['labeled']
        if len(cache['ids']) != 6300 or sorted(cache['ids']) != sorted(packed['contract']['role_ids']['L_s']):
            raise ValueError('Counterfactual source physical IDs differ')
        del packed  # Existing V cache is never forwarded or used for fitting.
        payload = torch.load(prepared / 'actions.pt', map_location='cpu', weights_only=False)
        if payload['seed'] != seed:
            raise ValueError('Frozen actions belong to another model seed')
        diagnostics = d.read(prepared / 'receiver_diagnostics.json')
        with numerical_context(d.FULL_FP32_POLICY):
            model = build_model(c, device)
            head = model.id_backbone.encoder.core.id_backbone.cls_head
            if head.scale != SCALE or torch.is_tensor(head.scale):
                raise ValueError('Expected original fixed scale30 cosine head')
            actions = load_actions(payload, device)
            result = audit_source(model, cache, actions, seed, device, diagnostics['tau_y'],
                                  progress=lambda event: print(json.dumps(event), flush=True))
            result.update(parent_provenance=provenance, parent_checkpoint=c['parent_checkpoint'],
                          source_inputs=[str(prepared / name) for name in ('source_cache.pt', 'actions.pt', 'receiver_diagnostics.json', 'preparation_complete.json')],
                          hardware=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU',
                          backend_flags=actual_flags())
    finally:
        rng.restore()
    result['entry_rng_restored'] = rng.matches()
    if not result['entry_rng_restored']:
        raise RuntimeError('CLI source audit changed entry RNG state')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], seed=seed, source_count=result['source_count'],
                          output=str(output), state_invariance=result['state_invariance'], target_access=False)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_seed(args.seed, args.output, args.device)


if __name__ == '__main__':
    main()
