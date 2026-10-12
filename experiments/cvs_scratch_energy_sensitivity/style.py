"""L-only discrete main-identity style search and same-mask replay.

Candidates are complete sequential shallow+middle forwards. Their predictions
are never spliced per packet: a single whole-batch CE selects one candidate.
"""
from contextlib import contextmanager
import math
import numbers
import weakref

import torch
from torch.nn import functional as F

from experiments.cvs_e20_cross_style.style import LEVELS, _metadata, _paths
from experiments.cvs_e20_budget_fishr.multilevel import weak_mixstyle_u


_ACTIVE = weakref.WeakSet()


def _devices(model):
    return sorted({v.device.index for v in list(model.parameters())+list(model.buffers())
                   if v.device.type == 'cuda'})


def capture_rng(model):
    """Call immediately before the ordinary clean identity forward."""
    return dict(cpu=torch.random.get_rng_state().clone(),
                cuda={device: torch.cuda.get_rng_state(device).clone() for device in _devices(model)})


def _same_rng(a, b):
    return (torch.equal(a['cpu'], b['cpu']) and a['cuda'].keys() == b['cuda'].keys()
            and all(torch.equal(a['cuda'][d], b['cuda'][d]) for d in a['cuda']))


def _set_rng(saved, model):
    if not isinstance(saved, dict) or set(saved) != {'cpu', 'cuda'} or not isinstance(saved['cuda'], dict):
        raise ValueError('clean_rng must be produced by capture_rng(model)')
    if set(saved['cuda']) != set(_devices(model)):
        raise ValueError('clean_rng CUDA devices differ from the identity model')
    current = capture_rng(model)
    for key, state, expected in [('cpu', saved['cpu'], current['cpu'])]+[
            (str(d), saved['cuda'][d], current['cuda'][d]) for d in current['cuda']]:
        if (not isinstance(state, torch.Tensor) or state.device.type != 'cpu'
                or state.dtype != torch.uint8 or state.shape != expected.shape):
            raise ValueError('Invalid clean RNG tensor: '+key)
    torch.random.set_rng_state(saved['cpu'])
    for device, state in saved['cuda'].items():
        torch.cuda.set_rng_state(state, device)


def candidate_seeds(model_seed, step):
    """Same four complete-forward candidates for every arm at a fixed step."""
    return [(int(model_seed)+int(step)*1000003+i*104729) % 2**63 for i in range(4)]


def _random_index(model_seed, step):
    generator = torch.Generator().manual_seed((int(model_seed)+int(step)*99991+81173) % 2**63)
    return int(torch.randint(4, (1,), generator=generator))


def _metrics(logits, labels):
    if logits.shape != (len(labels), 6) or not torch.isfinite(logits).all():
        raise ValueError('Expected finite six-class native L logits')
    truth = labels.to(device=logits.device)
    positive = logits.gather(1, truth[:, None]).squeeze(1)
    negative = logits.masked_fill(F.one_hot(truth, 6).bool(), -torch.inf).max(1).values
    return dict(CE=float(F.cross_entropy(logits, truth)),
                margin=float((positive-negative).mean()),
                accuracy=float((logits.argmax(1) == truth).float().mean()))


def _snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _same_state(model, initial):
    current = model.state_dict()
    return current.keys() == initial.keys() and all(torch.equal(current[k], v) for k, v in initial.items())


def _restore_changed_state(model, initial):
    # Used only to unwind a faulty eval forward. Never write unchanged parameters:
    # their version counters may already be referenced by the ordinary clean CE.
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if name in initial and not torch.equal(value, initial[name]):
                value.copy_(initial[name])


@contextmanager
def _hooks(model, labels, rx, day, ids, seed, eta):
    modules, paths = _paths(model)
    handles, levels, order = [], {}, []
    try:
        for level in LEVELS:
            def hook(module, inputs, value, level=level):
                if level in levels:
                    raise RuntimeError('Each candidate must execute exactly one whole identity forward')
                if (not isinstance(value, torch.Tensor) or value.ndim != 3 or value.is_complex()
                        or not value.is_floating_point() or len(value) != len(labels)):
                    raise ValueError('Boundary style requires real native [L batch,C,T] maps')
                layer_seed = (int(seed)+LEVELS.index(level)*1709) % 2**63
                output, record = weak_mixstyle_u(value, labels, rx, day, ids, layer_seed,
                                                 p=1., alpha=.1, eta=eta)
                donors = record.pop('donors')
                if len(donors) != len(labels):
                    raise ValueError('Missing style donor decisions')
                for i, j in enumerate(donors):
                    if not 0 <= j < len(labels):
                        raise ValueError('Style donor index outside the real L batch')
                    if i != j and (labels[i] != labels[j] or rx[i] == rx[j]
                            or day[i] != day[j] or ids[i] == ids[j]):
                        raise ValueError('Style donor violates real L TX/day/RX/PID constraints')
                record.update(seed=layer_seed, path=paths[level], shape=list(value.shape), eta=eta,
                              donor_rules_verified=True, donor_self_fallbacks=sum(i == j for i, j in enumerate(donors)),
                              input_requires_grad=value.requires_grad, output_requires_grad=output.requires_grad)
                levels[level] = record
                order.append(level)
                return output
            handles.append(modules[paths[level]].register_forward_hook(hook))
        yield levels, order
        if tuple(order) != LEVELS:
            raise RuntimeError('Candidate bypassed or reordered the two native style layers')
    finally:
        for handle in handles:
            handle.remove()


def styled_forward(model, x, y, rx, day, ids, step, config, ramp, clean_rng=None):
    """Select by dropout-free L readouts, then replay one differentiable forward.

    ``config`` provides model_seed, style_selection=random/hard/replay and
    candidates=4 with include_identity_candidate=True. Clean is the fifth choice. Pass ``capture_rng(model)`` taken before clean forward to use
    matching clean/style dropout masks. Caller owns source-role/PID provenance;
    this helper validates legal L metadata and never accesses a data provider.
    Only selected training logits retain a graph. Readout scalars are detached.
    """
    labels, receivers, days, physical_ids = _metadata(y, rx, day, ids)
    if (not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1] != 2 or len(x) != len(labels)
            or not x.is_floating_point() or x.is_complex()):
        raise ValueError('Expected native real L IQ [batch,2,time] and matching metadata')
    if config.get('source_role', 'L_s') != 'L_s':
        raise ValueError('Boundary style only accepts labeled source L_s')
    selection = config.get('style_selection')
    seed = config.get('model_seed')
    if config.get('include_identity_candidate') is not True or selection != 'hard':
        raise ValueError('Identity-inclusive run requires registered hard selection and clean candidate')
    if selection not in ('random', 'hard', 'replay') or config.get('candidates') != 4:
        raise ValueError('Expected random/hard/replay and exactly four candidates')
    if (isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or not 0 <= seed < 2**63
            or isinstance(step, bool) or not isinstance(step, numbers.Integral) or step < 0):
        raise ValueError('Invalid model seed or training step')
    eta = float(ramp)
    if not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError('Style ramp must be in [0,1]')
    if not model.training:
        raise ValueError('styled_forward is a training-only L identity operation')
    if model in _ACTIVE:
        raise RuntimeError('Nested boundary-style forwards are prohibited')
    modes = [(module, module.training) for module in model.modules()]
    before_rng = capture_rng(model)
    details = dict(role='L_style', source_role='L_s', style_selection=selection,
                   style_input_valid=True, target_access=False, V_access=False, U_access=False, U_truth=False,
                   batch_occurrences=len(labels), distinct_physical_ids=len(set(physical_ids)),
                   duplicate_occurrences=len(labels)-len(set(physical_ids)),
                   p=1., alpha=.1, eta=eta, candidates=4, candidate_seeds=[], candidate_losses=[],
                   selected_index=None, selected_seed=None, selected_levels=list(LEVELS) if selection != 'replay' else [],
                   levels={}, actual_hook_order=[], changed_packets=0,
                   changed_packets_semantics='sum over two layers; repeated batch occurrences retained',
                   eval_clean=None, eval_selected=None, eval_readouts=0, candidate_eval_forwards=0,
                   eval_forwards=0, train_forwards=0, total_extra_forwards=0,
                   candidate_state_unchanged=True, candidate_rng_unchanged=True,
                   original_training_flags_restored=False, main_rng_restored=False,
                   dropout_same_mask_requested=clean_rng is not None, normal_training_dropout=True,
                   selection_unit='one complete candidate for the whole L batch; mean true-label CE',
                   selection_dropout_free=True, tie_rule='smallest style index; style wins equality with clean',
                   clean_candidate_included=True,candidate_pool_size=5,identity_selected=False,style_winner_index=None,
                   train_diff_is_not_eval_selection_causal_effect=True)
    _ACTIVE.add(model)
    try:
        with torch.random.fork_rng(devices=_devices(model)):
            # Validate a supplied state even for candidate reads; this call stays
            # private, and selection readouts themselves must consume no RNG.
            if clean_rng is not None:
                _set_rng(clean_rng, model)
            if selection != 'replay':
                initial = _snapshot(model)
                readout_rng = capture_rng(model)
                model.eval()
                try:
                    with torch.no_grad():
                        details['eval_clean'] = _metrics(model(x), labels)
                        records = []
                        seeds = candidate_seeds(seed, step)
                        for candidate_seed in seeds:
                            with _hooks(model, labels, receivers, days, physical_ids, candidate_seed, eta):
                                records.append(_metrics(model(x), labels))
                    details['candidate_state_unchanged'] = _same_state(model, initial)
                    details['candidate_rng_unchanged'] = _same_rng(readout_rng, capture_rng(model))
                    if not details['candidate_state_unchanged'] or not details['candidate_rng_unchanged']:
                        raise RuntimeError('Candidate eval forward changed model state or global RNG')
                    losses = [record['CE'] for record in records]
                    chosen = max(range(4), key=lambda i: losses[i]) if selection == 'hard' else _random_index(seed, step)
                    identity = details['eval_clean']['CE'] > losses[chosen]
                    details.update(candidate_seeds=seeds, candidate_losses=losses,
                                   style_winner_index=chosen,identity_selected=identity,
                                   selected_index=-1 if identity else chosen,
                                   selected_seed=None if identity else seeds[chosen],
                                   eval_selected=details['eval_clean'] if identity else records[chosen],
                                   selected_levels=[] if identity else list(LEVELS),
                                   eval_readouts=5,candidate_eval_forwards=4,eval_forwards=5)
                finally:
                    _restore_changed_state(model, initial)
                    for module, mode in modes:
                        module.training = mode
            if clean_rng is not None:
                _set_rng(clean_rng, model)
            if selection == 'replay' or details['identity_selected']:
                logits = model(x)
            else:
                with _hooks(model, labels, receivers, days, physical_ids, details['selected_seed'], eta) as (levels, order):
                    logits = model(x)
                details.update(levels=levels, actual_hook_order=order,
                               changed_packets=sum(int(v['changed_packets']) for v in levels.values()))
            details.update(train_forwards=1, total_extra_forwards=details['eval_forwards']+1)
    finally:
        for module, mode in modes:
            module.training = mode
        _ACTIVE.discard(model)
    details['original_training_flags_restored'] = all(module.training == mode for module, mode in modes)
    details['main_rng_restored'] = _same_rng(before_rng, capture_rng(model))
    if not details['main_rng_restored']:
        raise RuntimeError('Extra style forward changed main-path RNG')
    return logits, details
