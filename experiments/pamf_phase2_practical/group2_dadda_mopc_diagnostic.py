"""Locate nonfinite DADDA→MoPC state with source and support only."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

from .io import activate, read, require


def nonfinite_tensors(tensors):
    items = [(name, value.detach()) for name, value in tensors if value is not None]
    if not items:
        return []
    finite = torch.stack([torch.isfinite(value).all() for _, value in items]).tolist()
    return [name for (name, _), ok in zip(items, finite) if not ok]


def trace_registration(prepared, x, y, seed, emit):
    from .group2_cross_fit import register_c
    from .vendor_native.comparison_suite import registration as native

    objective, record, correction = (native.mopc_hr_incremental_objective,
                                     native._record, native.correct_old_prototypes)
    model = first = None
    updates = 0
    position = {}

    def inspect(boundary, tensors, update):
        nonlocal first
        if first is not None:
            return
        bad = nonfinite_tensors(tensors)
        if bad:
            first = dict(event='FIRST_NONFINITE', boundary=boundary, update=update, tensors=bad)
            emit(first)

    def observe_objective(logits, labels, proto_logits, proto_labels, *args, **kwargs):
        inspect('objective_inputs', [('support_logits', logits), ('prototype_logits', proto_logits)], updates + 1)
        terms = objective(logits, labels, proto_logits, proto_labels, *args, **kwargs)
        inspect('objective_terms', vars(terms).items(), updates + 1)
        return terms

    def observe_record(trace, current, terms, callback, start, **fields):
        nonlocal model
        model = current
        position.update({key: fields[key] for key in ('stage', 'epoch', 'iteration')})
        inspect('forward_buffers', current.named_buffers(), updates + 1)
        inspect('backward', ((name, p.grad) for name, p in current.named_parameters()), updates + 1)
        return record(trace, current, terms, callback, start, **fields)

    def observe_step(optimizer, _args, _kwargs):
        nonlocal updates
        updates += 1
        tensors = [(name, p) for name, p in model.named_parameters()]
        tensors.extend((name, b) for name, b in model.named_buffers())
        tensors.extend((f'{name}.momentum', optimizer.state.get(p, {}).get('momentum_buffer'))
                       for name, p in model.named_parameters())
        inspect('after_sgd', tensors, updates)

    def observe_correction(old, previous, current, *args, **kwargs):
        inspect('correction_inputs', [('historical', old), ('previous', previous), ('current', current)], updates)
        corrected = correction(old, previous, current, *args, **kwargs)
        inspect('correction_output', [('corrected', corrected)], updates)
        return corrected

    with ExitStack() as stack:
        stack.enter_context(patch.object(native, 'mopc_hr_incremental_objective', observe_objective))
        stack.enter_context(patch.object(native, '_record', observe_record))
        stack.enter_context(patch.object(native, 'correct_old_prototypes', observe_correction))
        stack.callback(register_optimizer_step_post_hook(observe_step).remove)
        try:
            state = register_c(prepared, x, y, seed=seed, methods=('mopc_hr',))['mopc_hr']
        except FloatingPointError as exc:
            emit(dict(event='REGISTRATION_FAILURE', error=str(exc), completed_updates=updates, **position))
            return None
    emit(dict(event='REGISTRATION_RETURNED', observed_nonfinite=first is not None, completed_updates=updates))
    return state


def split_metadata(capsule, split_id):
    require(len(split_id) == 24 and all(char in '0123456789abcdef' for char in split_id), 'Invalid split ID')
    return read(capsule / 'splits' / (split_id + '.json'))


def support_pair(capsule, capsule_id, points, old_split, new_split):
    from .group2_c_backbone_worker import support_only

    old, new = split_metadata(capsule, old_split), split_metadata(capsule, new_split)
    k, n = int(new['k']), int(new['new_count'])
    require(old['capsule_id'] == new['capsule_id'] == capsule_id
            and int(old['k']) == k and int(old['new_count']) == 0 and n > 0 and [k, n] in points,
            'Support does not match the registered job')
    old_x, old_y, old_ids, _ = support_only(capsule, old_split)
    all_x, all_y, all_ids, _ = support_only(capsule, new_split)
    mask = all_y < 6
    require(np.array_equal(old_ids, all_ids[mask]), 'B/C old physical support differs')
    return old_x, old_y, all_x[~mask], all_y[~mask], k, n


def run(args, emit):
    from .external_execution import seed_for
    from .fast_transfer import to_tensor
    from .group1 import source_scope
    from .group1_source import embedding, load_source, source_data
    from .group1_target import fit_B
    from .group2_cross_fit import prepare_b

    runtime, job = read(args.runtime), read(args.job)
    profile, seed = job['profile'], int(job['seed'])
    require(profile == 'dadda_sda' and 'mopc_hr' in job['methods'], 'Not a DADDA→MoPC job')
    activate(runtime)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = source_scope(runtime)
    source['run_root'] = runtime['source_run_root']
    bundle = load_source(source, profile, seed, args.device)

    def extractor(model, x):
        return embedding(model, bundle['method'], x)

    with torch.no_grad():
        labeled = source_data(source, 'L_s')
        labels = np.asarray(labeled.metadata['y'])
        indices = [int(np.flatnonzero(labels == index)[0]) for index in range(6)]
        x = labeled.x[indices].to(args.device)
        y = torch.arange(6, device=args.device)
        model = bundle['model']
        smoke = prepare_b(model, profile, extractor, x, y)['mopc_hr']
        require(torch.allclose(bundle['score_fn'](model, x), smoke.model(x)['tx_logits'], rtol=1e-4, atol=1e-4),
                'Source-only DADDA bridge smoke failed')
    emit(dict(event='SOURCE_SMOKE_PASS', device=args.device, target_rows=0, query_rows=0))
    del labeled, smoke, x, y

    capsule = Path(runtime['data_root']) / job['capsule']
    old_x, old_y, new_x, new_y, k, n = support_pair(capsule, args.capsule_id, job['points'],
                                                    args.old_split, args.new_split)
    old_x = to_tensor(old_x, args.device, np.float32)
    old_y = to_tensor(old_y, args.device, np.int64)
    new_x = to_tensor(new_x, args.device, np.float32)
    new_y = to_tensor(new_y, args.device, np.int64)
    b_seed = seed_for(seed, job['capsule'], k, 'B')
    c_seed = seed_for(seed, job['capsule'], k, n, 'C')
    state = fit_B(runtime, bundle, old_x, old_y, b_seed, None)
    bad = nonfinite_tensors(state['model'].state_dict().items())
    emit(dict(event='B_STATE', nonfinite_tensors=bad, steps=state['resources']['optimizer_steps'],
              model_seed=seed, b_seed=b_seed, c_seed=c_seed, k=k, n=n))
    if bad:
        return
    prepared = prepare_b(state['model'], profile, extractor, old_x, old_y, b_scores=state['score_fn'])
    with torch.no_grad():
        expected = state['score_fn'](state['model'], old_x)
        actual = prepared['mopc_hr'].model(old_x)['tx_logits']
        require(torch.allclose(expected, actual, rtol=1e-4, atol=1e-4), 'B scores changed at C bridge')
    trace_registration(prepared, new_x, new_y, c_seed, emit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime', 'job', 'capsule-id', 'old-split', 'new-split'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), required=True)
    args = parser.parse_args()
    run(args, lambda event: print(json.dumps(event, allow_nan=False), flush=True))


if __name__ == '__main__':
    main()
