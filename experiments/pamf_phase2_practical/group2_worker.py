"""Utilities for fixed-state Group2 C prediction; no truth is read here."""
from __future__ import annotations

import numpy as np
import torch
import json
import time
from pathlib import Path

from .fast_transfer import to_tensor, to_numpy


def require_all_rows(rows, points, profile, methods):
    expected = {(int(k), int(n), profile + '__' + method) for k, n in points for method in methods}
    actual = [(int(row['k']), int(row['new_count']), row['arm']) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Incomplete or duplicate requested prediction matrix')


def score_fixed(state, query, *, count: int, old_count: int, batch_size: int, device: str):
    if not 0 < old_count < count or batch_size <= 0 or len(query) == 0:
        raise ValueError('invalid evaluation dimensions')
    model = state.model.eval()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    pred = np.empty(len(query), dtype=np.int16)
    old = np.empty(len(query), dtype=np.int16)
    with torch.inference_mode():
        for start in range(0, len(query), batch_size):
            x = to_tensor(query[start:start + batch_size], device, np.float32)
            scores = state.score(x)
            if scores.shape != (len(x), count) or not bool(torch.isfinite(scores).all()):
                raise ValueError('invalid fixed-state C scores')
            pred[start:start + len(x)] = to_numpy(scores.argmax(1), np.int16)
            old[start:start + len(x)] = to_numpy(scores[:, :old_count].argmax(1), np.int16)
    if any(not torch.equal(before[key], value) for key, value in model.state_dict().items()):
        raise ValueError('query prediction updated the registered model')
    return pred, old


def predict(rt, job):
    """One immutable capsule/seed/B profile; one B fit per K, C forks per N."""
    from .io import activate, read, require, write
    from .group1 import row, source_scope
    from .group1_source import load_source, embedding
    from .group1_target import fit_B
    from .group2_cross_fit import prepare_b, register_c
    from .episodic_io import load_task
    from .episodic_experiment import split_rows
    from .external_execution import plain, seed_for

    activate(rt)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    profile, seed = job['profile'], int(job['seed'])
    require(row(profile)['category'] == 'DA', 'Group2 worker requires a B-stage DA method')
    points = sorted(set((int(k), int(n)) for k, n in job['points']))
    require(points and all(k in (1, 5, 10, 20) and n in (2, 5, 10, 20) for k, n in points), 'Invalid requested K/N')
    methods = tuple(job['methods'])
    require(methods and len(set(methods)) == len(methods) and set(methods) <= {'lora', 'mopc_hr', 'orthogonal'}, 'Invalid C methods')
    device = rt.get('device', 'cuda:0')
    source_rt = source_scope(rt)
    source_rt['run_root'] = rt['source_run_root']
    bundle = load_source(source_rt, profile, seed, device)
    source_before = {key: value.detach().clone() for key, value in bundle['model'].state_dict().items()}
    cap = Path(rt['data_root']) / job['capsule']
    manifest = read(cap / 'manifest.json')
    require(manifest['protocol_schema'] == 'p2_min_v1' and manifest['phase2_data_status'] == 'VALIDATED_ONCE', 'Wrong capsule')
    splits = [read(path) for path in (cap / 'splits').glob('*.json')]
    lookup = {(item['k'], item['new_count']): item for item in splits}
    require(all((k, 0) in lookup and (k, n) in lookup for k, n in points), 'Missing registered split')
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    log = (output / 'fit.jsonl').open('x', encoding='utf-8')
    def synchronize():
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
    def emit(value):
        log.write(json.dumps(plain(value), allow_nan=False) + '\n')
        log.flush()
    try:
        for k in sorted({p[0] for p in points}):
            task = load_task(cap, lookup[k, 0]['split_id'], stage='B')
            old_x = to_tensor(task['support_x'], device, np.float32)
            old_y = to_tensor(task['support_y'], device, np.int64).long()
            synchronize(); started = time.perf_counter()
            bstate = fit_B(rt, bundle, old_x, old_y, seed_for(seed, job['capsule'], k, 'B'), emit)
            synchronize()
            b_seconds = time.perf_counter() - started
            old_ids = task['support_ids'].copy()
            native = bstate.get('native_state')
            old_prototypes = getattr(native, 'prototypes', None) if profile == 'protonet_cda' else None
            extractor = lambda model, x: embedding(model, row(profile)['method'], x)
            prepared = prepare_b(bstate['model'], profile, extractor, old_x, old_y,
                                 old_count=6, old_prototypes=old_prototypes, b_scores=bstate.get('score_fn'))
            require(set(methods) <= set(prepared), 'Requested B/C bridge is not bound')
            with torch.no_grad():
                original_scores = bstate['score_fn'](bstate['model'], old_x)
                bridge_scores = prepared['mopc_hr'].model(old_x)['tx_logits']
                require(torch.allclose(original_scores, bridge_scores, rtol=1e-4, atol=1e-4), 'B scores changed at C bridge')
                b_bridge_max_abs_diff = float((original_scores - bridge_scores).abs().max().item())
            for _, n in [point for point in points if point[0] == k]:
                spec = lookup[k, n]
                current = load_task(cap, spec['split_id'], stage='C')
                mask = current['support_y'] < 6
                require(np.array_equal(old_ids, current['support_ids'][mask]), 'B/C old physical support differs')
                new_x = to_tensor(current['support_x'][~mask], device, np.float32)
                new_y = to_tensor(current['support_y'][~mask], device, np.int64).long()
                for method in methods:
                    synchronize(); started = time.perf_counter()
                    state = register_c(prepared, new_x, new_y,
                        seed=seed_for(seed, job['capsule'], k, n, 'C'), methods=(method,))[method]
                    synchronize()
                    c_seconds = time.perf_counter() - started
                    synchronize(); started = time.perf_counter()
                    pred, restricted = score_fixed(state, current['query_x'], count=6+n, old_count=6,
                                                   batch_size=rt.get('batch_size', 128), device=device)
                    synchronize(); q_seconds = time.perf_counter() - started
                    name = f'C-K{k}-N{n}-{method}.npz'
                    np.savez_compressed(output / name, query_indices=split_rows(cap, spec),
                                        predicted_indices=pred, old_restricted_indices=restricted)
                    rows.append(dict(split_id=spec['split_id'], stage='C', arm=profile+'__'+method,
                        input_path='P0', prediction=name, k=k, new_count=n,
                        registered_classes=spec['registered_classes'], query_count=len(pred),
                        claim='DA_B_to_CIL_core_transplant', resources=dict(B_seconds=b_seconds,
                        C_seconds=c_seconds, query_seconds=q_seconds, B_fit_shared_across_C=True,
                        target_old_labels=6*k, target_new_labels=n*k, query_fit=False,
                        B_bridge_max_abs_diff=b_bridge_max_abs_diff,
                        B_native=plain(bstate.get('resources',{})), C_native=plain(getattr(state,'resource',{})),
                        C_protocol=plain(getattr(state,'protocol_metadata',{})))))
                    print('GROUP2_PREDICTED', profile, method, k, n, len(pred), flush=True)
            del prepared, bstate, old_x, old_y
        require_all_rows(rows, points, profile, methods)
        require(all(torch.equal(source_before[key], value) for key, value in bundle['model'].state_dict().items()), 'A source changed')
        write(output / 'predictions.json', dict(capsule=job['capsule'], capsule_id=manifest['capsule_id'],
              profile=profile, model_seed=seed, rows=rows))
        write(output / 'complete.json', dict(status='PREDICTIONS_COMPLETE', capsule_id=manifest['capsule_id'],
              rows=len(rows), truth_read=False, query_fit=False, profile=profile, model_seed=seed))
    finally:
        log.close()
