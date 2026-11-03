"""Source-C-backbone B→C worker; B/C fit sees support only, scoring is separate."""
from pathlib import Path
import time

import numpy as np
import torch

from .fast_transfer import to_numpy, to_tensor
from .io import read, require


def support_only(capsule, split_id):
    """Read only source-defined target support, never target query IQ or truth."""
    root = Path(capsule)
    require(len(split_id) == 24 and all(c in '0123456789abcdef' for c in split_id), 'Invalid split ID')
    manifest = read(root/'manifest.json')
    split = read(root/'splits'/(split_id+'.json'))
    require(manifest['storage'] == 'episodic-pool-complement-v1' and
            manifest['protocol_schema'] == split['protocol_schema'] == 'p2_min_v1' and
            manifest['phase2_data_status'] == split['phase2_data_status'] == 'VALIDATED_ONCE' and
            split['capsule_id'] == manifest['capsule_id'] and split['split_id'] == split_id,
            'Wrong capsule or split')
    n, k = int(split['new_count']), int(split['k'])
    classes = split['registered_classes']
    require(n in (0, 2, 5, 10, 20) and k in (1, 5, 10, 20) and len(classes) == 6+n,
            'Wrong support class count or K')
    indices = np.asarray(split['support_indices'], dtype=np.int64)
    labels = np.load(root/'support_labels'/(split_id+'.npy'), allow_pickle=False)
    bank = np.load(root/'received.npy', mmap_mode='r', allow_pickle=False)
    require(bank.dtype == np.float32 and bank.shape == (manifest['received_count'], 2, 256), 'Wrong IQ bank')
    require(len(indices) == k*len(classes) and len(np.unique(indices)) == len(indices) and
            bool((indices >= 0).all()) and bool((indices < len(bank)).all()), 'Illegal support indices')
    require(labels.tolist() == np.repeat(np.arange(len(classes)), k).tolist(), 'Wrong support labels')
    return np.asarray(bank[indices], dtype=np.float32), labels.astype(np.int64), indices, split


def score_indexed(state, bank, indices, *, count, old_count, batch_size, device):
    """Read frozen query rows one batch at a time after C registration is fixed."""
    if not 0 < old_count < count or batch_size <= 0 or len(indices) == 0:
        raise ValueError('invalid query inference dimensions')
    model = state.model.eval()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    pred = np.empty(len(indices), dtype=np.int16)
    old = np.empty(len(indices), dtype=np.int16)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            x = to_tensor(bank[indices[start:start+batch_size]], device, np.float32)
            scores = state.score(x)
            if scores.shape != (len(x), count) or not bool(torch.isfinite(scores).all()):
                raise ValueError('invalid frozen C scores')
            pred[start:start+len(x)] = to_numpy(scores.argmax(1), np.int16)
            old[start:start+len(x)] = to_numpy(scores[:, :old_count].argmax(1), np.int16)
    if any(not torch.equal(before[key], value) for key, value in model.state_dict().items()):
        raise ValueError('query inference changed C state')
    return pred, old


def predict(rt, job):
    """A source C checkpoint, one B fit per K, and a fork per fixed N."""
    from .io import activate, write
    from .group1 import source_scope
    from .group1_source import load_source
    from .external_execution import plain, seed_for
    from .external_native import _source_L
    from .episodic_experiment import split_rows
    from .group2_c_backbone_fit import REPLAY_B, fit_b_from_c_source, register_native_c
    from .group2_c_backbone_reuse import hu_aux_folder, reuse_pilot_prediction

    activate(rt)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = rt.get('device', 'cuda:0')
    source_rt = source_scope(rt)
    source_rt['run_root'] = rt['c_source_roots'][job['c_method']]
    bundle = load_source(source_rt, job['c_method'], job['seed'], device)
    bundle['source_model_seed'] = int(job['seed'])
    source_before = {key: value.detach().clone() for key, value in bundle['model'].state_dict().items()}
    source_x = source_y = None
    source_aux = None
    if job['b_method'] == 'feature_separation_hu2024':
        folder = hu_aux_folder(rt, job['c_method'], job['seed'])
        require(read(folder/'complete.json')['status'] == 'SOURCE_AUX_COMPLETE', 'Hu source auxiliary incomplete')
        source_aux = torch.load(folder/'hu_aux.pt', map_location='cpu', weights_only=False)
    if job['b_method'] in REPLAY_B:
        source_x, source_y = _source_L(rt, bundle)
        source_x = source_x.to(device=device, dtype=torch.float32)
        source_y = source_y.to(device=device, dtype=torch.long)
    cap = Path(rt['data_root'])/job['capsule']
    manifest = read(cap/'manifest.json')
    specs = {(s['k'], s['new_count']): s for s in (read(p) for p in (cap/'splits').glob('*.json'))}
    points = sorted(set((int(k), int(n)) for k, n in job['points']))
    require(points and all((k, n) in specs and (k, 0) in specs for k, n in points), 'Missing fixed split')
    out = Path(job['output'])
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    reused = set()
    if job.get('reuse_prediction'):
        require((5, 20) in points and job.get('reuse_score'), 'Wrong fixed pilot reuse plan')
        spec = specs[5, 20]
        rows.append(reuse_pilot_prediction(rt, job, manifest, spec, split_rows(cap, spec), out))
        reused.add((5, 20))
    pending = [point for point in points if point not in reused]
    def sync():
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
    for k in sorted({k for k, _ in pending}):
        x, y, old_ids, _ = support_only(cap, specs[k, 0]['split_id'])
        old_x, old_y = to_tensor(x, device, np.float32), to_tensor(y, device, np.int64).long()
        sync(); start = time.perf_counter()
        fitted = fit_b_from_c_source(rt, bundle, job['b_method'], old_x, old_y,
            seed=seed_for(job['seed'], job['capsule'], k, job['b_method'], job['c_method'], 'B'),
            model_seed=job['seed'], source_x=source_x, source_y=source_y, source_aux=source_aux)
        sync(); b_seconds = time.perf_counter()-start
        for _, n in [p for p in pending if p[0] == k]:
            x, y, ids, spec = support_only(cap, specs[k, n]['split_id'])
            old = y < 6
            require(np.array_equal(ids[old], old_ids), 'B/C old physical support differs')
            new_x, new_y = to_tensor(x[~old], device, np.float32), to_tensor(y[~old], device, np.int64).long()
            sync(); start = time.perf_counter()
            state = register_native_c(rt, fitted['c_state'], new_x, new_y, range(6, 6+n),
                seed=seed_for(job['seed'], job['capsule'], k, n, job['b_method'], job['c_method'], 'C'))
            sync(); c_seconds = time.perf_counter()-start
            query = split_rows(cap, spec)
            require(len(query) == spec['query_count'], 'Wrong query complement count')
            bank = np.load(cap/'received.npy', mmap_mode='r', allow_pickle=False)
            sync(); start = time.perf_counter()
            pred, restricted = score_indexed(state, bank, query, count=6+n, old_count=6,
                batch_size=int(rt.get('batch_size', 128)), device=device)
            sync(); q_seconds = time.perf_counter()-start
            name = f'C-K{k}-N{n}-{job["c_method"]}.npz'
            np.savez_compressed(out/name, query_indices=query, predicted_indices=pred, old_restricted_indices=restricted)
            resources = dict(B_seconds=b_seconds, C_seconds=c_seconds, query_seconds=q_seconds,
                    B_fit_shared_across_N=True, target_old_labels=6*k, target_new_labels=n*k,
                    query_fit=False, B_and_C=plain(fitted['metadata']),
                    C_native=plain(getattr(state, 'resource', {})),
                    C_protocol=plain(getattr(state, 'protocol_metadata', {})))
            if k == 5 and reused:
                resources['B_state_relation_to_reused_pilot'] = 'not_verified_against_pilot_B'
                resources['B_state_scope'] = 'newly_predicted_N_only'
            rows.append(dict(split_id=spec['split_id'], stage='C',
                arm=job['b_method']+'__'+job['c_method']+'__C_SOURCE', input_path='P0', prediction=name,
                k=k, new_count=n, registered_classes=spec['registered_classes'], query_count=len(pred),
                claim='C_source_backbone_with_B_core_then_native_C_transplant',
                resources=resources))
            print('C_SOURCE_PREDICTED', job['b_method'], job['c_method'], k, n, len(pred), flush=True)
        del fitted, old_x, old_y
    require(len(rows) == len(points) and {(r['k'], r['new_count']) for r in rows} == set(points),
            'Incomplete prediction matrix')
    require(all(torch.equal(source_before[k], v) for k, v in bundle['model'].state_dict().items()),
            'C source checkpoint was mutated')
    rows.sort(key=lambda row: (row['k'], row['new_count']))
    write(out/'predictions.json', dict(capsule=job['capsule'], capsule_id=manifest['capsule_id'],
        profile=job['b_method'], c_method=job['c_method'], model_seed=job['seed'], rows=rows))
    write(out/'complete.json', dict(status='PREDICTIONS_COMPLETE', capsule_id=manifest['capsule_id'],
        rows=len(rows), truth_read=False, query_fit=False, model_seed=job['seed']))
