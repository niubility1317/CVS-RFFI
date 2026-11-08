"""Run MoPC from the actual B anchors saved by a two-core subset run."""
from pathlib import Path
import time

import numpy as np
import torch

from .io import activate, read, require, write
from .external_execution import plain, seed_for
from .fast_transfer import to_tensor
from .group2_b_state import load_b_state
from .group2_c_backbone_worker import support_only
from .group2_cross_fit import register_c
from .group2_worker import require_all_rows, score_fixed
from .episodic_io import load_task
from .episodic_experiment import split_rows


def predict(rt, job):
    require(job['methods'] == ['mopc_hr'] and not job.get('save_B_for_mopc') and
            job.get('load_B_for_mopc'), 'Invalid saved-B MoPC job')
    activate(rt)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    profile, seed, device = job['profile'], int(job['seed']), rt.get('device', 'cuda:0')
    require(profile in ('protonet_cda', 'dadda_sda'), 'Unsupported saved-B profile')
    points = sorted(set((int(k), int(n)) for k, n in job['points']))
    require(points and all(k in (1, 5, 10, 20) and n in (2, 5, 10, 20) for k, n in points),
            'Invalid requested K/N')
    cap = Path(rt['data_root'])/job['capsule']
    manifest = read(cap/'manifest.json')
    require(manifest['protocol_schema'] == 'p2_min_v1' and manifest['phase2_data_status'] == 'VALIDATED_ONCE',
            'Wrong capsule')
    lookup = {(v['k'], v['new_count']): v for v in (read(p) for p in (cap/'splits').glob('*.json'))}
    subset_output = Path(job['load_B_for_mopc'])
    from .group2_schedule import closed_prediction
    subset_job = dict(job, output=str(subset_output), methods=['orthogonal', 'lora'], save_B_for_mopc=True)
    subset_job.pop('load_B_for_mopc')
    closed_prediction(subset_job, len(points)*2)
    subset_index = read(subset_output/'predictions.json')
    subset_done = read(subset_output/'complete.json')
    require(subset_done['status'] == 'SUBSET_PREDICTIONS_FIXED' and subset_done['truth_read'] is False and
            subset_done['query_fit'] is False and subset_index['capsule_id'] == manifest['capsule_id'] and
            subset_index['profile'] == profile and subset_index['model_seed'] == seed,
            'Saved-B subset binding differs')
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for k in sorted({p[0] for p in points}):
        old_split = lookup[k, 0]['split_id']
        _, _, old_ids, _ = support_only(cap, old_split)
        b_seed = seed_for(seed, job['capsule'], k, 'B')
        b_ref = subset_output/f'B-K{k}.pt'
        saved = load_b_state(b_ref, job, k, old_split, b_seed, device)
        anchor = saved['anchor']
        prior = [r for r in subset_index['rows'] if r['k'] == k]
        require(prior and all(r['resources'].get('B_state_ref') == str(b_ref) for r in prior),
                'Subset does not bind the saved B state')
        for _, n in [p for p in points if p[0] == k]:
            spec = lookup[k, n]
            task = load_task(cap, spec['split_id'], stage='C')
            mask = task['support_y'] < 6
            require(np.array_equal(old_ids, task['support_ids'][mask]), 'B/C old physical support differs')
            new_x = to_tensor(task['support_x'][~mask], device, np.float32)
            new_y = to_tensor(task['support_y'][~mask], device, np.int64).long()
            if str(device).startswith('cuda'): torch.cuda.synchronize()
            started = time.perf_counter()
            state = register_c({'mopc_hr': anchor}, new_x, new_y,
                               seed=seed_for(seed, job['capsule'], k, n, 'C'), methods=('mopc_hr',))['mopc_hr']
            if str(device).startswith('cuda'): torch.cuda.synchronize()
            c_seconds = time.perf_counter()-started
            started = time.perf_counter()
            pred, restricted = score_fixed(state, task['query_x'], count=6+n, old_count=6,
                                           batch_size=rt.get('batch_size', 128), device=device)
            if str(device).startswith('cuda'): torch.cuda.synchronize()
            q_seconds = time.perf_counter()-started
            name = f'C-K{k}-N{n}-mopc_hr.npz'
            np.savez_compressed(output/name, query_indices=split_rows(cap, spec),
                                predicted_indices=pred, old_restricted_indices=restricted)
            rows.append(dict(split_id=spec['split_id'], stage='C', arm=profile+'__mopc_hr', input_path='P0',
                prediction=name, k=k, new_count=n, registered_classes=spec['registered_classes'],
                query_count=len(pred), claim='DA_B_to_CIL_core_transplant',
                resources=dict(B_seconds=prior[0]['resources']['B_seconds'], C_seconds=c_seconds,
                    query_seconds=q_seconds, B_fit_shared_across_C=True,
                    target_old_labels=6*k, target_new_labels=n*k, query_fit=False, B_state_ref=str(b_ref),
                    B_native=prior[0]['resources']['B_native'], C_native=plain(state.resource),
                    C_protocol=plain(state.protocol_metadata))))
        del saved, anchor
    require_all_rows(rows, points, profile, ('mopc_hr',))
    write(output/'predictions.json', dict(capsule=job['capsule'], capsule_id=manifest['capsule_id'],
          profile=profile, model_seed=seed, rows=rows))
    write(output/'complete.json', dict(status='MOPC_ONLY_PREDICTIONS_FIXED', capsule_id=manifest['capsule_id'],
          rows=len(rows), truth_read=False, query_fit=False, profile=profile, model_seed=seed))
