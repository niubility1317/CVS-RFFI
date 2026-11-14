"""Load an existing ProtoNet B when replacing the MoPC bridge version."""
from pathlib import Path

from .group2_b_state import load_b_state
from .io import read, require


def prior_b(job, rt, k, split_id, b_seed, device):
    ref = job.get('reuse_original_b')
    if ref is None:
        return None
    require(job.get('mopc_bridge') == 'proto_affine_v1' and job.get('profile') == 'protonet_cda' and
            job.get('methods') == ['mopc_hr'], 'Wrong B reuse method')
    old = read(ref['job']); old_rt = read(ref['runtime'])
    require(all(old.get(field) == job.get(field) for field in ('key', 'capsule', 'profile', 'seed', 'methods', 'points')) and
            old.get('mopc_bridge', 'original') == 'original' and old.get('save_B_for_mopc') is True,
            'Original B task identity mismatch')
    for field in ('data_root', 'clean_root', 'source_run_root', 'source_contract'):
        require(Path(rt[field]).resolve() == Path(old_rt[field]).resolve(), 'Original B source/data mismatch')
    root = Path(old['output']).resolve()
    require(root.parent.name == 'tasks', 'Original B output is not a task output')
    parent_state = read(root.parent.parent/'scheduler'/'state.json')
    require(parent_state.get('status') in ('FAILED','STOPPED') and parent_state.get('active') == [],
            'Original B owner is not drained')
    path = root/f'B-K{k}.pt'
    if not path.is_file():
        require(not (root/'complete.json').exists() and not list(root.glob(f'C-K{k}-N*-mopc_hr.npz')),
                'Previously processed K has missing saved B')
        return None
    saved = load_b_state(path, old, k, split_id, b_seed, device)
    require(saved.get('old_prototypes') is not None, 'Saved ProtoNet B lacks prototypes')
    seconds = None; native = {'state_reused_from': str(path)}
    if (root/'complete.json').is_file():
        done = read(root/'complete.json'); index = read(root/'predictions.json')
        require(done.get('status') == 'PREDICTIONS_COMPLETE' and done.get('truth_read') is False and
                done.get('query_fit') is False and index.get('capsule') == job['capsule'] and
                index.get('profile') == job['profile'] and index.get('model_seed') == job['seed'],
                'Original completed B reference is invalid')
        prior = [r['resources'] for r in index['rows'] if r['k'] == k]
        require(prior and all(p.get('B_state_ref') == str(path) for p in prior), 'Original B reference differs')
        seconds, native = prior[0]['B_seconds'], prior[0]['B_native']
    return dict(payload=saved, path=str(path), seconds=seconds, native=native)
