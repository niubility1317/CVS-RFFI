"""Reuse the exact K5 B state from this author's completed satellite pilot."""
from pathlib import Path

from .group2_b_state import load_b_state
from .io import read, require
from .mopc_identity import require_author_mopc_row, require_author_sampling


def pilot_b(job,rt,k,split_id,pilot_c_split,b_seed,device):
    if k!=5 or not job.get('reuse_author_b'):
        return None
    root=Path(job['reuse_author_b'])
    previous=read(job['reuse_author_runtime'])
    for field in ('data_root','source_run_root','source_contract','source_cache','input_condition'):
        require(previous.get(field)==rt.get(field), 'Pilot B source runtime differs')
    if job['profile']=='tweak_m1':
        seed=str(job['seed'])
        require(previous['additional_sources']['tweak_m1'][seed]==rt['additional_sources']['tweak_m1'][seed],
                'Pilot B Tweak source differs')
    done=read(root/'complete.json');idx=read(root/'predictions.json')
    require(done.get('status')=='PREDICTIONS_COMPLETE' and done.get('truth_read') is False and
            done.get('query_fit') is False and done.get('profile')==job['profile'] and
            done.get('model_seed')==job['seed'] and idx.get('capsule')==job['capsule'] and
            idx.get('profile')==job['profile'] and idx.get('model_seed')==job['seed'],
            'Pilot B source identity differs')
    rows=idx['rows']
    from .group2_worker import method_arm
    expected_arm=method_arm(job['profile'],'mopc_hr',job.get('mopc_bridge','original'),'none')
    require(len(rows)==1 and done.get('rows')==1 and rows[0].get('stage')=='C' and
            rows[0].get('k')==5 and rows[0].get('new_count')==5 and
            rows[0].get('split_id')==pilot_c_split and rows[0].get('arm')==expected_arm and
            rows[0].get('input_condition')=='satellite_received' and
            (root/rows[0]['prediction']).is_file() and
            (rows[0].get('resources') or {}).get('B_state_ref')==str(root/'B-K5.pt'),
            'Pilot B source does not match fixed K5/N5 prediction')
    require_author_mopc_row(rows[0]);require_author_sampling(rows[0])
    path=root/'B-K5.pt'
    payload=load_b_state(path,job,k,split_id,b_seed,device)
    return dict(payload=payload,path=str(path),seconds=None,
                native={'state_reused_from':str(path),'reuse_scope':'same_pilot_B_state'})
