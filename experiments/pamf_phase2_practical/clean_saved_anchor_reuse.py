"""Reuse an actual saved B anchor for a separately versioned MoPC run."""
from pathlib import Path

from .group2_b_state import load_b_state
from .io import read, require


def prior_anchor(job, rt, k, split_id, b_seed, device):
    ref=job.get('reuse_saved_b')
    if ref is None:return None
    require(job.get('methods')==['mopc_hr'] and job.get('mopc_stabilization')=='clip1000_v1',
            'Saved anchor requires a versioned MoPC-only task')
    old=read(ref['job']);old_rt=read(ref['runtime'])
    require(all(old.get(field)==job.get(field) for field in ('key','capsule','profile','seed','methods','points')) and
            old.get('mopc_bridge','original')==job.get('mopc_bridge','original') and old.get('save_B_for_mopc') is True,
            'Saved B task identity differs')
    for field in ('data_root','clean_root','source_run_root','source_contract'):
        require(Path(rt[field]).resolve()==Path(old_rt[field]).resolve(),'Saved B source or data differs')
    if job['profile']=='tweak_m1':
        seed=str(job['seed'])
        require(Path(rt['additional_sources']['tweak_m1'][seed]).resolve()==
                Path(old_rt['additional_sources']['tweak_m1'][seed]).resolve(), 'Saved B Tweak source differs')
    root=Path(old['output']).resolve();require(root.parent.name=='tasks','Saved B is outside a task root')
    state=read(root.parent.parent/'scheduler'/'state.json')
    completed=root.parent.parent/'scheduler'/'predictions_complete.json'
    fixed=completed.is_file() and read(completed).get('status')=='SELECTED_PREDICTIONS_FIXED'
    require((state.get('status') in ('FAILED','STOPPED','COMPLETED_WITH_QUARANTINE','COMPLETE') or
             (state.get('status')=='RUNNING' and fixed)) and state.get('active')==[],
            'Saved B owner is not drained')
    path=root/f'B-K{k}.pt'
    if not path.is_file():
        require(not (root/'complete.json').exists() and not list(root.glob(f'C-K{k}-N*-mopc_hr.npz')),
                'Previously processed K has missing saved B')
        return None
    payload=load_b_state(path,old,k,split_id,b_seed,device)
    seconds=None;native={'state_reused_from':str(path)}
    if (root/'complete.json').is_file():
        done=read(root/'complete.json');index=read(root/'predictions.json')
        require(done.get('status')=='PREDICTIONS_COMPLETE' and done.get('truth_read') is False and
                done.get('query_fit') is False and index.get('capsule')==job['capsule'] and
                index.get('profile')==job['profile'] and index.get('model_seed')==job['seed'],
                'Saved B completed task identity differs')
        rows=[r['resources'] for r in index['rows'] if r['k']==k]
        require(rows and all(r.get('B_state_ref')==str(path) for r in rows), 'Saved B row reference differs')
        seconds=rows[0].get('B_seconds');native=dict(rows[0].get('B_native',{}),state_reused_from=str(path))
    return dict(payload=payload,path=str(path),seconds=seconds,native=native)
