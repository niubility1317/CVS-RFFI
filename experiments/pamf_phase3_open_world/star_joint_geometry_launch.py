"""One frozen cached-adapter export, then label-blind neighbors/truth-side CPU diagnostics."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from .star_representation_launch import _read,_write,_env,_gpu_free,_job

RUN='20261030-phase3-star-joint-geometry-s2026092701-r01'
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus','minimum_free_mb',
    'max_added_workers_per_gpu','export_config','L_public_root','novel_truth'}


def validate(r):
    from .star_joint_geometry_export import validate_config
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN
            or any(not isinstance(r.get(k),str) or not r[k].strip() for k in
                ('code_commit','run_root','release_root','python','launch_owner','L_public_root','novel_truth'))
            or re.fullmatch('[0-9a-f]{40}',r['code_commit']) is None or r.get('allowed_gpus')!=list(range(8))
            or r.get('minimum_free_mb')!=4096 or r.get('max_added_workers_per_gpu')!=2):
        raise ValueError('Only the fixed one-worker source diagnostic and scorer-only label paths allowed')
    validate_config(r['export_config']);c=r['export_config']
    if (c['run_id']!=RUN or c['code_commit']!=r['code_commit'] or Path(c['output_root'])!=Path(r['run_root'])/'features'
            or Path(r['L_public_root'])!=Path(c['parent_eval_config']['cache_root'])/'L'):
        raise ValueError('Owned export and exact old L public-label source handle required')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe existing run; never relaunch or replace outputs')
    root.mkdir();(root/'configs').mkdir();(root/'logs').mkdir();_write(root/'configs/export.json',r['export_config'])
    command=[r['python'],'-B','-u','-m',__package__+'.star_joint_geometry_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=command,cwd=r['release_root'],launch_owner=r['launch_owner'],time=time.time())
    _write(root/'launch.json',value);print(json.dumps(value),flush=True)


def _terminal(r,stage):
    folder='features' if stage=='export' else 'score';m=_read(Path(r['run_root'])/folder/'complete.json')
    required=dict(run_id=RUN,code_commit=r['code_commit'],counts=dict(L=6300,novel=3150,union=9450),source_only=True,target_reads=0,optimizer_updates=0,final_phase3_complete=False)
    if stage=='export':required.update(status='STAR_JOINT_GEOMETRY_FEATURES_COMPLETE',private_truth_reads=0,model_states_unchanged=True,
        actual_adapter_checkpoint_loaded=True,native_F0_checkpoint_loaded=False,G_checkpoint_loaded=False,
        real_F_forward_rows=0,synthetic_F_forward_rows=0,synthetic_adapter_rows=2,adapter_L_feature_rows=6300,novel_feature_rows_reused=3150)
    else:required.update(status='STAR_JOINT_GEOMETRY_SCORES_COMPLETE',truth_last=True,all_16_neighbors_fixed_before_truth=True,
        neighbor_selection_TX_or_K_used=False,diagnostic_only=True,eligible_primary=False,full_goal_status='ACTIVE')
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Joint source '+stage+' permission/identity terminal differs')
    return m


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    if path.exists():raise FileExistsError('Observe existing pipeline; never restart')
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={});_write(path,state)
    try:
        available=_gpu_free(r)
        while not available:
            state['status']='WAITING_MEMORY';_write(path,state);time.sleep(20);available=_gpu_free(r)
        state.update(status='RUNNING',gpu=available[0]);_write(path,state)
        command=[r['python'],'-B','-u','-m',__package__+'.star_joint_geometry_export','--config',str(root/'configs/export.json'),'--device','cuda:0']
        if not _job(r,state,path,'export',command,available[0]):raise RuntimeError('Export failed; preserve output, never score incomplete arrays')
        _terminal(r,'export')
        command=[r['python'],'-B','-u','-m',__package__+'.star_joint_geometry_score','--feature-root',str(root/'features'),
            '--L-public-root',r['L_public_root'],'--novel-truth',r['novel_truth'],'--output',str(root/'score')]
        if not _job(r,state,path,'score',command,None):raise RuntimeError('Independent score failed; preserve frozen export')
        _terminal(r,'score');state.update(status='ANALYZED_SOURCE_DIAGNOSTIC',all_stages_complete=True,eligible_primary=False,
            target_reads=0,final_phase3_complete=False,full_goal_status='ACTIVE')
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc),final_phase3_complete=False)
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','driver'));p.add_argument('--runtime',required=True);a=p.parse_args()
    {'start':start,'driver':driver}[a.mode](a.runtime)


if __name__=='__main__':main()
