"""One GPU source feature export followed by a separate truth-side CPU scorer."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from .star_representation_launch import _read,_write,_env,_gpu_free,_job

RUN='20261030-phase3-star-open-feature-diagnostic-s2026092701-r01'
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus','minimum_free_mb','max_added_workers_per_gpu','export_config','novel_truth'}


def validate(r):
    from .star_open_feature_export import validate_config
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN
            or any(not isinstance(r.get(k),str) or not r[k].strip() for k in ('code_commit','run_root','release_root','python','launch_owner','novel_truth'))
            or re.fullmatch('[0-9a-f]{40}',r['code_commit']) is None or r.get('allowed_gpus')!=list(range(8))
            or r.get('minimum_free_mb')!=4096 or r.get('max_added_workers_per_gpu')!=2):
        raise ValueError('Only the fixed one-GPU source diagnostic and scorer-only truth field are allowed')
    validate_config(r['export_config']);c=r['export_config']
    if (c['run_id']!=RUN or c['code_commit']!=r['code_commit'] or Path(c['output_root'])!=Path(r['run_root'])/'features'):
        raise ValueError('One owned source feature output and execution commit required')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe the same root; never relaunch existing output')
    root.mkdir();(root/'configs').mkdir();(root/'logs').mkdir();_write(root/'configs/export.json',r['export_config'])
    command=[r['python'],'-B','-u','-m',__package__+'.star_open_feature_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=command,cwd=r['release_root'],launch_owner=r['launch_owner'],time=time.time())
    _write(root/'launch.json',value);print(json.dumps(value),flush=True)


def _terminal(r,stage):
    root=Path(r['run_root']);folder='features' if stage=='export' else 'score';m=_read(root/folder/'complete.json')
    expected=dict(run_id=RUN,code_commit=r['code_commit'],source_count=3150,target_reads=0,optimizer_updates=0)
    if stage=='export':expected.update(status='STAR_OPEN_FEATURES_COMPLETE',private_truth_reads=0,model_states_unchanged=True,source_role='novel')
    else:expected.update(status='STAR_OPEN_FEATURE_SCORES_COMPLETE',truth_last=True,all_neighbors_fixed_before_truth=True,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen source '+stage+' terminal identity/permission differs')
    return m


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={});_write(path,state)
    try:
        available=_gpu_free(r)
        while not available:
            state['status']='WAITING_MEMORY';_write(path,state);time.sleep(20);available=_gpu_free(r)
        state['status']='RUNNING';state['gpu']=available[0];_write(path,state)
        command=[r['python'],'-B','-u','-m',__package__+'.star_open_feature_export','--config',str(root/'configs/export.json'),'--device','cuda:0']
        if not _job(r,state,path,'export',command,available[0]):raise RuntimeError('Export failed; preserve output and do not score')
        _terminal(r,'export')
        command=[r['python'],'-B','-u','-m',__package__+'.star_open_feature_score','--feature-root',str(root/'features'),'--novel-truth',r['novel_truth'],'--output',str(root/'score')]
        if not _job(r,state,path,'score',command,None):raise RuntimeError('Score failed; preserve frozen source features')
        _terminal(r,'score');state.update(status='ANALYZED_SOURCE_DIAGNOSTIC',all_stages_complete=True,eligible_primary=False,final_phase3_complete=False)
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc),final_phase3_complete=False)
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','driver'));p.add_argument('--runtime',required=True);a=p.parse_args()
    (start if a.mode=='start' else driver)(a.runtime)


if __name__=='__main__':main()
