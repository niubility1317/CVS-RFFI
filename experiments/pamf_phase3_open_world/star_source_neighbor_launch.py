"""One frozen source export and one CPU-only neighbor mining job; no truth input."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from .star_representation_launch import _read,_write,_env,_gpu_free,_job
from . import star_source_neighbor_export as export

RUN=export.RUN_ID
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus',
        'minimum_free_mb','max_added_workers_per_gpu','export_config'}


def validate(r):
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN
            or any(not isinstance(r.get(k),str) or not r[k].strip() for k in
                ('code_commit','run_root','release_root','python','launch_owner'))
            or re.fullmatch('[0-9a-f]{40}',r['code_commit']) is None or r.get('allowed_gpus')!=list(range(8))
            or r.get('minimum_free_mb')!=4096 or r.get('max_added_workers_per_gpu')!=2):
        raise ValueError('Only this source-only one-export/one-miner runtime is allowed; no truth paths')
    export.validate_config(r['export_config']);c=r['export_config']
    if (c['code_commit']!=r['code_commit'] or Path(c['output_root'])!=Path(r['run_root'])/'features'
            or Path(r['run_root']).resolve().is_relative_to(Path(r['release_root']).resolve())
            or Path(r['release_root']).resolve().is_relative_to(Path(r['run_root']).resolve())):
        raise ValueError('Single run and execution commit with separate immutable release/output required')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe existing run; never relaunch or overwrite')
    root.mkdir();(root/'configs').mkdir();(root/'logs').mkdir();_write(root/'configs/export.json',r['export_config'])
    command=[r['python'],'-B','-u','-m',__package__+'.star_source_neighbor_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=command,cwd=r['release_root'],
        launch_owner=r['launch_owner'],time=time.time())
    _write(root/'launch.json',value);print(json.dumps(value),flush=True)


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={})
    _write(path,state)
    try:
        available=_gpu_free(r)
        while not available:
            state['status']='WAITING_MEMORY';_write(path,state);time.sleep(20);available=_gpu_free(r)
        state.update(status='RUNNING',gpu=available[0]);_write(path,state)
        command=[r['python'],'-B','-u','-m',__package__+'.star_source_neighbor_export',
            '--config',str(root/'configs/export.json'),'--device','cuda:0']
        if not _job(r,state,path,'export',command,available[0]):raise RuntimeError('Export failed; no mining')
        marker=_read(root/'features/complete.json')
        if any(marker.get(k)!=v for k,v in dict(status=export.STATUS,run_id=RUN,code_commit=r['code_commit'],
                target_reads=0,private_truth_reads=0,model_states_unchanged=True).items()):
            raise ValueError('Matching complete frozen source features required')
        command=[r['python'],'-B','-u','-m',__package__+'.star_source_neighbor_mine','--feature-root',str(root/'features'),
            '--output',str(root/'mining'),'--code-commit',r['code_commit']]
        if not _job(r,state,path,'mine',command,None):raise RuntimeError('Mining failed; preserve features')
        marker=_read(root/'mining/complete.json')
        if any(marker.get(k)!=v for k,v in dict(status='SOURCE_NEIGHBOR_MINING_COMPLETE',run_id=RUN,
                code_commit=r['code_commit'],target_reads=0,private_truth_reads=0,neural_updates=0).items()):
            raise ValueError('Matching complete source-only neighbor result required')
        state.update(status='ANALYZED_SOURCE_INPUT_DIAGNOSTIC',all_stages_complete=True,
            next_step=marker['next_step'],final_phase3_complete=False)
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc),final_phase3_complete=False)
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','driver'));p.add_argument('--runtime',required=True);a=p.parse_args()
    (start if a.mode=='start' else driver)(a.runtime)


if __name__=='__main__':main()
