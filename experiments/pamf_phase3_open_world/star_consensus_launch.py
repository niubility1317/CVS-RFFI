"""Owned source-only star consensus prediction and independent score chain."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .star_representation_launch import _read, _write, _env, _gpu_free, _job

RUN = '20261029-phase3-star-consensus-s2026092701-r01'
FIELDS = {'run_id','code_commit','run_root','release_root','python','launch_owner',
          'allowed_gpus','minimum_free_mb','max_added_workers_per_gpu','source_config','novel_truth'}


def validate(r):
    from .star_consensus_run import validate_config
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN
            or r.get('allowed_gpus')!=list(range(8)) or r.get('minimum_free_mb')!=4096
            or r.get('max_added_workers_per_gpu')!=2
            or any(not isinstance(r[k],str) or not r[k].strip()
                   for k in ('run_root','release_root','python','launch_owner','novel_truth'))):
        raise ValueError('Only the fixed single-worker source-only launch is supported')
    validate_config(r['source_config'])
    c=r['source_config'];root=Path(r['run_root'])
    if (c['code_commit']!=r['code_commit'] or c['run_id']!=RUN
            or Path(c['output_root'])!=root/'prediction'
            or root.resolve().is_relative_to(Path(c['cache_root']).resolve())):
        raise ValueError('Owned code and immutable prediction/cache roots differ')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe the existing run; never overwrite or relaunch')
    root.mkdir();(root/'logs').mkdir();(root/'configs').mkdir()
    _write(root/'configs/source.json',r['source_config'])
    command=[r['python'],'-B','-u','-m',__package__+'.star_consensus_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=command,
                cwd=r['release_root'],time=time.time(),launch_owner=r['launch_owner'])
    _write(root/'launch.json',record);print(json.dumps(record),flush=True)


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',
               started_at=time.time(),jobs={})
    _write(path,state)
    try:
        available=_gpu_free(r)
        if not available:raise RuntimeError('No card has the declared actual free memory; preserve root')
        gpu=available[0]
        commands=[
            ('source_predict',[r['python'],'-B','-u','-m',__package__+'.star_consensus_run','run',
                               '--config',str(root/'configs/source.json'),'--device','cuda:0'],gpu),
            ('source_score',[r['python'],'-B','-u','-m',__package__+'.star_consensus_score',
                             '--prediction-root',str(root/'prediction'),'--novel-truth',r['novel_truth'],
                             '--output',str(root/'score')],None)]
        # Exactly one GPU child is ever live in this chain. Other users and
        # owners remain running; source fitting and scoring cannot launch rows.
        for name,command,device in commands:
            if not _job(r,state,path,name,command,device):
                raise RuntimeError(name+' failed; preserve partial output without automatic restart')
        prediction=_read(root/'prediction/complete.json');scored=_read(root/'score/complete.json')
        from .star_consensus_run import STATUS as PREDICTED
        from .star_consensus_score import STATUS as SCORED
        if (prediction.get('status')!=PREDICTED or scored.get('status')!=SCORED
                or any(row.get('run_id')!=RUN or row.get('code_commit')!=r['code_commit']
                       for row in (prediction,scored))
                or type(scored.get('source_gate_passed')) is not bool):
            raise ValueError('Prediction and independent score terminal identities differ')
        state.update(status='ANALYZED_SOURCE_ONLY',source_gate_passed=scored['source_gate_passed'],
                     next_step=scored['next_step'],target_reads=0,final_phase3_complete=False)
    except Exception as exc:
        state.update(status='FAILED_PRESERVED',error=str(exc))
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['start','driver']);p.add_argument('--runtime',required=True)
    a=p.parse_args();return dict(start=start,driver=driver)[a.mode](a.runtime)


if __name__=='__main__':main()
