"""One frozen original-F0 source row followed by independent CPU scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .star_representation_launch import _read,_write,_env,_gpu_free,_job

RUN='20261030-phase3-star-local-support-s2026092701-r01'
ARMS=('original_F0',)
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus',
    'minimum_free_mb','max_added_workers_per_gpu','max_concurrent_rows','rows','novel_truth'}


def validate(r):
    from .star_local_support_run import validate_config
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN
            or r.get('allowed_gpus')!=list(range(8)) or r.get('minimum_free_mb')!=4096
            or r.get('max_added_workers_per_gpu')!=2 or r.get('max_concurrent_rows')!=1 or set(r.get('rows',{}))!=set(ARMS)):
        raise ValueError('Only the fixed original-F0 single-row local-support runtime is allowed')
    root=Path(r['run_root']);row=r['rows']['original_F0']
    if set(row)!={'source_config'}:raise ValueError('One exact source configuration required')
    c=validate_config(row['source_config'])
    if c['run_id']!=RUN or c['code_commit']!=r['code_commit'] or Path(c['output_root'])!=root/'rows/original_F0/predict':
        raise ValueError('Owned code or output root differs')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe existing output; never relaunch or overwrite')
    root.mkdir();(root/'configs').mkdir();(root/'logs').mkdir();(root/'rows').mkdir();(root/'rows/original_F0').mkdir()
    _write(root/'configs/original_F0_source_config.json',r['rows']['original_F0']['source_config'])
    command=[r['python'],'-B','-u','-m',__package__+'.star_local_support_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=command,cwd=r['release_root'],time=time.time(),launch_owner=r['launch_owner'])
    _write(root/'launch.json',value);print(json.dumps(value),flush=True)


def row_driver(runtime,arm,gpu):
    r=validate(_read(runtime))
    if arm!='original_F0' or gpu not in r['allowed_gpus']:raise ValueError('Only the declared source row on an allowed GPU can run')
    root=Path(r['run_root']);path=root/'rows'/arm/'state.json'
    state=dict(run_id=RUN,arm=arm,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={});_write(path,state)
    commands=[(arm+'_source',[r['python'],'-B','-u','-m',__package__+'.star_local_support_run',
        '--config',str(root/'configs/original_F0_source_config.json'),'--device','cuda:0'],gpu),
        (arm+'_score',[r['python'],'-B','-u','-m',__package__+'.star_local_support_score',
        '--prediction-root',str(root/'rows'/arm/'predict'),'--novel-truth',r['novel_truth'],'--output',str(root/'rows'/arm/'score')],None)]
    try:
        for name,command,device in commands:
            if device is not None:
                while device not in _gpu_free(r):
                    state['status']='WAITING_MEMORY';_write(path,state);time.sleep(5)
                state['status']='RUNNING';_write(path,state)
            if not _job(r,state,path,name,command,device):raise RuntimeError(name+' failed; preserve partial, no restart')
        from .star_local_support_score import STATUS
        result=_read(root/'rows'/arm/'score/complete.json')
        if result.get('status')!=STATUS or result.get('run_id')!=RUN:raise ValueError('Score terminal identity differs')
        state.update(status='ANALYZED_SOURCE_ONLY',source_gate_passed=result['source_gate_passed'],eligible_primary=result['eligible_primary'],next_step=result['next_step'])
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc))
    state['finished_at']=time.time();_write(path,state);print(state['status'],arm,flush=True)


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={});_write(path,state)
    try:
        available=_gpu_free(r)
        while not available:
            state['status']='WAITING_MEMORY';_write(path,state);time.sleep(5);available=_gpu_free(r)
        gpu=available[0];arm='original_F0'
        command=[r['python'],'-B','-u','-m',__package__+'.star_local_support_launch','row','--runtime',str(runtime),'--arm',arm,'--gpu',str(gpu)]
        if not _job(r,state,path,'original_F0_driver',command,None):raise RuntimeError('Source row failed; preserve output, no restart')
        row=_read(root/'rows/original_F0/state.json')
        if row.get('status')!='ANALYZED_SOURCE_ONLY':raise RuntimeError('Source row did not complete independent scoring')
        state.update(status='ANALYZED_SOURCE_ONLY',source_gate_passed=row['source_gate_passed'],eligible_primary=row['eligible_primary'],next_step=row['next_step'],
            primary='original_F0::rx_consensus_relative',target_reads=0,final_phase3_complete=False)
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc))
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['start','driver','row']);p.add_argument('--runtime',required=True)
    p.add_argument('--arm',choices=ARMS);p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='row':return row_driver(a.runtime,a.arm,a.gpu)
    return dict(start=start,driver=driver)[a.mode](a.runtime)


if __name__=='__main__':main()
