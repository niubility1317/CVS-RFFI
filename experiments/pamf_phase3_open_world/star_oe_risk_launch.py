"""Two frozen-source rows, each immediately followed by independent scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .star_representation_launch import _read,_write,_env,_gpu_free,_job

RUN='20261030-phase3-star-oe-risk-budget-s2026092701-r01'
ARMS=('detach_control','joint')
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus',
    'minimum_free_mb','max_added_workers_per_gpu','max_concurrent_rows','rows','novel_truth'}


def validate(r):
    from .star_oe_risk_run import validate_config
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN
            or r.get('allowed_gpus')!=list(range(8)) or r.get('minimum_free_mb')!=4096
            or r.get('max_added_workers_per_gpu')!=2 or r.get('max_concurrent_rows')!=2
            or set(r.get('rows',{}))!=set(ARMS)):
        raise ValueError('Only the fixed two-row source OE risk-budget runtime is allowed')
    root=Path(r['run_root'])
    for arm,row in r['rows'].items():
        if set(row)!={'source_config'}:raise ValueError('One source prediction configuration per arm required')
        c=validate_config(row['source_config'])
        if (c['arm']!=arm or c['run_id']!=RUN or c['code_commit']!=r['code_commit']
                or Path(c['output_root'])!=root/'rows'/arm/'predict'):
            raise ValueError('Owned arm/code/prediction root differ')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe existing output; never relaunch or overwrite')
    root.mkdir();(root/'configs').mkdir();(root/'logs').mkdir();(root/'rows').mkdir()
    for arm,row in r['rows'].items():
        (root/'rows'/arm).mkdir();_write(root/'configs'/(arm+'_source_config.json'),row['source_config'])
    command=[r['python'],'-B','-u','-m',__package__+'.star_oe_risk_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=command,cwd=r['release_root'],
        time=time.time(),launch_owner=r['launch_owner'])
    _write(root/'launch.json',value);print(json.dumps(value),flush=True)


def row_driver(runtime,arm,gpu):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'rows'/arm/'state.json'
    state=dict(run_id=RUN,arm=arm,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={})
    _write(path,state)
    commands=[(arm+'_source',[r['python'],'-B','-u','-m',__package__+'.star_oe_risk_run',
        '--config',str(root/'configs'/(arm+'_source_config.json')),'--device','cuda:0'],gpu),
        (arm+'_score',[r['python'],'-B','-u','-m',__package__+'.star_oe_risk_score',
        '--prediction-root',str(root/'rows'/arm/'predict'),'--novel-truth',r['novel_truth'],
        '--output',str(root/'rows'/arm/'score')],None)]
    try:
        for name,command,device in commands:
            if device is not None:
                while device not in _gpu_free(r):
                    state['status']='WAITING_MEMORY';_write(path,state);time.sleep(5)
                state['status']='RUNNING';_write(path,state)
            if not _job(r,state,path,name,command,device):raise RuntimeError(name+' failed; preserve partial, no restart')
        from .star_oe_risk_score import STATUS
        result=_read(root/'rows'/arm/'score/complete.json')
        if result.get('status')!=STATUS or result.get('run_id')!=RUN or result.get('arm')!=arm:
            raise ValueError('Independent score terminal identity differs')
        state.update(status='ANALYZED_SOURCE_ONLY',source_gate_passed=result['arm_source_gate_passed'],next_step=result['next_step'])
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc))
    state['finished_at']=time.time();_write(path,state);print(state['status'],arm,flush=True)


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={})
    _write(path,state);pending=list(ARMS);active={}
    try:
        while pending or active:
            available=[g for g in _gpu_free(r) if g not in {v['gpu'] for v in active.values()}]
            while pending and available and len(active)<2:
                arm=pending.pop(0);gpu=available.pop(0)
                command=[r['python'],'-B','-u','-m',__package__+'.star_oe_risk_launch','row',
                    '--runtime',str(runtime),'--arm',arm,'--gpu',str(gpu)]
                log=(root/'logs'/(arm+'_driver.log')).open('xb')
                child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                active[arm]=dict(process=child,gpu=gpu,log=log)
                state['jobs'][arm]=dict(status='RUNNING',pid=child.pid,gpu=gpu,argv=command,started_at=time.time());_write(path,state)
            for arm,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                item['log'].close();sp=root/'rows'/arm/'state.json';row=_read(sp) if sp.is_file() else {}
                state['jobs'][arm].update(status='COMPLETE' if code==0 and row.get('status')=='ANALYZED_SOURCE_ONLY' else 'FAILED',
                    returncode=code,finished_at=time.time())
                del active[arm];_write(path,state)
            if pending or active:time.sleep(2)
        if any(state['jobs'][arm]['status']!='COMPLETE' for arm in ARMS):raise RuntimeError('A row failed; independent healthy output retained')
        primary=_read(root/'rows/joint/score/complete.json')
        state.update(status='ANALYZED_SOURCE_ONLY',source_gate_passed=primary['eligible_primary'],
            next_step=primary['next_step'],primary='joint::oe_risk_budget',target_reads=0,final_phase3_complete=False)
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc))
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['start','driver','row']);p.add_argument('--runtime',required=True)
    p.add_argument('--arm',choices=ARMS);p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='row':return row_driver(a.runtime,a.arm,a.gpu)
    return dict(start=start,driver=driver)[a.mode](a.runtime)


if __name__=='__main__':main()
