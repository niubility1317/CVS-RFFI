"""Two owned relation rows: train, source/target prediction, then truth-side score."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from .star_joint_ova_launch import _read,_write,_env,_gpu_free,_job

RUN='20261030-phase3-star-relation-oe-s2026092701-r01'
ARMS=('relation_control','relation_oe')
PRIMARY='relation_oe::entropy_global'
SCORE_STATUS='STAR_RELATION_OE_SCORES_COMPLETE'
TEACHER_RUN='20261029-phase3-star-joint-ova-s2026092701-r02'
CACHE_RUN='20261029-phase3-star-representation-s2026092701-r01'
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus',
    'minimum_free_mb','max_added_workers_per_gpu','max_concurrent_rows','rows','novel_truth','target_truth'}


def validate(r):
    from .star_relation_oe_train import validate_config as train_validate
    from .star_relation_oe_eval import validate_config as eval_validate
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN
            or any(not isinstance(r.get(k),str) or not r[k].strip() for k in ('code_commit','run_root','release_root','python','launch_owner','novel_truth','target_truth'))
            or re.fullmatch('[0-9a-f]{40}',r['code_commit']) is None
            or r.get('allowed_gpus')!=list(range(8)) or r.get('minimum_free_mb')!=4096
            or r.get('max_added_workers_per_gpu')!=2 or r.get('max_concurrent_rows')!=2 or set(r.get('rows',{}))!=set(ARMS)):
        raise ValueError('Only the fixed two-row development runtime and scorer-only truth fields are allowed')
    root=Path(r['run_root']);teacher=root.parent/TEACHER_RUN/'teacher';cache=root.parent/CACHE_RUN/'cache'
    for arm,row in r['rows'].items():
        if not isinstance(row,dict) or set(row)!={'train_config','eval_config'}:raise ValueError('One independent train/eval pair per arm required')
        t,e=row['train_config'],row['eval_config'];train_validate(t);eval_validate(e)
        if (t['arm']!=arm or e['arm']!=arm or any(c['run_id']!=RUN or c['code_commit']!=r['code_commit'] for c in (t,e))
                or Path(t['output_root'])!=root/'rows'/arm/'train' or Path(t['teacher_root'])!=teacher
                or Path(t['L_root'])!=cache/'L' or Path(t['OE_root'])!=cache/'OE'
                or Path(e['cache_root'])!=cache or Path(e['output_root'])!=root/'rows'/arm/'predict' or e['train_config']!=t
                or Path(e['checkpoint'])!=root/'rows'/arm/'train/final.pt'):
            raise ValueError('Owned arm/run/code/paths or fixed original-F0 teacher/source cache differ')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe existing output; never relaunch or overwrite')
    root.mkdir();(root/'configs').mkdir();(root/'logs').mkdir();(root/'rows').mkdir()
    for arm,row in r['rows'].items():
        (root/'rows'/arm).mkdir()
        for role,config in row.items():_write(root/'configs'/(arm+'_'+role+'.json'),config)
    command=[r['python'],'-B','-u','-m',__package__+'.star_relation_oe_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=command,cwd=r['release_root'],time=time.time(),launch_owner=r['launch_owner'])
    _write(root/'launch.json',value);print(json.dumps(value),flush=True)


def _score_terminal(r,arm):
    result=_read(Path(r['run_root'])/'rows'/arm/'score/complete.json')
    if (result.get('status')!=SCORE_STATUS or result.get('run_id')!=RUN or result.get('code_commit')!=r['code_commit'] or result.get('arm')!=arm
            or type(result.get('eligible_primary')) is not bool or not isinstance(result.get('next_step'),str) or not result['next_step']):
        raise ValueError('Independent development score terminal identity or result fields differ')
    return result


def row_driver(runtime,arm,gpu):
    r=validate(_read(runtime))
    if arm not in ARMS or gpu not in r['allowed_gpus']:raise ValueError('Only an owned arm and declared GPU may run')
    root=Path(r['run_root']);path=root/'rows'/arm/'state.json'
    if path.exists():raise FileExistsError('Existing owned row state must be observed, never restarted')
    state=dict(run_id=RUN,arm=arm,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={});_write(path,state)
    commands=[(arm+'_train',[r['python'],'-B','-u','-m',__package__+'.star_relation_oe_train','train','--config',str(root/'configs'/(arm+'_train_config.json')),'--device','cuda:0'],gpu),
        (arm+'_eval',[r['python'],'-B','-u','-m',__package__+'.star_relation_oe_eval','run','--config',str(root/'configs'/(arm+'_eval_config.json')),'--device','cuda:0'],gpu),
        (arm+'_score',[r['python'],'-B','-u','-m',__package__+'.star_relation_oe_score','--prediction-root',str(root/'rows'/arm/'predict'),
            '--novel-truth',r['novel_truth'],'--target-truth',r['target_truth'],'--output',str(root/'rows'/arm/'score')],None)]
    try:
        for name,command,device in commands:
            if device is not None:
                while device not in _gpu_free(r):
                    state['status']='WAITING_MEMORY';_write(path,state);time.sleep(5)
                state['status']='RUNNING';_write(path,state)
            if not _job(r,state,path,name,command,device):raise RuntimeError(name+' failed; preserve partial, no automatic restart')
        result=_score_terminal(r,arm)
        state.update(status='ANALYZED_DEVELOPMENT',eligible_primary=result['eligible_primary'],next_step=result['next_step'],final_phase3_complete=False)
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc))
    state['finished_at']=time.time();_write(path,state);print(state['status'],arm,flush=True)


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    if path.exists():raise FileExistsError('Existing pipeline state must be observed, never restarted')
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={});_write(path,state)
    pending=list(ARMS);active={}
    try:
        while pending or active:
            available=[g for g in _gpu_free(r) if g not in {v['gpu'] for v in active.values()}]
            while pending and available and len(active)<2:
                arm=pending.pop(0);gpu=available.pop(0)
                command=[r['python'],'-B','-u','-m',__package__+'.star_relation_oe_launch','row','--runtime',str(runtime),'--arm',arm,'--gpu',str(gpu)]
                log=(root/'logs'/(arm+'_driver.log')).open('xb')
                child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                active[arm]=dict(process=child,gpu=gpu,log=log)
                state['jobs'][arm]=dict(status='RUNNING',pid=child.pid,gpu=gpu,argv=command,started_at=time.time());_write(path,state)
            for arm,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                item['log'].close();sp=root/'rows'/arm/'state.json';row=_read(sp) if sp.is_file() else {}
                state['jobs'][arm].update(status='COMPLETE' if code==0 and row.get('status')=='ANALYZED_DEVELOPMENT' else 'FAILED',returncode=code,finished_at=time.time())
                del active[arm];_write(path,state)
            if pending or active:time.sleep(2)
        if any(state['jobs'][arm]['status']!='COMPLETE' for arm in ARMS):raise RuntimeError('A row failed; independent healthy rows and outputs retained')
        primary=_score_terminal(r,'relation_oe')
        state.update(status='ANALYZED_DEVELOPMENT',eligible_primary=primary['eligible_primary'],next_step=primary['next_step'],primary=PRIMARY,final_phase3_complete=False)
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc))
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['start','driver','row']);parser.add_argument('--runtime',required=True)
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--gpu',type=int);args=parser.parse_args()
    if args.mode=='row':return row_driver(args.runtime,args.arm,args.gpu)
    return dict(start=start,driver=driver)[args.mode](args.runtime)


if __name__=='__main__':main()
