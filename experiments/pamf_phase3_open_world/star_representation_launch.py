"""One source cache and three independent fixed-budget training/eval rows."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

RUN = '20261029-phase3-star-representation-s2026092701-r01'
ARMS = ('anchor', 'oe', 'oe_vic')
FIELDS = {'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus',
          'minimum_free_mb','max_added_workers_per_gpu','max_concurrent_rows','cache_config','rows','novel_truth'}


def _read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def _write(p, v):
    p=Path(p); q=p.with_name(p.name+'.tmp')
    q.write_text(json.dumps(v, indent=2, allow_nan=False)+'\n', encoding='utf-8'); q.replace(p)


def validate(r):
    from .star_representation_cache import validate_config as cache_validate
    from .star_representation_train import validate_config as train_validate
    from .star_representation_eval import validate_config as eval_validate
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN or r.get('allowed_gpus')!=list(range(8))
            or r.get('minimum_free_mb')!=4096 or r.get('max_added_workers_per_gpu')!=2
            or r.get('max_concurrent_rows')!=3 or set(r.get('rows',{}))!=set(ARMS)):
        raise ValueError('Only the fixed source-only three-arm launch matrix is supported')
    root=Path(r['run_root']); cache_validate(r['cache_config'])
    if Path(r['cache_config']['output_root'])!=root/'cache':raise ValueError('Owned cache root differs')
    for arm,row in r['rows'].items():
        if set(row)!={'train_config','eval_config'}:raise ValueError('Only train and evaluation configs per row')
        train_validate(row['train_config']); eval_validate(row['eval_config'])
        t,e=row['train_config'],row['eval_config']
        if (t['arm']!=arm or e['arm']!=arm or t['run_id']!=RUN or e['run_id']!=RUN
                or t['code_commit']!=r['code_commit'] or e['code_commit']!=r['code_commit']
                or Path(t['output_root'])!=root/'rows'/arm/'train'
                or Path(e['output_root'])!=root/'rows'/arm/'predict'
                or e['train_config']!=t or Path(e['cache_root'])!=root/'cache'
                or Path(e['checkpoint'])!=root/'rows'/arm/'train/final.pt'):
            raise ValueError('Independent row identity/roots do not match the immutable matrix')
    return r


def _env(r,gpu=None):
    return dict(os.environ,PYTHONPATH=r['release_root'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))


def _gpu_free(r):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    rows=[tuple(map(int,line.split(','))) for line in raw.splitlines()]
    return [g for m,g in sorted(((m,g) for g,m in rows if g in r['allowed_gpus'] and m>=r['minimum_free_mb']),reverse=True)]


def _job(r,state,path,name,command,gpu):
    entry=dict(status='RUNNING',started_at=time.time(),argv=command,gpu=gpu)
    with (Path(r['run_root'])/'logs'/(name+'.log')).open('xb') as log:
        child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r,gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        entry['pid']=child.pid;state['jobs'][name]=entry;_write(path,state)
        entry['returncode']=child.wait()
    entry.update(status='COMPLETE' if entry['returncode']==0 else 'FAILED',finished_at=time.time());_write(path,state)
    return entry['returncode']==0


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe the existing run; never relaunch or replace its outputs')
    root.mkdir();(root/'logs').mkdir();(root/'configs').mkdir();(root/'rows').mkdir()
    _write(root/'configs/cache.json',r['cache_config'])
    for arm,row in r['rows'].items():
        (root/'rows'/arm).mkdir()
        for role,cfg in row.items():_write(root/'configs'/(arm+'_'+role+'.json'),cfg)
    command=[r['python'],'-B','-u','-m',__package__+'.star_representation_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(pid=child.pid,run_id=RUN,argv=command,cwd=r['release_root'],time=time.time(),code_commit=r['code_commit'],launch_owner=r['launch_owner'])
    _write(root/'launch.json',value);print(json.dumps(value),flush=True)


def row_driver(runtime,arm,gpu):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'rows'/arm/'state.json'
    state=dict(run_id=RUN,arm=arm,pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={})
    _write(path,state)
    commands=[
        (arm+'_train',[r['python'],'-B','-u','-m',__package__+'.star_representation_train','train','--config',str(root/'configs'/(arm+'_train_config.json')),'--device','cuda:0'],gpu),
        (arm+'_eval',[r['python'],'-B','-u','-m',__package__+'.star_representation_eval','run','--config',str(root/'configs'/(arm+'_eval_config.json')),'--device','cuda:0'],gpu),
        (arm+'_score',[r['python'],'-B','-u','-m',__package__+'.star_representation_score','--prediction-root',str(root/'rows'/arm/'predict'),'--novel-truth',r['novel_truth'],'--output',str(root/'rows'/arm/'score')],None)]
    try:
        for name,command,device in commands:
            if not _job(r,state,path,name,command,device):raise RuntimeError(name+' failed; keep partial, do not restart')
        state['status']='ANALYZED_SOURCE_ONLY'
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc))
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={})
    _write(path,state)
    try:
        available=_gpu_free(r)
        if not available:raise RuntimeError('No card has the actual preregistered free memory')
        command=[r['python'],'-B','-u','-m',__package__+'.star_representation_cache','--config',str(root/'configs/cache.json'),'--device','cuda:0']
        if not _job(r,state,path,'cache',command,available[0]):raise RuntimeError('Cache failed; no training row launched')
        pending=list(ARMS);active={}
        while pending or active:
            used={x['gpu'] for x in active.values()}
            available=[g for g in _gpu_free(r) if g not in used]
            while pending and available and len(active)<r['max_concurrent_rows']:
                arm=pending.pop(0);gpu=available.pop(0)
                command=[r['python'],'-B','-u','-m',__package__+'.star_representation_launch','row','--runtime',str(runtime),'--arm',arm,'--gpu',str(gpu)]
                log=(root/'logs'/(arm+'_driver.log')).open('xb')
                child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                active[arm]=dict(process=child,gpu=gpu,log=log)
                state['jobs'][arm]=dict(status='RUNNING',pid=child.pid,gpu=gpu,argv=command,started_at=time.time());_write(path,state)
            for arm,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                item['log'].close();rowpath=root/'rows'/arm/'state.json'
                row=_read(rowpath) if rowpath.is_file() else {}
                state['jobs'][arm].update(status='COMPLETE' if row.get('status')=='ANALYZED_SOURCE_ONLY' and code==0 else 'FAILED',finished_at=time.time(),returncode=code)
                del active[arm];_write(path,state)
            if pending or active:time.sleep(3)
        complete=all(state['jobs'][arm]['status']=='COMPLETE' for arm in ARMS)
        primary=root/'rows/oe_vic/score/complete.json';primary_result=_read(primary) if primary.is_file() else {}
        passed=complete and primary_result.get('arm_source_gate_passed') is True
        _write(root/'selection.json',dict(primary_arm='oe_vic',primary_path='P1',primary_score='energy',all_rows_complete=complete,
            source_gate_passed=passed,eligible_candidates=['oe_vic'] if passed else [],
            next_step='SOURCE_ELIGIBLE_FOR_FROZEN_TARGET_VALIDATION' if passed else 'NO_PROMOTION',
            controls_cannot_replace_primary=True,target_reads=0,final_phase3_complete=False))
        state['status']='ANALYZED_SOURCE_ONLY' if complete else 'PARTIAL_FAILED_PRESERVED'
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc))
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['start','driver','row']);p.add_argument('--runtime',required=True)
    p.add_argument('--arm',choices=ARMS);p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='row':
        if a.arm is None or a.gpu is None:p.error('row requires --arm and --gpu')
        return row_driver(a.runtime,a.arm,a.gpu)
    return dict(start=start,driver=driver)[a.mode](a.runtime)


if __name__=='__main__':main()
