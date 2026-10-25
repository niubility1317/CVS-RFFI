"""One owner for matched continuations and immediate frozen truth-last testing."""
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import time
from experiments.cric_dura_comparison.io import read
from experiments.practical_full_2x2.cache import atomic
from .training_plan import TRAIN_RUN


def worker(c,kind):
    if kind=='score':
        from .evaluation import score
        return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind=='train':
        from .training import train
        return train(c)
    if kind=='predict':
        from .evaluation import predict
        return predict(c)
    if kind=='smoke':
        from .training import smoke
        return smoke(c,read(c['smoke_train_config']))
    raise ValueError('Unknown ChannelPL worker')


def launch(rt,job,gpu):
    log=Path(rt['run_root'])/'logs'/(job['id']+'.log');log.parent.mkdir(exist_ok=True)
    command=[rt['python'],'-u','-m','experiments.pamf_channel_pl.training_runner','worker',
        '--kind',job['kind'],'--config',job['config']]
    env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with log.open('x') as stream:
        child=subprocess.Popen(command,cwd=rt['release_root'],env=env,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    return child,dict(status='RUNNING',pid=child.pid,cwd=rt['release_root'],argv=command,gpu=gpu,log=str(log),started_at=time.time())


def dispatch(rt):
    from .prepare_training import prepare
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.pamf_repl_completion.runner import capacity,processes,foreign_CPU_score
    if rt['run_id']!=TRAIN_RUN or rt['allowed_gpus']!=[5,6,7]:raise ValueError('Fixed candidate/resources required')
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    state=dict(status='PREPARING',pid=os.getpid(),code_commit=rt['code_commit'],launch_owner=rt['launch_owner'],started_at=time.time(),jobs={})
    atomic(root/'pipeline_state.json',state)
    jobs=prepare(rt);by={j['id']:j for j in jobs};first=next(j for j in jobs if j['kind']=='train')
    smoke_rt=dict(rt,smoke_train_config=first['config']);path=root/'configs/smoke.json';atomic(path,smoke_rt)
    state['status']='WAITING_FOR_SMOKE_GPU';atomic(root/'pipeline_state.json',state)
    while True:
        pids,free=capacity(5);pending=processes()
        if len(pids|{p for p,g in pending.items() if g==5})<3 and free>=10000:break
        time.sleep(15)
    child,item=launch(rt,dict(id='smoke',kind='smoke',config=str(path)),5)
    state.update(status='SMOKE_RUNNING',smoke=item);atomic(root/'pipeline_state.json',state)
    child.wait()
    if not (root/'smoke.json').exists() or read(root/'smoke.json')['status']!='PASS':
        state.update(status='STOPPED_FOR_SMOKE_FAILURE',ended_at=time.time());atomic(root/'pipeline_state.json',state);return
    state.update(status='RUNNING',jobs={j['id']:dict(status='QUEUED') for j in jobs});active={};stop=False
    while True:
        for key,(child,item) in list(active.items()):
            code=child.poll()
            if code is None:continue
            good=code==0 and done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code,ended_at=time.time())
            if not good:stop=True
            del active[key]
        pending=processes()
        for job in sorted(jobs,key=lambda j:dict(score=0,predict=1,train=2)[j['kind']]):
            key=job['id']
            if stop or state['jobs'][key]['status']!='QUEUED' or not ready(job):continue
            if job['kind']=='score':
                if any(by[k]['kind']=='score' for k in active) or foreign_CPU_score({v[1]['pid'] for v in active.values()}):continue
                gpu=None
            else:
                gpu=job['gpu']
                if sum(v[1]['gpu']==gpu for v in active.values())>=2:continue
                pids,free=capacity(gpu)
                if len(pids|{p for p,g in pending.items() if g==gpu})>=3 or free<10000:continue
            child,item=launch(rt,job,gpu);active[key]=(child,item);state['jobs'][key]=item
            if gpu is not None:pending[child.pid]=gpu
        state['updated_at']=time.time();atomic(root/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(15)
    if stop:
        state.update(status='STOPPED_FOR_TECHNICAL_FAILURE',ended_at=time.time());atomic(root/'pipeline_state.json',state);return
    # Scores never enter source training or admission decisions. Aggregate only
    # after all predeclared rows finish, in this CPU-only controller.
    from .results import analyze
    analyze(rt)
    state.update(status='ANALYZED',ended_at=time.time());atomic(root/'pipeline_state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('dispatch','worker'))
    p.add_argument('--kind',choices=('train','predict','score','smoke'));p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config)
    dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
