"""One priority owner across all GPUs; frozen source training then immediate truth-last tests."""
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import time
from experiments.cric_dura_comparison.io import read,write
from experiments.practical_full_2x2.cache import atomic
from .plan import RUN


def worker(c,kind):
    if kind=='score':
        from experiments.pamf_channel_pl.evaluation import score
        return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind=='train':
        from .training import train
        return train(c)
    if kind=='predict':
        from experiments.pamf_channel_pl.evaluation import predict
        return predict(c)
    if kind=='smoke':
        import torch
        torch.set_num_threads(2)
        from .training import smoke
        return smoke(c,read(c['smoke_train_config']),device='cpu')
    raise ValueError('Unknown peer worker')


def launch(rt,job,gpu):
    release=job['execution_release'];folder=Path(job.get('log_root',str(Path(rt['run_root'])/'logs')))
    folder.mkdir(exist_ok=True);log=folder/(job.get('original_id',job['id'])+'.log')
    command=[job['execution_python'],'-u','-m',job['entrypoint'],'worker','--kind',job['kind'],'--config',job['config']]
    env=dict(os.environ,PYTHONPATH=release,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with log.open('x') as stream:
        child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    return child,dict(status='RUNNING',pid=child.pid,cwd=release,argv=command,gpu=gpu,log=str(log),started_at=time.time())


def dispatch(rt):
    from .prepare import prepare
    from .priority import handoff,defer,mirror_old
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.pamf_repl_completion.runner import capacity,processes,foreign_CPU_score
    if rt['run_id']!=RUN or rt['allowed_gpus']!=list(range(8)):raise ValueError('Fixed all-GPU research matrix required')
    context=read(rt['context']);rt=dict(rt,previous_runtime=context['previous_runtime'],
        queue_priority_authorization=context['queue_priority_authorization'])
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    state=dict(status='PREPARING',pid=os.getpid(),code_commit=rt['code_commit'],launch_owner=rt['launch_owner'],started_at=time.time(),jobs={})
    atomic(root/'pipeline_state.json',state);newjobs=prepare(rt)
    first=next(j for j in newjobs if j['kind']=='train');path=root/'configs/smoke.json'
    write(path,dict(rt,smoke_train_config=first['config']))
    smokejob=dict(id='smoke',kind='smoke',config=str(path),execution_release=rt['release_root'],
        execution_python=rt['python'],entrypoint='experiments.pamf_peer_supervised.runner')
    child,item=launch(rt,smokejob,None);state.update(status='SMOKE_RUNNING',smoke=item);atomic(root/'pipeline_state.json',state)
    code=child.wait()
    if code or not (root/'smoke.json').exists() or read(root/'smoke.json')['status']!='PASS':
        state.update(status='STOPPED_FOR_SMOKE_FAILURE',ended_at=time.time());atomic(root/'pipeline_state.json',state);return
    oldjobs,oldstates,active,children,deferred_rows,oldrt=handoff(rt)
    jobs=newjobs+oldjobs;by={j['id']:j for j in jobs};write(root/'control_jobs.json',jobs)
    state.update(status='RUNNING',jobs=dict({j['id']:dict(status='QUEUED') for j in newjobs},**oldstates),
        deferred_old_rows=sorted(deferred_rows),healthy_old_workers_adopted=len(active),allowed_gpus=rt['allowed_gpus'])
    atomic(root/'pipeline_state.json',state);stop=False;new_analyzed=False;old_analyzed=False
    while True:
        for key,item in list(active.items()):
            code=children[key].poll()
            if code is None:continue
            good=code==0 and done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code,ended_at=time.time())
            if not good:stop=True
            del active[key];del children[key]
        allnew=all(state['jobs'][j['id']]['status']=='COMPLETE' for j in newjobs)
        if allnew and not new_analyzed:
            from .results import analyze
            analyze(rt);new_analyzed=True;state['new_candidate_status']='ANALYZED'
        pending=processes()
        candidates=[j for j in jobs if state['jobs'][j['id']]['status']=='QUEUED' and ready(j) and not defer(j,allnew,deferred_rows)]
        candidates.sort(key=lambda j:(dict(score=0,predict=1,train=2)[j['kind']],0 if j['lane']=='new' else 1))
        if not stop:
            score=next((j for j in candidates if j['kind']=='score'),None)
            if score and not any(by[k]['kind']=='score' for k in active) and not foreign_CPU_score({v['pid'] for v in active.values()}):
                child,item=launch(rt,score,None);key=score['id'];active[key]=item;children[key]=child;state['jobs'][key]=item
            for gpu in rt['allowed_gpus']:
                if sum(v['gpu']==gpu for v in active.values())>=2:continue
                job=next((j for j in candidates if j['kind']!='score' and j['gpu']==gpu),None)
                if job is None:continue
                pids,free=capacity(gpu);reserved={p for p,g in pending.items() if g==gpu}
                if len(pids|reserved)>=2 or free<10000:continue
                child,item=launch(rt,job,gpu);key=job['id'];active[key]=item;children[key]=child;state['jobs'][key]=item;pending[child.pid]=gpu
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),
            new_counts=dict(Counter(state['jobs'][j['id']]['status'] for j in newjobs)),
            systemic_stop_new_launches=stop,old_extensions_deferred=not allnew)
        atomic(root/'pipeline_state.json',state);mirror_old(state,jobs,oldrt,rt)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    if not stop:
        from experiments.pamf_channel_pl.results import analyze as analyze_old
        analyze_old(oldrt);old_analyzed=True
    state.update(status='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'ANALYZED',ended_at=time.time(),old_analyzed=old_analyzed)
    atomic(root/'pipeline_state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('dispatch','worker'));p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
