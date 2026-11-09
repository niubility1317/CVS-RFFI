"""One immutable owner, bounded GPU resources and truth-last worker pipeline."""
import argparse
from collections import Counter
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from experiments.cric_dura_comparison.io import read,write
from experiments.pamf_branch_factorial.runner import resource_sample,resource_guard
from .plan import RUN, limits


def can_start(gpu, own, free_mib):
    return own < limits(gpu) and free_mib >= 12000


def worker(c,kind):
    if kind=='score':
        from .evaluation import score
        return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind=='smoke':
        from .smoke import smoke
        return smoke(c)
    if kind=='train':
        from .model import train
        return train(c)
    if kind=='predict':
        from .evaluation import predict
        return predict(c)
    raise ValueError('Unknown worker')


def dispatch(rt):
    from experiments.cric_dura_comparison.pipeline import done,ready,block_failed_dependencies
    from experiments.practical_full_2x2.cache import atomic
    from experiments.pamf_repl_completion.runner import capacity,foreign_CPU_score
    from .prepare import prepare
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir()
    state=dict(status='PREPARING',pid=os.getpid(),cwd=os.getcwd(),code_commit=rt['code_commit'],owner=rt['launch_owner'],jobs={})
    atomic(root/'pipeline_state.json',state)
    try: jobs=prepare(rt)
    except Exception as e:
        state.update(status='FAILED_PREPARATION',error=repr(e),ended_at=time.time());atomic(root/'pipeline_state.json',state);raise
    by={j['id']:j for j in jobs}; active={}; failures=Counter()
    state.update(status='SMOKE_PENDING',jobs={j['id']:dict(status='QUEUED') for j in jobs})
    atomic(root/'pipeline_state.json',state)
    def launch(job,gpu):
        path=root/'logs'/(job['id']+'.log')
        cmd=[rt['python'],'-u','-m','experiments.pamf_split_spectrum_paper.runner','worker','--kind',job['kind'],'--config',job['config']]
        env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                 OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with path.open('x') as log:
            p=subprocess.Popen(cmd,cwd=rt['release_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        return p,dict(status='RUNNING',pid=p.pid,gpu=gpu,cwd=rt['release_root'],argv=cmd,log=str(path),started_at=time.time())
    previous=resource_sample();time.sleep(1)
    while True:
        current=resource_sample();guard=resource_guard(previous,current);previous=current
        gpu=next((g for g in range(8) if guard['allowed'] and capacity(g)[1]>=12000),None)
        if gpu is not None: break
        time.sleep(15)
    p,item=launch(dict(id='smoke',kind='smoke',config=rt['runtime_file']),gpu);state['smoke']=item
    atomic(root/'pipeline_state.json',state);status=p.wait()
    state['smoke'].update(status='PASS' if status==0 else 'FAILED',exit_code=status)
    if status or read(root/'smoke.json')['status']!='PASS':
        state['status']='FAILED_NO_QUERY_SMOKE';atomic(root/'pipeline_state.json',state);return
    state['status']='RUNNING';turn=0
    while True:
        current=resource_sample();guard=resource_guard(previous,current);previous=current
        for key,(p,item) in list(active.items()):
            if p.poll() is None:continue
            ok=done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if ok else 'FAILED',ended_at=time.time())
            if not ok:
                lines=Path(item['log']).read_text(errors='replace').splitlines()
                reason=re.sub(r's202609270[1-8]','sSEED',lines[-1] if lines else 'missing completion')
                failures[reason]+=1;state['jobs'][key]['failure']=reason
            del active[key]
        block_failed_dependencies(jobs,state,[])
        systemic=any(n>=2 or any(x in reason for x in ('CONTRACT_MISMATCH','PROVENANCE','Truth closed','U truth','Output collision'))
                     for reason,n in failures.items())
        pending=[j for j in jobs if state['jobs'][j['id']]['status']=='QUEUED']
        eligible=[j for j in pending if ready(j)] if not systemic and guard['allowed'] else []
        score=next((j for j in eligible if j['kind']=='score'),None)
        if score and not any(by[k]['kind']=='score' for k in active) and not foreign_CPU_score({p.pid for p,_ in active.values()}):
            p,item=launch(score,None);active[score['id']]=(p,item);state['jobs'][score['id']]=item;eligible.remove(score)
        for gpu in range(8):
            own=sum(item['gpu']==gpu for p,item in active.values());_,free=capacity(gpu)
            if not can_start(gpu,own,free):continue
            available=[j for j in eligible if j['kind']!='score']
            if not available:continue
            preferred='predict' if turn%3==2 else 'train';turn+=1
            job=next((j for j in available if j['kind']==preferred),available[0]);p,item=launch(job,gpu)
            active[job['id']]=(p,item);state['jobs'][job['id']]=item;eligible.remove(job)
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),
                     failures=dict(failures),systemic_stop_new_launches=systemic,resource_guard=guard,
                     own_gpu_limits={'0-6':3,'7':1},other_owners_counted=False)
        atomic(root/'pipeline_state.json',state)
        if not active and (not pending or systemic):break
        time.sleep(15)
    if all(v['status']=='COMPLETE' for v in state['jobs'].values()):
        state['status']='ANALYZING';atomic(root/'pipeline_state.json',state)
        from .analyze import run
        run(rt);state['status']='ANALYZED'
    else: state['status']='PARTIAL_TECHNICAL_FAILURE'
    state['ended_at']=time.time();atomic(root/'pipeline_state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--config',required=True);p.add_argument('--kind');a=p.parse_args()
    c=read(a.config); dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
