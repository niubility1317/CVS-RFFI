"""Independent all-GPU owner; source cache, real backward smoke, row tests."""
import argparse
from collections import Counter
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read
from experiments.practical_full_2x2.cache import atomic
from experiments.pamf_peer_supervised.runner import launch
from .plan import RUN


def worker(c,kind):
    if kind=='score':
        from experiments.pamf_channel_pl.evaluation import score
        return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    import torch
    torch.set_num_threads(2)
    if kind=='cache':
        from .cache import build
        return build(c)
    if kind=='smoke':
        from .training import smoke
        return smoke(read(c['train_config']),c['output'])
    if kind=='train':
        from .training import train
        return train(c)
    if kind=='predict':
        from experiments.pamf_channel_pl.evaluation import predict
        return predict(c)
    raise ValueError('Unknown fixed restored-PL stage')


def dispatch(rt):
    from .prepare import prepare
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.pamf_repl_completion.runner import capacity,processes,foreign_CPU_score
    if rt['run_id']!=RUN or rt['allowed_gpus']!=list(range(8)):raise ValueError('Fixed independent all-GPU matrix required')
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);jobs=prepare(rt);by={j['id']:j for j in jobs}
    state=dict(status='RUNNING',pid=os.getpid(),code_commit=rt['code_commit'],launch_owner=rt['launch_owner'],
        jobs={j['id']:dict(status='QUEUED') for j in jobs},started_at=time.time())
    atomic(root/'pipeline_state.json',state);active={};stop=False
    while True:
        for key,(child,item) in list(active.items()):
            code=child.poll()
            if code is None:continue
            good=code==0 and done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code)
            if not good:stop=True
            if good and by[key]['kind']=='score':
                from .results import partial
                partial(rt)
            del active[key]
        pending=processes()
        order=dict(cache=0,smoke=1,score=2,predict=3,train=4)
        for job in sorted(jobs,key=lambda j:order[j['kind']]):
            key=job['id']
            if stop or state['jobs'][key]['status']!='QUEUED' or not ready(job):continue
            gpu=None
            if job['kind']=='score':
                if any(by[k]['kind']=='score' for k in active) or foreign_CPU_score({v[1]['pid'] for v in active.values()}):continue
            else:
                for candidate in job['gpu_pool']:
                    if sum(v[1]['gpu']==candidate for v in active.values())>=2:continue
                    pids,free=capacity(candidate)
                    if len(pids|{pid for pid,g in pending.items() if g==candidate})<2 and free>=10000:
                        gpu=candidate;break
                if gpu is None:continue
            child,item=launch(rt,job,gpu);active[key]=(child,item);state['jobs'][key]=item
            if gpu is not None:pending[child.pid]=gpu
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),systemic_stop_new_launches=stop)
        atomic(root/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    if not stop:
        from .results import partial
        partial(rt,require_all=True)
    state['status']='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'ANALYZED'
    atomic(root/'pipeline_state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
