"""Independent source-selected low-label replication; never touches the active owner."""
import argparse
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write
from experiments.practical_full_2x2.cache import atomic
from .plan import EXTRA_RUN
from .runner import launch,worker


def dispatch(rt):
    from .prepare import prepare
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.pamf_repl_completion.runner import capacity,processes,foreign_CPU_score
    if rt['run_id']!=EXTRA_RUN or rt['allowed_gpus']!=[5,6,7]:raise ValueError('Fixed three low-label students and GPUs required')
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    state=dict(status='PREPARING',pid=os.getpid(),code_commit=rt['code_commit'],launch_owner=rt['launch_owner'],jobs={})
    atomic(root/'pipeline_state.json',state);jobs=prepare(rt);by={j['id']:j for j in jobs}
    first=next(j for j in jobs if j['kind']=='train');path=root/'configs/smoke.json';write(path,dict(rt,smoke_train_config=first['config']))
    job=dict(id='smoke',kind='smoke',config=str(path),execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_peer_supervised.runner')
    child,item=launch(rt,job,None);state.update(status='SMOKE_RUNNING',smoke=item);atomic(root/'pipeline_state.json',state)
    code=child.wait()
    if code or not (root/'smoke.json').exists() or read(root/'smoke.json')['status']!='PASS':
        state.update(status='STOPPED_FOR_SMOKE_FAILURE');atomic(root/'pipeline_state.json',state);return
    active={};state.update(status='RUNNING',jobs={j['id']:dict(status='QUEUED') for j in jobs});stop=False
    while True:
        for key,(child,item) in list(active.items()):
            code=child.poll()
            if code is None:continue
            good=code==0 and done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code)
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
                if any(v[1]['gpu']==gpu for v in active.values()):continue
                pids,free=capacity(gpu)
                if len(pids|{p for p,g in pending.items() if g==gpu})>=2 or free<10000:continue
            child,item=launch(rt,job,gpu);active[key]=(child,item);state['jobs'][key]=item
            if gpu is not None:pending[child.pid]=gpu
        state['updated_at']=time.time();atomic(root/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(15)
    if stop:state['status']='STOPPED_FOR_TECHNICAL_FAILURE'
    else:
        from .results import analyze
        analyze(rt);state['status']='ANALYZED'
    atomic(root/'pipeline_state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('dispatch','worker'));p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
