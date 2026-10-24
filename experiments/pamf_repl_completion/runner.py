"""One immutable owner, all eight GPUs, bounded admission and truth-last CPU work."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from experiments.cric_dura_comparison.io import read,write

def activate(c):
    from experiments.cric_dura_comparison.bootstrap import activate
    from experiments.practical_full_2x2.compat import numerical_policy
    activate(c);numerical_policy()

def worker(c,kind):
    # Independent CPU scoring intentionally has no model/source release handles.
    if kind=='score':
        from .evaluation import score
        return score(c)
    activate(c)
    if kind in ('prefix','F'):
        from .source import train
        return train(c,prefix_only=kind=='prefix')
    if kind=='predict':
        from .evaluation import predict
        return predict(c)
    if kind=='timing':
        from .timing import run
        return run(c)
    raise ValueError('Unknown stage')

def smoke(rt):
    activate(rt)
    from .plan import make_plan,configuration,validate_config
    from .source import parent
    from .objectives import objective,portable_scope,loader_scope
    from experiments.pamf_ssl_research.hard_pl_worker import smoke
    context=read(rt['context']);row=next(r for r in make_plan()['rows'] if r['protocol_id']=='F0' and r['method']=='REPL')
    cfg=configuration(context,row,rt['project'],rt['code_commit']);path=Path(rt['run_root'])/'configs/smoke_train.json';write(path,cfg)
    def parents(c,validate_config=validate_config):
        validate_config(c);prefix=context['sources'][str(c['model_seed'])]['F21']+'/segment20'
        return [parent(c,prefix)],read(c['source_contract'])
    with portable_scope(),loader_scope():smoke(dict(rt,rows=[dict(train_config=str(path),method='PAMF-REPL')]),validate_config=validate_config,max_rows=1,
        objective_cls=objective,source_parents_fn=parents,parent_epoch=20)

def processes():
    """Count our user's pending GPU workers as well as all CUDA compute PIDs."""
    result={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
            if 'python' not in args or 'experiments.' not in args:continue
            env=dict(x.split(b'=',1) for x in p.joinpath('environ').read_bytes().split(b'\0') if b'=' in x)
            gpu=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode()
            if gpu not in map(str,range(8)):continue
            if p.joinpath('stat').read_text().split()[2]=='Z':continue
            result[int(p.name)]=int(gpu)
        except (PermissionError,FileNotFoundError,ProcessLookupError):continue
    return result

def capacity(gpu):
    ids=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
    free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    return {int(p.strip()) for p in ids if p.strip().isdigit()},free

def foreign_CPU_score(owned):
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) in owned:continue
        try:
            argv=[x.decode() for x in p.joinpath('cmdline').read_bytes().split(b'\0') if x]
            text=' '.join(argv)
            if 'python' not in text or 'experiments.' not in text:continue
            if ' score ' in text or '--kind score' in text:return True
            if '--config' in argv:
                path=Path(argv[argv.index('--config')+1])
                if path.is_file() and read(path).get('stage')=='score':return True
        except (PermissionError,FileNotFoundError,ProcessLookupError,ValueError,IndexError):continue
    return False

def launch(rt,job,gpu):
    path=Path(rt['run_root'])/'logs'/(job['id']+'.log');path.parent.mkdir(exist_ok=True)
    cmd=[rt['python'],'-u','-m','experiments.pamf_repl_completion.runner','worker','--kind',job['kind'],'--config',job['config']]
    release=job.get('execution_release',rt['release_root'])
    env=dict(os.environ,PYTHONPATH=release,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with path.open('x') as log:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return child,dict(status='RUNNING',pid=child.pid,gpu=gpu,cwd=release,argv=cmd,log=str(path),started_at=time.time())

def dispatch(rt):
    from .prepare import prepare
    from experiments.practical_full_2x2.cache import atomic
    from experiments.cric_dura_comparison.pipeline import done,ready,block_failed_dependencies
    from experiments.cric_dura_comparison.resource_balance import LoadPolicy
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    state=dict(status='PREPARING',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,code_commit=rt['code_commit'],
        owner='codex-pamf-repl-completion-20261024',started_at=time.time(),jobs={})
    atomic(root/'pipeline_state.json',state)
    if rt.get('recover_previous_runtime'):
        from .recovery import prepare as recovery_prepare
        jobs,oldstate=recovery_prepare(rt)
    else:jobs=prepare(rt);oldstate=None
    by={j['id']:j for j in jobs}
    # No F or target prediction is launched before the one actual-checkpoint smoke.
    gpu=None
    if oldstate is not None:
        from .recovery import attach
        active,children,failures=attach(jobs,oldstate)
        state.update(status='RUNNING',jobs=oldstate['jobs'],recovered_from=rt['recover_previous_runtime'],
            resolved_technical_failures=oldstate['resolved_technical_failures'])
        atomic(root/'pipeline_state.json',state)
    else:
        active={};children={};failures=Counter()
    while oldstate is None and gpu is None:
        pending=processes()
        for g in rt['allowed_gpus']:
            pids,free=capacity(g);count=len(pids|{p for p,v in pending.items() if v==g})
            if count<3 and free>=12000:gpu=g;break
        if gpu is None:time.sleep(20)
    if oldstate is None:
        cmd=[rt['python'],'-u','-m','experiments.pamf_repl_completion.runner','smoke','--config',rt['runtime_file']]
        env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with (root/'smoke.log').open('x') as log:subprocess.run(cmd,cwd=rt['release_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        if read(root/'smoke.json')['status']!='PASS':raise ValueError('Source smoke did not pass')
        state.update(status='RUNNING',jobs={j['id']:dict(status='QUEUED') for j in jobs});atomic(root/'pipeline_state.json',state)
    turn=0
    # Reuse the established host telemetry. This owner never uses its fourth-slot pilot.
    policy=LoadPolicy(dict(boost_gpus=[0,6],max_gpu_workers=26))
    while True:
        for key,item in list(active.items()):
            child=children[key];status=child.poll()
            if status is None:continue
            ok=done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if ok else 'FAILED',ended_at=time.time())
            if not ok:
                lines=Path(item['log']).read_text(errors='replace')[-7000:].splitlines();reason=lines[-1] if lines else 'missing completion'
                reason=re.sub(r's202609270[123]','sSEED',reason);failures[reason]+=1;state['jobs'][key]['failure']=reason
            del active[key];del children[key]
        block_failed_dependencies(jobs,state,[]);systemic=bool(failures and max(failures.values())>=2)
        pending=[j for j in jobs if state['jobs'][j['id']]['status']=='QUEUED']
        candidates=[j for j in pending if ready(j)] if not systemic else []
        score=next((j for j in candidates if j['kind']=='score'),None)
        if score and not any(by[k]['kind']=='score' for k in active) and not foreign_CPU_score({p.pid for p in children.values()}):
            child,item=launch(rt,score,None);active[score['id']]=item;children[score['id']]=child;state['jobs'][score['id']]=item;candidates.remove(score)
        policy.poll(by);foreign=processes()
        # One admission/card/iteration reserves new PIDs before their first CUDA call.
        for gpu in policy.order(rt['allowed_gpus']):
            if policy.last_error or not policy.host_ok():continue
            owned=[by[k] for k,v in active.items() if v['gpu']==gpu]
            if len(owned)>=3 or any(j.get('exclusive') for j in owned):continue
            pids,free=capacity(gpu);reserved={p for p,g in foreign.items() if g==gpu}|{v['pid'] for v in active.values() if v['gpu']==gpu}
            if len(pids|reserved)>=3 or free<12000:continue
            allpids={p for d in policy.devices.values() for p in d.get('cuda_pids',[])}|set(foreign)|{v['pid'] for v in active.values() if v['gpu'] is not None}
            if len(allpids)>=24:continue
            ownpids={v['pid'] for v in active.values() if v['gpu']==gpu}
            available=[j for j in candidates if j['kind']!='score' and not(j.get('heavy_G') and (any(x.get('heavy_G') for x in owned) or bool((pids|reserved)-ownpids)))
                and not(j.get('exclusive') and (owned or pids or reserved))]
            if not available:continue
            wanted=('prefix','F','F','predict')[turn%4];turn+=1
            job=next((j for j in available if j['kind']==wanted),available[0])
            budget=lambda j:10240 if j.get('heavy_G') else 4096 if j['kind'] in ('prefix','F') else 2048
            total_mem=policy.devices.get(gpu,{}).get('total_mib',0)
            if not job.get('exclusive') and sum(budget(j) for j in owned)+budget(job)+3072>total_mem:continue
            child,item=launch(rt,job,gpu);active[job['id']]=item;children[job['id']]=child;state['jobs'][job['id']]=item;candidates.remove(job)
            foreign[item['pid']]=gpu
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),
            failures=dict(failures),systemic_stop_new_launches=systemic,GPU_workers=sum(v['gpu'] is not None for v in active.values()),
            host=policy.host,telemetry_error=policy.last_error,allowed_gpus=list(range(8)))
        atomic(root/'pipeline_state.json',state)
        if not active and (not pending or systemic):break
        time.sleep(12)
    if any(v['status']!='COMPLETE' for v in state['jobs'].values()):
        state['status']='PARTIAL_TECHNICAL_FAILURE'
    else:
        state['status']='ANALYZING';atomic(root/'pipeline_state.json',state)
        from .analyze import run
        run(rt,jobs);state['status']='ANALYZED'
    state['ended_at']=time.time();atomic(root/'pipeline_state.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','smoke','worker']);p.add_argument('--config',required=True);p.add_argument('--kind')
    a=p.parse_args();c=read(a.config)
    if a.action=='dispatch':dispatch(c)
    elif a.action=='smoke':smoke(c)
    else:worker(c,a.kind)
