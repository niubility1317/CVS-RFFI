"""One owner for adopted healthy F workers, matching G and truth-last evaluation."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .io import read,write
from .source_queue import capacity,allow_shared


def alive(pid):
    p=Path('/proc')/str(pid)/'stat'
    try:return p.read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def final(job):
    return Path(job['completion'])


def done(job):
    path=final(job)
    return path.is_file() and read(path).get('status') in job['successful_statuses']


def ready(job):
    return all(Path(p).is_file() for p in job.get('dependencies',[]))


def block_failed_dependencies(jobs,state,failed_paths):
    blocked=set(failed_paths)
    blocked.update(j['completion'] for j in jobs if state['jobs'][j['id']]['status'] in ('FAILED','BLOCKED_TECHNICAL_DEPENDENCY'))
    changed=True
    while changed:
        changed=False
        for job in jobs:
            if state['jobs'][job['id']]['status']=='QUEUED' and any(p in blocked for p in job.get('dependencies',[])):
                state['jobs'][job['id']]=dict(status='BLOCKED_TECHNICAL_DEPENDENCY',failed_inputs=[p for p in job.get('dependencies',[]) if p in blocked])
                blocked.add(job['completion']);changed=True


def run(rt):
    root=Path(rt['control_root']);root.mkdir(parents=True,exist_ok=False)
    jobs=read(rt['jobs'])
    if len({j['id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate continuation jobs')
    previous=read(rt['source_state'])
    adopted={k:v for k,v in previous['rows'].items() if v['status']=='RUNNING' and alive(v['pid'])}
    pid=previous['pid'];process=Path('/proc')/str(pid)
    # Only this source dispatcher is replaced. Its detached workers continue
    # unchanged, and are adopted from the exact row/config mapping below.
    if alive(pid):
        argv=(process/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
        if os.readlink(process/'cwd')!=rt['source_release'] or 'source_queue dispatch' not in argv or rt['source_runtime'] not in argv:
            raise ValueError('Source controller identity mismatch; no process stopped')
        import signal
        os.kill(pid,signal.SIGTERM)
        for _ in range(100):
            if not alive(pid):break
            time.sleep(.1)
        if alive(pid):raise RuntimeError('Source dispatcher still alive; do not create second owner')
    # Reconcile the stopped dispatcher's pre-CUDA/state-write gap from exact
    # config paths. Detached healthy workers are never stopped or relaunched.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or not alive(int(proc.name)):continue
        try:
            args=proc.joinpath('cmdline').read_bytes().split(bytes([0]))
            argv=[a.decode() for a in args if a]
            if 'experiments.cric_dura_comparison.source_queue' not in argv or 'worker' not in argv or '--config' not in argv:continue
            cfg=argv[argv.index('--config')+1]
            if not cfg.startswith(rt['source_run_root']+'/F_source_configs/'):continue
            env=dict(item.split(b'=',1) for item in proc.joinpath('environ').read_bytes().split(bytes([0])) if b'=' in item)
            c=read(cfg);key=c['run_id']+'/'+c['row_id']
            adopted[key]=dict(status='RUNNING',pid=int(proc.name),gpu=int(env[b'CUDA_VISIBLE_DEVICES']),config=cfg,
                log=rt['source_log_root']+'/'+c['run_id']+'__'+c['row_id']+'.log')
        except (FileNotFoundError,ProcessLookupError):continue
    state=dict(status='RUNNING',pid=os.getpid(),owner=rt['launch_owner'],jobs={},adopted=len(adopted),
        source_model_commit=rt['source_commit'],continuation_commit=rt['code_commit'],
        shared_GPU_policy='one own GPU worker/card; at most two total; free>=12000MiB; timing exclusive')
    by_id={j['id']:j for j in jobs};active={};logs={};children={};failures=Counter()
    for key,item in adopted.items():
        if key not in by_id or by_id[key]['kind']!='F':raise ValueError('Unregistered adopted row')
        if not alive(item['pid']):
            state['jobs'][key]=dict(status='COMPLETE' if done(by_id[key]) else 'FAILED',adopted=True,log=item['log'])
            if not done(by_id[key]):failures['adopted_source_missing_completion']+=1
            continue
        cfg=by_id[key]['config']
        child=Path('/proc')/str(item['pid'])
        try:command=(child/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
        except (FileNotFoundError,ProcessLookupError):
            state['jobs'][key]=dict(status='COMPLETE' if done(by_id[key]) else 'FAILED',adopted=True,log=item['log'])
            if not done(by_id[key]):failures['adopted_source_missing_completion']+=1
            continue
        if not command and not alive(item['pid']):
            state['jobs'][key]=dict(status='COMPLETE' if done(by_id[key]) else 'FAILED',adopted=True,log=item['log'])
            if not done(by_id[key]):failures['adopted_source_missing_completion']+=1
            continue
        if cfg not in command:raise ValueError('Adopted worker config mismatch')
        active[item['gpu']]=(key,item['pid'])
        state['jobs'][key]=dict(status='RUNNING',pid=item['pid'],gpu=item['gpu'],adopted=True,log=item['log'])
    write(Path(rt['source_run_root'])/'continuation_owner.json',dict(pid=os.getpid(),control_root=str(root),runtime=rt['runtime_file'],source_dispatcher_stopped=pid,healthy_workers_adopted=list(adopted)))
    write(root/'adoption.json',dict(status='ADOPTED_HEALTHY_SOURCE_WORKERS',source_pid=pid,workers=adopted))
    for job in jobs:
        if job['id'] not in state['jobs']:
            state['jobs'][job['id']]=dict(status='COMPLETE' if done(job) else 'QUEUED')
    kinds=('G','F','G_student');turn=0;scorer=None;cache_failures={}
    while True:
        for gpu,(key,pid) in list(active.items()):
            if alive(pid):continue
            job=by_id[key]
            state['jobs'][key]['status']='COMPLETE' if done(job) else 'FAILED'
            if key in logs:logs.pop(key).close()
            if key in children:children.pop(key).poll()
            if not done(job):
                path=state['jobs'][key]['log'];tail=Path(path).read_text(errors='replace')[-3000:]
                fingerprint=tail.splitlines()[-1] if tail.splitlines() else 'empty log'
                failures[fingerprint]+=1;state['jobs'][key]['failure']=fingerprint
            del active[gpu]
        if scorer is not None and not alive(scorer[1]):
            key,_=scorer;job=by_id[key]
            state['jobs'][key]['status']='COMPLETE' if done(job) else 'FAILED'
            logs.pop(key).close();children.pop(key).poll()
            if not done(job):
                tail=Path(state['jobs'][key]['log']).read_text(errors='replace')[-3000:]
                failures[tail.splitlines()[-1] if tail.splitlines() else 'empty scorer log']+=1
            scorer=None
        failed_cache_paths=[]
        for dataset in ('ManySig','ManyRx'):
            receipt=read(Path(rt['run_root'])/('observations-'+dataset+'-submit.json'))
            complete=Path(rt['run_root'])/'g_data'/dataset/'complete.json'
            if not complete.is_file() and not alive(receipt['pid']):
                failed_cache_paths.append(str(complete))
                if dataset not in cache_failures:
                    tail=(Path(rt['run_root'])/('observations-'+dataset+'.log')).read_text(errors='replace')[-3000:]
                    fingerprint=tail.splitlines()[-1] if tail.splitlines() else 'empty observation log'
                    cache_failures[dataset]=fingerprint;failures[fingerprint]+=1
        block_failed_dependencies(jobs,state,failed_cache_paths)
        systemic=bool(failures and max(failures.values())>=2)
        pending=[j for j in jobs if state['jobs'][j['id']]['status']=='QUEUED']
        eligible=[j for j in pending if ready(j)] if not systemic else []
        if scorer is None:
            score=next((j for j in eligible if j['kind']=='score'),None)
            if score:
                child,log,path=launch(rt,score,None)
                scorer=(score['id'],child.pid);children[score['id']]=child;logs[score['id']]=log
                state['jobs'][score['id']]=dict(status='RUNNING',pid=child.pid,gpu=None,log=path)
                eligible.remove(score)
        for gpu in rt['allowed_gpus']:
            if gpu in active:continue
            apps,free=capacity(gpu)
            if not allow_shared(apps,free):continue
            candidates=[j for j in eligible if j['kind']!='score' and (not j.get('exclusive') or (apps==0 and gpu not in (2,3)))]
            if not candidates:continue
            # Predict frozen rows as soon as matching G exists. Interleave
            # method, restorer and mechanism training without score feedback.
            job=next((j for j in candidates if j.get('exclusive')),None)
            if job is None:job=next((j for j in candidates if j['kind']=='predict'),None)
            if job is None:
                wanted=kinds[turn%len(kinds)];turn+=1
                job=next((j for j in candidates if j['kind']==wanted),candidates[0])
            child,log,path=launch(rt,job,gpu)
            key=job['id'];active[gpu]=(key,child.pid);children[key]=child;logs[key]=log
            state['jobs'][key]=dict(status='RUNNING',pid=child.pid,gpu=gpu,log=path)
            eligible.remove(job)
            deadline=time.monotonic()+40
            while child.poll() is None and time.monotonic()<deadline:
                apps,_=capacity(gpu)
                if str(child.pid) in subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split():break
                time.sleep(.5)
        state.update(counts=dict(Counter(v['status'] for v in state['jobs'].values())),failures=dict(failures),cache_failures=cache_failures,systemic_stop_new_launches=systemic)
        write(root/'pipeline_state.json',state,exclusive=False)
        if not active and scorer is None and (not pending or systemic):break
        time.sleep(30)
    state['status']='COMPLETE' if all(v['status']=='COMPLETE' for v in state['jobs'].values()) else 'PARTIAL_TECHNICAL_FAILURE'
    write(root/'pipeline_state.json',state,exclusive=False)


def launch(rt,job,gpu):
    path=Path(rt['control_root'])/'logs'/(job['log_name']+'.log');path.parent.mkdir(exist_ok=True)
    log=path.open('x')
    if job['kind']=='F':
        cmd=[rt['python'],'-u','-m','experiments.cric_dura_comparison.source_queue','worker','--config',job['config']]
        cwd=rt['source_release']
    else:
        cmd=[rt['python'],'-u','-m','experiments.cric_dura_comparison.pipeline','worker','--config',job['config']]
        cwd=rt['release_root']
    env=dict(os.environ,PYTHONPATH=cwd,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    child=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return child,log,str(path)


def worker(c):
    from .bootstrap import activate,numerical_policy
    activate(c);numerical_policy()
    if c['stage']!='score':
        import torch
        torch.cuda.init();torch.cuda.set_per_process_memory_fraction(.45)
        reserve=torch.empty(1,device='cuda:0')
    if c['stage']=='G':
        from .g_train import train
        train(c)
    elif c['stage']=='G_student':
        from .g_student import train
        train(c)
    elif c['stage']=='predict':
        from .evaluation import predict
        predict(c)
    elif c['stage']=='score':
        from .evaluation import score
        score(c)
    elif c['stage']=='timing':
        from .timing import run as timing
        timing(c)
    else:raise ValueError('Unregistered continuation stage')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['dispatch','worker']);p.add_argument('--config',required=True)
    a=p.parse_args();(run if a.mode=='dispatch' else worker)(read(a.config))
