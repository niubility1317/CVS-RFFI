"""Requeue precisely identified workers and reserve physical GPU 0 and 1."""
import argparse
from collections import Counter
from copy import deepcopy
import importlib
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from experiments.cric_dura_comparison.io import read, write
from experiments.cric_dura_comparison.pipeline import alive, done, ready, block_failed_dependencies
from experiments.practical_full_2x2.cache import atomic
from experiments.pamf_repl_completion.runner import capacity, foreign_CPU_score
from experiments.pamf_branch_factorial.runner import resource_sample, resource_guard


def dispatch_gpus():
    return range(2,8)


def limits(gpu, gpu7_limit=1):
    return 0 if gpu in (0,1) else gpu7_limit if gpu == 7 else 3


def command(rt, job):
    return [rt['python'],'-u','-m',rt['worker_module'],'worker','--kind',job['kind'],'--config',job['config']]


def matches_dispatcher(rt,cwd,argv):
    return cwd==rt['source_control_release'] and argv==[rt['python'],'-u','-m',rt['source_control_module'],
        '--config',rt['source_control_release']+'/runtime.json']


def matches_worker(rt,job,cwd,argv):
    return cwd==job.get('worker_release',rt['source_release']) and argv==command(rt,job)


def current_workers(rt,jobs):
    byconfig={j['config']:j for j in jobs};found={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            pid=int(p.name)
            if not alive(pid):continue
            cwd=os.readlink(p/'cwd');argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if '--config' not in argv:continue
            job=byconfig.get(argv[argv.index('--config')+1])
            if not job or not matches_worker(rt,job,cwd,argv):continue
            if job['id'] in found:raise ValueError('Duplicate matching worker')
            env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
            gpu=None if job['kind']=='score' else int(env[b'CUDA_VISIBLE_DEVICES'])
            found[job['id']]=dict(status='RUNNING',pid=pid,gpu=gpu,cwd=cwd,argv=argv,
                log=job.get('recovery_log',str(Path(rt['run_root'])/'logs'/(job['id']+'.log'))),adopted=True)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return found


def replacement_plan(job, dependent, source_entry, cfg, dependent_cfg, control):
    """Prepare unique retry paths without changing scientific row identity."""
    job=deepcopy(job);dependent=deepcopy(dependent);source_entry=deepcopy(source_entry)
    cfg=deepcopy(cfg);dependent_cfg=deepcopy(dependent_cfg)
    old=Path(cfg['output_root']);new=old.with_name(old.name+'-gpu01-r01')
    if new.exists():raise FileExistsError('Replacement output exists')
    job['id']+='--gpu01-r01'; job['config']=str(control/'configs'/(job['id']+'.json'))
    job['recovery_log']=str(control/'logs'/(job['id']+'.log'))
    cfg['output_root']=str(new);job['completion']=str(new/'complete.json')
    dependent['config']=str(control/'configs'/(dependent['id']+'--gpu01-r01.json'))
    if job['kind']=='train':
        dependent_cfg['source_output']=str(new);source_entry['source_output']=str(new)
    elif job['kind']=='predict':
        dependent_cfg['prediction_output']=str(new);source_entry['predictions']=str(new)
    else:raise ValueError('Only GPU training or prediction can be moved')
    dependent['dependencies']=[job['completion'] if p==str(old/'complete.json') else p for p in dependent['dependencies']]
    if job['completion'] not in dependent['dependencies']:raise ValueError('Missing downstream dependency')
    return job,dependent,source_entry,cfg,dependent_cfg


def run(rt):
    try:
        return takeover(rt)
    except Exception as error:
        control=Path(rt['control_root'])
        if control.exists():
            snapshot=control/'pipeline_state.json'
            state=read(snapshot) if snapshot.exists() else {}
            state.update(status='FAILED_HANDOFF_RECOVERABLE',error=repr(error),ended_at=time.time())
            atomic(snapshot,state)
        raise


def takeover(rt):
    root=Path(rt['run_root']);control=Path(rt['control_root']);control.mkdir(parents=True,exist_ok=False)
    (control/'configs').mkdir();(control/'logs').mkdir()
    jobs=read(rt['source_jobs']);previous=read(rt['source_control_state']);old=previous['pid'];proc=Path('/proc')/str(old)
    argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    if previous['status']!='RUNNING' or not alive(old) or not matches_dispatcher(rt,os.readlink(proc/'cwd'),argv):
        raise ValueError('Controller identity changed; no process stopped')
    if any(x.get('status')=='RECONCILING_LAUNCH' for x in previous['jobs'].values()):
        raise ValueError('Existing unresolved launch; no process stopped')
    pre_workers=current_workers(rt,jobs);pre_sources=read(root/'sources.json');pre_by={j['id']:j for j in jobs}
    for key,item in pre_workers.items():
        if item['gpu'] not in (0,1):continue
        job=pre_by[key];cfg=read(job['config']);rid=cfg['row_id']
        target='predict' if job['kind']=='train' else 'score' if job['kind']=='predict' else None
        deps=[x for x in jobs if x['kind']==target and read(x['config']).get('row_id')==rid]
        if len(deps)!=1 or previous['jobs'][deps[0]['id']]['status']!='QUEUED':
            raise ValueError('Downstream preflight differs; no process stopped')
        replacement_plan(job,deps[0],pre_sources[rid],cfg,read(deps[0]['config']),control)
        if os.getpgid(item['pid'])!=item['pid'] or os.getsid(item['pid'])!=item['pid']:
            raise ValueError('Worker group preflight differs; no process stopped')
    # Preserve the exact prior queue before stopping its owner, so a failed
    # handoff has a concrete recovery starting point.
    write(control/'jobs.json',jobs)
    write(control/'sources_before.json',read(root/'sources.json'))
    atomic(control/'pipeline_state.json',dict(status='TAKEOVER_PENDING',previous_pid=old,jobs=previous['jobs']))
    write(control/'handoff_started.json',dict(status='TAKEOVER_STARTED',previous_pid=old))
    os.kill(old,signal.SIGTERM)
    for _ in range(100):
        if not alive(old):break
        time.sleep(.1)
    if alive(old):raise RuntimeError('Old controller remains alive')
    previous=read(rt['source_control_state']); observed={}
    for _ in range(13):
        observed.update(current_workers(rt,jobs));time.sleep(1)
    observed={k:v for k,v in observed.items() if alive(v['pid'])}
    state=dict(status='RUNNING',pid=os.getpid(),owner=rt['launch_owner'],code_commit=rt['code_commit'],
        source_model_commit=rt['source_model_commit'],previous_pid=old,jobs={},failures=previous.get('failures',{}))
    failures=Counter(state['failures']);sources=read(root/'sources.json')
    for j in jobs:
        k=j['id'];prior=previous['jobs'][k];log=Path(j.get('recovery_log',str(root/'logs'/(k+'.log'))))
        if k in observed:state['jobs'][k]=observed[k]
        elif done(j):state['jobs'][k]=dict(prior,status='COMPLETE')
        elif ((prior['status']=='QUEUED' and log.exists()) or
              (prior['status']=='RUNNING' and alive(prior.get('pid',-1)))):
            state['jobs'][k]=dict(status='RECONCILING_LAUNCH',log=str(log))
        elif prior['status'] in ('RUNNING','COMPLETE'):
            lines=log.read_text(errors='replace').splitlines() if log.exists() else []
            reason=re.sub(r's202609270[1-8]','sSEED',lines[-1] if lines else 'missing completion')
            failures[reason]+=1;state['jobs'][k]=dict(prior,status='FAILED',failure=reason)
        else:state['jobs'][k]=prior
    atomic(control/'pipeline_state.json',state)
    if any(v['status']=='RECONCILING_LAUNCH' for v in state['jobs'].values()):
        raise RuntimeError('Live worker not reconciled; no worker stopped')
    by={j['id']:j for j in jobs};plans=[]
    for key,item in list(state['jobs'].items()):
        if item['status']!='RUNNING' or item['gpu'] not in (0,1):continue
        job=by[key]
        if job['kind'] not in ('train','predict'):raise ValueError('Unexpected GPU worker type')
        cfg=read(job['config']);rid=cfg['row_id'];target='predict' if job['kind']=='train' else 'score'
        candidates=[x for x in jobs if x['kind']==target and read(x['config']).get('row_id')==rid]
        if len(candidates)!=1 or state['jobs'][candidates[0]['id']]['status']!='QUEUED':
            raise ValueError('Downstream job is not uniquely queued')
        dep=candidates[0];dcfg=read(dep['config'])
        plan=replacement_plan(job,dep,sources[rid],cfg,dcfg,control)
        pid=item['pid'];p=Path('/proc')/str(pid)
        if (not alive(pid) or not matches_worker(rt,job,os.readlink(p/'cwd'),
            [x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]) or
            os.getpgid(pid)!=pid or os.getsid(pid)!=pid):
            raise ValueError('Worker identity or isolated process group differs')
        plans.append(dict(key=key,pid=pid,gpu=item['gpu'],rid=rid,old_output=cfg['output_root'],
                          old_completion=job['completion'],new_job=plan[0],new_dep=plan[1],
                          new_source=plan[2],new_cfg=plan[3],new_dep_cfg=plan[4]))
    write(control/'interrupt_plan.json',plans)
    for number,p in enumerate(plans):
        job=by[p['key']]
        if done(job):
            state['jobs'][p['key']]=dict(status='COMPLETE',completed_before_interrupt=True);continue
        if not alive(p['pid']):raise RuntimeError('Worker exited before interrupt; reconcile')
        journal=control/('interrupt-%03d.json'%number)
        atomic(journal,dict(status='SIGNAL_PENDING',**p))
        os.killpg(p['pid'],signal.SIGTERM)
        atomic(journal,dict(status='SIGNAL_SENT',**p))
        for _ in range(100):
            if not alive(p['pid']):break
            time.sleep(.1)
        if alive(p['pid']):raise RuntimeError('Worker remains alive; reconcile')
        # The worker might have frozen all outputs between our first check and
        # SIGTERM. Completed scientific output must never be rerun.
        if done(job):
            state['jobs'][p['key']]=dict(status='COMPLETE',completed_before_interrupt=True)
            atomic(control/'pipeline_state.json',state)
            atomic(journal,dict(status='ALREADY_COMPLETE',**p))
            continue
        new,dep=p['new_job'],p['new_dep']
        write(new['config'],p['new_cfg']);write(dep['config'],p['new_dep_cfg'])
        jobs[jobs.index(job)]=new
        olddep=by[dep['id']];jobs[jobs.index(olddep)]=dep
        sources[p['rid']]=p['new_source']
        del state['jobs'][p['key']]
        state['jobs'][new['id']]=dict(status='QUEUED',interrupted_job=p['key'],preserved_output=p['old_output'])
        by.pop(p['key']);by[new['id']]=new;by[dep['id']]=dep
        atomic(root/'sources.json',sources);atomic(control/'jobs.json',jobs);atomic(control/'pipeline_state.json',state)
        atomic(journal,dict(status='REQUEUED',**p))
    atomic(root/'sources.json',sources);atomic(control/'jobs.json',jobs)
    state.update(interrupted=[dict(row=p['rid'],gpu=p['gpu'],old_output=p['old_output'],new_output=p['new_cfg']['output_root'])
                              for p in plans if p['new_job']['id'] in state['jobs']],dispatch_gpus=list(dispatch_gpus()))
    atomic(control/'pipeline_state.json',state)
    write(control/'handoff.json',dict(status='HANDOFF',previous_pid=old,new_pid=os.getpid(),adopted=list(observed)))
    return loop(rt,control,state,jobs,failures)


def loop(rt,control,state,jobs,failures):
    root=Path(rt['run_root']);by={j['id']:j for j in jobs};prev=resource_sample();time.sleep(1);turn=0
    while True:
        sample=resource_sample();guard=resource_guard(prev,sample);prev=sample
        for k,v in list(state['jobs'].items()):
            if v['status']!='RUNNING' or alive(v['pid']):continue
            ok=done(by[k]);v.update(status='COMPLETE' if ok else 'FAILED',ended_at=time.time())
            if not ok:
                log=Path(v['log']);lines=log.read_text(errors='replace').splitlines() if log.exists() else []
                reason=re.sub(r's202609270[1-8]','sSEED',lines[-1] if lines else 'missing completion')
                failures[reason]+=1;v['failure']=reason
        block_failed_dependencies(jobs,state,[])
        systemic=any(n>=2 or any(x in reason for x in ('CONTRACT_MISMATCH','PROVENANCE','Truth closed','U truth','Output collision')) for reason,n in failures.items())
        pending=[j for j in jobs if state['jobs'][j['id']]['status']=='QUEUED']
        eligible=[j for j in pending if ready(j)] if not systemic and guard['allowed'] else []
        active={k:v for k,v in state['jobs'].items() if v['status']=='RUNNING'}
        def start(job,gpu):
            path=Path(job.get('recovery_log',str(root/'logs'/(job['id']+'.log'))));release=job.get('worker_release',rt['source_release'])
            env=dict(os.environ,PYTHONPATH=release,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                     OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            args=command(rt,job)
            with path.open('x') as log:
                child=subprocess.Popen(args,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            item=dict(status='RUNNING',pid=child.pid,gpu=gpu,cwd=release,argv=args,log=str(path),started_at=time.time())
            state['jobs'][job['id']]=item;active[job['id']]=item
        score=next((j for j in eligible if j['kind']=='score'),None)
        if score and not any(by[k]['kind']=='score' for k in active) and not foreign_CPU_score({v['pid'] for v in active.values()}):
            start(score,None);eligible.remove(score)
        for gpu in dispatch_gpus():
            _,free=capacity(gpu);own=sum(v.get('gpu')==gpu for v in active.values())
            if own>=limits(gpu,rt['gpu7_limit']) or free<12000:continue
            pool=[j for j in eligible if j['kind']!='score']
            if not pool:continue
            preferred='predict' if turn%3==2 else 'train';turn+=1
            job=next((j for j in pool if j['kind']==preferred),pool[0]);start(job,gpu);eligible.remove(job)
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),
            failures=dict(failures),systemic_stop_new_launches=systemic,resource_guard=guard,
            own_gpu_limits={'0':0,'1':0,'2-6':3,'7':rt['gpu7_limit']},other_owners_counted=False)
        atomic(control/'pipeline_state.json',state)
        if not active and (not pending or systemic):break
        time.sleep(15)
    if all(v['status']=='COMPLETE' for v in state['jobs'].values()):
        state['status']='ANALYZING';atomic(control/'pipeline_state.json',state)
        importlib.import_module(rt['analyze_module']).run(dict(rt,release_root=rt['source_release']))
        state['status']='ANALYZED'
    else:state['status']='PARTIAL_TECHNICAL_FAILURE'
    state['ended_at']=time.time();atomic(control/'pipeline_state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(read(a.config))
