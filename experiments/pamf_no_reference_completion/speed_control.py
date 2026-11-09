"""Adopt the current owner and accelerate only unstarted source workers."""
import argparse
from collections import Counter
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from experiments.cric_dura_comparison.io import read,write
from experiments.cric_dura_comparison.pipeline import alive,done,ready,block_failed_dependencies
from experiments.practical_full_2x2.cache import atomic
from experiments.pamf_repl_completion.runner import capacity,foreign_CPU_score
from experiments.pamf_branch_factorial.runner import resource_sample,resource_guard
from experiments.pamf_branch_factorial.continuation import can_start


def command(rt,job):
    return [rt['python'],'-u','-m','experiments.pamf_no_reference_completion.runner','worker',
            '--kind',job['kind'],'--config',job['config']]


def matches_dispatch(rt,cwd,argv):
    return cwd==rt['source_release'] and argv==[rt['python'],'-u','-m',
        'experiments.pamf_no_reference_completion.runner','dispatch','--config',rt['source_release']+'/runtime.json']


def current_workers(rt,jobs):
    by_config={j['config']:j for j in jobs};found={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            pid=int(p.name)
            if not alive(pid) or os.readlink(p/'cwd')!=rt['source_release']:continue
            argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if '--config' not in argv:continue
            job=by_config.get(argv[argv.index('--config')+1])
            if not job or argv!=command(rt,job):continue
            if job['id'] in found:raise ValueError('Duplicate original worker: '+job['id'])
            env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
            gpu=None if job['kind']=='score' else int(env[b'CUDA_VISIBLE_DEVICES'])
            found[job['id']]=dict(status='RUNNING',pid=pid,gpu=gpu,cwd=rt['source_release'],argv=argv,
                log=str(Path(rt['run_root'])/'logs'/(job['id']+'.log')),adopted=True)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return found


def log_has_no_owner_process(log):
    """A forked worker holds its log even before exec; uncertainty blocks retry."""
    log=str(log)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            for fd in (proc/'fd').iterdir():
                try:
                    if os.readlink(fd)==log:return False
                except FileNotFoundError:pass
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:return False
    return True


def source_smoke(rt,control):
    import numpy as np
    import torch
    from experiments.pamf_repl_completion.runner import activate
    activate(rt)
    from experiments.pamf_no_reference_completion.model import load_reader
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.t1_practical import Practical as Original
    from experiments.pamf_no_reference_completion.fast_practical import Practical as Fast
    ctx=read(rt['context']);sources=read(Path(rt['run_root'])/'sources.json')
    row=next(v for v in sources.values() if v['row']['reuse_key']=='ssl/2026092701')
    h=ctx['handles']['F0'];contract=read(h['source_contract'])
    reader,_=load_reader(row['source_output'],contract,row['row'],'cpu')
    ds=SourceDataset(h['source_cache'],'L_s',contract);x=ds.x[:64]
    with torch.inference_mode():pred=reader.classify_features(reader.features(x[:2]))
    if pred.shape!=(2,6) or not bool(torch.isfinite(pred).all()):raise ValueError('Actual checkpoint source smoke failed')
    from experiments.cric_dura_comparison.io import tensor
    meta={k:tensor(np.asarray(ds.metadata[k][:64],dtype=np.int64)) for k in ('eq_i','rx_i','day_i','sig_i','base_index')}
    c=dict(source_rxs=contract['source_rxs'],source_days=contract['source_days'],fast_as_cache=str(control/'smoke_cache'))
    original=Original(c);fast=Fast(c);numpy_state=np.random.get_state();torch_state=torch.random.get_rng_state().clone()
    start=time.perf_counter();a=original(x,meta,7,1500);original_seconds=time.perf_counter()-start
    start=time.perf_counter();b=fast(x,meta,7,1500);miss_seconds=time.perf_counter()-start
    start=time.perf_counter();d=fast(x,meta,7,1500);hit_seconds=time.perf_counter()-start
    now=np.random.get_state()
    if (not torch.equal(a,b) or not torch.equal(a,d) or fast.cache_hits!=1 or fast.coverage!=Counter({k:2*v for k,v in original.coverage.items()}) or
        original.sessions!=fast.sessions or not torch.equal(torch_state,torch.random.get_rng_state()) or
        numpy_state[0]!=now[0] or not np.array_equal(numpy_state[1],now[1]) or numpy_state[2:]!=now[2:]):
        raise ValueError('Source simulation output or RNG state differs')
    write(control/'source_smoke.json',dict(status='PASS',source_samples=64,query_reads=0,U_truth_reads=0,
        optimizer_updates=0,outputs_bit_exact=True,coverage_and_session_equal=True,global_rng_equal=True,
        original_seconds=original_seconds,cache_miss_seconds=miss_seconds,cache_hit_seconds=hit_seconds))


def run(rt):
    root=Path(rt['run_root']);control=Path(rt['control_root']);control.mkdir(parents=True,exist_ok=False)
    source_smoke(rt,control)
    jobs=read(root/'jobs.json');by={j['id']:j for j in jobs};sources=read(root/'sources.json')
    previous=read(root/'pipeline_state.json');old=previous['pid'];proc=Path('/proc')/str(old)
    argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    if previous['status']!='RUNNING' or not alive(old) or not matches_dispatch(rt,os.readlink(proc/'cwd'),argv):
        raise ValueError('Original dispatcher identity/status changed; no process stopped')
    if (root/'handoff.json').exists():raise ValueError('Existing handoff; reconcile before stop')
    write(control/'handoff_started.json',dict(status='TAKEOVER_STARTED',previous_pid=old))
    os.kill(old,signal.SIGTERM)
    for _ in range(100):
        if not alive(old):break
        time.sleep(.1)
    if alive(old):raise RuntimeError('Original dispatcher remains alive')
    previous=read(root/'pipeline_state.json');observed={}
    for _ in range(13):observed.update(current_workers(rt,jobs));time.sleep(1)
    observed={k:v for k,v in observed.items() if alive(v['pid'])}
    failures=Counter(previous.get('failures',{}));state=dict(status='RUNNING',pid=os.getpid(),owner=rt['launch_owner'],
        code_commit=rt['code_commit'],source_model_commit=rt['source_model_commit'],previous_pid=old,jobs={})
    for j in jobs:
        k=j['id'];prior=previous['jobs'][k];log=root/'logs'/(k+'.log')
        if k in observed:state['jobs'][k]=observed[k]
        elif done(j):state['jobs'][k]=dict(prior,status='COMPLETE')
        elif ((prior['status']=='QUEUED' and log.exists()) or (prior['status']=='RUNNING' and alive(prior.get('pid',-1)))):
            state['jobs'][k]=dict(status='RECONCILING_LAUNCH',log=str(log),was_queued=prior['status']=='QUEUED',empty_scans=0)
        elif prior['status'] in ('RUNNING','COMPLETE'):
            lines=log.read_text(errors='replace').splitlines() if log.exists() else []
            reason=re.sub(r's202609270[1-8]','sSEED',lines[-1] if lines else 'missing completion')
            failures[reason]+=1;state['jobs'][k]=dict(prior,status='FAILED',failure=reason)
        else:state['jobs'][k]=prior
    # Never edit old configs. Only future augmented factorial training gets a
    # new control-specific config and new release; other jobs use original code.
    (control/'configs').mkdir()
    accelerated=[]
    for j in jobs:
        if j['kind']!='train' or state['jobs'][j['id']]['status']!='QUEUED':continue
        c=read(j['config']);row=c['paper_row']
        if row['family']=='factorial' and row['channel_augmentation']:
            c['fast_as_cache']=str(control/'shared_source_as_v1');c['code_commit']=rt['code_commit']
            new=control/'configs'/(j['id']+'.json');write(new,c);j['config']=str(new);j['worker_release']=rt['release_root'];accelerated.append(j['id'])
    write(control/'jobs.json',jobs);atomic(control/'pipeline_state.json',state)
    write(root/'handoff.json',dict(status='HANDOFF',previous_pid=old,new_pid=os.getpid(),control_root=str(control),
        source_release=rt['source_release'],adopted=list(observed),accelerated_queued=len(accelerated)))
    previous_sample=resource_sample();time.sleep(1);turn=0
    while True:
        current=resource_sample();guard=resource_guard(previous_sample,current);previous_sample=current
        reconciling=[k for k,v in state['jobs'].items() if v['status']=='RECONCILING_LAUNCH']
        if reconciling:
            observed=current_workers(rt,jobs)
            for k in reconciling:
                if k in observed:state['jobs'][k]=observed[k]
                elif done(by[k]):state['jobs'][k]=dict(status='COMPLETE',reconciled=True)
                else:
                    v=state['jobs'][k];log=Path(v['log'])
                    if v['was_queued'] and log.exists() and log.stat().st_size==0 and log_has_no_owner_process(log):
                        v['empty_scans']+=1
                        if v['empty_scans']>=6:
                            recovery=log.with_name(log.stem+'.handoff-recovery.log')
                            if recovery.exists():raise FileExistsError('Recovery log exists')
                            by[k]['recovery_log']=str(recovery)
                            atomic(control/'jobs.json',jobs)
                            state['jobs'][k]=dict(status='QUEUED',reconciled_empty_log=str(log))
                    else:v['empty_scans']=0
        for k,v in list(state['jobs'].items()):
            if v['status']!='RUNNING' or alive(v['pid']):continue
            ok=done(by[k]);v.update(status='COMPLETE' if ok else 'FAILED',ended_at=time.time())
            if not ok:
                lines=Path(v['log']).read_text(errors='replace').splitlines();reason=re.sub(r's202609270[1-8]','sSEED',lines[-1] if lines else 'missing completion')
                failures[reason]+=1;v['failure']=reason
        block_failed_dependencies(jobs,state,[])
        systemic=any(n>=2 or any(x in reason for x in ('CONTRACT_MISMATCH','PROVENANCE','Truth closed','U truth','Output collision')) for reason,n in failures.items())
        pending=[j for j in jobs if state['jobs'][j['id']]['status']=='QUEUED']
        uncertain=any(v['status']=='RECONCILING_LAUNCH' for v in state['jobs'].values())
        eligible=[j for j in pending if ready(j)] if not systemic and not uncertain and guard['allowed'] else []
        active={k:v for k,v in state['jobs'].items() if v['status']=='RUNNING'}
        def start(job,gpu):
            path=Path(job.get('recovery_log',str(root/'logs'/(job['id']+'.log'))));release=job.get('worker_release',rt['source_release'])
            env=dict(os.environ,PYTHONPATH=release,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                     OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            args=command(rt,job)
            with path.open('x') as log:
                child=subprocess.Popen(args,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            item=dict(status='RUNNING',pid=child.pid,gpu=gpu,cwd=release,argv=args,log=str(path),started_at=time.time(),
                      accelerated=release==rt['release_root'])
            state['jobs'][job['id']]=item;active[job['id']]=item
        score=next((j for j in eligible if j['kind']=='score'),None)
        if score and not any(by[k]['kind']=='score' for k in active) and not foreign_CPU_score({v['pid'] for v in active.values()}):
            start(score,None);eligible.remove(score)
        for gpu in range(8):
            _,free=capacity(gpu);own=sum(v.get('gpu')==gpu for v in active.values())
            if not can_start(gpu,own,free):continue
            pool=[j for j in eligible if j['kind']!='score']
            if not pool:continue
            want='predict' if turn%3==2 else 'train';turn+=1
            matches=[j for j in pool if j['kind']==want] or pool
            def rank(job):
                if job['kind']!='train':return (0,'',0,'')
                row=sources[job['id'].split('__',1)[1]]['row']
                if not (row['family']=='factorial' and row['channel_augmentation']):return (1,row['data_setting'],row['model_seed'],row['method'])
                return (0,row['data_setting'],row['model_seed'],{'sup':0,'pl':1,'ssl':2}.get(row['method'],3))
            job=min(matches,key=rank);start(job,gpu);eligible.remove(job)
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),failures=dict(failures),
            systemic_stop_new_launches=systemic,resource_guard=guard,own_gpu_limits={'0-6':3,'7':2},other_owners_counted=False,
            accelerated_queued=sum(state['jobs'][k]['status']=='QUEUED' for k in accelerated))
        atomic(control/'pipeline_state.json',state)
        if not active and (not pending or systemic) and not uncertain:break
        time.sleep(15)
    if all(v['status']=='COMPLETE' for v in state['jobs'].values()):
        from .analyze import run as analyze
        analyze(dict(rt,release_root=rt['source_release']));state['status']='ANALYZED'
    else:state['status']='PARTIAL_TECHNICAL_FAILURE'
    state['ended_at']=time.time();atomic(control/'pipeline_state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();run(read(args.config))
