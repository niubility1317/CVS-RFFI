"""Two-process-per-GPU queue: source smoke, data audit, pilot, full fixed matrix."""
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import time
from .io import read,write,atomic,require
from .runner import occupancy


def eligible(device,own):
    # Visible existing processes count toward the two-process cap; never stop them.
    return own<2 and len(device['pids'])<2 and device['free']>=5000


def jobs(rt):
    root=Path(rt['run_root']);rows=[]
    def add(key,kind,out,status,deps=(),gpu=False,**args):
        rows.append(dict(id=key,kind=kind,output=str(out),success=status,dependencies=list(deps),gpu=gpu,**args))
    add('source-smoke','smoke',root/'source_smoke.json','PASS',gpu=True)
    add('threepool-audit','audit',root/'data_audit.json','PASS',('source-smoke',))
    pilot=None
    capsules=[f'capsules/{p}/{e["id"]}/{r}' for p in rt['physical_pools'] for e in rt['episodes'] for r in rt['target_receivers']]
    for ci,cap in enumerate(capsules):
        first=None
        for si,seed in enumerate(rt['model_seeds']):
            key=f'predict-c{ci:04d}-s{seed}';deps=['threepool-audit']
            if si:deps.append(first)
            if ci or si:deps.append('predict-c0000-s'+str(rt['model_seeds'][0]))
            add(key,'predict',root/'predictions'/f'c{ci:04d}-s{seed}','PREDICTIONS_COMPLETE',deps,gpu=True,capsule=cap,seed=seed)
            if si==0:first=key
            add('score-'+key,'score',root/'scores'/f'c{ci:04d}-s{seed}','SCORED_COMPLETE',(key,),prediction=str(root/'predictions'/f'c{ci:04d}-s{seed}'))
    return rows


def worker(rt,job):
    kind=job['kind']
    if kind=='smoke':
        from .features import source_smoke
        return source_smoke(rt,job['output'])
    if kind=='audit':
        from .episodic_validation import run
        return run(rt,job['output'])
    if kind=='predict':
        from .episodic_experiment import run
        return run(rt,job)
    if kind=='score':
        from .episodic_score import run
        return run(rt,job)
    raise ValueError('Unknown stage')


def completed(j):
    p=Path(j['output']) if j['kind'] in ('smoke','audit') else Path(j['output'])/'complete.json'
    return p.is_file() and read(p)['status']==j['success']


def dispatch(rt):
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir();matrix=jobs(rt);write(root/'jobs.json',matrix)
    state=dict(status='RUNNING',controller_pid=os.getpid(),code_commit=rt['code_commit'],jobs={j['id']:dict(status='QUEUED') for j in matrix})
    by={j['id']:j for j in matrix};active={};failed=[]
    def launch(j,gpu):
        command=[rt['python'],'-u','-m','experiments.pamf_phase2_practical.episodic_dispatch','--runtime',rt['runtime_file'],'--job',j['id']]
        env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        log=root/'logs'/(j['id']+'.log')
        with log.open('x') as f:process=subprocess.Popen(command,cwd=rt['release_root'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        item=dict(status='RUNNING',pid=process.pid,gpu=gpu,argv=command,cwd=rt['release_root'],log=str(log),started=time.time())
        active[j['id']]=(process,item);state['jobs'][j['id']]=item;print('LAUNCH',j['id'],process.pid,gpu,flush=True)
    while True:
        for key,(process,item) in list(active.items()):
            if process.poll() is None:continue
            ok=process.returncode==0 and completed(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if ok else 'FAILED',exit_code=process.returncode)
            if not ok:failed.append(key)
            del active[key]
        pending=[j for j in matrix if state['jobs'][j['id']]['status']=='QUEUED']
        candidates=[j for j in pending if all(state['jobs'][x]['status']=='COMPLETE' for x in j['dependencies'])] if not failed else []
        for j in list(candidates):
            if j['gpu']:continue
            cpu=[by[k]['kind'] for k,(p,i) in active.items() if i['gpu'] is None]
            if j['kind']=='score' and 'score' in cpu:continue
            if len(cpu)>=2:continue
            launch(j,None);candidates.remove(j)
        devices=occupancy()
        for slot in range(2):
            for gpu in rt['allowed_gpus']:
                own=sum(i['gpu']==gpu for p,i in active.values())
                if not eligible(devices[gpu],own):continue
                j=next((j for j in candidates if j['gpu']),None)
                if j is None:continue
                launch(j,gpu);candidates.remove(j);devices[gpu]['pids'].add(active[j['id']][0].pid);devices[gpu]['free']-=5000
        state.update(counts=dict(Counter(x['status'] for x in state['jobs'].values())),updated_at=time.time())
        if failed:state.update(status='STOP_NEW_DISPATCH_TECHNICAL_FAILURE',failed_jobs=failed)
        elif not pending and not active:state['status']='COMPLETE'
        atomic(root/'pipeline_state.json',state)
        if not active and (failed or not pending):break
        time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',required=True);p.add_argument('--job');a=p.parse_args();rt=read(a.runtime)
    if a.job:worker(rt,next(j for j in jobs(rt) if j['id']==a.job))
    else:dispatch(rt)
