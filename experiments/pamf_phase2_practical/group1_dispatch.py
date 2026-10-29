"""Single-owner Group1 dispatcher; no legacy queue takeover or score feedback."""
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import time
from .io import read,write,atomic,require
from .group1 import catalog,source_scope,row


def jobs(rt):
    root=Path(rt['run_root']);result=[]
    selected=[row(p) for p in rt['method_profiles']] if rt.get('method_profiles') is not None else catalog()
    for r in selected:
        if not r['active']:continue
        if rt.get('method_profiles') is not None and r['id'] not in rt['method_profiles']:continue
        first=f'predict-{r["id"]}-s{rt["model_seeds"][0]}-c0000'
        for seed in rt['model_seeds']:
            sk=f'source-{r["id"]}-s{seed}';folder=root/'source'/f'{r["id"]}-s{seed}'
            result.append(dict(id=sk,kind='source',profile=r['id'],seed=seed,output=str(folder),
                completion=str(folder/'group1_complete.json'),status='GROUP1_SOURCE_READY',gpu=True,dependencies=[],
                priority=0 if seed==rt['model_seeds'][0] else 2))
            for index,cap in enumerate(rt['capsules']):
                key=f'predict-{r["id"]}-s{seed}-c{index:04d}';folder=root/'predictions'/r['id']/f's{seed}'/f'c{index:04d}'
                deps=[sk]
                if key!=first:deps.append(first)
                result.append(dict(id=key,kind='predict',profile=r['id'],seed=seed,capsule=cap,output=str(folder),
                    completion=str(folder/'complete.json'),status='PREDICTIONS_COMPLETE',gpu=True,
                    dependencies=deps,priority=1 if key==first else 3,scope='pilot_main' if key==first else 'full'))
                result.append(dict(id='score-'+key,kind='score',profile=r['id'],seed=seed,prediction=str(folder),
                    output=str(root/'scores'/r['id']/f's{seed}'/f'c{index:04d}'),
                    completion=str(root/'scores'/r['id']/f's{seed}'/f'c{index:04d}'/'complete.json'),
                    status='SCORED_COMPLETE',gpu=False,dependencies=[key],priority=0))
                if key==first:
                    extra=key+'-curves';efolder=folder.with_name(folder.name+'-curves')
                    result.append(dict(id=extra,kind='predict',profile=r['id'],seed=seed,capsule=cap,output=str(efolder),
                        completion=str(efolder/'complete.json'),status='PREDICTIONS_COMPLETE',gpu=True,
                        dependencies=[first],priority=3,scope='curve_extra'))
                    result.append(dict(id='score-'+extra,kind='score',profile=r['id'],seed=seed,prediction=str(efolder),
                        output=str(root/'scores'/r['id']/f's{seed}'/'c0000-curves'),
                        completion=str(root/'scores'/r['id']/f's{seed}'/'c0000-curves/complete.json'),
                        status='SCORED_COMPLETE',gpu=False,dependencies=[extra],priority=0))
    return result


def worker(rt,j):
    if j['kind']=='source':
        from .group1_source import train
        return train(source_scope(rt),j['profile'],j['seed'],j['output'])
    if j['kind']=='predict':
        from .group1_target import predict
        return predict(rt,j)
    if j['kind']=='score':
        from .episodic_score import run
        return run(rt,j)
    raise ValueError('Unknown Group1 job kind')


def completed(j):
    p=Path(j['completion'])
    return p.is_file() and read(p).get('status')==j['status']


def dispatch(rt):
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    (root/'logs').mkdir();(root/'job_results').mkdir()
    write(root/'owner.json',dict(owner=rt['launch_owner'],pid=os.getpid(),runtime=rt['runtime_file'],
        authorization='User2026-10-30_release_new_Group1_with_source_concat',legacy_queues_resumed=False))
    write(root/'method_matrix.json',dict(methods=catalog(),source_concat=True,PL='midpoint0.95',
        input_data='VALIDATED_ONCE_r10',query_fit=False))
    matrix=jobs(rt);write(root/'jobs.json',matrix)
    by={j['id']:j for j in matrix};remaining=set(by);done=set();failed={};blocked=set();active={};profile_errors=set()
    # Existing source-only practical cache machinery; new owner/cache, no old service changes.
    cache_script=('import json;from experiments.pamf_phase2_practical.group1 import source_scope;'
        'from experiments.pamf_phase2_practical.io import activate;'
        'from experiments.practical_full_2x2.cache import service;'
        'c=source_scope(json.load(open('+repr(rt['runtime_file'])+')));activate(c);service(c)')
    env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (root/'logs/augmentation_service.log').open('x') as f:
        augmentation=subprocess.Popen([rt['python'],'-u','-c',cache_script],cwd=rt['release_root'],env=env,
            stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
    write(root/'augmentation_owner.json',dict(pid=augmentation.pid,cwd=rt['release_root'],command=cache_script))
    while True:
        stop=(root/'USER_STOP.json').is_file()
        if augmentation.poll() is not None and any(j['kind']=='source' and j['id'] in remaining for j in matrix):
            raise RuntimeError('Source augmentation service failed; preserve active workers and partials')
        for key,(process,item) in list(active.items()):
            code=process.poll()
            if code is None:continue
            good=code==0 and completed(by[key]);result=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code,ended_at=time.time())
            write(root/'job_results'/(key+'.json'),result)
            if good:done.add(key)
            else:failed[key]=result;profile_errors.add(by[key]['profile'])
            del active[key]
        for key in list(remaining):
            j=by[key]
            if j['profile'] in profile_errors or any(d in failed or d in blocked for d in j['dependencies']):
                blocked.add(key);remaining.remove(key)
        available=[j for j in matrix if j['id'] in remaining and all(d in done for d in j['dependencies'])]
        available.sort(key=lambda j:(j['priority'],j.get('seed')!=rt['model_seeds'][0],j['id']))
        def launch(j,gpu):
            args=[rt['python'],'-u','-m','experiments.pamf_phase2_practical.group1_dispatch','--runtime',rt['runtime_file'],'--job',j['id']]
            env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            log=root/'logs'/(j['id']+'.log')
            with log.open('x') as stream:
                child=subprocess.Popen(args,cwd=rt['release_root'],env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            item=dict(pid=child.pid,gpu=gpu,kind=j['kind'],profile=j['profile'],command=args,cwd=rt['release_root'],log=str(log),started_at=time.time())
            active[j['id']]=(child,item);remaining.remove(j['id']);available.remove(j)
            print('GROUP1_LAUNCH',j['id'],child.pid,gpu,flush=True)
        if not stop:
            if not any(item['kind']=='score' for _,item in active.values()):
                scorer=next((j for j in available if not j['gpu']),None)
                if scorer:launch(scorer,None)
            from .runner import occupancy
            devices=occupancy()
            for slot in (0,1):
                for gpu in rt['allowed_gpus']:
                    own=sum(item['gpu']==gpu for _,item in active.values())
                    if own>=2 or devices[gpu]['free']<rt.get('minimum_free_MiB',6000):continue
                    job=next((j for j in available if j['gpu']),None)
                    if job is None:continue
                    launch(job,gpu);devices[gpu]['free']-=rt.get('minimum_free_MiB',6000)
        counts=dict(COMPLETE=len(done),FAILED=len(failed),BLOCKED=len(blocked),RUNNING=len(active),QUEUED=len(remaining))
        status='STOP_REQUESTED' if stop else 'RUNNING'
        if not active and not remaining:status='COMPLETE' if not failed else 'PARTIAL_TECHNICAL_FAILURE'
        atomic(root/'pipeline_state.json',dict(status=status,run_id=rt['run_id'],updated_at=time.time(),
            active={k:v for k,(_,v) in active.items()},counts=counts,unavailable_methods={r['id']:r['unavailable_reason'] for r in catalog() if not r['active']},
            failed_jobs=list(failed),legacy_queues_resumed=False,source_concat=True))
        if (not active and not remaining) or (stop and not active):break
        time.sleep(rt.get('poll_seconds',10))
    # Stop our producer through its own documented marker, never other owners.
    Path(rt['augmentation_cache'],'stop').write_text('group1 dispatch completed\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--job');a=p.parse_args();rt=read(a.runtime)
    if a.job:worker(rt,next(j for j in jobs(rt) if j['id']==a.job))
    else:dispatch(rt)
