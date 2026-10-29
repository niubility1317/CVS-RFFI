"""All registered runnable external versions, native C and named extensions.

Large matrices are described once; live state stores counts and active workers,
not a repeated copy of every queued row. Failures block only the affected
profile/source descendants. No query score controls scheduling or expansion.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import time
from .io import read,write,atomic,require
from .runner import occupancy


def source_id(profile,seed):return f'source-{profile}-s{seed}'
def prediction_id(profile,seed,index,path):return f'predict-{profile}-s{seed}-c{index:04d}-{path}'


def priority_key(job,first_seed):
    return job['priority'],job.get('seed')!=first_seed,job['id']


def make_jobs(rt):
    root=Path(rt['run_root']);jobs=[]
    for item in rt['source_plan']:
        for seed in rt['model_seeds']:
            key=source_id(item['id'],seed)
            jobs.append(dict(id=key,kind='source',profile=item['profile'],seed=seed,action=item['action'],
                source_output=item['paths'][str(seed)],output=str(root/'source_checks'/f'{item["id"]}-s{seed}'),
                source_key=item['id'],dependencies=[],gpu=True,priority=0 if item['action']=='reuse' else 1))
    for profile,p in rt['profile_matrix'].items():
        if not p['active']:continue
        first=[prediction_id(profile,rt['model_seeds'][0],0,path) for path in ('P0','P1')]
        for seed in rt['model_seeds']:
            for ci,cap in enumerate(rt['capsules']):
                parent=root/'predictions'/profile/f's{seed}'/f'c{ci:04d}'
                pair=[]
                for path in ('P0','P1'):
                    key=prediction_id(profile,seed,ci,path);pair.append(key)
                    deps=[source_id(p['source_id'],seed)]
                    pilot=seed==rt['model_seeds'][0] and ci==0
                    if not pilot:deps.extend(first)
                    jobs.append(dict(id=key,kind='predict',profile=profile,seed=seed,capsule=cap,input_path=path,
                        output=str(parent/path),dependencies=deps,gpu=True,priority=2 if pilot else 3,
                        core_producer_id=f'core__predict-c{ci:04d}-s{rt["model_seeds"][0]}'))
                jobs.append(dict(id=f'score-{profile}-s{seed}-c{ci:04d}',kind='score',profile=profile,seed=seed,capsule=cap,
                    prediction=str(parent),output=str(root/'scores'/profile/f's{seed}'/f'c{ci:04d}'),
                    dependencies=pair,gpu=False,priority=1))
    if rt.get('continue_core'):
        old=read(rt['shared_core_runtime'])
        for item in read(Path(old['run_root'])/'jobs.json'):
            jobs.append(dict(item,id='core__'+item['id'],original_id=item['id'],original_kind=item['kind'],
                kind='core',profile='PAMF_core',dependencies=['core__'+x for x in item['dependencies']],
                priority=4,core_runtime=old['runtime_file'],core_release=old['release_root']))
    return jobs


def external_ready(rt,job,done=None):
    if job['kind']!='predict' or job['input_path']=='P0':return True
    if rt.get('continue_core') and job['core_producer_id'] not in (done or set()):return False
    marker=Path(rt['shared_dura_root'])/job['capsule']/'complete.json'
    return marker.is_file() and read(marker)['status']=='RESTORED_COMPLETE'


def gpu_ready(device,own):
    return own<2 and len(device['pids'])<2 and device['free']>=5000


def worker(rt,job):
    from . import external_execution as execute
    if job['kind']=='source':return execute.source_job(rt,job)
    if job['kind']=='predict':return execute.predict(rt,job)
    if job['kind']=='score':return execute.score_pair(rt,job)
    raise ValueError('Unknown external job')


def is_complete(job):
    if job['kind']=='core':
        path=Path(job['output']) if job['original_kind'] in ('smoke','audit') else Path(job['output'])/'complete.json'
        return path.is_file() and read(path)['status']==job['success']
    path=Path(job['output'])/'complete.json'
    expected={'source':'SOURCE_READY','predict':'PREDICTIONS_COMPLETE','score':'SCORED_COMPLETE'}[job['kind']]
    return path.is_file() and read(path)['status']==expected


def dispatch(rt):
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir()
    # Reuse the completed data audit. Neither method/config changes nor a new
    # source checkpoint require a second scan of VALIDATED_ONCE data.
    data=read(Path(rt['data_root'])/'builder_report.json');audit=read(rt['data_audit_ref'])
    require(data['status']=='VALIDATED_ONCE' and data['capsules']==483 and audit['status']=='PASS','Input matrix not ready')
    shared=read(rt['shared_core_runtime'])
    require(shared['data_root']==rt['data_root'] and shared['DURA']==rt['DURA'],'Shared P1 input mismatch')
    matrix=make_jobs(rt);write(root/'jobs.json',matrix);by={j['id']:j for j in matrix}
    remaining={j['id'] for j in matrix};done=set();blocked=set();failed={};active={};profile_errors=set();source_errors=set()
    if rt.get('continue_core'):
        from .external_takeover import adopt
        old,adopted=adopt(rt,root/'core_controller_handoff.json')
        for oldid,item in adopted.items():
            key='core__'+oldid;require(key in by,'Unknown adopted worker job')
            item['original_kind']=by[key]['original_kind']
            active[key]=(None,item);remaining.remove(key)
        for j in matrix:
            if j['kind']=='core' and j['id'] not in active and is_complete(j):
                done.add(j['id']);remaining.remove(j['id'])
    write(root/'admission_policy.json',dict(scope=rt.get('gpu_count_scope','phase2'),per_gpu=2,minimum_free_MiB=5000))
    state=dict(status='RUNNING',controller_pid=os.getpid(),code_commit=rt['code_commit'],owner=rt['launch_owner'],
               active={},counts={},profiles={},inactive_profiles={p:x['inactive_reason'] for p,x in rt['profile_matrix'].items() if not x['active']})
    atomic(root/'pipeline_state.json',state)
    def launch(j,gpu):
        cwd=j['core_release'] if j['kind']=='core' else rt['release_root']
        command=([rt['python'],'-u','-m','experiments.pamf_phase2_practical.episodic_dispatch','--runtime',j['core_runtime'],'--job',j['original_id']]
                 if j['kind']=='core' else [rt['python'],'-u','-m','experiments.pamf_phase2_practical.external_dispatch','--runtime',rt['runtime_file'],'--job',j['id']])
        env=dict(os.environ,PYTHONPATH=cwd,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        log=root/'logs'/(j['id']+'.log')
        with log.open('x') as f:process=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        item=dict(pid=process.pid,gpu=gpu,argv=command,cwd=cwd,log=str(log),started=time.time(),kind=j['kind'],profile=j['profile'],original_kind=j.get('original_kind'))
        active[j['id']]=(process,item);remaining.remove(j['id']);print('LAUNCH',j['id'],process.pid,gpu,flush=True)
    while True:
        for key,(process,item) in list(active.items()):
            if process is None:
                from .external_takeover import process as identity
                rc=None if identity(item['pid']) else (0 if is_complete(by[key]) else 1)
            else:rc=process.poll()
            if rc is None:continue
            good=rc==0 and is_complete(by[key]);result=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=rc,ended=time.time())
            write(root/'job_results'/(key+'.json'),result)
            if good:done.add(key)
            else:
                failed[key]=result
                if by[key]['kind']=='source':source_errors.add(by[key]['source_key'])
                else:profile_errors.add(by[key]['profile'])
            del active[key]
        for key in list(remaining):
            j=by[key];src=rt['profile_matrix'].get(j['profile'],{}).get('source_id',j.get('source_key'))
            dependency_failed=any(d in failed or d in blocked for d in j['dependencies'])
            no_future_G=('PAMF_core' in profile_errors and not any(item['kind']=='core' and item['gpu'] is not None for process,item in active.values())
                         and j['kind']=='predict' and j['input_path']=='P1' and not external_ready(rt,j,done))
            if j['profile'] in profile_errors or src in source_errors or dependency_failed or no_future_G:
                remaining.remove(key);blocked.add(key)
        candidates=[j for j in matrix if j['id'] in remaining and all(d in done for d in j['dependencies'])]
        candidates.sort(key=lambda j:priority_key(j,rt['model_seeds'][0]))
        for j in list(candidates):
            if j['gpu']:continue
            if any(item['kind']=='score' or item.get('original_kind')=='score' for process,item in active.values()):break
            launch(j,None);candidates.remove(j)
        from .external_takeover import resources
        devices=resources(rt);policy=read(root/'admission_policy.json')
        # Keep the original G-producing core progressing while additional
        # families train/adapt; scientific budgets and outputs are unchanged.
        if sum(item['kind']=='core' and item['gpu'] is not None for process,item in active.values())<rt.get('minimum_core_workers',4):
            candidates.sort(key=lambda j:(j['kind']!='core',j.get('seed')!=rt['model_seeds'][0],j['priority'],j['id']))
        for slot in range(2):
            for gpu in rt['allowed_gpus']:
                own=sum(item['gpu']==gpu for process,item in active.values())
                scope_pids=devices[gpu]['phase2_pids'] if policy['scope']=='phase2' else devices[gpu]['pids']
                if len(scope_pids)>=2 or devices[gpu]['free']<5000:continue
                j=next((j for j in candidates if j['gpu'] and external_ready(rt,j,done)
                        and not (j['kind']=='source' and j['action']=='train' and any(by[k].get('source_key')==j['source_key'] for k in active))),None)
                if j is None:continue
                launch(j,gpu);candidates.remove(j);devices[gpu]['pids'].add(active[j['id']][0].pid)
                devices[gpu]['phase2_pids'].add(active[j['id']][0].pid);devices[gpu]['free']-=5000
        profile_status={}
        for profile,p in rt['profile_matrix'].items():
            if not p['active']:profile_status[profile]=dict(status='N_A_BY_ORIGINAL_TABLE',reason=p['inactive_reason']);continue
            counts=Counter('COMPLETE' if j['id'] in done else 'FAILED' if j['id'] in failed else 'BLOCKED' if j['id'] in blocked else 'RUNNING' if j['id'] in active else 'QUEUED'
                           for j in matrix if j['profile']==profile)
            profile_status[profile]=dict(counts=counts,status='FAILED_TECHNICAL' if profile in profile_errors or p['source_id'] in source_errors else 'RUNNING_OR_QUEUED')
        if rt.get('continue_core'):
            profile_status['PAMF_core']=dict(status='CONTINUED_WITHOUT_RESTART',counts=dict(Counter(
                'COMPLETE' if j['id'] in done else 'RUNNING' if j['id'] in active else 'FAILED' if j['id'] in failed else 'BLOCKED' if j['id'] in blocked else 'QUEUED'
                for j in matrix if j['kind']=='core')))
        state.update(active={k:item for k,(process,item) in active.items()},counts=dict(COMPLETE=len(done),RUNNING=len(active),QUEUED=len(remaining),BLOCKED=len(blocked),FAILED=len(failed)),
            profiles=profile_status,failed_jobs=list(failed),updated_at=time.time())
        if not active and not remaining:state['status']='COMPLETE' if not failed else 'PARTIAL_TECHNICAL_FAILURE'
        atomic(root/'pipeline_state.json',state)
        if not active and not remaining:break
        time.sleep(rt.get('poll_seconds',5))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',required=True);p.add_argument('--job');a=p.parse_args();rt=read(a.runtime)
    if a.job:worker(rt,next(j for j in make_jobs(rt) if j['id']==a.job))
    else:dispatch(rt)
