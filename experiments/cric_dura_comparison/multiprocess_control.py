"""Resource-only handover; run immutable scientific workers with more GPU slots."""
import argparse
from collections import Counter
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .io import read,write
from .pipeline import alive,done,ready,block_failed_dependencies
from .source_queue import capacity
from .resource_balance import LoadPolicy


HEAVY={'G','G_student'}


def completion_status(state):
    effective=[v for k,v in state['jobs'].items() if k not in state.get('excluded_job_ids',[])]
    return 'COMPLETE' if effective and all(v['status']=='COMPLETE' for v in effective) else 'PARTIAL_TECHNICAL_FAILURE'


def gpu_admits(gpu,job,active,jobs,apps,free,reserve_external=True,own_limit=3,total_limit=3):
    owned=[jobs[key] for key,item in active.items() if item.get('gpu')==gpu]
    if any(j.get('exclusive') for j in owned):return False
    if job.get('exclusive'):return not owned and apps==0 and gpu not in (2,3)
    limit=1 if reserve_external and gpu in (2,3) else own_limit
    if len(owned)>=limit or apps>=total_limit or free<12000:return False
    if job['kind'] in HEAVY and any(j['kind'] in HEAVY for j in owned):return False
    return True


def command_for(rt,job,gpu):
    cmd=[rt['python'],'-u','-m','experiments.cric_dura_comparison.pipeline','worker','--config',job['config']]
    cwd=job.get('worker_release',rt['worker_release'])
    env=dict(os.environ,PYTHONPATH=cwd,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    return cmd,cwd,env


def inspect_worker(rt,job,pid):
    if not alive(pid):return None
    proc=Path('/proc')/str(pid)
    try:
        args=[v.decode() for v in proc.joinpath('cmdline').read_bytes().split(bytes([0])) if v]
        if not args and not alive(pid):return None  # A worker may finish during handover.
        if args!=command_for(rt,job,None)[0] or os.readlink(proc/'cwd')!=command_for(rt,job,None)[1]:
            raise ValueError('Owned worker PID/CWD/argv mismatch: '+job['id'])
        env=dict(s.split(b'=',1) for s in proc.joinpath('environ').read_bytes().split(bytes([0])) if b'=' in s)
        raw=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode()
        return None if job['kind']=='score' else int(raw)
    except (FileNotFoundError,ProcessLookupError):return None


def handover(rt,jobs):
    previous=read(rt['previous_state']);pid=previous['pid'];process=Path('/proc')/str(pid)
    if alive(pid):
        argv=[a.decode() for a in process.joinpath('cmdline').read_bytes().split(bytes([0])) if a]
        module=rt.get('previous_controller_module','experiments.cric_dura_comparison.start_continuation')
        expected=[rt['python'],'-u','-m',module,'--config',rt['previous_runtime']]
        if argv!=expected or os.readlink(process/'cwd')!=rt.get('previous_controller_cwd',rt['worker_release']):
            raise ValueError('Previous controller identity differs; no stop')
        os.kill(pid,signal.SIGTERM)
        for _ in range(50):
            if not alive(pid):break
            time.sleep(.1)
        if alive(pid):raise RuntimeError('Previous controller still alive; no second owner')
    try:previous=read(rt['previous_state'])
    except (ValueError,FileNotFoundError):pass  # Preserve last valid snapshot plus exact process reconciliation.
    by_config={j['config']:j for j in jobs};found={}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or not alive(int(process.name)):continue
        try:
            argv=[v.decode() for v in process.joinpath('cmdline').read_bytes().split(bytes([0])) if v]
            if len(argv)!=7 or argv[1:5]!=['-u','-m','experiments.cric_dura_comparison.pipeline','worker'] or argv[5]!='--config':continue
            job=by_config.get(argv[6])
            if job is None:continue
            gpu=inspect_worker(rt,job,int(process.name))
            if job['id'] in found:raise ValueError('Duplicate existing job worker')
            old=previous['jobs'].get(job['id'],{})
            log=old.get('log',str(Path(rt['previous_control_root'])/'logs'/(job['log_name']+'.log')))
            found[job['id']]=dict(pid=int(process.name),gpu=gpu,log=log,adopted=True)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return previous,found


def external_pending(rt):
    path=Path(rt['external_pipeline_state'])
    if not path.is_file():return True
    try:s=read(path)
    except (ValueError,FileNotFoundError):return True  # A concurrent state rewrite must not stop this owner.
    required=s.get('effective_required_row_ids')
    if required is not None:return any(s.get('rows',{}).get(key,{}).get('status')!='ANALYZED' for key in required)
    return s.get('status') not in ('COMPLETE','ANALYZED','ARTIFACTS_COMPLETE')


def failure(job,item):
    path=Path(item['log']);tail=path.read_text(errors='replace')[-3000:] if path.is_file() else ''
    return tail.splitlines()[-1] if tail.splitlines() else 'missing_completion_without_log'


def run(rt):
    own_limit=rt.get('max_own_per_gpu',3);total_limit=rt.get('max_compute_per_gpu',3)
    balance=LoadPolicy(rt['load_balance']) if rt.get('load_balance') else None
    if own_limit not in (2,3) or total_limit!=3:raise ValueError('Unsupported baseline resource capacity')
    root=Path(rt['control_root']);root.mkdir(parents=True,exist_ok=False)
    jobs=read(rt.get('source_jobs',rt['jobs']));by_id={j['id']:j for j in jobs}
    if len(by_id)!=len(jobs):raise ValueError('Duplicate registered jobs')
    if len({j['config'] for j in jobs})!=len(jobs):raise ValueError('Duplicate registered worker configs')
    if rt.get('G_teacher_acceleration'):
        from .acceleration_smoke import run as acceleration_smoke
        acceleration_smoke(rt,root/'prelaunch_source_smoke.json')
    if rt.get('F_source_binding_recovery'):
        from .f_binding_recovery import smoke as binding_smoke
        binding_smoke(rt,root/'prelaunch_F_binding_smoke.json')
    if rt.get('supervised_extension'):
        from .bootstrap import activate,numerical_policy
        activate(rt);numerical_policy()
        from .supervised_extension import smoke as supervised_smoke
        supervised_smoke(rt,root/'prelaunch_strict_supervised_smoke.json')
    if rt.get('core_extension'):
        from .bootstrap import activate,numerical_policy
        activate(rt);numerical_policy()
        from .core_extension import smoke as core_smoke
        core_smoke(rt,root/'prelaunch_core_smoke.json')
    if rt.get('external_dura_extension'):
        from .external_dura import smoke as external_smoke
        from .external_dura_extension import recovery_needed
        recovery_needed(read(rt['previous_state']))
        external_smoke(rt,root/'prelaunch_external_DURA_smoke.json')
    previous,active=handover(rt,jobs)
    if balance:balance.restore_handover(previous.get('load_balance',{}))
    admitted=previous.get('accelerated_G_jobs',[])
    if rt.get('external_dura_extension'):
        from .external_dura_extension import prepare as extend_external
        jobs=extend_external(rt,jobs,previous,active)
        by_id={j['id']:j for j in jobs}
        if len(by_id)!=len(jobs):raise ValueError('Duplicate external frozen readout jobs')
        write(rt['jobs'],jobs)
    if rt.get('G_teacher_acceleration'):
        from .execution_migration import promote_pending_G
        admitted=promote_pending_G(rt,jobs,previous,active)
        write(rt['jobs'],jobs)
    if rt.get('supervised_extension'):
        from .supervised_extension import prepare as extend_supervised
        jobs=extend_supervised(rt,jobs,previous,active)
        by_id={j['id']:j for j in jobs}
        if len(by_id)!=len(jobs):raise ValueError('Duplicate supervised supplement jobs')
        write(rt['jobs'],jobs)
    if rt.get('core_extension'):
        from .core_extension import prepare as extend_core
        jobs=extend_core(rt,jobs,previous,active)
        by_id={j['id']:j for j in jobs}
        if len(by_id)!=len(jobs):raise ValueError('Duplicate core factorial jobs')
        write(rt['jobs'],jobs)
    if rt.get('F_source_binding_recovery'):
        from .f_binding_recovery import prepare as repair_F_binding
        jobs=repair_F_binding(rt,jobs,previous,active)
        by_id={j['id']:j for j in jobs}
        if len(by_id)!=len(jobs):raise ValueError('Duplicate repaired jobs')
        write(rt['jobs'],jobs)
    state=dict(previous,pid=os.getpid(),owner=rt['launch_owner'],status='RUNNING',control_commit=rt['code_commit'],
        scientific_worker_commit=rt['worker_commit'],previous_owner_pid=previous['pid'],jobs=dict(previous['jobs']),
        accelerated_G_jobs=admitted,accelerated_G_commit=rt['code_commit'] if rt.get('G_teacher_acceleration') else previous.get('accelerated_G_commit'),
        resource_policy=str(own_limit)+'own/card,3total,free>=12000MiB;at most1G/G_student per card;2/3 reserve external until complete;timingexclusive',adopted=len(active))
    failures=Counter(previous.get('failures',{}));cache_failures=dict(previous.get('cache_failures',{}));children={};logs={}
    for job in jobs:
        key=job['id'];old=state['jobs'].get(key,{})
        if key in active:state['jobs'][key]=dict(status='RUNNING',**active[key]);continue
        if done(job):state['jobs'][key]=dict(old,status='COMPLETE');continue
        if old.get('status')=='RUNNING':
            reason=failure(job,old);failures[reason]+=1;state['jobs'][key]=dict(old,status='FAILED',failure=reason)
        elif not old:state['jobs'][key]=dict(status='QUEUED')
    if sum(by_id[k]['kind']=='score' for k in active)>1:raise ValueError('More than one existing scorer')
    write(root/'adoption.json',dict(status='VERIFIED_OWNED_CONTROLLER_HANDOVER',previous_owner_pid=previous['pid'],healthy_workers=active,scientific_worker_release=rt['worker_release']))
    write(Path(rt['previous_control_root'])/'multiprocess_handoff.json',dict(new_owner_pid=os.getpid(),new_control_root=str(root),new_runtime=rt['runtime_file']))
    turn=fturn=0
    while True:
        for key,item in list(active.items()):
            if alive(item['pid']):continue
            job=by_id[key]
            if key in logs:logs.pop(key).close()
            if key in children:children.pop(key).poll()
            state['jobs'][key].update(status='COMPLETE' if done(job) else 'FAILED')
            if not done(job):
                reason=failure(job,item);failures[reason]+=1;state['jobs'][key]['failure']=reason
            del active[key]
        failed_cache=[]
        for dataset in ('ManySig','ManyRx'):
            receipt=read(Path(rt['run_root'])/('observations-'+dataset+'-submit.json'))
            completion=Path(rt['run_root'])/'g_data'/dataset/'complete.json'
            if not completion.is_file() and not alive(receipt['pid']):
                failed_cache.append(str(completion))
                if dataset not in cache_failures:
                    tail=(Path(rt['run_root'])/('observations-'+dataset+'.log')).read_text(errors='replace')[-3000:]
                    reason=tail.splitlines()[-1] if tail.splitlines() else 'empty observation log';cache_failures[dataset]=reason;failures[reason]+=1
        block_failed_dependencies(jobs,state,failed_cache)
        systemic=bool(failures and max(failures.values())>=2)
        pending=[j for j in jobs if state['jobs'][j['id']]['status']=='QUEUED']
        eligible=[j for j in pending if ready(j)] if not systemic else []
        def launch(job,gpu):
            key=job['id'];path=root/'logs'/(job['log_name']+'.log');path.parent.mkdir(exist_ok=True)
            cmd,cwd,env=command_for(rt,job,gpu);log=path.open('x')
            child=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[key]=dict(pid=child.pid,gpu=gpu,log=str(path),adopted=False);children[key]=child;logs[key]=log
            state['jobs'][key]=dict(status='RUNNING',**active[key]);eligible.remove(job)
            if gpu is not None:
                deadline=time.monotonic()+40
                while child.poll() is None and time.monotonic()<deadline:
                    pids=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
                    if str(child.pid) in pids:break
                    time.sleep(.5)
        if not any(by_id[key]['kind']=='score' for key in active):
            job=next((j for j in eligible if j['kind']=='score'),None)
            if job:launch(job,None)
        reserve=external_pending(rt)
        if balance:balance.poll(by_id)
        gpu_order=balance.order(rt['allowed_gpus']) if balance else rt['allowed_gpus']
        for gpu in gpu_order:
            apps,free=capacity(gpu)
            candidates=[j for j in eligible if j['kind']!='score' and
                (balance.admits(gpu,j,active,by_id,apps,free,reserve) if balance else
                 gpu_admits(gpu,j,active,by_id,apps,free,reserve,own_limit,total_limit))]
            if not candidates:continue
            job=next((j for j in candidates if j.get('exclusive')),None)
            if job is None:job=next((j for j in candidates if j['kind']=='predict'),None)
            if job is None:
                wanted=('G','F','G_student')[turn%3];turn+=1
                group=[j for j in candidates if j['kind']==wanted]
                if wanted=='F' and group:
                    want_ablation=fturn%2==0;fturn+=1
                    job=next((j for j in group if ('F-ablation-' in j['id'])==want_ablation),group[0])
                else:job=group[0] if group else candidates[0]
            if balance and sum(v.get('gpu')==gpu for v in active.values())==3:
                balance.start_trial(gpu,active,by_id)
            launch(job,gpu)
        state.update(counts=dict(Counter(v['status'] for v in state['jobs'].values())),failures=dict(failures),cache_failures=cache_failures,
            systemic_stop_new_launches=systemic,reserve_external_GPUs_2_3=reserve,active_own_per_gpu=dict(Counter(v['gpu'] for v in active.values() if v['gpu'] is not None)))
        state['effective_counts']=dict(Counter(v['status'] for k,v in state['jobs'].items() if k not in state.get('excluded_job_ids',[])))
        if balance:
            state.update(load_balance=balance.snapshot(),resource_policy=balance.snapshot()['policy'])
        write(root/'pipeline_state.json',state,exclusive=False)
        if not active and (not pending or systemic):break
        time.sleep(30)
    state['status']=completion_status(state)
    write(root/'pipeline_state.json',state,exclusive=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(read(a.config))
