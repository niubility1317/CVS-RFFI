"""Adopt verified workers and immediately dispatch remaining registered jobs."""
import argparse
from collections import Counter
from functools import partial
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write

GPUS=(0,1,2,3,5,7)  # 4 and 6 belong to the independent low-weight KL dispatcher.


def live(pid):
    try:
        # A reparented zombie has exited and must not keep the queue alive.
        return Path('/proc',str(pid),'stat').read_text().split(') ',1)[1][0] != 'Z'
    except FileNotFoundError:
        return False


class Adopted:
    def __init__(self,pid):self.pid=pid;self.returncode=None
    def poll(self):
        if live(self.pid):return None
        # Completion artifacts, rather than an unavailable exit status, decide success.
        self.returncode=0;return 0


def verify_worker(item,job,rt):
    pid=item['pid'];proc=Path('/proc')/str(pid)
    if not live(pid):return False
    argv=proc.joinpath('cmdline').read_bytes().decode().split('\0')
    if (os.readlink(proc/'cwd')!=rt['release_root'] or job['config'] not in argv
            or 'experiments.pamf_repl_priority.runner' not in argv or 'worker' not in argv):
        raise ValueError('Running worker identity changed: '+job['id'])
    return True


def dispatch(rt):
    from experiments.practical_full_2x2.cache import atomic
    from experiments.cric_dura_comparison.pipeline import done,ready,block_failed_dependencies
    from experiments.pamf_repl_completion.runner import foreign_CPU_score
    from experiments.pamf_physics_branch.runner import admits as physical_admits
    from experiments.pamf_physics_route.runner import launch
    from experiments.pamf_branch_training.runner import choose_gpu_job
    from experiments.pamf_repl_component_guard import runner as base
    from experiments.pamf_repl_priority import plan
    old=read(rt['original_runtime_file']);root=Path(old['run_root'])
    state=read(root/'pipeline_state.json');jobs=read(root/'jobs.json');by={j['id']:j for j in jobs}
    if state['status']!='RUNNING' or state.get('failed_jobs') or state.get('smoke',{}).get('status')!='PASS':
        raise ValueError('Only a healthy smoke-passed original queue may be adopted')
    if live(state['pid']):raise ValueError('Original controller is still alive')
    if len(by)!=len(jobs) or set(by)!=set(state['jobs']):raise ValueError('Exact registered job set required')
    active={}
    for key,item in state['jobs'].items():
        if item['status']=='RUNNING':
            if verify_worker(item,by[key],old):active[key]=(Adopted(item['pid']),item)
            elif done(by[key]):state['jobs'][key]=dict(item,status='COMPLETE',exit_code=None,adopted_exit_status='unavailable')
            else:raise ValueError('Original running worker disappeared without complete artifact: '+key)
        elif item['status']=='QUEUED':
            if (root/'logs'/(key+'.log')).exists():
                raise ValueError('Queued job already has a log; reconcile before takeover: '+key)
    old_pid=state['pid'];state.update(pid=os.getpid(),cwd=os.getcwd(),argv=__import__('sys').argv,
        owner=rt['launch_owner'],previous_controller_pid=old_pid,takeover_release=rt['release_root'],
        takeover_code_commit=rt['code_commit'],adopted_workers=len(active),
        allowed_gpus=list(GPUS),max_own_per_gpu=2,admission_floor_mib=12288)
    out=Path(rt['run_root']);out.mkdir(parents=True,exist_ok=False)
    write(out/'takeover.json',dict(status='ADOPTED',original_run_root=str(root),
        previous_controller_pid=old_pid,adopted_workers={k:v[1] for k,v in active.items()},
        code_commit=rt['code_commit'],original_runtime=old))
    atomic(root/'pipeline_state.json',state)
    failed=[];nonblocking_failed=[];launch_worker=partial(launch,module='experiments.pamf_repl_priority.runner')
    while True:
        for key,(process,item) in list(active.items()):
            if process.poll() is None:continue
            ok=process.returncode==0 and done(by[key])
            adopted=isinstance(process,Adopted)
            state['jobs'][key]=dict(item,status='COMPLETE' if ok else 'FAILED',ended_at=time.time(),
                exit_code=None if adopted else process.returncode,
                **({'adopted_exit_status':'unavailable'} if adopted else {}))
            if not ok:(nonblocking_failed if by[key].get('nonblocking') is True else failed).append(key)
            del active[key]
        block_failed_dependencies(jobs,state,[])
        pending=[j for j in jobs if state['jobs'][j['id']]['status']=='QUEUED']
        candidates=[j for j in pending if ready(j)] if not failed else []
        scorer=next((j for j in candidates if j['kind'] in ('score','weak_score')),None)
        if scorer and not any(item['gpu'] is None for _,item in active.values()) and not foreign_CPU_score({p.pid for p,_ in active.values()}):
            process,item=launch_worker(old,scorer,None);active[scorer['id']]=(process,item)
            state['jobs'][scorer['id']]=item;candidates.remove(scorer)
        for gpu in GPUS:
            if sum(item['gpu']==gpu for _,item in active.values())>=2:continue
            if not physical_admits(gpu,active,max(state['admission_mib'],12288),max_total=5):continue
            available=[j for j in candidates if j['kind'] not in ('score','weak_score')]
            if not available:continue
            job=choose_gpu_job(available,True)
            process,item=launch_worker(old,job,gpu);active[job['id']]=(process,item)
            state['jobs'][job['id']]=item;candidates.remove(job)
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),
            failed_jobs=failed,nonblocking_failed_jobs=nonblocking_failed,stop_new_dispatch=bool(failed))
        atomic(root/'pipeline_state.json',state)
        if not active and (not pending or failed):break
        time.sleep(15)
    if failed or any(v['status']!='COMPLETE' for v in state['jobs'].values()):state['status']='PARTIAL_TECHNICAL_FAILURE'
    else:
        state['status']='ANALYZING';atomic(root/'pipeline_state.json',state)
        with plan.scoped():base.analyze(old)
        state['status']='ANALYZED'
    atomic(root/'pipeline_state.json',state)
    write(out/'complete.json',dict(status=state['status'],original_run_root=str(root),
        failed_jobs=failed,nonblocking_failed_jobs=nonblocking_failed))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch']);p.add_argument('--config',required=True)
    args=p.parse_args();dispatch(read(args.config))
