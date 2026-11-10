"""Transfer the existing controller while draining reserved physical GPUs."""
import argparse,os,signal,time
from collections import Counter
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from experiments.cric_dura_comparison.pipeline import alive,done
from experiments.practical_full_2x2.cache import atomic
from .speed_control import current_workers,run_loop


def run(rt):
    previous=Path(rt['previous_control']);control=Path(rt['control_root']);root=Path(rt['run_root'])
    if rt['dispatch_gpus']!=[2,3,4,5,6,7]:raise ValueError('Unexpected device reservation')
    state=read(previous/'pipeline_state.json');old=state['pid'];proc=Path('/proc')/str(old)
    expected=[rt['python'],'-u','-m','experiments.pamf_no_reference_completion.speed_control','--config',rt['previous_release']+'/runtime.json']
    argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    if state['status']!='RUNNING' or not alive(old) or os.readlink(proc/'cwd')!=rt['previous_release'] or argv!=expected:
        raise ValueError('Previous owner changed; no process stopped')
    if (previous/'handoff.json').exists():raise ValueError('Existing handoff; reconcile first')
    control.mkdir(parents=True,exist_ok=False)
    write(control/'handoff_started.json',dict(status='TAKEOVER_STARTED',previous_pid=old,dispatch_gpus=rt['dispatch_gpus']))
    os.kill(old,signal.SIGTERM)
    for _ in range(100):
        if not alive(old):break
        time.sleep(.1)
    if alive(old):raise RuntimeError('Prior controller remains alive')
    state=read(previous/'pipeline_state.json');jobs=read(previous/'jobs.json');sources=read(root/'sources.json');observed={}
    for _ in range(13):observed.update(current_workers(rt,jobs));time.sleep(1)
    observed={k:v for k,v in observed.items() if alive(v['pid'])};failures=Counter(state.get('failures',{}))
    for job in jobs:
        k=job['id'];prior=state['jobs'][k];log=Path(job.get('recovery_log',str(root/'logs'/(k+'.log'))))
        if k in observed:state['jobs'][k]=observed[k]
        elif done(job):state['jobs'][k]=dict(prior,status='COMPLETE')
        elif ((prior['status']=='QUEUED' and log.exists()) or (prior['status']=='RUNNING' and alive(prior.get('pid',-1)))):
            state['jobs'][k]=dict(status='RECONCILING_LAUNCH',log=str(log),was_queued=prior['status']=='QUEUED',empty_scans=0)
        elif prior['status']=='RUNNING':
            # No known completion is never silently retried or discarded.
            failures['worker exited without completion during handoff']+=1
            state['jobs'][k]=dict(prior,status='FAILED',failure='worker exited without completion during handoff')
    accelerated=[j['id'] for j in jobs if j.get('worker_release')==rt['release_root']]
    state.update(status='RUNNING',pid=os.getpid(),previous_pid=old,owner=rt['launch_owner'],controller_commit=rt['code_commit'],dispatch_gpus=rt['dispatch_gpus'])
    write(control/'jobs.json',jobs);atomic(control/'pipeline_state.json',state)
    write(previous/'handoff.json',dict(status='HANDOFF',previous_pid=old,new_pid=os.getpid(),control_root=str(control),
        dispatch_gpus=rt['dispatch_gpus'],adopted=list(observed),draining=[k for k,v in observed.items() if v['gpu'] in (0,1)]))
    run_loop(rt,control,state,jobs,sources,failures,accelerated)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(read(p.parse_args().config))
