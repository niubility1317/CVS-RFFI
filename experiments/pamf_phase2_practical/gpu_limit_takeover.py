"""Transfer one Phase2 queue to three GPUs, preserving running workers."""
import argparse
from collections import Counter
import os
from pathlib import Path
import signal
import subprocess
import time

from .cuda_takeover import Adopted, _check_children
from .external_accel_takeover import _workers, _check_pending
from .external_takeover import process
from .group1_dispatch import completed
from .gpu_limit_state import partition, stable_workers
from .io import atomic, read, require, write


def dispatch(rt):
    parent_rt = read(rt['takeover_runtime'])
    old_root = Path(parent_rt['run_root'])
    require(parent_rt['run_id'] == '20261030-phase2-issl-lora-refine-m4-r08', 'Unexpected queue owner')
    require(rt['allowed_gpus'] == [1, 5, 7], 'Three GPU selection differs')
    root = Path(rt['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    (root/'logs').mkdir(); (root/'job_results').mkdir()
    # Real source-only checkpoints; no target/query/truth access.
    from experiments.phase1_external_matched.runner import smoke
    smoke(dict(external_runtime=parent_rt['runtime_file'], run_root=str(root)))
    items = read(old_root/'jobs.json'); by = {j['id']: j for j in items}
    require(len(items) == len(by), 'Duplicate job ID')
    configs = {path: read(path) for path in {j['runtime_file'] for j in items}}
    state = read(old_root/'pipeline_state.json'); parent = state['controller_pid']
    owner = process(parent)
    require(state['status'] == 'RUNNING' and owner and owner['uid'] == os.getuid()
            and owner['cwd'] == parent_rt['release_root']
            and parent_rt['runtime_file'] in owner['argv']
            and 'experiments.pamf_phase2_practical.external_accel_takeover' in owner['argv']
            and '--job' not in owner['argv'], 'Old owner identity mismatch')
    os.kill(parent, signal.SIGSTOP); terminated = False
    try:
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            stat=Path('/proc',str(parent),'stat').read_text().split(') ',1)[1].split()[0]
            if stat in ('T','t'):break
            time.sleep(.05)
        else:raise RuntimeError('Old controller did not enter stopped state')
        time.sleep(.3)  # Already-forked children can reach exec before inspection.
        state = read(old_root/'pipeline_state.json')
        require(state['controller_pid'] == parent and state['status'] == 'RUNNING', 'Owner changed')
        observed = stable_workers(lambda: _workers(configs, by),
                                  lambda workers: _check_children(parent, workers), time.sleep)
        failed = {}
        for key in state['failed_jobs']:
            result = read(old_root/'job_results'/(key+'.json'))
            require(key in by and result['status'] == 'FAILED', 'Failed job evidence differs')
            failed[key] = result
        done = {j['id'] for j in items if j['id'] not in observed and completed(j)}
        require(not set(failed).intersection(done) and not set(failed).intersection(observed),
                'Failed job also completed or running')
        for key in state['active']:
            require(key in observed or key in done or key in failed, 'Unresolved active job')
        remaining, blocked, profile_errors = partition(by, observed, done, failed)
        _check_pending((by[k] for k in remaining | blocked), configs)
        write(root/'jobs.json', items)
        write(root/'owner.json', dict(pid=os.getpid(), owner=rt['launch_owner'],
              runtime=rt['runtime_file'], replacement_for=parent_rt['run_id']))
        write(root/'handoff_prepared.json', dict(old_controller_pid=parent,
              adopted_workers=list(observed.values()), complete=len(done), failed=list(failed),
              blocked=len(blocked), queued=len(remaining), allowed_gpus=rt['allowed_gpus']))
        os.kill(parent, signal.SIGTERM); os.kill(parent, signal.SIGCONT)
        deadline=time.monotonic()+5
        while process(parent) and time.monotonic()<deadline: time.sleep(.1)
        require(process(parent) is None, 'Old controller did not exit')
        terminated=True
    finally:
        if not terminated and process(parent): os.kill(parent, signal.SIGCONT)
    write(root/'handoff.json', dict(status='VERIFIED', old_controller_pid=parent,
          old_controller_exited=True, adopted_workers=list(observed.values()),
          model_workers_restarted=0, allowed_gpus=rt['allowed_gpus'],
          existing_failures_preserved=list(failed), scientific_config_unchanged=True))
    active={k:(Adopted(v['pid']),v) for k,v in observed.items()}
    while True:
        stop=(root/'USER_STOP.json').exists() or any((Path(c['run_root'])/'USER_STOP.json').exists() for c in configs.values())
        for key,(child,item) in list(active.items()):
            code=child.poll()
            if code is None: continue
            good=code==0 and completed(by[key])
            result=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code,ended_at=time.time(),
                        output=by[key]['output'],completion=by[key]['completion'],prediction=by[key].get('prediction'))
            write(root/'job_results'/(key+'.json'),result)
            if good: done.add(key)
            else:
                failed[key]=result;profile_errors.add((by[key]['origin'],by[key]['profile']))
            del active[key]
        for key in list(remaining):
            if ((by[key]['origin'],by[key]['profile']) in profile_errors
                or any(dep in failed or dep in blocked for dep in by[key]['dependencies'])):
                blocked.add(key);remaining.remove(key)
        available=sorted((by[k] for k in remaining if all(d in done for d in by[k]['dependencies'])),
                         key=lambda j:(j['priority'],j['origin']!='replacement',j['id']))
        def launch(job,gpu):
            cfg=configs[job['runtime_file']]
            if job['kind']=='score':
                predecessor=by[job['dependencies'][0]]
                require(completed(predecessor) and job['prediction']==predecessor['output'],
                        'Scorer prediction root mismatch')
            args=[cfg['python'],'-u','-m','experiments.pamf_phase2_practical.group1_dispatch',
                  '--runtime',cfg['runtime_file'],'--job',job['original_id']]
            env=dict(os.environ,PYTHONPATH=cfg['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                     OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            log=Path(cfg['run_root'])/'logs'/(job['original_id']+'.log')
            with log.open('x') as stream:
                child=subprocess.Popen(args,cwd=cfg['release_root'],env=env,stdin=subprocess.DEVNULL,
                                       stdout=stream,stderr=subprocess.STDOUT)
            item=dict(pid=child.pid,gpu=gpu,kind=job['kind'],profile=job['profile'],command=args,
                      cwd=cfg['release_root'],runtime_file=cfg['runtime_file'],log=str(log),started_at=time.time(),
                      origin=job['origin'],adopted=False,output=job['output'],completion=job['completion'],
                      prediction=job.get('prediction'))
            active[job['id']]=(child,item);remaining.remove(job['id']);available.remove(job)
            print('GPU_LIMIT_LAUNCH',job['id'],child.pid,gpu,flush=True)
        if not stop:
            if not any(v['kind']=='score' for _,v in active.values()):
                scorer=next((j for j in available if not j['gpu']),None)
                if scorer:launch(scorer,None)
            from .runner import occupancy
            devices=occupancy()
            for _ in (0,1):
                for gpu in rt['allowed_gpus']:
                    own=sum(v['gpu']==gpu for _,v in active.values())
                    if own>=2 or devices[gpu]['free']<rt['minimum_free_MiB']:continue
                    job=next((j for j in available if j['gpu']),None)
                    if job is None:continue
                    launch(job,gpu);devices[gpu]['free']-=rt['minimum_free_MiB']
        counts=dict(COMPLETE=len(done),FAILED=len(failed),BLOCKED=len(blocked),RUNNING=len(active),QUEUED=len(remaining))
        status='STOP_REQUESTED' if stop else 'RUNNING'
        if not active and not remaining:status='COMPLETE' if not failed else 'PARTIAL_TECHNICAL_FAILURE'
        atomic(root/'pipeline_state.json',dict(status=status,run_id=rt['run_id'],controller_pid=os.getpid(),
            counts=counts,active={k:v for k,(_,v) in active.items()},updated_at=time.time(),
            failed_jobs=list(failed),previous_run=parent_rt['run_id'],allowed_gpus=rt['allowed_gpus'],
            draining_gpus=sorted({v['gpu'] for _,v in active.values() if v['gpu'] is not None and v['gpu'] not in rt['allowed_gpus']})))
        if (not active and not remaining) or (stop and not active):break
        time.sleep(rt.get('poll_seconds',10))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',required=True);args=parser.parse_args()
    dispatch(read(args.runtime))
