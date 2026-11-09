"""Adopt this run's live workers and pilot one observed low-memory row on GPU 4."""
import argparse
from collections import Counter
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

PILOT = 'train__main_split-branch-ce-no_path_no_spectrum-s2026092702'


def select_pilot(jobs, *, gpu, own, free_mib, busy, io, mem_kib, used):
    if gpu != 4 or own != 0 or used or free_mib < 8500 or busy >= 85 or io >= 5 or mem_kib < 64*1024**2:
        return None
    return next((j for j in jobs if j['id'] == PILOT and j['kind'] == 'train'), None)


def limits(gpu):
    return 1 if gpu == 7 else 3


def can_start(gpu, own, free_mib):
    return own < limits(gpu) and free_mib >= 12000


def command(rt, job):
    return [rt['python'], '-u', '-m', 'experiments.pamf_split_spectrum_paper.runner',
            'worker', '--kind', job['kind'], '--config', job['config']]


def matches_worker(rt, job, cwd, argv):
    return cwd == rt['source_release'] and argv == command(rt, job)


def matches_dispatcher(rt, cwd, argv):
    return cwd == rt['source_release'] and argv == [rt['python'], '-u', '-m',
        'experiments.pamf_split_spectrum_paper.runner', 'dispatch', '--config', rt['source_release']+'/runtime.json']


def current_workers(rt, jobs):
    """Find only exact original release/CWD/config workers, including launch gaps."""
    by_config = {j['config']: j for j in jobs}
    found = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            pid = int(p.name)
            if not alive(pid) or os.readlink(p/'cwd') != rt['source_release']:
                continue
            argv = [x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if '--config' not in argv:
                continue
            cfg = argv[argv.index('--config') + 1]
            job = by_config.get(cfg)
            if not job or not matches_worker(rt, job, os.readlink(p/'cwd'), argv):
                continue
            if job['id'] in found:
                raise ValueError('Duplicate exact worker for ' + job['id'])
            env = dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
            value = env.get(b'CUDA_VISIBLE_DEVICES', b'').decode()
            gpu = None if job['kind'] == 'score' else int(value)
            found[job['id']] = dict(status='RUNNING', pid=pid, gpu=gpu, cwd=rt['source_release'],
                argv=argv, log=str(Path(rt['run_root'])/'logs'/(job['id']+'.log')), adopted=True)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def run(rt):
    root = Path(rt['run_root']); control = Path(rt['control_root'])
    control.mkdir(parents=True, exist_ok=False)
    jobs = read(root/'jobs.json'); by = {j['id']: j for j in jobs}
    previous = read(root/'pipeline_state.json'); old = previous['pid']; proc = Path('/proc')/str(old)
    argv = [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    if previous['status'] != 'RUNNING' or not alive(old) or not matches_dispatcher(rt, os.readlink(proc/'cwd'), argv):
        raise ValueError('Original dispatcher identity/status changed; no process stopped')
    write(control/'handoff_started.json', dict(status='TAKEOVER_STARTED', previous_pid=old))
    os.kill(old, signal.SIGTERM)
    for _ in range(100):
        if not alive(old): break
        time.sleep(.1)
    if alive(old): raise RuntimeError('Original dispatcher remains alive; abort takeover')
    previous = read(root/'pipeline_state.json')
    # A detached child may be between fork and exec when the old dispatcher is
    # stopped. Allow its argv to settle before treating a pre-created log as a
    # failed launch. Never dispatch from the new owner during this interval.
    observed = {}
    for _ in range(13):
        observed.update(current_workers(rt, jobs))
        time.sleep(1)
    observed = {key:item for key,item in observed.items() if alive(item['pid'])}
    state = dict(status='RUNNING', pid=os.getpid(), owner=rt['launch_owner'], code_commit=rt['code_commit'],
                 source_model_commit=rt['source_model_commit'], previous_pid=old, jobs={}, failures=previous.get('failures', {}))
    failures = Counter(state['failures'])
    for j in jobs:
        key = j['id']; prior = previous['jobs'][key]; logfile = root/'logs'/(key+'.log')
        if key in observed:
            state['jobs'][key] = observed[key]
        elif done(j):
            state['jobs'][key] = dict(prior, status='COMPLETE')
        elif ((prior['status'] == 'QUEUED' and logfile.exists()) or
              (prior['status'] == 'RUNNING' and alive(prior.get('pid', -1)))):
            # A child could still be paused before exec. Do not misclassify or
            # relaunch this row; continue looking for the exact worker.
            state['jobs'][key] = dict(status='RECONCILING_LAUNCH', log=str(logfile))
        elif prior['status'] in ('RUNNING', 'COMPLETE'):
            lines = logfile.read_text(errors='replace').splitlines() if logfile.exists() else []
            reason = re.sub(r's202609270[123]', 'sSEED', lines[-1] if lines else 'missing completion')
            failures[reason] += 1
            state['jobs'][key] = dict(prior, status='FAILED', failure=reason)
        else:
            state['jobs'][key] = prior
    atomic(control/'pipeline_state.json', state)
    write(root/'handoff.json', dict(status='HANDOFF', previous_pid=old, new_pid=os.getpid(),
          control_root=str(control), previous_release=rt['source_release'], adopted=list(observed)))
    previous_sample = resource_sample(); time.sleep(1)
    turn = 0
    pilot_used = state['jobs'][PILOT]['status'] != 'QUEUED'
    while True:
        sample = resource_sample(); guard = resource_guard(previous_sample, sample); previous_sample = sample
        reconciling = [key for key,item in state['jobs'].items() if item['status']=='RECONCILING_LAUNCH']
        if reconciling:
            observed = current_workers(rt, jobs)
            for key in reconciling:
                if key in observed:
                    state['jobs'][key] = observed[key]
                elif done(by[key]):
                    state['jobs'][key] = dict(status='COMPLETE', reconciled=True)
        for key, item in list(state['jobs'].items()):
            if item['status'] != 'RUNNING' or alive(item['pid']): continue
            ok = done(by[key]); item.update(status='COMPLETE' if ok else 'FAILED', ended_at=time.time())
            if not ok:
                lines = Path(item['log']).read_text(errors='replace').splitlines()
                reason = re.sub(r's202609270[123]', 'sSEED', lines[-1] if lines else 'missing completion')
                failures[reason] += 1; item['failure'] = reason
        block_failed_dependencies(jobs, state, [])
        systemic = any(n >= 2 or any(t in reason for t in ('CONTRACT_MISMATCH','PROVENANCE','Truth closed','U truth','Output collision'))
                       for reason,n in failures.items())
        pending = [j for j in jobs if state['jobs'][j['id']]['status'] == 'QUEUED']
        uncertain = any(v['status']=='RECONCILING_LAUNCH' for v in state['jobs'].values())
        ready_jobs = [j for j in pending if ready(j)] if not systemic and not uncertain else []
        candidates = list(ready_jobs) if guard['allowed'] else []
        active = {k:v for k,v in state['jobs'].items() if v['status']=='RUNNING'}
        def start(job, gpu):
            path = root/'logs'/(job['id']+'.log')
            env = dict(os.environ, PYTHONPATH=rt['source_release'], CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            args = command(rt, job)
            with path.open('x') as log:
                child = subprocess.Popen(args, cwd=rt['source_release'], env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            item = dict(status='RUNNING', pid=child.pid, gpu=gpu, cwd=rt['source_release'], argv=args,
                        log=str(path), started_at=time.time())
            state['jobs'][job['id']] = item; active[job['id']] = item
        score = next((j for j in candidates if j['kind']=='score'), None)
        if score and not any(by[k]['kind']=='score' for k in active) and not foreign_CPU_score({v['pid'] for v in active.values()}):
            start(score, None); candidates.remove(score)
        for gpu in range(8):
            _, free = capacity(gpu)
            own = sum(v.get('gpu')==gpu for v in active.values())
            pilot = select_pilot(ready_jobs, gpu=gpu, own=own, free_mib=free,
                busy=guard.get('cpu_busy_percent', 100), io=guard.get('iowait_percent', 100),
                mem_kib=sample['available_kib'], used=pilot_used)
            if pilot:
                start(pilot, gpu)
                pilot_used = True
                if pilot in candidates:
                    candidates.remove(pilot)
                ready_jobs.remove(pilot)
                continue
            if not can_start(gpu, own, free): continue
            usable = [j for j in candidates if j['kind']!='score']
            if not usable: continue
            preferred = 'predict' if turn%3==2 else 'train'
            job = next((j for j in usable if j['kind']==preferred), usable[0]); turn += 1
            start(job, gpu); candidates.remove(job)
            ready_jobs.remove(job)
            if job['id'] == PILOT:
                pilot_used = True
        state.update(updated_at=time.time(), counts=dict(Counter(v['status'] for v in state['jobs'].values())),
                     failures=dict(failures), systemic_stop_new_launches=systemic, resource_guard=guard,
                     own_gpu_limits={'0-6':3,'7':1}, other_owners_counted=False,
                     pilot_id=PILOT, pilot_used=pilot_used, pilot_min_free_mib=8500, pilot_max_cpu_percent=85)
        atomic(control/'pipeline_state.json', state)
        if not active and (not pending or systemic) and not uncertain: break
        time.sleep(15)
    if all(v['status']=='COMPLETE' for v in state['jobs'].values()):
        from .analyze import run as analyze
        analyze(rt); state['status']='ANALYZED'
    else: state['status']='PARTIAL_TECHNICAL_FAILURE'
    state['ended_at']=time.time(); atomic(control/'pipeline_state.json', state)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); args=p.parse_args(); run(read(args.config))
