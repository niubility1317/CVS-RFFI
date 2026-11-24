"""One owner for source training, frozen prediction and independent CPU scoring."""

import argparse
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from experiments.cric_dura_comparison.pipeline import block_failed_dependencies, done, ready
from experiments.pamf_repl_view_shuffle.runner import activate, eligible, resources
from experiments.practical_full_2x2.cache import atomic

from experiments.pamf_wisig_branch_style.plan import (
    ARMS, ORIGINAL_ROW, ORIGINAL_RUN, RUN, SEED, TASKS, row_id, source_config,
)

WORKER_MODULE = "experiments.pamf_wisig_branch_style.runner"
WORKER_MEMORY_MIB = 5600
CARD_HEADROOM_MIB = 768


def prepare(runtime):
    context = read(runtime['context'])
    root = Path(runtime['run_root'])
    original = context['original_prediction']
    jobs = []
    for arm in ARMS:
        row = row_id(arm)
        train = source_config(runtime, context, arm)
        folder = root / 'readouts' / row
        predict = dict(
            run_id=RUN, row_id=row, model_seed=SEED, source_output=train['output_root'],
            source_contract=train['source_contract'], classes=context['classes'],
            readouts=['raw'], tests={key: {'views_root': context['tests'][key]['views_root']} for key in TASKS},
            output_root=str(folder / 'predictions'), frozen_sources=runtime['frozen_sources'])
        score = dict(run_id=RUN, row_id=row, model_seed=SEED, classes=context['classes'],
                     readouts=['raw'], tests=context['tests'], source_days=train['source_days'],
                     prediction_output=predict['output_root'], output_root=str(folder / 'scores'),
                     reference_prediction=dict(run_id=ORIGINAL_RUN, row_id=ORIGINAL_ROW,
                         classes=context['classes'], readouts=['raw'], tests=predict['tests'],
                         prediction_output=original))
        for kind, c, status in (
            ('train', train, 'SOURCE_FROZEN'),
            ('predict', predict, 'ALL_PREDICTIONS_COMPLETE'),
            ('score', score, 'SCORED_COMPLETE'),
        ):
            key = kind + '__' + row
            path = root / 'configs' / (key + '.json')
            dependency = [] if kind == 'train' else [jobs[-1]['completion']]
            write(path, c)
            jobs.append(dict(id=key, kind=kind, arm=arm, config=str(path), dependencies=dependency,
                             completion=str(Path(c['output_root']) / 'complete.json'),
                             successful_statuses=[status]))
    write(root / 'jobs.json', jobs)
    return jobs


def launch(runtime, key, kind, config, gpu):
    cwd = runtime['release_root']
    argv = [runtime['python'], '-u', '-m', WORKER_MODULE, 'worker', '--kind', kind, '--config', config]
    env = dict(os.environ, PYTHONPATH=cwd, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    path = Path(runtime['run_root']) / 'logs' / (key + '.log')
    with path.open('x') as stream:
        child = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    return child, dict(status='RUNNING', pid=child.pid, gpu=gpu, cwd=cwd, argv=argv, log=str(path))


def failure_label(log):
    lines = Path(log).read_text(encoding='utf-8', errors='replace').splitlines()
    return lines[-1] if lines else 'Empty worker log'


def dispatch(runtime):
    root = Path(runtime['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    jobs = prepare(runtime)
    state_path = root / 'pipeline_state.json'
    state = dict(status='WAITING_FOR_SMOKE_GPU', pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv,
                 code_commit=runtime['code_commit'], jobs={j['id']: {'status': 'QUEUED'} for j in jobs})
    atomic(state_path, state)
    print('WAITING_FOR_SMOKE_GPU', runtime['run_id'], flush=True)
    while True:
        available = resources()
        gpu = next((g for g in runtime['allowed_gpus'] if eligible(available[g], 0, 1)), None)
        if gpu is not None:
            break
        state['updated_at'] = time.time()
        state['gpu_availability'] = {str(g): dict(free_mib=available[g]['free'],
                                                  compute_slots=len(available[g]['pids']),
                                                  foreign_or_unknown=available[g]['blocked'])
                                     for g in runtime['allowed_gpus']}
        atomic(state_path, state)
        time.sleep(15)
    child, details = launch(runtime, 'smoke', 'smoke', runtime['runtime_file'], gpu)
    state.update(status='SMOKE_RUNNING', smoke=details)
    atomic(state_path, state)
    if child.wait() or read(root / 'smoke.json')['status'] != 'PASS':
        state.update(status='FAILED_NO_QUERY_SMOKE')
        atomic(state_path, state)
        return
    state['smoke']['status'] = 'PASS'

    run_jobs(runtime, jobs, state, {})


def run_jobs(runtime, jobs, state, active, *, shared_memory=False):
    from experiments.pamf_repl_completion.runner import foreign_CPU_score

    root = Path(runtime['run_root'])
    state_path = root / 'pipeline_state.json'
    by_id = {job['id']: job for job in jobs}
    failures = Counter(state.get('failed_fingerprints', {}))
    while True:
        for key, (child, details) in list(active.items()):
            status = child.poll()
            if status is None:
                continue
            success = (status == 0 or isinstance(child, AdoptedWorker)) and done(by_id[key])
            entry = state['jobs'][key] = dict(details, status='COMPLETE' if success else 'FAILED',
                                              exit_code=None if isinstance(child, AdoptedWorker) else status)
            if not success:
                label = failure_label(details['log'])
                failures[label] += 1
                entry['failure'] = label
            del active[key]
        block_failed_dependencies(jobs, state, [])
        systemic = any(n >= 2 for n in failures.values())
        queued = [j for j in jobs if state['jobs'][j['id']]['status'] == 'QUEUED']
        candidates = [j for j in queued if ready(j)] if not systemic else []
        scores = [j for j in candidates if j['kind'] == 'score']
        if (scores and not any(by_id[key]['kind'] == 'score' for key in active)
                and not foreign_CPU_score({p.pid for p, _ in active.values()})):
            job = scores[0]
            active[job['id']] = launch(runtime, job['id'], job['kind'], job['config'], None)
            state['jobs'][job['id']] = active[job['id']][1]
        gpu_jobs = sorted((j for j in candidates if j['kind'] != 'score'),
                          key=lambda j: j['kind'] != 'predict')
        if shared_memory:
            if gpu_jobs:
                free, resident = gpu_memory()
                assignments = assign_by_memory(runtime['allowed_gpus'], free, resident, active, len(gpu_jobs))
                for gpu in assignments:
                    job = gpu_jobs.pop(0)
                    active[job['id']] = launch(runtime, job['id'], job['kind'], job['config'], gpu)
                    state['jobs'][job['id']] = active[job['id']][1]
                    print('GPU_DISPATCH', job['id'], gpu, active[job['id']][0].pid, flush=True)
        else:
            available = resources() if gpu_jobs else {}
            for gpu in runtime['allowed_gpus']:
                if not gpu_jobs:
                    break
                own = sum(details['gpu'] == gpu for _, details in active.values())
                if eligible(available[gpu], own, 1):
                    job = gpu_jobs.pop(0)
                    active[job['id']] = launch(runtime, job['id'], job['kind'], job['config'], gpu)
                    state['jobs'][job['id']] = active[job['id']][1]
        state.update(status='RUNNING', updated_at=time.time(),
                     counts=dict(Counter(v['status'] for v in state['jobs'].values())),
                     failed_fingerprints=dict(failures), systemic_stop_new_launches=systemic)
        atomic(state_path, state)
        if not active and (not queued or systemic):
            break
        time.sleep(15)

    from experiments.pamf_wisig_branch_style.summary import analyze
    if any(done(j) for j in jobs if j['kind'] == 'score'):
        analyze(runtime, jobs)
    state['status'] = 'PARTIAL_TECHNICAL_FAILURE' if failures else 'ANALYZED'
    atomic(state_path, state)


def worker(c, kind):
    from experiments.pamf_repl_view_shuffle import evaluation
    if kind == 'score':
        return evaluation.score(c, reference=c['reference_prediction'])
    activate(c)
    from . import source
    if kind == 'smoke':
        jobs = read(Path(c['run_root']) / 'jobs.json')
        return source.smoke(c, [j['config'] for j in jobs if j['kind'] == 'train'])
    if kind == 'train':
        return source.train(c)
    return evaluation.predict(c, load_source=source.load_reader)


def gpu_memory():
    def query(fields):
        data = subprocess.check_output(['nvidia-smi', fields, '--format=csv,noheader,nounits'], text=True)
        return [tuple(x.strip() for x in line.split(',')) for line in data.splitlines() if line.strip()]

    free = {int(gpu): int(memory) for gpu, memory in query('--query-gpu=index,memory.free')}
    resident = {int(pid): int(memory) for pid, memory in query('--query-compute-apps=pid,used_gpu_memory')
                if memory.isdecimal()}
    return free, resident


def assign_by_memory(gpus, free, resident, active, requested):
    available = {gpu: free[gpu] - CARD_HEADROOM_MIB for gpu in gpus}
    workers = Counter()
    for child, details in active.values():
        gpu = details['gpu']
        if gpu is not None:
            available[gpu] -= max(0, WORKER_MEMORY_MIB - resident.get(child.pid, 0))
            workers[gpu] += 1
    assignments = []
    for _ in range(requested):
        candidates = [g for g in gpus if available[g] >= WORKER_MEMORY_MIB]
        if not candidates:
            break
        gpu = min(candidates, key=lambda g: (workers[g], -available[g], g))
        assignments.append(gpu)
        workers[gpu] += 1
        available[gpu] -= WORKER_MEMORY_MIB
    return assignments


def process_status(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None


def argv_of(pid):
    try:
        return (Path('/proc') / str(pid) / 'cmdline').read_bytes().decode().rstrip(chr(0)).split(chr(0))
    except FileNotFoundError:
        return []


class AdoptedWorker:
    def __init__(self, pid, birth):
        self.pid = pid
        self.birth = birth

    def poll(self):
        status = process_status(self.pid)
        return None if status and status[0] != 'Z' and status[1] == self.birth else 0


class DispatchInProgress(RuntimeError):
    pass


def handoff(runtime, previous_pid, controller_commit):
    root = Path(runtime['run_root'])
    state_path = root / 'pipeline_state.json'
    jobs = read(root / 'jobs.json')
    proc = Path('/proc') / str(previous_pid)
    original = [runtime['python'], '-u', '-m', WORKER_MODULE, 'dispatch', '--config', runtime['runtime_file']]
    before = process_status(previous_pid)
    if (not before or before[0] in ('T', 't', 'Z') or proc.stat().st_uid != os.getuid()
            or argv_of(previous_pid) != original or os.readlink(proc / 'cwd') != runtime['release_root']):
        raise ValueError('Original dispatcher identity differs')
    for attempt in range(3):
        os.kill(previous_pid, signal.SIGSTOP)
        retired = False
        try:
            for _ in range(100):
                if process_status(previous_pid) == ('T', before[1]):
                    break
                time.sleep(.02)
            else:
                raise RuntimeError('Original dispatcher did not pause')
            state = read(state_path)
            if (state['pid'] != previous_pid or state['status'] != 'RUNNING'
                    or state['code_commit'] != runtime['code_commit']
                    or state.get('smoke', {}).get('status') != 'PASS'
                    or read(root / 'smoke.json')['status'] != 'PASS'
                    or {job['id'] for job in jobs} != set(state['jobs'])):
                raise ValueError('Original running queue or completed smoke differs')

            children = {int(x) for x in (proc / 'task' / str(previous_pid) / 'children').read_text().split()}
            active = {}
            for job in jobs:
                key = job['id']
                details = state['jobs'][key]
                log = root / 'logs' / (key + '.log')
                if details['status'] == 'QUEUED' and (log.exists() or Path(job['completion']).exists()):
                    raise DispatchInProgress('Original dispatcher is publishing a new task')
                if details['status'] != 'RUNNING':
                    continue
                pid = details['pid']
                status = process_status(pid)
                command = [runtime['python'], '-u', '-m', WORKER_MODULE, 'worker',
                           '--kind', job['kind'], '--config', job['config']]
                if details['argv'] != command or details['log'] != str(log):
                    raise ValueError('Worker record does not belong to this queue')
                if status is not None and status[0] != 'Z':
                    if pid not in children:
                        raise ValueError('Recorded live worker is outside the original dispatcher')
                    try:
                        cwd = os.readlink(Path('/proc') / str(pid) / 'cwd')
                        argv = argv_of(pid)
                    except FileNotFoundError:
                        cwd, argv = None, []
                    after = process_status(pid)
                    if after and after[0] != 'Z' and (cwd != runtime['release_root'] or argv != command):
                        raise ValueError('Recorded worker command or working directory differs')
                active[key] = (AdoptedWorker(pid, status[1] if status else None), details)
            adopted = {child.pid for child, _ in active.values()}
            for pid in children - adopted:
                args = argv_of(pid)
                if WORKER_MODULE in args and 'worker' in args:
                    raise DispatchInProgress('Original dispatcher has an unpublished worker')

            # Only the exact paused controller is retired. Its workers were
            # launched as independent sessions and continue from their PIDs.
            os.kill(previous_pid, signal.SIGKILL)
            retired = True
            state.update(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, updated_at=time.time(),
                         controller_commit=controller_commit, previous_controller_pid=previous_pid,
                         adopted_workers=sorted(adopted), scheduling='free_memory',
                         worker_memory_mib=WORKER_MEMORY_MIB, card_headroom_mib=CARD_HEADROOM_MIB)
            atomic(state_path, state)
            print('CONTROLLER_ADOPTED', previous_pid, sorted(adopted), flush=True)
            return jobs, state, active
        except DispatchInProgress:
            if attempt == 2:
                raise
        finally:
            if not retired and process_status(previous_pid) == ('T', before[1]):
                os.kill(previous_pid, signal.SIGCONT)
        time.sleep(.5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'worker', 'resume'])
    parser.add_argument('--kind', choices=['smoke', 'train', 'predict', 'score'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--previous-pid', type=int)
    parser.add_argument('--controller-commit')
    args = parser.parse_args()
    config = read(args.config)
    if args.action == 'dispatch':
        dispatch(config)
    elif args.action == 'resume':
        if args.previous_pid is None or not args.controller_commit:
            parser.error('resume requires --previous-pid and --controller-commit')
        jobs, state, active = handoff(config, args.previous_pid, args.controller_commit)
        run_jobs(config, jobs, state, active, shared_memory=True)
    else:
        worker(config, args.kind)
