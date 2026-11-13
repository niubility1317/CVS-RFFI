"""Adopt two owned queues without restarting workers; respect shared GPU limits."""
import argparse
from collections import Counter
import os
from pathlib import Path
import signal
import sys
import time

PROJECT = Path('/home/szu2070436088/2510044040/CV-SincNet')
RUNS = (
    ('20261112-phase1-pamf-prototype-fullsource-tuning-m33-r01',
     'experiments.pamf_physics_tri.target_tuning_runner', 1843699),
    ('20261112-phase1-pamf-coverage-tenclass-m9-r01',
     'experiments.pamf_physics_tri.coverage_runner', 1881305),
)
LIMITS = {gpu: (4 if gpu == 7 else 6) for gpu in range(8)}


def may_launch(cuda, pending, own, free_mib, admission_mib, limit, young_own=frozenset()):
    known = cuda | pending | own
    # CUDA becomes visible before the worker reaches its model-memory peak.
    initializing = len(((pending | own) - cuda) | young_own)
    return len(known) < limit and free_mib >= admission_mib * (1 + initializing)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def process_info(pid):
    root = Path('/proc') / str(pid)
    try:
        if not alive(pid):
            return None
        argv = [x.decode() for x in root.joinpath('cmdline').read_bytes().split(b'\0') if x]
        env = dict(x.split(b'=', 1) for x in root.joinpath('environ').read_bytes().split(b'\0') if b'=' in x)
        return dict(argv=argv, cwd=str(root.joinpath('cwd').resolve()),
                    gpu=env.get(b'CUDA_VISIBLE_DEVICES', b'').decode())
    except (FileNotFoundError, ProcessLookupError):
        return None


def child_pids(parent):
    children = Path(f'/proc/{parent}/task/{parent}/children').read_text().split()
    return {int(pid) for pid in children if alive(int(pid))}


def check_owner(rt, state, module, pid):
    expected = [rt['python'], '-u', '-m', module, 'dispatch', '--config', rt['runtime_file']]
    info = process_info(pid)
    if (state['status'] != 'RUNNING' or state['pid'] != pid or info is None
            or info['argv'] != expected or info['cwd'] != rt['release_root']):
        raise ValueError(f'Exact owned running controller required: {pid}')


def adopt_worker(item):
    info = process_info(item['pid'])
    if info is None:
        return
    if (info['cwd'] != item['cwd'] or info['argv'] != item['argv']
            or info['gpu'] != ('' if item['gpu'] is None else str(item['gpu']))):
        raise ValueError(f"Running worker identity differs: {item['pid']}")


def read_queues(read):
    queues = []
    for run, module, pid in RUNS:
        release = PROJECT / 'releases' / (run + '-release')
        rt = read(release / 'runtime.json')
        root = PROJECT / 'runs' / run
        state = read(root / 'pipeline_state.json')
        check_owner(rt, state, module, pid)
        queues.append(dict(run=run, module=module, old_pid=pid, rt=rt, root=root, state=state))
    return queues


def handoff(queues, read):
    stopped = []
    try:
        for queue in queues:
            os.kill(queue['old_pid'], signal.SIGSTOP)
            stopped.append(queue['old_pid'])
        for queue in queues:
            pid = queue['old_pid']
            deadline = time.monotonic() + 2
            while Path(f'/proc/{pid}/stat').read_text().split()[2] not in ('T', 't') and time.monotonic() < deadline:
                time.sleep(.02)
            if Path(f'/proc/{pid}/stat').read_text().split()[2] not in ('T', 't'):
                raise ValueError(f'Owner did not pause: {pid}')
            state = read(queue['root'] / 'pipeline_state.json')
            check_owner(queue['rt'], state, queue['module'], pid)
            running = {item['pid'] for item in state['jobs'].values() if item['status'] == 'RUNNING'}
            if not child_pids(pid) <= running:
                raise ValueError(f'Owner has an unrecorded live child: {pid}')
            for item in state['jobs'].values():
                if item['status'] == 'RUNNING':
                    adopt_worker(item)
            queue['state'] = state
            queue['jobs'] = read(queue['root'] / 'jobs.json')
        for pid in stopped:
            os.kill(pid, signal.SIGKILL)
        stopped.clear()
    finally:
        for pid in stopped:
            os.kill(pid, signal.SIGCONT)


def run(queues, read, atomic):
    from experiments.cric_dura_comparison.pipeline import block_failed_dependencies, done, ready
    from experiments.pamf_physics_route.runner import launch
    from experiments.pamf_repl_completion.runner import capacity, foreign_CPU_score, processes

    for q in queues:
        q['by'] = {job['id']: job for job in q['jobs']}
        q['active'] = {key: (None, item) for key, item in q['state']['jobs'].items()
                       if item['status'] == 'RUNNING'}
        q['failed'] = list(q['state'].get('failed_jobs', []))
        q['state'].update(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv,
                          prior_controller_pid=q['old_pid'], resource_limits=LIMITS,
                          scheduler='shared takeover; original worker code and output roots unchanged')
        atomic(q['root'] / 'pipeline_state.json', q['state'])
    turn = 0
    while queues:
        for q in list(queues):
            state, active = q['state'], q['active']
            for key, (process, item) in list(active.items()):
                if process is None:
                    finished = not alive(item['pid'])
                    exit_code = None
                else:
                    exit_code = process.poll()
                    finished = exit_code is not None
                if not finished:
                    continue
                ok = (exit_code in (None, 0)) and done(q['by'][key])
                state['jobs'][key] = dict(item, status='COMPLETE' if ok else 'FAILED',
                                          ended_at=time.time(), exit_code=exit_code)
                if ok and key == 'smoke':
                    peak = max(entry['peak_allocated_bytes'] for entry in read(q['root'] / 'smoke.json')['checks'])
                    state['admission_mib'] = max(6000, int(peak / 1024**2) + 2000)
                if not ok:
                    q['failed'].append(key)
                del active[key]
            block_failed_dependencies(q['jobs'], state, [])

        active_items = [(q, key, process, item) for q in queues
                        for key, (process, item) in q['active'].items()]
        owned_pids = {item['pid'] for _, _, _, item in active_items}
        available = [(q, job) for q in queues if not q['failed'] for job in q['jobs']
                     if q['state']['jobs'][job['id']]['status'] == 'QUEUED' and ready(job)
                     and ('smoke' not in q['state']['jobs'] or job['kind'] != 'train'
                          or q['state']['jobs']['smoke']['status'] == 'COMPLETE')]
        scoring = [(q, j) for q, j in available if j['kind'] in ('score', 'weak_score')]
        if scoring and not any(item['gpu'] is None for _, _, _, item in active_items) and not foreign_CPU_score(owned_pids):
            q, job = scoring[0]
            process, item = launch(q['rt'], job, None, module=q['module'])
            q['active'][job['id']] = (process, item)
            q['state']['jobs'][job['id']] = item
            available.remove((q, job))

        for gpu in range(8):
            while True:
                candidates = [(q, j) for q, j in available if j['kind'] not in ('score', 'weak_score')]
                if not candidates:
                    break
                urgent = next(((q, j) for q, j in candidates if j['kind'] == 'predict'), None)
                if urgent is None:
                    urgent = next(((q, j) for q, j in candidates if j['kind'] == 'smoke'), None)
                if urgent is None:
                    for offset in range(len(queues)):
                        index = (turn + offset) % len(queues)
                        selected = next(((q, j) for q, j in candidates if q is queues[index]), None)
                        if selected is not None:
                            urgent = selected
                            turn = (index + 1) % len(queues)
                            break
                q, job = urgent
                cuda, free = capacity(gpu)
                pending = {pid for pid, card in processes().items() if card == gpu}
                own = {item['pid'] for other in queues for _, item in other['active'].values() if item['gpu'] == gpu}
                young = {item['pid'] for other in queues for _, item in other['active'].values()
                         if item['gpu'] == gpu and time.time() - item['started_at'] < 90}
                if not may_launch(cuda, pending, own, free, q['state']['admission_mib'], LIMITS[gpu], young):
                    break
                process, item = launch(q['rt'], job, gpu, module=q['module'])
                q['active'][job['id']] = (process, item)
                q['state']['jobs'][job['id']] = item
                available.remove((q, job))

        for q in list(queues):
            state = q['state']
            state.update(updated_at=time.time(), counts=dict(Counter(item['status'] for item in state['jobs'].values())),
                         failed_jobs=q['failed'], stop_new_dispatch=bool(q['failed']))
            atomic(q['root'] / 'pipeline_state.json', state)
            pending = [item for item in state['jobs'].values() if item['status'] == 'QUEUED']
            if q['active'] or (pending and not q['failed']):
                continue
            if q['failed'] or any(item['status'] != 'COMPLETE' for item in state['jobs'].values()):
                state['status'] = 'PARTIAL_TECHNICAL_FAILURE'
            else:
                state['status'] = 'ANALYZING'
                atomic(q['root'] / 'pipeline_state.json', state)
                if q['module'].endswith('target_tuning_runner'):
                    from experiments.pamf_physics_tri.target_tuning_report import analyze
                elif q['module'].endswith('hybrid_fullbranch_runner'):
                    from experiments.pamf_physics_tri.hybrid_fullbranch_report import analyze
                elif q['module'].endswith('hybrid_structure_runner'):
                    from experiments.pamf_physics_tri.hybrid_structure_report import analyze
                elif q['module'].endswith('hybrid_tracked_runner'):
                    from experiments.pamf_physics_tri.hybrid_tracked_report import analyze
                elif q['module'].endswith('hybrid_low_runner'):
                    from experiments.pamf_physics_tri.hybrid_low_report import analyze
                elif q['module'].endswith('hybrid_weight_runner'):
                    from experiments.pamf_physics_tri.hybrid_weight_report import analyze
                elif q['module'].endswith('hybrid_confidence_runner'):
                    from experiments.pamf_physics_tri.hybrid_confidence_report import analyze
                elif q['module'].endswith('confidence_runner'):
                    from experiments.pamf_physics_tri.confidence_runner import analyze
                else:
                    from experiments.pamf_physics_tri.coverage_runner import analyze
                analyze(q['rt'])
                state['status'] = 'ANALYZED'
            atomic(q['root'] / 'pipeline_state.json', state)
            queues.remove(q)
        if queues:
            time.sleep(15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(PROJECT / 'releases/20261112-phase1-pamf-coverage-tenclass-m9-r01-release'))
    from experiments.cric_dura_comparison.io import read
    from experiments.practical_full_2x2.cache import atomic
    queues = read_queues(read)
    if args.check:
        print('READY', flush=True)
        return
    handoff(queues, read)
    run(queues, read, atomic)


if __name__ == '__main__':
    main()
