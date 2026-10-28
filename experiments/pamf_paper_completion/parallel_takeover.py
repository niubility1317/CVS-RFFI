"""Transfer the paper queue to one owner while retaining all live workers."""
import argparse
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def slots(cuda, pending, foreign, free_mib):
    if foreign or free_mib < 12000:
        return 0
    return max(0, 3 - len(cuda | pending))


def reconcile(jobs, state, live, done, log_exists):
    result = deepcopy(state)
    configs = {j['config']: j for j in jobs}
    if len(configs) != len(jobs) or set(live) - set(configs):
        raise ValueError('Unregistered or duplicate worker config')
    for j in jobs:
        item = result['jobs'][j['id']]
        current = live.get(j['config'])
        if current:
            if item['status'] not in ('QUEUED', 'RUNNING'):
                raise ValueError('Live worker conflicts with recorded status')
            if item['status'] == 'RUNNING' and item['pid'] != current['pid']:
                raise ValueError('Worker PID conflict')
            result['jobs'][j['id']] = dict(item, **current, status='RUNNING')
        elif done(j):
            result['jobs'][j['id']] = dict(item, status='COMPLETE')
        elif item['status'] == 'RUNNING' or (item['status'] == 'QUEUED' and log_exists(j)):
            raise ValueError('Unfinished worker or partial output needs reconciliation')
        elif item['status'] != 'QUEUED':
            raise ValueError('Unexpected incomplete job status')
    return result


def process_info(pid):
    p = Path('/proc') / str(pid)
    try:
        if p.joinpath('stat').read_text().split()[2] == 'Z':
            return None
        argv = [x.decode() for x in p.joinpath('cmdline').read_bytes().split(b'\0') if x]
        env = dict(x.split(b'=', 1) for x in p.joinpath('environ').read_bytes().split(b'\0') if b'=' in x)
        return {'argv': argv, 'cwd': str(p.joinpath('cwd').resolve()),
                'gpu': env.get(b'CUDA_VISIBLE_DEVICES', b'').decode()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def scan_live(rt, jobs):
    by_config = {j['config']: j for j in jobs}
    live = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            info = process_info(int(proc.name))
            if not info or 'experiments.pamf_paper_completion.runner' not in info['argv'] or 'worker' not in info['argv']:
                continue
            argv = info['argv']
            config = argv[argv.index('--config') + 1]
            if not config.startswith(rt['run_root'] + '/configs/'):
                continue
            if config not in by_config or config in live or info['cwd'] != rt['release_root']:
                raise ValueError('Worker identity or config is ambiguous')
            j = by_config[config]
            if argv[argv.index('--kind') + 1] != j['kind']:
                raise ValueError('Worker kind mismatch')
            gpu = None if info['gpu'] == '' else int(info['gpu'])
            live[config] = dict(pid=int(proc.name), gpu=gpu, cwd=info['cwd'], argv=argv,
                                log=str(Path(rt['run_root']) / 'logs' / (j['id'] + '.log')))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return live


def run(runtime_path):
    import json
    rt = json.loads(Path(runtime_path).read_text())
    sys.path.insert(0, rt['release_root'])
    from experiments.cric_dura_comparison.io import read
    from experiments.cric_dura_comparison.pipeline import done, ready, block_failed_dependencies
    from experiments.pamf_paper_completion.runner import launch, analyze
    from experiments.pamf_repl_completion.runner import capacity, processes, foreign_CPU_score
    from experiments.practical_full_2x2.cache import atomic

    root = Path(rt['run_root'])
    state_path = root / 'pipeline_state.json'
    jobs = read(root / 'jobs.json')
    initial = read(state_path)
    old_pid = initial['pid']
    info = process_info(old_pid)
    expected = [rt['python'], '-u', '-m', 'experiments.pamf_paper_completion.runner', 'dispatch', '--config', runtime_path]
    if not info or info['cwd'] != rt['release_root'] or info['argv'] != expected or initial['status'] != 'RUNNING':
        raise ValueError('Original owner is not the expected running dispatcher')
    os.kill(old_pid, signal.SIGSTOP)
    stopped = True
    try:
        for _ in range(100):
            try:
                if (Path('/proc') / str(old_pid) / 'stat').read_text().split()[2] in ('T', 't'):
                    break
            except FileNotFoundError:
                raise RuntimeError('Original dispatcher disappeared before handoff')
            time.sleep(.1)
        else:
            raise RuntimeError('Original dispatcher did not pause')
        state = read(state_path)
        live = scan_live(rt, jobs)
        state = reconcile(jobs, state, live, done, lambda j: (root / 'logs' / (j['id'] + '.log')).exists())
        if any(x['status'] not in ('QUEUED', 'RUNNING', 'COMPLETE') for x in state['jobs'].values()):
            raise ValueError('Existing failure blocks dispatch')
        os.kill(old_pid, signal.SIGTERM)
        os.kill(old_pid, signal.SIGCONT)
        stopped = False
        for _ in range(100):
            if process_info(old_pid) is None:
                break
            time.sleep(.1)
        if process_info(old_pid) is not None:
            raise RuntimeError('Old dispatcher did not exit; no new dispatch')
    finally:
        if stopped:
            os.kill(old_pid, signal.SIGCONT)

    state.update(pid=os.getpid(), previous_owner_pid=old_pid, status='RUNNING',
                 concurrency_limit_per_gpu=3, takeover='parallel_takeover')
    atomic(state_path, state)
    by = {j['id']: j for j in jobs}
    children = {}
    active = {j['id']: state['jobs'][j['id']] for j in jobs if state['jobs'][j['id']]['status'] == 'RUNNING'}
    failures = []
    turn = 0
    while True:
        for key, item in list(active.items()):
            if process_info(item['pid']) is not None:
                continue
            child = children.pop(key, None)
            if child:
                child.poll()
            ok = done(by[key])
            state['jobs'][key] = dict(item, status='COMPLETE' if ok else 'FAILED', ended_at=time.time())
            if not ok:
                failures.append(key)
            del active[key]
        block_failed_dependencies(jobs, state, [])
        pending = [j for j in jobs if state['jobs'][j['id']]['status'] == 'QUEUED']
        candidates = [j for j in pending if ready(j)] if not failures else []
        scorer = next((j for j in candidates if j['kind'] == 'score'), None)
        if scorer and not any(by[k]['kind'] == 'score' for k in active) and not foreign_CPU_score({i['pid'] for i in active.values()}):
            child, item = launch(rt, scorer, None)
            active[scorer['id']] = item
            children[scorer['id']] = child
            state['jobs'][scorer['id']] = item
            candidates.remove(scorer)
        reserved = processes()
        for gpu in rt['allowed_gpus']:
            cuda, free = capacity(gpu)
            try:
                foreign = any(Path('/proc', str(pid)).stat().st_uid != os.getuid() for pid in cuda)
            except FileNotFoundError:
                continue
            pending_pids = {pid for pid, g in reserved.items() if g == gpu} | {i['pid'] for i in active.values() if i['gpu'] == gpu}
            if not slots(cuda, pending_pids, foreign, free):
                continue
            mem = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
            if mem < 32 * 1024 ** 2:
                continue
            available = [j for j in candidates if j['kind'] != 'score']
            if not available:
                continue
            wanted = ('train', 'train', 'predict')[turn % 3]
            turn += 1
            job = next((j for j in available if j['kind'] == wanted), available[0])
            child, item = launch(rt, job, gpu)
            children[job['id']] = child
            active[job['id']] = item
            state['jobs'][job['id']] = item
            candidates.remove(job)
            reserved[child.pid] = gpu
        state.update(updated_at=time.time(), counts=dict(Counter(v['status'] for v in state['jobs'].values())),
                     failed_jobs=failures, stop_new_dispatch=bool(failures),
                     GPU_workers=sum(i['gpu'] is not None for i in active.values()))
        atomic(state_path, state)
        if not active and (not pending or failures):
            break
        time.sleep(5)
    if failures or any(v['status'] != 'COMPLETE' for v in state['jobs'].values()):
        state['status'] = 'PARTIAL_TECHNICAL_FAILURE'
    else:
        state['status'] = 'ANALYZING'
        atomic(state_path, state)
        analyze(rt)
        state['status'] = 'ANALYZED'
    atomic(state_path, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    run(parser.parse_args().runtime)
