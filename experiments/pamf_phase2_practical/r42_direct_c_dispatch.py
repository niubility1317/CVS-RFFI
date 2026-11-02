"""One-owner MIX01 direct-C queue, with global truth-last scoring barrier."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .io import read, write, atomic, require


def jobs(rt):
    rx = rt['target_receivers']
    require(len(rx) == 7 and len(set(rx)) == 7 and rt['model_seeds'] == [2026092701, 2026092702, 2026092703, 2026092704], 'Wrong MIX01 dimensions')
    capsules = [p for p in rt['capsules'] if '/MIX01/' in p]
    expected = {f'capsules/P{pool:02d}/MIX01/{r}' for pool in (1, 2, 3) for r in rx}
    require(len(capsules) == 21 and set(capsules) == expected, 'Wrong MIX01 capsules')
    points = [[k, n] for k in (1, 5, 10, 20) for n in (2, 5, 10, 20)]
    return [dict(key=f's{seed}-c{i:02d}', seed=seed, capsule=cap, points=points)
            for seed in rt['model_seeds'] for i, cap in enumerate(sorted(capsules))]


def gpu_free():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    return {int(a.strip()): int(b.strip()) for a, b in (line.split(',') for line in output.splitlines())}


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    for name in ('sources', 'tasks', 'jobs', 'logs', 'scores'):
        (root / name).mkdir()
    work = jobs(rt)
    gpu_ids = rt['gpu_ids']
    require(len(gpu_ids) == len(set(gpu_ids)) and set(gpu_ids) <= set(range(8)), 'Invalid GPU list')
    active = {}
    pending = []
    for seed in rt['model_seeds']:
        pending.append(('source', str(seed), [sys.executable, '-u', '-m',
            'experiments.pamf_phase2_practical.r42_direct_c_worker', 'source', '--runtime', runtime, '--seed', str(seed)]))
    phase = 'sources'
    completed = {'sources': 0, 'predictions': 0, 'scores': 0}
    while True:
        for key, (process, logfile, device) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            logfile.close()
            del active[key]
            if code:
                write(root / 'failed.json', dict(phase=phase, job=key, exit_code=code))
                # In-flight owned workers are allowed to finish; no outside process is touched.
                for p, log, _ in active.values():
                    p.wait()
                    log.close()
                raise RuntimeError('Owned direct-C worker failed: ' + key)
            completed[phase] += 1
        if not pending and not active:
            if phase == 'sources':
                for task in work:
                    job = dict(task, output=str(root / 'tasks' / task['key'] / 'predictions'))
                    jobpath = root / 'jobs' / (task['key'] + '.json')
                    write(jobpath, job)
                    pending.append(('prediction', task['key'], [sys.executable, '-u', '-m',
                        'experiments.pamf_phase2_practical.r42_direct_c_worker', 'predict', '--runtime', runtime, '--job', str(jobpath)]))
                phase = 'predictions'
            elif phase == 'predictions':
                require(completed['predictions'] == len(work), 'Missing prediction tasks')
                for task in work:
                    pred = root / 'tasks' / task['key'] / 'predictions'
                    done = read(pred / 'complete.json')
                    require(done['status'] == 'PREDICTIONS_COMPLETE' and done['rows'] == 128
                            and done['truth_read'] is False and done['query_fit'] is False
                            and done['target_old_support_read'] is False, 'Incomplete prediction closure')
                write(root / 'predictions_complete.json', dict(status='ALL_PREDICTIONS_FIXED', units=len(work), rows=len(work)*128))
                for task in work:
                    pending.append(('score', task['key'], [sys.executable, '-u', '-m',
                        'experiments.pamf_phase2_practical.r42_direct_c_dispatch', 'score', '--runtime', runtime, '--key', task['key']]))
                phase = 'scores'
            else:
                require(completed['scores'] == len(work), 'Missing scoring tasks')
                write(root / 'scoring_complete.json', dict(status='SCORED_COMPLETE', units=len(work), rows=len(work)*128))
                atomic(root / 'worker_state.json', dict(phase='complete', completed=completed, active=0, pending=0))
                return
        free = gpu_free() if phase in ('sources', 'predictions') and pending else {}
        available = [g for g in gpu_ids if g not in {entry[2] for entry in active.values()} and free.get(g, 0) >= 6000]
        capacity = len(available) if phase != 'scores' else max(0, 4 - len(active))
        while pending and capacity:
            kind, key, argv = pending.pop(0)
            device = available.pop(0) if phase != 'scores' else None
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '1'
            if device is not None:
                env['CUDA_VISIBLE_DEVICES'] = str(device)
            log = (root / 'logs' / (kind + '-' + key + '.log')).open('x', encoding='utf-8')
            process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            active[kind + '-' + key] = (process, log, device)
            capacity -= 1
        atomic(root / 'worker_state.json', dict(phase=phase, completed=completed, pending=len(pending),
              active=[dict(key=k, pid=v[0].pid, gpu=v[2]) for k, v in active.items()]))
        time.sleep(5)


def score(runtime, key):
    from .episodic_score import run
    rt = read(runtime)
    root = Path(rt['run_root'])
    require(read(root / 'predictions_complete.json')['status'] == 'ALL_PREDICTIONS_FIXED', 'Global truth-last barrier not closed')
    run(rt, dict(prediction=str(root / 'tasks' / key / 'predictions'), output=str(root / 'scores' / key)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('dispatch', 'score'))
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--key')
    args = parser.parse_args()
    if args.action == 'dispatch':
        dispatch(args.runtime)
    else:
        require(args.key is not None, 'Missing score key')
        score(args.runtime, args.key)


if __name__ == '__main__':
    main()
