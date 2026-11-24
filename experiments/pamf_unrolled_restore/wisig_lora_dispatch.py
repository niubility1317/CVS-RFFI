"""One-owner source training and truth-last WiSig comparison dispatcher."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ARMS = ('expanded', 'guard')
ROUTES = ('full', 'residual')
SPLITS = ('cross_rx', 'cross_both')


def available_gpus(devices, occupied, assigned, share, min_free):
    return [index for index, uuid, free in devices
            if index not in assigned and free >= min_free and (share or uuid not in occupied)]


def inventory():
    command = ['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits']
    rows = subprocess.check_output(command, text=True).splitlines()
    devices = [(int(index), uuid.strip(), int(free)) for index, uuid, free
               in (row.split(',') for row in rows)]
    command = ['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader,nounits']
    occupied = set(subprocess.check_output(command, text=True).splitlines())
    return devices, occupied


def state(root, status, **details):
    path = root / 'state.json'
    tmp = root / 'state.next.json'
    tmp.write_text(json.dumps(dict(status=status, **details)), encoding='utf-8')
    tmp.replace(path)


def schedule(rt, jobs):
    root = Path(rt['run_root'])
    jobs, running = list(jobs), {}
    failure = None
    while jobs or running:
        for pid, (process, gpu, name) in list(running.items()):
            status = process.poll()
            if status is not None:
                del running[pid]
                if status != 0:
                    failure = f'Worker {name} exited {status}; completed and partial outputs are preserved'
                    jobs.clear()
        if failure and not running:
            raise RuntimeError(failure)
        if not failure and jobs and len(running) < 2:
            devices, occupied = inventory()
            free = available_gpus(devices, occupied, {item[1] for item in running.values()},
                                  rt['share_existing'], rt['min_free_mib'])
            if free:
                gpu = free[0]
                action, arm, mode, route, split = jobs.pop(0)
                name = '-'.join(part for part in (action, arm, mode, route, split) if part)
                command = [rt['python'], '-u', str(Path(rt['release_root']) / 'tools/wisig_lora_replay.py'),
                           action, str(Path(rt['release_root']) / 'runtime.json'), arm, mode]
                if route:
                    command += ['--route', route, '--split', split]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                with (root / 'logs' / (name + '.log')).open('x', encoding='utf-8') as log:
                    process = subprocess.Popen(command, cwd=rt['release_root'], env=env,
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                running[process.pid] = (process, gpu, name)
                print('WISIG_LORA_STARTED', name, process.pid, gpu, flush=True)
                continue
        time.sleep(10)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    (root / 'logs').mkdir()
    try:
        state(root, 'SOURCE_SMOKE')
        schedule(rt, [('smoke', arm, 'frozen', None, None) for arm in ARMS])
        schedule(rt, [('fit-smoke', arm, 'retrained', None, None) for arm in ARMS])
        state(root, 'SOURCE_TRAINING')
        schedule(rt, [('fit', arm, 'retrained', None, None) for arm in ARMS])
        state(root, 'FROZEN_SOURCE_SMOKE')
        schedule(rt, [('smoke', arm, 'retrained', None, None) for arm in ARMS])
        state(root, 'TARGET_PREDICTING')
        schedule(rt, [('predict', arm, mode, route, split) for arm in ARMS
                      for mode in ('frozen', 'retrained') for route in ROUTES for split in SPLITS])
        state(root, 'PREDICTIONS_FROZEN', truth_read=False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=rt['release_root'])
        command = [rt['python'], '-c', 'import json,sys; from experiments.pamf_unrolled_restore.wisig_lora_score import score; score(json.load(open(sys.argv[1])))',
                   str(Path(rt['release_root']) / 'runtime.json')]
        with (root / 'logs' / 'score.log').open('x', encoding='utf-8') as log:
            subprocess.run(command, cwd=rt['release_root'], env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        state(root, 'ANALYZED')
    except Exception as exc:
        state(root, 'FAILED', error=str(exc))
        raise


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != 'dispatch':
        raise ValueError('Expected dispatch and an immutable runtime path')
    with Path(sys.argv[2]).open(encoding='utf-8') as stream:
        dispatch(json.load(stream))
