"""Run the preregistered three-seed ManySig Main split comparison on N607."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.cric_dura_comparison.io import read, write
from experiments.cric_dura_comparison.source_queue import allow_shared
from experiments.pamf_repl_view_shuffle.plan import TASKS
from .runner import CLASSES, SOURCE_DAYS


SEEDS = (2026092701, 2026092702, 2026092703)


def wait_gpu(root, stage, seed, shared_gpu):
    while True:
        gpu_data = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'],
            text=True)
        apps = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
            text=True)
        by_uuid = {}
        for line in apps.splitlines():
            uuid, pid = (part.strip() for part in line.split(','))
            by_uuid.setdefault(uuid, []).append(int(pid))
        available = []
        for line in gpu_data.splitlines():
            gpu, uuid, mem = (part.strip() for part in line.split(','))
            pids = by_uuid.get(uuid, [])
            if shared_gpu is not None:
                admitted = gpu == shared_gpu and int(mem) >= 6000 and sum(
                    project_training(pid) for pid in pids
                ) < 2
            else:
                admitted = allow_shared(len(pids), int(mem)) and all(owned_process(pid) for pid in pids)
            if admitted:
                available.append((int(mem), gpu))
        if available:
            return max(available)[1]
        state(root, 'WAITING_FOR_GPU', stage=stage, seed=seed)
        time.sleep(30)


def owned_process(pid):
    try:
        return Path('/proc', str(pid)).stat().st_uid == os.getuid()
    except FileNotFoundError:
        return False


def project_training(pid):
    if not owned_process(pid):
        return False
    try:
        argv = Path('/proc', str(pid), 'cmdline').read_bytes().decode(errors='replace').split('\0')
    except FileNotFoundError:
        return False
    return any(part in ('train', 'F', '--kind=train', '--stage=train') for part in argv) or any(
        part.endswith(('_train.py', '.train')) and 'predict' not in argv and 'score' not in argv
        for part in argv
    )


def state(root, status, **fields):
    path = root / 'state.json'
    temporary = root / 'state.next.json'
    write(temporary, dict(status=status, pid=os.getpid(), **fields), exclusive=False)
    temporary.replace(path)


def invoke(root, stage, config, seed, gpu=None, *, bounded_gpu=False):
    log = root / 'logs' / f'{stage}-s{seed}.log'
    command = [sys.executable, '-u', '-m', 'experiments.shen_wisig.runner', stage,
               '--config', str(config), '--device', 'cpu' if gpu is None else 'cuda:0']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu or '', OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    if bounded_gpu:
        environment['SHEN_WISIG_CUDA_MEMORY_FRACTION'] = '0.12'
    with log.open('x', encoding='utf-8') as stream:
        child = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        state(root, 'RUNNING', stage=stage, seed=seed, gpu=gpu, worker=child.pid, log=str(log),
              cwd=os.getcwd(), argv=command)
        if child.wait():
            raise RuntimeError(f'{stage} s{seed} failed. See {log}')


def configurations(root, inputs, commit, seed):
    folder = root / f's{seed}'
    folder.mkdir()
    identity = dict(run_id=root.name, row_id=f'shen-wisig-main-s{seed}', model_seed=seed)
    source = folder / 'source'
    raw = folder / 'predictions'
    tests = {name: inputs['tests'][name] for name in TASKS}
    configs = {
        'train': dict(identity, source_cache=inputs['source_cache'], code_commit=commit,
                      source_contract=inputs['source_contract'], output_root=str(source)),
        'predict': dict(identity, source_output=str(source), source_contract=inputs['source_contract'],
                        classes=CLASSES, readouts=['raw'], output_root=str(raw),
                        tests={name: {'views_root': t['views_root']} for name, t in tests.items()}),
        'score': dict(identity, classes=CLASSES, source_days=SOURCE_DAYS, readouts=['raw'],
                      tests=tests, prediction_output=str(raw), output_root=str(folder / 'scores')),
    }
    paths = {stage: folder / f'{stage}.json' for stage in configs}
    for stage, config in configs.items():
        write(paths[stage], config)
    return paths


def launch(root, handles, commit, shared_gpu):
    inputs = read(handles)['F0']
    root.mkdir(exist_ok=False)
    (root / 'logs').mkdir()
    jobs = {seed: configurations(root, inputs, commit, seed) for seed in SEEDS}
    try:
        invoke(root, 'smoke', jobs[SEEDS[0]]['train'], SEEDS[0])
        for seed, cfg in jobs.items():
            invoke(root, 'train', cfg['train'], seed, wait_gpu(root, 'train', seed, shared_gpu),
                   bounded_gpu=shared_gpu is not None)
            frozen = read(root / f's{seed}' / 'source' / 'complete.json')
            eligible = frozen['val_accuracy'] > 1 / len(CLASSES)
            invoke(root, 'predict', cfg['predict'], seed, wait_gpu(root, 'predict', seed, shared_gpu),
                   bounded_gpu=shared_gpu is not None)
            invoke(root, 'score', cfg['score'], seed)
            if seed == SEEDS[0] and not eligible:
                state(root, 'FIRST_SEED_SCORED_SOURCE_THRESHOLD_UNMET', source_v=frozen['val_accuracy'])
                return
    except Exception as error:
        state(root, 'TECHNICAL_FAILURE', reason=str(error))
        raise
    state(root, 'SCORED_COMPLETE', seeds=list(jobs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--handles', required=True, type=Path)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--shared-gpu', choices=tuple(map(str, range(8))))
    args = parser.parse_args()
    launch(args.root, args.handles, args.commit, args.shared_gpu)
