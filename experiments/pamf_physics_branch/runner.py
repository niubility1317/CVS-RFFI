"""PTBR workers reuse the fixed pilot dispatcher and independent scorer."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from experiments.cric_dura_comparison.io import read
from experiments.pamf_branch_training import runner as base


def worker(c, kind):
    if kind in ('score', 'weak_score'):
        if kind == 'score':
            from experiments.pamf_paper_completion.evaluation import score
        else:
            from experiments.pamf_branch_training.weak_evaluation import score
        return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind == 'smoke':
        from .smoke import smoke
        return smoke(c)
    if kind == 'train':
        from .source import train
        return train(c)
    if kind == 'source_diagnostic':
        from .source_diagnostic import run
        return run(c)
    if kind == 'predict':
        from experiments.pamf_paper_completion.evaluation import predict
        return predict(c)
    if kind == 'weak_predict':
        from experiments.pamf_branch_training.weak_evaluation import predict
        return predict(c)
    raise ValueError('Unknown PTBR stage')


def launch(rt, job, gpu):
    path = Path(rt['run_root']) / 'logs' / (job['id'] + '.log')
    cmd = [rt['python'], '-u', '-m', 'experiments.pamf_physics_branch.runner',
           'worker', '--kind', job['kind'], '--config', job['config']]
    env = dict(os.environ, PYTHONPATH=rt['release_root'], CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with path.open('x') as log:
        process = subprocess.Popen(cmd, cwd=rt['release_root'], env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return process, dict(status='RUNNING', pid=process.pid, gpu=gpu, cwd=rt['release_root'],
                         argv=cmd, log=str(path), started_at=time.time())


def admits(gpu, active, required_mib):
    from experiments.pamf_repl_completion.runner import capacity, processes
    cuda, free = capacity(gpu)
    pending = {pid for pid, card in processes().items() if card == gpu}
    if len(cuda | pending) >= 2:
        return False
    own = [p for p, item in active.values() if item['gpu'] == gpu]
    initializing = sum(p.pid not in cuda for p in own)
    return len(own) < 2 and free >= required_mib * (1 + initializing)


def dispatch(rt):
    from experiments.pamf_branch_training import prepare as old_prepare
    from .prepare import prepare
    previous = old_prepare.prepare, base.admits, base.launch
    old_prepare.prepare, base.admits, base.launch = prepare, admits, launch
    try:
        base.dispatch(rt)
    finally:
        old_prepare.prepare, base.admits, base.launch = previous


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'worker'])
    parser.add_argument('--kind')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    dispatch(read(args.config)) if args.action == 'dispatch' else worker(read(args.config), args.kind)
