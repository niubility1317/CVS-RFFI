"""One owner for the reference-budget candidate; previous owner must be terminal."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from experiments.cric_dura_comparison.io import read
from experiments.pamf_branch_training import runner as base
from experiments.pamf_physics_branch import runner as physical
from .prepare import prepare
from .plan import candidate_rows, configuration
from .source import validate_source


def worker(c, kind):
    if kind not in ('smoke', 'train'):
        return physical.worker(c, kind)
    from experiments.pamf_physics_branch import source, smoke
    original = source.validate_source, smoke.candidate_rows, smoke.configuration
    source.validate_source, smoke.candidate_rows, smoke.configuration = validate_source, candidate_rows, configuration
    try:
        if kind == 'smoke':
            from experiments.pamf_repl_completion.runner import activate
            activate(c)
            return smoke.smoke(c)
        from experiments.pamf_repl_completion.runner import activate
        activate(c)
        return source.train(c)
    finally:
        source.validate_source, smoke.candidate_rows, smoke.configuration = original


def launch(rt, job, gpu):
    path = Path(rt['run_root']) / 'logs' / (job['id'] + '.log')
    cmd = [rt['python'], '-u', '-m', 'experiments.pamf_physics_branch_balanced.runner',
           'worker', '--kind', job['kind'], '--config', job['config']]
    env = dict(os.environ, PYTHONPATH=rt['release_root'], CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with path.open('x') as log:
        process = subprocess.Popen(cmd, cwd=rt['release_root'], env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return process, dict(status='RUNNING', pid=process.pid, gpu=gpu, cwd=rt['release_root'],
                         argv=cmd, log=str(path), started_at=time.time())


def dispatch(rt):
    from experiments.pamf_branch_training import prepare as old_prepare
    previous = old_prepare.prepare, base.admits, base.launch
    old_prepare.prepare, base.admits, base.launch = prepare, physical.admits, launch
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
