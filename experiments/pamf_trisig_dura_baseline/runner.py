"""One owner, two authorized GPUs, frozen baseline, then truth-last scoring."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.cric_dura_comparison.io import read, write
from experiments.practical_full_2x2.cache import atomic
from .evaluation import RUN, SEEDS, predict_config, predict, smoke, score

MODULE = 'experiments.pamf_trisig_dura_baseline.runner'


def available():
    from experiments.pamf_repl_completion.runner import capacity, processes
    pending = processes()
    for gpu in (0, 1):
        pids, free = capacity(gpu)
        reserved = {pid for pid, assigned in pending.items() if assigned == gpu}
        if len(pids | reserved) < 2 and free >= 6000:
            return gpu
    return None


def launch(root, project, kind, gpu, seed=None):
    command = [sys.executable, '-u', '-m', MODULE, kind, '--project', str(project)]
    if seed is not None:
        command += ['--seed', str(seed)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    name = kind if seed is None else f'{kind}-{seed}'
    with (root / 'logs' / (name + '.log')).open('x') as log:
        proc = subprocess.Popen(command, cwd=Path.cwd(), env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return proc, dict(pid=proc.pid, gpu=gpu, cwd=str(Path.cwd()), argv=command, status='RUNNING')


def dispatch(project, commit):
    project = Path(project)
    root = project / 'runs' / RUN
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    state = dict(status='WAITING_FOR_SMOKE', pid=os.getpid(), cwd=str(Path.cwd()),
                 code_commit=commit, jobs={})
    atomic(root / 'pipeline_state.json', state)
    while (gpu := available()) is None:
        time.sleep(30)
    process, info = launch(root, project, 'smoke', gpu)
    state['smoke'] = info; atomic(root / 'pipeline_state.json', state)
    code = process.wait()
    state['smoke'].update(status='PASS' if code == 0 else 'FAILED', exit_code=code)
    if code:
        state['status'] = 'FAILED_NO_QUERY_SMOKE'; atomic(root / 'pipeline_state.json', state)
        return
    state['status'] = 'RUNNING'; atomic(root / 'pipeline_state.json', state)
    pending = list(SEEDS); active = {}; failed = []
    while pending or active:
        for seed, (process, info) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            completed = root / 'predictions' / str(seed) / 'complete.json'
            good = code == 0 and completed.is_file()
            state['jobs'][str(seed)].update(status='COMPLETE' if good else 'FAILED', exit_code=code)
            if not good:
                failed.append(seed)
            del active[seed]
        if not failed and pending:
            gpu = available()
            if gpu is not None:
                seed = pending.pop(0)
                process, info = launch(root, project, 'predict', gpu, seed)
                active[seed] = process, info; state['jobs'][str(seed)] = info
        state['failed'] = failed; atomic(root / 'pipeline_state.json', state)
        if failed and not active:
            state['status'] = 'PARTIAL_TECHNICAL_FAILURE'; atomic(root / 'pipeline_state.json', state)
            return
        if pending or active:
            time.sleep(30)
    process, info = launch(root, project, 'score', None)
    state['score'] = info; atomic(root / 'pipeline_state.json', state)
    code = process.wait()
    state['score'].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
    state['status'] = 'SCORED_COMPLETE' if code == 0 else 'PARTIAL_TECHNICAL_FAILURE'
    atomic(root / 'pipeline_state.json', state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'smoke', 'predict', 'score'])
    parser.add_argument('--project', required=True)
    parser.add_argument('--commit')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.action == 'dispatch':
        if not args.commit or (Path.cwd() / 'release_commit.txt').read_text().strip() != args.commit:
            raise ValueError('Committed release identity required')
        dispatch(args.project, args.commit)
    elif args.action == 'smoke':
        smoke(args.project)
    elif args.action == 'predict':
        predict(predict_config(args.project, args.seed))
    else:
        score(args.project)


if __name__ == '__main__':
    main()
