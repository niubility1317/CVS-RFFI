"""One immutable run; serial GPU4 workers, then independent CPU truth-last."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity


def execute(mode, row):
    env = dict(os.environ)
    if mode == 'score':
        env['CUDA_VISIBLE_DEVICES'] = ''
    os.execve(sys.executable, [sys.executable, '-u', str(d.ROOT / 'experiments/cvs_t1_e50/runner.py'),
                             '--mode', mode, '--row', row], env)


def worker(row):
    with lock(d.BASE / (row + '.owner.lock'), nonblocking=True):
        gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
        if gpu not in d.GPU_IDS:
            raise ValueError('Only registered GPU4')
        with lock(Path(d.PROJECT) / 'runs/receiver_residual_capacity4.lock'):
            if len(capacity()[gpu]['pids'] | {os.getpid()}) > 1:
                raise ValueError('GPU must have no other compute owner at worker start')
        from .observer import train
        train(d.config(next(r for r in d.rows() if r['row_id'] == row)))
    execute('predict', row)


def collect():
    results = []
    for row in d.rows():
        root = d.row_root(row['row_id'])
        complete = d.read(root / 'complete.json')
        if complete['status'] != 'TRAINED_AND_SCORED':
            raise ValueError('Incomplete row cannot close the run')
        results.append(dict(row=row, training=d.read(root / 'source/result.json'),
                            scores=d.read(root / 'scoring/scores.json')))
    d.write(d.BASE / 'results.json', dict(rows=results, target_feedback_forbidden=True,
            comparison_limit='Different L/U and LR calendars; model seed count1; clean only'))
    d.write(d.BASE / 'complete.json', dict(status='TRAINED_AND_SCORED', rows=[r['row_id'] for r in d.rows()],
            model_seed_count=1, views=list(d.VIEWS), newly_trained_epochs=70, cumulative_epochs_per_model=50))


def dispatch():
    if d.BASE.exists():
        raise FileExistsError('Reconcile existing run; never overwrite or duplicate')
    d.BASE.mkdir()
    (d.BASE / 'logs').mkdir()
    d.write(d.BASE / 'dispatcher.json', dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, owner=d.OWNER))
    try:
        from .preflight import smoke
        smoke()
        pending = d.rows()
        child = None
        launched = []
        completed = []
        active_row = None
        while pending or child is not None:
            if child is not None and child.poll() is not None:
                code = child.returncode
                done = d.row_root(active_row) / 'complete.json'
                if code or not done.is_file() or d.read(done)['status'] != 'TRAINED_AND_SCORED':
                    raise RuntimeError(f'Worker {active_row} failed with {code}; preserve artifacts; no auto-retry')
                completed.append(active_row)
                child = None
                active_row = None
            if pending and child is None:
                with lock(Path(d.PROJECT) / 'runs/receiver_residual_capacity4.lock'):
                    states = capacity()
                    slots = [g for g in d.GPU_IDS if len(states[g]['pids']) == 0 and states[g]['free_mb'] >= 12000]
                    if slots:
                        gpu = slots[0]
                        row = pending.pop(0)
                        active_row = row['row_id']
                        cmd = [sys.executable, '-u', str(d.ROOT / 'experiments/cvs_t1_e50/runner.py'),
                               '--mode', 'worker', '--row', active_row]
                        log = d.BASE / 'logs' / (active_row + '.log')
                        with log.open('x', encoding='utf-8') as handle:
                            child = subprocess.Popen(cmd, cwd=d.ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                        launched.append(dict(row_id=active_row, pid=child.pid, gpu=gpu, cwd=str(d.ROOT), argv=cmd, log=str(log)))
                        d.write(d.BASE / 'launch.json', dict(rows=launched))
            d.write(d.BASE / 'queue_state.json', dict(active=active_row, pending=pending, completed=completed,
                    updated_at=time.time(), GPU_IDS=list(d.GPU_IDS), maximum_own_workers=1))
            if pending or child is not None:
                time.sleep(5)
        collect()
    except BaseException as error:
        d.write(d.BASE / 'failure.json', dict(status='SYSTEMIC_TECHNICAL_FAILURE', error=repr(error),
                traceback=traceback.format_exc(), preserve_outputs=True, no_automatic_retry=True))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['worker', 'predict', 'score'])
    parser.add_argument('--row')
    args = parser.parse_args()
    if args.mode and args.row not in {r['row_id'] for r in d.rows()}:
        raise ValueError('Unregistered row')
    if args.mode == 'worker':
        worker(args.row)
    elif args.mode == 'predict':
        from .evaluate import predict
        predict(args.row)
        execute('score', args.row)
    elif args.mode == 'score':
        from .evaluate import score
        score(args.row)
        d.write(d.row_root(args.row) / 'complete.json', dict(status='TRAINED_AND_SCORED', row_id=args.row))
    else:
        dispatch()
