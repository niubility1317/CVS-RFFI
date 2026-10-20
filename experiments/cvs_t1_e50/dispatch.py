"""Two distinct least-loaded GPUs; preserve the run's frozen scientific rows."""
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
            raise ValueError('GPU outside registered pool')
        with lock(Path(d.PROJECT) / 'runs/receiver_residual_capacity4.lock'):
            if len(capacity()[gpu]['pids'] | {os.getpid()}) > d.MAX_TOTAL_GPU_PROCESSES:
                raise ValueError('Total process capacity includes every owner')
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


def dispatch(adopt=False):
    if adopt:
        previous=d.read(d.BASE/'dispatcher.json')
        if Path('/proc',str(previous['pid'])).exists():
            raise RuntimeError('Original controller still alive; do not duplicate')
        state=d.read(d.BASE/'queue_state.json')
        if state['active'] is not None or state['completed'] or (d.BASE/'launch.json').exists():
            raise RuntimeError('Adopt only the verified untouched pending queue')
        if state['pending']!=d.rows() or any(d.row_root(r['row_id']).exists() for r in d.rows()):
            raise RuntimeError('Existing row output or pending matrix differs')
        if any(d.read(d.BASE/name)['status']!='PASS' for name in ['real_checkpoint_smoke.json','synthetic_smoke.json']):
            raise ValueError('Original launcher smoke did not pass')
    else:
        if d.BASE.exists():
            raise FileExistsError('Reconcile existing run; never overwrite or duplicate')
        d.BASE.mkdir()
        (d.BASE / 'logs').mkdir()
    d.write(d.BASE / 'dispatcher.json', dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, owner=d.OWNER))
    try:
        if not adopt:
            from .preflight import smoke
            smoke()
        pending = d.rows()
        active = {}
        launched = []
        completed = []
        while pending or active:
            for active_row,info in list(active.items()):
                child=info['child']
                if child.poll() is None:continue
                code = child.returncode
                done = d.row_root(active_row) / 'complete.json'
                if code or not done.is_file() or d.read(done)['status'] != 'TRAINED_AND_SCORED':
                    raise RuntimeError(f'Worker {active_row} failed with {code}; preserve artifacts; no auto-retry')
                completed.append(active_row)
                del active[active_row]
            if pending and len(active)<2:
                with lock(Path(d.PROJECT) / 'runs/receiver_residual_capacity4.lock'):
                    states = capacity()
                    used={v['gpu'] for v in active.values()}
                    slots=sorted((len(states[g]['pids']),-states[g]['free_mb'],g) for g in d.GPU_IDS
                          if g not in used and len(states[g]['pids'])<d.MAX_TOTAL_GPU_PROCESSES and states[g]['free_mb']>=12000)
                    while slots and pending and len(active)<2:
                        gpu=slots.pop(0)[2]
                        row = pending.pop(0)
                        active_row = row['row_id']
                        cmd = [sys.executable, '-u', str(d.ROOT / 'experiments/cvs_t1_e50/runner.py'),
                               '--mode', 'worker', '--row', active_row]
                        log = d.BASE / 'logs' / (active_row + '.log')
                        with log.open('x', encoding='utf-8') as handle:
                            child = subprocess.Popen(cmd, cwd=d.ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                        launched.append(dict(row_id=active_row, pid=child.pid, gpu=gpu, cwd=str(d.ROOT), argv=cmd, log=str(log)))
                        active[active_row]=dict(child=child,gpu=gpu)
                        d.write(d.BASE / 'launch.json', dict(rows=launched))
            d.write(d.BASE / 'queue_state.json', dict(active=list(active), pending=pending, completed=completed,
                    updated_at=time.time(), GPU_IDS=list(d.GPU_IDS), maximum_own_workers=2))
            if pending or active:
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
    parser.add_argument('--adopt-empty-queue',action='store_true')
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
        with lock(d.BASE/'CPU_scorer.lock'):
            score(args.row)
        d.write(d.row_root(args.row) / 'complete.json', dict(status='TRAINED_AND_SCORED', row_id=args.row))
    else:
        dispatch(args.adopt_empty_queue)
