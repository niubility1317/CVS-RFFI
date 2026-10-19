"""One row owner; fixed E10, clean prediction, separate truth-last score."""
import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity


def execute(mode, row):
    env = dict(os.environ)
    if mode == 'score':
        env['CUDA_VISIBLE_DEVICES'] = ''
    os.execve(sys.executable, [sys.executable, '-u', str(d.ROOT / 'experiments/cvs_ce_half_replicate/runner.py'),
                             '--mode', mode, '--row', row], env)


def worker(row):
    c = d.config(next(r for r in d.rows() if r['row_id'] == row))
    with lock(d.BASE / (row + '.owner.lock'), nonblocking=True):
        gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
        if gpu not in d.GPU_IDS:
            raise ValueError('Unregistered physical GPU')
        # Another owner can enter after dispatch but before this worker starts.
        # Resource contention waits; it must not consume a scientific run.
        while True:
            with lock(Path(d.PROJECT) / 'runs/receiver_residual_capacity4.lock'):
                state = capacity()[gpu]
                pids = state['pids'] | {os.getpid()}
                ready = len(pids) <= 1 and state['free_mb'] >= 12000
                d.write(d.BASE / (row + '.capacity.json'), dict(
                    status='CAPACITY_READY' if ready else 'WAITING_CAPACITY',
                    gpu=gpu, pid=os.getpid(), total_pids=sorted(pids),
                    max_total_GPU_workers=1, free_mb=state['free_mb']))
            if ready:
                break
            time.sleep(5)
        from .observer import train
        train(c)
    execute('predict', row)


def dispatch():
    if d.BASE.exists():
        raise FileExistsError('Reconcile existing run; never overwrite or duplicate')
    d.BASE.mkdir()
    (d.BASE / 'logs').mkdir()
    d.write(d.BASE / 'dispatcher.json', dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, owner=d.OWNER))
    from .preflight import smoke
    try:
        smoke()
    except BaseException as error:
        d.write(d.BASE / 'failure.json', dict(status='PREFLIGHT_FAILED', error=repr(error),
                                            preserve_outputs=True, no_automatic_retry=True))
        raise
    pending, active, launched, completed, failures = d.rows(), {}, [], [], []
    while pending or active:
        for row, child in list(active.items()):
            code = child.poll()
            if code is None:
                continue
            done = d.BASE / row / 'complete.json'
            if code or not done.is_file() or d.read(done).get('status') != 'TRAINED_AND_SCORED':
                failures.append(dict(row_id=row, exit_code=code))
            else:
                completed.append(row)
            del active[row]
        if pending and not failures:
            with lock(Path(d.PROJECT) / 'runs/receiver_residual_capacity4.lock'):
                states = capacity()
                for gpu in d.GPU_IDS:
                    if not pending:
                        break
                    reserved = {x['pid'] for x in launched if x['row_id'] in active and x['gpu'] == gpu}
                    if states[gpu]['pids'] | reserved or states[gpu]['free_mb'] < 12000:
                        continue
                    row = pending.pop(0)['row_id']
                    cmd = [sys.executable, '-u', str(d.ROOT / 'experiments/cvs_ce_half_replicate/runner.py'),
                           '--mode', 'worker', '--row', row]
                    log = d.BASE / 'logs' / (row + '.log')
                    with log.open('x', encoding='utf8') as f:
                        child = subprocess.Popen(cmd, cwd=d.ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                                 stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                                                 start_new_session=True)
                    active[row] = child
                    launched.append(dict(row_id=row, pid=child.pid, gpu=gpu, cwd=str(d.ROOT), argv=cmd, log=str(log)))
                    d.write(d.BASE / 'launch.json', dict(rows=launched))
        d.write(d.BASE / 'queue_state.json', dict(active=list(active), pending=pending, completed=completed,
                                                failures=failures, gpu_ids=list(d.GPU_IDS), max_total_GPU_workers=1,
                                                updated_at=time.time()))
        if failures and not active:
            break
        if pending or active:
            time.sleep(5)
    if failures:
        d.write(d.BASE / 'failure.json', dict(status='TECHNICAL_PARTIAL', failures=failures, pending=pending,
                                            healthy_rows_preserved=True, no_automatic_retry=True))
        raise RuntimeError('Technical failure preserved; no automatic retry')
    from .report import collect
    result = collect()
    d.write(d.BASE / 'complete.json', dict(status='ANALYZED', rows=completed, reused_seed01=True,
                                          model_seed_count=3, views=list(d.VIEWS), clean_mean=result['CE_HALF']['mean']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['worker', 'predict', 'score'])
    p.add_argument('--row')
    a = p.parse_args()
    if a.mode is not None and a.row not in {r['row_id'] for r in d.rows()}:
        raise ValueError('Only registered CE_HALF seeds02/03 permitted')
    if a.mode == 'worker':
        worker(a.row)
    elif a.mode == 'predict':
        from .evaluate import predict
        predict(a.row)
        execute('score', a.row)
    elif a.mode == 'score':
        with lock(Path(d.PROJECT) / 'runs/ce_half_three_seeds_CPU_scorer.lock'):
            from .evaluate import score
            score(a.row)
        d.write(d.BASE / a.row / 'complete.json', dict(status='TRAINED_AND_SCORED', row_id=a.row))
    else:
        dispatch()
