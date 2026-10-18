"""Two As CE aggregation rows; source failure never suppresses fixed formal tests."""
import argparse
import json
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
    os.execve(sys.executable, [sys.executable, '-u', str(d.ROOT / 'experiments/cvs_as_marginal73/runner.py'), '--mode', mode, '--row', row], env)

def row_config(row):
    return d.config(next(r for r in d.rows() if r['row_id'] == row))

def decide_source(terminal=False):
    if (d.BASE/'source_matrix_decision.json').is_file():
        return
    if not terminal and not all((d.BASE/r['row_id']/'source_qualification/complete.json').is_file() for r in d.rows()):
        return
    from .report import source_decision,source_partial
    try:
        source_decision()
    except Exception as error:
        source_partial(error=repr(error))

def worker(row):
    c = row_config(row)
    with lock(d.BASE / (row + '.owner.lock'), nonblocking=True):
        gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
        with lock(Path(d.PROJECT) / 'runs/receiver_residual_capacity4.lock'):
            if len(capacity()[gpu]['pids'] | {os.getpid()}) > 2:
                raise ValueError('Total GPU process cap2 includes other owners')
        from .observer import train
        train(c)
        from experiments.cvs_equivariant_identity.precision import numerical_context
        from .qualification import run as qualify
        try:
            with numerical_context(d.FULL_FP32_POLICY):
                qualify(row)
        except Exception as error:
            d.write(d.BASE / row / 'source_qualification_failure.json', dict(
                status='SOURCE_READOUT_FAILED', message=str(error), traceback=traceback.format_exc(),
                formal_evaluation_still_required=True, no_automatic_retry=True))
        execute('predict', row)

def dispatch():
    if d.BASE.exists():
        raise FileExistsError('Reconcile existing run; never duplicate or overwrite')
    d.BASE.mkdir()
    (d.BASE / 'logs').mkdir()
    d.write(d.BASE / 'dispatcher.json', dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, owner=d.OWNER))
    from .preflight import smoke
    try:
        smoke()
    except BaseException as error:
        d.write(d.BASE / 'failure.json', dict(status='PREFLIGHT_FAILED', error=repr(error), preserve_outputs=True, no_automatic_retry=True))
        raise
    pending = [r['row_id'] for r in d.rows()]
    active, launched, completed, failures = {}, [], [], []
    while pending or active:
        for row, child in list(active.items()):
            if child.poll() is None:
                continue
            done = d.BASE / row / 'complete.json'
            if child.returncode or not done.is_file() or d.read(done).get('status') != 'TRAINED_AND_SCORED':
                failures.append(dict(row_id=row, exit_code=child.returncode))
            else:
                completed.append(row)
            del active[row]
        decide_source()
        # Technical failures preserve every output and let already healthy
        # children finish. Never restart a failed row or kill another owner.
        if pending:
            with lock(Path(d.PROJECT) / 'runs/receiver_residual_capacity4.lock'):
                states = capacity()
                # Own workers reserve a slot before their first CUDA call is
                # visible in nvidia-smi; count those PIDs without double count.
                reservations = {gpu: {item['pid'] for item in launched
                                      if item['gpu'] == gpu and item['row_id'] in active}
                                for gpu in states}
                slots = [(len(s['pids'] | reservations[gpu]), -s['free_mb'], gpu)
                         for gpu, s in states.items()
                         if len(s['pids'] | reservations[gpu]) < 2 and s['free_mb'] >= 12000]
                if slots:
                    gpu = min(slots)[2]
                    row = pending.pop(0)
                    cmd = [sys.executable, '-u', str(d.ROOT / 'experiments/cvs_as_marginal73/runner.py'), '--mode', 'worker', '--row', row]
                    log = d.BASE / 'logs' / (row + '.log')
                    with log.open('x', encoding='utf8') as f:
                        child = subprocess.Popen(cmd, cwd=d.ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                    active[row] = child
                    launched.append(dict(row_id=row, pid=child.pid, gpu=gpu, cwd=str(d.ROOT), argv=cmd, log=str(log)))
                    d.write(d.BASE / 'launch.json', dict(rows=launched))
        d.write(d.BASE / 'queue_state.json', dict(active=list(active), pending=pending, completed=completed, failures=failures, updated_at=time.time(), max_total_processes_per_GPU=2))
        if pending or active:
            time.sleep(3)
    # Cross-row source decision occurs only after all terminal workers. A
    # merely unfinished peer never becomes a source technical failure.
    from .report import collect
    decide_source(terminal=True)
    result = collect()
    d.write(d.BASE / 'complete.json', dict(status=result['status'], rows=completed, failures=failures, new_model_rows=2, reused_controls=2, model_seed_count=1, seven_views_per_completed_row=True, crossseed_confirmed=False, goal_complete=False))
    if failures:
        d.write(d.BASE / 'failure.json', dict(status='TECHNICAL_PARTIAL', failures=failures, preserve_outputs=True, healthy_other_runs_untouched=True, all_other_fixed_rows_finished=True, no_automatic_retry=True))
        raise RuntimeError('Technical failure preserved; remaining fixed rows finished')
    command=[sys.executable,'-u',str(d.ROOT/'analysis/cvs_as_marginal73_analysis_20261019.py'),'--project',str(d.PROJECT),'--out',str(d.BASE/'analysis_r01'),'--local-nn',str(d.LOCAL_NN),'--smoke-nn',str(d.SMOKE_NN),'--code-commit',(d.ROOT/'release_commit.txt').read_text().strip()]
    d.write(d.BASE/'analysis_launch.json',dict(argv=command,NN=0,IQ=0,CP=0,optimizer_updates=0,after_all_seven_views=True,logical_rows=4))
    os.execve(sys.executable,command,dict(os.environ,CUDA_VISIBLE_DEVICES=''))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['worker', 'predict', 'score'])
    parser.add_argument('--row')
    a = parser.parse_args()
    if a.mode is not None and a.row not in [r['row_id'] for r in d.rows()]:
        raise ValueError('Only the single registered two As CE aggregation rows allowed')
    if a.mode == 'worker':
        worker(a.row)
    elif a.mode == 'predict':
        from .evaluate import predict
        predict(a.row)
        execute('score', a.row)
    elif a.mode == 'score':
        from .evaluate import score
        score(a.row)
        d.write(d.BASE / a.row / 'complete.json', dict(status='TRAINED_AND_SCORED', row_id=a.row))
    else:
        dispatch()
