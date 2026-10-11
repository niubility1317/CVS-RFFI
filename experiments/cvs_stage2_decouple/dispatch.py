"""One owner, three shared preparations, twelve rows, immediate clean tests."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

from . import design as d
from experiments.cvs_multi_state_action.dispatch import capacity
from experiments.cvs_receiver_residual_capacity4.control import lock, process
from .registration import record

LIMIT = 4
FREE_MB = 12000


def choose_gpu(capacities):
    eligible = [g for g, info in capacities.items()
                if len(info['pids']) < LIMIT and info['free_mb'] >= FREE_MB]
    return min(eligible, key=lambda g: (len(capacities[g]['pids']), -capacities[g]['free_mb'], g)) if eligible else None


def preparation_path(seed):
    return d.BASE / 'prepared' / ('s' + str(seed)) / 'preparation_complete.json'


def preparation_ready(seed):
    path = preparation_path(seed)
    if not path.is_file():
        return False
    value = d.read(path)
    # Preparation has no target access; source owns its detailed completeness contract.
    return value.get('status') in ('PREPARATION_COMPLETE', 'PREPARED') and value.get('target_access') is False


def await_capacity(kind, identifier):
    gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
    stable = 0
    while stable < 3:
        with lock(Path(d.PROJECT) / 'runs/receiver_residual_capacity4.lock'):
            observed = capacity()[gpu]
            owners = observed['pids'] | {os.getpid()}
            ready = len(owners) <= LIMIT and observed['free_mb'] >= FREE_MB
            stable = stable + 1 if ready else 0
            d.write(d.BASE / 'capacity_leases' / (kind + '-' + identifier + '.json'),
                dict(pid=os.getpid(), gpu=gpu, observed_pids=sorted(owners), stable_checks=stable,
                     per_gpu_limit=LIMIT, status='CAPACITY_READY' if stable == 3 else 'WAITING_CAPACITY'))
        if stable < 3:
            time.sleep(1 if ready else 5)


def invoke_source(args):
    # Execute in the reservation owner process, avoiding a duplicate GPU owner child.
    old = sys.argv
    sys.argv = ['experiments.cvs_stage2_decouple.source', *args]
    try:
        runpy.run_module('experiments.cvs_stage2_decouple.source', run_name='__main__')
    finally:
        sys.argv = old


def test_row(row_id):
    from . import evaluate
    # Prediction uses no truth. The independent CPU scorer alone opens target labels.
    evaluate.predict(row_id)
    record('PREDICTIONS_COMPLETE', row_id=row_id, views=['clean'])
    subprocess.run([sys.executable, '-m', 'experiments.cvs_stage2_decouple.evaluate',
        '--mode', 'score', '--row', row_id], cwd=d.ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
    result = d.read(d.BASE / 'row_scoring' / row_id / 'scoring_complete.json')
    require_scored(result, 1)
    d.write(d.BASE / row_id / 'completion.json', dict(status='ANALYZED', views=['clean'],
        independent_recount='VERIFIED', target_feedback_forbidden=True))
    record('ANALYZED', row_id=row_id, views=['clean'])


def require_scored(result, count):
    if (result['status'] != 'SCORED_COMPLETE' or result['models'] != count or result['views'] != 1
            or result['metric_records'] != count * 14 or result['day_metric_records'] != count * 4
            or result['query_count_per_view'] != 168000 or result['prediction_count'] != count * 168000
            or not result['truth_last'] or result['independent_recount'] != 'VERIFIED'):
        raise ValueError('Clean score coverage or recount differs')


def worker(kind, identifier):
    await_capacity(kind, identifier)
    try:
        if kind == 'prepare':
            seed = int(identifier)
            if seed not in d.SEEDS:
                raise ValueError('Unknown preparation seed')
            invoke_source(['--prepare', '--seed', str(seed)])
            if not preparation_ready(seed):
                raise ValueError('Incomplete shared preparation')
            record('PREPARATION_COMPLETE', model_seed=seed)
            return
        row = next(r for r in d.rows() if r['row_id'] == identifier)
        config = d.config(row)
        d.validate(config)
        path = d.BASE / 'configs' / (identifier + '.json')
        if d.read(path) != config or not preparation_ready(config['model_seed']):
            raise ValueError('Frozen config or prepared parent differs')
        invoke_source(['--config', str(path)])
        from .evaluate import source_provenance, snapshot
        source_provenance(config)
        snapshot(identifier)  # source writes the fixed E40 freeze before evaluation.
        record('SOURCE_FROZEN', row_id=identifier, target_access=False)
        test_row(identifier)
    except Exception as exc:
        destination = (d.BASE / 'prepared' / ('s' + identifier) if kind == 'prepare' else d.BASE / identifier)
        d.write(destination / 'failure.json', dict(status='FAILED', error=repr(exc), automatic_retry=False))
        record('FAILED', row_id=identifier if kind == 'train' else None, job_kind=kind, error=repr(exc))
        raise


def launch(kind, identifier, gpu):
    command = [sys.executable, '-u', '-m', 'experiments.cvs_stage2_decouple.dispatch',
               '--worker', kind, '--identifier', identifier, '--train-stage2']
    log = d.BASE / 'logs' / (kind + '-' + identifier + '.log')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(d.ROOT) + os.pathsep + str(d.ROOT / 'code'),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with log.open('x', encoding='utf-8') as stream:
        child = subprocess.Popen(command, cwd=d.ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    actual = process(child.pid)
    if actual is None:
        raise RuntimeError('Worker absent immediately after launch')
    receipt = dict(kind=kind, identifier=identifier, pid=child.pid, start_ticks=actual['start_ticks'],
                   gpu=gpu, cwd=str(d.ROOT), argv=command, log=str(log))
    return child, receipt


def dispatch():
    from .registration import initialize, finalize_report
    with lock(d.ROOT / 'owner.lock', nonblocking=True):
        d.BASE.mkdir(parents=True, exist_ok=False)
        (d.BASE / 'logs').mkdir()
        initialize()
        for row in d.rows():
            d.write(d.BASE / 'configs' / (row['row_id'] + '.json'), d.config(row))
        current = process(os.getpid())
        d.write(d.BASE / 'dispatcher.json', dict(pid=os.getpid(), start_ticks=current['start_ticks'],
            cwd=str(d.ROOT), argv=sys.argv, launch_owner=d.OWNER,
            commit=(d.ROOT / 'release_commit.txt').read_text().strip()))
        pending = [('prepare', str(s)) for s in d.SEEDS]
        active, receipts, completed, failures, released_seeds = {}, [], [], [], set()
        try:
            while pending or active:
                for key, child in list(active.items()):
                    exit_code = child.poll()
                    if exit_code is None:
                        continue
                    kind, identifier = key
                    artifact = (preparation_path(int(identifier)) if kind == 'prepare'
                                else d.BASE / identifier / 'completion.json')
                    valid = (preparation_ready(int(identifier)) if kind == 'prepare'
                             else artifact.is_file() and d.read(artifact)['status'] == 'ANALYZED')
                    if exit_code or not valid:
                        failures.append(dict(kind=kind, identifier=identifier, exit_code=exit_code, artifact_valid=valid))
                    else:
                        completed.append(dict(kind=kind, identifier=identifier))
                        if kind == 'prepare':
                            seed = int(identifier)
                            if seed in released_seeds:
                                raise RuntimeError('Preparation released duplicate rows')
                            released_seeds.add(seed)
                            pending.extend(('train', r['row_id']) for r in d.rows() if r['model_seed'] == seed)
                    del active[key]
                while pending and not failures:
                    with lock(Path(d.PROJECT) / 'runs/receiver_residual_capacity4.lock'):
                        gpu = choose_gpu(capacity())
                        if gpu is None:
                            break
                        kind, identifier = pending.pop(0)
                        child, receipt = launch(kind, identifier, gpu)
                        receipts.append(receipt)
                        d.write(d.BASE / 'launch.json', dict(rows=receipts))
                        active[kind, identifier] = child
                    record('RUNNING', row_id=identifier if kind == 'train' else None,
                           job_kind=kind, identifier=identifier, pid=child.pid, gpu=gpu, log=receipt['log'])
                d.write(d.BASE / 'queue_state.json', dict(active=[list(k) for k in active], pending=pending,
                    completed=completed, failures=failures, total_identity_rows=12, total_preparations=3,
                    per_gpu_limit=LIMIT, views=['clean'], updated_at=time.time()))
                if failures and not active:
                    raise RuntimeError('Technical failures retained; no automatic retry: ' + repr(failures))
                if pending or active:
                    time.sleep(10)
            subprocess.run([sys.executable, '-m', 'experiments.cvs_stage2_decouple.evaluate', '--mode', 'score'],
                cwd=d.ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
            require_scored(d.read(d.BASE / 'scoring_complete.json'), 12)
            d.write(d.BASE / 'completion.json', dict(status='ANALYZED', trained_models=12, model_seeds=list(d.SEEDS),
                tested_models=12, views=['clean'], independent_recount='VERIFIED', all_rows_tested_automatically=True))
            finalize_report()
            record('ANALYZED', trained_models=12, tested_models=12, views=['clean'])
        except Exception as exc:
            d.write(d.BASE / 'failure.json', dict(status='FAILED', error=repr(exc), automatic_retry=False))
            record('FAILED', error=repr(exc))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=['prepare', 'train'])
    parser.add_argument('--identifier')
    parser.add_argument('--train-stage2', action='store_true')
    args = parser.parse_args()
    if args.worker:
        if not args.identifier:
            parser.error('--worker requires --identifier')
        worker(args.worker, args.identifier)
    else:
        dispatch()
