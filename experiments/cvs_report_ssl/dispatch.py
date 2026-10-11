"""One owner, REUSED source audit, 24 E40 rows, immediate clean-only tests."""
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

LIMIT = 3
FREE_MB = 12000


def record(*args, **kwargs):
    from .registration import record as write_record
    return write_record(*args, **kwargs)


def choose_gpu(capacities):
    eligible = [g for g, info in capacities.items()
                if len(info['pids']) < LIMIT and info['free_mb'] >= FREE_MB]
    return min(eligible, key=lambda g: (len(capacities[g]['pids']), -capacities[g]['free_mb'], g)) if eligible else None


def preparation_path(seed):
    return d.AUDIT_BASE / ('s' + str(seed)) / 'preparation_complete.json'


def preparation_ready(seed):
    path = preparation_path(seed)
    if not path.is_file():
        return False
    value = d.read(path)
    expected = dict(status='SOURCE_GEOMETRY_AUDIT_COMPLETE', seed=seed, count=6300,
                    target_access=False, model_state_restored=True, identity_updates=0,
                    parent_contract=d.parent_config(seed))
    if any(value.get(k) != v for k,v in expected.items()) or not isinstance(value.get('source_contract'),dict):
        return False
    folder = path.parent
    if value.get('audit') != str(folder/'audit.json'):
        return False
    if not all((folder/name).is_file() and (folder/name).stat().st_size > 0
               for name in ('source_cache.pt','actions.pt','receiver.pt','source_responses.pt','audit.json')):
        return False
    audit = d.read(folder/'audit.json')
    return all(audit.get(k) == v for k,v in expected.items() if k != 'parent_contract')


def require_preparations():
    missing = [s for s in d.SEEDS if not preparation_ready(s)]
    if missing:
        raise ValueError('Complete source-only geometry audit required for all seeds; never rerun preparation: '+str(missing))
    u=d.read(d.UNLABELED_CACHE.parent/'complete.json')
    if u['status']!='SOURCE_UNLABELED_READY' or u['count']!=56700 or u['TX_truth_exposed'] or u['target_loaders_constructed']:raise ValueError('Truth-blind U preparation incomplete')
    return [dict(seed=s,preparation=str(preparation_path(s).parent),reused_IQ_only=True) for s in d.SEEDS]


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
    sys.argv = ['experiments.cvs_report_ssl.source', *args]
    try:
        runpy.run_module('experiments.cvs_report_ssl.source', run_name='__main__')
    finally:
        sys.argv = old


def predict_row(row_id):
    from . import evaluate
    # Prediction uses no truth. The independent CPU scorer alone opens target labels.
    c = evaluate.snapshot(row_id)[0]
    output = d.BASE / row_id / 'prediction'
    if output.exists():
        # Existing complete predictions are reusable; partial outputs are retained.
        evaluate.prediction_preflight([c])
    else:
        evaluate.predict(row_id)
    record('PREDICTIONS_COMPLETE', row_id=row_id, views=['clean'])


def score_row(row_id):
    from . import evaluate
    c=evaluate.snapshot(row_id)[0]
    evaluate.prediction_preflight([c])
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Scoring must not reserve a GPU')
    evaluate.score(row_id)
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
    if kind not in ('train','score'):
        raise ValueError('Only training workers exist; geometry preparation is immutable and reused')
    row = next((r for r in d.rows() if r['row_id'] == identifier), None)
    if row is None:
        raise ValueError('Unknown geometry row')
    destination = d.BASE / identifier
    destination.mkdir(parents=True, exist_ok=True)
    with lock(destination/'owner.lock', nonblocking=True):
      try:
        if kind=='score':
            score_row(identifier)
            return
        await_capacity(kind, identifier)
        config = d.config(row)
        d.validate(config)
        path = d.BASE / 'configs' / (identifier + '.json')
        if d.read(path) != config:
            raise ValueError('Frozen config or prepared parent differs')
        require_preparations()
        if Path(config['output_root']).exists() or (destination/'prediction').exists():
            raise FileExistsError('Existing row output must be reconciled, never retrained or overwritten')
        invoke_source(['--config', str(path)])
        from .evaluate import source_provenance, snapshot
        source_provenance(config)
        snapshot(identifier)  # source writes the fixed E40 freeze before evaluation.
        record('SOURCE_FROZEN', row_id=identifier, target_access=False)
        predict_row(identifier)
        d.write(destination/'gpu_complete.json',dict(status='PREDICTIONS_COMPLETE',views=['clean'],truth_read=False,gpu_worker_releases_before_control_wait=True))
      except Exception as exc:
        if not (destination/'failure.json').exists():
            d.write(destination / 'failure.json', dict(status='FAILED', error=repr(exc), automatic_retry=False))
        record('FAILED', row_id=identifier, job_kind=kind, error=repr(exc))
        raise


def launch(kind, identifier, gpu):
    if kind not in ('train','score'):
        raise ValueError('Preparation relaunch is forbidden')
    command = [sys.executable, '-u', '-m', 'experiments.cvs_report_ssl.dispatch',
               '--worker', kind, '--identifier', identifier, '--train-stage2']
    log = d.BASE / 'logs' / (kind + '-' + identifier + '.log')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu) if kind=='train' else '', PYTHONPATH=str(d.ROOT) + os.pathsep + str(d.ROOT / 'code'),
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
        preparations = require_preparations()
        d.BASE.mkdir(parents=True, exist_ok=False)
        (d.BASE / 'logs').mkdir()
        initialize()
        for row in d.rows():
            d.write(d.BASE / 'configs' / (row['row_id'] + '.json'), d.config(row))
        current = process(os.getpid())
        d.write(d.BASE / 'dispatcher.json', dict(pid=os.getpid(), start_ticks=current['start_ticks'],
            cwd=str(d.ROOT), argv=sys.argv, launch_owner=d.OWNER,
            commit=(d.ROOT / 'release_commit.txt').read_text().strip()))
        d.write(d.BASE/'preparation_reuse.json',dict(audit_run=d.AUDIT_RUN,seeds=preparations,
                                                   preparation_launched=False,target_access=False))
        pending = [('train', r['row_id']) for r in d.rows()]
        active, receipts, completed, failures = {}, [], [], []
        pending_scores,active_scores,predicted=[],{},[]
        try:
            while pending or active or pending_scores or active_scores:
                for key, child in list(active.items()):
                    exit_code = child.poll()
                    if exit_code is None:
                        continue
                    kind, identifier = key
                    artifact = d.BASE / identifier / 'gpu_complete.json'
                    valid = artifact.is_file() and d.read(artifact)['status'] == 'PREDICTIONS_COMPLETE'
                    if exit_code or not valid:
                        failures.append(dict(kind=kind, identifier=identifier, exit_code=exit_code, artifact_valid=valid))
                    else:
                        predicted.append(dict(kind=kind, identifier=identifier))
                        pending_scores.append(identifier)
                    del active[key]
                for identifier,child in list(active_scores.items()):
                    exit_code=child.poll()
                    if exit_code is None:continue
                    artifact=d.BASE/identifier/'completion.json'
                    valid=artifact.is_file() and d.read(artifact)['status']=='ANALYZED'
                    if exit_code or not valid:failures.append(dict(kind='score',identifier=identifier,exit_code=exit_code,artifact_valid=valid))
                    else:completed.append(dict(kind='score',identifier=identifier))
                    del active_scores[identifier]
                while pending_scores and len(active_scores)<4:
                    identifier=pending_scores.pop(0)
                    child,receipt=launch('score',identifier,None)
                    receipts.append(receipt);d.write(d.BASE/'launch.json',dict(rows=receipts));active_scores[identifier]=child
                    record('SCORING',row_id=identifier,pid=child.pid,gpu=None,log=receipt['log'])
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
                    record('RUNNING', row_id=identifier,
                           job_kind=kind, identifier=identifier, pid=child.pid, gpu=gpu, log=receipt['log'])
                d.write(d.BASE / 'queue_state.json', dict(active=[list(k) for k in active], pending=pending,
                    completed=completed, completed_predictions=predicted,active_scores=list(active_scores),pending_scores=pending_scores,
                    failures=failures, total_identity_rows=len(d.rows()), preparations_reused=3,
                    preparations_launched=0,
                    per_gpu_limit=LIMIT, views=['clean'], updated_at=time.time()))
                from .source_selection import select_if_ready
                select_if_ready()
                if failures and not active and not active_scores and not pending_scores:
                    raise RuntimeError('Technical failures retained; no automatic retry: ' + repr(failures))
                if pending or active or pending_scores or active_scores:
                    time.sleep(10)
            subprocess.run([sys.executable, '-m', 'experiments.cvs_report_ssl.evaluate', '--mode', 'score'],
                cwd=d.ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
            require_scored(d.read(d.BASE / 'scoring_complete.json'), len(d.rows()))
            d.write(d.BASE / 'completion.json', dict(status='ANALYZED', trained_models=len(d.rows()), model_seeds=list(d.SEEDS),
                tested_models=len(d.rows()), views=['clean'], independent_recount='VERIFIED', all_rows_tested_automatically=True,
                preparation_audit_run=d.AUDIT_RUN, preparations_reused=3, preparations_launched=0))
            finalize_report()
            record('ANALYZED', trained_models=len(d.rows()), tested_models=len(d.rows()), views=['clean'])
        except Exception as exc:
            d.write(d.BASE / 'failure.json', dict(status='FAILED', error=repr(exc), automatic_retry=False))
            record('FAILED', error=repr(exc))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=['train','score'])
    parser.add_argument('--identifier')
    parser.add_argument('--train-stage2', action='store_true')
    args = parser.parse_args()
    if args.worker:
        if not args.identifier:
            parser.error('--worker requires --identifier')
        worker(args.worker, args.identifier)
    else:
        dispatch()
