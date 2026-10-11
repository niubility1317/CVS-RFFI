"""Independent TEST_ONLY recovery; source weights and old failures are immutable."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import design as d
from . import evaluate
from .dispatch import choose_gpu, require_scored
from .registration import record, finalize_report
from experiments.cvs_multi_state_action.dispatch import capacity
from experiments.cvs_receiver_residual_capacity4.control import lock, process

RECOVERY_RUN = '20261011-phase1-stage2-clean-recovery-manysig-m12-r01'
RECOVERY_BASE = Path(d.PROJECT) / 'runs' / RECOVERY_RUN
LIMIT = 4
FREE_MB = 12000


def commit():
    return (d.ROOT / 'release_commit.txt').read_text(encoding='utf-8').strip()


def recovery_record(status, **values):
    with lock(RECOVERY_BASE / 'registration.lock'):
        item = dict(at=time.time(), status=status, scope='TEST_ONLY', source_run=d.RUN,
                    recovery_run=RECOVERY_RUN, execution_commit=commit(), **values)
        with (RECOVERY_BASE / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
        path = RECOVERY_BASE / 'experiment.json'
        if path.is_file():
            spec = d.read(path)
            spec.update(status=status, latest_event=item, execution_commit=commit())
            if isinstance(spec.get('code'), dict):
                spec['code']['commit'] = commit()
            if isinstance(spec.get('execution'), dict):
                spec['execution']['release_commit'] = commit()
            d.write(path, spec)


def initialize_registration():
    folder = d.ROOT / 'recovery_registration'
    spec = d.read(folder / 'experiment.json')
    if spec['run_id'] != RECOVERY_RUN:
        raise ValueError('Recovery preregistration ID differs')
    for name in ('experiment.json', 'report.md', 'events.jsonl'):
        original, destination = folder / name, RECOVERY_BASE / name
        value = original.read_text(encoding='utf-8')
        if destination.exists():
            if destination.read_text(encoding='utf-8') != value:
                raise FileExistsError('Existing recovery registration differs: ' + name)
        else:
            with destination.open('x', encoding='utf-8') as stream:
                stream.write(value)


def require_exited(receipt):
    """A reused PID belongs to a different process and is never signalled."""
    actual = process(receipt['pid'])
    if actual is None:
        return dict(pid=receipt['pid'], status='EXITED')
    if 'start_ticks' not in receipt:
        raise ValueError('Cannot establish former worker identity without start_ticks')
    if actual['start_ticks'] == receipt['start_ticks']:
        raise RuntimeError('Existing owner still active; recovery must wait: ' + str(receipt['pid']))
    return dict(pid=receipt['pid'], status='EXITED_PID_REUSED', current_start_ticks=actual['start_ticks'])


def preflight():
    dispatcher = d.read(d.BASE / 'dispatcher.json')
    launches = d.read(d.BASE / 'launch.json')['rows']
    expected = {r['row_id'] for r in d.rows()}
    actual = [r['identifier'] for r in launches if r['kind'] == 'train']
    if len(actual) != 12 or set(actual) != expected:
        raise ValueError('Former launch does not contain exactly the twelve fixed training rows')
    exited = [require_exited(dispatcher)] + [require_exited(row) for row in launches]
    configs = evaluate.snapshot()
    for c in configs:
        if d.read(d.BASE / 'configs' / (c['row_id'] + '.json')) != c:
            raise ValueError('Original row config differs')
        source = Path(c['output_root'])
        done = d.read(source / 'completion.json')
        frozen = d.read(d.BASE / c['row_id'] / 'source_frozen.json')
        if (done['status'] != 'SOURCE_TRAINED' or done['config'] != c
                or done['epoch'] != 40 or done['optimizer_steps'] != 8880
                or done['logged_steps'] != 8880 or done['target_access']
                or done['checkpoint_selection'] != 'final_only'
                or frozen['epoch'] != 40 or frozen['optimizer_steps'] != 8880
                or frozen['checkpoint'] != str(source / 'final_ssdg.pth')
                or not (source / 'final_ssdg.pth').is_file()):
            raise ValueError('Complete fixed E40 source required before test recovery')
    return dict(status='VERIFIED', scope='TEST_ONLY', source_run=d.RUN, recovery_run=RECOVERY_RUN,
                old_owners=exited, rows=len(configs), retraining=False,
                source_weights_unchanged=True, target_truth_read=False, configs=configs)


def reconcile_prediction(c):
    """Only a known metadata-only failure may be moved, never a partial result."""
    output = d.BASE / c['row_id'] / 'prediction'
    if not output.exists():
        return 'MISSING'
    predictions, complete = output / 'predictions.npz', output / 'complete.json'
    if predictions.is_file() and complete.is_file():
        evaluate.prediction_preflight([c])
        return 'REUSED_COMPLETE'
    if predictions.exists() or complete.exists():
        raise ValueError('Partial prediction evidence requires explicit reconciliation; not archived')
    allowed = {'provenance.json', 'resolved_config.json'}
    children = list(output.iterdir())
    if not children or any(not p.is_file() or p.name not in allowed for p in children):
        raise ValueError('Unexpected failed prediction contents; refusing to move')
    provenance = d.read(output / 'provenance.json')
    resolved = d.read(output / 'resolved_config.json')
    if (provenance['config'] != c or provenance['checkpoint'] != str(Path(c['output_root']) / 'final_ssdg.pth')
            or provenance['truth_read'] or provenance['query_fit'] or resolved['row_id'] != c['row_id']):
        raise ValueError('Failed prediction metadata belongs to a different row')
    archive = RECOVERY_BASE / c['row_id'] / 'recovery_archive' / 'prediction-r02-failed'
    source_root, archive_root = d.BASE.resolve(), RECOVERY_BASE.resolve()
    if not output.resolve().is_relative_to(source_root) or not archive.resolve().is_relative_to(archive_root):
        raise ValueError('Prediction archive escaped intended source/recovery roots')
    if output.is_symlink() or archive.exists():
        raise FileExistsError('Archive collision or symlink; preserving both paths')
    archive.parent.mkdir(parents=True, exist_ok=True)
    output.rename(archive)
    if output.exists() or not archive.is_dir():
        raise RuntimeError('Metadata archive post-state differs')
    d.write(archive.parent / 'archive_receipt.json', dict(status='VERIFIED', original=str(output),
        archive=str(archive), files=sorted(p.name for p in archive.iterdir()),
        predictions_existed=False, complete_existed=False, deleted=False))
    return 'ARCHIVED_METADATA_ONLY'


def score_complete(row_id=None):
    base = d.BASE / 'row_scoring' / row_id if row_id else d.BASE
    path = base / 'scoring_complete.json'
    if not path.is_file():
        return False
    require_scored(d.read(path), 1 if row_id else 12)
    for name in ('scores.json', 'scores.csv', 'day_scores.json', 'summary.json', 'paired_results.json', 'analysis.md'):
        if not (base / name).is_file():
            raise ValueError('Scoring completion is missing artifact ' + name)
    return True


def independent_score(row_id=None):
    command = [sys.executable, '-m', 'experiments.cvs_stage2_decouple.evaluate', '--mode', 'score']
    if row_id:
        command.extend(['--row', row_id])
    subprocess.run(command, cwd=d.ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
    if not score_complete(row_id):
        raise RuntimeError('Independent scorer did not produce completion')


def await_capacity(row_id):
    gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
    stable = 0
    while stable < 3:
        with lock(Path(d.PROJECT) / 'runs/receiver_residual_capacity4.lock'):
            observed = capacity()[gpu]
            owners = observed['pids'] | {os.getpid()}
            ready = len(owners) <= LIMIT and observed['free_mb'] >= FREE_MB
            stable = stable + 1 if ready else 0
            d.write(RECOVERY_BASE / 'capacity_leases' / (row_id + '.json'),
                dict(pid=os.getpid(), gpu=gpu, observed_pids=sorted(owners), stable_checks=stable,
                     per_gpu_limit=LIMIT, status='CAPACITY_READY' if stable == 3 else 'WAITING_CAPACITY'))
        if stable < 3:
            time.sleep(1 if ready else 5)


def worker(row_id):
    c = next((d.config(r) for r in d.rows() if r['row_id'] == row_id), None)
    if c is None:
        raise ValueError('Unregistered test recovery row')
    root = RECOVERY_BASE / row_id
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / 'owner.lock', nonblocking=True):
        try:
            await_capacity(row_id)
            state = reconcile_prediction(c)
            if state != 'REUSED_COMPLETE':
                evaluate.predict(row_id)
            evaluate.prediction_preflight([c])
            reused_score = score_complete(row_id)
            if not reused_score:
                independent_score(row_id)
            result = dict(status='ANALYZED', scope='TEST_ONLY', views=['clean'],
                recovery_run=RECOVERY_RUN, source_run=d.RUN, row_id=row_id,
                prediction_recovery=state, score_reused=reused_score, technical_recovery=True,
                execution_commit=commit(), independent_recount='VERIFIED', retraining=False,
                target_feedback_forbidden=True, old_failure_preserved=True)
            d.write(root / 'completion.json', result)
            old_row_completion = d.BASE / row_id / 'completion.json'
            if not old_row_completion.exists():
                d.write(old_row_completion, result)
            if (d.BASE / row_id / 'failure.json').is_file():
                d.write(d.BASE / row_id / 'failure_resolution.json', dict(status='RESOLVED',
                    original_failure_preserved=True, evidence=str(root / 'completion.json'), recovery_run=RECOVERY_RUN))
            record('ANALYZED', row_id=row_id, technical_recovery=RECOVERY_RUN, views=['clean'])
        except Exception as exc:
            d.write(root / 'failure.json', dict(status='FAILED', error=repr(exc),
                    original_failure_preserved=True, automatic_retry=False, execution_commit=commit()))
            raise


def launch(row_id, gpu):
    command = [sys.executable, '-u', '-m', 'experiments.cvs_stage2_decouple.test_recovery',
               '--worker', row_id, '--train-stage2']
    log = RECOVERY_BASE / 'logs' / (row_id + '.log')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONPATH=str(d.ROOT) + os.pathsep + str(d.ROOT / 'code'), OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with log.open('x', encoding='utf-8') as stream:
        child = subprocess.Popen(command, cwd=d.ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    actual = process(child.pid)
    if actual is None:
        raise RuntimeError('Recovery worker absent immediately after launch')
    return child, dict(row_id=row_id, pid=child.pid, start_ticks=actual['start_ticks'],
        gpu=gpu, cwd=str(d.ROOT), argv=command, log=str(log))


def run():
    RECOVERY_BASE.mkdir(parents=True, exist_ok=True)
    with lock(RECOVERY_BASE / 'owner.lock', nonblocking=True):
        if (RECOVERY_BASE / 'controller.json').exists():
            raise FileExistsError('Recovery already submitted; reconcile its independent evidence before restarting')
        verified = preflight()
        initialize_registration()
        (RECOVERY_BASE / 'logs').mkdir(exist_ok=True)
        d.write(RECOVERY_BASE / 'preflight.json', verified)
        own = process(os.getpid())
        d.write(RECOVERY_BASE / 'controller.json', dict(pid=os.getpid(), start_ticks=own['start_ticks'],
            cwd=str(d.ROOT), argv=sys.argv, commit=commit(), source_run=d.RUN,
            recovery_run=RECOVERY_RUN, scope='TEST_ONLY', launch_owner=d.OWNER))
        recovery_record('RUNNING', rows=12, views=['clean'], retraining=False)
        pending = [c['row_id'] for c in verified['configs']]
        active, receipts, completed, failures = {}, [], [], []
        try:
            while pending or active:
                for row_id, child in list(active.items()):
                    result = child.poll()
                    if result is None:
                        continue
                    path = RECOVERY_BASE / row_id / 'completion.json'
                    valid = path.is_file() and d.read(path)['status'] == 'ANALYZED'
                    if result or not valid:
                        failures.append(dict(row_id=row_id, exit_code=result, artifact_valid=valid))
                    else:
                        completed.append(row_id)
                    del active[row_id]
                while pending and not failures:
                    with lock(Path(d.PROJECT) / 'runs/receiver_residual_capacity4.lock'):
                        gpu = choose_gpu(capacity())
                        if gpu is None:
                            break
                        row_id = pending.pop(0)
                        child, receipt = launch(row_id, gpu)
                        active[row_id] = child
                        receipts.append(receipt)
                        d.write(RECOVERY_BASE / 'launch.json', dict(rows=receipts))
                d.write(RECOVERY_BASE / 'queue_state.json', dict(active=list(active), pending=pending,
                    completed=completed, failures=failures, per_gpu_limit=LIMIT, scope='TEST_ONLY'))
                if failures and not active:
                    raise RuntimeError('Test recovery failure preserved without retry: ' + repr(failures))
                if pending or active:
                    time.sleep(5)
            if not score_complete():
                independent_score()
            result = dict(status='ANALYZED', scope='TEST_ONLY', recovery_run=RECOVERY_RUN, source_run=d.RUN,
                trained_models=12, tested_models=12, model_seeds=list(d.SEEDS), views=['clean'],
                technical_recovery=True, all_rows_tested_automatically=True, retraining=False,
                execution_commit=commit(), independent_recount='VERIFIED', old_failure_preserved=True)
            if not (d.BASE / 'completion.json').exists():
                d.write(d.BASE / 'completion.json', result)
            finalize_report()
            recovery_report = RECOVERY_BASE / 'report.md'
            recovery_report.write_text(recovery_report.read_text(encoding='utf-8') + '\n\n'
                + (d.BASE / 'analysis.md').read_text(encoding='utf-8'), encoding='utf-8')
            d.write(RECOVERY_BASE / 'completion.json', result)
            if (d.BASE / 'failure.json').is_file():
                d.write(d.BASE / 'failure_resolution.json', dict(status='RESOLVED', recovery_run=RECOVERY_RUN,
                    original_failure_preserved=True, evidence=str(RECOVERY_BASE / 'completion.json')))
            record('ANALYZED', technical_recovery=RECOVERY_RUN, trained_models=12, tested_models=12, views=['clean'])
            recovery_record('ANALYZED', rows=12, views=['clean'])
        except Exception as exc:
            d.write(RECOVERY_BASE / 'failure.json', dict(status='FAILED', error=repr(exc),
                automatic_retry=False, source_run=d.RUN, old_failure_preserved=True, execution_commit=commit()))
            recovery_record('FAILED', error=repr(exc))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker')
    parser.add_argument('--train-stage2', action='store_true')
    args = parser.parse_args()
    worker(args.worker) if args.worker else run()
