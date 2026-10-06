"""Deferred evaluation repair; source workers and their immutable run stay untouched."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

EVAL_RUN = '20261006-phase1-reference-stack-sixscene-manysig-m136-r03'
RELEASE = 'cvs_reference_stack_evaluation_repair_20261006_r01'
PROJECT = '/home/szu2070436088/2510044040/CV-SincNet'
BASE = Path(PROJECT) / 'runs' / EVAL_RUN
ROOT = Path(__file__).resolve().parents[2]
WAIT_SECONDS = 30
COMPLETE = dict(status='ANALYZED', rows=136, independent_recount='VERIFIED',
                target_feedback_forbidden=True)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def parent_ready(d, process):
    """Read only source/control artifacts; only the known pre-query failure opens evaluation."""
    parent = Path(d.BASE)
    active = d.read(parent / 'dispatcher_active.json')
    worker_root = Path(d.PROJECT) / 'releases' / d.RELEASE
    if (active.get('run_id') != d.RUN or active.get('release') != d.RELEASE
            or active.get('cwd') != str(worker_root)
            or active.get('argv', [])[1:] != ['-u', '-m', 'experiments.cvs_phase1_stack.recover']):
        raise ValueError('Unexpected original controller identity')
    now = process(active['pid'])
    if now and now['start_ticks'] == active['start_ticks']:
        if now['cwd'] != active['cwd'] or now['argv'] != active['argv']:
            raise ValueError('Original controller identity changed')
        return False
    if (parent / 'completion.json').exists():
        raise ValueError('Original run already completed; repair is not authorized for this state')
    required = dict(status='ALL_SOURCE_FROZEN', run_id=d.RUN,
                    rows=[r['row_id'] for r in d.rows()], stages=list(d.STAGES), target_access=False)
    if not (parent / 'all_sources_frozen.json').exists() or d.read(parent / 'all_sources_frozen.json') != required:
        raise ValueError('Original controller exited before complete source freeze')
    for stage in d.STAGES:
        d.validate_freeze(d.read(parent / (stage + '_source_frozen.json')), stage)
    state = d.read(parent / 'queue_state.json')
    failures = state.get('failures', [])
    if (state.get('phase') != 'predict' or state.get('kind') != 'predict'
            or state.get('active') != [] or not failures
            or state.get('controller_pid') != active['pid']
            or state.get('controller_release') != d.RELEASE):
        raise ValueError('Original failure was not a drained prediction queue')
    for failure in failures:
        if (set(failure) != {'row_id', 'exit_code', 'error'} or failure['exit_code'] != 1
                or failure['error'] != 'Worker failed or missing valid completion'):
            raise ValueError('Unexpected original worker failure')
    error = repr(RuntimeError('Failed rows preserved; no retry ' + repr(failures)))
    if d.read(parent / 'failure.json') != dict(status='FAILED', error=error, no_retry=True, release=d.RELEASE):
        raise ValueError('Original failure fingerprint differs')
    receipts = d.read(parent / 'launch_predict.json')['rows']
    launched = [r['row_id'] for r in receipts]
    failed = [r['row_id'] for r in failures]
    all_rows = required['rows']
    if (len(set(launched)) != len(launched) or len(set(failed)) != len(failed)
            or set(launched) != set(failed) or not set(launched) <= set(all_rows)
            or state.get('pending') != [rid for rid in all_rows if rid not in launched]):
        raise ValueError('Original prediction launch/failure coverage differs')
    for receipt in receipts:
        rid = receipt['row_id']
        log = Path(d.PROJECT) / 'logs' / d.RUN / ('predict-' + rid + '.log')
        argv_tail = ['-u', '-m', 'experiments.cvs_phase1_stack.predict', '--config',
                     str(parent / 'configs' / ('predict-' + rid + '.json'))]
        if (receipt.get('cwd') != str(worker_root) or receipt.get('argv', [])[1:] != argv_tail
                or receipt.get('log') != str(log) or receipt.get('controller_release') != d.RELEASE):
            raise ValueError('Original prediction launch identity differs')
        if process(receipt['pid']) is not None:
            raise ValueError('Original prediction worker has not exited')
        text = log.read_text(encoding='utf-8')
        if (not text.rstrip().endswith("KeyError: 'classes'")
                or str(worker_root / 'experiments/cvs_phase1_stack/predict.py') not in text
                or 'in predict' not in text
                or "if contract[k]!=expected_contract[k]:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH '+k)" not in text):
            raise ValueError('Prediction log does not prove the known pre-query schema failure')
    for rid in all_rows:
        output = parent / rid / 'prediction'
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError('Original query artifacts exist; automatic recovery forbidden')
    for name in ['scoring_p1_complete.json', 'phase1_scored_results.json', 'independent_recount.json']:
        if (parent / name).exists():
            raise ValueError('Original scoring artifacts exist; automatic recovery forbidden')
    return True


def mixed(d, q):
    from experiments.cvs_phase1_stack.predict import EVALUATION_RUN, PREDICTION_BASE
    output = BASE / 'mixed'
    if EVALUATION_RUN != EVAL_RUN or Path(PREDICTION_BASE) != output or d.ROOT != ROOT:
        raise ValueError('Evaluation output or release import mismatch')
    output.mkdir(exist_ok=False)
    try:
        jobs = []
        for row in d.rows():
            rid = row['row_id']
            source_path = Path(d.BASE) / 'configs' / ('source-' + rid + '.json')
            c = d.read(source_path)
            d.validate(c)
            config_path = output / 'configs' / ('predict-' + rid + '.json')
            prediction = dict(row_id=rid, source_config=str(source_path), source_output=c['output_root'],
                              output_root=str(output / rid / 'prediction'), p1_capsule=d.CAPSULE)
            write(config_path, prediction)
            jobs.append(dict(row_id=rid, config=str(config_path), output_root=prediction['output_root'],
                             model_seed=c['model_seed'], method=c['stage'] + '-' + c['arm'], stage='phase12'))
        spec = dict(run_id=EVAL_RUN, runtime_root=str(output), p1_truth=d.TRUTH, rows=jobs)
        spec_path = output / 'evaluation_spec.json'
        write(spec_path, spec)
        queue_design = SimpleNamespace(BASE=str(output), RUN=EVAL_RUN, ROOT=ROOT,
                                       PROJECT=d.PROJECT, read=d.read, write=write)
        q.CONTROL_RELEASE = RELEASE
        q.queue(queue_design, jobs, 'predict', 'predict', {})
        # The independent scorer validates every fixed prediction before reading truth.
        for module, args in [('comparison_suite.score', ['--stage', 'p1']),
                             ('experiments.cvs_phase1_stack.analyze', [])]:
            subprocess.run([sys.executable, '-m', module, '--spec', str(spec_path), *args], cwd=ROOT, check=True)
        write(output / 'completion.json', COMPLETE)
    except Exception as error:
        write(output / 'failure.json', dict(status='FAILED', error=repr(error), no_retry=True))
        raise


def dispatch():
    from experiments.cvs_phase1_stack import dispatch as d, capacity16 as q
    BASE.mkdir(parents=True, exist_ok=False)
    try:
        (Path(PROJECT) / 'logs' / EVAL_RUN).mkdir(parents=True, exist_ok=False)
        if d.ROOT != ROOT or ROOT != Path(PROJECT) / 'releases' / RELEASE:
            raise ValueError('Unexpected evaluation release')
        write(BASE / 'dispatcher.json', dict(**q.proc(os.getpid()), run_id=EVAL_RUN,
              owner='codex/root/reference-stack-evaluation-recovery-20261006', release=RELEASE,
              commit=(ROOT / 'release_commit.txt').read_text().strip(), parent_run=d.RUN))
        while True:
            write(BASE / 'state.json', dict(status='WAITING_PARENT_SCHEMA_FAILURE', parent_run=d.RUN,
                  updated_at=time.time(), target_read=False, gpu_reserved=False))
            if parent_ready(d, q.proc):
                break
            time.sleep(WAIT_SECONDS)
        write(BASE / 'parent_gate.json', dict(status='VERIFIED', parent_run=d.RUN,
              all_sources_frozen=True, original_controller_exited=True,
              failure='PRE_QUERY_CONTRACT_KEYERROR_CLASSES', original_prediction_outputs='EMPTY', target_read=False))
        write(BASE / 'state.json', dict(status='MIXED_EVALUATION', parent_run=d.RUN, updated_at=time.time()))
        mixed(d, q)
        from experiments.cvs_phase1_stack import sixscene_after
        if sixscene_after.BASE != BASE / 'seven_views' or sixscene_after.RELEASE != RELEASE:
            raise ValueError('Seven-view evaluation location differs')
        write(BASE / 'state.json', dict(status='SEVEN_VIEW_EVALUATION', parent_run=d.RUN, updated_at=time.time()))
        sixscene_after.dispatch()
        expected = dict(COMPLETE, views=list(sixscene_after.VIEWS))
        if d.read(BASE / 'seven_views' / 'completion.json') != expected:
            raise ValueError('Seven-view completion mismatch')
        write(BASE / 'completion.json', dict(COMPLETE, mixed=True, views=list(sixscene_after.VIEWS)))
        write(BASE / 'state.json', dict(status='ANALYZED', updated_at=time.time()))
    except Exception as error:
        write(BASE / 'failure.json', dict(status='FAILED', error=repr(error), no_retry=True, release=RELEASE))
        raise


if __name__ == '__main__':
    dispatch()
