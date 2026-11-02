"""Run both preregistered source candidates without selecting by score."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from experiments.phase1_dura_posterior.resources import wait_for_cpu, wait_for_gpu


ROOT = Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN = '20261102-phase1-dura-posterior-twoarm-s2026092701-r01'
RELEASE = ROOT/'releases'/(RUN+'-release')
OUTPUT = ROOT/'runs'/(RUN+'-source-controller')
REAL_SMOKE = ROOT/'runs/20261102-phase1-dura-posterior-internal-smoke-s2026092701-r01/status.json'
REAL_SMOKE_PID = 1524603
PYTHON = '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
ARMS = {
    'mean': '20261102-phase1-dura-posterior-internal-source-s2026092701-r01',
    'nonlocal': '20261102-phase1-dura-nonlocal-internal-source-s2026092701-r01',
}


def status(state, **extra):
    temporary = OUTPUT/'status.tmp'
    temporary.write_text(json.dumps(dict(state=state, **extra), indent=2)+'\n', encoding='utf8')
    os.replace(temporary, OUTPUT/'status.json')


def environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    releases = ROOT/'releases'
    env['PYTHONPATH'] = ':'.join(map(str, (RELEASE,
        releases/'20261101-phase1-physical-rich-router-s2026092701-r01-release',
        releases/'20261027-phase1-pamf-repl-shuffled-views-m16-r02-release')))
    return env


def run_module(module, args, log_path, gpu):
    command = [PYTHON, '-X', 'utf8', '-u', '-m', module, *args]
    with log_path.open('x', encoding='utf8') as log:
        result = subprocess.run(command, cwd=RELEASE, env=environment(gpu), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(module+' exited '+str(result.returncode))


def wait_for_real_smoke():
    while True:
        result = json.loads(REAL_SMOKE.read_text(encoding='utf8'))
        state = result.get('state')
        if state == 'REAL_SOURCE_SMOKE_COMPLETE':
            value = result.get('result', {})
            if (value.get('status') != 'REAL_PHYSICAL_DURA_INSERTION_PASS'
                    or any(value.get(k) != 0 for k in ('TX_labels_read', 'U_s_truth_reads', 'V_reads', 'test_reads'))):
                raise ValueError('Original real-source DURA interface check is invalid')
            return
        if state == 'FAILED':
            raise RuntimeError('Original real-source DURA interface check failed')
        proc = Path('/proc')/str(REAL_SMOKE_PID)/'cmdline'
        if not proc.is_file() or 'phase1_dura_posterior.smoke_controller' not in proc.read_bytes().decode(errors='replace'):
            raise RuntimeError('Original real-source smoke stopped without a completed result')
        status('WAITING_REAL_SOURCE_SMOKE', upstream_state=state, upstream_pid=REAL_SMOKE_PID)
        time.sleep(60)


def run_arm(kind, name, commit):
    smoke = ROOT/'runs'/(name+'-training-smoke')
    source = ROOT/'runs'/name
    score = ROOT/'runs'/(name+'-score')/'source_score.json'
    for path in (smoke, source, score.parent):
        if path.exists():
            raise FileExistsError('Existing arm output requires reconciliation: '+str(path))
    common = ['--root', str(ROOT), '--commit', commit, '--device', 'cuda:0', '--posterior-kind', kind]
    gpu = wait_for_gpu(lambda state, **extra: status(state, arm=kind, phase='training_smoke', **extra))
    status('RUNNING_TRAINING_SMOKE', arm=kind, gpu=gpu)
    run_module('experiments.phase1_dura_posterior.train_source', [*common, '--out', str(smoke), '--smoke'],
               OUTPUT/(kind+'.smoke.log'), gpu)
    result = json.loads((smoke/'smoke.json').read_text(encoding='utf8'))
    expected = dict(status='SOURCE_TRAINING_SMOKE_PASS', fit_ids=64, eval_records=28,
                    posterior_kind=kind, F_frozen=True, DURA_frozen=True,
                    U_s_truth_reads=0, TX_labels_read=0, V_reads=0, test_reads=0)
    if any(result.get(k) != v for k, v in expected.items()) or result.get('active_exposures', 0) < 8:
        raise ValueError('The actual source training smoke did not pass')
    gpu = wait_for_gpu(lambda state, **extra: status(state, arm=kind, phase='source_train', **extra))
    status('RUNNING_SOURCE_TRAIN', arm=kind, gpu=gpu)
    run_module('experiments.phase1_dura_posterior.train_source', [*common, '--out', str(source)],
               OUTPUT/(kind+'.train.log'), gpu)
    result = json.loads((source/'prediction_complete.json').read_text(encoding='utf8'))
    expected = dict(status='SOURCE_PREDICTIONS_FIXED', commit=commit, count=672, fit_ids=8192,
                    posterior_kind=kind, F_frozen=True, DURA_frozen=True, gate_frozen=True,
                    posterior_frozen=True, U_s_truth_reads=0, TX_labels_read=0, V_reads=0,
                    test_reads=0, model_updates_after_prediction=0)
    if any(result.get(k) != v for k, v in expected.items()) or not (source/'checkpoint.pt').is_file():
        raise ValueError('Complete fixed source model and predictions required')
    wait_for_cpu(lambda state, **extra: status(state, arm=kind, phase='source_score', **extra))
    status('SCORING_FIXED_SOURCE', arm=kind)
    run_module('experiments.phase1_dura_posterior.score_source',
               ['--root', str(ROOT), '--source', str(source), '--out', str(score), '--commit', commit],
               OUTPUT/(kind+'.score.log'), None)
    result = json.loads(score.read_text(encoding='utf8'))
    if result.get('status') != 'SOURCE_SCORE_COMPLETE' or result.get('posterior_kind') != kind:
        raise ValueError('Fixed-source score output is invalid')
    return dict(source=str(source), score=str(score), L_s_scored=result['L_s_scored'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if not OUTPUT.is_dir() or any(OUTPUT.iterdir()):
        raise ValueError('One empty immutable source-controller root is required')
    if (RELEASE/'release_commit.txt').read_text(encoding='ascii').strip() != args.commit:
        raise ValueError('The registered release commit is required')
    try:
        wait_for_real_smoke()
        results, failures = {}, {}
        for kind, name in ARMS.items():
            try:
                results[kind] = run_arm(kind, name, args.commit)
            except Exception as error:
                failures[kind] = dict(error=str(error), traceback=traceback.format_exc())
                status('ARM_TECHNICAL_FAILURE', arm=kind, failures=failures, completed=results)
        if failures:
            status('PARTIAL_TECHNICAL_FAILURE', failures=failures, completed=results)
        else:
            status('BOTH_SOURCE_ARMS_SCORED', completed=results)
    except Exception as error:
        status('FAILED', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
