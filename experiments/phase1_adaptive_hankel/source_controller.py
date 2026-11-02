"""Run the preregistered independent adaptive Hankel source experiment."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import traceback

from experiments.phase1_dura_posterior.resources import gpu_lease, wait_for_cpu


ROOT = Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN = '20261102-phase1-adaptive-hankel-source-s2026092701-r01'
RELEASE = ROOT/'releases'/(RUN+'-release')
CONTROLLER = ROOT/'runs'/(RUN+'-controller')
GPU_LOCK = ROOT/'runs/20261102-phase1-dura-posterior-twoarm-s2026092701-r01-gpu.lock'
PYTHON = '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'


def status(state, **extra):
    temporary = CONTROLLER/'status.tmp'
    temporary.write_text(json.dumps(dict(state=state, **extra), indent=2)+'\n', encoding='utf8')
    os.replace(temporary, CONTROLLER/'status.json')


def environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    releases = ROOT/'releases'
    env['PYTHONPATH'] = ':'.join(map(str, (RELEASE,
        releases/'20261101-phase1-physical-rich-router-s2026092701-r01-release',
        releases/'20261027-phase1-pamf-repl-shuffled-views-m16-r02-release')))
    return env


def run_module(module, args, log_path, gpu, lock_fd=None):
    command = [PYTHON, '-X', 'utf8', '-u', '-m', module, *args]
    with log_path.open('x', encoding='utf8') as log:
        result = subprocess.run(command, cwd=RELEASE, env=environment(gpu), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT,
                                pass_fds=() if lock_fd is None else (lock_fd,))
    if result.returncode:
        raise RuntimeError(module+' exited '+str(result.returncode))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    source = ROOT/'runs'/RUN
    smoke = ROOT/'runs'/(RUN+'-training-smoke')
    score = ROOT/'runs'/(RUN+'-score')/'source_score.json'
    if (not CONTROLLER.is_dir() or any(CONTROLLER.iterdir())
            or any(path.exists() for path in (source, smoke, score.parent))):
        raise ValueError('Unique source experiment outputs are required')
    if (RELEASE/'release_commit.txt').read_text(encoding='ascii').strip() != args.commit:
        raise ValueError('Registered release commit required')
    try:
        wait_for_cpu(lambda state, **extra: status(state, phase='gpu_lock_check', **extra))
        status('CHECKING_GPU_LOCK_INHERITANCE')
        run_module('unittest',
                   ['experiments.phase1_dura_posterior.test_resources.ResourceTest.test_worker_inherits_lock_after_controller_closes_its_descriptor'],
                   CONTROLLER/'gpu-lock-test.log', None)
        common = ['--root', str(ROOT), '--commit', args.commit, '--device', 'cuda:0']
        with gpu_lease(GPU_LOCK, lambda state, **extra: status(state, phase='training_smoke', **extra)) as (gpu, lock_fd):
            status('RUNNING_TRAINING_SMOKE', gpu=gpu)
            run_module('experiments.phase1_adaptive_hankel.train_source',
                       [*common, '--out', str(smoke), '--smoke'], CONTROLLER/'smoke.log', gpu, lock_fd)
        result = json.loads((smoke/'smoke.json').read_text(encoding='utf8'))
        expected = dict(status='SOURCE_TRAINING_SMOKE_PASS', fit_ids=64, eval_records=28,
                        F_frozen=True, blind_frozen=True, gate_frozen=True, DURA_loaded=False,
                        TX_labels_read=0, U_s_truth_reads=0, V_reads=0, test_reads=0)
        if any(result.get(key) != value for key, value in expected.items()) or result.get('adaptive_gradient_batches', 0) < 1:
            raise ValueError('Real-source adaptive Hankel training smoke did not pass')
        with gpu_lease(GPU_LOCK, lambda state, **extra: status(state, phase='source_train', **extra)) as (gpu, lock_fd):
            status('RUNNING_SOURCE_TRAIN', gpu=gpu)
            run_module('experiments.phase1_adaptive_hankel.train_source',
                       [*common, '--out', str(source)], CONTROLLER/'train.log', gpu, lock_fd)
        result = json.loads((source/'prediction_complete.json').read_text(encoding='utf8'))
        expected = dict(status='SOURCE_PREDICTIONS_FIXED', commit=args.commit, count=672, fit_ids=8192,
                        F_frozen=True, blind_frozen=True, gate_frozen=True, DURA_loaded=False,
                        TX_labels_read=0, U_s_truth_reads=0, V_reads=0, test_reads=0, model_updates_after_prediction=0)
        if any(result.get(key) != value for key, value in expected.items()) or not (source/'checkpoint.pt').is_file():
            raise ValueError('Complete frozen source model and predictions required')
        wait_for_cpu(lambda state, **extra: status(state, phase='source_score', **extra))
        status('SCORING_FIXED_SOURCE')
        run_module('experiments.phase1_adaptive_hankel.score_source',
                   ['--root', str(ROOT), '--source', str(source), '--out', str(score), '--commit', args.commit],
                   CONTROLLER/'score.log', None)
        result = json.loads(score.read_text(encoding='utf8'))
        if result.get('status') != 'SOURCE_SCORE_COMPLETE':
            raise ValueError('Complete truth-last source score required')
        status('SOURCE_SCORE_COMPLETE', source=str(source), score=str(score), L_s_scored=result['L_s_scored'])
    except Exception as error:
        status('FAILED', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
