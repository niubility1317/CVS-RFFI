"""One-owner GPU queue and immutable source-to-seven-view execution."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.phase1_cumulant_rank.source_pair_queue import wait_for_project_slot



def execute(root, source, ground, commit, cache, gpu):
    if (Path.cwd()/'release_commit.txt').read_text(encoding='ascii').strip() != commit:
        raise ValueError('Release must contain the registered code commit')
    runs = root/'runs'
    controller = runs/(source+'-controller')
    source_dir, smoke_dir, ground_dir = runs/source, runs/(source+'-training-smoke'), runs/ground
    if any(path.exists() for path in (controller, source_dir, smoke_dir, ground_dir)):
        raise FileExistsError('All run outputs must be new')
    if not cache.is_file():
        raise FileNotFoundError('Fixed source context cache required')
    # Resource ordering only: allow the already queued source pair diagnostic to
    # finish first, preventing two workers from this task sharing GPU4.
    prior = runs/'20261109-phase1-cumulant-rank-paired-source-s2026092701-r01-controller'/'status.json'
    while True:
        if not prior.is_file():
            raise FileNotFoundError('Existing source-pair resource queue required')
        state = json.loads(prior.read_text(encoding='utf8')).get('state')
        if state in ('SOURCE_DIAGNOSTIC_COMPLETE', 'FAILED'):
            break
        print('TRUE_RESPONSE_WAITING_FOR_PRIOR_GPU_OWNER', state, flush=True)
        time.sleep(20)
    wait_for_project_slot(root, gpu)
    controller.mkdir(parents=True)
    gpu_env = os.environ.copy()
    gpu_env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    cpu_env = gpu_env.copy()
    cpu_env['CUDA_VISIBLE_DEVICES'] = ''

    def status(state, **extra):
        tmp = controller/'status.tmp'
        tmp.write_text(json.dumps(dict(state=state, **extra)), encoding='utf8')
        os.replace(tmp, controller/'status.json')

    def stage(name, state, command, *, use_gpu):
        if use_gpu:
            status('WAITING_GPU', next_stage=name)
            wait_for_project_slot(root, gpu)
        with (controller/(name+'.log')).open('x', encoding='utf8') as log:
            process = subprocess.Popen(command, cwd=Path.cwd(), env=gpu_env if use_gpu else cpu_env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            status(state, pid=process.pid)
            if process.wait() != 0:
                raise RuntimeError(name+' exited '+str(process.returncode))

    common = ['--root', str(root), '--commit', commit]
    python = [sys.executable, '-X', 'utf8', '-u', '-m']
    try:
        stage('source_smoke', 'RUNNING_SOURCE_SMOKE', python+['experiments.phase1_cumulant_true_response.train',
              *common, '--out', str(smoke_dir), '--cache', str(cache), '--device', 'cuda:0', '--smoke'], use_gpu=True)
        stage('ground_smoke', 'RUNNING_SOURCE_PREDICTION_SMOKE', python+['experiments.phase1_cumulant_true_response.ground',
              'smoke', *common, '--out', str(smoke_dir/'prediction-check'), '--source', source, '--device', 'cuda:0'], use_gpu=True)
        stage('source_train', 'RUNNING_SOURCE_TRAINING', python+['experiments.phase1_cumulant_true_response.train',
              *common, '--out', str(source_dir), '--cache', str(cache), '--device', 'cuda:0'], use_gpu=True)
        stage('ground_predict', 'RUNNING_SEVEN_VIEW_PREDICTION', python+['experiments.phase1_cumulant_true_response.ground',
              'predict', *common, '--out', str(ground_dir), '--source', source, '--device', 'cuda:0'], use_gpu=True)
        stage('ground_score', 'RUNNING_INDEPENDENT_TRUTH_LAST_SCORE', python+['experiments.phase1_cumulant_true_response.ground',
              'score', *common, '--out', str(ground_dir), '--source', source, '--device', 'cpu'], use_gpu=False)
        status('GROUND_SCORE_COMPLETE')
    except BaseException as error:
        (controller/'failure.json').write_text(json.dumps(dict(error=repr(error), traceback=traceback.format_exc())),
                                               encoding='utf8')
        status('FAILED', error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--ground', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    execute(args.root, args.source, args.ground, args.commit, args.cache, args.gpu)


if __name__ == '__main__':
    main()
