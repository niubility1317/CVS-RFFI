"""Source-only training, frozen seven-view inference and truth-last scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def training_environments(device, inherited):
    """Require an explicitly selected physical GPU and hide it from scoring."""
    env = inherited.copy()
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    cpu_env = env.copy()
    cpu_env['CUDA_VISIBLE_DEVICES'] = ''
    if device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    elif device == 'cuda:0':
        visible = env.get('CUDA_VISIBLE_DEVICES', '').strip()
        if not visible or visible == '-1' or ',' in visible:
            raise ValueError('Select exactly one physical GPU via CUDA_VISIBLE_DEVICES')
    else:
        raise ValueError('Use cpu or cuda:0 mapped to one explicitly selected GPU')
    return env, cpu_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--ground', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--prepare-batch', type=int, default=32)
    parser.add_argument('--source-context-cache', type=Path)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--objective', choices=('iq', 'dura_guard', 'band_guard', 'nonlinear_guard', 'crossview_guard', 'sim_action_guard', 'sim_action_fixed_prior', 'moment_guard', 'geometry_guard', 'ssl_joint_guard', 'ssl_fixed_guard', 'distill_iq', 'posterior_teacher_guard', 'ssl_crosshalf_guard', 'tail_excess_guard'), default='iq')
    parser.add_argument('--architecture', choices=('orthogonal', 'dual_domain', 'dura_internal', 'adaptive_inverse', 'timevarying_inverse', 'dura_timevarying', 'dura_timevarying_gate', 'dura_multiscale', 'direct_spectral_student'), default='orthogonal')
    args = parser.parse_args()
    env, cpu_env = training_environments(args.device, os.environ)
    controller = args.root/'runs'/(args.source+'-controller')
    source = args.root/'runs'/args.source
    training_smoke = args.root/'runs'/(args.source+'-training-smoke')
    ground = args.root/'runs'/args.ground
    if controller.exists() or source.exists() or training_smoke.exists() or ground.exists():
        raise FileExistsError('Unique controller, training and ground outputs required')
    controller.mkdir(parents=True)
    def status(state, **extra):
        tmp = controller/'status.tmp'
        tmp.write_text(json.dumps(dict(state=state, **extra)), encoding='utf8')
        os.replace(tmp, controller/'status.json')

    def execute(name, command, *, cpu=False):
        with (controller/(name+'.log')).open('x', encoding='utf8') as log:
            process = subprocess.run(command, cwd=Path.cwd(), env=cpu_env if cpu else env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(name+' exited '+str(process.returncode))

    common = ['--root', str(args.root), '--commit', args.commit]
    try:
        status('RUNNING_SOURCE_TRAINING_SMOKE')
        execute('source_smoke', [sys.executable, '-X', 'utf8', '-u', '-m',
                'experiments.phase1_dura_orthogonal.train_source', *common, '--out', str(training_smoke), '--smoke',
                '--device', args.device, '--prepare-batch', str(args.prepare_batch), '--objective', args.objective,
                '--architecture', args.architecture, *(['--deterministic'] if args.deterministic else [])])
        smoke = json.loads((training_smoke/'smoke.json').read_text(encoding='utf8'))
        if smoke.get('status') != 'SOURCE_TRAINING_SMOKE_PASS' or smoke.get('F_weights_read') != 0:
            raise ValueError('Source-only training smoke required')
        status('RUNNING_SOURCE_TRAINING')
        execute('source_train', [sys.executable, '-X', 'utf8', '-u', '-m',
                'experiments.phase1_dura_orthogonal.train_source', *common, '--out', str(source),
                '--device', args.device, '--prepare-batch', str(args.prepare_batch), '--objective', args.objective,
                '--architecture', args.architecture, *(['--source-context-cache', str(args.source_context_cache)]
                                                       if args.source_context_cache else []),
                *(['--deterministic'] if args.deterministic else [])])
        status('RUNNING_SOURCE_PREDICTION_SMOKE')
        fixed = ['--source', args.source, '--out', str(ground), *common]
        execute('ground_smoke', [sys.executable, '-X', 'utf8', '-u', '-m',
                'experiments.phase1_dura_orthogonal.ground', 'smoke', *fixed], cpu=True)
        status('RUNNING_SEVEN_VIEW_PREDICTION')
        execute('ground_predict', [sys.executable, '-X', 'utf8', '-u', '-m',
                'experiments.phase1_dura_orthogonal.ground', 'predict', *fixed], cpu=True)
        prediction = json.loads((ground/'flow/test_prediction/complete.json').read_text(encoding='utf8'))
        if (prediction.get('status') != 'PREDICTIONS_COMPLETE' or prediction.get('count') != 168000
                or prediction.get('TX_truth_reads') != 0 or prediction.get('model_updates') != 0):
            raise ValueError('All seven predictions must be fixed before scoring')
        status('RUNNING_INDEPENDENT_TRUTH_LAST_SCORE')
        execute('ground_score', [sys.executable, '-X', 'utf8', '-u', '-m',
                'experiments.phase1_dura_orthogonal.ground', 'score', *fixed], cpu=True)
        summary = json.loads((ground/'flow/test_summary.json').read_text(encoding='utf8'))
        pair = json.loads((ground/'flow/paired_dura.json').read_text(encoding='utf8'))
        if (summary.get('status') != 'ANALYZED' or summary.get('all_predictions_before_truth') is not True
                or summary.get('target_feedback') is not False or pair.get('status') != 'PAIRED_COMPLETE'):
            raise ValueError('Completed independent paired score required')
        status('GROUND_SCORE_COMPLETE')
    except Exception as exc:
        status('FAILED', error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
