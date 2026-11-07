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
    parser.add_argument('--save-source-context-cache', type=Path)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--fast-flow-guards', action='store_true')
    parser.add_argument('--cuda-graph-flow', action='store_true')
    parser.add_argument('--epoch-flow-times', action='store_true')
    parser.add_argument('--objective', choices=('iq', 'dura_guard', 'band_guard', 'nonlinear_guard', 'crossview_guard', 'sim_action_guard', 'sim_action_fixed_prior', 'moment_guard', 'geometry_guard', 'ssl_joint_guard', 'ssl_fixed_guard', 'distill_iq', 'posterior_teacher_guard', 'ssl_crosshalf_guard', 'tail_excess_guard', 'wide_source_guard', 'wide_source_distill', 'wide_mixture_guard', 'wide_expert_distill', 'wide_masked_prior_guard', 'wide_temporal_prior_guard', 'wide_student_fir_guard', 'wide_frequency_regression', 'wide_reversible_guard', 'wide_balanced_spectral_guard', 'wide_multiview_dura_guard', 'wide_counterfactual_guard', 'wide_instance_contrast', 'wide_multiview_distill', 'wide_conditional_flow', 'wide_direct_flow', 'wide_direct_phase', 'wide_direct_circular', 'wide_conditional_flow_t0', 'wide_envelope_guard', 'wide_quartic_guard', 'wide_antithetic_flow', 'wide_phase_equivariant_guard', 'wide_dura_circular', 'wide_conditional_phase', 'wide_conditional_phase_moments', 'wide_supervised_phase', 'wide_lag_context', 'wide_large_context', 'wide_large_context_tail', 'wide_large_context_stable', 'wide_large_context_stable_tail', 'wide_large_context_stable_observable', 'wide_large_context_stable_crossview', 'wide_large_context_stable_receiver', 'wide_large_context_stable_adaptive', 'wide_large_context_stable_timefreq', 'wide_large_context_stable_arctan', 'wide_large_context_stable_wide', 'wide_large_direct_timefreq', 'wide_large_direct_timefreq_stable', 'wide_large_direct_timefreq_stable_physics', 'wide_large_direct_timefreq_stable_diverse', 'wide_large_direct_timefreq_stable_wide', 'wide_large_direct_timefreq_stable_wide_physics', 'wide_conditional_adversary'), default='iq')
    parser.add_argument('--architecture', choices=('orthogonal', 'dual_domain', 'dura_internal', 'adaptive_inverse', 'timevarying_inverse', 'dura_timevarying', 'dura_timevarying_gate', 'dura_multiscale', 'direct_spectral_student', 'dura_experts', 'direct_student_fir', 'dura_frequency_regression', 'dura_reversible', 'conditional_iq_flow', 'direct_iq_flow', 'direct_phase_recovery', 'direct_circular_recovery', 'dura_envelope', 'dura_quartic', 'antithetic_iq_flow', 'phase_equivariant_dura', 'dura_circular_recovery', 'conditional_phase_dura', 'conditional_phase_moments', 'phase_supervised_dura', 'lag_context_dura', 'large_context_dura', 'large_context_stable', 'large_direct_timefreq', 'large_direct_timefreq_stable'), default='orthogonal')
    args = parser.parse_args()
    if args.epoch_flow_times and ((args.objective, args.architecture) != ('wide_conditional_flow', 'conditional_iq_flow')
                                  or not args.deterministic or args.device != 'cuda:0'):
        raise ValueError('Epoch flow times require deterministic conditional flow on cuda:0')
    if args.cuda_graph_flow and ((args.objective, args.architecture) != ('wide_conditional_flow', 'conditional_iq_flow')
                                 or not args.deterministic or args.device != 'cuda:0' or args.fast_flow_guards):
        raise ValueError('CUDA graph requires deterministic conditional flow on cuda:0 without fast-flow-guards')
    if args.fast_flow_guards and (args.objective, args.architecture) not in (
            ('wide_conditional_flow', 'conditional_iq_flow'), ('wide_antithetic_flow', 'antithetic_iq_flow')):
        raise ValueError('Fast flow guards require conditional or antithetic flow')
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
    training_module = ('experiments.phase1_dura_orthogonal.train_frequency_source' if args.architecture == 'dura_frequency_regression'
                       else 'experiments.phase1_dura_orthogonal.train_adversarial_source' if args.objective == 'wide_conditional_adversary'
                       else 'experiments.phase1_dura_orthogonal.train_source')
    try:
        status('RUNNING_SOURCE_TRAINING_SMOKE')
        execute('source_smoke', [sys.executable, '-X', 'utf8', '-u', '-m',
                training_module, *common, '--out', str(training_smoke), '--smoke',
                '--device', args.device, '--prepare-batch', str(args.prepare_batch), '--objective', args.objective,
                '--architecture', args.architecture, *(['--source-context-cache', str(args.source_context_cache)]
                    if args.architecture == 'dura_frequency_regression' and args.source_context_cache else []),
                *(['--deterministic'] if args.deterministic else []),
                *(['--fast-flow-guards'] if args.fast_flow_guards else []),
                *(['--cuda-graph-flow'] if args.cuda_graph_flow else []),
                *(['--epoch-flow-times'] if args.epoch_flow_times else [])])
        smoke = json.loads((training_smoke/'smoke.json').read_text(encoding='utf8'))
        if smoke.get('status') != 'SOURCE_TRAINING_SMOKE_PASS' or smoke.get('F_weights_read') != 0:
            raise ValueError('Source-only training smoke required')
        status('RUNNING_SOURCE_TRAINING')
        execute('source_train', [sys.executable, '-X', 'utf8', '-u', '-m',
                training_module, *common, '--out', str(source),
                '--device', args.device, '--prepare-batch', str(args.prepare_batch), '--objective', args.objective,
                '--architecture', args.architecture, *(['--source-context-cache', str(args.source_context_cache)]
                                                       if args.source_context_cache else []),
                *(['--save-source-context-cache', str(args.save_source_context_cache)] if args.save_source_context_cache else []),
                *(['--deterministic'] if args.deterministic else []),
                *(['--fast-flow-guards'] if args.fast_flow_guards else []),
                *(['--cuda-graph-flow'] if args.cuda_graph_flow else []),
                *(['--epoch-flow-times'] if args.epoch_flow_times else [])])
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
