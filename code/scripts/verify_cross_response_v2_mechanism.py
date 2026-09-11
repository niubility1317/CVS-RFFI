"""Real CORE90 trainer: full-graph source audit, fail-closed gate and main parity.

Only synthetic IQ and deliberately impossible test-only thresholds are used.
No real checkpoint/target qualification or formal experiment is produced.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def run(root, *, device='cuda:0', amp=False, initial_scale=None):
    import torch
    from scripts.register_cross_response_v2 import configuration
    from scripts.verify_core90_cross_response import run_synthetic_variant
    from cvsrffi.cross_response.integration import CrossResponseRuntime
    from cvsrffi.cross_response.replay_audit import freeze, capture_rng_state, first_divergence
    from scripts.verify_cross_response_v2_entrypoint import compare_main_traces
    from SSDG import train_ssdg as train
    actual_scaler = train.GradScaler
    def make_scaler(**kwargs):
        if initial_scale is not None:
            kwargs['init_scale'] = initial_scale
        return actual_scaler(**kwargs)
    root.mkdir(parents=True, exist_ok=False)
    traces, results = {}, {}
    actual_init = CrossResponseRuntime.__init__
    for label, enabled in (('control', False), ('audited', True)):
        folder = root / label
        folder.mkdir()
        c = configuration()
        c['cross_response'].update(source_fit_max_records=128, source_eval_max_blocks=1, gate_min_blocks=1)
        if enabled:
            c['cross_response'].update(source_audit_enabled=True,
                source_baseline_fit_steps=2, source_baseline_fit_lr=.001,
                mechanism_gate=dict(thresholds={key: 1e6 for key in (
                    'capability_min_improvement', 'necessity_min_gap', 'update_min_improvement',
                    'max_hard_group_degradation', 'max_leo_degradation')},
                    stable_observations=1, min_samples=2,
                    source_freeze_id='SYNTHETIC_INTEGRATION_TEST_ONLY', source_frozen=True,
                    terminal_identity_module='time_fuse'),
                mechanism_audit=dict(interval_steps=1, pairs_per_observation=2,
                    query_records_per_cell=1, seed=392005, identity_scope='cls_head'))
        (folder / 'synthetic_cross_response_config.json').write_text(json.dumps(c, indent=2), encoding='utf-8')
        trace = []
        def observer(step, stage, values):
            trace.append(dict(step=step, stage=stage, values=freeze(values), rng=capture_rng_state()))
        def initialize(self, *args, **kwargs):
            actual_init(self, *args, **kwargs)
            self.replay_observer = observer
        def check_activation(report):
            counts = report['counts']
            assert counts['successful_steps'] > 0 and counts['domain_gradient_steps'] > 0
            assert counts['joint_gradient_steps'] == 0
            if enabled:
                state = report['source_mechanism_audit']
                assert state['attempts'] > 0 and state['paired_updates'] > 0
                if not amp or initial_scale is not None:
                    assert state['observations']
                else:
                    assert not report['mechanism_gate']['last_result']['authorized']
                assert all(not item['result']['authorized'] for item in state['observations'])
        with patch.object(CrossResponseRuntime, '__init__', initialize), \
             patch('scripts.verify_core90_cross_response.assert_activation', check_activation), \
             patch.object(train, 'GradScaler', make_scaler):
            result = run_synthetic_variant('U2', folder, device=device, epochs=4 if amp else 2,
                deterministic=True, amp=amp, transmitters=6, source_receivers=5, records_per_cell=128)
        results[label] = result
        traces[label] = trace
        print(label, 'TRAIN_AND_READBACK_VERIFIED', flush=True)
    # This helper removes only expected AMP nonfinite values by comparing their
    # exact masks/signs. Both branches otherwise have identical optimizer groups.
    divergence = compare_main_traces({'U1': traces['control'], 'head_only': traces['audited']}, amp=amp)
    torch.save(traces, root / 'actual_main_traces.pt')
    observed = bool(results['audited']['activation']['source_mechanism_audit']['observations'])
    report = dict(status=('MAIN_PARITY_AND_REAL_SOURCE_GATE_VERIFIED' if observed else
                          'MAIN_PARITY_AND_SOURCE_FAIL_CLOSED_VERIFIED') if divergence is None else 'FIRST_DIVERGENCE',
        first_divergence=divergence, amp=amp, device=device,
        evidence_scope='synthetic_actual_CORE90_trainer_not_real_source_benefit',
        events={key: len(value) for key, value in traces.items()},
        source_mechanism=results['audited']['activation']['source_mechanism_audit'],
        counters={key: value['activation']['counts'] for key, value in results.items()},
        training_context='actual_label_and_pseudo_phase_loss_graphs',
        epochs=4 if amp else 2,
        synthetic_initial_scale_override=initial_scale,
        amp_budget_reason='default_scaler_warmup_can_skip_first_three_updates' if amp else None,
        risk_model='student_after_one_adamw_update',
        test_only_thresholds='deliberately_impossible_to_verify_fail_closed',
        main_model_optimizer_ema_prototype_pseudo_rng_parity=divergence is None,
        formal_experiment=False, target_evaluation='OMITTED_SYNTHETIC_TEST')
    (root / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    assert divergence is None, divergence
    print(report['status'], flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--initial-scale', type=float, help='Synthetic fixture only; identical for both branches')
    args = parser.parse_args()
    run(args.output_root, device=args.device, amp=args.amp, initial_scale=args.initial_scale)
