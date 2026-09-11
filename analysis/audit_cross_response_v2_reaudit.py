"""Bounded CPU re-audit probes; synthetic fixtures are not source qualification."""
import ast
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'tests')]
import torch
from scripts.register_cross_response_v2 import configuration
from scripts.core90_cross_response_matrix import resolve_variant
from cvsrffi.cross_response.config import validate_runtime_config
from cvsrffi.cross_response.decision_calibration import calibrated_decision_tensors
from test_cross_response_v2_integration import runtime
from test_cross_response_v2_geometry_gate import make_gate
from test_cross_response_v2_scheduler import evidence_config


def run():
    config = configuration()
    rows = {}
    for variant in config['variants']:
        c = resolve_variant(config, variant)
        try:
            validate_runtime_config(c)
            validation = 'ACCEPTED_CONFIG_ONLY'
        except ValueError as exc:
            validation = str(exc)
        rows[variant] = dict(validation=validation, **{k: c.get(k) for k in (
            'response_enabled', 'decision_enabled', 'identity_interaction_enabled',
            'head_only', 'permanent_detach', 'lambda_resp', 'lambda_dec', 'lambda_cross',
            'response_routing', 'source_audit_enabled', 'decision_mode', 'gain_strategy',
            'mechanism_gate', 'direction_shrinkage', 'diagnostic_interval')})
    calls = []
    for path in (ROOT / 'code').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == 'observe_source_mechanism':
                    calls.append(dict(file=path.relative_to(ROOT).as_posix(), line=node.lineno))
    result = dict(scope='SYNTHETIC_CPU_AUDIT_NO_CHECKPOINT_NO_TARGET', variants=rows,
                  production_observe_source_mechanism_calls=calls)
    with tempfile.TemporaryDirectory(prefix='v2_reaudit_') as temp:
        folder = Path(temp)
        gate_config = make_gate(1).config
        gate_config['terminal_identity_module'] = 'time_fuse'
        _, rt, _ = runtime(folder, 'U2', mechanism_gate=gate_config)
        rt.gate.opened = True
        result['legacy_gate_vs_new_gate'] = dict(legacy_open=rt.gate.opened,
            new_gate_configured=rt.mechanism_gate is not None,
            new_gate_last_result=rt.mechanism_gate.last_result, joint_open=rt.joint_open)
        _, feedback, ctx = runtime(folder, 'U5_reliable', evidence_config=evidence_config())
        next(iter(ctx['train_loader']))
        result['feedback_with_closed_joint_gate'] = dict(joint_open=feedback.joint_open,
            mechanism_gate=feedback.mechanism_gate,
            gain_strategy=feedback.sampler.scheduler.gain_strategy,
            probability_audit_rows=len(feedback.sampler.scheduler.last_probability_audit))
        _, base, _ = runtime(folder, 'U3')
        calibration = folder / 'calibration.pt'
        noise = dict(source_role='L_s', source_frozen=True, num_classes=4,
            conditional={}, pair={}, global_noise=dict(delta=.1, groups=8))
        torch.save(dict(source_contract=base.source_contract, noise=noise,
            validation_evidence=dict(source_role='V', target_used=False,
                metrics=dict(synthetic_risk=1.))), calibration)
        _, pairs, _ = runtime(folder, 'U3_delta_pairs', decision_calibration=str(calibration))
        _, weights, _ = calibrated_decision_tensors(noise,
            pairs.decision_calibration.get('competitors'), torch.arange(4), [0]*4, ['clean']*4)
        result['delta_pairs_without_competitors'] = dict(runtime_accepted=True,
            selected_mode=pairs.config['decision_mode'],
            competitors_present='competitors' in pairs.decision_calibration,
            available_competitors_per_record=weights.sum(1).tolist())
    return result


if __name__ == '__main__':
    result = run()
    destination = Path(sys.argv[1])
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'variants'}, ensure_ascii=False, indent=2))
