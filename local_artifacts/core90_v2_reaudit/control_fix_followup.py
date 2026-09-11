"""Independent bounded CPU verification of A01/A02 fixes and config boundary."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from cvsrffi.game_tracking.config import parse_args
from cvsrffi.game_tracking.controller import ControllerConfig, GameController, GameControllerV2
from cvsrffi.game_tracking.runtime_control import calibrate_game_v2
from cvsrffi.game_tracking.audit_evidence import make_evidence_v2
from repro_high_gap_correction import run as actual_probe

cap = dict(schema='game_capability_v2', valid=True, identity_valid=True, collapsed=False,
           identity=.9, margin=.3, next_identity=.9, next_margin=.2, next_worst_tx=.8)

def evidence(status, gap, step):
    return make_evidence_v2(observation_id=str(step), step=step, encoder_version=step,
        data_valid=True, coverage_valid=True,
        lag=dict(status=status, quality_pass=True, control_ready=True, gap_normalized=gap, readability=.9),
        gradient=dict(valid=True, representative=True, scope='full_current_training_objective', direction_imbalance=.8),
        capability=dict(cap, step=step, encoder_version=step))

def confirmed(controller, status, gap):
    return [controller.decide(evidence(status, gap, step), step=step, encoder_version=step)
            for step in (1000, 1250)]

results = {}
for status, gap, expected in [('RELIABLE_HIGH_GAP', .2, 'NORMAL'),
                              ('RELIABLE_LOW_GAP', 2., 'NORMAL'),
                              ('RELIABLE_LOW_GAP', .2, 'CORRECT'),
                              ('RELIABLE_HIGH_GAP', 2., 'CATCHUP')]:
    c = GameControllerV2(ControllerConfig(lag_enter=1., lag_exit=.5))
    actions = confirmed(c, status, gap)
    assert actions[-1]['action'] == expected, actions
    results[f'{status}_{gap}'] = actions

args = parse_args(['--output_dir', 'unused', '--game_control', 'correction', '--game_max_extra_head', '0'])
c = calibrate_game_v2([evidence('RELIABLE_HIGH_GAP', 1., i) for i in range(3)], [cap]*3, args)
assert c is not None and c.config.catchup_steps == 0
results['zero_head_correction'] = confirmed(c, 'RELIABLE_LOW_GAP', .2)
assert results['zero_head_correction'][-1]['action'] == 'CORRECT'
c = GameControllerV2(ControllerConfig(catchup_steps=0))
results['zero_head_v2_high'] = confirmed(c, 'RELIABLE_HIGH_GAP', 2.)
assert results['zero_head_v2_high'][-1]['action'] == 'NORMAL'
c = GameController(ControllerConfig(catchup_steps=0))
results['zero_head_v1_high'] = [c.decide(dict(valid=True, step=s, encoder_version=s, G_lag=2.,
    identity=.9, margin=.3, S_rx=.9, direction_imbalance=.8, gradient_valid=True), step=s, encoder_version=s)
    for s in (1000, 1250)]
assert results['zero_head_v1_high'][-1]['action'] == 'NORMAL'

rejections = {}
for name, flags in [('control', ['--game_control', 'correction']),
                    ('curriculum', ['--game_curriculum', 'capability']),
                    ('response', ['--game_response_tracking']),
                    ('jacobian', ['--game_jacobian_interval', '10'])]:
    try:
        parse_args(['--output_dir', 'unused', '--game_no_audit'] + flags)
    except ValueError as exc:
        assert 'requires source audits' in str(exc)
        rejections[name] = str(exc)
    else:
        raise AssertionError(name)
try:
    parse_args(['--output_dir', 'unused', '--game_max_extra_head', '-1'])
except ValueError as exc:
    rejections['negative_head'] = str(exc)
else:
    raise AssertionError('negative head accepted')
args = parse_args(['--output_dir', 'unused', '--game_no_audit', '--game_solver', 'head_lookahead'])
assert args.game_no_audit and args.game_control == 'off'
results['config'] = dict(rejections=rejections, b8_no_audit_accepted=True)
results['actual_probe'] = actual_probe()
assert not results['actual_probe']['defect_reproduced']
assert results['actual_probe']['evidence_lag']['status'] == 'RELIABLE_HIGH_GAP'
results['status'] = 'VERIFIED'
out = Path(__file__).with_suffix('.json')
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(dict(status=results['status'], output=str(out), cases=12)))
