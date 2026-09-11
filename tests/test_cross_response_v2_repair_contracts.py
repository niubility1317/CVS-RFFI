"""Regression tests for the concrete second-review findings."""
import copy
import pytest
import torch

from test_cross_response_v2_integration import runtime, update
from test_cross_response_v2_scheduler import evidence_config, evidence_scheduler, evidence_inputs
from test_cross_response_v2_geometry_gate import make_gate, evidence
from scripts.register_cross_response_v2 import configuration
from scripts.core90_cross_response_matrix import resolve_variant
from cvsrffi.cross_response.config import validate_runtime_config


def test_delta_pairs_missing_competitors_rejected_at_initialization(tmp_path):
    _, base, _ = runtime(tmp_path, 'U3')
    artifact = tmp_path / 'noise_only.pt'
    noise = dict(source_role='L_s', source_frozen=True, num_classes=4,
        conditional={}, pair={}, global_noise=dict(delta=.1, groups=8))
    torch.save(dict(source_contract=base.source_contract, noise=noise,
        validation_evidence=dict(source_role='V', target_used=False,
            metrics=dict(synthetic_risk=1.))), artifact)
    with pytest.raises(ValueError, match='competitor'):
        runtime(tmp_path, 'U3_delta_pairs', decision_calibration=str(artifact))
    # The very same noise-only artifact remains valid in delta-only mode.
    _, delta, _ = runtime(tmp_path, 'U3_delta', decision_calibration=str(artifact))
    assert delta.decision_calibration['noise']['source_frozen']


def test_feedback_rejects_explicitly_unauthorized_gate():
    cfg = evidence_config()
    cfg['source_evidence']['gate_result'].update(authorized=False, stable_observations=1)
    with pytest.raises(ValueError, match='authoriz'):
        evidence_scheduler(cfg)


@pytest.mark.parametrize('variant', ['U1', 'head_only', 'permanent_detach'])
def test_reliable_feedback_requires_current_joint_objective(variant):
    c = resolve_variant(configuration(), variant)
    c.update(scheduler_mode='feedback', gain_strategy='reliable_evidence', evidence_config=evidence_config())
    with pytest.raises(ValueError, match='joint'):
        validate_runtime_config(c)


def test_nonfinite_response_forces_diagnostic_on_unscheduled_step(tmp_path):
    model, rt, ctx = runtime(tmp_path, 'head_only')
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    update(model, rt, ctx, opt)
    for p in rt.auxiliary['predictor'].parameters():
        with torch.no_grad():
            p.fill_(float('nan'))
    x, y, d, meta = next(iter(ctx['train_loader']))
    plan = rt.begin_batch((d, meta))
    assert not rt.audit_due
    out = model(x)
    assert torch.isfinite(out['z_id']).all()
    rt.losses(model, out, y, plan)
    assert rt.audit_due
    assert rt.counts['forced_audit_steps'] == 1
    assert any(row['kind'] == 'nonfinite_diagnostic' for row in rt.detailed_audits)


def test_necessity_requires_rx_only_and_head_only_comparisons():
    gate = make_gate(1)
    row = evidence(0)
    row['necessity'].update(rx_only_error=.9, head_only_error=2.)
    assert not gate.observe(row)['authorized']
    row = evidence(1)
    row['necessity'].update(rx_only_error=2., head_only_error=.9)
    assert not gate.observe(row)['authorized']


def test_reliable_gain_binds_live_qualification_and_closes_to_uniform():
    cfg = evidence_config()
    cfg['source_evidence'].update(source_contract={'source': 'fixture'},
                                 joint_objective={'scope': 'cls_head', 'response_enabled': True})
    s = evidence_scheduler(cfg)
    live = dict(authorized=True, scope='cls_head')
    s.bind_qualification(source_contract={'source': 'fixture'},
        joint_objective=cfg['source_evidence']['joint_objective'], provider=lambda: live)
    candidates, roles, physical = evidence_inputs()
    physical[candidates[0].key]['new_physical_estimate'] = 0
    before = s.probabilities(candidates, 0, 128, role_plans=roles, candidate_evidence=physical)
    assert before[0] != before[1]
    live['authorized'] = False
    after = s.probabilities(candidates, 0, 128, role_plans=roles, candidate_evidence=physical)
    assert after == [.5, .5]
    assert all(row['qualification'] == 'CLOSED_UNIFORM' for row in s.last_probability_audit)
    with pytest.raises(ValueError, match='contract'):
        s.bind_qualification(source_contract={'source': 'different'},
            joint_objective=cfg['source_evidence']['joint_objective'], provider=lambda: live)
