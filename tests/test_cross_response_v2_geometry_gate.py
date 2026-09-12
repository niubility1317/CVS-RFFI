import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from cvsrffi.cross_response.tensor_ops import (unit_identity_geometry,
    normalized_identity_interaction_loss, decomposed_response_losses, double_center)
from cvsrffi.cross_response.gate_v2 import SourceMechanismGate


def test_unit_records_before_mean_scale_invariance_and_source_class_metrics():
    torch.manual_seed(12)
    z = torch.randn(3, 4, 2, 8, dtype=torch.float64)
    z[:, :, 0] *= 10
    labels = torch.arange(3)[:, None, None].expand(3, 4, 2)
    logits = torch.zeros(3, 4, 2, 3)
    logits[..., 0] = 1
    a = unit_identity_geometry(z, min_norm=1e-9, labels=labels, logits=logits)
    b = unit_identity_geometry(z * 17, min_norm=1e-9, labels=labels, logits=logits)
    expected = (z / z.norm(dim=-1, keepdim=True)).mean(2)
    assert torch.allclose(a['normalized_grid'], expected)
    assert a['unit_interaction'] == pytest.approx(b['unit_interaction'])
    assert a['raw_relative_interaction'] == pytest.approx(b['raw_relative_interaction'])
    assert b['raw_interaction'] == pytest.approx(a['raw_interaction'] * 17**2)
    assert a['raw_interaction_per_160_dimensions'] == pytest.approx(a['raw_interaction'] * 20)
    assert a['inter_class_angles_deg'] == pytest.approx(b['inter_class_angles_deg'])
    assert a['error_class_pairs'] == {(1, 0): 8, (2, 0): 8}
    assert len(a['intra_class_tail']) == 3
    for label in a['intra_class_tail']:
        assert a['intra_class_tail'][label]['angle_p95_deg'] == pytest.approx(b['intra_class_tail'][label]['angle_p95_deg'])
    wrong = z.mean(2) / z.mean(2).norm(dim=-1, keepdim=True)
    assert not torch.allclose(expected, wrong)


@pytest.mark.parametrize('tiny', [0., 1e-20])
def test_invalid_norm_is_unavailable_not_an_epsilon_value(tiny):
    z = torch.ones(2, 2, 2, 4, dtype=torch.float64, requires_grad=True)
    z = z.clone()
    z[0, 0, 0] = tiny
    loss, audit = normalized_identity_interaction_loss(z, min_norm=1e-12)
    assert loss is None and not audit['available']
    assert audit['unit_interaction'] is None and audit['normalized_grid'] is None
    assert audit['invalid_records'] == 1
    assert not audit['valid_record_mask'][0, 0, 0]


def test_normalized_candidate_gradient_finite_and_scale_behavior():
    torch.manual_seed(3)
    z = torch.randn(2, 3, 2, 5, dtype=torch.float64, requires_grad=True)
    loss, _ = normalized_identity_interaction_loss(z, min_norm=1e-12)
    grad, = torch.autograd.grad(loss, z)
    scaled = (z.detach() * 4).requires_grad_()
    other, _ = normalized_identity_interaction_loss(scaled, min_norm=1e-12)
    other_grad, = torch.autograd.grad(other, scaled)
    assert torch.allclose(loss, other)
    assert torch.allclose(grad, other_grad * 4)
    assert torch.isfinite(grad).all()


def test_reduced_precision_inputs_use_finite_float_statistics():
    z = torch.randn(2, 2, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    loss, audit = normalized_identity_interaction_loss(z, min_norm=1e-8)
    assert loss.dtype == torch.float32 and audit['normalized_grid'].dtype == torch.float32
    assert torch.isfinite(torch.autograd.grad(loss, z)[0]).all()
    prediction = torch.randn(2, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    losses, audit = decomposed_response_losses(prediction, torch.zeros_like(prediction), alpha=.2, beta=.3,
                                               interaction_reliability=.5, weight_bounds=(0., 1.))
    assert losses['predictor_full'].dtype == torch.float32
    assert float(audit['energy_identity_error']) < 1e-6


def test_error_decomposition_reconstruction_energy_and_route_gradients():
    torch.manual_seed(6)
    prediction = torch.randn(3, 4, 7, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(prediction, requires_grad=True)
    reliability = torch.tensor(.6, dtype=torch.float64, requires_grad=True)
    losses, audit = decomposed_response_losses(prediction, target, alpha=.3, beta=.7,
                                               interaction_reliability=reliability, weight_bounds=(0., 1.))
    assert float(audit['reconstruction_max_error']) < 1e-14
    assert float(audit['energy_identity_error']) < 1e-14
    assert torch.allclose(sum(audit['error_components'].values()), prediction-target)
    assert set(losses) == {'predictor_full', 'identity', 'domain'}
    gradient = torch.autograd.grad(losses['identity'], (prediction, target, reliability),
                                   allow_unused=True, retain_graph=True)
    assert gradient[1] is None and gradient[2] is None
    # The identity route excludes grand/RX components, the domain route excludes grand/TX.
    assert torch.allclose(gradient[0].mean(0), torch.zeros_like(gradient[0].mean(0)), atol=1e-15)
    domain_gradient, = torch.autograd.grad(losses['domain'], prediction)
    assert torch.allclose(domain_gradient.mean(1), torch.zeros_like(domain_gradient.mean(1)), atol=1e-15)
    _, clamped = decomposed_response_losses(prediction, target, alpha=1, beta=1,
                                            interaction_reliability=10., weight_bounds=(.1, .8))
    assert float(clamped['interaction_reliability']) == .8


def make_gate(stability=2):
    return SourceMechanismGate(thresholds=dict(capability_min_improvement=.1, necessity_min_gap=.1,
        update_min_improvement=.01, max_hard_group_degradation=.05, max_leo_degradation=.05),
        stable_observations=stability, min_samples=2, source_freeze_id='SYNTHETIC_TEST_ONLY',
        source_frozen=True, terminal_identity_module='identity_last_projection')


def evidence(index, scope='cls_head'):
    group = dict(sample_count=2, base_risk=1., joint_risk=.99)
    return dict(observation_id=str(index), scope=scope, source_freeze_id='SYNTHETIC_TEST_ONLY',
        source_role='source_validation', target_used=False,
        capability=dict(sample_count=4, baseline_error=2., full_error=1.),
        necessity=dict(sample_count=4, shuffled_tx_error=2., rx_only_error=2., head_only_error=2., full_error=1.),
        update_value=dict(sample_count=4, base_risk=1., joint_risk=.95,
            training_physical_ids=['t1', 't2'], query_physical_ids=['q1', 'q2', 'q3', 'q4'],
            hard_groups={'weak_rx': copy.deepcopy(group)}, leo_groups={'leo_weak': copy.deepcopy(group)}))


def test_gate_three_conditions_no_latch_and_fresh_extension_scope():
    gate = make_gate()
    assert not gate.observe(evidence(0))['authorized']
    assert gate.observe(evidence(1))['authorized']
    module = 'identity_last_projection'
    # Existing base-head stability is not inherited by the newly requested module.
    assert not gate.request_extension(module, evidence(2, module))['authorized']
    assert gate.request_extension(module, evidence(3, module))['authorized']
    failed = evidence(4, module)
    failed['necessity']['shuffled_tx_error'] = .8
    result = gate.request_extension(module, failed)
    assert not result['authorized'] and result['failed_conditions'] == ['necessity']
    with pytest.raises(ValueError, match='fresh'):
        gate.request_extension(module, evidence(3, module))
    with pytest.raises(ValueError, match='one registered'):
        gate.request_extension('shared_backbone', evidence(5))


@pytest.mark.parametrize('failure', ['capability', 'necessity', 'update_value', 'hard_groups', 'leo_groups'])
def test_each_scientific_condition_blocks_opening(failure):
    gate = make_gate(1)
    data = evidence(0)
    if failure == 'capability':
        data[failure]['baseline_error'] = .5
    elif failure == 'necessity':
        data[failure]['shuffled_tx_error'] = .5
    elif failure == 'update_value':
        data[failure]['joint_risk'] = 1.1
    else:
        next(iter(data['update_value'][failure].values()))['joint_risk'] = 2.
    result = gate.observe(data)
    assert not result['authorized'] and not result['passed']


@pytest.mark.parametrize('failure', ['target', 'overlap', 'missing_leo', 'no_scope', 'no_evidence'])
def test_invalid_evidence_is_rejected(failure):
    data = evidence(0)
    if failure == 'target':
        data['target_used'] = True
    elif failure == 'overlap':
        data['update_value']['query_physical_ids'][0] = 't1'
    elif failure == 'missing_leo':
        del data['update_value']['leo_groups']
    elif failure == 'no_scope':
        del data['scope']
    else:
        data = None
    gate = make_gate(1)
    assert gate.observe(evidence('previous'))['authorized']
    with pytest.raises(ValueError):
        gate.observe(data)
    assert not gate.last_result['authorized'] and gate.streak == 0


def test_gate_resume_and_explicit_frozen_configuration():
    gate = make_gate()
    gate.observe(evidence(0))
    restored = make_gate()
    restored.load_state_dict(gate.state_dict())
    assert restored.observe(evidence(1)) == gate.observe(evidence(1))
    with pytest.raises(ValueError, match='configuration'):
        make_gate(1).load_state_dict(gate.state_dict())
    gate.config['thresholds']['necessity_min_gap'] = 0.
    with pytest.raises(ValueError, match='configuration changed'):
        gate.observe(evidence(2))
