"""Analytical source recovery, no-mutation, controller and resume contracts."""
from copy import deepcopy
import math
import random

import numpy as np
import pytest
import torch
from torch import nn

from cvsrffi.game_tracking.source_audit import (AuditConfig, SourceAuditor, classification_metrics,
                                               partition_groups, validate_groups)
from cvsrffi.game_tracking.gradient_audit import gradient_diagnostics
from cvsrffi.game_tracking.capability import (CapabilityConfig, evaluate_capability,
                                              cross_rx_margin, BranchCapabilityRegistry)
from cvsrffi.game_tracking.budget import BudgetConfig, ComputeBudget
from cvsrffi.game_tracking.controller import ControllerConfig, GameController
from cvsrffi.game_tracking.curriculum import CurriculumConfig, CapabilityCurriculum
from cvsrffi.game_tracking.coverage_audit import coverage_audit


def source_data(classes=2):
    labels = torch.arange(classes).repeat_interleave(8)
    features = torch.eye(classes)[labels] * 3.
    groups = ['fit-' + str(i) for i in range(len(labels))]
    monitor_groups = ['monitor-' + str(i) for i in range(len(labels))]
    return features, labels, groups, monitor_groups


def audit(head, x, y, groups, monitor, **kwargs):
    return SourceAuditor(AuditConfig(steps=25, lr=.15)).run(
        head, x, y, x.clone(), y.clone(), fit_groups=groups, monitor_groups=monitor,
        step=10, encoder_version=3, independence_verified=True, **kwargs)


def test_recovery_analytical_and_no_online_or_rng_mutation():
    x, y, fit, monitor = source_data()
    x.requires_grad_()
    head = nn.Linear(2, 2)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    head.train()
    head.weight.grad = torch.ones_like(head.weight)
    before, grad = deepcopy(head.state_dict()), head.weight.grad.clone()
    random.seed(35)
    np.random.seed(36)
    torch.manual_seed(37)
    rng, py, npr = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    result = audit(head, x, y, fit, monitor)
    m = result.metrics
    assert m['valid'] and m['recovered']['accuracy'] == 1.
    assert m['online']['ce'] == pytest.approx(math.log(2), abs=1e-6)
    assert m['G_lag'] == pytest.approx(m['ce_gap_raw'] / math.log(2), abs=1e-6)
    assert m['S_rx'] > .8 and m['probe_steps'] == 75
    assert head.training and x.grad is None
    assert all(torch.equal(head.state_dict()[k], value) for k, value in before.items())
    assert torch.equal(grad, head.weight.grad)
    assert torch.equal(rng, torch.get_rng_state()) and random.getstate() == py
    after = np.random.get_state()
    assert after[0] == npr[0] and np.array_equal(after[1], npr[1]) and after[2:] == npr[2:]


def test_group_split_and_overlap_and_unverified():
    fit, monitor = partition_groups(['a', 'a', 'b', 'b', 'c', 'c'], seed=3)
    assert not {['a', 'a', 'b', 'b', 'c', 'c'][i] for i in fit} & {['a', 'a', 'b', 'b', 'c', 'c'][i] for i in monitor}
    with pytest.raises(ValueError, match='overlap'):
        validate_groups(['record'], ['record'], 1, 1)
    x, y, f, m = source_data()
    result = SourceAuditor(AuditConfig(steps=1)).run(nn.Linear(2, 2), x, y, x, y,
             fit_groups=f, monitor_groups=m, step=0, encoder_version=0)
    assert not result.metrics['valid']
    assert result.metrics['reason'] == 'acquisition_independence_unverified'


def test_imbalanced_metrics_are_not_chance_accuracy():
    logits = torch.tensor([[2., 0.]]).repeat(10, 1)
    labels = torch.tensor([0] * 9 + [1])
    m = classification_metrics(logits, labels)
    assert m['accuracy'] == pytest.approx(.9)
    assert m['balanced_accuracy'] == .5
    assert m['majority_accuracy'] == .9
    assert m['prior_sampling_accuracy'] == pytest.approx(.82)
    assert m['prior_entropy'] == pytest.approx(-.9 * math.log(.9) - .1 * math.log(.1))


def test_native_domain_task_is_not_rx_and_single_class_is_invalid():
    x, y, f, m = source_data(classes=4)
    result = audit(nn.Linear(4, 4), x, y, f, m, label_kind='domain', rx_fit_labels=y % 2, rx_monitor_labels=y % 2)
    assert 'S_domain' in result.metrics and 'S_rx' not in result.metrics
    assert result.metrics['online']['effective_classes'] == 4
    assert result.metrics['independent']['rx']['linear']['effective_classes'] == 2
    result = audit(nn.Linear(4, 4), x, y * 0, f, m)
    assert not result.metrics['valid'] and result.metrics['reason'] == 'single_class'


def test_probe_cannot_fit_validation_or_missing_monitor_classes():
    x, y, f, m = source_data()
    with pytest.raises(ValueError, match='V/target forbidden'):
        audit(nn.Linear(2, 2), x, y, f, m, source_role='V')
    result = SourceAuditor().run(nn.Linear(2, 3), x, y, x, y + 1, fit_groups=f,
        monitor_groups=m, step=0, encoder_version=0, independence_verified=True)
    assert not result.metrics['valid']
    assert result.metrics['reason'] == 'fit_monitor_label_coverage_mismatch'


def test_gradient_conflict_and_zero_norm_invalid():
    m = gradient_diagnostics(torch.tensor([1., 0.]), torch.tensor([-2., 0.]), torch.tensor([0., 3.]),
                             g_nonadv=torch.tensor([1., 1.]))
    assert m['valid']
    assert m['id_online']['cosine'] == -1.
    assert m['id_online']['norm_ratio'] == 2.
    assert m['online_recovered']['cosine'] == 0.
    zero = gradient_diagnostics(torch.zeros(2), torch.ones(2), torch.ones(2))
    assert not zero['valid'] and zero['id_online']['cosine'] is None


def capability_data():
    tx = torch.tensor([0, 0, 1, 1] * 4)
    rx = torch.tensor([0, 1, 0, 1] * 4)
    z = torch.eye(2)[tx] * 3.
    return z, tx, rx, ['f' + str(i) for i in range(16)], ['m' + str(i) for i in range(16)]


def test_capability_cross_rx_margin_and_collapse_counterexample():
    z, tx, rx, f, m = capability_data()
    result = evaluate_capability(z, tx, rx, z, tx, rx, fit_groups=f, monitor_groups=m,
        config=CapabilityConfig(steps=25, lr=.15), monitor_aug=z, independence_verified=True)
    assert result['valid'] and result['identity'] == 1.
    assert result['margin'] == pytest.approx(1.)
    assert result['cross_rx']['coverage'] == 1. and result['consistency'] == pytest.approx(1.)
    collapsed = evaluate_capability(torch.ones_like(z), tx, rx, torch.ones_like(z), tx, rx,
        fit_groups=f, monitor_groups=m, config=CapabilityConfig(steps=1), monitor_aug=torch.ones_like(z),
        independence_verified=True)
    assert collapsed['consistency'] == pytest.approx(1.)
    assert collapsed['collapsed'] and not collapsed['valid']


def test_margin_missing_coverage_and_branch_interface():
    z, tx, rx, f, m = capability_data()
    only_tx_zero = tx == 0
    margin = cross_rx_margin(z[only_tx_zero], tx[only_tx_zero], rx[only_tx_zero], z, tx, rx)
    assert not margin['valid'] and margin['coverage'] == 0.
    registry = BranchCapabilityRegistry()
    assert registry.fusion_gate()['reason'] == 'NOT_APPLICABLE_NO_RESPONSE_FUSION'
    registry.register('response', dict(valid=True, identity=.9, margin=.2, collapsed=False),
                      identity_threshold=.8, margin_threshold=.1)
    assert registry.fusion_gate('response')['allowed']
    registry.register('response', dict(valid=True, identity=.4, margin=.2, collapsed=False),
                      identity_threshold=.8, margin_threshold=.1)
    assert not registry.fusion_gate('response')['allowed']


def signal(step, **values):
    return dict(valid=True, step=step, encoder_version=1, identity=.9, identity_valid=True,
                margin=.2, G_lag=.3, S_rx=.5, consistency=.95, **values)


def test_controller_confirmation_stale_budget_and_resume():
    c = GameController(ControllerConfig(cooldown_steps=2))
    budget = ComputeBudget(BudgetConfig(max_head_steps=2))
    assert c.decide(signal(0), step=0, encoder_version=1, budget=budget)['action'] == 'NORMAL'
    # Repeated evaluation of one audit is not a second independent observation.
    assert c.decide(signal(0), step=1, encoder_version=1)['action'] == 'NORMAL'
    assert c.decide(signal(2), step=2, encoder_version=1, budget=budget)['action'] == 'CATCHUP'
    assert budget.lifetime['head_steps'] == 0
    budget.commit(2, head_steps=2)
    assert c.decide(signal(3), step=3, encoder_version=1, budget=budget)['action'] == 'NORMAL'
    assert c.decide(signal(3), step=4, encoder_version=2)['hold_curriculum']
    restored = GameController(c.config)
    restored.load_state_dict(c.state_dict())
    assert restored.decide(signal(5), step=5, encoder_version=1) == c.decide(signal(5), step=5, encoder_version=1)


def test_controller_correct_and_hold_states_and_hysteresis():
    c = GameController(ControllerConfig(confirmation_windows=1, cooldown_steps=0))
    healthy = signal(0)
    healthy.update(G_lag=.02, direction_imbalance=.8, gradient_valid=True)
    assert c.decide(healthy, step=0, encoder_version=1)['action'] == 'CORRECT'
    healthy.update(step=1, direction_imbalance=.4)
    assert c.decide(healthy, step=1, encoder_version=1)['action'] == 'CORRECT'
    healthy.update(step=2, direction_imbalance=.2)
    assert c.decide(healthy, step=2, encoder_version=1)['action'] == 'NORMAL'
    healthy.update(step=3, identity=.1)
    assert c.decide(healthy, step=3, encoder_version=1)['action'] == 'HOLD_CURRICULUM'


def test_budget_actual_cost_window_cap_fraction_and_resume():
    budget = ComputeBudget(BudgetConfig(window_steps=5, max_head_steps=3, max_field_evaluations=2,
                                       max_audit_seconds=1., max_correction_fraction=.5))
    assert not budget.can_afford(0, corrections=1, base_steps=1)[0]
    budget.commit(0, base_steps=1)
    assert budget.can_afford(1, corrections=1, base_steps=1)[0]
    budget.commit(1, field_evaluations=1, base_steps=1, corrections=1, head_steps=3, audit_seconds=1.1)
    assert not budget.can_afford(2, head_steps=1)[0]
    assert not budget.can_afford(2, audit_seconds=.1)[0]
    # Actual audit time is recorded even when estimate was too small.
    assert budget.lifetime['audit_seconds'] == 1.1
    restored = ComputeBudget(budget.config)
    restored.load_state_dict(budget.state_dict())
    assert restored.usage(2) == budget.usage(2)
    assert restored.can_afford(6, head_steps=3)[0]
    assert restored.lifetime['head_steps'] == 3


def test_curriculum_no_flapping_forced_upgrade_or_repeated_evidence():
    c = CapabilityCurriculum(CurriculumConfig(confirmation_windows=2, cooldown_steps=3, max_increment=.2))
    good = signal(0)
    good['G_lag'] = .01
    assert not c.update(good, step=0, encoder_version=1)['changed']
    assert not c.update(good, step=1, encoder_version=1)['changed']
    good['step'] = 2
    changed = c.update(good, step=2, encoder_version=1)
    assert changed['changed'] and changed['level'] == .2 and changed['reset_optimistic']
    for step in range(3, 20):
        noisy = dict(good, step=step, identity=.6 if step % 2 else .8)
        assert not c.update(noisy, step=step, encoder_version=1)['changed']
    assert c.level == .2
    restored = CapabilityCurriculum(c.config)
    restored.load_state_dict(c.state_dict())
    assert restored.update(dict(good, step=20), step=20, encoder_version=1) == c.update(dict(good, step=20), step=20, encoder_version=1)
    assert not c.update(dict(good, step=21), step=21, encoder_version=2)['changed']


def test_coverage_forbids_hidden_tx_and_reports_association():
    with pytest.raises(ValueError, match='hidden TX'):
        coverage_audit([0, 1], tx=[0, 1], role='U_s')
    report = coverage_audit([0, 0, 1, 1], tx=[0, 0, 1, 1], view=['clean', 'clean', 'leo', 'leo'])
    assert report['tx_rx']['coverage'] == .5
    assert report['tx_rx']['mutual_information_nats'] == pytest.approx(math.log(2))
    assert report['view_rx']['mutual_information_nats'] == pytest.approx(math.log(2))
    assert report['day_rx'] == 'N/A' and report['snr_rx'] == 'N/A'
    assert coverage_audit([0, 1], role='U_s')['tx_rx'] == 'N/A'


@pytest.mark.parametrize('consistency', [-.4, .59, .79, float('nan'), float('inf'), None, 'missing'])
def test_curriculum_requires_measured_finite_enter_consistency(consistency):
    c = CapabilityCurriculum(CurriculumConfig(confirmation_windows=1, cooldown_steps=0))
    for step in range(3):
        good_identity = signal(step)
        good_identity['G_lag'] = .01
        if consistency == 'missing':
            good_identity.pop('consistency')
        else:
            good_identity['consistency'] = consistency
        result = c.update(good_identity, step=step, encoder_version=1)
        assert not result['changed'] and c.level == 0.


def test_curriculum_consistency_hysteresis_and_resume():
    c = CapabilityCurriculum(CurriculumConfig(confirmation_windows=2, cooldown_steps=10))
    c.last_change = 0
    for step in (1, 2):
        measured = dict(signal(step), G_lag=.01, consistency=.9)
        assert not c.update(measured, step=step, encoder_version=1)['changed']
    assert c.ready
    between = dict(signal(3), G_lag=.01, consistency=.7)
    assert not c.update(between, step=3, encoder_version=1)['changed'] and c.ready
    low = dict(signal(4), G_lag=.01, consistency=.5)
    assert not c.update(low, step=4, encoder_version=1)['changed'] and not c.ready
    assert c.streak == 0
    restored = CapabilityCurriculum(c.config)
    restored.load_state_dict(c.state_dict())
    assert restored.config.consistency_enter == .8 and restored.config.consistency_exit == .6
    for step in (10, 11):
        measured = dict(signal(step), G_lag=.01, consistency=.9)
        assert restored.update(measured, step=step, encoder_version=1) == c.update(measured, step=step, encoder_version=1)
    assert c.level == c.config.max_increment
