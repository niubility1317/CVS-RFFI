from dataclasses import replace

import pytest
import torch

from cvsrffi.muse_ssdg import (
    build_rc4_calibration,
    rc4_identity_losses,
    rc4_tail_transition_scale,
    route_fasttrust_rc4,
)


def _calibration_fixture():
    anchor_logits = torch.tensor(
        [
            [6.0, 0.0, -1.0],
            [5.0, 0.2, -1.0],
            [0.0, 5.0, -1.0],
            [0.0, 4.0, 0.2],
            [-1.0, 0.0, 5.0],
            [3.0, 2.8, 0.0],
            [2.4, 2.5, 0.0],
            [0.0, 2.6, 2.5],
            [2.2, 0.0, 2.3],
            [0.0, 2.1, 2.2],
        ],
        dtype=torch.float32,
    )
    ema1 = anchor_logits + torch.tensor([0.1, -0.1, 0.0])
    ema2 = anchor_logits + torch.tensor([-0.1, 0.1, 0.0])
    labels = torch.tensor([0, 0, 1, 1, 2, 0, 1, 1, 2, 2])
    domains = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    z_norm = torch.linspace(1.0, 2.0, steps=10)
    return anchor_logits, ema1, ema2, labels, domains, z_norm


def test_rc4_calibration_is_source_only_frozen_and_finite():
    anchor, ema1, ema2, labels, domains, z_norm = _calibration_fixture()
    package = build_rc4_calibration(
        anchor,
        ema1,
        ema2,
        labels,
        domains,
        z_norm,
        num_classes=3,
        num_domains=2,
        folds=2,
        min_stratum_samples=2,
    )

    assert 0.25 <= package.temperature <= 4.0
    assert package.feature_mean.shape == (7,)
    assert package.feature_scale.shape == (7,)
    assert package.correctness_weight.shape == (1 + 7 + 3 + 2,)
    assert torch.isfinite(package.correctness_weight).all()
    assert 0.0 < package.aps_global <= 1.0
    assert package.calibration_rows == labels.numel()


def test_rc4_route_partitions_h_p_n_r_without_any_fill():
    anchor, ema1, ema2, labels, domains, z_norm = _calibration_fixture()
    package = build_rc4_calibration(
        anchor,
        ema1,
        ema2,
        labels,
        domains,
        z_norm,
        num_classes=3,
        num_domains=2,
        folds=2,
        min_stratum_samples=2,
        hard_precision_target=0.80,
        partial_coverage_target=0.80,
        negative_false_exclusion_target=0.20,
    )
    route = route_fasttrust_rc4(
        anchor,
        ema1,
        ema2,
        domains=domains,
        receivers=domains,
        z_norm=z_norm,
        calibration=package,
        hard_max_fraction=1.0,
        candidate_max_classes=3,
        enable_hard=True,
        enable_partial=True,
        enable_negative=True,
        class_receiver_cap=False,
    )

    union = route.hard | route.partial | route.negative | route.representation
    assert union.all()
    assert not (route.hard & route.partial).any()
    assert not (route.hard & route.negative).any()
    assert not (route.partial & route.negative).any()
    assert torch.equal(route.representation, ~(route.hard | route.partial | route.negative))
    assert torch.equal(route.excluded_mask, ~route.candidate_mask)
    assert torch.all(route.weights >= 0)
    assert torch.all(route.weights <= 4.0)


def test_rc4_class_receiver_balance_preserves_amp_dtype():
    anchor, ema1, ema2, labels, domains, z_norm = _calibration_fixture()
    package = build_rc4_calibration(
        anchor,
        ema1,
        ema2,
        labels,
        domains,
        z_norm,
        num_classes=3,
        num_domains=2,
        folds=2,
        min_stratum_samples=2,
        hard_precision_target=0.80,
        partial_coverage_target=0.80,
        negative_false_exclusion_target=0.20,
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        route = route_fasttrust_rc4(
            anchor,
            ema1,
            ema2,
            domains=domains,
            receivers=domains,
            z_norm=z_norm,
            calibration=package,
            hard_max_fraction=1.0,
            candidate_max_classes=3,
            class_receiver_cap=True,
        )

    assert route.hard.any()
    assert torch.isfinite(route.weights).all()


def test_rc4_losses_use_full_u_denominator_and_do_not_train_teacher():
    student = torch.tensor(
        [[3.0, 0.0, -1.0], [0.0, 2.0, 1.0], [0.0, 1.0, 2.0], [1.0, 1.0, 1.0]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[0.9, 0.08, 0.02], [0.1, 0.6, 0.3], [0.1, 0.4, 0.5], [0.34, 0.33, 0.33]],
        requires_grad=True,
    )
    candidate = torch.tensor(
        [[True, False, False], [False, True, True], [False, True, True], [True, True, True]]
    )
    losses = rc4_identity_losses(
        student,
        teacher,
        pseudo=torch.tensor([0, 1, 2, 0]),
        candidate_mask=candidate,
        hard_mask=torch.tensor([True, False, False, False]),
        partial_mask=torch.tensor([False, True, False, False]),
        negative_mask=torch.tensor([False, False, True, False]),
        weights=torch.ones(4),
        full_unlabeled_batch_size=4,
    )
    losses["total"].backward()

    expected_hard = torch.nn.functional.cross_entropy(student[:1], torch.tensor([0])) / 4.0
    assert torch.allclose(losses["hard"], expected_hard)
    assert student.grad is not None and student.grad.abs().sum().item() > 0
    assert teacher.grad is None


def test_rc4_empty_identity_routes_are_graph_safe_zero():
    logits = torch.randn(5, 3, requires_grad=True)
    losses = rc4_identity_losses(
        logits,
        torch.softmax(torch.randn(5, 3), dim=-1),
        pseudo=torch.zeros(5, dtype=torch.long),
        candidate_mask=torch.ones(5, 3, dtype=torch.bool),
        hard_mask=torch.zeros(5, dtype=torch.bool),
        partial_mask=torch.zeros(5, dtype=torch.bool),
        negative_mask=torch.zeros(5, dtype=torch.bool),
        weights=torch.ones(5),
        full_unlabeled_batch_size=5,
    )
    losses["total"].backward()

    assert losses["total"].item() == 0.0
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() == 0.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rc4_set_losses_are_finite_for_saturated_logits(dtype):
    student = torch.tensor(
        [[100.0, 0.0, -100.0], [1000.0, -1000.0, -1000.0]],
        dtype=dtype,
        requires_grad=True,
    )
    candidate = torch.tensor([[False, True, True], [True, True, False]])
    losses = rc4_identity_losses(
        student,
        torch.full_like(student, 1.0 / 3.0),
        pseudo=torch.tensor([1, 0]),
        candidate_mask=candidate,
        hard_mask=torch.tensor([False, False]),
        partial_mask=torch.tensor([True, False]),
        negative_mask=torch.tensor([False, True]),
        weights=torch.ones(2),
        full_unlabeled_batch_size=2,
    )

    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_rc4_partial_set_mass_has_hand_checked_boundaries():
    singleton_logits = torch.tensor([[3.0, 1.0, -2.0]], requires_grad=True)
    singleton = rc4_identity_losses(
        singleton_logits,
        torch.tensor([[0.7, 0.2, 0.1]]),
        pseudo=torch.tensor([0]),
        candidate_mask=torch.tensor([[True, False, False]]),
        hard_mask=torch.tensor([False]),
        partial_mask=torch.tensor([True]),
        negative_mask=torch.tensor([False]),
        weights=torch.ones(1),
        full_unlabeled_batch_size=1,
        enable_partial_conditional=False,
    )
    expected = torch.nn.functional.cross_entropy(singleton_logits, torch.tensor([0]))
    assert torch.allclose(singleton["partial_set"], expected)

    all_classes = rc4_identity_losses(
        singleton_logits,
        torch.tensor([[0.7, 0.2, 0.1]]),
        pseudo=torch.tensor([0]),
        candidate_mask=torch.tensor([[True, True, True]]),
        hard_mask=torch.tensor([False]),
        partial_mask=torch.tensor([True]),
        negative_mask=torch.tensor([False]),
        weights=torch.ones(1),
        full_unlabeled_batch_size=1,
        enable_partial_conditional=False,
    )
    assert all_classes["partial_set"].item() == pytest.approx(0.0, abs=1e-7)


def test_rc4_selected_set_route_rejects_empty_or_all_excluded_candidate():
    logits = torch.zeros(1, 3, requires_grad=True)
    with pytest.raises(ValueError, match="non-empty allowed set"):
        rc4_identity_losses(
            logits,
            torch.full_like(logits, 1.0 / 3.0),
            pseudo=torch.tensor([0]),
            candidate_mask=torch.tensor([[False, False, False]]),
            hard_mask=torch.tensor([False]),
            partial_mask=torch.tensor([True]),
            negative_mask=torch.tensor([False]),
            weights=torch.ones(1),
            full_unlabeled_batch_size=1,
        )


def test_rc4_partial_effective_weight_budget_caps_quality_mass():
    anchor, ema1, ema2, labels, domains, z_norm = _calibration_fixture()
    package = build_rc4_calibration(
        anchor, ema1, ema2, labels, domains, z_norm,
        num_classes=3, num_domains=2, folds=2, min_stratum_samples=2,
        hard_precision_target=0.80, partial_coverage_target=0.80,
        negative_false_exclusion_target=0.20,
    )
    partial_weight = torch.zeros_like(package.partial_safety_weight)
    partial_weight[0] = 8.0
    package = replace(
        package,
        partial_ready=True,
        aps_global=0.80,
        aps_by_class=torch.full_like(package.aps_by_class, 0.80),
        aps_by_domain=torch.full_like(package.aps_by_domain, 0.80),
        partial_safety_weight=partial_weight,
    )
    route = route_fasttrust_rc4(
        anchor, ema1, ema2, domains=domains, receivers=domains, z_norm=z_norm,
        calibration=package, hard_max_fraction=0.0, candidate_max_classes=3,
        partial_min_risk=0.0, partial_effective_budget=0.10,
        enable_hard=False, enable_partial=True, enable_negative=False,
        class_receiver_cap=False,
    )

    assert route.partial.any()
    assert route.weights[route.partial].sum().item() <= 1.0 + 1e-6
    assert torch.allclose(route.p_correct, route.risk)
    assert torch.isfinite(route.p_set_safe).all()
    assert torch.isfinite(route.p_exclusion_safe).all()


def test_rc4_tail_transition_restarts_all_u_weight_without_changing_core90_schedule():
    assert rc4_tail_transition_scale(90, start_epoch=91, ramp_epochs=20, floor=0.25) == 1.0
    assert rc4_tail_transition_scale(91, start_epoch=91, ramp_epochs=20, floor=0.25) == 0.25
    assert rc4_tail_transition_scale(101, start_epoch=91, ramp_epochs=20, floor=0.25) == pytest.approx(0.625)
    assert rc4_tail_transition_scale(111, start_epoch=91, ramp_epochs=20, floor=0.25) == 1.0
