import torch

from cvsrffi.muse_ssdg import (
    build_rc4_calibration,
    rc4_identity_losses,
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
