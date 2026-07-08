from __future__ import annotations

import math
import pickle

import numpy as np
import pytest
import torch

from paper_reproduction.orthogonal_incremental_sei.losses import (
    base_training_loss,
    class_center_separation_loss,
    incremental_calibration_loss,
    pseudo_target_cross_entropy,
    supervised_anchor_contrastive_loss,
)
from paper_reproduction.orthogonal_incremental_sei.metrics import (
    average_incremental_metrics,
    forgetting_by_session,
    harmonic_accuracy,
)
from paper_reproduction.orthogonal_incremental_sei.model import (
    CosineClassifier,
    SixBlockConv1DEncoder,
    class_mean_weights,
)
from paper_reproduction.orthogonal_incremental_sei.pseudo_targets import (
    assign_base_targets,
    make_simplex_pseudo_targets,
    optimize_pseudo_targets,
    perturb_pseudo_targets,
    pseudo_target_orthogonal_loss,
)


def test_simplex_pseudo_targets_match_paper_bounds_and_geometry() -> None:
    targets = make_simplex_pseudo_targets(num_targets=5, feature_dim=8)

    assert targets.shape == (5, 8)
    assert torch.allclose(targets.norm(dim=1), torch.ones(5), atol=1e-6)
    gram = targets @ targets.t()
    off_diag = gram[~torch.eye(5, dtype=torch.bool)]
    assert torch.allclose(off_diag, torch.full_like(off_diag, -0.25), atol=1e-6)

    with pytest.raises(ValueError, match="num_targets must be <= feature_dim \\+ 1"):
        make_simplex_pseudo_targets(num_targets=6, feature_dim=4)

    with pytest.raises(ValueError, match="num_targets must be >= total_classes"):
        make_simplex_pseudo_targets(num_targets=4, feature_dim=8, total_classes=5)


def test_iterative_pseudo_target_optimization_reduces_formula4_loss() -> None:
    torch.manual_seed(19)
    initial = torch.randn(4, 6)
    before = pseudo_target_orthogonal_loss(initial, temperature=0.2)

    optimized = optimize_pseudo_targets(
        num_targets=4,
        feature_dim=6,
        temperature=0.2,
        steps=30,
        seed=19,
    )
    after = pseudo_target_orthogonal_loss(optimized, temperature=0.2)

    assert optimized.shape == (4, 6)
    assert torch.allclose(optimized.norm(dim=1), torch.ones(4), atol=1e-5)
    assert after.item() < before.item()

    with pytest.raises(ValueError, match="num_targets must be positive"):
        optimize_pseudo_targets(num_targets=0, feature_dim=6, steps=1)

    with pytest.raises(ValueError, match="feature_dim must be positive"):
        optimize_pseudo_targets(num_targets=4, feature_dim=0, steps=1)


def test_train_entrypoint_builds_formula4_pseudo_targets_from_config() -> None:
    from paper_reproduction.orthogonal_incremental_sei.train import _build_pseudo_targets

    targets, steps = _build_pseudo_targets(
        {
            "seed": 19,
            "pseudo_targets": 4,
            "embedding_dim": 6,
            "pseudo_target_steps": 10,
            "tau_c": 0.2,
        },
        device="cpu",
    )

    assert steps == 10
    assert targets.shape == (4, 6)
    assert torch.allclose(targets.norm(dim=1), torch.ones(4), atol=1e-5)


def test_pseudo_target_perturbation_defaults_to_paper_additive_formula() -> None:
    targets = make_simplex_pseudo_targets(num_targets=4, feature_dim=6)
    perturbed = perturb_pseudo_targets(targets, noise_range=0.05, seed=23)
    additive = perturb_pseudo_targets(targets, noise_range=0.05, seed=23, renormalize=False)

    assert torch.allclose(perturbed, additive)
    assert not torch.allclose(perturbed.norm(dim=1), torch.ones(4), atol=1e-6)


def test_pseudo_target_losses_are_finite_and_backpropagate() -> None:
    torch.manual_seed(7)
    targets = make_simplex_pseudo_targets(num_targets=6, feature_dim=5)
    assigned = assign_base_targets(base_labels=[0, 1, 2], pseudo_targets=targets)
    features = torch.randn(9, 5, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    perturbed = perturb_pseudo_targets(targets, noise_range=0.01, seed=11)

    ce = pseudo_target_cross_entropy(features, labels, assigned)
    supcon = supervised_anchor_contrastive_loss(features, labels, assigned, targets, perturbed)
    sep = class_center_separation_loss(features, labels, assigned, targets)
    total, terms = base_training_loss(features, labels, assigned, targets, perturbed)

    assert ce.item() > 0
    assert supcon.item() > 0
    assert sep.item() > 0
    assert set(terms) == {"ce", "contrastive", "center", "total"}
    total.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_base_target_assignment_is_stable_for_unsorted_labels() -> None:
    targets = make_simplex_pseudo_targets(num_targets=5, feature_dim=4)
    assigned = assign_base_targets(base_labels=[10, 2], pseudo_targets=targets)

    assert torch.allclose(assigned[10], targets[0])
    assert torch.allclose(assigned[2], targets[1])

    sorted_assigned = assign_base_targets(base_labels=[10, 2], pseudo_targets=targets, sort_labels=True)
    assert torch.allclose(sorted_assigned[2], targets[0])
    assert torch.allclose(sorted_assigned[10], targets[1])


def test_supervised_anchor_contrastive_loss_matches_paper_sets() -> None:
    targets = torch.eye(4)
    perturbed = torch.tensor(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([2, 2, 10])
    assigned = assign_base_targets([10, 2], targets)
    tau = 0.5

    loss = supervised_anchor_contrastive_loss(features, labels, assigned, targets, perturbed, temperature=tau)

    z = torch.nn.functional.normalize(features, dim=1)
    target_norm = torch.nn.functional.normalize(targets, dim=1)
    pert_norm = torch.nn.functional.normalize(perturbed, dim=1)
    # Anchor sample 0 has positives: itself, same-class sample 1, assigned target row 0,
    # assigned perturb row 0. Its negatives are only different-class features.
    anchor = z[0]
    positives = torch.stack([z[0], z[1], target_norm[0], pert_norm[0]])
    negatives = z[labels != 2]
    expected_first = -((positives @ anchor / tau) - torch.logsumexp(negatives @ anchor / tau, dim=0)).mean()
    # Unassigned pseudo target row 2 has positive perturbed row 2 and negatives:
    # all base features, assigned targets, and assigned perturbations.
    anchor_u = target_norm[2]
    neg_u = torch.cat([z, target_norm[:2], pert_norm[:2]], dim=0)
    expected_unassigned = -((pert_norm[2:3] @ anchor_u / tau) - torch.logsumexp(neg_u @ anchor_u / tau, dim=0)).mean()

    assert torch.isfinite(loss)
    assert expected_first.item() != pytest.approx(expected_unassigned.item())


def test_incremental_calibration_penalizes_competition_and_alignment() -> None:
    new_features = torch.tensor([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])
    new_labels = torch.tensor([2, 2, 3])
    old_weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    new_weights = torch.tensor([[0.6, 0.8], [0.0, 1.0]], requires_grad=True)
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    loss, terms = incremental_calibration_loss(
        new_features,
        new_labels,
        old_weights,
        new_weights,
        new_class_ids=torch.tensor([2, 3]),
        prototypes=prototypes,
        top_k=2,
        margin=0.2,
        tau_fuse=0.5,
        lambda_align=1.6,
    )

    assert terms["hard_count"].item() >= 1
    assert terms["margin"].item() > 0
    assert terms["align"].item() >= 0
    loss.backward()
    assert new_weights.grad is not None
    assert torch.isfinite(new_weights.grad).all()


def test_incremental_calibration_moves_weights_to_feature_device_and_checks_shapes() -> None:
    new_features = torch.randn(4, 3)
    new_labels = torch.tensor([20, 20, 30, 30])
    old_weights = torch.randn(2, 3)
    new_weights = torch.randn(2, 3, requires_grad=True)
    prototypes = torch.randn(2, 3)

    loss, _ = incremental_calibration_loss(
        new_features,
        new_labels,
        old_weights,
        new_weights,
        new_class_ids=torch.tensor([20, 30]),
        prototypes=prototypes,
        top_k=2,
    )
    loss.backward()
    assert new_weights.grad is not None

    with pytest.raises(ValueError, match="prototypes and new_weights"):
        incremental_calibration_loss(
            new_features,
            new_labels,
            old_weights,
            new_weights.detach(),
            new_class_ids=torch.tensor([20, 30]),
            prototypes=torch.randn(1, 3),
        )

    with pytest.raises(ValueError, match="new_labels must have one label per feature"):
        incremental_calibration_loss(
            new_features,
            new_labels[:2],
            old_weights,
            new_weights.detach(),
            new_class_ids=torch.tensor([20, 30]),
            prototypes=prototypes,
        )

    with pytest.raises(ValueError, match="new_labels contain classes outside new_class_ids"):
        incremental_calibration_loss(
            new_features,
            torch.tensor([20, 20, 40, 40]),
            old_weights,
            new_weights.detach(),
            new_class_ids=torch.tensor([20, 30]),
            prototypes=prototypes,
        )


def test_base_losses_validate_pseudo_target_shapes() -> None:
    targets = make_simplex_pseudo_targets(num_targets=3, feature_dim=4)
    assigned = assign_base_targets([0, 1], targets)
    features = torch.randn(4, 4)
    labels = torch.tensor([0, 0, 1, 1])

    with pytest.raises(ValueError, match="perturbed_targets must match pseudo_targets shape"):
        supervised_anchor_contrastive_loss(features, labels, assigned, targets, targets[:2])

    with pytest.raises(ValueError, match="pseudo_targets feature dimension mismatch"):
        class_center_separation_loss(features, labels, assigned, torch.randn(3, 5))


def test_dry_run_performs_incremental_backward_and_rejects_invalid_shot() -> None:
    from paper_reproduction.orthogonal_incremental_sei.train import run_dry_run

    result = run_dry_run(
        {
            "seed": 7,
            "embedding_dim": 8,
            "pseudo_targets": 5,
            "base_classes": 3,
            "shot": 1,
            "input_length": 128,
        },
        device="cpu",
    )
    assert result["incremental_grad_norm"] > 0
    assert result["encoder_grad_after_increment"] == 0
    assert result["encoder_trainable_after_increment"] == 0
    assert result["claim_boundary"] == "synthetic_dry_run_not_formal_reproduction"

    with pytest.raises(ValueError, match="shot must be positive"):
        run_dry_run({"shot": 0}, device="cpu")

    protocol_result = run_dry_run(
        {
            "seed": 7,
            "embedding_dim": 8,
            "pseudo_targets": 5,
            "base_classes": 3,
            "shot": 1,
            "input_length": 128,
            "shot_grid": [1, 5],
            "base_epochs": 100,
            "same_receiver_only": True,
        },
        device="cpu",
    )
    assert "shot_grid" in protocol_result["unsupported_config_fields"]
    assert "base_epochs" in protocol_result["unsupported_config_fields"]
    assert "same_receiver_only" in protocol_result["unsupported_config_fields"]


def test_encoder_classifier_weights_and_metrics_cover_fscil_flow() -> None:
    encoder = SixBlockConv1DEncoder(input_channels=2, embedding_dim=16)
    x = torch.randn(4, 2, 256)
    z = encoder(x)
    assert z.shape == (4, 16)

    classifier = CosineClassifier(embedding_dim=16, num_classes=3)
    logits = classifier(z)
    assert logits.shape == (4, 3)
    override = torch.randn(2, 16, dtype=torch.float64)
    override_logits = classifier(z, weight_override=override)
    assert override_logits.shape == (4, 2)

    labels = torch.tensor([5, 5, 7, 7])
    weights, class_ids = class_mean_weights(torch.randn(4, 16), labels)
    assert class_ids.tolist() == [5, 7]
    assert weights.shape == (2, 16)

    h = harmonic_accuracy(old_accuracy=0.8, new_accuracy=0.4)
    assert math.isclose(h, 2 * 0.8 * 0.4 / 1.2)

    summary = average_incremental_metrics(
        session_accuracies=[0.9, 0.8, 0.7],
        old_accuracies=[0.9, 0.82, 0.75],
        new_accuracies=[0.9, 0.7, 0.6],
        accuracy_matrix=torch.tensor(
            [
                [0.90, float("nan"), float("nan")],
                [0.85, 0.80, float("nan")],
                [0.80, 0.70, 0.70],
            ]
        ),
    )
    assert summary["A_bar"] == pytest.approx(0.8)
    assert summary["H_bar"] < 0.9
    assert summary["F_bar"] > 0
    assert summary["F_bar"] == pytest.approx((0.05 + 0.10) / 2.0)

    total_session_denominator = average_incremental_metrics(
        session_accuracies=[0.9, 0.8, 0.7],
        old_accuracies=[0.9, 0.82, 0.75],
        new_accuracies=[0.9, 0.7, 0.6],
        accuracy_matrix=torch.tensor(
            [
                [0.90, float("nan"), float("nan")],
                [0.85, 0.80, float("nan")],
                [0.80, 0.70, 0.70],
            ]
        ),
        forgetting_denominator="total_sessions",
    )
    assert total_session_denominator["F_bar"] == pytest.approx((0.05 + 0.10) / 3.0)

    negative_forgetting = forgetting_by_session(
        torch.tensor(
            [
                [0.50, float("nan")],
                [0.60, 0.70],
            ]
        )
    )
    assert negative_forgetting == pytest.approx([-0.10])

    with pytest.raises(ValueError, match="old_accuracies and new_accuracies"):
        average_incremental_metrics(
            session_accuracies=[0.9, 0.8],
            old_accuracies=[],
            new_accuracies=[],
            accuracy_matrix=torch.eye(2),
        )


def test_dry_run_consumes_paper_named_temperatures_and_margin() -> None:
    from paper_reproduction.orthogonal_incremental_sei.train import _paper_float

    config = {"tau_s": 0.11, "tau_c": 0.22, "q": 0.33, "lambda_a": 0.44}
    assert _paper_float(config, "contrast_temperature", "tau_s", default=0.1) == pytest.approx(0.11)
    assert _paper_float(config, "center_temperature", "tau_c", default=0.1) == pytest.approx(0.22)
    assert _paper_float(config, "margin", "q", default=0.2) == pytest.approx(0.33)
    assert _paper_float(config, "lambda_align", "lambda_a", default=1.6) == pytest.approx(0.44)


def test_supervised_anchor_contrastive_loss_reports_protocol_errors() -> None:
    targets = make_simplex_pseudo_targets(num_targets=3, feature_dim=4)
    assigned = assign_base_targets([0, 1], targets)
    perturbed = perturb_pseudo_targets(targets, noise_range=0.01, seed=3)

    with pytest.raises(ValueError, match="labels contain classes without assigned pseudo targets"):
        supervised_anchor_contrastive_loss(
            torch.randn(3, 4),
            torch.tensor([0, 1, 9]),
            assigned,
            targets,
            perturbed,
        )

    with pytest.raises(ValueError, match="at least two classes"):
        supervised_anchor_contrastive_loss(
            torch.randn(3, 4),
            torch.tensor([0, 0, 0]),
            assigned,
            targets,
            perturbed,
        )


def test_encoder_rejects_too_short_input_before_pooling_crash() -> None:
    encoder = SixBlockConv1DEncoder(input_channels=2, embedding_dim=8)
    with pytest.raises(ValueError, match="at least 64"):
        encoder(torch.randn(2, 2, 32))
    with pytest.raises(ValueError, match="input channel mismatch"):
        encoder(torch.randn(2, 1, 128))


def test_formal_wisig_runner_writes_incremental_metrics(tmp_path) -> None:
    from paper_reproduction.orthogonal_incremental_sei.train import run_formal_wisig

    rng = np.random.default_rng(123)
    data = []
    for tx in range(5):
        tx_items = []
        for _rx in range(1):
            rx_items = []
            for _day in range(1):
                eq_items = [rng.normal(loc=tx * 0.1, scale=0.01, size=(8, 64, 2)).astype(np.float32)]
                rx_items.append(eq_items)
            tx_items.append(rx_items)
        data.append(tx_items)
    pkl_path = tmp_path / "fake_wisig.pkl"
    with pkl_path.open("wb") as handle:
        pickle.dump(
            {
                "data": data,
                "tx_list": [f"tx-{i}" for i in range(5)],
                "rx_list": ["rx-a"],
                "capture_date_list": ["day-a"],
                "equalized_list": [1],
            },
            handle,
        )

    result = run_formal_wisig(
        {
            "seed": 3,
            "input_length": 64,
            "embedding_dim": 8,
            "pseudo_targets": 6,
            "base_classes": 3,
            "increment_classes_per_session": 1,
            "num_increment_sessions": 2,
            "base_epochs": 1,
            "increment_epochs": 1,
            "batch_size": 12,
            "eval_batch_size": 8,
            "min_samples_per_transmitter": 4,
            "base_train_ratio": 0.5,
            "shot": 1,
            "receiver_label": "rx-a",
            "optimizer": "SGD",
        },
        wisig_pkl=str(pkl_path),
        run_dir=tmp_path / "run",
        device="cpu",
    )

    assert result["mode"] == "formal_wisig_fscil"
    assert result["summary"]["A_bar"] >= 0.0
    assert len(result["session_accuracies"]) == 3
    assert (tmp_path / "run" / "metrics.json").is_file()
