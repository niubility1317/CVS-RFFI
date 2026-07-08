from __future__ import annotations

import math

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


def test_encoder_classifier_weights_and_metrics_cover_fscil_flow() -> None:
    encoder = SixBlockConv1DEncoder(input_channels=2, embedding_dim=16)
    x = torch.randn(4, 2, 256)
    z = encoder(x)
    assert z.shape == (4, 16)

    classifier = CosineClassifier(embedding_dim=16, num_classes=3)
    logits = classifier(z)
    assert logits.shape == (4, 3)

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
