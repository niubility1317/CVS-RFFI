import pytest
import torch

from paper_reproduction.gaskin_tweak_2023.calibration import (
    aggregate_embeddings,
    calibrate_domains,
    closed_set_predict_grouped,
)
from paper_reproduction.gaskin_tweak_2023.evaluation import open_set_trial_metrics
from paper_reproduction.gaskin_tweak_2023.model import TweakEncoder
from paper_reproduction.gaskin_tweak_2023.training import shared_triplet_loss


def test_shared_encoder_triplet_backpropagates_through_all_three_inputs():
    encoder = TweakEncoder()
    anchor = torch.randn(2, 2, 128, requires_grad=True)
    positive = torch.randn(2, 2, 128, requires_grad=True)
    negative = torch.randn(2, 2, 128, requires_grad=True)
    loss = shared_triplet_loss(encoder, anchor, positive, negative)
    loss.backward()
    assert anchor.grad is not None
    assert positive.grad is not None
    assert negative.grad is not None
    assert any(parameter.grad is not None for parameter in encoder.parameters())


def test_domain_calibration_requires_exactly_n_examples_per_class_and_groups_m_embeddings():
    features = torch.tensor([[0.0], [2.0], [10.0], [12.0], [20.0], [22.0], [30.0], [32.0]])
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    domains = ["rx1", "rx1", "rx1", "rx1", "rx2", "rx2", "rx2", "rx2"]
    bank = calibrate_domains(features, labels, domains, samples_per_class=2)
    assert bank.by_domain["rx1"].centroids.squeeze(1).tolist() == [1.0, 11.0]
    assert bank.by_domain["rx2"].centroids.squeeze(1).tolist() == [21.0, 31.0]
    grouped = aggregate_embeddings(torch.arange(20, dtype=torch.float32).reshape(20, 1), group_size=10)
    assert grouped.squeeze(1).tolist() == [4.5, 14.5]
    assert closed_set_predict_grouped(torch.tensor([[0.0], [2.0]]), bank.by_domain["rx1"], group_size=2).tolist() == [0]
    with pytest.raises(ValueError, match="exactly 2"):
        calibrate_domains(features[:-1], labels[:-1], domains[:-1], samples_per_class=2)


def test_open_set_metrics_use_minimum_centroid_distance_and_paper_admit_rule():
    state = calibrate_domains(
        torch.tensor([[0.0], [2.0], [10.0], [12.0]]),
        torch.tensor([0, 0, 1, 1]),
        ["rx", "rx", "rx", "rx"],
        samples_per_class=2,
    ).by_domain["rx"]
    metrics = open_set_trial_metrics(
        known_points=torch.tensor([[1.0], [11.0]]),
        known_labels=torch.tensor([0, 1]),
        unknown_points=torch.tensor([[5.0], [20.0]]),
        state=state,
    )
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["tpr"] == pytest.approx(1.0)
    assert metrics["fpr"] == pytest.approx(0.0)
    assert metrics["accepted_known_accuracy"] == pytest.approx(1.0)
