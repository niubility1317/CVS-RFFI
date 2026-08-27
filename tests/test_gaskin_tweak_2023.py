import torch
import pytest

from paper_reproduction.gaskin_tweak_2023.calibration import calibrate, closed_set_predict, open_set_admit
from paper_reproduction.gaskin_tweak_2023.metrics import average_trials
from paper_reproduction.gaskin_tweak_2023.model import TweakEncoder
from paper_reproduction.gaskin_tweak_2023.triplet import batch_hard_triplet_loss, hard_positive_negative_indices


def test_encoder_produces_paper_embedding_shape():
    model = TweakEncoder()
    assert model(torch.randn(3, 2, 128)).shape == (3, 12)


def test_encoder_rejects_nonpaper_input_shape():
    with pytest.raises(ValueError, match="2, 128"):
        TweakEncoder()(torch.randn(2, 2, 127))


def test_hard_mining_selects_farthest_positive_and_nearest_negative():
    embeddings = torch.tensor([[0.0], [2.0], [1.0], [5.0]])
    labels = torch.tensor([0, 0, 1, 1])
    positive, negative = hard_positive_negative_indices(embeddings, labels)
    assert positive.tolist() == [1, 0, 3, 2]
    assert negative.tolist() == [2, 2, 0, 1]


def test_batch_hard_triplet_loss_uses_paper_margin():
    embeddings = torch.tensor([[0.0], [2.0], [1.0], [5.0]], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(embeddings, labels, margin=0.1)
    assert loss.item() == pytest.approx(1.6, abs=1e-6)
    loss.backward()
    assert embeddings.grad is not None


def test_calibration_uses_centroid_and_mean_radius_without_gradients():
    features = torch.tensor([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0]], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    state = calibrate(features, labels)
    assert state.centroids.tolist() == [[1.0, 0.0], [11.0, 0.0]]
    assert state.radii.tolist() == pytest.approx([1.0, 1.0])
    assert not state.centroids.requires_grad
    assert not state.radii.requires_grad


def test_closed_set_follows_inside_then_excess_distance_rules():
    state = calibrate(torch.tensor([[0.0], [2.0], [10.0], [12.0]]), torch.tensor([0, 0, 1, 1]))
    assert closed_set_predict(torch.tensor([[0.5], [7.0]]), state).tolist() == [0, 1]


def test_open_set_admits_only_if_any_class_radius_contains_input_point():
    state = calibrate(torch.tensor([[0.0], [2.0], [10.0], [12.0]]), torch.tensor([0, 0, 1, 1]))
    assert open_set_admit(torch.tensor([[0.0], [5.0]]), state).tolist() == [True, False]


def test_trial_average_reports_mean_of_paper_open_set_metrics():
    result = average_trials([
        {"auroc": 0.8, "tpr": 0.9, "fpr": 0.2},
        {"auroc": 1.0, "tpr": 0.7, "fpr": 0.4},
    ])
    assert result == {"auroc": pytest.approx(0.9), "tpr": pytest.approx(0.8), "fpr": pytest.approx(0.3)}
