import torch
import pytest

from paper_reproduction.gaskin_tweak_2023.calibration import calibrate, closed_set_predict, open_set_admit
from paper_reproduction.gaskin_tweak_2023.metrics import average_trials
from paper_reproduction.gaskin_tweak_2023.model import TweakEncoder
from paper_reproduction.gaskin_tweak_2023.triplet import batch_hard_triplet_loss, hard_positive_negative_indices
from paper_reproduction.gaskin_tweak_2023.triplet import all_strict_hard_triplet_loss, margin_violating_triplet_loss, random_triplet_indices, strict_hard_triplet_loss


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


def test_batch_hard_triplet_loss_uses_paper_squared_l2_distances_and_margin():
    embeddings = torch.tensor([[0.0], [2.0], [1.0], [5.0]], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(embeddings, labels, margin=0.1)
    # Break caught: Equation (1) squares each L2 distance rather than using raw L2 distance.
    assert loss.item() == pytest.approx(7.1, abs=1e-6)
    loss.backward()
    assert embeddings.grad is not None


def test_margin_violating_triplet_loss_discards_random_candidates_that_already_meet_the_margin():
    # Break caught: averaging every candidate, including easy ones, changes the cited online hard-negative rule.
    embeddings = torch.tensor([[0.0], [0.3], [0.4], [0.1]], requires_grad=True)
    loss = margin_violating_triplet_loss(
        embeddings,
        anchors=torch.tensor([0, 1]),
        positives=torch.tensor([1, 0]),
        negatives=torch.tensor([2, 3]),
        margin=0.1,
    )

    assert loss.item() == pytest.approx(0.09, abs=1e-6)
    loss.backward()
    assert embeddings.grad is not None


def test_random_triplet_indices_choose_a_same_class_positive_and_different_class_negative():
    # Break caught: online mining must sample candidates, not silently revert to batch extrema.
    labels = torch.tensor([0, 0, 1, 1])
    anchors, positives, negatives = random_triplet_indices(labels, generator=torch.Generator().manual_seed(7))

    assert anchors.tolist() == [0, 1, 2, 3]
    assert torch.equal(labels[positives], labels[anchors])
    assert torch.all(positives.ne(anchors))
    assert torch.all(labels[negatives].ne(labels[anchors]))


def test_strict_hard_mining_excludes_a_margin_violating_triplet_when_negative_is_not_closer_than_positive():
    # Break caught: Tweak's IV-A definition is d(A,N)<d(A,P), not merely a positive margin loss.
    embeddings = torch.tensor([[0.0], [1.0], [1.05]], requires_grad=True)
    loss, has_hard_triplets = strict_hard_triplet_loss(
        embeddings,
        anchors=torch.tensor([0]),
        positives=torch.tensor([1]),
        negatives=torch.tensor([2]),
        margin=0.1,
    )

    assert not has_hard_triplets
    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert embeddings.grad is not None


def test_strict_hard_mining_keeps_triplets_with_negative_closer_than_positive():
    embeddings = torch.tensor([[0.0], [2.0], [1.0]], requires_grad=True)
    loss, has_hard_triplets = strict_hard_triplet_loss(
        embeddings,
        anchors=torch.tensor([0]),
        positives=torch.tensor([1]),
        negatives=torch.tensor([2]),
        margin=0.1,
    )

    assert has_hard_triplets
    assert loss.item() == pytest.approx(3.1)
    loss.backward()
    assert embeddings.grad is not None


def test_all_strict_hard_mining_uses_every_and_only_batch_triplet_with_negative_closer_than_positive():
    # Break caught: reducing online mining to one random candidate per anchor omits nearly all hard mini-batch triplets.
    embeddings = torch.tensor([[0.0], [3.0], [1.0], [5.0]], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])

    loss, has_hard_triplets = all_strict_hard_triplet_loss(embeddings, labels, margin=0.1)

    # The six strict-hard directed triplets have squared-loss values 8.1, 5.1, 5.1, 15.1, 12.1, 12.1.
    assert has_hard_triplets
    assert loss.item() == pytest.approx(9.6)
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
