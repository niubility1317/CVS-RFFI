import pytest
import torch

from cvsrffi.muse_ssdg import (
    candidate_set_cross_entropy,
    candidate_set_mask,
    weighted_soft_cross_entropy,
)


def test_candidate_set_caps_at_three_and_rejects_unreachable_mass():
    prob = torch.tensor(
        [[0.40, 0.30, 0.20, 0.10, 0.00], [0.24, 0.23, 0.22, 0.16, 0.15]]
    )
    mask, active = candidate_set_mask(prob, mass=0.75, max_classes=3)
    assert mask[0].sum().item() == 3
    assert active.tolist() == [True, False]


@pytest.mark.parametrize("mass", [0.0, 0.749999])
def test_candidate_set_rejects_mass_below_fixed_quality_floor(mass):
    with pytest.raises(ValueError, match="mass"):
        candidate_set_mask(torch.tensor([[0.8, 0.1, 0.1]]), mass=mass)


def test_candidate_set_rejects_class_cap_above_three():
    with pytest.raises(ValueError, match="max_classes"):
        candidate_set_mask(torch.tensor([[0.8, 0.1, 0.1]]), max_classes=4)


def test_inactive_candidate_from_mask_and_active_cannot_backpropagate_identity():
    logits = torch.tensor(
        [[2.0, 0.0, -1.0, -2.0, -3.0], [0.2, 0.1, 0.0, -0.1, -0.2]],
        requires_grad=True,
    )
    probability = torch.tensor(
        [[0.80, 0.10, 0.05, 0.03, 0.02], [0.24, 0.23, 0.22, 0.16, 0.15]]
    )
    candidate, active = candidate_set_mask(probability, mass=0.75, max_classes=3)
    assert active.tolist() == [True, False]
    loss = candidate_set_cross_entropy(
        logits,
        (candidate, active),
        torch.ones(2),
        torch.ones(2, dtype=torch.bool),
    )
    loss.backward()
    assert torch.equal(logits.grad[1], torch.zeros_like(logits.grad[1]))
    assert logits.grad[0].abs().sum().item() > 0.0


def test_inactive_low_confidence_row_has_zero_identity_gradient():
    logits = torch.tensor([[0.2, 0.1, 0.0, -0.1]], requires_grad=True)
    candidate = torch.zeros_like(logits, dtype=torch.bool)
    loss = candidate_set_cross_entropy(
        logits, candidate, torch.ones(1), torch.tensor([False])
    )
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_weighted_soft_cross_entropy_uses_only_masked_weight_mass():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    teacher_prob = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    weights = torch.tensor([2.0, 100.0])
    mask = torch.tensor([True, False])

    loss = weighted_soft_cross_entropy(logits, teacher_prob, weights, mask)
    expected = -torch.log_softmax(logits.detach(), dim=-1)[0, 0]
    assert torch.allclose(loss, expected, atol=1e-7)


def test_weighted_soft_cross_entropy_empty_mask_keeps_zero_gradient_graph():
    logits = torch.tensor([[0.5, -0.5]], requires_grad=True)
    teacher_prob = torch.tensor([[0.5, 0.5]])
    loss = weighted_soft_cross_entropy(
        logits, teacher_prob, torch.tensor([2.0]), torch.tensor([False])
    )
    assert loss.requires_grad
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_candidate_set_cross_entropy_scores_total_candidate_mass_not_uniform_targets():
    logits = torch.log(torch.tensor([[0.70, 0.20, 0.10]])).requires_grad_()
    candidate = torch.tensor([[True, True, False]])
    loss = candidate_set_cross_entropy(
        logits, candidate, torch.tensor([3.0]), torch.tensor([True])
    )
    assert torch.allclose(loss, -torch.log(torch.tensor(0.90)), atol=1e-6)


def test_candidate_set_cross_entropy_empty_sample_mask_keeps_zero_gradient_graph():
    logits = torch.tensor([[0.3, -0.2, 0.1]], requires_grad=True)
    candidate = torch.tensor([[True, False, True]])
    loss = candidate_set_cross_entropy(
        logits, candidate, torch.tensor([1.0]), torch.tensor([False])
    )
    assert loss.requires_grad
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
