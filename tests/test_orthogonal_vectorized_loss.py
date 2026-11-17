import pytest
import torch
from experiments.orthogonal_paper_repro.vectorized_loss import contrastive_loss
from experiments.pamf_phase2_practical.vendor_native.paper_reproduction.orthogonal_incremental_sei.losses import supervised_anchor_contrastive_loss


@pytest.mark.parametrize('extra_targets', [0, 4])
def test_vectorized_value_and_gradients_match_existing_reference(extra_targets):
    generator = torch.Generator().manual_seed(420)
    features = torch.randn(9, 12, generator=generator, requires_grad=True)
    targets = torch.randn(3 + extra_targets, 12, generator=generator, requires_grad=True)
    perturbed = torch.randn(3 + extra_targets, 12, generator=generator, requires_grad=True)
    labels = torch.tensor([7, 3, 9, 7, 7, 3, 9, 3, 9])
    assigned = {7: targets[0], 3: targets[1], 9: targets[2]}
    expected = supervised_anchor_contrastive_loss(features, labels, assigned, targets, perturbed)
    actual = contrastive_loss(features, labels, assigned, targets, perturbed)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)
    expected_grads = torch.autograd.grad(expected, (features, targets, perturbed), retain_graph=True)
    actual_grads = torch.autograd.grad(actual, (features, targets, perturbed))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=1e-5)


def test_vectorized_rejects_single_class_batches():
    targets = torch.randn(3, 4)
    with pytest.raises(ValueError):
        contrastive_loss(torch.randn(2, 4), torch.tensor([1, 1]), {1: targets[0]}, targets, targets)
