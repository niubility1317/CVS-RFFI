import torch

from experiments.pamf_physics_tri.hybrid_margin import capped_margin_loss


def test_capped_margin_stops_when_satisfied_and_ignores_unaccepted_packets():
    raw = torch.tensor([[1., 0.], [1., 0.], [1., 0.]], requires_grad=True)
    augmented = torch.tensor([[1., 0.], [0., 1.], [0., 1.]], requires_grad=True)
    weights = torch.eye(2, requires_grad=True)
    pseudo = torch.tensor([0, 0, 0])
    accepted = torch.tensor([True, True, False])
    loss, diagnostics = capped_margin_loss(raw, augmented, weights, pseudo, accepted, .5)
    torch.testing.assert_close(loss, torch.tensor(.5))
    assert diagnostics['violating'].tolist() == [False, True, False]
    loss.backward()
    assert raw.grad is None and weights.grad is None
    assert augmented.grad[0].abs().sum() == 0
    assert augmented.grad[1].abs().sum() > 0
    assert augmented.grad[2].abs().sum() == 0


def test_negative_raw_margin_has_zero_reference_but_corrects_negative_augmented_margin():
    raw = torch.tensor([[0., 1.]])
    augmented = torch.tensor([[0., 1.]], requires_grad=True)
    loss, diagnostics = capped_margin_loss(raw, augmented, torch.eye(2), torch.tensor([0]), torch.tensor([True]), .5)
    torch.testing.assert_close(diagnostics['reference'], torch.tensor([0.]))
    torch.testing.assert_close(loss, torch.tensor(1.))
