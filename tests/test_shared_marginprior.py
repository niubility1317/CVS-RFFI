import pytest
import torch

from experiments.pamf_lora_main.shared_marginprior_run import margin_preservation


def test_margin_loss_masks_unconfident_and_disagreeing_packets():
    clean = torch.tensor([[5., 0.], [0., 0.], [5., 0.]])
    anchor = torch.tensor([[3., 0.], [3., 0.], [0., 3.]])
    output = torch.tensor([[1., 0.], [0., 2.], [0., 2.]], requires_grad=True)
    loss, eligible = margin_preservation(clean, anchor, output)
    torch.testing.assert_close(loss, torch.tensor(2/3))
    assert eligible.tolist() == [True, False, False]
    grad, = torch.autograd.grad(loss, output)
    torch.testing.assert_close(grad[1:], torch.zeros_like(grad[1:]))
    assert grad[0, 0] < 0 and grad[0, 1] > 0


def test_margin_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        margin_preservation(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 4))
