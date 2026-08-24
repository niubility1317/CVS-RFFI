import math

import pytest
import torch

from cvsrffi.bounded_domain_confusion import (
    bounded_domain_confusion_loss,
    bounded_domain_objectives,
)


def test_bounded_confusion_has_uniform_minimum_and_log_domain_upper_bound():
    uniform = torch.zeros(4, 7)
    saturated = torch.full((4, 7), -1000.0)
    saturated[:, 0] = 1000.0

    uniform_loss = bounded_domain_confusion_loss(uniform)
    saturated_loss = bounded_domain_confusion_loss(saturated)

    assert uniform_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert 0.0 <= saturated_loss.item() <= math.log(7) + 1e-6
    assert saturated_loss.item() == pytest.approx(math.log(7), rel=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bounded_confusion_is_finite_for_saturated_amp_logits(dtype):
    logits = torch.tensor(
        [[1000.0, -1000.0, -1000.0], [-1000.0, 1000.0, -1000.0]],
        dtype=dtype,
        requires_grad=True,
    )
    loss = bounded_domain_confusion_loss(logits)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_domain_discriminator_and_confusion_have_isolated_gradients():
    torch.manual_seed(3)
    head = torch.nn.Linear(5, 3)
    z_id = torch.randn(8, 5, requires_grad=True)
    domains = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    losses = bounded_domain_objectives(head, z_id, domains)
    losses["discriminator"].backward(retain_graph=True)
    assert z_id.grad is None or z_id.grad.abs().sum().item() == 0.0
    assert head.weight.grad is not None and head.weight.grad.abs().sum().item() > 0.0

    head.zero_grad(set_to_none=True)
    z_id.grad = None
    losses["confusion"].backward()
    assert z_id.grad is not None and z_id.grad.abs().sum().item() > 0.0
    assert head.weight.grad is None


def test_domain_head_forward_is_forced_to_float32_inside_outer_autocast():
    head = torch.nn.Linear(4, 3)
    with torch.no_grad():
        head.weight.fill_(1.0e10)
        head.bias.zero_()
    z_id = torch.full((6, 4), 1.0e10, requires_grad=True)
    domains = torch.tensor([0, 1, 2, 0, 1, 2])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        losses = bounded_domain_objectives(head, z_id, domains)

    assert losses["discriminator_logits"].dtype == torch.float32
    assert losses["confusion_logits"].dtype == torch.float32
    assert torch.isfinite(losses["discriminator"])
    assert torch.isfinite(losses["confusion"])
    (losses["discriminator"] + losses["confusion"]).backward()
    assert z_id.grad is not None and torch.isfinite(z_id.grad).all()


def test_bounded_confusion_rejects_nonfinite_logits_before_log_softmax():
    with pytest.raises(FloatingPointError, match="non-finite"):
        bounded_domain_confusion_loss(torch.tensor([[float("inf"), 0.0]]))
