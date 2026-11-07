import pytest
import torch

from experiments.phase1_dura_orthogonal.observable_source_risk import observable_risk_penalty


def test_observable_risk_uses_only_physically_difficult_source_items():
    clean = torch.ones(3, 2, 256)
    anchor = torch.ones_like(clean)
    innovation = torch.zeros_like(clean)
    innovation[1] = .2
    innovation[2] = .3
    output = (clean + torch.tensor([1., 3., 2.])[:, None, None]).requires_grad_()
    loss, fraction = observable_risk_penalty(output, anchor, innovation, clean, return_fraction=True)
    assert loss.item() == pytest.approx(3.25)
    assert fraction.item() == pytest.approx(2 / 3)
    loss.backward()
    assert output.grad[0].count_nonzero() == 0
    assert output.grad[1].count_nonzero() > 0
    assert output.grad[2].count_nonzero() > 0


def test_no_high_risk_items_gives_finite_zero_loss_and_gradient():
    clean = torch.ones(2, 2, 256)
    output = (2 * clean).requires_grad_()
    loss = observable_risk_penalty(output, clean, torch.zeros_like(clean), clean)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(output.grad).all() and output.grad.count_nonzero() == 0


def test_mismatched_source_shapes_are_rejected():
    x = torch.ones(2, 2, 256)
    with pytest.raises(ValueError):
        observable_risk_penalty(x, x, x[:1], x)
