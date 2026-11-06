import pytest
import torch

from experiments.phase1_dura_orthogonal.fused_gradient_guard import checked_backward_and_clip


def test_checked_update_matches_original_clipped_update():
    torch.manual_seed(11)
    first = torch.nn.Sequential(torch.nn.Linear(3, 7), torch.nn.SiLU(), torch.nn.Linear(7, 2))
    second = torch.nn.Sequential(torch.nn.Linear(3, 7), torch.nn.SiLU(), torch.nn.Linear(7, 2))
    second.load_state_dict(first.state_dict())
    x, y = torch.randn(8, 3), torch.randn(8, 2)
    loss1 = (first(x)-y).square().mean()
    assert torch.isfinite(loss1)
    loss1.backward()
    assert torch.isfinite(first[-1].weight.grad).all()
    torch.nn.utils.clip_grad_norm_(first.parameters(), 1., error_if_nonfinite=True)
    loss2 = (second(x)-y).square().mean()
    checked_backward_and_clip(loss2, second, (second[-1].weight,))
    assert torch.equal(loss1, loss2)
    assert all(torch.equal(a.grad, b.grad) for a,b in zip(first.parameters(), second.parameters()))


def test_nonfinite_loss_with_finite_gradient_fails_before_update():
    model = torch.nn.Linear(2, 1)
    initial = [p.detach().clone() for p in model.parameters()]
    loss = model(torch.ones(1, 2)).sum() * 0 + torch.tensor(float('inf'))
    with pytest.raises(FloatingPointError, match='nonfinite'):
        checked_backward_and_clip(loss, model, (model.weight,))
    assert all(torch.equal(a, b) for a,b in zip(initial, model.parameters()))


def test_nonfinite_gradient_and_missing_required_gradient_fail():
    model = torch.nn.Linear(2, 1)
    loss = model(torch.ones(1, 2)).sum() * torch.tensor(float('nan'))
    with pytest.raises(FloatingPointError, match='nonfinite'):
        checked_backward_and_clip(loss, model, (model.weight,))
    model.zero_grad(set_to_none=True)
    with pytest.raises(ValueError, match='gradient'):
        checked_backward_and_clip(model.bias.sum(), model, (model.weight,))


def test_finite_loss_with_infinite_gradient_fails_before_update():
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.zero_()
    initial = [p.detach().clone() for p in model.parameters()]
    loss = model.weight.sqrt().sum()
    assert torch.isfinite(loss)
    with pytest.raises(FloatingPointError, match='nonfinite'):
        checked_backward_and_clip(loss, model, (model.weight,))
    assert all(torch.equal(a, b) for a,b in zip(initial, model.parameters()))
