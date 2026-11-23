import torch

from experiments.pamf_repl_view_shuffle.observable_action import (
    ObservableAction, adjoint, estimate_delay, fit_action, forward_action, objective,
)


def test_widely_linear_action_has_the_real_adjoint_with_edges():
    model = ObservableAction(True)
    torch.manual_seed(9)
    x = torch.randn(3, 256, dtype=torch.complex64)
    y = torch.randn_like(x)
    h = torch.randn(3, 7, dtype=x.dtype)
    delay = torch.tensor([-33, 0, 15])
    forward = forward_action(x, h, delay, model.index, model.ramp)
    transpose = adjoint(y, h, delay, model.index, model.ramp)
    lhs = (forward.conj() * y).sum(dim=-1).real
    rhs = (x.conj() * transpose).sum(dim=-1).real
    torch.testing.assert_close(lhs, rhs, atol=1e-4, rtol=1e-5)


def test_current_iq_fit_detects_lag_and_reduces_held_error():
    model = ObservableAction(True)
    torch.manual_seed(1)
    x = torch.randn(3, 256, dtype=torch.complex64)
    delay = torch.tensor([-34, 12, 35])
    h = torch.tensor([[.02j, -.1, 1.+.45j, -.1j, -.08, .2, .1j]]).repeat(3, 1)
    y = forward_action(x, h, delay, model.index, model.ramp)
    guessed = estimate_delay(x, y, model.delays)
    assert torch.equal(guessed, delay)
    _, variance = fit_action(x, y, guessed, model.index, model.ramp, model.regularizer, model.center)
    assert torch.all(variance[:, 0] < .035 * (x - y).abs().square().mean(dim=-1))


def test_both_stages_receive_matched_source_gradient():
    source = torch.randn(6, 256, dtype=torch.complex64, generator=torch.Generator().manual_seed(7))
    for fitted in (False, True):
        torch.manual_seed(11)
        model = ObservableAction(fitted)
        loss, risk = objective(model, source, torch.Generator().manual_seed(12))
        assert torch.isfinite(risk).all()
        loss.backward()
        assert model.coarse.out.weight.grad.norm() > 0
        assert model.fine.out.weight.grad.norm() > 0
