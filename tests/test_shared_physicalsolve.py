import torch

from experiments.pamf_lora_main.shared_physicalsolve_run import (
    ActionWithVariance, SharedFullSpectrumGain, SharedPhysicalSolveGain, make_eval_gain,
)
from experiments.pamf_action_restore.models import SCALE


class IdentityAction:
    def __call__(self, x):
        return x

    def adjoint(self, x):
        return x


def test_identity_channel_solver_projects_correction_with_known_solution():
    torch.manual_seed(11)
    y = torch.randn(2, 256, dtype=torch.complex64)
    a, b = y*.8, y*1.2
    action = ActionWithVariance(IdentityAction(), torch.full((2,), .1))
    plain = SharedFullSpectrumGain(32, anchor_real=True, correction_cap=.05)
    with torch.no_grad():
        plain.net[-1].weight.normal_(0, .005)
    model = SharedPhysicalSolveGain(32, anchor_real=True, correction_cap=.05)
    model.load_state_dict(plain.state_dict(), strict=True)
    expected = a + (.1/(.1+SCALE**2))*(plain(y, a, b, action)-a)
    result = model(y, a, b, action)
    torch.testing.assert_close(result, expected, rtol=2e-4, atol=2e-5)
    gradient, = torch.autograd.grad(result.abs().square().mean(), model.net[-1].weight)
    assert torch.isfinite(gradient).all() and gradient.norm() > 0


def test_prediction_replaces_old_action_for_both_data_shapes():
    for dataset, width in [('lora', 1024), ('wisig', 32)]:
        y = torch.randn(1, 8*width, dtype=torch.complex64)
        action = ActionWithVariance(IdentityAction(), torch.full((1,), .1))
        model = make_eval_gain(dataset, lambda x: (x*1.1, action))(width)
        torch.testing.assert_close(model(y, y*.8, y*9, IdentityAction()), y*.8 if dataset == 'lora' else y*1.1)


def test_zero_gain_keeps_anchor_and_has_trainable_gradient():
    y = torch.randn(2, 256, dtype=torch.complex64)
    action = ActionWithVariance(IdentityAction(), torch.full((2,), .1))
    model = SharedPhysicalSolveGain(32, anchor_real=True, correction_cap=.05)
    output = model(y, y*.8, y*1.2, action)
    torch.testing.assert_close(output, y*.8, rtol=0, atol=0)
    grad, = torch.autograd.grad((output-y).abs().square().mean(), model.net[-1].weight)
    assert torch.isfinite(grad).all() and grad.norm() > 0
