import torch

from experiments.pamf_lora_main.shared_fullfusion_run import ActionWithVariance, SharedPhysicalSolveGain


class IdentityAction:
    def __call__(self, x):
        return x

    def adjoint(self, x):
        return x


def test_zero_initialization_preserves_anchor_and_trains_both_heads():
    torch.manual_seed(9)
    y = torch.randn(2, 256, dtype=torch.complex64)
    anchor, physical = y*.8, y*1.2
    action = ActionWithVariance(IdentityAction(), torch.full((2,), .1))
    model = SharedPhysicalSolveGain(32, anchor_real=True, correction_cap=.05)
    output = model(y, anchor, physical, action)
    torch.testing.assert_close(output, anchor, rtol=0, atol=0)
    target = y*torch.exp(1j*torch.sin(torch.arange(256)/17))
    grads = torch.autograd.grad((output-target).abs().square().mean(),
                                (model.net[-1].weight, model.phase_net[-1].weight))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)
