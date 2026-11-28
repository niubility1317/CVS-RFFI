import torch
from torch import nn

from experiments.pamf_action_restore.operators import Action
from experiments.pamf_lora_main.dura_real_gain_amplitude import R17AmplitudeGain
from experiments.pamf_lora_main.dura_real_gain_unroll import fused_anchor_unroll, fused_unroll
from experiments.pamf_restoration.models import complex_iq, pair
from experiments.pamf_unrolled_restore.models import UnrolledRestorer


def test_zero_gain_reproduces_r17_and_solver_backpropagates_to_each_gain():
    class Frozen(nn.Module):
        def estimate(self, received):
            y = complex_iq(received)
            action = Action(torch.ones(len(y), 1, dtype=y.dtype), torch.ones_like(y), torch.zeros_like(y))
            return .8 * received, action, torch.full((len(y),), .1)

    torch.manual_seed(12)
    r17 = UnrolledRestorer('unroll_unshared', Frozen()).eval().requires_grad_(False)
    gain = R17AmplitudeGain(32)
    y = torch.randn(1, 256, dtype=torch.complex64)
    reference = complex_iq(r17(pair(y)))
    fused = fused_unroll(r17, gain, y, .5 * y)
    torch.testing.assert_close(fused, reference, rtol=1e-5, atol=1e-6)
    fused.abs().square().mean().backward()
    gradients = gain.net[-1].weight.grad.flatten(1).norm(dim=1)
    assert torch.isfinite(gradients).all() and (gradients > 0).all()


def test_anchor_unroll_starts_at_real_gain_prior_and_trains_all_directions():
    class Native(nn.Module):
        def __init__(self):
            super().__init__()
            self.priors = nn.ModuleList([nn.Conv1d(9, 2, 1) for _ in range(3)])

        def estimate(self, received):
            y = complex_iq(received)
            action = Action(torch.ones(len(y), 1, dtype=y.dtype), torch.ones_like(y), torch.zeros_like(y))
            return .8 * received, action, torch.full((len(y),), .1)

    torch.manual_seed(13)
    native = Native().eval().requires_grad_(False)
    gain = R17AmplitudeGain(32)
    y = torch.randn(1, 256, dtype=torch.complex64)
    anchor = .6 * y
    output = fused_anchor_unroll(native, gain, y, anchor)
    torch.testing.assert_close(output, anchor, rtol=1e-5, atol=1e-6)
    output.abs().square().mean().backward()
    gradients = gain.net[-1].weight.grad.flatten(1).norm(dim=1)
    assert torch.isfinite(gradients).all() and (gradients > 0).all()
