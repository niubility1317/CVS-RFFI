import torch

from experiments.pamf_action_restore.operators import Action
from experiments.pamf_lora_main.dura_real_gain_amplitude import R17AmplitudeGain


def test_zero_initialization_preserves_prior_and_all_real_gain_directions_train():
    torch.manual_seed(14)
    y = torch.randn(2, 256, dtype=torch.complex64)
    prior, r17 = .7 * y, -.2 * y
    action = Action(torch.ones(2, 1, dtype=y.dtype), torch.ones_like(y), torch.zeros_like(y))
    model = R17AmplitudeGain(32)
    output = model(y, prior, r17, action)
    torch.testing.assert_close(output, prior, rtol=0, atol=0)
    output.abs().square().mean().backward()
    gradients = model.net[-1].weight.grad.flatten(1).norm(dim=1)
    assert torch.isfinite(gradients).all() and (gradients > 0).all()
