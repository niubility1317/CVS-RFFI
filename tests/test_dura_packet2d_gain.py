import torch

from experiments.pamf_action_restore.operators import Action
from experiments.pamf_lora_main.dura_real_gain_packet2d import PacketR17Gain


def test_packet2d_starts_as_identity_and_trains_all_real_directions():
    torch.manual_seed(23)
    for width in (32, 1024):
        model = PacketR17Gain(width)
        observed = torch.randn(1, 8 * width, dtype=torch.complex64)
        prior = .8 * observed
        proposal = .6 * observed + .1 * torch.randn_like(observed)
        action = Action(torch.ones(1, 1, dtype=observed.dtype),
                        torch.ones_like(observed), torch.zeros_like(observed))
        output = model(observed, prior, proposal, action)
        torch.testing.assert_close(output, prior, rtol=1e-5, atol=1e-6)
        output.abs().square().mean().backward()
        gradient = model.net[-1].weight.grad.flatten(1).norm(dim=1)
        assert torch.isfinite(gradient).all() and (gradient > 0).all()
