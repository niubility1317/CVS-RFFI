import torch

from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.phase_dose import PhaseDose
from experiments.pamf_lora_main.phase_radius_diagnostic import (
    CONTEXT, MODES, RECEIVER, radial_ratio, radius_path, render,
)


def test_radial_path_caps_actual_packet_edit_and_keeps_per_chirp_power():
    rng = torch.Generator().manual_seed(21856)
    amplitude = torch.rand((5, 6, 256), generator=rng) + .1
    guide = torch.rand(amplitude.shape, generator=rng) + .1
    limits = torch.tensor([0., .02, .04, .06, .08])[:, None, None]
    output = radius_path(amplitude, guide, limits)
    assert torch.equal(output[0], amplitude[0]) and (output >= 0).all()
    torch.testing.assert_close(output.norm(dim=-1), amplitude.norm(dim=-1), rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(radial_ratio(amplitude, output)[1:], limits[1:, 0, 0],
                               rtol=3e-4, atol=5e-6)


@torch.no_grad()
def test_radial_receiver_branches_are_isolated_from_true_noise():
    rng = torch.Generator().manual_seed(5161)
    iq = real_iq(torch.randn((3, 6, 256), dtype=torch.complex64, generator=rng))
    noise = .4 * torch.randn(iq.shape, generator=rng)
    tracker = PacketRestorer('consensus').eval()
    model = PhaseDose(True).eval()
    args = (lambda a: (a, None), tracker, (model,))
    evidence = []
    plain, count = render(iq, torch.zeros_like(iq), *args, threshold=0., measurements=evidence)
    diagnostic, active = render(iq, noise, *args, threshold=0.)
    cutoff = len(CONTEXT) + len(RECEIVER)
    assert plain.shape == (len(MODES), *iq.shape) and count == active == 3
    assert torch.equal(plain[:cutoff], diagnostic[:cutoff])
    assert not torch.equal(plain[cutoff:], diagnostic[cutoff:])
    match_power = CONTEXT.index('local_full')
    assert torch.equal(plain[MODES.index('exact_radius_all')], iq)
    assert torch.equal(plain[MODES.index('exact_joint_equal')], plain[match_power])
    comparison = MODES.index('power_equal'), MODES.index('fold_equal')
    movements = torch.stack([(plain[j] - plain[match_power]).flatten(1).norm(dim=1) for j in comparison])
    torch.testing.assert_close(movements[0], movements[1], rtol=3e-4, atol=3e-5)
