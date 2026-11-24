import torch

from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.phase_observation_diagnostic import (
    CONTEXT, RECEIVED, MODES, phase_budget, phase_ratio, render, repeat_angle,
)
from experiments.pamf_lora_main.received_update import ReceivedUpdate
from experiments.pamf_lora_main.received_update_source import recover


def test_phase_budget_matches_physical_edit_without_changing_amplitude():
    generator = torch.Generator().manual_seed(741)
    received = torch.randn((4, 6, 256), dtype=torch.complex64, generator=generator)
    angle = 6.28 * torch.rand(received.shape, generator=generator) - 3.14
    angle[0].zero_()
    target = torch.tensor([.10, .025, .10, .23]).reshape(-1, 1, 1)
    output = phase_budget(received, angle, target)
    torch.testing.assert_close(output.abs(), received.abs(), atol=4e-7, rtol=2e-7)
    assert torch.equal(output[0], received[0])
    torch.testing.assert_close(phase_ratio(received, output)[1:], target[1:, 0, 0],
                               rtol=1e-4, atol=3e-6)


def test_repeat_angle_is_current_packet_based_and_zero_for_perfect_repeats():
    rng = torch.Generator().manual_seed(8195)
    a = torch.randn((3, 1, 256), dtype=torch.complex64, generator=rng).expand(-1, 6, -1)
    quality = torch.full((3, 6, 1), .7)
    assert repeat_angle(a, quality).abs().max() < 1e-6
    changed = a.clone()
    changed[1, 0, 17:31] *= 1j
    output = repeat_angle(changed, quality)
    assert output[1, 0, 17:31].abs().max() > .1
    assert output[[0, 2]].abs().max() < 1e-6


@torch.no_grad()
def test_exact_noise_changes_only_the_isolated_rows():
    generator = torch.Generator().manual_seed(729)
    iq = real_iq(torch.randn((3, 6, 256), dtype=torch.complex64, generator=generator))
    noise = .25 * torch.randn(iq.shape, generator=generator)
    tracker = PacketRestorer('consensus').eval()
    model = ReceivedUpdate(False).eval()
    measurements = []
    args = (lambda x: (x, None), tracker, (model,))
    plain, active = render(iq, torch.zeros_like(iq), *args, threshold=0., measurements=measurements)
    diagnosed, second = render(iq, noise, *args, threshold=0.)
    cutoff = len(CONTEXT) + len(RECEIVED)
    assert plain.shape == (len(MODES), *iq.shape) and active == second == len(iq)
    assert measurements[0].shape == (3, 13)
    inherited, _, _ = recover(iq, tracker, (model,), 0.)
    assert torch.equal(plain[2:4], inherited)
    assert torch.equal(plain[:cutoff], diagnosed[:cutoff])
    assert torch.equal(plain[cutoff:], iq.expand_as(plain[cutoff:]))
    assert not torch.equal(plain[cutoff:], diagnosed[cutoff:])
