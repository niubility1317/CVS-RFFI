import torch

from experiments.pamf_lora_main.dura_repair import PacketRestorer
from experiments.pamf_lora_main.receive_frame_action import delay, features, fit, restore_pair
from experiments.pamf_lora_main.receive_frame_action_source import render


def test_channel_action_transports_conjugation_and_delay_on_both_grids():
    rng = torch.Generator().manual_seed(42)
    for n in (256, 384):
        tracked = torch.randn((3, 6, n), dtype=torch.complex64, generator=rng)
        phase = torch.polar(torch.ones_like(tracked.real), torch.randn(tracked.shape, generator=rng))
        coefficients = torch.randn((3, 6, 1, 3), dtype=tracked.dtype, generator=rng)
        a, b, c = coefficients.unbind(-1)
        raw = phase*tracked
        expected = (a*raw+b*raw.conj()+c*delay(raw))*phase.conj()
        actual = (features(tracked, phase, raw_frame=True)*coefficients).sum(-1)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_identical_bypass_for_direct_and_composed_controls():
    tracker = PacketRestorer('consensus')
    iq = torch.randn((2, 6, 2, 256), generator=torch.Generator().manual_seed(21))
    a, b, active = restore_pair(iq, tracker, 1.)
    assert a is iq and b is iq and not active.any()

    def expanded(wave):
        return wave*.8

    outputs, count = render(iq, lambda x: (x*.9, None), tracker, expanded, 1.)
    assert count == 0 and torch.equal(outputs[0], outputs[3]) and torch.equal(outputs[0], outputs[4])
    assert torch.equal(outputs[2], outputs[5]) and torch.equal(outputs[2], outputs[6])


def test_relative_bound_preserves_zero_mean_across_chirps():
    design = torch.ones((1, 6, 32, 1), dtype=torch.complex64)
    intended = torch.tensor([.8, -.16, -.16, -.16, -.16, -.16]).view(1, 6, 1)
    correction = fit(design, (intended*design[..., 0]), torch.arange(32), relative=True)
    torch.testing.assert_close(correction.abs().max(), torch.tensor(.55))
    torch.testing.assert_close(correction.mean(1), torch.zeros_like(correction.mean(1)), atol=1e-7, rtol=0)
