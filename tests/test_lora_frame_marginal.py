import torch

from experiments.pamf_lora_main.frame_marginal import (
    marginal_readouts, multipliers, split_uncertainty,
)


def test_coordinate_points_preserve_physical_waveform():
    rng = torch.Generator().manual_seed(4)
    signal = torch.randn(3, 6, 256, dtype=torch.complex64, generator=rng)
    phase, slope = split_uncertainty(signal)
    points = multipliers(signal, phase, slope)
    torch.testing.assert_close(points[:, 1], points[:, 2].conj())
    decoded = signal[:, None] * points.conj() * points
    torch.testing.assert_close(decoded, signal[:, None].expand_as(decoded))


def test_interleaved_pure_tone_has_negligible_disagreement():
    t = torch.linspace(-1, 1, 256)
    rot = torch.polar(torch.ones_like(t), .22 + .31 * t)[None, None].expand(3, 6, -1)
    phase, slope = split_uncertainty(rot)
    assert float(phase.abs().max()) < 1e-5 and float(slope.abs().max()) < 1e-5


def test_zero_uncertainty_recovers_nominal_for_carrier_free_edit():
    class BoundEdit:
        def __call__(self, wave, quality, rotation):
            change = wave - wave.mean(-1, keepdim=True)
            return wave + .2 * change, wave + .1 * change

    rng = torch.Generator().manual_seed(5)
    wave = torch.randn(3, 6, 256, generator=rng, dtype=torch.complex64)
    q = torch.ones_like(wave.real[..., :1])
    z = torch.zeros_like(q)
    variants, _ = marginal_readouts(wave, torch.ones_like(wave), q, q, BoundEdit(), None,
                                    disagreement=(z, z))
    for item in variants[1:]:
        torch.testing.assert_close(item, variants[0], atol=1e-6, rtol=1e-5)
