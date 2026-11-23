import torch

from experiments.pamf_lora_main.rf_noise_confidence import RFNoiseConfidence, repeatability
from experiments.pamf_lora_main.rf_noise_transport import FFT, RFNoiseTransport


def test_repeatability_requires_frequency_aligned_split_evidence():
    frequencies = torch.arange(FFT)
    peak = 1 + 3 * torch.exp(-.5 * ((frequencies - 16) / 2.0).square())
    profiles = torch.stack((torch.ones(FFT), peak, peak))
    partner = torch.stack((torch.ones(FFT), peak.roll(12), peak))
    confidence = repeatability(profiles, partner)
    torch.testing.assert_close(confidence[:2], torch.zeros(2))
    assert confidence[2] > .7


def test_phase_kernel_has_the_expected_signed_frequency_shift_and_mean():
    t = torch.arange(256) / FFT
    angle = 2 * torch.pi * 5 * t
    rotation = torch.polar(torch.ones_like(t), angle).expand(2, 6, -1)
    bins = torch.arange(FFT)
    profile = (1 + 3 * torch.exp(-.5 * ((bins - 12) / 2).square())).expand(2, 6, -1)
    scalar = profile.mean(dim=-1, keepdim=True)
    transport = RFNoiseConfidence(False).transport(profile, scalar, rotation)
    # Except for the reflected boundary windows, receiver peak 12 -> tracked bin 7.
    core = transport[:, :, 2:-2]
    assert (core.argmax(-1) == 7).all()
    torch.testing.assert_close(transport.mean(-1), scalar.expand_as(transport.mean(-1)))


def test_receiving_variance_matches_flat_and_carrier_is_unchanged():
    rng = torch.Generator().manual_seed(904)
    t = torch.arange(256) / 256
    angle = torch.pi * 256 * t.square()
    rotation = torch.polar(torch.ones_like(t), angle).expand(2, 6, -1)
    y = .65 + .2 * torch.randn(rotation.shape, generator=rng, dtype=torch.complex64)
    quality = torch.full((2, 6, 1), .7)
    candidate = RFNoiseConfidence(True).eval()
    estimate, confidence = candidate.estimate_with_confidence(y, rotation)
    flat = RFNoiseTransport(False).estimate_noise(y, rotation)
    torch.testing.assert_close(estimate.mean(-1), flat.mean(-1))
    assert ((0 <= confidence) & (confidence <= 1)).all()
    for u in candidate(y, quality, rotation):
        torch.testing.assert_close(u.mean(-1), y.mean(-1))
        assert torch.isfinite(u).all()
