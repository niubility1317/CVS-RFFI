import torch

from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.variance_transport import VarianceTransport, posterior_gain
from experiments.pamf_lora_main.variance_transport_source import MODES, recover


def test_covariance_gain_and_mode_basis():
    signal = torch.tensor([.03, .20, .50])
    noise = torch.tensor([.50, .01, .30])
    gain = posterior_gain(signal, noise)
    torch.testing.assert_close(gain * (signal + noise), signal)
    assert (gain.square() * (signal + noise) < signal).all()
    basis = VarianceTransport().basis
    torch.testing.assert_close(basis @ basis.mT, torch.eye(6), atol=1e-6, rtol=0)


def test_clean_bypass_weak_support_and_exact_physical_dose_match():
    rng = torch.Generator().manual_seed(74)
    tracker = PacketRestorer('consensus').eval()
    model = VarianceTransport()
    noise = torch.randn(120, 6, 256, dtype=torch.complex64, generator=rng)
    clean = real_iq((1. + .025 * noise) * tracker.reference.chirp)
    model.calibrate(coordinates(tracker, clean)[0])
    gains = .4 + torch.rand(4, 6, 1, generator=rng)
    disturbed = real_iq((.6 * noise[:4] + 1.) * tracker.reference.chirp * gains)
    iq = torch.cat((clean[-1:], disturbed))
    waves, n, measures = recover(iq, tracker, model, .03)
    assert n == 4 and waves.shape == (len(MODES) - 2, *iq.shape)
    assert torch.equal(waves[:, 0], iq[0].expand_as(waves[:, 0]))
    _, phase, _, rms = coordinates(tracker, iq)
    delta = torch.stack([complex_iq(wave - iq) for wave in waves])
    frequency = torch.fft.fft(delta / (phase * rms), norm='ortho')
    torch.testing.assert_close(frequency[..., ~model.weak], torch.zeros_like(frequency[..., ~model.weak]),
                               atol=1e-6, rtol=0)
    energy = delta.abs().square().sum((-1, -2))
    torch.testing.assert_close(energy[1], energy[2], atol=1e-5, rtol=1e-6)
    assert (measures[:, 3:] <= .100001).all()
