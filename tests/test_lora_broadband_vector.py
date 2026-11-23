import torch

from experiments.pamf_lora_main.broadband_vector import CAP, BroadbandVector, band_risk, target_vectors
from experiments.pamf_lora_main.broadband_vector_source import MODES, readouts
from experiments.pamf_lora_main.complex_vector import rms
from experiments.pamf_lora_main.dura_repair import PacketRestorer


def test_broadband_supervision_uses_all_noncarrier_bins_and_limits_whole_chirp():
    torch.manual_seed(17)
    raw = torch.randn(2, 6, 256, dtype=torch.complex64)
    expected = target_vectors(raw, -2.*raw)
    f = torch.fft.fftfreq(256).abs()
    for key, value in expected.items():
        assert torch.count_nonzero(value[..., f == 0]) == 0
        torch.testing.assert_close(rms(value), torch.full_like(rms(value), CAP))
        assert bool(torch.count_nonzero(value[..., f > 1/8])) == (key == 'broad')
        model = BroadbandVector(wide=key == 'broad')
        proposal = model(raw, raw, torch.ones(2, 6, 1))
        band_risk(proposal, value, key == 'broad').backward()
        assert all(m.out.weight.grad.norm() > 0 for m in (model.spectral, model.temporal))


def test_only_source_oracles_change_when_true_noise_is_replaced():
    torch.manual_seed(6)
    sample = torch.randn(2, 6, 2, 256)
    epsilon = .05*torch.randn_like(sample)

    def baseline(x):
        return .8*x, torch.tensor([True, False])

    tracker = PacketRestorer('consensus').eval()
    arms = (BroadbandVector(wide=False).eval(), BroadbandVector(wide=True).eval())
    values, active = readouts(sample, epsilon, baseline, tracker, arms)
    contrasts, _ = readouts(sample, -epsilon, baseline, tracker, arms)
    assert active == 1 and values.shape == (len(MODES), *sample.shape)
    assert torch.equal(values[:6], contrasts[:6])
    assert not torch.equal(values[6:], contrasts[6:])
    torch.testing.assert_close(values[1:, 1], (.8*sample[1]).expand_as(values[1:, 1]))
