import torch

from experiments.pamf_lora_main.cross_frequency import CrossFrequencyPrior
from experiments.pamf_lora_main.variance_transport import physical_norm


def test_same_bin_covariances_and_equal_physical_dose():
    model = CrossFrequencyPrior()
    generator = torch.Generator().manual_seed(53)
    wave = torch.randn(167, 6, 256, dtype=torch.complex64, generator=generator)
    freq = torch.fft.fft(wave, norm='ortho')
    freq[..., 12] = .7j * freq[..., 3] + .15 * freq[..., 12]
    wave = torch.fft.ifft(freq, norm='ortho')
    model.calibrate(wave.split(35))

    joint = (model.joint_axes * model.joint_variance[:, None, :]) @ model.joint_axes.mT
    diagonal = joint.reshape(2, 255, 2, 255, 2).diagonal(dim1=1, dim2=3).movedim(-1, 1)
    blocks = (model.bin_axes * model.bin_variance[:, :, None, :]) @ model.bin_axes.mT
    torch.testing.assert_close(diagonal, blocks, rtol=3e-5, atol=3e-6)
    assert joint[0, 4, 23].abs() > .1

    samples = wave[:4]
    rms = torch.linspace(.5, 1.5, 6)[None, :, None]
    edits, noise = model.changes(samples, rms)
    torch.testing.assert_close(edits.mean(-1), torch.zeros_like(edits.mean(-1)), rtol=0., atol=1e-7)
    assert torch.isfinite(edits).all() and (noise > 0).all()
    doses = physical_norm(edits, rms)
    torch.testing.assert_close(doses[4], doses[5], rtol=1e-5, atol=1e-7)
    assert (doses <= .30001 * physical_norm(samples, rms)).all()
    assert (edits[4] - edits[5]).abs().mean() > .001


def test_bypassed_iq_does_not_read_any_prior_state():
    from experiments.pamf_lora_main.cross_frequency_source import recover
    from experiments.pamf_lora_main.dura_repair import PacketRestorer

    iq = torch.randn(2, 6, 2, 256, generator=torch.Generator().manual_seed(17))
    result, active, estimates = recover(iq, PacketRestorer('consensus').eval(), None, float('inf'))
    assert active == 0
    assert torch.equal(result, iq.expand_as(result))
    assert not estimates.any()
