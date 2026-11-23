import torch

from experiments.pamf_lora_main.rf_noise_transport import FFT, RFNoiseTransport, colored_observation


def test_native_psd_is_warped_and_matches_flat_mean_power():
    rng = torch.Generator().manual_seed(45)
    t = torch.arange(256) / 256
    angle = torch.pi * 256 * t.square()
    rotation = torch.polar(torch.ones_like(t), angle).expand(3, 6, -1)
    additive = .2 * torch.polar(torch.ones_like(t), 2 * torch.pi * 64 * t)
    additive = additive + .01 * torch.randn(rotation.shape, generator=rng, dtype=torch.complex64)
    observed = 1 + additive * rotation.conj()
    model = RFNoiseTransport(True)
    aligned = model.estimate_noise(observed, rotation)
    flat = RFNoiseTransport(False).estimate_noise(observed, rotation)
    torch.testing.assert_close(aligned.mean(-1), flat.mean(-1))
    assert not torch.allclose(aligned, flat)
    ridge = model.analyze(rotation).abs().argmax(-1)
    native_estimate = (aligned.argmax(-1) + ridge) % FFT
    error = (native_estimate - 8 + FFT // 2) % FFT - FFT // 2
    assert (error.abs() <= 1).float().mean() > .8
    assert aligned.argmax(-1).float().std(dim=-1).min() > 4


def test_joint_rf_prior_preserves_carrier_and_updates_both_heads():
    rng = torch.Generator().manual_seed(20)
    shape = (3, 6, 256)
    t = torch.arange(256) / 256
    rotation = torch.polar(torch.ones_like(t), torch.pi * 256 * t.square()).expand(shape)
    clean = .4 + .1 * torch.randn(shape, generator=rng, dtype=torch.complex64)
    observed = colored_observation(clean * rotation, rng) * rotation.conj()
    quality = torch.full((3, 6, 1), .7)
    for aligned in (True, False):
        model = RFNoiseTransport(aligned)
        stages = model(observed, quality, rotation)
        for result in stages:
            assert torch.isfinite(result).all()
            torch.testing.assert_close(result.mean(-1), observed.mean(-1))
        sum((u-clean).abs().square().mean() for u in stages).backward()
        for head in (model.coarse, model.fine):
            grad = head[-1].weight.grad
            assert torch.isfinite(grad).all() and grad.norm() > 0
