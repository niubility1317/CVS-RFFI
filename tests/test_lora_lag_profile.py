import torch

from experiments.pamf_lora_main.lag_profile import lag_profile, lag_profile_risk


def test_profile_is_invariant_to_packet_phase_and_constant_frequency_shift():
    rng = torch.Generator().manual_seed(45)
    wave = 1 + .15 * torch.randn(3, 6, 256, generator=rng, dtype=torch.complex64)
    angle = torch.arange(256) * .17 + 2.4
    oscillator = torch.polar(torch.ones_like(angle), angle)
    torch.testing.assert_close(lag_profile(wave * oscillator), lag_profile(wave),
                               atol=2e-6, rtol=1e-5)
    assert lag_profile(oscillator).abs().max() < 2e-6


def test_lag_profiles_compare_actual_time_order_and_preserve_private_difference():
    t = torch.arange(256) / 256
    clean = (1 + .2 * torch.sin(2 * torch.pi * t))[None, None].expand(2, 6, -1).to(torch.complex64).clone()
    clean[:, 1::2] *= torch.exp(.25j * torch.cos(2 * torch.pi * t))
    perfect = clean[:, None].expand(-1, 2, -1, -1)
    assert not lag_profile_risk(perfect, clean).count_nonzero()
    collapsed = clean.mean(1, keepdim=True).expand_as(clean)
    assert float(lag_profile_risk(collapsed[:, None], clean)[1]) > .01
    moved = lag_profile_risk(perfect.roll(64, dims=-1), clean)
    assert (moved > 0).all()


def test_weak_coherent_support_has_finite_gradients():
    rng = torch.Generator().manual_seed(5)
    shape = (3, 2, 6, 256)
    prediction = (1e-5 * torch.randn(shape, generator=rng, dtype=torch.complex64)).requires_grad_()
    reference = torch.zeros(3, 6, 256, dtype=torch.complex64)
    risk = lag_profile_risk(prediction, reference).sum()
    risk.backward()
    assert risk.isfinite() and prediction.grad.isfinite().all()
