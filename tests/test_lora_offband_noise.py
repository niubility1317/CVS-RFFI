import torch

from experiments.pamf_lora_main.offband_noise import conditional_noise, covariance_action, received_profile


def inputs():
    g = torch.Generator().manual_seed(276)
    size = 32
    angle = torch.randn((4, 3, size), generator=g)
    rotation = torch.polar(torch.ones_like(angle), angle)
    measured = torch.randn(rotation.shape, generator=g, dtype=torch.complex64)
    density = torch.linspace(.3, 2., size).expand(rotation.shape)
    keep = torch.arange(size) % 3 != 0
    return measured, rotation, density, keep


def test_flat_rf_noise_cannot_be_predicted_outside_witness():
    x, rot, density, keep = inputs()
    prediction = conditional_noise(x, rot, torch.ones_like(density), keep, steps=16)
    assert prediction[..., ~keep].abs().max() < 4e-7
    assert prediction[..., keep].abs().max() == 0.


def test_cg_prediction_matches_complex_dense_conditional_mean():
    x, rot, density, keep = inputs()
    x, rot, density = (v[0, 0] for v in (x, rot, density))
    ridge = .05
    spectrum = covariance_action(torch.eye(len(x), dtype=x.dtype), rot, density)
    matrix = spectrum.T.contiguous()
    chosen = matrix[keep][:, keep] + ridge * torch.eye(int(keep.sum()))
    expected = matrix[~keep][:, keep] @ torch.linalg.solve(chosen, x[keep])
    got = conditional_noise(x, rot, density, keep, ridge=ridge, steps=40)
    torch.testing.assert_close(got[~keep], expected, rtol=3e-4, atol=5e-6)
    deleted = x.clone()
    deleted[~keep] *= -123.
    torch.testing.assert_close(got, conditional_noise(deleted, rot, density, keep, ridge=ridge, steps=40))


def test_received_profile_stays_finite_and_does_not_read_protected_bins():
    x, rot, _, keep = inputs()
    fixed = received_profile(x, rot, keep, share=.5)
    changed = x.clone()
    changed[..., ~keep] *= 91.
    torch.testing.assert_close(fixed, received_profile(changed, rot, keep, share=.5))
    assert torch.isfinite(fixed).all() and fixed.min() > 0.
    torch.testing.assert_close(fixed.mean(-1), torch.ones_like(fixed[..., 0]))
    quiet = received_profile(torch.zeros_like(x), rot, keep, share=.5)
    torch.testing.assert_close(quiet, torch.ones_like(quiet))


def test_subtracting_the_signal_reduces_the_reported_signal_gain():
    from experiments.pamf_lora_main.offband_noise_source import measurements

    signal = torch.ones((2, 6, 4), dtype=torch.complex64)
    mask = torch.ones(4, dtype=torch.bool)
    result = measurements(signal, torch.zeros_like(signal), signal, signal.abs()[..., :1], mask, 2)
    assert result['signal_real_gain_shift_median'] == -1.
    assert result['signal_projection_abs_median'] == 1.
