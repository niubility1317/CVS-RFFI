import torch

from experiments.pamf_repl_view_shuffle.phase_magnitude import candidates, paired_reference_diagnostic, weighted_phase


def fft(x):
    return torch.fft.fft(x, norm='ortho')


def test_phase_magnitude_exchange_preserves_the_intended_fourier_components():
    rng = torch.Generator().manual_seed(12)
    raw = torch.randn(3, 256, dtype=torch.complex64, generator=rng)
    restored = torch.randn(3, 256, dtype=raw.dtype, generator=rng) * .5
    y, x = fft(raw), fft(restored)
    out = candidates(raw, restored, torch.zeros(len(y)))
    torch.testing.assert_close(fft(out['fit_phase']), y.abs() * x / x.abs())
    torch.testing.assert_close(fft(out['fit_magnitude']), x.abs() * y / y.abs())
    torch.testing.assert_close(out['fit_rms'].norm(dim=-1), raw.norm(dim=-1))
    torch.testing.assert_close(fft(out['weighted_phase']).abs(), y.abs(), atol=1e-6, rtol=1e-5)


def test_guarded_phase_kernel_preserves_received_magnitude_and_backs_off_with_noise():
    rng = torch.Generator().manual_seed(9)
    raw = torch.randn(2, 256, dtype=torch.complex64, generator=rng)
    fit = raw * 1j
    noise = torch.tensor([0., 1e8])
    out = weighted_phase(raw, fit, noise)
    torch.testing.assert_close(out[0], fit[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[1], raw[1], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(out, candidates(raw, fit, noise)['weighted_phase'])
    silent = torch.zeros_like(fit)
    out = candidates(raw, silent, torch.zeros(len(raw)))
    torch.testing.assert_close(out['fit_phase'], raw)
    assert all(torch.isfinite(value).all() for value in out.values())


def test_paired_reference_diagnostic_has_a_clean_endpoint_and_separate_magnitude():
    raw = torch.randn(2, 64, dtype=torch.complex64, generator=torch.Generator().manual_seed(9))
    clean = 2j * raw
    out = paired_reference_diagnostic(raw, clean)
    assert torch.equal(out['paired_clean'], clean)
    torch.testing.assert_close(out['reference_phase'], raw * 1j)
    torch.testing.assert_close(out['reference_magnitude'], 2 * raw)
