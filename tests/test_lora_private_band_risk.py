import torch

from experiments.pamf_lora_main.private_band_risk_source import (
    band_inner,
    minimizing_dose,
    stein_trace,
)


def test_trace_matches_weighted_complex_diagonal_jacobian():
    generator = torch.Generator().manual_seed(36)
    spectrum = torch.randn(3, 6, 9, dtype=torch.complex128, generator=generator)
    slope = torch.rand(3, 6, 9, dtype=torch.float64, generator=generator) - .65
    variance = torch.rand(3, 6, 9, dtype=torch.float64, generator=generator)
    weights = torch.rand(3, 6, 1, dtype=torch.float64, generator=generator)
    masks = torch.eye(3, dtype=torch.float64).repeat_interleave(3, dim=1)
    trace = stein_trace(lambda y: y * slope, spectrum, variance, weights, masks, generator)
    exact = (variance * slope * weights).sum(1) @ masks.T
    torch.testing.assert_close(trace, exact, rtol=1e-10, atol=1e-10)


def test_band_quadratic_matches_waveform_errors_for_mixed_doses():
    gen = torch.Generator().manual_seed(56)
    base = torch.randn(2, 6, 8, dtype=torch.complex64, generator=gen)
    edit = torch.randn(2, 6, 8, dtype=torch.complex64, generator=gen)
    weight = torch.rand(2, 6, 1, generator=gen)
    mask = torch.eye(2).repeat_interleave(4, dim=1)
    dose = torch.tensor([[0, .25], [.5, 1]])
    residual = base + edit * (dose @ mask)[:, None, :]
    actual = band_inner(residual, residual, weight, mask)
    expected = (band_inner(base, base, weight, mask)
                + 2 * dose * band_inner(base, edit, weight, mask)
                + dose.square() * band_inner(edit, edit, weight, mask))
    torch.testing.assert_close(actual, expected)


def test_doses_are_bounded_and_the_zero_edit_stays_off():
    linear = torch.tensor([0., .1, -.2, -1.8])
    quadratic = torch.tensor([0., 1., .8, 1.])
    torch.testing.assert_close(minimizing_dose(linear,quadratic), torch.tensor([0.,0.,.25,1.]))
