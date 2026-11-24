import torch

from experiments.pamf_lora_main.spectral_covariance import SharedSpectralPrior
from experiments.pamf_lora_main.transported_covariance import (
    ac, inner, native_psd, noise_action, signal_action, solve, transported_diagonal,
)


def example():
    rng = torch.Generator().manual_seed(13)
    data = torch.randn(2, 6, 256, generator=rng, dtype=torch.complex64)
    reference = torch.randn(35, 6, 256, generator=rng, dtype=torch.complex64)
    phase = torch.linspace(-4, 9, 256).square()
    rotation = torch.polar(torch.ones_like(phase), phase).expand_as(data)
    marginal = .75 + torch.rand(data.shape, generator=rng)
    prior = SharedSpectralPrior()
    prior.fit(reference)
    return ac(data), rotation, marginal, prior


def test_transport_marginals_against_dense_basis_and_white_limit():
    size = 16
    phase = torch.linspace(-.5, 1.7, size).square() * 4
    rot = torch.polar(torch.ones_like(phase), phase)[None, None]
    psd = torch.linspace(.4, 1.6, size)[None, None]
    basis = torch.eye(size, dtype=torch.complex64)[:, None]
    dense = noise_action(basis, rot, psd, torch.ones_like(psd))
    expected = dense[:, 0].diagonal().real[1:]
    torch.testing.assert_close(transported_diagonal(rot, psd).flatten()[1:], expected)
    white = torch.ones_like(psd)
    torch.testing.assert_close(noise_action(basis, rot, white, white), ac(basis))
    # AC subspace cross-terms are Hermitian and non-trivial.
    subspace = dense[1:, 0, 1:]
    torch.testing.assert_close(subspace, subspace.mH)
    assert (subspace - torch.diag_embed(subspace.diagonal())).abs().max() > .02


def test_real_prior_and_flat_transport_agree_with_frozen_posteriors():
    residual, rotation, marginal, prior = example()
    original = prior.innovation(residual, marginal, True)
    reconstructed = signal_action((residual - original) / marginal, prior)
    torch.testing.assert_close(reconstructed, original, atol=2e-6, rtol=3e-5)
    transported, error = solve(residual, prior, marginal, rotation, torch.ones_like(marginal))
    torch.testing.assert_close(transported, original, atol=2e-6, rtol=3e-5)
    assert float(error.max()) < 1e-6


def test_colored_solver_reduces_actual_covariance_system_residual():
    residual, rotation, marginal, prior = example()
    psd = native_psd(residual, rotation)
    assert psd.shape == residual.shape and psd.isfinite().all() and (psd > 0).all()
    assert not torch.allclose(psd, torch.ones_like(psd))
    torch.testing.assert_close(native_psd(.4 * residual, rotation), psd)
    initial, initial_error = solve(residual, prior, marginal, rotation, psd, steps=0)
    transported, error = solve(residual, prior, marginal, rotation, psd)
    assert (error < .03 * initial_error).all() and float(error.max()) < .02
    assert transported.isfinite().all() and not transported[..., 0].count_nonzero()
    assert inner(transported - initial, transported - initial).min() > 0
