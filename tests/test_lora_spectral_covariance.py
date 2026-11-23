import torch

from experiments.pamf_lora_main.spectral_covariance import residual_posterior, covariance_factors


def test_rank_four_posterior_matches_dense_heteroscedastic_conditioning():
    rng = torch.Generator().manual_seed(276)
    observed = torch.randn((2, 6, 9), generator=rng, dtype=torch.float64)
    noise = 0.01 + torch.rand(observed.shape, generator=rng, dtype=observed.dtype)
    private = 0.07 + torch.rand(9, generator=rng, dtype=observed.dtype)
    shared = 0.02 + torch.rand(9, generator=rng, dtype=observed.dtype)
    factors = 0.2 * torch.randn((9, 4), generator=rng, dtype=observed.dtype)
    repeats = len(observed[0])
    common_cov = torch.diag(shared) + factors @ factors.T
    prior = torch.kron(torch.ones((repeats, repeats), dtype=observed.dtype), common_cov)
    prior += torch.kron(torch.eye(repeats, dtype=observed.dtype).contiguous(), torch.diag(private))
    covariance = prior + torch.diag_embed(noise.flatten(1))
    reference = (prior @ torch.linalg.solve(covariance, observed.reshape(len(noise), -1, 1)))
    actual = residual_posterior(observed, noise, private, shared, factors)
    torch.testing.assert_close(actual, reference.reshape_as(actual))


def test_low_rank_components_retain_marginal_power_and_inactive_directions():
    u = torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64)
    diagonal = torch.tensor([.35, .8, .28], dtype=u.dtype)
    independent, factors = covariance_factors(torch.diag(diagonal) + u[:, None] * u[None], 3)
    torch.testing.assert_close(independent + factors.square().sum(-1), diagonal + u.square())
    assert bool((independent >= .05 * (diagonal + u.square()) - 1e-12).all())
    _, factors = covariance_factors(torch.diag(diagonal), 3)
    torch.testing.assert_close(factors, torch.zeros_like(factors), atol=1e-12, rtol=0)
