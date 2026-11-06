from unittest.mock import patch

import numpy as np
import pytest
import torch

from experiments.pamf_phase3_restoration import phase_mixture_source as probe


def fixture():
    rng = torch.Generator().manual_seed(37)
    a = torch.randn(2, 6, 6, generator=rng, dtype=torch.float64)
    p = torch.randn(6, 6, generator=rng, dtype=torch.float64)
    c = p @ p.T / 6 + torch.eye(6, dtype=torch.float64)
    y = torch.randn(2, 6, generator=rng, dtype=torch.float64)
    mean = torch.arange(6, dtype=torch.float64) * .3
    var = torch.tensor([.03, .8], dtype=torch.float64)
    rotation = torch.kron(torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64),
                          torch.eye(3, dtype=torch.float64))
    return a, y, mean, c, var, rotation


def test_shared_factor_mixture_matches_four_independent_gaussian_updates():
    a, y, mean, cov, var, j = fixture()
    within = .5 * (cov + j @ cov @ j.T)
    results, likelihoods = [], []
    for k in range(4):
        center = torch.linalg.matrix_power(j, k) @ mean
        observed_mean = a @ center
        sigma = a @ within @ a.transpose(-1, -2) + torch.diag_embed(var[:, None].expand(-1, 6))
        innovation = y - observed_mean
        likelihoods.append(torch.distributions.MultivariateNormal(observed_mean, sigma).log_prob(y))
        posterior = center + (within @ a.transpose(-1, -2) @
                              torch.linalg.solve(sigma, innovation[:, :, None])).squeeze(-1)
        results.append(posterior)
    expected_weights = torch.stack(likelihoods, dim=1).softmax(dim=1)
    expected = (torch.stack(results, dim=2) @ expected_weights[:, :, None]).squeeze(-1)
    with patch.object(torch.linalg, 'cholesky', wraps=torch.linalg.cholesky) as cholesky:
        mixed, anchor, weights = probe.phase_mixture_posterior(a, y, mean, within, var)
        assert cholesky.call_count == 1
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(mixed, expected)
    torch.testing.assert_close(anchor, results[0])
    assert (mixed-anchor).abs().max() > .01


def test_mixture_rotates_back_with_nontrivial_action_and_shifted_weights():
    a, y, mean, cov, var, j = fixture()
    within = probe.shared_covariance(cov)
    torch.testing.assert_close(within, .5*(cov + j @ cov @ j.T))
    mixed, _, weights = probe.phase_mixture_posterior(a, y, mean, within, var)
    rotated, _, rotated_weights = probe.phase_mixture_posterior(j @ a @ j.T, y @ j.T, mean, within, var)
    torch.testing.assert_close(rotated, mixed @ j.T)
    torch.testing.assert_close(rotated_weights, weights.roll(1, dims=1))


def test_uniform_gaussian_prior_has_same_unconditional_moments():
    _, _, mean, cov, _, j = fixture()
    c = probe.shared_covariance(cov)
    means = torch.stack([torch.linalg.matrix_power(j,k) @ mean for k in range(4)])
    gaussian_mean, gaussian_covariance = probe.coordinate_prior(mean, cov)
    torch.testing.assert_close(gaussian_mean, means.mean(0))
    torch.testing.assert_close(gaussian_covariance, c + means.T @ means / 4)


def test_source_panel_is_fixed_and_rejects_wrong_strata():
    rx = np.repeat([1,3,4,6,8], 360)
    day = np.tile(np.repeat([1,2,3], 120), 5)
    np.testing.assert_array_equal(probe.source_indices(rx, day), (np.arange(0,1800,120)[:,None] + np.arange(8)).ravel())
    with pytest.raises(ValueError):
        probe.source_indices(rx[1:], day[1:])
    day[-1] = 4
    with pytest.raises(ValueError):
        probe.source_indices(rx, day)
