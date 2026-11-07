import numpy as np
import torch

from experiments.pamf_phase3_restoration.covariance_inverse import gaussian_posterior
from experiments.pamf_phase3_restoration.waveform_mixture_prior import fit_source_modes, mixture_posterior


def test_source_modes_match_original_moments():
    rng = np.random.default_rng(145)
    samples = rng.normal(size=(800, 8))
    samples[:250, 0] += 5
    samples[250:450, 1] -= 4
    mean = samples.mean(axis=0)
    covariance = np.cov(samples, rowvar=False, bias=True)
    covariance = .999 * covariance + .001 * np.eye(8) * covariance.trace() / 8
    offsets, within, weights = fit_source_modes(samples, mean, covariance)
    assert offsets.shape == (4, 8)
    assert (weights > 0).all()
    assert np.linalg.eigvalsh(within).min() > 0
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-14)
    np.testing.assert_allclose(weights @ offsets, 0, atol=1e-14)
    np.testing.assert_allclose(within + (offsets.T * weights) @ offsets, covariance, atol=1e-14)


def test_shared_solve_matches_independent_component_posteriors():
    rng = torch.Generator().manual_seed(27)
    def rand(*shape):
        return torch.randn(*shape, generator=rng, dtype=torch.float64)
    action, observed, mean = rand(3, 6, 6), rand(3, 6), rand(6)
    factor = rand(6, 6)
    within = factor @ factor.T + .2 * torch.eye(6, dtype=torch.float64)
    weights = torch.tensor([.15, .24, .1, .51], dtype=torch.float64)
    offsets = rand(4, 6)
    variance = torch.tensor([.1, .9, 2.2], dtype=torch.float64)
    result, probabilities = mixture_posterior(action, observed, mean, within, offsets, weights, variance)
    obs_cov = action @ within @ action.transpose(-1, -2) + variance[:, None, None] * torch.eye(6, dtype=torch.float64)
    posterior_means, log_weights = [], []
    for offset, weight in zip(offsets, weights):
        mode = mean + offset
        posterior_means.append(gaussian_posterior(action, observed, mode, within, variance)[0])
        distribution = torch.distributions.MultivariateNormal(action @ mode, obs_cov)
        log_weights.append(weight.log() + distribution.log_prob(observed))
    expected = torch.stack(log_weights, dim=1).softmax(dim=1)
    torch.testing.assert_close(probabilities, expected, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(result, (torch.stack(posterior_means, dim=1) * expected[:, :, None]).sum(dim=1),
                               rtol=1e-11, atol=1e-12)
