"""Exact posterior mean for joint observation phase and clean waveform modes."""

import torch


def joint_phase_mean(action, observed, mean, within, offsets, weights, variance, phase, log_prior):
    """Share one Cholesky factor across all phase and waveform hypotheses."""
    centers = mean + offsets
    ac = action @ within
    covariance = ac @ action.transpose(-1, -2)
    covariance.diagonal(dim1=-2, dim2=-1).add_(variance[:, None])
    factor = torch.linalg.cholesky(covariance)
    i, q = observed.chunk(2, dim=1)
    quarter_turn = torch.cat((-q, i), dim=1)
    rhs = torch.cat((observed[:, :, None], quarter_turn[:, :, None], action @ centers.T), dim=2)
    whitened = torch.linalg.solve_triangular(factor, rhs, upper=False)
    p, jp, anchor = whitened.split((1, 1, len(offsets)), dim=2)
    c, s = phase.cos(), phase.sin()
    observation_energy = (p.square().sum(1)*c.square() + jp.square().sum(1)*s.square()
                          - 2*(p*jp).sum(1)*c*s)
    center_energy = anchor.square().sum(1)
    inner = (p*anchor).sum(1)[:, :, None]*c - (jp*anchor).sum(1)[:, :, None]*s
    scores = inner - .5*(observation_energy[:, None, :] + center_energy[:, :, None])
    scores = scores + weights.log()[None, :, None] + log_prior
    posterior = scores.flatten(1).softmax(dim=1).reshape_as(scores)
    mode_probability = posterior.sum(2)
    phase_moments = posterior.sum(1) @ torch.stack((c, s), dim=1)
    expected_innovation = (p*phase_moments[:, None, :1] - jp*phase_moments[:, None, 1:]
                           - anchor @ mode_probability[:, :, None])
    correction = torch.linalg.solve_triangular(factor.transpose(-1, -2), expected_innovation, upper=True)
    restored = mode_probability @ centers + (ac.transpose(-1, -2) @ correction).squeeze(-1)
    return restored, mode_probability, phase_moments
