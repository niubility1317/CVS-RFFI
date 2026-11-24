"""A single-observation diagonal-plus-rank-four spectral update."""

import torch


def posterior_increment(residual, diagonal, factors, noise):
    """Return covariance @ inverse(covariance + noise I) @ residual."""
    precision = (diagonal + noise).reciprocal()
    weighted = precision.unsqueeze(-1) * factors
    system = torch.eye(factors.shape[-1], device=residual.device, dtype=residual.dtype)
    system = system + factors.T @ weighted
    rhs = (factors.T @ (precision * residual).unsqueeze(-1))
    inverse_residual = precision * residual - (weighted @ torch.linalg.solve(system, rhs)).squeeze(-1)
    return residual - noise * inverse_residual
