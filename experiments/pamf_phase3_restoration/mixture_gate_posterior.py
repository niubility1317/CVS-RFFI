"""A common-covariance multi-hypothesis posterior and differentiable gate."""
import torch


def mixture_components(action, observed, mean, within, offsets, weights, variance):
    """Exact component posterior means and log evidence from one factorization.

    All inputs share dtype/device. Each row of offsets is a source-only prior
    mean offset; neither transmitter labels nor target information are inputs.
    """
    if (action.ndim != 3 or action.shape[1] != action.shape[2]
            or observed.shape != action.shape[:2] or mean.shape != action.shape[1:2]
            or within.shape != action.shape[1:] or offsets.ndim != 2
            or offsets.shape[1] != action.shape[-1] or weights.shape != offsets.shape[:1]
            or variance.shape != action.shape[:1]):
        raise ValueError('Aligned batched linear-Gaussian mixture required')
    if not all(torch.isfinite(v).all() for v in (action, observed, mean, within, offsets, weights, variance)):
        raise ValueError('Finite inputs required')
    if not (weights > 0).all() or not (variance > 0).all():
        raise ValueError('Positive mixture weights and noise variances required')
    ac = action @ within
    observation_covariance = ac @ action.transpose(-1, -2)
    observation_covariance.diagonal(dim1=-2, dim2=-1).add_(variance[:, None])
    centered = observed - action @ mean
    projected = action @ offsets.T
    rhs = torch.cat((centered[:, :, None], projected), dim=-1)
    solved = torch.cholesky_solve(rhs, torch.linalg.cholesky(observation_covariance))
    residual = (observation_covariance @ solved - rhs).norm(dim=1) / rhs.norm(dim=1).clamp_min(1e-30)
    if not torch.isfinite(residual).all() or residual.max() > 1e-8:
        raise ValueError('Mixture posterior solve residual exceeds 1e-8')
    filtered = ac.transpose(-1, -2) @ solved
    base = mean + filtered[:, :, 0]
    innovations = offsets.T - filtered[:, :, 1:]
    components = base[:, :, None] + innovations
    evidence = (projected * (solved[:, :, :1] - .5*solved[:, :, 1:])).sum(dim=1)
    return components, weights.log() + evidence, residual


def combine_components(components, evidence, adjustment):
    """Blend posterior candidates with a learned observation-only adjustment."""
    if (components.ndim != 3 or evidence.shape != (components.shape[0], components.shape[2])
            or adjustment.shape != evidence.shape):
        raise ValueError('Matched component posterior and gate logits required')
    probabilities = (evidence + adjustment.to(evidence)).softmax(dim=1)
    restored = (components @ probabilities[:, :, None]).squeeze(-1)
    return restored, probabilities
