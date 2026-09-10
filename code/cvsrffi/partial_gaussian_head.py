"""Gaussian observed marginals; missing coordinates are never zero evidence."""
import math
import torch


def _inputs(z, means, observed, log_prior):
    if z.ndim != 2 or observed.shape != z.shape or observed.dtype != torch.bool:
        raise ValueError('z and boolean observed must have shape [B,D]')
    if means.ndim == 2:
        means = means.unsqueeze(0).expand(z.shape[0], -1, -1)
    if means.ndim != 3 or means.shape[0] != z.shape[0] or means.shape[2] != z.shape[1]:
        raise ValueError('means must have shape [C,D] or [B,C,D]')
    prior = z.new_zeros(means.shape[1]) if log_prior is None else log_prior
    if prior.shape != (means.shape[1],) or not torch.isfinite(prior).all():
        raise ValueError('log_prior must be finite [C]')
    return means, prior


def _chol(matrix):
    if not torch.isfinite(matrix).all() or not torch.allclose(matrix, matrix.transpose(-1, -2), rtol=1e-5, atol=1e-7):
        raise ValueError('observed covariance must be finite and symmetric')
    # No hidden jitter: callers must supply a positive definite covariance.
    return torch.linalg.cholesky(matrix)


def _result(mahal, logdet, count, prior):
    mahal, logdet = torch.stack(mahal), torch.stack(logdet)
    return dict(scores=-0.5 * (mahal + logdet + count[:, None] * math.log(2 * math.pi)) + prior,
                mahalanobis=mahal, logdet=logdet, observed_count=count)


def gaussian_scores(z, means, covariance, observed, log_prior=None):
    """Full normalized log density plus prior, selecting covariance BEFORE solve.

    Covariance is [B,D,D] (shared across classes) or [B,C,D,D].
    Masked values may be NaN. Nonmissing covariance must be positive definite.
    """
    means, prior = _inputs(z, means, observed, log_prior)
    b, c, d = means.shape
    if covariance.shape == (b, d, d):
        covariance = covariance[:, None].expand(-1, c, -1, -1)
    elif covariance.shape != (b, c, d, d):
        raise ValueError('covariance must be [B,D,D] or [B,C,D,D]')
    mahal, logdet = [], []
    for i in range(b):
        ix = observed[i].nonzero().flatten()
        residual = z[i].index_select(0, ix)[None] - means[i].index_select(1, ix)
        cov = covariance[i].index_select(1, ix).index_select(2, ix)
        if not ix.numel():
            zero = residual.sum(-1) + cov.sum((-1, -2))
            mahal.append(zero); logdet.append(zero)
            continue
        if not torch.isfinite(residual).all():
            raise ValueError('observed z and means must be finite')
        chol = _chol(cov)
        white = torch.linalg.solve_triangular(chol, residual[..., None], upper=False).squeeze(-1)
        mahal.append(white.square().sum(-1))
        logdet.append(2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1))
    return _result(mahal, logdet, observed.sum(-1), prior)


def lowrank_gaussian_scores(z, means, diagonal, factor, observed, log_prior=None):
    """Observed D+LL' marginal using a shared rank-r Woodbury solve per sample.

    diagonal: positive [B,D]; factor: [B,D,r]. Complexity avoids a D by D solve.
    """
    means, prior = _inputs(z, means, observed, log_prior)
    b, c, d = means.shape
    if diagonal.shape != (b, d) or factor.ndim != 3 or factor.shape[:2] != (b, d):
        raise ValueError('diagonal/factor must be [B,D]/[B,D,r]')
    mahal, logdet = [], []
    for i in range(b):
        ix = observed[i].nonzero().flatten()
        residual = z[i].index_select(0, ix)[None] - means[i].index_select(1, ix)
        diag = diagonal[i].index_select(0, ix)
        low = factor[i].index_select(0, ix)
        if not ix.numel():
            zero = residual.sum(-1) + diag.sum() + low.sum()
            mahal.append(zero); logdet.append(zero)
            continue
        if not torch.isfinite(residual).all() or not torch.isfinite(diag).all() or not (diag > 0).all() or not torch.isfinite(low).all():
            raise ValueError('observed inputs must be finite with positive diagonal')
        white = residual / diag.sqrt()
        low = low / diag.sqrt()[:, None]
        inner = torch.eye(low.shape[1], device=z.device, dtype=z.dtype) + low.T @ low
        chol = _chol(inner)
        latent = torch.cholesky_solve((white @ low).T, chol).T
        # Equivalent to Woodbury quadratic, without subtracting large near-equal terms.
        mahal.append((white - latent @ low.T).square().sum(-1) + latent.square().sum(-1))
        ld = diag.log().sum() + 2 * chol.diagonal().log().sum()
        logdet.append(ld.expand(c))
    return _result(mahal, logdet, observed.sum(-1), prior)
