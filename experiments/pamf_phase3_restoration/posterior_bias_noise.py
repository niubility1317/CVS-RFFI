"""Separate observation-noise leakage and shrinkage in a fixed Gaussian inverse."""

import torch


def posterior_bias_noise(action, observed, clean, mean, covariance, variance):
    """Return original, exact noise-removed, and exact bias-removed IQ.

    Each intervention retains the original true operator, estimated prior, and
    noise variance in the Gaussian gain. The latter two use clean source IQ.
    """
    ac = action @ covariance
    normal = ac @ action.transpose(-1,-2)
    normal.diagonal(dim1=-2,dim2=-1).add_(variance[:,None])
    innovation = observed-action@mean
    realized_noise = observed-(action@clean[:,:,None]).squeeze(-1)
    rhs = torch.stack((innovation,realized_noise),dim=2)
    solved = torch.cholesky_solve(rhs,torch.linalg.cholesky(normal))
    error = (normal@solved-rhs).norm(dim=1)/rhs.norm(dim=1).clamp_min(1e-30)
    if not torch.isfinite(error).all() or error.max()>1e-8:
        raise ValueError('Bias/noise posterior solve residual exceeds 1e-8')
    correction,noise = (ac.transpose(-1,-2)@solved).unbind(2)
    original = mean+correction
    return original,original-noise,clean+noise
