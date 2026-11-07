"""Exact conditional-prior posterior with only a small differentiable solve.

The physical operator, noise, source covariance and source modes are fixed for
the prior loss. Both the conditional mean and the low-rank factor may learn.
All posterior arithmetic uses the real I/Q representation in FP64.
"""
import torch


def conditional_prior_posterior(action,observed,mean,covariance,modes,factor,variance):
    if any(value.requires_grad for value in (action,covariance,modes,variance)):
        raise ValueError('This prior-only solver requires fixed physical and source-prior inputs')
    action,observed,mean,covariance,modes,variance=(value.double() for value in (action,observed,mean,covariance,modes,variance))
    k=factor.double()
    with torch.no_grad():
        residual_covariance=covariance-modes@modes.transpose(-1,-2)
        ac=action@residual_covariance
        normal=ac@action.transpose(-1,-2)
        normal.diagonal(dim1=-2,dim2=-1).add_(variance[:,None])
        base_factor=torch.linalg.cholesky(normal)
        observed_modes=action@modes
        inverse_modes=torch.cholesky_solve(observed_modes,base_factor)
        small_precision=observed_modes.transpose(-1,-2)@inverse_modes
        posterior_modes=modes-ac.transpose(-1,-2)@inverse_modes
    innovation=(observed-(action@mean.unsqueeze(-1)).squeeze(-1))[:,:,None]
    solved=torch.cholesky_solve(innovation,base_factor)
    core=torch.eye(k.shape[-1],device=k.device,dtype=k.dtype)+k.transpose(-1,-2)@small_precision@k
    core=(core+core.transpose(-1,-2))*.5
    core_factor=torch.linalg.cholesky(core)
    rhs=k.transpose(-1,-2)@(observed_modes.transpose(-1,-2)@solved)
    coefficient=torch.cholesky_solve(rhs,core_factor)
    latent=k@coefficient
    restored=mean+(ac.transpose(-1,-2)@solved+posterior_modes@latent).squeeze(-1)
    with torch.no_grad():
        full_solution=solved-inverse_modes@latent
        error=normal@full_solution+observed_modes@(k@(k.transpose(-1,-2)@(observed_modes.transpose(-1,-2)@full_solution)))-innovation
        relative_error=error.squeeze(-1).norm(dim=1)/innovation.squeeze(-1).norm(dim=1).clamp_min(1e-30)
    if not torch.isfinite(restored).all() or not torch.isfinite(relative_error).all():
        raise ValueError('Nonfinite low-rank conditional posterior')
    return restored,relative_error
