"""Phase estimation from a coherent IQ mean with fixed observation covariance."""

import torch

from .covariance_inverse import gaussian_posterior
from .phase_ramp_posterior import calibrated_action


def rotate_anchor(main):
    i, q = main.chunk(2, dim=1)
    return torch.cat((-q,i),dim=1)


def phase_weights(whitened, phase, log_prior):
    """Score fixed phase candidates after whitening [y-image, main, j*main]."""
    observed,main,orthogonal = whitened.unbind(2)
    c,s = phase.cos(),phase.sin()
    inner = (observed*main).sum(1)[:,None]*c + (observed*orthogonal).sum(1)[:,None]*s
    energy = (main.square().sum(1)[:,None]*c.square() + orthogonal.square().sum(1)[:,None]*s.square()
              + 2*(main*orthogonal).sum(1)[:,None]*c*s)
    return (inner-.5*energy+log_prior).softmax(dim=1)


def phase_basis(action, main_action, observed, mean, covariance, variance):
    normal = (action @ covariance) @ action.transpose(-1,-2)
    normal.diagonal(dim1=-2,dim2=-1).add_(variance[:,None])
    main = main_action @ mean
    image = action @ mean - main
    return normal, torch.stack((observed-image,main,rotate_anchor(main)),dim=2)


def phase_direction(whitened, phase, log_prior):
    direction = torch.stack((phase.cos(),phase.sin()),dim=1)
    return torch.complex(*((phase_weights(whitened,phase,log_prior) @ direction).unbind(1)))


def full_phase_direction(normal, basis, phase, log_prior):
    full = torch.linalg.solve_triangular(torch.linalg.cholesky(normal),basis,upper=False)
    return phase_direction(full,phase,log_prior)


def coherent_phase(action, main_action, observed, mean, covariance, variance, phase, log_prior):
    """Compare equal-trace scalar and full fixed covariances for the same mean."""
    normal,basis = phase_basis(action,main_action,observed,mean,covariance,variance)
    trace_per_dim = normal.diagonal(dim1=-2,dim2=-1).mean(1)
    isotropic = basis / trace_per_dim.sqrt()[:,None,None]
    return phase_direction(isotropic,phase,log_prior), full_phase_direction(normal,basis,phase,log_prior)


def restore_main_direction(main_action, image_action, observed, mean, covariance, variance, direction):
    angle = direction.angle()
    rotation = torch.polar(torch.ones_like(angle),angle)
    action = calibrated_action(main_action,rotation[:,None].expand(-1,observed.shape[1]//2))+image_action
    restored,residual = gaussian_posterior(action,observed,mean,covariance,variance)
    if residual.max() > 1e-8:
        raise ValueError('Main-only posterior residual exceeds 1e-8')
    return restored,angle
