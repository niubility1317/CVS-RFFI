"""Phase estimation from a coherent IQ mean with fixed observation covariance."""

import torch


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


def coherent_phase(action, main_action, observed, mean, covariance, variance, phase, log_prior):
    """Compare equal-trace scalar and full fixed covariances for the same mean."""
    normal = (action @ covariance) @ action.transpose(-1,-2)
    normal.diagonal(dim1=-2,dim2=-1).add_(variance[:,None])
    main = main_action @ mean
    image = action @ mean - main
    basis = torch.stack((observed-image,main,rotate_anchor(main)),dim=2)
    trace_per_dim = normal.diagonal(dim1=-2,dim2=-1).mean(1)
    isotropic = basis / trace_per_dim.sqrt()[:,None,None]
    full = torch.linalg.solve_triangular(torch.linalg.cholesky(normal),basis,upper=False)
    direction = torch.stack((phase.cos(),phase.sin()),dim=1)
    return tuple(torch.complex(*((phase_weights(basis,phase,log_prior) @ direction).unbind(1)))
                 for basis in (isotropic,full))
