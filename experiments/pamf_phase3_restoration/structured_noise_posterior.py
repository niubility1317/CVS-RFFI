"""Full per-sample I/Q noise blocks and a Gaussian posterior in real coordinates."""
import torch


def iq_noise_blocks(main, image, variance):
    """Recover real I/Q noise blocks at fixed mean real noise variance.

    For proper complex noise w the receiver maps w to main*w + image*conj(w).
    Normalizing by average coefficient power preserves the supplied trace.
    """
    power = main.abs().square() + image.abs().square()
    q = variance[:, None] / power.mean(dim=1, keepdim=True)
    correlation = main * image
    in_phase = q * (power + 2*correlation.real)
    quadrature = q * (power - 2*correlation.real)
    cross = 2*q*correlation.imag
    determinant = in_phase*quadrature - cross.square()
    if not torch.isfinite(determinant).all() or (determinant <= 0).any():
        raise ValueError('Finite positive definite I/Q noise blocks required')
    return in_phase, quadrature, cross


def posterior(action, observed, mean, covariance, variance, *, blocks=None, risk=False):
    """One Cholesky for the posterior mean and optional conditional MSE."""
    d = observed.shape[1]
    ac = action @ covariance
    normal = ac @ action.transpose(-1,-2)
    if blocks is None:
        normal.diagonal(dim1=-2,dim2=-1).add_(variance[:,None])
    else:
        i, q, cross = blocks
        half = d//2
        normal.diagonal(dim1=-2,dim2=-1).add_(torch.cat((i,q),dim=1))
        normal[:, :half, half:].diagonal(dim1=-2,dim2=-1).add_(cross)
        normal[:, half:, :half].diagonal(dim1=-2,dim2=-1).add_(cross)
    rhs = observed - action @ mean
    factor = torch.linalg.cholesky(normal)
    solved = torch.cholesky_solve(rhs[:,:,None],factor)
    estimate = mean + (ac.transpose(-1,-2) @ solved).squeeze(-1)
    relative = ((normal @ solved).squeeze(-1)-rhs).norm(dim=1)/rhs.norm(dim=1).clamp_min(1e-30)
    trace = None
    if risk:
        whitened = torch.linalg.solve_triangular(factor,ac,upper=False)
        trace = covariance.trace() - whitened.square().sum(dim=(1,2))
        if not torch.isfinite(trace).all() or (trace < 0).any():
            raise ValueError('Invalid posterior covariance trace')
    if not torch.isfinite(estimate).all() or not torch.isfinite(relative).all():
        raise ValueError('Invalid Gaussian posterior')
    return estimate,relative,trace
