"""Paired-source half-window cross-fit diagnostic, never a deployed receiver."""
import torch
from torch.nn import functional as F


def crossfit_paired_fir(anchor, clean, *, taps, ridge=1e-4):
    """Predict each window half using coefficients fitted on the other clean half."""
    if (anchor.ndim != 3 or anchor.shape[1:] != (2,256) or clean.shape != anchor.shape
            or not torch.is_floating_point(anchor) or anchor.dtype != clean.dtype):
        raise ValueError('Aligned floating-point Phase1 source IQ required')
    if type(taps) is not int or taps < 1 or taps > 33 or taps % 2 != 1 or not 0 < ridge <= 1:
        raise ValueError('Odd bounded tap count and positive ridge required')
    x = torch.complex(anchor[:,0],anchor[:,1])
    target = torch.complex(clean[:,0],clean[:,1])
    windows = F.pad(x,(taps//2,taps//2)).unfold(1,taps,1)
    design = torch.cat((windows,windows.conj()),dim=2)
    def fit_and_predict(fit,predict):
        h = design[:,fit].conj().transpose(1,2)
        gram = h@design[:,fit]
        scale = gram.diagonal(dim1=-2,dim2=-1).real.mean(-1).clamp_min(1e-8)
        eye = torch.eye(2*taps,dtype=gram.dtype,device=gram.device)
        weights = torch.linalg.solve(gram+ridge*scale[:,None,None]*eye,h@target[:,fit,None])
        return (design[:,predict]@weights).squeeze(-1)
    left = fit_and_predict(slice(128,256),slice(0,128))
    right = fit_and_predict(slice(0,128),slice(128,256))
    fitted = torch.cat((left,right),dim=1)
    result = torch.stack((fitted.real,fitted.imag),dim=1)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Finite paired-source cross-fit output required')
    return result
