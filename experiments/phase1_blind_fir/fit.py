"""Per-record regularized FIR adjustment for a frozen widely-linear action.

Fits only the current observed window and current IQ proposal. This routine
is an uncalibrated optimizer, not a proven posterior or an identity metric.
"""
import torch


def reflected_taps(signal, count):
    """Reflect left boundary exactly as the frozen DURA FIR action does."""
    if signal.ndim != 2 or not signal.is_complex() or not 1 <= count <= signal.shape[1]:
        raise ValueError('Batched complex IQ and valid tap count required')
    pos = torch.arange(signal.shape[1], device=signal.device)[:, None]
    tap = torch.arange(count, device=signal.device)[None, :]
    return signal[:, (pos-tap).abs()]


def fir_design(signal, main, image, count):
    """Complex output changes for real and imaginary tap perturbations."""
    if main.shape != signal.shape or image.shape != signal.shape or not (main.is_complex() and image.is_complex()):
        raise ValueError('Matching complex main and image coefficients required')
    taps = reflected_taps(signal, count)
    a, b = main.unsqueeze(-1)*taps, image.unsqueeze(-1)*taps.conj()
    return torch.cat((a+b, 1j*(a-b)), dim=-1)


def fit_fir_delta(signal, residual, main, image, kernel, *, count, ridge=.1, max_relative=.25):
    """Fit a bounded complex FIR perturbation by real ridge regression.

    residual is observed minus the estimated action applied to signal.
    A relative norm cap protects the fixed estimated channel; no other record,
    clean target, transmitter label, classifier, or simulator truth is used.
    """
    if (residual.shape != signal.shape or not residual.is_complex() or kernel.ndim != 2 or
            not kernel.is_complex() or len(kernel) != len(signal) or count > kernel.shape[1] or
            ridge <= 0 or max_relative <= 0):
        raise ValueError('Matching complex inputs and positive regularization required')
    if not all(torch.isfinite(x).all() for x in (signal, residual, main, image, kernel)):
        raise ValueError('Finite signal and action required')
    basis = fir_design(signal, main, image, count)
    gram = torch.einsum('bnk,bnl->bkl', basis.conj(), basis).real
    rhs = torch.einsum('bnk,bn->bk', basis.conj(), residual).real
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
    identity = torch.eye(2*count, device=signal.device, dtype=gram.dtype)
    coefficients = torch.linalg.solve(gram + ridge*scale[:, None, None]*identity, rhs.unsqueeze(-1)).squeeze(-1)
    delta = torch.complex(coefficients[:, :count], coefficients[:, count:])
    bound = max_relative*kernel.abs().square().sum(-1).sqrt()
    size = delta.abs().square().sum(-1).sqrt().clamp_min(1e-12)
    return delta * (bound/size).clamp(max=1)[:, None]
