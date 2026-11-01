"""Privileged paired-source FIR fits, used only as a research diagnostic."""
import torch
from torch.nn import functional as F


def fit_paired_fir(received, clean, *, taps, widely_linear, ridge=1e-4):
    """Fit a per-record complex FIR to paired clean IQ; never valid at inference.

    The fitted waveform minimizes a regularized IQ error, not classifier error.
    Centered taps make this a bounded-window oracle, not a causal receiver.
    """
    if (received.ndim != 3 or received.shape[1:] != (2, 256) or clean.shape != received.shape
            or not torch.is_floating_point(received) or received.dtype != clean.dtype):
        raise ValueError('Aligned floating-point Phase1 source IQ windows required')
    if type(taps) is not int or taps < 1 or taps > 65 or taps % 2 != 1 or type(widely_linear) is not bool or not 0 < ridge <= 1:
        raise ValueError('Odd bounded tap count and positive ridge required')
    x = torch.complex(received[:, 0], received[:, 1])
    target = torch.complex(clean[:, 0], clean[:, 1])
    half = taps // 2
    design = F.pad(x, (half, half)).unfold(1, taps, 1)
    if widely_linear:
        design = torch.cat((design, design.conj()), dim=-1)
    adjoint = design.conj().transpose(1, 2)
    gram = adjoint @ design
    scale = gram.diagonal(dim1=-2, dim2=-1).real.mean(-1).clamp_min(1e-8)
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    coefficients = torch.linalg.solve(gram + ridge * scale[:, None, None] * eye, adjoint @ target.unsqueeze(-1))
    fitted = (design @ coefficients).squeeze(-1)
    result = torch.stack((fitted.real, fitted.imag), dim=1)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Finite paired-source FIR output required')
    return result
