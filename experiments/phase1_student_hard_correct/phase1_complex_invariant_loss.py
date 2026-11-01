"""Stable per-record complex-gain-invariant source reconstruction objective."""
import torch


def complex_gain_invariant_error(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-record normalized central-128 IQ error after optimal complex gain.

    Reference is a paired source waveform already time-aligned to the anchor;
    no classification model or transmitter label is an argument.
    """
    if prediction.shape != reference.shape or prediction.ndim != 3 or prediction.shape[1:] != (2, 256):
        raise ValueError('matching [batch,2,256] IQ tensors required')
    p = torch.complex(prediction[:, 0, 64:-64], prediction[:, 1, 64:-64])
    y = torch.complex(reference[:, 0, 64:-64], reference[:, 1, 64:-64])
    energy = p.abs().square().sum(1).clamp_min(1e-12)
    gain = (p.conj() * y).sum(1) / energy
    error = (gain[:, None]*p - y).abs().square().mean(1)
    return error / y.abs().square().mean(1).clamp_min(1e-12)
