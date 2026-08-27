from __future__ import annotations

import torch


def synchronize_iq(iq: torch.Tensor, preamble: torch.Tensor, *, output_length: int = 256) -> torch.Tensor:
    """Locate a supplied leading preamble by complex correlation and extract its following window."""
    if iq.ndim != 3 or iq.shape[1] != 2 or preamble.ndim != 2 or preamble.shape[0] != 2:
        raise ValueError("iq must be [batch, 2, samples] and preamble must be [2, samples]")
    if output_length <= 0 or iq.shape[-1] < preamble.shape[-1] + output_length:
        raise ValueError("iq must contain a full output window after the preamble")
    signal = torch.complex(iq[:, 0], iq[:, 1])
    known = torch.complex(preamble[0], preamble[1]).to(device=iq.device, dtype=signal.dtype)
    valid_starts = iq.shape[-1] - known.numel() - output_length + 1
    windows = signal.unfold(1, known.numel(), 1)[:, :valid_starts]
    scores = (windows * known.conj()).sum(dim=-1).abs()
    starts = scores.argmax(dim=1) + known.numel()
    offsets = torch.arange(output_length, device=iq.device)
    indices = starts[:, None] + offsets[None, :]
    extracted = signal.gather(1, indices)
    return torch.stack((extracted.real, extracted.imag), dim=1)


def preprocess_iq(
    iq: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
    preamble: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply default scalar equalization and per-segment zero-mean unit-RMS normalization."""
    if preamble is not None:
        iq = synchronize_iq(iq, preamble)
    if iq.ndim != 3 or iq.shape[1] != 2 or iq.shape[-1] != 256:
        raise ValueError("iq must have shape [batch, 2, 256]")
    complex_iq = torch.complex(iq[:, 0], iq[:, 1])
    if reference is not None:
        if reference.shape != iq.shape:
            raise ValueError("reference must have the same shape as iq")
        complex_reference = torch.complex(reference[:, 0], reference[:, 1])
        denominator = complex_reference.abs().square().sum(dim=1).clamp_min(torch.finfo(iq.dtype).eps)
        channel = (complex_iq * complex_reference.conj()).sum(dim=1) / denominator
        complex_iq = complex_iq / channel[:, None].abs().clamp_min(torch.finfo(iq.dtype).eps)
    stacked = torch.stack((complex_iq.real, complex_iq.imag), dim=1)
    centered = stacked - stacked.mean(dim=-1, keepdim=True)
    rms = centered.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(torch.finfo(iq.dtype).eps)
    return centered / rms
