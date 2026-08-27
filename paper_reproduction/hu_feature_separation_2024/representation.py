from __future__ import annotations

import torch


def welch_psd_256(iq: torch.Tensor) -> torch.Tensor:
    """Return the paper's 256-bin PSD stream from 2x256 IQ input.

    The paper fixes a 256-point Welch FFT but does not specify segment overlap
    or window. A single Hann-windowed 256-point segment is the documented
    implementation choice for this reproducible core.
    """
    if iq.ndim != 3 or tuple(iq.shape[1:]) != (2, 256):
        raise ValueError("iq must have shape [batch, 2, 256]")
    complex_iq = torch.complex(iq[:, 0], iq[:, 1])
    window = torch.hann_window(256, dtype=iq.dtype, device=iq.device)
    spectrum = torch.fft.fft(complex_iq * window, n=256, dim=-1)
    return spectrum.abs().square().unsqueeze(1) / window.square().sum().clamp_min(torch.finfo(iq.dtype).eps)


def build_fusion_representation(iq: torch.Tensor) -> torch.Tensor:
    """Concatenate untouched I/Q streams with their 256-bin PSD stream."""
    return torch.cat((iq, welch_psd_256(iq)), dim=1)

