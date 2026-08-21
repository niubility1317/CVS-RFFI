"""Source-only spectral identifiability features for Phase1 SID-FFT96."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SID_FFT96_DIM = 96
SID_GROUP_DIMS = (24, 24, 16, 16, 16)
SID_GROUP_NAMES = (
    "amplitude_residual",
    "phase_residual",
    "phase_curvature_coherence",
    "mirror_coupling",
    "band_edge_residual",
)
SID_MODES = {"center", "phase", "sid"}


def build_center_mask(fft_bins: int, half_width: int, dc_notch: int = 0) -> Tensor:
    """Build a centered FFT mask with an optional symmetric DC notch."""
    if fft_bins <= 0:
        raise ValueError("fft_bins must be positive")
    if half_width <= 0 or half_width > fft_bins // 2:
        raise ValueError("half_width must be in (0, fft_bins // 2]")
    if dc_notch < 0 or dc_notch >= half_width:
        raise ValueError("dc_notch must be non-negative and smaller than half_width")
    center = fft_bins // 2
    start = max(0, center - half_width)
    stop = min(fft_bins, center + half_width)
    mask = torch.zeros(fft_bins, dtype=torch.bool)
    mask[start:stop] = True
    if dc_notch:
        mask[max(0, center - dc_notch) : min(fft_bins, center + dc_notch + 1)] = False
    return validate_sid_mask(mask, fft_bins)


def validate_sid_mask(mask: Tensor, fft_bins: int) -> Tensor:
    """Validate and canonicalize a one-dimensional spectral mask."""
    if not isinstance(mask, Tensor):
        mask = torch.as_tensor(mask)
    if mask.ndim != 1 or mask.numel() != fft_bins:
        raise ValueError(f"SID mask fft_bins mismatch: expected {fft_bins}, got {tuple(mask.shape)}")
    if not torch.isfinite(mask.to(dtype=torch.float32)).all():
        raise ValueError("SID mask must contain only finite values")
    mask = mask.to(dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError("SID mask must be non-empty")
    return mask


def _pool(sequence: Tensor, output_dim: int) -> Tensor:
    if sequence.ndim != 2:
        raise ValueError("spectral sequence must have shape [B, F]")
    return F.adaptive_avg_pool1d(sequence.unsqueeze(1), output_dim).squeeze(1)


def _safe_difference(sequence: Tensor) -> Tensor:
    if sequence.shape[1] <= 1:
        return torch.zeros_like(sequence)
    difference = sequence[:, 1:] - sequence[:, :-1]
    return F.pad(difference, (1, 0))


def _complex_iq(iq: Tensor) -> Tensor:
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError(f"iq must have shape [B, 2, T], got {tuple(iq.shape)}")
    if iq.shape[-1] < 4:
        raise ValueError("iq must contain at least four time samples")
    if not torch.isfinite(iq).all():
        raise ValueError("iq must contain only finite values")
    return torch.complex(iq[:, 0], iq[:, 1])


def extract_sid_fft96(
    iq: Tensor,
    mask: Tensor,
    mode: str = "sid",
    eps: float = 1e-6,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Extract the fixed 96-dimensional amplitude/phase-aware SID descriptor."""
    if mode not in SID_MODES:
        raise ValueError(f"unsupported SID mode: {mode!r}")
    if eps <= 0:
        raise ValueError("eps must be positive")

    signal = _complex_iq(iq)
    fft_bins = signal.shape[-1]
    mask = validate_sid_mask(mask, fft_bins).to(device=signal.device)

    signal = signal - signal.mean(dim=-1, keepdim=True)
    rms = signal.abs().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    signal = signal / rms
    window = torch.hann_window(fft_bins, dtype=signal.real.dtype, device=signal.device)
    spectrum = torch.fft.fftshift(torch.fft.fft(signal * window, dim=-1), dim=-1)
    selected = spectrum[:, mask]

    amplitude = selected.abs().clamp_min(eps)
    log_amplitude = amplitude.log()
    amplitude_residual = log_amplitude - _pool(log_amplitude, min(8, log_amplitude.shape[1])).mean(
        dim=1, keepdim=True
    )

    phase_step = torch.angle(selected[:, 1:] * selected[:, :-1].conj())
    if phase_step.shape[1] == 0:
        phase_step = torch.zeros_like(log_amplitude)
    else:
        phase_step = F.pad(phase_step, (1, 0))
    phase_residual = phase_step - phase_step.mean(dim=1, keepdim=True)
    phase_curvature = _safe_difference(phase_step)
    phase_coherence = torch.cos(phase_curvature)

    mirrored = torch.flip(selected, dims=(1,)).conj()
    mirror_coupling = (selected * mirrored.conj()).real / (amplitude * mirrored.abs()).clamp_min(eps)
    band_edge = _safe_difference(log_amplitude)

    groups = [
        _pool(amplitude_residual, 24),
        _pool(phase_residual, 24),
        _pool(phase_curvature * phase_coherence, 16),
        _pool(mirror_coupling, 16),
        _pool(band_edge, 16),
    ]
    feature = torch.cat(groups, dim=1)
    if mode == "center":
        active_groups = (True, False, False, False, False)
    elif mode == "phase":
        active_groups = (False, True, True, False, False)
    else:
        active_groups = (True, True, True, True, True)

    active = torch.cat(
        [torch.full((dim,), enabled, dtype=torch.bool, device=feature.device) for dim, enabled in zip(SID_GROUP_DIMS, active_groups)]
    )
    feature = torch.nan_to_num(feature)
    active_mean = feature[:, active].mean(dim=1, keepdim=True)
    feature = torch.where(active.unsqueeze(0), feature - active_mean, torch.zeros_like(feature))
    feature = F.normalize(feature, dim=1, eps=eps)
    feature = torch.where(active.unsqueeze(0), feature, torch.zeros_like(feature))

    offsets = (0, 24, 48, 64, 80, 96)
    group_norms = torch.stack(
        [feature[:, offsets[index] : offsets[index + 1]].norm(dim=1) for index in range(5)],
        dim=1,
    )
    diagnostics = {
        "group_norms": group_norms,
        "valid_bin_ratio": feature.new_full((feature.shape[0],), float(mask.float().mean())),
    }
    return feature, diagnostics


class SIDFFT96Residual(nn.Module):
    """Zero-initialized residual from SID-FFT96 into the identity embedding."""

    def __init__(
        self,
        embedding_dim: int,
        mode: str,
        mask: Tensor,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if mode not in SID_MODES:
            raise ValueError(f"unsupported SID mode: {mode!r}")
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        mask = validate_sid_mask(mask, int(torch.as_tensor(mask).numel()))
        self.mode = mode
        self.residual_scale = float(residual_scale)
        self.register_buffer("mask", mask, persistent=True)
        self.projector = nn.Sequential(
            nn.Linear(SID_FFT96_DIM, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        nn.init.zeros_(self.projector[-1].weight)
        nn.init.zeros_(self.projector[-1].bias)

    def forward(self, iq: Tensor, z_raw: Tensor) -> Dict[str, Tensor]:
        if z_raw.ndim != 2:
            raise ValueError("z_raw must have shape [B, D]")
        features, diagnostics = extract_sid_fft96(iq, self.mask, mode=self.mode)
        delta = self.projector(features)
        z_sid = F.normalize(z_raw + self.residual_scale * delta, dim=1)
        return {
            "z_raw": z_raw,
            "z_sid": z_sid,
            "sid_fft96": features,
            "sid_delta": delta,
            "sid_group_norms": diagnostics["group_norms"],
            "sid_valid_bin_ratio": diagnostics["valid_bin_ratio"],
        }
