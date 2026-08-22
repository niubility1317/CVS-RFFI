"""Source-only spectral identifiability features for Phase1 SID-FFT96."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
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


class SpectralIdentifiabilityAccumulator:
    """Accumulate source-label separability and source-domain scatter by band."""

    def __init__(self, num_bands: int, feature_dim: int, eps: float = 1e-8) -> None:
        if num_bands <= 0 or feature_dim <= 0:
            raise ValueError("num_bands and feature_dim must be positive")
        self.num_bands = int(num_bands)
        self.feature_dim = int(feature_dim)
        self.eps = float(eps)
        self._descriptors: List[np.ndarray] = []
        self._tx: List[int] = []
        self._domain: List[Tuple[int, int, int]] = []

    def update(
        self,
        descriptor: np.ndarray,
        *,
        tx: int,
        rx: int,
        day: int,
        view: int,
    ) -> None:
        descriptor = np.asarray(descriptor, dtype=np.float64)
        expected = (self.num_bands, self.feature_dim)
        if descriptor.shape != expected:
            raise ValueError(f"descriptor must have shape {expected}, got {descriptor.shape}")
        if not np.isfinite(descriptor).all():
            raise ValueError("descriptor must contain only finite values")
        self._descriptors.append(descriptor)
        self._tx.append(int(tx))
        self._domain.append((int(rx), int(day), int(view)))

    @staticmethod
    def _scatter(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
        means = [values[groups == group].mean(axis=0) for group in np.unique(groups)]
        if len(means) <= 1:
            return np.zeros(values.shape[1], dtype=np.float64)
        return np.stack(means, axis=0).var(axis=0).mean(axis=-1)

    @classmethod
    def _tx_conditioned_domain_scatter(
        cls,
        values: np.ndarray,
        tx: np.ndarray,
        domain: np.ndarray,
    ) -> np.ndarray:
        per_tx = [
            cls._scatter(values[tx == tx_value], domain[tx == tx_value])
            for tx_value in np.unique(tx)
        ]
        return np.stack(per_tx, axis=0).mean(axis=0)

    def finalize(self) -> Dict[str, np.ndarray]:
        if not self._descriptors:
            raise ValueError("cannot finalize an empty identifiability accumulator")
        values = np.stack(self._descriptors, axis=0)
        tx = np.asarray(self._tx, dtype=np.int64)
        domain_values = np.asarray(self._domain, dtype=np.int64)
        _, domain = np.unique(domain_values, axis=0, return_inverse=True)
        tx_scatter = self._scatter(values, tx)
        domain_scatter = self._tx_conditioned_domain_scatter(values, tx, domain)

        residual = np.empty_like(values)
        for tx_value in np.unique(tx):
            selected = tx == tx_value
            residual[selected] = values[selected] - values[selected].mean(axis=0, keepdims=True)
        noise_scatter = residual.var(axis=0).mean(axis=-1)
        j_score = tx_scatter / (domain_scatter + noise_scatter + self.eps)
        return {
            "j_score": j_score,
            "tx_scatter": tx_scatter,
            "domain_scatter": domain_scatter,
            "noise_scatter": noise_scatter,
            "count": np.asarray([values.shape[0]], dtype=np.int64),
        }


def select_sid_mask(
    stats: Dict[str, np.ndarray],
    keep_fraction: float,
    dc_notch: int = 0,
) -> np.ndarray:
    """Select the highest-scoring bands with deterministic index tie-breaking."""
    scores = np.asarray(stats.get("j_score"), dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("j_score must be a non-empty finite one-dimensional array")
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    if dc_notch < 0:
        raise ValueError("dc_notch must be non-negative")
    eligible = np.ones(scores.size, dtype=bool)
    if dc_notch:
        center = scores.size // 2
        eligible[max(0, center - dc_notch) : min(scores.size, center + dc_notch + 1)] = False
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size == 0:
        raise ValueError("dc_notch removes every spectral band")
    keep = max(1, int(np.ceil(scores.size * keep_fraction)))
    keep = min(keep, eligible_indices.size)
    order = np.lexsort((eligible_indices, -scores[eligible_indices]))
    selected = eligible_indices[order[:keep]]
    mask = np.zeros(scores.size, dtype=bool)
    mask[selected] = True
    return mask


def load_sid_mask(path: str | Path, fft_bins: int, key: str = "mask") -> Tensor:
    """Load the fixed P0 FFT mask without accepting pickled objects."""
    mask_path = Path(path)
    if not mask_path.is_file():
        raise FileNotFoundError(f"SID mask does not exist: {mask_path}")
    with np.load(mask_path, allow_pickle=False) as payload:
        if key not in payload.files:
            raise ValueError(f"SID mask artifact lacks {key!r}: {mask_path}")
        mask = np.asarray(payload[key])
    return validate_sid_mask(torch.tensor(mask.tolist()), fft_bins)


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


def extract_band_descriptors(iq: Tensor, num_bands: int, eps: float = 1e-6) -> Tensor:
    """Return per-band amplitude and phase descriptors with shape [B, bands, 5]."""
    if num_bands <= 0:
        raise ValueError("num_bands must be positive")
    signal = _complex_iq(iq)
    fft_bins = signal.shape[-1]
    if num_bands > fft_bins:
        raise ValueError("num_bands cannot exceed the IQ sequence length")
    signal = signal - signal.mean(dim=-1, keepdim=True)
    signal = signal / signal.abs().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    window = torch.hann_window(fft_bins, dtype=signal.real.dtype, device=signal.device)
    spectrum = torch.fft.fftshift(torch.fft.fft(signal * window, dim=-1), dim=-1)
    amplitude = spectrum.abs().clamp_min(eps)
    log_amplitude = amplitude.log()
    phase_step = torch.angle(spectrum[:, 1:] * spectrum[:, :-1].conj())
    phase_step = F.pad(phase_step, (1, 0))
    curvature = _safe_difference(phase_step)
    mirror = torch.flip(spectrum, dims=(1,)).conj()
    mirror_coupling = (spectrum * mirror.conj()).real / (amplitude * mirror.abs()).clamp_min(eps)
    channels = torch.stack(
        (log_amplitude, phase_step.sin(), phase_step.cos(), curvature.cos(), mirror_coupling),
        dim=1,
    )
    pooled = F.adaptive_avg_pool1d(channels, num_bands)
    return torch.nan_to_num(pooled.transpose(1, 2))


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
        # The shared CosFace head performs the final direction normalization.
        # Keeping this residual in the backbone's native scale makes the
        # zero-initialized candidate exactly identical to the mature path.
        z_sid = z_raw + self.residual_scale * delta
        return {
            "z_raw": z_raw,
            "z_sid": z_sid,
            "sid_fft96": features,
            "sid_delta": delta,
            "sid_group_norms": diagnostics["group_norms"],
            "sid_valid_bin_ratio": diagnostics["valid_bin_ratio"],
        }
