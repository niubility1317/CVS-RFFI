"""Source-only spectral identifiability features for Phase1 SID-FFT96."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Tuple

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
        self._cluster: List[int] = []

    def update(
        self,
        descriptor: np.ndarray,
        *,
        tx: int,
        rx: int,
        day: int,
        view: int,
        cluster: int | None = None,
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
        self._cluster.append(len(self._cluster) if cluster is None else int(cluster))

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

    @staticmethod
    def _main_effect_scatter(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
        means = [values[groups == group].mean(axis=0) for group in np.unique(groups)]
        if len(means) <= 1:
            return np.zeros(values.shape[1], dtype=np.float64)
        return np.stack(means, axis=0).var(axis=0).mean(axis=-1)

    @staticmethod
    def _interaction_components(
        values: np.ndarray,
        tx: np.ndarray,
        factor: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cells = []
        cell_tx = []
        cell_factor = []
        for tx_value in np.unique(tx):
            for factor_value in np.unique(factor[tx == tx_value]):
                selected = (tx == tx_value) & (factor == factor_value)
                if selected.any():
                    cells.append(values[selected].mean(axis=0))
                    cell_tx.append(int(tx_value))
                    cell_factor.append(int(factor_value))
        if len(cells) <= 1:
            empty = np.zeros((1, *values.shape[1:]), dtype=np.float64)
            return empty, np.asarray([0], dtype=np.int64)
        cell_values = np.stack(cells, axis=0)
        cell_tx_array = np.asarray(cell_tx, dtype=np.int64)
        cell_factor_array = np.asarray(cell_factor, dtype=np.int64)
        grand = cell_values.mean(axis=0)
        tx_means = {
            value: cell_values[cell_tx_array == value].mean(axis=0)
            for value in np.unique(cell_tx_array)
        }
        factor_means = {
            value: cell_values[cell_factor_array == value].mean(axis=0)
            for value in np.unique(cell_factor_array)
        }
        residuals = np.stack(
            [
                cell - tx_means[tx_value] - factor_means[factor_value] + grand
                for cell, tx_value, factor_value in zip(cell_values, cell_tx_array, cell_factor_array)
            ],
            axis=0,
        )
        return residuals, cell_factor_array

    @classmethod
    def _interaction_scatter(
        cls,
        values: np.ndarray,
        tx: np.ndarray,
        factor: np.ndarray,
    ) -> np.ndarray:
        residuals, _ = cls._interaction_components(values, tx, factor)
        return np.square(residuals).mean(axis=0).mean(axis=-1)

    @classmethod
    def _interaction_tail_scatter(
        cls,
        values: np.ndarray,
        tx: np.ndarray,
        factor: np.ndarray,
        tail_fraction: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        residuals, cell_factor = cls._interaction_components(values, tx, factor)
        risks = np.square(residuals).mean(axis=-1)
        grouped = [risks[cell_factor == value].mean(axis=0) for value in np.unique(cell_factor)]
        grouped_risk = np.stack(grouped, axis=0)
        worst = grouped_risk.max(axis=0)
        tail_count = max(1, int(np.ceil(grouped_risk.shape[0] * float(tail_fraction))))
        cvar = np.sort(grouped_risk, axis=0)[-tail_count:].mean(axis=0)
        return worst, cvar

    @staticmethod
    def _within_cell_scatter(
        values: np.ndarray,
        tx: np.ndarray,
        rx: np.ndarray,
        day: np.ndarray,
        view: np.ndarray,
    ) -> np.ndarray:
        residuals = np.zeros_like(values)
        cells = np.stack((tx, rx, day, view), axis=1)
        _, inverse = np.unique(cells, axis=0, return_inverse=True)
        for cell in np.unique(inverse):
            selected = inverse == cell
            residuals[selected] = values[selected] - values[selected].mean(axis=0, keepdims=True)
        return residuals.var(axis=0).mean(axis=-1)

    def _statistics_from_arrays(
        self,
        values: np.ndarray,
        tx: np.ndarray,
        domain_values: np.ndarray,
        *,
        lambda_rx: float = 2.0,
        lambda_day: float = 1.0,
        lambda_view: float = 1.5,
        tail_fraction: float = 0.30,
    ) -> Dict[str, np.ndarray]:
        rx = domain_values[:, 0]
        day = domain_values[:, 1]
        view = domain_values[:, 2]
        _, domain = np.unique(domain_values, axis=0, return_inverse=True)
        tx_scatter = self._scatter(values, tx)
        domain_scatter = self._tx_conditioned_domain_scatter(values, tx, domain)
        rx_main_scatter = self._main_effect_scatter(values, rx)
        day_main_scatter = self._main_effect_scatter(values, day)
        view_main_scatter = self._main_effect_scatter(values, view)
        tx_rx_interaction = self._interaction_scatter(values, tx, rx)
        tx_day_interaction = self._interaction_scatter(values, tx, day)
        tx_view_interaction = self._interaction_scatter(values, tx, view)
        rx_worst, rx_cvar = self._interaction_tail_scatter(values, tx, rx, tail_fraction)
        noise_scatter = self._within_cell_scatter(values, tx, rx, day, view)
        denominator = (
            float(lambda_rx) * rx_cvar
            + float(lambda_day) * tx_day_interaction
            + float(lambda_view) * tx_view_interaction
            + noise_scatter
            + self.eps
        )
        j_score = tx_scatter / denominator
        nonlinear_score = tx_scatter / (
            tx_day_interaction + tx_view_interaction + noise_scatter + self.eps
        )
        domain_score = rx_main_scatter + day_main_scatter + view_main_scatter
        return {
            "j_score": j_score,
            "nonlinear_score": nonlinear_score,
            "domain_score": domain_score,
            "tx_scatter": tx_scatter,
            "domain_scatter": domain_scatter,
            "noise_scatter": noise_scatter,
            "rx_main_scatter": rx_main_scatter,
            "day_main_scatter": day_main_scatter,
            "view_main_scatter": view_main_scatter,
            "tx_rx_interaction_scatter": tx_rx_interaction,
            "tx_day_interaction_scatter": tx_day_interaction,
            "tx_view_interaction_scatter": tx_view_interaction,
            "rx_worst_interaction_scatter": rx_worst,
            "rx_cvar_interaction_scatter": rx_cvar,
            "count": np.asarray([values.shape[0]], dtype=np.int64),
        }

    def finalize(
        self,
        *,
        lambda_rx: float = 2.0,
        lambda_day: float = 1.0,
        lambda_view: float = 1.5,
        tail_fraction: float = 0.30,
        bootstrap_repeats: int = 64,
        bootstrap_keep_fraction: float = 0.30,
        bootstrap_seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        if not self._descriptors:
            raise ValueError("cannot finalize an empty identifiability accumulator")
        if min(lambda_rx, lambda_day, lambda_view) < 0.0:
            raise ValueError("hierarchical identifiability weights must be non-negative")
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")
        if int(bootstrap_repeats) <= 0:
            raise ValueError("bootstrap_repeats must be positive")
        if not 0.0 < float(bootstrap_keep_fraction) <= 1.0:
            raise ValueError("bootstrap_keep_fraction must be in (0, 1]")
        values = np.stack(self._descriptors, axis=0)
        tx = np.asarray(self._tx, dtype=np.int64)
        domain_values = np.asarray(self._domain, dtype=np.int64)
        clusters = np.asarray(self._cluster, dtype=np.int64)
        statistics = self._statistics_from_arrays(
            values,
            tx,
            domain_values,
            lambda_rx=lambda_rx,
            lambda_day=lambda_day,
            lambda_view=lambda_view,
            tail_fraction=tail_fraction,
        )

        unique_clusters = np.unique(clusters)
        rng = np.random.default_rng(int(bootstrap_seed))
        selection_count = np.zeros(self.num_bands, dtype=np.float64)
        keep_count = max(1, int(np.ceil(self.num_bands * float(bootstrap_keep_fraction))))
        for _ in range(int(bootstrap_repeats)):
            sampled_clusters = rng.choice(unique_clusters, size=unique_clusters.size, replace=True)
            sampled_indices = np.concatenate([np.flatnonzero(clusters == cluster) for cluster in sampled_clusters])
            sampled = self._statistics_from_arrays(
                values[sampled_indices],
                tx[sampled_indices],
                domain_values[sampled_indices],
                lambda_rx=lambda_rx,
                lambda_day=lambda_day,
                lambda_view=lambda_view,
                tail_fraction=tail_fraction,
            )
            indices = np.arange(self.num_bands)
            order = np.lexsort((indices, -sampled["j_score"]))
            selection_count[order[:keep_count]] += 1.0
        statistics["bootstrap_selection_probability"] = selection_count / float(bootstrap_repeats)
        return statistics


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


def select_hsid_role_masks(
    stats: Mapping[str, np.ndarray],
    *,
    common_fraction: float = 0.30,
    nonlinear_fraction: float = 0.20,
    domain_fraction: float = 0.30,
    stability_threshold: float = 0.80,
    dc_notch: int = 0,
) -> Dict[str, np.ndarray]:
    """Select deterministic, disjoint common/nonlinear/domain spectral roles."""

    keys = ("j_score", "nonlinear_score", "domain_score")
    scores = [np.asarray(stats.get(key), dtype=np.float64) for key in keys]
    if any(score.ndim != 1 or score.size == 0 for score in scores):
        raise ValueError("HSID role scores must be non-empty one-dimensional arrays")
    if any(score.shape != scores[0].shape or not np.isfinite(score).all() for score in scores):
        raise ValueError("HSID role scores must share one finite shape")
    fractions = (common_fraction, nonlinear_fraction, domain_fraction)
    if any(not 0.0 <= float(value) <= 1.0 for value in fractions):
        raise ValueError("HSID role fractions must be in [0, 1]")
    stability = np.asarray(
        stats.get("bootstrap_selection_probability", np.ones_like(scores[0])),
        dtype=np.float64,
    )
    if stability.shape != scores[0].shape or not np.isfinite(stability).all():
        raise ValueError("bootstrap_selection_probability must match role scores")
    if int(dc_notch) < 0:
        raise ValueError("dc_notch must be non-negative")
    eligible = stability >= float(stability_threshold)
    if int(dc_notch):
        center = scores[0].size // 2
        eligible[
            max(0, center - int(dc_notch)) : min(scores[0].size, center + int(dc_notch) + 1)
        ] = False
    if not eligible.any():
        raise ValueError("stability threshold and dc_notch remove every spectral band")
    available = eligible.copy()
    result = {}
    for name, score, fraction in zip(("common_mask", "nonlinear_mask", "domain_mask"), scores, fractions):
        count = min(int(np.ceil(score.size * float(fraction))), int(available.sum()))
        mask = np.zeros(score.size, dtype=bool)
        if count > 0:
            indices = np.flatnonzero(available)
            order = np.lexsort((indices, -score[indices]))
            chosen = indices[order[:count]]
            mask[chosen] = True
            available[chosen] = False
        result[name] = mask
    return result


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


def _masked_frequency_pool(sequence: Tensor, valid: Tensor, output_dim: int) -> Tensor:
    """Pool fixed full-spectrum coordinates without closing gaps in a mask."""

    if sequence.ndim != 2:
        raise ValueError("spectral sequence must have shape [B, F]")
    if valid.ndim == 1:
        valid = valid.unsqueeze(0).expand(sequence.shape[0], -1)
    if valid.shape != sequence.shape:
        raise ValueError("frequency validity must align with the spectral sequence")
    edges = torch.linspace(0, sequence.shape[1], output_dim + 1, device=sequence.device)
    edges = edges.round().to(dtype=torch.long)
    pooled = []
    for index in range(output_dim):
        start = int(edges[index].item())
        stop = max(start + 1, int(edges[index + 1].item()))
        stop = min(stop, sequence.shape[1])
        weights = valid[:, start:stop].to(dtype=sequence.dtype)
        numerator = (sequence[:, start:stop] * weights).sum(dim=1)
        pooled.append(numerator / weights.sum(dim=1).clamp_min(1.0))
    return torch.stack(pooled, dim=1)


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


def exact_fftshift_mirror_indices(fft_bins: int, *, device=None) -> Tensor:
    """Return the exact f-to-negative-f mapping for an fftshift-ordered spectrum."""

    if fft_bins <= 0:
        raise ValueError("fft_bins must be positive")
    indices = torch.arange(fft_bins, dtype=torch.long, device=device)
    return torch.remainder(-indices, fft_bins)


def quadratic_log_amplitude_residual(
    log_amplitude: Tensor,
    mask: Tensor,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """Remove a weighted quadratic passband trend in full FFT coordinates."""

    if log_amplitude.ndim != 2:
        raise ValueError("log_amplitude must have shape [B, F]")
    mask = validate_sid_mask(mask, log_amplitude.shape[1]).to(device=log_amplitude.device)
    frequency = torch.linspace(
        -1.0,
        1.0,
        log_amplitude.shape[1],
        dtype=log_amplitude.dtype,
        device=log_amplitude.device,
    )
    design = torch.stack((torch.ones_like(frequency), frequency, frequency.square()), dim=1)
    selected = mask.unsqueeze(0).to(dtype=log_amplitude.dtype)
    relative_amplitude = torch.exp(
        log_amplitude - log_amplitude.detach().amax(dim=1, keepdim=True)
    )
    weights = selected * relative_amplitude.clamp_min(eps)
    weighted_design = weights.unsqueeze(-1) * design.unsqueeze(0)
    gram = torch.einsum("bfi,fj->bij", weighted_design, design)
    ridge = torch.eye(3, dtype=gram.dtype, device=gram.device).unsqueeze(0) * eps
    rhs = torch.einsum("bfi,bf->bi", weighted_design, log_amplitude)
    coefficients = torch.linalg.solve(gram + ridge, rhs.unsqueeze(-1)).squeeze(-1)
    trend = torch.einsum("fi,bi->bf", design, coefficients)
    residual = torch.where(mask.unsqueeze(0), log_amplitude - trend, torch.zeros_like(log_amplitude))
    trend_error = (
        (weights * residual.square()).sum(dim=1) / weights.sum(dim=1).clamp_min(eps)
    ).clamp_min(0.0).sqrt()
    return residual, trend_error


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

    device_type = iq.device.type if iq.device.type in {"cpu", "cuda"} else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        signal = _complex_iq(iq.float())
        fft_bins = signal.shape[-1]
        mask = validate_sid_mask(mask, fft_bins).to(device=signal.device)

        signal = signal - signal.mean(dim=-1, keepdim=True)
        raw_peak = signal.abs().amax(dim=-1).clamp_min(eps)
        rms = (signal.abs().square().mean(dim=-1, keepdim=True) + eps * eps).sqrt()
        signal = signal / rms
        window = torch.hann_window(fft_bins, dtype=signal.real.dtype, device=signal.device)
        spectrum = torch.fft.fftshift(torch.fft.fft(signal * window, dim=-1), dim=-1)
        amplitude_unclamped = spectrum.abs()
        amplitude = amplitude_unclamped.clamp_min(eps)
        log_amplitude = amplitude.log()
        amplitude_residual, trend_error = quadratic_log_amplitude_residual(log_amplitude, mask, eps)

        phase_product = spectrum[:, 1:] * spectrum[:, :-1].conj()
        phase_q_inner = phase_product / phase_product.abs().clamp_min(eps)
        phase_q = torch.cat((torch.ones_like(spectrum[:, :1]), phase_q_inner), dim=1)
        adjacent_valid = torch.cat(
            (
                torch.zeros(1, dtype=torch.bool, device=mask.device),
                mask[1:] & mask[:-1],
            ),
            dim=0,
        )
        energy_floor = amplitude_unclamped[:, mask].median(dim=1).values.unsqueeze(1) * 0.10
        energetic = amplitude_unclamped > energy_floor.clamp_min(eps)
        phase_valid = adjacent_valid.unsqueeze(0) & energetic & F.pad(energetic[:, :-1], (1, 0))

        curvature_inner = phase_q[:, 1:] * phase_q[:, :-1].conj()
        curvature_q = torch.cat((torch.ones_like(spectrum[:, :1]), curvature_inner), dim=1)
        curvature_valid = phase_valid & F.pad(phase_valid[:, :-1], (1, 0))

        mirror_indices = exact_fftshift_mirror_indices(fft_bins, device=spectrum.device)
        mirrored = spectrum[:, mirror_indices]
        mirror_product = spectrum * mirrored
        mirror_q = mirror_product / (amplitude * amplitude[:, mirror_indices]).clamp_min(eps)
        mirror_asymmetry = (amplitude - amplitude[:, mirror_indices]) / (
            amplitude + amplitude[:, mirror_indices]
        ).clamp_min(eps)
        mirror_valid = (mask & mask[mirror_indices]).unsqueeze(0) & energetic & energetic[:, mirror_indices]

        band_edge = torch.cat(
            (torch.zeros_like(amplitude_residual[:, :1]), amplitude_residual[:, 1:] - amplitude_residual[:, :-1]),
            dim=1,
        )
        band_edge_valid = adjacent_valid.unsqueeze(0) & energetic

        groups = [
            _masked_frequency_pool(amplitude_residual, mask, 24),
            torch.cat(
                (
                    _masked_frequency_pool(phase_q.real, phase_valid, 12),
                    _masked_frequency_pool(phase_q.imag, phase_valid, 12),
                ),
                dim=1,
            ),
            torch.cat(
                (
                    _masked_frequency_pool(curvature_q.real, curvature_valid, 8),
                    _masked_frequency_pool(curvature_q.imag, curvature_valid, 8),
                ),
                dim=1,
            ),
            torch.cat(
                (
                    _masked_frequency_pool(mirror_q.real, mirror_valid, 6),
                    _masked_frequency_pool(mirror_q.imag, mirror_valid, 5),
                    _masked_frequency_pool(mirror_asymmetry, mirror_valid, 5),
                ),
                dim=1,
            ),
            _masked_frequency_pool(band_edge, band_edge_valid, 16),
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
    offsets = (0, 24, 48, 64, 80, 96)
    normalized_groups = []
    for index in range(5):
        group = feature[:, offsets[index] : offsets[index + 1]]
        if active_groups[index]:
            group = group - group.mean(dim=1, keepdim=True)
            group = group / (group.square().mean(dim=1, keepdim=True) + eps * eps).sqrt()
        else:
            group = torch.zeros_like(group)
        normalized_groups.append(group)
    feature = torch.cat(normalized_groups, dim=1)
    feature = F.normalize(feature, dim=1, eps=eps)
    feature = torch.where(active.unsqueeze(0), feature, torch.zeros_like(feature))

    group_norms = torch.stack(
        [feature[:, offsets[index] : offsets[index + 1]].norm(dim=1) for index in range(5)],
        dim=1,
    )
    selected_count = mask.sum().clamp_min(1)
    valid_bin_ratio = (energetic & mask.unsqueeze(0)).sum(dim=1).to(dtype=feature.dtype) / selected_count
    fade_ratio = 1.0 - valid_bin_ratio
    phase_coherence = (
        phase_q.real.abs() * phase_valid.to(dtype=phase_q.real.dtype)
    ).sum(dim=1) / phase_valid.sum(dim=1).clamp_min(1)
    clip_ratio = (signal.abs() >= (0.98 * raw_peak).unsqueeze(1)).float().mean(dim=1)
    center = fft_bins // 2
    dc_ratio = amplitude[:, center] / amplitude.sum(dim=1).clamp_min(eps)
    selected_power = amplitude_unclamped[:, mask].square()
    snr_proxy = 10.0 * torch.log10(
        selected_power.mean(dim=1).clamp_min(eps)
        / selected_power.median(dim=1).values.clamp_min(eps)
    )
    quality = torch.stack(
        (valid_bin_ratio, fade_ratio, phase_coherence, trend_error, clip_ratio, dc_ratio, snr_proxy),
        dim=1,
    )
    diagnostics = {
        "group_norms": group_norms,
        "valid_bin_ratio": torch.nan_to_num(valid_bin_ratio),
        "quality": torch.nan_to_num(quality),
        "mirror_valid_ratio": mirror_valid.float().mean(dim=1),
        "phase_valid_ratio": phase_valid.float().mean(dim=1),
    }
    return feature, diagnostics


class HSIDPrototypeEvidence(nn.Module):
    """Independent normalized spectral prototypes with bounded Raw-first fusion."""

    def __init__(
        self,
        num_classes: int,
        spectral_dim: int = 48,
        alpha_max: float = 0.20,
        temperature: float = 0.10,
    ) -> None:
        super().__init__()
        if num_classes <= 1 or spectral_dim <= 0:
            raise ValueError("num_classes and spectral_dim must be positive")
        if not 0.0 <= alpha_max <= 1.0:
            raise ValueError("alpha_max must be in [0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.alpha_max = float(alpha_max)
        self.temperature = float(temperature)
        self.encoder = nn.Sequential(
            nn.Linear(SID_FFT96_DIM, SID_FFT96_DIM),
            nn.GELU(),
            nn.Linear(SID_FFT96_DIM, spectral_dim),
        )
        self.prototypes = nn.Parameter(torch.empty(num_classes, spectral_dim))
        nn.init.normal_(self.prototypes, std=0.02)
        self.quality_gate = nn.Sequential(
            nn.Linear(11, 24),
            nn.GELU(),
            nn.Linear(24, 1),
        )
        self.fusion_alpha = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _margin(logits: Tensor) -> Tensor:
        if logits.shape[1] < 2:
            return logits.new_zeros(logits.shape[0])
        top2 = logits.topk(2, dim=1).values
        return top2[:, 0] - top2[:, 1]

    def forward(self, features: Tensor, raw_logits: Tensor, quality: Tensor) -> Dict[str, Tensor]:
        if features.ndim != 2 or features.shape[1] != SID_FFT96_DIM:
            raise ValueError("features must have shape [B, 96]")
        if raw_logits.ndim != 2 or raw_logits.shape[0] != features.shape[0]:
            raise ValueError("raw_logits must have shape [B, C]")
        if quality.shape != (features.shape[0], 7):
            raise ValueError("quality must have shape [B, 7]")
        spectral_embedding = F.normalize(self.encoder(features), dim=1, eps=1e-6)
        prototypes = F.normalize(self.prototypes, dim=1, eps=1e-6)
        spectral_logits = spectral_embedding @ prototypes.transpose(0, 1) / self.temperature
        raw_probability = raw_logits.softmax(dim=1)
        spectral_probability = spectral_logits.softmax(dim=1)
        mean_probability = 0.5 * (raw_probability + spectral_probability)
        js_divergence = 0.5 * (
            F.kl_div(mean_probability.clamp_min(1e-8).log(), raw_probability, reduction="none").sum(dim=1)
            + F.kl_div(mean_probability.clamp_min(1e-8).log(), spectral_probability, reduction="none").sum(dim=1)
        )
        agreement = (raw_logits.argmax(dim=1) == spectral_logits.argmax(dim=1)).to(dtype=features.dtype)
        gate_input = torch.cat(
            (
                quality,
                self._margin(raw_logits).unsqueeze(1),
                self._margin(spectral_logits).unsqueeze(1),
                js_divergence.unsqueeze(1),
                agreement.unsqueeze(1),
            ),
            dim=1,
        )
        reliability = torch.sigmoid(self.quality_gate(gate_input)).squeeze(1)
        bounded_alpha = self.fusion_alpha.clamp(min=0.0, max=self.alpha_max)
        # Straight-through projection keeps the forward gate exactly bounded while
        # allowing a temporarily negative optimizer state to recover later.
        alpha = self.fusion_alpha + (bounded_alpha - self.fusion_alpha).detach()
        fusion_gate = alpha * reliability
        centered = spectral_logits - spectral_logits.mean(dim=1, keepdim=True)
        calibrated = centered / centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        fused_logits = raw_logits + fusion_gate.unsqueeze(1) * calibrated
        return {
            "spectral_embedding": spectral_embedding,
            "spectral_logits": spectral_logits,
            "fused_logits": fused_logits,
            "fusion_gate": fusion_gate,
            "fusion_reliability": reliability,
            "raw_margin": self._margin(raw_logits),
            "spectral_margin": self._margin(spectral_logits),
            "js_divergence": js_divergence,
            "agreement": agreement,
        }


class HSIDFFT96Evidence(nn.Module):
    """Extract corrected FFT96 descriptors and emit independent prototype evidence."""

    def __init__(
        self,
        *,
        num_classes: int,
        mode: str,
        mask: Tensor,
        spectral_dim: int = 48,
        alpha_max: float = 0.20,
    ) -> None:
        super().__init__()
        if mode not in SID_MODES:
            raise ValueError(f"unsupported SID mode: {mode!r}")
        self.mode = mode
        self.register_buffer(
            "mask",
            validate_sid_mask(mask, int(torch.as_tensor(mask).numel())),
            persistent=True,
        )
        self.evidence = HSIDPrototypeEvidence(
            num_classes=num_classes,
            spectral_dim=spectral_dim,
            alpha_max=alpha_max,
        )

    def forward(self, iq: Tensor, raw_logits: Tensor) -> Dict[str, Tensor]:
        features, diagnostics = extract_sid_fft96(iq, self.mask, mode=self.mode)
        output = self.evidence(features, raw_logits, diagnostics["quality"])
        output.update(
            {
                "sid_fft96": features,
                "sid_group_norms": diagnostics["group_norms"],
                "sid_valid_bin_ratio": diagnostics["valid_bin_ratio"],
                "sid_quality": diagnostics["quality"],
            }
        )
        return output


class SIDFFT96Residual(nn.Module):
    """Zero-initialized residual from SID-FFT96 into the identity embedding."""

    def __init__(
        self,
        embedding_dim: int,
        mode: str,
        mask: Tensor,
        residual_scale: float = 1.0,
        max_residual_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if mode not in SID_MODES:
            raise ValueError(f"unsupported SID mode: {mode!r}")
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        if max_residual_ratio < 0:
            raise ValueError("max_residual_ratio must be non-negative")
        mask = validate_sid_mask(mask, int(torch.as_tensor(mask).numel()))
        self.mode = mode
        self.residual_scale = float(residual_scale)
        self.max_residual_ratio = float(max_residual_ratio)
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
        delta_raw = self.projector(features)
        delta = self.residual_scale * delta_raw
        if self.max_residual_ratio > 0.0:
            raw_norm = z_raw.detach().norm(dim=1, keepdim=True)
            delta_norm = delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
            max_norm = self.max_residual_ratio * raw_norm
            delta = delta * (max_norm / delta_norm).clamp(max=1.0)
        z_sid = z_raw + delta
        return {
            "z_raw": z_raw,
            "z_sid": z_sid,
            "sid_fft96": features,
            "sid_delta": delta,
            "sid_delta_raw": delta_raw,
            "sid_group_norms": diagnostics["group_norms"],
            "sid_valid_bin_ratio": diagnostics["valid_bin_ratio"],
        }
