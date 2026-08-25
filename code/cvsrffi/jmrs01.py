"""JMRS01 source-only mechanism bank.

The module intentionally contains only the S0 mechanisms.  Core90 is an
external frozen baseline (M0); no fusion gate or PA sidecar lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ALLOWED_S0_ROWS = ("M0", "R1", "R2", "D1", "P1", "P2", "S1")
_REMOVED_SYMBOL_DEPENDENT_ROWS = {"D2"}


@dataclass(frozen=True)
class JMRS01Config:
    z_dim: int
    num_classes: int
    embedding_dim: int = 32
    correction_rank: int = 16
    correction_radius: float = 0.25
    smooth_basis: int = 16
    spectral_mask_ratio: float = 0.10
    amplitude_mask_ratio: float = 0.20
    quotient_clip: float = 8.0
    eps: float = 1e-6
    seed: int = 20260826

    def __post_init__(self) -> None:
        if self.z_dim <= 0 or self.num_classes <= 1:
            raise ValueError("z_dim must be positive and num_classes must exceed one")
        if self.embedding_dim != 32:
            raise ValueError("JMRS01 S0 uses a frozen 32-dimensional mechanism budget")
        if not 0.0 < self.correction_radius <= 1.0:
            raise ValueError("correction_radius must be in (0, 1]")
        if self.smooth_basis not in range(8, 17):
            raise ValueError("smooth_basis must be between 8 and 16")


@dataclass
class MechanismOutput:
    embedding: Tensor
    logits: Tensor
    reliability: Tensor
    diagnostics: Mapping[str, Tensor] = field(default_factory=dict)


def validate_s0_rows(rows: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(row).upper() for row in rows)
    removed = sorted(set(normalized).intersection(_REMOVED_SYMBOL_DEPENDENT_ROWS))
    if removed:
        raise ValueError(
            "D2 requires known transmitted symbols and verified same-symbol, "
            "different-time observations; it is removed from JMRS01"
        )
    unknown = sorted(set(normalized).difference(ALLOWED_S0_ROWS))
    if unknown:
        raise ValueError(f"unsupported JMRS01 S0 rows: {unknown}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("JMRS01 S0 rows must be unique")
    return normalized


def _as_complex(iq: Tensor) -> Tensor:
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError("iq must have shape [batch, 2, samples]")
    return torch.complex(iq[:, 0], iq[:, 1])


def _wrap_phase(value: Tensor) -> Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def _pool(sequence: Tensor, bins: int) -> Tensor:
    return F.adaptive_avg_pool1d(sequence.unsqueeze(1), bins).squeeze(1)


def _waveform_summary(iq: Tensor) -> Tensor:
    x = _as_complex(iq)
    spectrum = torch.fft.fft(x, dim=-1)
    log_mag = torch.log(torch.abs(spectrum) + 1e-6)
    phase_step = torch.angle(x[:, 1:] * x[:, :-1].conj())
    return torch.cat(
        (
            _pool(iq[:, 0], 16),
            _pool(iq[:, 1], 16),
            _pool(log_mag, 16),
            _pool(phase_step, 16),
        ),
        dim=1,
    )


class _Head(nn.Module):
    def __init__(self, input_dim: int, cfg: JMRS01Config) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, cfg.embedding_dim),
            nn.LayerNorm(cfg.embedding_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(cfg.embedding_dim, cfg.num_classes)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        embedding = self.embedding(features)
        return embedding, self.classifier(embedding)


class RCFeature32(nn.Module):
    def __init__(self, cfg: JMRS01Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.nuisance = nn.Linear(cfg.z_dim, cfg.correction_rank)
        self.correction = nn.Linear(cfg.correction_rank, cfg.z_dim, bias=False)
        self.quality = nn.Linear(cfg.z_dim, 1)
        self.head = _Head(cfg.z_dim, cfg)

    def forward(self, *, iq: Tensor, z_id: Tensor) -> MechanismOutput:
        del iq
        raw = self.correction(torch.tanh(self.nuisance(z_id)))
        raw_norm = raw.norm(dim=1, keepdim=True).clamp_min(self.cfg.eps)
        scale = torch.clamp(self.cfg.correction_radius / raw_norm, max=1.0)
        bounded = raw * scale
        corrected = z_id - bounded
        embedding, logits = self.head(corrected)
        reliability = torch.sigmoid(self.quality(z_id)).squeeze(1)
        return MechanismOutput(
            embedding,
            logits,
            reliability,
            {"correction_norm": bounded.norm(dim=1)},
        )


def _dct_basis(terms: int, length: int = 256) -> Tensor:
    n = torch.arange(length, dtype=torch.float32) + 0.5
    k = torch.arange(terms, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos(math.pi * k * n / float(length))
    return F.normalize(basis, dim=1)


class RCSmooth16(nn.Module):
    def __init__(self, cfg: JMRS01Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("basis", _dct_basis(cfg.smooth_basis), persistent=True)
        self.estimator = nn.Sequential(
            nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 2 * cfg.smooth_basis)
        )
        self.head = _Head(64, cfg)

    def forward(self, *, iq: Tensor, z_id: Tensor) -> MechanismOutput:
        del z_id
        x = _as_complex(iq)
        spectrum = torch.fft.fft(x, dim=-1)
        magnitude = torch.abs(spectrum)
        phase_step = torch.angle(x[:, 1:] * x[:, :-1].conj())
        summary = torch.cat(
            (_pool(torch.log(magnitude + self.cfg.eps), 16), _pool(phase_step, 16)), dim=1
        )
        coeff = torch.tanh(self.estimator(summary))
        amp_coeff, phase_coeff = coeff.chunk(2, dim=1)
        amp_curve = 0.35 * (amp_coeff @ self.basis)
        phase_curve = 0.50 * (phase_coeff @ self.basis)
        corrected_spectrum = spectrum * torch.exp(-amp_curve) * torch.exp(
            torch.complex(torch.zeros_like(phase_curve), -phase_curve)
        )
        corrected = torch.fft.ifft(corrected_spectrum, dim=-1)
        corrected_iq = torch.stack((corrected.real, corrected.imag), dim=1)
        embedding, logits = self.head(_waveform_summary(corrected_iq))
        scale = magnitude.square().mean(dim=1).sqrt()
        valid = magnitude > torch.maximum(
            self.cfg.spectral_mask_ratio * scale[:, None],
            torch.full_like(magnitude, self.cfg.eps),
        )
        reliability = valid.float().mean(dim=1)
        smooth = amp_curve.diff(n=2, dim=1).square().mean(dim=1)
        smooth = smooth + phase_curve.diff(n=2, dim=1).square().mean(dim=1)
        return MechanismOutput(
            embedding,
            logits,
            reliability,
            {
                "valid_bin_fraction": reliability,
                "smooth_penalty": smooth,
                "amplitude_curve_max": amp_curve.abs().amax(dim=1),
                "phase_curve_max": phase_curve.abs().amax(dim=1),
            },
        )


class MultiScaleDSQ(nn.Module):
    _SHIFTS = (1, 2, 4, 8)

    def __init__(self, cfg: JMRS01Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.head = _Head(4 * 6 * 8, cfg)

    def _stable_quotient(self, numerator: Tensor, denominator: Tensor, mask: Tensor) -> Tensor:
        safe = torch.where(mask, denominator, torch.ones_like(denominator))
        quotient = numerator / safe
        magnitude = quotient.abs().clamp(max=self.cfg.quotient_clip)
        quotient = torch.polar(magnitude, torch.angle(quotient))
        return torch.where(mask, quotient, torch.zeros_like(quotient))

    def forward(self, *, iq: Tensor, z_id: Tensor) -> MechanismOutput:
        del z_id
        spectrum = torch.fft.fft(_as_complex(iq), dim=-1)
        magnitude = spectrum.abs()
        scale = magnitude.square().mean(dim=1).sqrt()
        threshold = torch.maximum(
            self.cfg.spectral_mask_ratio * scale[:, None],
            torch.full_like(magnitude, self.cfg.eps),
        )
        valid = magnitude > threshold
        features: list[Tensor] = []
        pair_coverages: list[Tensor] = []
        for shift in self._SHIFTS:
            plus = torch.roll(spectrum, shifts=-shift, dims=1)
            minus = torch.roll(spectrum, shifts=shift, dims=1)
            plus_mask = valid & torch.roll(valid, shifts=-shift, dims=1)
            minus_mask = valid & torch.roll(valid, shifts=shift, dims=1)
            q_plus = self._stable_quotient(plus, spectrum, plus_mask)
            q_minus = self._stable_quotient(minus, spectrum, minus_mask)
            features.extend(
                (
                    _pool(q_plus.abs(), 8),
                    _pool(q_minus.abs(), 8),
                    _pool(torch.log1p(q_plus.abs()), 8),
                    _pool(torch.log1p(q_minus.abs()), 8),
                    _pool(torch.angle(q_plus), 8),
                    _pool(torch.angle(q_minus), 8),
                )
            )
            pair_coverages.extend((plus_mask.float().mean(1), minus_mask.float().mean(1)))
        embedding, logits = self.head(torch.cat(features, dim=1))
        reliability = torch.stack(pair_coverages, dim=1).mean(dim=1)
        return MechanismOutput(
            embedding,
            logits,
            reliability,
            {
                "valid_bin_fraction": reliability,
                "feature_family_count": torch.full_like(reliability, 6.0),
            },
        )


def _phase_statistics(values: Tensor, mask: Tensor) -> Tensor:
    outputs: list[Tensor] = []
    for window in (4, 8, 16, 32):
        usable = (values.shape[1] // window) * window
        v = values[:, :usable].reshape(values.shape[0], -1, window)
        m = mask[:, :usable].reshape(mask.shape[0], -1, window).float()
        count = m.sum(dim=2).clamp_min(1.0)
        mean = (v * m).sum(dim=2) / count
        centered = (v - mean.unsqueeze(2)) * m
        variance = centered.square().sum(dim=2) / count
        mad = centered.abs().sum(dim=2) / count
        energy = (v.square() * m).sum(dim=2) / count
        fourth = centered.pow(4).sum(dim=2) / count
        excess_kurtosis = fourth / variance.square().clamp_min(1e-8) - 3.0
        masked_for_quantile = v.masked_fill(~m.bool(), float("nan"))
        quantiles = torch.nanquantile(
            masked_for_quantile,
            torch.tensor((0.10, 0.50, 0.90), device=v.device, dtype=v.dtype),
            dim=2,
        ).nan_to_num(0.0)
        spectrum = torch.fft.rfft(centered, dim=2).abs().square()
        spectrum_denominator = spectrum.sum(dim=2).clamp_min(1e-8)
        split1 = max(1, spectrum.size(2) // 3)
        split2 = max(split1 + 1, 2 * spectrum.size(2) // 3)
        psd_low = spectrum[:, :, :split1].sum(dim=2) / spectrum_denominator
        psd_mid = spectrum[:, :, split1:split2].sum(dim=2) / spectrum_denominator
        psd_high = spectrum[:, :, split2:].sum(dim=2) / spectrum_denominator
        if window > 1:
            pair = m[:, :, 1:] * m[:, :, :-1]
            pair_count = pair.sum(dim=2).clamp_min(1.0)
            differences = v[:, :, 1:] - v[:, :, :-1]
            ac1 = (v[:, :, 1:] * v[:, :, :-1] * pair).sum(dim=2) / pair_count
            diff_energy = (differences.square() * pair).sum(dim=2) / pair_count
            jump = ((differences.abs() > (math.pi / 2.0)).float() * pair).sum(dim=2) / pair_count
            allan = 0.5 * diff_energy
        else:  # pragma: no cover - current preregistered windows are all >1
            ac1 = torch.zeros_like(mean)
            diff_energy = torch.zeros_like(mean)
            jump = torch.zeros_like(mean)
            allan = torch.zeros_like(mean)
        coverage = m.mean(dim=2)
        outputs.append(
            torch.stack(
                (
                    variance.mean(1),
                    mad.mean(1),
                    ac1.mean(1),
                    psd_low.mean(1),
                    psd_mid.mean(1),
                    psd_high.mean(1),
                    diff_energy.mean(1),
                    jump.mean(1),
                    allan.mean(1),
                    quantiles[0].mean(1),
                    quantiles[1].mean(1),
                    quantiles[2].mean(1),
                    excess_kurtosis.mean(1),
                    coverage.mean(1),
                ),
                dim=1,
            )
        )
    return torch.cat(outputs, dim=1)


class MultiScalePhaseInnovation(nn.Module):
    def __init__(self, cfg: JMRS01Config, include_second_order: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.include_second_order = include_second_order
        self.head = _Head(112 if include_second_order else 56, cfg)

    def forward(self, *, iq: Tensor, z_id: Tensor) -> MechanismOutput:
        del z_id
        x = _as_complex(iq)
        amplitude = x.abs()
        rms = amplitude.square().mean(dim=1).sqrt()
        threshold = torch.maximum(
            self.cfg.amplitude_mask_ratio * rms[:, None],
            torch.full_like(amplitude, self.cfg.eps),
        )
        valid = amplitude > threshold
        first = torch.angle(x[:, 1:] * x[:, :-1].conj())
        first_mask = valid[:, 1:] & valid[:, :-1]
        n = torch.linspace(-1.0, 1.0, first.shape[1], device=first.device, dtype=first.dtype)
        design = torch.stack((torch.ones_like(n), n, n.square()), dim=1)
        weights = first_mask.float()
        gram = torch.einsum("bl,li,lj->bij", weights, design, design)
        regularizer = self.cfg.eps * torch.eye(3, device=first.device, dtype=first.dtype).unsqueeze(0)
        right = torch.einsum("bl,bl,li->bi", weights, first, design)
        coefficients = torch.linalg.solve(gram + regularizer, right.unsqueeze(2)).squeeze(2)
        trend = torch.einsum("li,bi->bl", design, coefficients)
        innovation1 = _wrap_phase(first - trend) * first_mask.float()
        features = [_phase_statistics(innovation1, first_mask)]
        if self.include_second_order:
            phase = torch.angle(x)
            innovation2 = _wrap_phase(phase[:, 2:] - 2.0 * phase[:, 1:-1] + phase[:, :-2])
            second_mask = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
            innovation2 = innovation2 * second_mask.float()
            features.append(_phase_statistics(innovation2, second_mask))
        embedding, logits = self.head(torch.cat(features, dim=1))
        reliability = first_mask.float().mean(dim=1)
        return MechanismOutput(
            embedding,
            logits,
            reliability,
            {
                "valid_sample_fraction": reliability,
                "statistic_count_per_scale": torch.full_like(reliability, 14.0),
            },
        )


class Sham32(nn.Module):
    def __init__(self, cfg: JMRS01Config) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(cfg.seed + 1009)
        random_projection = torch.randn(64, 192, generator=generator) / 8.0
        self.register_buffer("random_projection", random_projection, persistent=True)
        self.head = _Head(192, cfg)

    def forward(self, *, iq: Tensor, z_id: Tensor) -> MechanismOutput:
        del z_id
        features = torch.tanh(_waveform_summary(iq) @ self.random_projection)
        embedding, logits = self.head(features)
        reliability = torch.ones(iq.shape[0], device=iq.device, dtype=iq.dtype)
        return MechanismOutput(embedding, logits, reliability, {"sham_coverage": reliability})


def build_mechanism(row: str, cfg: JMRS01Config) -> nn.Module:
    normalized = validate_s0_rows([row])[0]
    if normalized == "M0":
        raise ValueError("M0 is the external frozen Core90 baseline, not a trainable mechanism")
    constructors = {
        "R1": lambda: RCFeature32(cfg),
        "R2": lambda: RCSmooth16(cfg),
        "D1": lambda: MultiScaleDSQ(cfg),
        "P1": lambda: MultiScalePhaseInnovation(cfg, include_second_order=False),
        "P2": lambda: MultiScalePhaseInnovation(cfg, include_second_order=True),
        "S1": lambda: Sham32(cfg),
    }
    row_seed = cfg.seed + ALLOWED_S0_ROWS.index(normalized) * 101
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(row_seed)
        return constructors[normalized]()


def _class_cond_receiver_loss(embedding: Tensor, labels: Tensor, receivers: Tensor) -> Tensor:
    terms: list[Tensor] = []
    for label in labels.unique():
        class_mask = labels == label
        class_receivers = receivers[class_mask].unique()
        means = [embedding[class_mask & (receivers == receiver)].mean(0) for receiver in class_receivers]
        for left in range(len(means)):
            for right in range(left + 1, len(means)):
                terms.append((means[left] - means[right]).square().mean())
    return torch.stack(terms).mean() if terms else embedding.sum() * 0.0


def _tx_margin_loss(embedding: Tensor, labels: Tensor, margin: float = 1.0) -> Tensor:
    centers = [embedding[labels == label].mean(0) for label in labels.unique()]
    terms: list[Tensor] = []
    for left in range(len(centers)):
        for right in range(left + 1, len(centers)):
            terms.append(F.relu(margin - torch.linalg.vector_norm(centers[left] - centers[right])))
    return torch.stack(terms).mean() if terms else embedding.sum() * 0.0


def mechanism_loss(
    clean: MechanismOutput,
    satellite: MechanismOutput,
    labels: Tensor,
    receivers: Tensor,
    *,
    base_logits: Tensor | None = None,
    clean_sat_weight: float = 0.20,
    receiver_weight: float = 0.05,
    margin_weight: float = 0.05,
    preserve_weight: float = 0.10,
    quality_weight: float = 0.02,
    regularization_weight: float = 0.01,
) -> dict[str, Tensor]:
    ce = 0.5 * (F.cross_entropy(clean.logits, labels) + F.cross_entropy(satellite.logits, labels))
    clean_sat = F.mse_loss(clean.embedding, satellite.embedding)
    class_cond_rx = _class_cond_receiver_loss(clean.embedding, labels, receivers)
    tx_margin = _tx_margin_loss(clean.embedding, labels)
    preserve = clean.embedding.sum() * 0.0
    if base_logits is not None:
        probability = F.softmax(base_logits.detach(), dim=1)
        preserve_mask = base_logits.detach().argmax(dim=1).eq(labels) & probability.max(dim=1).values.ge(0.80)
        if bool(preserve_mask.any()):
            preserve = F.kl_div(
                F.log_softmax(clean.logits[preserve_mask], dim=1),
                probability[preserve_mask],
                reduction="batchmean",
            )
    quality_target = clean.logits.detach().argmax(dim=1).eq(labels) & satellite.logits.detach().argmax(dim=1).eq(labels)
    quality = F.binary_cross_entropy(
        clean.reliability.clamp(1e-6, 1.0 - 1e-6), quality_target.float()
    )
    mechanism_regularization = clean.embedding.sum() * 0.0
    if "smooth_penalty" in clean.diagnostics:
        mechanism_regularization = mechanism_regularization + clean.diagnostics["smooth_penalty"].mean()
    total = (
        ce
        + clean_sat_weight * clean_sat
        + receiver_weight * class_cond_rx
        + margin_weight * tx_margin
        + preserve_weight * preserve
        + quality_weight * quality
        + regularization_weight * mechanism_regularization
    )
    return {
        "total": total,
        "ce": ce,
        "clean_sat": clean_sat,
        "class_cond_rx": class_cond_rx,
        "tx_margin": tx_margin,
        "preserve": preserve,
        "quality": quality,
        "mechanism_regularization": mechanism_regularization,
    }


__all__ = [
    "ALLOWED_S0_ROWS",
    "JMRS01Config",
    "MechanismOutput",
    "build_mechanism",
    "mechanism_loss",
    "validate_s0_rows",
]
