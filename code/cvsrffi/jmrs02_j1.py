"""Role-correct JMRS02 J1 single-module mechanisms.

J1 deliberately contains no joint rows.  Every identity residual is an exact
Core90 bypass at initialization, RX1 is an identity waveform transform, and P0
is a nuisance estimator rather than a transmitter classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


J1_ROWS = ("B0", "RZ0", "RZ1", "RX1", "D1P", "P0")


@dataclass(frozen=True)
class J1Config:
    z_dim: int
    num_classes: int
    hidden_dim: int = 32
    nuisance_dim: int = 16
    smooth_basis: int = 12
    correction_radius: float = 0.20
    spectral_mask_ratio: float = 0.10
    cepstral_trend_bins: int = 12
    eps: float = 1e-6
    seed: int = 20260826

    def __post_init__(self) -> None:
        if self.z_dim <= 0 or self.num_classes <= 1:
            raise ValueError("z_dim must be positive and num_classes must exceed one")
        if not 8 <= self.smooth_basis <= 16:
            raise ValueError("smooth_basis must be between 8 and 16")
        if not 0.0 < self.correction_radius <= 1.0:
            raise ValueError("correction_radius must be in (0, 1]")


@dataclass
class J1Output:
    final_logits: Tensor
    residual_logits: Tensor
    gate_logits: Tensor
    embedding: Tensor
    corrected_iq: Tensor | None = None
    nuisance_prediction: Tensor | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def validate_j1_rows(rows: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(row).strip().upper() for row in rows)
    unknown = sorted(set(normalized).difference(J1_ROWS))
    if unknown:
        raise ValueError(f"unsupported or joint JMRS02 J1 rows: {unknown}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("JMRS02 J1 rows must be unique")
    return normalized


def _as_complex(iq: Tensor) -> Tensor:
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError("iq must have shape [batch, 2, samples]")
    return torch.complex(iq[:, 0], iq[:, 1])


def _pool(values: Tensor, bins: int) -> Tensor:
    return F.adaptive_avg_pool1d(values.unsqueeze(1), bins).squeeze(1)


def _zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _safe_zero_rms(values: Tensor, eps: float = 1e-12) -> Tensor:
    """RMS that is exactly zero and has a finite derivative at zero."""
    return torch.sqrt(values.square().mean(1) + eps) - math.sqrt(eps)


def _dct_basis(terms: int, length: int = 256) -> Tensor:
    n = torch.arange(length, dtype=torch.float32) + 0.5
    k = torch.arange(terms, dtype=torch.float32).unsqueeze(1)
    return F.normalize(torch.cos(math.pi * k * n / float(length)), dim=1)


def receiver_condition(iq: Tensor) -> Tensor:
    x = _as_complex(iq)
    shifted = torch.fft.fftshift(torch.fft.fft(x, dim=-1), dim=-1)
    log_magnitude = torch.log(shifted.abs().clamp_min(1e-6))
    step = x[:, 1:] * x[:, :-1].conj()
    amplitude = x.abs()
    rms = amplitude.square().mean(1).sqrt().clamp_min(1e-6)
    papr = amplitude.amax(1) / rms
    clipping = (amplitude > 2.5 * rms[:, None]).float().mean(1)
    return torch.cat(
        (
            _pool(log_magnitude, 16),
            _pool(step.real, 8),
            _pool(step.imag, 8),
            rms[:, None],
            papr[:, None],
            clipping[:, None],
        ),
        dim=1,
    )


class _ResidualBase(nn.Module):
    def __init__(self, feature_dim: int, cfg: J1Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.project = nn.Sequential(
            nn.Linear(feature_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
        )
        self.residual = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.gate = nn.Linear(cfg.hidden_dim + 3, 2)
        _zero_linear(self.residual)
        _zero_linear(self.gate)

    def _finish(self, features: Tensor, base_logits: Tensor, diagnostics: Mapping[str, Any]) -> J1Output:
        embedding = self.project(features)
        residual = self.residual(embedding)
        probability = F.softmax(base_logits.detach(), dim=1)
        top2 = torch.topk(probability, k=min(2, probability.size(1)), dim=1).values
        margin = top2[:, 0] - (top2[:, 1] if top2.size(1) > 1 else 0.0)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(1)
        disagreement = residual.detach().square().mean(1).sqrt()
        gate_logits = self.gate(torch.cat((embedding.detach(), margin[:, None], entropy[:, None], disagreement[:, None]), 1))
        return J1Output(base_logits + residual, residual, gate_logits, embedding, diagnostics=diagnostics)


class RZ0Control(_ResidualBase):
    def __init__(self, cfg: J1Config) -> None:
        super().__init__(cfg.z_dim, cfg)

    def forward(self, *, iq: Tensor, z_id: Tensor, base_logits: Tensor, domain: Tensor) -> J1Output:
        del iq, domain
        return self._finish(z_id, base_logits, {"conditioning": z_id.detach(), "role": "same_head_no_correction"})


class IQConditionedRCZ(_ResidualBase):
    def __init__(self, cfg: J1Config) -> None:
        super().__init__(cfg.z_dim, cfg)
        self.condition_encoder = nn.Sequential(nn.Linear(35, 32), nn.GELU(), nn.Linear(32, cfg.nuisance_dim))
        self.correction = nn.Linear(cfg.nuisance_dim, cfg.z_dim, bias=False)
        _zero_linear(self.correction)

    def forward(self, *, iq: Tensor, z_id: Tensor, base_logits: Tensor, domain: Tensor) -> J1Output:
        del domain
        condition = self.condition_encoder(receiver_condition(iq))
        raw = self.correction(torch.tanh(condition))
        radius = self.cfg.correction_radius * z_id.norm(p=1, dim=1, keepdim=True).clamp_min(self.cfg.eps) / z_id.size(1)
        scale = torch.clamp(radius / raw.norm(p=1, dim=1, keepdim=True).clamp_min(self.cfg.eps), max=1.0)
        correction = raw * scale
        output = self._finish(
            z_id - correction,
            base_logits,
            {"conditioning": condition.detach(), "correction_norm": correction.norm(dim=1), "role": "iq_conditioned_feature_correction"},
        )
        return output


class IdentityInitRCX(nn.Module):
    def __init__(self, cfg: J1Config, *, conditioning_enabled: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.conditioning_enabled = bool(conditioning_enabled)
        self.register_buffer("basis", _dct_basis(cfg.smooth_basis), persistent=True)
        self.estimator = nn.Sequential(nn.Linear(35, 32), nn.GELU(), nn.Linear(32, 2 * cfg.smooth_basis))
        _zero_linear(self.estimator[-1])
        self.gate = nn.Linear(6, 2)
        _zero_linear(self.gate)

    def forward(self, *, iq: Tensor, z_id: Tensor, base_logits: Tensor, domain: Tensor) -> J1Output:
        del domain
        condition = receiver_condition(iq)
        estimator_condition = condition if self.conditioning_enabled else torch.zeros_like(condition)
        coeff = torch.tanh(self.estimator(estimator_condition))
        amp_coeff, phase_coeff = coeff.chunk(2, dim=1)
        amplitude_curve = 0.30 * (amp_coeff @ self.basis)
        phase_curve = 0.40 * (phase_coeff @ self.basis)
        spectrum = torch.fft.fftshift(torch.fft.fft(_as_complex(iq), dim=-1), dim=-1)
        magnitude = spectrum.abs()
        floor = self.cfg.spectral_mask_ratio * magnitude.square().mean(1).sqrt()[:, None]
        observable = magnitude > floor.clamp_min(self.cfg.eps)
        amplitude_curve = torch.where(observable, amplitude_curve, torch.zeros_like(amplitude_curve))
        phase_curve = torch.where(observable, phase_curve, torch.zeros_like(phase_curve))
        corrected_shifted = spectrum * torch.exp(-amplitude_curve) * torch.polar(torch.ones_like(phase_curve), -phase_curve)
        corrected = torch.fft.ifft(torch.fft.ifftshift(corrected_shifted, dim=-1), dim=-1)
        corrected_iq = torch.stack((corrected.real, corrected.imag), 1)
        raw_power = iq.square().mean((1, 2)).sqrt().clamp_min(self.cfg.eps)
        corrected_power = corrected_iq.square().mean((1, 2)).sqrt().clamp_min(self.cfg.eps)
        corrected_iq = corrected_iq * (raw_power / corrected_power)[:, None, None]
        zeros = torch.zeros_like(base_logits)
        gate_features = torch.stack(
            (
                observable.float().mean(1),
                _safe_zero_rms(amplitude_curve),
                _safe_zero_rms(phase_curve),
                condition[:, -3],
                condition[:, -2],
                condition[:, -1],
            ),
            1,
        )
        return J1Output(
            base_logits,
            zeros,
            self.gate(gate_features.detach()),
            z_id,
            corrected_iq=corrected_iq,
            diagnostics={
                "fftshifted": torch.ones(iq.size(0), device=iq.device, dtype=torch.bool),
                "valid_bin_fraction": observable.float().mean(1),
                "correction_norm": _safe_zero_rms(amplitude_curve) + _safe_zero_rms(phase_curve),
                "role": "iq_conditioned_waveform_canonicalizer" if self.conditioning_enabled else "same_capacity_global_waveform_control",
            },
        )


def cepstral_spectral_residual(iq: Tensor, trend_bins: int) -> tuple[Tensor, Tensor]:
    shifted = torch.fft.fftshift(torch.fft.fft(_as_complex(iq), dim=-1), dim=-1)
    magnitude = shifted.abs()
    floor = 0.10 * magnitude.square().mean(1).sqrt()[:, None]
    valid = magnitude > floor.clamp_min(1e-6)
    log_magnitude = torch.log(magnitude.clamp_min(floor.clamp_min(1e-6)))
    cepstrum = torch.fft.ifft(log_magnitude.to(torch.complex64), dim=-1).real
    keep = torch.ones_like(cepstrum)
    keep[:, :trend_bins] = 0.0
    keep[:, -trend_bins + 1 :] = 0.0
    residual = torch.fft.fft((cepstrum * keep).to(torch.complex64), dim=-1).real
    residual = residual * valid.float()
    return residual, valid.float().mean(1)


class RobustSpectralResidual(_ResidualBase):
    def __init__(self, cfg: J1Config) -> None:
        super().__init__(cfg.z_dim + 32, cfg)

    def forward(self, *, iq: Tensor, z_id: Tensor, base_logits: Tensor, domain: Tensor) -> J1Output:
        del domain
        residual_spectrum, coverage = cepstral_spectral_residual(iq, self.cfg.cepstral_trend_bins)
        features = torch.cat((z_id, _pool(residual_spectrum, 32)), dim=1)
        return self._finish(
            features,
            base_logits,
            {
                "feature_family": "cepstral_log_spectrum_residual_no_ratio",
                "unknown_symbol_invariant_claim": False,
                "valid_bin_fraction": coverage,
                "role": "bounded_identity_residual_expert",
            },
        )


def circular_phase_features(iq: Tensor) -> Tensor:
    x = _as_complex(iq)
    amplitude = x.abs()
    threshold = 0.20 * amplitude.square().mean(1).sqrt()[:, None]
    valid = (amplitude[:, 1:] > threshold) & (amplitude[:, :-1] > threshold)
    step = x[:, 1:] * x[:, :-1].conj()
    unit = step / step.abs().clamp_min(1e-6)
    mask = valid.float()
    count = mask.sum(1).clamp_min(1.0)
    mean_real = (unit.real * mask).sum(1) / count
    mean_imag = (unit.imag * mask).sum(1) / count
    concentration = torch.sqrt(mean_real.square() + mean_imag.square())
    phase = torch.atan2(unit.imag, unit.real)
    centered = torch.atan2(torch.sin(phase - torch.atan2(mean_imag, mean_real)[:, None]), torch.cos(phase - torch.atan2(mean_imag, mean_real)[:, None]))
    variance = (centered.square() * mask).sum(1) / count
    jumps = ((centered.abs() > math.pi / 2).float() * mask).sum(1) / count
    coverage = mask.mean(1)
    return torch.stack((mean_real, mean_imag, concentration, variance, jumps, coverage), 1)


class PhaseNuisanceOnly(nn.Module):
    def __init__(self, cfg: J1Config) -> None:
        super().__init__()
        self.predictor = nn.Sequential(nn.Linear(6, 24), nn.GELU(), nn.Linear(24, 4))
        self.gate = nn.Linear(9, 2)
        _zero_linear(self.gate)

    def forward(self, *, iq: Tensor, z_id: Tensor, base_logits: Tensor, domain: Tensor) -> J1Output:
        del domain
        features = circular_phase_features(iq)
        nuisance = self.predictor(features)
        probability = F.softmax(base_logits.detach(), 1)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(1, keepdim=True)
        margin = torch.topk(probability, 2, dim=1).values.diff(dim=1).abs()
        gate_logits = self.gate(torch.cat((features, nuisance.detach()[:, :1], entropy, margin), 1))
        zeros = torch.zeros_like(base_logits)
        return J1Output(
            base_logits,
            zeros,
            gate_logits,
            z_id,
            nuisance_prediction=nuisance,
            diagnostics={"phase_representation": "unit_phasor_circular_statistics", "role": "nuisance_and_quality_only"},
        )


def build_j1_module(row: str, cfg: J1Config) -> nn.Module:
    normalized = validate_j1_rows((row,))[0]
    if normalized == "B0":
        raise ValueError("B0 is the external frozen Core90 baseline")
    constructors = {
        "RZ0": RZ0Control,
        "RZ1": IQConditionedRCZ,
        "RX1": IdentityInitRCX,
        "D1P": RobustSpectralResidual,
        "P0": PhaseNuisanceOnly,
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed + J1_ROWS.index(normalized) * 101)
        return constructors[normalized](cfg)


__all__ = ["J1Config", "J1Output", "J1_ROWS", "build_j1_module", "validate_j1_rows"]
