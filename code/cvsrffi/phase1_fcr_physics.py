from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn


FEATURE_BLOCKS = (
    "iq_non_circularity",
    "am_am",
    "am_pm",
    "memory_residual",
    "spectral_shoulder",
    "phase_noise_psd",
    "amplitude_conditioned_residual",
    "cyclostationary",
)


def _as_complex_iq(iq: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(iq):
        if iq.ndim != 2:
            raise ValueError("complex IQ must have shape [B,T]")
        return iq
    if iq.ndim != 3 or iq.size(1) != 2:
        raise ValueError("real IQ must have shape [B,2,T]")
    return torch.complex(iq[:, 0], iq[:, 1])


def _safe_complex(value: torch.Tensor) -> torch.Tensor:
    return torch.complex(
        torch.nan_to_num(value.real, nan=0.0, posinf=1e6, neginf=-1e6),
        torch.nan_to_num(value.imag, nan=0.0, posinf=1e6, neginf=-1e6),
    )


@dataclass(frozen=True)
class FingerprintFeatures:
    """Fixed named physical summaries, one tensor per feature family."""

    blocks: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class FisherGateOutput:
    block_weights: Mapping[str, torch.Tensor]
    quality: Mapping[str, torch.Tensor]


class FrozenFingerprintFeatureBank(nn.Module):
    """Deterministic, parameter-free physical fingerprint summaries."""

    def __init__(self, *, eps: float = 1e-8, noise_floor: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.noise_floor = float(noise_floor)

    def forward(self, iq: torch.Tensor) -> FingerprintFeatures:
        signal = _safe_complex(_as_complex_iq(iq))
        amplitude = signal.abs()
        power = amplitude.square()
        mean_power = power.mean(dim=-1, keepdim=True).clamp_min(self.eps)
        centered = signal - signal.mean(dim=-1, keepdim=True)
        normalized = centered / centered.abs().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(self.eps)
        pseudo_covariance = normalized.square().mean(dim=-1)
        covariance = (normalized.real * normalized.imag).mean(dim=-1)
        iq_non_circularity = torch.stack(
            (pseudo_covariance.real, pseudo_covariance.imag, covariance), dim=-1
        )

        amp_normalized = amplitude / mean_power.sqrt()
        am_am = torch.stack(
            (amp_normalized.mean(dim=-1), amp_normalized.square().mean(dim=-1), amp_normalized.pow(3).mean(dim=-1)),
            dim=-1,
        )
        increments = signal[:, 1:] * signal[:, :-1].conj()
        unit_increment = increments / increments.abs().clamp_min(self.eps)
        amp_pair = amplitude[:, 1:] * amplitude[:, :-1]
        am_pm = torch.stack(
            (
                (unit_increment.real * amp_pair).mean(dim=-1),
                (unit_increment.imag * amp_pair).mean(dim=-1),
                (unit_increment.real * amp_normalized[:, 1:]).mean(dim=-1),
            ),
            dim=-1,
        )

        delayed = torch.cat((signal[:, :1], signal[:, :-1]), dim=-1)
        memory_error = signal - delayed
        memory_residual = torch.stack(
            (
                memory_error.abs().square().mean(dim=-1) / mean_power.squeeze(-1),
                (memory_error * signal.conj()).real.mean(dim=-1) / mean_power.squeeze(-1),
            ),
            dim=-1,
        )

        spectrum = torch.fft.fft(signal, dim=-1)
        spectral_power = spectrum.abs().square() / float(max(1, signal.size(-1)))
        frequency = torch.fft.fftfreq(signal.size(-1), device=signal.device).abs()
        shoulder = spectral_power[:, frequency >= 0.30].mean(dim=-1)
        core = spectral_power[:, frequency <= 0.15].mean(dim=-1).clamp_min(self.noise_floor)
        spectral_shoulder = torch.stack((shoulder / core, shoulder.log1p()), dim=-1)

        phase_psd = torch.fft.fft(unit_increment, dim=-1).abs().square() / float(max(1, unit_increment.size(-1)))
        phase_frequency = torch.fft.fftfreq(unit_increment.size(-1), device=signal.device).abs()
        low_noise = phase_psd[:, (phase_frequency > 0.02) & (phase_frequency <= 0.15)].mean(dim=-1)
        high_noise = phase_psd[:, phase_frequency > 0.15].mean(dim=-1)
        phase_noise_psd = torch.stack((low_noise, high_noise, high_noise / low_noise.clamp_min(self.noise_floor)), dim=-1)

        centered_amplitude = amp_normalized - amp_normalized.mean(dim=-1, keepdim=True)
        amplitude_conditioned_residual = torch.stack(
            (
                (centered_amplitude[:, 1:] * memory_error[:, 1:].abs()).mean(dim=-1),
                (centered_amplitude[:, 1:] * unit_increment.imag).mean(dim=-1),
            ),
            dim=-1,
        )

        cyclostationary_parts = []
        for lag in (1, 4, 8):
            if signal.size(-1) <= lag:
                cyclostationary_parts.append(torch.zeros_like(mean_power.squeeze(-1)))
            else:
                correlation = (signal[:, lag:] * signal[:, :-lag].conj()).mean(dim=-1)
                cyclostationary_parts.append(correlation.abs() / mean_power.squeeze(-1))
        cyclostationary = torch.stack(cyclostationary_parts, dim=-1)
        blocks = {
            "iq_non_circularity": iq_non_circularity,
            "am_am": am_am,
            "am_pm": am_pm,
            "memory_residual": memory_residual,
            "spectral_shoulder": spectral_shoulder,
            "phase_noise_psd": phase_noise_psd,
            "amplitude_conditioned_residual": amplitude_conditioned_residual,
            "cyclostationary": cyclostationary,
        }
        return FingerprintFeatures({name: torch.nan_to_num(value) for name, value in blocks.items()})


class FisherIdentifiabilityGate(nn.Module):
    """Stop-gradient confidence gate for fixed physical loss blocks."""

    def __init__(self, *, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        iq: torch.Tensor,
        gram: torch.Tensor | None,
        snr_db: torch.Tensor | float | None,
        noise_floor: torch.Tensor | float | None = None,
        excitation_coverage: torch.Tensor | float | None = None,
    ) -> FisherGateOutput:
        signal = _as_complex_iq(iq).detach()
        batch_size = signal.size(0)
        power = signal.abs().square()
        mean_power = power.mean(dim=-1).clamp_min(self.eps)
        papr = (power.max(dim=-1).values / mean_power).detach()
        papr_quality = ((papr - 1.0) / 4.0).clamp(0.0, 1.0)
        if excitation_coverage is None:
            amplitude = signal.abs()
            spread = amplitude.std(dim=-1, unbiased=False) / amplitude.mean(dim=-1).clamp_min(self.eps)
            coverage = (0.15 + spread).clamp(0.0, 1.0)
        else:
            coverage = torch.as_tensor(excitation_coverage, device=signal.device, dtype=signal.real.dtype).detach().reshape(-1)
            coverage = coverage.expand(batch_size) if coverage.numel() == 1 else coverage
            coverage = torch.nan_to_num(coverage, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        gram_rank = self._effective_rank(gram, batch_size, signal.device, signal.real.dtype)
        snr_quality = self._snr_quality(snr_db, batch_size, signal.device, signal.real.dtype)
        if noise_floor is None:
            floor_quality = (mean_power / (mean_power + 1e-4)).detach()
        else:
            floor = torch.as_tensor(noise_floor, device=signal.device, dtype=signal.real.dtype).detach().reshape(-1)
            floor = floor.expand(batch_size) if floor.numel() == 1 else floor
            floor_quality = (mean_power / (mean_power + torch.nan_to_num(floor, nan=1e6, posinf=1e6, neginf=1e6).clamp_min(self.eps))).clamp(0.0, 1.0)
        base = (gram_rank * snr_quality * floor_quality).clamp(0.0, 1.0).detach()
        pa = (base * papr_quality * coverage).clamp(0.0, 1.0).detach()
        weights = {
            "pa": pa,
            "iq_non_circularity": base,
            "am_am": pa,
            "am_pm": pa,
            "memory_residual": pa,
            "spectral_shoulder": base,
            "phase_noise_psd": base,
            "amplitude_conditioned_residual": pa,
            "cyclostationary": base * coverage,
        }
        quality = {
            "gram_effective_rank": gram_rank.detach(),
            "excitation_coverage": coverage.detach(),
            "papr": papr.detach(),
            "papr_quality": papr_quality.detach(),
            "snr_quality": snr_quality.detach(),
            "noise_floor_quality": floor_quality.detach(),
        }
        return FisherGateOutput(weights, quality)

    def _effective_rank(self, gram: torch.Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if gram is None:
            return torch.zeros(batch_size, device=device, dtype=dtype)
        matrix = torch.as_tensor(gram, device=device, dtype=dtype).detach()
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        if matrix.ndim != 3 or matrix.size(-1) != matrix.size(-2):
            return torch.zeros(batch_size, device=device, dtype=dtype)
        matrix = torch.nan_to_num(matrix)
        singular = torch.linalg.svdvals(matrix)
        total = singular.sum(dim=-1, keepdim=True)
        probabilities = singular / total.clamp_min(self.eps)
        entropy = -(probabilities * probabilities.clamp_min(self.eps).log()).sum(dim=-1)
        rank = entropy.exp() / float(max(1, matrix.size(-1)))
        rank = torch.where(total.squeeze(-1) > self.eps, rank, torch.zeros_like(rank))
        return rank.expand(batch_size) if rank.numel() == 1 else rank[:batch_size]

    @staticmethod
    def _snr_quality(snr_db: torch.Tensor | float | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if snr_db is None:
            return torch.zeros(batch_size, device=device, dtype=dtype)
        value = torch.as_tensor(snr_db, device=device, dtype=dtype).detach().reshape(-1)
        value = value.expand(batch_size) if value.numel() == 1 else value
        finite = torch.isfinite(value)
        quality = torch.sigmoid((torch.nan_to_num(value, nan=-100.0, posinf=100.0, neginf=-100.0) - 3.0) / 6.0)
        return torch.where(finite, quality, torch.zeros_like(quality)).clamp(0.0, 1.0)


def fingerprint_energy_penalty(delta_f: torch.Tensor, reference: torch.Tensor, *, ratio_limit: float = 0.10) -> torch.Tensor:
    delta = _as_complex_iq(delta_f)
    baseline = _as_complex_iq(reference)
    ratio = delta.norm(dim=-1) / baseline.norm(dim=-1).clamp_min(1e-8)
    return (ratio - float(ratio_limit)).clamp_min(0.0).square().mean()


def response_smoothness_penalty(response: torch.Tensor) -> torch.Tensor:
    if response.size(-1) < 2:
        return response.real.new_zeros(())
    return (response[..., 1:] - response[..., :-1]).abs().square().mean()


def parameter_boundary_penalty(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return ((float(lower) - value).clamp_min(0.0).square() + (value - float(upper)).clamp_min(0.0).square()).mean()
