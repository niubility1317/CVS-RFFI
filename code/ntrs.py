"""Nuisance-Tangent Robust System primitives for ADVB02.

The module keeps the Phase1 identity anchor intact and exposes only bounded,
source-trained interventions. Slow context, source support, and tangent-basis
buffers are writable only while the module is training and the caller marks
the current samples as source updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _finite_detached_iq(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3 or int(x.size(1)) != 2:
        raise ValueError("NTRS expects IQ tensors shaped [B, 2, T]")
    return torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)


def _safe_moments(value: torch.Tensor, eps: float) -> tuple[torch.Tensor, ...]:
    mean = value.mean(dim=1)
    centered = value - mean[:, None]
    variance = centered.square().mean(dim=1)
    std = variance.clamp_min(eps).sqrt()
    skew = centered.pow(3).mean(dim=1) / std.pow(3).clamp_min(eps)
    kurtosis = centered.pow(4).mean(dim=1) / variance.square().clamp_min(eps)
    return mean, std, skew.clamp(-20.0, 20.0), kurtosis.clamp(0.0, 100.0)


def compute_grouped_physical_descriptors(
    x: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return 40 detached physical descriptors in four ten-value groups."""

    iq = _finite_detached_iq(x)
    i = iq[:, 0, :]
    q = iq[:, 1, :]
    amplitude_squared = i.square() + q.square()
    amplitude = amplitude_squared.add(float(eps)).sqrt()
    rms = amplitude_squared.mean(dim=1).add(float(eps)).sqrt()
    median = amplitude.median(dim=1).values
    mad = (amplitude - median[:, None]).abs().median(dim=1).values
    _amp_mean, _amp_std, amp_skew, amp_kurtosis = _safe_moments(amplitude, float(eps))
    clipping_threshold = median + 3.0 * mad + float(eps)
    clipping_ratio = (amplitude > clipping_threshold[:, None]).float().mean(dim=1)
    low_ratio = (amplitude < (0.10 * rms)[:, None]).float().mean(dim=1)

    if int(iq.size(-1)) > 1:
        delta_i = i[:, 1:] - i[:, :-1]
        delta_q = q[:, 1:] - q[:, :-1]
        difference_noise = 0.5 * (delta_i.square() + delta_q.square()).mean(dim=1)
    else:
        difference_noise = amplitude_squared.new_zeros(amplitude_squared.size(0))
    snr_proxy = torch.log1p(amplitude_squared.mean(dim=1) / difference_noise.clamp_min(float(eps)))

    complex_iq = torch.complex(i, q)
    spectrum = torch.fft.fftshift(torch.fft.fft(complex_iq, dim=-1), dim=-1)
    spectral_power = spectrum.abs().square().clamp_min(float(eps))
    spectral_flatness = torch.exp(spectral_power.log().mean(dim=1)) / spectral_power.mean(dim=1).clamp_min(float(eps))
    amplitude_group = torch.stack(
        [
            rms,
            median,
            mad,
            amp_skew,
            amp_kurtosis,
            clipping_ratio,
            low_ratio,
            snr_proxy,
            spectral_flatness,
            torch.log1p(amplitude_squared.mean(dim=1)),
        ],
        dim=1,
    )

    i_mean, i_std, _i_skew, _i_kurtosis = _safe_moments(i, float(eps))
    q_mean, q_std, _q_skew, _q_kurtosis = _safe_moments(q, float(eps))
    i_centered = i - i_mean[:, None]
    q_centered = q - q_mean[:, None]
    iq_corr = (i_centered * q_centered).mean(dim=1) / (i_std * q_std).clamp_min(float(eps))
    total_power = amplitude_squared.mean(dim=1).clamp_min(float(eps))
    pseudo_real = (i.square() - q.square()).mean(dim=1) / total_power
    pseudo_imag = (2.0 * i * q).mean(dim=1) / total_power
    pseudo_magnitude = (pseudo_real.square() + pseudo_imag.square()).sqrt()
    dc_magnitude = (i_mean.square() + q_mean.square()).sqrt() / rms.clamp_min(float(eps))
    iq_group = torch.stack(
        [
            i_mean,
            q_mean,
            i_std,
            q_std,
            torch.log((i_std + float(eps)) / (q_std + float(eps))),
            iq_corr.clamp(-5.0, 5.0),
            pseudo_real.clamp(-5.0, 5.0),
            pseudo_imag.clamp(-5.0, 5.0),
            pseudo_magnitude.clamp(0.0, 5.0),
            dc_magnitude.clamp(0.0, 10.0),
        ],
        dim=1,
    )

    if int(iq.size(-1)) > 1:
        cross = q[:, 1:] * i[:, :-1] - i[:, 1:] * q[:, :-1]
        dot = i[:, 1:] * i[:, :-1] + q[:, 1:] * q[:, :-1]
        phase_delta = torch.atan2(cross, dot + float(eps))
    else:
        phase_delta = i.new_zeros((i.size(0), 1))
    phase_mean, phase_std, _phase_skew, _phase_kurtosis = _safe_moments(phase_delta, float(eps))
    phase_abs = phase_delta.abs().mean(dim=1)
    midpoint = max(1, int(phase_delta.size(1)) // 2)
    first_mean = phase_delta[:, :midpoint].mean(dim=1)
    second_mean = phase_delta[:, midpoint:].mean(dim=1) if midpoint < int(phase_delta.size(1)) else first_mean
    local_drift = second_mean - first_mean
    if int(phase_delta.size(1)) > 1:
        second_delta = phase_delta[:, 1:] - phase_delta[:, :-1]
    else:
        second_delta = phase_delta.new_zeros((phase_delta.size(0), 1))
    second_mean_value, second_std, _second_skew, _second_kurtosis = _safe_moments(second_delta, float(eps))
    jump_ratio = (phase_delta.abs() > (0.5 * torch.pi)).float().mean(dim=1)
    sin_mean = phase_delta.sin().mean(dim=1)
    cos_mean = phase_delta.cos().mean(dim=1)
    concentration = (sin_mean.square() + cos_mean.square()).sqrt()
    phase_group = torch.stack(
        [
            phase_mean.clamp(-torch.pi, torch.pi),
            phase_std.clamp(0.0, torch.pi),
            phase_abs.clamp(0.0, torch.pi),
            local_drift.clamp(-torch.pi, torch.pi),
            second_mean_value.clamp(-torch.pi, torch.pi),
            second_std.clamp(0.0, torch.pi),
            jump_ratio,
            sin_mean,
            cos_mean,
            concentration,
        ],
        dim=1,
    )

    bins = int(spectral_power.size(1))
    coordinates = torch.linspace(-1.0, 1.0, bins, device=spectral_power.device, dtype=spectral_power.dtype)
    normalized_power = spectral_power / spectral_power.sum(dim=1, keepdim=True).clamp_min(float(eps))
    centroid = (normalized_power * coordinates[None, :]).sum(dim=1)
    spread = (normalized_power * (coordinates[None, :] - centroid[:, None]).square()).sum(dim=1).sqrt()
    spectral_skew = (
        normalized_power * (coordinates[None, :] - centroid[:, None]).pow(3)
    ).sum(dim=1) / spread.pow(3).clamp_min(float(eps))
    slope = (normalized_power * coordinates[None, :]).sum(dim=1) / coordinates.square().mean().clamp_min(float(eps))
    band_ratios = []
    for band in torch.tensor_split(normalized_power, 4, dim=1):
        band_ratios.append(band.sum(dim=1))
    lag_values = []
    max_lag = min(8, max(1, int(complex_iq.size(1)) - 1))
    for lag in range(1, max_lag + 1):
        numerator = (complex_iq[:, lag:] * complex_iq[:, :-lag].conj()).mean(dim=1).abs()
        lag_values.append(numerator / total_power)
    autocorr_side_peak = torch.stack(lag_values, dim=1).max(dim=1).values if lag_values else total_power.new_zeros(total_power.size(0))
    selectivity = spectral_power.std(dim=1, unbiased=False) / spectral_power.mean(dim=1).clamp_min(float(eps))
    frequency_group = torch.stack(
        [
            centroid,
            spread,
            spectral_skew.clamp(-20.0, 20.0),
            slope.clamp(-10.0, 10.0),
            band_ratios[0],
            band_ratios[1],
            band_ratios[2],
            band_ratios[3],
            autocorr_side_peak.clamp(0.0, 10.0),
            selectivity.clamp(0.0, 100.0),
        ],
        dim=1,
    )
    descriptors = torch.cat([amplitude_group, iq_group, phase_group, frequency_group], dim=1)
    return torch.nan_to_num(descriptors, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0).detach()


@dataclass
class NTRSContext:
    q: torch.Tensor
    q_fast: torch.Tensor
    q_slow: torch.Tensor
    q_meta: torch.Tensor
    uncertainty: torch.Tensor
    descriptors: torch.Tensor
    metadata_valid: torch.Tensor


class FastSlowContext(nn.Module):
    """Per-packet fast context plus source-domain EMA slow context."""

    def __init__(
        self,
        *,
        descriptor_dim: int = 40,
        q_dim: int = 32,
        fast_dim: int = 24,
        slow_dim: int = 24,
        metadata_dim: int = 9,
        num_domains: int = 1,
        slow_ema_decay: float = 0.95,
    ):
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.q_dim = int(q_dim)
        self.fast_dim = int(fast_dim)
        self.slow_dim = int(slow_dim)
        self.metadata_dim = int(metadata_dim)
        self.num_domains = int(num_domains)
        self.slow_ema_decay = float(slow_ema_decay)
        if min(self.descriptor_dim, self.q_dim, self.fast_dim, self.slow_dim, self.num_domains) <= 0:
            raise ValueError("NTRS context dimensions and num_domains must be positive")
        if self.metadata_dim < 0:
            raise ValueError("metadata_dim cannot be negative")
        hidden = max(64, 2 * max(self.fast_dim, self.slow_dim, self.q_dim))
        self.fast_encoder = nn.Sequential(
            nn.Linear(self.descriptor_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, self.fast_dim),
        )
        self.slow_observer = nn.Sequential(
            nn.Linear(self.descriptor_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, self.slow_dim),
        )
        self.meta_encoder = nn.Sequential(
            nn.Linear(self.metadata_dim + 1, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, self.slow_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.fast_dim + 2 * self.slow_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, self.q_dim),
        )
        self.uncertainty_head = nn.Linear(self.q_dim + 1, 1)
        self.register_buffer("slow_centers", torch.zeros(self.num_domains, self.slow_dim))
        self.register_buffer("slow_counts", torch.zeros(self.num_domains, dtype=torch.long))

    @torch.no_grad()
    def _update_slow(
        self,
        observation: torch.Tensor,
        domains: torch.Tensor,
        update_mask: Optional[torch.Tensor],
    ) -> None:
        domains = domains.detach().view(-1).long().to(device=observation.device)
        if int(domains.numel()) != int(observation.size(0)):
            raise ValueError("NTRS slow-context domains must align with the batch")
        valid = (domains >= 0) & (domains < self.num_domains)
        if update_mask is not None:
            mask = update_mask.detach().view(-1).bool().to(device=observation.device)
            if int(mask.numel()) != int(observation.size(0)):
                raise ValueError("NTRS slow-context update mask must align with the batch")
            valid = valid & mask
        for domain in torch.unique(domains[valid]).tolist():
            domain_index = int(domain)
            values = observation[valid & (domains == domain_index)].detach().float()
            if values.numel() == 0:
                continue
            mean = values.mean(dim=0)
            if int(self.slow_counts[domain_index].item()) == 0:
                self.slow_centers[domain_index].copy_(mean)
            else:
                decay = self.slow_ema_decay
                self.slow_centers[domain_index].mul_(decay).add_(mean, alpha=1.0 - decay)
            self.slow_counts[domain_index].add_(int(values.size(0)))

    def _read_slow(self, observation: torch.Tensor, domains: Optional[torch.Tensor]) -> torch.Tensor:
        slow = observation
        if domains is None:
            return slow
        domain_values = domains.detach().view(-1).long().to(device=observation.device)
        if int(domain_values.numel()) != int(observation.size(0)):
            raise ValueError("NTRS slow-context domains must align with the batch")
        valid = (domain_values >= 0) & (domain_values < self.num_domains)
        if bool(valid.any()):
            active = valid & (self.slow_counts.to(device=observation.device)[domain_values.clamp(0, self.num_domains - 1)] > 0)
            if bool(active.any()):
                slow = slow.clone()
                slow[active] = self.slow_centers.to(device=observation.device, dtype=observation.dtype)[domain_values[active]]
        return slow

    def forward(
        self,
        x: torch.Tensor,
        *,
        metadata: Optional[torch.Tensor] = None,
        metadata_valid: Optional[torch.Tensor] = None,
        domains: Optional[torch.Tensor] = None,
        update_slow: bool = False,
        update_mask: Optional[torch.Tensor] = None,
    ) -> NTRSContext:
        descriptors = compute_grouped_physical_descriptors(x)
        if int(descriptors.size(1)) != self.descriptor_dim:
            raise ValueError(f"NTRS descriptor_dim={self.descriptor_dim} does not match {descriptors.size(1)}")
        dtype = next(self.parameters()).dtype
        descriptors = descriptors.to(device=x.device, dtype=dtype)
        q_fast = torch.tanh(self.fast_encoder(descriptors))
        slow_observation = torch.tanh(self.slow_observer(descriptors))
        if self.training and bool(update_slow) and domains is not None:
            self._update_slow(slow_observation, domains, update_mask)
        q_slow = self._read_slow(slow_observation, domains)

        batch = int(x.size(0))
        if metadata is None:
            metadata_tensor = descriptors.new_zeros((batch, self.metadata_dim))
            valid = torch.zeros(batch, dtype=torch.bool, device=x.device)
        else:
            metadata_tensor = torch.nan_to_num(
                metadata.detach().to(device=x.device, dtype=dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if metadata_tensor.dim() != 2 or metadata_tensor.shape != (batch, self.metadata_dim):
                raise ValueError("NTRS metadata must be shaped [B, metadata_dim]")
            if metadata_valid is None:
                valid = torch.isfinite(metadata.detach().to(device=x.device)).all(dim=1)
            else:
                valid = metadata_valid.detach().to(device=x.device).view(-1).bool()
                if int(valid.numel()) != batch:
                    raise ValueError("NTRS metadata_valid must align with the batch")
        meta_input = torch.cat([metadata_tensor, valid.to(dtype=dtype).unsqueeze(1)], dim=1)
        q_meta = torch.tanh(self.meta_encoder(meta_input))
        q = torch.tanh(self.fusion(torch.cat([q_fast, q_slow, q_meta], dim=1)))
        missing = (~valid).to(dtype=dtype)
        uncertainty = F.softplus(self.uncertainty_head(torch.cat([q, missing[:, None]], dim=1))).view(-1)
        uncertainty = (uncertainty + 0.25 * missing).clamp(0.0, 10.0)
        return NTRSContext(
            q=q,
            q_fast=q_fast,
            q_slow=q_slow,
            q_meta=q_meta,
            uncertainty=uncertainty,
            descriptors=descriptors.detach(),
            metadata_valid=valid,
        )


@dataclass
class PhysicalCorrection:
    corrected: torch.Tensor
    energy: torch.Tensor
    gate: torch.Tensor
    coefficients: torch.Tensor


class BoundedWidelyLinearCorrector(nn.Module):
    """Short, context-conditioned, near-identity widely linear IQ filter."""

    def __init__(
        self,
        *,
        q_dim: int,
        taps: int = 3,
        center_bound: float = 0.05,
        side_bound: float = 0.03,
        conjugate_bound: float = 0.03,
        phase_offset_bound: float = 0.15,
        phase_slope_bound: float = 0.05,
        phase_curve_bound: float = 0.02,
    ):
        super().__init__()
        self.q_dim = int(q_dim)
        self.taps = int(taps)
        if self.q_dim <= 0 or self.taps <= 0 or self.taps % 2 == 0:
            raise ValueError("NTRS q_dim must be positive and taps must be positive odd")
        self.center_bound = float(center_bound)
        self.side_bound = float(side_bound)
        self.conjugate_bound = float(conjugate_bound)
        self.phase_bounds = (float(phase_offset_bound), float(phase_slope_bound), float(phase_curve_bound))
        self.parameter_head = nn.Linear(self.q_dim, 4 * self.taps + 3)
        self.gate_head = nn.Linear(self.q_dim, 1)
        nn.init.zeros_(self.parameter_head.weight)
        nn.init.zeros_(self.parameter_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    @staticmethod
    def _delay(signal: torch.Tensor, lag: int) -> torch.Tensor:
        if lag <= 0:
            return signal
        zeros = signal.new_zeros((signal.size(0), lag))
        return torch.cat([zeros, signal[:, :-lag]], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        *,
        stage_scale: float | torch.Tensor = 1.0,
    ) -> PhysicalCorrection:
        source_dtype = x.dtype
        iq = _finite_detached_iq(x).to(device=q.device, dtype=q.dtype)
        if q.dim() != 2 or int(q.size(0)) != int(iq.size(0)) or int(q.size(1)) != self.q_dim:
            raise ValueError("NTRS corrector q must be shaped [B, q_dim]")
        params = self.parameter_head(q)
        h_real_raw, h_imag_raw, g_real_raw, g_imag_raw, phase_raw = params.split(
            [self.taps, self.taps, self.taps, self.taps, 3], dim=1
        )
        side_bounds = params.new_full((self.taps,), self.side_bound)
        side_bounds[0] = self.center_bound
        h_real = torch.tanh(h_real_raw) * side_bounds[None, :]
        h_real[:, 0] = h_real[:, 0] + 1.0
        h_imag = torch.tanh(h_imag_raw) * side_bounds[None, :]
        g_real = torch.tanh(g_real_raw) * self.conjugate_bound
        g_imag = torch.tanh(g_imag_raw) * self.conjugate_bound
        h = torch.complex(h_real.float(), h_imag.float())
        g = torch.complex(g_real.float(), g_imag.float())
        signal = torch.complex(iq[:, 0, :].float(), iq[:, 1, :].float())
        candidate = torch.zeros_like(signal)
        for lag in range(self.taps):
            delayed = self._delay(signal, lag)
            candidate = candidate + h[:, lag : lag + 1] * delayed + g[:, lag : lag + 1] * delayed.conj()
        coordinate = torch.linspace(-1.0, 1.0, int(signal.size(1)), device=signal.device, dtype=signal.real.dtype)
        phase = (
            self.phase_bounds[0] * torch.tanh(phase_raw[:, 0:1]).float()
            + self.phase_bounds[1] * torch.tanh(phase_raw[:, 1:2]).float() * coordinate[None, :]
            + self.phase_bounds[2] * torch.tanh(phase_raw[:, 2:3]).float() * coordinate[None, :].square()
        )
        candidate = candidate * torch.exp(torch.complex(torch.zeros_like(phase), -phase))
        scale = torch.as_tensor(stage_scale, device=q.device, dtype=q.dtype).clamp(0.0, 1.0)
        gate = (torch.sigmoid(self.gate_head(q)).view(-1) * scale).clamp(0.0, 1.0)
        corrected_complex = signal + gate[:, None].float() * (candidate - signal)
        corrected = torch.stack([corrected_complex.real, corrected_complex.imag], dim=1).to(dtype=source_dtype)
        reference = iq.to(dtype=corrected.dtype)
        energy = (corrected - reference).float().square().mean(dim=(1, 2)).sqrt()
        return PhysicalCorrection(
            corrected=corrected,
            energy=energy,
            gate=gate,
            coefficients=params,
        )


class NuisanceTangentBasis(nn.Module):
    """EMA covariance and orthonormal basis learned from paired source deltas."""

    def __init__(self, *, embedding_dim: int, rank: int = 8, momentum: float = 0.95):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.rank = int(rank)
        self.momentum = float(momentum)
        if self.embedding_dim <= 0 or self.rank <= 0 or self.rank > self.embedding_dim:
            raise ValueError("NTRS tangent rank must be in [1, embedding_dim]")
        initial = torch.eye(self.embedding_dim, dtype=torch.float32)[:, : self.rank]
        self.register_buffer("covariance", torch.zeros(self.embedding_dim, self.embedding_dim))
        self.register_buffer("basis", initial)
        self.register_buffer("update_count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, clean_z: torch.Tensor, satellite_z: torch.Tensor) -> None:
        if not self.training:
            return
        if clean_z.shape != satellite_z.shape or clean_z.dim() != 2 or int(clean_z.size(1)) != self.embedding_dim:
            raise ValueError("NTRS tangent pairs must share shape [B, embedding_dim]")
        delta = torch.nan_to_num(
            satellite_z.detach().float() - clean_z.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if int(delta.size(0)) == 0:
            return
        delta = delta - delta.mean(dim=0, keepdim=True)
        covariance = delta.transpose(0, 1) @ delta / float(max(1, int(delta.size(0))))
        if int(self.update_count.item()) == 0:
            self.covariance.copy_(covariance)
        else:
            self.covariance.mul_(self.momentum).add_(covariance, alpha=1.0 - self.momentum)
        eigenvalues, eigenvectors = torch.linalg.eigh(self.covariance)
        order = torch.argsort(eigenvalues, descending=True)[: self.rank]
        selected = eigenvectors[:, order]
        orthogonal, _ = torch.linalg.qr(selected, mode="reduced")
        self.basis.copy_(orthogonal[:, : self.rank])
        self.update_count.add_(1)

    def project(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.dim() != 2 or int(coefficients.size(1)) != self.rank:
            raise ValueError("NTRS tangent coefficients must be shaped [B, rank]")
        basis = self.basis.to(device=coefficients.device, dtype=coefficients.dtype)
        return coefficients @ basis.transpose(0, 1)

    def off_subspace_energy(self, delta: torch.Tensor) -> torch.Tensor:
        if delta.dim() != 2 or int(delta.size(1)) != self.embedding_dim:
            raise ValueError("NTRS delta must be shaped [B, embedding_dim]")
        basis = self.basis.to(device=delta.device, dtype=delta.dtype)
        projected = (delta @ basis) @ basis.transpose(0, 1)
        return (delta - projected).square().mean(dim=1).sqrt()


class NTRSSourceSupport(nn.Module):
    """Source-only multi-domain diagonal support model."""

    def __init__(self, q_dim: int, num_domains: int, tau: float = 1.0, momentum: float = 0.95, eps: float = 1e-5):
        super().__init__()
        self.q_dim = int(q_dim)
        self.num_domains = int(num_domains)
        self.tau = float(tau)
        self.momentum = float(momentum)
        self.eps = float(eps)
        if self.q_dim <= 0 or self.num_domains <= 0 or self.tau <= 0.0:
            raise ValueError("NTRS source-support dimensions and tau must be positive")
        self.register_buffer("centers", torch.zeros(self.num_domains, self.q_dim))
        self.register_buffer("scales", torch.ones(self.num_domains, self.q_dim))
        self.register_buffer("counts", torch.zeros(self.num_domains, dtype=torch.long))

    @torch.no_grad()
    def update(self, q: torch.Tensor, domains: torch.Tensor, update_mask: Optional[torch.Tensor]) -> None:
        domains = domains.detach().view(-1).long().to(device=q.device)
        if int(domains.numel()) != int(q.size(0)):
            raise ValueError("NTRS source-support domains must align with q")
        valid = (domains >= 0) & (domains < self.num_domains)
        if update_mask is not None:
            mask = update_mask.detach().view(-1).bool().to(device=q.device)
            if int(mask.numel()) != int(q.size(0)):
                raise ValueError("NTRS source-support update mask must align with q")
            valid = valid & mask
        for domain in torch.unique(domains[valid]).tolist():
            domain_index = int(domain)
            values = q[valid & (domains == domain_index)].detach().float()
            if values.numel() == 0:
                continue
            center = values.mean(dim=0)
            scale = values.std(dim=0, unbiased=False).clamp_min(self.eps)
            if int(self.counts[domain_index].item()) == 0:
                self.centers[domain_index].copy_(center)
                self.scales[domain_index].copy_(scale)
            else:
                self.centers[domain_index].mul_(self.momentum).add_(center, alpha=1.0 - self.momentum)
                self.scales[domain_index].mul_(self.momentum).add_(scale, alpha=1.0 - self.momentum)
            self.counts[domain_index].add_(int(values.size(0)))

    def forward(
        self,
        q: torch.Tensor,
        *,
        update_source: bool = False,
        domains: Optional[torch.Tensor] = None,
        update_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and bool(update_source) and domains is not None:
            self.update(q, domains, update_mask)
        active = self.counts > 0
        if not bool(active.any()):
            return q.new_ones(q.size(0)), q.new_zeros(q.size(0))
        centers = self.centers.to(device=q.device, dtype=q.dtype)
        scales = self.scales.to(device=q.device, dtype=q.dtype).clamp_min(self.eps)
        distance_squared = ((q[:, None, :] - centers[None, :, :]) / scales[None, :, :]).square().mean(dim=2)
        distance_squared = distance_squared.masked_fill(~active.to(device=q.device)[None, :], float("inf"))
        nearest = distance_squared.min(dim=1).values.clamp_min(0.0)
        distance = nearest.sqrt()
        support = torch.exp(-nearest / self.tau).clamp(0.0, 1.0)
        return support, distance


def ntrs_stage_scale(epoch: int) -> float:
    """S1=0, S2-a rises to 0.5, S2-b rises to 1, S3 stays at 1."""

    epoch = int(epoch)
    if epoch <= 16:
        return 0.0
    if epoch <= 40:
        return 0.5 * float(epoch - 16) / 24.0
    if epoch <= 68:
        return 0.5 + 0.5 * float(epoch - 40) / 28.0
    return 1.0


@dataclass
class NTRSOutput:
    z_rob: torch.Tensor
    correction: torch.Tensor
    coefficients: torch.Tensor
    alpha: torch.Tensor
    gate: torch.Tensor
    correctability: torch.Tensor
    correction_energy: torch.Tensor
    support: torch.Tensor
    support_distance: torch.Tensor
    uncertainty: torch.Tensor
    subspace_residual: torch.Tensor


class NTRSRobustifier(nn.Module):
    """Bounded 160-D identity correction constrained to a nuisance basis."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        q_dim: int,
        rank: int = 8,
        alpha_max: float = 0.20,
        support_domains: int = 1,
        support_tau: float = 1.0,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.q_dim = int(q_dim)
        self.rank = int(rank)
        self.alpha_max = float(alpha_max)
        if self.embedding_dim <= 0 or self.q_dim <= 0 or self.rank <= 0 or self.alpha_max < 0.0:
            raise ValueError("NTRS robustifier dimensions must be positive and alpha_max non-negative")
        hidden = max(64, self.embedding_dim)
        self.tangent = NuisanceTangentBasis(embedding_dim=self.embedding_dim, rank=self.rank)
        self.coefficient_head = nn.Sequential(
            nn.Linear(2 * self.embedding_dim + self.q_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, self.rank),
        )
        self.alpha_head = nn.Linear(self.q_dim, 1)
        self.correctability_head = nn.Sequential(
            nn.Linear(self.q_dim + 3, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.support_model = NTRSSourceSupport(self.q_dim, int(support_domains), tau=float(support_tau))
        nn.init.zeros_(self.coefficient_head[-1].weight)
        nn.init.zeros_(self.coefficient_head[-1].bias)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        nn.init.zeros_(self.correctability_head[-1].weight)
        nn.init.zeros_(self.correctability_head[-1].bias)

    def forward(
        self,
        z_anchor: torch.Tensor,
        z_phys: torch.Tensor,
        q: torch.Tensor,
        *,
        uncertainty: torch.Tensor,
        raw_margin: torch.Tensor,
        epoch: int,
        update_source_support: bool = False,
        source_domains: Optional[torch.Tensor] = None,
        source_support_mask: Optional[torch.Tensor] = None,
    ) -> NTRSOutput:
        if z_anchor.shape != z_phys.shape or z_anchor.dim() != 2 or int(z_anchor.size(1)) != self.embedding_dim:
            raise ValueError("NTRS anchor and physical embeddings must share [B, embedding_dim]")
        if q.shape != (z_anchor.size(0), self.q_dim):
            raise ValueError("NTRS q must align with the embedding batch")
        uncertainty = uncertainty.view(-1).to(device=z_anchor.device, dtype=z_anchor.dtype)
        raw_margin = raw_margin.view(-1).to(device=z_anchor.device, dtype=z_anchor.dtype)
        if int(uncertainty.numel()) != int(z_anchor.size(0)) or int(raw_margin.numel()) != int(z_anchor.size(0)):
            raise ValueError("NTRS uncertainty and raw_margin must align with the batch")
        support, support_distance = self.support_model(
            q.detach(),
            update_source=bool(update_source_support),
            domains=source_domains,
            update_mask=source_support_mask,
        )
        coefficient_input = torch.cat([z_anchor, z_phys - z_anchor, q], dim=1)
        coefficients = torch.tanh(self.coefficient_head(coefficient_input))
        tangent_delta = self.tangent.project(coefficients)
        alpha = (self.alpha_max * torch.sigmoid(self.alpha_head(q))).view(-1)
        bounded_delta = alpha[:, None] * tangent_delta
        pre_energy = bounded_delta.square().mean(dim=1).sqrt()
        correctability_input = torch.cat(
            [q, uncertainty[:, None], pre_energy[:, None], raw_margin[:, None]],
            dim=1,
        )
        correctability = torch.sigmoid(self.correctability_head(correctability_input)).view(-1)
        scale = z_anchor.new_tensor(ntrs_stage_scale(int(epoch)))
        gate = (
            scale
            * correctability
            * torch.exp(-uncertainty.clamp_min(0.0))
            * support.to(device=z_anchor.device, dtype=z_anchor.dtype)
        ).clamp(0.0, 1.0)
        correction = gate[:, None] * bounded_delta
        z_corrected = z_anchor - correction
        z_rob = F.normalize(F.layer_norm(z_corrected, (self.embedding_dim,)), dim=1, eps=1e-6)
        correction_energy = correction.square().mean(dim=1).sqrt()
        subspace_residual = self.tangent.off_subspace_energy(correction)
        return NTRSOutput(
            z_rob=z_rob,
            correction=correction,
            coefficients=coefficients,
            alpha=alpha,
            gate=gate,
            correctability=correctability,
            correction_energy=correction_energy,
            support=support,
            support_distance=support_distance,
            uncertainty=uncertainty,
            subspace_residual=subspace_residual,
        )


__all__ = [
    "BoundedWidelyLinearCorrector",
    "FastSlowContext",
    "NTRSContext",
    "NTRSOutput",
    "NTRSRobustifier",
    "NTRSSourceSupport",
    "NuisanceTangentBasis",
    "PhysicalCorrection",
    "compute_grouped_physical_descriptors",
    "ntrs_stage_scale",
]
