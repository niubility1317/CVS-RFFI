from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import torch
import torch.nn as nn


def _as_complex(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.is_complex():
        z = x
    elif x.dim() >= 2 and int(x.size(-2)) == 2:
        z = torch.complex(x[..., 0, :].float(), x[..., 1, :].float())
    else:
        z = torch.complex(x.float(), torch.zeros_like(x.float()))
    if z.dim() == 1:
        z = z.unsqueeze(0)
    if z.dim() != 2:
        raise ValueError("IQ input must be complex [B,T] or real [B,2,T]")
    return torch.complex(
        torch.nan_to_num(z.real, nan=0.0, posinf=0.0, neginf=0.0),
        torch.nan_to_num(z.imag, nan=0.0, posinf=0.0, neginf=0.0),
    )


def _unwrap_phase(phase: torch.Tensor) -> torch.Tensor:
    if phase.numel() <= 1:
        return phase
    difference = phase[1:] - phase[:-1]
    wrapped = torch.remainder(difference + math.pi, 2.0 * math.pi) - math.pi
    wrapped = torch.where(
        (wrapped == -math.pi) & (difference > 0),
        torch.full_like(wrapped, math.pi),
        wrapped,
    )
    return torch.cat([phase[:1], phase[:1] + wrapped.cumsum(dim=0)])


@dataclass
class NuisanceEstimate:
    log_gain: torch.Tensor
    phase0: torch.Tensor
    normalized_cfo: torch.Tensor
    time_shift: torch.Tensor
    confidence: torch.Tensor
    observed_valid_mask: torch.Tensor


@dataclass
class ContentExcitationOutput:
    s_hat: torch.Tensor
    content_confidence: torch.Tensor
    reconstruction_nmse: torch.Tensor
    condition: torch.Tensor
    uncertainty: torch.Tensor


@dataclass
class CanonicalExcitationOutput(ContentExcitationOutput):
    canonical_iq: torch.Tensor
    valid_mask: torch.Tensor
    nuisance: NuisanceEstimate


class NuisanceEstimator(nn.Module):
    """Conservative analytic gain/phase/CFO/coarse-shift estimator."""

    def __init__(self, max_time_shift: int = 8, eps: float = 1e-6):
        super().__init__()
        self.max_time_shift = int(max_time_shift)
        self.eps = float(eps)

    @staticmethod
    def _aligned_pair(
        received: torch.Tensor, reference: torch.Tensor, shift: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = int(received.numel())
        if shift >= 0:
            observed_index = torch.arange(
                shift, length, device=received.device, dtype=torch.long
            )
            return received[shift:], reference[: length - shift], observed_index
        observed_index = torch.arange(
            0, length + shift, device=received.device, dtype=torch.long
        )
        return received[: length + shift], reference[-shift:], observed_index

    def _reference_aided(
        self,
        received: torch.Tensor,
        reference: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        best = None
        length = int(received.numel())
        max_shift = min(self.max_time_shift, max(0, length - 2))
        for shift in range(-max_shift, max_shift + 1):
            r_aligned, s_aligned, observed_index = self._aligned_pair(
                received, reference, shift
            )
            mask = valid_mask[observed_index] & (s_aligned.abs() > self.eps)
            if int(mask.sum()) < 2:
                score = received.real.new_tensor(0.0)
            else:
                cross = r_aligned[mask] * s_aligned[mask].conj()
                coherence = cross / cross.abs().clamp_min(self.eps)
                score = coherence.mean().abs()
            if best is None or float(score.detach()) > float(best[0].detach()):
                best = (score, shift, r_aligned, s_aligned, observed_index, mask)
        assert best is not None
        score, shift, r_aligned, s_aligned, observed_index, mask = best
        r_valid = r_aligned[mask]
        s_valid = s_aligned[mask]
        t_valid = observed_index[mask].to(dtype=received.real.dtype)
        if int(mask.sum()) < 2 or float(
            s_valid.abs().square().sum().detach()
        ) <= self.eps:
            zero = received.real.new_tensor(0.0)
            return zero, zero, zero, torch.tensor(shift, device=received.device), zero
        gain = (
            r_valid.abs().square().sum()
            / s_valid.abs().square().sum().clamp_min(self.eps)
        ).sqrt().clamp_min(self.eps)
        phase = _unwrap_phase(torch.angle(r_valid * s_valid.conj()))
        t_centered = t_valid - t_valid.mean()
        omega = (
            t_centered * (phase - phase.mean())
        ).sum() / t_centered.square().sum().clamp_min(self.eps)
        phase0 = phase.mean() - omega * t_valid.mean()
        return gain.log(), phase0, omega, torch.tensor(shift, device=received.device), score

    def _blind(
        self, received: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        values = received[valid_mask]
        zero = received.real.new_tensor(0.0)
        if values.numel() < 2:
            return zero, zero, zero, torch.tensor(0, device=received.device), zero
        gain = values.abs().square().mean().sqrt()
        if float(gain.detach()) <= self.eps:
            return gain.clamp_min(self.eps).log(), zero, zero, torch.tensor(0, device=received.device), zero
        lag_product = values[1:] * values[:-1].conj()
        coherence = lag_product.mean()
        confidence = coherence.abs() / lag_product.abs().mean().clamp_min(self.eps)
        omega = torch.angle(coherence)
        phase0 = torch.angle((values / values.abs().clamp_min(self.eps)).mean())
        return gain.log(), phase0, omega, torch.tensor(0, device=received.device), confidence

    def forward(
        self,
        received_iq: torch.Tensor,
        reference_iq: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> NuisanceEstimate:
        received = _as_complex(received_iq)
        reference = _as_complex(reference_iq) if reference_iq is not None else None
        if reference is not None and reference.shape != received.shape:
            raise ValueError("reference_iq must match received_iq")
        if valid_mask is None:
            valid_mask = torch.ones_like(received.real, dtype=torch.bool)
        elif valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0).expand(received.size(0), -1)
        valid_mask = valid_mask.to(device=received.device, dtype=torch.bool)
        outputs = []
        for batch_index in range(received.size(0)):
            if reference is None:
                outputs.append(self._blind(received[batch_index], valid_mask[batch_index]))
            else:
                outputs.append(
                    self._reference_aided(
                        received[batch_index], reference[batch_index], valid_mask[batch_index]
                    )
                )
        return NuisanceEstimate(
            log_gain=torch.stack([item[0] for item in outputs]),
            phase0=torch.stack([item[1] for item in outputs]),
            normalized_cfo=torch.stack([item[2] for item in outputs]),
            time_shift=torch.stack([item[3] for item in outputs]).long(),
            confidence=torch.stack([item[4] for item in outputs]).clamp(0.0, 1.0),
            observed_valid_mask=valid_mask,
        )


class AnalyticCanonicalizer(nn.Module):
    """Invert only the estimated scalar gain, common phase, CFO and shift."""

    def forward(
        self, received_iq: torch.Tensor, estimate: NuisanceEstimate
    ) -> tuple[torch.Tensor, torch.Tensor]:
        received = _as_complex(received_iq)
        batch, length = received.shape
        time = torch.arange(length, device=received.device, dtype=received.real.dtype)
        phase = estimate.phase0.unsqueeze(-1) + estimate.normalized_cfo.unsqueeze(-1) * time
        corrected = received * torch.exp(-1j * phase) / estimate.log_gain.exp().unsqueeze(-1).clamp_min(1e-6)
        canonical = torch.zeros_like(corrected)
        valid = torch.zeros_like(received.real)
        for batch_index in range(batch):
            shift = int(estimate.time_shift[batch_index].item())
            observed_mask = estimate.observed_valid_mask[batch_index]
            if shift >= 0:
                count = length - shift
                canonical[batch_index, :count] = corrected[batch_index, shift:]
                valid[batch_index, :count] = observed_mask[shift:].to(valid.dtype)
            else:
                count = length + shift
                canonical[batch_index, -shift:] = corrected[batch_index, :count]
                valid[batch_index, -shift:] = observed_mask[:count].to(valid.dtype)
        canonical_iq = torch.stack([canonical.real, canonical.imag], dim=1)
        return canonical_iq, valid


class ContentExcitationEstimator(nn.Module):
    """Low-capacity analytic reconstruction of the excitation presented to the gate."""

    def __init__(self, detach_gate_input: bool = True, eps: float = 1e-6):
        super().__init__()
        self.detach_gate_input = bool(detach_gate_input)
        self.eps = float(eps)

    def forward(
        self, canonical_iq: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> ContentExcitationOutput:
        canonical = _as_complex(canonical_iq)
        if valid_mask is None:
            valid_mask = torch.ones_like(canonical.real)
        elif valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0).expand(canonical.size(0), -1)
        valid_mask = valid_mask.to(device=canonical.device, dtype=canonical.real.dtype).clamp(0.0, 1.0)
        reconstructed = []
        confidences = []
        nmse_values = []
        conditions = []
        uncertainties = []
        for batch_index in range(canonical.size(0)):
            z = canonical[batch_index]
            mask = valid_mask[batch_index] > 0.5
            amplitude = z.abs()
            valid_amplitude = amplitude[mask]
            if valid_amplitude.numel() == 0 or float(
                valid_amplitude.square().mean().detach()
            ) <= self.eps:
                s_hat = torch.zeros_like(z)
                confidence = torch.zeros_like(amplitude)
                condition = amplitude.new_tensor(1.0 / self.eps)
                uncertainty = amplitude.new_tensor(1.0)
            else:
                median = valid_amplitude.median().clamp_min(self.eps)
                low = torch.quantile(valid_amplitude, 0.10)
                high = torch.quantile(valid_amplitude, 0.90).clamp_min(low + self.eps)
                clipped_amplitude = amplitude.clamp(min=low, max=high)
                unit_phase = z / amplitude.clamp_min(self.eps)
                s_hat = unit_phase * clipped_amplitude
                log_deviation = (amplitude.clamp_min(self.eps) / median).log().abs()
                confidence = torch.exp(-log_deviation) * valid_mask[batch_index]
                condition = high / low.clamp_min(self.eps)
                uncertainty = 1.0 - confidence.sum() / valid_mask[batch_index].sum().clamp_min(1.0)
            denominator = (z.abs().square() * valid_mask[batch_index]).sum().clamp_min(self.eps)
            nmse = ((z - s_hat).abs().square() * valid_mask[batch_index]).sum() / denominator
            reconstructed.append(s_hat)
            confidences.append(confidence)
            nmse_values.append(nmse)
            conditions.append(condition)
            uncertainties.append(uncertainty.clamp(0.0, 1.0))
        s_hat = torch.stack(reconstructed)
        content_confidence = torch.stack(confidences)
        if self.detach_gate_input:
            s_hat = s_hat.detach()
            content_confidence = content_confidence.detach()
        return ContentExcitationOutput(
            s_hat=s_hat,
            content_confidence=content_confidence,
            reconstruction_nmse=torch.stack(nmse_values),
            condition=torch.stack(conditions),
            uncertainty=torch.stack(uncertainties),
        )


class CanonicalExcitationEstimator(nn.Module):
    def __init__(
        self, max_time_shift: int = 8, detach_gate_input: bool = True
    ):
        super().__init__()
        self.nuisance_estimator = NuisanceEstimator(max_time_shift=max_time_shift)
        self.canonicalizer = AnalyticCanonicalizer()
        self.content_estimator = ContentExcitationEstimator(
            detach_gate_input=detach_gate_input
        )

    def forward(
        self,
        received_iq: torch.Tensor,
        reference_iq: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> CanonicalExcitationOutput:
        nuisance = self.nuisance_estimator(
            received_iq, reference_iq=reference_iq, valid_mask=valid_mask
        )
        canonical_iq, canonical_valid = self.canonicalizer(received_iq, nuisance)
        content = self.content_estimator(canonical_iq, valid_mask=canonical_valid)
        return CanonicalExcitationOutput(
            s_hat=content.s_hat,
            content_confidence=content.content_confidence,
            reconstruction_nmse=content.reconstruction_nmse,
            condition=content.condition,
            uncertainty=content.uncertainty,
            canonical_iq=canonical_iq,
            valid_mask=canonical_valid,
            nuisance=nuisance,
        )


def unique_physical_sample_mask(physical_sample_ids: Sequence[object]) -> torch.Tensor:
    seen = set()
    keep = []
    for sample_id in physical_sample_ids:
        key = str(sample_id)
        keep.append(key not in seen)
        seen.add(key)
    return torch.tensor(keep, dtype=torch.bool)
