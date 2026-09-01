from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .identifiability_stats import (
    complex_excitation_stats,
    effective_fisher_summary,
    hos_confidence_stats,
    memory_polynomial_gram_stats,
    phase_residual_stats,
    spectral_occupancy_stats,
)


BRANCH_NAMES = ("raw", "hom", "phase", "pa", "hos")


@dataclass
class BranchOutput:
    embedding: torch.Tensor
    local_mask: torch.Tensor
    direction_gate: torch.Tensor
    identifiability: torch.Tensor
    stability: torch.Tensor
    uncertainty: torch.Tensor
    evidence: torch.Tensor


def _finite_unit(value: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    return torch.nan_to_num(
        value.float(), nan=default, posinf=default, neginf=default
    ).clamp(0.0, 1.0)


def _safe_feature(value: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    value = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tanh(value / float(scale))


class FisherBranchBank(nn.Module):
    """Build five physical-feature branches and their three-level gate evidence.

    All Fisher and confidence statistics are functions of the fixed canonical
    reception and reconstructed excitation. Class labels and query-level state
    are deliberately absent from this interface.
    """

    def __init__(
        self,
        embedding_dim: int,
        pa_orders: Sequence[int] = (1, 3, 5),
        pa_memory_depth: int = 1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.pa_orders = tuple(int(value) for value in pa_orders)
        self.pa_memory_depth = int(pa_memory_depth)
        self.eps = float(eps)
        self.phase_projection = nn.Sequential(
            nn.Linear(5, self.embedding_dim), nn.LayerNorm(self.embedding_dim)
        )
        self.hos_projection = nn.Sequential(
            nn.Linear(5, self.embedding_dim), nn.LayerNorm(self.embedding_dim)
        )

    def _validate_embedding(self, name: str, value: torch.Tensor, batch: int) -> None:
        if tuple(value.shape) != (batch, self.embedding_dim):
            raise ValueError(
                f"{name}_embedding must have shape [B,{self.embedding_dim}]"
            )

    def forward(
        self,
        canonical_iq: torch.Tensor,
        s_hat: torch.Tensor,
        *,
        raw_embedding: torch.Tensor,
        hom_embedding: torch.Tensor,
        pa_embedding: torch.Tensor,
        content_confidence: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, BranchOutput]:
        if canonical_iq.dim() != 3 or canonical_iq.size(1) != 2:
            raise ValueError("canonical_iq must have shape [B,2,T]")
        batch, _, length = canonical_iq.shape
        for name, embedding in (
            ("raw", raw_embedding),
            ("hom", hom_embedding),
            ("pa", pa_embedding),
        ):
            self._validate_embedding(name, embedding, batch)

        device = canonical_iq.device
        dtype = canonical_iq.dtype
        if content_confidence is None:
            content_confidence = torch.ones((batch, length), device=device, dtype=dtype)
        if content_confidence.dim() == 1:
            content_confidence = content_confidence.unsqueeze(0)
        if valid_mask is None:
            valid_mask = torch.ones((batch, length), device=device, dtype=dtype)
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0)
        confidence = _finite_unit(content_confidence) * _finite_unit(valid_mask)
        confidence_mean = confidence.mean(dim=-1)

        excitation = complex_excitation_stats(s_hat, eps=self.eps)
        spectrum = spectral_occupancy_stats(s_hat, confidence, eps=self.eps)
        pa = memory_polynomial_gram_stats(
            s_hat,
            order=self.pa_orders,
            memory_depth=self.pa_memory_depth,
            weight=confidence,
            eps=self.eps,
        )
        phase = phase_residual_stats(canonical_iq, confidence, eps=self.eps)
        hos = hos_confidence_stats(canonical_iq, eps=self.eps)

        time_axis = torch.linspace(
            -1.0, 1.0, length, device=s_hat.device, dtype=s_hat.real.dtype
        ).unsqueeze(0)
        iq_target = torch.stack([s_hat.conj(), 1j * s_hat.conj()], dim=-1)
        nuisance_jacobian = torch.stack(
            [s_hat, 1j * s_hat, time_axis * s_hat], dim=-1
        )
        iq_effective = effective_fisher_summary(
            iq_target, nuisance_jacobian, confidence, eps=self.eps
        )

        nonlinear_orders = tuple(order for order in self.pa_orders if order > 1)
        if nonlinear_orders:
            pa_target = torch.stack(
                [s_hat * s_hat.abs().pow(order - 1) for order in nonlinear_orders],
                dim=-1,
            )
        else:
            pa_target = (
                s_hat * (s_hat.abs() - s_hat.abs().mean(dim=-1, keepdim=True))
            ).unsqueeze(-1)
        pa_effective = effective_fisher_summary(
            pa_target, nuisance_jacobian, confidence, eps=self.eps
        )

        iq_eigen = iq_effective["eigenvalues"]
        iq_direction = iq_eigen / iq_eigen.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        raw_i = _finite_unit(
            iq_effective["lambda_min"]
            / (iq_effective["trace"] / float(iq_eigen.size(-1))).clamp_min(self.eps)
            * confidence_mean
        )
        raw_s = _finite_unit(confidence_mean)
        raw_u = _finite_unit(1.0 - raw_i * raw_s, default=1.0)

        hom_direction = torch.stack(
            [
                spectrum["effective_bandwidth"],
                spectrum["occupied_fraction"],
                spectrum["spectral_entropy"],
                1.0 - spectrum["edge_energy"],
            ],
            dim=-1,
        )
        hom_direction = _finite_unit(hom_direction)
        hom_i = _finite_unit(
            0.5 * spectrum["effective_bandwidth"]
            + 0.5 * spectrum["spectral_entropy"]
        )
        hom_s = _finite_unit(confidence_mean * (1.0 - spectrum["edge_energy"]))
        hom_u = _finite_unit(1.0 - hom_i * hom_s, default=1.0)
        excitation_spectrum = torch.fft.fftshift(torch.fft.fft(s_hat, dim=-1), dim=-1)
        spectral_power = excitation_spectrum.abs().square()
        hom_local_mask = spectral_power / spectral_power.amax(
            dim=-1, keepdim=True
        ).clamp_min(self.eps)
        hom_local_mask = _finite_unit(hom_local_mask)

        phase_s = _finite_unit(phase["stability"] * (1.0 - phase["cycle_slip_rate"]))
        phase_i = _finite_unit(phase_s * confidence_mean)
        phase_u = _finite_unit(
            1.0 - phase_s * (1.0 - phase["cycle_slip_rate"]), default=1.0
        )
        phase_features = torch.stack(
            [
                _safe_feature(phase["residual_rms"], 3.14159265),
                phase_s,
                _finite_unit(phase["cycle_slip_rate"]),
                _safe_feature(phase["phase_snr"], 20.0),
                confidence_mean,
            ],
            dim=-1,
        )
        phase_embedding = self.phase_projection(phase_features)
        canonical_complex = torch.complex(canonical_iq[:, 0], canonical_iq[:, 1])
        if length > 2:
            phase_step = torch.angle(
                canonical_complex[:, 1:] * canonical_complex[:, :-1].conj()
            )
            phase_curvature = torch.remainder(
                phase_step[:, 1:] - phase_step[:, :-1] + torch.pi,
                2.0 * torch.pi,
            ) - torch.pi
            phase_local_mask = F.pad(
                torch.exp(-phase_curvature.abs()), (2, 0), value=1.0
            )
        else:
            phase_local_mask = confidence.new_ones((batch, length))
        phase_local_mask = _finite_unit(phase_local_mask * confidence)

        pa_eigen = pa_effective["eigenvalues"]
        pa_direction = pa_eigen / pa_eigen.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        rank_fraction = pa_effective["effective_rank"] / float(pa_eigen.size(-1))
        weakest_direction = pa_effective["lambda_min"] / (
            pa_effective["trace"] / float(pa_eigen.size(-1))
        ).clamp_min(self.eps)
        pa_i = _finite_unit(
            rank_fraction
            * weakest_direction
            * pa["amplitude_entropy"]
            * torch.tanh(pa["amplitude_dynamic_range"])
        )
        pa_s = _finite_unit(confidence_mean * (1.0 - pa["clipping_rate"]))
        pa_u = _finite_unit(1.0 - pa_i * pa_s, default=1.0)
        amplitude = canonical_complex.abs()
        amplitude_mask = amplitude / amplitude.amax(dim=-1, keepdim=True).clamp_min(self.eps)
        amplitude_mask = _finite_unit(amplitude_mask * confidence)

        hos_s = _finite_unit(hos["confidence"])
        hos_strength = torch.tanh(torch.log1p(hos["c40_abs"] + hos["c42_abs"]))
        hos_i = _finite_unit(hos_s * hos_strength * confidence_mean)
        hos_u = _finite_unit(1.0 - hos_s, default=1.0)
        hos_features = torch.stack(
            [
                _safe_feature(torch.log1p(hos["c40_abs"]), 2.0),
                _safe_feature(torch.log1p(hos["c42_abs"]), 2.0),
                _safe_feature(hos["segment_variance"], 1.0),
                hos_s,
                confidence_mean,
            ],
            dim=-1,
        )
        hos_embedding = self.hos_projection(hos_features)
        amplitude_mean = amplitude.mean(dim=-1, keepdim=True).clamp_min(self.eps)
        hos_local_mask = torch.exp(-(amplitude / amplitude_mean - 1.0).abs())
        hos_local_mask = _finite_unit(hos_local_mask * confidence)

        outputs = OrderedDict()
        branch_values = (
            (
                "raw",
                raw_embedding,
                confidence,
                iq_direction,
                raw_i,
                raw_s,
                raw_u,
            ),
            (
                "hom",
                hom_embedding,
                hom_local_mask,
                hom_direction,
                hom_i,
                hom_s,
                hom_u,
            ),
            (
                "phase",
                phase_embedding,
                phase_local_mask,
                torch.stack([phase_i, phase_s, 1.0 - phase_u], dim=-1),
                phase_i,
                phase_s,
                phase_u,
            ),
            (
                "pa",
                pa_embedding,
                amplitude_mask,
                pa_direction,
                pa_i,
                pa_s,
                pa_u,
            ),
            (
                "hos",
                hos_embedding,
                hos_local_mask,
                torch.stack([hos_i, hos_s], dim=-1),
                hos_i,
                hos_s,
                hos_u,
            ),
        )
        for name, embedding, local_mask, direction, ident, stable, uncertainty in branch_values:
            ident = _finite_unit(ident)
            stable = _finite_unit(stable)
            uncertainty = _finite_unit(uncertainty, default=1.0)
            local_mask = _finite_unit(local_mask)
            direction = _finite_unit(direction)
            direction_profile = F.interpolate(
                direction.unsqueeze(1),
                size=self.embedding_dim,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            gated_embedding = (
                torch.nan_to_num(embedding)
                * direction_profile
                * local_mask.mean(dim=-1, keepdim=True)
            )
            outputs[name] = BranchOutput(
                embedding=gated_embedding,
                local_mask=local_mask,
                direction_gate=direction,
                identifiability=ident,
                stability=stable,
                uncertainty=uncertainty,
                evidence=torch.stack([ident, stable, uncertainty], dim=-1),
            )
        return outputs
