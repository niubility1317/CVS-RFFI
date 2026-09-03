from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as functional
from torch import nn

from .phase1_fcr_factors import ContentGenerator, ContentSequenceEncoder, excitation_features
from .phase1_fcr_types import FCRConfig, FCRV2FactorOutput
from .phase1_fcr_v2_physics import complex_gram


def _as_complex_iq(iq: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(iq):
        if iq.ndim != 2:
            raise ValueError("complex IQ must have shape [B,T]")
        return iq.to(torch.complex64)
    if iq.ndim != 3 or iq.size(1) != 2:
        raise ValueError("real IQ must have shape [B,2,T]")
    return torch.complex(iq[:, 0].float(), iq[:, 1].float())


def orthogonal_response_basis(s: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    signal = _as_complex_iq(s)
    delayed = torch.roll(signal, shifts=1, dims=-1)
    raw_basis = torch.stack(
        (
            signal,
            signal.conj(),
            signal * signal.abs().square(),
            delayed,
        ),
        dim=-1,
    )
    orthogonal: list[torch.Tensor] = []
    for basis_index in range(raw_basis.size(-1)):
        vector = raw_basis[..., basis_index]
        for previous in orthogonal:
            projection = (vector * previous.conj()).sum(dim=-1, keepdim=True)
            denom = previous.abs().square().sum(dim=-1, keepdim=True).clamp_min(eps)
            vector = vector - projection / denom * previous
        norm = vector.abs().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        orthogonal.append(vector / norm)
    return torch.stack(orthogonal, dim=-1)


def _limit_delta_energy(delta: torch.Tensor, reference: torch.Tensor, *, ratio_max: float) -> torch.Tensor:
    signal_energy = reference.norm(dim=-1, keepdim=True)
    delta_energy = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (float(ratio_max) * signal_energy / delta_energy).clamp(max=1.0)
    return delta * scale


class FCRV2FactorEncoder(nn.Module):
    def __init__(
        self,
        config: FCRConfig,
        *,
        content_dim_limit: int = 16,
        residual_ratio_max: float = 0.10,
        residual_rank: int = 4,
        multipath_taps: int = 3,
    ) -> None:
        super().__init__()
        self.config = config
        self.content_dim_limit = int(min(content_dim_limit, config.content_dim))
        self.residual_ratio_max = float(residual_ratio_max)
        self.multipath_taps = int(max(1, multipath_taps))
        reduced = replace(config, content_dim=self.content_dim_limit)
        self.sequence_encoder = ContentSequenceEncoder(reduced)
        self.generator = ContentGenerator(reduced)

        self.delta_z_head = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
            nn.Linear(64, 160),
        )
        self.response_head = nn.Linear(160, 8)
        self.residual_local = nn.Conv1d(4, residual_rank, kernel_size=3, padding=1, bias=False)
        self.residual_state = nn.Linear(160, residual_rank, bias=False)
        self.nuisance_head = nn.Sequential(
            nn.Linear(8, 32),
            nn.GELU(),
            nn.Linear(32, 7 + 2 * (self.multipath_taps - 1)),
        )

    def _summary(
        self,
        canonical_iq: torch.Tensor,
        residual_iq: torch.Tensor,
        excitation: torch.Tensor,
    ) -> torch.Tensor:
        canonical = canonical_iq.float()
        residual = residual_iq.float()
        return torch.cat(
            (
                canonical.mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                canonical.square().mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                residual.mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                residual.square().mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                excitation.float().mean(dim=1),
            ),
            dim=1,
        )

    def _response_delta(self, s_hat: torch.Tensor, z_f_dev: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        basis = orthogonal_response_basis(s_hat)
        response_raw = 0.05 * torch.tanh(self.response_head(z_f_dev.float()))
        response_coef = torch.complex(response_raw[:, :4], response_raw[:, 4:])
        delta_physical = torch.einsum("btn,bn->bt", basis, response_coef)
        excitation = excitation_features(s_hat)
        local = self.residual_local(excitation.float().transpose(1, 2))
        state = self.residual_state(z_f_dev.float()).unsqueeze(-1)
        amplitude_residual = torch.tanh((local * state).sum(dim=1))
        carrier = s_hat / s_hat.abs().clamp_min(1e-8)
        carrier = torch.where(
            s_hat.abs() > 0,
            carrier,
            torch.ones_like(carrier),
        )
        delta_small = torch.complex(amplitude_residual, torch.zeros_like(amplitude_residual)) * carrier
        delta_f = _limit_delta_energy(
            delta_physical + delta_small,
            s_hat,
            ratio_max=self.residual_ratio_max,
        ).to(torch.complex64)
        gram = complex_gram(basis)
        quality = {
            "energy_ratio": delta_f.norm(dim=-1) / s_hat.norm(dim=-1).clamp_min(1e-8),
            "gram_trace": gram.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1),
        }
        return delta_f, quality

    def _nuisance(self, summary: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.nuisance_head(summary.float())
        alpha = torch.complex(
            1.0 + 0.05 * torch.tanh(raw[:, 0]),
            0.05 * torch.tanh(raw[:, 1]),
        )
        beta = torch.complex(
            0.05 * torch.tanh(raw[:, 2]),
            0.05 * torch.tanh(raw[:, 3]),
        )
        sto = 2.0 * torch.tanh(raw[:, 4])
        sfo = 0.01 * torch.tanh(raw[:, 5])
        phase = 0.10 * torch.tanh(raw[:, 6])
        taps = [torch.ones(raw.size(0), 1, dtype=torch.complex64, device=raw.device)]
        tap_offset = 7
        for tap_index in range(self.multipath_taps - 1):
            real = 0.15 * torch.tanh(raw[:, tap_offset + 2 * tap_index])
            imag = 0.15 * torch.tanh(raw[:, tap_offset + 2 * tap_index + 1])
            taps.append(torch.complex(real, imag).unsqueeze(1))
        return {
            "alpha": alpha.to(torch.complex64),
            "beta": beta.to(torch.complex64),
            "sto": sto.float(),
            "sfo": sfo.float(),
            "phase": phase.float(),
            "taps": torch.cat(taps, dim=1),
        }

    def forward(
        self,
        canonical_iq: torch.Tensor,
        residual_iq: torch.Tensor,
        z_adv: torch.Tensor,
    ) -> FCRV2FactorOutput:
        batch_size = canonical_iq.size(0)
        expected = (batch_size, 2, self.config.input_len)
        if canonical_iq.shape != expected or residual_iq.shape != expected:
            raise ValueError("canonical_iq and residual_iq must be [B,2,input_len]")
        if z_adv.shape != (batch_size, 160):
            raise ValueError("z_adv must be [B,160]")

        z_s = self.sequence_encoder(canonical_iq.float())
        s_hat = self.generator(z_s)
        summary_excitation = excitation_features(s_hat)
        summary = self._summary(canonical_iq, residual_iq, summary_excitation)
        delta_z_f = 0.05 * torch.tanh(self.delta_z_head(summary.float()))
        z_f_dev = z_adv.float() + delta_z_f
        z_f_id = functional.normalize(z_f_dev, dim=1, eps=1e-8)
        canonical_residual = _as_complex_iq(canonical_iq) - s_hat
        delta_f, response_quality = self._response_delta(s_hat, z_f_dev)
        return FCRV2FactorOutput(
            z_s=z_s,
            z_f_id=z_f_id,
            z_f_dev=z_f_dev,
            z_n=self._nuisance(summary),
            s_hat=s_hat,
            delta_f=delta_f,
            canonical_residual=canonical_residual,
            response_quality=response_quality,
        )
