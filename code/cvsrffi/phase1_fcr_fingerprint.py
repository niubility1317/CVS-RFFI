from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from .phase1_fcr_factors import excitation_features
from .phase1_fcr_types import FCRConfig


def fixed_response_basis(s: torch.Tensor) -> torch.Tensor:
    """Fixed, non-learned PA, conjugate-IQ, and one-step memory basis."""

    if not torch.is_complex(s) or s.ndim != 2:
        raise ValueError("s must be complex [B,input_len]")
    delayed = torch.roll(s, shifts=1, dims=-1)
    return torch.stack((s, s.conj(), s * s.abs().square(), delayed * delayed.abs().square()), dim=-1)


@dataclass
class FingerprintFactorOutput:
    z_f_id: torch.Tensor
    z_tx_state: torch.Tensor


@dataclass
class FingerprintResponseOutput:
    delta_f: torch.Tensor
    response_coef: torch.Tensor
    response_quality: dict[str, torch.Tensor]


class FingerprintFactorEncoder(nn.Module):
    """Keep the ADV3B02 identity feature while deriving a separate TX state."""

    def __init__(self, config: FCRConfig) -> None:
        super().__init__()
        self.config = config
        summary_dim = 8
        self.identity_context = nn.Linear(summary_dim, 160, bias=False)
        # A legacy ADV3B02/ADV3B03 warm start must preserve its identity
        # geometry at optimizer step zero. The context branch learns only the
        # residual correction introduced by FCR.
        nn.init.zeros_(self.identity_context.weight)
        self.state_head = nn.Sequential(
            nn.Linear(summary_dim, 32),
            nn.GELU(),
            nn.Linear(32, config.tx_state_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        id_feature_raw: torch.Tensor,
        canonical_iq: torch.Tensor,
        residual_iq: torch.Tensor,
        excitation: torch.Tensor,
    ) -> FingerprintFactorOutput:
        batch_size = id_feature_raw.size(0)
        if id_feature_raw.shape != (batch_size, 160):
            raise ValueError("id_feature_raw must have shape [B,160]")
        expected_iq = (batch_size, 2, self.config.input_len)
        if canonical_iq.shape != expected_iq or residual_iq.shape != expected_iq:
            raise ValueError("canonical_iq and residual_iq must be [B,2,input_len]")
        if excitation.shape != (batch_size, self.config.input_len, 4):
            raise ValueError("excitation must be [B,input_len,4]")

        canonical = canonical_iq.float()
        residual = residual_iq.float()
        summary = torch.cat(
            (
                canonical.mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                canonical.square().mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                residual.mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                residual.square().mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                excitation.float().mean(dim=1),
            ),
            dim=1,
        )
        z_f_id = functional.normalize(id_feature_raw.float() + self.identity_context(summary), dim=1, eps=1e-8)
        return FingerprintFactorOutput(z_f_id=z_f_id, z_tx_state=self.state_head(summary))


class ExcitationConditionedFingerprintOperator(nn.Module):
    """Fixed physical response plus a short, low-rank excitation-only residual."""

    def __init__(
        self,
        config: FCRConfig,
        *,
        residual_ratio_max: float = 0.10,
        residual_rank: int = 4,
    ) -> None:
        super().__init__()
        if not 0.0 < residual_ratio_max <= 1.0:
            raise ValueError("residual_ratio_max must be in (0,1]")
        self.config = config
        self.residual_ratio_max = float(residual_ratio_max)
        self.residual_rank = int(residual_rank)
        self.response_head = nn.Linear(config.tx_state_dim, 8)
        self.residual_local = nn.Conv1d(4, residual_rank, kernel_size=3, padding=1, bias=False)
        self.residual_state = nn.Linear(config.tx_state_dim, residual_rank, bias=False)

    def bounded_residual(self, excitation: torch.Tensor, z_tx_state: torch.Tensor) -> torch.Tensor:
        """Return a real short-receptive-field amplitude residual from allowed inputs only."""

        if excitation.ndim != 3 or excitation.size(-1) != 4:
            raise ValueError("excitation must be [B,input_len,4]")
        if z_tx_state.shape != (excitation.size(0), self.config.tx_state_dim):
            raise ValueError("z_tx_state has an unexpected shape")
        local = self.residual_local(excitation.float().transpose(1, 2))
        state = self.residual_state(z_tx_state.float()).unsqueeze(-1)
        return torch.tanh((local * state).sum(dim=1))

    @staticmethod
    def _phase_reference(s: torch.Tensor) -> torch.Tensor:
        index = s.abs().argmax(dim=1, keepdim=True)
        reference = s.gather(1, index)
        return reference / reference.abs().clamp_min(1e-8)

    def _response_coefficients(self, z_tx_state: torch.Tensor) -> torch.Tensor:
        raw = 0.05 * torch.tanh(self.response_head(z_tx_state.float()))
        return torch.complex(raw[:, :4], raw[:, 4:])

    @staticmethod
    def _bandlimit(delta: torch.Tensor) -> torch.Tensor:
        kernel = delta.real.new_tensor([0.25, 0.5, 0.25]).view(1, 1, 3)
        real = functional.conv1d(delta.real.unsqueeze(1), kernel, padding=1).squeeze(1)
        imag = functional.conv1d(delta.imag.unsqueeze(1), kernel, padding=1).squeeze(1)
        return torch.complex(real, imag)

    def limit_energy_and_bandwidth(self, delta: torch.Tensor, s_hat: torch.Tensor) -> torch.Tensor:
        filtered = self._bandlimit(delta)
        signal_energy = s_hat.norm(dim=1, keepdim=True)
        delta_energy = filtered.norm(dim=1, keepdim=True).clamp_min(1e-8)
        scale = (self.residual_ratio_max * signal_energy / delta_energy).clamp(max=1.0)
        return filtered * scale

    def forward(self, s_hat: torch.Tensor, factor: FingerprintFactorOutput) -> FingerprintResponseOutput:
        if not torch.is_complex(s_hat) or s_hat.shape != (factor.z_f_id.size(0), self.config.input_len):
            raise ValueError("s_hat must be complex [B,input_len]")
        with torch.autocast(device_type=s_hat.device.type, enabled=False):
            s_hat_fp32 = s_hat.to(torch.complex64)
            z_tx_state_fp32 = factor.z_tx_state.float()
            excitation = excitation_features(s_hat_fp32)
            response_coef = self._response_coefficients(z_tx_state_fp32)
            basis = fixed_response_basis(s_hat_fp32)
            phase_reference = self._phase_reference(s_hat_fp32)
            phase_equivariant_basis = basis.clone()
            phase_equivariant_basis[:, :, 1] = (
                phase_equivariant_basis[:, :, 1] * phase_reference.square()
            )
            delta_physical = torch.einsum("btn,bn->bt", phase_equivariant_basis, response_coef)
            residual_amplitude = self.bounded_residual(excitation, z_tx_state_fp32)
            phase_carrier = s_hat_fp32 / s_hat_fp32.abs().clamp_min(1e-8)
            delta_small = (
                torch.complex(residual_amplitude, torch.zeros_like(residual_amplitude))
                * phase_carrier
            )
            delta_f = self.limit_energy_and_bandwidth(
                delta_physical + delta_small, s_hat_fp32
            ).to(torch.complex64)
            energy_ratio = delta_f.norm(dim=1) / s_hat_fp32.norm(dim=1).clamp_min(1e-8)
            response_quality = {
                "energy_ratio": energy_ratio,
                "state_norm": z_tx_state_fp32.norm(dim=1),
            }
            return FingerprintResponseOutput(
                delta_f=delta_f,
                response_coef=response_coef,
                response_quality=response_quality,
            )
