from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .phase1_fcr_types import FCRConfig


@dataclass
class CanonicalOutput:
    canonical_iq: torch.Tensor
    eta_hat: torch.Tensor
    residual_iq: torch.Tensor
    quality: dict[str, torch.Tensor]


class ConservativeCanonicalizer(nn.Module):
    """Remove only common scalar gain, phase and CFO from complex IQ.

    This intentionally has no trainable waveform path.  Its three estimates are
    bounded before applying the analytic inverse, so fine non-common transmitter
    residuals remain in ``residual_iq`` for the fingerprint branch.
    """

    _MAX_LOG_GAIN = math.log(4.0)
    _MAX_OMEGA = 0.20

    def __init__(self, config: FCRConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, iq: torch.Tensor) -> CanonicalOutput:
        if iq.ndim != 3 or iq.size(1) != 2 or iq.size(2) != self.config.input_len:
            raise ValueError(
                "iq must have shape [B,2,{}]".format(self.config.input_len)
            )
        if not torch.is_floating_point(iq):
            raise TypeError("iq must be a floating-point tensor")

        complex_iq = torch.complex(iq[:, 0].float(), iq[:, 1].float())
        magnitude_sq = complex_iq.abs().square()
        log_gain = (0.5 * magnitude_sq.mean(dim=-1).clamp_min(1.0e-8).log()).clamp(
            -self._MAX_LOG_GAIN, self._MAX_LOG_GAIN
        )
        gain = log_gain.exp()

        adjacent = (complex_iq[:, 1:] * complex_iq[:, :-1].conj()).mean(dim=-1)
        raw_omega = torch.atan2(adjacent.imag, adjacent.real)
        omega = self._MAX_OMEGA * torch.tanh(raw_omega / self._MAX_OMEGA)

        sample_index = torch.arange(
            self.config.input_len, device=iq.device, dtype=complex_iq.real.dtype
        )
        cfo_removed = complex_iq * torch.exp(-1j * omega[:, None] * sample_index)
        common = cfo_removed.mean(dim=-1)
        raw_phase0 = torch.atan2(common.imag, common.real)
        phase0 = math.pi * torch.tanh(raw_phase0 / math.pi)

        phase = phase0[:, None] + omega[:, None] * sample_index[None, :]
        canonical = complex_iq * torch.exp(-1j * phase) / gain[:, None].clamp_min(1.0e-4)
        residual = complex_iq - canonical
        coherence = adjacent.abs() / magnitude_sq[:, :-1].mean(dim=-1).clamp_min(1.0e-8)
        quality = {
            "signal_rms": magnitude_sq.mean(dim=-1).sqrt(),
            "adjacent_coherence": coherence.clamp(0.0, 1.0),
            "residual_rms": residual.abs().square().mean(dim=-1).sqrt(),
        }
        eta_hat = torch.stack((log_gain, phase0, omega), dim=-1)
        return CanonicalOutput(
            canonical_iq=torch.stack((canonical.real, canonical.imag), dim=1),
            eta_hat=eta_hat,
            residual_iq=torch.stack((residual.real, residual.imag), dim=1),
            quality=quality,
        )
