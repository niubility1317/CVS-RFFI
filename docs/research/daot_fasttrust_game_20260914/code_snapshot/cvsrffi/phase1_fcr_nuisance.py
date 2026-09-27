from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .phase1_fcr_types import FCRConfig


@dataclass
class NuisanceOutput:
    z_ch: torch.Tensor
    z_rx: torch.Tensor
    z_sync: torch.Tensor
    z_gain: torch.Tensor
    eta_pred: torch.Tensor


class StructuredNuisanceEncoder(nn.Module):
    """Compress received IQ statistics into bounded physical nuisance parts."""

    def __init__(self, config: FCRConfig) -> None:
        super().__init__()
        self.config = config
        self.statistics_encoder = nn.Sequential(
            nn.Linear(11, 24),
            nn.GELU(),
            nn.Linear(24, 24),
            nn.GELU(),
        )
        self.channel_head = nn.Linear(24, config.channel_dim)
        self.receiver_head = nn.Linear(24, config.receiver_dim)
        self.sync_head = nn.Linear(24, config.sync_dim)
        self.gain_head = nn.Linear(24, config.gain_dim)
        self.eta_head = nn.Linear(24, 3)

    def _statistics(self, x: torch.Tensor, eta_hat: torch.Tensor) -> torch.Tensor:
        real = x[:, 0].float()
        imag = x[:, 1].float()
        power = real.square() + imag.square()
        adjacent_real = (real[:, 1:] * real[:, :-1] + imag[:, 1:] * imag[:, :-1]).mean(dim=1)
        adjacent_imag = (imag[:, 1:] * real[:, :-1] - real[:, 1:] * imag[:, :-1]).mean(dim=1)
        return torch.stack(
            (
                real.mean(dim=1),
                imag.mean(dim=1),
                real.square().mean(dim=1).sqrt(),
                imag.square().mean(dim=1).sqrt(),
                power.mean(dim=1).sqrt(),
                power.std(dim=1, unbiased=False),
                adjacent_real,
                adjacent_imag,
                eta_hat[:, 0].float(),
                eta_hat[:, 1].float(),
                eta_hat[:, 2].float(),
            ),
            dim=1,
        )

    def forward(self, x: torch.Tensor, eta_hat: torch.Tensor) -> NuisanceOutput:
        expected_iq = (x.size(0), 2, self.config.input_len)
        if x.ndim != 3 or x.shape != expected_iq or not torch.is_floating_point(x):
            raise ValueError("x must be a floating-point tensor shaped [B,2,input_len]")
        if eta_hat.shape != (x.size(0), 3):
            raise ValueError("eta_hat must have shape [B,3]")

        encoded = self.statistics_encoder(self._statistics(x, eta_hat))
        raw_sync = self.sync_head(encoded)
        z_sync = torch.stack(
            (
                math.pi * torch.tanh(raw_sync[:, 0]),
                0.25 * torch.tanh(raw_sync[:, 1]),
                0.01 * torch.tanh(raw_sync[:, 2]),
                8.0 * torch.tanh(raw_sync[:, 3]),
                0.02 * torch.tanh(raw_sync[:, 4]),
                0.10 * torch.tanh(raw_sync[:, 5]),
            ),
            dim=1,
        )
        return NuisanceOutput(
            z_ch=0.5 * torch.tanh(self.channel_head(encoded)),
            z_rx=0.25 * torch.tanh(self.receiver_head(encoded)),
            z_sync=z_sync,
            z_gain=2.0 * torch.tanh(self.gain_head(encoded)),
            eta_pred=eta_hat.float() + 0.10 * torch.tanh(self.eta_head(encoded)),
        )
