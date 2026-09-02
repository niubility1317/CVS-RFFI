from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn

from .phase1_fcr_nuisance import NuisanceOutput
from .phase1_fcr_types import FCRConfig, FCRDecodeOutput


class PhysicsOrderedDecoder(nn.Module):
    """Decode only through content, fingerprint, and structured nuisance factors."""

    def __init__(self, config: FCRConfig) -> None:
        super().__init__()
        if not 0.0 < config.variance_floor <= config.variance_ceiling:
            raise ValueError("variance bounds must satisfy 0 < floor <= ceiling")
        mode = str(config.decoder_mode).strip().lower()
        if mode not in {"control", "full_physics"}:
            raise ValueError("decoder_mode must be 'control' or 'full_physics'")
        self.config = config
        self.mode = mode
        nuisance_dim = config.channel_dim + config.receiver_dim + config.sync_dim + config.gain_dim
        self.variance_head = nn.Linear(nuisance_dim, 1)
        self.call_trace: tuple[str, ...] = ()

    def _check_nuisance(self, nuisance: NuisanceOutput, batch_size: int) -> None:
        expected = (
            (nuisance.z_ch, self.config.channel_dim, "z_ch"),
            (nuisance.z_rx, self.config.receiver_dim, "z_rx"),
            (nuisance.z_sync, self.config.sync_dim, "z_sync"),
            (nuisance.z_gain, self.config.gain_dim, "z_gain"),
        )
        for value, width, name in expected:
            if value.shape != (batch_size, width):
                raise ValueError(f"{name} must have shape [B,{width}]")

    @staticmethod
    def _complex_taps(z_ch: torch.Tensor) -> torch.Tensor:
        grouped = z_ch.float().reshape(z_ch.size(0), 4, 4)
        real = 0.15 * torch.tanh(grouped[:, :, :2].mean(dim=-1))
        imag = 0.15 * torch.tanh(grouped[:, :, 2:].mean(dim=-1))
        taps = torch.complex(real, imag)
        direct = torch.complex(torch.ones_like(real[:, :1]), torch.zeros_like(imag[:, :1]))
        return torch.cat((direct + taps[:, :1], taps[:, 1:]), dim=1)

    def apply_short_channel(self, u_hat: torch.Tensor, z_ch: torch.Tensor) -> torch.Tensor:
        taps = self._complex_taps(z_ch).to(u_hat.dtype)
        padded = functional.pad(u_hat, (taps.size(1) - 1, 0))
        windows = padded.unfold(-1, taps.size(1), 1)
        return (windows * taps[:, None, :]).sum(dim=-1)

    @staticmethod
    def apply_rx_residual(linked: torch.Tensor, z_rx: torch.Tensor) -> torch.Tensor:
        grouped = z_rx.float().reshape(z_rx.size(0), 2, 4)
        direct = 0.15 * torch.tanh(grouped[:, 0].mean(dim=-1))
        conjugate = 0.05 * torch.tanh(grouped[:, 1].mean(dim=-1))
        return linked * (1.0 + direct[:, None]) + linked.conj() * conjugate[:, None]

    def apply_sync_and_gain(
        self, linked: torch.Tensor, z_sync: torch.Tensor, z_gain: torch.Tensor
    ) -> torch.Tensor:
        sample_count = self.config.input_len
        index = torch.arange(sample_count, device=linked.device, dtype=linked.real.dtype)
        normalized = index / max(sample_count - 1, 1)
        phase0 = math.pi * torch.tanh(z_sync[:, 0].float() / math.pi)
        cfo = 0.25 * torch.tanh(z_sync[:, 1].float() / 0.25)
        doppler_rate = 0.01 * torch.tanh(z_sync[:, 2].float() / 0.01)
        sto = 8.0 * torch.tanh(z_sync[:, 3].float() / 8.0)
        sfo = 0.02 * torch.tanh(z_sync[:, 4].float() / 0.02)
        sync_residual = 0.10 * torch.tanh(z_sync[:, 5].float() / 0.10)

        fractional_delay = 0.5 * torch.tanh(sto / 8.0)
        delayed = linked + fractional_delay[:, None] * (
            torch.roll(linked, shifts=1, dims=-1) - torch.roll(linked, shifts=-1, dims=-1)
        )
        phase = (
            phase0[:, None]
            + cfo[:, None] * index[None, :]
            + doppler_rate[:, None] * normalized.square()[None, :]
            + sfo[:, None] * normalized[None, :] * index[None, :]
            + sync_residual[:, None] * torch.sin(2.0 * math.pi * normalized)[None, :]
        )
        synchronized = delayed * torch.exp(1j * phase)
        agc = torch.exp(0.5 * torch.tanh(z_gain[:, 0].float()))
        amplitude_offset = 0.10 * torch.tanh(z_gain[:, 1].float())
        iq_imbalance = 0.10 * torch.tanh(z_gain[:, 2].float())
        return synchronized * (agc * (1.0 + amplitude_offset))[:, None] + synchronized.conj() * iq_imbalance[:, None]

    def bounded_variance_head(self, nuisance: NuisanceOutput) -> torch.Tensor:
        features = torch.cat(
            (nuisance.z_ch, nuisance.z_rx, nuisance.z_sync, nuisance.z_gain), dim=1
        ).float()
        unit_interval = torch.sigmoid(self.variance_head(features))
        variance = self.config.variance_floor + (
            self.config.variance_ceiling - self.config.variance_floor
        ) * unit_interval
        return variance.clamp(self.config.variance_floor, self.config.variance_ceiling).log()

    def forward(self, s_hat: torch.Tensor, delta_f: torch.Tensor, nuisance: NuisanceOutput) -> FCRDecodeOutput:
        expected_shape = (s_hat.size(0), self.config.input_len)
        if not torch.is_complex(s_hat) or s_hat.shape != expected_shape:
            raise ValueError("s_hat must be complex [B,input_len]")
        if not torch.is_complex(delta_f) or delta_f.shape != expected_shape:
            raise ValueError("delta_f must be complex [B,input_len]")
        self._check_nuisance(nuisance, s_hat.size(0))

        with torch.autocast(device_type=s_hat.device.type, enabled=False):
            s_hat_fp32 = s_hat.to(torch.complex64)
            delta_f_fp32 = delta_f.to(torch.complex64)
            nuisance_fp32 = NuisanceOutput(
                z_ch=nuisance.z_ch.float(),
                z_rx=nuisance.z_rx.float(),
                z_sync=nuisance.z_sync.float(),
                z_gain=nuisance.z_gain.float(),
                eta_pred=nuisance.eta_pred.float(),
            )
            u_hat = s_hat_fp32 + delta_f_fp32
            if self.mode == "control":
                mu = u_hat
                self.call_trace = ("content", "fingerprint", "control")
            else:
                linked = self.apply_short_channel(u_hat, nuisance_fp32.z_ch)
                linked = self.apply_rx_residual(linked, nuisance_fp32.z_rx)
                mu = self.apply_sync_and_gain(
                    linked, nuisance_fp32.z_sync, nuisance_fp32.z_gain
                )
                self.call_trace = ("content", "fingerprint", "channel_receiver")
            log_variance = self.bounded_variance_head(nuisance_fp32).expand(
                -1, self.config.input_len
            )
            return FCRDecodeOutput(
                mu_iq=torch.stack((mu.real, mu.imag), dim=1),
                log_variance=log_variance,
                delta_f=delta_f_fp32,
                decoder_mode=self.mode,
            )
