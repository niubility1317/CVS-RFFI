from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class AmplitudeNormalizer(nn.Module):
    """Per-sample RMS normalization for two-channel IQ tensors."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_iq(x)
        rms = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True) + self.eps)
        return x / rms

    @staticmethod
    def _check_iq(x: torch.Tensor) -> None:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError("SGC-Adapter expects IQ tensors shaped [B, 2, L]")


class FrequencyOffsetCompensator(nn.Module):
    """Lightweight normalized CFO/Doppler estimator and compensator."""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 32,
        max_norm_freq_offset: float = 0.05,
    ):
        super().__init__()
        self.max_norm_freq_offset = float(max_norm_freq_offset)
        self.estimator = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        AmplitudeNormalizer._check_iq(x)
        bsz, _, length = x.shape
        delta_f = self.estimator(x).mean(dim=-1) * self.max_norm_freq_offset
        t = torch.arange(length, device=x.device, dtype=x.dtype) / max(1, length)
        phase = -2.0 * math.pi * delta_f * t.unsqueeze(0)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        i_part = x[:, 0, :]
        q_part = x[:, 1, :]
        i_comp = i_part * cos_phase - q_part * sin_phase
        q_comp = i_part * sin_phase + q_part * cos_phase
        return torch.stack([i_comp, q_comp], dim=1).reshape(bsz, 2, length)


class SpectralInterferenceSuppressor(nn.Module):
    """Soft FFT-domain mask with residual blending."""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 32,
        residual_alpha: float = 0.5,
    ):
        super().__init__()
        if int(in_channels) != 2:
            raise ValueError("SpectralInterferenceSuppressor currently expects two IQ channels")
        self.residual_alpha = float(residual_alpha)
        self.mask_net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        AmplitudeNormalizer._check_iq(x)
        z = torch.complex(x[:, 0, :], x[:, 1, :])
        spectrum = torch.fft.fft(z, dim=-1)
        mag = torch.abs(spectrum)
        mask = self.mask_net(mag.unsqueeze(1)).squeeze(1)
        filtered = torch.fft.ifft(spectrum * mask, dim=-1)
        x_filtered = torch.stack([filtered.real, filtered.imag], dim=1).to(dtype=x.dtype)
        return x + self.residual_alpha * (x_filtered - x)


class ResidualChannelCompensator(nn.Module):
    """Depthwise-separable residual channel compensator."""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 32,
        num_blocks: int = 2,
        kernel_size: int = 5,
        init_gamma: float = 0.0,
    ):
        super().__init__()
        layers = []
        ch = int(in_channels)
        blocks = max(1, int(num_blocks))
        for idx in range(blocks):
            out_ch = int(hidden_channels) if idx < blocks - 1 else int(in_channels)
            layers.append(self._make_block(ch, out_ch, int(kernel_size)))
            ch = out_ch
        self.net = nn.Sequential(*layers)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    @staticmethod
    def _make_block(in_ch: int, out_ch: int, kernel_size: int) -> nn.Sequential:
        padding = int(kernel_size) // 2
        return nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=False),
            nn.BatchNorm1d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        AmplitudeNormalizer._check_iq(x)
        return x + self.gamma * self.net(x)


class SGCAdapter(nn.Module):
    """Satellite-ground-channel-aware adapter for IQ inputs."""

    def __init__(
        self,
        in_channels: int = 2,
        use_amp_norm: bool = True,
        use_freq_comp: bool = True,
        use_spectral_suppressor: bool = True,
        use_residual_comp: bool = True,
        freq_hidden_dim: int = 32,
        max_norm_freq_offset: float = 0.05,
        spectral_hidden_dim: int = 32,
        spectral_residual_alpha: float = 0.5,
        residual_channels: int = 32,
        residual_blocks: int = 2,
        residual_kernel_size: int = 5,
        residual_init_gamma: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.use_amp_norm = bool(use_amp_norm)
        self.use_freq_comp = bool(use_freq_comp)
        self.use_spectral_suppressor = bool(use_spectral_suppressor)
        self.use_residual_comp = bool(use_residual_comp)

        self.amp_norm = AmplitudeNormalizer(eps=eps) if self.use_amp_norm else nn.Identity()
        self.freq_comp = (
            FrequencyOffsetCompensator(in_channels, freq_hidden_dim, max_norm_freq_offset)
            if self.use_freq_comp
            else nn.Identity()
        )
        self.spectral_sup = (
            SpectralInterferenceSuppressor(in_channels, spectral_hidden_dim, spectral_residual_alpha)
            if self.use_spectral_suppressor
            else nn.Identity()
        )
        self.residual_comp = (
            ResidualChannelCompensator(
                in_channels,
                residual_channels,
                residual_blocks,
                residual_kernel_size,
                residual_init_gamma,
            )
            if self.use_residual_comp
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        AmplitudeNormalizer._check_iq(x)
        x_in = x
        x = self.amp_norm(x)
        x = self.freq_comp(x)
        x = self.spectral_sup(x)
        x = self.residual_comp(x)

        if not return_aux:
            return x, None

        aux: Dict[str, torch.Tensor] = {
            "adapter_input": x_in,
            "adapter_output": x,
            "adapter_delta_rms": torch.sqrt(torch.mean((x - x_in).square()) + 1e-12),
        }
        if self.use_residual_comp and hasattr(self.residual_comp, "gamma"):
            aux["residual_gamma"] = self.residual_comp.gamma.abs()
        return x, aux

    def get_channel_feature(self, x: torch.Tensor) -> torch.Tensor:
        AmplitudeNormalizer._check_iq(x)
        return x.mean(dim=-1)

    @property
    def submodule_status(self) -> Dict[str, bool]:
        return {
            "amp_norm": self.use_amp_norm,
            "freq_comp": self.use_freq_comp,
            "spectral_suppressor": self.use_spectral_suppressor,
            "residual_comp": self.use_residual_comp,
        }
