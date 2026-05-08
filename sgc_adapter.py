from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def _as_int_list(value: Optional[Iterable[int]], fallback: Sequence[int]) -> List[int]:
    if value is None:
        return [int(v) for v in fallback]
    if isinstance(value, str):
        items = [v.strip() for v in value.replace(";", ",").replace("+", ",").split(",")]
        out = [int(v) for v in items if v]
    else:
        out = [int(v) for v in value]
    return out or [int(v) for v in fallback]


class ResidualChannelCompensator(nn.Module):
    """Depthwise-separable residual channel compensator."""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 32,
        num_blocks: int = 2,
        kernel_size: int = 5,
        init_gamma: float = 0.0,
        max_gamma: Optional[float] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_gamma = None if max_gamma is None else float(max_gamma)
        layers = []
        ch = int(in_channels)
        blocks = max(1, int(num_blocks))
        for idx in range(blocks):
            out_ch = int(hidden_channels) if idx < blocks - 1 else int(in_channels)
            layers.append(self._make_block(ch, out_ch, int(kernel_size), float(dropout)))
            ch = out_ch
        self.net = nn.Sequential(*layers)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    @staticmethod
    def _make_block(in_ch: int, out_ch: int, kernel_size: int, dropout: float = 0.0) -> nn.Sequential:
        padding = int(kernel_size) // 2
        layers = [
            nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=False),
            nn.BatchNorm1d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(inplace=True),
        ]
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(p=min(0.5, max(0.0, float(dropout)))))
        return nn.Sequential(*layers)

    def effective_gamma(self) -> torch.Tensor:
        if self.max_gamma is None or self.max_gamma <= 0.0:
            return self.gamma
        return torch.tanh(self.gamma) * self.max_gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        AmplitudeNormalizer._check_iq(x)
        return x + self.effective_gamma() * self.net(x)


class MultiScaleResidualChannelCompensator(nn.Module):
    """Conservative residual channel compensator with local and wider kernels.

    This block keeps the residual path explicit: the input is always added back,
    while the learned branch is gated by a small scalar gamma and optionally by
    per-sample channel statistics. The intent is to correct satellite-channel
    nuisance distortions without replacing transmitter fingerprints.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 32,
        kernel_sizes: Optional[Iterable[int]] = None,
        dilations: Optional[Iterable[int]] = None,
        init_gamma: float = 0.0,
        max_gamma: float = 0.20,
        dropout: float = 0.05,
        stat_gate: bool = False,
    ):
        super().__init__()
        if int(in_channels) != 2:
            raise ValueError("MultiScaleResidualChannelCompensator expects two IQ channels")
        self.max_gamma = float(max_gamma)
        self.stat_gate = bool(stat_gate)
        hidden = int(hidden_channels)
        kernels = _as_int_list(kernel_sizes, (3, 5, 9))
        dilations_list = _as_int_list(dilations, (1, 2, 4))
        if len(dilations_list) < len(kernels):
            dilations_list = (dilations_list * len(kernels))[: len(kernels)]

        self.pre = nn.Sequential(
            nn.Conv1d(int(in_channels), hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.SiLU(inplace=True),
        )
        branches = []
        for kernel, dilation in zip(kernels, dilations_list):
            k = max(1, int(kernel))
            if k % 2 == 0:
                k += 1
            d = max(1, int(dilation))
            pad = d * (k // 2)
            branches.append(
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, kernel_size=k, padding=pad, dilation=d, groups=hidden, bias=False),
                    nn.BatchNorm1d(hidden),
                    nn.SiLU(inplace=True),
                    nn.Conv1d(hidden, hidden, kernel_size=1, bias=False),
                    nn.BatchNorm1d(hidden),
                    nn.SiLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden * len(self.branches), hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p=min(0.5, max(0.0, float(dropout)))) if float(dropout) > 0.0 else nn.Identity(),
            nn.Conv1d(hidden, int(in_channels), kernel_size=1, bias=True),
        )
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))
        self.gate_mlp = (
            nn.Sequential(
                nn.Linear(6, hidden),
                nn.SiLU(inplace=True),
                nn.Linear(hidden, 1),
                nn.Sigmoid(),
            )
            if self.stat_gate
            else None
        )
        self.last_gate_mean: Optional[torch.Tensor] = None
        self.last_delta_rms: Optional[torch.Tensor] = None

    def _summary_stats(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[:, 0, :], x[:, 1, :])
        amp = torch.abs(z)
        phase_step = torch.angle(z[:, 1:] * torch.conj(z[:, :-1])) if z.size(1) > 1 else amp[:, :1] * 0.0
        spectrum = torch.fft.fft(z, dim=-1)
        mag = torch.abs(spectrum)
        mag_prob = mag / mag.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        spec_entropy = -(mag_prob * torch.log(mag_prob.clamp_min(1e-6))).sum(dim=-1, keepdim=True)
        return torch.cat(
            [
                amp.mean(dim=-1, keepdim=True),
                amp.std(dim=-1, keepdim=True, unbiased=False),
                x[:, 0, :].mean(dim=-1, keepdim=True),
                x[:, 1, :].mean(dim=-1, keepdim=True),
                phase_step.std(dim=-1, keepdim=True, unbiased=False),
                spec_entropy,
            ],
            dim=1,
        )

    def effective_gamma(self) -> torch.Tensor:
        if self.max_gamma <= 0.0:
            return self.gamma
        return torch.tanh(self.gamma) * self.max_gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        AmplitudeNormalizer._check_iq(x)
        h = self.pre(x)
        residual = self.fuse(torch.cat([branch(h) for branch in self.branches], dim=1))
        scale = self.effective_gamma()
        if self.gate_mlp is not None:
            gate = self.gate_mlp(self._summary_stats(x)).unsqueeze(-1)
            self.last_gate_mean = gate.detach().mean()
            scale = scale * (0.5 + gate)
        else:
            self.last_gate_mean = None
        delta = scale * residual
        self.last_delta_rms = torch.sqrt(torch.mean(delta.square()) + 1e-12).detach()
        return x + delta


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
        residual_kernel_sizes: Optional[Iterable[int]] = None,
        residual_dilations: Optional[Iterable[int]] = None,
        residual_init_gamma: float = 0.0,
        residual_max_gamma: Optional[float] = None,
        residual_dropout: float = 0.0,
        residual_mode: str = "plain",
        residual_stat_gate: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.use_amp_norm = bool(use_amp_norm)
        self.use_freq_comp = bool(use_freq_comp)
        self.use_spectral_suppressor = bool(use_spectral_suppressor)
        self.use_residual_comp = bool(use_residual_comp)
        self.residual_mode = str(residual_mode or "plain").lower().strip()

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
        if self.use_residual_comp:
            if self.residual_mode in ("multiscale", "ms", "gated_multiscale", "msg"):
                self.residual_comp = MultiScaleResidualChannelCompensator(
                    in_channels=in_channels,
                    hidden_channels=residual_channels,
                    kernel_sizes=residual_kernel_sizes,
                    dilations=residual_dilations,
                    init_gamma=residual_init_gamma,
                    max_gamma=0.20 if residual_max_gamma is None else float(residual_max_gamma),
                    dropout=residual_dropout,
                    stat_gate=bool(residual_stat_gate or self.residual_mode in ("gated_multiscale", "msg")),
                )
            elif self.residual_mode in ("plain", "std", "depthwise"):
                self.residual_comp = ResidualChannelCompensator(
                    in_channels,
                    residual_channels,
                    residual_blocks,
                    residual_kernel_size,
                    residual_init_gamma,
                    max_gamma=residual_max_gamma,
                    dropout=residual_dropout,
                )
            else:
                raise ValueError(f"Unknown SGC residual_mode={residual_mode}")
        else:
            self.residual_comp = nn.Identity()

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
        x_before_residual = x
        x = self.residual_comp(x)

        if not return_aux:
            return x, None

        aux: Dict[str, torch.Tensor] = {
            "adapter_input": x_in,
            "adapter_output": x,
            "adapter_delta_rms": torch.sqrt(torch.mean((x - x_in).square()) + 1e-12),
            "adapter_input_rms": torch.sqrt(torch.mean(x_in.square()) + 1e-12),
            "residual_delta_rms": torch.sqrt(torch.mean((x - x_before_residual).square()) + 1e-12),
        }
        if self.use_residual_comp and hasattr(self.residual_comp, "gamma"):
            aux["residual_gamma"] = self.residual_comp.gamma.abs()
        if self.use_residual_comp and hasattr(self.residual_comp, "effective_gamma"):
            aux["residual_effective_gamma"] = self.residual_comp.effective_gamma().abs()
        if self.use_residual_comp and getattr(self.residual_comp, "last_gate_mean", None) is not None:
            aux["residual_gate_mean"] = self.residual_comp.last_gate_mean
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
            "residual_mode": self.residual_mode if self.use_residual_comp else "none",
        }
