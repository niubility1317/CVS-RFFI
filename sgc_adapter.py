from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _iq_to_complex(x: torch.Tensor) -> torch.Tensor:
    AmplitudeNormalizer._check_iq(x)
    real = x[:, 0, :]
    imag = x[:, 1, :]
    if real.dtype not in (torch.float32, torch.float64):
        real = real.float()
        imag = imag.float()
    return torch.complex(real, imag)


def _complex_to_iq(z: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.stack([z.real, z.imag], dim=1).to(dtype=dtype)


def _safe_rms(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.mean(x.square()) + float(eps))


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


class FingerprintPreservingChannelProjector(nn.Module):
    """Conservative frequency-domain channel projection for FPCR-SGC.

    The projector estimates only a low-quefrency log-spectral envelope,
    which corresponds to a smooth multiplicative channel term. The correction
    is shrinkage-limited so high-order transmitter details such as spectral
    regrowth and IQ image leakage are not aggressively equalized away.
    """

    def __init__(
        self,
        shrinkage: float = 0.35,
        cepstral_lifter: int = 8,
        occupied_band_fraction: float = 0.70,
        log_correction_clip: float = 1.25,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.shrinkage = float(shrinkage)
        self.cepstral_lifter = int(cepstral_lifter)
        self.occupied_band_fraction = float(occupied_band_fraction)
        self.log_correction_clip = float(log_correction_clip)
        self.eps = float(eps)

    def _low_quefrency_envelope(self, log_mag: torch.Tensor) -> torch.Tensor:
        length = int(log_mag.size(-1))
        lifter = max(1, min(int(self.cepstral_lifter), max(1, length // 2)))
        cep = torch.fft.ifft(torch.complex(log_mag, torch.zeros_like(log_mag)), dim=-1).real
        idx = torch.arange(length, device=log_mag.device)
        mask = ((idx <= lifter) | (idx >= length - lifter)).to(dtype=log_mag.dtype)
        cep_low = cep * mask.view(1, -1)
        return torch.fft.fft(torch.complex(cep_low, torch.zeros_like(cep_low)), dim=-1).real

    def _fingerprint_stats(self, x: torch.Tensor, spectrum: torch.Tensor, log_mag: torch.Tensor) -> Dict[str, torch.Tensor]:
        length = int(spectrum.size(-1))
        power = spectrum.real.square() + spectrum.imag.square()
        shifted = torch.fft.fftshift(power, dim=-1)
        band_fraction = min(0.98, max(0.05, float(self.occupied_band_fraction)))
        band_bins = max(1, min(length, int(round(length * band_fraction))))
        start = max(0, (length - band_bins) // 2)
        stop = min(length, start + band_bins)
        inband = shifted[:, start:stop].sum(dim=-1)
        total = shifted.sum(dim=-1)
        outband = (total - inband).clamp_min(0.0)
        spectral_regrowth = (outband / inband.clamp_min(self.eps)).mean()

        mirror = torch.flip(torch.conj(spectrum), dims=(-1,))
        image_num = torch.abs(spectrum * mirror).sum(dim=-1)
        image_den = power.sum(dim=-1).clamp_min(self.eps)
        iq_image = (image_num / image_den).mean()

        cep = torch.fft.ifft(torch.complex(log_mag, torch.zeros_like(log_mag)), dim=-1).real
        lifter = max(1, min(int(self.cepstral_lifter), max(1, length // 2)))
        idx = torch.arange(length, device=x.device)
        detail_mask = ((idx > lifter + 1) & (idx < length - lifter - 1)).to(dtype=x.dtype)
        cep_detail = torch.sqrt((cep.square() * detail_mask.view(1, -1)).sum(dim=-1).clamp_min(self.eps)).mean()

        z = _iq_to_complex(x)
        cubic = (torch.abs(z).square() * z)
        cubic_corr = torch.abs((z.conj() * cubic).mean(dim=-1)) / (
            torch.sqrt((torch.abs(z).square().mean(dim=-1) * torch.abs(cubic).square().mean(dim=-1)).clamp_min(self.eps))
        )
        return {
            "spectral_regrowth_ratio": spectral_regrowth,
            "iq_image_ratio": iq_image,
            "cepstral_detail_energy": cep_detail,
            "cubic_nonlinearity_corr": cubic_corr.mean(),
        }

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        AmplitudeNormalizer._check_iq(x)
        z = _iq_to_complex(x)
        spectrum = torch.fft.fft(z, dim=-1)
        log_mag = torch.log(torch.abs(spectrum).clamp_min(self.eps))
        smooth_log = self._low_quefrency_envelope(log_mag)
        smooth_centered = smooth_log - smooth_log.mean(dim=-1, keepdim=True)
        correction_log = (-float(self.shrinkage) * smooth_centered).clamp(
            min=-float(self.log_correction_clip),
            max=float(self.log_correction_clip),
        )
        projected_spectrum = spectrum * torch.exp(correction_log)
        projected = torch.fft.ifft(projected_spectrum, dim=-1)
        x_projected = _complex_to_iq(projected, x.dtype)

        if not return_aux:
            return x_projected, None

        out_log_mag = torch.log(torch.abs(projected_spectrum).clamp_min(self.eps))
        stats_in = self._fingerprint_stats(x, spectrum, log_mag)
        stats_out = self._fingerprint_stats(x_projected, projected_spectrum, out_log_mag)
        delta_rms = _safe_rms(x_projected - x)
        input_rms = _safe_rms(x)
        aux: Dict[str, torch.Tensor] = {
            "fpcr_projected": x_projected,
            "fpcr_smooth_log_channel": smooth_log,
            "fpcr_projection_delta_rms": delta_rms,
            "fpcr_projection_ratio": delta_rms / input_rms.clamp_min(self.eps),
            "fpcr_projection_shrinkage": x.new_tensor(float(self.shrinkage)),
        }
        for key, value in stats_in.items():
            aux[f"fpcr_{key}_in"] = value
        for key, value in stats_out.items():
            aux[f"fpcr_{key}_out"] = value
        return x_projected, aux


class BoundedFingerprintResidual(nn.Module):
    """Small TCN residual branch with an explicit L2 budget."""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 24,
        num_blocks: int = 2,
        kernel_size: int = 5,
        max_residual_ratio: float = 0.06,
        max_gamma: float = 0.25,
        init_gamma: float = 0.0,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.max_residual_ratio = float(max_residual_ratio)
        self.max_gamma = float(max_gamma)
        self.eps = float(eps)
        layers: List[nn.Module] = []
        ch = int(in_channels)
        hidden = int(hidden_channels)
        blocks = max(1, int(num_blocks))
        for idx in range(blocks):
            dilation = 2 ** idx
            pad = dilation * (int(kernel_size) // 2)
            out_ch = hidden if idx < blocks - 1 else int(in_channels)
            layers.extend(
                [
                    nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=pad, dilation=dilation, groups=ch, bias=False),
                    nn.BatchNorm1d(ch),
                    nn.SiLU(inplace=True),
                    nn.Conv1d(ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.SiLU(inplace=True) if idx < blocks - 1 else nn.Identity(),
                ]
            )
            if float(dropout) > 0.0 and idx < blocks - 1:
                layers.append(nn.Dropout(p=min(0.5, max(0.0, float(dropout)))))
            ch = out_ch
        self.net = nn.Sequential(*layers)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def effective_gamma(self) -> torch.Tensor:
        if self.max_gamma <= 0.0:
            return self.gamma
        return torch.tanh(self.gamma) * self.max_gamma

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        AmplitudeNormalizer._check_iq(x)
        raw_delta = self.effective_gamma() * self.net(x)
        delta_rms = _safe_rms(raw_delta, self.eps)
        input_rms = _safe_rms(x, self.eps)
        raw_ratio = delta_rms / input_rms.clamp_min(self.eps)
        if self.max_residual_ratio > 0.0:
            budget_scale = torch.clamp(x.new_tensor(float(self.max_residual_ratio)) / raw_ratio.clamp_min(self.eps), max=1.0)
        else:
            budget_scale = x.new_tensor(1.0)
        delta = raw_delta * budget_scale
        out = x + delta
        if not return_aux:
            return out, None
        final_delta_rms = _safe_rms(delta, self.eps)
        final_ratio = final_delta_rms / input_rms.clamp_min(self.eps)
        aux = {
            "fpcr_residual_delta_rms": final_delta_rms,
            "fpcr_residual_raw_ratio": raw_ratio,
            "fpcr_residual_ratio": final_ratio,
            "fpcr_residual_budget": x.new_tensor(float(self.max_residual_ratio)),
            "fpcr_residual_budget_scale": budget_scale,
            "fpcr_effective_gamma": self.effective_gamma().abs(),
            "fpcr_budget_loss": torch.relu(final_ratio - x.new_tensor(float(self.max_residual_ratio))),
        }
        return out, aux


class FPCRSGCReconstructor(nn.Module):
    """Fingerprint-Preserving Constrained Reconstruction for SGC inputs."""

    def __init__(
        self,
        in_channels: int = 2,
        shrinkage: float = 0.35,
        cepstral_lifter: int = 8,
        occupied_band_fraction: float = 0.70,
        log_correction_clip: float = 1.25,
        use_learned_residual: bool = True,
        residual_channels: int = 24,
        residual_blocks: int = 2,
        residual_kernel_size: int = 5,
        max_residual_ratio: float = 0.06,
        residual_max_gamma: float = 0.25,
        residual_init_gamma: float = 0.0,
        residual_dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if int(in_channels) != 2:
            raise ValueError("FPCRSGCReconstructor expects two IQ channels")
        self.projector = FingerprintPreservingChannelProjector(
            shrinkage=shrinkage,
            cepstral_lifter=cepstral_lifter,
            occupied_band_fraction=occupied_band_fraction,
            log_correction_clip=log_correction_clip,
            eps=eps,
        )
        self.use_learned_residual = bool(use_learned_residual)
        self.residual = (
            BoundedFingerprintResidual(
                in_channels=in_channels,
                hidden_channels=residual_channels,
                num_blocks=residual_blocks,
                kernel_size=residual_kernel_size,
                max_residual_ratio=max_residual_ratio,
                max_gamma=residual_max_gamma,
                init_gamma=residual_init_gamma,
                dropout=residual_dropout,
                eps=eps,
            )
            if self.use_learned_residual
            else nn.Identity()
        )
        self.eps = float(eps)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        AmplitudeNormalizer._check_iq(x)
        projected, proj_aux = self.projector(x, return_aux=True)
        if self.use_learned_residual:
            out, res_aux = self.residual(projected, return_aux=True)
        else:
            out, res_aux = projected, {
                "fpcr_residual_delta_rms": x.new_tensor(0.0),
                "fpcr_residual_ratio": x.new_tensor(0.0),
                "fpcr_residual_budget": x.new_tensor(0.0),
                "fpcr_budget_loss": x.new_tensor(0.0),
            }

        if not return_aux:
            return out, None
        aux: Dict[str, torch.Tensor] = {}
        aux.update(proj_aux or {})
        aux.update(res_aux or {})
        aux.update(
            {
                "adapter_input": x,
                "adapter_output": out,
                "adapter_delta_rms": _safe_rms(out - x, self.eps),
                "adapter_input_rms": _safe_rms(x, self.eps),
                "fpcr_projected_input_rms": _safe_rms(projected, self.eps),
                "fpcr_total_ratio": _safe_rms(out - x, self.eps) / _safe_rms(x, self.eps).clamp_min(self.eps),
            }
        )
        return out, aux


class SGCAdapter(nn.Module):
    """Satellite-ground-channel-aware adapter for IQ inputs."""

    def __init__(
        self,
        in_channels: int = 2,
        adapter_mode: str = "legacy",
        use_fpcr: bool = False,
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
        fpcr_shrinkage: float = 0.35,
        fpcr_cepstral_lifter: int = 8,
        fpcr_occupied_band_fraction: float = 0.70,
        fpcr_log_correction_clip: float = 1.25,
        fpcr_use_learned_residual: bool = True,
        fpcr_residual_channels: int = 24,
        fpcr_residual_blocks: int = 2,
        fpcr_residual_kernel_size: int = 5,
        fpcr_max_residual_ratio: float = 0.06,
        fpcr_residual_max_gamma: float = 0.25,
        fpcr_residual_init_gamma: float = 0.0,
        fpcr_residual_dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.adapter_mode = "fpcr" if bool(use_fpcr) else str(adapter_mode or "legacy").lower().strip()
        if self.adapter_mode not in ("legacy", "fpcr"):
            raise ValueError(f"Unknown SGC adapter_mode={adapter_mode}")
        self.use_amp_norm = bool(use_amp_norm)
        self.use_freq_comp = bool(use_freq_comp)
        self.use_spectral_suppressor = bool(use_spectral_suppressor)
        self.use_residual_comp = bool(use_residual_comp)
        self.residual_mode = str(residual_mode or "plain").lower().strip()

        if self.adapter_mode == "fpcr":
            self.fpcr_reconstructor = FPCRSGCReconstructor(
                in_channels=in_channels,
                shrinkage=fpcr_shrinkage,
                cepstral_lifter=fpcr_cepstral_lifter,
                occupied_band_fraction=fpcr_occupied_band_fraction,
                log_correction_clip=fpcr_log_correction_clip,
                use_learned_residual=fpcr_use_learned_residual,
                residual_channels=fpcr_residual_channels,
                residual_blocks=fpcr_residual_blocks,
                residual_kernel_size=fpcr_residual_kernel_size,
                max_residual_ratio=fpcr_max_residual_ratio,
                residual_max_gamma=fpcr_residual_max_gamma,
                residual_init_gamma=fpcr_residual_init_gamma,
                residual_dropout=fpcr_residual_dropout,
                eps=eps,
            )
            self.amp_norm = nn.Identity()
            self.freq_comp = nn.Identity()
            self.spectral_sup = nn.Identity()
            self.residual_comp = nn.Identity()
            return

        self.fpcr_reconstructor = None
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
        if self.adapter_mode == "fpcr":
            return self.fpcr_reconstructor(x, return_aux=return_aux)

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
            "adapter_mode": self.adapter_mode,
            "amp_norm": self.use_amp_norm,
            "freq_comp": self.use_freq_comp,
            "spectral_suppressor": self.use_spectral_suppressor,
            "residual_comp": self.use_residual_comp,
            "residual_mode": self.residual_mode if self.use_residual_comp else "none",
        }
