from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

from .spectrogram import ensure_iq_2xl


class OnlineRFChannelAugment:
    """Approximate TDL/Doppler/AWGN augmentation for IQ tensors.

    This is a practical reproduction of the channel/noise augmentation used by
    the compared papers. It keeps the sample length fixed and avoids operations
    that intentionally erase transmitter hardware fingerprints.
    """

    def __init__(
        self,
        sample_rate: float,
        rms_delay_ns_range: Tuple[float, float] = (5.0, 300.0),
        doppler_hz_range: Tuple[float, float] = (0.0, 5.0),
        snr_db_range: Tuple[float, float] = (10.0, 40.0),
        max_taps: int = 8,
        p: float = 1.0,
        rms_normalize: bool = True,
    ):
        self.sample_rate = float(sample_rate)
        self.rms_delay_ns_range = tuple(float(x) for x in rms_delay_ns_range)
        self.doppler_hz_range = tuple(float(x) for x in doppler_hz_range)
        self.snr_db_range = tuple(float(x) for x in snr_db_range)
        self.max_taps = max(1, int(max_taps))
        self.p = float(p)
        self.rms_normalize = bool(rms_normalize)

    @staticmethod
    def _rand_uniform(lo: float, hi: float, device) -> torch.Tensor:
        return torch.empty((), device=device).uniform_(float(lo), float(hi))

    def _random_taps(self, device, dtype) -> torch.Tensor:
        n_taps = int(torch.randint(1, self.max_taps + 1, (1,), device=device).item())
        delay_ns = self._rand_uniform(*self.rms_delay_ns_range, device=device).clamp_min(1e-3)
        tau = torch.arange(n_taps, device=device, dtype=dtype)
        decay = torch.exp(-tau / delay_ns.to(dtype=dtype).clamp_min(1.0))
        phase = 2.0 * math.pi * torch.rand(n_taps, device=device, dtype=dtype)
        real = torch.randn(n_taps, device=device, dtype=dtype) * torch.cos(phase)
        imag = torch.randn(n_taps, device=device, dtype=dtype) * torch.sin(phase)
        taps = torch.complex(real, imag) * decay
        return taps / torch.sqrt((taps.real.square() + taps.imag.square()).sum().clamp_min(1e-8))

    @staticmethod
    def _same_length_conv(x: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
        # x: [1, 2, L], taps: complex [K]
        k = int(taps.numel())
        pad = k // 2
        xr = x[:, 0:1]
        xi = x[:, 1:2]
        hr = taps.real.flip(0).view(1, 1, k)
        hi = taps.imag.flip(0).view(1, 1, k)
        yr = F.conv1d(xr, hr, padding=pad) - F.conv1d(xi, hi, padding=pad)
        yi = F.conv1d(xr, hi, padding=pad) + F.conv1d(xi, hr, padding=pad)
        y = torch.cat([yr, yi], dim=1)
        length = x.size(-1)
        if y.size(-1) > length:
            start = (y.size(-1) - length) // 2
            y = y[..., start : start + length]
        elif y.size(-1) < length:
            y = F.pad(y, (0, length - y.size(-1)))
        return y

    @staticmethod
    def _rms_normalize(iq: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        rms = torch.sqrt(iq[:, 0].square().add(iq[:, 1].square()).mean(dim=-1, keepdim=True).clamp_min(eps))
        return iq / rms.unsqueeze(1)

    def __call__(self, iq: torch.Tensor) -> torch.Tensor:
        x = ensure_iq_2xl(iq)
        was_single = iq.dim() == 2
        if self.p <= 0 or float(torch.rand(())) > self.p:
            return iq.clone()

        out = []
        for sample in x:
            y = sample.unsqueeze(0)
            taps = self._random_taps(y.device, y.dtype)
            y = self._same_length_conv(y, taps)

            fd = self._rand_uniform(*self.doppler_hz_range, device=y.device).to(dtype=y.dtype)
            if float(fd.abs()) > 0:
                t = torch.arange(y.size(-1), device=y.device, dtype=y.dtype) / self.sample_rate
                phase = 2.0 * math.pi * fd * t
                cr = torch.cos(phase)
                ci = torch.sin(phase)
                yr = y[:, 0] * cr - y[:, 1] * ci
                yi = y[:, 0] * ci + y[:, 1] * cr
                y = torch.stack([yr, yi], dim=1)

            snr_db = self._rand_uniform(*self.snr_db_range, device=y.device).to(dtype=y.dtype)
            sig_power = y[:, 0].square().add(y[:, 1].square()).mean().clamp_min(1e-8)
            noise_power = sig_power / (10.0 ** (snr_db / 10.0))
            noise = torch.randn_like(y) * torch.sqrt(noise_power / 2.0)
            y = y + noise
            if self.rms_normalize:
                y = self._rms_normalize(y)
            out.append(torch.nan_to_num(y.squeeze(0), nan=0.0, posinf=0.0, neginf=0.0))
        yb = torch.stack(out, dim=0)
        return yb[0] if was_single else yb
