from __future__ import annotations

from typing import Optional

import torch

from DataAugmentation import (
    _apply_agc_softclip,
    _apply_awgn,
    _apply_cfo,
    _apply_multipath_fir,
    _apply_phase_noise,
    _apply_random_lowpass,
    _apply_sro_resample,
    _from_complex,
    _nan_to_num_,
    _to_complex,
)
from sat_channel import apply_iq_imbalance

from .style_packet import StylePacket


class StyleConditionedReceiverDG:
    """Maps remote RF style statistics to conservative receiver-chain perturbations."""

    def __init__(
        self,
        *,
        max_gain_delta: float = 0.15,
        max_noise_std: float = 0.03,
        eps: float = 1e-6,
        sample_rate_hz: float = 25_000_000.0,
        style_jitter_scale: float = 1.0,
        max_cfo_hz: float = 35_000.0,
        max_sro_ppm: float = 150.0,
        max_agc_gain_db: float = 8.0,
        max_iq_gain_db: float = 3.0,
        max_iq_phase_deg: float = 3.0,
        max_phase_noise_std: float = 2e-3,
        min_awgn_snr_db: float = 10.0,
        p_lowpass: float = 0.7,
        p_multipath: float = 0.7,
        max_multipath_taps: int = 9,
    ):
        self.max_gain_delta = max(0.0, float(max_gain_delta))
        self.max_noise_std = max(0.0, float(max_noise_std))
        self.eps = float(eps)
        self.sample_rate_hz = max(float(sample_rate_hz or 0.0), 1.0)
        self.style_jitter_scale = max(0.0, float(style_jitter_scale))
        self.max_cfo_hz = max(0.0, float(max_cfo_hz))
        self.max_sro_ppm = max(0.0, float(max_sro_ppm))
        self.max_agc_gain_db = max(0.0, float(max_agc_gain_db))
        self.max_iq_gain_db = max(0.0, float(max_iq_gain_db))
        self.max_iq_phase_deg = max(0.0, float(max_iq_phase_deg))
        self.max_phase_noise_std = max(0.0, float(max_phase_noise_std))
        self.min_awgn_snr_db = max(0.0, float(min_awgn_snr_db))
        self.p_lowpass = max(0.0, min(1.0, float(p_lowpass)))
        self.p_multipath = max(0.0, min(1.0, float(p_multipath)))
        self.max_multipath_taps = max(3, int(max_multipath_taps))

    def transform(
        self,
        x: torch.Tensor,
        style: StylePacket,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        stats = style.stats
        if _has_physical_stats(stats) and x.dim() >= 3 and x.size(1) >= 2:
            return self._transform_physical(x, style, generator=generator)
        return self._transform_legacy(x, style, generator=generator)

    def _transform_legacy(
        self,
        x: torch.Tensor,
        style: StylePacket,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        stats = style.stats
        rms = _finite_float(stats.get("iq_rms", 1.0), 1.0)
        amp_std = _finite_float(stats.get("amp_std", 0.0), 0.0)
        phase_shift = _finite_float(stats.get("phase_diff_mean", 0.0), 0.0)
        gain_delta = max(-self.max_gain_delta, min(self.max_gain_delta, rms - 1.0))
        noise_std = max(0.0, min(self.max_noise_std, amp_std * self.max_noise_std))
        y = x.float() * (1.0 + gain_delta)
        if x.dim() >= 3 and x.size(1) >= 2 and abs(phase_shift) > self.eps:
            c = torch.cos(torch.tensor(phase_shift, device=x.device, dtype=y.dtype))
            s = torch.sin(torch.tensor(phase_shift, device=x.device, dtype=y.dtype))
            i = y[:, 0, :].clone()
            q = y[:, 1, :].clone()
            y[:, 0, :] = c * i - s * q
            y[:, 1, :] = s * i + c * q
        if noise_std > 0.0:
            noise = torch.randn(y.shape, generator=generator, device=y.device, dtype=y.dtype) * float(noise_std)
            y = y + noise
        return torch.nan_to_num(y, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0).to(dtype=x.dtype)

    def _transform_physical(
        self,
        x: torch.Tensor,
        style: StylePacket,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        stats = style.stats
        x_f = x.float()
        z = _to_complex(x_f)
        batch = int(z.size(0))
        device = z.device
        dtype = z.real.dtype
        fs = _finite_float(style.metadata.get("sample_rate_hz", self.sample_rate_hz), self.sample_rate_hz)
        fs = max(fs, 1.0)
        scale = self.style_jitter_scale

        cfo_hz = _finite_float(stats.get("phys_cfo_hz", 0.0), 0.0)
        if abs(cfo_hz) <= self.eps:
            cfo_cycles = _finite_float(stats.get("phys_cfo_cycles_per_sample", 0.0), 0.0)
        else:
            cfo_cycles = cfo_hz / fs
        max_cfo_cycles = min(0.49, self.max_cfo_hz / fs) if self.max_cfo_hz > 0.0 else 0.0
        cfo_cycles = _clip(cfo_cycles * scale, -max_cfo_cycles, max_cfo_cycles)

        sro_ppm = _clip(_finite_float(stats.get("phys_sro_ppm", 0.0), 0.0) * scale, -self.max_sro_ppm, self.max_sro_ppm)
        if abs(sro_ppm) > self.eps:
            z = _apply_sro_resample(z, _full_param(batch, sro_ppm, device, dtype))
        if abs(cfo_cycles) > self.eps:
            z = _apply_cfo(z, _full_param(batch, cfo_cycles, device, dtype))

        phase_noise = _clip(_finite_float(stats.get("phys_phase_noise_std", 0.0), 0.0) * scale, 0.0, self.max_phase_noise_std)
        if phase_noise > self.eps:
            z = _apply_phase_noise(z, _full_param(batch, phase_noise, device, dtype))

        iq_amp_db = _clip(_finite_float(stats.get("phys_iq_gain_imbalance_db", 0.0), 0.0) * scale, -self.max_iq_gain_db, self.max_iq_gain_db)
        iq_phase_deg = _clip(_finite_float(stats.get("phys_iq_phase_imbalance_deg", 0.0), 0.0) * scale, -self.max_iq_phase_deg, self.max_iq_phase_deg)
        if abs(iq_amp_db) > self.eps or abs(iq_phase_deg) > self.eps:
            z = apply_iq_imbalance(
                z,
                amp_db=torch.full((batch,), float(iq_amp_db), device=device, dtype=dtype),
                phase_deg=torch.full((batch,), float(iq_phase_deg), device=device, dtype=dtype),
            )

        gain_db = _clip(_finite_float(stats.get("phys_agc_gain_db", 0.0), 0.0) * scale, -self.max_agc_gain_db, self.max_agc_gain_db)
        softclip = max(self.eps, _finite_float(stats.get("phys_softclip_level", 8.0), 8.0))
        if abs(gain_db) > self.eps or softclip < 8.0:
            z = _apply_agc_softclip(
                z,
                _full_param(batch, gain_db, device, dtype),
                _full_param(batch, softclip, device, dtype),
            )

        mp_strength = _clip(_finite_float(stats.get("phys_multipath_strength", 0.0), 0.0) * scale, 0.0, 1.0)
        if mp_strength > self.eps and self.p_multipath > 0.0:
            z = self._apply_style_multipath(z, mp_strength, generator=generator)

        cutoff_frac = _clip(_finite_float(stats.get("phys_lowpass_cutoff_frac", 1.0), 1.0), 0.05, 1.0)
        transition_frac = _clip(_finite_float(stats.get("phys_lowpass_transition_frac", 0.05), 0.05), 0.005, 0.20)
        if cutoff_frac < 0.995 and self.p_lowpass > 0.0:
            nyq = fs / 2.0
            z = _apply_random_lowpass(
                z,
                fs,
                _full_param(batch, cutoff_frac * nyq, device, dtype),
                _full_param(batch, transition_frac * nyq, device, dtype),
            )

        snr_db = _finite_float(stats.get("phys_awgn_snr_db", 80.0), 80.0)
        if self.max_noise_std > 0.0 and snr_db < 60.0:
            z = _apply_awgn(z, _full_param(batch, max(self.min_awgn_snr_db, snr_db), device, dtype))

        y = _from_complex(_nan_to_num_(z))
        return torch.nan_to_num(y, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0).to(dtype=x.dtype)

    def _apply_style_multipath(
        self,
        z: torch.Tensor,
        strength: float,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        batch = int(z.size(0))
        device = z.device
        dtype = z.real.dtype
        max_taps = self.max_multipath_taps
        if max_taps % 2 == 0:
            max_taps += 1
        taps_len = max(3, min(max_taps, 3 + int(round(float(strength) * float(max_taps - 3)))))
        if taps_len % 2 == 0:
            taps_len += 1
        taps = torch.zeros((batch, taps_len), device=device, dtype=z.dtype)
        taps[:, 0] = torch.complex(
            torch.full((batch,), 1.0 - 0.35 * float(strength), device=device, dtype=dtype),
            torch.zeros((batch,), device=device, dtype=dtype),
        )
        if taps_len > 1:
            delay = torch.arange(1, taps_len, device=device, dtype=dtype).view(1, -1)
            decay = torch.exp(-delay / max(1.0, 1.5 + 2.0 * float(strength)))
            rand_r = torch.randn((batch, taps_len - 1), generator=generator, device=device, dtype=dtype)
            rand_i = torch.randn((batch, taps_len - 1), generator=generator, device=device, dtype=dtype)
            tail = torch.complex(rand_r * decay, rand_i * decay)
            tail = tail / tail.abs().sum(dim=1, keepdim=True).clamp_min(self.eps)
            taps[:, 1:] = tail * (0.35 * float(strength))
        taps = taps / taps.abs().sum(dim=1, keepdim=True).clamp_min(self.eps)
        return _apply_multipath_fir(z, taps)


def _finite_float(value, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out or out in (float("inf"), -float("inf")):
        return float(default)
    return out


def _has_physical_stats(stats) -> bool:
    return any(str(key).startswith("phys_") for key in (stats or {}).keys())


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(value))))


def _full_param(batch: int, value: float, device, dtype) -> torch.Tensor:
    return torch.full((int(batch), 1), float(value), device=device, dtype=dtype)
