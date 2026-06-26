"""Star-ground proxy view generation for IQ tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SGVStrengthConfig:
    cfo_range: tuple[float, float]
    phase_noise_std: float
    snr_db: tuple[float, float]
    amp_scale: tuple[float, float]
    time_shift: tuple[int, int]
    multipath_taps: tuple[int, int]
    rician_k_db: tuple[float, float]


def sgv_strength_config(strength: str = "mid") -> SGVStrengthConfig:
    strength = str(strength or "mid").lower().strip().replace("sat_", "")
    if strength == "low":
        return SGVStrengthConfig((-1.0e-4, 1.0e-4), 0.001, (22.0, 30.0), (0.85, 1.15), (-4, 4), (1, 2), (10.0, 20.0))
    if strength == "high":
        return SGVStrengthConfig((-4.0e-4, 4.0e-4), 0.004, (10.0, 18.0), (0.50, 1.50), (-16, 16), (3, 5), (0.0, 10.0))
    return SGVStrengthConfig((-2.0e-4, 2.0e-4), 0.002, (16.0, 24.0), (0.70, 1.30), (-8, 8), (2, 3), (6.0, 15.0))


def _rand_uniform(shape, lo: float, hi: float, device) -> torch.Tensor:
    return torch.empty(shape, device=device).uniform_(float(lo), float(hi))


def _to_complex_iq(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        return x
    if x.dim() < 3 or x.size(1) != 2:
        raise ValueError("SGV expects complex IQ or real tensor shaped [B,2,T].")
    return torch.complex(x[:, 0].float(), x[:, 1].float())


def _from_complex_iq(z: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(like):
        return z
    return torch.stack([z.real, z.imag], dim=1).to(dtype=like.dtype)


class StarGroundViewGenerator(nn.Module):
    """Generate paired clean/satellite proxy IQ views."""

    def __init__(self, sample_rate_hz: float = 25e6, normalize: bool = True) -> None:
        super().__init__()
        self.sample_rate_hz = float(sample_rate_hz)
        self.normalize = bool(normalize)

    def forward(self, x_clean: torch.Tensor, strength: str = "mid") -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = sgv_strength_config(strength)
        z = _to_complex_iq(x_clean)
        bsz, steps = z.shape[0], z.shape[-1]
        device = z.device
        shift = torch.randint(cfg.time_shift[0], cfg.time_shift[1] + 1, (bsz,), device=device)
        shifted = torch.stack([torch.roll(z[i], int(shift[i].item()), dims=-1) for i in range(bsz)], dim=0)
        cfo = _rand_uniform((bsz, 1), *cfg.cfo_range, device=device)
        phase0 = _rand_uniform((bsz, 1), -3.141592653589793, 3.141592653589793, device=device)
        n = torch.arange(steps, device=device, dtype=torch.float32).view(1, -1)
        phase_noise = torch.randn((bsz, steps), device=device) * float(cfg.phase_noise_std)
        phase = phase0 + 2.0 * torch.pi * cfo * n + phase_noise
        amp = _rand_uniform((bsz, 1), *cfg.amp_scale, device=device)
        faded = amp * shifted * torch.exp(1j * phase)
        taps = int(torch.randint(cfg.multipath_taps[0], cfg.multipath_taps[1] + 1, (1,), device=device).item())
        if taps > 1:
            accum = faded
            for tap in range(1, taps):
                gain = (0.15 / tap) * torch.exp(1j * _rand_uniform((bsz, 1), -torch.pi, torch.pi, device=device))
                accum = accum + gain * torch.roll(faded, tap, dims=-1)
            faded = accum
        snr = _rand_uniform((bsz, 1), *cfg.snr_db, device=device)
        sig_power = faded.abs().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8)
        noise_power = sig_power / (10.0 ** (snr / 10.0))
        noise = torch.sqrt(noise_power / 2.0) * (torch.randn_like(faded.real) + 1j * torch.randn_like(faded.real))
        sat = faded + noise
        if self.normalize:
            clean_power = z.abs().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            sat_power = sat.abs().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            sat = sat * (clean_power / sat_power)
        params = {
            "strength_id": torch.full((bsz,), {"low": 0, "mid": 1, "high": 2}.get(str(strength).replace("sat_", ""), 1), device=device),
            "cfo": cfo.view(-1),
            "snr_db": snr.view(-1),
            "amp": amp.view(-1),
            "time_shift": shift,
            "multipath_taps": torch.full((bsz,), taps, device=device),
        }
        return _from_complex_iq(sat, x_clean), params


def estimate_sat_reliability(
    *,
    x_clean: Optional[torch.Tensor] = None,
    x_sat: Optional[torch.Tensor] = None,
    base_clean: Optional[torch.Tensor] = None,
    base_sat: Optional[torch.Tensor] = None,
    sat_params: Optional[Mapping[str, torch.Tensor]] = None,
    base_acc_drop: Optional[float] = None,
) -> torch.Tensor:
    """Estimate a sample-level reliability mask in [0, 1]."""

    ref = base_clean if base_clean is not None else (x_clean if x_clean is not None else None)
    if ref is None:
        raise ValueError("At least one tensor reference is required.")
    score = torch.ones((ref.size(0),), device=ref.device, dtype=torch.float32)
    if base_clean is not None and base_sat is not None:
        kl = F.kl_div(
            F.log_softmax(base_sat.float(), dim=1),
            F.softmax(base_clean.float().detach(), dim=1),
            reduction="none",
        ).sum(dim=1)
        score = score * torch.exp(-kl.clamp_min(0.0))
    if x_clean is not None and x_sat is not None:
        clean_c = _to_complex_iq(x_clean)
        sat_c = _to_complex_iq(x_sat)
        corr = (clean_c.conj() * sat_c).real.mean(dim=-1).abs()
        denom = clean_c.abs().pow(2).mean(dim=-1).sqrt() * sat_c.abs().pow(2).mean(dim=-1).sqrt()
        score = score * (corr / denom.clamp_min(1e-6)).clamp(0.0, 1.0)
    if base_acc_drop is not None:
        drop = float(base_acc_drop)
        if drop > 0.05:
            score = score * 0.0
        elif drop > 0.03:
            score = score * 0.5
    if sat_params and "snr_db" in sat_params:
        snr = sat_params["snr_db"].to(device=score.device).float()
        score = score * ((snr - 10.0) / 12.0).clamp(0.0, 1.0)
    return torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


__all__ = ["SGVStrengthConfig", "StarGroundViewGenerator", "estimate_sat_reliability", "sgv_strength_config"]
