from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn


WEATHER_TO_INDEX = {"clear": 0, "cloudy": 1, "rain": 2, "storm": 3}
ORBIT_TO_INDEX = {"LEO": 0, "MEO": 1, "GEO": 2, "leo": 0, "meo": 1, "geo": 2}
STATE_TO_INDEX = {"LOS": 0, "LOO": 1, "Rayleigh": 2, "los": 0, "loo": 1, "rayleigh": 2}


def _as_tensor(value, *, device, dtype=torch.float32) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _as_long(value, *, device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device).long()
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
        mapped = [ORBIT_TO_INDEX.get(v, STATE_TO_INDEX.get(v, WEATHER_TO_INDEX.get(v, 0))) for v in value]
        return torch.tensor(mapped, device=device, dtype=torch.long)
    if isinstance(value, str):
        return torch.tensor([ORBIT_TO_INDEX.get(value, STATE_TO_INDEX.get(value, WEATHER_TO_INDEX.get(value, 0)))], device=device)
    return torch.as_tensor(value, device=device).long()


def normalize_sat_meta(meta: Mapping[str, object], *, device=None, pl_mean: float = 180.0, pl_std: float = 12.0) -> dict[str, torch.Tensor]:
    first = next((v for v in meta.values() if torch.is_tensor(v)), None)
    if device is None:
        device = first.device if first is not None else torch.device("cpu")
    batch = None
    for key in ("theta_deg", "h_km", "orbit", "state"):
        if key in meta:
            value = meta[key]
            batch = int(value.numel()) if torch.is_tensor(value) else (len(value) if isinstance(value, (list, tuple)) else 1)
            break
    batch = int(batch or 1)

    def cont(key: str, default: float) -> torch.Tensor:
        value = _as_tensor(meta.get(key, torch.full((batch,), default)), device=device).view(-1)
        return value.expand(batch) if value.numel() == 1 and batch > 1 else value

    def disc(key: str, default: int, modulo: int) -> torch.Tensor:
        value = _as_long(meta.get(key, torch.full((batch,), default)), device=device).view(-1)
        value = value.expand(batch) if value.numel() == 1 and batch > 1 else value
        return value.clamp_min(0).remainder(modulo)

    theta = cont("theta_deg", 45.0)
    h = cont("h_km", 1_000.0)
    d = cont("d_km", 2_000.0)
    pl = cont("pl_db", pl_mean)
    fD = cont("fD_hz", 0.0)
    cfo = cont("cfo_hz", 0.0)
    snr = cont("snr_db", 20.0)
    K = cont("K_db", 9.0)
    return {
        "orbit": disc("orbit", 0, 3),
        "state": disc("state", 0, 3),
        "weather": disc("weather", 0, 4),
        "theta_norm": (theta / 90.0).clamp(0.0, 1.5),
        "h_norm": torch.log(h.clamp_min(0.0) + 1.0) / math.log(40_000.0),
        "d_norm": torch.log(d.clamp_min(0.0) + 1.0) / math.log(50_000.0),
        "pl_norm": (pl - float(pl_mean)) / max(float(pl_std), 1e-6),
        "fD_norm": fD / 50_000.0,
        "cfo_norm": cfo / 1_000.0,
        "snr_norm": (snr - 10.0) / 20.0,
        "K_norm": K / 18.0,
    }


def estimate_phy_proxy(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if y.ndim != 3 or y.size(1) != 2:
        raise ValueError("y must be shaped [B, 2, T].")
    z = torch.complex(y[:, 0].float(), y[:, 1].float())
    power = z.real.square() + z.imag.square()
    rms = torch.sqrt(power.mean(dim=-1) + eps)
    papr = power.amax(dim=-1) / (power.mean(dim=-1) + eps)
    phase = torch.angle(z[:, 1:] * torch.conj(z[:, :-1]))
    cfo_proxy = phase.mean(dim=-1)
    pn_proxy = phase.std(dim=-1)
    iq_corr = (y[:, 0].float() * y[:, 1].float()).mean(dim=-1)
    spec = torch.fft.fft(z, dim=-1)
    mag = spec.abs().clamp_min(eps)
    freqs = torch.linspace(-1.0, 1.0, steps=mag.size(-1), device=y.device)
    weights = mag / mag.sum(dim=-1, keepdim=True).clamp_min(eps)
    centroid = (weights * freqs).sum(dim=-1)
    bandwidth = torch.sqrt((weights * (freqs.view(1, -1) - centroid.view(-1, 1)).square()).sum(dim=-1) + eps)
    split = max(1, mag.size(-1) // 4)
    low = mag[:, :split].mean(dim=-1)
    high = mag[:, -split:].mean(dim=-1)
    band_ratio = high / (low + eps)
    return torch.stack([rms, papr, cfo_proxy, pn_proxy, iq_corr, centroid, bandwidth, band_ratio], dim=-1)


class PhyConditionEncoder(nn.Module):
    def __init__(self, num_orbit: int = 3, num_state: int = 3, num_weather: int = 4, out_dim: int = 24) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.orbit_emb = nn.Embedding(int(num_orbit), 4)
        self.state_emb = nn.Embedding(int(num_state), 4)
        self.weather_emb = nn.Embedding(int(num_weather), 4)
        self.mlp = nn.Sequential(nn.Linear(4 + 4 + 4 + 8, 48), nn.SiLU(), nn.Linear(48, self.out_dim))
        self.proxy_mlp = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, self.out_dim))

    def forward(self, meta: Mapping[str, torch.Tensor]) -> torch.Tensor:
        discrete = torch.cat(
            [
                self.orbit_emb(meta["orbit"].long()),
                self.state_emb(meta["state"].long()),
                self.weather_emb(meta["weather"].long()),
            ],
            dim=-1,
        )
        continuous = torch.stack(
            [
                meta["theta_norm"],
                meta["h_norm"],
                meta["d_norm"],
                meta["pl_norm"],
                meta["fD_norm"],
                meta["cfo_norm"],
                meta["snr_norm"],
                meta["K_norm"],
            ],
            dim=-1,
        ).to(dtype=discrete.dtype)
        return self.mlp(torch.cat([discrete, continuous], dim=-1))

    def from_raw_meta(self, meta: Mapping[str, object], *, device=None) -> torch.Tensor:
        return self(normalize_sat_meta(meta, device=device or next(self.parameters()).device))

    def from_iq_proxy(self, y: torch.Tensor) -> torch.Tensor:
        return self.proxy_mlp(estimate_phy_proxy(y).to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype))
