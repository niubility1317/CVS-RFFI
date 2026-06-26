from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorDict = Dict[str, torch.Tensor]


STAT_NAMES = (
    "rms_power",
    "papr",
    "phase_step_mean",
    "phase_step_var",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
    "autocorr_lag1_real",
    "autocorr_lag1_imag",
    "short_energy_var",
    "envelope_kurtosis",
    "cfo_magnitude",
)


def _safe(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)


def compute_channel_statistics(x: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, TensorDict]:
    if x.dim() != 3 or x.size(1) != 2:
        raise ValueError("IQ input must be shaped [B, 2, T].")
    s = torch.complex(x[:, 0].float(), x[:, 1].float())
    power = s.abs().square()
    rms_power = power.mean(dim=-1)
    papr = power.max(dim=-1).values / rms_power.clamp_min(eps)
    phase_step = torch.angle(s[:, 1:] * torch.conj(s[:, :-1]))
    phase_step_mean = phase_step.mean(dim=-1)
    phase_step_var = phase_step.var(dim=-1, unbiased=False)

    spec = torch.fft.fft(s, dim=-1)
    mag = spec.abs().clamp_min(eps)
    freq = torch.linspace(-1.0, 1.0, steps=s.size(-1), device=s.device).view(1, -1)
    weight = mag / mag.sum(dim=-1, keepdim=True).clamp_min(eps)
    centroid = (weight * freq).sum(dim=-1)
    bandwidth = torch.sqrt((weight * (freq - centroid.view(-1, 1)).square()).sum(dim=-1).clamp_min(0.0))
    spectral_flatness = torch.exp(mag.log().mean(dim=-1)) / mag.mean(dim=-1).clamp_min(eps)

    autocorr = (s[:, 1:] * torch.conj(s[:, :-1])).mean(dim=-1)
    chunks = min(8, max(1, x.size(-1) // 4))
    short_energy = power[:, : chunks * (x.size(-1) // chunks)].reshape(x.size(0), chunks, -1).mean(dim=-1)
    short_energy_var = short_energy.var(dim=-1, unbiased=False)
    env = s.abs()
    env_centered = env - env.mean(dim=-1, keepdim=True)
    env_kurt = env_centered.pow(4).mean(dim=-1) / env_centered.pow(2).mean(dim=-1).clamp_min(eps).square()
    cfo_magnitude = phase_step_mean.abs()

    values = (
        rms_power,
        papr,
        phase_step_mean,
        phase_step_var,
        centroid,
        bandwidth,
        spectral_flatness,
        autocorr.real,
        autocorr.imag,
        short_energy_var,
        env_kurt,
        cfo_magnitude,
    )
    stats = {name: _safe(value).unsqueeze(-1) for name, value in zip(STAT_NAMES, values)}
    matrix = torch.cat([stats[name] for name in STAT_NAMES], dim=-1)
    matrix = (matrix - matrix.mean(dim=-1, keepdim=True)) / matrix.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
    return _safe(matrix), stats


class SatelliteEvidenceEncoder(nn.Module):
    """Small channel-state encoder fed by physics statistics, not identity features."""

    def __init__(
        self,
        num_views: int,
        scenario_dim: int = 16,
        num_experts: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_views = int(num_views)
        self.scenario_dim = int(scenario_dim)
        self.num_experts = int(num_experts)
        self.net = nn.Sequential(
            nn.Linear(len(STAT_NAMES), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.scenario = nn.Linear(int(hidden_dim), self.scenario_dim)
        self.sat_score = nn.Linear(int(hidden_dim), 1)
        self.view_head = nn.Linear(int(hidden_dim), self.num_views)
        self.expert_head = nn.Linear(int(hidden_dim), self.num_experts)
        self.quality_head = nn.Linear(int(hidden_dim), 1)

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        stat_matrix, stat_dict = compute_channel_statistics(x)
        h = self.net(stat_matrix)
        code = torch.tanh(self.scenario(h))
        sat_logit = self.sat_score(h)
        return {
            "scenario_code": code,
            "sat_logit": sat_logit,
            "sat_score": torch.sigmoid(sat_logit),
            "view_weights": F.softmax(self.view_head(h), dim=-1),
            "expert_weights": F.softmax(self.expert_head(h), dim=-1),
            "quality": torch.sigmoid(self.quality_head(h)),
            "channel_stats": stat_dict,
            "stat_matrix": stat_matrix,
        }
