from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


TensorDict = Dict[str, torch.Tensor]


def clamp_by_norm(delta: torch.Tensor, reference: torch.Tensor, ratio: float, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    ref_norm = reference.float().norm(dim=-1, keepdim=True).clamp_min(eps)
    delta_norm = delta.float().norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = float(ratio) * ref_norm
    scale = (max_norm / delta_norm).clamp(max=1.0)
    clamped = delta * scale.to(dtype=delta.dtype)
    applied_ratio = clamped.float().norm(dim=-1, keepdim=True) / ref_norm
    return clamped, applied_ratio


class IdentityPreservingFeatureAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        scenario_dim: int,
        p_stats_dim: int = 4,
        rank: int = 16,
        hidden_dim: int = 128,
        epsilon_z: float = 0.02,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.scenario_dim = int(scenario_dim)
        self.p_stats_dim = int(p_stats_dim)
        self.epsilon_z = float(epsilon_z)
        in_dim = self.feature_dim + self.scenario_dim + self.p_stats_dim
        self.norm = nn.LayerNorm(in_dim)
        self.down = nn.Linear(in_dim, int(hidden_dim))
        self.mid = nn.Linear(int(hidden_dim), int(rank))
        self.up = nn.Linear(int(rank), self.feature_dim)
        self.dropout = nn.Dropout(float(dropout))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        z_base: torch.Tensor,
        scenario_code: torch.Tensor,
        p_stats: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TensorDict]:
        if p_stats is None:
            p_stats = z_base.new_zeros(z_base.size(0), self.p_stats_dim)
        if p_stats.size(-1) != self.p_stats_dim:
            raise ValueError(f"p_stats must have dim {self.p_stats_dim}, got {p_stats.size(-1)}.")
        u = torch.cat([z_base, scenario_code, p_stats.to(dtype=z_base.dtype)], dim=-1)
        raw = self.up(torch.nn.functional.gelu(self.mid(self.dropout(torch.nn.functional.gelu(self.down(self.norm(u)))))))
        delta, ratio = clamp_by_norm(raw, z_base, self.epsilon_z)
        if gate is not None:
            delta = delta * gate.to(device=delta.device, dtype=delta.dtype).view(-1, 1)
            ratio = ratio * gate.to(device=ratio.device, dtype=ratio.dtype).view(-1, 1).abs()
        z_sgc = z_base + delta
        return z_sgc, {
            "delta_z": delta,
            "delta_z_raw": raw,
            "delta_z_ratio": ratio.squeeze(-1),
        }
