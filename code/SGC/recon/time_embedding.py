from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    if t.ndim == 0:
        t = t.view(1)
    t = t.float().view(-1)
    half = int(dim) // 2
    freqs = torch.exp(
        -math.log(float(max_period))
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(1, half)
    )
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, emb.new_zeros((emb.size(0), 1))], dim=-1)
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or dim)
        self.dim = int(dim)
        self.net = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(sinusoidal_time_embedding(t, self.dim).to(next(self.parameters()).device))
