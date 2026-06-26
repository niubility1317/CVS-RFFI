from __future__ import annotations

import torch
import torch.nn as nn


class ResidualSafetyGate(nn.Module):
    def __init__(self, cond_dim: int = 24) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.net = nn.Sequential(
            nn.Linear(self.cond_dim + 4, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if y.ndim != 3 or y.size(1) != 2:
            raise ValueError("y must be shaped [B, 2, T].")
        if c.ndim != 2 or c.size(0) != y.size(0):
            raise ValueError("c must be shaped [B, cond_dim].")
        yf = y.float()
        rms = torch.sqrt(yf.square().mean(dim=(1, 2)) + 1e-6)
        papr = yf.square().amax(dim=(1, 2)) / (yf.square().mean(dim=(1, 2)) + 1e-6)
        dy = yf[:, :, 1:] - yf[:, :, :-1]
        rough = torch.sqrt(dy.square().mean(dim=(1, 2)) + 1e-6)
        bias = torch.ones_like(rms)
        stats = torch.stack([rms, papr, rough, bias], dim=-1).to(dtype=c.dtype)
        return self.net(torch.cat([c, stats], dim=-1)).view(-1, 1, 1)
