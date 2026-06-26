from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


TensorDict = Dict[str, torch.Tensor]


class BaseAnchoredLogitCalibrator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        scenario_dim: int,
        num_classes: int,
        p_stats_dim: int = 4,
        topk_only: int = 3,
        epsilon_logit: float = 0.5,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.topk_only = int(topk_only)
        self.epsilon_logit = float(epsilon_logit)
        self.p_stats_dim = int(p_stats_dim)
        in_dim = int(feature_dim) + int(scenario_dim) + self.num_classes + self.p_stats_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.num_classes),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _topk_mask(self, logits_base: torch.Tensor) -> torch.Tensor:
        k = min(max(1, self.topk_only), logits_base.size(-1))
        index = logits_base.topk(k, dim=-1).indices
        mask = torch.zeros_like(logits_base, dtype=torch.bool)
        return mask.scatter(1, index, True)

    def _clip_delta(self, delta: torch.Tensor) -> torch.Tensor:
        norm = delta.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = (self.epsilon_logit / norm).clamp(max=1.0)
        return delta * scale.to(dtype=delta.dtype)

    def forward(
        self,
        z_sgc: torch.Tensor,
        scenario_code: torch.Tensor,
        logits_base: torch.Tensor,
        p_stats: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TensorDict]:
        probs = torch.softmax(logits_base.detach(), dim=-1)
        if p_stats is None:
            p_stats = logits_base.new_zeros(logits_base.size(0), self.p_stats_dim)
        u = torch.cat([z_sgc, scenario_code, probs.to(dtype=z_sgc.dtype), p_stats.to(dtype=z_sgc.dtype)], dim=-1)
        raw = self.net(u)
        raw = raw.masked_fill(~self._topk_mask(logits_base.detach()), 0.0)
        delta = self._clip_delta(raw)
        if gate is not None:
            delta = delta * gate.to(device=delta.device, dtype=delta.dtype).view(-1, 1)
        logits_sgc = logits_base + delta
        return logits_sgc, {
            "delta_logits": delta,
            "delta_logits_raw": raw,
            "delta_logit_norm": delta.float().norm(dim=-1),
        }
