"""FJMP-v2 base-anchored safe residual prototype head."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorDict = Dict[str, torch.Tensor]


class SafeResidualProtoHead(nn.Module):
    """Base-anchored residual multi-prototype head.

    The head treats the frozen base logits as the decision anchor.  Prototype
    logits only supply a clipped residual correction, then a bounded sample-wise
    gate controls how much of that correction reaches the final logits.
    """

    def __init__(
        self,
        in_dim: int,
        proto_dim: int,
        num_classes: int,
        K: int = 3,
        rho_max: float = 0.15,
        delta_clip: float = 3.0,
        proto_dropout: float = 0.10,
        logit_scale_init: float = 10.0,
        logit_scale_min: float = 3.0,
        logit_scale_max: float = 20.0,
        gate_hidden_dim: int = 128,
        dynamic_rho_cap: bool = False,
        rho_easy_cap: float = 0.03,
        rho_boundary_cap: float = 0.30,
        rho_easy_conf: float = 0.95,
        rho_easy_margin: float = 5.0,
        rho_boundary_margin: float = 3.0,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or proto_dim <= 0 or num_classes <= 0 or K <= 0:
            raise ValueError("in_dim, proto_dim, num_classes and K must be positive.")
        self.in_dim = int(in_dim)
        self.proto_dim = int(proto_dim)
        self.num_classes = int(num_classes)
        self.K = int(K)
        self.rho_max = float(rho_max)
        self.delta_clip = float(delta_clip)
        self.proto_dropout = float(proto_dropout)
        self.logit_scale_min = float(logit_scale_min)
        self.logit_scale_max = float(logit_scale_max)
        self.dynamic_rho_cap = bool(dynamic_rho_cap)
        self.rho_easy_cap = float(rho_easy_cap)
        self.rho_boundary_cap = float(rho_boundary_cap)
        self.rho_easy_conf = float(rho_easy_conf)
        self.rho_easy_margin = float(rho_easy_margin)
        self.rho_boundary_margin = float(rho_boundary_margin)

        self.id_proj = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.proto_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(self.proto_dim, self.proto_dim),
            nn.LayerNorm(self.proto_dim),
        )
        self.skip_proj = nn.Linear(self.in_dim, self.proto_dim, bias=False)
        self.prototypes = nn.Parameter(torch.randn(self.num_classes, self.K, self.proto_dim) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init), dtype=torch.float32))
        self.gate_net = nn.Sequential(
            nn.LayerNorm(self.proto_dim + 2),
            nn.Linear(self.proto_dim + 2, int(gate_hidden_dim)),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(int(gate_hidden_dim), 1),
        )

    def set_rho_max(self, value: float) -> None:
        self.rho_max = float(value)

    def set_dynamic_rho_cap(self, enabled: bool) -> None:
        self.dynamic_rho_cap = bool(enabled)

    def set_stage3_gate_only(self, enabled: bool = True, *, train_logit_scale: bool = True) -> None:
        """Freeze prototype feature path in stabilization stage.

        This is intentionally coarse-grained: stage 3 should calibrate the gate
        and optionally the logit scale, not keep reshaping prototypes.
        """

        frozen_modules = [self.id_proj, self.skip_proj]
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = not enabled
        self.prototypes.requires_grad = not enabled
        self.logit_scale.requires_grad = (not enabled) or bool(train_logit_scale)
        for param in self.gate_net.parameters():
            param.requires_grad = True

    def _validate_inputs(self, z_id: torch.Tensor, base_logits: torch.Tensor) -> None:
        if z_id.dim() != 2 or z_id.size(1) != self.in_dim:
            raise ValueError(f"Expected z_id shaped [B,{self.in_dim}], got {tuple(z_id.shape)}.")
        if base_logits.dim() != 2 or base_logits.size(1) != self.num_classes:
            raise ValueError(
                f"Expected base_logits shaped [B,{self.num_classes}], got {tuple(base_logits.shape)}."
            )
        if z_id.size(0) != base_logits.size(0):
            raise ValueError("z_id and base_logits must have the same batch size.")

    def forward(self, z_id: torch.Tensor, base_logits: torch.Tensor) -> tuple[torch.Tensor, TensorDict]:
        self._validate_inputs(z_id, base_logits)

        h = self.id_proj(z_id.float()) + self.skip_proj(z_id.float())
        h = F.normalize(h, dim=-1, eps=1e-6)
        prototypes = F.normalize(self.prototypes.float(), dim=-1, eps=1e-6)

        sim = torch.einsum("bd,ckd->bck", h, prototypes)
        scale = self.logit_scale.clamp(self.logit_scale_min, self.logit_scale_max)
        sim = sim * scale

        if self.training and self.proto_dropout > 0.0 and self.K > 1:
            drop_mask = torch.rand(self.num_classes, self.K, device=sim.device) < self.proto_dropout
            all_drop = drop_mask.all(dim=1)
            if bool(all_drop.any()):
                drop_mask[all_drop, 0] = False
            sim = sim.masked_fill(drop_mask.unsqueeze(0), -1e4)

        proto_logits = torch.logsumexp(sim, dim=-1)

        with torch.no_grad():
            base_prob = base_logits.softmax(dim=-1)
            base_conf = base_prob.max(dim=-1).values
            if self.num_classes > 1:
                top2 = torch.topk(base_logits, k=2, dim=-1).values
                base_margin = top2[:, 0] - top2[:, 1]
            else:
                base_margin = torch.zeros_like(base_conf)

        gate_input = torch.cat([h, base_conf.unsqueeze(-1), base_margin.unsqueeze(-1)], dim=-1)
        rho_cap = h.new_full((h.size(0), 1), float(self.rho_max))
        if self.dynamic_rho_cap:
            easy = (base_conf >= self.rho_easy_conf) & (base_margin >= self.rho_easy_margin)
            boundary = base_margin <= self.rho_boundary_margin
            easy_cap = min(float(self.rho_easy_cap), float(self.rho_max))
            boundary_cap = min(float(self.rho_boundary_cap), float(self.rho_max))
            rho_cap = torch.where(easy.unsqueeze(-1), h.new_full((h.size(0), 1), easy_cap), rho_cap)
            rho_cap = torch.where(boundary.unsqueeze(-1), h.new_full((h.size(0), 1), boundary_cap), rho_cap)
        rho = rho_cap * torch.sigmoid(self.gate_net(gate_input))

        base_anchor = base_logits.detach()
        delta_logits = (proto_logits - base_anchor).clamp(min=-self.delta_clip, max=self.delta_clip)
        fused_logits = base_anchor + rho * delta_logits

        aux = {
            "proto_logits": proto_logits,
            "h": h,
            "prototypes": prototypes,
            "rho": rho,
            "rho_cap": rho_cap,
            "delta_logits": delta_logits,
            "delta": rho * delta_logits,
            "sim": sim,
            "proto_scores": sim,
            "base_conf": base_conf,
            "base_margin": base_margin,
            "logit_scale": scale.detach(),
            "z_joint": h,
            "logits": fused_logits,
            "fused_logits": fused_logits,
        }
        return fused_logits, aux


__all__ = ["SafeResidualProtoHead"]
