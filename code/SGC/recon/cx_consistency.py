from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from .complex_ops import apply_bounded_residual, residual_ratio
from .condition_encoder import PhyConditionEncoder, normalize_sat_meta
from .cx_unet_1d import CxResUNet1D
from .residual_gate import ResidualSafetyGate


def pseudo_huber_loss(input: torch.Tensor, target: torch.Tensor, c: float = 0.03) -> torch.Tensor:
    diff = input - target
    c_t = input.new_tensor(float(c))
    return (torch.sqrt(diff.square() + c_t.square()) - c_t).mean()


class CxConsistency(nn.Module):
    def __init__(
        self,
        *,
        model: CxResUNet1D | None = None,
        condition_encoder: PhyConditionEncoder | None = None,
        residual_gate: ResidualSafetyGate | None = None,
        condition_dim: int = 24,
        train_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.train_timesteps = int(train_timesteps)
        self.model = model if model is not None else CxResUNet1D(condition_dim=self.condition_dim)
        self.condition_encoder = condition_encoder if condition_encoder is not None else PhyConditionEncoder(out_dim=self.condition_dim)
        self.residual_gate = residual_gate if residual_gate is not None else ResidualSafetyGate(cond_dim=self.condition_dim)

    def encode_condition(self, y: torch.Tensor, meta: Mapping[str, object] | None = None) -> torch.Tensor:
        if meta is None:
            return self.condition_encoder.from_iq_proxy(y)
        return self.condition_encoder(normalize_sat_meta(meta, device=y.device))

    def forward(self, *, x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.model(x_t=x_t, y=y, t=t, c=c)

    def correct(
        self,
        y: torch.Tensor,
        *,
        meta: Mapping[str, object] | None = None,
        c: torch.Tensor | None = None,
        steps: int = 2,
        rho: float = 0.15,
    ) -> dict[str, torch.Tensor]:
        c = c if c is not None else self.encode_condition(y, meta)
        steps = int(max(1, steps))
        x_t = y
        delta = None
        for step in range(steps):
            frac = 1.0 - (step / max(1, steps))
            t_value = int(max(1, round(frac * self.train_timesteps)))
            t = torch.full((y.size(0),), t_value, device=y.device, dtype=torch.long)
            delta = self.model(x_t=x_t, y=y, t=t, c=c)
            interim_gate = self.residual_gate(y, c)
            x_t = apply_bounded_residual(y, delta, interim_gate, rho=float(rho))
        gate = self.residual_gate(y, c)
        x_hat = apply_bounded_residual(y, delta if delta is not None else torch.zeros_like(y), gate, rho=float(rho))
        ratio = residual_ratio(x_hat, y)
        return {"x_hat": x_hat, "delta": delta, "gate": gate, "condition": c, "residual_ratio": ratio}

    def consistency_loss(self, online: torch.Tensor, target: torch.Tensor, *, loss_type: str = "pseudo_huber") -> torch.Tensor:
        if str(loss_type).lower() == "mse":
            return torch.mean((online - target.detach()).square())
        return pseudo_huber_loss(online, target.detach())
