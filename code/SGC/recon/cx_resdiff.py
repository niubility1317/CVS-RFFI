from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_ops import apply_bounded_residual, residual_ratio
from .condition_encoder import PhyConditionEncoder, normalize_sat_meta
from .cx_unet_1d import CxResUNet1D
from .residual_gate import ResidualSafetyGate


class CosineNoiseSchedule(nn.Module):
    def __init__(self, timesteps: int = 1000, s: float = 0.008) -> None:
        super().__init__()
        self.timesteps = int(timesteps)
        steps = torch.arange(self.timesteps + 1, dtype=torch.float32)
        f = torch.cos(((steps / self.timesteps) + float(s)) / (1.0 + float(s)) * math.pi * 0.5).square()
        alpha_bar = f / f[0].clamp_min(1e-12)
        self.register_buffer("alpha_bar", alpha_bar.clamp(1e-6, 1.0))

    def alpha_sigma(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = t.long().clamp(0, self.timesteps)
        alpha_bar = self.alpha_bar.to(device=t.device)[idx]
        return torch.sqrt(alpha_bar), torch.sqrt((1.0 - alpha_bar).clamp_min(1e-8))


def sample_timesteps(batch_size: int, timesteps: int, device) -> torch.Tensor:
    return torch.randint(0, int(timesteps), (int(batch_size),), device=device)


def q_sample(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, schedule: CosineNoiseSchedule) -> torch.Tensor:
    alpha, sigma = schedule.alpha_sigma(t)
    return alpha.view(-1, 1, 1) * x0 + sigma.view(-1, 1, 1) * noise


def target_v(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, schedule: CosineNoiseSchedule) -> torch.Tensor:
    alpha, sigma = schedule.alpha_sigma(t)
    return alpha.view(-1, 1, 1) * noise - sigma.view(-1, 1, 1) * x0


def v_prediction_loss(pred_v: torch.Tensor, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, schedule: CosineNoiseSchedule) -> torch.Tensor:
    return F.mse_loss(pred_v, target_v(x0, noise, t, schedule))


class CxResDiff(nn.Module):
    def __init__(
        self,
        *,
        model: CxResUNet1D | None = None,
        condition_encoder: PhyConditionEncoder | None = None,
        residual_gate: ResidualSafetyGate | None = None,
        train_timesteps: int = 1000,
        condition_dim: int = 24,
    ) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.model = model if model is not None else CxResUNet1D(condition_dim=self.condition_dim)
        self.condition_encoder = condition_encoder if condition_encoder is not None else PhyConditionEncoder(out_dim=self.condition_dim)
        self.residual_gate = residual_gate if residual_gate is not None else ResidualSafetyGate(cond_dim=self.condition_dim)
        self.schedule = CosineNoiseSchedule(timesteps=int(train_timesteps))

    def encode_condition(self, y: torch.Tensor, meta: Mapping[str, object] | None = None) -> torch.Tensor:
        if meta is None:
            return self.condition_encoder.from_iq_proxy(y)
        return self.condition_encoder(normalize_sat_meta(meta, device=y.device))

    def forward(self, *, x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.model(x_t=x_t, y=y, t=t, c=c)

    def diffusion_loss(self, x_clean: torch.Tensor, y_sat: torch.Tensor, c: torch.Tensor, t: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if t is None:
            t = sample_timesteps(x_clean.size(0), self.schedule.timesteps, x_clean.device)
        noise = torch.randn_like(x_clean)
        x_t = q_sample(x_clean, noise, t, self.schedule)
        pred_v = self.forward(x_t=x_t, y=y_sat, t=t, c=c)
        loss = v_prediction_loss(pred_v, x_clean, noise, t, self.schedule)
        return loss, {"loss_diff": loss.detach()}

    @torch.no_grad()
    def _default_t(self, y: torch.Tensor) -> torch.Tensor:
        return torch.full((y.size(0),), max(1, self.schedule.timesteps // 4), device=y.device, dtype=torch.long)

    def correct(
        self,
        y: torch.Tensor,
        *,
        meta: Mapping[str, object] | None = None,
        c: torch.Tensor | None = None,
        rho: float = 0.15,
        t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        c = c if c is not None else self.encode_condition(y, meta)
        t = t if t is not None else self._default_t(y)
        delta = self.model(x_t=y, y=y, t=t, c=c)
        gate = self.residual_gate(y, c)
        x_hat = apply_bounded_residual(y, delta, gate, rho=float(rho))
        ratio = residual_ratio(x_hat, y)
        return {"x_hat": x_hat, "delta": delta, "gate": gate, "condition": c, "residual_ratio": ratio}
