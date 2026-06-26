"""Logit calibration helpers for SGV-BP-FJMP."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def center_logits(logits: torch.Tensor) -> torch.Tensor:
    """Remove each sample's mean logit before comparing different heads."""

    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return logits - logits.mean(dim=1, keepdim=True)


def clip_delta_norm(delta: torch.Tensor, max_delta_norm: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """Clip per-sample delta vectors by L2 norm without changing direction."""

    max_norm = float(max_delta_norm)
    if max_norm <= 0:
        return torch.zeros_like(delta)
    norm = delta.float().norm(dim=1, keepdim=True).clamp_min(float(eps))
    scale = (max_norm / norm).clamp(max=1.0)
    return delta * scale.to(dtype=delta.dtype)


class CenteredTemperatureCalibrator(nn.Module):
    """Centered temperature calibration with an optional learnable temperature."""

    def __init__(
        self,
        temperature_init: float = 1.0,
        learn_temperature: bool = True,
        temperature_range: Tuple[float, float] = (0.5, 5.0),
    ) -> None:
        super().__init__()
        lo, hi = float(temperature_range[0]), float(temperature_range[1])
        if lo <= 0 or hi <= lo:
            raise ValueError("temperature_range must be positive and increasing.")
        self.temperature_min = lo
        self.temperature_max = hi
        init = min(max(float(temperature_init), lo), hi)
        ratio = (init - lo) / max(hi - lo, 1e-8)
        ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
        raw = math.log(ratio / (1.0 - ratio))
        if bool(learn_temperature):
            self.raw_temperature = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        else:
            self.register_buffer("raw_temperature", torch.tensor(raw, dtype=torch.float32))
        self.learn_temperature = bool(learn_temperature)

    def temperature(self) -> torch.Tensor:
        raw = torch.nan_to_num(self.raw_temperature, nan=0.0, posinf=8.0, neginf=-8.0)
        return self.temperature_min + (self.temperature_max - self.temperature_min) * torch.sigmoid(raw)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return center_logits(logits) / self.temperature().to(device=logits.device, dtype=logits.dtype).clamp_min(1e-6)


def calibrated_delta(
    head_logits: torch.Tensor,
    base_logits: torch.Tensor,
    head_calibrator: CenteredTemperatureCalibrator,
    base_calibrator: CenteredTemperatureCalibrator,
    max_delta_norm: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return clipped calibrated residual and both calibrated logit tensors."""

    head_cal = head_calibrator(head_logits)
    base_cal = base_calibrator(base_logits.detach())
    delta = clip_delta_norm(head_cal - base_cal.detach(), max_delta_norm=max_delta_norm)
    return delta, head_cal, base_cal


__all__ = [
    "CenteredTemperatureCalibrator",
    "calibrated_delta",
    "center_logits",
    "clip_delta_norm",
]
