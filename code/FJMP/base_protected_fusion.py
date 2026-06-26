"""Base-protected safe fusion for SGV-BP-FJMP."""

from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn as nn

from .logit_calibration import CenteredTemperatureCalibrator, calibrated_delta


def stage_rho_max(epoch: int, *, stage1: float = 0.10, stage2: float = 0.25, stage3: float = 0.30) -> float:
    """Documented rho cap schedule: E1-E5, E6-E15, E16+."""

    epoch = int(epoch)
    if epoch <= 5:
        return float(stage1)
    if epoch <= 15:
        return float(stage2)
    return float(stage3)


class BaseProtectedFusion(nn.Module):
    """Compute ``safe_logits`` from frozen base logits and FJMP head logits."""

    def __init__(
        self,
        num_classes: int,
        gate_input_dim: int = 0,
        *,
        rho_init: float = 0.03,
        rho_max: float = 0.30,
        max_delta_norm: float = 3.0,
        learn_temperature: bool = True,
        temperature_range: tuple[float, float] = (0.5, 5.0),
    ) -> None:
        super().__init__()
        if int(num_classes) <= 0:
            raise ValueError("num_classes must be positive.")
        self.num_classes = int(num_classes)
        self.rho_max = float(rho_max)
        self.max_delta_norm = float(max_delta_norm)
        self.head_calibrator = CenteredTemperatureCalibrator(1.0, learn_temperature, temperature_range)
        self.base_calibrator = CenteredTemperatureCalibrator(1.0, learn_temperature, temperature_range)
        if int(gate_input_dim) > 0:
            hidden = max(8, min(64, int(gate_input_dim) * 2))
            self.gate = nn.Sequential(
                nn.Linear(int(gate_input_dim), hidden),
                nn.SiLU(inplace=True),
                nn.Linear(hidden, 1),
            )
        else:
            self.gate = None
        init_ratio = min(max(float(rho_init) / max(self.rho_max, 1e-8), 1e-6), 1.0 - 1e-6)
        self.raw_rho_bias = nn.Parameter(torch.logit(torch.tensor(init_ratio, dtype=torch.float32)))

    def alpha(self) -> torch.Tensor:
        return self.current_rho_max_tensor()

    def beta(self) -> torch.Tensor:
        return torch.tensor(1.0, device=self.raw_rho_bias.device)

    def eta(self) -> torch.Tensor:
        return self.current_rho_max_tensor()

    def current_rho_max_tensor(self) -> torch.Tensor:
        return torch.tensor(float(self.rho_max), device=self.raw_rho_bias.device, dtype=self.raw_rho_bias.dtype)

    def set_rho_max(self, rho_max: float) -> None:
        self.rho_max = float(rho_max)

    def forward(
        self,
        *,
        base_logits: torch.Tensor,
        head_logits: Optional[torch.Tensor] = None,
        proto_logits: Optional[torch.Tensor] = None,
        gate_input: Optional[torch.Tensor] = None,
        accept_proto: Optional[torch.Tensor] = None,
        rho_max: Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        if head_logits is None:
            if proto_logits is None:
                raise ValueError("head_logits or proto_logits is required.")
            head_logits = proto_logits
        cap = self.current_rho_max_tensor() if rho_max is None else torch.tensor(float(rho_max), device=base_logits.device)
        delta, head_cal, base_cal = calibrated_delta(
            head_logits,
            base_logits,
            self.head_calibrator,
            self.base_calibrator,
            max_delta_norm=self.max_delta_norm,
        )
        if self.gate is not None and gate_input is not None:
            gate_logit = self.gate(gate_input.float()).view(-1, 1)
        else:
            gate_logit = self.raw_rho_bias.to(device=base_logits.device, dtype=base_logits.dtype).expand(base_logits.size(0), 1)
        gate_logit = torch.nan_to_num(gate_logit, nan=-20.0, posinf=20.0, neginf=-20.0)
        rho = cap.to(device=base_logits.device, dtype=base_logits.dtype) * torch.sigmoid(gate_logit)
        rho = torch.nan_to_num(rho, nan=0.0, posinf=float(cap), neginf=0.0)
        safe_logits = base_logits.float() + rho * delta
        safe_logits = torch.where(torch.isfinite(safe_logits), safe_logits, base_logits.float())
        if accept_proto is not None:
            mask = accept_proto.to(device=base_logits.device, dtype=torch.bool).view(-1, 1)
            safe_logits = torch.where(mask, safe_logits, base_logits.float())
            rho = torch.where(mask, rho, torch.zeros_like(rho))
        return {
            "logits": safe_logits,
            "safe_logits": safe_logits,
            "head_logits": head_logits,
            "base_logits": base_logits,
            "head_calibrated": head_cal,
            "base_calibrated": base_cal,
            "delta": delta,
            "delta_norm": delta.norm(dim=1),
            "gate": torch.sigmoid(gate_logit).view(-1),
            "rho": rho.view(-1),
            "rho_max": cap.detach(),
            "T_head": self.head_calibrator.temperature().detach(),
            "T_base": self.base_calibrator.temperature().detach(),
            "eta": cap.detach(),
            "alpha": cap.detach(),
            "beta": torch.tensor(1.0, device=base_logits.device),
        }


__all__ = ["BaseProtectedFusion", "stage_rho_max"]
