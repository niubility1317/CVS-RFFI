from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DAOTScheduleState:
    stage: str
    orbit_scale: float
    tangent_scale: float
    tail_scale: float


def _linear_ramp(epoch: int, start: int, end: int) -> float:
    if int(epoch) < int(start):
        return 0.0
    if int(end) <= int(start):
        return 1.0
    return min(1.0, max(0.0, (float(epoch) - float(start) + 1.0) / (float(end) - float(start) + 1.0)))


def adv3b02_daot_schedule(epoch: int, *, total_epochs: int = 200) -> DAOTScheduleState:
    if int(epoch) < 1 or int(epoch) > int(total_epochs):
        raise ValueError("epoch must be inside the configured training budget")
    if int(epoch) <= 20:
        return DAOTScheduleState("A", 0.0, 0.0, 0.0)
    if int(epoch) <= 60:
        return DAOTScheduleState("B", _linear_ramp(epoch, 21, 60), 0.0, 0.0)
    if int(epoch) <= 140:
        return DAOTScheduleState("C", 1.0, _linear_ramp(epoch, 61, 140), 0.0)
    return DAOTScheduleState("D", 1.0, 1.0, _linear_ramp(epoch, 141, int(total_epochs)))


def _masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum() / weights.expand_as(values).sum().clamp_min(1e-12)


def robust_spherical_orbit_target(
    features: torch.Tensor,
    *,
    reliability: torch.Tensor,
    importance: torch.Tensor,
    coverage_floor: float = 0.15,
    huber_beta_min: float = 0.30,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Estimate a reliability-weighted robust orbit center on the unit sphere."""

    if features.ndim != 3:
        raise ValueError("features must have shape [batch, views, embedding]")
    if reliability.shape != features.shape[:2] or importance.shape != features.shape[:2]:
        raise ValueError("reliability and importance must align with batch and views")
    gamma = float(coverage_floor)
    beta_floor = float(huber_beta_min)
    if not (0.0 <= gamma < 1.0):
        raise ValueError("coverage_floor must be in [0,1)")
    if not (0.0 < beta_floor <= 1.0):
        raise ValueError("huber_beta_min must be in (0,1]")
    normalized = F.normalize(features.float(), dim=-1)
    physical = torch.nan_to_num(reliability.float(), nan=0.0).clamp(0.0, 1.0)
    deployment = torch.nan_to_num(importance.float(), nan=0.0).clamp_min(0.0)
    base = gamma + (1.0 - gamma) * physical * deployment
    initial = F.normalize((base.unsqueeze(-1) * normalized).sum(dim=1), dim=-1)
    residual = torch.acos(
        (normalized * initial.unsqueeze(1)).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    )
    scale = residual.detach().median(dim=1, keepdim=True).values.clamp_min(1e-3)
    huber = torch.where(residual <= scale, torch.ones_like(residual), scale / residual.clamp_min(1e-6))
    huber = huber.clamp_min(beta_floor)
    raw_weights = base * huber
    weights = raw_weights / raw_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    target = F.normalize((weights.unsqueeze(-1) * normalized).sum(dim=1), dim=-1)
    dispersion = (weights * (1.0 - (normalized * target.unsqueeze(1)).sum(dim=-1))).sum(dim=1)
    effective_views = 1.0 / weights.square().sum(dim=1).clamp_min(1e-12)
    return target, weights, {
        "orbit_dispersion": dispersion,
        "effective_views": effective_views,
        "clean_weight": weights[:, 0],
    }


def orbit_feature_loss(
    student: torch.Tensor,
    target: torch.Tensor,
    *,
    recoverability: torch.Tensor,
) -> torch.Tensor:
    if student.shape != target.shape:
        raise ValueError("student and target features must have identical shape")
    per_row = 1.0 - (F.normalize(student.float(), dim=-1) * F.normalize(target.detach().float(), dim=-1)).sum(dim=-1)
    return _masked_mean(per_row, recoverability.float().clamp_min(0.0))


def orbit_logit_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    consensus: torch.Tensor,
    temperature: float = 3.0,
    confidence_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    teacher_prob = F.softmax(teacher_logits.detach().float() / temperature, dim=-1)
    per_row = F.kl_div(
        F.log_softmax(student_logits.float() / temperature, dim=-1),
        teacher_prob,
        reduction="none",
    ).sum(dim=-1) * (temperature**2)
    weights = consensus.to(device=per_row.device, dtype=per_row.dtype)
    if confidence_weight is not None:
        weights = weights * confidence_weight.to(device=per_row.device, dtype=per_row.dtype).clamp_min(0.0)
    return _masked_mean(per_row, weights)


def orbit_prototype_distillation_loss(
    student_similarity: torch.Tensor,
    teacher_similarity: torch.Tensor,
    *,
    consensus: torch.Tensor,
    temperature: float = 3.0,
) -> torch.Tensor:
    return orbit_logit_distillation_loss(
        student_similarity,
        teacher_similarity,
        consensus=consensus,
        temperature=temperature,
    )


def orbit_relation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    pairs: torch.Tensor,
) -> torch.Tensor:
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape [count,2]")
    student_n = F.normalize(student.float(), dim=-1)
    teacher_n = F.normalize(teacher.detach().float(), dim=-1)
    left, right = pairs[:, 0].long(), pairs[:, 1].long()
    student_rel = (student_n[left] * student_n[right]).sum(dim=-1)
    teacher_rel = (teacher_n[left] * teacher_n[right]).sum(dim=-1)
    return F.mse_loss(student_rel, teacher_rel)


class TemporalOrbitMemory:
    """Small keyed EMA memory used only by the A8 efficiency ablation."""

    def __init__(self, *, momentum: float = 0.85) -> None:
        if not (0.0 <= float(momentum) < 1.0):
            raise ValueError("memory momentum must be in [0,1)")
        self.momentum = float(momentum)
        self._features: dict[int, torch.Tensor] = {}

    def update(self, *, keys: torch.Tensor, features: torch.Tensor, valid: torch.Tensor) -> None:
        keys_cpu = keys.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        features_cpu = features.detach().to(device="cpu", dtype=torch.float32)
        valid_cpu = valid.detach().reshape(-1).to(device="cpu", dtype=torch.bool)
        if keys_cpu.numel() != features_cpu.shape[0] or valid_cpu.numel() != features_cpu.shape[0]:
            raise ValueError("memory keys, features, and validity must align")
        for key, feature, keep in zip(keys_cpu.tolist(), features_cpu, valid_cpu.tolist()):
            if not keep:
                continue
            old = self._features.get(int(key))
            self._features[int(key)] = feature.clone() if old is None else self.momentum * old + (1.0 - self.momentum) * feature

    def lookup(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = keys.detach().reshape(-1)
        stored = next(iter(self._features.values()), None)
        dim = int(stored.numel()) if stored is not None else 0
        values = torch.zeros((flat.numel(), dim), device=keys.device, dtype=torch.float32)
        found = torch.zeros(flat.numel(), device=keys.device, dtype=torch.bool)
        for index, key in enumerate(flat.to(device="cpu", dtype=torch.long).tolist()):
            feature = self._features.get(int(key))
            if feature is not None:
                values[index] = feature.to(device=keys.device)
                found[index] = True
        return values, found

    def state_dict(self) -> dict[str, Any]:
        keys = sorted(self._features)
        features = torch.stack([self._features[key] for key in keys], dim=0) if keys else torch.empty((0, 0))
        return {"momentum": self.momentum, "keys": torch.tensor(keys, dtype=torch.long), "features": features}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        momentum = float(state["momentum"])
        if not (0.0 <= momentum < 1.0):
            raise ValueError("memory momentum must be in [0,1)")
        keys = torch.as_tensor(state["keys"], dtype=torch.long).reshape(-1)
        features = torch.as_tensor(state["features"], dtype=torch.float32)
        if keys.numel() != features.shape[0]:
            raise ValueError("memory state keys and features must align")
        self.momentum = momentum
        self._features = {int(key): feature.clone() for key, feature in zip(keys.tolist(), features)}


class EMALossScaleNormalizer:
    """Normalize heterogeneous DAOT auxiliaries without changing their gradients."""

    def __init__(self, *, momentum: float = 0.95, epsilon: float = 1e-6) -> None:
        if not (0.0 <= float(momentum) < 1.0):
            raise ValueError("loss-scale momentum must be in [0,1)")
        if float(epsilon) <= 0.0:
            raise ValueError("loss-scale epsilon must be positive")
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self._scales: dict[str, float] = {}

    def normalize(
        self,
        components: Mapping[str, torch.Tensor],
        *,
        active: Optional[Mapping[str, bool]] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        normalized: dict[str, torch.Tensor] = {}
        reported: dict[str, float] = {}
        for name, loss in components.items():
            enabled = True if active is None else bool(active.get(name, False))
            current = float(loss.detach().float().abs().cpu().item())
            old = self._scales.get(str(name))
            if enabled and current > self.epsilon:
                scale = current if old is None else self.momentum * old + (1.0 - self.momentum) * current
                self._scales[str(name)] = float(scale)
            else:
                scale = old if old is not None else 1.0
            normalized[str(name)] = loss / max(float(scale), self.epsilon)
            reported[str(name)] = float(scale)
        return normalized, reported

    def state_dict(self) -> dict[str, Any]:
        return {
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "scales": dict(sorted(self._scales.items())),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        momentum = float(state["momentum"])
        epsilon = float(state["epsilon"])
        if not (0.0 <= momentum < 1.0):
            raise ValueError("loss-scale momentum must be in [0,1)")
        if epsilon <= 0.0:
            raise ValueError("loss-scale epsilon must be positive")
        scales = {str(name): float(value) for name, value in dict(state.get("scales", {})).items()}
        if any((not torch.isfinite(torch.tensor(value))) or value <= 0.0 for value in scales.values()):
            raise ValueError("loss scales must be finite and positive")
        self.momentum = momentum
        self.epsilon = epsilon
        self._scales = scales
