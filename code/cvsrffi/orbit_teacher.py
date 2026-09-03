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


@dataclass(frozen=True)
class DAOTRXV2ScheduleState:
    stage: str
    scales: Mapping[str, float]


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


def adv3b02_daot_rx_v2_schedule(epoch: int, *, total_epochs: int = 200) -> DAOTRXV2ScheduleState:
    if int(epoch) < 1 or int(epoch) > int(total_epochs):
        raise ValueError("epoch must be inside the configured training budget")
    scales = {
        "base": 1.0,
        "orbit_feature": _linear_ramp(epoch, 10, 30),
        "soft": _linear_ramp(epoch, 25, 50),
        "rx": _linear_ramp(epoch, 25, 50),
        "tangent": _linear_ramp(epoch, 40, 70),
        "route": _linear_ramp(epoch, 55, 80),
        "tail": _linear_ramp(epoch, 70, int(total_epochs)),
    }
    active = [name for name, scale in scales.items() if scale > 0.0]
    return DAOTRXV2ScheduleState(active[-1], scales)


def _masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum() / weights.expand_as(values).sum().clamp_min(1e-12)


def coverage_mixture_weights(
    recoverability: torch.Tensor,
    deployment_weight: torch.Tensor,
    *,
    prior: torch.Tensor,
    coverage_floor: float,
) -> torch.Tensor:
    """Mix evidence weights with an explicit deployment-view coverage prior."""

    if recoverability.shape != deployment_weight.shape:
        raise ValueError("recoverability and deployment_weight must align")
    if recoverability.ndim != 2:
        raise ValueError("coverage weights require [batch,views] inputs")
    gamma = float(coverage_floor)
    if not 0.0 <= gamma < 1.0:
        raise ValueError("coverage_floor must be in [0,1)")
    prior = torch.as_tensor(prior, device=recoverability.device, dtype=torch.float32).reshape(-1)
    if prior.numel() != recoverability.shape[1] or bool((prior < 0.0).any()) or float(prior.sum()) <= 0.0:
        raise ValueError("prior must provide non-negative mass for every view")
    prior = prior / prior.sum()
    evidence = torch.nan_to_num(recoverability.float(), nan=0.0).clamp_min(0.0)
    evidence = evidence * torch.nan_to_num(deployment_weight.float(), nan=0.0).clamp_min(0.0)
    evidence_mass = evidence.sum(dim=1, keepdim=True)
    normalized = evidence / evidence_mass.clamp_min(1e-12)
    normalized = torch.where(evidence_mass > 0.0, normalized, prior.unsqueeze(0).expand_as(evidence))
    return (1.0 - gamma) * normalized + gamma * prior.unsqueeze(0)


def _sphere_log_map(base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    base = F.normalize(base.float(), dim=-1)
    point = F.normalize(point.float(), dim=-1)
    cosine = (base * point).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.acos(cosine)
    tangent = point - cosine * base
    return tangent * (angle / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-6))


def _sphere_exp_map(base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    base = F.normalize(base.float(), dim=-1)
    length = tangent.norm(dim=-1, keepdim=True)
    direction = tangent / length.clamp_min(1e-6)
    moved = torch.cos(length) * base + torch.sin(length) * direction
    return F.normalize(torch.where(length > 1e-6, moved, base), dim=-1)


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


def anchored_spherical_orbit_target(
    features: torch.Tensor,
    *,
    reliability: torch.Tensor,
    importance: torch.Tensor,
    prior: torch.Tensor,
    coverage_floor: float = 0.15,
    huber_beta_min: float = 0.30,
    anchor_strength: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Estimate the orbit center in the clean teacher's tangent space."""

    anchor_strength = float(anchor_strength)
    if not 0.0 <= anchor_strength <= 1.0:
        raise ValueError("anchor_strength must be in [0,1]")
    initial_weights = coverage_mixture_weights(
        reliability,
        importance,
        prior=prior,
        coverage_floor=coverage_floor,
    )
    target, _, diagnostics = robust_spherical_orbit_target(
        features,
        reliability=initial_weights,
        importance=torch.ones_like(initial_weights),
        coverage_floor=0.0,
        huber_beta_min=huber_beta_min,
    )
    clean = F.normalize(features[:, 0].float(), dim=-1)
    dispersion = diagnostics["orbit_dispersion"].detach().clamp(0.0, 1.0)
    travel = (1.0 - anchor_strength * dispersion).clamp(1.0 - anchor_strength, 1.0)
    anchored = _sphere_exp_map(clean, travel.unsqueeze(-1) * _sphere_log_map(clean, target))
    diagnostics = dict(diagnostics)
    diagnostics["anchor_travel_fraction"] = travel
    diagnostics["coverage_entropy"] = -(
        initial_weights * initial_weights.clamp_min(1e-12).log()
    ).sum(dim=1)
    diagnostics["clean_weight"] = initial_weights[:, 0]
    return anchored, initial_weights, diagnostics


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


class TensorTemporalOrbitMemory:
    """Bounded tensor-backed orbit memory with reliability-aware EMA and TTL."""

    def __init__(
        self,
        *,
        feature_dim: int,
        capacity: int,
        base_momentum: float = 0.85,
        ttl: int = 64,
    ) -> None:
        if int(feature_dim) <= 0 or int(capacity) <= 0:
            raise ValueError("feature_dim and capacity must be positive")
        if not 0.0 <= float(base_momentum) < 1.0:
            raise ValueError("base_momentum must be in [0,1)")
        if int(ttl) < 0:
            raise ValueError("ttl must be non-negative")
        self.feature_dim = int(feature_dim)
        self.capacity = int(capacity)
        self.base_momentum = float(base_momentum)
        self.ttl = int(ttl)
        self.keys = torch.full((self.capacity,), -1, dtype=torch.long)
        self.features = torch.zeros((self.capacity, self.feature_dim), dtype=torch.float32)
        self.reliability = torch.zeros(self.capacity, dtype=torch.float32)
        self.scenario_bin = torch.full((self.capacity,), -1, dtype=torch.long)
        self.receiver_bin = torch.full((self.capacity,), -1, dtype=torch.long)
        self.last_seen = torch.full((self.capacity,), -1, dtype=torch.long)
        self._next_slot = 0

    def _slot_for_update(self, key: int) -> int:
        matches = torch.nonzero(self.keys == int(key), as_tuple=False).reshape(-1)
        if matches.numel():
            return int(matches[0])
        free = torch.nonzero(self.keys < 0, as_tuple=False).reshape(-1)
        if free.numel():
            return int(free[0])
        slot = int(self._next_slot % self.capacity)
        self._next_slot = (slot + 1) % self.capacity
        return slot

    def update(
        self,
        *,
        keys: torch.Tensor,
        features: torch.Tensor,
        reliability: torch.Tensor,
        scenario_bin: torch.Tensor,
        receiver_bin: torch.Tensor,
        step: int,
    ) -> None:
        flat_keys = keys.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        values = features.detach().to(device="cpu", dtype=torch.float32)
        confidence = reliability.detach().reshape(-1).to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
        scenarios = scenario_bin.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        receivers = receiver_bin.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        count = int(flat_keys.numel())
        if values.shape != (count, self.feature_dim):
            raise ValueError("memory features must have shape [count,feature_dim]")
        if confidence.numel() != count or scenarios.numel() != count or receivers.numel() != count:
            raise ValueError("memory metadata must align with keys")
        values = F.normalize(values, dim=-1)
        for index, key in enumerate(flat_keys.tolist()):
            slot = self._slot_for_update(int(key))
            is_new = int(self.keys[slot]) != int(key)
            if is_new:
                merged = values[index]
            else:
                momentum = self.base_momentum * float(confidence[index])
                merged = momentum * self.features[slot] + (1.0 - momentum) * values[index]
            self.keys[slot] = int(key)
            self.features[slot] = F.normalize(merged.reshape(1, -1), dim=-1).reshape(-1)
            self.reliability[slot] = confidence[index]
            self.scenario_bin[slot] = scenarios[index]
            self.receiver_bin[slot] = receivers[index]
            self.last_seen[slot] = int(step)

    def lookup(
        self,
        keys: torch.Tensor,
        *,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        requested = keys.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        values = torch.zeros((requested.numel(), self.feature_dim), dtype=torch.float32)
        found = torch.zeros(requested.numel(), dtype=torch.bool)
        metadata = {
            "reliability": torch.zeros(requested.numel(), dtype=torch.float32),
            "scenario_bin": torch.full((requested.numel(),), -1, dtype=torch.long),
            "receiver_bin": torch.full((requested.numel(),), -1, dtype=torch.long),
            "last_seen": torch.full((requested.numel(),), -1, dtype=torch.long),
        }
        for index, key in enumerate(requested.tolist()):
            matches = torch.nonzero(self.keys == int(key), as_tuple=False).reshape(-1)
            if not matches.numel():
                continue
            slot = int(matches[0])
            if int(step) - int(self.last_seen[slot]) > self.ttl:
                continue
            found[index] = True
            values[index] = self.features[slot]
            for name in metadata:
                metadata[name][index] = getattr(self, name)[slot]
        device = keys.device
        return values.to(device), found.to(device), {name: value.to(device) for name, value in metadata.items()}

    def state_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "capacity": self.capacity,
            "base_momentum": self.base_momentum,
            "ttl": self.ttl,
            "keys": self.keys.clone(),
            "features": self.features.clone(),
            "reliability": self.reliability.clone(),
            "scenario_bin": self.scenario_bin.clone(),
            "receiver_bin": self.receiver_bin.clone(),
            "last_seen": self.last_seen.clone(),
            "next_slot": self._next_slot,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["feature_dim"]) != self.feature_dim or int(state["capacity"]) != self.capacity:
            raise ValueError("memory state dimensions do not match")
        for name in ("keys", "features", "reliability", "scenario_bin", "receiver_bin", "last_seen"):
            target = getattr(self, name)
            value = torch.as_tensor(state[name], dtype=target.dtype)
            if value.shape != target.shape:
                raise ValueError(f"memory state field {name} has the wrong shape")
            target.copy_(value)
        self.base_momentum = float(state["base_momentum"])
        self.ttl = int(state["ttl"])
        self._next_slot = int(state.get("next_slot", 0)) % self.capacity


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
