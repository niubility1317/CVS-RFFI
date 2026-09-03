from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from .phase1_fcr_types import FCRDecodeOutput, FCRV2CapabilityState, FCRV2FactorOutput, FCRV2LossOutput
from .phase1_fcr_v2_schedule import BASE_LOSS_WEIGHTS, FCRV2Schedule
from .phase1_fcr_v2_physics import complex_gram


class LossMagnitudeEMA:
    """Persistent source-only loss-scale normalizer."""

    def __init__(self, decay: float = 0.95, eps: float = 1.0e-6) -> None:
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.eps = float(eps)
        self.magnitudes: dict[str, float] = {}
        self.updates: dict[str, int] = {}

    def normalize(self, name: str, value: torch.Tensor) -> torch.Tensor:
        magnitude = float(value.detach().abs().mean().cpu())
        if not torch.isfinite(torch.tensor(magnitude)):
            raise ValueError(f"non-finite loss magnitude for {name}")
        previous = self.magnitudes.get(str(name))
        current = magnitude if previous is None else self.decay * previous + (1.0 - self.decay) * magnitude
        self.magnitudes[str(name)] = max(float(current), self.eps)
        self.updates[str(name)] = self.updates.get(str(name), 0) + 1
        return value / self.magnitudes[str(name)]

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "eps": self.eps,
            "magnitudes": dict(self.magnitudes),
            "updates": dict(self.updates),
        }


def asymmetric_clean_teacher_loss(clean_teacher: torch.Tensor, leo_student: torch.Tensor) -> torch.Tensor:
    if clean_teacher.shape != leo_student.shape:
        raise ValueError("clean teacher and LEO student features must be shape matched")
    return F.mse_loss(leo_student, clean_teacher.detach())


def per_class_tail_cvar_loss(
    per_sample_loss: torch.Tensor,
    labels: torch.Tensor,
    *,
    tail_fraction: float = 0.25,
) -> torch.Tensor:
    losses = per_sample_loss.reshape(-1)
    target = labels.to(device=losses.device, dtype=torch.long).reshape(-1)
    if losses.numel() != target.numel():
        raise ValueError("per-sample loss and labels must align")
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    class_tails: list[torch.Tensor] = []
    for label in torch.unique(target[target >= 0], sorted=True):
        values = losses[target == label]
        if values.numel() == 0:
            continue
        count = max(1, int(torch.ceil(values.new_tensor(values.numel() * float(tail_fraction))).item()))
        class_tails.append(values.topk(count, largest=True).values.mean())
    if not class_tails:
        return losses.sum() * 0.0
    return torch.stack(class_tails).max()


def _complex_physical_features(signal: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(signal) or signal.ndim != 2:
        raise ValueError("physical feature input must be complex [B,T]")
    delayed = torch.roll(signal, shifts=1, dims=-1)
    derivative = signal - delayed
    return torch.stack((signal, signal.conj(), delayed, derivative), dim=-1)


def complex_physical_gram_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("complex Gram inputs must be shape matched")
    delta = complex_gram(_complex_physical_features(left)) - complex_gram(_complex_physical_features(right))
    return delta.abs().square().mean()


def response_surface_smoothness(delta_f: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(delta_f) or delta_f.ndim != 2:
        raise ValueError("response surface must be complex [B,T]")
    if delta_f.size(-1) < 3:
        return delta_f.abs().square().mean() * 0.0
    second = delta_f[:, 2:] - 2.0 * delta_f[:, 1:-1] + delta_f[:, :-2]
    return second.abs().square().mean()


def fingerprint_separation_loss(
    source: torch.Tensor,
    destination: torch.Tensor,
    *,
    cosine_margin: float = 0.2,
) -> torch.Tensor:
    if source.shape != destination.shape:
        raise ValueError("fingerprint features must be shape matched")
    cosine = F.cosine_similarity(source, destination, dim=-1)
    # Different-TX pairs are valid only when cosine similarity is below the
    # positive margin.  Minimization therefore separates, never aligns, them.
    return F.relu(cosine - float(cosine_margin)).mean()


def nuisance_cycle_loss(
    recycled: FCRV2FactorOutput,
    destination: FCRV2FactorOutput,
) -> torch.Tensor:
    losses = []
    for name in sorted(destination.z_n):
        left = recycled.z_n[name]
        right = destination.z_n[name].detach()
        losses.append((left - right).abs().square().mean())
    if not losses:
        return recycled.z_f_id.sum() * 0.0
    return torch.stack(losses).mean()


def cross_decode(
    source: FCRV2FactorOutput,
    destination: FCRV2FactorOutput,
    decoder,
) -> FCRDecodeOutput:
    return decoder(source.s_hat, destination.delta_f, destination.z_n)


def necessity_loss(
    *,
    full_error: torch.Tensor,
    drop_error: torch.Tensor,
    relative_margin: float = 0.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    baseline = full_error.detach().clamp_min(float(eps))
    relative_gap = (drop_error - baseline) / baseline
    return (float(relative_margin) - relative_gap).clamp_min(0.0)


def _extract_components(inputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    payload = inputs.get("components", inputs)
    components: dict[str, torch.Tensor] = {}
    for name in BASE_LOSS_WEIGHTS:
        value = payload.get(name)
        if value is None:
            continue
        components[name] = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
    if not components:
        raise ValueError("compute_fcr_v2_losses requires at least one loss component tensor")
    return components


def _normalize(name: str, value: torch.Tensor, ema_normalizer) -> torch.Tensor:
    if ema_normalizer is None:
        return value
    if hasattr(ema_normalizer, "normalize"):
        normalized = ema_normalizer.normalize(name, value)
    elif hasattr(ema_normalizer, "value"):
        scale = float(ema_normalizer.value(name, fallback=float(value.detach().abs().mean().cpu())))
        normalized = value / max(scale, 1e-6)
    elif callable(ema_normalizer):
        normalized = ema_normalizer(name, value)
    else:
        raise TypeError("ema_normalizer must be callable or expose normalize/value")
    return normalized if isinstance(normalized, torch.Tensor) else torch.as_tensor(normalized, device=value.device)


def _zero_like(components: Mapping[str, torch.Tensor]) -> torch.Tensor:
    first = next(iter(components.values()))
    return first.reshape(-1).sum() * 0.0


def _default_capabilities() -> FCRV2CapabilityState:
    return FCRV2CapabilityState(
        eta_ready=True,
        decoder_ready=True,
        swap_ready=True,
        fingerprint_ready=True,
        reasons={},
    )


def compute_fcr_v2_losses(
    inputs: Mapping[str, Any],
    row: str,
    ema_normalizer,
) -> FCRV2LossOutput:
    components = _extract_components(inputs)
    zero = _zero_like(components)
    role = str(inputs.get("role", "L_s"))
    epoch = int(inputs.get("epoch", 1))
    capabilities = inputs.get("capabilities") or _default_capabilities()
    if not isinstance(capabilities, FCRV2CapabilityState):
        raise TypeError("capabilities must be an FCRV2CapabilityState")
    schedule = inputs.get("schedule") or FCRV2Schedule()
    if not isinstance(schedule, FCRV2Schedule):
        raise TypeError("schedule must be an FCRV2Schedule")
    stage = schedule.state(epoch=epoch, row=row, capabilities=capabilities)

    weighted_components: dict[str, torch.Tensor] = {}
    weights: dict[str, float] = {}
    metrics: dict[str, float | str] = {}

    normalized_supervised_scale = 1.0 if role.strip().lower() in {"l_s", "labeled", "source_labeled"} else 0.0
    for name in ("identity_ce", "prototype", "tail"):
        raw = components.get(name, zero)
        weight = BASE_LOSS_WEIGHTS[name] * normalized_supervised_scale
        normalized = _normalize(name, raw, ema_normalizer) if weight > 0.0 else raw
        weights[name] = float(weight)
        weighted_components[name] = normalized * weight
        metrics[f"{name}_normalized"] = float(normalized.detach().cpu())

    unlabeled_scale = schedule.unlabeled_fcr_weight if role.strip().lower() in {"u_s", "unlabeled", "source_unlabeled"} else 1.0
    if normalized_supervised_scale == 0.0 and unlabeled_scale == 1.0 and role.strip().lower() not in {"v", "validation", "source_validation"}:
        unlabeled_scale = 0.0

    for name in schedule.base_weights:
        if name in {"identity_ce", "prototype", "tail"}:
            continue
        raw = components.get(name, zero)
        base_weight = schedule.base_weights[name]
        stage_scale = float(stage.scales.get(name, 0.0))
        effective = base_weight * stage_scale * unlabeled_scale
        if role.strip().lower() in {"l_s", "labeled", "source_labeled"}:
            effective = base_weight * stage_scale
        if name not in stage.active_losses:
            effective = 0.0
        normalized = _normalize(name, raw, ema_normalizer) if effective > 0.0 else raw
        weights[name] = float(effective)
        weighted_components[name] = normalized * effective
        metrics[f"{name}_normalized"] = float(normalized.detach().cpu())

    total = sum(weighted_components.values(), zero)
    return FCRV2LossOutput(
        total=total,
        components=weighted_components,
        metrics=metrics,
        active_losses=stage.active_losses,
        weights=weights,
        blocked=dict(stage.blocked),
    )
