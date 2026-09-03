from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .phase1_fcr_types import FCRDecodeOutput, FCRV2CapabilityState, FCRV2FactorOutput, FCRV2LossOutput
from .phase1_fcr_v2_schedule import BASE_LOSS_WEIGHTS, FCRV2Schedule


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
    eps: float = 1e-6,
) -> torch.Tensor:
    baseline = full_error.detach().clamp_min(float(eps))
    return ((drop_error - baseline) / baseline).clamp_min(0.0)


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
        normalized = _normalize(name, raw, ema_normalizer)
        weight = BASE_LOSS_WEIGHTS[name] * normalized_supervised_scale
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
        normalized = _normalize(name, raw, ema_normalizer)
        base_weight = schedule.base_weights[name]
        stage_scale = float(stage.scales.get(name, 0.0))
        effective = base_weight * stage_scale * unlabeled_scale
        if role.strip().lower() in {"l_s", "labeled", "source_labeled"}:
            effective = base_weight * stage_scale
        if name not in stage.active_losses:
            effective = 0.0
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
