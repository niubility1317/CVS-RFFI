"""Frozen configuration and stage scheduling for ADV3B02-BiCAD-XR.

This module is intentionally independent from the training implementation.  A
candidate is resolved once into :class:`BiCADXRConfig`; later stages consume the
dataclass rather than reinterpreting candidate names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from enum import Enum
from math import isclose
from typing import Any


LEO_WEAK_SCENARIOS: tuple[str, str, str] = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)


class BiCADXRStage(str, Enum):
    """The five update-based training stages.

    Lower-case members are canonical so ``stage_for_update(...).name`` is the
    report-facing value.  Upper-case aliases are provided for callers that use
    conventional enum member spelling.
    """

    stage0 = "stage0"
    stage1 = "stage1"
    stage2 = "stage2"
    stage3 = "stage3"
    stage4 = "stage4"

    STAGE0 = stage0
    STAGE1 = stage1
    STAGE2 = stage2
    STAGE3 = stage3
    STAGE4 = stage4


_CANDIDATE_SWITCHES = (
    "factorized_domains",
    "conditional_cdan",
    "zdom_tx_adversary",
    "conditional_xcov",
    "gradient_firewall",
    "task_protected_gradient",
    "sparse_xdc",
    "xdc_kd",
    "paired_satellite",
    "margin_tail",
    "receiver_tangent",
    "swad",
)

_FORBIDDEN_LEGACY_FEATURES = (
    "use_fasttrust",
    "use_pseudo_label",
    "use_csd",
    "use_hcf_transport",
    "use_content_lodo",
    "use_hdro",
    "use_proxy_unknown",
    "use_soft_unknown_mixup",
    "use_open_world_feature_loss",
    "use_fishr",
    "use_generic_mixup",
    "use_mixstyle",
)


@dataclass(frozen=True)
class BiCADXRConfig:
    """One immutable, fully resolved BiCAD-XR candidate configuration."""

    candidate_id: str

    # Candidate mechanism switches.
    factorized_domains: bool = False
    conditional_cdan: bool = False
    zdom_tx_adversary: bool = False
    conditional_xcov: bool = False
    gradient_firewall: bool = False
    task_protected_gradient: bool = False
    sparse_xdc: bool = False
    xdc_kd: bool = False
    paired_satellite: bool = False
    margin_tail: bool = False
    receiver_tangent: str = "off"
    swad: bool = False

    # Legacy mechanisms are explicit so an accidental carry-over fails closed.
    use_fasttrust: bool = False
    use_pseudo_label: bool = False
    use_csd: bool = False
    use_hcf_transport: bool = False
    use_content_lodo: bool = False
    use_hdro: bool = False
    use_proxy_unknown: bool = False
    use_soft_unknown_mixup: bool = False
    use_open_world_feature_loss: bool = False
    use_fishr: bool = False
    use_generic_mixup: bool = False
    use_mixstyle: bool = False

    # Frozen protocol, schedule and loss defaults.
    phase1_method: str = "bicad_xr"
    optimizer_updates: int = 5000
    batch_size: int = 96
    xdc_interval: int = 4
    pair_interval: int = 4
    lambda_sat_cls: float = 0.68
    lambda_sat_cons: float = 0.0
    lambda_cond_xcov: float = 0.02
    lambda_orth: float = 0.0
    gradient_firewall_scale: float = 0.05
    concat_sat_ce_only: bool = True
    concat_sat_start_epoch: int = 80
    sat_train_scenarios: tuple[str, str, str] = LEO_WEAK_SCENARIOS

    # Fixed mechanism constants used by later modules.
    xdc_ridge: float = 1e-2
    xdc_temperature: float = 2.0
    xdc_min_support_accuracy: float = 0.25
    xdc_microepisode_tx: int = 6
    xdc_microepisode_receivers: int = 4
    xdc_samples_per_cell: int = 2
    margin_tail_cvar_fraction: float = 0.2
    margin_tail_weights: tuple[float, float, float] = (0.6, 0.3, 0.1)
    margin_tail_ema: float = 0.9
    receiver_tangent_rank: int = 4
    receiver_tangent_start_progress: float = 0.70
    stage4_domain_scale: float = 0.6
    stage4_shared_stem_lr_scale: float = 0.1

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")

        for field_name in _CANDIDATE_SWITCHES + _FORBIDDEN_LEGACY_FEATURES:
            value = getattr(self, field_name)
            if field_name != "receiver_tangent" and not isinstance(value, bool):
                raise ValueError(f"{field_name} must be a bool")

        incompatible = [
            field_name
            for field_name in _FORBIDDEN_LEGACY_FEATURES
            if getattr(self, field_name)
        ]
        if incompatible:
            names = ", ".join(incompatible)
            raise ValueError(f"incompatible legacy features: {names}")

        if self.receiver_tangent not in {"off", "factual", "worst"}:
            raise ValueError("receiver_tangent must be one of: off, factual, worst")
        if self.phase1_method != "bicad_xr":
            raise ValueError("incompatible phase1_method for BiCAD-XR")
        if self.optimizer_updates <= 0:
            raise ValueError("optimizer_updates must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.xdc_interval <= 0 or self.pair_interval <= 0:
            raise ValueError("update intervals must be positive")
        if self.concat_sat_ce_only is not True:
            raise ValueError("incompatible satellite contract: concat_sat_ce_only must be true")
        if self.lambda_sat_cls != 0.68 or self.lambda_sat_cons != 0.0:
            raise ValueError("incompatible satellite loss weights")
        if self.concat_sat_start_epoch != 80:
            raise ValueError("incompatible satellite start epoch")
        if tuple(self.sat_train_scenarios) != LEO_WEAK_SCENARIOS:
            raise ValueError("incompatible satellite scenarios")
        if self.lambda_orth != 0.0:
            raise ValueError("incompatible orthogonal loss weight")
        if self.gradient_firewall_scale != 0.05:
            raise ValueError("incompatible gradient firewall scale")
        if not 0.0 < self.margin_tail_cvar_fraction <= 1.0:
            raise ValueError("margin_tail_cvar_fraction must be in (0,1]")
        if len(self.margin_tail_weights) != 3 or not isclose(
            sum(self.margin_tail_weights), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("margin_tail_weights must contain three components summing to 1")


def _candidate_registry() -> dict[str, BiCADXRConfig]:
    """Construct the frozen D0--F3 registry and the V1 compatibility alias."""

    d0 = BiCADXRConfig(candidate_id="D0")
    d1 = replace(d0, candidate_id="D1", factorized_domains=True)
    d2 = replace(d1, candidate_id="D2", conditional_cdan=True)
    d3 = replace(d2, candidate_id="D3", zdom_tx_adversary=True)
    d4 = replace(d3, candidate_id="D4", conditional_xcov=True)
    d5 = replace(d4, candidate_id="D5", gradient_firewall=True)
    d6 = replace(d5, candidate_id="D6", task_protected_gradient=True)

    e0 = replace(d5, candidate_id="E0")
    e1 = replace(e0, candidate_id="E1", sparse_xdc=True)
    e2 = replace(e1, candidate_id="E2", xdc_kd=True)
    e3 = replace(e2, candidate_id="E3", paired_satellite=True)
    e4 = replace(e3, candidate_id="E4", margin_tail=True)

    f0 = replace(e4, candidate_id="F0")
    f1 = replace(f0, candidate_id="F1", receiver_tangent="factual")
    f2 = replace(f1, candidate_id="F2", receiver_tangent="worst")
    f3 = replace(f2, candidate_id="F3", swad=True)

    v1 = replace(
        d5,
        candidate_id="ADV3B02-BiCAD-XDC-V1",
        sparse_xdc=True,
        margin_tail=True,
    )
    return {
        config.candidate_id.upper(): config
        for config in (d0, d1, d2, d3, d4, d5, d6, e0, e1, e2, e3, e4, f0, f1, f2, f3, v1)
    }


_CANDIDATE_CONFIGS = _candidate_registry()
CANDIDATE_IDS: tuple[str, ...] = tuple(_CANDIDATE_CONFIGS)


def _candidate_key(candidate_id: str) -> str:
    if not isinstance(candidate_id, str):
        raise ValueError(f"unknown candidate: {candidate_id!r}")
    key = candidate_id.strip().upper()
    if key not in _CANDIDATE_CONFIGS:
        raise ValueError(f"unknown candidate: {candidate_id}")
    return key


_CONFIG_FIELD_NAMES = frozenset(field.name for field in fields(BiCADXRConfig))


def candidate_config(
    candidate_id: str,
    overrides: Mapping[str, Any] | None = None,
) -> BiCADXRConfig:
    """Return an immutable config for a registered candidate.

    Overrides are useful for source-only parameter screening, but candidate
    identity and protocol invariants remain frozen.  Unknown fields and
    forbidden legacy mechanisms are rejected instead of being silently ignored.
    """

    base = _CANDIDATE_CONFIGS[_candidate_key(candidate_id)]
    if overrides is None:
        return base
    if not isinstance(overrides, Mapping):
        raise ValueError("overrides must be a mapping")

    override_values = dict(overrides)
    unknown = sorted(set(override_values) - _CONFIG_FIELD_NAMES)
    if unknown:
        raise ValueError(f"unknown config override(s): {', '.join(unknown)}")
    if "candidate_id" in override_values:
        raise ValueError("candidate_id is frozen and cannot be overridden")
    incompatible = [
        name
        for name in _FORBIDDEN_LEGACY_FEATURES
        if override_values.get(name, False)
    ]
    if incompatible:
        names = ", ".join(incompatible)
        raise ValueError(f"incompatible legacy features: {names}")
    return replace(base, **override_values)


def stage_for_update(update: int, total_updates: int) -> BiCADXRStage:
    """Resolve an inclusive optimizer-update boundary into Stage0--Stage4."""

    if not isinstance(total_updates, int) or isinstance(total_updates, bool):
        raise ValueError("total_updates must be a positive integer")
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    if not isinstance(update, int) or isinstance(update, bool):
        raise ValueError("update must be an integer in [1,total_updates]")
    if not 1 <= update <= total_updates:
        raise ValueError("update must be in [1,total_updates]")

    progress = update / total_updates
    boundaries = (
        (0.10, BiCADXRStage.stage0),
        (0.35, BiCADXRStage.stage1),
        (0.70, BiCADXRStage.stage2),
        (0.90, BiCADXRStage.stage3),
        (1.00, BiCADXRStage.stage4),
    )
    for limit, stage in boundaries:
        if progress <= limit:
            return stage
    raise RuntimeError("stage boundary resolution failed")


def _resolve_config(value: str | BiCADXRConfig) -> BiCADXRConfig:
    if isinstance(value, BiCADXRConfig):
        return value
    return candidate_config(value)


def candidate_diff(
    left: str | BiCADXRConfig,
    right: str | BiCADXRConfig,
) -> dict[str, tuple[Any, Any]]:
    """Return differing resolved fields as ``field -> (left, right)``.

    ``candidate_id`` is metadata and is intentionally omitted, so the result
    describes actual mechanism, protocol or weight differences.
    """

    left_config = _resolve_config(left)
    right_config = _resolve_config(right)
    return {
        field.name: (getattr(left_config, field.name), getattr(right_config, field.name))
        for field in fields(BiCADXRConfig)
        if field.name != "candidate_id"
        and getattr(left_config, field.name) != getattr(right_config, field.name)
    }


__all__ = [
    "BiCADXRConfig",
    "BiCADXRStage",
    "CANDIDATE_IDS",
    "LEO_WEAK_SCENARIOS",
    "candidate_config",
    "candidate_diff",
    "stage_for_update",
]
