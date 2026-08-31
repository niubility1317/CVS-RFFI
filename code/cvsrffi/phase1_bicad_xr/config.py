"""Frozen configuration and stage scheduling for ADV3B02-BiCAD-XR.

This module is intentionally independent from the training implementation.  A
candidate is resolved once into :class:`BiCADXRConfig`; later stages consume the
dataclass rather than reinterpreting candidate names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from enum import Enum
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
    "strict_pair_concat",
    "pair_identity",
    "pair_vicreg",
    "pair_delta",
    "dynamic_adversarial_dose",
    "coverage_convergence",
    "reduce_lr_on_plateau",
    "no_early_freeze",
    "detached_adversarial",
    "adversarial_two_time_scale",
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

_SOURCE_SEARCH_OVERRIDE_FIELDS = frozenset({"lambda_cond_xcov"})

CV2_CANDIDATE_IDS: tuple[str, ...] = (
    "CV2-B0",
    "CV2-B1",
    "CV2-B2",
    "CV2-B3",
    "CV2-D0",
    "CV2-D1",
    "CV2-D2",
    "CV2-D3",
    "CV2-T0",
    "CV2-T1",
    "CV2-T2",
    "CV2-T3",
)
CV2_CANDIDATES = CV2_CANDIDATE_IDS

CV2_DEFERRED_FEATURES: tuple[str, ...] = (
    "vicreg",
    "pair_delta",
    "soft_u_cdan",
    "ema_teacher",
    "robust_class_reference",
    "sparse_xdc",
    "receiver_frontend_augmentation",
    "hard_leo_mining",
    "third_view",
    "hcf_counterfactual_transport",
    "rank4_common_specific",
    "content_conditioned_lodo",
    "iq_mixup",
    "fishr",
    "fasttrust_pseudolabel",
)
DEFERRED_FEATURES = CV2_DEFERRED_FEATURES

_CV2_DEFERRED_CONFIG_FIELDS = frozenset(
    {
        "pair_vicreg",
        "pair_delta",
        "sparse_xdc",
        "use_fishr",
        "use_fasttrust",
        "use_pseudo_label",
        "use_generic_mixup",
        "use_hcf_transport",
        "use_content_lodo",
    }
) | frozenset(CV2_DEFERRED_FEATURES) | frozenset(_FORBIDDEN_LEGACY_FEATURES)


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
    strict_pair_concat: bool = False
    pair_identity: bool = False
    pair_vicreg: bool = False
    pair_delta: bool = False
    dynamic_adversarial_dose: bool = False
    coverage_convergence: bool = False
    reduce_lr_on_plateau: bool = False
    no_early_freeze: bool = False
    detached_adversarial: bool = False
    adversarial_two_time_scale: bool = False

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
    satellite_supervision_mode: str = "ce_only"
    pair_projector_dim: int = 128
    factor_interaction_dim: int = 24
    lambda_sat_cls_start: float = 0.68
    lambda_sat_cls_end: float = 0.68
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

        if self.candidate_id.strip().upper().startswith("CV2-"):
            deferred = [
                name
                for name, enabled in (
                    ("vicreg", self.pair_vicreg),
                    ("pair_delta", self.pair_delta),
                    ("sparse_xdc", self.sparse_xdc),
                )
                if enabled
            ]
            if deferred:
                raise ValueError(
                    "incompatible frozen/deferred features are disabled: "
                    + ", ".join(deferred)
                )

        if self.receiver_tangent not in {"off", "factual", "worst"}:
            raise ValueError("receiver_tangent must be one of: off, factual, worst")
        if self.reduce_lr_on_plateau and not self.coverage_convergence:
            raise ValueError("reduce_lr_on_plateau requires coverage_convergence")
        if self.no_early_freeze and not self.coverage_convergence:
            raise ValueError("no_early_freeze requires coverage_convergence")
        if self.adversarial_two_time_scale and not self.detached_adversarial:
            raise ValueError("adversarial_two_time_scale requires detached_adversarial")
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
        if self.satellite_supervision_mode not in {
            "ce_only",
            "ce_only_plus_pair_selfsup",
        }:
            raise ValueError("incompatible satellite supervision mode")
        if self.satellite_supervision_mode == "ce_only":
            if self.concat_sat_start_epoch != 80:
                raise ValueError("incompatible satellite start epoch")
            if self.lambda_sat_cls_start != 0.68 or self.lambda_sat_cls_end != 0.68:
                raise ValueError("incompatible satellite classification schedule")
        else:
            if not self.strict_pair_concat or self.concat_sat_start_epoch != 1:
                raise ValueError("incompatible strict pair satellite contract")
            if self.lambda_sat_cls_start != 0.5 or self.lambda_sat_cls_end != 1.0:
                raise ValueError("incompatible satellite classification schedule")
        for field_name in ("pair_projector_dim", "factor_interaction_dim"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if not isinstance(self.sat_train_scenarios, tuple):
            raise ValueError("sat_train_scenarios must be a tuple")
        if self.sat_train_scenarios != LEO_WEAK_SCENARIOS:
            raise ValueError("incompatible satellite scenarios")
        if self.lambda_orth != 0.0:
            raise ValueError("incompatible orthogonal loss weight")
        if self.gradient_firewall_scale != 0.05:
            raise ValueError("incompatible gradient firewall scale")
        if not 0.0 < self.margin_tail_cvar_fraction <= 1.0:
            raise ValueError("margin_tail_cvar_fraction must be in (0,1]")
        if not isinstance(self.margin_tail_weights, tuple):
            raise ValueError("margin_tail_weights must be a tuple")
        if self.margin_tail_weights != (0.6, 0.3, 0.1):
            raise ValueError("margin_tail_weights must be exactly (0.6,0.3,0.1)")


def _candidate_registry() -> dict[str, BiCADXRConfig]:
    """Construct the frozen legacy and PairBiCAD candidate registry."""

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

    p0 = replace(
        d0,
        candidate_id="P0",
        optimizer_updates=4000,
        batch_size=48,
        concat_sat_start_epoch=1,
        satellite_supervision_mode="ce_only_plus_pair_selfsup",
        strict_pair_concat=True,
        lambda_sat_cls_start=0.5,
        lambda_sat_cls_end=1.0,
    )
    p1 = replace(
        p0,
        candidate_id="P1",
        factorized_domains=True,
        gradient_firewall=True,
    )
    p2 = replace(
        p1,
        candidate_id="P2",
        conditional_cdan=True,
        zdom_tx_adversary=True,
    )
    p3 = replace(
        p2,
        candidate_id="P3",
        pair_identity=True,
        pair_vicreg=True,
    )
    p4 = replace(
        p3,
        candidate_id="P4",
        pair_delta=True,
        dynamic_adversarial_dose=True,
    )

    cv2_b0 = replace(d0, candidate_id="CV2-B0")
    cv2_b1 = replace(
        p1,
        candidate_id="CV2-B1",
        optimizer_updates=6500,
    )
    cv2_b2 = replace(
        cv2_b1,
        candidate_id="CV2-B2",
        coverage_convergence=True,
        reduce_lr_on_plateau=True,
        no_early_freeze=True,
    )
    cv2_b3 = replace(cv2_b2, candidate_id="CV2-B3", swad=True)

    # The D/T stage-zero rows are frozen copies of the fully formed B3 row.
    # They are independent static registry entries, not aliases selected from
    # a live champion or copies of the historical short-id D0.
    cv2_d0 = replace(cv2_b3, candidate_id="CV2-D0")
    cv2_d1 = replace(
        cv2_d0,
        candidate_id="CV2-D1",
        conditional_cdan=True,
        detached_adversarial=True,
        adversarial_two_time_scale=True,
    )
    cv2_d2 = replace(
        cv2_d1,
        candidate_id="CV2-D2",
        zdom_tx_adversary=True,
        conditional_xcov=True,
    )
    cv2_d3 = replace(
        cv2_d2,
        candidate_id="CV2-D3",
        dynamic_adversarial_dose=True,
        task_protected_gradient=True,
    )

    cv2_t0 = replace(cv2_b3, candidate_id="CV2-T0")
    cv2_t1 = replace(cv2_t0, candidate_id="CV2-T1", pair_identity=True)
    cv2_t2 = replace(cv2_t0, candidate_id="CV2-T2", margin_tail=True)
    cv2_t3 = replace(
        cv2_t0,
        candidate_id="CV2-T3",
        pair_identity=True,
        margin_tail=True,
    )

    return {
        config.candidate_id.upper(): config
        for config in (
            d0,
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            e0,
            e1,
            e2,
            e3,
            e4,
            f0,
            f1,
            f2,
            f3,
            v1,
            p0,
            p1,
            p2,
            p3,
            p4,
            cv2_b0,
            cv2_b1,
            cv2_b2,
            cv2_b3,
            cv2_d0,
            cv2_d1,
            cv2_d2,
            cv2_d3,
            cv2_t0,
            cv2_t1,
            cv2_t2,
            cv2_t3,
        )
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
    deferred = sorted(
        name
        for name in override_values
        if name in _CV2_DEFERRED_CONFIG_FIELDS and override_values[name]
    )
    if deferred:
        raise ValueError(
            "incompatible frozen/deferred features are disabled: "
            + ", ".join(deferred)
        )
    unknown = sorted(set(override_values) - _CONFIG_FIELD_NAMES)
    if unknown:
        raise ValueError(f"unknown config override(s): {', '.join(unknown)}")
    incompatible = [
        name
        for name in _FORBIDDEN_LEGACY_FEATURES
        if override_values.get(name, False)
    ]
    if incompatible:
        names = ", ".join(incompatible)
        raise ValueError(f"incompatible legacy features: {names}")
    frozen = sorted(set(override_values) - _SOURCE_SEARCH_OVERRIDE_FIELDS)
    if frozen:
        raise ValueError(f"frozen config override(s): {', '.join(frozen)}")
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


def _cv2_identity(candidate_id: str) -> tuple[str, int]:
    key = _candidate_key(candidate_id)
    if not key.startswith("CV2-") or len(key) != 6 or key[5] not in "0123":
        raise ValueError(f"not a CV2 candidate: {candidate_id}")
    family = key[4]
    if family not in {"B", "D", "T"}:
        raise ValueError(f"not a CV2 candidate: {candidate_id}")
    return family, int(key[5])


def method_lock_payload(value: str | BiCADXRConfig) -> dict[str, Any]:
    """Return a source-only, immutable-by-convention lock for one CV2 row.

    The returned payload contains no target, support, query, or truth input.
    Each call returns a fresh dictionary so callers cannot mutate the registry.
    """

    config = _resolve_config(value)
    family, level = _cv2_identity(config.candidate_id)
    disabled = {
        name: False
        for name in CV2_DEFERRED_FEATURES
    }
    if config.pair_vicreg or config.pair_delta or config.sparse_xdc:
        raise ValueError("deferred features are disabled for CV2")
    return {
        "schema": "pairbicad_cv2_method_lock_v1",
        "candidate_id": config.candidate_id,
        "candidate_family": family,
        "candidate_level": level,
        "frozen": True,
        "dynamic_alias": False,
        "phase1_method": config.phase1_method,
        "source_only": True,
        "target_access": False,
        "phase2_access": False,
        "support_access": False,
        "query_access": False,
        "truth_access": False,
        "deferred_features": disabled,
        "configuration": {
            field.name: getattr(config, field.name)
            for field in fields(BiCADXRConfig)
        },
    }


def cv2_method_lock(value: str | BiCADXRConfig) -> dict[str, Any]:
    """Compatibility alias for :func:`method_lock_payload`."""

    return method_lock_payload(value)


candidate_method_lock = method_lock_payload


def cv2_candidate_config(
    candidate_id: str,
    overrides: Mapping[str, Any] | None = None,
) -> BiCADXRConfig:
    """Resolve a CV2 candidate and reject non-CV2 aliases."""

    config = candidate_config(candidate_id, overrides=overrides)
    _cv2_identity(config.candidate_id)
    return config


__all__ = [
    "BiCADXRConfig",
    "BiCADXRStage",
    "CANDIDATE_IDS",
    "CV2_CANDIDATES",
    "CV2_CANDIDATE_IDS",
    "CV2_DEFERRED_FEATURES",
    "DEFERRED_FEATURES",
    "LEO_WEAK_SCENARIOS",
    "candidate_config",
    "candidate_diff",
    "cv2_candidate_config",
    "cv2_method_lock",
    "candidate_method_lock",
    "method_lock_payload",
    "stage_for_update",
]
