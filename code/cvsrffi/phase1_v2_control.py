from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping


@dataclass(frozen=True)
class ControlDecision:
    fired: bool
    reason: str = ""
    details: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TailSafetyConfig:
    p95_target_deg: float = 54.0
    p99_target_deg: float = 70.0
    tail_cvar_target_deg: float = 56.0
    proxy_vaccept_target: float = 0.35
    warning_patience: int = 2
    rollback_patience: int = 1
    max_rollbacks: int = 1
    p99_expansion_block_final_delta: float = 2.0
    p99_expansion_block_best_delta: float = 3.5


@dataclass(frozen=True)
class TailSafetyDecision(ControlDecision):
    state: str = "NORMAL"
    action: str = "NONE"
    blocks_best: bool = False
    blocks_final: bool = False


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _source_only_phase(row: Mapping[str, Any]) -> bool:
    phase = str(row.get("phase", row.get("claim_level", ""))).lower()
    if "phase1" in phase or "source_only" in phase:
        return True
    return _truthy(row.get("source_only", False))


def assess_endpoint_contract(row: Mapping[str, Any]) -> ControlDecision:
    """Fail closed when Phase1 proxy/loss gates are exported as final evidence."""
    reasons: list[str] = []
    source_only = _source_only_phase(row)
    policy = str(row.get("endpoint_policy_id", row.get("endpoint_accept_policy_id", ""))).strip()
    final_keys = {
        "final_accept_rate",
        "final_reject_rate",
        "final_accepted",
        "endpoint_decision",
        "endpoint_accept_rate",
        "endpoint_reject_rate",
    }
    exports_final_boundary = any(key in row for key in final_keys) or _truthy(row.get("endpoint_accept_boundary_exported", False))
    if exports_final_boundary and policy != "endpoint_accept_v1":
        reasons.append("missing_endpoint_accept_v1")
    if _truthy(row.get("loss_gate_exported", False)):
        reasons.append("loss_gate_exported")
    if any(key in row for key in ("unknown_FAR_proxy", "FPR95_proxy", "stage2_success_proxy")):
        reasons.append("ambiguous_proxy_final_metric")
    real_unknown_keys = ("unknown_FAR", "FPR95", "true_unknown_eval_FAR")
    if source_only and any(key in row for key in real_unknown_keys) and not _truthy(row.get("real_unknown_eval_available", False)):
        reasons.append("phase1_claim_contains_real_unknown_metric")
    if source_only and (_truthy(row.get("stage2_success_claim", False)) or _truthy(row.get("deployment_success_claim", False))):
        reasons.append("phase1_overclaims_stage2_or_deployment_success")
    if policy == "endpoint_accept_v1":
        for key in ("endpoint_threshold_source", "endpoint_calibration_split"):
            if not str(row.get(key, "")).strip():
                reasons.append(f"missing_{key}")

    fired = bool(reasons)
    return ControlDecision(
        fired=fired,
        reason=";".join(reasons),
        details={"endpoint_contract_pass": 0.0 if fired else 1.0},
    )


def _metric_value(metrics: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        value = finite_float(metrics.get(key))
        if math.isfinite(value):
            return value
    return float("nan")


class TailSafetyStateMachine:
    """Training-time tail guard for Phase1 known-domain expansion."""

    def __init__(self, config: TailSafetyConfig | None = None):
        self.config = config or TailSafetyConfig()
        self.state = "NORMAL"
        self.bad_windows = 0
        self.rollback_windows = 0
        self.rollback_count = 0
        self.safe_best = float("inf")
        self.best_p99 = float("inf")

    def update(self, metrics: Mapping[str, Any]) -> TailSafetyDecision:
        cfg = self.config
        p95 = _metric_value(metrics, "train/dm_accept_zid_p95_deg", "dm_accept_zid_p95_deg")
        p99 = _metric_value(metrics, "train/dm_accept_zid_p99_deg", "dm_accept_zid_p99_deg")
        cvar = _metric_value(metrics, "train/dm_accept_zid_tail_cvar_deg", "dm_accept_zid_tail_cvar_deg")
        proxy = _metric_value(metrics, "train/dm_accept_proxy_vaccept", "dm_accept_proxy_vaccept")
        if math.isfinite(p99):
            self.best_p99 = min(self.best_p99, float(p99))
        p99_delta = float("nan")
        if math.isfinite(p99) and math.isfinite(self.best_p99):
            p99_delta = max(0.0, float(p99) - float(self.best_p99))
        pairs = (
            ("p95", p95, cfg.p95_target_deg),
            ("p99", p99, cfg.p99_target_deg),
            ("tail_cvar", cvar, cfg.tail_cvar_target_deg),
            ("proxy_vaccept", proxy, cfg.proxy_vaccept_target),
        )
        ratios: Dict[str, float] = {}
        unsafe_parts: list[str] = []
        for name, value, target in pairs:
            if not math.isfinite(value) or float(target) <= 0.0:
                ratios[name] = float("inf")
                unsafe_parts.append(f"{name}_missing")
            else:
                ratios[name] = float(value) / float(target)
                if value > target:
                    unsafe_parts.append(f"{name}_over")
        finite_ratios = [value for value in ratios.values() if math.isfinite(value)]
        composite = sum(finite_ratios) / len(finite_ratios) if finite_ratios else float("inf")
        if composite < self.safe_best and len(unsafe_parts) < 2:
            self.safe_best = composite

        unsafe = len(unsafe_parts) >= 2 or not math.isfinite(composite)
        action = "NONE"
        if unsafe:
            self.bad_windows += 1
            if self.state == "NORMAL" and self.bad_windows >= max(1, int(cfg.warning_patience)):
                self.state = "WARNING"
                action = "WARNING"
            elif self.state == "WARNING" and self.bad_windows >= max(1, int(cfg.warning_patience)) + max(1, int(cfg.rollback_patience)):
                self.state = "ROLLBACK"
                self.rollback_count += 1
                self.rollback_windows = 0
                action = "ROLLBACK"
            elif self.state == "ROLLBACK":
                self.rollback_windows += 1
                if self.rollback_count >= max(0, int(cfg.max_rollbacks)) and self.rollback_windows >= max(1, int(cfg.rollback_patience)):
                    self.state = "STOP"
                    action = "STOP"
        else:
            self.bad_windows = 0
            self.rollback_windows = 0
            if self.state in {"WARNING", "ROLLBACK"}:
                self.state = "NORMAL"

        expansion_reasons: list[str] = []
        expansion_blocks_final = False
        expansion_blocks_best = False
        if math.isfinite(p99_delta):
            if p99_delta > float(cfg.p99_expansion_block_final_delta):
                expansion_blocks_final = True
                expansion_reasons.append("tail_expansion_blocks_final")
            if p99_delta > float(cfg.p99_expansion_block_best_delta):
                expansion_blocks_best = True
                expansion_reasons.append("tail_expansion_blocks_promotion")

        reasons: list[str] = []
        if unsafe:
            reasons.append("tail_safety:" + ",".join(sorted(unsafe_parts)))
        reasons.extend(expansion_reasons)
        reason = ";".join(reasons)
        if self.state == "STOP":
            reason = (reason + ";" if reason else "") + "tail_stop_blocks_final"
        details = {
            "tail_safety_composite_v1": composite,
            "tail_safety_bad_windows": float(self.bad_windows),
            "tail_safety_rollbacks": float(self.rollback_count),
            "tail_expansion_p99_best": self.best_p99 if math.isfinite(self.best_p99) else float("nan"),
            "tail_expansion_p99_current": p99,
            "tail_expansion_p99_delta": p99_delta,
            "tail_expansion_block_final_delta": float(cfg.p99_expansion_block_final_delta),
            "tail_expansion_block_best_delta": float(cfg.p99_expansion_block_best_delta),
            **{f"tail_ratio_{key}": value for key, value in ratios.items()},
        }
        return TailSafetyDecision(
            fired=unsafe or expansion_blocks_final or expansion_blocks_best or self.state in {"WARNING", "ROLLBACK", "STOP"},
            reason=reason,
            details=details,
            state=self.state,
            action=action,
            blocks_best=expansion_blocks_best or self.state in {"WARNING", "ROLLBACK", "STOP"},
            blocks_final=expansion_blocks_final or self.state == "STOP",
        )


def _sum_abs(metrics: Mapping[str, Any], keys: Iterable[str]) -> float:
    total = 0.0
    for key in keys:
        value = finite_float(metrics.get(key), 0.0)
        if math.isfinite(value):
            total += abs(float(value))
    return total


def assess_open_set_effective_budget(metrics: Mapping[str, Any], *, min_budget: float = 0.15) -> ControlDecision:
    os_keys = (
        "train/w_loss_direct_metric_accept",
        "train/w_loss_source_episode",
        "train/w_loss_proxy_unknown",
        "train/w_loss_open_world_feat",
        "train/w_loss_zid_compact",
        "train/w_loss_u_direct_metric_accept",
        "train/w_loss_u_quarantine_accept",
    )
    closed_keys = (
        "train/w_loss_tx_labeled",
        "train/w_loss_group_ce_labeled",
        "train/w_loss_sat_cls_labeled",
        "train/w_loss_sat_cons_labeled",
        "train/w_loss_teacher_clean_kl",
        "train/w_loss_teacher_sat_kl",
        "train/w_loss_teacher_zid_mse",
        "train/w_loss_u_domain",
        "train/w_loss_u_adv",
        "train/w_loss_u_sat_cons",
    )
    os_total = _sum_abs(metrics, os_keys)
    closed_total = _sum_abs(metrics, closed_keys)
    denom = os_total + closed_total
    budget = os_total / denom if denom > 0.0 else 0.0
    fired = budget < max(0.0, float(min_budget))
    reason = "B_os_eff_below_min" if fired else ""
    return ControlDecision(
        fired=fired,
        reason=reason,
        details={"B_os_eff": budget, "B_os_total": os_total, "B_closed_total": closed_total},
    )


def assess_unlabeled_tri_state(
    metrics: Mapping[str, Any],
    *,
    required: bool,
    min_selected: int = 16,
) -> ControlDecision:
    if not required:
        return ControlDecision(False, "", {"promotable": 1.0})
    w_loss = finite_float(metrics.get("train/w_loss_u_direct_metric_accept"), 0.0)
    active = finite_float(metrics.get("train/u_dm_accept_active"), 0.0)
    selected = finite_float(metrics.get("train/u_dm_accept_selected"), 0.0)
    pseudo_selected = finite_float(metrics.get("train/pseudo_selected"), 0.0)
    reasons: list[str] = []
    if w_loss <= 0.0 or active <= 0.0 or selected < float(min_selected):
        reasons.append("US_DIRECT_LOSS_IDLE")
    tri_keys = (
        "train/u_tri_trusted_core_count",
        "train/u_tri_ambiguous_tail_count",
        "train/u_tri_outside_reject_count",
    )
    counts = [finite_float(metrics.get(key), 0.0) for key in tri_keys]
    if any(key in metrics for key in tri_keys):
        if sum(max(0.0, value) for value in counts) <= 0.0:
            reasons.append("US_TRI_STATE_EMPTY")
    else:
        reasons.append("US_TRI_STATE_COUNTS_MISSING")
    fired = bool(reasons)
    return ControlDecision(
        fired=fired,
        reason=";".join(reasons),
        details={
            "promotable": 0.0 if fired else 1.0,
            "u_direct_weighted_loss": w_loss,
            "u_direct_active": active,
            "u_direct_selected": selected,
            "u_pseudo_selected": pseudo_selected,
            "u_tri_trusted_core_count": counts[0] if len(counts) > 0 else 0.0,
            "u_tri_ambiguous_tail_count": counts[1] if len(counts) > 1 else 0.0,
            "u_tri_outside_reject_count": counts[2] if len(counts) > 2 else 0.0,
        },
    )


def assess_source_episode_density_gate(
    metrics: Mapping[str, Any],
    *,
    overflow_warn: float = 0.90,
    min_local_components: int = 1,
) -> ControlDecision:
    overflow = _metric_value(
        metrics,
        "train/source_episode_overflow_rate",
        "train/source_overflow",
        "source_episode_overflow_rate",
    )
    local_components = _metric_value(
        metrics,
        "train/source_episode_receiver_local_component_count",
        "source_episode_receiver_local_component_count",
    )
    core_tail_outside_ready = _metric_value(
        metrics,
        "train/source_episode_core_tail_outside_ready",
        "source_episode_core_tail_outside_ready",
    )
    density_gate_active = _metric_value(
        metrics,
        "train/source_episode_density_gate_active",
        "source_episode_density_gate_active",
    )
    reasons: list[str] = []
    if not math.isfinite(overflow):
        reasons.append("SOURCE_EPISODE_OVERFLOW_MISSING")
    elif overflow > float(overflow_warn):
        reasons.append("SOURCE_EPISODE_OVERFLOW_HIGH")
    if (not math.isfinite(local_components)) or local_components < float(max(1, int(min_local_components))):
        reasons.append("RECEIVER_AWARE_LOCAL_COMPONENT_MISSING")
    if (not math.isfinite(core_tail_outside_ready)) or core_tail_outside_ready <= 0.0:
        reasons.append("CORE_TAIL_OUTSIDE_NOT_READY")
    if (not math.isfinite(density_gate_active)) or density_gate_active <= 0.0:
        reasons.append("SOURCE_EPISODE_DENSITY_GATE_INACTIVE")
    fired = bool(reasons)
    return ControlDecision(
        fired=fired,
        reason=";".join(reasons),
        details={
            "source_episode_overflow_rate": overflow,
            "source_episode_overflow_warn": float(overflow_warn),
            "source_episode_receiver_local_component_count": local_components,
            "source_episode_core_tail_outside_ready": core_tail_outside_ready,
            "source_episode_density_gate_active": density_gate_active,
        },
    )


def assess_feasibility_gate(state: Mapping[str, Any]) -> ControlDecision:
    stage = str(state.get("stage", "audit")).strip().lower()
    relaxed_pass = _truthy(state.get("relaxed_pass", False))
    local_pass = _truthy(state.get("local_pass", False))
    slope = finite_float(state.get("loss_response_slope"), 0.0)
    excess_delta = finite_float(state.get("overflow_excess_cvar95_delta"), 0.0)
    reasons: list[str] = []
    if stage == "full" and not relaxed_pass:
        reasons.append("RELAXED_UNREACHABLE_STOP_FULL_TARGET")
    if stage == "full" and relaxed_pass and not local_pass:
        reasons.append("LOCAL_UPPER_BOUND_BELOW_TARGET")
    if slope >= 0.0 and excess_delta >= 0.0 and stage in {"relaxed", "local", "full"}:
        reasons.append("LOSS_GEOMETRY_DECOUPLED")
    fired = bool(reasons)
    return ControlDecision(
        fired=fired,
        reason=";".join(reasons),
        details={
            "relaxed_pass": 1.0 if relaxed_pass else 0.0,
            "local_pass": 1.0 if local_pass else 0.0,
            "loss_response_slope": slope,
            "overflow_excess_cvar95_delta": excess_delta,
        },
    )
