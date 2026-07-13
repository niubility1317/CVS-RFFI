from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import median
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
    tail_cvar_expansion_block_final_delta: float = 4.0
    tail_cvar_expansion_block_best_delta: float = 6.0
    reference_window: int = 5
    absolute_violation_drives_state: bool = True
    training_stop_enabled: bool = True
    reference_requires_absolute_safe: bool = True


@dataclass(frozen=True)
class TailSafetyDecision(ControlDecision):
    state: str = "NORMAL"
    action: str = "NONE"
    blocks_best: bool = False
    blocks_final: bool = False


@dataclass(frozen=True)
class OpenSetBudgetAction:
    active: bool
    reason: str
    os_scale: float
    closed_scale: float
    pre_budget: float
    post_budget: float
    target_budget: float
    max_budget: float = 1.0


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


def assess_endpoint_artifact(manifest: Any) -> ControlDecision:
    """Validate the versioned endpoint boundary shared by export/eval/runtime."""

    reasons: list[str] = []
    package = manifest if isinstance(manifest, Mapping) and "endpoint_accept_v1" in manifest else None
    obj: Mapping[str, Any] = {}
    if package is None:
        reasons.append("endpoint_artifact_unverified_manifest_only")
        obj = manifest if isinstance(manifest, Mapping) else {}
    else:
        try:
            from cvsrffi.phase2_prototypes import verify_endpoint_accept_v1_manifest

            obj = verify_endpoint_accept_v1_manifest(package)
        except Exception:
            reasons.append("endpoint_artifact_package_verification_failed")
            candidate = package.get("endpoint_accept_v1", {})
            obj = candidate if isinstance(candidate, Mapping) else {}
    if not obj:
        reasons.append("endpoint_artifact_missing")
    if str(obj.get("policy_id", "")).strip() != "endpoint_accept_v1":
        reasons.append("endpoint_artifact_policy_mismatch")
    if int(finite_float(obj.get("schema_version"), 0.0)) != 1:
        reasons.append("endpoint_artifact_schema_mismatch")
    boundary_version = str(obj.get("boundary_version", "")).strip()
    boundary_hash = str(obj.get("boundary_hash", "")).strip()
    if boundary_version != "endpoint_accept_v1.1":
        reasons.append("endpoint_boundary_version_unsupported")
    if not boundary_hash:
        reasons.append("endpoint_boundary_hash_missing")
    if str(obj.get("threshold_source", "")).strip() != "source_val_only":
        reasons.append("endpoint_artifact_threshold_source_invalid")
    if str(obj.get("calibration_split", "")).strip() != "source_val":
        reasons.append("endpoint_artifact_calibration_split_invalid")
    if not _truthy(obj.get("fail_closed", False)):
        reasons.append("endpoint_artifact_not_fail_closed")
    if _truthy(obj.get("loss_gate_exported", False)):
        reasons.append("endpoint_artifact_exports_loss_gate")
    if str(obj.get("accept_policy", "")).strip() != "local_component":
        reasons.append("endpoint_artifact_accept_policy_invalid")
    if _truthy(obj.get("global_ball_accept", False)):
        reasons.append("endpoint_artifact_global_ball_accept_forbidden")
    if str(obj.get("component_radius_key", "")).strip() != "r_accept_deg":
        reasons.append("endpoint_artifact_radius_key_invalid")
    reason_codes = obj.get("reason_codes", [])
    if not isinstance(reason_codes, (list, tuple)) or not reason_codes:
        reasons.append("endpoint_artifact_reason_codes_missing")
    gate_thresholds = obj.get("gate_thresholds")
    if not isinstance(gate_thresholds, Mapping) or not gate_thresholds:
        reasons.append("endpoint_artifact_gate_thresholds_missing")
    calibration = obj.get("calibration_evidence")
    if not isinstance(calibration, Mapping):
        reasons.append("endpoint_artifact_calibration_evidence_missing")
    else:
        if str(calibration.get("threshold_source", "")).strip() != "source_val_only":
            reasons.append("endpoint_artifact_calibration_source_invalid")
        if str(calibration.get("calibration_split", "")).strip() != "source_val":
            reasons.append("endpoint_artifact_calibration_split_invalid")
        if int(finite_float(calibration.get("num_samples"), 0.0)) <= 0:
            reasons.append("endpoint_artifact_calibration_samples_missing")

    entry_points = obj.get("entry_points", {})
    required_entries = ("train_export", "offline_eval", "runtime_inference")
    if not isinstance(entry_points, Mapping):
        reasons.append("endpoint_entry_parity_missing")
        entry_points = {}
    for entry in required_entries:
        row = entry_points.get(entry, {}) if isinstance(entry_points, Mapping) else {}
        if not isinstance(row, Mapping):
            reasons.append(f"endpoint_entry_{entry}_missing")
            continue
        if str(row.get("boundary_version", "")).strip() != boundary_version:
            reasons.append(f"endpoint_entry_{entry}_version_mismatch")
        if str(row.get("boundary_hash", "")).strip() != boundary_hash:
            reasons.append(f"endpoint_entry_{entry}_hash_mismatch")

    fired = bool(reasons)
    return ControlDecision(
        fired=fired,
        reason=";".join(reasons),
        details={
            "endpoint_artifact_ready": 0.0 if fired else 1.0,
            "endpoint_entry_parity_pass": 0.0 if fired else 1.0,
        },
    )


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
    artifact_required = _truthy(row.get("endpoint_artifact_required", False)) or exports_final_boundary
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
        threshold_source = str(row.get("endpoint_threshold_source", "")).strip()
        calibration_split = str(row.get("endpoint_calibration_split", "")).strip()
        if not threshold_source:
            reasons.append("missing_endpoint_threshold_source")
        elif threshold_source != "source_val_only":
            reasons.append("invalid_endpoint_threshold_source")
        if not calibration_split:
            reasons.append("missing_endpoint_calibration_split")
        elif calibration_split != "source_val":
            reasons.append("invalid_endpoint_calibration_split")

    artifact_decision = None
    if artifact_required:
        artifact_decision = assess_endpoint_artifact(row.get("endpoint_artifact"))
        if artifact_decision.fired:
            reasons.extend(part for part in artifact_decision.reason.split(";") if part)

    fired = bool(reasons)
    artifact_ready = (
        float(artifact_decision.details.get("endpoint_artifact_ready", 0.0))
        if artifact_decision is not None
        else 0.0
    )
    return ControlDecision(
        fired=fired,
        reason=";".join(reasons),
        details={
            "endpoint_contract_pass": 0.0 if fired else 1.0,
            "endpoint_config_pass": 0.0 if any(reason.startswith(("missing_endpoint", "invalid_endpoint", "loss_gate")) for reason in reasons) else 1.0,
            "endpoint_artifact_required": 1.0 if artifact_required else 0.0,
            "endpoint_artifact_ready": artifact_ready,
        },
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
        self.reference_best = float("inf")
        self.best_p99 = float("inf")
        self.best_tail_cvar = float("inf")
        window = max(1, int(self.config.reference_window))
        self._p95_window = deque(maxlen=window)
        self._p99_window = deque(maxlen=window)
        self._cvar_window = deque(maxlen=window)
        self._proxy_window = deque(maxlen=window)

    def update(self, metrics: Mapping[str, Any]) -> TailSafetyDecision:
        cfg = self.config
        p95 = _metric_value(metrics, "train/dm_accept_zid_p95_deg", "dm_accept_zid_p95_deg")
        p99 = _metric_value(metrics, "train/dm_accept_zid_p99_deg", "dm_accept_zid_p99_deg")
        cvar = _metric_value(metrics, "train/dm_accept_zid_tail_cvar_deg", "dm_accept_zid_tail_cvar_deg")
        proxy = _metric_value(metrics, "train/dm_accept_proxy_vaccept", "dm_accept_proxy_vaccept")
        values = (p95, p99, cvar, proxy)
        if not all(math.isfinite(value) for value in values):
            return TailSafetyDecision(
                fired=True,
                reason="tail_reference_insufficient",
                details={
                    "tail_reference_ready": 0.0,
                    "tail_reference_observation_count": float(len(self._p99_window)),
                },
                state="INSUFFICIENT",
                action="NONE",
                blocks_best=True,
                blocks_final=True,
            )
        self._p95_window.append(float(p95))
        self._p99_window.append(float(p99))
        self._cvar_window.append(float(cvar))
        self._proxy_window.append(float(proxy))
        if len(self._p99_window) < max(1, int(self.config.reference_window)):
            return TailSafetyDecision(
                fired=True,
                reason="tail_reference_insufficient",
                details={
                    "tail_reference_ready": 0.0,
                    "tail_reference_observation_count": float(len(self._p99_window)),
                },
                state="INSUFFICIENT",
                action="NONE",
                blocks_best=True,
                blocks_final=True,
            )
        p95 = float(median(self._p95_window))
        p99 = float(median(self._p99_window))
        cvar = float(median(self._cvar_window))
        proxy = float(median(self._proxy_window))
        p99_delta = float("nan")
        if math.isfinite(p99) and math.isfinite(self.best_p99):
            p99_delta = max(0.0, float(p99) - float(self.best_p99))
        cvar_delta = float("nan")
        if math.isfinite(cvar) and math.isfinite(self.best_tail_cvar):
            cvar_delta = max(0.0, float(cvar) - float(self.best_tail_cvar))
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
        p99_non_degrading = (not math.isfinite(self.best_p99)) or p99 <= self.best_p99 + 1e-8
        cvar_non_degrading = (not math.isfinite(self.best_tail_cvar)) or cvar <= self.best_tail_cvar + 1e-8
        reference_improved = (
            math.isfinite(composite)
            and composite < self.reference_best
            and p99_non_degrading
            and cvar_non_degrading
        )
        if composite < self.safe_best and not unsafe_parts:
            self.safe_best = composite

        absolute_unsafe = bool(unsafe_parts) or not math.isfinite(composite)
        if (
            (not cfg.absolute_violation_drives_state or not cfg.training_stop_enabled)
            and self.state in {"WARNING", "ROLLBACK", "STOP"}
        ):
            self.state = "NORMAL"
            self.bad_windows = 0
            self.rollback_windows = 0
        state_unsafe = bool(
            absolute_unsafe
            and cfg.absolute_violation_drives_state
            and cfg.training_stop_enabled
        )
        action = "NONE"
        if state_unsafe:
            self.bad_windows += 1
            if self.state == "NORMAL" and self.bad_windows >= max(1, int(cfg.warning_patience)):
                self.state = "WARNING"
                action = "WARNING"
            elif self.state == "WARNING" and self.bad_windows >= max(1, int(cfg.warning_patience)) + max(1, int(cfg.rollback_patience)):
                if self.rollback_count < max(0, int(cfg.max_rollbacks)):
                    self.state = "ROLLBACK"
                    self.rollback_count += 1
                    self.rollback_windows = 0
                    action = "ROLLBACK"
                else:
                    self.state = "STOP"
                    action = "STOP"
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
        if math.isfinite(cvar_delta):
            if cvar_delta > float(cfg.tail_cvar_expansion_block_final_delta):
                expansion_blocks_final = True
                expansion_reasons.append("tail_cvar_expansion_blocks_final")
            if cvar_delta > float(cfg.tail_cvar_expansion_block_best_delta):
                expansion_blocks_best = True
                expansion_reasons.append("tail_cvar_expansion_blocks_promotion")

        reasons: list[str] = []
        if absolute_unsafe:
            reasons.append("tail_safety:" + ",".join(sorted(unsafe_parts)))
        reasons.extend(expansion_reasons)
        reason = ";".join(reasons)
        if self.state == "STOP":
            reason = (reason + ";" if reason else "") + "tail_stop_blocks_final"
        details = {
            "tail_reference_ready": 1.0,
            "tail_reference_observation_count": float(len(self._p99_window)),
            "tail_reference_window": float(max(1, int(cfg.reference_window))),
            "tail_reference_improved": 1.0 if reference_improved else 0.0,
            "tail_reference_p99_non_degrading": 1.0 if p99_non_degrading else 0.0,
            "tail_reference_cvar_non_degrading": 1.0 if cvar_non_degrading else 0.0,
            "tail_reference_best_composite": self.reference_best,
            "tail_absolute_violation": 1.0 if absolute_unsafe else 0.0,
            "tail_absolute_violation_drives_state": 1.0 if cfg.absolute_violation_drives_state else 0.0,
            "tail_training_stop_enabled": 1.0 if cfg.training_stop_enabled else 0.0,
            "tail_reference_requires_absolute_safe": 1.0 if cfg.reference_requires_absolute_safe else 0.0,
            "tail_safety_composite_v1": composite,
            "tail_safety_bad_windows": float(self.bad_windows),
            "tail_safety_rollbacks": float(self.rollback_count),
            "tail_expansion_p99_best": self.best_p99 if math.isfinite(self.best_p99) else float("nan"),
            "tail_expansion_p99_current": p99,
            "tail_expansion_p99_delta": p99_delta,
            "tail_expansion_cvar_best": self.best_tail_cvar if math.isfinite(self.best_tail_cvar) else float("nan"),
            "tail_expansion_cvar_current": cvar,
            "tail_expansion_cvar_delta": cvar_delta,
            "tail_expansion_block_final_delta": float(cfg.p99_expansion_block_final_delta),
            "tail_expansion_block_best_delta": float(cfg.p99_expansion_block_best_delta),
            "tail_cvar_expansion_block_final_delta": float(cfg.tail_cvar_expansion_block_final_delta),
            "tail_cvar_expansion_block_best_delta": float(cfg.tail_cvar_expansion_block_best_delta),
            "tail_expansion_blocks_final": 1.0 if expansion_blocks_final else 0.0,
            "tail_expansion_blocks_promotion": 1.0 if expansion_blocks_best else 0.0,
            **{f"tail_ratio_{key}": value for key, value in ratios.items()},
        }
        return TailSafetyDecision(
            fired=absolute_unsafe or expansion_blocks_final or expansion_blocks_best or self.state in {"WARNING", "ROLLBACK", "STOP"},
            reason=reason,
            details=details,
            state=self.state,
            action=action,
            blocks_best=absolute_unsafe or expansion_blocks_best or self.state in {"WARNING", "ROLLBACK", "STOP"},
            blocks_final=absolute_unsafe or expansion_blocks_final or self.state == "STOP",
        )

    def acknowledge_rollback(self) -> None:
        """Reset transient windows after the trainer restored a reference checkpoint."""

        self.state = "NORMAL"
        self.bad_windows = 0
        self.rollback_windows = 0
        self._p95_window.clear()
        self._p99_window.clear()
        self._cvar_window.clear()
        self._proxy_window.clear()

    def commit_reference(self, decision: TailSafetyDecision) -> None:
        """Commit one finite metric reference without confusing it with export safety."""

        details = decision.details
        composite = finite_float(details.get("tail_safety_composite_v1"))
        p99 = finite_float(details.get("tail_expansion_p99_current"))
        cvar = finite_float(details.get("tail_expansion_cvar_current"))
        reference_ready = finite_float(details.get("tail_reference_ready"), 0.0) >= 1.0
        reference_improved = finite_float(details.get("tail_reference_improved"), 0.0) >= 1.0
        absolute_unsafe = finite_float(details.get("tail_absolute_violation"), 0.0) >= 1.0
        expansion_blocked = (
            finite_float(details.get("tail_expansion_blocks_final"), 0.0) >= 1.0
            or finite_float(details.get("tail_expansion_blocks_promotion"), 0.0) >= 1.0
        )
        invalid_state = decision.state in {"WARNING", "ROLLBACK", "STOP", "INSUFFICIENT"}
        if not reference_ready or not reference_improved:
            raise ValueError("tail reference requires one complete, improved metric window")
        if not all(math.isfinite(value) for value in (composite, p99, cvar)):
            raise ValueError("tail reference metrics must be finite")
        if expansion_blocked or invalid_state:
            raise ValueError("tail reference cannot be committed from an expanding or stopped state")
        if absolute_unsafe and self.config.reference_requires_absolute_safe:
            raise ValueError("tail reference exceeds absolute safety targets")
        self.reference_best = composite
        self.best_p99 = p99
        self.best_tail_cvar = cvar

    def state_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "bad_windows": self.bad_windows,
            "rollback_windows": self.rollback_windows,
            "rollback_count": self.rollback_count,
            "safe_best": self.safe_best,
            "reference_best": self.reference_best,
            "best_p99": self.best_p99,
            "best_tail_cvar": self.best_tail_cvar,
            "p95_window": list(self._p95_window),
            "p99_window": list(self._p99_window),
            "cvar_window": list(self._cvar_window),
            "proxy_window": list(self._proxy_window),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.state = str(state.get("state", "NORMAL"))
        self.bad_windows = int(state.get("bad_windows", 0))
        self.rollback_windows = int(state.get("rollback_windows", 0))
        self.rollback_count = int(state.get("rollback_count", 0))
        self.safe_best = finite_float(state.get("safe_best"), float("inf"))
        self.reference_best = finite_float(state.get("reference_best"), float("inf"))
        self.best_p99 = finite_float(state.get("best_p99"), float("inf"))
        self.best_tail_cvar = finite_float(state.get("best_tail_cvar"), float("inf"))
        for queue, key in (
            (self._p95_window, "p95_window"),
            (self._p99_window, "p99_window"),
            (self._cvar_window, "cvar_window"),
            (self._proxy_window, "proxy_window"),
        ):
            queue.clear()
            queue.extend(float(value) for value in state.get(key, []))


def _sum_abs(metrics: Mapping[str, Any], keys: Iterable[str]) -> float:
    total = 0.0
    for key in keys:
        value = finite_float(metrics.get(key), 0.0)
        if math.isfinite(value):
            total += abs(float(value))
    return total


def compute_open_set_budget_action(
    *,
    os_total: float,
    closed_total: float,
    min_budget: float,
    max_budget: float = 0.0,
    max_os_scale: float = 4.0,
    min_closed_scale: float = 0.35,
) -> OpenSetBudgetAction:
    """Return bounded scales that make the open-set loss budget actionable."""

    os_value = max(0.0, finite_float(os_total, 0.0))
    closed_value = max(0.0, finite_float(closed_total, 0.0))
    target = max(0.0, min(0.95, finite_float(min_budget, 0.0)))
    max_target_raw = finite_float(max_budget, 0.0)
    max_target = 1.0 if max_target_raw <= 0.0 else max(target, min(0.99, max_target_raw))
    denom = os_value + closed_value
    pre = os_value / denom if denom > 0.0 else 0.0
    if pre > max_target and os_value > 0.0 and closed_value > 0.0:
        desired_os = (max_target * closed_value) / max(1e-8, 1.0 - max_target)
        os_scale = max(0.0, min(1.0, desired_os / os_value))
        scaled_os = os_value * os_scale
        post = scaled_os / max(1e-8, scaled_os + closed_value)
        return OpenSetBudgetAction(
            True,
            "B_os_eff_upper_controller_active",
            float(os_scale),
            1.0,
            float(pre),
            float(post),
            float(target),
            float(max_target),
        )
    if target <= 0.0 or pre >= target:
        return OpenSetBudgetAction(False, "", 1.0, 1.0, pre, pre, target, max_target)
    if os_value <= 0.0:
        return OpenSetBudgetAction(False, "OS_LOSS_IDLE", 1.0, 1.0, pre, pre, target, max_target)

    desired_os = (target * closed_value) / max(1e-8, 1.0 - target)
    os_scale = min(max(1.0, desired_os / os_value), max(1.0, float(max_os_scale)))
    closed_scale = 1.0
    scaled_os = os_value * os_scale
    post = scaled_os / max(1e-8, scaled_os + closed_value)
    if post < target and closed_value > 0.0:
        desired_closed = scaled_os * (1.0 - target) / max(1e-8, target)
        closed_scale = max(max(0.0, min(1.0, float(min_closed_scale))), min(1.0, desired_closed / closed_value))
        post = scaled_os / max(1e-8, scaled_os + closed_value * closed_scale)
    return OpenSetBudgetAction(
        True,
        "B_os_eff_controller_active",
        float(os_scale),
        float(closed_scale),
        float(pre),
        float(post),
        float(target),
        float(max_target),
    )


def assess_open_set_effective_budget(
    metrics: Mapping[str, Any],
    *,
    min_budget: float = 0.15,
    max_budget: float = 0.0,
) -> ControlDecision:
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
    effective_open_grad = finite_float(metrics.get("train/os_gradient_effective_open_norm"))
    effective_closed_grad = finite_float(
        metrics.get("train/os_gradient_effective_closed_norm", metrics.get("train/os_gradient_balanced_closed_norm"))
    )
    uses_gradient_budget = math.isfinite(effective_open_grad) and math.isfinite(effective_closed_grad)
    if uses_gradient_budget:
        os_total = max(0.0, effective_open_grad)
        closed_total = max(0.0, effective_closed_grad)
    else:
        os_total = _sum_abs(metrics, os_keys)
        closed_total = _sum_abs(metrics, closed_keys)
    denom = os_total + closed_total
    budget = os_total / denom if denom > 0.0 else 0.0
    min_target = max(0.0, float(min_budget))
    max_target = float(max_budget)
    below_min = budget < min_target
    above_max = max_target > 0.0 and budget > max_target
    fired = below_min or above_max
    reason = "B_os_eff_below_min" if below_min else "B_os_eff_above_max" if above_max else ""
    return ControlDecision(
        fired=fired,
        reason=reason,
        details={
            "B_os_eff": budget,
            "B_os_total": os_total,
            "B_closed_total": closed_total,
            "B_os_uses_gradient_norm": 1.0 if uses_gradient_budget else 0.0,
            "B_os_min_target": min_target,
            "B_os_max_target": max_target,
        },
    )


def assess_unlabeled_tri_state(
    metrics: Mapping[str, Any],
    *,
    required: bool,
    min_selected: int = 16,
    min_core_rate: float = 0.05,
    max_core_rate: float = 0.95,
    min_ambiguous_rate: float = 0.01,
    max_outside_rate: float = 0.80,
    min_class_coverage: int = 2,
    min_domain_coverage: int = 2,
    max_pair_disagreement_rate: float = 0.25,
    min_pseudo_component_agreement: float = 0.80,
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
    query_count = finite_float(metrics.get("train/u_tri_query_count"), float("nan"))
    source = str(metrics.get("train/u_tri_state_source", "")).strip().lower()
    source_code = finite_float(metrics.get("train/u_tri_state_source_code"), float("nan"))
    has_geometry_source = source == "geometry" or (math.isfinite(source_code) and source_code >= 0.5)
    count_sum = sum(max(0.0, value) for value in counts)
    if any(key in metrics for key in tri_keys):
        if count_sum <= 0.0:
            reasons.append("US_TRI_STATE_EMPTY")
    else:
        reasons.append("US_TRI_STATE_COUNTS_MISSING")
    if not has_geometry_source:
        reasons.append("US_TRI_STATE_NOT_GEOMETRY" if source else "US_TRI_STATE_SOURCE_MISSING")
    if not math.isfinite(query_count) or query_count <= 0.0:
        reasons.append("US_TRI_STATE_QUERY_COUNT_MISSING")
    elif abs(count_sum - query_count) > max(1e-6, 0.01 * query_count):
        reasons.append("US_TRI_STATE_COUNT_MISMATCH")
    core_rate = counts[0] / query_count if math.isfinite(query_count) and query_count > 0.0 else float("nan")
    ambiguous_rate = counts[1] / query_count if math.isfinite(query_count) and query_count > 0.0 else float("nan")
    outside_rate = counts[2] / query_count if math.isfinite(query_count) and query_count > 0.0 else float("nan")
    class_coverage = finite_float(metrics.get("train/u_tri_class_coverage"), float("nan"))
    domain_coverage = finite_float(metrics.get("train/u_tri_domain_coverage"), float("nan"))
    pair_disagreement = finite_float(metrics.get("train/u_tri_pair_disagreement_rate"), float("nan"))
    pseudo_component_agreement = finite_float(
        metrics.get("train/u_tri_pseudo_component_agreement_rate"), float("nan")
    )
    if not math.isfinite(core_rate) or core_rate < float(min_core_rate):
        reasons.append("US_TRI_STATE_CORE_COLLAPSE")
    if math.isfinite(core_rate) and core_rate > float(max_core_rate):
        reasons.append("US_TRI_STATE_ALL_CORE_DEGENERATE")
    if not math.isfinite(ambiguous_rate) or ambiguous_rate < float(min_ambiguous_rate):
        reasons.append("US_TRI_STATE_NO_AMBIGUOUS_QUARANTINE")
    if not math.isfinite(outside_rate) or outside_rate > float(max_outside_rate):
        reasons.append("US_TRI_STATE_OUTSIDE_COLLAPSE")
    if not math.isfinite(class_coverage) or class_coverage < float(max(1, int(min_class_coverage))):
        reasons.append("US_TRI_STATE_CLASS_COVERAGE_LOW")
    if not math.isfinite(domain_coverage) or domain_coverage < float(max(1, int(min_domain_coverage))):
        reasons.append("US_TRI_STATE_DOMAIN_COVERAGE_LOW")
    if not math.isfinite(pair_disagreement) or pair_disagreement > float(max_pair_disagreement_rate):
        reasons.append("US_TRI_STATE_CLEAN_SAT_PAIR_INCONSISTENT")
    if (
        not math.isfinite(pseudo_component_agreement)
        or pseudo_component_agreement < float(min_pseudo_component_agreement)
    ):
        reasons.append("US_TRI_STATE_PSEUDO_COMPONENT_MISMATCH")
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
            "u_tri_query_count": query_count,
            "u_tri_count_sum": count_sum,
            "u_tri_geometry_source": 1.0 if has_geometry_source else 0.0,
            "u_tri_trusted_core_count": counts[0] if len(counts) > 0 else 0.0,
            "u_tri_ambiguous_tail_count": counts[1] if len(counts) > 1 else 0.0,
            "u_tri_outside_reject_count": counts[2] if len(counts) > 2 else 0.0,
            "u_tri_trusted_core_rate": core_rate,
            "u_tri_ambiguous_tail_rate": ambiguous_rate,
            "u_tri_outside_reject_rate": outside_rate,
            "u_tri_class_coverage": class_coverage,
            "u_tri_domain_coverage": domain_coverage,
            "u_tri_pair_disagreement_rate": pair_disagreement,
            "u_tri_pseudo_component_agreement_rate": pseudo_component_agreement,
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
    p95 = _metric_value(metrics, "train/source_episode_zid_p95_deg", "source_episode_zid_p95_deg")
    p99 = _metric_value(metrics, "train/source_episode_zid_p99_deg", "source_episode_zid_p99_deg")
    tail_cvar = _metric_value(
        metrics,
        "train/source_episode_zid_tail_cvar_deg",
        "source_episode_zid_tail_cvar_deg",
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
    if not (math.isfinite(p95) and math.isfinite(p99) and math.isfinite(tail_cvar)):
        reasons.append("SOURCE_EPISODE_QUANTILES_MISSING")
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
            "source_episode_zid_p95_deg": p95,
            "source_episode_zid_p99_deg": p99,
            "source_episode_zid_tail_cvar_deg": tail_cvar,
        },
    )


def assess_phase1_v2_final_export_policy(
    reasons: Iterable[str] | str,
    *,
    tail_blocks_final: bool = False,
    fail_closed: bool = True,
) -> ControlDecision:
    if isinstance(reasons, str):
        reason_items = [part.strip() for part in reasons.split(";") if part.strip()]
    else:
        reason_items = [str(part).strip() for part in reasons if str(part).strip()]
    critical_prefixes = (
        "missing_endpoint",
        "invalid_endpoint",
        "loss_gate_exported",
        "ambiguous_proxy_final_metric",
        "phase1_claim_contains_real_unknown_metric",
        "phase1_overclaims_stage2_or_deployment_success",
        "B_os_eff_below_min",
        "US_DIRECT_LOSS_IDLE",
        "US_TRI_STATE",
        "SOURCE_EPISODE_",
        "RELAXED_UNREACHABLE",
        "LOCAL_UPPER_BOUND",
        "LOSS_GEOMETRY_DECOUPLED",
    )
    critical = [
        reason
        for item in reason_items
        for reason in item.split(";")
        if any(reason.startswith(prefix) for prefix in critical_prefixes)
    ]
    fired = bool(tail_blocks_final) or (bool(fail_closed) and bool(critical))
    final_reason = "phase1_v2_guard_blocks_final_export" if fired else ""
    return ControlDecision(
        fired=fired,
        reason=final_reason,
        details={
            "final_export_allowed": 0.0 if fired else 1.0,
            "tail_blocks_final": 1.0 if tail_blocks_final else 0.0,
            "critical_guard_count": float(len(critical)),
        },
    )


def should_skip_phase1_v2_final_export(
    *,
    phase1_v2_final_blocked: bool,
    tail_stop_blocks_final: bool = True,
) -> bool:
    """Return whether final prototype export must be skipped after any v2 block.

    ``tail_stop_blocks_final`` is retained for caller compatibility, but it must
    not weaken endpoint, OS-budget, U_s, source-episode, or feasibility blocks.
    """
    _ = tail_stop_blocks_final
    return bool(phase1_v2_final_blocked)


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
