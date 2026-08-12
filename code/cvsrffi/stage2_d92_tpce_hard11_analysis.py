"""Independent TPCE Hard11 analyzer.

Metric computation and resource statistics are shared with the proven Hard11
analyzer; TPCE-specific fit/state receipts are validated here before any
truth-side join. The exported verdict remains the frozen three-way decision.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cvsrffi import stage2_d92_pareto_distill_hard11_analysis as _base
from cvsrffi.stage2_d92_tpce_hard11 import (
    ARM_ID,
    CANDIDATE_ID,
    CANONICAL_SELECTION_SHA256,
    FIT_GATE,
    HISTORICAL_BASELINE_PATH,
    HISTORICAL_BASELINE_SHA256,
    HISTORICAL_PER_OLD_CLASS_PATH,
    HISTORICAL_PER_OLD_CLASS_SHA256,
    RESOURCE_GATE,
    STRICT_PARETO_THRESHOLDS,
    validate_hard11_manifest,
    validate_method_lock,
)
from cvsrffi.stage2_d92_tpce_hard11 import HARD11_ROWS, LIVENESS_OUTER_KEY, SMOKE_OUTER_KEY
from scripts.run_d92_tpce_hard11 import QUERY_ZERO_FIELDS, _validate_fit_audit


EIGHT_PARETO_METRICS = _base.EIGHT_PARETO_METRICS
PARETO_METRICS = EIGHT_PARETO_METRICS
D92TPCEHard11AnalysisError = ValueError

compute_confusion_rates = _base.compute_confusion_rates
compute_old_balanced_accuracy = _base.compute_old_balanced_accuracy
compute_score_metrics = _base.compute_score_metrics
strict_pareto_deltas = _base.strict_pareto_deltas
decide_verdict = _base.decide_verdict
evaluate_resource_gate = _base.evaluate_resource_gate
validate_truth_binding = _base.validate_truth_binding
validate_per_old_class_join = _base.validate_per_old_class_join


def _finite(value: Any, label: str, *, lower: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise D92TPCEHard11AnalysisError(f"{label} is not finite")
    result = float(value)
    if lower is not None and result < lower:
        raise D92TPCEHard11AnalysisError(f"{label} below lower bound")
    return result


def evaluate_component_fit_reduction_gate(candidate_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate the frozen D92 component-fit proxy, excluding K1 liveness."""
    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        if str(row.get("outer_role", "performance")) == "liveness" or int(row.get("k_shot", 0)) <= 2:
            continue
        k_shot = int(row.get("k_shot", 0))
        fit_count = int(row.get("fit_count", row.get("after_total_component_fit_count", -1)))
        actual = int(row.get("actual_fit_count", row.get("after_actual_component_fit_count", -1)))
        baseline = 8 * (k_shot + 1)
        reduction = 1.0 - float(fit_count) / float(baseline) if baseline > 0 else -math.inf
        rows.append({"outer_key": str(row.get("outer_key", "")), "k_shot": k_shot, "fit_count": fit_count, "actual_fit_count": actual, "d92_component_fit_baseline": baseline, "reduction_fraction_vs_d92": reduction})
    passed = bool(rows) and all(item["fit_count"] == FIT_GATE["k_gt_2_total"] and item["actual_fit_count"] == FIT_GATE["k_gt_2_actual"] and item["reduction_fraction_vs_d92"] >= float(RESOURCE_GATE["component_fit_reduction_min_fraction_vs_d92"]) for item in rows)
    return {"passed": passed, "rows": rows, "min_reduction_fraction": min((item["reduction_fraction_vs_d92"] for item in rows), default=-math.inf), "proxy": "D92_FULL_TWO_STATE_COMPONENT_FIT_COUNT_8*(K+1)"}


def _fit_resource(job_root: str | Path, k_shot: int, *, baseline: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path = Path(job_root) / "diag" / "after" / "fit_audit.json"
    try:
        _validate_fit_audit(path, k_shot=k_shot)
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as error:  # noqa: BLE001
        raise D92TPCEHard11AnalysisError(f"TPCE fit/resource receipt invalid: {path}") from error
    if not isinstance(rows, list) or len(rows) != 3:
        raise D92TPCEHard11AnalysisError("TPCE fit/resource scene closure drift")
    active = int(k_shot) > 2
    prefix = "d92_e0d_tpce_"
    wall: list[float] = []
    peak: list[float] = []
    query: set[int] = set()
    state: set[int] = set()
    fit_counts: set[int] = set()
    actual_counts: set[int] = set()
    for row in rows:
        resource = row.get("after_registration_resource")
        if not isinstance(resource, Mapping):
            raise D92TPCEHard11AnalysisError("TPCE registration resource receipt missing")
        wall.append(_finite(resource.get("registration_wall_time_ns"), "registration wall", lower=0.0))
        peak.append(_finite(resource.get("registration_incremental_peak_working_set_bytes"), "registration peak", lower=0.0))
        query.add(int(_finite(row.get("query_macs"), "query MACs", lower=0.0)))
        state.add(int(_finite(row.get("after_state_bytes"), "state bytes", lower=0.0)))
        fit_counts.add(int(row.get("after_total_component_fit_count", -1)))
        inventory = row.get("after_actual_component_inventory", {})
        actual_counts.add(int(inventory.get("actual_component_fit_count", -1)) if isinstance(inventory, Mapping) else -1)
        if row.get(prefix + "persistent_state_bytes_delta") not in ({0} if active else {None}):
            raise D92TPCEHard11AnalysisError("TPCE persistent state delta drift")
    if len(query) != 1 or len(state) != 1 or len(fit_counts) != 1 or len(actual_counts) != 1:
        raise D92TPCEHard11AnalysisError("TPCE receipt values differ across scenes")
    expected = (2, 1, "full_only") if active else (3, 3, "full_only")
    if (next(iter(fit_counts)), next(iter(actual_counts)), rows[0].get("after_registered_d_mode_effective")) != expected:
        raise D92TPCEHard11AnalysisError("TPCE fit inventory drift")
    return {"fit_count": next(iter(fit_counts)), "actual_fit_count": next(iter(actual_counts)), "registered_d_mode": "full_only", "query_macs": next(iter(query)), "state_bytes": next(iter(state)), "registration_wall_time_ns": statistics.median(wall), "registration_incremental_peak_working_set_bytes": statistics.median(peak), "support_macs": statistics.median([_finite(row.get(prefix + "support_macs_upper_bound"), "support MACs", lower=0.0) for row in rows]) if active else None, "support_transient_bytes": statistics.median([_finite(row.get(prefix + "support_transient_bytes_upper_bound"), "support transient bytes", lower=0.0) for row in rows]) if active else None, "persistent_state_bytes_delta": 0 if active else None}


def analyze_d92_tpce_hard11(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run the proven truth-last analyzer under TPCE identity.

    The shared analyzer is kept as the single implementation of paired metric,
    per-old, scene and verdict logic. TPCE receipt validation is performed by
    ``_fit_resource`` above; artifact schemas are expected to be TPCE-native.
    """
    # The full artifact join is intentionally delegated only after replacing
    # its frozen identity/resource hooks; no query truth is opened here.
    patched = {
        "ARM_ID": ARM_ID,
        "CANDIDATE_ID": CANDIDATE_ID,
        "CANONICAL_SELECTION_SHA256": CANONICAL_SELECTION_SHA256,
        "RESOURCE_GATE": RESOURCE_GATE,
        "STRICT_PARETO_THRESHOLDS": STRICT_PARETO_THRESHOLDS,
        "HISTORICAL_BASELINE_PATH": HISTORICAL_BASELINE_PATH,
        "HISTORICAL_BASELINE_SHA256": HISTORICAL_BASELINE_SHA256,
        "HISTORICAL_PER_OLD_CLASS_PATH": HISTORICAL_PER_OLD_CLASS_PATH,
        "HISTORICAL_PER_OLD_CLASS_SHA256": HISTORICAL_PER_OLD_CLASS_SHA256,
        "_fit_resource": _fit_resource,
        "validate_method_lock": validate_method_lock,
        "validate_hard11_manifest": validate_hard11_manifest,
        "HARD11_ROWS": HARD11_ROWS,
        "LIVENESS_OUTER_KEY": LIVENESS_OUTER_KEY,
        "SMOKE_OUTER_KEY": SMOKE_OUTER_KEY,
    }
    old = {name: getattr(_base, name) for name in patched}
    old_read_json = _base._read_json

    def _read_json_with_tpce_schema(path: Path) -> dict[str, Any]:
        payload = old_read_json(path)
        if path.name == "job_receipt.json" and payload.get("schema") == "cvs.phase2.d92_tpce_hard11.job_receipt.v1":
            payload = dict(payload)
            payload["schema"] = "cvs.phase2.d92_pareto_distill_hard11.job_receipt.v1"
        return payload
    try:
        for name, value in patched.items():
            setattr(_base, name, value)
        _base._read_json = _read_json_with_tpce_schema
        result = _base.analyze_d92_pareto_distill_hard11(*args, **kwargs)
    except ValueError as error:
        raise D92TPCEHard11AnalysisError(str(error)) from error
    finally:
        for name, value in old.items():
            setattr(_base, name, value)
        _base._read_json = old_read_json
    result["schema"] = "cvs.phase2.d92_tpce_hard11.analysis.v1"
    return result


analyze_tpce_hard11 = analyze_d92_tpce_hard11


__all__ = [
    "D92TPCEHard11AnalysisError", "EIGHT_PARETO_METRICS", "PARETO_METRICS", "HISTORICAL_BASELINE_SHA256", "HISTORICAL_PER_OLD_CLASS_SHA256", "analyze_d92_tpce_hard11", "analyze_tpce_hard11", "compute_confusion_rates", "compute_old_balanced_accuracy", "compute_score_metrics", "decide_verdict", "evaluate_component_fit_reduction_gate", "evaluate_resource_gate", "strict_pareto_deltas", "validate_per_old_class_join", "validate_truth_binding", "_fit_resource",
]
