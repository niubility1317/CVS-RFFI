"""Independent truth-last analyzer for D92 E0_FULL_CSOAS Hard9+K1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cvsrffi import stage2_d92_pareto_distill_hard11_analysis as _base
from cvsrffi.stage2_d92_csoas_hard10 import *  # noqa: F401,F403
from cvsrffi.stage2_d92_csoas_hard10 import _sha256_file

EIGHT_PARETO_METRICS = ("h_old_new", "old_balanced_accuracy", "c_old_acc", "old_floor", "seen_new_acc", "average_forgetting", "new_to_old_rate", "old_to_new_rate")
PARETO_METRICS = EIGHT_PARETO_METRICS
QUERY_ZERO_FIELDS = tuple(_base.QUERY_ZERO_FIELDS)
_TOLERANCE = 1.0e-12


class D92CSOASHard10AnalysisError(ValueError):
    """Raised when candidate/truth/receipt closure is incomplete or detached."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise D92CSOASHard10AnalysisError(f"missing JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92CSOASHard10AnalysisError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise D92CSOASHard10AnalysisError(f"JSON object required: {path}")
    return value


def _sha(path: Path) -> str:
    return _sha256_file(path)


def _finite(value: Any, label: str, *, lower: float | None = None, upper: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise D92CSOASHard10AnalysisError(f"non-numeric {label}") from error
    if not math.isfinite(result) or (lower is not None and result < lower) or (upper is not None and result > upper):
        raise D92CSOASHard10AnalysisError(f"out-of-range {label}")
    return result


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise D92CSOASHard10AnalysisError("empty mean")
    return float(statistics.fmean(items))


def compute_confusion_rates(score: Mapping[str, Any]) -> dict[str, float]:
    try:
        return _base.compute_confusion_rates(score)
    except ValueError as error:
        raise D92CSOASHard10AnalysisError(str(error)) from error


def compute_old_balanced_accuracy(by_tx: Mapping[str, Any]) -> float:
    try:
        return _base.compute_old_balanced_accuracy(by_tx)
    except ValueError as error:
        raise D92CSOASHard10AnalysisError(str(error)) from error


def compute_score_metrics(score: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return _base.compute_score_metrics(score)
    except ValueError as error:
        raise D92CSOASHard10AnalysisError(str(error)) from error


def validate_per_old_class_join(rows: Sequence[Mapping[str, Any]], raw_score: Mapping[str, Any], *, outer_key: str) -> dict[str, dict[str, float]]:
    try:
        return _base.validate_per_old_class_join(rows, raw_score, outer_key=outer_key)
    except ValueError as error:
        raise D92CSOASHard10AnalysisError(str(error)) from error


def validate_truth_binding(score: Mapping[str, Any], receipt: Mapping[str, Any], job: Mapping[str, Any], truth_path: str | Path) -> str:
    try:
        return _base.validate_truth_binding(score, receipt, job, truth_path)
    except ValueError as error:
        raise D92CSOASHard10AnalysisError(str(error)) from error


def strict_pareto_deltas(candidate: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, float]:
    return {metric: float(candidate[metric]) - float(baseline[metric]) for metric in EIGHT_PARETO_METRICS}


def _strict_ok(deltas: Mapping[str, float]) -> bool:
    return all((float(deltas[m]) > _TOLERANCE if m in EIGHT_PARETO_METRICS[:5] else float(deltas[m]) < -_TOLERANCE) for m in EIGHT_PARETO_METRICS)


def _magnitude_ok(deltas: Mapping[str, float]) -> bool:
    return all((float(deltas[m]) >= STRICT_PARETO_THRESHOLDS[m] - _TOLERANCE if m in EIGHT_PARETO_METRICS[:5] else float(deltas[m]) <= STRICT_PARETO_THRESHOLDS[m] + _TOLERANCE) for m in EIGHT_PARETO_METRICS)


def evaluate_resource_gate(candidate_rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]], *, query_state_exact: bool) -> dict[str, Any]:
    if not candidate_rows or len(candidate_rows) != len(baseline_rows):
        raise D92CSOASHard10AnalysisError("resource row closure drift")
    walls, ratios, peaks = [], [], []
    for candidate, baseline in zip(candidate_rows, baseline_rows):
        wall = _finite(candidate.get("registration_wall_time_ns"), "candidate registration wall", lower=0)
        base_wall = _finite(baseline.get("registration_wall_time_ns"), "E0 registration wall", lower=0)
        if base_wall <= 0:
            raise D92CSOASHard10AnalysisError("E0 registration wall is zero")
        peak = _finite(candidate.get("registration_incremental_peak_working_set_bytes"), "candidate registration peak", lower=0)
        base_peak = _finite(baseline.get("registration_incremental_peak_working_set_bytes"), "E0 registration peak", lower=0)
        walls.append(wall); ratios.append(wall / base_wall); peaks.append(peak - base_peak)
    p90 = sorted(walls)[max(0, math.ceil(0.9 * len(walls)) - 1)]
    ratio_median = float(statistics.median(ratios))
    peak_max = max(peaks)
    hard_limits = p90 <= RESOURCE_GATE["registration_wall_p90_max_ns"] and ratio_median <= RESOURCE_GATE["registration_wall_ratio_max"] and peak_max <= RESOURCE_GATE["registration_peak_delta_max_bytes"]
    target_limits = p90 <= RESOURCE_GATE["registration_wall_p90_target_max_ns"] and ratio_median <= RESOURCE_GATE["registration_wall_ratio_target_max"]
    return {"passed": bool(query_state_exact and hard_limits), "query_state_exact": bool(query_state_exact), "hard_limits_passed": hard_limits, "hard_passed": bool(query_state_exact and hard_limits), "target_limits_passed": target_limits, "target_passed": bool(query_state_exact and hard_limits and target_limits), "wall_p90": p90, "wall_ratio_median": ratio_median, "wall_ratio_p90": ratio_median, "peak_delta_p90_bytes": sorted(peaks)[max(0, math.ceil(0.9 * len(peaks)) - 1)], "peak_delta_max_bytes": peak_max, "candidate_wall_p90_ns": p90}


def _candidate_component_fit_count(row: Mapping[str, Any]) -> int:
    inventory = row.get("actual_component_inventory")
    value = (
        inventory.get("actual_component_fit_count")
        if isinstance(inventory, Mapping)
        else row.get("actual_component_fit_count", row.get("actual_fit_count"))
    )
    if isinstance(value, bool) or value is None:
        raise D92CSOASHard10AnalysisError("candidate component-fit receipt missing")
    count = int(value)
    if count < 0 or float(count) != float(value):
        raise D92CSOASHard10AnalysisError("candidate component-fit receipt invalid")
    return count


def evaluate_component_fit_reduction_gate(candidate_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in candidate_rows if str(row.get("outer_role", "performance")) != "liveness"]
    if not rows:
        raise D92CSOASHard10AnalysisError("component-fit row closure drift")
    evidence = []
    for row in rows:
        value = row.get("fit_count", row.get("total_component_fit_count"))
        if isinstance(value, bool) or value is None or int(value) <= 0 or float(int(value)) != float(value):
            raise D92CSOASHard10AnalysisError("candidate total component-fit receipt invalid")
        k_shot = int(row.get("k_shot", -1))
        baseline = 3 if k_shot <= 2 else 8 * (k_shot + 1)
        evidence.append({"outer_key": str(row.get("outer_key", "")), "k_shot": k_shot, "candidate_actual_component_fit_count": _candidate_component_fit_count(row), "candidate_total_component_fit_count": int(value), "original_d92_total_component_fit_count": baseline, "reduction_fraction_vs_d92": 1.0 - float(int(value)) / float(baseline), "baseline_source": "frozen_d92_full_two_state_component_fit_count_8*(K+1)"})
    reductions = [float(row["reduction_fraction_vs_d92"]) for row in evidence]
    return {"passed": all(value >= RESOURCE_GATE["component_fit_reduction_min_fraction_vs_d92"] - _TOLERANCE for value in reductions), "evidence_complete": True, "threshold": RESOURCE_GATE["component_fit_reduction_min_fraction_vs_d92"], "min_reduction_fraction": min(reductions), "mean_reduction_fraction": _mean(reductions), "rows": evidence, "baseline": RESOURCE_GATE["component_fit_baseline"]}


def decide_verdict(gate_state: Mapping[str, bool]) -> str:
    required = ("complete_artifact_closure", "performance_outer_closure", "all_strict_pareto", "all_magnitude", "stability", "resource_integrity", "resource_hard", "resource_target", "compute_reduction")
    if any(name not in gate_state for name in required):
        return "REJECT_ROUTE"
    if not all(bool(gate_state[name]) for name in ("complete_artifact_closure", "performance_outer_closure", "all_strict_pareto", "stability", "resource_integrity", "resource_hard", "compute_reduction")):
        return "REJECT_ROUTE"
    if not bool(gate_state["all_magnitude"]) or not bool(gate_state["resource_target"]):
        return "REVISE_ONCE"
    return "ADVANCE_TO_TARGET125_CANDIDATE"


def _key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return str(row["receiver"]), int(row["seed"]), int(row["k_shot"]), int(row.get("new_class_count", row.get("new_count")))


def _fit_resource(job_root: Path, k_shot: int) -> dict[str, Any]:
    rows = json.loads((job_root / "diag" / "after" / "fit_audit.json").read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or len(rows) != 3 or {str(row.get("scenario")) for row in rows} != set(SCENES):
        raise D92CSOASHard10AnalysisError("fit/resource scene closure drift")
    totals, actuals, modes, macs, states, walls, peaks = set(), set(), set(), set(), set(), [], []
    for row in rows:
        if any(row.get(field) is not False for field in QUERY_ZERO_FIELDS + tuple(CSOAS_QUERY_ZERO_FIELDS)):
            raise D92CSOASHard10AnalysisError("query access is not zero")
        active = int(k_shot) > 2
        expected = (2, 1, "csoas_full") if active else (3, 3, "d92_full_alias")
        inventory = row.get("after_actual_component_inventory", {})
        totals.add(int(row.get("after_total_component_fit_count", -1))); actuals.add(int(inventory.get("actual_component_fit_count", -1))); modes.add(str(row.get("after_registered_d_mode_effective", "")))
        if (int(row.get("after_total_component_fit_count", -1)), int(inventory.get("actual_component_fit_count", -1)), str(row.get("after_registered_d_mode_effective", ""))) != expected:
            raise D92CSOASHard10AnalysisError("fit/resource count closure drift")
        prefix = "d92_csoas_"
        if active:
            if row.get(prefix + "active") is not True or row.get(prefix + "fallback_active") is not False or row.get(prefix + "fallback_reason") is not None or int(row.get(prefix + "candidate_attempt_fit_count", -1)) != 1 or int(row.get(prefix + "fallback_reference_fit_count", -1)) != 0:
                raise D92CSOASHard10AnalysisError("CSOAS active receipt drift")
        else:
            if row.get(prefix + "active") is not False or row.get(prefix + "fallback_active") is not False or row.get(prefix + "fallback_reason") != "K1_K2_EXACT_D92_FULL_ALIAS":
                raise D92CSOASHard10AnalysisError("K1 alias receipt drift")
        macs.add(int(row.get("query_macs", -1))); states.add(int(row.get("after_state_bytes", -1)))
        class_count = row.get("registered_class_count", row.get("class_count"))
        if isinstance(class_count, bool) or not isinstance(class_count, (int, float)) or int(class_count) <= 0 or float(int(class_count)) != float(class_count):
            raise D92CSOASHard10AnalysisError("registered-class receipt drift")
        if int(row.get("query_macs", -1)) != int(class_count) * 288:
            raise D92CSOASHard10AnalysisError("query MAC receipt drift")
        if int(row.get("after_state_bytes", -1)) <= 0:
            raise D92CSOASHard10AnalysisError("state-byte receipt drift")
        resource = row.get("after_registration_resource", {})
        walls.append(_finite(resource.get("registration_wall_time_ns"), "registration wall", lower=0)); peaks.append(_finite(resource.get("registration_incremental_peak_working_set_bytes"), "registration peak", lower=0))
    if len(macs) != 1 or len(states) != 1 or min(macs) < 0 or min(states) < 0:
        raise D92CSOASHard10AnalysisError("query/state closure drift")
    return {"fit_count": next(iter(totals)), "actual_fit_count": next(iter(actuals)), "registered_d_mode": next(iter(modes)), "query_macs": next(iter(macs)), "state_bytes": next(iter(states)), "registration_wall_time_ns": float(statistics.median(walls)), "registration_incremental_peak_working_set_bytes": float(statistics.median(peaks))}


def _scenario_rows(outer_key: str, candidate: Mapping[str, Any], baseline: Mapping[str, Any], *, receiver: str, slice_name: str) -> list[dict[str, Any]]:
    return _base._scenario_rows(outer_key, candidate, baseline, receiver=receiver, slice_name=slice_name)


def _group_stability(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    result = {}
    for key, items in sorted(grouped.items()):
        h = _mean(float(item["candidate_h_old_new"]) - float(item["e0_h_old_new"]) for item in items)
        n = _mean(float(item["candidate_seen_new_acc"]) - float(item["e0_seen_new_acc"]) for item in items)
        result[key] = {"row_count": len(items), "mean_delta_h_old_new": h, "mean_delta_seen_new_acc": n, "passed": h >= -_TOLERANCE and n >= -_TOLERANCE}
    return result


def analyze_d92_csoas_hard10(matrix_manifest_path: str | Path, *, run_root: str | Path | None, method_lock_path: str | Path, baseline_paired_rows_path: str | Path = HISTORICAL_BASELINE_PATH, per_old_class_rows_path: str | Path = HISTORICAL_PER_OLD_CLASS_PATH, truth_sidecar_root: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(matrix_manifest_path).resolve(strict=True); manifest_sha = _sha(manifest_path); manifest = _read_json(manifest_path)
    lock_path = Path(method_lock_path).resolve(strict=True); lock_sha = _sha(lock_path); lock = _read_json(lock_path)
    try:
        validate_method_lock(lock); validate_hard10_manifest(manifest, expected_method_lock_sha256=lock_sha, require_package_hashes=True)
    except ValueError as error:
        raise D92CSOASHard10AnalysisError("matrix/method lock drift") from error
    root = Path(run_root or manifest["output_root"]).resolve(strict=True)
    if (root / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json").exists():
        raise D92CSOASHard10AnalysisError("systemic stop marker exists")
    baseline_path = Path(baseline_paired_rows_path).resolve(strict=True); per_old_path = Path(per_old_class_rows_path).resolve(strict=True)
    if str(baseline_path).replace("\\", "/").lower() != HISTORICAL_BASELINE_PATH.lower() or str(per_old_path).replace("\\", "/").lower() != HISTORICAL_PER_OLD_CLASS_PATH.lower():
        raise D92CSOASHard10AnalysisError("historical baseline path drift")
    baseline_rows = _base._read_csv(baseline_path, HISTORICAL_BASELINE_SHA256, expected_rows=125)
    per_old_source = _base._read_csv(per_old_path, HISTORICAL_PER_OLD_CLASS_SHA256, expected_rows=750)
    baseline_by_key = {_key(row): row for row in baseline_rows}; per_old_by_outer: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in per_old_source: per_old_by_outer[str(row.get("outer_key"))].append(row)
    paired_rows: list[dict[str, Any]] = []; per_old_rows: list[dict[str, Any]] = []; scenario_rows: list[dict[str, Any]] = []
    raw_specs = lock["historical_baseline"]["e0_raw_scores"]
    for job in manifest["jobs"]:
        key = _key(job); outer = str(job["outer_key"])
        if key not in baseline_by_key or outer not in raw_specs:
            raise D92CSOASHard10AnalysisError("job/baseline identity mismatch")
        raw_spec = raw_specs[outer]; raw_path = Path(str(raw_spec["path"])).resolve(strict=True)
        if _sha(raw_path) != str(raw_spec["sha256"]).lower():
            raise D92CSOASHard10AnalysisError("E0 raw score SHA drift")
        job_root = root / "jobs" / outer / ARM_ID; receipt = _read_json(job_root / "job_receipt.json"); score_path = job_root / "scorer" / "diag_cosine_score.json"; score = _read_json(score_path)
        before = job_root / "diag" / "before" / "prediction_artifact.npz"; after = job_root / "diag" / "after" / "prediction_artifact.npz"
        if receipt.get("schema") != "cvs.phase2.d92_csoas_hard10.job_receipt.v1" or receipt.get("status") != "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE" or receipt.get("job_id") != job["job_id"] or receipt.get("outer_key") != outer or receipt.get("arm_id") != ARM_ID or receipt.get("candidate") != CANDIDATE_ID or receipt.get("matrix_manifest_sha256") != manifest_sha or receipt.get("method_lock_sha256") != lock_sha or receipt.get("selection_sha256") != CANONICAL_SELECTION_SHA256 or receipt.get("truth_sidecar_exposed_to_predictor") is not False or receipt.get("query_truth_fed_back_to_predictor") is not False or receipt.get("query_truth_joined_only_after_immutable_predictions") is not True or receipt.get("prediction_and_scorer_processes_isolated") is not True or receipt.get("before_prediction_sha256") != _sha(before) or receipt.get("after_prediction_sha256") != _sha(after) or receipt.get("score_sha256") != _sha(score_path):
            raise D92CSOASHard10AnalysisError("job receipt binding drift")
        if _base._prediction_closure_status(job_root / "diag") != ("closed", "closed"):
            raise D92CSOASHard10AnalysisError("prediction closure drift")
        if score.get("candidate") != CANDIDATE_ID or score.get("query_truth_fed_back_to_predictor") is not False or score.get("query_truth_joined_only_after_immutable_predictions") is not True or score.get("before_prediction_sha256") != receipt.get("before_prediction_sha256") or score.get("after_prediction_sha256") != receipt.get("after_prediction_sha256"):
            raise D92CSOASHard10AnalysisError("score binding drift")
        truth_path = Path(str(job["truth_sidecar"])) if truth_sidecar_root is None else Path(truth_sidecar_root).resolve() / "jobs" / outer / "offline" / "scorer" / "truth_sidecar.json"
        validate_truth_binding(score, receipt, job, truth_path)
        candidate = compute_score_metrics(score); baseline = compute_score_metrics(_read_json(raw_path)); historical = validate_per_old_class_join(per_old_by_outer.get(outer, []), _read_json(raw_path), outer_key=outer); resource = _fit_resource(job_root, int(job["k_shot"]))
        base_row = baseline_by_key[key]
        row = {"outer_key": outer, "outer_role": job["outer_role"], "receiver": job["receiver"], "seed": job["seed"], "k_shot": job["k_shot"], "new_class_count": job["new_class_count"], "slice": f"K{job['k_shot']}_new{job['new_class_count']}", **resource}
        for metric in EIGHT_PARETO_METRICS: row[f"candidate_{metric}"] = candidate[metric]; row[f"e0_{metric}"] = baseline[metric]; row[f"delta_{metric}_vs_e0"] = candidate[metric] - baseline[metric]
        row.update({"full_only_query_macs": _finite(base_row.get("query_macs"), "E0 query MACs", lower=0), "full_only_state_bytes": _finite(base_row.get("state_bytes"), "E0 state bytes", lower=0), "full_only_registration_wall_time_ns": _finite(base_row.get("registration_wall_time_ns"), "E0 registration wall", lower=0), "full_only_registration_peak_working_set_bytes": _finite(base_row.get("registration_incremental_peak_working_set_bytes"), "E0 registration peak", lower=0)})
        row["registration_wall_ratio"] = row["registration_wall_time_ns"] / row["full_only_registration_wall_time_ns"]; row["registration_peak_delta_bytes"] = row["registration_incremental_peak_working_set_bytes"] - row["full_only_registration_peak_working_set_bytes"]
        paired_rows.append(row)
        for tx, values in historical.items(): per_old_rows.append({"outer_key": outer, "tx": tx, "candidate_accuracy": candidate["old_class_accuracy"][tx], "e0_accuracy": values["e0_accuracy"], "delta_accuracy": candidate["old_class_accuracy"][tx] - values["e0_accuracy"], "historical_baseline_accuracy": values["historical_baseline_accuracy"], "historical_delta_accuracy": values["historical_delta_accuracy"]})
        scenario_rows.extend(_scenario_rows(outer, score, _read_json(raw_path), receiver=str(job["receiver"]), slice_name=f"K{job['k_shot']}_new{job['new_class_count']}"))
    if len(paired_rows) != 10 or len(per_old_rows) != 60 or len(scenario_rows) != 30:
        raise D92CSOASHard10AnalysisError("Hard9 result row closure drift")
    performance = [row for row in paired_rows if row["outer_role"] == "performance"]; liveness = [row for row in paired_rows if row["outer_role"] == "liveness"]
    deltas = {metric: _mean(row[f"delta_{metric}_vs_e0"] for row in performance) for metric in EIGHT_PARETO_METRICS}; strict = _strict_ok(deltas); magnitude = _magnitude_ok(deltas)
    direction_counts = {metric: sum((row[f"delta_{metric}_vs_e0"] < -_TOLERANCE if metric in EIGHT_PARETO_METRICS[5:] else row[f"delta_{metric}_vs_e0"] > _TOLERANCE) for row in performance) for metric in EIGHT_PARETO_METRICS}
    by_tx: dict[str, list[float]] = defaultdict(list); by_outer: dict[str, list[float]] = defaultdict(list)
    for item in per_old_rows:
        if item["outer_key"] in {row["outer_key"] for row in performance}: by_tx[str(item["tx"])].append(float(item["delta_accuracy"])); by_outer[str(item["outer_key"])].append(float(item["delta_accuracy"]))
    per_old_summary = {tx: {"row_count": len(values), "mean_delta_accuracy": _mean(values), "min_delta_accuracy": min(values), "nondecrease_count": sum(value >= -0.01 for value in values)} for tx, values in sorted(by_tx.items())}
    per_outer_summary = {outer: {"row_count": len(values), "min_delta_accuracy": min(values), "nondecrease_count": sum(value >= -_TOLERANCE for value in values), "passed": len(values) == 6 and min(values) >= -0.01 and sum(value >= -_TOLERANCE for value in values) >= 5} for outer, values in sorted(by_outer.items())}
    per_old_stability = len(per_old_summary) == 6 and all(row["row_count"] == 9 and row["min_delta_accuracy"] >= -0.01 for row in per_old_summary.values()) and len(per_outer_summary) == 9 and all(row["passed"] for row in per_outer_summary.values())
    scene_perf = [row for row in scenario_rows if row["outer_key"] in {item["outer_key"] for item in performance}]; by_receiver = _group_stability(scene_perf, "receiver"); by_slice = _group_stability(scene_perf, "slice"); by_scene = _group_stability(scene_perf, "scenario"); group_stability = all(row["passed"] for groups in (by_receiver, by_slice, by_scene) for row in groups.values())
    stability = all(direction_counts[m] >= 8 for m in EIGHT_PARETO_METRICS) and per_old_stability and group_stability
    query_exact = all(int(row["query_macs"]) == int(row["full_only_query_macs"]) and int(row["state_bytes"]) == int(row["full_only_state_bytes"]) for row in performance)
    resource_eval = evaluate_resource_gate(performance, [{"registration_wall_time_ns": row["full_only_registration_wall_time_ns"], "registration_incremental_peak_working_set_bytes": row["full_only_registration_peak_working_set_bytes"]} for row in performance], query_state_exact=query_exact); compute_eval = evaluate_component_fit_reduction_gate(performance)
    gates = {"complete_artifact_closure": {"passed": True, "observed": {"paired": 10, "per_old": 60, "scene": 30}, "threshold": "10/60/30"}, "performance_outer_closure": {"passed": len(performance) == 9 and len(liveness) == 1, "observed": f"{len(performance)}+{len(liveness)}", "threshold": "9+1"}, "all_strict_pareto": {"passed": strict, "observed": deltas, "threshold": "strict directions"}, "all_magnitude": {"passed": magnitude, "observed": deltas, "threshold": STRICT_PARETO_THRESHOLDS}, "stability": {"passed": stability, "observed": {"direction_counts": direction_counts, "per_old_class": per_old_stability, "per_outer_old": per_outer_summary, "by_receiver": by_receiver, "by_slice": by_slice, "by_scene": by_scene}, "threshold": "paired/group/old-class stability"}, "resource_integrity": {"passed": query_exact, "observed": {"query_exact": query_exact}, "threshold": "query MACs/state bytes equal to E0"}, "resource_hard": {"passed": resource_eval["hard_limits_passed"], "observed": resource_eval, "threshold": RESOURCE_GATE}, "resource_target": {"passed": resource_eval["target_limits_passed"], "observed": resource_eval, "threshold": {"wall_p90_ns": RESOURCE_GATE["registration_wall_p90_target_max_ns"], "wall_ratio_median": RESOURCE_GATE["registration_wall_ratio_target_max"]}}, "compute_reduction": {"passed": compute_eval["passed"], "observed": compute_eval, "threshold": RESOURCE_GATE["component_fit_reduction_min_fraction_vs_d92"]}}
    gate_state = {name: bool(value["passed"]) for name, value in gates.items()}; verdict = decide_verdict(gate_state)
    aggregate = {"row_count": 10, "performance_row_count": 9, "liveness_row_count": 1, **{f"candidate_mean_{metric}": _mean(row[f"candidate_{metric}"] for row in performance) for metric in EIGHT_PARETO_METRICS}, **{f"e0_mean_{metric}": _mean(row[f"e0_{metric}"] for row in performance) for metric in EIGHT_PARETO_METRICS}, **{f"mean_delta_{metric}_vs_e0": deltas[metric] for metric in EIGHT_PARETO_METRICS}, "registration_wall_p90_ns": resource_eval["wall_p90"], "registration_wall_ratio_median": resource_eval["wall_ratio_median"], "registration_peak_delta_max_bytes": resource_eval["peak_delta_max_bytes"], "component_fit_reduction_min_fraction_vs_d92": compute_eval["min_reduction_fraction"]}
    return {"schema": "cvs.phase2.d92_csoas_hard10.analysis.v1", "status": "ANALYZED", "claim_scope": CLAIM_SCOPE, "matrix_manifest_sha256": manifest_sha, "method_lock_sha256": lock_sha, "selection_sha256": CANONICAL_SELECTION_SHA256, "baseline": {"paired_rows_path": str(baseline_path), "paired_rows_sha256": HISTORICAL_BASELINE_SHA256, "per_old_class_rows_path": str(per_old_path), "per_old_class_rows_sha256": HISTORICAL_PER_OLD_CLASS_SHA256}, "aggregate": aggregate, "paired_rows": paired_rows, "per_old_class_rows": per_old_rows, "per_old_class_summary": per_old_summary, "scenario_rows": scenario_rows, "by_receiver": by_receiver, "by_slice": by_slice, "by_scene": by_scene, "liveness_rows": liveness, "gates": gates, "gate_state": gate_state, "all_gates_pass": verdict == "ADVANCE_TO_TARGET125_CANDIDATE", "verdict": verdict}


analyze_csoas_hard10 = analyze_d92_csoas_hard10

__all__ = ["D92CSOASHard10AnalysisError", "EIGHT_PARETO_METRICS", "PARETO_METRICS", "analyze_d92_csoas_hard10", "analyze_csoas_hard10", "compute_confusion_rates", "compute_old_balanced_accuracy", "compute_score_metrics", "decide_verdict", "evaluate_component_fit_reduction_gate", "evaluate_resource_gate", "strict_pareto_deltas", "validate_per_old_class_join", "validate_truth_binding"]
