"""Independent post-prediction analyzer for D92 NewGuard Hard11."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cvsrffi.stage2_d92_newguard_hard11 import (
    ARM_ID,
    CANDIDATE_ID,
    CANONICAL_SELECTION_SHA256,
    HARD11_ROWS,
    HISTORICAL_BASELINE_PATH,
    HISTORICAL_BASELINE_SHA256,
    HISTORICAL_PER_OLD_CLASS_PATH,
    HISTORICAL_PER_OLD_CLASS_SHA256,
    LIVENESS_OUTER_KEY,
    RESOURCE_GATE,
    SCENES,
    SHARD_COUNT,
    SMOKE_OUTER_KEY,
    STRICT_PARETO_THRESHOLDS,
    validate_hard11_manifest,
    validate_method_lock,
)
from scripts.run_d92_newguard_hard11 import QUERY_ZERO_FIELDS, _prediction_closure_status


EIGHT_PARETO_METRICS = ("h_old_new", "old_balanced_accuracy", "c_old_acc", "old_floor", "seen_new_acc", "average_forgetting", "new_to_old_rate", "old_to_new_rate")
PARETO_METRICS = EIGHT_PARETO_METRICS
HISTORICAL_BASELINE_SHA256 = HISTORICAL_BASELINE_SHA256
HISTORICAL_PER_OLD_CLASS_SHA256 = HISTORICAL_PER_OLD_CLASS_SHA256
_TOLERANCE = 1.0e-12


class D92NewGuardHard11AnalysisError(ValueError):
    """Raised when frozen evidence is incomplete, detached or malformed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise D92NewGuardHard11AnalysisError(f"missing JSON artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92NewGuardHard11AnalysisError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise D92NewGuardHard11AnalysisError(f"JSON artifact is not an object: {path}")
    return payload


def _read_csv(path: Path, expected_sha256: str, *, expected_rows: int | None = None) -> list[dict[str, str]]:
    if not path.is_file() or path.is_symlink() or _sha256(path) != expected_sha256:
        raise D92NewGuardHard11AnalysisError(f"frozen CSV identity drift: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if expected_rows is not None and len(rows) != expected_rows:
        raise D92NewGuardHard11AnalysisError(f"frozen CSV row closure drift: {len(rows)}")
    return rows


def _finite(value: Any, label: str, *, lower: float | None = None, upper: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise D92NewGuardHard11AnalysisError(f"non-numeric {label}") from error
    if not math.isfinite(result) or (lower is not None and result < lower) or (upper is not None and result > upper):
        raise D92NewGuardHard11AnalysisError(f"out-of-range {label}")
    return result


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise D92NewGuardHard11AnalysisError("empty mean")
    return float(statistics.fmean(values))


def _rate(after: Mapping[str, Any], name: str) -> float:
    direct = after.get(name)
    if direct is not None:
        return _finite(direct, name, lower=0.0, upper=1.0)
    by_scene = after.get("by_scenario")
    if isinstance(by_scene, Mapping):
        values = [_finite(scene.get(name), name, lower=0.0, upper=1.0) for scene in by_scene.values() if isinstance(scene, Mapping) and scene.get(name) is not None]
        if values:
            return _mean(values)
    raise D92NewGuardHard11AnalysisError(f"missing {name}")


def compute_old_balanced_accuracy(by_tx: Mapping[str, Any]) -> float:
    values = []
    for tx, row in by_tx.items():
        if not isinstance(row, Mapping) or row.get("role") != "target_old":
            continue
        values.append(_finite(row.get("accuracy"), f"old accuracy {tx}", lower=0.0, upper=1.0))
    if len(values) != 6:
        raise D92NewGuardHard11AnalysisError(f"expected six old classes, got {len(values)}")
    return _mean(values)


def compute_score_metrics(score: Mapping[str, Any]) -> dict[str, Any]:
    before, after = score.get("before"), score.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise D92NewGuardHard11AnalysisError("score state surface missing")
    by_tx = after.get("by_tx")
    if not isinstance(by_tx, Mapping):
        raise D92NewGuardHard11AnalysisError("score by_tx surface missing")
    old_acc = {str(tx): _finite(row.get("accuracy"), f"old accuracy {tx}", lower=0.0, upper=1.0) for tx, row in by_tx.items() if isinstance(row, Mapping) and row.get("role") == "target_old"}
    if len(old_acc) != 6:
        raise D92NewGuardHard11AnalysisError("old class closure drift")
    new_to_old_rate = _rate(after, "new_to_old_rate")
    old_to_new_rate = _rate(after, "old_to_new_rate")
    return {
        "h_old_new": _finite(after.get("h_old_new"), "H_old_new", lower=0.0, upper=1.0),
        "old_balanced_accuracy": compute_old_balanced_accuracy(by_tx),
        "c_old_acc": _finite(after.get("old_acc"), "c_old_acc", lower=0.0, upper=1.0),
        "old_floor": min(old_acc.values()),
        "seen_new_acc": _finite(after.get("seen_new_acc"), "seen_new_acc", lower=0.0, upper=1.0),
        "average_forgetting": (_finite(score.get("old_forgetting_pp"), "average_forgetting") / 100.0 if score.get("old_forgetting_pp") is not None else _finite(before.get("old_acc"), "before old accuracy", lower=0.0, upper=1.0) - _finite(after.get("old_acc"), "after old accuracy", lower=0.0, upper=1.0)),
        "new_to_old_rate": new_to_old_rate,
        "old_to_new_rate": old_to_new_rate,
        "new_to_old_error": new_to_old_rate,
        "old_to_new_error": old_to_new_rate,
        "old_class_accuracy": old_acc,
        "old_class_count": len(old_acc),
        "query_macs": int(after.get("query_macs", score.get("query_macs", -1))),
        "state_bytes": int(after.get("after_state_bytes", score.get("state_bytes", -1))),
    }


def strict_pareto_deltas(candidate: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, float]:
    aliases = {"new_to_old_rate": "new_to_old_error", "old_to_new_rate": "old_to_new_error"}
    def value(source: Mapping[str, float], metric: str) -> float:
        if metric in source:
            return float(source[metric])
        return float(source[aliases[metric]])
    return {metric: value(candidate, metric) - value(baseline, metric) for metric in EIGHT_PARETO_METRICS}


# Compatibility names used by neighboring Hard11 analyzers.
_score_metrics = compute_score_metrics
_metrics_from_score = compute_score_metrics


def _strict_ok(deltas: Mapping[str, float]) -> bool:
    return all((deltas[m] > _TOLERANCE if m not in {"average_forgetting", "new_to_old_rate", "old_to_new_rate"} else deltas[m] < -_TOLERANCE) for m in EIGHT_PARETO_METRICS)


def _magnitude_ok(deltas: Mapping[str, float]) -> bool:
    return all(deltas[m] >= STRICT_PARETO_THRESHOLDS[m] - _TOLERANCE if m not in {"average_forgetting", "new_to_old_rate", "old_to_new_rate"} else deltas[m] <= STRICT_PARETO_THRESHOLDS[m] + _TOLERANCE for m in EIGHT_PARETO_METRICS)


def decide_verdict(gate_state: Mapping[str, bool]) -> str:
    if not bool(gate_state.get("complete_artifact_closure")) or not bool(gate_state.get("performance_outer_closure")) or not bool(gate_state.get("all_strict_pareto")):
        return "REJECT_ROUTE"
    if not bool(gate_state.get("all_magnitude")):
        return "REVISE_ONCE"
    if not bool(gate_state.get("stability", True)) or not bool(gate_state.get("resources", True)):
        return "REVISE_ONCE"
    return "ADVANCE_TO_TARGET125_CANDIDATE"


def _key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return str(row["receiver"]), int(row["seed"]), int(row["k_shot"]), int(row.get("new_class_count", row.get("new_count")))


def _fit_resource(job_root: Path, k_shot: int, *, baseline: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path = job_root / "diag" / "after" / "fit_audit.json"
    if not path.is_file() or path.is_symlink():
        raise D92NewGuardHard11AnalysisError("after fit audit missing")
    try:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92NewGuardHard11AnalysisError("after fit audit invalid") from error
    if not isinstance(rows, list) or len(rows) != 3:
        raise D92NewGuardHard11AnalysisError("after fit audit scene closure drift")
    totals, actuals, macs, states, modes, walls, peaks = set(), set(), set(), set(), [], [], []
    for row in rows:
        if any(row.get(field) is not False for field in QUERY_ZERO_FIELDS):
            raise D92NewGuardHard11AnalysisError("query access is not zero")
        totals.add(int(row.get("after_total_component_fit_count", row.get("fit_count", -1))))
        inventory = row.get("after_actual_component_inventory", {})
        actuals.add(int(inventory.get("actual_component_fit_count", row.get("actual_fit_count", -1))) if isinstance(inventory, Mapping) else -1)
        macs.add(int(row.get("query_macs", -1))); states.add(int(row.get("after_state_bytes", row.get("state_bytes", -1))))
        modes.add(str(row.get("after_registered_d_mode_effective", row.get("registered_d_mode", ""))))
        resource = row.get("after_registration_resource", {})
        walls.append(_finite(resource.get("registration_wall_time_ns"), "registration wall", lower=0.0)); peaks.append(_finite(resource.get("registration_incremental_peak_working_set_bytes"), "registration peak", lower=0.0))
    # NewGuard's K>2 gate is the exact 2/1 inventory (one FULL registration
    # fit plus the immutable base component); K1 remains the real D92 alias
    # inventory and is checked as 3/3.
    expected = (3, 3) if int(k_shot) <= 2 else (2, 1)
    if totals != {expected[0]} or actuals != {expected[1]} or len(macs) != 1 or len(states) != 1 or min(macs) < 0 or min(states) < 0:
        raise D92NewGuardHard11AnalysisError("fit/resource count closure drift")
    return {"fit_count": expected[0], "actual_fit_count": expected[1], "registered_d_mode": sorted(modes)[0] if modes else "", "query_macs": next(iter(macs)), "state_bytes": next(iter(states)), "registration_wall_time_ns": float(statistics.median(walls)), "registration_incremental_peak_working_set_bytes": float(statistics.median(peaks))}


def _baseline_raw_metrics(path: Path) -> dict[str, Any]:
    return compute_score_metrics(_read_json(path))


def analyze_d92_newguard_hard11(matrix_manifest_path: str | Path, *, run_root: str | Path | None, method_lock_path: str | Path, baseline_paired_rows_path: str | Path = HISTORICAL_BASELINE_PATH, per_old_class_rows_path: str | Path = HISTORICAL_PER_OLD_CLASS_PATH) -> dict[str, Any]:
    manifest_path = Path(matrix_manifest_path).resolve(strict=True); manifest_sha = _sha256(manifest_path); manifest = _read_json(manifest_path)
    lock_path = Path(method_lock_path).resolve(strict=True); lock_sha = _sha256(lock_path); lock = _read_json(lock_path)
    try:
        validate_method_lock(lock); validate_hard11_manifest(manifest, expected_method_lock_sha256=lock_sha, require_package_hashes=True)
    except ValueError as error:
        raise D92NewGuardHard11AnalysisError("matrix/method lock drift") from error
    root = Path(run_root or manifest["output_root"]).resolve(strict=True)
    if (root / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json").exists():
        raise D92NewGuardHard11AnalysisError("systemic stop marker exists")
    baseline_path = Path(baseline_paired_rows_path).resolve(strict=True); per_old_path = Path(per_old_class_rows_path).resolve(strict=True)
    if str(baseline_path).replace("\\", "/").lower() != HISTORICAL_BASELINE_PATH.lower() or str(per_old_path).replace("\\", "/").lower() != HISTORICAL_PER_OLD_CLASS_PATH.lower():
        raise D92NewGuardHard11AnalysisError("historical baseline path drift")
    baseline_rows = _read_csv(baseline_path, HISTORICAL_BASELINE_SHA256, expected_rows=125); _read_csv(per_old_path, HISTORICAL_PER_OLD_CLASS_SHA256, expected_rows=750)
    baseline_by_key = {_key(row): row for row in baseline_rows}
    raw_specs = lock["historical_baseline"]["e0_raw_scores"]
    paired_rows, per_old_rows, scenario_rows = [], [], []
    for job in manifest["jobs"]:
        key = _key(job)
        if key not in baseline_by_key or str(job["outer_key"]) not in raw_specs:
            raise D92NewGuardHard11AnalysisError("job/baseline identity mismatch")
        raw_spec = raw_specs[str(job["outer_key"])]; raw_path = Path(str(raw_spec["path"])).resolve(strict=True)
        if _sha256(raw_path) != str(raw_spec["sha256"]).lower():
            raise D92NewGuardHard11AnalysisError("E0 raw score SHA drift")
        job_root = root / "jobs" / str(job["outer_key"]) / ARM_ID; receipt = _read_json(job_root / "job_receipt.json"); score_path = job_root / "scorer" / "diag_cosine_score.json"; score = _read_json(score_path)
        before = job_root / "diag" / "before" / "prediction_artifact.npz"; after = job_root / "diag" / "after" / "prediction_artifact.npz"
        if receipt.get("schema") != "cvs.phase2.d92_newguard_hard11.job_receipt.v1" or receipt.get("job_id") != job["job_id"] or receipt.get("outer_key") != job["outer_key"] or receipt.get("arm_id") != ARM_ID or receipt.get("candidate") != CANDIDATE_ID or receipt.get("matrix_manifest_sha256") != manifest_sha or receipt.get("method_lock_sha256") != lock_sha or receipt.get("selection_sha256") != CANONICAL_SELECTION_SHA256 or receipt.get("truth_sidecar_exposed_to_predictor") is not False or receipt.get("query_truth_fed_back_to_predictor") is not False or receipt.get("before_prediction_sha256") != _sha256(before) or receipt.get("after_prediction_sha256") != _sha256(after) or receipt.get("score_sha256") != _sha256(score_path):
            raise D92NewGuardHard11AnalysisError("job receipt binding drift")
        if _prediction_closure_status(job_root / "diag") != ("closed", "closed"):
            raise D92NewGuardHard11AnalysisError("prediction closure drift")
        if score.get("candidate") != CANDIDATE_ID or score.get("query_truth_fed_back_to_predictor") is not False:
            raise D92NewGuardHard11AnalysisError("score binding drift")
        candidate = compute_score_metrics(score); baseline = _baseline_raw_metrics(raw_path); resource = _fit_resource(job_root, int(job["k_shot"]), baseline=baseline)
        row = {"outer_key": job["outer_key"], "outer_role": job["outer_role"], "receiver": job["receiver"], "seed": job["seed"], "k_shot": job["k_shot"], "new_class_count": job["new_class_count"], "slice": f"K{job['k_shot']}_new{job['new_class_count']}", **resource}
        for metric in EIGHT_PARETO_METRICS:
            row[f"candidate_{metric}"] = candidate[metric]; row[f"e0_{metric}"] = baseline[metric]; row[f"delta_{metric}_vs_e0"] = candidate[metric] - baseline[metric]
        row["full_only_query_macs"] = baseline.get("query_macs", -1); row["full_only_state_bytes"] = baseline.get("state_bytes", -1)
        paired_rows.append(row)
        for tx, value in candidate["old_class_accuracy"].items():
            per_old_rows.append({"outer_key": job["outer_key"], "tx": tx, "candidate_accuracy": value, "e0_accuracy": baseline["old_class_accuracy"].get(tx), "delta_accuracy": value - baseline["old_class_accuracy"].get(tx, value)})
        after_by_scene = score.get("after", {}).get("by_scenario", {})
        for scene in SCENES:
            if isinstance(after_by_scene, Mapping) and isinstance(after_by_scene.get(scene), Mapping):
                item = after_by_scene[scene]; scenario_rows.append({"outer_key": job["outer_key"], "scenario": scene, "candidate_h_old_new": item.get("h_old_new"), "candidate_old_acc": item.get("old_acc"), "candidate_seen_new_acc": item.get("seen_new_acc"), "candidate_new_to_old_rate": item.get("new_to_old_rate"), "candidate_old_to_new_rate": item.get("old_to_new_rate")})
    if len(paired_rows) != 11 or len(per_old_rows) != 66:
        raise D92NewGuardHard11AnalysisError("Hard11 result row closure drift")
    performance = [row for row in paired_rows if row["outer_role"] == "performance"]; liveness = [row for row in paired_rows if row["outer_role"] == "liveness"]
    deltas = {metric: _mean(row[f"delta_{metric}_vs_e0"] for row in performance) for metric in EIGHT_PARETO_METRICS}
    strict = _strict_ok(deltas); magnitude = _magnitude_ok(deltas)
    direction_counts = {metric: sum((row[f"delta_{metric}_vs_e0"] > -_TOLERANCE if metric in {"average_forgetting", "new_to_old_rate", "old_to_new_rate"} else row[f"delta_{metric}_vs_e0"] >= -_TOLERANCE) for row in performance) for metric in EIGHT_PARETO_METRICS}
    stability = all(direction_counts[m] >= (9 if m in {"seen_new_acc", "new_to_old_rate"} else 8) for m in ("h_old_new", "old_balanced_accuracy", "old_floor", "seen_new_acc", "average_forgetting", "new_to_old_rate"))
    query_exact = all(int(row["query_macs"]) == int(row["full_only_query_macs"]) and int(row["state_bytes"]) == int(row["full_only_state_bytes"]) for row in performance)
    wall_p90 = sorted(row["registration_wall_time_ns"] for row in performance)[max(0, math.ceil(0.9 * len(performance)) - 1)]
    peak_p90 = sorted(row["registration_incremental_peak_working_set_bytes"] for row in performance)[max(0, math.ceil(0.9 * len(performance)) - 1)]
    resources = query_exact and wall_p90 <= RESOURCE_GATE["registration_wall_p90_max_ns"] and peak_p90 <= max(float(row["full_only_state_bytes"] for row in performance)) + RESOURCE_GATE["registration_peak_delta_max_bytes"]
    gates = {"complete_artifact_closure": {"passed": len(paired_rows) == 11, "observed": len(paired_rows), "threshold": 11}, "performance_outer_closure": {"passed": len(performance) == 10 and len(liveness) == 1, "observed": f"{len(performance)}+{len(liveness)}", "threshold": "10+1"}, "all_strict_pareto": {"passed": strict, "observed": deltas, "threshold": "strict directions"}, "all_magnitude": {"passed": magnitude, "observed": deltas, "threshold": STRICT_PARETO_THRESHOLDS}, "stability": {"passed": stability, "observed": direction_counts, "threshold": "H/old/floor/forget>=8, seen-new/new-old>=9"}, "resources": {"passed": resources, "observed": {"query_exact": query_exact, "wall_p90_ns": wall_p90, "peak_p90_bytes": peak_p90}, "threshold": RESOURCE_GATE}}
    gate_state = {name: bool(value["passed"]) for name, value in gates.items()}; verdict = decide_verdict(gate_state)
    aggregate = {"row_count": len(paired_rows), "performance_row_count": len(performance), "liveness_row_count": len(liveness), **{f"candidate_mean_{m}": _mean(row[f"candidate_{m}"] for row in performance) for m in EIGHT_PARETO_METRICS}, **{f"e0_mean_{m}": _mean(row[f"e0_{m}"] for row in performance) for m in EIGHT_PARETO_METRICS}, **{f"mean_delta_{m}_vs_e0": deltas[m] for m in EIGHT_PARETO_METRICS}, "registration_wall_p90_ns": wall_p90, "registration_peak_p90_bytes": peak_p90}
    return {"schema": "cvs.phase2.d92_newguard_hard11.analysis.v1", "status": "ANALYZED", "claim_scope": manifest.get("claim_scope"), "matrix_manifest_sha256": manifest_sha, "method_lock_sha256": lock_sha, "selection_sha256": CANONICAL_SELECTION_SHA256, "baseline": {"paired_rows_path": str(baseline_path), "paired_rows_sha256": HISTORICAL_BASELINE_SHA256, "per_old_class_rows_path": str(per_old_path), "per_old_class_rows_sha256": HISTORICAL_PER_OLD_CLASS_SHA256}, "aggregate": aggregate, "paired_rows": paired_rows, "per_old_class_rows": per_old_rows, "scenario_rows": scenario_rows, "liveness_rows": liveness, "gates": gates, "gate_state": gate_state, "all_gates_pass": verdict == "ADVANCE_TO_TARGET125_CANDIDATE", "verdict": verdict}


analyze_newguard_hard11 = analyze_d92_newguard_hard11

__all__ = ["D92NewGuardHard11AnalysisError", "EIGHT_PARETO_METRICS", "PARETO_METRICS", "HISTORICAL_BASELINE_SHA256", "HISTORICAL_PER_OLD_CLASS_SHA256", "analyze_d92_newguard_hard11", "analyze_newguard_hard11", "compute_old_balanced_accuracy", "compute_score_metrics", "decide_verdict", "strict_pareto_deltas"]
