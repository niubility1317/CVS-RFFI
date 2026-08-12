"""Independent post-prediction analyzer for D92 NewGuard Hard11."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
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


def compute_confusion_rates(score: Mapping[str, Any]) -> dict[str, float]:
    """Compute bidirectional confusion using old/new query counts, not scene means."""
    before = score.get("before")
    after = score.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise D92NewGuardHard11AnalysisError("score state surface missing")
    before_scene = before.get("by_scenario")
    after_scene = after.get("by_scenario")
    if not isinstance(before_scene, Mapping) or not isinstance(after_scene, Mapping):
        return {
            "new_to_old_rate": _rate(after, "new_to_old_rate"),
            "old_to_new_rate": _rate(after, "old_to_new_rate"),
        }
    if set(before_scene) != set(after_scene) or not before_scene:
        raise D92NewGuardHard11AnalysisError("bidirectional scenario closure drift")
    old_total = 0.0
    new_total = 0.0
    old_to_new = 0.0
    new_to_old = 0.0
    for scene, after_row in after_scene.items():
        before_row = before_scene.get(scene)
        if not isinstance(before_row, Mapping) or not isinstance(after_row, Mapping):
            raise D92NewGuardHard11AnalysisError(f"scenario row missing: {scene}")
        old_count = _finite(before_row.get("query_count"), f"before query count {scene}", lower=0.0)
        total_count = _finite(after_row.get("query_count"), f"after query count {scene}", lower=0.0)
        new_count = total_count - old_count
        if new_count < -_TOLERANCE:
            raise D92NewGuardHard11AnalysisError(f"scenario query count regressed: {scene}")
        new_count = max(0.0, new_count)
        new_rate = _finite(after_row.get("new_to_old_rate"), f"new-to-old rate {scene}", lower=0.0, upper=1.0)
        old_rate = _finite(after_row.get("old_to_new_rate"), f"old-to-new rate {scene}", lower=0.0, upper=1.0)
        old_total += old_count
        new_total += new_count
        old_to_new += old_rate * old_count
        new_to_old += new_rate * new_count
    if old_total <= 0.0 or new_total <= 0.0:
        raise D92NewGuardHard11AnalysisError("old/new query count closure is empty")
    result = {"new_to_old_rate": new_to_old / new_total, "old_to_new_rate": old_to_new / old_total}
    # When an aggregate is present, ensure it agrees with the count-weighted
    # scene calculation. Never replace the weighted result with the aggregate.
    for name, value in result.items():
        aggregate = after.get(name)
        if aggregate is not None and abs(_finite(aggregate, name, lower=0.0, upper=1.0) - value) > 1.0e-9:
            raise D92NewGuardHard11AnalysisError(f"aggregate {name} disagrees with scenario weighting")
    return result


def validate_per_old_class_join(rows: Sequence[Mapping[str, Any]], raw_score: Mapping[str, Any], *, outer_key: str) -> dict[str, dict[str, float]]:
    """Join frozen historical rows to E0 raw by_tx without candidate backfill."""
    if not isinstance(rows, Sequence) or not rows:
        raise D92NewGuardHard11AnalysisError(f"per-old rows missing for {outer_key}")
    by_tx = raw_score.get("after", {}).get("by_tx") if isinstance(raw_score.get("after"), Mapping) else None
    if not isinstance(by_tx, Mapping):
        raise D92NewGuardHard11AnalysisError(f"per-old raw by_tx missing for {outer_key}")
    raw_old = {
        str(tx): _finite(item.get("accuracy"), f"raw old accuracy {tx}", lower=0.0, upper=1.0)
        for tx, item in by_tx.items()
        if isinstance(item, Mapping) and item.get("role") == "target_old"
    }
    if not raw_old:
        raise D92NewGuardHard11AnalysisError(f"per-old raw old-class closure missing for {outer_key}")
    joined: dict[str, dict[str, float]] = {}
    for row in rows:
        if str(row.get("outer_key")) != str(outer_key):
            raise D92NewGuardHard11AnalysisError(f"per-old outer join drift for {outer_key}")
        tx = str(row.get("tx"))
        if tx in joined or tx not in raw_old:
            raise D92NewGuardHard11AnalysisError(f"per-old TX join drift for {outer_key}")
        historical_candidate = _finite(row.get("candidate_accuracy"), f"historical candidate {tx}", lower=0.0, upper=1.0)
        if abs(historical_candidate - raw_old[tx]) > 1.0e-9:
            raise D92NewGuardHard11AnalysisError(f"per-old E0 accuracy mismatch for {outer_key}/{tx}")
        joined[tx] = {
            "candidate_accuracy": historical_candidate,
            "e0_accuracy": raw_old[tx],
            "historical_baseline_accuracy": _finite(row.get("baseline_accuracy"), f"historical baseline {tx}", lower=0.0, upper=1.0),
            "historical_delta_accuracy": _finite(row.get("delta_accuracy", historical_candidate - float(row.get("baseline_accuracy"))), f"historical delta {tx}"),
        }
    if set(joined) != set(raw_old):
        raise D92NewGuardHard11AnalysisError(f"per-old TX set closure drift for {outer_key}")
    return joined


def evaluate_resource_gate(candidate_rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]], *, query_state_exact: bool) -> dict[str, Any]:
    """Compare matched registration resources against the same E0 rows."""
    if not candidate_rows or len(candidate_rows) != len(baseline_rows):
        raise D92NewGuardHard11AnalysisError("resource row closure drift")
    wall_ratios: list[float] = []
    peak_deltas: list[float] = []
    candidate_walls: list[float] = []
    for candidate, baseline in zip(candidate_rows, baseline_rows):
        wall = _finite(candidate.get("registration_wall_time_ns"), "candidate registration wall", lower=0.0)
        base_wall = _finite(baseline.get("registration_wall_time_ns"), "E0 registration wall", lower=0.0)
        if base_wall <= 0.0:
            raise D92NewGuardHard11AnalysisError("E0 registration wall is zero")
        peak = _finite(candidate.get("registration_incremental_peak_working_set_bytes"), "candidate registration peak", lower=0.0)
        base_peak = _finite(baseline.get("registration_incremental_peak_working_set_bytes"), "E0 registration peak", lower=0.0)
        candidate_walls.append(wall)
        wall_ratios.append(wall / base_wall)
        peak_deltas.append(peak - base_peak)
    p90_index = max(0, math.ceil(0.9 * len(candidate_walls)) - 1)
    wall_p90 = sorted(candidate_walls)[p90_index]
    wall_ratio_p90 = sorted(wall_ratios)[p90_index]
    peak_delta_p90 = sorted(peak_deltas)[p90_index]
    passed = bool(query_state_exact) and wall_p90 <= RESOURCE_GATE["registration_wall_p90_max_ns"] and wall_ratio_p90 <= RESOURCE_GATE["registration_wall_ratio_max"] and peak_delta_p90 <= RESOURCE_GATE["registration_peak_delta_max_bytes"]
    return {"passed": passed, "query_state_exact": bool(query_state_exact), "wall_p90": wall_p90, "wall_ratio_p90": wall_ratio_p90, "peak_delta_p90_bytes": peak_delta_p90, "candidate_wall_p90_ns": wall_p90}


def validate_truth_binding(score: Mapping[str, Any], receipt: Mapping[str, Any], job: Mapping[str, Any], truth_path: str | Path) -> str:
    """Require manifest, receipt, scorer and actual truth-sidecar hashes to agree."""
    manifest_value = str(job.get("truth_sidecar", ""))
    actual_path = Path(truth_path).resolve()
    expected = str(job.get("truth_sidecar_sha256", "")).lower()
    same_absolute_path = Path(manifest_value).resolve() == actual_path
    outer_key = str(job.get("outer_key", ""))
    expected_suffix = ("jobs", outer_key, "offline", "scorer", "truth_sidecar.json")

    def logical_suffix(value: str | Path) -> tuple[str, ...]:
        return tuple(part for part in str(value).replace("\\", "/").split("/") if part)[-5:]

    same_logical_path = bool(outer_key) and logical_suffix(manifest_value) == expected_suffix and logical_suffix(actual_path) == expected_suffix
    if not (same_absolute_path or same_logical_path) or not actual_path.is_file() or actual_path.is_symlink() or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise D92NewGuardHard11AnalysisError("truth sidecar path/hash closure drift")
    actual = _sha256(actual_path)
    if actual != expected or str(receipt.get("truth_sidecar_sha256", "")).lower() != actual or str(score.get("truth_sidecar_sha256", "")).lower() != actual:
        raise D92NewGuardHard11AnalysisError("truth sidecar hash binding drift")
    return actual


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
    confusion = compute_confusion_rates(score)
    new_to_old_rate = confusion["new_to_old_rate"]
    old_to_new_rate = confusion["old_to_new_rate"]
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
    required = ("complete_artifact_closure", "performance_outer_closure", "all_strict_pareto", "all_magnitude", "stability", "resources")
    if any(name not in gate_state for name in required):
        return "REJECT_ROUTE"
    if not bool(gate_state.get("complete_artifact_closure")) or not bool(gate_state.get("performance_outer_closure")) or not bool(gate_state.get("all_strict_pareto")):
        return "REJECT_ROUTE"
    if not bool(gate_state.get("all_magnitude")):
        return "REVISE_ONCE"
    if not bool(gate_state.get("stability")) or not bool(gate_state.get("resources")):
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
    totals, actuals, macs, states, modes = set(), set(), set(), set(), set()
    walls, peaks = [], []
    scenarios = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise D92NewGuardHard11AnalysisError("after fit audit row invalid")
        scenarios.add(str(row.get("scenario")))
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
    expected = (3, 3, "d92_full_alias") if int(k_shot) <= 2 else (2, 1, "newguard_maxmin")
    if scenarios != set(SCENES) or totals != {expected[0]} or actuals != {expected[1]} or modes != {expected[2]} or len(macs) != 1 or len(states) != 1 or min(macs) < 0 or min(states) < 0:
        raise D92NewGuardHard11AnalysisError("fit/resource count closure drift")
    return {"fit_count": expected[0], "actual_fit_count": expected[1], "registered_d_mode": expected[2], "query_macs": next(iter(macs)), "state_bytes": next(iter(states)), "registration_wall_time_ns": float(statistics.median(walls)), "registration_incremental_peak_working_set_bytes": float(statistics.median(peaks))}


def _baseline_raw_metrics(path: Path) -> dict[str, Any]:
    return compute_score_metrics(_read_json(path))


def _scenario_closure(score: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    before = score.get("before")
    after = score.get("after")
    before_scene = before.get("by_scenario") if isinstance(before, Mapping) else None
    after_scene = after.get("by_scenario") if isinstance(after, Mapping) else None
    if not isinstance(before_scene, Mapping) or not isinstance(after_scene, Mapping) or set(before_scene) != set(SCENES) or set(after_scene) != set(SCENES):
        raise D92NewGuardHard11AnalysisError(f"{label} 3-scene closure drift")
    for scene in SCENES:
        if not isinstance(before_scene[scene], Mapping) or not isinstance(after_scene[scene], Mapping):
            raise D92NewGuardHard11AnalysisError(f"{label} scene row invalid: {scene}")
        _finite(before_scene[scene].get("query_count"), f"{label} before query count {scene}", lower=0.0)
        _finite(after_scene[scene].get("query_count"), f"{label} after query count {scene}", lower=0.0)
        for field in ("h_old_new", "old_acc", "seen_new_acc", "new_to_old_rate", "old_to_new_rate"):
            _finite(after_scene[scene].get(field), f"{label} {field} {scene}", lower=0.0, upper=1.0)
    return {scene: after_scene[scene] for scene in SCENES}


def _scenario_rows(outer_key: str, candidate: Mapping[str, Any], baseline: Mapping[str, Any], *, receiver: str, slice_name: str) -> list[dict[str, Any]]:
    candidate_after = _scenario_closure(candidate, f"candidate {outer_key}")
    baseline_after = _scenario_closure(baseline, f"E0 {outer_key}")
    candidate_before = candidate["before"]["by_scenario"]
    baseline_before = baseline["before"]["by_scenario"]
    rows: list[dict[str, Any]] = []
    for scene in SCENES:
        c, b = candidate_after[scene], baseline_after[scene]
        c_old = _finite(candidate_before[scene]["query_count"], f"candidate old count {scene}", lower=0.0)
        b_old = _finite(baseline_before[scene]["query_count"], f"E0 old count {scene}", lower=0.0)
        c_total = _finite(c["query_count"], f"candidate total count {scene}", lower=0.0)
        b_total = _finite(b["query_count"], f"E0 total count {scene}", lower=0.0)
        rows.append({
            "outer_key": outer_key,
            "receiver": receiver,
            "slice": slice_name,
            "scenario": scene,
            "old_query_count": c_old,
            "new_query_count": c_total - c_old,
            "e0_old_query_count": b_old,
            "e0_new_query_count": b_total - b_old,
            "candidate_h_old_new": _finite(c["h_old_new"], f"candidate H {scene}", lower=0.0, upper=1.0),
            "e0_h_old_new": _finite(b["h_old_new"], f"E0 H {scene}", lower=0.0, upper=1.0),
            "candidate_old_acc": _finite(c["old_acc"], f"candidate old acc {scene}", lower=0.0, upper=1.0),
            "e0_old_acc": _finite(b["old_acc"], f"E0 old acc {scene}", lower=0.0, upper=1.0),
            "candidate_seen_new_acc": _finite(c["seen_new_acc"], f"candidate new acc {scene}", lower=0.0, upper=1.0),
            "e0_seen_new_acc": _finite(b["seen_new_acc"], f"E0 new acc {scene}", lower=0.0, upper=1.0),
            "candidate_new_to_old_rate": _finite(c["new_to_old_rate"], f"candidate new-to-old {scene}", lower=0.0, upper=1.0),
            "e0_new_to_old_rate": _finite(b["new_to_old_rate"], f"E0 new-to-old {scene}", lower=0.0, upper=1.0),
            "candidate_old_to_new_rate": _finite(c["old_to_new_rate"], f"candidate old-to-new {scene}", lower=0.0, upper=1.0),
            "e0_old_to_new_rate": _finite(b["old_to_new_rate"], f"E0 old-to-new {scene}", lower=0.0, upper=1.0),
        })
    return rows


def _group_stability(rows: Sequence[Mapping[str, Any]], group_field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_field])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for key, items in sorted(grouped.items()):
        h_delta = _mean(float(item["candidate_h_old_new"]) - float(item["e0_h_old_new"]) for item in items)
        new_delta = _mean(float(item["candidate_seen_new_acc"]) - float(item["e0_seen_new_acc"]) for item in items)
        result[key] = {"row_count": len(items), "mean_delta_h_old_new": h_delta, "mean_delta_seen_new_acc": new_delta, "passed": h_delta >= -_TOLERANCE and new_delta >= -_TOLERANCE}
    return result


def analyze_d92_newguard_hard11(matrix_manifest_path: str | Path, *, run_root: str | Path | None, method_lock_path: str | Path, baseline_paired_rows_path: str | Path = HISTORICAL_BASELINE_PATH, per_old_class_rows_path: str | Path = HISTORICAL_PER_OLD_CLASS_PATH, truth_sidecar_root: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(matrix_manifest_path).resolve(strict=True); manifest_sha = _sha256(manifest_path); manifest = _read_json(manifest_path)
    lock_path = Path(method_lock_path).resolve(strict=True); lock_sha = _sha256(lock_path); lock = _read_json(lock_path)
    try:
        validate_method_lock(lock); validate_hard11_manifest(manifest, expected_method_lock_sha256=lock_sha, require_package_hashes=True)
    except ValueError as error:
        raise D92NewGuardHard11AnalysisError("matrix/method lock drift") from error
    root = Path(run_root or manifest["output_root"]).resolve(strict=True)
    truth_root = Path(truth_sidecar_root).resolve(strict=True) if truth_sidecar_root is not None else None
    if (root / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json").exists():
        raise D92NewGuardHard11AnalysisError("systemic stop marker exists")
    baseline_path = Path(baseline_paired_rows_path).resolve(strict=True); per_old_path = Path(per_old_class_rows_path).resolve(strict=True)
    if str(baseline_path).replace("\\", "/").lower() != HISTORICAL_BASELINE_PATH.lower() or str(per_old_path).replace("\\", "/").lower() != HISTORICAL_PER_OLD_CLASS_PATH.lower():
        raise D92NewGuardHard11AnalysisError("historical baseline path drift")
    baseline_rows = _read_csv(baseline_path, HISTORICAL_BASELINE_SHA256, expected_rows=125)
    per_old_source_rows = _read_csv(per_old_path, HISTORICAL_PER_OLD_CLASS_SHA256, expected_rows=750)
    baseline_by_key = {_key(row): row for row in baseline_rows}
    if len(baseline_by_key) != len(baseline_rows):
        raise D92NewGuardHard11AnalysisError("historical baseline key collision")
    per_old_by_outer: dict[str, list[dict[str, str]]] = defaultdict(list)
    for historical_row in per_old_source_rows:
        per_old_by_outer[str(historical_row.get("outer_key"))].append(historical_row)
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
        if receipt.get("schema") != "cvs.phase2.d92_newguard_hard11.job_receipt.v1" or receipt.get("status") != "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE" or receipt.get("job_id") != job["job_id"] or receipt.get("outer_key") != job["outer_key"] or receipt.get("arm_id") != ARM_ID or receipt.get("candidate") != CANDIDATE_ID or receipt.get("matrix_manifest_sha256") != manifest_sha or receipt.get("method_lock_sha256") != lock_sha or receipt.get("selection_sha256") != CANONICAL_SELECTION_SHA256 or receipt.get("truth_sidecar_exposed_to_predictor") is not False or receipt.get("query_truth_fed_back_to_predictor") is not False or receipt.get("query_truth_joined_only_after_immutable_predictions") is not True or receipt.get("prediction_and_scorer_processes_isolated") is not True or receipt.get("before_prediction_sha256") != _sha256(before) or receipt.get("after_prediction_sha256") != _sha256(after) or receipt.get("score_sha256") != _sha256(score_path):
            raise D92NewGuardHard11AnalysisError("job receipt binding drift")
        if _prediction_closure_status(job_root / "diag") != ("closed", "closed"):
            raise D92NewGuardHard11AnalysisError("prediction closure drift")
        if score.get("candidate") != CANDIDATE_ID or score.get("query_truth_fed_back_to_predictor") is not False or score.get("query_truth_joined_only_after_immutable_predictions") is not True or score.get("before_prediction_sha256") != receipt.get("before_prediction_sha256") or score.get("after_prediction_sha256") != receipt.get("after_prediction_sha256"):
            raise D92NewGuardHard11AnalysisError("score binding drift")
        truth_path = Path(str(job["truth_sidecar"])) if truth_root is None else truth_root / "jobs" / str(job["outer_key"]) / "offline" / "scorer" / "truth_sidecar.json"
        validate_truth_binding(score, receipt, job, truth_path)
        candidate = compute_score_metrics(score); baseline_score = _read_json(raw_path); baseline = compute_score_metrics(baseline_score)
        historical_join = validate_per_old_class_join(per_old_by_outer.get(str(job["outer_key"]), []), baseline_score, outer_key=str(job["outer_key"]))
        resource = _fit_resource(job_root, int(job["k_shot"]), baseline=baseline)
        baseline_row = baseline_by_key[key]
        baseline_wall = _finite(baseline_row.get("registration_wall_time_ns"), "E0 registration wall", lower=0.0)
        baseline_peak = _finite(baseline_row.get("registration_incremental_peak_working_set_bytes"), "E0 registration peak", lower=0.0)
        row = {"outer_key": job["outer_key"], "outer_role": job["outer_role"], "receiver": job["receiver"], "seed": job["seed"], "k_shot": job["k_shot"], "new_class_count": job["new_class_count"], "slice": f"K{job['k_shot']}_new{job['new_class_count']}", **resource}
        for metric in EIGHT_PARETO_METRICS:
            row[f"candidate_{metric}"] = candidate[metric]; row[f"e0_{metric}"] = baseline[metric]; row[f"delta_{metric}_vs_e0"] = candidate[metric] - baseline[metric]
        row["full_only_query_macs"] = _finite(baseline_row.get("query_macs"), "E0 query MACs", lower=0.0); row["full_only_state_bytes"] = _finite(baseline_row.get("state_bytes"), "E0 state bytes", lower=0.0)
        row["full_only_registration_wall_time_ns"] = baseline_wall; row["full_only_registration_peak_working_set_bytes"] = baseline_peak
        row["registration_wall_ratio"] = resource["registration_wall_time_ns"] / baseline_wall if baseline_wall else math.inf; row["registration_peak_delta_bytes"] = resource["registration_incremental_peak_working_set_bytes"] - baseline_peak
        paired_rows.append(row)
        if set(candidate["old_class_accuracy"]) != set(historical_join) or set(candidate["old_class_accuracy"]) != set(baseline["old_class_accuracy"]):
            raise D92NewGuardHard11AnalysisError("candidate/E0 six-old TX closure drift")
        for tx in sorted(historical_join):
            value = _finite(candidate["old_class_accuracy"].get(tx), f"candidate old accuracy {tx}", lower=0.0, upper=1.0)
            e0_value = historical_join[tx]["e0_accuracy"]
            per_old_rows.append({"outer_key": job["outer_key"], "tx": tx, "candidate_accuracy": value, "e0_accuracy": e0_value, "delta_accuracy": value - e0_value, "historical_baseline_accuracy": historical_join[tx]["historical_baseline_accuracy"], "historical_delta_accuracy": historical_join[tx]["historical_delta_accuracy"]})
        scenario_rows.extend(_scenario_rows(str(job["outer_key"]), score, baseline_score, receiver=str(job["receiver"]), slice_name=f"K{job['k_shot']}_new{job['new_class_count']}"))
    if len(paired_rows) != 11 or len(per_old_rows) != 66 or len(scenario_rows) != 33:
        raise D92NewGuardHard11AnalysisError("Hard11 result row closure drift")
    performance = [row for row in paired_rows if row["outer_role"] == "performance"]; liveness = [row for row in paired_rows if row["outer_role"] == "liveness"]
    deltas = {metric: _mean(row[f"delta_{metric}_vs_e0"] for row in performance) for metric in EIGHT_PARETO_METRICS}
    strict = _strict_ok(deltas); magnitude = _magnitude_ok(deltas)
    direction_counts = {metric: sum((row[f"delta_{metric}_vs_e0"] > -_TOLERANCE if metric in {"average_forgetting", "new_to_old_rate", "old_to_new_rate"} else row[f"delta_{metric}_vs_e0"] >= -_TOLERANCE) for row in performance) for metric in EIGHT_PARETO_METRICS}
    by_old_group: dict[str, list[float]] = defaultdict(list)
    by_outer_old: dict[str, list[float]] = defaultdict(list)
    for item in per_old_rows:
        if str(item["outer_key"]) in {str(row["outer_key"]) for row in performance}:
            by_old_group[str(item["tx"])].append(float(item["delta_accuracy"]))
            by_outer_old[str(item["outer_key"])].append(float(item["delta_accuracy"]))
    per_old_summary = {tx: {"row_count": len(values), "mean_delta_accuracy": _mean(values), "min_delta_accuracy": min(values), "nondecrease_count": sum(value >= -_TOLERANCE for value in values)} for tx, values in sorted(by_old_group.items())}
    per_outer_old_summary = {outer: {"row_count": len(values), "min_delta_accuracy": min(values), "nondecrease_count": sum(value >= -_TOLERANCE for value in values), "passed": len(values) == 6 and min(values) >= -0.01 and sum(value >= -_TOLERANCE for value in values) >= 5} for outer, values in sorted(by_outer_old.items())}
    per_old_stability = len(per_old_summary) == 6 and all(item["row_count"] == 10 for item in per_old_summary.values()) and all(item["min_delta_accuracy"] >= -0.01 for item in per_old_summary.values()) and len(per_outer_old_summary) == 10 and all(item["passed"] for item in per_outer_old_summary.values())
    group_base = [item for item in scenario_rows if item["outer_key"] in {row["outer_key"] for row in performance}]
    by_receiver = _group_stability(group_base, "receiver"); by_slice = _group_stability(group_base, "slice"); by_scene = _group_stability(group_base, "scenario")
    group_stability = all(item["passed"] for groups in (by_receiver, by_slice, by_scene) for item in groups.values())
    stability = all(direction_counts[m] >= (9 if m in {"seen_new_acc", "new_to_old_rate"} else 8) for m in ("h_old_new", "old_balanced_accuracy", "old_floor", "seen_new_acc", "average_forgetting", "new_to_old_rate")) and per_old_stability and group_stability
    query_exact = all(int(row["query_macs"]) == int(row["full_only_query_macs"]) and int(row["state_bytes"]) == int(row["full_only_state_bytes"]) for row in performance)
    resource_eval = evaluate_resource_gate(performance, [{"registration_wall_time_ns": row["full_only_registration_wall_time_ns"], "registration_incremental_peak_working_set_bytes": row["full_only_registration_peak_working_set_bytes"]} for row in performance], query_state_exact=query_exact)
    wall_p90 = resource_eval["wall_p90"]; peak_p90 = _finite(resource_eval["peak_delta_p90_bytes"], "peak delta p90")
    gates = {"complete_artifact_closure": {"passed": len(paired_rows) == 11 and len(per_old_rows) == 66 and len(scenario_rows) == 33, "observed": {"paired": len(paired_rows), "per_old": len(per_old_rows), "scene": len(scenario_rows)}, "threshold": "11/66/33"}, "performance_outer_closure": {"passed": len(performance) == 10 and len(liveness) == 1, "observed": f"{len(performance)}+{len(liveness)}", "threshold": "10+1"}, "all_strict_pareto": {"passed": strict, "observed": deltas, "threshold": "strict directions"}, "all_magnitude": {"passed": magnitude, "observed": deltas, "threshold": STRICT_PARETO_THRESHOLDS}, "stability": {"passed": stability, "observed": {"direction_counts": direction_counts, "per_old_class": per_old_stability, "per_outer_old": per_outer_old_summary, "by_receiver": by_receiver, "by_slice": by_slice, "by_scene": by_scene}, "threshold": "paired/group/old-class stability"}, "resources": {"passed": bool(resource_eval["passed"]), "observed": {"query_exact": query_exact, "wall_p90_ns": wall_p90, "wall_ratio_p90": resource_eval["wall_ratio_p90"], "peak_delta_p90_bytes": peak_p90}, "threshold": RESOURCE_GATE}}
    gate_state = {name: bool(value["passed"]) for name, value in gates.items()}; verdict = decide_verdict(gate_state)
    aggregate = {"row_count": len(paired_rows), "performance_row_count": len(performance), "liveness_row_count": len(liveness), **{f"candidate_mean_{m}": _mean(row[f"candidate_{m}"] for row in performance) for m in EIGHT_PARETO_METRICS}, **{f"e0_mean_{m}": _mean(row[f"e0_{m}"] for row in performance) for m in EIGHT_PARETO_METRICS}, **{f"mean_delta_{m}_vs_e0": deltas[m] for m in EIGHT_PARETO_METRICS}, "registration_wall_p90_ns": wall_p90, "registration_wall_ratio_p90": resource_eval["wall_ratio_p90"], "registration_peak_delta_p90_bytes": peak_p90}
    return {"schema": "cvs.phase2.d92_newguard_hard11.analysis.v1", "status": "ANALYZED", "claim_scope": manifest.get("claim_scope"), "matrix_manifest_sha256": manifest_sha, "method_lock_sha256": lock_sha, "selection_sha256": CANONICAL_SELECTION_SHA256, "baseline": {"paired_rows_path": str(baseline_path), "paired_rows_sha256": HISTORICAL_BASELINE_SHA256, "per_old_class_rows_path": str(per_old_path), "per_old_class_rows_sha256": HISTORICAL_PER_OLD_CLASS_SHA256}, "aggregate": aggregate, "paired_rows": paired_rows, "per_old_class_rows": per_old_rows, "per_old_class_summary": per_old_summary, "scenario_rows": scenario_rows, "by_receiver": by_receiver, "by_slice": by_slice, "by_scene": by_scene, "liveness_rows": liveness, "gates": gates, "gate_state": gate_state, "all_gates_pass": verdict == "ADVANCE_TO_TARGET125_CANDIDATE", "verdict": verdict}


analyze_newguard_hard11 = analyze_d92_newguard_hard11

__all__ = ["D92NewGuardHard11AnalysisError", "EIGHT_PARETO_METRICS", "PARETO_METRICS", "HISTORICAL_BASELINE_SHA256", "HISTORICAL_PER_OLD_CLASS_SHA256", "analyze_d92_newguard_hard11", "analyze_newguard_hard11", "compute_confusion_rates", "compute_old_balanced_accuracy", "compute_score_metrics", "decide_verdict", "evaluate_resource_gate", "strict_pareto_deltas", "validate_per_old_class_join", "validate_truth_binding"]
