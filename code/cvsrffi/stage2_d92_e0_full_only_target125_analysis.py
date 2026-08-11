"""Analyze the complete E0_FULL_ONLY Target125 run against frozen D92 rows."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cvsrffi.stage2_d92_e0_full_only_target125 import (
    ARM_ID,
    CANDIDATE_ID,
    CANONICAL_SELECTION_SHA256,
    SCENES,
    SHARD_COUNT,
    validate_method_lock,
    validate_target125_manifest,
)
from scripts.run_d92_e0_full_only_target125 import (
    QUERY_ZERO_FIELDS,
    _prediction_closure_status,
)


BASELINE_ROW_SHA256 = "bc8070cd9235ab41eda5bafd2ec66e9afad48b6466d2066508d0bab46980fa62"
BASELINE_SCENARIO_SHA256 = "34fbe22ff7aca1f98fce127bb31f731078bc234fd8e793e194e1a07b89e446d4"
BASELINE_PER_TX_SHA256 = "3d68876873458d5ae91a6d8018242e31ef7574b5cccd32dec6726c0203646cc2"
_TOLERANCE = 1.0e-12
_METRICS = ("h_old_new", "old_acc", "old_floor", "seen_new_acc", "forgetting")
_SCENARIO_METRICS = ("h_old_new", "old_acc", "seen_new_acc", "forgetting")


class D92E0FullOnlyTarget125AnalysisError(ValueError):
    """Raised when complete Target125 evidence is missing or detached."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise D92E0FullOnlyTarget125AnalysisError(f"missing JSON artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92E0FullOnlyTarget125AnalysisError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise D92E0FullOnlyTarget125AnalysisError(f"JSON artifact is not an object: {path}")
    return payload


def _read_csv(path: Path, expected_sha256: str) -> list[dict[str, str]]:
    if not path.is_file() or path.is_symlink() or _sha256(path) != expected_sha256:
        raise D92E0FullOnlyTarget125AnalysisError(f"frozen baseline identity drift: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise D92E0FullOnlyTarget125AnalysisError(f"empty baseline: {path}")
    return rows


def _finite(value: Any, label: str, *, lower: float | None = None, upper: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise D92E0FullOnlyTarget125AnalysisError(f"non-numeric {label}") from error
    if not math.isfinite(result) or (lower is not None and result < lower) or (upper is not None and result > upper):
        raise D92E0FullOnlyTarget125AnalysisError(f"out-of-range {label}")
    return result


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        raise D92E0FullOnlyTarget125AnalysisError("empty mean")
    return float(statistics.fmean(rows))


def _key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["receiver"]),
        int(row["seed"]),
        int(row["k_shot"]),
        int(row.get("new_class_count", row.get("new_count"))),
    )


def _score_metrics(score: Mapping[str, Any]) -> dict[str, float]:
    before, after = score.get("before"), score.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise D92E0FullOnlyTarget125AnalysisError("score state surface missing")
    return {
        "h_old_new": _finite(after.get("h_old_new"), "H", lower=0.0, upper=1.0),
        "old_acc": _finite(after.get("old_acc"), "old accuracy", lower=0.0, upper=1.0),
        "old_floor": _finite(score.get("per_old_class_floor_after"), "old floor", lower=0.0, upper=1.0),
        "seen_new_acc": _finite(after.get("seen_new_acc"), "seen-new accuracy", lower=0.0, upper=1.0),
        "forgetting": _finite(score.get("old_forgetting_pp"), "forgetting") / 100.0,
        "da1_reg0_old_acc": _finite(before.get("old_acc"), "DA1_REG0 old accuracy", lower=0.0, upper=1.0),
        "da1_reg0_old_floor": _finite(score.get("per_old_class_floor_before"), "DA1_REG0 old floor", lower=0.0, upper=1.0),
    }


def _baseline_metrics(row: Mapping[str, str]) -> dict[str, float]:
    return {
        "h_old_new": _finite(row.get("h_old_new"), "baseline H", lower=0.0, upper=1.0),
        "old_acc": _finite(row.get("c_old_acc"), "baseline old accuracy", lower=0.0, upper=1.0),
        "old_floor": _finite(row.get("c_old_floor"), "baseline old floor", lower=0.0, upper=1.0),
        "seen_new_acc": _finite(row.get("seen_new_acc"), "baseline seen-new", lower=0.0, upper=1.0),
        "forgetting": _finite(row.get("average_forgetting"), "baseline forgetting"),
        "da1_reg0_old_acc": _finite(row.get("b_old_acc"), "baseline DA1_REG0 old accuracy", lower=0.0, upper=1.0),
        "da1_reg0_old_floor": _finite(row.get("b_old_floor"), "baseline DA1_REG0 old floor", lower=0.0, upper=1.0),
    }


def _fit_resource(job_root: Path, k_shot: int) -> dict[str, float | int | str]:
    path = job_root / "diag" / "after" / "fit_audit.json"
    if not path.is_file() or path.is_symlink():
        raise D92E0FullOnlyTarget125AnalysisError("after fit audit missing")
    try:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92E0FullOnlyTarget125AnalysisError("after fit audit invalid") from error
    if not isinstance(rows, list) or len(rows) != len(SCENES) or any(not isinstance(row, Mapping) for row in rows):
        raise D92E0FullOnlyTarget125AnalysisError("after fit audit scene closure drift")
    if {str(row.get("scenario")) for row in rows} != set(SCENES):
        raise D92E0FullOnlyTarget125AnalysisError("after fit audit scenario identity drift")
    expected_total, expected_actual = (3, 3) if k_shot <= 2 else (2, 1)
    totals: set[int] = set()
    actuals: set[int] = set()
    query_macs: set[int] = set()
    state_bytes: set[int] = set()
    modes: set[str] = set()
    walls: list[float] = []
    cpus: list[float] = []
    peaks: list[float] = []
    for row in rows:
        if row.get("arm_id") != ARM_ID or row.get("candidate_id") != CANDIDATE_ID:
            raise D92E0FullOnlyTarget125AnalysisError("fit audit arm identity drift")
        if any(row.get(field) is not False for field in QUERY_ZERO_FIELDS):
            raise D92E0FullOnlyTarget125AnalysisError("query access is not zero")
        totals.add(int(row.get("after_total_component_fit_count", -1)))
        inventory = row.get("after_actual_component_inventory")
        if not isinstance(inventory, Mapping):
            raise D92E0FullOnlyTarget125AnalysisError("actual fit inventory missing")
        actuals.add(int(inventory.get("actual_component_fit_count", -1)))
        query_macs.add(int(row.get("query_macs", -1)))
        state_bytes.add(int(row.get("after_state_bytes", -1)))
        modes.add(str(row.get("after_registered_d_mode_effective")))
        resource = row.get("after_registration_resource")
        if not isinstance(resource, Mapping):
            raise D92E0FullOnlyTarget125AnalysisError("registration resource receipt missing")
        walls.append(_finite(resource.get("registration_wall_time_ns"), "wall", lower=0.0))
        cpus.append(_finite(resource.get("registration_process_cpu_time_ns"), "CPU time", lower=0.0))
        peaks.append(_finite(resource.get("registration_incremental_peak_working_set_bytes"), "peak", lower=0.0))
    if totals != {expected_total} or actuals != {expected_actual} or len(query_macs) != 1 or len(state_bytes) != 1 or min(query_macs) < 0 or min(state_bytes) < 0:
        raise D92E0FullOnlyTarget125AnalysisError("fit/resource count closure drift")
    expected_mode = "d92_full_alias" if k_shot <= 2 else "full_only"
    if modes != {expected_mode}:
        raise D92E0FullOnlyTarget125AnalysisError("registered D mode drift")
    return {
        "fit_count": expected_total,
        "actual_fit_count": expected_actual,
        "registered_d_mode": expected_mode,
        "query_macs": next(iter(query_macs)),
        "state_bytes": next(iter(state_bytes)),
        "registration_wall_time_ns": float(statistics.median(walls)),
        "registration_process_cpu_time_ns": float(statistics.median(cpus)),
        "registration_incremental_peak_working_set_bytes": float(statistics.median(peaks)),
    }


def _group_summary(
    rows: Sequence[Mapping[str, Any]],
    key_name: str,
    *,
    metrics: Sequence[str] = _METRICS,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key_name])].append(row)
    result: list[dict[str, Any]] = []
    for group, members in sorted(grouped.items()):
        item: dict[str, Any] = {key_name: group, "row_count": len(members)}
        for metric in metrics:
            item[f"candidate_{metric}"] = _mean(float(row[f"candidate_{metric}"]) for row in members)
            item[f"baseline_{metric}"] = _mean(float(row[f"baseline_{metric}"]) for row in members)
            item[f"delta_{metric}"] = _mean(float(row[f"delta_{metric}"]) for row in members)
        result.append(item)
    return result


def _scenario_pairs(
    score: Mapping[str, Any],
    job: Mapping[str, Any],
    baseline: Mapping[tuple[str, int, int, int, str], Mapping[str, str]],
) -> list[dict[str, Any]]:
    before, after = score["before"]["by_scenario"], score["after"]["by_scenario"]
    result: list[dict[str, Any]] = []
    for scene in SCENES:
        base = baseline[(*_key(job), scene)]
        candidate = {
            "h_old_new": _finite(after[scene].get("h_old_new"), "scenario H", lower=0.0, upper=1.0),
            "old_acc": _finite(after[scene].get("old_acc"), "scenario old", lower=0.0, upper=1.0),
            "seen_new_acc": _finite(after[scene].get("seen_new_acc"), "scenario new", lower=0.0, upper=1.0),
            "forgetting": _finite(before[scene].get("old_acc"), "scenario before old", lower=0.0, upper=1.0)
            - _finite(after[scene].get("old_acc"), "scenario after old", lower=0.0, upper=1.0),
        }
        baseline_values = {
            "h_old_new": _finite(base["h_old_new"], "baseline scenario H"),
            "old_acc": _finite(base["c_old_acc"], "baseline scenario old"),
            "seen_new_acc": _finite(base["seen_new_acc"], "baseline scenario new"),
            "forgetting": _finite(base["average_forgetting"], "baseline scenario forgetting"),
        }
        row: dict[str, Any] = {
            "outer_key": str(job["outer_key"]),
            "receiver": str(job["receiver"]),
            "seed": int(job["seed"]),
            "k_shot": int(job["k_shot"]),
            "new_class_count": int(job["new_class_count"]),
            "scenario": scene,
        }
        for metric in _SCENARIO_METRICS:
            row[f"candidate_{metric}"] = candidate[metric]
            row[f"baseline_{metric}"] = baseline_values[metric]
            row[f"delta_{metric}"] = candidate[metric] - baseline_values[metric]
        result.append(row)
    return result


def analyze_d92_e0_full_only_target125(
    matrix_manifest_path: str | Path,
    *,
    run_root: str | Path | None,
    method_lock_path: str | Path,
    baseline_row_metrics_path: str | Path,
    baseline_scenario_metrics_path: str | Path,
    baseline_per_tx_metrics_path: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(matrix_manifest_path).resolve(strict=True)
    manifest_sha = _sha256(manifest_path)
    manifest = _read_json(manifest_path)
    lock_path = Path(method_lock_path).resolve(strict=True)
    lock_sha = _sha256(lock_path)
    lock = _read_json(lock_path)
    try:
        validate_method_lock(lock)
        validate_target125_manifest(
            manifest,
            expected_method_lock_sha256=lock_sha,
            require_package_hashes=True,
        )
    except ValueError as error:
        raise D92E0FullOnlyTarget125AnalysisError("matrix/method lock drift") from error
    root = Path(run_root).resolve(strict=True) if run_root is not None else Path(str(manifest["output_root"])).resolve(strict=True)
    if (root / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json").exists():
        raise D92E0FullOnlyTarget125AnalysisError("systemic stop marker exists")

    expected_by_shard: dict[int, list[str]] = {index: [] for index in range(SHARD_COUNT)}
    for job in manifest["jobs"]:
        expected_by_shard[int(job["planned_shard_index"])].append(str(job["job_id"]))
    for shard, expected_ids in expected_by_shard.items():
        summary = _read_json(root / "summaries" / f"shard_{shard}.json")
        if (
            summary.get("schema") != "cvs.phase2.d92_e0_full_only_target125.shard_summary.v1"
            or summary.get("status") != "PASS"
            or int(summary.get("shard_index", -1)) != shard
            or int(summary.get("selected_job_count", -1)) != len(expected_ids)
            or int(summary.get("completed_job_count", -1)) != len(expected_ids)
            or int(summary.get("failed_job_count", -1)) != 0
            or summary.get("performance_result_allowed") is not True
            or not isinstance(summary.get("completed_job_ids"), list)
            or len(summary["completed_job_ids"]) != len(set(summary["completed_job_ids"]))
            or set(summary["completed_job_ids"]) != set(expected_ids)
        ):
            raise D92E0FullOnlyTarget125AnalysisError("shard summary closure drift")

    baseline_rows = _read_csv(Path(baseline_row_metrics_path), BASELINE_ROW_SHA256)
    baseline_scenarios = _read_csv(Path(baseline_scenario_metrics_path), BASELINE_SCENARIO_SHA256)
    baseline_tx = _read_csv(Path(baseline_per_tx_metrics_path), BASELINE_PER_TX_SHA256)
    baseline_by_key = {_key(row): row for row in baseline_rows}
    scenario_by_key = {(*_key(row), str(row["scenario"])): row for row in baseline_scenarios}
    if len(baseline_by_key) != 125 or len(scenario_by_key) != 375:
        raise D92E0FullOnlyTarget125AnalysisError("baseline Cartesian closure drift")
    baseline_tx_group: dict[tuple[str, int, int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in baseline_tx:
        if row.get("state") == "after" and row.get("role") == "target_old":
            baseline_tx_group[(*_key(row), str(row["tx"]))].append(row)

    paired_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        key = _key(job)
        if key not in baseline_by_key:
            raise D92E0FullOnlyTarget125AnalysisError("job/baseline key mismatch")
        job_root = root / "jobs" / str(job["outer_key"]) / ARM_ID
        receipt = _read_json(job_root / "job_receipt.json")
        score_path = job_root / "scorer" / "diag_cosine_score.json"
        score = _read_json(score_path)
        before_path = job_root / "diag" / "before" / "prediction_artifact.npz"
        after_path = job_root / "diag" / "after" / "prediction_artifact.npz"
        if (
            receipt.get("schema") != "cvs.phase2.d92_e0_full_only_target125.job_receipt.v1"
            or receipt.get("status") != "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE"
            or receipt.get("job_id") != job.get("job_id")
            or receipt.get("outer_key") != job.get("outer_key")
            or receipt.get("arm_id") != ARM_ID
            or receipt.get("candidate") != CANDIDATE_ID
            or receipt.get("matrix_manifest_sha256") != manifest_sha
            or receipt.get("method_lock_sha256") != lock_sha
            or receipt.get("selection_sha256") != CANONICAL_SELECTION_SHA256
            or receipt.get("truth_sidecar_exposed_to_predictor") is not False
            or receipt.get("query_truth_joined_only_after_immutable_predictions") is not True
            or receipt.get("query_truth_fed_back_to_predictor") is not False
            or receipt.get("before_prediction_sha256") != _sha256(before_path)
            or receipt.get("after_prediction_sha256") != _sha256(after_path)
            or receipt.get("score_sha256") != _sha256(score_path)
        ):
            raise D92E0FullOnlyTarget125AnalysisError("job receipt binding drift")
        if _prediction_closure_status(job_root / "diag") != ("closed", "closed"):
            raise D92E0FullOnlyTarget125AnalysisError("prediction closure drift")
        if (
            score.get("schema") != "cvs.phase2.diag_cosine_dev_pair_score.v1"
            or score.get("candidate") != CANDIDATE_ID
            or score.get("before_prediction_sha256") != receipt.get("before_prediction_sha256")
            or score.get("after_prediction_sha256") != receipt.get("after_prediction_sha256")
            or score.get("query_truth_joined_only_after_immutable_predictions") is not True
            or score.get("query_truth_fed_back_to_predictor") is not False
        ):
            raise D92E0FullOnlyTarget125AnalysisError("score binding drift")
        candidate = _score_metrics(score)
        baseline = _baseline_metrics(baseline_by_key[key])
        resource = _fit_resource(job_root, int(job["k_shot"]))
        row: dict[str, Any] = {
            "outer_key": str(job["outer_key"]),
            "receiver": str(job["receiver"]),
            "seed": int(job["seed"]),
            "k_shot": int(job["k_shot"]),
            "new_class_count": int(job["new_class_count"]),
            "slice": f"K{int(job['k_shot'])}_new{int(job['new_class_count'])}",
            **resource,
        }
        for metric in (*_METRICS, "da1_reg0_old_acc", "da1_reg0_old_floor"):
            row[f"candidate_{metric}"] = candidate[metric]
            row[f"baseline_{metric}"] = baseline[metric]
            row[f"delta_{metric}"] = candidate[metric] - baseline[metric]
        paired_rows.append(row)
        scenario_rows.extend(_scenario_pairs(score, job, scenario_by_key))
        for tx, tx_row in score["after"]["by_tx"].items():
            if tx_row.get("role") != "target_old":
                continue
            source = baseline_tx_group.get((*key, str(tx)), [])
            total = sum(int(item["count"]) for item in source)
            if len(source) != len(SCENES) or total <= 0:
                raise D92E0FullOnlyTarget125AnalysisError("baseline per-old-class closure drift")
            baseline_accuracy = sum(int(item["count"]) * float(item["accuracy"]) for item in source) / total
            candidate_accuracy = _finite(tx_row.get("accuracy"), "candidate old-class accuracy", lower=0.0, upper=1.0)
            per_class_rows.append({
                "outer_key": str(job["outer_key"]),
                "tx": str(tx),
                "candidate_accuracy": candidate_accuracy,
                "baseline_accuracy": baseline_accuracy,
                "delta_accuracy": candidate_accuracy - baseline_accuracy,
            })

    if len(paired_rows) != 125 or len(scenario_rows) != 375 or len(per_class_rows) != 750:
        raise D92E0FullOnlyTarget125AnalysisError("complete result row closure drift")
    k_gt_2 = [row for row in paired_rows if int(row["k_shot"]) > 2]
    k1 = [row for row in paired_rows if int(row["k_shot"]) == 1]
    gates = {
        "complete_artifact_closure": {"passed": len(paired_rows) == 125, "observed": len(paired_rows), "threshold": 125},
        "k_gt_2_mean_delta_h": {"passed": _mean(row["delta_h_old_new"] for row in k_gt_2) > _TOLERANCE, "observed": _mean(row["delta_h_old_new"] for row in k_gt_2), "threshold": ">0"},
        "k_gt_2_nonnegative_h_rows": {"passed": sum(row["delta_h_old_new"] >= -_TOLERANCE for row in k_gt_2) >= 80, "observed": sum(row["delta_h_old_new"] >= -_TOLERANCE for row in k_gt_2), "threshold": ">=80/100"},
        "all125_mean_delta_old_acc": {"passed": _mean(row["delta_old_acc"] for row in paired_rows) >= -_TOLERANCE, "observed": _mean(row["delta_old_acc"] for row in paired_rows), "threshold": ">=0"},
        "all125_mean_delta_old_floor": {"passed": _mean(row["delta_old_floor"] for row in paired_rows) >= -_TOLERANCE, "observed": _mean(row["delta_old_floor"] for row in paired_rows), "threshold": ">=0"},
        "all125_mean_delta_seen_new": {"passed": _mean(row["delta_seen_new_acc"] for row in paired_rows) >= -_TOLERANCE, "observed": _mean(row["delta_seen_new_acc"] for row in paired_rows), "threshold": ">=0"},
        "all125_mean_delta_forgetting": {"passed": _mean(row["delta_forgetting"] for row in paired_rows) <= _TOLERANCE, "observed": _mean(row["delta_forgetting"] for row in paired_rows), "threshold": "<=0"},
        "fit_count_exact": {"passed": all((row["fit_count"], row["actual_fit_count"]) == ((3, 3) if row["k_shot"] == 1 else (2, 1)) for row in paired_rows), "observed": "K1=3/3,K5/K10=2/1", "threshold": "exact"},
        "query_protocol_zero_access": {"passed": True, "observed": True, "threshold": True},
    }
    aggregate = {
        "row_count": 125,
        **{f"candidate_mean_{metric}": _mean(row[f"candidate_{metric}"] for row in paired_rows) for metric in _METRICS},
        **{f"baseline_mean_{metric}": _mean(row[f"baseline_{metric}"] for row in paired_rows) for metric in _METRICS},
        **{f"mean_delta_{metric}": _mean(row[f"delta_{metric}"] for row in paired_rows) for metric in _METRICS},
        "k_gt_2_mean_delta_h": _mean(row["delta_h_old_new"] for row in k_gt_2),
        "k_gt_2_nonnegative_h_row_count": sum(row["delta_h_old_new"] >= -_TOLERANCE for row in k_gt_2),
        "k1_mean_delta_h": _mean(row["delta_h_old_new"] for row in k1),
        "median_registration_wall_time_ns": float(statistics.median(row["registration_wall_time_ns"] for row in paired_rows)),
        "median_registration_process_cpu_time_ns": float(statistics.median(row["registration_process_cpu_time_ns"] for row in paired_rows)),
        "median_registration_incremental_peak_working_set_bytes": float(statistics.median(row["registration_incremental_peak_working_set_bytes"] for row in paired_rows)),
    }
    class_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_class_rows:
        class_group[str(row["tx"])].append(row)
    per_old_class = [
        {
            "tx": tx,
            "row_count": len(rows),
            "candidate_accuracy": _mean(row["candidate_accuracy"] for row in rows),
            "baseline_accuracy": _mean(row["baseline_accuracy"] for row in rows),
            "delta_accuracy": _mean(row["delta_accuracy"] for row in rows),
        }
        for tx, rows in sorted(class_group.items())
    ]
    scenario_group = _group_summary(scenario_rows, "scenario", metrics=_SCENARIO_METRICS)
    resource_by_slice = []
    for group in _group_summary(paired_rows, "slice"):
        members = [row for row in paired_rows if row["slice"] == group["slice"]]
        resource_by_slice.append({
            "slice": group["slice"],
            "row_count": len(members),
            "median_wall_time_ns": float(statistics.median(row["registration_wall_time_ns"] for row in members)),
            "median_cpu_time_ns": float(statistics.median(row["registration_process_cpu_time_ns"] for row in members)),
            "median_incremental_peak_bytes": float(statistics.median(row["registration_incremental_peak_working_set_bytes"] for row in members)),
            "fit_count": int(members[0]["fit_count"]),
            "actual_fit_count": int(members[0]["actual_fit_count"]),
            "query_macs": int(statistics.median(row["query_macs"] for row in members)),
            "state_bytes": int(statistics.median(row["state_bytes"] for row in members)),
        })
    all_gates_pass = all(bool(item["passed"]) for item in gates.values())
    return {
        "schema": "cvs.phase2.d92_e0_full_only_target125.analysis.v1",
        "status": "ANALYZED",
        "claim_scope": manifest.get("claim_scope"),
        "matrix_manifest_sha256": manifest_sha,
        "method_lock_sha256": lock_sha,
        "selection_sha256": CANONICAL_SELECTION_SHA256,
        "baseline": {
            "row_metrics_sha256": BASELINE_ROW_SHA256,
            "scenario_metrics_sha256": BASELINE_SCENARIO_SHA256,
            "per_tx_metrics_sha256": BASELINE_PER_TX_SHA256,
        },
        "aggregate": aggregate,
        "by_receiver": _group_summary(paired_rows, "receiver"),
        "by_seed": _group_summary(paired_rows, "seed"),
        "by_slice": _group_summary(paired_rows, "slice"),
        "by_scenario": scenario_group,
        "resource_by_slice": resource_by_slice,
        "per_old_class": per_old_class,
        "paired_rows": paired_rows,
        "scenario_rows": scenario_rows,
        "per_old_class_rows": per_class_rows,
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "verdict": "PROMOTE_E0_FULL_ONLY_TARGET125" if all_gates_pass else "NO_TARGET125_PROMOTION",
    }


__all__ = [
    "BASELINE_PER_TX_SHA256",
    "BASELINE_ROW_SHA256",
    "BASELINE_SCENARIO_SHA256",
    "D92E0FullOnlyTarget125AnalysisError",
    "analyze_d92_e0_full_only_target125",
]
