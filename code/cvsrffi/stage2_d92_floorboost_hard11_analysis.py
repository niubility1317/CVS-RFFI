"""Analyze the immutable D92 floor-boost Hard11 screen.

The ten performance rows are paired with the frozen E0_FULL_ONLY/D92
``paired_rows.csv``.  The K1 row is retained as liveness evidence only and is
never included in performance means or promotion gates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cvsrffi.stage2_d92_floorboost_hard11 import (
    ARM_ID,
    CANDIDATE_ID,
    CANONICAL_SELECTION_SHA256,
    CONTRAST_LAMBDA,
    HARD11_ROWS,
    HISTORICAL_BASELINE_PATH,
    HISTORICAL_BASELINE_SHA256,
    MARGIN_QUANTILE,
    QUANTILE_METHOD,
    RETENTION_BIAS_KAPPA,
    SCENES,
    SHARD_COUNT,
    validate_hard11_manifest,
    validate_method_lock,
)
from scripts.run_d92_floorboost_hard11 import QUERY_ZERO_FIELDS, _prediction_closure_status


_TOLERANCE = 1.0e-12
_METRICS = ("h_old_new", "old_acc", "old_floor", "seen_new_acc", "forgetting")
HISTORICAL_BASELINE_SHA256 = HISTORICAL_BASELINE_SHA256


class D92FloorboostHard11AnalysisError(ValueError):
    """Raised when Hard11 evidence is incomplete, detached or malformed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise D92FloorboostHard11AnalysisError(f"missing JSON artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92FloorboostHard11AnalysisError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise D92FloorboostHard11AnalysisError(f"JSON artifact is not an object: {path}")
    return payload


def _read_csv(path: Path, expected_sha256: str) -> list[dict[str, str]]:
    if not path.is_file() or path.is_symlink() or _sha256(path) != expected_sha256:
        raise D92FloorboostHard11AnalysisError(f"frozen historical baseline identity drift: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != 125:
        raise D92FloorboostHard11AnalysisError("historical paired-row closure drift")
    return rows


def _finite(value: Any, label: str, *, lower: float | None = None, upper: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise D92FloorboostHard11AnalysisError(f"non-numeric {label}") from error
    if not math.isfinite(result) or (lower is not None and result < lower) or (upper is not None and result > upper):
        raise D92FloorboostHard11AnalysisError(f"out-of-range {label}")
    return result


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        raise D92FloorboostHard11AnalysisError("empty mean")
    return float(statistics.fmean(rows))


def _p90(values: Iterable[float]) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        raise D92FloorboostHard11AnalysisError("empty percentile")
    index = max(0, min(len(rows) - 1, math.ceil(0.90 * len(rows)) - 1))
    return rows[index]


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
        raise D92FloorboostHard11AnalysisError("score state surface missing")
    return {
        "h_old_new": _finite(after.get("h_old_new"), "H", lower=0.0, upper=1.0),
        "old_acc": _finite(after.get("old_acc"), "old accuracy", lower=0.0, upper=1.0),
        "old_floor": _finite(score.get("per_old_class_floor_after"), "old floor", lower=0.0, upper=1.0),
        "seen_new_acc": _finite(after.get("seen_new_acc"), "seen-new accuracy", lower=0.0, upper=1.0),
        "forgetting": _finite(score.get("old_forgetting_pp"), "forgetting") / 100.0,
        "da1_reg0_old_acc": _finite(before.get("old_acc"), "DA1_REG0 old accuracy", lower=0.0, upper=1.0),
        "da1_reg0_old_floor": _finite(score.get("per_old_class_floor_before"), "DA1_REG0 old floor", lower=0.0, upper=1.0),
    }


def _historical_metrics(row: Mapping[str, str], prefix: str) -> dict[str, float]:
    names = {
        "h_old_new": f"{prefix}_h_old_new",
        "old_acc": f"{prefix}_old_acc",
        "old_floor": f"{prefix}_old_floor",
        "seen_new_acc": f"{prefix}_seen_new_acc",
        "forgetting": f"{prefix}_forgetting",
    }
    return {
        "h_old_new": _finite(row.get(names["h_old_new"]), f"{prefix} H", lower=0.0, upper=1.0),
        "old_acc": _finite(row.get(names["old_acc"]), f"{prefix} old accuracy", lower=0.0, upper=1.0),
        "old_floor": _finite(row.get(names["old_floor"]), f"{prefix} old floor", lower=0.0, upper=1.0),
        "seen_new_acc": _finite(row.get(names["seen_new_acc"]), f"{prefix} seen-new", lower=0.0, upper=1.0),
        "forgetting": _finite(row.get(names["forgetting"]), f"{prefix} forgetting"),
    }


def _floorboost_receipt(row: Mapping[str, Any], k_shot: int) -> None:
    """Validate the exact flat receipt emitted by the floorboost core."""

    prefix = "d92_e0d_floorboost_"
    # K1/K2 is an exact D92 alias.  The alias path intentionally has no
    # floorboost statistics, so only its state triplet is required here.
    active = row.get(prefix + "active")
    fallback_active = row.get(prefix + "fallback_active")
    fallback_reason = row.get(prefix + "fallback_reason")
    if int(k_shot) <= 2:
        if (
            active is not False
            or fallback_active is not False
            or fallback_reason != "K1_K2_EXACT_D92_FULL_ALIAS"
        ):
            raise D92FloorboostHard11AnalysisError("floorboost K1/K2 fallback drift")
        return

    # K>2 has two legal states: a successful active receipt, or a numeric
    # fail-close receipt that falls back to the full baseline.  Both states
    # retain scalar parameters/costs; only the per-old-class vectors may be
    # absent in the fallback state.
    if prefix + "fallback_reason" not in row:
        raise D92FloorboostHard11AnalysisError("floorboost support receipt missing: fallback_reason")
    if active is True and fallback_active is False and fallback_reason in (None, ""):
        receipt_state = "success"
    elif active is False and fallback_active is True and isinstance(fallback_reason, str) and fallback_reason:
        receipt_state = "fallback"
    else:
        raise D92FloorboostHard11AnalysisError("floorboost active/fallback drift")

    required = {
        "active": active,
        "lambda": row.get(prefix + "lambda"),
        "quantile": row.get(prefix + "quantile"),
        "quantile_method": row.get(prefix + "quantile_method"),
        "kappa": row.get(prefix + "kappa"),
        "fallback_active": fallback_active,
        "fallback_reason": fallback_reason,
        "new_rows_byte_exact": row.get(prefix + "new_rows_byte_exact"),
        "old_bias_zero_sum_residual_abs": row.get(prefix + "old_bias_zero_sum_residual_abs"),
        "old_intercept_mean_residual_abs": row.get(prefix + "old_intercept_mean_residual_abs"),
        "max_abs_delta_over_rms": row.get(prefix + "max_abs_delta_over_rms"),
        "full_old_rms": row.get(prefix + "full_old_rms"),
        "retention_score_by_old_class": row.get(prefix + "retention_score_by_old_class"),
        "registration_drift_by_old_class": row.get(prefix + "registration_drift_by_old_class"),
        "delta_bias_by_old_class": row.get(prefix + "delta_bias_by_old_class"),
        "support_ocf_alignment_macs_upper_bound": row.get(prefix + "support_ocf_alignment_macs_upper_bound"),
        "support_retention_affine_macs_upper_bound": row.get(prefix + "support_retention_affine_macs_upper_bound"),
        "support_bias_calibration_macs_upper_bound": row.get(prefix + "support_bias_calibration_macs_upper_bound"),
        "support_macs_upper_bound": row.get(prefix + "support_macs_upper_bound"),
        "support_transient_bytes_upper_bound": row.get(prefix + "support_transient_bytes_upper_bound"),
        "persistent_state_bytes_delta": row.get(prefix + "persistent_state_bytes_delta"),
    }
    scalar_names = (
        "lambda",
        "quantile",
        "quantile_method",
        "kappa",
        "new_rows_byte_exact",
        "old_bias_zero_sum_residual_abs",
        "old_intercept_mean_residual_abs",
        "max_abs_delta_over_rms",
        "full_old_rms",
        "support_ocf_alignment_macs_upper_bound",
        "support_retention_affine_macs_upper_bound",
        "support_bias_calibration_macs_upper_bound",
        "support_macs_upper_bound",
        "support_transient_bytes_upper_bound",
        "persistent_state_bytes_delta",
    )
    missing = [name for name in scalar_names if required[name] is None]
    vector_names = (
        "retention_score_by_old_class",
        "registration_drift_by_old_class",
        "delta_bias_by_old_class",
    )
    if receipt_state == "success":
        missing.extend(name for name in vector_names if required[name] is None)
    if missing:
        raise D92FloorboostHard11AnalysisError(
            "floorboost support receipt missing: " + ",".join(missing)
        )
    if abs(float(required["lambda"]) - CONTRAST_LAMBDA) > _TOLERANCE:
        raise D92FloorboostHard11AnalysisError("floorboost lambda drift")
    if abs(float(required["quantile"]) - MARGIN_QUANTILE) > _TOLERANCE:
        raise D92FloorboostHard11AnalysisError("floorboost quantile drift")
    if required["quantile_method"] != QUANTILE_METHOD:
        raise D92FloorboostHard11AnalysisError("floorboost quantile method drift")
    if abs(float(required["kappa"]) - RETENTION_BIAS_KAPPA) > _TOLERANCE:
        raise D92FloorboostHard11AnalysisError("floorboost kappa drift")
    if required["new_rows_byte_exact"] is not True:
        raise D92FloorboostHard11AnalysisError("floorboost new-row identity drift")
    if abs(float(required["old_bias_zero_sum_residual_abs"])) > 1.0e-5 or abs(float(required["old_intercept_mean_residual_abs"])) > 1.0e-5:
        raise D92FloorboostHard11AnalysisError("floorboost old-class mean residual drift")
    if float(required["full_old_rms"]) < 0.0 or float(required["max_abs_delta_over_rms"]) > RETENTION_BIAS_KAPPA + 1.0e-6:
        raise D92FloorboostHard11AnalysisError("floorboost bias RMS/cap drift")
    for name in vector_names:
        # Numeric fail-close receipts may omit these diagnostics.  If a
        # fallback does provide one, validate it exactly as a success receipt.
        if receipt_state == "fallback" and required[name] is None:
            continue
        if not isinstance(required[name], (list, tuple, Mapping)) or len(required[name]) != 6:
            raise D92FloorboostHard11AnalysisError(f"floorboost {name} missing")
        values = required[name].values() if isinstance(required[name], Mapping) else required[name]
        for value in values:
            if isinstance(value, Mapping):
                value = value.get("value", value.get("score", value.get("delta", 0.0)))
            if not math.isfinite(float(value)):
                raise D92FloorboostHard11AnalysisError(f"floorboost {name} non-finite")
    for name in (
        "support_ocf_alignment_macs_upper_bound",
        "support_retention_affine_macs_upper_bound",
        "support_bias_calibration_macs_upper_bound",
        "support_macs_upper_bound",
        "support_transient_bytes_upper_bound",
    ):
        if int(required[name]) < 0:
            raise D92FloorboostHard11AnalysisError(f"floorboost {name} invalid")
    if int(required["persistent_state_bytes_delta"]) != 0:
        raise D92FloorboostHard11AnalysisError("floorboost persistent state drift")


def _fit_resource(job_root: Path, k_shot: int) -> dict[str, float | int | str]:
    path = job_root / "diag" / "after" / "fit_audit.json"
    if not path.is_file() or path.is_symlink():
        raise D92FloorboostHard11AnalysisError("after fit audit missing")
    try:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92FloorboostHard11AnalysisError("after fit audit invalid") from error
    if not isinstance(rows, list) or len(rows) != len(SCENES) or any(not isinstance(row, Mapping) for row in rows):
        raise D92FloorboostHard11AnalysisError("after fit audit scene closure drift")
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
            raise D92FloorboostHard11AnalysisError("fit audit arm identity drift")
        if any(row.get(field) is not False for field in QUERY_ZERO_FIELDS):
            raise D92FloorboostHard11AnalysisError("query access is not zero")
        _floorboost_receipt(row, k_shot)
        totals.add(int(row.get("after_total_component_fit_count", row.get("d92_component_fit_count", -1))))
        inventory = row.get("after_actual_component_inventory", row.get("d92_component_fit_inventory"))
        if not isinstance(inventory, Mapping):
            raise D92FloorboostHard11AnalysisError("actual fit inventory missing")
        actuals.add(int(inventory.get("actual_component_fit_count", -1)))
        query_macs.add(int(row.get("query_macs", -1)))
        state_bytes.add(int(row.get("after_state_bytes", -1)))
        modes.add(str(row.get("after_registered_d_mode_effective", row.get("d92_registered_d_mode_effective"))))
        resource = row.get("after_registration_resource")
        if not isinstance(resource, Mapping):
            raise D92FloorboostHard11AnalysisError("registration resource receipt missing")
        walls.append(_finite(resource.get("registration_wall_time_ns"), "wall", lower=0.0))
        cpus.append(_finite(resource.get("registration_process_cpu_time_ns"), "CPU time", lower=0.0))
        peaks.append(_finite(resource.get("registration_incremental_peak_working_set_bytes"), "peak", lower=0.0))
    expected_total, expected_actual = (3, 3) if k_shot <= 2 else (4, 2)
    if totals != {expected_total} or actuals != {expected_actual} or len(query_macs) != 1 or len(state_bytes) != 1 or min(query_macs) < 0 or min(state_bytes) < 0:
        raise D92FloorboostHard11AnalysisError("fit/resource count closure drift")
    expected_modes = {"d92_full_alias"} if int(k_shot) <= 2 else {"floorboost", "full_only", "floorboost_fallback"}
    if not modes or not modes.issubset(expected_modes):
        raise D92FloorboostHard11AnalysisError("registered floorboost mode drift")
    return {
        "fit_count": expected_total,
        "actual_fit_count": expected_actual,
        "registered_d_mode": sorted(modes)[0],
        "query_macs": next(iter(query_macs)),
        "state_bytes": next(iter(state_bytes)),
        "registration_wall_time_ns": float(statistics.median(walls)),
        "registration_process_cpu_time_ns": float(statistics.median(cpus)),
        "registration_incremental_peak_working_set_bytes": float(statistics.median(peaks)),
    }


def _group_summary(rows: Sequence[Mapping[str, Any]], key_name: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key_name])].append(row)
    result: list[dict[str, Any]] = []
    for group, members in sorted(grouped.items()):
        item: dict[str, Any] = {key_name: group, "row_count": len(members)}
        for metric in _METRICS:
            item[f"candidate_{metric}"] = _mean(float(row[f"candidate_{metric}"]) for row in members)
            item[f"d92_{metric}"] = _mean(float(row[f"d92_{metric}"]) for row in members)
            item[f"full_only_{metric}"] = _mean(float(row[f"full_only_{metric}"]) for row in members)
            item[f"delta_{metric}_vs_d92"] = _mean(float(row[f"delta_{metric}_vs_d92"]) for row in members)
            item[f"delta_{metric}_vs_full_only"] = _mean(float(row[f"delta_{metric}_vs_full_only"]) for row in members)
        result.append(item)
    return result


def _candidate_scenario_rows(score: Mapping[str, Any], job: Mapping[str, Any]) -> list[dict[str, Any]]:
    before = score.get("before")
    after = score.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return []
    before_by_scene = before.get("by_scenario")
    after_by_scene = after.get("by_scenario")
    if not isinstance(before_by_scene, Mapping) or not isinstance(after_by_scene, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for scene in SCENES:
        old_before = before_by_scene.get(scene)
        old_after = after_by_scene.get(scene)
        if not isinstance(old_before, Mapping) or not isinstance(old_after, Mapping):
            raise D92FloorboostHard11AnalysisError("candidate scenario closure drift")
        rows.append(
            {
                "outer_key": str(job["outer_key"]),
                "outer_role": str(job["outer_role"]),
                "receiver": str(job["receiver"]),
                "seed": int(job["seed"]),
                "k_shot": int(job["k_shot"]),
                "new_class_count": int(job["new_class_count"]),
                "scenario": scene,
                "candidate_h_old_new": _finite(old_after.get("h_old_new"), "scenario H", lower=0.0, upper=1.0),
                "candidate_old_acc": _finite(old_after.get("old_acc"), "scenario old accuracy", lower=0.0, upper=1.0),
                "candidate_seen_new_acc": _finite(old_after.get("seen_new_acc"), "scenario seen-new", lower=0.0, upper=1.0),
                "candidate_forgetting": _finite(old_before.get("old_acc"), "scenario before old", lower=0.0, upper=1.0) - _finite(old_after.get("old_acc"), "scenario after old", lower=0.0, upper=1.0),
            }
        )
    return rows


def decide_verdict(gate_state: Mapping[str, bool]) -> str:
    """Apply the frozen three-way Hard11 decision rule."""

    complete = bool(gate_state.get("complete_artifact_closure"))
    performance = bool(gate_state.get("performance_outer_closure", True))
    if complete and performance and gate_state.get("all_advance_core") and gate_state.get("all_forgetting"):
        return "ADVANCE_TO_FULL125"
    if complete and performance and gate_state.get("revision_gate_passed") and not gate_state.get("hard_reject"):
        return "REVISE_ONCE_FLOORBOOST"
    return "REJECT_FLOORBOOST"


def analyze_d92_floorboost_hard11(
    matrix_manifest_path: str | Path,
    *,
    run_root: str | Path | None,
    method_lock_path: str | Path,
    baseline_paired_rows_path: str | Path = HISTORICAL_BASELINE_PATH,
) -> dict[str, Any]:
    manifest_path = Path(matrix_manifest_path).resolve(strict=True)
    manifest_sha = _sha256(manifest_path)
    manifest = _read_json(manifest_path)
    lock_path = Path(method_lock_path).resolve(strict=True)
    lock_sha = _sha256(lock_path)
    lock = _read_json(lock_path)
    try:
        validate_method_lock(lock)
        validate_hard11_manifest(manifest, expected_method_lock_sha256=lock_sha, require_package_hashes=True)
    except ValueError as error:
        raise D92FloorboostHard11AnalysisError("matrix/method lock drift") from error
    root = Path(run_root).resolve(strict=True) if run_root is not None else Path(str(manifest["output_root"])).resolve(strict=True)
    if (root / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json").exists():
        raise D92FloorboostHard11AnalysisError("systemic stop marker exists")
    expected_by_shard: dict[int, list[str]] = {index: [] for index in range(SHARD_COUNT)}
    for job in manifest["jobs"]:
        expected_by_shard[int(job["planned_shard_index"])].append(str(job["job_id"]))
    for shard, expected_ids in expected_by_shard.items():
        summary = _read_json(root / "summaries" / f"shard_{shard}.json")
        if (
            summary.get("schema") != "cvs.phase2.d92_floorboost_hard11.shard_summary.v1"
            or summary.get("status") != "PASS"
            or int(summary.get("shard_index", -1)) != shard
            or int(summary.get("selected_job_count", -1)) != len(expected_ids)
            or int(summary.get("completed_job_count", -1)) != len(expected_ids)
            or int(summary.get("failed_job_count", -1)) != 0
            or summary.get("performance_result_allowed") is not True
            or set(summary.get("completed_job_ids", [])) != set(expected_ids)
        ):
            raise D92FloorboostHard11AnalysisError("shard summary closure drift")
    baseline_path = Path(baseline_paired_rows_path).resolve(strict=True)
    if str(baseline_path).replace("\\", "/").lower() != HISTORICAL_BASELINE_PATH.lower():
        raise D92FloorboostHard11AnalysisError("historical baseline path drift")
    baseline_rows = _read_csv(baseline_path, HISTORICAL_BASELINE_SHA256)
    baseline_by_key = {_key(row): row for row in baseline_rows}
    if len(baseline_by_key) != 125:
        raise D92FloorboostHard11AnalysisError("historical baseline key closure drift")
    paired_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        key = _key(job)
        if key not in baseline_by_key:
            raise D92FloorboostHard11AnalysisError("job/historical baseline key mismatch")
        job_root = root / "jobs" / str(job["outer_key"]) / ARM_ID
        receipt = _read_json(job_root / "job_receipt.json")
        score_path = job_root / "scorer" / "diag_cosine_score.json"
        score = _read_json(score_path)
        scenario_rows.extend(_candidate_scenario_rows(score, job))
        before_path = job_root / "diag" / "before" / "prediction_artifact.npz"
        after_path = job_root / "diag" / "after" / "prediction_artifact.npz"
        if (
            receipt.get("schema") != "cvs.phase2.d92_floorboost_hard11.job_receipt.v1"
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
            raise D92FloorboostHard11AnalysisError("job receipt binding drift")
        if _prediction_closure_status(job_root / "diag") != ("closed", "closed"):
            raise D92FloorboostHard11AnalysisError("prediction closure drift")
        if (
            score.get("schema") != "cvs.phase2.diag_cosine_dev_pair_score.v1"
            or score.get("candidate") != CANDIDATE_ID
            or score.get("before_prediction_sha256") != receipt.get("before_prediction_sha256")
            or score.get("after_prediction_sha256") != receipt.get("after_prediction_sha256")
            or score.get("query_truth_joined_only_after_immutable_predictions") is not True
            or score.get("query_truth_fed_back_to_predictor") is not False
        ):
            raise D92FloorboostHard11AnalysisError("score binding drift")
        candidate = _score_metrics(score)
        d92 = _historical_metrics(baseline_by_key[key], "baseline")
        full = _historical_metrics(baseline_by_key[key], "candidate")
        resource = _fit_resource(job_root, int(job["k_shot"]))
        full_only_query_macs = int(float(baseline_by_key[key].get("query_macs", -1)))
        full_only_state_bytes = int(float(baseline_by_key[key].get("state_bytes", -1)))
        if full_only_query_macs < 0 or full_only_state_bytes < 0:
            raise D92FloorboostHard11AnalysisError("historical query/resource identity missing")
        row: dict[str, Any] = {
            "outer_key": str(job["outer_key"]),
            "outer_role": str(job["outer_role"]),
            "receiver": str(job["receiver"]),
            "seed": int(job["seed"]),
            "k_shot": int(job["k_shot"]),
            "new_class_count": int(job["new_class_count"]),
            "slice": f"K{int(job['k_shot'])}_new{int(job['new_class_count'])}",
            "full_only_query_macs": full_only_query_macs,
            "full_only_state_bytes": full_only_state_bytes,
            **resource,
        }
        for metric in _METRICS:
            row[f"candidate_{metric}"] = candidate[metric]
            row[f"d92_{metric}"] = d92[metric]
            row[f"full_only_{metric}"] = full[metric]
            row[f"delta_{metric}_vs_d92"] = candidate[metric] - d92[metric]
            row[f"delta_{metric}_vs_full_only"] = candidate[metric] - full[metric]
        paired_rows.append(row)
    if len(paired_rows) != 11:
        raise D92FloorboostHard11AnalysisError("complete result row closure drift")
    if scenario_rows and len(scenario_rows) != 33:
        raise D92FloorboostHard11AnalysisError("candidate scenario row closure drift")
    performance = [row for row in paired_rows if row["outer_role"] == "performance"]
    liveness = [row for row in paired_rows if row["outer_role"] == "liveness"]
    if len(performance) != 10 or len(liveness) != 1:
        raise D92FloorboostHard11AnalysisError("Hard11 performance/liveness closure drift")
    floor_vs_full = [row["delta_old_floor_vs_full_only"] for row in performance]
    h_vs_full = [row["delta_h_old_new_vs_full_only"] for row in performance]
    old_vs_full = [row["delta_old_acc_vs_full_only"] for row in performance]
    new_vs_full = [row["delta_seen_new_acc_vs_full_only"] for row in performance]
    forget_vs_full = [row["delta_forgetting_vs_full_only"] for row in performance]
    h_vs_d92 = [row["delta_h_old_new_vs_d92"] for row in performance]
    old_vs_d92 = [row["delta_old_acc_vs_d92"] for row in performance]
    floor_vs_d92 = [row["delta_old_floor_vs_d92"] for row in performance]
    new_vs_d92 = [row["delta_seen_new_acc_vs_d92"] for row in performance]
    forget_vs_d92 = [row["delta_forgetting_vs_d92"] for row in performance]
    historical_floor_drop_count = sum(row["full_only_old_floor"] < row["d92_old_floor"] - _TOLERANCE for row in performance)
    candidate_floor_drop_count = sum(row["candidate_old_floor"] < row["d92_old_floor"] - _TOLERANCE for row in performance)
    resource_wall_p90 = _p90(row["registration_wall_time_ns"] for row in performance)
    resource_peak_p90 = _p90(row["registration_incremental_peak_working_set_bytes"] for row in performance)
    query_state_exact = all(
        int(row["query_macs"]) == int(row["full_only_query_macs"])
        and int(row["state_bytes"]) == int(row["full_only_state_bytes"])
        for row in performance
    )
    gates: dict[str, dict[str, Any]] = {
        "complete_artifact_closure": {"passed": len(paired_rows) == 11, "observed": len(paired_rows), "threshold": 11},
        "performance_outer_closure": {"passed": len(performance) == 10 and len(liveness) == 1, "observed": "10+1", "threshold": "10+1"},
        "mean_delta_old_floor_vs_full_only": {"passed": _mean(floor_vs_full) >= 0.04 - _TOLERANCE, "observed": _mean(floor_vs_full), "threshold": ">=0.04"},
        "mean_delta_h_vs_full_only": {"passed": _mean(h_vs_full) >= 0.008 - _TOLERANCE, "observed": _mean(h_vs_full), "threshold": ">=0.008"},
        "mean_delta_old_balanced_vs_full_only": {"passed": _mean(old_vs_full) >= 0.01 - _TOLERANCE, "observed": _mean(old_vs_full), "threshold": ">=0.01"},
        "mean_delta_seen_new_vs_full_only": {"passed": _mean(new_vs_full) >= -_TOLERANCE, "observed": _mean(new_vs_full), "threshold": ">=0"},
        "mean_delta_forgetting_vs_full_only": {"passed": _mean(forget_vs_full) <= -0.018 + _TOLERANCE, "observed": _mean(forget_vs_full), "threshold": "<=-0.018"},
        "mean_delta_h_vs_d92": {"passed": _mean(h_vs_d92) >= -_TOLERANCE, "observed": _mean(h_vs_d92), "threshold": ">=0"},
        "h_nonnegative_vs_d92": {"passed": sum(value >= -_TOLERANCE for value in h_vs_d92) >= 8, "observed": sum(value >= -_TOLERANCE for value in h_vs_d92), "threshold": ">=8/10"},
        "mean_delta_old_balanced_vs_d92": {"passed": _mean(old_vs_d92) >= -_TOLERANCE, "observed": _mean(old_vs_d92), "threshold": ">=0"},
        "mean_delta_old_floor_vs_d92": {"passed": _mean(floor_vs_d92) >= -_TOLERANCE, "observed": _mean(floor_vs_d92), "threshold": ">=0"},
        "old_floor_nonnegative_vs_d92": {"passed": sum(value >= -_TOLERANCE for value in floor_vs_d92) >= 8, "observed": sum(value >= -_TOLERANCE for value in floor_vs_d92), "threshold": ">=8/10"},
        "worst_delta_old_floor_vs_d92": {"passed": min(floor_vs_d92) >= -0.02 - _TOLERANCE, "observed": min(floor_vs_d92), "threshold": ">=-0.02"},
        "mean_delta_seen_new_vs_d92": {"passed": _mean(new_vs_d92) >= -_TOLERANCE, "observed": _mean(new_vs_d92), "threshold": ">=0"},
        "mean_delta_forgetting_vs_d92": {"passed": _mean(forget_vs_d92) <= -0.005 + _TOLERANCE, "observed": _mean(forget_vs_d92), "threshold": "<=-0.005"},
        "forgetting_nonincrease_vs_d92": {"passed": sum(value <= _TOLERANCE for value in forget_vs_d92) >= 8, "observed": sum(value <= _TOLERANCE for value in forget_vs_d92), "threshold": ">=8/10"},
        "worst_delta_forgetting_vs_d92": {"passed": max(forget_vs_d92) <= 0.005 + _TOLERANCE, "observed": max(forget_vs_d92), "threshold": "<=0.005"},
        "fit_count_exact": {"passed": all((row["fit_count"], row["actual_fit_count"]) == (4, 2) for row in performance), "observed": "K5/K10=4/2", "threshold": "exact"},
        "query_protocol_zero_access": {"passed": True, "observed": True, "threshold": True},
        "registration_wall_p90": {"passed": resource_wall_p90 <= 180_000_000, "observed": resource_wall_p90, "threshold": "<=180e6 ns"},
        "registration_peak_p90": {"passed": resource_peak_p90 <= 3 * 1024 * 1024, "observed": resource_peak_p90, "threshold": "<=3MiB"},
        "query_state_equal_full_only": {"passed": query_state_exact, "observed": query_state_exact, "threshold": True},
    }
    all_forgetting = all(gates[name]["passed"] for name in ("mean_delta_forgetting_vs_full_only", "mean_delta_forgetting_vs_d92", "forgetting_nonincrease_vs_d92", "worst_delta_forgetting_vs_d92"))
    all_advance_core = all(gates[name]["passed"] for name in gates if name not in {"mean_delta_forgetting_vs_full_only", "mean_delta_forgetting_vs_d92", "forgetting_nonincrease_vs_d92", "worst_delta_forgetting_vs_d92", "complete_artifact_closure", "performance_outer_closure"})
    revision_gate = {
        "floor_vs_full_only_ge_0_02": _mean(floor_vs_full) >= 0.02 - _TOLERANCE,
        "floor_drop_count_reduced_by_half": candidate_floor_drop_count <= max(0, historical_floor_drop_count // 2),
        "h_vs_full_only_nonnegative": _mean(h_vs_full) >= -_TOLERANCE,
        "seen_new_vs_full_only_ge_minus_0_002": _mean(new_vs_full) >= -0.002 - _TOLERANCE,
        "forgetting_vs_full_only_le_minus_0_01": _mean(forget_vs_full) <= -0.01 + _TOLERANCE,
        "resources": gates["registration_wall_p90"]["passed"] and gates["registration_peak_p90"]["passed"] and gates["query_state_equal_full_only"]["passed"],
    }
    hard_reject = _mean(floor_vs_full) < 0.02 - _TOLERANCE or _mean(new_vs_full) < -0.005 - _TOLERANCE
    revision_gate_passed = all(bool(value) for value in revision_gate.values())
    gate_state = {
        "complete_artifact_closure": gates["complete_artifact_closure"]["passed"],
        "performance_outer_closure": gates["performance_outer_closure"]["passed"],
        "all_advance_core": all_advance_core,
        "all_forgetting": all_forgetting,
        "revision_gate_passed": revision_gate_passed,
        "hard_reject": hard_reject,
    }
    verdict = decide_verdict(gate_state)
    aggregate = {
        "row_count": len(paired_rows),
        "performance_row_count": len(performance),
        "liveness_row_count": len(liveness),
        **{f"candidate_mean_{metric}": _mean(row[f"candidate_{metric}"] for row in performance) for metric in _METRICS},
        **{f"d92_mean_{metric}": _mean(row[f"d92_{metric}"] for row in performance) for metric in _METRICS},
        **{f"full_only_mean_{metric}": _mean(row[f"full_only_{metric}"] for row in performance) for metric in _METRICS},
        **{f"mean_delta_{metric}_vs_d92": _mean(row[f"delta_{metric}_vs_d92"] for row in performance) for metric in _METRICS},
        **{f"mean_delta_{metric}_vs_full_only": _mean(row[f"delta_{metric}_vs_full_only"] for row in performance) for metric in _METRICS},
        "worst_delta_forgetting_vs_d92": max(forget_vs_d92),
        "historical_floor_drop_count": historical_floor_drop_count,
        "candidate_floor_drop_count": candidate_floor_drop_count,
        "registration_wall_p90_ns": resource_wall_p90,
        "registration_peak_p90_bytes": resource_peak_p90,
    }
    resource_by_slice: list[dict[str, Any]] = []
    for slice_name in sorted({str(row["slice"]) for row in performance}):
        members = [row for row in performance if row["slice"] == slice_name]
        resource_by_slice.append(
            {
                "slice": slice_name,
                "row_count": len(members),
                "median_wall_time_ns": float(statistics.median(row["registration_wall_time_ns"] for row in members)),
                "median_process_cpu_time_ns": float(statistics.median(row["registration_process_cpu_time_ns"] for row in members)),
                "median_peak_bytes": float(statistics.median(row["registration_incremental_peak_working_set_bytes"] for row in members)),
                "query_macs": int(statistics.median(row["query_macs"] for row in members)),
                "state_bytes": int(statistics.median(row["state_bytes"] for row in members)),
            }
        )
    return {
        "schema": "cvs.phase2.d92_floorboost_hard11.analysis.v1",
        "status": "ANALYZED",
        "claim_scope": manifest.get("claim_scope"),
        "matrix_manifest_sha256": manifest_sha,
        "method_lock_sha256": lock_sha,
        "selection_sha256": CANONICAL_SELECTION_SHA256,
        "baseline": {"paired_rows_sha256": HISTORICAL_BASELINE_SHA256, "paired_rows_path": str(baseline_path)},
        "aggregate": aggregate,
        "by_receiver": _group_summary(paired_rows, "receiver"),
        "by_seed": _group_summary(paired_rows, "seed"),
        "by_slice": _group_summary(paired_rows, "slice"),
        "paired_rows": paired_rows,
        "scenario_rows": scenario_rows,
        "resource_by_slice": resource_by_slice,
        "liveness_rows": liveness,
        "gates": gates,
        "revision_gate": revision_gate,
        "gate_state": gate_state,
        "all_gates_pass": verdict == "ADVANCE_TO_FULL125",
        "verdict": verdict,
    }


analyze_floorboost_hard11 = analyze_d92_floorboost_hard11


__all__ = [
    "D92FloorboostHard11AnalysisError",
    "HISTORICAL_BASELINE_SHA256",
    "analyze_d92_floorboost_hard11",
    "analyze_floorboost_hard11",
    "decide_verdict",
]
