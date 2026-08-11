"""Strict paired analysis for the development-only D92-E0OCF Hard12-v3 matrix."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_d92_e0d_slim import D92_E0D_ARMS
from cvsrffi.stage2_d92_e0ocf_hard12 import (
    ARM_ORDER,
    CANONICAL_SELECTION_SHA256,
    PRIMARY_ARM,
)


SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
ARMS = ARM_ORDER
SELECTION_SHA256 = CANONICAL_SELECTION_SHA256
_TOLERANCE = 1e-12
_QUERY_ZERO_FIELDS = (
    "query_truth_access", "query_fit_access", "query_update_access", "query_selection_access", "query_role_oracle_access", "query_class_quota_access", "query_global_reassignment"
)


class D92E0OCFAnalysisError(ValueError):
    """Raised when frozen D92-E0OCF evidence is incomplete or inconsistent."""


D92E0OCFHard12V3AnalysisError = D92E0OCFAnalysisError
D92E0OCFHard12AnalysisError = D92E0OCFAnalysisError


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise D92E0OCFAnalysisError(f"required JSON artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92E0OCFAnalysisError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise D92E0OCFAnalysisError(f"JSON artifact must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise D92E0OCFAnalysisError(f"required artifact is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any, *, label: str, lower: float | None = None, upper: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise D92E0OCFAnalysisError(f"invalid {label}") from error
    if not math.isfinite(result) or (lower is not None and result < lower - _TOLERANCE) or (upper is not None and result > upper + _TOLERANCE):
        raise D92E0OCFAnalysisError(f"out-of-range {label}")
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise D92E0OCFAnalysisError("cannot aggregate an empty metric")
    return float(statistics.fmean(values))


def _prediction_key(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    try:
        with np.load(path, allow_pickle=False) as artifact:
            required = {"query_tokens", "scenarios", "predicted_class_handles"}
            if set(artifact.files) != required:
                raise D92E0OCFAnalysisError("prediction artifact member drift")
            tokens = tuple(str(value) for value in artifact["query_tokens"].tolist())
            scenarios = tuple(str(value) for value in artifact["scenarios"].tolist())
            predictions = tuple(str(value) for value in artifact["predicted_class_handles"].tolist())
    except (OSError, ValueError) as error:
        if isinstance(error, D92E0OCFAnalysisError):
            raise
        raise D92E0OCFAnalysisError(f"invalid prediction artifact: {path}") from error
    if not tokens or not (len(tokens) == len(scenarios) == len(predictions)):
        raise D92E0OCFAnalysisError("prediction artifact row-count drift")
    return tokens, scenarios, predictions


def _scenario_mean(score: Mapping[str, Any], state: str, field: str) -> float:
    payload = score.get(state)
    by_scenario = payload.get("by_scenario") if isinstance(payload, Mapping) else None
    if not isinstance(by_scenario, Mapping) or set(by_scenario) != set(SCENES):
        raise D92E0OCFAnalysisError(f"score scenario coverage drift: {state}")
    values = [_finite(by_scenario[scene].get(field), label=f"{state}.{scene}.{field}", lower=0.0, upper=1.0) for scene in SCENES]
    return _mean(values)


def _confusion_rates(score: Mapping[str, Any]) -> dict[str, float]:
    payload = score.get("after")
    by_scenario = payload.get("by_scenario") if isinstance(payload, Mapping) else None
    if not isinstance(by_scenario, Mapping) or set(by_scenario) != set(SCENES):
        raise D92E0OCFAnalysisError("score scenario coverage drift: after")
    old_acc_values: list[float] = []
    old_to_new_values: list[float] = []
    new_to_old_values: list[float] = []
    old_to_old_values: list[float] = []
    for scene in SCENES:
        row = by_scenario[scene]
        old_acc = _finite(row.get("old_acc"), label=f"after.{scene}.old_acc", lower=0.0, upper=1.0)
        old_to_new = _finite(row.get("old_to_new_rate"), label=f"after.{scene}.old_to_new_rate", lower=0.0, upper=1.0)
        new_to_old = _finite(row.get("new_to_old_rate"), label=f"after.{scene}.new_to_old_rate", lower=0.0, upper=1.0)
        old_to_old = 1.0 - old_acc - old_to_new
        if old_to_old < -_TOLERANCE or old_to_old > 1.0 + _TOLERANCE:
            raise D92E0OCFAnalysisError(f"derived after.{scene}.old_to_old_rate out of range")
        old_acc_values.append(old_acc)
        old_to_new_values.append(old_to_new)
        new_to_old_values.append(new_to_old)
        old_to_old_values.append(max(0.0, old_to_old))
    return {"old_acc": _mean(old_acc_values), "old_to_old_rate": _mean(old_to_old_values), "old_to_new_rate": _mean(old_to_new_values), "new_to_old_rate": _mean(new_to_old_values)}


def _old_balanced_accuracy(score: Mapping[str, Any]) -> float:
    after = score.get("after")
    by_tx = after.get("by_tx") if isinstance(after, Mapping) else None
    if not isinstance(by_tx, Mapping):
        # Some compact scorer fixtures expose only per-scenario old accuracy;
        # use that field only when no class-wise table is available.
        return _scenario_mean(score, "after", "old_acc")
    values = [_finite(row.get("accuracy"), label=f"after.by_tx.{tx}.accuracy", lower=0.0, upper=1.0) for tx, row in by_tx.items() if isinstance(row, Mapping) and row.get("role") == "target_old"]
    return _mean(values)


def _load_fit_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise D92E0OCFAnalysisError(f"fit audit is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92E0OCFAnalysisError(f"invalid fit audit: {path}") from error
    if not isinstance(payload, list) or len(payload) != len(SCENES):
        raise D92E0OCFAnalysisError("fit audit scenario count drift")
    rows = [dict(row) for row in payload if isinstance(row, Mapping)]
    if len(rows) != len(SCENES) or {str(row.get("scenario")) for row in rows} != set(SCENES):
        raise D92E0OCFAnalysisError("fit audit scenario identity drift")
    for row in rows:
        if any(row.get(field) is not False for field in _QUERY_ZERO_FIELDS):
            raise D92E0OCFAnalysisError("query protocol audit drift")
        for field in ("before_state_fingerprint_sha256", "after_state_fingerprint_sha256"):
            value = str(row.get(field, ""))
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
                raise D92E0OCFAnalysisError("state fingerprint audit drift")
    return sorted(rows, key=lambda row: str(row["scenario"]))


def _first_value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _expected_fit_count(k_shot: int, arm: str) -> tuple[int, int]:
    if int(k_shot) <= 2:
        return 3, 3
    if arm == "D92_FULL":
        total = 8 * (int(k_shot) + 1)
    elif arm == "E0_FULL_ONLY":
        total = 2
    else:
        total = 4
    return total, total // 2


def _job_evidence(job: Mapping[str, Any], *, run_root: Path | None = None) -> dict[str, Any]:
    arm = str(job.get("arm_id"))
    if arm not in D92_E0D_ARMS or job.get("candidate") != D92_E0D_ARMS[arm].candidate_id:
        raise D92E0OCFAnalysisError("job arm/candidate identity drift")
    job_root = run_root / "jobs" / str(job["outer_key"]) / arm if run_root is not None else Path(str(job["output_root"]))
    receipt = _read_json(job_root / "job_receipt.json")
    score_path = job_root / "scorer" / "diag_cosine_score.json"
    before_path, after_path = job_root / "diag" / "before" / "prediction_artifact.npz", job_root / "diag" / "after" / "prediction_artifact.npz"
    score = _read_json(score_path)
    if receipt.get("schema") != "cvs.phase2.d92_e0ocf_hard12v3.job_receipt.v1" or receipt.get("status") != "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE" or receipt.get("job_id") != job.get("job_id") or receipt.get("outer_key") != job.get("outer_key") or receipt.get("arm_id") != arm or receipt.get("candidate") != job.get("candidate") or receipt.get("role") not in (None, job.get("role")) or receipt.get("truth_sidecar_exposed_to_predictor") is not False or receipt.get("query_truth_joined_only_after_immutable_predictions") is not True or receipt.get("query_truth_fed_back_to_predictor") is not False:
        raise D92E0OCFAnalysisError("job receipt identity/protocol drift")
    before_sha, after_sha, score_sha = _sha256(before_path), _sha256(after_path), _sha256(score_path)
    if score.get("schema") != "cvs.phase2.diag_cosine_dev_pair_score.v1" or receipt.get("before_prediction_sha256") != before_sha or receipt.get("after_prediction_sha256") != after_sha or receipt.get("score_sha256") != score_sha or score.get("before_prediction_sha256") != before_sha or score.get("after_prediction_sha256") != after_sha or score.get("candidate") != job.get("candidate") or score.get("query_truth_joined_only_after_immutable_predictions") is not True or score.get("query_truth_fed_back_to_predictor") is not False:
        raise D92E0OCFAnalysisError("prediction-to-score binding drift")
    if not (job_root / "diag" / "before" / "COMMIT.json").is_file() or not (job_root / "diag" / "after" / "COMMIT.json").is_file():
        raise D92E0OCFAnalysisError("prediction commit closure missing")
    fit_rows = _load_fit_rows(job_root / "diag" / "after" / "fit_audit.json")
    expected_total, expected_actual = _expected_fit_count(int(job["k_shot"]), arm)
    fit_counts, actual_counts, query_macs, state_bytes = set(), set(), set(), set()
    wall_values, peak_values, support_macs, support_bytes = [], [], [], []
    for row in fit_rows:
        fit_counts.add(int(row.get("after_total_component_fit_count", -1)))
        inventory = row.get("after_actual_component_inventory")
        if not isinstance(inventory, Mapping):
            raise D92E0OCFAnalysisError("actual component inventory missing")
        actual_counts.add(int(inventory.get("actual_component_fit_count", -1)))
        try:
            query_macs.add(int(_first_value(row, "query_macs", "d92_e0d_query_macs")))
        except (TypeError, ValueError) as error:
            raise D92E0OCFAnalysisError("query MAC receipt drift") from error
        state = _first_value(row, "state_bytes", "after_state_bytes", "d92_e0d_head_state_bytes")
        try:
            state_bytes.add(int(state))
        except (TypeError, ValueError) as error:
            raise D92E0OCFAnalysisError("state bytes receipt drift") from error
        resource = row.get("after_registration_resource")
        if not isinstance(resource, Mapping):
            raise D92E0OCFAnalysisError("registered-state resource receipt missing")
        wall_values.append(_finite(resource.get("registration_wall_time_ns"), label="registration wall", lower=0.0))
        peak_values.append(_finite(resource.get("registration_incremental_peak_working_set_bytes"), label="incremental peak working set", lower=0.0))
        ocf_active = row.get("d92_e0d_ocf_active", row.get("ocf_active"))
        ocf_lambda = row.get("d92_e0d_ocf_lambda", row.get("ocf_lambda"))
        if arm in {"E0_OCF25", "E0_OCF50"} and int(job["k_shot"]) > 2:
            expected_lambda = 0.25 if arm == "E0_OCF25" else 0.50
            try:
                lambda_value = float(ocf_lambda)
            except (TypeError, ValueError) as error:
                raise D92E0OCFAnalysisError("OCF active/lambda receipt drift") from error
            if ocf_active is not True or lambda_value != expected_lambda:
                raise D92E0OCFAnalysisError("OCF active/lambda receipt drift")
        else:
            if ocf_active is not False or ocf_lambda is not None:
                raise D92E0OCFAnalysisError("non-OCF arm must be inactive with no lambda")
        support_macs_value = _first_value(row, "ocf_support_alignment_macs", "d92_e0d_ocf_support_alignment_macs_upper_bound")
        support_bytes_value = _first_value(row, "ocf_support_alignment_transient_bytes", "d92_e0d_ocf_support_alignment_transient_bytes_upper_bound")
        if arm in {"E0_OCF25", "E0_OCF50"} and int(job["k_shot"]) > 2:
            if support_macs_value is None or support_bytes_value is None:
                raise D92E0OCFAnalysisError("OCF support-side cost fields missing")
            support_macs.append(_finite(support_macs_value, label="OCF support alignment MACs", lower=0.0))
            support_bytes.append(_finite(support_bytes_value, label="OCF support alignment transient bytes", lower=0.0))
        else:
            for value, label in (
                (support_macs_value, "OCF support alignment MACs"),
                (support_bytes_value, "OCF support alignment transient bytes"),
            ):
                if value is None:
                    continue
                numeric = _finite(value, label=label, lower=0.0)
                if abs(numeric) > _TOLERANCE:
                    raise D92E0OCFAnalysisError("non-OCF support-side cost must be zero")
            support_macs.append(0.0)
            support_bytes.append(0.0)
    if fit_counts != {expected_total} or actual_counts != {expected_actual} or len(query_macs) != 1 or len(state_bytes) != 1:
        raise D92E0OCFAnalysisError("fit inventory/count/state closure drift")
    rates = _confusion_rates(score)
    return {
        "job_id": str(job["job_id"]), "outer_key": str(job["outer_key"]), "outer_role": str(job["outer_role"]), "k_shot": int(job["k_shot"]), "arm_id": arm, "candidate": str(job["candidate"]),
        "h_old_new": _scenario_mean(score, "after", "h_old_new"), "old_balanced_accuracy": _old_balanced_accuracy(score), "seen_new_accuracy": _scenario_mean(score, "after", "seen_new_acc"), "old_floor": _finite(score.get("per_old_class_floor_after"), label="old floor", lower=0.0, upper=1.0), "forgetting": _finite(score.get("old_forgetting_pp"), label="forgetting") / 100.0,
        **rates, "registration_wall_time_ns": float(statistics.median(wall_values)), "registration_incremental_peak_working_set_bytes": float(statistics.median(peak_values)), "fit_count": expected_total, "actual_fit_count": expected_actual, "query_macs": next(iter(query_macs)), "state_bytes": next(iter(state_bytes)), "ocf_support_alignment_macs": float(statistics.median(support_macs)), "ocf_support_alignment_transient_bytes": float(statistics.median(support_bytes)),
        "_before_prediction": _prediction_key(before_path), "_after_prediction": _prediction_key(after_path), "_before_state": tuple((str(row["scenario"]), str(row["before_state_fingerprint_sha256"])) for row in fit_rows), "_after_state": tuple((str(row["scenario"]), str(row["after_state_fingerprint_sha256"])) for row in fit_rows),
    }


def _gate(passed: bool, observed: Any, threshold: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "observed": observed, "threshold": threshold}


def _paired_summary(by_outer: Mapping[str, Mapping[str, Mapping[str, Any]]], performance_outers: Sequence[str], *, candidate_arm: str, reference_arm: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for outer_key in sorted(performance_outers):
        candidate, reference = by_outer[outer_key][candidate_arm], by_outer[outer_key][reference_arm]
        denominator = float(reference["registration_wall_time_ns"])
        if denominator <= 0:
            raise D92E0OCFAnalysisError("non-positive wall-time denominator")
        row: dict[str, Any] = {"outer_key": outer_key, "k_shot": int(candidate["k_shot"])}
        for metric in ("h_old_new", "old_balanced_accuracy", "seen_new_accuracy", "old_floor", "forgetting", "old_to_old_rate", "old_to_new_rate", "new_to_old_rate"):
            row[f"{candidate_arm}_{metric}"] = candidate[metric]
            row[f"{reference_arm}_{metric}"] = reference[metric]
        row.update({"old_to_old_rate": candidate["old_to_old_rate"], "old_to_new_rate": candidate["old_to_new_rate"], "new_to_old_rate": candidate["new_to_old_rate"]})
        row.update({f"delta_{candidate_arm}_vs_{reference_arm}_{metric}": candidate[metric] - reference[metric] for metric in ("h_old_new", "old_balanced_accuracy", "seen_new_accuracy", "old_floor", "forgetting")})
        row[f"wall_reduction_{candidate_arm}_vs_{reference_arm}"] = 1.0 - float(candidate["registration_wall_time_ns"]) / denominator
        row[f"incremental_peak_increase_{candidate_arm}_vs_{reference_arm}_bytes"] = float(candidate["registration_incremental_peak_working_set_bytes"]) - float(reference["registration_incremental_peak_working_set_bytes"])
        row[f"query_macs_difference_{candidate_arm}_vs_{reference_arm}"] = int(candidate["query_macs"]) - int(reference["query_macs"])
        row[f"state_bytes_difference_{candidate_arm}_vs_{reference_arm}"] = int(candidate["state_bytes"]) - int(reference["state_bytes"])
        rows.append(row)
    prefix = f"{candidate_arm}_vs_{reference_arm}"
    return ({
        "mean_delta_h": _mean([row[f"delta_{candidate_arm}_vs_{reference_arm}_h_old_new"] for row in rows]),
        "nonnegative_delta_h_outer_count": sum(row[f"delta_{candidate_arm}_vs_{reference_arm}_h_old_new"] >= -_TOLERANCE for row in rows),
        "mean_delta_old_balanced": _mean([row[f"delta_{candidate_arm}_vs_{reference_arm}_old_balanced_accuracy"] for row in rows]),
        "mean_delta_seen_new": _mean([row[f"delta_{candidate_arm}_vs_{reference_arm}_seen_new_accuracy"] for row in rows]),
        "mean_delta_old_floor": _mean([row[f"delta_{candidate_arm}_vs_{reference_arm}_old_floor"] for row in rows]),
        "nonnegative_delta_old_floor_outer_count": sum(row[f"delta_{candidate_arm}_vs_{reference_arm}_old_floor"] >= -_TOLERANCE for row in rows),
        "mean_delta_forgetting": _mean([row[f"delta_{candidate_arm}_vs_{reference_arm}_forgetting"] for row in rows]),
        "median_wall_reduction": float(statistics.median(row[f"wall_reduction_{candidate_arm}_vs_{reference_arm}"] for row in rows)),
        "median_incremental_peak_increase_bytes": float(statistics.median(row[f"incremental_peak_increase_{candidate_arm}_vs_{reference_arm}_bytes"] for row in rows)),
        "max_abs_query_macs_difference": max(abs(row[f"query_macs_difference_{candidate_arm}_vs_{reference_arm}"]) for row in rows),
        "max_abs_state_bytes_difference": max(abs(row[f"state_bytes_difference_{candidate_arm}_vs_{reference_arm}"]) for row in rows),
    }, rows)


def analyze_d92_e0ocf_hard12v3(matrix_manifest_path: str | Path, *, run_root: str | Path | None = None, method_lock_path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(matrix_manifest_path).resolve(strict=True)
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != "cvs.phase2.d92_e0ocf_hard12v3.matrix.v1" or manifest.get("status") != "FROZEN_DEVELOPMENT_MATRIX" or manifest.get("claim_scope") != "DEVELOPMENT_ONLY_PSEUDO_BLIND_DISJOINT_STRESS_SCREEN" or manifest.get("protocol_schema") != "p2_min_v1" or manifest.get("selection_sha256") != SELECTION_SHA256 or int(manifest.get("outer_count", -1)) != 12 or int(manifest.get("performance_outer_count", -1)) != 10 or int(manifest.get("liveness_outer_count", -1)) != 2 or int(manifest.get("job_count", -1)) != 60 or int(manifest.get("scene_arm_count", -1)) != 180 or int(manifest.get("shard_count", -1)) != 8 or manifest.get("arms") != list(ARMS) or manifest.get("primary_arm") != PRIMARY_ARM:
        raise D92E0OCFAnalysisError("matrix identity/count drift")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 60:
        raise D92E0OCFAnalysisError("matrix jobs are incomplete")
    lock_path = Path(method_lock_path) if method_lock_path is not None else Path(str(manifest["method_lock"]))
    if _sha256(lock_path) != manifest.get("method_lock_sha256"):
        raise D92E0OCFAnalysisError("method lock identity drift")
    method_lock = _read_json(lock_path)
    if method_lock.get("schema") != "cvs.phase2.d92_e0ocf.method_lock.v1" or method_lock.get("only_promotion_candidate") != PRIMARY_ARM:
        raise D92E0OCFAnalysisError("promotion candidate drift")
    thresholds = method_lock.get("strict_geometry_gate")
    if not isinstance(thresholds, Mapping):
        raise D92E0OCFAnalysisError("strict geometry gate missing")
    output_root = Path(run_root) if run_root is not None else Path(str(manifest["output_root"]))
    expected_by_shard: dict[int, list[str]] = {shard: [] for shard in range(8)}
    seen_job_ids: set[str] = set()
    for job in jobs:
        if not isinstance(job, Mapping):
            raise D92E0OCFAnalysisError("matrix job identity drift")
        job_id = str(job.get("job_id", ""))
        try:
            shard_index = int(job.get("planned_shard_index", -1))
        except (TypeError, ValueError) as error:
            raise D92E0OCFAnalysisError("matrix planned shard identity drift") from error
        if not job_id or job_id in seen_job_ids or shard_index not in expected_by_shard:
            raise D92E0OCFAnalysisError("matrix planned shard identity drift")
        seen_job_ids.add(job_id)
        expected_by_shard[shard_index].append(job_id)
    selected_total = 0
    completed_total = 0
    for shard in range(8):
        summary = _read_json(output_root / "summaries" / f"shard_{shard}.json")
        expected_ids = expected_by_shard[shard]
        completed_ids = summary.get("completed_job_ids")
        try:
            selected_count = int(summary.get("selected_job_count", -1))
            completed_count = int(summary.get("completed_job_count", -1))
            failed_count = int(summary.get("failed_job_count", -1))
            summary_shard = int(summary.get("shard_index", -1))
        except (TypeError, ValueError) as error:
            raise D92E0OCFAnalysisError("shard summary schema drift") from error
        if (
            summary.get("schema") != "cvs.phase2.d92_e0ocf_hard12v3.shard_summary.v1"
            or summary.get("status") != "PASS"
            or summary_shard != shard
            or failed_count != 0
            or summary.get("performance_result_allowed") is not True
            or selected_count != len(expected_ids)
            or completed_count != len(expected_ids)
            or not isinstance(completed_ids, list)
            or any(not isinstance(job_id, str) for job_id in completed_ids)
            or len(completed_ids) != len(set(completed_ids))
            or set(completed_ids) != set(expected_ids)
        ):
            raise D92E0OCFAnalysisError("shard completed job IDs/summary closure drift")
        selected_total += selected_count
        completed_total += completed_count
    if selected_total != 60 or completed_total != 60:
        raise D92E0OCFAnalysisError("shard job receipt closure drift")
    evidence = [_job_evidence(job, run_root=output_root if run_root is not None else None) for job in jobs]
    if len(evidence) != 60:
        raise D92E0OCFAnalysisError("job receipt closure drift")
    by_outer: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in evidence:
        if row["arm_id"] in by_outer[row["outer_key"]]:
            raise D92E0OCFAnalysisError("duplicate outer/arm evidence")
        by_outer[row["outer_key"]][row["arm_id"]] = row
    if len(by_outer) != 12 or any(set(rows) != set(ARMS) for rows in by_outer.values()):
        raise D92E0OCFAnalysisError("outer/arm closure drift")
    performance_outers: list[str] = []
    for outer_key, rows in sorted(by_outer.items()):
        reference = rows["D92_FULL"]
        if any(rows[arm]["_before_prediction"] != reference["_before_prediction"] for arm in ARMS[1:]):
            raise D92E0OCFAnalysisError(f"DA0_REG0 prediction drift: {outer_key}")
        if any(rows[arm]["_before_state"] != reference["_before_state"] for arm in ARMS[1:]):
            raise D92E0OCFAnalysisError(f"DA0_REG0 state drift: {outer_key}")
        roles, ks = {row["outer_role"] for row in rows.values()}, {row["k_shot"] for row in rows.values()}
        if len(roles) != 1 or len(ks) != 1:
            raise D92E0OCFAnalysisError("same-outer role/K drift")
        role, k_shot = next(iter(roles)), next(iter(ks))
        if role == "performance":
            performance_outers.append(outer_key)
        elif role == "liveness":
            if k_shot != 1:
                raise D92E0OCFAnalysisError("liveness outer is not K1")
            if any(rows[arm]["_after_prediction"] != reference["_after_prediction"] for arm in ARMS[1:]):
                raise D92E0OCFAnalysisError(f"K1 prediction alias drift: {outer_key}")
            if any(rows[arm]["_after_state"] != reference["_after_state"] for arm in ARMS[1:]):
                raise D92E0OCFAnalysisError(f"K1 state alias drift: {outer_key}")
        else:
            raise D92E0OCFAnalysisError("unknown outer role")
    if len(performance_outers) != 10:
        raise D92E0OCFAnalysisError("performance outer closure drift")
    public = [{key: value for key, value in row.items() if not key.startswith("_")} for row in evidence]
    aggregate: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        rows = [row for row in public if row["outer_role"] == "performance" and row["arm_id"] == arm]
        aggregate[arm] = {"mean_h_old_new": _mean([row["h_old_new"] for row in rows]), "mean_old_balanced_accuracy": _mean([row["old_balanced_accuracy"] for row in rows]), "mean_seen_new_accuracy": _mean([row["seen_new_accuracy"] for row in rows]), "mean_old_floor": _mean([row["old_floor"] for row in rows]), "mean_forgetting": _mean([row["forgetting"] for row in rows]), "mean_old_to_old_rate": _mean([row["old_to_old_rate"] for row in rows]), "mean_old_to_new_rate": _mean([row["old_to_new_rate"] for row in rows]), "mean_new_to_old_rate": _mean([row["new_to_old_rate"] for row in rows]), "median_registration_wall_time_ns": float(statistics.median(row["registration_wall_time_ns"] for row in rows)), "median_registration_incremental_peak_working_set_bytes": float(statistics.median(row["registration_incremental_peak_working_set_bytes"] for row in rows)), "mean_query_macs": _mean([float(row["query_macs"]) for row in rows]), "mean_state_bytes": _mean([float(row["state_bytes"]) for row in rows]), "median_ocf_support_alignment_macs": float(statistics.median(row["ocf_support_alignment_macs"] for row in rows)), "median_ocf_support_alignment_transient_bytes": float(statistics.median(row["ocf_support_alignment_transient_bytes"] for row in rows))}
    paired_full_only, rows_full_only = _paired_summary(by_outer, performance_outers, candidate_arm=PRIMARY_ARM, reference_arm="E0_FULL_ONLY")
    paired_d92, rows_d92 = _paired_summary(by_outer, performance_outers, candidate_arm=PRIMARY_ARM, reference_arm="D92_FULL")
    full_only_by_outer = {row["outer_key"]: row for row in evidence if row["arm_id"] == "E0_FULL_ONLY" and row["outer_role"] == "performance"}
    query_state_exact = paired_full_only["max_abs_query_macs_difference"] == 0 and paired_full_only["max_abs_state_bytes_difference"] == 0
    for diagnostic_arm in ("E0_OCF25", "E0_OCF50"):
        for row in evidence:
            if row["arm_id"] == diagnostic_arm and row["outer_role"] == "performance":
                reference = full_only_by_outer.get(row["outer_key"])
                if reference is None or row["query_macs"] != reference["query_macs"] or row["state_bytes"] != reference["state_bytes"]:
                    query_state_exact = False
    ocf_fit_exact = all(row["fit_count"] == 4 and row["actual_fit_count"] == 2 for row in evidence if row["arm_id"] in {"E0_OCF25", "E0_OCF50"} and row["k_shot"] > 2)
    gates = {
        "mean_delta_old_floor_vs_full_only": _gate(paired_full_only["mean_delta_old_floor"] > float(thresholds.get("mean_delta_old_floor_vs_full_only_min_exclusive", 0.0)) + _TOLERANCE, paired_full_only["mean_delta_old_floor"], {">": float(thresholds.get("mean_delta_old_floor_vs_full_only_min_exclusive", 0.0))}),
        "old_floor_nonnegative_vs_full_only_outer_count": _gate(paired_full_only["nonnegative_delta_old_floor_outer_count"] >= int(thresholds.get("old_floor_nonnegative_vs_full_only_min", 8)), paired_full_only["nonnegative_delta_old_floor_outer_count"], {">=": int(thresholds.get("old_floor_nonnegative_vs_full_only_min", 8))}),
        "mean_delta_h_vs_full_only": _gate(paired_full_only["mean_delta_h"] >= float(thresholds.get("mean_delta_h_vs_full_only_min", 0.0)) - _TOLERANCE, paired_full_only["mean_delta_h"], {">=": float(thresholds.get("mean_delta_h_vs_full_only_min", 0.0))}),
        "mean_delta_old_balanced_vs_full_only": _gate(paired_full_only["mean_delta_old_balanced"] >= float(thresholds.get("mean_delta_old_balanced_vs_full_only_min", 0.0)) - _TOLERANCE, paired_full_only["mean_delta_old_balanced"], {">=": float(thresholds.get("mean_delta_old_balanced_vs_full_only_min", 0.0))}),
        "mean_delta_seen_new_vs_full_only": _gate(paired_full_only["mean_delta_seen_new"] >= float(thresholds.get("mean_delta_seen_new_vs_full_only_min", 0.0)) - _TOLERANCE, paired_full_only["mean_delta_seen_new"], {">=": float(thresholds.get("mean_delta_seen_new_vs_full_only_min", 0.0))}),
        "mean_delta_forgetting_vs_full_only": _gate(paired_full_only["mean_delta_forgetting"] <= float(thresholds.get("mean_delta_forgetting_vs_full_only_max", 0.0)) + _TOLERANCE, paired_full_only["mean_delta_forgetting"], {"<=": float(thresholds.get("mean_delta_forgetting_vs_full_only_max", 0.0))}),
        "mean_delta_h_vs_d92_full": _gate(paired_d92["mean_delta_h"] >= float(thresholds.get("mean_delta_h_vs_d92_full_min", 0.005)) - _TOLERANCE, paired_d92["mean_delta_h"], {">=": float(thresholds.get("mean_delta_h_vs_d92_full_min", 0.005))}),
        "h_nonnegative_vs_d92_full_outer_count": _gate(paired_d92["nonnegative_delta_h_outer_count"] >= int(thresholds.get("h_nonnegative_vs_d92_full_min", 8)), paired_d92["nonnegative_delta_h_outer_count"], {">=": int(thresholds.get("h_nonnegative_vs_d92_full_min", 8))}),
        "mean_delta_old_balanced_vs_d92_full": _gate(paired_d92["mean_delta_old_balanced"] >= float(thresholds.get("mean_delta_old_balanced_vs_d92_full_min", 0.0)) - _TOLERANCE, paired_d92["mean_delta_old_balanced"], {">=": float(thresholds.get("mean_delta_old_balanced_vs_d92_full_min", 0.0))}),
        "mean_delta_old_floor_vs_d92_full": _gate(paired_d92["mean_delta_old_floor"] >= float(thresholds.get("mean_delta_old_floor_vs_d92_full_min", 0.0)) - _TOLERANCE, paired_d92["mean_delta_old_floor"], {">=": float(thresholds.get("mean_delta_old_floor_vs_d92_full_min", 0.0))}),
        "mean_delta_seen_new_vs_d92_full": _gate(paired_d92["mean_delta_seen_new"] >= float(thresholds.get("mean_delta_seen_new_vs_d92_full_min", 0.0)) - _TOLERANCE, paired_d92["mean_delta_seen_new"], {">=": float(thresholds.get("mean_delta_seen_new_vs_d92_full_min", 0.0))}),
        "mean_delta_forgetting_vs_d92_full": _gate(paired_d92["mean_delta_forgetting"] <= float(thresholds.get("mean_delta_forgetting_vs_d92_full_max", 0.0)) + _TOLERANCE, paired_d92["mean_delta_forgetting"], {"<=": float(thresholds.get("mean_delta_forgetting_vs_d92_full_max", 0.0))}),
        "median_wall_reduction_vs_d92_full": _gate(paired_d92["median_wall_reduction"] >= float(thresholds.get("median_wall_reduction_vs_d92_full_min", 0.6)) - _TOLERANCE, paired_d92["median_wall_reduction"], {">=": float(thresholds.get("median_wall_reduction_vs_d92_full_min", 0.6))}),
        "median_incremental_peak_vs_d92_full": _gate(paired_d92["median_incremental_peak_increase_bytes"] <= float(thresholds.get("median_incremental_peak_vs_d92_full_max", 0.0)) + _TOLERANCE, paired_d92["median_incremental_peak_increase_bytes"], {"<=": float(thresholds.get("median_incremental_peak_vs_d92_full_max", 0.0))}),
        "ocf_fit_count_exact": _gate(ocf_fit_exact, ocf_fit_exact, "K5/K10 two-state=4/4,after actual=2/2"),
        "query_macs_and_state_bytes_exact": _gate(query_state_exact, query_state_exact, True),
        "query_protocol_zero_access": _gate(True, True, True),
        "complete_artifact_closure": _gate(len(evidence) == 60, len(evidence), 60),
    }
    all_gates_pass = all(bool(row["passed"]) for row in gates.values())
    paired_rows: list[dict[str, Any]] = []
    d92_by_outer = {row["outer_key"]: row for row in rows_d92}
    for row in rows_full_only:
        paired_rows.append({**row, **{f"d92_{key}": value for key, value in d92_by_outer[row["outer_key"]].items() if key not in {"outer_key", "k_shot"}}})
    return {"schema": "cvs.phase2.d92_e0ocf_hard12v3.analysis.v1", "status": "ANALYZED", "claim_scope": manifest.get("claim_scope"), "matrix_manifest": str(manifest_path), "matrix_manifest_sha256": _sha256(manifest_path), "method_lock_sha256": manifest.get("method_lock_sha256"), "selection_sha256": manifest.get("selection_sha256"), "performance_outer_count": 10, "liveness_outer_count": 2, "job_count": 60, "promotion_candidate": PRIMARY_ARM, "diagnostic_only_arm": "E0_OCF50", "aggregate": aggregate, "paired": {"E0_OCF25_minus_E0_FULL_ONLY": paired_full_only, "E0_OCF25_minus_D92_FULL": paired_d92}, "paired_rows": paired_rows, "gates": gates, "all_gates_pass": all_gates_pass, "verdict": "PROMOTE_E0_OCF25_TO_TARGET125_CONFIRMATION" if all_gates_pass else "NO_E0_OCF25_PROMOTION"}


analyze_d92_e0ocf_hard12 = analyze_d92_e0ocf_hard12v3


__all__ = ["ARMS", "D92E0OCFAnalysisError", "D92E0OCFHard12AnalysisError", "D92E0OCFHard12V3AnalysisError", "analyze_d92_e0ocf_hard12", "analyze_d92_e0ocf_hard12v3"]
