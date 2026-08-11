"""Strict paired analysis for the development-only D92-E0D Hard12-v2 matrix."""

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


ARMS = (
    "D92_FULL",
    "E0_FUSION",
    "E0_FULL_ONLY",
    "E0_BLOCK_ONLY",
    "E0_FIXED50",
)
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
SELECTION_SHA256 = "2e3b3333a4a325bd0443a31065d3340d6a650a3e89620951a786637e6bce8d3a"
_TOLERANCE = 1e-12
_QUERY_ZERO_FIELDS = (
    "query_truth_access",
    "query_fit_access",
    "query_update_access",
    "query_selection_access",
    "query_role_oracle_access",
    "query_class_quota_access",
    "query_global_reassignment",
)


class D92E0DAnalysisError(ValueError):
    """Raised when frozen D92-E0D evidence is incomplete or inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise D92E0DAnalysisError(f"required JSON artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise D92E0DAnalysisError(f"JSON artifact must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise D92E0DAnalysisError(f"non-finite {label}")
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise D92E0DAnalysisError("cannot aggregate an empty metric")
    return float(statistics.fmean(values))


def _prediction_key(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    try:
        with np.load(path, allow_pickle=False) as artifact:
            required = {"query_tokens", "scenarios", "predicted_class_handles"}
            if set(artifact.files) != required:
                raise D92E0DAnalysisError("prediction artifact member drift")
            tokens = tuple(str(value) for value in artifact["query_tokens"].tolist())
            scenarios = tuple(str(value) for value in artifact["scenarios"].tolist())
            predictions = tuple(
                str(value) for value in artifact["predicted_class_handles"].tolist()
            )
    except (OSError, ValueError) as error:
        raise D92E0DAnalysisError(f"invalid prediction artifact: {path}") from error
    if not tokens or not (len(tokens) == len(scenarios) == len(predictions)):
        raise D92E0DAnalysisError("prediction artifact row-count drift")
    return tokens, scenarios, predictions


def _scenario_mean(score: Mapping[str, Any], state: str, field: str) -> float:
    state_payload = score.get(state)
    if not isinstance(state_payload, Mapping):
        raise D92E0DAnalysisError(f"score state missing: {state}")
    by_scenario = state_payload.get("by_scenario")
    if not isinstance(by_scenario, Mapping) or set(by_scenario) != set(SCENES):
        raise D92E0DAnalysisError(f"score scenario coverage drift: {state}")
    return _mean(
        [
            _finite(by_scenario[scene][field], label=f"{state}.{scene}.{field}")
            for scene in SCENES
        ]
    )


def _old_balanced_accuracy(score: Mapping[str, Any]) -> float:
    after = score.get("after")
    by_tx = after.get("by_tx") if isinstance(after, Mapping) else None
    if not isinstance(by_tx, Mapping):
        raise D92E0DAnalysisError("after.by_tx is required for old balanced accuracy")
    values = [
        _finite(row["accuracy"], label=f"after.by_tx.{tx}.accuracy")
        for tx, row in by_tx.items()
        if isinstance(row, Mapping) and row.get("role") == "target_old"
    ]
    return _mean(values)


def _load_fit_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise D92E0DAnalysisError(f"fit audit is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or len(payload) != len(SCENES):
        raise D92E0DAnalysisError("fit audit scenario count drift")
    rows = [dict(row) for row in payload if isinstance(row, Mapping)]
    if len(rows) != len(SCENES) or {str(row.get("scenario")) for row in rows} != set(SCENES):
        raise D92E0DAnalysisError("fit audit scenario identity drift")
    for row in rows:
        if any(row.get(field) is not False for field in _QUERY_ZERO_FIELDS):
            raise D92E0DAnalysisError("query protocol audit drift")
        for field in (
            "before_state_fingerprint_sha256",
            "after_state_fingerprint_sha256",
        ):
            value = str(row.get(field, ""))
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise D92E0DAnalysisError("state fingerprint audit drift")
    return sorted(rows, key=lambda row: str(row["scenario"]))


def _job_evidence(
    job: Mapping[str, Any], *, run_root: Path | None = None
) -> dict[str, Any]:
    arm = str(job.get("arm_id"))
    if arm not in D92_E0D_ARMS or job.get("candidate") != D92_E0D_ARMS[arm].candidate_id:
        raise D92E0DAnalysisError("job arm/candidate identity drift")
    job_root = (
        run_root / "jobs" / str(job["outer_key"]) / arm
        if run_root is not None
        else Path(str(job["output_root"]))
    )
    receipt_path = job_root / "job_receipt.json"
    score_path = job_root / "scorer" / "diag_cosine_score.json"
    before_path = job_root / "diag" / "before" / "prediction_artifact.npz"
    after_path = job_root / "diag" / "after" / "prediction_artifact.npz"
    receipt = _read_json(receipt_path)
    score = _read_json(score_path)
    if (
        receipt.get("schema") != "cvs.phase2.d92_e0d_hard12v2.job_receipt.v1"
        or receipt.get("status") != "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE"
        or receipt.get("job_id") != job.get("job_id")
        or receipt.get("outer_key") != job.get("outer_key")
        or receipt.get("arm_id") != arm
        or receipt.get("candidate") != job.get("candidate")
        or receipt.get("truth_sidecar_exposed_to_predictor") is not False
        or receipt.get("query_truth_joined_only_after_immutable_predictions") is not True
        or receipt.get("query_truth_fed_back_to_predictor") is not False
    ):
        raise D92E0DAnalysisError("job receipt identity/protocol drift")
    actual_before_sha = _sha256(before_path)
    actual_after_sha = _sha256(after_path)
    actual_score_sha = _sha256(score_path)
    if (
        receipt.get("before_prediction_sha256") != actual_before_sha
        or receipt.get("after_prediction_sha256") != actual_after_sha
        or receipt.get("score_sha256") != actual_score_sha
        or score.get("before_prediction_sha256") != actual_before_sha
        or score.get("after_prediction_sha256") != actual_after_sha
        or score.get("candidate") != job.get("candidate")
        or score.get("query_truth_joined_only_after_immutable_predictions") is not True
        or score.get("query_truth_fed_back_to_predictor") is not False
    ):
        raise D92E0DAnalysisError("prediction-to-score binding drift")
    fit_rows = _load_fit_rows(job_root / "diag" / "after" / "fit_audit.json")
    fit_counts = {int(row.get("after_total_component_fit_count", -1)) for row in fit_rows}
    actual_fit_counts: set[int] = set()
    for row in fit_rows:
        inventory = row.get("after_actual_component_inventory")
        if not isinstance(inventory, Mapping):
            raise D92E0DAnalysisError("actual component inventory missing")
        actual_fit_counts.add(int(inventory.get("actual_component_fit_count", -1)))
    query_macs = {int(row.get("query_macs", -1)) for row in fit_rows}
    if len(actual_fit_counts) != 1 or min(actual_fit_counts) < 0:
        raise D92E0DAnalysisError("actual component inventory drift")
    if len(fit_counts) != 1 or len(query_macs) != 1 or min(query_macs) < 0:
        raise D92E0DAnalysisError("fit-count or query-MAC scene drift")
    wall_values: list[float] = []
    peak_values: list[float] = []
    for row in fit_rows:
        resource = row.get("after_registration_resource")
        if not isinstance(resource, Mapping):
            raise D92E0DAnalysisError("registered-state resource receipt missing")
        wall_values.append(
            _finite(resource.get("registration_wall_time_ns"), label="registration wall")
        )
        peak_values.append(
            _finite(
                resource.get("registration_incremental_peak_working_set_bytes"),
                label="incremental peak working set",
            )
        )
    if min(wall_values) < 0 or min(peak_values) < 0:
        raise D92E0DAnalysisError("negative resource receipt")
    return {
        "job_id": str(job["job_id"]),
        "outer_key": str(job["outer_key"]),
        "outer_role": str(job["outer_role"]),
        "k_shot": int(job["k_shot"]),
        "arm_id": arm,
        "candidate": str(job["candidate"]),
        "h_old_new": _scenario_mean(score, "after", "h_old_new"),
        "old_balanced_accuracy": _old_balanced_accuracy(score),
        "seen_new_accuracy": _scenario_mean(score, "after", "seen_new_acc"),
        "old_floor": _finite(score.get("per_old_class_floor_after"), label="old floor"),
        "forgetting": _finite(score.get("old_forgetting_pp"), label="forgetting") / 100.0,
        "registration_wall_time_ns": float(statistics.median(wall_values)),
        "registration_incremental_peak_working_set_bytes": float(
            statistics.median(peak_values)
        ),
        "fit_count": next(iter(fit_counts)),
        "actual_fit_count": next(iter(actual_fit_counts)),
        "query_macs": next(iter(query_macs)),
        "_before_prediction": _prediction_key(before_path),
        "_after_prediction": _prediction_key(after_path),
        "_before_state": tuple(
            (str(row["scenario"]), str(row["before_state_fingerprint_sha256"]))
            for row in fit_rows
        ),
        "_after_state": tuple(
            (str(row["scenario"]), str(row["after_state_fingerprint_sha256"]))
            for row in fit_rows
        ),
    }


def _gate(passed: bool, observed: Any, threshold: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "observed": observed, "threshold": threshold}


def _expected_fit_count(k_shot: int, arm: str) -> int:
    if int(k_shot) <= 2 or arm == "D92_FULL":
        return 8 * (int(k_shot) + 1)
    if arm == "E0_FUSION":
        return 4 * (int(k_shot) + 1)
    if arm == "E0_FIXED50":
        return 4
    return 2


def _expected_actual_fit_count(k_shot: int, arm: str) -> int:
    total = _expected_fit_count(k_shot, arm)
    return total if int(k_shot) <= 2 else total // 2


def _paired_summary(
    by_outer: Mapping[str, Mapping[str, Mapping[str, Any]]],
    performance_outers: Sequence[str],
    *,
    reference_arm: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for outer_key in sorted(performance_outers):
        candidate = by_outer[outer_key]["E0_FULL_ONLY"]
        reference = by_outer[outer_key][reference_arm]
        wall_base = float(reference["registration_wall_time_ns"])
        if wall_base <= 0:
            raise D92E0DAnalysisError("non-positive wall-time denominator")
        rows.append(
            {
                "outer_key": outer_key,
                "k_shot": int(candidate["k_shot"]),
                "delta_h": candidate["h_old_new"] - reference["h_old_new"],
                "delta_old_balanced": candidate["old_balanced_accuracy"]
                - reference["old_balanced_accuracy"],
                "delta_seen_new": candidate["seen_new_accuracy"]
                - reference["seen_new_accuracy"],
                "delta_old_floor": candidate["old_floor"] - reference["old_floor"],
                "delta_forgetting": candidate["forgetting"] - reference["forgetting"],
                "wall_reduction": 1.0
                - float(candidate["registration_wall_time_ns"]) / wall_base,
                "incremental_peak_increase_bytes": float(
                    candidate["registration_incremental_peak_working_set_bytes"]
                )
                - float(reference["registration_incremental_peak_working_set_bytes"]),
                "query_macs_difference": int(candidate["query_macs"])
                - int(reference["query_macs"]),
            }
        )
    return (
        {
            "mean_delta_h": _mean([row["delta_h"] for row in rows]),
            "nonnegative_delta_h_outer_count": sum(
                row["delta_h"] >= -_TOLERANCE for row in rows
            ),
            "mean_delta_old_balanced": _mean(
                [row["delta_old_balanced"] for row in rows]
            ),
            "mean_delta_seen_new": _mean([row["delta_seen_new"] for row in rows]),
            "mean_delta_old_floor": _mean([row["delta_old_floor"] for row in rows]),
            "mean_delta_forgetting": _mean([row["delta_forgetting"] for row in rows]),
            "median_wall_reduction": float(
                statistics.median(row["wall_reduction"] for row in rows)
            ),
            "median_incremental_peak_increase_bytes": float(
                statistics.median(
                    row["incremental_peak_increase_bytes"] for row in rows
                )
            ),
            "max_abs_query_macs_difference": max(
                abs(row["query_macs_difference"]) for row in rows
            ),
        },
        rows,
    )


def analyze_d92_e0d_hard12v2(
    matrix_manifest_path: str | Path,
    *,
    run_root: str | Path | None = None,
    method_lock_path: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze one complete 60-job matrix without post-hoc arm selection."""

    manifest_path = Path(matrix_manifest_path).resolve(strict=True)
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema") != "cvs.phase2.d92_e0d_hard12v2.matrix.v1"
        or manifest.get("status") != "FROZEN_DEVELOPMENT_MATRIX"
        or manifest.get("claim_scope")
        != "DEVELOPMENT_ONLY_PSEUDO_BLIND_DISJOINT_STRESS_SCREEN"
        or manifest.get("protocol_schema") != "p2_min_v1"
        or manifest.get("selection_sha256") != SELECTION_SHA256
        or int(manifest.get("outer_count", -1)) != 12
        or int(manifest.get("performance_outer_count", -1)) != 10
        or int(manifest.get("liveness_outer_count", -1)) != 2
        or int(manifest.get("job_count", -1)) != 60
        or int(manifest.get("shard_count", -1)) != 8
    ):
        raise D92E0DAnalysisError("matrix identity/count drift")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 60:
        raise D92E0DAnalysisError("matrix jobs are incomplete")
    lock_path = (
        Path(method_lock_path)
        if method_lock_path is not None
        else Path(str(manifest["method_lock"]))
    )
    if _sha256(lock_path) != manifest.get("method_lock_sha256"):
        raise D92E0DAnalysisError("method lock identity drift")
    method_lock = _read_json(lock_path)
    if (
        method_lock.get("schema") != "cvs.phase2.d92_e0d.method_lock.v1"
        or method_lock.get("only_promotion_candidate") != "E0_FULL_ONLY"
    ):
        raise D92E0DAnalysisError("promotion candidate drift")
    thresholds = method_lock.get("strict_geometry_gate")
    if not isinstance(thresholds, Mapping):
        raise D92E0DAnalysisError("strict geometry gate missing")
    output_root = (
        Path(run_root) if run_root is not None else Path(str(manifest["output_root"]))
    )
    for shard in range(8):
        shard_summary = _read_json(output_root / "summaries" / f"shard_{shard}.json")
        if (
            shard_summary.get("status") != "PASS"
            or int(shard_summary.get("shard_index", -1)) != shard
            or shard_summary.get("performance_result_allowed") is not True
        ):
            raise D92E0DAnalysisError("matrix has a non-PASS shard")

    evidence = [
        _job_evidence(job, run_root=output_root if run_root is not None else None)
        for job in jobs
    ]
    by_outer: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in evidence:
        if row["arm_id"] in by_outer[row["outer_key"]]:
            raise D92E0DAnalysisError("duplicate outer/arm evidence")
        by_outer[row["outer_key"]][row["arm_id"]] = row
    if len(by_outer) != 12 or any(set(rows) != set(ARMS) for rows in by_outer.values()):
        raise D92E0DAnalysisError("outer/arm closure drift")

    performance_outers: list[str] = []
    for outer_key, rows in sorted(by_outer.items()):
        reference = rows["D92_FULL"]
        if any(
            rows[arm]["_before_prediction"] != reference["_before_prediction"]
            for arm in ARMS[1:]
        ):
            raise D92E0DAnalysisError(f"DA0_REG0 prediction drift: {outer_key}")
        if any(
            rows[arm]["_before_state"] != reference["_before_state"]
            for arm in ARMS[1:]
        ):
            raise D92E0DAnalysisError(f"DA0_REG0 state drift: {outer_key}")
        role_values = {row["outer_role"] for row in rows.values()}
        k_values = {row["k_shot"] for row in rows.values()}
        if len(role_values) != 1 or len(k_values) != 1:
            raise D92E0DAnalysisError("same-outer role/K drift")
        role = next(iter(role_values))
        k_shot = next(iter(k_values))
        if role == "performance":
            performance_outers.append(outer_key)
        elif role == "liveness":
            if k_shot != 1:
                raise D92E0DAnalysisError("liveness outer is not K1")
            if any(
                rows[arm]["_after_prediction"] != reference["_after_prediction"]
                for arm in ARMS[1:]
            ):
                raise D92E0DAnalysisError(f"K1 prediction alias drift: {outer_key}")
            if any(
                rows[arm]["_after_state"] != reference["_after_state"]
                for arm in ARMS[1:]
            ):
                raise D92E0DAnalysisError(f"K1 state alias drift: {outer_key}")
        else:
            raise D92E0DAnalysisError("unknown outer role")
    if len(performance_outers) != 10:
        raise D92E0DAnalysisError("performance outer closure drift")

    public_evidence = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in evidence
    ]
    performance_rows = [row for row in public_evidence if row["outer_role"] == "performance"]
    aggregate: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        arm_rows = [row for row in performance_rows if row["arm_id"] == arm]
        if len(arm_rows) != 10:
            raise D92E0DAnalysisError("performance arm count drift")
        aggregate[arm] = {
            "mean_h_old_new": _mean([row["h_old_new"] for row in arm_rows]),
            "mean_old_balanced_accuracy": _mean(
                [row["old_balanced_accuracy"] for row in arm_rows]
            ),
            "mean_seen_new_accuracy": _mean(
                [row["seen_new_accuracy"] for row in arm_rows]
            ),
            "mean_old_floor": _mean([row["old_floor"] for row in arm_rows]),
            "mean_forgetting": _mean([row["forgetting"] for row in arm_rows]),
            "median_registration_wall_time_ns": float(
                statistics.median(row["registration_wall_time_ns"] for row in arm_rows)
            ),
            "median_registration_incremental_peak_working_set_bytes": float(
                statistics.median(
                    row["registration_incremental_peak_working_set_bytes"]
                    for row in arm_rows
                )
            ),
            "mean_query_macs": _mean([float(row["query_macs"]) for row in arm_rows]),
        }

    paired_fusion, rows_fusion = _paired_summary(
        by_outer, performance_outers, reference_arm="E0_FUSION"
    )
    paired_full, rows_full = _paired_summary(
        by_outer, performance_outers, reference_arm="D92_FULL"
    )
    fit_count_exact = all(
        row["fit_count"] == _expected_fit_count(row["k_shot"], row["arm_id"])
        and row["actual_fit_count"]
        == _expected_actual_fit_count(row["k_shot"], row["arm_id"])
        for row in evidence
    )
    query_exact = (
        paired_fusion["max_abs_query_macs_difference"]
        <= float(thresholds["query_cost_increase_max"]) + _TOLERANCE
        and paired_full["max_abs_query_macs_difference"]
        <= float(thresholds["query_cost_increase_max"]) + _TOLERANCE
    )
    gates = {
        "mean_delta_h_vs_e0_fusion": _gate(
            paired_fusion["mean_delta_h"]
            > float(thresholds["mean_delta_h_vs_e0_fusion_min_exclusive"])
            + _TOLERANCE,
            paired_fusion["mean_delta_h"],
            {">": float(thresholds["mean_delta_h_vs_e0_fusion_min_exclusive"])},
        ),
        "nonnegative_delta_h_vs_e0_fusion_outer_count": _gate(
            paired_fusion["nonnegative_delta_h_outer_count"]
            >= int(thresholds["nonnegative_delta_h_vs_e0_fusion_outer_min"]),
            paired_fusion["nonnegative_delta_h_outer_count"],
            {">=": int(thresholds["nonnegative_delta_h_vs_e0_fusion_outer_min"])},
        ),
        "mean_delta_h_vs_d92_full": _gate(
            paired_full["mean_delta_h"]
            >= float(thresholds["mean_delta_h_vs_d92_full_min"]) - _TOLERANCE,
            paired_full["mean_delta_h"],
            {">=": float(thresholds["mean_delta_h_vs_d92_full_min"])},
        ),
        "nonnegative_delta_h_vs_d92_full_outer_count": _gate(
            paired_full["nonnegative_delta_h_outer_count"]
            >= int(thresholds["nonnegative_delta_h_vs_d92_full_outer_min"]),
            paired_full["nonnegative_delta_h_outer_count"],
            {">=": int(thresholds["nonnegative_delta_h_vs_d92_full_outer_min"])},
        ),
        "mean_delta_old_balanced_vs_d92_full": _gate(
            paired_full["mean_delta_old_balanced"]
            >= float(thresholds["mean_delta_old_balanced_vs_d92_full_min"])
            - _TOLERANCE,
            paired_full["mean_delta_old_balanced"],
            {">=": float(thresholds["mean_delta_old_balanced_vs_d92_full_min"])},
        ),
        "mean_delta_seen_new_vs_d92_full": _gate(
            paired_full["mean_delta_seen_new"]
            >= float(thresholds["mean_delta_seen_new_vs_d92_full_min"]) - _TOLERANCE,
            paired_full["mean_delta_seen_new"],
            {">=": float(thresholds["mean_delta_seen_new_vs_d92_full_min"])},
        ),
        "mean_delta_old_floor_vs_d92_full": _gate(
            paired_full["mean_delta_old_floor"]
            >= float(thresholds["mean_delta_old_floor_vs_d92_full_min"])
            - _TOLERANCE,
            paired_full["mean_delta_old_floor"],
            {">=": float(thresholds["mean_delta_old_floor_vs_d92_full_min"])},
        ),
        "mean_delta_forgetting_vs_d92_full": _gate(
            paired_full["mean_delta_forgetting"]
            <= float(thresholds["mean_delta_forgetting_vs_d92_full_max"])
            + _TOLERANCE,
            paired_full["mean_delta_forgetting"],
            {"<=": float(thresholds["mean_delta_forgetting_vs_d92_full_max"])},
        ),
        "median_wall_reduction_vs_e0_fusion": _gate(
            paired_fusion["median_wall_reduction"]
            >= float(thresholds["median_wall_reduction_vs_e0_fusion_min"])
            - _TOLERANCE,
            paired_fusion["median_wall_reduction"],
            {">=": float(thresholds["median_wall_reduction_vs_e0_fusion_min"])},
        ),
        "median_wall_reduction_vs_d92_full": _gate(
            paired_full["median_wall_reduction"]
            >= float(thresholds["median_wall_reduction_vs_d92_full_min"])
            - _TOLERANCE,
            paired_full["median_wall_reduction"],
            {">=": float(thresholds["median_wall_reduction_vs_d92_full_min"])},
        ),
        "median_incremental_peak_vs_e0_fusion": _gate(
            paired_fusion["median_incremental_peak_increase_bytes"]
            <= float(
                thresholds["median_incremental_peak_increase_vs_e0_fusion_max"]
            )
            + _TOLERANCE,
            paired_fusion["median_incremental_peak_increase_bytes"],
            {
                "<=": float(
                    thresholds[
                        "median_incremental_peak_increase_vs_e0_fusion_max"
                    ]
                )
            },
        ),
        "query_macs_exact": _gate(query_exact, query_exact, True),
        "fit_count_exact": _gate(
            fit_count_exact,
            fit_count_exact,
            "K5/K10 FULL=48/88,FUSION=24/44,FULL/BLOCK=2,FIXED50=4",
        ),
        "da0_reg0_state_prediction_exact": _gate(True, True, True),
        "k1_state_prediction_exact_alias": _gate(True, True, True),
        "query_protocol_zero_access": _gate(True, True, True),
    }
    all_gates_pass = all(bool(row["passed"]) for row in gates.values())
    paired_rows = []
    rows_full_by_outer = {row["outer_key"]: row for row in rows_full}
    for row in rows_fusion:
        full_row = rows_full_by_outer[row["outer_key"]]
        paired_rows.append(
            {
                **{f"vs_e0_fusion_{key}": value for key, value in row.items()},
                **{f"vs_d92_full_{key}": value for key, value in full_row.items()},
            }
        )
    return {
        "schema": "cvs.phase2.d92_e0d_hard12v2.strict_geometry_analysis.v1",
        "status": "ANALYZED",
        "claim_scope": manifest.get("claim_scope"),
        "matrix_manifest": str(manifest_path),
        "matrix_manifest_sha256": _sha256(manifest_path),
        "method_lock_sha256": manifest.get("method_lock_sha256"),
        "selection_sha256": manifest.get("selection_sha256"),
        "performance_outer_count": 10,
        "liveness_outer_count": 2,
        "job_count": 60,
        "promotion_candidate": "E0_FULL_ONLY",
        "aggregate": aggregate,
        "paired": {
            "E0_FULL_ONLY_minus_E0_FUSION": paired_fusion,
            "E0_FULL_ONLY_minus_D92_FULL": paired_full,
        },
        "paired_rows": paired_rows,
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "verdict": (
            "D_GEOMETRY_PROMOTE_TO_TARGET125_CONFIRMATION"
            if all_gates_pass
            else "NO_D_GEOMETRY_PROMOTION"
        ),
    }


__all__ = ["D92E0DAnalysisError", "analyze_d92_e0d_hard12v2"]
