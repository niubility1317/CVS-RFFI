"""Strict paired analysis for the development-only D92-BE Hard12 matrix."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_d92_be_slim import D92_BE_ARMS


ARMS = ("FULL", "B0", "E0", "B0E0")
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
_GATE_TOLERANCE = 1e-12


class D92BEAnalysisError(ValueError):
    """Raised when the frozen matrix or its paired evidence is incomplete."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise D92BEAnalysisError(f"required JSON artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise D92BEAnalysisError(f"JSON artifact must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise D92BEAnalysisError(f"non-finite {label}")
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise D92BEAnalysisError("cannot aggregate an empty metric")
    return float(statistics.fmean(values))


def _prediction_key(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    try:
        with np.load(path, allow_pickle=False) as artifact:
            required = {"query_tokens", "scenarios", "predicted_class_handles"}
            if set(artifact.files) != required:
                raise D92BEAnalysisError("prediction artifact member drift")
            tokens = tuple(str(value) for value in artifact["query_tokens"].tolist())
            scenarios = tuple(str(value) for value in artifact["scenarios"].tolist())
            predictions = tuple(
                str(value) for value in artifact["predicted_class_handles"].tolist()
            )
    except (OSError, ValueError) as error:
        raise D92BEAnalysisError(f"invalid prediction artifact: {path}") from error
    if not tokens or not (len(tokens) == len(scenarios) == len(predictions)):
        raise D92BEAnalysisError("prediction artifact row-count drift")
    return tokens, scenarios, predictions


def _scenario_mean(score: Mapping[str, Any], state: str, field: str) -> float:
    state_payload = score.get(state)
    if not isinstance(state_payload, Mapping):
        raise D92BEAnalysisError(f"score state missing: {state}")
    by_scenario = state_payload.get("by_scenario")
    if not isinstance(by_scenario, Mapping) or set(by_scenario) != set(SCENES):
        raise D92BEAnalysisError(f"score scenario coverage drift: {state}")
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
        raise D92BEAnalysisError("after.by_tx is required for old balanced accuracy")
    values = [
        _finite(row["accuracy"], label=f"after.by_tx.{tx}.accuracy")
        for tx, row in by_tx.items()
        if isinstance(row, Mapping) and row.get("role") == "target_old"
    ]
    return _mean(values)


def _load_fit_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise D92BEAnalysisError(f"fit audit is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or len(payload) != len(SCENES):
        raise D92BEAnalysisError("fit audit scenario count drift")
    rows = [dict(row) for row in payload if isinstance(row, Mapping)]
    if len(rows) != len(SCENES) or {str(row.get("scenario")) for row in rows} != set(SCENES):
        raise D92BEAnalysisError("fit audit scenario identity drift")
    return rows


def _job_evidence(
    job: Mapping[str, Any], *, run_root: Path | None = None
) -> dict[str, Any]:
    arm = str(job.get("arm_id"))
    if arm not in D92_BE_ARMS or job.get("candidate") != D92_BE_ARMS[arm].candidate_id:
        raise D92BEAnalysisError("job arm/candidate identity drift")
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
        receipt.get("schema") != "cvs.phase2.d92_be_hard12.job_receipt.v1"
        or receipt.get("status") != "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE"
        or receipt.get("job_id") != job.get("job_id")
        or receipt.get("outer_key") != job.get("outer_key")
        or receipt.get("arm_id") != arm
        or receipt.get("candidate") != job.get("candidate")
        or receipt.get("truth_sidecar_exposed_to_predictor") is not False
        or receipt.get("query_truth_joined_only_after_immutable_predictions") is not True
        or receipt.get("query_truth_fed_back_to_predictor") is not False
    ):
        raise D92BEAnalysisError("job receipt identity/protocol drift")
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
        raise D92BEAnalysisError("prediction-to-score binding drift")
    fit_rows = _load_fit_rows(job_root / "diag" / "after" / "fit_audit.json")
    fit_counts = {
        int(row.get("after_total_component_fit_count", -1)) for row in fit_rows
    }
    query_macs = {int(row.get("query_macs", -1)) for row in fit_rows}
    if len(fit_counts) != 1 or len(query_macs) != 1 or min(query_macs) < 0:
        raise D92BEAnalysisError("fit-count or query-MAC scene drift")
    wall_values: list[float] = []
    peak_values: list[float] = []
    for row in fit_rows:
        resource = row.get("after_registration_resource")
        if not isinstance(resource, Mapping):
            raise D92BEAnalysisError("registered-state resource receipt missing")
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
        raise D92BEAnalysisError("negative resource receipt")
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
        "query_macs": next(iter(query_macs)),
        "_before_prediction": _prediction_key(before_path),
        "_after_prediction": _prediction_key(after_path),
    }


def _gate(passed: bool, observed: Any, threshold: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "observed": observed, "threshold": threshold}


def _expected_fit_count(k_shot: int, arm: str) -> int:
    if int(k_shot) <= 2 or arm in {"FULL", "B0"}:
        return 8 * (int(k_shot) + 1)
    return 4 + 4 * int(k_shot)


def analyze_d92_be_hard12(
    matrix_manifest_path: str | Path,
    *,
    run_root: str | Path | None = None,
    method_lock_path: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze only a complete 48-job frozen matrix with same-outer pairing."""

    manifest_path = Path(matrix_manifest_path).resolve(strict=True)
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema") != "cvs.phase2.d92_be_hard12.matrix.v1"
        or manifest.get("status") != "FROZEN_DEVELOPMENT_MATRIX"
        or manifest.get("protocol_schema") != "p2_min_v1"
        or int(manifest.get("outer_count", -1)) != 12
        or int(manifest.get("performance_outer_count", -1)) != 10
        or int(manifest.get("liveness_outer_count", -1)) != 2
        or int(manifest.get("job_count", -1)) != 48
        or int(manifest.get("shard_count", -1)) != 8
    ):
        raise D92BEAnalysisError("matrix identity/count drift")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 48:
        raise D92BEAnalysisError("matrix jobs are incomplete")
    lock_path = (
        Path(method_lock_path)
        if method_lock_path is not None
        else Path(str(manifest["method_lock"]))
    )
    if _sha256(lock_path) != manifest.get("method_lock_sha256"):
        raise D92BEAnalysisError("method lock identity drift")
    method_lock = _read_json(lock_path)
    if method_lock.get("only_promotion_candidate") != "B0E0":
        raise D92BEAnalysisError("promotion candidate drift")
    thresholds = method_lock.get("strict_pareto_gate")
    if not isinstance(thresholds, Mapping):
        raise D92BEAnalysisError("strict Pareto gate missing")
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
            raise D92BEAnalysisError("matrix has a non-PASS shard")

    evidence = [
        _job_evidence(job, run_root=output_root if run_root is not None else None)
        for job in jobs
    ]
    by_outer: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in evidence:
        if row["arm_id"] in by_outer[row["outer_key"]]:
            raise D92BEAnalysisError("duplicate outer/arm evidence")
        by_outer[row["outer_key"]][row["arm_id"]] = row
    if len(by_outer) != 12 or any(set(rows) != set(ARMS) for rows in by_outer.values()):
        raise D92BEAnalysisError("outer/arm closure drift")

    da0_exact = True
    k1_exact = True
    performance_outers: list[str] = []
    for outer_key, rows in sorted(by_outer.items()):
        reference_before = rows["FULL"]["_before_prediction"]
        if any(rows[arm]["_before_prediction"] != reference_before for arm in ARMS[1:]):
            raise D92BEAnalysisError(f"DA0_REG0 prediction drift: {outer_key}")
        role_values = {row["outer_role"] for row in rows.values()}
        k_values = {row["k_shot"] for row in rows.values()}
        if len(role_values) != 1 or len(k_values) != 1:
            raise D92BEAnalysisError("same-outer role/K drift")
        role = next(iter(role_values))
        k_shot = next(iter(k_values))
        if role == "performance":
            performance_outers.append(outer_key)
        elif role == "liveness":
            if k_shot != 1:
                raise D92BEAnalysisError("liveness outer is not K1")
            reference_after = rows["FULL"]["_after_prediction"]
            k1_exact = k1_exact and all(
                rows[arm]["_after_prediction"] == reference_after for arm in ARMS[1:]
            )
        else:
            raise D92BEAnalysisError("unknown outer role")
    if len(performance_outers) != 10:
        raise D92BEAnalysisError("performance outer closure drift")

    public_evidence = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in evidence
    ]
    performance_rows = [row for row in public_evidence if row["outer_role"] == "performance"]
    aggregate: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        arm_rows = [row for row in performance_rows if row["arm_id"] == arm]
        if len(arm_rows) != 10:
            raise D92BEAnalysisError("performance arm count drift")
        aggregate[arm] = {
            "mean_h_old_new": _mean([row["h_old_new"] for row in arm_rows]),
            "mean_old_balanced_accuracy": _mean(
                [row["old_balanced_accuracy"] for row in arm_rows]
            ),
            "mean_seen_new_accuracy": _mean([row["seen_new_accuracy"] for row in arm_rows]),
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

    paired_rows: list[dict[str, Any]] = []
    wall_reductions: list[float] = []
    peak_reductions: list[float] = []
    resource_denominator_valid = True
    fit_count_exact = True
    for outer_key in sorted(performance_outers):
        full = by_outer[outer_key]["FULL"]
        candidate = by_outer[outer_key]["B0E0"]
        wall_base = float(full["registration_wall_time_ns"])
        peak_base = float(full["registration_incremental_peak_working_set_bytes"])
        if wall_base > 0:
            wall_reduction = 1.0 - float(candidate["registration_wall_time_ns"]) / wall_base
            wall_reductions.append(wall_reduction)
        else:
            wall_reduction = None
            resource_denominator_valid = False
        if peak_base > 0:
            peak_reduction = (
                1.0
                - float(candidate["registration_incremental_peak_working_set_bytes"])
                / peak_base
            )
            peak_reductions.append(peak_reduction)
        else:
            peak_reduction = None
            resource_denominator_valid = False
        for arm in ARMS:
            row = by_outer[outer_key][arm]
            fit_count_exact = fit_count_exact and row["fit_count"] == _expected_fit_count(
                row["k_shot"], arm
            )
        paired_rows.append(
            {
                "outer_key": outer_key,
                "k_shot": int(full["k_shot"]),
                "delta_h": candidate["h_old_new"] - full["h_old_new"],
                "delta_old_balanced": candidate["old_balanced_accuracy"]
                - full["old_balanced_accuracy"],
                "delta_seen_new": candidate["seen_new_accuracy"]
                - full["seen_new_accuracy"],
                "delta_old_floor": candidate["old_floor"] - full["old_floor"],
                "delta_forgetting": candidate["forgetting"] - full["forgetting"],
                "wall_reduction": wall_reduction,
                "incremental_peak_reduction": peak_reduction,
                "query_macs_increase": candidate["query_macs"] - full["query_macs"],
                "full_fit_count": full["fit_count"],
                "b0e0_fit_count": candidate["fit_count"],
            }
        )
    mean_delta_h = _mean([row["delta_h"] for row in paired_rows])
    mean_delta_old = _mean([row["delta_old_balanced"] for row in paired_rows])
    mean_delta_new = _mean([row["delta_seen_new"] for row in paired_rows])
    mean_delta_floor = _mean([row["delta_old_floor"] for row in paired_rows])
    mean_delta_forgetting = _mean([row["delta_forgetting"] for row in paired_rows])
    nonnegative_count = sum(row["delta_h"] >= -1e-12 for row in paired_rows)
    median_wall_reduction = (
        float(statistics.median(wall_reductions)) if len(wall_reductions) == 10 else None
    )
    median_peak_reduction = (
        float(statistics.median(peak_reductions)) if len(peak_reductions) == 10 else None
    )
    max_query_increase = max(row["query_macs_increase"] for row in paired_rows)
    gates = {
        "mean_delta_h": _gate(
            mean_delta_h
            >= float(thresholds["mean_delta_h_min"]) - _GATE_TOLERANCE,
            mean_delta_h,
            {">=": float(thresholds["mean_delta_h_min"])},
        ),
        "nonnegative_delta_h_outer_count": _gate(
            nonnegative_count >= int(thresholds["nonnegative_delta_h_outer_min"]),
            nonnegative_count,
            {">=": int(thresholds["nonnegative_delta_h_outer_min"])},
        ),
        "mean_delta_old_balanced": _gate(
            mean_delta_old
            >= float(thresholds["mean_delta_old_balanced_min"])
            - _GATE_TOLERANCE,
            mean_delta_old,
            {">=": float(thresholds["mean_delta_old_balanced_min"])},
        ),
        "mean_delta_seen_new": _gate(
            mean_delta_new
            >= float(thresholds["mean_delta_seen_new_min"]) - _GATE_TOLERANCE,
            mean_delta_new,
            {">=": float(thresholds["mean_delta_seen_new_min"])},
        ),
        "mean_delta_old_floor": _gate(
            mean_delta_floor
            >= float(thresholds["mean_delta_old_floor_min"]) - _GATE_TOLERANCE,
            mean_delta_floor,
            {">=": float(thresholds["mean_delta_old_floor_min"])},
        ),
        "mean_delta_forgetting": _gate(
            mean_delta_forgetting
            <= float(thresholds["mean_delta_forgetting_max"]) + _GATE_TOLERANCE,
            mean_delta_forgetting,
            {"<=": float(thresholds["mean_delta_forgetting_max"])},
        ),
        "median_wall_reduction": _gate(
            resource_denominator_valid
            and median_wall_reduction is not None
            and median_wall_reduction
            >= float(thresholds["median_wall_reduction_min"]) - _GATE_TOLERANCE,
            median_wall_reduction,
            {">=": float(thresholds["median_wall_reduction_min"])},
        ),
        "median_incremental_peak_reduction": _gate(
            resource_denominator_valid
            and median_peak_reduction is not None
            and median_peak_reduction
            >= float(thresholds["median_incremental_peak_reduction_min"])
            - _GATE_TOLERANCE,
            median_peak_reduction,
            {">=": float(thresholds["median_incremental_peak_reduction_min"])},
        ),
        "query_cost_nonincrease": _gate(
            max_query_increase
            <= float(thresholds["query_cost_increase_max"]) + _GATE_TOLERANCE,
            max_query_increase,
            {"<=": float(thresholds["query_cost_increase_max"])},
        ),
        "fit_count_exact": _gate(fit_count_exact, fit_count_exact, "K5 48/24; K10 88/44"),
        "da0_reg0_prediction_exact": _gate(da0_exact, da0_exact, True),
        "k1_exact_alias_liveness": _gate(k1_exact, k1_exact, True),
    }
    all_gates_pass = all(bool(row["passed"]) for row in gates.values())
    paired = {
        "B0E0_minus_FULL": {
            "mean_delta_h": mean_delta_h,
            "nonnegative_delta_h_outer_count": nonnegative_count,
            "mean_delta_old_balanced": mean_delta_old,
            "mean_delta_seen_new": mean_delta_new,
            "mean_delta_old_floor": mean_delta_floor,
            "mean_delta_forgetting": mean_delta_forgetting,
            "median_wall_reduction": median_wall_reduction,
            "median_incremental_peak_reduction": median_peak_reduction,
            "max_query_macs_increase": max_query_increase,
        }
    }
    return {
        "schema": "cvs.phase2.d92_be_hard12.strict_pareto_analysis.v1",
        "status": "ANALYZED",
        "claim_scope": manifest.get("claim_scope"),
        "matrix_manifest": str(manifest_path),
        "matrix_manifest_sha256": _sha256(manifest_path),
        "method_lock_sha256": manifest.get("method_lock_sha256"),
        "performance_outer_count": 10,
        "liveness_outer_count": 2,
        "job_count": 48,
        "promotion_candidate": "B0E0",
        "aggregate": aggregate,
        "paired": paired,
        "paired_rows": paired_rows,
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "verdict": (
            "STRICT_PARETO_PROMOTE_TO_TARGET125_CONFIRMATION"
            if all_gates_pass
            else "NO_STRICT_PARETO_PROMOTION"
        ),
    }


__all__ = ["D92BEAnalysisError", "analyze_d92_be_hard12"]
