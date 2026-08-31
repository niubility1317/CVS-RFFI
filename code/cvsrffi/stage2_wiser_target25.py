"""Immutable historical Target25/K10 validation for WISER-RF P3-primary.

This module only filters the already-bound Target125 surface and analyzes
truth-last paired rows.  It never reconstructs a Cartesian matrix or opens
packages, query payloads, or truth sidecars.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import PurePosixPath
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_bisage_target125 import (
    RECEIVERS,
    SCENARIOS,
    SEEDS,
    canonical_target125_rows,
)


class WISERTarget25Error(ValueError):
    """Raised when the historical matrix, marker, or paired evidence drifts."""


_PHASES = {"target25", "k10"}
_PACKAGE_ROLES = (
    "before_enrollment", "before_apply", "after_enrollment", "after_apply"
)
_CHAMPION_ARMS = {"N2", "N3", "N4", "N5", "N6"}
_CHAMPION_FIELDS = ("arm", "commit", "config_id", "checkpoint_id", "capsule_id", "split_id")
_P3_PROBE = "P3_OLD_D92"


def canonical_target25_rows() -> tuple[dict[str, Any], ...]:
    """Return the historical K10/new5 rows, in Target125 order."""

    return tuple(
        deepcopy(row)
        for row in canonical_target125_rows()
        if int(row["k_shot"]) == 10 and int(row["new_class_count"]) == 5
    )


def canonical_k10_expansion_rows() -> tuple[dict[str, Any], ...]:
    """Return the historical K10/new5,new10,new20 rows, in Target125 order."""

    return tuple(
        deepcopy(row)
        for row in canonical_target125_rows()
        if int(row["k_shot"]) == 10 and int(row["new_class_count"]) in {5, 10, 20}
    )


def _phase_rows(phase: str) -> tuple[dict[str, Any], ...]:
    value = str(phase).lower()
    if value == "target25":
        return canonical_target25_rows()
    if value == "k10":
        return canonical_k10_expansion_rows()
    raise WISERTarget25Error("validation phase must be target25 or k10")


def _as_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise WISERTarget25Error(f"{field} is missing")
    return value


def validate_p3_primary_marker(marker: Mapping[str, Any]) -> dict[str, str]:
    """Require the analyzed, single-champion Task6 authorization marker."""

    if not isinstance(marker, Mapping):
        raise WISERTarget25Error("P3 pilot marker must be an object")
    if marker.get("schema") != "cvs.phase2.wiser_rf.p3_primary.target25_authorization.v1":
        raise WISERTarget25Error("P3 pilot marker schema drift")
    if marker.get("status") != "ANALYZED" or marker.get("full_target25_authorized") is not True:
        raise WISERTarget25Error("P3 pilot marker does not authorize Target25")
    arm = _as_nonempty_string(marker.get("p3_primary_champion"), "P3 champion")
    identity = marker.get("champion_identity")
    if arm not in _CHAMPION_ARMS or not isinstance(identity, Mapping):
        raise WISERTarget25Error("P3 pilot marker champion is invalid")
    result = {field: _as_nonempty_string(identity.get(field), f"champion {field}") for field in _CHAMPION_FIELDS}
    if result["arm"] != arm:
        raise WISERTarget25Error("P3 champion identity drift")
    return result


def _bound_source_jobs(source_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Validate the full Target125 bound surface before filtering a single row."""

    if source_manifest.get("schema") != "cvs.phase2.bisage_d92_target125.manifest.v1":
        raise WISERTarget25Error("bound Target125 manifest schema drift")
    if source_manifest.get("protocol_schema") != "p2_min_v1" or source_manifest.get("phase2_data_status") != "VALIDATED_ONCE":
        raise WISERTarget25Error("bound Target125 protocol/data status drift")
    jobs = source_manifest.get("jobs")
    expected = canonical_target125_rows()
    if not isinstance(jobs, list) or len(jobs) != len(expected):
        raise WISERTarget25Error("bound Target125 outer coverage drift")
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, (job, canonical) in enumerate(zip(jobs, expected)):
        if not isinstance(job, Mapping) or job.get("outer_key") != canonical["outer_key"]:
            raise WISERTarget25Error("bound Target125 outer order drift")
        if job["outer_key"] in indexed:
            raise WISERTarget25Error("bound Target125 outer duplication")
        for field in ("receiver", "seed", "k_shot", "new_class_count", "scenarios"):
            if job.get(field) != canonical[field]:
                raise WISERTarget25Error(f"bound Target125 {field} drift")
        if job.get("planned_shard_index") != index % 8:
            raise WISERTarget25Error("bound Target125 shard assignment drift")
        if job.get("protocol_schema") != "p2_min_v1" or job.get("phase2_data_status") != "VALIDATED_ONCE":
            raise WISERTarget25Error("bound Target125 job protocol/data status drift")
        for field in ("capsule_id", "split_id", "source_capsule_id", "source_split_id", "source_job_root", "truth_sidecar"):
            _as_nonempty_string(job.get(field), f"bound Target125 {field}")
        if job.get("capsule_id") != job.get("source_capsule_id") or job.get("split_id") != job.get("source_split_id"):
            raise WISERTarget25Error("bound Target125 source capsule/split drift")
        packages = job.get("packages")
        if not isinstance(packages, Mapping) or tuple(packages) != _PACKAGE_ROLES:
            raise WISERTarget25Error("bound Target125 package role drift")
        for role in _PACKAGE_ROLES:
            package = packages[role]
            if not isinstance(package, Mapping) or not package.get("package_root"):
                raise WISERTarget25Error("bound Target125 package binding missing")
        indexed[str(job["outer_key"])] = job
    return indexed


def build_wiser_target25_manifest(
    source_manifest: Mapping[str, Any],
    pilot_marker: Mapping[str, Any],
    output_root: str,
    *,
    phase: str = "target25",
) -> dict[str, Any]:
    """Filter the exact bound Target125 jobs into a new immutable phase root."""

    identity = validate_p3_primary_marker(pilot_marker)
    sources = _bound_source_jobs(source_manifest)
    root = PurePosixPath(str(output_root))
    if str(root) in {"", ".", "/"}:
        raise WISERTarget25Error("output root must be a dedicated immutable path")
    rows = _phase_rows(phase)
    jobs: list[dict[str, Any]] = []
    for canonical in rows:
        source = sources.get(str(canonical["outer_key"]))
        if source is None:
            raise WISERTarget25Error("bound Target125 join is incomplete")
        job = deepcopy(dict(source))
        job.update({
            "validation_phase": str(phase).lower(),
            "output_root": str(root / "jobs" / str(canonical["outer_key"])),
            "champion_arm": identity["arm"], "champion_commit": identity["commit"],
            "champion_config_id": identity["config_id"], "champion_checkpoint_id": identity["checkpoint_id"],
            "champion_capsule_id": identity["capsule_id"], "champion_split_id": identity["split_id"],
            "query_rows_used": 0,
        })
        jobs.append(job)
    manifest = {
        "schema": "cvs.phase2.wiser_rf.target25.manifest.v1",
        "status": "PREPARED", "validation_phase": str(phase).lower(),
        "protocol_schema": "p2_min_v1", "phase2_data_status": "VALIDATED_ONCE",
        "champion_identity": identity, "source_manifest_schema": source_manifest["schema"],
        "output_root": str(root), "query_opened": False, "truth_opened": False,
        "jobs": jobs,
    }
    validate_wiser_target25_manifest(manifest)
    return manifest


def validate_wiser_target25_manifest(manifest: Mapping[str, Any]) -> dict[str, int]:
    if manifest.get("schema") != "cvs.phase2.wiser_rf.target25.manifest.v1" or manifest.get("status") != "PREPARED":
        raise WISERTarget25Error("Target25 manifest schema/status drift")
    if manifest.get("protocol_schema") != "p2_min_v1" or manifest.get("phase2_data_status") != "VALIDATED_ONCE":
        raise WISERTarget25Error("Target25 manifest protocol/data status drift")
    phase = str(manifest.get("validation_phase"))
    expected_rows = _phase_rows(phase)
    identity = manifest.get("champion_identity")
    if not isinstance(identity, Mapping) or set(identity) != set(_CHAMPION_FIELDS):
        raise WISERTarget25Error("Target25 champion identity drift")
    if identity.get("arm") not in _CHAMPION_ARMS:
        raise WISERTarget25Error("Target25 champion arm drift")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(expected_rows):
        raise WISERTarget25Error("Target25 outer coverage drift")
    roots: set[str] = set()
    for source_index, (job, canonical) in enumerate(zip(jobs, expected_rows)):
        if not isinstance(job, Mapping) or job.get("outer_key") != canonical["outer_key"]:
            raise WISERTarget25Error("Target25 outer order drift")
        historical_index = canonical_target125_rows().index(canonical)
        for field in ("receiver", "seed", "k_shot", "new_class_count", "scenarios"):
            if job.get(field) != canonical[field]:
                raise WISERTarget25Error(f"Target25 {field} drift")
        if job.get("planned_shard_index") != historical_index % 8:
            raise WISERTarget25Error("Target25 historical shard drift")
        for field in ("capsule_id", "split_id", "source_capsule_id", "source_split_id", "source_job_root", "truth_sidecar"):
            _as_nonempty_string(job.get(field), f"Target25 {field}")
        if job.get("capsule_id") != job.get("source_capsule_id") or job.get("split_id") != job.get("source_split_id"):
            raise WISERTarget25Error("Target25 source capsule/split drift")
        if job.get("validation_phase") != phase or job.get("query_rows_used") != 0:
            raise WISERTarget25Error("Target25 query policy drift")
        for field, identity_field in (("champion_arm", "arm"), ("champion_commit", "commit"), ("champion_config_id", "config_id"), ("champion_checkpoint_id", "checkpoint_id"), ("champion_capsule_id", "capsule_id"), ("champion_split_id", "split_id")):
            if job.get(field) != identity[identity_field]:
                raise WISERTarget25Error("Target25 champion binding drift")
        root = _as_nonempty_string(job.get("output_root"), "Target25 output root")
        roots.add(root)
    if len(roots) != len(jobs):
        raise WISERTarget25Error("Target25 output-root collision")
    return {"outer_count": len(jobs), "scene_unit_count": len(jobs) * len(SCENARIOS)}


def _finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise WISERTarget25Error(f"Target25 {field} is nonfinite")
    return result


def _linear_quantile(values: Sequence[float], q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise WISERTarget25Error("Target25 quantile evidence is invalid")
    return float(np.quantile(array, float(q), method="linear"))


def _failed(phase: str, reason: str) -> dict[str, Any]:
    return {
        "schema": "cvs.phase2.wiser_rf.target25.analysis.v1", "status": "PENDING_EVIDENCE",
        "validation_phase": phase, "evidence_complete": False, "reason": reason,
        "gates": {}, "passed": False, "k10_expansion_authorized": False,
        "stage_b_authorized": False, "stage_b_eligible": False,
    }


def _expected_scene_units(phase: str) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for outer in _phase_rows(phase):
        for scenario in SCENARIOS:
            result[(str(outer["outer_key"]), scenario)] = outer
    return result


def _p3_values(row: Mapping[str, Any]) -> Mapping[str, Any]:
    p3 = row.get("p3")
    if not isinstance(p3, Mapping):
        probes = row.get("probes")
        if isinstance(probes, Mapping):
            p3 = probes.get(_P3_PROBE)
    if not isinstance(p3, Mapping):
        raise WISERTarget25Error("Target25 P3 paired evidence is missing")
    required = ("balanced_accuracy_delta_pp", "floor_delta_pp", "net_help_minus_harm", "help_count", "harm_count", "accuracy_delta_pp", "nll_delta", "per_class_accuracy_delta_pp", "control_metrics", "candidate_metrics")
    if any(field not in p3 for field in required):
        raise WISERTarget25Error("Target25 detailed P3 evidence is incomplete")
    for field in ("balanced_accuracy_delta_pp", "floor_delta_pp", "accuracy_delta_pp", "nll_delta"):
        _finite(p3[field], field)
    help_count, harm_count, net = int(p3["help_count"]), int(p3["harm_count"]), int(p3["net_help_minus_harm"])
    if help_count < 0 or harm_count < 0 or net != help_count - harm_count:
        raise WISERTarget25Error("Target25 help/harm evidence is invalid")
    per_class = p3["per_class_accuracy_delta_pp"]
    if not isinstance(per_class, Mapping) or set(per_class) != {str(index) for index in range(6)}:
        raise WISERTarget25Error("Target25 per-class evidence is invalid")
    for value in per_class.values(): _finite(value, "per-class delta")
    for side in ("control_metrics", "candidate_metrics"):
        metrics = p3[side]
        if not isinstance(metrics, Mapping): raise WISERTarget25Error("Target25 absolute metrics are missing")
        for field in ("accuracy", "balanced_accuracy", "floor", "nll"):
            _finite(metrics.get(field), f"{side}.{field}")
    return p3


def _validate_rows(rows: Sequence[Mapping[str, Any]], phase: str) -> list[Mapping[str, Any]]:
    expected = _expected_scene_units(phase)
    if len(rows) != len(expected):
        raise WISERTarget25Error("Target25 paired grid is incomplete")
    observed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("schema") != "cvs.phase2.wiser_rf.paired_query_delta.v1":
            raise WISERTarget25Error("Target25 paired schema drift")
        key = (str(row.get("outer_key")), str(row.get("scenario")))
        canonical = expected.get(key)
        if canonical is None or key in observed:
            raise WISERTarget25Error("Target25 paired grid coverage drift")
        if row.get("control_arm") != "N0" or str(row.get("candidate_arm")) not in _CHAMPION_ARMS:
            raise WISERTarget25Error("Target25 paired arm drift")
        for field in ("receiver", "seed", "k_shot", "new_class_count"):
            if row.get(field) != canonical[field]:
                raise WISERTarget25Error("Target25 paired binding drift")
        if row.get("scenario") not in SCENARIOS or not row.get("capsule_id") or not row.get("split_id"):
            raise WISERTarget25Error("Target25 paired binding drift")
        count = row.get("query_rows")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or row.get("query_rows_used") != 0:
            raise WISERTarget25Error("Target25 actual query-row evidence is invalid")
        if row.get("planned_shard_index") != canonical_target125_rows().index(canonical) % 8:
            raise WISERTarget25Error("Target25 paired shard drift")
        audit = row.get("candidate_training_audit")
        if not isinstance(audit, Mapping):
            raise WISERTarget25Error("Target25 candidate support audit is missing")
        zero = audit.get("final_zero_identity_count")
        baseline = _finite(audit.get("baseline_joint_condition_number"), "baseline condition")
        final = _finite(audit.get("final_joint_condition_number"), "final condition")
        if isinstance(zero, bool) or not isinstance(zero, int) or zero != 0 or baseline <= 0.0 or final < 0.0:
            raise WISERTarget25Error("Target25 support safety evidence is invalid")
        _p3_values(row)
        observed[key] = row
    return [observed[key] for key in expected]


def _aggregate(values: Sequence[float]) -> dict[str, float]:
    checked = [_finite(value, "aggregate") for value in values]
    return {"mean": float(np.mean(checked)), "median": float(median(checked)), "worst": float(min(checked)), "linear_q10": _linear_quantile(checked, .10)}


def _bootstrap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blocks: dict[str, list[float]] = {}
    for row in rows:
        blocks.setdefault(str(row["outer_key"]), []).append(_finite(_p3_values(row)["balanced_accuracy_delta_pp"], "P3 BA"))
    if not blocks or any(len(values) != 3 for values in blocks.values()):
        raise WISERTarget25Error("Target25 bootstrap blocks are incomplete")
    rng = np.random.default_rng(713102)
    keys = tuple(blocks)
    medians = np.empty(2000, dtype=np.float64)
    for index in range(len(medians)):
        sample = rng.integers(0, len(keys), size=len(keys))
        values = [value for block_index in sample for value in blocks[keys[int(block_index)]]]
        medians[index] = float(np.median(values))
    return {"seed": 713102, "iterations": 2000, "confidence": 0.95,
            "p3_ba_median_ci_pp": [_linear_quantile(medians, .025), _linear_quantile(medians, .975)]}


def target25_promotion_decision(rows: Sequence[Mapping[str, Any]], *, phase: str = "target25") -> dict[str, Any]:
    """Analyze a complete paired grid and apply Target25 or K10 promotion gates."""

    phase_value = str(phase).lower()
    try:
        selected = _validate_rows(rows, phase_value)
        values = [_p3_values(row) for row in selected]
        ba = [_finite(item["balanced_accuracy_delta_pp"], "P3 BA") for item in values]
        floor = [_finite(item["floor_delta_pp"], "P3 floor") for item in values]
        scenario_ba = {scene: _aggregate([_finite(_p3_values(row)["balanced_accuracy_delta_pp"], "P3 BA") for row in selected if row["scenario"] == scene]) for scene in SCENARIOS}
        scenario_floor = {scene: _aggregate([_finite(_p3_values(row)["floor_delta_pp"], "P3 floor") for row in selected if row["scenario"] == scene]) for scene in SCENARIOS}
        receiver = {name: float(median([_finite(_p3_values(row)["balanced_accuracy_delta_pp"], "P3 BA") for row in selected if row["receiver"] == name])) for name in RECEIVERS}
        seed = {str(name): float(median([_finite(_p3_values(row)["balanced_accuracy_delta_pp"], "P3 BA") for row in selected if row["seed"] == name])) for name in SEEDS}
        net = {scene: int(sum(int(_p3_values(row)["net_help_minus_harm"]) for row in selected if row["scenario"] == scene)) for scene in SCENARIOS}
        all_safe = all(
            int(row["candidate_training_audit"]["final_zero_identity_count"]) == 0
            and _finite(row["candidate_training_audit"]["final_joint_condition_number"], "final condition") <= 2.0 * _finite(row["candidate_training_audit"]["baseline_joint_condition_number"], "baseline condition")
            for row in selected
        )
        per_class = {str(index): _aggregate([_finite(_p3_values(row)["per_class_accuracy_delta_pp"][str(index)], "per-class") for row in selected]) for index in range(6)}
        absolute = {side: {metric: _aggregate([_finite(_p3_values(row)[side][metric], metric) for row in selected]) for metric in ("accuracy", "balanced_accuracy", "floor", "nll")} for side in ("control_metrics", "candidate_metrics")}
        resources = {key: [row.get("candidate_training_audit", {}).get(key) for row in selected if key in row.get("candidate_training_audit", {})] for key in ("optimizer_steps", "final_joint_condition_number", "baseline_joint_condition_number")}
    except (KeyError, TypeError, ValueError, WISERTarget25Error) as error:
        return _failed(phase_value, str(error))
    overall = _aggregate(ba)
    gates = {
        "overall_p3_ba_median_ge_3pp": overall["median"] >= 3.0,
        "every_scene_p3_ba_median_ge_0pp": all(item["median"] >= 0.0 for item in scenario_ba.values()),
        "linear_p3_ba_q10_ge_minus_2pp": overall["linear_q10"] >= -2.0,
        "overall_p3_floor_median_ge_0pp": float(median(floor)) >= 0.0,
        "low_elev_p3_floor_median_ge_0pp": scenario_floor["leo_low_elev_weak"]["median"] >= 0.0,
        "receiver_median_positive_at_least_4of5": sum(value > 0.0 for value in receiver.values()) >= 4,
        "seed_median_positive_at_least_4of5": sum(value > 0.0 for value in seed.values()) >= 4,
        "zero_identity_and_condition_safety_all_units": all_safe,
        "positive_scene_net_help_at_least_2of3": sum(value > 0 for value in net.values()) >= 2,
    }
    passed = all(gates.values())
    return {
        "schema": "cvs.phase2.wiser_rf.target25.analysis.v1", "status": "ANALYZED",
        "validation_phase": phase_value, "evidence_complete": True, "scene_unit_count": len(selected),
        "outer_count": len(_phase_rows(phase_value)), "actual_query_rows": int(sum(int(row["query_rows"]) for row in selected)),
        "per_unit_p3": [
            {
                **{field: row[field] for field in ("outer_key", "capsule_id", "split_id", "receiver", "seed", "k_shot", "new_class_count", "scenario", "query_rows", "planned_shard_index")},
                "p3": deepcopy(dict(_p3_values(row))),
                "candidate_training_audit": deepcopy(dict(row["candidate_training_audit"])),
            }
            for row in selected
        ],
        "overall_p3_ba_delta_pp": overall, "scenario_p3_ba_delta_pp": scenario_ba,
        "scenario_p3_floor_delta_pp": scenario_floor, "overall_p3_floor_median_pp": float(median(floor)),
        "receiver_p3_ba_median_pp": receiver, "seed_p3_ba_median_pp": seed,
        "per_class_p3_delta_pp": per_class, "absolute_metrics": absolute,
        "help_harm": {"help_count": int(sum(int(item["help_count"]) for item in values)), "harm_count": int(sum(int(item["harm_count"]) for item in values)), "net_help_minus_harm_by_scene": net},
        "resource_distributions": resources, "paired_cluster_bootstrap": _bootstrap(selected),
        "gates": gates, "passed": passed,
        "k10_expansion_authorized": bool(passed and phase_value == "target25"),
        "stage_b_authorized": False, "stage_b_eligible": bool(passed and phase_value == "k10"),
    }


__all__ = [
    "SCENARIOS", "WISERTarget25Error", "build_wiser_target25_manifest",
    "canonical_k10_expansion_rows", "canonical_target25_rows", "target25_promotion_decision",
    "validate_p3_primary_marker", "validate_wiser_target25_manifest",
]
