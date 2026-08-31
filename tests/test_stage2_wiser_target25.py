from __future__ import annotations

from copy import deepcopy

import pytest

from cvsrffi.stage2_bisage_target125 import canonical_target125_rows
from cvsrffi.stage2_wiser_target25 import (
    SCENARIOS,
    WISERTarget25Error,
    build_wiser_target25_manifest,
    canonical_k10_expansion_rows,
    canonical_target25_rows,
    target25_promotion_decision,
)


def _marker() -> dict[str, object]:
    return {
        "schema": "cvs.phase2.wiser_rf.p3_primary.score_collection.v1",
        "status": "ANALYZED",
        "full_target25_authorized": True,
        "p3_primary_champion": "N6",
        "champion_identity": {
            "arm": "N6", "runtime_commit": "abc123", "p3_config_sha256": "a" * 64,
            "checkpoint_id": "ADV3B02_CORE90_SOFT_E200", "checkpoint_sha256": "b" * 64,
            "source_summary_sha256": "c" * 64, "source_binding_sha256": "d" * 64,
            "outer_key": "pilot", "capsule_id": "pilot-capsule", "split_id": "pilot-split", "receiver": "3-19", "seed": 713102, "k_shot": 10, "new_class_count": 5,
        },
    }


def _source_manifest() -> dict[str, object]:
    jobs = []
    for index, row in enumerate(canonical_target125_rows()):
        jobs.append({
            **deepcopy(row), "planned_shard_index": index % 8,
            "protocol_schema": "p2_min_v1", "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": f"cap-{row['outer_key']}", "split_id": f"split-{row['outer_key']}",
            "source_capsule_id": f"cap-{row['outer_key']}", "source_split_id": f"split-{row['outer_key']}",
            "source_job_root": f"/source/{row['outer_key']}",
            "packages": {name: {"package_root": f"/pkg/{row['outer_key']}/{name}"} for name in (
                "before_enrollment", "before_apply", "after_enrollment", "after_apply")},
            "truth_sidecar": f"/truth/{row['outer_key']}.json",
        })
    return {"schema": "cvs.phase2.bisage_d92_target125.manifest.v1", "protocol_schema": "p2_min_v1", "phase2_data_status": "VALIDATED_ONCE", "jobs": jobs}


def _safe_rows(*, phase: str = "target25") -> list[dict[str, object]]:
    source = canonical_target25_rows() if phase == "target25" else canonical_k10_expansion_rows()
    rows: list[dict[str, object]] = []
    for outer in source:
        for scenario in SCENARIOS:
            rows.append({
                "schema": "cvs.phase2.wiser_rf.paired_query_delta.v1", "control_arm": "N0", "candidate_arm": "N6",
                "outer_key": outer["outer_key"], "capsule_id": f"cap-{outer['outer_key']}", "split_id": f"split-{outer['outer_key']}",
                "receiver": outer["receiver"], "seed": outer["seed"], "k_shot": outer["k_shot"], "new_class_count": outer["new_class_count"],
                "scenario": scenario, "query_rows": 17, "expected_query_tokens": [f"{outer['outer_key']}-{scenario}"],
                "query_rows_used": 0, "planned_shard_index": canonical_target125_rows().index(outer) % 8,
                "p3": {"balanced_accuracy_delta_pp": 3.0, "floor_delta_pp": 0.0, "net_help_minus_harm": 1,
                       "help_count": 1, "harm_count": 0, "accuracy_delta_pp": 3.0, "nll_delta": -0.1,
                       "per_class_accuracy_delta_pp": {str(i): 3.0 for i in range(6)},
                       "control_metrics": {"accuracy": .5, "balanced_accuracy": .5, "floor": .4, "nll": 1.0, "per_class_accuracy": {str(i): .5 for i in range(6)}},
                       "candidate_metrics": {"accuracy": .53, "balanced_accuracy": .53, "floor": .4, "nll": .9, "per_class_accuracy": {str(i): .53 for i in range(6)}}},
                "candidate_training_audit": {"final_zero_identity_count": 0, "baseline_joint_condition_number": 2.0, "final_joint_condition_number": 2.0},
            })
    return rows


def test_target25_and_k10_are_exact_historical_filters_in_historical_order() -> None:
    target25 = canonical_target25_rows()
    k10 = canonical_k10_expansion_rows()
    assert len(target25) == 25
    assert len(k10) == 75
    assert sum(len(row["scenarios"]) for row in target25) == 75
    assert sum(len(row["scenarios"]) for row in k10) == 225
    assert {(row["k_shot"], row["new_class_count"]) for row in target25} == {(10, 5)}
    assert {(row["k_shot"], row["new_class_count"]) for row in k10} == {(10, 5), (10, 10), (10, 20)}
    assert tuple(target25) == tuple(row for row in canonical_target125_rows() if row["k_shot"] == 10 and row["new_class_count"] == 5)
    assert tuple(k10) == tuple(row for row in canonical_target125_rows() if row["k_shot"] == 10 and row["new_class_count"] in {5, 10, 20})


def test_manifest_joins_bound_jobs_without_renumbering_or_package_drift() -> None:
    source = _source_manifest()
    manifest = build_wiser_target25_manifest(source, _marker(), "/immutable/target25")
    assert len(manifest["jobs"]) == 25
    assert {job["planned_shard_index"] for job in manifest["jobs"]} <= set(range(8))
    for job in manifest["jobs"]:
        historical = next(row for row in source["jobs"] if row["outer_key"] == job["outer_key"])
        for field in ("capsule_id", "split_id", "source_capsule_id", "source_split_id", "source_job_root", "packages", "truth_sidecar", "planned_shard_index"):
            assert job[field] == historical[field]
        assert job["champion_arm"] == "N6"
        assert job["query_rows_used"] == 0
    with pytest.raises(WISERTarget25Error):
        build_wiser_target25_manifest(source, _marker(), "/")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonfinite", "binding"])
def test_target25_gate_fails_closed_for_incomplete_duplicate_nonfinite_or_binding_grid(mutation: str) -> None:
    rows = _safe_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif mutation == "nonfinite":
        rows[0]["p3"]["balanced_accuracy_delta_pp"] = float("nan")
    else:
        rows[0]["seed"] = 1
    decision = target25_promotion_decision(rows, phase="target25")
    assert decision["passed"] is False
    assert decision["stage_b_authorized"] is False


@pytest.mark.parametrize("field,value", [
    ("ba", 2.99), ("scene", -0.01), ("q10", -2.01), ("floor", -0.01), ("receiver", -0.01), ("seed", -0.01), ("net", 0),
])
def test_target25_gate_has_each_preregistered_threshold(field: str, value: float) -> None:
    rows = _safe_rows()
    if field == "ba":
        for row in rows: row["p3"]["balanced_accuracy_delta_pp"] = value
    elif field == "scene":
        for row in rows:
            if row["scenario"] == "leo_low_elev_weak": row["p3"]["balanced_accuracy_delta_pp"] = value
    elif field == "q10":
        for row in rows[:9]: row["p3"]["balanced_accuracy_delta_pp"] = value
    elif field == "floor":
        for row in rows:
            if row["scenario"] == "leo_low_elev_weak": row["p3"]["floor_delta_pp"] = value
    elif field in {"receiver", "seed"}:
        key = "receiver" if field == "receiver" else "seed"
        selected = sorted({row[key] for row in rows})[:2]
        for row in rows:
            if row[key] in selected: row["p3"]["balanced_accuracy_delta_pp"] = value
    else:
        for row in rows:
            if row["scenario"] != "leo_clear_weak": row["p3"].update(help_count=0, harm_count=0, net_help_minus_harm=0)
    assert target25_promotion_decision(rows, phase="target25")["passed"] is False


def test_target25_pass_only_authorizes_k10_and_k10_pass_is_stage_b_eligible() -> None:
    target25 = target25_promotion_decision(_safe_rows(), phase="target25")
    k10 = target25_promotion_decision(_safe_rows(phase="k10"), phase="k10")
    assert target25["passed"] is True
    assert target25["k10_expansion_authorized"] is True
    assert target25["stage_b_authorized"] is False
    assert k10["passed"] is True
    assert k10["stage_b_eligible"] is True
    assert k10["actual_query_rows"] == 225 * 17
