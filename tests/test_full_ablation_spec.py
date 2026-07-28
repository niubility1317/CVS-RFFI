from __future__ import annotations

import copy

import pytest

from cvsrffi.full_ablation_spec import (
    LEO_SCENARIOS,
    PHASE1_T1_ARMS,
    PHASE2_T1_ARMS,
    REQUIRED_RUN_ARTIFACT_FIELDS,
    ArmSpec,
    FullAblationSpecError,
    SeedBundle,
    assign_worker_slots,
    build_phase1_t1_rows,
    build_phase2_rows,
    validate_artifact_record,
)


def _bundles(count: int, start: int = 820001) -> list[SeedBundle]:
    return [
        SeedBundle(
            train_seed=start + 3 * index,
            support_seed=start + 3 * index + 1,
            query_seed=start + 3 * index + 2,
        )
        for index in range(count)
    ]


def test_phase1_t1_is_30_paired_runs_on_current_split() -> None:
    rows = build_phase1_t1_rows(
        [810001, 810002, 810003, 810004, 810005],
        git_commit="abcdef0123456789",
    )
    assert len(PHASE1_T1_ARMS) == 6
    assert len(rows) == 30
    assert len({row["row_key"] for row in rows}) == 30
    assert {
        tuple(row["split_fractions"].values()) for row in rows
    } == {(0.07, 0.63, 0.30)}
    for seed in {row["train_seed"] for row in rows}:
        assert len([row for row in rows if row["train_seed"] == seed]) == 6


def test_screening_is_75_rows_per_arm_and_three_scenarios_per_row() -> None:
    arm = ArmSpec("P2-TEST", "stage2c", "M", "test", "P2-FULL")
    rows = build_phase2_rows(
        stage="screening",
        arms=[arm],
        seed_bundles=_bundles(3),
        class_draw_seeds=[830001],
        git_commit="abcdef0123456789",
    )
    assert len(rows) == 75
    assert all(tuple(row["scenarios"]) == LEO_SCENARIOS for row in rows)
    assert sum(len(row["scenarios"]) for row in rows) == 225
    assert {(row["k_shot"], row["new_class_count"]) for row in rows} == {
        (1, 20),
        (2, 20),
        (5, 20),
        (10, 5),
        (10, 20),
    }
    assert all(row["formal_launch_authority"] is False for row in rows)


def test_confirmation_is_900_rows_per_arm() -> None:
    arm = ArmSpec("P2-TEST", "stage2c", "M", "test", "P2-FULL")
    rows = build_phase2_rows(
        stage="confirmation",
        arms=[arm],
        seed_bundles=_bundles(5),
        class_draw_seeds=[830001, 830002, 830003],
        git_commit="abcdef0123456789",
    )
    assert len(rows) == 900
    assert sum(len(row["scenarios"]) for row in rows) == 2700
    assert {row["k_shot"] for row in rows} == {1, 2, 5, 10}
    assert {row["new_class_count"] for row in rows} == {5, 10, 20}


def test_stage2_t1_arm_ids_are_unique() -> None:
    ids = [arm.ablation_id for arm in PHASE2_T1_ARMS]
    assert len(ids) == len(set(ids))
    assert {
        "P2-FULL",
        "P2-A0",
        "P2-B0",
        "P2-C3",
        "P2-D0",
        "P2-D1",
        "P2-D2",
        "P2-E0",
        "P2-F0",
        "P2-F1",
        "P2-F2",
        "P2-F3",
    }.issubset(ids)
    full = next(arm for arm in PHASE2_T1_ARMS if arm.ablation_id == "P2-FULL")
    f3 = next(arm for arm in PHASE2_T1_ARMS if arm.ablation_id == "P2-F3")
    assert full.physical_config_id is None
    assert f3.physical_config_id == "P2-FULL"


def test_fresh_stage2_rejects_observed_or_aliased_seeds() -> None:
    arm = ArmSpec("P2-TEST", "stage2c", "M", "test", "P2-FULL")
    observed = _bundles(3)
    observed[0] = SeedBundle(713102, 820002, 820003)
    with pytest.raises(FullAblationSpecError, match="observed"):
        build_phase2_rows(
            stage="screening",
            arms=[arm],
            seed_bundles=observed,
            class_draw_seeds=[830001],
            git_commit="abcdef0123456789",
        )
    aliased = _bundles(3)
    aliased[0] = SeedBundle(820001, 820001, 820003)
    with pytest.raises(FullAblationSpecError, match="separately"):
        build_phase2_rows(
            stage="screening",
            arms=[arm],
            seed_bundles=aliased,
            class_draw_seeds=[830001],
            git_commit="abcdef0123456789",
        )


def test_worker_assignment_never_exceeds_eight_by_two() -> None:
    slots = assign_worker_slots(1000)
    assert {(slot.gpu, slot.slot) for slot in slots} == {
        (gpu, slot) for gpu in range(8) for slot in range(2)
    }
    assert all(0 <= slot.gpu < 8 and 0 <= slot.slot < 2 for slot in slots)


def test_artifact_validator_is_fail_closed() -> None:
    record = {field: f"value-{field}" for field in REQUIRED_RUN_ARTIFACT_FIELDS}
    record.update(
        {
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "support_physical_ids_hash": "support-hash",
            "query_physical_ids_hash": "query-hash",
        }
    )
    validate_artifact_record(record)
    missing = copy.deepcopy(record)
    missing.pop("predictions_hash")
    with pytest.raises(FullAblationSpecError, match="predictions_hash"):
        validate_artifact_record(missing)
    overlap = copy.deepcopy(record)
    overlap["query_physical_ids_hash"] = overlap["support_physical_ids_hash"]
    with pytest.raises(FullAblationSpecError, match="must differ"):
        validate_artifact_record(overlap)
