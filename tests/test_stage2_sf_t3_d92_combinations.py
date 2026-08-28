from __future__ import annotations

import pytest

from cvsrffi.stage2_sf_t3_d92_combinations import (
    CANDIDATES,
    FORMAL_SCENES,
    NEW_CLASS_COUNTS,
    build_candidate_config,
    build_experiment_rows,
    validate_combo_plan,
)


def _plan() -> dict:
    scenes = {}
    for gpu, scene in enumerate(FORMAL_SCENES):
        scenes[scene] = {
            "gpu": gpu,
            "split_id": f"split-{scene}",
            "old_support": f"/input/{scene}/old_support.npz",
            "registered_support_pattern": f"/input/{scene}/new{{new_count}}/registered_support.npz",
            "query_pattern": f"/input/{scene}/new{{new_count}}/query.npz",
            "data_handle_pattern": f"/input/{scene}/new{{new_count}}/data_handle.json",
        }
    return {
        "schema": "cvs.sf_t3_d92.combo_plan.v1",
        "run_id": "combo-run",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule",
        "base_checkpoint_path": "/checkpoint.pth",
        "phase1_bundle": {"package_root": "/phase1"},
        "old_class_count": 6,
        "k_shot": 10,
        "d92_method_lock": "D92-E0-NORF32",
        "rf32_used": False,
        "new_class_counts": list(NEW_CLASS_COUNTS),
        "candidates": list(CANDIDATES),
        "scenes": scenes,
    }


def test_combo_plan_expands_exactly_three_by_three_by_three() -> None:
    plan = validate_combo_plan(_plan())
    rows = build_experiment_rows(plan)

    assert len(rows) == 27
    assert {row["candidate_id"] for row in rows} == set(CANDIDATES)
    assert {row["scenario"] for row in rows} == set(FORMAL_SCENES)
    assert {row["new_class_count"] for row in rows} == set(NEW_CLASS_COUNTS)
    assert len({row["row_id"] for row in rows}) == 27
    assert all(row["four_states"] == [
        "DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1"
    ] for row in rows)
    assert all(row["d92_method_lock"] == "D92-E0-NORF32" for row in rows)
    assert all(row["rf32_used"] is False and row["query_fit_access"] is False for row in rows)


def test_combo_plan_rejects_protocol_or_matrix_drift() -> None:
    wrong_protocol = _plan()
    wrong_protocol["protocol_schema"] = "legacy"
    with pytest.raises(ValueError, match="protocol"):
        validate_combo_plan(wrong_protocol)

    wrong_counts = _plan()
    wrong_counts["new_class_counts"] = [1, 2, 10, 20]
    with pytest.raises(ValueError, match="new-class"):
        validate_combo_plan(wrong_counts)

    wrong_d92 = _plan()
    wrong_d92["rf32_used"] = True
    with pytest.raises(ValueError, match="D92"):
        validate_combo_plan(wrong_d92)


def test_candidate_configs_lock_d0_s02_and_r3_training_routes() -> None:
    d0 = build_candidate_config("D0_T3_D92")
    s02 = build_candidate_config("S02_T3_D92")
    r3 = build_candidate_config("R3_DUALDELTA_T3_D92_INLOOP")

    assert d0.phase_steps == (300, 150, 70)
    assert d0.norm_rules == (("t3", "weight_bias"),)
    assert d0.cache_storage_dtype == "float32"
    assert d0.checkpoint_average_top_k == 1

    assert s02.phase_steps == (4500, 0, 0)
    assert s02.norm_rules == (("t3", "weight_bias"),)
    assert s02.checkpoint_average_top_k == 3
    assert s02.validation_steps == ()

    assert r3.phase_steps == (300, 150, 70)
    assert r3.norm_rules == (("t3", "weight_bias"),)
    assert r3.checkpoint_average_top_k == 1
