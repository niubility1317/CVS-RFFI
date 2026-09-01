from __future__ import annotations

import pytest

from cvsrffi.stage2_marc_ot_pilot import (
    FORMAL_ARMS,
    SCENARIOS,
    normalize_formal_arms,
    run_support_then_query,
    validate_mrior_controls,
    validate_pilot_config,
)


def _mrior_controls():
    return {
        "MRIOR-H": {
            "permission_scope": "TARGET_SUPPORT_ONLY_HEAD_CONTROL_DIAGNOSTIC_NON_FORMAL",
            "claim_scope": "MECHANISM_CONTROL_ONLY",
        },
        "MRIOR-B": {
            "permission_scope": "P2_MIN_V1_TARGET_SUPPORT_ONLY_BACKBONE_CONTROL",
            "claim_scope": "MATCHED_PERMISSION_CONTROL",
        },
        "MRIOR-HB": {
            "permission_scope": "TARGET_SUPPORT_ONLY_HEAD_BACKBONE_CONTROL_DIAGNOSTIC_NON_FORMAL",
            "claim_scope": "MECHANISM_CONTROL_ONLY",
        },
    }


def _config():
    return {
        "schema": "cvs.phase2.marc_ot.pilot_config.v1",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule",
        "split_id": "split",
        "pilot_outer_key": "outer",
        "checkpoint_id": "ADV3B02_CORE90_SOFT_E200",
        "receiver": "3-19",
        "seed": 713102,
        "k_shot": 10,
        "software_supported_k": [1, 2, 5, 10, 20],
        "pilot_k": 10,
        "pilot_executed": False,
        "training_coverage_k": [],
        "scenarios": list(SCENARIOS),
        "arms": list(FORMAL_ARMS),
        "fold_count": 5,
        "stage_steps": [2, 2, 2, 2],
        "learning_rate_bounds": {"min": 1e-5, "max": 3e-4},
        "ot": {"epsilon": 0.1, "iterations": 80},
        "supcon": {"temperature": 0.07, "weight": 0.1},
        "ratio_cap": 0.5,
        "interpolation_grid": [1.0, 0.75, 0.5, 0.25, 0.0],
        "promotion_gates": {
            "median_p3_ba_delta_pp": 3.0,
            "worst_scene_p3_ba_delta_pp": -0.5,
            "median_p3_floor_delta_pp": 0.0,
            "low_elev_p3_floor_delta_pp": 0.0,
            "max_p1_p2_scene_drop_pp": 2.0,
            "minimum_help_gt_harm_scenes": 2,
        },
        "mrior_controls": _mrior_controls(),
    }


def test_formal_arm_registry_is_exact_and_not_expandable() -> None:
    assert FORMAL_ARMS == ("R0", "R1", "R2", "R4", "R6", "R8")
    assert normalize_formal_arms(FORMAL_ARMS) == FORMAL_ARMS
    with pytest.raises(ValueError, match="exactly"):
        normalize_formal_arms(("R0", "R1"))
    with pytest.raises(ValueError, match="exactly"):
        normalize_formal_arms((*FORMAL_ARMS, "R10"))


def test_mrior_controls_require_permission_scope_and_reject_historical_backfill() -> None:
    validate_mrior_controls(_mrior_controls())
    missing = _mrior_controls()
    del missing["MRIOR-H"]["permission_scope"]
    with pytest.raises(ValueError, match="permission_scope"):
        validate_mrior_controls(missing)
    historical = _mrior_controls()
    historical["MRIOR-H"]["historical_accuracy"] = 0.91
    with pytest.raises(ValueError, match="historical numerical"):
        validate_mrior_controls(historical)
    backfill = _mrior_controls()
    backfill["MRIOR-HB"]["mrior_sda_result"] = {"ba": 0.9}
    with pytest.raises(ValueError, match="MRIOR-SDA"):
        validate_mrior_controls(backfill)


@pytest.mark.parametrize("injection", ("MRIOR-SDA", "history", "backfill"))
def test_mrior_permission_scope_rejects_historical_string_injection(injection) -> None:
    controls = _mrior_controls()
    controls["MRIOR-B"]["permission_scope"] += f"_{injection}"
    with pytest.raises(ValueError, match="permission_scope|historical|backfill|MRIOR-SDA"):
        validate_mrior_controls(controls)


def test_config_reuses_validated_once_handles_without_matrix_expansion() -> None:
    validated = validate_pilot_config(_config())
    assert validated["protocol_schema"] == "p2_min_v1"
    assert validated["phase2_data_status"] == "VALIDATED_ONCE"
    assert tuple(validated["arms"]) == FORMAL_ARMS
    assert tuple(validated["scenarios"]) == SCENARIOS
    assert validated["interpolation_grid"][-1] == 0.0
    invalid = _config()
    invalid["phase2_data_status"] = "REVALIDATE"
    with pytest.raises(ValueError, match="VALIDATED_ONCE"):
        validate_pilot_config(invalid)


def test_all_support_states_are_written_before_first_query_load() -> None:
    events: list[str] = []

    def support_loader(scene):
        events.append(f"support-load:{scene}")
        return {"scene": scene}

    def adapt(scene, arm, _support):
        events.append(f"adapt:{scene}:{arm}")
        return {"scene": scene, "arm": arm}

    def write_frozen(scene, arm, _state):
        events.append(f"SUPPORT_STATE_FROZEN:{scene}:{arm}")

    def query_loader(scene):
        assert sum(event.startswith("SUPPORT_STATE_FROZEN:") for event in events) == (
            len(SCENARIOS) * len(FORMAL_ARMS)
        )
        events.append("QUERY_LOAD")
        return {"scene": scene}

    result = run_support_then_query(
        scenarios=SCENARIOS,
        arms=FORMAL_ARMS,
        support_loader=support_loader,
        adapt_and_freeze=adapt,
        support_state_writer=write_frozen,
        query_loader=query_loader,
        predict_and_write=lambda scene, arm, _support, _query, _state: events.append(
            f"predict:{scene}:{arm}"
        ),
    )

    first_query = events.index("QUERY_LOAD")
    assert sum(event.startswith("SUPPORT_STATE_FROZEN:") for event in events[:first_query]) == 18
    assert result["support_frozen_unit_count"] == 18
    assert result["prediction_unit_count"] == 18
    assert result["truth_opened"] is False
