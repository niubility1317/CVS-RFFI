from __future__ import annotations

import inspect

import pytest

from cvsrffi import stage2_next_r2_matrix as matrix


RECEIVERS = tuple(f"rx-{index}" for index in range(7))
CLASSES = tuple(f"class-{index}" for index in range(6))
SOURCE_SHA = "7" * 64


def _plan():
    return matrix.build_next_r2_proxy24_plan(
        RECEIVERS, CLASSES, source_identity_sha256=SOURCE_SHA
    )


def test_source_only_hash_selection_is_deterministic_and_has_no_performance_input() -> None:
    first = matrix.select_source_receivers(
        RECEIVERS, source_identity_sha256=SOURCE_SHA
    )
    second = matrix.select_source_receivers(
        tuple(RECEIVERS), source_identity_sha256=SOURCE_SHA
    )
    assert first == second
    assert len(set(first)) == 2
    signature = inspect.signature(matrix.select_source_receivers)
    assert tuple(signature.parameters) == ("receiver_registry", "source_identity_sha256")


def test_proxy24_has_exact_outer_and_four_state_coverage() -> None:
    plan = _plan()
    matrix.validate_next_r2_proxy24_plan(plan)
    assert plan["outer_key_count"] == 24
    assert plan["state_prediction_count"] == 96
    assert tuple(plan["state_ids"]) == matrix.STATE_IDS
    assert tuple(plan["k_values"]) == (1, 5)
    assert len(plan["keys"]) == 24
    assert len({item["outer_key_id"] for item in plan["keys"]}) == 24
    assert {item["held_receiver"] for item in plan["keys"]} == set(
        plan["receiver_selection"]["selected_receivers"]
    )
    assert {item["held_class"] for item in plan["keys"]} == set(CLASSES)
    assert {item["active_k"] for item in plan["keys"]} == {1, 5}


def test_registration_and_query_counts_are_frozen() -> None:
    outer_key = matrix.outer_key_from_mapping(_plan()["keys"][0])
    for state_id in matrix.STATE_IDS:
        registered = matrix.registered_classes_for_state(outer_key, state_id)
        if state_id in matrix.REG1_STATES:
            assert registered == CLASSES
            assert matrix.query_count_for_state(outer_key, state_id) == 54
        else:
            assert len(registered) == 5
            assert outer_key.held_class not in registered
            assert matrix.query_count_for_state(outer_key, state_id) == 45


def test_plan_excludes_target_k10_cn20_and_truth_selected_receivers() -> None:
    plan = _plan()
    assert plan["target_receiver_count"] == 0
    assert plan["target_cn20"] is False
    assert plan["k10"] is False
    assert plan["truth_or_accuracy_receiver_selection"] is False
    assert plan["receiver_selection"]["performance_inputs"] == 0


def test_plan_tamper_fails_closed() -> None:
    plan = dict(_plan())
    plan["target_receiver_count"] = 1
    with pytest.raises(matrix.NextR2MatrixError):
        matrix.validate_next_r2_proxy24_plan(plan)


def test_receiver_selection_requires_exact_source_registry_and_sha() -> None:
    with pytest.raises(matrix.NextR2MatrixError):
        matrix.select_source_receivers(RECEIVERS[:-1], source_identity_sha256=SOURCE_SHA)
    with pytest.raises(matrix.NextR2MatrixError):
        matrix.select_source_receivers(RECEIVERS, source_identity_sha256="not-a-sha")
