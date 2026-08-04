from __future__ import annotations

import json

import pytest

from cvsrffi import stage2_d106_matrix_protocol as d106
from cvsrffi import stage2_next_r5_target5_plan as target5


def _sha(index: int) -> str:
    return f"{index:064x}"


def _valid_bindings(plan):
    bindings = []
    for index, surface in enumerate(plan["state_surfaces"]):
        scenario_index = index // len(target5.STATE_IDS)
        state = surface["state"]
        bindings.append(
            {
                "surface_id": surface["surface_id"],
                "effective_seed": 713103,
                "old_query_id_root_sha256": _sha(1_000 + scenario_index),
                "new_query_id_root_sha256": (
                    _sha(1_500 + scenario_index)
                    if state in {"DA0_REG1", "DA1_REG1"}
                    else "N/A"
                ),
                "fa_state_binding_sha256": (
                    _sha(2_000 + scenario_index)
                    if state in {"DA1_REG0", "DA1_REG1"}
                    else "N/A"
                ),
            }
        )
    return bindings


def test_target5_is_exact_d106_receiver_k5_q_only_four_state_plan() -> None:
    plan = target5.build_next_r5_target5_plan()
    assert tuple(plan["receivers"]) == d106.RECEIVERS
    assert plan["receivers"] == tuple(d106.RECEIVERS)
    assert plan["candidate_id"] == "NEXT-R5-K5-FA-RDCE3-Q"
    assert plan["k_shot"] == 5
    assert plan["new_class_count"] == 20
    assert tuple(plan["arm_ids"]) == ("Q",)
    assert tuple(plan["prohibited_components"]) == ("K1", "CER", "H", "D92-Lite", "K10")
    assert plan["job_count"] == 5
    assert plan["scenario_row_count"] == 15
    assert plan["state_surface_count"] == 60
    assert tuple(plan["state_ids"]) == target5.STATE_IDS
    assert plan["seed_policy"]["preferred_seed"] == 713103
    assert plan["seed_policy"]["allowed_seeds"] == (713103,)
    assert plan["seed_policy"]["fallback_allowed"] is False
    assert all(value is False for value in plan["negative_protocol_flags"].values())
    for state in ("DA0_REG0", "DA1_REG0"):
        assert plan["metric_availability_by_state"][state] == {
            "seen_new_acc": "N/A",
            "H_old_new": "N/A",
        }
    target5.validate_next_r5_target5_plan(plan)


def test_plan_is_canonical_and_drift_fails_closed() -> None:
    frozen = target5.build_next_r5_target5_plan()
    plan = json.loads(target5.canonical_bytes(frozen))
    assert plan["plan_receipt_sha256"] == target5.canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_receipt_sha256"}
    )
    plan["state_surfaces"][0]["arm_id"] = "H"
    with pytest.raises(target5.NextR5Target5PlanError):
        target5.validate_next_r5_target5_plan(plan)

    with pytest.raises(TypeError):
        frozen["state_surfaces"][0]["arm_id"] = "H"


def test_binding_receipt_requires_same_scene_query_root_and_da1_reuse() -> None:
    plan = target5.build_next_r5_target5_plan()
    bindings = _valid_bindings(plan)
    receipt = target5.build_next_r5_target5_binding_receipt(plan, bindings)
    assert receipt["binding_receipt_sha256"] == target5.canonical_sha256(
        {key: value for key, value in receipt.items() if key != "binding_receipt_sha256"}
    )
    target5.validate_next_r5_target5_binding_receipt(plan, receipt)

    query_drift = _valid_bindings(plan)
    query_drift[1]["old_query_id_root_sha256"] = _sha(9_999)
    with pytest.raises(target5.NextR5Target5PlanError, match="old-query-ID root reuse"):
        target5.build_next_r5_target5_binding_receipt(plan, query_drift)

    new_query_drift = _valid_bindings(plan)
    new_query_drift[3]["new_query_id_root_sha256"] = _sha(9_998)
    with pytest.raises(target5.NextR5Target5PlanError, match="new-query-ID root reuse"):
        target5.build_next_r5_target5_binding_receipt(plan, new_query_drift)

    fa_drift = _valid_bindings(plan)
    fa_drift[3]["fa_state_binding_sha256"] = _sha(8_888)
    with pytest.raises(target5.NextR5Target5PlanError, match="FA state reuse"):
        target5.build_next_r5_target5_binding_receipt(plan, fa_drift)


def test_binding_receipt_rejects_da0_binding_and_partial_coverage() -> None:
    plan = target5.build_next_r5_target5_plan()
    bindings = _valid_bindings(plan)
    bindings[0]["fa_state_binding_sha256"] = _sha(7_777)
    with pytest.raises(target5.NextR5Target5PlanError, match="DA0"):
        target5.build_next_r5_target5_binding_receipt(plan, bindings)
    with pytest.raises(target5.NextR5Target5PlanError, match="coverage"):
        target5.build_next_r5_target5_binding_receipt(plan, _valid_bindings(plan)[:-1])

    reg0_new_query = _valid_bindings(plan)
    reg0_new_query[0]["new_query_id_root_sha256"] = _sha(6_666)
    with pytest.raises(target5.NextR5Target5PlanError, match="REG0"):
        target5.build_next_r5_target5_binding_receipt(plan, reg0_new_query)

    wrong_seed = _valid_bindings(plan)
    wrong_seed[0]["effective_seed"] = 713104
    with pytest.raises(target5.NextR5Target5PlanError, match="effective seed"):
        target5.build_next_r5_target5_binding_receipt(plan, wrong_seed)


def test_binding_receipt_is_deeply_immutable() -> None:
    plan = target5.build_next_r5_target5_plan()
    receipt = target5.build_next_r5_target5_binding_receipt(plan, _valid_bindings(plan))
    with pytest.raises(TypeError):
        receipt["state_bindings"][0]["effective_seed"] = 713104
