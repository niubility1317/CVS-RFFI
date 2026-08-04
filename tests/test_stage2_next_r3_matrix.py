from __future__ import annotations

import hashlib

import pytest

from cvsrffi import stage2_next_r3_matrix as matrix


CLASSES = tuple(f"tx-{index}" for index in range(6))


def test_frozen_receivers_and_explicit_six_class_proxy_matrix() -> None:
    plan = matrix.build_next_r3_proxy24_plan(CLASSES)
    assert tuple(plan["held_receivers"]) == ("1-1", "18-2")
    assert tuple(plan["held_classes"]) == CLASSES
    assert plan["candidate_count"] == 1
    assert plan["row_count"] == 24
    assert plan["state_prediction_count"] == 96
    assert plan["arm_prediction_count"] == 576
    assert tuple(plan["state_ids"]) == matrix.STATE_IDS
    assert tuple(plan["arm_ids"]) == matrix.ARM_IDS
    assert all(set(row["registrations"]) == {"REG0", "REG1"} for row in plan["rows"])
    assert all(row["formal_new_registration_claim"] is False for row in plan["rows"])
    matrix.validate_next_r3_proxy24_plan(plan)


def test_d129_style_pair_call_and_receiver_override_fail_closed() -> None:
    plan = matrix.build_next_r3_proxy24_plan(matrix.HELD_RECEIVERS, CLASSES)
    assert plan["row_count"] == 24
    with pytest.raises(matrix.NextR3MatrixError):
        matrix.build_next_r3_proxy24_plan(
            CLASSES, held_receivers=("18-2", "1-1")
        )
    with pytest.raises(matrix.NextR3MatrixError):
        matrix.build_next_r3_proxy24_plan(CLASSES[:-1])


def test_physical_binding_checks_prefix_and_disjointness() -> None:
    plan = matrix.build_next_r3_proxy24_plan(CLASSES)
    rows = [matrix.outer_key_from_mapping(value) for value in plan["rows"][:2]]
    assert [row.active_k for row in rows] == [1, 5]
    support5 = {
        cls: [f"s-{cls}-{index}" for index in range(5)] for cls in CLASSES
    }
    support1 = {cls: values[:1] for cls, values in support5.items()}
    query = {cls: [f"q-{cls}-{index}" for index in range(9)] for cls in CLASSES}
    phase1 = [f"p-{index}" for index in range(matrix.PHASE1_FIT_COUNT)]
    receipt = matrix.bind_next_r3_physical_ids(
        row_k1=rows[0],
        row_k5=rows[1],
        loco_fold_receipt={
            "held_receiver": rows[0].held_receiver,
            "held_class": rows[0].held_class,
            "phase1_fit_count": matrix.PHASE1_FIT_COUNT,
            "phase1_fit_physical_root_sha256": hashlib.sha256(
                "\n".join(phase1).encode()
            ).hexdigest(),
        },
        phase1_fit_ids=phase1,
        k1_support_ids_by_class=support1,
        k5_support_ids_by_class=support5,
        query_ids_by_class=query,
    )
    assert receipt["k1_is_exact_k5_prefix"] is True
    assert receipt["support_query_physical_ids_disjoint"] is True
    assert matrix.validate_next_r3_binding(receipt)["binding_sha256"] == receipt[
        "binding_sha256"
    ]
    with pytest.raises(matrix.NextR3MatrixError, match="prefix"):
        matrix.bind_next_r3_physical_ids(
            row_k1=rows[0],
            row_k5=rows[1],
            loco_fold_receipt={},
            phase1_fit_ids=phase1,
            k1_support_ids_by_class={cls: [values[1]] for cls, values in support5.items()},
            k5_support_ids_by_class=support5,
            query_ids_by_class=query,
        )


def test_plan_digest_is_immutable() -> None:
    plan = dict(matrix.build_next_r3_proxy24_plan(CLASSES))
    plan["row_count"] = 23
    with pytest.raises(matrix.NextR3MatrixError):
        matrix.validate_next_r3_proxy24_plan(plan)
