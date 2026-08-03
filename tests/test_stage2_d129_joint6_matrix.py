from __future__ import annotations

import hashlib

import pytest

from cvsrffi import stage2_d129_joint6_da as da
from cvsrffi import stage2_d129_joint6_matrix as matrix


def _plan():
    return matrix.build_joint6_loco_plan(
        [f"rx{index}" for index in range(7)],
        [f"tx{index}" for index in range(6)],
    )


def _binding_inputs(rows):
    receivers = [f"rx{index}" for index in range(7)]
    classes = [f"tx{index}" for index in range(6)]
    cells = {
        (receiver, class_id): [
            f"p-{receiver}-{class_id}-{index}" for index in range(14)
        ]
        for receiver in receivers
        for class_id in classes
    }

    def ordered(receiver, class_id):
        return sorted(
            cells[(receiver, class_id)],
            key=lambda physical_id: hashlib.sha256(
                f"{da.LOCO_SALT}|{receiver}|{class_id}|{physical_id}".encode()
            ).hexdigest(),
        )

    records = [
        da.D129LOCORecord(receiver, class_id, physical_id)
        for (receiver, class_id), physical_ids in cells.items()
        for physical_id in physical_ids
    ]
    loco = da.build_d129_loco_plan(records)
    fold = next(
        value
        for value in loco.folds
        if value.held_receiver == rows[0].held_receiver
        and value.held_class == rows[0].held_class
    )
    registry = rows[0].registered_classes
    phase1_fit = [
        physical_id
        for receiver in receivers
        if receiver != rows[0].held_receiver
        for class_id in classes
        if class_id != rows[0].held_class
        for physical_id in ordered(receiver, class_id)
    ]
    support5 = {
        class_id: ordered(rows[0].held_receiver, class_id)[:5]
        for class_id in registry
    }
    support1 = {class_id: values[:1] for class_id, values in support5.items()}
    query = {
        class_id: ordered(rows[0].held_receiver, class_id)[5:]
        for class_id in registry
    }
    return fold.as_dict(), phase1_fit, support1, support5, query


def test_complete_receiver_class_loco_matrix_is_frozen_without_truth() -> None:
    plan = _plan()
    assert plan["fold_count"] == 42
    assert plan["row_count_per_candidate"] == 84
    assert len(plan["rows"]) == 84
    assert plan["candidate_ids"] == ["CSPAR-2", "SRDH-2"]
    assert plan["arm_ids"] == list(matrix.ARM_IDS)
    assert plan["truth_loaded"] is False
    assert plan["query_rows_used_for_fit"] == 0
    assert plan["query_state_updates"] == 0
    assert plan["query_selection_count"] == 0
    keys = {
        (row["held_receiver"], row["held_class"], row["active_k"])
        for row in plan["rows"]
    }
    assert len(keys) == 84
    assert plan["formal_new_registration_claim"] is False
    assert all(len(row["retained_classes"]) == 5 for row in plan["rows"])
    assert all(
        row["held_proxy_classes"] == [row["held_class"]]
        for row in plan["rows"]
    )
    assert all(row["registered_classes"][-1] == row["held_class"] for row in plan["rows"])


def test_plan_order_is_invariant_to_input_order() -> None:
    forward = _plan()
    reverse = matrix.build_joint6_loco_plan(
        list(reversed([f"rx{index}" for index in range(7)])),
        list(reversed([f"tx{index}" for index in range(6)])),
    )
    assert forward["matrix_sha256"] == reverse["matrix_sha256"]
    assert forward["rows"] == reverse["rows"]


def test_physical_binding_proves_prefix_and_disjointness() -> None:
    plan = _plan()
    first = plan["rows"][:2]
    rows = [
        matrix.Joint6LocoRow(
            row_id=value["row_id"],
            held_receiver=value["held_receiver"],
            held_class=value["held_class"],
            active_k=value["active_k"],
            retained_classes=tuple(value["retained_classes"]),
            registered_classes=tuple(value["registered_classes"]),
        )
        for value in first
    ]
    assert [row.active_k for row in rows] == [1, 5]
    fold, phase1_fit, support1, support5, query = _binding_inputs(rows)
    receipt = matrix.bind_joint6_physical_ids(
        row_k1=rows[0],
        row_k5=rows[1],
        loco_fold_receipt=fold,
        phase1_fit_ids=phase1_fit,
        k1_support_ids_by_class=support1,
        k5_support_ids_by_class=support5,
        query_ids_by_class=query,
    )
    assert receipt["k1_is_exact_k5_prefix"] is True
    assert receipt["support_query_physical_ids_disjoint"] is True
    assert receipt["k1_support_count"] == 6
    assert receipt["k5_support_count"] == 30
    assert receipt["query_count"] == 54
    assert receipt["phase1_fit_count"] == 420
    assert len(receipt["phase1_seal_sha256"]) == 64
    assert receipt["data_revalidated"] is False
    assert matrix.validate_joint6_binding(receipt)["binding_sha256"] == receipt[
        "binding_sha256"
    ]
    tampered = dict(receipt)
    tampered["phase1_fit_count"] = 419
    with pytest.raises(matrix.D129Joint6MatrixError, match="SHA256 drift"):
        matrix.validate_joint6_binding(tampered)


def test_prefix_and_overlap_fail_closed() -> None:
    plan = _plan()
    first = plan["rows"][:2]
    rows = [
        matrix.Joint6LocoRow(
            row_id=value["row_id"],
            held_receiver=value["held_receiver"],
            held_class=value["held_class"],
            active_k=value["active_k"],
            retained_classes=tuple(value["retained_classes"]),
            registered_classes=tuple(value["registered_classes"]),
        )
        for value in first
    ]
    classes = rows[0].registered_classes
    fold, phase1_fit, _support1, support5, query = _binding_inputs(rows)
    bad_prefix = {class_id: [values[1]] for class_id, values in support5.items()}
    with pytest.raises(matrix.D129Joint6MatrixError, match="exact K5 prefix"):
        matrix.bind_joint6_physical_ids(
            row_k1=rows[0],
            row_k5=rows[1],
            loco_fold_receipt=fold,
            phase1_fit_ids=phase1_fit,
            k1_support_ids_by_class=bad_prefix,
            k5_support_ids_by_class=support5,
            query_ids_by_class=query,
        )
    good_prefix = {class_id: values[:1] for class_id, values in support5.items()}
    query[classes[0]][0] = support5[classes[0]][2]
    with pytest.raises(matrix.D129Joint6MatrixError, match="support/query"):
        matrix.bind_joint6_physical_ids(
            row_k1=rows[0],
            row_k5=rows[1],
            loco_fold_receipt=fold,
            phase1_fit_ids=phase1_fit,
            k1_support_ids_by_class=good_prefix,
            k5_support_ids_by_class=support5,
            query_ids_by_class=query,
        )


@pytest.mark.parametrize(
    "receivers,classes",
    [(["rx"] * 7, [f"tx{i}" for i in range(6)]), ([f"rx{i}" for i in range(7)], ["tx"] * 6)],
)
def test_matrix_registry_drift_fails_closed(receivers, classes) -> None:
    with pytest.raises(matrix.D129Joint6MatrixError):
        matrix.build_joint6_loco_plan(receivers, classes)
