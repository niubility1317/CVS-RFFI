from __future__ import annotations

import hashlib

import pytest

from cvsrffi import stage2_next_r1_matrix as matrix


def _plan():
    return matrix.build_next_r1_loco_plan(
        [f"rx{index}" for index in range(7)],
        [f"tx{index}" for index in range(6)],
    )


def _rows(plan):
    return [
        matrix.NextR1LocoRow(
            row_id=value["row_id"],
            held_receiver=value["held_receiver"],
            held_class=value["held_class"],
            active_k=value["active_k"],
            retained_classes=tuple(value["retained_classes"]),
            registered_classes=tuple(value["registered_classes"]),
            candidate_id=value["candidate_id"],
        )
        for value in plan["rows"]
    ]


def _binding_inputs(row_k1, row_k5):
    receivers = [f"rx{index}" for index in range(7)]
    classes = [f"tx{index}" for index in range(6)]
    cells = {
        (receiver, class_id): [
            f"p-{receiver}-{class_id}-{index}" for index in range(matrix.PHYSICAL_PER_CELL)
        ]
        for receiver in receivers
        for class_id in classes
    }
    held_receiver = row_k1.held_receiver
    held_class = row_k1.held_class

    def ordered(receiver, class_id):
        return sorted(
            cells[(receiver, class_id)],
            key=lambda physical_id: hashlib.sha256(
                f"next-r1|{receiver}|{class_id}|{physical_id}".encode()
            ).hexdigest(),
        )

    phase1_fit = [
        physical_id
        for receiver in receivers
        if receiver != held_receiver
        for class_id in classes
        if class_id != held_class
        for physical_id in ordered(receiver, class_id)
    ]
    support5 = {
        class_id: ordered(held_receiver, class_id)[:5]
        for class_id in row_k1.registered_classes
    }
    support1 = {class_id: values[:1] for class_id, values in support5.items()}
    query = {
        class_id: ordered(held_receiver, class_id)[5:]
        for class_id in row_k1.registered_classes
    }

    root = lambda values: hashlib.sha256("\n".join(values).encode()).hexdigest()
    support1_ordered = tuple(item for key in row_k1.registered_classes for item in support1[key])
    support5_ordered = tuple(item for key in row_k1.registered_classes for item in support5[key])
    query_ordered = tuple(item for key in row_k1.registered_classes for item in query[key])
    receipt = {
        "held_receiver": held_receiver,
        "held_class": held_class,
        "phase1_fit_count": len(phase1_fit),
        "phase1_fit_physical_root_sha256": root(phase1_fit),
        "support_k1_count": len(support1_ordered),
        "support_k1_physical_root_sha256": root(support1_ordered),
        "support_k5_count": len(support5_ordered),
        "support_k5_physical_root_sha256": root(support5_ordered),
        "outer_query_count": len(query_ordered),
        "outer_query_physical_root_sha256": root(query_ordered),
        "k1_is_k5_prefix": True,
    }
    return receipt, phase1_fit, support1, support5, query


def test_next_r1_is_one_candidate_84_rows_with_six_arms() -> None:
    plan = _plan()
    assert plan["candidate_ids"] == ["NEXT-R1"]
    assert plan["candidate_id"] == "NEXT-R1"
    assert plan["method_lock"] == "NEXT-R1 FABR-TSL"
    assert plan["fold_count"] == 42
    assert plan["row_count"] == 84
    assert plan["row_count_per_candidate"] == 84
    assert plan["arm_ids"] == list(matrix.ARM_IDS)
    assert plan["common_arm_ids"] == ["R0Q", "R0F", "R0L"]
    assert plan["adapted_arm_ids"] == ["R1Q", "R1F", "R1L"]
    assert len(plan["rows"]) == 84
    assert all(row["candidate_id"] == "NEXT-R1" for row in plan["rows"])
    assert all(row["arm_ids"] == list(matrix.ARM_IDS) for row in plan["rows"])
    assert len({(row["held_receiver"], row["held_class"], row["active_k"]) for row in plan["rows"]}) == 84
    assert all(len(row["retained_classes"]) == 5 for row in plan["rows"])
    assert all(row["registered_classes"][-1] == row["held_class"] for row in plan["rows"])


def test_plan_order_invariant_and_digest_is_immutable_contract() -> None:
    forward = _plan()
    reverse = matrix.build_next_r1_matrix(
        list(reversed([f"rx{index}" for index in range(7)])),
        list(reversed([f"tx{index}" for index in range(6)])),
    )
    assert forward["matrix_sha256"] == reverse["matrix_sha256"]
    assert forward["rows"] == reverse["rows"]
    assert matrix.validate_next_r1_plan(forward)["matrix_sha256"] == forward["matrix_sha256"]
    tampered = dict(forward)
    tampered["rows"] = list(forward["rows"])
    tampered["rows"][0] = dict(tampered["rows"][0], active_k=5)
    with pytest.raises(matrix.NextR1MatrixError, match="SHA256 drift"):
        matrix.validate_next_r1_plan(tampered)


def test_forbidden_oracle_assignment_fields_are_rejected() -> None:
    plan = _plan()
    tampered = dict(plan)
    tampered.pop("matrix_sha256")
    tampered["truth_loaded"] = False
    with pytest.raises(matrix.NextR1MatrixError, match="forbidden matrix field"):
        matrix.validate_next_r1_plan(
            {**tampered, "matrix_sha256": matrix.canonical_sha256(tampered)}
        )


def test_binding_proves_exact_prefix_and_three_way_disjointness() -> None:
    plan = _plan()
    row_pair = _rows(plan)[:2]
    assert [row.active_k for row in row_pair] == [1, 5]
    receipt, phase1_fit, support1, support5, query = _binding_inputs(*row_pair)
    binding = matrix.bind_next_r1_physical_ids(
        row_k1=row_pair[0],
        row_k5=row_pair[1],
        loco_fold_receipt=receipt,
        phase1_fit_ids=phase1_fit,
        k1_support_ids_by_class=support1,
        k5_support_ids_by_class=support5,
        query_ids_by_class=query,
    )
    assert binding["candidate_id"] == "NEXT-R1"
    assert binding["k1_is_exact_k5_prefix"] is True
    assert binding["support_query_phase1_physical_ids_disjoint"] is True
    assert binding["phase1_fit_count"] == 420
    assert binding["support_k1_count"] == 6
    assert binding["support_k5_count"] == 30
    assert binding["query_count"] == 54
    assert len(binding["phase1_seal_sha256"]) == 64
    assert matrix.validate_next_r1_binding(binding)["binding_sha256"] == binding["binding_sha256"]


def test_prefix_and_overlap_fail_closed() -> None:
    row_pair = _rows(_plan())[:2]
    receipt, phase1_fit, support1, support5, query = _binding_inputs(*row_pair)
    bad_prefix = {class_id: [values[1]] for class_id, values in support5.items()}
    with pytest.raises(matrix.NextR1MatrixError, match="exact K5 prefix"):
        matrix.bind_next_r1_physical_ids(
            row_k1=row_pair[0], row_k5=row_pair[1], loco_fold_receipt=receipt,
            phase1_fit_ids=phase1_fit, k1_support_ids_by_class=bad_prefix,
            k5_support_ids_by_class=support5, query_ids_by_class=query,
        )
    query[row_pair[0].registered_classes[0]][0] = support5[row_pair[0].registered_classes[0]][2]
    with pytest.raises(matrix.NextR1MatrixError, match="support/query"):
        matrix.bind_next_r1_physical_ids(
            row_k1=row_pair[0], row_k5=row_pair[1], loco_fold_receipt=receipt,
            phase1_fit_ids=phase1_fit, k1_support_ids_by_class=support1,
            k5_support_ids_by_class=support5, query_ids_by_class=query,
        )


def test_binding_digest_and_row_pair_drift_fail_closed() -> None:
    row_pair = _rows(_plan())[:2]
    receipt, phase1_fit, support1, support5, query = _binding_inputs(*row_pair)
    binding = matrix.bind_next_r1_physical_ids(
        row_k1=row_pair[0], row_k5=row_pair[1], loco_fold_receipt=receipt,
        phase1_fit_ids=phase1_fit, k1_support_ids_by_class=support1,
        k5_support_ids_by_class=support5, query_ids_by_class=query,
    )
    tampered = dict(binding)
    tampered["query_count"] = 53
    with pytest.raises(matrix.NextR1MatrixError, match="SHA256 drift"):
        matrix.validate_next_r1_binding(tampered)
    with pytest.raises(matrix.NextR1MatrixError, match="K1/K5 row pairing"):
        matrix.bind_next_r1_physical_ids(
            row_k1=row_pair[0], row_k5=_rows(_plan())[2], loco_fold_receipt=receipt,
            phase1_fit_ids=phase1_fit, k1_support_ids_by_class=support1,
            k5_support_ids_by_class=support5, query_ids_by_class=query,
        )


@pytest.mark.parametrize(
    "receivers,classes",
    [
        (["rx"] * 7, [f"tx{i}" for i in range(6)]),
        ([f"rx{i}" for i in range(7)], ["tx"] * 6),
    ],
)
def test_registry_drift_fails_closed(receivers, classes) -> None:
    with pytest.raises(matrix.NextR1MatrixError):
        matrix.build_next_r1_plan(receivers, classes)
