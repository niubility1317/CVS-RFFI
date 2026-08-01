from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from cvsrffi.stage2_d106_k_conditioned_router import (
    ARMS,
    CANDIDATE_ID,
    D106KConditionedRouterError,
    G1_ROW_KEYS,
    G1_ROW_SCHEMA,
    ROUTE_BY_K,
    TARGET25_ROW_KEYS,
    TARGET25_ROW_SCHEMA,
    route_d106_k_conditioned_prediction,
)


def _sha(character: str) -> str:
    return character * 64


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sign(row: dict[str, Any]) -> dict[str, Any]:
    row.pop("prediction_receipt_sha256", None)
    row["prediction_receipt_sha256"] = _canonical_sha256(row)
    return row


def _common(active_k: int) -> dict[str, Any]:
    return {
        "K": active_k,
        "arm_predictions": {
            "M0": ["old_a", "new_b", "old_a"],
            "M_DA": ["new_b", "new_b", "old_a"],
            "M_HEAD": ["old_a", "old_a", "new_b"],
            "M_JOINT": ["new_b", "old_a", "new_b"],
        },
        "query_physical_ids": ["q2", "q0", "q1"],
        "registered_classes": ["old_a", "new_b"],
        "shared_component_receipts": {
            "rdce_state_sha256": _sha("b"),
            "rcmr_state_sha256": _sha("c"),
        },
        "query_state_updates": 0,
        "query_truth_access": False,
    }


def _g1_row(active_k: int = 1) -> dict[str, Any]:
    return _sign(
        {
            **_common(active_k),
            "schema": G1_ROW_SCHEMA,
            "held_receiver": "source-rx",
            "held_class": None,
            "package_id": "source-held-package",
            "formal_p2_authority": False,
            "target_access": False,
        }
    )


def _target_row(active_k: int = 1) -> dict[str, Any]:
    return _sign(
        {
            **_common(active_k),
            "schema": TARGET25_ROW_SCHEMA,
            "row_id": "target25-row",
            "receiver": "20-1",
            "scene": "leo_rain_weak",
            "query_role_access": False,
            "query_selection": False,
        }
    )


@pytest.mark.parametrize("builder", [_g1_row, _target_row])
@pytest.mark.parametrize(
    ("active_k", "expected_arm"),
    [(1, "M_DA"), (5, "M0"), (10, "M_HEAD")],
)
def test_exact_schema_and_k_route_preserve_same_row_bindings(
    builder, active_k: int, expected_arm: str
) -> None:
    row = builder(active_k)
    original = copy.deepcopy(row)

    result = route_d106_k_conditioned_prediction(
        active_k=active_k, row_prediction=row
    )

    assert CANDIDATE_ID == "D106-KCR/r1"
    assert dict(ROUTE_BY_K) == {1: "M_DA", 5: "M0", 10: "M_HEAD"}
    assert result.selected_arm == expected_arm
    assert result.predictions == tuple(original["arm_predictions"][expected_arm])
    assert result.query_order == tuple(original["query_physical_ids"])
    assert result.registered_classes == tuple(original["registered_classes"])
    assert result.source_prediction_receipt_sha256 == original[
        "prediction_receipt_sha256"
    ]
    assert dict(result.shared_component_receipts) == original[
        "shared_component_receipts"
    ]
    assert result.query_state_updates == 0
    assert row == original


def test_schema_key_sets_are_exact_and_distinct() -> None:
    assert set(_g1_row()) == G1_ROW_KEYS
    assert set(_target_row()) == TARGET25_ROW_KEYS
    assert "held_receiver" in G1_ROW_KEYS and "receiver" not in G1_ROW_KEYS
    assert {"row_id", "receiver", "scene"}.issubset(TARGET25_ROW_KEYS)
    assert "held_receiver" not in TARGET25_ROW_KEYS


@pytest.mark.parametrize("active_k", [True, 0, 2, 1.0, "1"])
def test_rejects_non_frozen_k(active_k: object) -> None:
    with pytest.raises(D106KConditionedRouterError, match="active_k"):
        route_d106_k_conditioned_prediction(
            active_k=active_k, row_prediction=_g1_row()
        )


def test_rejects_row_k_binding_drift() -> None:
    with pytest.raises(D106KConditionedRouterError, match="K binding"):
        route_d106_k_conditioned_prediction(
            active_k=5, row_prediction=_g1_row(1)
        )


@pytest.mark.parametrize(
    "extra_key",
    [
        "truth",
        "query_truth",
        "query_truth_labels",
        "metric",
        "metrics",
        "score",
        "scores",
        "metadata",
        "receiver",
        "scene",
    ],
)
def test_g1_schema_rejects_every_extra_key_even_with_valid_receipt(
    extra_key: str,
) -> None:
    row = _g1_row()
    row[extra_key] = {"nested": {"truth": ["old_a"]}}
    _sign(row)
    with pytest.raises(D106KConditionedRouterError, match="field closure"):
        route_d106_k_conditioned_prediction(active_k=1, row_prediction=row)


def test_schema_closure_rejects_missing_key_and_unknown_schema() -> None:
    row = _g1_row()
    del row["held_class"]
    _sign(row)
    with pytest.raises(D106KConditionedRouterError, match="field closure"):
        route_d106_k_conditioned_prediction(active_k=1, row_prediction=row)

    row = _g1_row()
    row["schema"] = "cvs.phase2.d106.unknown.v1"
    _sign(row)
    with pytest.raises(D106KConditionedRouterError, match="unsupported"):
        route_d106_k_conditioned_prediction(active_k=1, row_prediction=row)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_requires_exact_complete_four_arm_closure(mutation: str) -> None:
    row = _g1_row()
    arms = row["arm_predictions"]
    if mutation == "missing":
        del arms["M_JOINT"]
    else:
        arms["M_OTHER"] = ["old_a", "old_a", "old_a"]
    _sign(row)

    with pytest.raises(D106KConditionedRouterError, match="four-arm closure"):
        route_d106_k_conditioned_prediction(active_k=1, row_prediction=row)


def test_rejects_query_order_registry_or_prediction_drift() -> None:
    row = _g1_row(10)
    row["arm_predictions"]["M_HEAD"] = ["old_a", "new_b"]
    _sign(row)
    with pytest.raises(D106KConditionedRouterError, match="query order"):
        route_d106_k_conditioned_prediction(active_k=10, row_prediction=row)

    row = _g1_row(5)
    row["arm_predictions"]["M0"][1] = "unregistered"
    _sign(row)
    with pytest.raises(D106KConditionedRouterError, match="registry"):
        route_d106_k_conditioned_prediction(active_k=5, row_prediction=row)

    row = _g1_row()
    row["query_physical_ids"] = ["q0", "q0", "q1"]
    _sign(row)
    with pytest.raises(D106KConditionedRouterError, match="unique IDs"):
        route_d106_k_conditioned_prediction(active_k=1, row_prediction=row)

    row = _g1_row()
    row["registered_classes"] = ["old_a", "old_a"]
    _sign(row)
    with pytest.raises(D106KConditionedRouterError, match="must be unique"):
        route_d106_k_conditioned_prediction(active_k=1, row_prediction=row)


@pytest.mark.parametrize("value", [True, 1, "false"])
def test_both_schemas_require_query_truth_access_exact_false(value: object) -> None:
    for builder in (_g1_row, _target_row):
        row = builder()
        row["query_truth_access"] = value
        _sign(row)
        with pytest.raises(D106KConditionedRouterError, match="query_truth_access"):
            route_d106_k_conditioned_prediction(active_k=1, row_prediction=row)


@pytest.mark.parametrize("field", ["query_role_access", "query_selection"])
@pytest.mark.parametrize("value", [True, 0, "false"])
def test_target_schema_requires_role_and_selection_exact_false(
    field: str, value: object
) -> None:
    row = _target_row()
    row[field] = value
    _sign(row)
    with pytest.raises(D106KConditionedRouterError, match="role/selection"):
        route_d106_k_conditioned_prediction(active_k=1, row_prediction=row)


def test_receipt_types_and_zero_update_fail_closed() -> None:
    row = _g1_row()
    row["prediction_receipt_sha256"] = "bad"
    with pytest.raises(D106KConditionedRouterError, match="lowercase SHA256"):
        route_d106_k_conditioned_prediction(active_k=1, row_prediction=row)

    row = _g1_row()
    row["shared_component_receipts"]["rdce_state_sha256"] = "F" * 64
    _sign(row)
    with pytest.raises(D106KConditionedRouterError, match="lowercase SHA256"):
        route_d106_k_conditioned_prediction(active_k=1, row_prediction=row)

    for value in (1, False):
        row = _g1_row()
        row["query_state_updates"] = value
        _sign(row)
        with pytest.raises(D106KConditionedRouterError, match="exactly zero"):
            route_d106_k_conditioned_prediction(active_k=1, row_prediction=row)


def test_full_row_receipt_detects_payload_tampering() -> None:
    mutations = []

    row = _g1_row()
    row["arm_predictions"]["M_JOINT"][0] = "old_a"
    mutations.append(row)

    row = _g1_row()
    row["query_physical_ids"] = ["q0", "q2", "q1"]
    mutations.append(row)

    row = _g1_row()
    row["shared_component_receipts"]["rdce_state_sha256"] = _sha("d")
    mutations.append(row)

    row = _target_row()
    row["receiver"] = "3-19"
    mutations.append(row)

    row = _g1_row()
    row["prediction_receipt_sha256"] = _sha("f")
    mutations.append(row)

    for row in mutations:
        with pytest.raises(D106KConditionedRouterError, match="receipt.*drift"):
            route_d106_k_conditioned_prediction(
                active_k=row["K"], row_prediction=row
            )


@pytest.mark.parametrize(
    "forbidden_name",
    ["receiver", "scene", "class", "class_id", "truth", "query_truth", "metric"],
)
def test_rejects_non_schema_routing_parameters(forbidden_name: str) -> None:
    with pytest.raises(TypeError):
        route_d106_k_conditioned_prediction(
            active_k=1,
            row_prediction=_g1_row(),
            **{forbidden_name: "must-not-route"},
        )


def test_target_neutral_bindings_cannot_change_the_frozen_route() -> None:
    row = _target_row(5)
    row.update(row_id="other-row", receiver="3-19", scene="leo_clear_weak")
    _sign(row)
    result = route_d106_k_conditioned_prediction(active_k=5, row_prediction=row)
    assert result.selected_arm == "M0"
    assert result.predictions == tuple(row["arm_predictions"]["M0"])


def test_route_result_is_immutable_and_has_exact_arm_closure() -> None:
    result = route_d106_k_conditioned_prediction(
        active_k=10, row_prediction=_target_row(10)
    )
    assert ARMS == ("M0", "M_DA", "M_HEAD", "M_JOINT")
    with pytest.raises(TypeError):
        ROUTE_BY_K[5] = "M_JOINT"
    with pytest.raises(AttributeError):
        result.selected_arm = "M_JOINT"
    assert result.as_dict()["query_state_updates"] == 0
