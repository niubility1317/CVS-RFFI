from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import cvsrffi.stage2_d106_target25_evaluator as evaluator
from cvsrffi.stage2_d106_k_conditioned_router import (
    TARGET25_ROW_KEYS,
    TARGET25_ROW_SCHEMA,
    route_d106_k_conditioned_prediction,
)
from cvsrffi.stage2_d106_rcmr_2v_qknn import (
    load_d106_rcmr_2v_method_lock,
)
import test_stage2_d106_rdce_runtime as rdce_fixture


ROOT = Path(__file__).resolve().parents[1]
RCMR_LOCK_PATH = ROOT / "configs" / "d106_rcmr_2v_method_lock_20260801.json"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    active_k: int = 1,
) -> dict[str, object]:
    query_ids = ("query-class-a", "query-class-b", "query-class-c")
    support = rdce_fixture._support_rows(active_k)
    split_handle = replace(
        support.split_handle,
        query_physical_root_sha256=_canonical_sha256(sorted(query_ids)),
    )
    support = replace(support, split_handle=split_handle)
    authority, _path, _document = rdce_fixture._write_row_authority(
        tmp_path, support
    )
    asset = rdce_fixture._formal_asset(tmp_path, monkeypatch)
    rcmr_lock = load_d106_rcmr_2v_method_lock(
        RCMR_LOCK_PATH,
        expected_sha256=_file_sha(RCMR_LOCK_PATH),
    )

    support_plus = support.support_z_id
    support_signed = support_plus.copy()
    for index in range(len(support_signed)):
        support_signed[index, 80 + index % 3] = np.float32(-0.15 - 0.01 * index)

    query_plus_rows: list[np.ndarray] = []
    query_signed_rows: list[np.ndarray] = []
    support_labels = support.support_labels.astype(str).tolist()
    for class_index, class_name in enumerate(support.qknn_bank.classes):
        source_index = support_labels.index(class_name)
        plus = support_plus[source_index].copy()
        plus[120 + class_index] = np.float32(0.017 + 0.003 * class_index)
        signed = plus.copy()
        signed[90 + class_index] = np.float32(-0.21 - 0.01 * class_index)
        query_plus_rows.append(plus)
        query_signed_rows.append(signed)

    return {
        "row_id": support.row_id,
        "receiver": "20-1",
        "scene": "leo_rain_weak",
        "active_k": active_k,
        "support_rows": support,
        "support_signed": np.ascontiguousarray(support_signed, dtype=np.float32),
        "query_plus": np.ascontiguousarray(
            np.stack(query_plus_rows), dtype=np.float32
        ),
        "query_signed": np.ascontiguousarray(
            np.stack(query_signed_rows), dtype=np.float32
        ),
        "query_physical_ids": query_ids,
        "registered_classes": tuple(support.qknn_bank.classes),
        "rdce_asset": asset,
        "rdce_row_authority": authority,
        "rcmr_method_lock": rcmr_lock,
    }


def test_real_public_api_four_arm_row_and_router_direct_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _formal_inputs(tmp_path, monkeypatch)
    fit_calls = 0
    rcmr_calls: list[tuple[object, str]] = []
    original_fit = evaluator.fit_d106_rdce_runtime
    original_build = evaluator.build_d106_rcmr_2v_state

    def counted_fit(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(*args, **kwargs)

    def counted_build(*args, **kwargs):
        rcmr_calls.append(
            (kwargs["method_lock"], kwargs["binding"].da_receipt_sha256)
        )
        return original_build(*args, **kwargs)

    monkeypatch.setattr(evaluator, "fit_d106_rdce_runtime", counted_fit)
    monkeypatch.setattr(evaluator, "build_d106_rcmr_2v_state", counted_build)

    row = evaluator.evaluate_d106_target25_state(**inputs)

    assert set(row) == TARGET25_ROW_KEYS
    assert row["schema"] == TARGET25_ROW_SCHEMA
    assert set(row["arm_predictions"]) == set(evaluator.ARMS)
    assert all(len(value) == 3 for value in row["arm_predictions"].values())
    assert all(
        prediction in row["registered_classes"]
        for values in row["arm_predictions"].values()
        for prediction in values
    )
    assert row["query_truth_access"] is False
    assert row["query_role_access"] is False
    assert row["query_selection"] is False
    assert row["query_state_updates"] == 0
    assert row["prediction_receipt_sha256"] == _canonical_sha256(
        {
            key: value
            for key, value in row.items()
            if key != "prediction_receipt_sha256"
        }
    )
    assert fit_calls == 1
    assert len(rcmr_calls) == 2
    assert rcmr_calls[0][0] is rcmr_calls[1][0] is inputs["rcmr_method_lock"]
    assert rcmr_calls[0][1] != rcmr_calls[1][1]
    assert row["shared_component_receipts"][
        "M_DA_M_JOINT_rdce_state_sha256"
    ] == rcmr_calls[1][1]

    routed = route_d106_k_conditioned_prediction(
        active_k=inputs["active_k"], row_prediction=row
    )
    assert routed.selected_arm == "M_DA"
    assert routed.predictions == tuple(row["arm_predictions"]["M_DA"])


@pytest.mark.parametrize("forbidden", ["truth", "metric", "receiver_selector"])
def test_api_has_no_truth_metric_or_receiver_selection_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forbidden: str,
) -> None:
    inputs = _formal_inputs(tmp_path, monkeypatch)
    inputs[forbidden] = "forbidden"
    with pytest.raises(TypeError):
        evaluator.evaluate_d106_target25_state(**inputs)


def test_plus_must_be_relu_of_same_signed_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _formal_inputs(tmp_path, monkeypatch)
    changed = inputs["query_plus"].copy()
    changed[0, 50] = np.float32(0.25)
    inputs["query_plus"] = changed
    with pytest.raises(evaluator.D106Target25EvaluatorError, match="ReLU"):
        evaluator.evaluate_d106_target25_state(**inputs)


def test_query_physical_root_and_disjointness_fail_closed_before_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _formal_inputs(tmp_path, monkeypatch)
    inputs["query_physical_ids"] = ("wrong-a", "wrong-b", "wrong-c")
    with pytest.raises(
        evaluator.D106Target25EvaluatorError, match="physical-root"
    ):
        evaluator.evaluate_d106_target25_state(**inputs)

    inputs = _formal_inputs(tmp_path, monkeypatch)
    support_id = str(inputs["support_rows"].support_physical_ids[0])
    inputs["query_physical_ids"] = (
        support_id,
        "query-class-b",
        "query-class-c",
    )
    with pytest.raises(Exception, match="disjoint"):
        evaluator.evaluate_d106_target25_state(**inputs)


def test_binding_metadata_does_not_select_an_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _formal_inputs(tmp_path, monkeypatch)
    first = evaluator.evaluate_d106_target25_state(**inputs)
    inputs.update(receiver="3-19", scene="leo_clear_weak")
    second = evaluator.evaluate_d106_target25_state(**inputs)
    assert first["arm_predictions"] == second["arm_predictions"]
    assert first["receiver"] != second["receiver"]
    assert first["scene"] != second["scene"]
