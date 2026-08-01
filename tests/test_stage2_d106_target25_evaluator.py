from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cvsrffi.stage2_d106_target25_evaluator as evaluator
from cvsrffi.stage2_d106_k_conditioned_router import (
    TARGET25_ROW_KEYS,
    TARGET25_ROW_SCHEMA,
    route_d106_k_conditioned_prediction,
)
from cvsrffi.stage2_d106_rcmr_2v_qknn import load_d106_rcmr_2v_method_lock
import test_stage2_d106_rdce_runtime as rdce_fixture


ROOT = Path(__file__).resolve().parents[1]
RCMR_LOCK_PATH = ROOT / "configs" / "d106_rcmr_2v_method_lock_20260801.json"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feature_arrays(support) -> dict[str, Any]:
    query_ids = ("query-class-a", "query-class-b", "query-class-c")
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
        "support_plus": np.ascontiguousarray(support_plus, dtype=np.float32),
        "support_signed": np.ascontiguousarray(support_signed, dtype=np.float32),
        "query_plus": np.ascontiguousarray(np.stack(query_plus_rows), dtype=np.float32),
        "query_signed": np.ascontiguousarray(np.stack(query_signed_rows), dtype=np.float32),
        "support_physical_ids": tuple(support.support_physical_ids.astype(str).tolist()),
        "query_physical_ids": query_ids,
    }


def _strict_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    suffix: str = "one",
) -> tuple[dict[str, object], dict[str, Path]]:
    support = rdce_fixture._support_rows(1)
    arrays = _feature_arrays(support)
    support = replace(
        support,
        split_handle=replace(
            support.split_handle,
            query_physical_root_sha256=_canonical_sha256(
                sorted(arrays["query_physical_ids"])
            ),
        ),
    )
    authority, _authority_path, _document = rdce_fixture._write_row_authority(
        tmp_path, support, name=f"row-{suffix}.json"
    )
    asset = rdce_fixture._formal_asset(tmp_path, monkeypatch)
    lock = load_d106_rcmr_2v_method_lock(
        RCMR_LOCK_PATH, expected_sha256=_file_sha(RCMR_LOCK_PATH)
    )
    feature_path = tmp_path / f"features-{suffix}.npz"
    feature_receipt_path = tmp_path / f"features-{suffix}.receipt.json"
    published = evaluator.publish_d106_paired_features(
        feature_path,
        feature_receipt_path,
        received_iq_package_seal_sha256="1" * 64,
        checkpoint_sha256="2" * 64,
        runtime_sha256="3" * 64,
        forward_receipt_sha256="4" * 64,
        **arrays,
    )
    features = evaluator.load_d106_paired_features(
        feature_path,
        feature_receipt_path,
        expected_receipt_sha256=published["feature_receipt_sha256"],
    )
    plan_path = tmp_path / f"plan-{suffix}.json"
    projection = {
        "schema": evaluator.PLAN_STATE_SCHEMA,
        "row_id": support.row_id,
        "receiver": "20-1",
        "scene": "leo_rain_weak",
        "active_k": 1,
        "registered_classes": list(support.qknn_bank.classes),
        "capsule_id": support.split_handle.capsule_id,
        "split_id": support.split_handle.split_id,
        "validator_receipt_sha256": support.split_handle.validator_receipt_sha256,
        "seed": support.seed,
        "support_physical_root_sha256": support.split_handle.support_physical_root_sha256,
        "query_physical_root_sha256": support.split_handle.query_physical_root_sha256,
        "paired_feature_receipt_sha256": features.receipt_sha256,
    }
    plan_sha = evaluator.publish_d106_target25_plan_state(
        plan_path, projection=projection
    )
    plan = evaluator.load_d106_target25_plan_state(
        plan_path, expected_receipt_sha256=plan_sha
    )
    return (
        {
            "plan_state": plan,
            "paired_features": features,
            "support_rows": support,
            "rdce_asset": asset,
            "rdce_row_authority": authority,
            "rcmr_method_lock": lock,
        },
        {
            "feature": feature_path,
            "feature_receipt": feature_receipt_path,
            "plan": plan_path,
        },
    )


def test_strict_typed_public_api_four_arm_row_and_router_direct_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, _paths = _strict_inputs(tmp_path, monkeypatch)
    fit_calls = 0
    rcmr_calls: list[tuple[object, str]] = []
    original_fit = evaluator.fit_d106_rdce_runtime
    original_build = evaluator.build_d106_rcmr_2v_state

    def counted_fit(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(*args, **kwargs)

    def counted_build(*args, **kwargs):
        rcmr_calls.append((kwargs["method_lock"], kwargs["binding"].da_receipt_sha256))
        return original_build(*args, **kwargs)

    monkeypatch.setattr(evaluator, "fit_d106_rdce_runtime", counted_fit)
    monkeypatch.setattr(evaluator, "build_d106_rcmr_2v_state", counted_build)
    row = evaluator.evaluate_d106_target25_state(**inputs)

    assert set(row) == TARGET25_ROW_KEYS
    assert row["schema"] == TARGET25_ROW_SCHEMA
    assert set(row["arm_predictions"]) == set(evaluator.ARMS)
    assert all(len(value) == 3 for value in row["arm_predictions"].values())
    assert row["query_truth_access"] is False
    assert row["query_role_access"] is False
    assert row["query_selection"] is False
    assert row["query_state_updates"] == 0
    assert row["prediction_receipt_sha256"] == _canonical_sha256(
        {key: value for key, value in row.items() if key != "prediction_receipt_sha256"}
    )
    assert fit_calls == 1
    assert len(rcmr_calls) == 2
    assert rcmr_calls[0][0] is rcmr_calls[1][0] is inputs["rcmr_method_lock"]
    assert row["shared_component_receipts"][
        "M_DA_M_JOINT_rdce_state_sha256"
    ] == rcmr_calls[1][1]
    assert row["shared_component_receipts"]["target25_plan_state_sha256"] == inputs[
        "plan_state"
    ].receipt_sha256
    assert row["shared_component_receipts"]["forward_receipt_sha256"] == "4" * 64

    routed = route_d106_k_conditioned_prediction(
        active_k=inputs["plan_state"].active_k, row_prediction=row
    )
    assert routed.selected_arm == "M_DA"
    assert routed.predictions == tuple(row["arm_predictions"]["M_DA"])


def test_publisher_is_non_overwriting_and_enforces_same_forward_relu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _inputs, paths = _strict_inputs(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError):
        evaluator.publish_d106_paired_features(
            paths["feature"],
            tmp_path / "unused-receipt.json",
            received_iq_package_seal_sha256="1" * 64,
            checkpoint_sha256="2" * 64,
            runtime_sha256="3" * 64,
            forward_receipt_sha256="4" * 64,
            support_plus=np.ones((1, 160), dtype=np.float32),
            support_signed=np.ones((1, 160), dtype=np.float32),
            query_plus=np.ones((1, 160), dtype=np.float32),
            query_signed=np.ones((1, 160), dtype=np.float32),
            support_physical_ids=("s",),
            query_physical_ids=("q",),
        )

    bad_plus = np.ones((1, 160), dtype=np.float32)
    bad_signed = bad_plus.copy()
    bad_signed[0, 0] = np.float32(-1.0)
    with pytest.raises(evaluator.D106Target25EvaluatorError, match="ReLU"):
        evaluator.publish_d106_paired_features(
            tmp_path / "bad.npz",
            tmp_path / "bad.json",
            received_iq_package_seal_sha256="1" * 64,
            checkpoint_sha256="2" * 64,
            runtime_sha256="3" * 64,
            forward_receipt_sha256="4" * 64,
            support_plus=bad_plus,
            support_signed=bad_signed,
            query_plus=bad_plus,
            query_signed=bad_plus,
            support_physical_ids=("s",),
            query_physical_ids=("q",),
        )


def test_feature_loader_rejects_external_sha_extra_keys_and_npz_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _inputs, paths = _strict_inputs(tmp_path, monkeypatch)
    with pytest.raises(evaluator.D106Target25EvaluatorError, match="external receipt"):
        evaluator.load_d106_paired_features(
            paths["feature"], paths["feature_receipt"], expected_receipt_sha256="f" * 64
        )

    document = json.loads(paths["feature_receipt"].read_text(encoding="utf-8"))
    document["metadata"] = {"truth": ["forbidden"]}
    extra_receipt = tmp_path / "extra.receipt.json"
    extra_receipt.write_bytes(_canonical_bytes(document))
    with pytest.raises(evaluator.D106Target25EvaluatorError, match="field closure"):
        evaluator.load_d106_paired_features(
            paths["feature"],
            extra_receipt,
            expected_receipt_sha256=_file_sha(extra_receipt),
        )

    with np.load(paths["feature"], allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["metadata"] = np.asarray(["forbidden"], dtype=np.str_)
    extra_npz = tmp_path / "extra.npz"
    np.savez_compressed(extra_npz, **arrays)
    document.pop("metadata")
    document["feature_archive_name"] = extra_npz.name
    document["feature_archive_sha256"] = _file_sha(extra_npz)
    extra_npz_receipt = tmp_path / "extra-npz.receipt.json"
    extra_npz_receipt.write_bytes(_canonical_bytes(document))
    with pytest.raises(evaluator.D106Target25EvaluatorError, match="member closure"):
        evaluator.load_d106_paired_features(
            extra_npz,
            extra_npz_receipt,
            expected_receipt_sha256=_file_sha(extra_npz_receipt),
        )


def test_feature_loader_detects_array_and_ordered_id_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _inputs, paths = _strict_inputs(tmp_path, monkeypatch)
    with np.load(paths["feature"], allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["query_plus"][0, 0] += np.float32(0.01)
    tampered_npz = tmp_path / "tampered.npz"
    np.savez_compressed(tampered_npz, **arrays)
    document = json.loads(paths["feature_receipt"].read_text(encoding="utf-8"))
    document["feature_archive_name"] = tampered_npz.name
    document["feature_archive_sha256"] = _file_sha(tampered_npz)
    receipt = tmp_path / "tampered.receipt.json"
    receipt.write_bytes(_canonical_bytes(document))
    with pytest.raises(evaluator.D106Target25EvaluatorError, match="query_plus receipt"):
        evaluator.load_d106_paired_features(
            tampered_npz, receipt, expected_receipt_sha256=_file_sha(receipt)
        )

    arrays["query_plus"][0, 0] -= np.float32(0.01)
    arrays["query_physical_ids"] = arrays["query_physical_ids"][::-1]
    reordered_npz = tmp_path / "reordered.npz"
    np.savez_compressed(reordered_npz, **arrays)
    document["feature_archive_name"] = reordered_npz.name
    document["feature_archive_sha256"] = _file_sha(reordered_npz)
    receipt = tmp_path / "reordered.receipt.json"
    receipt.write_bytes(_canonical_bytes(document))
    with pytest.raises(evaluator.D106Target25EvaluatorError, match="ordered physical-ID"):
        evaluator.load_d106_paired_features(
            reordered_npz, receipt, expected_receipt_sha256=_file_sha(receipt)
        )


def test_plan_loader_rejects_extra_fields_and_receiver_receipt_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, paths = _strict_inputs(tmp_path, monkeypatch)
    document = json.loads(paths["plan"].read_text(encoding="utf-8"))
    document["metadata"] = {"scene_override": "forbidden"}
    extra = tmp_path / "plan-extra.json"
    extra.write_bytes(_canonical_bytes(document))
    with pytest.raises(evaluator.D106Target25EvaluatorError, match="field closure"):
        evaluator.load_d106_target25_plan_state(
            extra, expected_receipt_sha256=_file_sha(extra)
        )

    plan = inputs["plan_state"]
    object.__setattr__(plan, "receiver", "3-19")
    with pytest.raises(evaluator.D106Target25EvaluatorError, match="revalidation"):
        evaluator.evaluate_d106_target25_state(**inputs)


def test_plan_feature_support_mismatch_fails_before_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, _paths = _strict_inputs(tmp_path, monkeypatch)
    plan = inputs["plan_state"]
    features = inputs["paired_features"]
    object.__setattr__(plan, "paired_feature_receipt_sha256", "f" * 64)
    object.__setattr__(plan, "receipt_sha256", _canonical_sha256(plan.receipt_payload))
    with pytest.raises(evaluator.D106Target25EvaluatorError, match="exact binding"):
        evaluator.evaluate_d106_target25_state(**inputs)

    inputs, _paths = _strict_inputs(tmp_path, monkeypatch, suffix="two")
    features = inputs["paired_features"]
    changed = features.query_plus.copy()
    changed[0, 0] += np.float32(0.01)
    object.__setattr__(features, "query_plus", changed)
    with pytest.raises(evaluator.D106Target25EvaluatorError, match="revalidation"):
        evaluator.evaluate_d106_target25_state(**inputs)


@pytest.mark.parametrize(
    "forbidden",
    [
        "truth",
        "metric",
        "receiver",
        "scene",
        "row_id",
        "support_signed",
        "query_plus",
        "query_signed",
    ],
)
def test_evaluator_has_no_loose_feature_or_selection_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forbidden: str,
) -> None:
    inputs, _paths = _strict_inputs(tmp_path, monkeypatch)
    inputs[forbidden] = "forbidden"
    with pytest.raises(TypeError):
        evaluator.evaluate_d106_target25_state(**inputs)


def test_published_feature_arrays_are_bytes_backed_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, _paths = _strict_inputs(tmp_path, monkeypatch)
    features = inputs["paired_features"]
    for name in evaluator.FEATURE_ARRAY_NAMES:
        array = getattr(features, name)
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.setflags(write=True)
