from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from cvsrffi import stage2_d106_rcmr_g0 as g0
from cvsrffi.stage2_d106_phase1_tap import (
    D106Phase1TapRows,
    PROTOCOL_SCHEMA,
    TAP_ARCHIVE_NAME,
    TAP_RECEIPT_SCHEMA,
    TAP_RECEIPT_NAME,
)
from cvsrffi.stage2_zid_student_t_qknn import Phase1ZIDStudentTLock


ROOT = Path(__file__).resolve().parents[1]
METHOD_LOCK = ROOT / "configs" / "d106_rcmr_2v_method_lock_20260801.json"
PHASE1_LODO_SHA = "3" * 64


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _synthetic_tap(*, seed: int = 106) -> D106Phase1TapRows:
    rng = np.random.default_rng(seed)
    class_ids = tuple(f"tx-{index}" for index in range(6))
    receiver_ids = tuple(f"rx-{index}" for index in range(7))
    day_ids = tuple(f"day-{index}" for index in range(4))
    rows: list[np.ndarray] = []
    labels: list[str] = []
    receivers: list[str] = []
    days: list[str] = []
    physical: list[str] = []
    scenarios: list[str] = []
    observations: list[str] = []
    for receiver_index, receiver in enumerate(receiver_ids):
        for day_index, day in enumerate(day_ids):
            cell_index = receiver_index * len(day_ids) + day_index
            for class_index, class_id in enumerate(class_ids):
                count = 4 if (cell_index + class_index) % 2 == 0 else 3
                for sample_index in range(count):
                    signed = rng.normal(0.0, 0.002, size=160).astype(np.float32)
                    signed[0] = np.float32(1.0)
                    signed[10 + class_index] = np.float32(0.24 + 0.002 * sample_index)
                    signed_class = (
                        (class_index + 1) % len(class_ids)
                        if cell_index == 0
                        else class_index
                    )
                    if cell_index == 0:
                        signed[10 + signed_class] = np.float32(0.235)
                    signed[30 + signed_class] = np.float32(-6.0)
                    rows.append(signed)
                    labels.append(class_id)
                    receivers.append(receiver)
                    days.append(day)
                    physical.append(
                        f"p-{receiver_index:02d}-{day_index:02d}-"
                        f"{class_index:02d}-{sample_index:02d}"
                    )
                    scenarios.append(
                        ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")[
                            (cell_index + class_index + sample_index) % 3
                        ]
                    )
                    observations.append(f"obs-{len(observations):04d}")
    pre_relu = _readonly(np.stack(rows).astype(np.float32))
    assert pre_relu.shape == (588, 160)
    z_id = _readonly(np.maximum(pre_relu, np.float32(0.0)).astype(np.float32))
    receipt = MappingProxyType(
        {
            "schema": TAP_RECEIPT_SCHEMA,
            "protocol_schema": PROTOCOL_SCHEMA,
            "row_count": 588,
            "exact_inner_join": True,
            "same_received_iq_for_zid_zdom": True,
            "z_id_storage_policy": "derive_relu_pre_relu",
            "feature_stage_source_pool_access": False,
            "clean_iq_access": False,
            "target_access": False,
            "formal_query_access": False,
        }
    )
    return D106Phase1TapRows(
        pre_relu=pre_relu,
        z_dom=_readonly(np.zeros_like(pre_relu, dtype=np.float32)),
        tx_labels=_readonly(np.asarray(labels, dtype=np.str_)),
        receiver_ids=_readonly(np.asarray(receivers, dtype=np.str_)),
        day_ids=_readonly(np.asarray(days, dtype=np.str_)),
        physical_ids=_readonly(np.asarray(physical, dtype=np.str_)),
        scenario_names=_readonly(np.asarray(scenarios, dtype=np.str_)),
        observation_ids=_readonly(np.asarray(observations, dtype=np.str_)),
        z_id=z_id,
        receipt=receipt,
    )


def _method_lock_sha() -> str:
    return hashlib.sha256(METHOD_LOCK.read_bytes()).hexdigest()


def _tap_receipt_sha(rows: D106Phase1TapRows) -> str:
    return hashlib.sha256(g0._canonical_bytes(rows.receipt)).hexdigest()


def _predecessor_lock(k_shot: int, *, tap_sha: str, shared_h0: float = 0.35):
    return Phase1ZIDStudentTLock(
        active_k=k_shot,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=shared_h0,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=0.85,
        phase1_lodo_receipt_sha256=PHASE1_LODO_SHA,
        quantization_margin_audit_sha256=tap_sha,
    )


def _locks(rows: D106Phase1TapRows):
    tap_sha = _tap_receipt_sha(rows)
    return tuple(_predecessor_lock(k, tap_sha=tap_sha) for k in g0.K_VALUES)


def _production_request(*, authority: str | None = None):
    return g0.D106RCMRG0ProductionRequest(
        registered_classes=tuple(f"tx-{index}" for index in range(6)),
        expected_release_commit="a" * 40,
        expected_code_sha256=tuple(g0._current_code_sha256().items()),
        expected_d105_lock_authority_sha256=authority,
    )


def _source_spoofable_loader(
    archive_path, receipt_path, *, expected_archive_sha256, expected_receipt_sha256
):
    """A same-signature replacement whose source coordinates can be spoofed."""

    del archive_path, receipt_path, expected_archive_sha256, expected_receipt_sha256
    return _P0_SYNTHETIC_ROWS


def _fixture_sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_minimal_strict_tap(tmp_path: Path, rows: D106Phase1TapRows):
    """Build a loader-valid local fixture without patching the strict loader."""

    output = tmp_path / "strict_tap"
    output.mkdir()
    arrays = {
        "pre_relu": np.asarray(rows.pre_relu, dtype=np.float32),
        "z_dom": np.asarray(rows.z_dom, dtype=np.float32),
        "tx_labels": np.asarray(rows.tx_labels),
        "receiver_ids": np.asarray(rows.receiver_ids),
        "day_ids": np.asarray(rows.day_ids),
        "physical_ids": np.asarray(rows.physical_ids),
        "scenario_names": np.asarray(rows.scenario_names),
        "observation_ids": np.asarray(rows.observation_ids),
    }
    tap = g0._tap_module
    archive = output / TAP_ARCHIVE_NAME
    archive.write_bytes(tap._deterministic_npz_bytes(arrays))
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    export_closure = tap._execution_closure("export")
    extract_closure = tap._execution_closure("extract")
    input_ls_sha = _fixture_sha256("fixture-input-ls")
    selected_root = _fixture_sha256("fixture-selected-root")
    receipt = {
        "schema": TAP_RECEIPT_SCHEMA,
        "candidate_id": tap.CANDIDATE_ID,
        "split_id": tap.SPLIT_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "selected_iq_archive_sha256": _fixture_sha256("fixture-selected-archive"),
        "selected_iq_receipt_sha256": _fixture_sha256("fixture-selected-receipt"),
        "storage_validator_receipt_sha256": _fixture_sha256("fixture-validator"),
        "storage_validation_binding": {
            "schema": tap.LS_IQ_VALIDATOR_SCHEMA,
            "storage_validation_root_sha256": _fixture_sha256("fixture-validator-root"),
            "selected_content_root_sha256": selected_root,
            "all_8400x3_storage_semantics_verified": True,
        },
        "extraction_binding": {
            "schema": tap.LS_IQ_RECEIPT_SCHEMA,
            "row_count": 588,
            "selection_salt_sha256": _fixture_sha256("fixture-selection-salt"),
            "selected_content_root_sha256": selected_root,
            "input_ls_archive_sha256": input_ls_sha,
            "execution_root_sha256": extract_closure["execution_content_root_sha256"],
        },
        "input_ls_archive_sha256": input_ls_sha,
        "checkpoint_sha256": tap.EXPECTED_CHECKPOINT_SHA256,
        "runtime_sha256": _fixture_sha256("fixture-runtime"),
        "execution_closure": export_closure,
        "execution_pre_root_sha256": export_closure["execution_content_root_sha256"],
        "execution_post_root_sha256": export_closure["execution_content_root_sha256"],
        "tap_archive_name": TAP_ARCHIVE_NAME,
        "tap_archive_sha256": archive_sha,
        "tap_archive_members": list(tap.TAP_MEMBERS),
        "array_sha256": {name: tap._array_sha256(value) for name, value in arrays.items()},
        "row_count": 588,
        "source_split_counts": dict(tap.EXPECTED_COUNTS),
        "rho_label": tap.RHO_LABEL,
        "physical_id_root_sha256": tap._ordered_id_root(
            arrays["physical_ids"].astype(str).tolist()
        ),
        "scenario_order": list(tap.FORMAL_LEO_WEAK_SCENARIOS),
        "method_visible_received_iq_rows": 588,
        "method_visible_tx_label_rows": 588,
        "checkpoint_loader": {
            "policy": "legacy_pickle_exact_frozen_sha_only",
            "torch_version": "fixture",
            "safe_globals_available": False,
            "weights_only": False,
            "exact_frozen_checkpoint_sha256_required": tap.EXPECTED_CHECKPOINT_SHA256,
            "caller_selected_checkpoint_allowed": False,
        },
        "model_reconstruction": {
            "loader": "d105_minimal_cvsincnet_checkpoint_reconstruction_v1",
            "model_factory": "model_dual_cvsincnet.build_dual_model",
            "backbone_factory": "model.build_model",
            "checkpoint_load_strict": True,
            "missing_keys": 0,
            "unexpected_keys": 0,
            "skipped_mismatch": 0,
            "state_tensor_count": 1,
            "num_domains_from_state": 1,
            "input_len": 256,
            "eval_mode": True,
        },
        "forward": {
            "forward_batch_capacity": 256,
            "forward_invocation_count": 3,
            "last_batch_real_rows": 76,
            "last_batch_padding_rows": 180,
            "same_iq_dual_forward": True,
            "fixed256_zero_pad_then_slice": True,
        },
        "exact_inner_join": True,
        "same_received_iq_for_zid_zdom": True,
        "z_id_storage_policy": "derive_relu_pre_relu",
        "feature_stage_source_pool_access": False,
        "received_iq_persisted": False,
        "raw_iq_persisted": False,
        "clean_iq_access": False,
        "target_access": False,
        "formal_query_access": False,
    }
    receipt_path = output / TAP_RECEIPT_NAME
    receipt_path.write_bytes(tap._canonical_bytes(receipt))
    tap._write_completion_marker(
        output,
        artifact_kind="d106_ls_strict_tap",
        members=(TAP_ARCHIVE_NAME, TAP_RECEIPT_NAME),
    )
    return archive, receipt_path, archive_sha, hashlib.sha256(receipt_path.read_bytes()).hexdigest()


def _document(payload: bytes) -> dict:
    return json.loads(payload.decode("utf-8"))


def _walk_keys(value):
    if isinstance(value, dict):
        for key, member in value.items():
            yield str(key)
            yield from _walk_keys(member)
    elif isinstance(value, list):
        for member in value:
            yield from _walk_keys(member)


def _reseal_production_document(document: dict) -> bytes:
    document.pop("result_receipt_sha256")
    manifest = document["execution_manifest"]
    for per_k in manifest["per_k"]:
        for fold in per_k["fold_execution_receipts"]:
            fold_payload = dict(fold)
            fold_payload.pop("execution_receipt_sha256")
            fold["execution_receipt_sha256"] = hashlib.sha256(
                g0._canonical_bytes(fold_payload)
            ).hexdigest()
        per_k["fold_execution_receipts_root_sha256"] = hashlib.sha256(
            g0._canonical_bytes(
                [item["execution_receipt_sha256"] for item in per_k["fold_execution_receipts"]]
            )
        ).hexdigest()
        per_k["fold_changed_bitmap_roots_root_sha256"] = hashlib.sha256(
            g0._canonical_bytes(
                [
                    item["argmax_changed_bitmap_root_sha256"]
                    for item in per_k["fold_execution_receipts"]
                ]
            )
        ).hexdigest()
        per_k_payload = dict(per_k)
        per_k_payload.pop("per_k_receipt_sha256")
        per_k["per_k_receipt_sha256"] = hashlib.sha256(
            g0._canonical_bytes(per_k_payload)
        ).hexdigest()
    manifest["per_k_execution_receipts_root_sha256"] = hashlib.sha256(
        g0._canonical_bytes([item["per_k_receipt_sha256"] for item in manifest["per_k"]])
    ).hexdigest()
    manifest["canonical_execution_root_sha256"] = manifest[
        "per_k_execution_receipts_root_sha256"
    ]
    manifest_payload = dict(manifest)
    manifest_payload.pop("execution_manifest_receipt_sha256")
    manifest["execution_manifest_receipt_sha256"] = hashlib.sha256(
        g0._canonical_bytes(manifest_payload)
    ).hexdigest()
    document["execution_manifest_root_sha256"] = hashlib.sha256(
        g0._canonical_bytes(manifest)
    ).hexdigest()
    document["result_receipt_sha256"] = hashlib.sha256(
        g0._canonical_bytes(document)
    ).hexdigest()
    return g0._canonical_bytes(document)


@pytest.fixture(scope="module")
def full_synthetic_result() -> bytes:
    rows = _synthetic_tap()
    return g0.run_d106_rcmr_g0_synthetic_test(
        rows,
        registered_classes=tuple(f"tx-{index}" for index in range(6)),
        predecessor_locks=_locks(rows),
        rcmr_method_lock_sha256=_method_lock_sha(),
        synthetic_test_id="d106-g0-unit-synthetic",
    )


@pytest.fixture(scope="module")
def full_internal_rows_result() -> bytes:
    rows = _synthetic_tap()
    request = _production_request()
    return g0._execute_rows_internal_test(
        rows,
        registered_classes=tuple(f"tx-{index}" for index in range(6)),
        predecessor_locks=_locks(rows),
        rcmr_method_lock_sha256=_method_lock_sha(),
        tap_archive_sha256="a" * 64,
        tap_receipt_sha256=_tap_receipt_sha(rows),
        predecessor_lock_bundle_root_sha256="b" * 64,
        code_sha256=g0._current_code_sha256(),
        request=request,
    )


@pytest.fixture(scope="module")
def full_production_result(tmp_path_factory) -> bytes:
    rows = _synthetic_tap()
    archive, receipt, archive_sha, receipt_sha = _write_minimal_strict_tap(
        tmp_path_factory.mktemp("d106-strict-tap"), rows
    )
    return g0.run_d106_rcmr_g0_from_formal_tap(
        archive,
        receipt,
        expected_archive_sha256=archive_sha,
        expected_receipt_sha256=receipt_sha,
        request=_production_request(),
    )


def test_public_surface_has_single_production_call_and_no_handle_rows_or_token():
    forbidden = {
        "load_d106_rcmr_g0_formal_tap",
        "build_d106_rcmr_g0_fold_plan",
        "build_d106_rcmr_g0_request",
        "run_d106_rcmr_g0",
        "D106RCMRG0Fold",
        "D106RCMRG0FoldPlan",
        "D106RCMRG0Request",
    }
    assert forbidden.isdisjoint(g0.__all__)
    assert not any("token" in name.lower() for name in g0.__all__)
    signature = inspect.signature(g0.run_d106_rcmr_g0_from_formal_tap)
    assert list(signature.parameters) == [
        "archive_path",
        "receipt_path",
        "expected_archive_sha256",
        "expected_receipt_sha256",
        "request",
    ]
    assert signature.return_annotation in {"bytes", bytes}


def test_production_rejects_wrong_caller_sha_through_the_strict_loader(tmp_path):
    archive = tmp_path / TAP_ARCHIVE_NAME
    receipt = tmp_path / TAP_RECEIPT_NAME
    archive.write_bytes(b"strict-loader-must-not-see-this-for-wrong-sha")
    receipt.write_bytes(b'{"not":"loaded"}')
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(g0.D106RCMRG0Error, match="strict D106 tap loader rejected"):
        g0.run_d106_rcmr_g0_from_formal_tap(
            archive,
            receipt,
            expected_archive_sha256="f" * 64,
            expected_receipt_sha256=receipt_sha,
            request=_production_request(),
        )


def test_production_mechanical_manifest_is_real_nonformal_and_root_only(
    full_production_result,
):
    assert g0.verify_d106_rcmr_g0_production_result_bytes(
        full_production_result,
        expected_sha256=hashlib.sha256(full_production_result).hexdigest(),
    ) is full_production_result
    document = _document(full_production_result)
    assert document["schema"] == g0.PRODUCTION_SCHEMA
    assert document["status"] == g0.PRODUCTION_EXECUTED_STATUS
    assert document["real_g0_executed"] is True
    assert document["authority_flags"] == {
        "d105_formal_authority": False,
        "held_truth_access": False,
        "performance_metrics_computed": False,
        "phase2_promotion_authority": False,
        "runner_authority": False,
        "target_access": False,
    }
    manifest = document["execution_manifest"]
    assert manifest["K_values"] == [1, 5, 10]
    assert manifest["fold_count"] == 28
    assert manifest["query_count_per_k"] == 588
    assert [item["fold_count"] for item in manifest["per_k"]] == [28, 28, 28]
    assert [item["query_count"] for item in manifest["per_k"]] == [588, 588, 588]
    assert manifest["functional_gate_status"] == "REJECT_NO_FUNCTION_K_ZERO_CHANGED"
    forbidden_exact = {"query_ids", "candidate_argmax", "baseline_argmax", "truth", "score"}
    assert not any(key.lower() in forbidden_exact for key in _walk_keys(document))
    assert not any("accuracy" in key.lower() or "confusion" in key.lower() for key in _walk_keys(document))


def test_direct_rows_executor_is_internal_only_and_public_verifier_rejects(
    full_internal_rows_result,
):
    document = _document(full_internal_rows_result)
    assert document["schema"] == g0.INTERNAL_ROWS_SCHEMA
    assert document["status"] == g0.INTERNAL_ROWS_STATUS
    assert document["internal_test_only"] is True
    assert document["real_g0_executed"] is False
    with pytest.raises(g0.D106RCMRG0Error, match="schema is not accepted"):
        g0.verify_d106_rcmr_g0_production_result_bytes(
            full_internal_rows_result,
            expected_sha256=hashlib.sha256(full_internal_rows_result).hexdigest(),
        )


def test_production_and_synthetic_verifiers_are_schema_separated(
    full_production_result, full_synthetic_result
):
    with pytest.raises(g0.D106RCMRG0Error, match="schema is not accepted"):
        g0.verify_d106_rcmr_g0_production_result_bytes(
            full_synthetic_result,
            expected_sha256=hashlib.sha256(full_synthetic_result).hexdigest(),
        )
    with pytest.raises(g0.D106RCMRG0Error, match="schema is not accepted"):
        g0.verify_d106_rcmr_g0_synthetic_test_result_bytes(
            full_production_result,
            expected_sha256=hashlib.sha256(full_production_result).hexdigest(),
        )


def test_production_recomputes_fold_bitmap_and_count_on_verify(full_production_result):
    document = _document(full_production_result)
    fold = document["execution_manifest"]["per_k"][0]["fold_execution_receipts"][0]
    bitmap = fold["argmax_changed_bitmap"]
    replacement = "1" if bitmap[0] == "0" else "0"
    fold["argmax_changed_bitmap"] = replacement + bitmap[1:]
    resealed = _reseal_production_document(document)
    with pytest.raises(g0.D106RCMRG0Error, match="bitmap/count"):
        g0.verify_d106_rcmr_g0_production_result_bytes(
            resealed, expected_sha256=hashlib.sha256(resealed).hexdigest()
        )


def test_production_derives_588_query_rows_from_fold_bitmaps(full_production_result):
    document = _document(full_production_result)
    fold = document["execution_manifest"]["per_k"][0]["fold_execution_receipts"][0]
    shortened = fold["argmax_changed_bitmap"][:-1]
    assert shortened
    fold["argmax_changed_bitmap"] = shortened
    fold["argmax_changed_count"] = shortened.count("1")
    fold["argmax_changed_bitmap_root_sha256"] = hashlib.sha256(
        g0._canonical_bytes(
            {
                "encoding": "ascii01_query_order",
                "query_count": len(shortened),
                "bits": shortened,
            }
        )
    ).hexdigest()
    resealed = _reseal_production_document(document)
    with pytest.raises(g0.D106RCMRG0Error, match="per-K aggregate closure"):
        g0.verify_d106_rcmr_g0_production_result_bytes(
            resealed, expected_sha256=hashlib.sha256(resealed).hexdigest()
        )


def test_caller_unverified_bundle_cannot_replace_strict_loader_or_fresh_locks():
    receipt_sha = _tap_receipt_sha(_synthetic_tap())
    bundle = g0._load_predecessor_lock_bundle(
        tap_archive_sha256="a" * 64,
        tap_receipt_sha256=receipt_sha,
    )
    document = _document(bundle)
    assert document["tap_hash_provenance"] == "CALLER_SUPPLIED_UNVERIFIED"
    first = g0._strict_reconstruct_predecessor_locks(bundle)
    fresh = g0._strict_reconstruct_predecessor_locks(bundle)
    assert first is not fresh
    assert all(left is not right for left, right in zip(first, fresh, strict=True))
    signature = inspect.signature(g0.run_d106_rcmr_g0_from_formal_tap)
    assert "bundle" not in " ".join(signature.parameters)
    source = inspect.getsource(g0.run_d106_rcmr_g0_from_formal_tap)
    assert "_load_d106_phase1_ls_tap" in source
    assert "_strict_reconstruct_predecessor_locks" in source


def test_production_reaches_actual_strict_loader_not_blocked_shortcut(tmp_path):
    archive = tmp_path / TAP_ARCHIVE_NAME
    receipt = tmp_path / TAP_RECEIPT_NAME
    archive.write_bytes(b"a")
    receipt.write_bytes(b"b")
    with pytest.raises(g0.D106RCMRG0Error, match="strict D106 tap loader rejected"):
        g0.run_d106_rcmr_g0_from_formal_tap(
            archive,
            receipt,
            expected_archive_sha256=hashlib.sha256(b"a").hexdigest(),
            expected_receipt_sha256=hashlib.sha256(b"b").hexdigest(),
            request=_production_request(authority="f" * 64),
        )


def test_monkeypatch_cannot_produce_a_production_document(monkeypatch, tmp_path):
    archive = tmp_path / "tap.npz"
    receipt = tmp_path / "receipt.json"
    archive.write_bytes(b"a")
    receipt.write_bytes(b"b")
    request = _production_request()
    monkeypatch.setattr(g0, "_load_d106_phase1_ls_tap", lambda *args, **kwargs: _synthetic_tap())
    with pytest.raises(g0.D106RCMRG0Error, match="production callable .* drift"):
        g0.run_d106_rcmr_g0_from_formal_tap(
            archive,
            receipt,
            expected_archive_sha256=hashlib.sha256(b"a").hexdigest(),
            expected_receipt_sha256=hashlib.sha256(b"b").hexdigest(),
            request=request,
        )


def test_loader_code_replacement_with_spoofed_source_coordinates_is_rejected(
    monkeypatch, tmp_path
):
    """A preserved function object/source view cannot bypass the code-object pin."""

    loader = g0._load_d106_phase1_ls_tap
    original_source = inspect.getsource(loader)
    replacement = _source_spoofable_loader.__code__.replace(
        co_filename=loader.__code__.co_filename,
        co_firstlineno=loader.__code__.co_firstlineno,
    )
    monkeypatch.setattr(g0._tap_module, "_P0_SYNTHETIC_ROWS", _synthetic_tap(), raising=False)
    monkeypatch.setattr(loader, "__code__", replacement)
    # ``inspect`` still resolves the original source range, which is exactly
    # why production pins executable code identity and bytes as well.
    assert inspect.getsource(loader) == original_source
    archive = tmp_path / TAP_ARCHIVE_NAME
    receipt = tmp_path / TAP_RECEIPT_NAME
    archive.write_bytes(b"not-a-real-strict-tap")
    receipt.write_bytes(g0._canonical_bytes({"schema": "forged"}))
    with pytest.raises(g0.D106RCMRG0Error, match="callable code identity drift"):
        g0.run_d106_rcmr_g0_from_formal_tap(
            archive,
            receipt,
            expected_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            expected_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
            request=_production_request(),
        )


def test_production_runtime_closure_pins_transitive_execution_helpers():
    labels = set(g0._PRODUCTION_RUNTIME_PINS)
    assert {
        f"{g0.__name__}:_snapshot_from_rows",
        f"{g0.__name__}:_build_fold_plan",
        f"{g0.__name__}:_execute_fold",
        f"{g0.__name__}:_nonformal_state_from_support",
        f"{g0._tap_module.__name__}:load_d106_phase1_ls_tap",
        f"{g0._predecessor_module.__name__}:load_d106_train_only_predecessor_lock_bundle",
        f"{g0._qknn_module.__name__}:score_zid_student_t_logits",
        f"{g0._rcmr_module.__name__}:serialize_d106_rcmr_2v_state",
    }.issubset(labels)


@pytest.mark.parametrize(
    ("field", "replacement", "pattern"),
    (
        ("expected_release_commit", "b" * 40, "request receipt/value closure"),
        ("expected_code_sha256", [], "expected-code type drift"),
        (
            "expected_d105_lock_authority_sha256",
            "c" * 64,
            "request receipt/value closure",
        ),
    ),
)
def test_production_request_object_setattr_tampering_is_rejected(
    field, replacement, pattern
):
    """Frozen dataclasses must not trust same-process object.__setattr__."""

    request = _production_request()
    object.__setattr__(request, field, replacement)
    with pytest.raises(g0.D106RCMRG0Error, match=pattern):
        g0._verify_production_code_closure(request)


def test_public_release_code_map_is_fresh_and_matches_production_request_closure():
    first = g0.get_d106_rcmr_g0_release_expected_code_sha256()
    assert type(first) is dict
    request = g0.D106RCMRG0ProductionRequest(
        registered_classes=tuple(f"tx-{index}" for index in range(6)),
        expected_release_commit="a" * 40,
        expected_code_sha256=tuple(first.items()),
    )
    assert dict(request.expected_code_sha256) == first
    first["g0_executor_module"] = "f" * 64
    second = g0.get_d106_rcmr_g0_release_expected_code_sha256()
    assert second == g0._current_code_sha256()
    assert second["g0_executor_module"] != first["g0_executor_module"]


def test_synthetic_full_matrix_is_nonformal_and_52_2_0_is_overall_no_go(
    full_synthetic_result,
):
    result = full_synthetic_result
    assert g0.verify_d106_rcmr_g0_synthetic_test_result_bytes(
        result, expected_sha256=hashlib.sha256(result).hexdigest()
    ) is result
    document = _document(result)
    assert document["schema"] == g0.SYNTHETIC_SCHEMA
    assert document["status"] == g0.SYNTHETIC_STATUS
    assert document["argmax_changed_count_by_k"] == {"1": 52, "5": 2, "10": 0}
    assert document["argmax_changed_count"] == 54
    assert document["functional_gate_status"] == "REJECT_NO_FUNCTION_K_ZERO_CHANGED"
    assert document["functional_gate_pass"] is False
    assert document["zero_changed_k_values"] == [10]
    assert document["held_label_audit_status"] == g0.HELD_LABEL_AUDIT_STATUS
    assert document["same_process_held_label_capability_absence_claimed"] is False
    assert document["runner_authority"] is False
    assert document["deployable"] is False
    assert document["formal_performance_claim"] is False
    assert document["real_g0_executed"] is False
    assert document["external_execution_manifest_root_sha256"] is None
    assert document["opaque_claim_not_independently_verified"] is True
    assert document["promotion_or_runner_consumption_allowed"] is False
    assert document["canonical_execution_root_sha256"]
    assert [item["K"] for item in document["per_k"]] == [1, 5, 10]
    required = {
        "query_root_sha256",
        "query_plus_root_sha256",
        "query_signed_root_sha256",
        "support_root_sha256",
        "baseline_bank_receipt_sha256",
        "paired_view_receipt_sha256",
        "rcmr_state_receipt_sha256",
        "rcmr_wire_sha256",
        "registry_root_sha256",
        "common_query_order_root_sha256",
        "tap_snapshot_root_sha256",
        "fold_identity_root_sha256",
        "receiver_id",
        "day_id",
        "argmax_changed_bitmap",
        "argmax_changed_bitmap_root_sha256",
        "argmax_changed_count",
        "execution_receipt_sha256",
    }
    for per_k in document["per_k"]:
        assert len(per_k["fold_execution_receipts"]) == 28
        assert per_k["fold_execution_receipts_root_sha256"]
        for fold in per_k["fold_execution_receipts"]:
            assert required.issubset(fold)
            assert fold["algorithm_execution_scope"] == g0.ALGORITHM_SCOPE
            assert fold["p2_validated_or_deployable_claimed"] is False
            assert fold["runner_authority"] is False
            assert fold["opaque_claim_not_independently_verified"] is True
            assert fold["promotion_or_runner_consumption_allowed"] is False
            assert set(fold["argmax_changed_bitmap"]) <= {"0", "1"}
            assert fold["argmax_changed_count"] == fold["argmax_changed_bitmap"].count("1")
    forbidden = {"accuracy", "correctness", "truth", "confusion", "held_label", "score"}
    assert not any(key.lower() in forbidden for key in _walk_keys(document))


def test_query_view_byte_perturbation_changes_receipt_even_when_argmax_is_same():
    rows = _synthetic_tap()
    snapshot = g0._snapshot_from_rows(rows, tap_receipt_sha256=_tap_receipt_sha(rows))
    fold = g0._build_fold_plan(snapshot)[0]
    common_root = hashlib.sha256(
        g0._canonical_bytes(
            [query for member in g0._build_fold_plan(snapshot) for query in member.query_ids]
        )
    ).hexdigest()
    original = g0._execute_fold(
        snapshot,
        fold,
        active_k=1,
        predecessor_lock=_locks(rows)[0],
        rcmr_method_lock_sha256=_method_lock_sha(),
        registry=tuple(f"tx-{index}" for index in range(6)),
        common_query_order_root_sha256=common_root,
    )
    query_index = list(rows.physical_ids).index(fold.query_ids[0])
    pre_relu = np.array(rows.pre_relu, copy=True)
    pre_relu[query_index, 10] += np.float32(1e-6)
    z_id = np.maximum(pre_relu, np.float32(0.0)).astype(np.float32)
    perturbed_rows = replace(
        rows,
        pre_relu=_readonly(pre_relu),
        z_id=_readonly(z_id),
    )
    perturbed_snapshot = g0._snapshot_from_rows(
        perturbed_rows, tap_receipt_sha256=_tap_receipt_sha(perturbed_rows)
    )
    perturbed_fold = g0._build_fold_plan(perturbed_snapshot)[0]
    perturbed = g0._execute_fold(
        perturbed_snapshot,
        perturbed_fold,
        active_k=1,
        predecessor_lock=_locks(perturbed_rows)[0],
        rcmr_method_lock_sha256=_method_lock_sha(),
        registry=tuple(f"tx-{index}" for index in range(6)),
        common_query_order_root_sha256=common_root,
    )
    assert original[1:3] == perturbed[1:3]
    assert original[3]["query_plus_root_sha256"] != perturbed[3]["query_plus_root_sha256"]
    assert original[3]["query_signed_root_sha256"] != perturbed[3]["query_signed_root_sha256"]
    assert original[3]["tap_snapshot_root_sha256"] != perturbed[3]["tap_snapshot_root_sha256"]
    assert original[3]["execution_receipt_sha256"] != perturbed[3]["execution_receipt_sha256"]


def test_synthetic_mechanics_cannot_capture_formal_state_or_validated_once(monkeypatch):
    rows = _synthetic_tap()
    snapshot = g0._snapshot_from_rows(rows, tap_receipt_sha256=_tap_receipt_sha(rows))
    fold = g0._build_fold_plan(snapshot)[0]
    common_root = hashlib.sha256(
        g0._canonical_bytes(
            [query for member in g0._build_fold_plan(snapshot) for query in member.query_ids]
        )
    ).hexdigest()

    def forbidden(*args, **kwargs):
        raise AssertionError("formal RCMR authority surface was called")

    monkeypatch.setattr(g0._rcmr_module, "build_d106_rcmr_2v_state", forbidden)
    monkeypatch.setattr(g0._rcmr_module, "serialize_d106_rcmr_2v_state", forbidden)
    monkeypatch.setattr(g0._rcmr_module, "deserialize_d106_rcmr_2v_state", forbidden)
    captured = []
    original_scorer = g0._score_nonformal_query

    def capture(state, context, query_plus, query_signed):
        captured.append(state)
        return original_scorer(state, context, query_plus, query_signed)

    monkeypatch.setattr(g0, "_score_nonformal_query", capture)
    result = g0._execute_fold(
        snapshot,
        fold,
        active_k=1,
        predecessor_lock=_locks(rows)[0],
        rcmr_method_lock_sha256=_method_lock_sha(),
        registry=tuple(f"tx-{index}" for index in range(6)),
        common_query_order_root_sha256=common_root,
    )
    assert result[3]["algorithm_execution_scope"] == g0.ALGORITHM_SCOPE
    assert captured
    assert {type(state).__name__ for state in captured} == {"_NonFormalRCMRState"}
    assert all(state.lifecycle_status == g0.ALGORITHM_SCOPE for state in captured)
    assert all(state.payload()["formal_authority"] is False for state in captured)
    assert all(state.payload()["phase2_validated_once"] is False for state in captured)
    assert b"VALIDATED_ONCE" not in g0._serialize_nonformal_state(captured[0])


def test_deeply_resealed_nested_tamper_is_rejected(full_synthetic_result):
    document = _document(full_synthetic_result)
    document.pop("result_receipt_sha256")
    per_k = document["per_k"][0]
    fold = per_k["fold_execution_receipts"][0]
    fold["common_query_order_root_sha256"] = "f" * 64
    fold_payload = dict(fold)
    fold_payload.pop("execution_receipt_sha256")
    fold["execution_receipt_sha256"] = hashlib.sha256(
        g0._canonical_bytes(fold_payload)
    ).hexdigest()
    per_k["fold_execution_receipts_root_sha256"] = hashlib.sha256(
        g0._canonical_bytes(
            [item["execution_receipt_sha256"] for item in per_k["fold_execution_receipts"]]
        )
    ).hexdigest()
    per_k_payload = dict(per_k)
    per_k_payload.pop("per_k_receipt_sha256")
    per_k["per_k_receipt_sha256"] = hashlib.sha256(
        g0._canonical_bytes(per_k_payload)
    ).hexdigest()
    document["canonical_execution_root_sha256"] = hashlib.sha256(
        g0._canonical_bytes([item["per_k_receipt_sha256"] for item in document["per_k"]])
    ).hexdigest()
    document["result_receipt_sha256"] = hashlib.sha256(
        g0._canonical_bytes(document)
    ).hexdigest()
    resealed = g0._canonical_bytes(document)
    with pytest.raises(g0.D106RCMRG0Error, match="fold lifecycle/top binding"):
        g0.verify_d106_rcmr_g0_synthetic_test_result_bytes(
            resealed, expected_sha256=hashlib.sha256(resealed).hexdigest()
        )


def test_external_result_anchor_is_required_and_wrong_anchor_rejects(
    full_synthetic_result,
):
    with pytest.raises(TypeError):
        g0.verify_d106_rcmr_g0_synthetic_test_result_bytes(full_synthetic_result)  # type: ignore[call-arg]
    with pytest.raises(g0.D106RCMRG0Error, match="external SHA256"):
        g0.verify_d106_rcmr_g0_synthetic_test_result_bytes(
            full_synthetic_result, expected_sha256="f" * 64
        )
    assert "not an authority issuer" in g0.verify_d106_rcmr_g0_synthetic_test_result_bytes.__doc__


def test_self_resealed_wire_sha_is_rejected_by_original_external_anchor(
    full_synthetic_result,
):
    original_anchor = hashlib.sha256(full_synthetic_result).hexdigest()
    document = _document(full_synthetic_result)
    document.pop("result_receipt_sha256")
    per_k = document["per_k"][0]
    fold = per_k["fold_execution_receipts"][0]
    fold["rcmr_wire_sha256"] = "f" * 64
    fold_payload = dict(fold)
    fold_payload.pop("execution_receipt_sha256")
    fold["execution_receipt_sha256"] = hashlib.sha256(
        g0._canonical_bytes(fold_payload)
    ).hexdigest()
    per_k["fold_execution_receipts_root_sha256"] = hashlib.sha256(
        g0._canonical_bytes(
            [item["execution_receipt_sha256"] for item in per_k["fold_execution_receipts"]]
        )
    ).hexdigest()
    per_k_payload = dict(per_k)
    per_k_payload.pop("per_k_receipt_sha256")
    per_k["per_k_receipt_sha256"] = hashlib.sha256(
        g0._canonical_bytes(per_k_payload)
    ).hexdigest()
    document["canonical_execution_root_sha256"] = hashlib.sha256(
        g0._canonical_bytes([item["per_k_receipt_sha256"] for item in document["per_k"]])
    ).hexdigest()
    document["result_receipt_sha256"] = hashlib.sha256(
        g0._canonical_bytes(document)
    ).hexdigest()
    resealed = g0._canonical_bytes(document)
    with pytest.raises(g0.D106RCMRG0Error, match="external SHA256"):
        g0.verify_d106_rcmr_g0_synthetic_test_result_bytes(
            resealed, expected_sha256=original_anchor
        )


def test_k10_changed_count_self_reseal_is_rejected_by_fold_bitmaps(
    full_synthetic_result,
):
    document = _document(full_synthetic_result)
    document.pop("result_receipt_sha256")
    per_k = document["per_k"][2]
    assert per_k["K"] == 10
    assert sum(
        fold["argmax_changed_bitmap"].count("1")
        for fold in per_k["fold_execution_receipts"]
    ) == 0
    per_k["argmax_changed_count"] = 1
    per_k_payload = dict(per_k)
    per_k_payload.pop("per_k_receipt_sha256")
    per_k["per_k_receipt_sha256"] = hashlib.sha256(
        g0._canonical_bytes(per_k_payload)
    ).hexdigest()
    document["argmax_changed_count_by_k"]["10"] = 1
    document["argmax_changed_count"] += 1
    document["zero_changed_k_values"] = []
    document["functional_gate_pass"] = True
    document["functional_gate_status"] = "G0_EVERY_K_ARGMAX_CHANGED_NO_PERFORMANCE_CLAIM"
    document["canonical_execution_root_sha256"] = hashlib.sha256(
        g0._canonical_bytes([item["per_k_receipt_sha256"] for item in document["per_k"]])
    ).hexdigest()
    document["result_receipt_sha256"] = hashlib.sha256(
        g0._canonical_bytes(document)
    ).hexdigest()
    resealed = g0._canonical_bytes(document)
    with pytest.raises(g0.D106RCMRG0Error, match="changed count/bitmap"):
        g0.verify_d106_rcmr_g0_synthetic_test_result_bytes(
            resealed, expected_sha256=hashlib.sha256(resealed).hexdigest()
        )


def test_snapshot_arrays_are_bytes_backed_and_revalidated_before_execution():
    rows = _synthetic_tap()
    snapshot = g0._snapshot_from_rows(rows, tap_receipt_sha256=_tap_receipt_sha(rows))
    for name in g0._SNAPSHOT_ARRAY_NAMES:
        value = getattr(snapshot, name)
        assert value.flags.writeable is False
        base = value
        while isinstance(base, np.ndarray):
            base = base.base
        assert isinstance(base, bytes)
    tampered = np.array(snapshot.pre_relu, copy=True)
    tampered[0, 0] += np.float32(1.0)
    tampered.setflags(write=False)
    object.__setattr__(snapshot, "pre_relu", tampered)
    with pytest.raises(g0.D106RCMRG0Error, match="root drift"):
        g0._build_fold_plan(snapshot)


def test_result_bytes_and_nested_content_are_immutable_and_receipt_checked(
    full_production_result,
):
    result = full_production_result
    with pytest.raises(TypeError):
        result[0] = 0  # type: ignore[index]
    document = _document(result)
    document["code_sha256"]["g0_executor_module"] = "0" * 64
    mutated = g0._canonical_bytes(document)
    with pytest.raises(
        g0.D106RCMRG0Error,
        match="strict-loader evidence closure drift|nested result receipt",
    ):
        g0.verify_d106_rcmr_g0_production_result_bytes(
            mutated, expected_sha256=hashlib.sha256(mutated).hexdigest()
        )


def test_every_k_must_change_and_all_zero_is_rejected(monkeypatch):
    rows = _synthetic_tap()

    def identity_fold(snapshot, fold, *, active_k, common_query_order_root_sha256, **kwargs):
        labels = tuple("tx-0" for _ in fold.query_ids)
        receipt = {
            "schema": g0.FOLD_EXECUTION_SCHEMA,
            "fold_id": fold.fold_id,
            "K": active_k,
            "argmax_changed_bitmap": "0" * len(fold.query_ids),
            "argmax_changed_bitmap_root_sha256": hashlib.sha256(
                g0._canonical_bytes(
                    {
                        "encoding": "ascii01_query_order",
                        "query_count": len(fold.query_ids),
                        "bits": "0" * len(fold.query_ids),
                    }
                )
            ).hexdigest(),
            "argmax_changed_count": 0,
            "execution_receipt_sha256": "1" * 64,
        }
        return fold.query_ids, labels, labels, receipt

    monkeypatch.setattr(g0, "_execute_fold", identity_fold)
    document = _document(
        g0.run_d106_rcmr_g0_synthetic_test(
            rows,
            registered_classes=tuple(f"tx-{index}" for index in range(6)),
            predecessor_locks=_locks(rows),
            rcmr_method_lock_sha256=_method_lock_sha(),
            synthetic_test_id="identity-only",
        )
    )
    assert document["argmax_changed_count_by_k"] == {"1": 0, "5": 0, "10": 0}
    assert document["functional_gate_pass"] is False
    assert document["zero_changed_k_values"] == [1, 5, 10]


def test_fold_query_hard_bound_is_enforced_before_execution():
    rows = _synthetic_tap()
    receivers = np.array(rows.receiver_ids, copy=True)
    days = np.array(rows.day_ids, copy=True)
    target = np.flatnonzero((receivers == "rx-1") & (days == "day-0"))[:4]
    receivers[target] = "rx-0"
    days[target] = "day-0"
    drifted = replace(rows, receiver_ids=_readonly(receivers), day_ids=_readonly(days))
    with pytest.raises(g0.D106RCMRG0Error, match="1..24"):
        g0.run_d106_rcmr_g0_synthetic_test(
            drifted,
            registered_classes=tuple(f"tx-{index}" for index in range(6)),
            predecessor_locks=_locks(drifted),
            rcmr_method_lock_sha256=_method_lock_sha(),
            synthetic_test_id="oversized-fold",
        )


def test_synthetic_three_k_lock_and_numeric_closure_are_strict():
    rows = _synthetic_tap()
    locks = _locks(rows)
    with pytest.raises(g0.D106RCMRG0Error, match="K1/K5/K10"):
        g0.run_d106_rcmr_g0_synthetic_test(
            rows,
            registered_classes=tuple(f"tx-{index}" for index in range(6)),
            predecessor_locks=locks[:2],
            rcmr_method_lock_sha256=_method_lock_sha(),
            synthetic_test_id="missing-k",
        )
    bad = (locks[0], replace(locks[1], shared_h0=0.36), locks[2])
    with pytest.raises(g0.D106RCMRG0Error, match="numeric lock drift"):
        g0.run_d106_rcmr_g0_synthetic_test(
            rows,
            registered_classes=tuple(f"tx-{index}" for index in range(6)),
            predecessor_locks=bad,
            rcmr_method_lock_sha256=_method_lock_sha(),
            synthetic_test_id="numeric-drift",
        )


def test_resource_gate_accounts_requested_arrays_without_rss_claim():
    resource = g0.audit_d106_rcmr_g0_resources()
    assert resource["fold_query_rows_hard_max"] == 24
    components = resource["known_incremental_numeric_array_analysis_estimate_bytes"]
    for field in (
        "query_indices_int64",
        "support_pool_indices_int64",
        "selected_support_indices_int64",
        "baseline_logits_float32",
        "nonformal_wire_numeric_payload",
        "query_class_scores_float64",
    ):
        assert components[field] > 0
    assert resource["incremental_numeric_array_peak_analysis_estimate_bytes"] <= resource["analysis_numeric_array_budget_bytes"]
    assert not any("upper_bound" in key for key in resource)
    assert resource["analysis_budget_is_process_rss_cap"] is False
    assert resource["process_rss_measured"] is False
    assert resource["analysis_estimate_is_measured_peak"] is False
    assert "RSS" in resource["unaccounted_overhead"]


def test_design_traceability_is_complete_and_has_no_formal_promotion_claim():
    assert len(g0.DESIGN_TRACEABILITY) == 10
    assert all(status == "implemented" for _identifier, _claim, status in g0.DESIGN_TRACEABILITY)
    assert g0.D105_CANONICAL_THREE_K_LOCK_AUTHORITY_SHA256 is None
    assert g0.ALGORITHM_SCOPE == "NON_FORMAL_TRAIN_ONLY_MECHANICAL"
