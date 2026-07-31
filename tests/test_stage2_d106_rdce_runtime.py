from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

import cvsrffi.stage2_d106_rdce_asset as rdce_asset
import cvsrffi.stage2_d106_rdce_runtime as rdce_runtime
from cvsrffi.stage2_d106_phase1_tap import (
    D106Phase1TapRows,
    TAP_ARCHIVE_NAME,
    TAP_MEMBERS,
    TAP_RECEIPT_SCHEMA,
)
from cvsrffi.stage2_d106_rdce_asset import (
    CANDIDATE_ID,
    D104_SPLIT_ID,
    D106RDCEBuildLock,
    FORMAL_DEPLOYMENT_STATUS,
    NON_DEPLOYABLE_MATH_STATUS,
    Z_DIM,
    build_d106_rdce_asset,
)
from cvsrffi.stage2_d106_rdce_runtime import (
    D106RDCERuntimeError,
    D106RDCESupportRows,
    K1_ATTENUATION,
    RUNTIME_WIRE_MAGIC,
    ROW_AUTHORITY_SCHEMA,
    audit_d106_rdce_runtime,
    deserialize_d106_rdce_runtime,
    fit_d106_rdce_runtime,
    load_d106_rdce_row_authority,
    prepare_d106_rdce_scoring_context,
    serialize_d106_rdce_runtime,
    transform_d106_rdce_query,
)
from cvsrffi.stage2_lpo_rc_qknn import TypedValidatedOnceP2SplitHandle
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
)


def _sha(character: str) -> str:
    return character * 64


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lock() -> D106RDCEBuildLock:
    return D106RDCEBuildLock(
        method_lock_sha256=_sha("c"), construction_code_sha256=_sha("e")
    )


def _source_tap() -> D106Phase1TapRows:
    rows: list[np.ndarray] = []
    tx: list[str] = []
    receiver: list[str] = []
    day: list[str] = []
    for receiver_index in range(7):
        for day_index in range(4):
            for class_index in range(6):
                count = 4 if (receiver_index + day_index + class_index) % 2 == 0 else 3
                for sample_index in range(count):
                    row = np.zeros(Z_DIM, dtype=np.float32)
                    row[class_index] = 1.2
                    centred = float(sample_index) - float(count - 1) / 2.0
                    row[20] = 0.22 + 0.018 * (receiver_index - 3) + 0.004 * centred
                    row[21] = 0.19 + 0.016 * day_index + 0.003 * centred
                    row[22] = 0.17 + 0.008 * (receiver_index - 3) * (day_index * 2 - 3)
                    rows.append(row)
                    tx.append(f"tx_{class_index}")
                    receiver.append(f"rx_{receiver_index}")
                    day.append(f"day_{day_index}")
    pre_relu = np.stack(rows).astype(np.float32)
    arrays = {
        "pre_relu": pre_relu,
        "z_dom": np.zeros_like(pre_relu),
        "tx_labels": np.asarray(tx, dtype="<U16"),
        "receiver_ids": np.asarray(receiver, dtype="<U16"),
        "day_ids": np.asarray(day, dtype="<U16"),
        "physical_ids": np.asarray(
            [f"physical_{index:04d}" for index in range(len(pre_relu))], dtype="<U20"
        ),
        "scenario_names": np.full(len(pre_relu), "leo_clear_weak", dtype="<U20"),
        "observation_ids": np.asarray(
            [f"obs_{index:04d}" for index in range(len(pre_relu))], dtype="<U20"
        ),
    }
    receipt = {
        "schema": TAP_RECEIPT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "split_id": D104_SPLIT_ID,
        "protocol_schema": "p2_min_v1",
        "selected_iq_archive_sha256": _sha("6"),
        "selected_iq_receipt_sha256": _sha("1"),
        "storage_validator_receipt_sha256": _sha("7"),
        "storage_validation_binding": {
            "schema": rdce_asset.LS_IQ_VALIDATOR_SCHEMA,
            "storage_validation_root_sha256": _sha("8"),
            "selected_content_root_sha256": _sha("9"),
            "all_8400x3_storage_semantics_verified": True,
        },
        "extraction_binding": {
            "schema": rdce_asset.LS_IQ_RECEIPT_SCHEMA,
            "row_count": len(pre_relu),
            "selection_salt_sha256": _sha("0"),
            "selected_content_root_sha256": _sha("9"),
            "input_ls_archive_sha256": _sha("a"),
            "execution_root_sha256": _sha("b"),
        },
        "input_ls_archive_sha256": _sha("a"),
        "checkpoint_sha256": _sha("a"),
        "runtime_sha256": _sha("b"),
        "tap_archive_name": TAP_ARCHIVE_NAME,
        "tap_archive_sha256": _sha("d"),
        "tap_archive_members": list(TAP_MEMBERS),
        "array_sha256": {
            name: rdce_asset._tap_array_sha256(value)
            for name, value in arrays.items()
        },
        "row_count": len(pre_relu),
        "physical_id_root_sha256": rdce_asset._tap_ordered_id_root(arrays["physical_ids"]),
    }
    return D106Phase1TapRows(
        **arrays,
        z_id=np.maximum(pre_relu, np.float32(0.0)).astype(np.float32, copy=False),
        receipt=receipt,
    )


def _formal_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    archive = tmp_path / "tap.npz"
    receipt = tmp_path / "tap.receipt.json"
    archive.write_bytes(b"formal-tap-archive")
    receipt.write_bytes(b"formal-tap-receipt")
    archive_sha, receipt_sha = _file_sha(archive), _file_sha(receipt)
    tap = _source_tap()
    loader_tap = replace(tap, receipt={**tap.receipt, "tap_archive_sha256": archive_sha})

    def _loader(*_args, **_kwargs):
        return loader_tap

    monkeypatch.setattr(rdce_asset, "load_d106_phase1_ls_tap", _loader)
    return build_d106_rdce_asset(
        archive,
        receipt,
        expected_tap_archive_sha256=archive_sha,
        expected_tap_receipt_sha256=receipt_sha,
        build_lock=_lock(),
    )


def _math_asset():
    asset = rdce_asset._try_build_d106_rdce_asset_math(_source_tap(), build_lock=_lock())
    assert not isinstance(asset, rdce_asset.D106RDCEScientificRejectReceipt)
    return asset


def _qknn_lock(k_shot: int) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        active_k=k_shot,
        student_nu=4.0,
        kernel_effective_dim=16,
        kernel_volume_gamma=1.0,
        shared_h0=0.35,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=1.0,
        phase1_lodo_receipt_sha256=_sha("8"),
        quantization_margin_audit_sha256=_sha("9"),
    )


def _support_rows(k_shot: int) -> D106RDCESupportRows:
    classes = ("registered_a", "registered_b", "registered_c")
    rows: list[np.ndarray] = []
    labels: list[str] = []
    physical_ids: list[str] = []
    for class_index, class_name in enumerate(classes):
        for sample_index in range(k_shot):
            row = np.zeros(Z_DIM, dtype=np.float32)
            row[class_index] = 1.0
            centred = float(sample_index) - float(k_shot - 1) / 2.0
            row[20] = (0.012 + 0.005 * class_index) * centred
            row[21] = (0.006 + 0.002 * class_index) * centred
            row[22] = (0.003 + 0.001 * class_index) * centred * centred
            rows.append(row)
            labels.append(class_name)
            physical_ids.append(f"p{class_index}-shot{sample_index}")
    support = np.stack(rows).astype(np.float32)
    label_array = np.asarray(labels, dtype="<U16")
    physical_array = np.asarray(physical_ids, dtype="<U32")
    bank = build_typed_zid_support_bank(
        support, label_array.tolist(), classes, config=_qknn_lock(k_shot)
    )
    physical = tuple(physical_array.tolist())
    handle = TypedValidatedOnceP2SplitHandle(
        capsule_id=_sha("2"),
        split_id=_sha("3"),
        validator_receipt_sha256=_sha("4"),
        support_physical_root_sha256=rdce_runtime._physical_root(physical),
        query_physical_root_sha256=_sha("5"),
        support_query_disjoint=True,
    )
    return D106RDCESupportRows(
        support_z_id=support,
        support_labels=label_array,
        support_physical_ids=physical_array,
        qknn_bank=bank,
        split_handle=handle,
        row_id=f"row-k{k_shot}",
        seed=104713,
    )


def _write_row_authority(
    tmp_path: Path, support: D106RDCESupportRows, *, name: str = "row.json"
):
    physical = tuple(str(value) for value in support.support_physical_ids.tolist())
    document = {
        "schema": ROW_AUTHORITY_SCHEMA,
        "capsule_id": support.split_handle.capsule_id,
        "split_id": support.split_handle.split_id,
        "validator_receipt_sha256": support.split_handle.validator_receipt_sha256,
        "row_id": support.row_id,
        "seed": support.seed,
        "active_k": support.qknn_bank.active_k,
        "registered_classes": list(support.qknn_bank.classes),
        "support_z_id_receipt": rdce_runtime._array_receipt(support.support_z_id),
        "support_labels_receipt": rdce_runtime._array_receipt(support.support_labels),
        "support_physical_ids_receipt": rdce_runtime._array_receipt(support.support_physical_ids),
        "ordered_support_physical_ids_sha256": rdce_runtime._ordered_physical_root(physical),
        "qknn_bank_sha256": support.qknn_bank.bank_receipt_sha256,
        "support_physical_root_sha256": support.split_handle.support_physical_root_sha256,
        "query_physical_root_sha256": support.split_handle.query_physical_root_sha256,
        "protocol_schema": support.split_handle.protocol_schema,
        "phase2_data_status": support.split_handle.phase2_data_status,
        "support_query_disjoint": True,
    }
    path = tmp_path / name
    path.write_bytes(rdce_runtime._canonical_bytes(document))
    authority = load_d106_rdce_row_authority(
        path, expected_authority_sha256=_file_sha(path)
    )
    return authority, path, document


def _wire_sha(wire: bytes) -> str:
    return hashlib.sha256(wire).hexdigest()


def _replace_wire_header(wire: bytes, replacement: bytes) -> bytes:
    old_size = struct.unpack(">I", wire[len(RUNTIME_WIRE_MAGIC) : len(RUNTIME_WIRE_MAGIC) + 4])[0]
    start = len(RUNTIME_WIRE_MAGIC) + 4
    return RUNTIME_WIRE_MAGIC + struct.pack(">I", len(replacement)) + replacement + wire[start + old_size :]


def test_only_loader_origin_asset_and_row_authority_produce_deployable_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = _formal_asset(tmp_path, monkeypatch)
    support = _support_rows(5)
    authority, _path, _document = _write_row_authority(tmp_path, support)
    state = fit_d106_rdce_runtime(asset, support, row_authority=authority)
    assert asset.deployment_status == FORMAL_DEPLOYMENT_STATUS
    assert state.deployment_status == FORMAL_DEPLOYMENT_STATUS
    assert state.is_formal_deployable
    assert state.row_authority_sha256 == authority.authority_document_sha256

    math_state = rdce_runtime._try_fit_d106_rdce_runtime_math(_math_asset(), support)
    assert not isinstance(math_state, rdce_asset.D106RDCEScientificRejectReceipt)
    assert math_state.deployment_status == NON_DEPLOYABLE_MATH_STATUS
    with pytest.raises(D106RDCERuntimeError, match="loader-authorized"):
        serialize_d106_rdce_runtime(math_state)
    with pytest.raises(TypeError):
        fit_d106_rdce_runtime(asset, support)  # type: ignore[call-arg]


@pytest.mark.parametrize("k_shot", (1, 5, 10))
def test_formal_kshot_rows_keep_balanced_registered_class_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, k_shot: int
) -> None:
    asset = _formal_asset(tmp_path, monkeypatch)
    support = _support_rows(k_shot)
    authority, _path, _document = _write_row_authority(
        tmp_path, support, name=f"row-k{k_shot}.json"
    )
    state = fit_d106_rdce_runtime(asset, support, row_authority=authority)
    assert state.active_k == k_shot
    assert state.registered_class_count == 3
    if k_shot == 1:
        assert np.array_equal(
            state.attenuation,
            np.full(3, np.float16(K1_ATTENUATION), dtype=np.float16),
        )


def test_row_authority_fails_feature_label_row_seed_and_external_document_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = _formal_asset(tmp_path, monkeypatch)
    support = _support_rows(5)
    authority, path, document = _write_row_authority(tmp_path, support)

    changed_features = support.support_z_id.copy()
    changed_features[0, 20] += np.float32(0.1)
    feature_bank = build_typed_zid_support_bank(
        changed_features,
        support.support_labels.tolist(),
        support.qknn_bank.classes,
        config=support.qknn_bank.config,
    )
    with pytest.raises(D106RDCERuntimeError, match="row authority"):
        fit_d106_rdce_runtime(
            asset,
            replace(support, support_z_id=changed_features, qknn_bank=feature_bank),
            row_authority=authority,
        )

    changed_labels = np.asarray(
        [
            {"registered_a": "registered_b", "registered_b": "registered_a"}.get(value, value)
            for value in support.support_labels.tolist()
        ],
        dtype="<U16",
    )
    label_bank = build_typed_zid_support_bank(
        support.support_z_id,
        changed_labels.tolist(),
        support.qknn_bank.classes,
        config=support.qknn_bank.config,
    )
    with pytest.raises(D106RDCERuntimeError, match="row authority"):
        fit_d106_rdce_runtime(
            asset,
            replace(support, support_labels=changed_labels, qknn_bank=label_bank),
            row_authority=authority,
        )
    with pytest.raises(D106RDCERuntimeError, match="row authority"):
        fit_d106_rdce_runtime(
            asset, replace(support, row_id="other-row"), row_authority=authority
        )
    with pytest.raises(D106RDCERuntimeError, match="row authority"):
        fit_d106_rdce_runtime(
            asset, replace(support, seed=104714), row_authority=authority
        )
    with pytest.raises(D106RDCERuntimeError, match="external SHA256"):
        load_d106_rdce_row_authority(path, expected_authority_sha256=_sha("0"))

    noncanonical = tmp_path / "noncanonical-row.json"
    noncanonical.write_bytes(json.dumps(document, sort_keys=True, indent=2).encode("utf-8"))
    with pytest.raises(D106RDCERuntimeError, match="original canonical"):
        load_d106_rdce_row_authority(
            noncanonical, expected_authority_sha256=_file_sha(noncanonical)
        )


@pytest.mark.parametrize(
    ("handle_field", "replacement_value"),
    (
        ("capsule_id", _sha("a")),
        ("split_id", _sha("b")),
        ("validator_receipt_sha256", _sha("c")),
        ("query_physical_root_sha256", _sha("d")),
    ),
)
def test_row_authority_rejects_each_split_handle_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    handle_field: str,
    replacement_value: str,
) -> None:
    asset = _formal_asset(tmp_path, monkeypatch)
    support = _support_rows(5)
    authority, _path, _document = _write_row_authority(tmp_path, support)
    changed_handle = replace(
        support.split_handle, **{handle_field: replacement_value}
    )
    with pytest.raises(D106RDCERuntimeError, match="row authority"):
        fit_d106_rdce_runtime(
            asset,
            replace(support, split_handle=changed_handle),
            row_authority=authority,
        )


def test_row_authority_rejects_ordered_physical_id_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = _formal_asset(tmp_path, monkeypatch)
    support = _support_rows(5)
    authority, _path, _document = _write_row_authority(tmp_path, support)
    changed_physical_ids = support.support_physical_ids.copy()
    changed_physical_ids[[0, 1]] = changed_physical_ids[[1, 0]]
    with pytest.raises(D106RDCERuntimeError, match="row authority"):
        fit_d106_rdce_runtime(
            asset,
            replace(support, support_physical_ids=changed_physical_ids),
            row_authority=authority,
        )


@pytest.mark.parametrize(
    ("authority_field", "replacement_value"),
    (
        ("active_k", 10),
        (
            "registered_classes",
            ["registered_c", "registered_b", "registered_a"],
        ),
        ("qknn_bank_sha256", _sha("f")),
    ),
)
def test_row_authority_rejects_k_registry_and_bank_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority_field: str,
    replacement_value: object,
) -> None:
    asset = _formal_asset(tmp_path, monkeypatch)
    support = _support_rows(5)
    _authority, _path, document = _write_row_authority(tmp_path, support)
    changed_document = {**document, authority_field: replacement_value}
    changed_path = tmp_path / f"row-{authority_field}.json"
    changed_path.write_bytes(rdce_runtime._canonical_bytes(changed_document))
    changed_authority = load_d106_rdce_row_authority(
        changed_path, expected_authority_sha256=_file_sha(changed_path)
    )
    with pytest.raises(D106RDCERuntimeError, match="row authority"):
        fit_d106_rdce_runtime(
            asset, support, row_authority=changed_authority
        )


def test_runtime_wire_replay_rejects_replacement_noncanonical_duplicate_and_trailing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = _formal_asset(tmp_path, monkeypatch)
    support = _support_rows(10)
    authority, _path, _document = _write_row_authority(tmp_path, support)
    state = fit_d106_rdce_runtime(asset, support, row_authority=authority)
    wire = serialize_d106_rdce_runtime(state)
    replay = deserialize_d106_rdce_runtime(
        wire,
        asset=asset,
        expected_wire_sha256=_wire_sha(wire),
        expected_binding=state.binding,
    )
    assert replay.is_formal_deployable
    assert np.array_equal(replay.attenuation, state.attenuation)
    header_size = struct.unpack(">I", wire[len(RUNTIME_WIRE_MAGIC) : len(RUNTIME_WIRE_MAGIC) + 4])[0]
    start = len(RUNTIME_WIRE_MAGIC) + 4
    raw_header = wire[start : start + header_size]
    header = json.loads(raw_header.decode("utf-8"))
    noncanonical = _replace_wire_header(
        wire, json.dumps(header, sort_keys=True, indent=2).encode("utf-8")
    )
    duplicate = _replace_wire_header(
        wire, b'{"schema":"ignored",' + raw_header[1:]
    )
    for malformed in (noncanonical, duplicate):
        with pytest.raises(D106RDCERuntimeError, match="original canonical"):
            deserialize_d106_rdce_runtime(
                malformed,
                asset=asset,
                expected_wire_sha256=_wire_sha(malformed),
                expected_binding=state.binding,
            )
    trailing = wire + b"\x00"
    with pytest.raises(D106RDCERuntimeError, match="payload length/trailing"):
        deserialize_d106_rdce_runtime(
            trailing,
            asset=asset,
            expected_wire_sha256=_wire_sha(trailing),
            expected_binding=state.binding,
        )

    other_support = _support_rows(5)
    other_authority, _other_path, _other_document = _write_row_authority(
        tmp_path, other_support, name="other-row.json"
    )
    other_state = fit_d106_rdce_runtime(
        asset, other_support, row_authority=other_authority
    )
    other_wire = serialize_d106_rdce_runtime(other_state)
    with pytest.raises(D106RDCERuntimeError, match="expected typed binding"):
        deserialize_d106_rdce_runtime(
            other_wire,
            asset=asset,
            expected_wire_sha256=_wire_sha(other_wire),
            expected_binding=state.binding,
        )


def test_ephemeral_context_decodes_basis_once_and_resource_scope_is_narrow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = _formal_asset(tmp_path, monkeypatch)
    support = _support_rows(5)
    authority, _path, _document = _write_row_authority(tmp_path, support)
    state = fit_d106_rdce_runtime(asset, support, row_authority=authority)
    real_decode = rdce_runtime.decode_d106_rdce_basis
    decoded_shapes: list[tuple[int, int]] = []

    def _spy_decode(value):
        basis = real_decode(value)
        decoded_shapes.append(basis.shape)
        return basis

    monkeypatch.setattr(rdce_runtime, "decode_d106_rdce_basis", _spy_decode)
    context = prepare_d106_rdce_scoring_context(state)
    query = support.support_z_id[:2]
    first = transform_d106_rdce_query(state, query, context=context)
    second = transform_d106_rdce_query(state, query, context=context)
    assert decoded_shapes == [(3, Z_DIM)]
    assert np.allclose(first, second, rtol=0.0, atol=1e-7)
    transform_d106_rdce_query(state, query)
    assert decoded_shapes == [(3, Z_DIM), (3, Z_DIM)]

    receipt = audit_d106_rdce_runtime(state)
    assert receipt["projection_mac_per_row"] == 960
    assert receipt["basis_decode_calls_per_scoring_context"] == 1
    assert receipt["basis_decode_calls_per_transform_with_context"] == 0
    assert receipt["basis_decode_calls_per_transform_without_context"] == 1
    assert receipt["runtime_wire_persistent_metric_eigenvalue"] is False
    assert "estimated_phi_multiply_accumulates" not in receipt

    with pytest.raises(D106RDCERuntimeError, match="immutable"):
        context.basis_content_sha256 = _sha("f")
    with pytest.raises(ValueError):
        context.basis.setflags(write=True)

    forged_basis = context.basis.copy()
    forged_basis[0, 0] += np.float64(0.125)
    object.__setattr__(
        context, "_basis_bytes", forged_basis.astype(np.float64).tobytes(order="C")
    )
    object.__setattr__(
        context,
        "_basis_content_sha256",
        rdce_runtime._basis_content_digest(forged_basis),
    )
    with pytest.raises(D106RDCERuntimeError, match="mint record"):
        transform_d106_rdce_query(state, query, context=context)

    with pytest.raises(D106RDCERuntimeError, match="loader-origin"):
        rdce_runtime._D106RDCEEphemeralScoringContext(
            runtime_receipt_sha256=state.runtime_receipt_sha256,
            asset_binding_sha256=state.asset.binding_sha256,
            basis=np.eye(3, Z_DIM, dtype=np.float64),
            attenuation=state.attenuation_fp16.astype(np.float64),
            basis_content_sha256=_sha("e"),
            _loader_token=None,
        )
