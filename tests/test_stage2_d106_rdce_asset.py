from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

import cvsrffi.stage2_d106_rdce_asset as rdce_asset
from cvsrffi.stage2_d106_phase1_tap import (
    D106Phase1TapRows,
    TAP_ARCHIVE_NAME,
    TAP_MEMBERS,
    TAP_RECEIPT_SCHEMA,
)
from cvsrffi.stage2_d106_rdce_asset import (
    ASSET_WIRE_NAME,
    CANDIDATE_ID,
    D104_RECEIVER_TX_FOUR_DAY_COUNT,
    D104_SOURCE_ROW_COUNT,
    D104_SPLIT_ID,
    D104_TX_ROW_COUNT,
    D106RDCEAssetError,
    D106RDCEBuildLock,
    D106RDCEScientificRejectReceipt,
    FORMAL_DEPLOYMENT_STATUS,
    NON_DEPLOYABLE_MATH_STATUS,
    RAW_DECODED_GRAM_ATOL,
    WIRE_MAGIC,
    Z_DIM,
    _canonical_tied_eigenspace,
    build_d106_rdce_asset,
    decode_d106_rdce_basis,
    decode_d106_rdce_tau,
    deserialize_d106_rdce_asset,
    load_d106_rdce_asset,
    save_d106_rdce_asset,
    serialize_d106_rdce_asset,
)


def _sha(character: str) -> str:
    return character * 64


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lock(*, method: str = "c") -> D106RDCEBuildLock:
    return D106RDCEBuildLock(
        method_lock_sha256=_sha(method), construction_code_sha256=_sha("e")
    )


def _make_tap(
    pre_relu: np.ndarray,
    z_dom: np.ndarray,
    tx_labels: np.ndarray,
    receiver_ids: np.ndarray,
    day_ids: np.ndarray,
    *,
    physical_ids: np.ndarray | None = None,
    receipt_override: dict[str, object] | None = None,
) -> D106Phase1TapRows:
    rows = len(pre_relu)
    arrays = {
        "pre_relu": np.ascontiguousarray(pre_relu, dtype=np.float32),
        "z_dom": np.ascontiguousarray(z_dom, dtype=np.float32),
        "tx_labels": np.asarray(tx_labels, dtype="<U16"),
        "receiver_ids": np.asarray(receiver_ids, dtype="<U16"),
        "day_ids": np.asarray(day_ids, dtype="<U16"),
        "physical_ids": (
            np.asarray([f"physical_{index:04d}" for index in range(rows)], dtype="<U20")
            if physical_ids is None
            else np.asarray(physical_ids, dtype="<U32")
        ),
        "scenario_names": np.full(rows, "leo_clear_weak", dtype="<U20"),
        "observation_ids": np.asarray(
            [f"obs_{index:04d}" for index in range(rows)], dtype="<U20"
        ),
    }
    receipt: dict[str, object] = {
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
        "row_count": rows,
        "physical_id_root_sha256": rdce_asset._tap_ordered_id_root(
            arrays["physical_ids"]
        ),
    }
    if receipt_override:
        receipt.update(receipt_override)
    return D106Phase1TapRows(
        **arrays,
        z_id=np.maximum(arrays["pre_relu"], np.float32(0.0)).astype(
            np.float32, copy=False
        ),
        receipt=receipt,
    )


def _phase1_tap() -> D106Phase1TapRows:
    rows: list[np.ndarray] = []
    domains: list[np.ndarray] = []
    tx: list[str] = []
    receiver: list[str] = []
    day: list[str] = []
    rng = np.random.default_rng(104713)
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
                    row[22] = (
                        0.17
                        + 0.008 * (receiver_index - 3) * (day_index * 2 - 3)
                        + 0.002 * centred * centred
                    )
                    row[60 + class_index] = 0.02 + 0.002 * centred
                    rows.append(row)
                    domains.append(rng.normal(size=Z_DIM).astype(np.float32))
                    tx.append(f"tx_{class_index}")
                    receiver.append(f"rx_{receiver_index}")
                    day.append(f"day_{day_index}")
    pre_relu = np.stack(rows).astype(np.float32)
    tx_array = np.asarray(tx, dtype="<U16")
    receiver_array = np.asarray(receiver, dtype="<U16")
    assert len(pre_relu) == D104_SOURCE_ROW_COUNT
    assert all(np.sum(tx_array == f"tx_{index}") == D104_TX_ROW_COUNT for index in range(6))
    assert all(
        np.sum(
            (receiver_array == f"rx_{receiver_index}")
            & (tx_array == f"tx_{class_index}")
        )
        == D104_RECEIVER_TX_FOUR_DAY_COUNT
        for receiver_index in range(7)
        for class_index in range(6)
    )
    return _make_tap(
        pre_relu,
        np.stack(domains).astype(np.float32),
        tx_array,
        receiver_array,
        np.asarray(day, dtype="<U16"),
    )


def _math_asset(tap: D106Phase1TapRows | None = None):
    result = rdce_asset._try_build_d106_rdce_asset_math(
        _phase1_tap() if tap is None else tap, build_lock=_lock()
    )
    assert not isinstance(result, D106RDCEScientificRejectReceipt)
    return result


def _formal_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, method: str = "c"
):
    archive = tmp_path / "formal-tap.npz"
    receipt = tmp_path / "formal-tap.receipt.json"
    archive.write_bytes(b"external-tap-archive-authority")
    receipt.write_bytes(b"external-tap-receipt-authority")
    archive_sha = _file_sha(archive)
    receipt_sha = _file_sha(receipt)
    tap = _phase1_tap()
    loader_result = replace(
        tap,
        receipt={**tap.receipt, "tap_archive_sha256": archive_sha},
    )
    seen: list[tuple[Path, Path, str, str]] = []

    def _data_loader(
        archive_path: str | Path,
        receipt_path: str | Path,
        *,
        expected_archive_sha256: str,
        expected_receipt_sha256: str,
    ) -> D106Phase1TapRows:
        seen.append(
            (
                Path(archive_path),
                Path(receipt_path),
                expected_archive_sha256,
                expected_receipt_sha256,
            )
        )
        return loader_result

    monkeypatch.setattr(rdce_asset, "load_d106_phase1_ls_tap", _data_loader)
    built = build_d106_rdce_asset(
        archive,
        receipt,
        expected_tap_archive_sha256=archive_sha,
        expected_tap_receipt_sha256=receipt_sha,
        build_lock=_lock(method=method),
    )
    return built, archive, receipt, archive_sha, receipt_sha, seen


def _wire_sha(wire: bytes) -> str:
    return hashlib.sha256(wire).hexdigest()


def _replace_header(wire: bytes, replacement: bytes) -> bytes:
    old_size = struct.unpack(">I", wire[len(WIRE_MAGIC) : len(WIRE_MAGIC) + 4])[0]
    offset = len(WIRE_MAGIC) + 4
    return WIRE_MAGIC + struct.pack(">I", len(replacement)) + replacement + wire[offset + old_size :]


def test_manual_rows_are_math_only_and_public_builder_requires_loader() -> None:
    math_asset = _math_asset()
    assert math_asset.deployment_status == NON_DEPLOYABLE_MATH_STATUS
    assert not math_asset.is_formal_deployable
    with pytest.raises(D106RDCEAssetError, match="loader-authorized"):
        serialize_d106_rdce_asset(math_asset)
    with pytest.raises(TypeError):
        build_d106_rdce_asset(_phase1_tap(), build_lock=_lock())  # type: ignore[call-arg]


def test_formal_asset_uses_external_sha_bound_data_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset, archive, receipt, archive_sha, receipt_sha, seen = _formal_asset(
        tmp_path, monkeypatch
    )
    assert asset.deployment_status == FORMAL_DEPLOYMENT_STATUS
    assert asset.is_formal_deployable
    assert asset.tap_sha256 == archive_sha
    assert asset.tap_receipt_sha256 == receipt_sha
    assert seen == [(archive, receipt, archive_sha, receipt_sha)]
    with pytest.raises(D106RDCEAssetError, match="external D106 tap SHA256"):
        build_d106_rdce_asset(
            archive,
            receipt,
            expected_tap_archive_sha256=_sha("0"),
            expected_tap_receipt_sha256=receipt_sha,
            build_lock=_lock(),
        )


def test_d104_geometry_and_actual_tap_hash_fail_closed() -> None:
    tap = _phase1_tap()
    forged = replace(tap, receipt={**tap.receipt, "array_sha256": {}})
    with pytest.raises(D106RDCEAssetError, match="array SHA256"):
        rdce_asset._try_build_d106_rdce_asset_math(forged, build_lock=_lock())

    moved = tap.receiver_ids.copy()
    index = int(np.flatnonzero((tap.receiver_ids == "rx_0") & (tap.tx_labels == "tx_0"))[0])
    moved[index] = "rx_1"
    result = rdce_asset._try_build_d106_rdce_asset_math(
        _make_tap(tap.pre_relu, tap.z_dom, tap.tx_labels, moved, tap.day_ids),
        build_lock=_lock(),
    )
    assert isinstance(result, D106RDCEScientificRejectReceipt)
    assert result.reason == "d104_receiver_tx_four_day_count_not_14"

    short = _make_tap(
        tap.pre_relu[:-1], tap.z_dom[:-1], tap.tx_labels[:-1], tap.receiver_ids[:-1], tap.day_ids[:-1]
    )
    short_result = rdce_asset._try_build_d106_rdce_asset_math(short, build_lock=_lock())
    assert isinstance(short_result, D106RDCEScientificRejectReceipt)
    assert short_result.reason == "d104_ls_row_count_not_588"


def test_closed_basis_tau_raw_gram_and_canonical_eigenspace() -> None:
    asset = _math_asset()
    tap = _phase1_tap()
    normalized = tap.z_id.astype(np.float64)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    basis = decode_d106_rdce_basis(asset)
    per_class = []
    for label in sorted(set(tap.tx_labels.tolist())):
        local = normalized[tap.tx_labels == label]
        residual = local - np.mean(local, axis=0, dtype=np.float64)
        per_class.append(np.sum(np.square(residual @ basis.T), axis=0) / float(len(local) - 1))
    assert np.allclose(decode_d106_rdce_tau(asset), np.mean(per_class, axis=0), rtol=0.025, atol=2e-8)

    codes = np.zeros((3, Z_DIM), dtype=np.int8)
    codes[0, 0], codes[1, 0], codes[1, 1], codes[2, 2] = 127, 64, 127, 127
    scales = np.full(3, np.float16(1.0 / 127.0), dtype=np.float16)
    raw_gram = (codes.astype(np.float64) * scales[:, None]) @ (codes.astype(np.float64) * scales[:, None]).T
    assert abs(raw_gram[0, 1]) > RAW_DECODED_GRAM_ATOL
    with pytest.raises(D106RDCEAssetError, match="raw Gram"):
        rdce_asset._orthogonal_closure(codes, scales)

    first = np.zeros((Z_DIM, 2), dtype=np.float64)
    first[0, 0], first[1, 1] = 1.0, 1.0
    rotation = np.asarray([[0.6, -0.8], [0.8, 0.6]])
    assert np.allclose(
        np.stack(_canonical_tied_eigenspace(first), axis=1),
        np.stack(_canonical_tied_eigenspace(first @ rotation), axis=1),
    )


def test_formal_wire_requires_pinned_lineage_canonical_header_and_atomic_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset, _archive, _receipt, _archive_sha, _receipt_sha, _seen = _formal_asset(
        tmp_path, monkeypatch
    )
    wire = serialize_d106_rdce_asset(asset)
    replay = deserialize_d106_rdce_asset(
        wire, expected_wire_sha256=_wire_sha(wire), expected_lineage=asset.lineage
    )
    assert replay.is_formal_deployable
    header_size = struct.unpack(">I", wire[len(WIRE_MAGIC) : len(WIRE_MAGIC) + 4])[0]
    start = len(WIRE_MAGIC) + 4
    header = json.loads(wire[start : start + header_size].decode("utf-8"))
    noncanonical = _replace_header(
        wire, json.dumps(header, sort_keys=True, indent=2).encode("utf-8")
    )
    with pytest.raises(D106RDCEAssetError, match="original canonical"):
        deserialize_d106_rdce_asset(
            noncanonical,
            expected_wire_sha256=_wire_sha(noncanonical),
            expected_lineage=asset.lineage,
        )
    with pytest.raises(D106RDCEAssetError, match="trust anchor"):
        deserialize_d106_rdce_asset(
            wire, expected_wire_sha256=_sha("0"), expected_lineage=asset.lineage
        )

    destination = tmp_path / "asset"
    saved = save_d106_rdce_asset(asset, destination)
    assert {entry.name for entry in destination.iterdir()} == {ASSET_WIRE_NAME}
    assert load_d106_rdce_asset(
        destination,
        expected_wire_sha256=saved["wire_sha256"],
        expected_lineage=asset.lineage,
    ).is_formal_deployable
    with pytest.raises(D106RDCEAssetError, match="absent"):
        save_d106_rdce_asset(asset, destination)
