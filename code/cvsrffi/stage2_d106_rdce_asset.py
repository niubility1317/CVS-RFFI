"""Immutable D106 RDCE Phase1 asset with strict tap and wire bindings.

The public builder accepts only archive/receipt paths plus external SHA256
anchors, then calls the DATA-owned ``load_d106_phase1_ls_tap`` boundary.  Its
private typed-row math helper is deliberately non-deployable.  A formal asset
wire retains only INT8 arrays, FP16 scales, and immutable loader lineage; it
contains no source rows, IDs, names, or FP32 source feature sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
from typing import Any, Mapping
import uuid

import numpy as np

from .stage2_d106_phase1_tap import (
    D106Phase1TapRows,
    LS_IQ_RECEIPT_SCHEMA,
    LS_IQ_VALIDATOR_SCHEMA,
    TAP_ARCHIVE_NAME,
    TAP_MEMBERS,
    TAP_RECEIPT_SCHEMA,
    load_d106_phase1_ls_tap,
)


Z_DIM = 160
RDCE_RANK = 3
RECEIVER_DAY_CELL_COUNT = 28
D104_SOURCE_ROW_COUNT = 588
D104_SOURCE_CLASS_COUNT = 6
D104_RECEIVER_COUNT = 7
D104_DAY_COUNT = 4
D104_TX_ROW_COUNT = 98
D104_RECEIVER_TX_FOUR_DAY_COUNT = 14
D104_SPLIT_ID = "d104_source_seed104713_v2"
D104_CELL_MIN_SAMPLES = 2
D104_CELL_MAX_SAMPLES = 4
INT8_MAX = 127
EPSILON = 1.0e-12
MIN_SPECTRUM = 1.0e-10
MIN_BASIS_SINGULAR_VALUE = 1.0e-7
# A unit row quantized with a per-row INT8 scale has a worst-case L2 error
# below sqrt(160)/(2*127).  0.125 is a fixed, conservative bound on its raw
# three-by-three Gram residual; closure is forbidden outside this envelope.
RAW_DECODED_GRAM_ATOL = 0.125

SCHEMA = "cvs.phase1.d106.rdce_gtsm_asset.v3"
BUILD_LOCK_SCHEMA = "cvs.phase1.d106.rdce_gtsm_build_lock.v1"
LINEAGE_SCHEMA = "cvs.phase1.d106.rdce_gtsm_lineage.v1"
TAP_AUTHORITY_SCHEMA = "cvs.phase1.d106.rdce_gtsm_tap_authority.v1"
TAP_CONTENT_ROOT_SCHEMA = "cvs.phase1.d106.rdce_gtsm_tap_content_root.v1"
WIRE_SCHEMA = "cvs.phase1.d106.rdce_gtsm_asset_wire.v3"
REJECT_SCHEMA = "cvs.phase1.d106.rdce_gtsm_scientific_reject.v3"
CANDIDATE_ID = "D106-RDCE/GTSM-r3-SCATTER02"
WIRE_MAGIC = b"CVSD106RDCE\x00\x03"
ASSET_WIRE_NAME = "d106_rdce_gtsm.asset.wire"
REJECT_NAME = "d106_rdce_gtsm.scientific_reject.json"
FORMAL_DEPLOYMENT_STATUS = "FORMAL_DEPLOYABLE"
NON_DEPLOYABLE_MATH_STATUS = "NON_DEPLOYABLE_MATH_ONLY"
_TAP_LOADER_TOKEN = object()


class D106RDCEAssetError(ValueError):
    """Raised when a D106 Phase1 asset boundary or wire drifts."""


class D106RDCEScientificRejectError(D106RDCEAssetError):
    """Raised by the strict builder for a scientific geometry reject."""

    def __init__(self, receipt: "D106RDCEScientificRejectReceipt") -> None:
        self.receipt = receipt
        super().__init__(f"D106 scientific reject: {receipt.reason}")


class _ScientificReject(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if type(value) is not str:
        raise D106RDCEAssetError(f"{name} must be an exact string SHA256")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise D106RDCEAssetError(f"{name} must be a lowercase SHA256")
    return value


def _require_d104_split_id(value: Any) -> str:
    if type(value) is not str or value != D104_SPLIT_ID:
        raise D106RDCEAssetError(
            f"split_id must be the frozen {D104_SPLIT_ID!r} binding"
        )
    return value


def _readonly(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _tap_array_sha256(value: np.ndarray) -> str:
    """Match the strict D106 tap receipt's typed-array digest exactly."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise D106RDCEAssetError("tap object arrays are forbidden")
    if array.dtype.kind in {"U", "S"}:
        descriptor = {"dtype": "utf8-string", "shape": list(array.shape)}
        body = _canonical_bytes(array.astype(str).tolist())
    else:
        array = np.ascontiguousarray(array)
        descriptor = {"dtype": array.dtype.str, "shape": list(array.shape)}
        body = array.tobytes(order="C")
    return _sha256_bytes(_canonical_bytes(descriptor) + b"\0" + body)


def _tap_ordered_id_root(values: np.ndarray) -> str:
    encoded = json.dumps(
        [str(value) for value in values.tolist()],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _require_float32_matrix(
    value: Any, name: str, rows: int | None = None
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise D106RDCEAssetError(f"{name} must be a numpy float32 array")
    if (
        value.dtype != np.float32
        or value.ndim != 2
        or value.shape[1] != Z_DIM
        or (rows is not None and value.shape[0] != rows)
        or value.shape[0] < 1
        or not np.isfinite(value).all()
    ):
        raise D106RDCEAssetError(f"{name} must be finite float32 [N,{Z_DIM}]")
    return np.ascontiguousarray(value)


def _token_bytes(value: Any, name: str) -> bytes:
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return f"i:{int(value)}".encode("ascii")
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise D106RDCEAssetError(f"{name} bytes must be UTF-8") from error
    elif isinstance(value, str):
        text = value
    else:
        raise D106RDCEAssetError(
            f"{name} values must be integer, UTF-8 bytes, or unicode strings"
        )
    if not text:
        raise D106RDCEAssetError(f"{name} values must be non-empty")
    return b"s:" + text.encode("utf-8")


def _typed_tokens(value: Any, name: str, rows: int) -> tuple[bytes, ...]:
    if not isinstance(value, np.ndarray) or value.ndim != 1 or len(value) != rows:
        raise D106RDCEAssetError(
            f"{name} must be a one-dimensional typed array aligned to z_id"
        )
    if value.dtype.kind not in {"i", "u", "U", "S"}:
        raise D106RDCEAssetError(f"{name} must use an integer or string numpy dtype")
    return tuple(
        _token_bytes(item.item() if isinstance(item, np.generic) else item, name)
        for item in value
    )


@dataclass(frozen=True, slots=True)
class D106RDCEBuildLock:
    """Method/code hashes supplied through one typed build-lock object."""

    method_lock_sha256: str
    construction_code_sha256: str
    schema: str = BUILD_LOCK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BUILD_LOCK_SCHEMA:
            raise D106RDCEAssetError("D106 build-lock schema drift")
        object.__setattr__(
            self,
            "method_lock_sha256",
            _require_sha256(self.method_lock_sha256, "method_lock_sha256"),
        )
        object.__setattr__(
            self,
            "construction_code_sha256",
            _require_sha256(
                self.construction_code_sha256, "construction_code_sha256"
            ),
        )


@dataclass(frozen=True, slots=True)
class _D106RDCETapAuthority:
    """Private proof that the public builder used the SHA-bound D106 loader."""

    archive_sha256: str
    receipt_sha256: str
    schema: str = TAP_AUTHORITY_SCHEMA
    loader: str = "load_d106_phase1_ls_tap"

    def __post_init__(self) -> None:
        if self.schema != TAP_AUTHORITY_SCHEMA or self.loader != "load_d106_phase1_ls_tap":
            raise D106RDCEAssetError("D106 tap authority schema/loader drift")
        object.__setattr__(
            self,
            "archive_sha256",
            _require_sha256(self.archive_sha256, "tap authority archive_sha256"),
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _require_sha256(self.receipt_sha256, "tap authority receipt_sha256"),
        )

    @property
    def authority_sha256(self) -> str:
        return _sha256_bytes(
            _canonical_bytes(
                {
                    "schema": self.schema,
                    "loader": self.loader,
                    "archive_sha256": self.archive_sha256,
                    "receipt_sha256": self.receipt_sha256,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class D106RDCEAssetLineage:
    """Exact immutable identity required to consume an asset wire."""

    checkpoint_sha256: str
    runtime_sha256: str
    method_lock_sha256: str
    split_id: str
    tap_sha256: str
    construction_code_sha256: str
    content_root_sha256: str
    source_receipt_sha256: str
    tap_receipt_sha256: str | None = None
    tap_authority_sha256: str | None = None
    schema: str = LINEAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LINEAGE_SCHEMA:
            raise D106RDCEAssetError("D106 lineage schema drift")
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _require_sha256(self.checkpoint_sha256, "checkpoint_sha256"),
        )
        object.__setattr__(
            self,
            "runtime_sha256",
            _require_sha256(self.runtime_sha256, "runtime_sha256"),
        )
        object.__setattr__(
            self,
            "method_lock_sha256",
            _require_sha256(self.method_lock_sha256, "method_lock_sha256"),
        )
        object.__setattr__(self, "split_id", _require_d104_split_id(self.split_id))
        object.__setattr__(
            self, "tap_sha256", _require_sha256(self.tap_sha256, "tap_sha256")
        )
        object.__setattr__(
            self,
            "construction_code_sha256",
            _require_sha256(
                self.construction_code_sha256, "construction_code_sha256"
            ),
        )
        object.__setattr__(
            self,
            "content_root_sha256",
            _require_sha256(self.content_root_sha256, "content_root_sha256"),
        )
        object.__setattr__(
            self,
            "source_receipt_sha256",
            _require_sha256(self.source_receipt_sha256, "source_receipt_sha256"),
        )

        if (self.tap_receipt_sha256 is None) != (self.tap_authority_sha256 is None):
            raise D106RDCEAssetError("D106 tap lineage authority fields must appear together")
        if self.tap_receipt_sha256 is not None:
            object.__setattr__(
                self,
                "tap_receipt_sha256",
                _require_sha256(self.tap_receipt_sha256, "tap_receipt_sha256"),
            )
            object.__setattr__(
                self,
                "tap_authority_sha256",
                _require_sha256(self.tap_authority_sha256, "tap_authority_sha256"),
            )

    @property
    def has_external_tap_authority(self) -> bool:
        return self.tap_receipt_sha256 is not None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "runtime_sha256": self.runtime_sha256,
            "method_lock_sha256": self.method_lock_sha256,
            "split_id": self.split_id,
            "tap_sha256": self.tap_sha256,
            "construction_code_sha256": self.construction_code_sha256,
            "content_root_sha256": self.content_root_sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
            "tap_receipt_sha256": self.tap_receipt_sha256,
            "tap_authority_sha256": self.tap_authority_sha256,
        }


def _tap_content_root(receipt: Mapping[str, Any], array_sha256: Mapping[str, str]) -> str:
    storage_validation = receipt["storage_validation_binding"]
    extraction = receipt["extraction_binding"]
    values = {
        name: receipt[name]
        for name in (
            "schema",
            "candidate_id",
            "split_id",
            "protocol_schema",
            "checkpoint_sha256",
            "runtime_sha256",
            "selected_iq_archive_sha256",
            "selected_iq_receipt_sha256",
            "storage_validator_receipt_sha256",
            "input_ls_archive_sha256",
            "tap_archive_sha256",
            "physical_id_root_sha256",
        )
    }
    return _sha256_bytes(
        _canonical_bytes(
            {
                "schema": TAP_CONTENT_ROOT_SCHEMA,
                "tap_receipt_identity": values,
                "storage_validation_binding": {
                    "schema": storage_validation["schema"],
                    "storage_validation_root_sha256": storage_validation[
                        "storage_validation_root_sha256"
                    ],
                    "selected_content_root_sha256": storage_validation[
                        "selected_content_root_sha256"
                    ],
                    "all_8400x3_storage_semantics_verified": storage_validation[
                        "all_8400x3_storage_semantics_verified"
                    ],
                },
                "extraction_binding": {
                    "schema": extraction["schema"],
                    "row_count": extraction["row_count"],
                    "selection_salt_sha256": extraction[
                        "selection_salt_sha256"
                    ],
                    "selected_content_root_sha256": extraction[
                        "selected_content_root_sha256"
                    ],
                    "input_ls_archive_sha256": extraction[
                        "input_ls_archive_sha256"
                    ],
                    "execution_root_sha256": extraction["execution_root_sha256"],
                },
                "actual_array_sha256": dict(array_sha256),
            }
        )
    )


def _verified_tap(
    tap: D106Phase1TapRows,
    build_lock: D106RDCEBuildLock,
    *,
    tap_authority: _D106RDCETapAuthority | None,
) -> tuple[np.ndarray, tuple[bytes, ...], tuple[bytes, ...], tuple[bytes, ...], D106RDCEAssetLineage]:
    """Verify every binding that must originate in the actual sealed tap."""

    if type(tap) is not D106Phase1TapRows:
        raise D106RDCEAssetError("D106 asset requires an exact D106Phase1TapRows tap")
    if type(build_lock) is not D106RDCEBuildLock:
        raise D106RDCEAssetError("D106 asset requires an exact typed build lock")
    if tap_authority is not None and type(tap_authority) is not _D106RDCETapAuthority:
        raise D106RDCEAssetError("D106 asset tap authority type drift")
    if not isinstance(tap.receipt, Mapping):
        raise D106RDCEAssetError("D106 tap requires a mapping receipt")
    receipt = dict(tap.receipt)
    required = {
        "schema",
        "candidate_id",
        "split_id",
        "protocol_schema",
        "checkpoint_sha256",
        "runtime_sha256",
        "selected_iq_archive_sha256",
        "selected_iq_receipt_sha256",
        "storage_validator_receipt_sha256",
        "storage_validation_binding",
        "extraction_binding",
        "input_ls_archive_sha256",
        "tap_archive_name",
        "tap_archive_sha256",
        "tap_archive_members",
        "array_sha256",
        "row_count",
        "physical_id_root_sha256",
    }
    if not required.issubset(receipt):
        raise D106RDCEAssetError("D106 tap receipt required lineage fields are missing")
    storage_validation = receipt["storage_validation_binding"]
    extraction = receipt["extraction_binding"]
    if (
        receipt["schema"] != TAP_RECEIPT_SCHEMA
        or receipt["candidate_id"] != CANDIDATE_ID
        or receipt["split_id"] != D104_SPLIT_ID
        or receipt["protocol_schema"] != "p2_min_v1"
        or type(receipt["row_count"]) is not int
        or receipt["tap_archive_name"] != TAP_ARCHIVE_NAME
        or tuple(receipt["tap_archive_members"]) != tuple(TAP_MEMBERS)
        or type(storage_validation) is not dict
        or set(storage_validation)
        != {
            "schema",
            "storage_validation_root_sha256",
            "selected_content_root_sha256",
            "all_8400x3_storage_semantics_verified",
        }
        or storage_validation["schema"] != LS_IQ_VALIDATOR_SCHEMA
        or storage_validation["all_8400x3_storage_semantics_verified"] is not True
        or type(extraction) is not dict
        or set(extraction)
        != {
            "schema",
            "row_count",
            "selection_salt_sha256",
            "selected_content_root_sha256",
            "input_ls_archive_sha256",
            "execution_root_sha256",
        }
        or extraction["schema"] != LS_IQ_RECEIPT_SCHEMA
        or extraction["row_count"] != receipt["row_count"]
        or extraction["input_ls_archive_sha256"]
        != receipt["input_ls_archive_sha256"]
        or storage_validation["selected_content_root_sha256"]
        != extraction["selected_content_root_sha256"]
    ):
        raise D106RDCEAssetError("D106 tap receipt schema/count/member drift")
    for name in (
        "checkpoint_sha256",
        "runtime_sha256",
        "selected_iq_archive_sha256",
        "selected_iq_receipt_sha256",
        "storage_validator_receipt_sha256",
        "input_ls_archive_sha256",
        "tap_archive_sha256",
        "physical_id_root_sha256",
    ):
        _require_sha256(receipt[name], f"tap receipt {name}")
    for value, name in (
        (
            storage_validation["storage_validation_root_sha256"],
            "tap storage_validation_root_sha256",
        ),
        (
            storage_validation["selected_content_root_sha256"],
            "tap storage selected_content_root_sha256",
        ),
        (extraction["selection_salt_sha256"], "tap extraction selection_salt_sha256"),
        (
            extraction["selected_content_root_sha256"],
            "tap extraction selected_content_root_sha256",
        ),
        (extraction["execution_root_sha256"], "tap extraction execution_root_sha256"),
    ):
        _require_sha256(value, name)
    arrays = {
        name: np.asarray(getattr(tap, name))
        for name in TAP_MEMBERS
    }
    actual_array_sha256 = {
        name: _tap_array_sha256(value) for name, value in arrays.items()
    }
    if receipt["array_sha256"] != actual_array_sha256:
        raise D106RDCEAssetError("D106 tap receipt actual array SHA256 drift")
    if tap_authority is not None and receipt["tap_archive_sha256"] != tap_authority.archive_sha256:
        raise D106RDCEAssetError("D106 loaded tap archive SHA256 authority drift")
    pre_relu = _require_float32_matrix(tap.pre_relu, "tap.pre_relu")
    z_dom = _require_float32_matrix(tap.z_dom, "tap.z_dom", rows=len(pre_relu))
    z_id = _require_float32_matrix(tap.z_id, "tap.z_id", rows=len(pre_relu))
    derived_z_id = np.maximum(pre_relu, np.float32(0.0)).astype(
        np.float32, copy=False
    )
    if receipt["row_count"] != len(pre_relu):
        raise D106RDCEAssetError("D106 tap receipt actual row count drift")
    if not np.array_equal(z_id, derived_z_id):
        raise D106RDCEAssetError("tap.z_id must be the exact ReLU(pre_relu) array")
    tx = _typed_tokens(tap.tx_labels, "tap.tx_labels", len(z_id))
    receiver = _typed_tokens(tap.receiver_ids, "tap.receiver_ids", len(z_id))
    day = _typed_tokens(tap.day_ids, "tap.day_ids", len(z_id))
    physical_ids = np.asarray(tap.physical_ids)
    _typed_tokens(physical_ids, "tap.physical_ids", len(z_id))
    if (
        len(set(physical_ids.astype(str).tolist())) != len(physical_ids)
        or receipt["physical_id_root_sha256"] != _tap_ordered_id_root(physical_ids)
    ):
        raise D106RDCEAssetError("D106 tap physical-ID content root drift")
    # z_dom is deliberately checked but omitted from the frozen z_id geometry.
    del z_dom
    lineage = D106RDCEAssetLineage(
        checkpoint_sha256=receipt["checkpoint_sha256"],
        runtime_sha256=receipt["runtime_sha256"],
        method_lock_sha256=build_lock.method_lock_sha256,
        split_id=receipt["split_id"],
        tap_sha256=receipt["tap_archive_sha256"],
        construction_code_sha256=build_lock.construction_code_sha256,
        content_root_sha256=_tap_content_root(receipt, actual_array_sha256),
        source_receipt_sha256=receipt["storage_validator_receipt_sha256"],
        tap_receipt_sha256=(
            None if tap_authority is None else tap_authority.receipt_sha256
        ),
        tap_authority_sha256=(
            None if tap_authority is None else tap_authority.authority_sha256
        ),
    )
    return z_id, tx, receiver, day, lineage


def _normalize_rows(rows: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= EPSILON):
        raise D106RDCEAssetError(f"{name} contains a zero-norm row")
    return np.ascontiguousarray(values / norms, dtype=np.float64)


def _canonical_rows(rows: np.ndarray) -> np.ndarray:
    order = sorted(range(len(rows)), key=lambda index: rows[index].tobytes(order="C"))
    return np.ascontiguousarray(rows[np.asarray(order, dtype=np.int64)], dtype=np.float64)


def _mean_canonical(rows: list[np.ndarray]) -> np.ndarray:
    ordered = _canonical_rows(np.stack(rows, axis=0))
    return np.mean(ordered, axis=0, dtype=np.float64)


def _content_key(rows: list[np.ndarray]) -> bytes:
    ordered = _canonical_rows(np.stack(rows, axis=0))
    return hashlib.sha256(ordered.tobytes(order="C")).digest()


def _canonical_sign(vector: np.ndarray) -> np.ndarray:
    pivot = int(np.argmax(np.abs(vector)))
    return -vector if vector[pivot] < 0.0 else vector


def _canonical_tied_eigenspace(vectors: np.ndarray) -> list[np.ndarray]:
    """Choose an order- and sign-stable basis for one repeated eigenspace."""

    if vectors.ndim != 2 or vectors.shape[0] != Z_DIM or vectors.shape[1] < 1:
        raise D106RDCEAssetError("eigenspace shape drift")
    projector = vectors @ vectors.T
    chosen: list[np.ndarray] = []
    tolerance = 1.0e-11
    for coordinate in range(Z_DIM):
        candidate = projector[:, coordinate].astype(np.float64, copy=True)
        for previous in chosen:
            candidate -= previous * float(previous @ candidate)
        norm = float(np.linalg.norm(candidate))
        if norm <= tolerance:
            continue
        candidate /= norm
        chosen.append(_canonical_sign(candidate))
        if len(chosen) == vectors.shape[1]:
            break
    if len(chosen) != vectors.shape[1]:
        raise D106RDCEAssetError("failed to canonicalize repeated eigenspace")
    return chosen


def _canonical_top_eigensystem(scatter: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (scatter + scatter.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(-eigenvalues, kind="stable")
    values = eigenvalues[order]
    vectors = eigenvectors[:, order]
    selected: list[np.ndarray] = []
    start = 0
    while start < len(values) and len(selected) < RDCE_RANK:
        value = float(values[start])
        tolerance = max(1.0e-12, abs(value) * 1.0e-10)
        end = start + 1
        while end < len(values) and abs(float(values[end]) - value) <= tolerance:
            end += 1
        for vector in _canonical_tied_eigenspace(vectors[:, start:end]):
            if len(selected) == RDCE_RANK:
                break
            selected.append(vector)
        start = end
    if len(selected) != RDCE_RANK:
        raise _ScientificReject("rank3_spectrum_unavailable")
    basis = np.stack(selected, axis=0)
    if not np.allclose(
        basis @ basis.T, np.eye(RDCE_RANK), rtol=0.0, atol=1.0e-9
    ):
        raise D106RDCEAssetError("canonical eigenbasis lost orthogonality")
    return basis


def _quantize_basis(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codes = np.zeros((RDCE_RANK, Z_DIM), dtype=np.int8)
    scales = np.zeros(RDCE_RANK, dtype=np.float16)
    for index, row in enumerate(np.asarray(rows, dtype=np.float64)):
        maximum = float(np.max(np.abs(row)))
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise D106RDCEAssetError("basis quantization received a degenerate direction")
        scale = np.float16(maximum / float(INT8_MAX))
        if not math.isfinite(float(scale)) or scale <= 0.0:
            raise D106RDCEAssetError("basis quantization scale drift")
        code = np.clip(
            np.rint(row / float(scale)), -INT8_MAX, INT8_MAX
        ).astype(np.int8)
        if np.any(code == np.int8(-128)):
            raise D106RDCEAssetError("basis quantization code range drift")
        codes[index] = code
        scales[index] = scale
    return codes, scales


def _quantize_positive(values: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values, dtype=np.float64)
    if raw.shape != (RDCE_RANK,) or not np.isfinite(raw).all() or np.any(raw <= 0.0):
        raise D106RDCEAssetError(f"{name} must be finite positive rank-three values")
    scale = np.float16(float(np.max(raw)) / float(INT8_MAX))
    if not math.isfinite(float(scale)) or scale <= 0.0:
        raise D106RDCEAssetError(f"{name} quantization scale drift")
    codes = np.clip(np.rint(raw / float(scale)), 1, INT8_MAX).astype(np.int8)
    scales = np.full(RDCE_RANK, scale, dtype=np.float16)
    return codes, scales


def _decode_positive(codes: np.ndarray, scales: np.ndarray, name: str) -> np.ndarray:
    values = codes.astype(np.float64) * scales.astype(np.float64)
    if values.shape != (RDCE_RANK,) or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise D106RDCEAssetError(f"{name} decoded values must remain positive")
    return values


def _orthogonal_closure(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Fail on a raw decoded Gram drift before deterministic closure."""

    decoded = codes.astype(np.float64) * scales.astype(np.float64)[:, None]
    if decoded.shape != (RDCE_RANK, Z_DIM) or not np.isfinite(decoded).all():
        raise D106RDCEAssetError("decoded basis shape/finite drift")
    gram = decoded @ decoded.T
    if not np.allclose(
        gram, np.eye(RDCE_RANK), rtol=0.0, atol=RAW_DECODED_GRAM_ATOL
    ):
        raise D106RDCEAssetError("INT8 decoded raw Gram exceeds quantization tolerance")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    if (
        not np.isfinite(eigenvalues).all()
        or float(np.min(eigenvalues)) <= MIN_BASIS_SINGULAR_VALUE**2
    ):
        raise D106RDCEAssetError("INT8 decoded basis lost rank")
    inverse_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    closed = inverse_sqrt @ decoded
    if not np.allclose(
        closed @ closed.T, np.eye(RDCE_RANK), rtol=0.0, atol=2.0e-10
    ):
        raise D106RDCEAssetError("INT8 decoded basis orthogonal closure drift")
    return np.ascontiguousarray(closed, dtype=np.float64)


def _asset_payload(
    *,
    lineage: D106RDCEAssetLineage,
    deployment_status: str,
    source_row_count: int,
    source_class_count: int,
    basis_codes_qint8: np.ndarray,
    basis_scales_fp16: np.ndarray,
    tau_codes_qint8: np.ndarray,
    tau_scales_fp16: np.ndarray,
    spectrum_codes_qint8: np.ndarray,
    spectrum_scales_fp16: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "candidate_id": CANDIDATE_ID,
        **lineage.as_dict(),
        "deployment_status": deployment_status,
        "source_row_count": source_row_count,
        "source_class_count": source_class_count,
        "receiver_day_cell_count": RECEIVER_DAY_CELL_COUNT,
        "rank": RDCE_RANK,
        "z_id_geometry_only": True,
        "z_dom_retained": False,
        "source_rows_retained": False,
        "source_names_retained": False,
        "basis_codes_qint8": _array_receipt(basis_codes_qint8),
        "basis_scales_fp16": _array_receipt(basis_scales_fp16),
        "tau_codes_qint8": _array_receipt(tau_codes_qint8),
        "tau_scales_fp16": _array_receipt(tau_scales_fp16),
        "spectrum_codes_qint8": _array_receipt(spectrum_codes_qint8),
        "spectrum_scales_fp16": _array_receipt(spectrum_scales_fp16),
    }


@dataclass(frozen=True, slots=True)
class D106RDCEAsset:
    """The deployable aggregate-only rank-three RDCE component."""

    checkpoint_sha256: str
    runtime_sha256: str
    method_lock_sha256: str
    split_id: str
    tap_sha256: str
    construction_code_sha256: str
    content_root_sha256: str
    source_receipt_sha256: str
    tap_receipt_sha256: str | None
    tap_authority_sha256: str | None
    source_row_count: int
    source_class_count: int
    basis_codes_qint8: np.ndarray
    basis_scales_fp16: np.ndarray
    tau_codes_qint8: np.ndarray
    tau_scales_fp16: np.ndarray
    spectrum_codes_qint8: np.ndarray
    spectrum_scales_fp16: np.ndarray
    asset_receipt_sha256: str
    deployment_status: str = NON_DEPLOYABLE_MATH_STATUS
    _tap_authority: _D106RDCETapAuthority | None = None
    _authority_token: object | None = None
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise D106RDCEAssetError("D106 asset schema drift")
        lineage = self.lineage
        formal = self.deployment_status == FORMAL_DEPLOYMENT_STATUS
        if self.deployment_status not in {
            FORMAL_DEPLOYMENT_STATUS,
            NON_DEPLOYABLE_MATH_STATUS,
        }:
            raise D106RDCEAssetError("D106 asset deployment status drift")
        if formal:
            if (
                type(self._tap_authority) is not _D106RDCETapAuthority
                or self._authority_token is not _TAP_LOADER_TOKEN
                or not lineage.has_external_tap_authority
                or self._tap_authority.archive_sha256 != lineage.tap_sha256
                or self._tap_authority.receipt_sha256 != lineage.tap_receipt_sha256
                or self._tap_authority.authority_sha256
                != lineage.tap_authority_sha256
            ):
                raise D106RDCEAssetError(
                    "formal D106 asset requires loader-origin external tap authority"
                )
        elif (
            self._tap_authority is not None
            or self._authority_token is not None
            or lineage.has_external_tap_authority
        ):
            raise D106RDCEAssetError(
                "non-deployable D106 math asset may not carry tap authority"
            )
        receipt = _require_sha256(self.asset_receipt_sha256, "asset_receipt_sha256")
        if (
            type(self.source_row_count) is not int
            or self.source_row_count != D104_SOURCE_ROW_COUNT
            or type(self.source_class_count) is not int
            or self.source_class_count != D104_SOURCE_CLASS_COUNT
        ):
            raise D106RDCEAssetError("D106 asset aggregate count drift")
        arrays = {
            "basis_codes_qint8": (
                self.basis_codes_qint8,
                np.dtype(np.int8),
                (RDCE_RANK, Z_DIM),
            ),
            "basis_scales_fp16": (
                self.basis_scales_fp16,
                np.dtype("<f2"),
                (RDCE_RANK,),
            ),
            "tau_codes_qint8": (
                self.tau_codes_qint8,
                np.dtype(np.int8),
                (RDCE_RANK,),
            ),
            "tau_scales_fp16": (
                self.tau_scales_fp16,
                np.dtype("<f2"),
                (RDCE_RANK,),
            ),
            "spectrum_codes_qint8": (
                self.spectrum_codes_qint8,
                np.dtype(np.int8),
                (RDCE_RANK,),
            ),
            "spectrum_scales_fp16": (
                self.spectrum_scales_fp16,
                np.dtype("<f2"),
                (RDCE_RANK,),
            ),
        }
        normalized: dict[str, np.ndarray] = {}
        for name, (value, dtype, shape) in arrays.items():
            array = np.asarray(value)
            if array.dtype != dtype or array.shape != shape or not np.isfinite(array).all():
                raise D106RDCEAssetError(f"{name} dtype/shape/finite drift")
            if name.endswith("codes_qint8") and (
                np.any(array == np.int8(-128))
                or (name != "basis_codes_qint8" and np.any(array <= 0))
            ):
                raise D106RDCEAssetError(f"{name} code range drift")
            if name.endswith("scales_fp16") and np.any(array <= 0.0):
                raise D106RDCEAssetError(f"{name} scale range drift")
            normalized[name] = np.ascontiguousarray(array)
        _orthogonal_closure(
            normalized["basis_codes_qint8"], normalized["basis_scales_fp16"]
        )
        _decode_positive(normalized["tau_codes_qint8"], normalized["tau_scales_fp16"], "tau")
        _decode_positive(
            normalized["spectrum_codes_qint8"],
            normalized["spectrum_scales_fp16"],
            "spectrum",
        )
        payload = _asset_payload(
            lineage=lineage,
            deployment_status=self.deployment_status,
            source_row_count=self.source_row_count,
            source_class_count=self.source_class_count,
            **normalized,
        )
        if _sha256_bytes(_canonical_bytes(payload)) != receipt:
            raise D106RDCEAssetError("D106 asset receipt drift")
        for name, value in lineage.as_dict().items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "asset_receipt_sha256", receipt)
        for name, value in normalized.items():
            object.__setattr__(self, name, _readonly(value, value.dtype))

    @property
    def lineage(self) -> D106RDCEAssetLineage:
        return D106RDCEAssetLineage(
            checkpoint_sha256=self.checkpoint_sha256,
            runtime_sha256=self.runtime_sha256,
            method_lock_sha256=self.method_lock_sha256,
            split_id=self.split_id,
            tap_sha256=self.tap_sha256,
            construction_code_sha256=self.construction_code_sha256,
            content_root_sha256=self.content_root_sha256,
            source_receipt_sha256=self.source_receipt_sha256,
            tap_receipt_sha256=self.tap_receipt_sha256,
            tap_authority_sha256=self.tap_authority_sha256,
        )

    @property
    def is_formal_deployable(self) -> bool:
        return self.deployment_status == FORMAL_DEPLOYMENT_STATUS

    @property
    def binding_sha256(self) -> str:
        return _sha256_bytes(
            _canonical_bytes(
                {
                    "candidate_id": CANDIDATE_ID,
                    "lineage": self.lineage.as_dict(),
                    "asset_receipt_sha256": self.asset_receipt_sha256,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class D106RDCEScientificRejectReceipt:
    """A no-wire receipt for a scientifically invalid RDCE construction."""

    stage: str
    reason: str
    lineage: D106RDCEAssetLineage
    source_row_count: int
    source_class_count: int
    deployable_wire_present: bool = False
    status: str = "SCIENTIFIC_REJECT_NO_DEPLOYABLE_WIRE"
    schema: str = REJECT_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != REJECT_SCHEMA
            or self.stage not in {"phase1_asset", "phase2_runtime"}
            or not isinstance(self.reason, str)
            or not self.reason
            or type(self.lineage) is not D106RDCEAssetLineage
            or self.deployable_wire_present is not False
            or self.status != "SCIENTIFIC_REJECT_NO_DEPLOYABLE_WIRE"
            or type(self.source_row_count) is not int
            or self.source_row_count < 0
            or type(self.source_class_count) is not int
            or self.source_class_count < 0
        ):
            raise D106RDCEAssetError("D106 scientific reject receipt drift")

    @property
    def receipt_sha256(self) -> str:
        return _sha256_bytes(_canonical_bytes(_reject_payload(self)))

    @property
    def checkpoint_sha256(self) -> str:
        return self.lineage.checkpoint_sha256

    @property
    def runtime_sha256(self) -> str:
        return self.lineage.runtime_sha256

    @property
    def method_lock_sha256(self) -> str:
        return self.lineage.method_lock_sha256

    @property
    def split_id(self) -> str:
        return self.lineage.split_id

    @property
    def tap_sha256(self) -> str:
        return self.lineage.tap_sha256

    @property
    def construction_code_sha256(self) -> str:
        return self.lineage.construction_code_sha256

    @property
    def content_root_sha256(self) -> str:
        return self.lineage.content_root_sha256

    @property
    def source_receipt_sha256(self) -> str:
        return self.lineage.source_receipt_sha256


def _reject_payload(receipt: D106RDCEScientificRejectReceipt) -> dict[str, Any]:
    return {
        "schema": receipt.schema,
        "candidate_id": CANDIDATE_ID,
        "stage": receipt.stage,
        "reason": receipt.reason,
        **receipt.lineage.as_dict(),
        "source_row_count": receipt.source_row_count,
        "source_class_count": receipt.source_class_count,
        "deployable_wire_present": False,
        "status": receipt.status,
        "source_rows_retained": False,
        "source_names_retained": False,
    }


def make_d106_rdce_scientific_reject(
    *,
    stage: str,
    reason: str,
    lineage: D106RDCEAssetLineage,
    source_row_count: int,
    source_class_count: int,
) -> D106RDCEScientificRejectReceipt:
    if type(lineage) is not D106RDCEAssetLineage:
        raise D106RDCEAssetError("scientific reject requires exact typed lineage")
    return D106RDCEScientificRejectReceipt(
        stage=stage,
        reason=reason,
        lineage=lineage,
        source_row_count=source_row_count,
        source_class_count=source_class_count,
    )


def _build_geometry(
    z_id: np.ndarray,
    tx: tuple[bytes, ...],
    receiver: tuple[bytes, ...],
    day: tuple[bytes, ...],
) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]], int]:
    rows = _normalize_rows(z_id, "z_id")
    cell_class_rows: dict[tuple[bytes, bytes, bytes], list[np.ndarray]] = {}
    class_rows: dict[bytes, list[np.ndarray]] = {}
    receiver_class_count: dict[tuple[bytes, bytes], int] = {}
    for index, row in enumerate(rows):
        key = (receiver[index], day[index], tx[index])
        cell_class_rows.setdefault(key, []).append(row)
        class_rows.setdefault(tx[index], []).append(row)
        receiver_class_count[(receiver[index], tx[index])] = (
            receiver_class_count.get((receiver[index], tx[index]), 0) + 1
        )
    if len(rows) != D104_SOURCE_ROW_COUNT:
        raise _ScientificReject("d104_ls_row_count_not_588")
    if len(class_rows) != D104_SOURCE_CLASS_COUNT:
        raise _ScientificReject("d104_ls_source_class_count_not_6")
    receiver_set = frozenset(receiver)
    day_set = frozenset(day)
    if len(receiver_set) != D104_RECEIVER_COUNT or len(day_set) != D104_DAY_COUNT:
        raise _ScientificReject("d104_receiver_day_cardinality_not_7x4")
    if any(len(values) != D104_TX_ROW_COUNT for values in class_rows.values()):
        raise _ScientificReject("d104_each_tx_row_count_not_98")
    if any(
        receiver_class_count.get((receiver_token, tx_token), 0)
        != D104_RECEIVER_TX_FOUR_DAY_COUNT
        for receiver_token in receiver_set
        for tx_token in class_rows
    ):
        raise _ScientificReject("d104_receiver_tx_four_day_count_not_14")
    cells = {(key[0], key[1]) for key in cell_class_rows}
    if len(cells) != RECEIVER_DAY_CELL_COUNT:
        raise _ScientificReject("receiver_day_cell_count_not_28")
    if len(cell_class_rows) != RECEIVER_DAY_CELL_COUNT * D104_SOURCE_CLASS_COUNT:
        raise _ScientificReject("receiver_day_class_grid_not_28x6")
    all_classes = frozenset(class_rows)
    if any(
        frozenset(
            tx_token
            for (receiver_token, day_token, tx_token) in cell_class_rows
            if (receiver_token, day_token) == cell
        )
        != all_classes
        for cell in cells
    ):
        raise _ScientificReject("receiver_day_class_grid_incomplete")
    if any(
        not D104_CELL_MIN_SAMPLES <= len(values) <= D104_CELL_MAX_SAMPLES
        for values in cell_class_rows.values()
    ):
        raise _ScientificReject("d104_cell_sample_count_not_2_to_4")
    m = {key: _mean_canonical(value) for key, value in cell_class_rows.items()}
    by_class: dict[bytes, list[tuple[tuple[bytes, bytes], np.ndarray]]] = {}
    by_cell: dict[tuple[bytes, bytes], list[tuple[bytes, np.ndarray]]] = {}
    for (receiver_token, day_token, tx_token), value in m.items():
        cell = (receiver_token, day_token)
        by_class.setdefault(tx_token, []).append((cell, value))
        by_cell.setdefault(cell, []).append((tx_token, value))
    mu: dict[bytes, np.ndarray] = {}
    for tx_token, items in by_class.items():
        if len(items) != RECEIVER_DAY_CELL_COUNT:
            raise _ScientificReject("class_not_seen_in_all_receiver_day_cells")
        ordered_means = [
            value
            for _, value in sorted(items, key=lambda item: _content_key([item[1]]))
        ]
        mu[tx_token] = _mean_canonical(ordered_means)
    g_rows: list[tuple[bytes, np.ndarray]] = []
    for cell, items in by_cell.items():
        if len(items) != D104_SOURCE_CLASS_COUNT:
            raise _ScientificReject("receiver_day_cell_class_count_not_6")
        deltas = [value - mu[tx_token] for tx_token, value in items]
        g = _mean_canonical(sorted(deltas, key=lambda value: _content_key([value])))
        g_rows.append((_content_key([value for _, value in items]), g))
    ordered_g = [value for _, value in sorted(g_rows, key=lambda item: item[0])]
    if len(ordered_g) != RECEIVER_DAY_CELL_COUNT:
        raise D106RDCEAssetError("receiver-day aggregation cardinality drift")
    G = np.stack(ordered_g, axis=0)
    G = G - _mean_canonical(ordered_g)
    scatter = (G.T @ G) / float(RECEIVER_DAY_CELL_COUNT)
    basis = _canonical_top_eigensystem(scatter)
    ordered_groups = sorted(class_rows.values(), key=_content_key)
    if any(len(group) != D104_TX_ROW_COUNT for group in ordered_groups):
        raise _ScientificReject("d104_each_tx_row_count_not_98")
    return basis, scatter, ordered_groups, len(class_rows)


def _balanced_tau_for_closed_basis(
    class_groups: list[list[np.ndarray]], basis: np.ndarray
) -> np.ndarray:
    per_class_scatter: list[np.ndarray] = []
    for rows_for_class in class_groups:
        ordered = _canonical_rows(np.stack(rows_for_class, axis=0))
        residual = ordered - np.mean(ordered, axis=0, dtype=np.float64)
        projected = residual @ basis.T
        per_class_scatter.append(
            np.sum(np.square(projected), axis=0) / float(len(ordered) - 1)
        )
    tau = np.mean(np.stack(per_class_scatter, axis=0), axis=0, dtype=np.float64)
    if not np.isfinite(tau).all() or np.any(tau <= EPSILON):
        raise _ScientificReject("class_balanced_projected_scatter_nonpositive")
    return tau


def _try_build_d106_rdce_asset_core(
    tap: D106Phase1TapRows,
    *,
    build_lock: D106RDCEBuildLock,
    tap_authority: _D106RDCETapAuthority | None,
    formal_loader_token: object | None,
) -> D106RDCEAsset | D106RDCEScientificRejectReceipt:
    """Internal construction core shared by math diagnostics and formal loading."""

    if tap_authority is not None and formal_loader_token is not _TAP_LOADER_TOKEN:
        raise D106RDCEAssetError("formal asset core requires the loader-only token")

    z_id, tx, receiver, day, lineage = _verified_tap(
        tap, build_lock, tap_authority=tap_authority
    )
    try:
        raw_basis, scatter, class_groups, class_count = _build_geometry(
            z_id, tx, receiver, day
        )
        basis_codes, basis_scales = _quantize_basis(raw_basis)
        closed_basis = _orthogonal_closure(basis_codes, basis_scales)
        # The deployed basis is the closed INT8 decode, so both support scale
        # and spectrum are intentionally computed in that deployed coordinate.
        tau = _balanced_tau_for_closed_basis(class_groups, closed_basis)
        spectrum = np.diag(closed_basis @ scatter @ closed_basis.T)
        if not np.isfinite(spectrum).all() or np.any(spectrum <= MIN_SPECTRUM):
            raise _ScientificReject("rank3_spectrum_nonpositive_after_closed_decode")
    except _ScientificReject as rejection:
        return make_d106_rdce_scientific_reject(
            stage="phase1_asset",
            reason=rejection.reason,
            lineage=lineage,
            source_row_count=len(z_id),
            source_class_count=len(set(tx)),
        )
    tau_codes, tau_scales = _quantize_positive(tau, "tau")
    spectrum_codes, spectrum_scales = _quantize_positive(spectrum, "spectrum")
    deployment_status = (
        FORMAL_DEPLOYMENT_STATUS
        if tap_authority is not None
        else NON_DEPLOYABLE_MATH_STATUS
    )
    payload = _asset_payload(
        lineage=lineage,
        deployment_status=deployment_status,
        source_row_count=len(z_id),
        source_class_count=class_count,
        basis_codes_qint8=basis_codes,
        basis_scales_fp16=basis_scales,
        tau_codes_qint8=tau_codes,
        tau_scales_fp16=tau_scales,
        spectrum_codes_qint8=spectrum_codes,
        spectrum_scales_fp16=spectrum_scales,
    )
    return D106RDCEAsset(
        **lineage.as_dict(),
        source_row_count=len(z_id),
        source_class_count=class_count,
        basis_codes_qint8=basis_codes,
        basis_scales_fp16=basis_scales,
        tau_codes_qint8=tau_codes,
        tau_scales_fp16=tau_scales,
        spectrum_codes_qint8=spectrum_codes,
        spectrum_scales_fp16=spectrum_scales,
        asset_receipt_sha256=_sha256_bytes(_canonical_bytes(payload)),
        deployment_status=deployment_status,
        _tap_authority=tap_authority,
        _authority_token=formal_loader_token,
    )


def _try_build_d106_rdce_asset_math(
    tap: D106Phase1TapRows,
    *,
    build_lock: D106RDCEBuildLock,
) -> D106RDCEAsset | D106RDCEScientificRejectReceipt:
    """Private pure-math path; its output is always NON_DEPLOYABLE."""

    return _try_build_d106_rdce_asset_core(
        tap,
        build_lock=build_lock,
        tap_authority=None,
        formal_loader_token=None,
    )


def _try_build_d106_rdce_asset_from_loaded_tap(
    tap: D106Phase1TapRows,
    *,
    build_lock: D106RDCEBuildLock,
    tap_authority: _D106RDCETapAuthority,
) -> D106RDCEAsset | D106RDCEScientificRejectReceipt:
    """Internal formal path reached only after the DATA loader boundary."""

    return _try_build_d106_rdce_asset_core(
        tap,
        build_lock=build_lock,
        tap_authority=tap_authority,
        formal_loader_token=_TAP_LOADER_TOKEN,
    )


def _sha256_regular_file(path: str | Path, name: str) -> tuple[Path, str]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise D106RDCEAssetError(f"{name} must be a regular non-symlink file")
    return source, _sha256_bytes(source.read_bytes())


def _load_formal_d106_tap(
    tap_archive_path: str | Path,
    tap_receipt_path: str | Path,
    *,
    expected_tap_archive_sha256: str,
    expected_tap_receipt_sha256: str,
) -> tuple[D106Phase1TapRows, _D106RDCETapAuthority]:
    """Use the DATA-owned loader after binding both external source files."""

    expected_archive = _require_sha256(
        expected_tap_archive_sha256, "expected_tap_archive_sha256"
    )
    expected_receipt = _require_sha256(
        expected_tap_receipt_sha256, "expected_tap_receipt_sha256"
    )
    archive, archive_sha = _sha256_regular_file(tap_archive_path, "D106 tap archive")
    receipt, receipt_sha = _sha256_regular_file(tap_receipt_path, "D106 tap receipt")
    if archive_sha != expected_archive or receipt_sha != expected_receipt:
        raise D106RDCEAssetError("external D106 tap SHA256 authority mismatch")
    try:
        tap = load_d106_phase1_ls_tap(
            archive,
            receipt,
            expected_archive_sha256=expected_archive,
            expected_receipt_sha256=expected_receipt,
        )
    except Exception as error:
        raise D106RDCEAssetError("D106 formal tap loader rejected the external artifact") from error
    if type(tap) is not D106Phase1TapRows:
        raise D106RDCEAssetError("D106 formal tap loader returned an unexpected type")
    return tap, _D106RDCETapAuthority(
        archive_sha256=expected_archive,
        receipt_sha256=expected_receipt,
    )


def try_build_d106_rdce_asset(
    tap_archive_path: str | Path,
    tap_receipt_path: str | Path,
    *,
    expected_tap_archive_sha256: str,
    expected_tap_receipt_sha256: str,
    build_lock: D106RDCEBuildLock,
) -> D106RDCEAsset | D106RDCEScientificRejectReceipt:
    """Formal deployable builder; bare D106Phase1TapRows are never accepted."""

    tap, authority = _load_formal_d106_tap(
        tap_archive_path,
        tap_receipt_path,
        expected_tap_archive_sha256=expected_tap_archive_sha256,
        expected_tap_receipt_sha256=expected_tap_receipt_sha256,
    )
    return _try_build_d106_rdce_asset_from_loaded_tap(
        tap, build_lock=build_lock, tap_authority=authority
    )


def build_d106_rdce_asset(
    tap_archive_path: str | Path,
    tap_receipt_path: str | Path,
    *,
    expected_tap_archive_sha256: str,
    expected_tap_receipt_sha256: str,
    build_lock: D106RDCEBuildLock,
) -> D106RDCEAsset:
    """Strict formal variant; only loader-origin assets can be deployable."""

    result = try_build_d106_rdce_asset(
        tap_archive_path,
        tap_receipt_path,
        expected_tap_archive_sha256=expected_tap_archive_sha256,
        expected_tap_receipt_sha256=expected_tap_receipt_sha256,
        build_lock=build_lock,
    )
    if isinstance(result, D106RDCEScientificRejectReceipt):
        raise D106RDCEScientificRejectError(result)
    return result


def decode_d106_rdce_basis(asset: D106RDCEAsset) -> np.ndarray:
    if type(asset) is not D106RDCEAsset:
        raise D106RDCEAssetError("D106 basis decode requires an exact asset")
    return _orthogonal_closure(asset.basis_codes_qint8, asset.basis_scales_fp16)


def decode_d106_rdce_tau(asset: D106RDCEAsset) -> np.ndarray:
    if type(asset) is not D106RDCEAsset:
        raise D106RDCEAssetError("D106 tau decode requires an exact asset")
    return _decode_positive(asset.tau_codes_qint8, asset.tau_scales_fp16, "tau")


def decode_d106_rdce_spectrum(asset: D106RDCEAsset) -> np.ndarray:
    if type(asset) is not D106RDCEAsset:
        raise D106RDCEAssetError("D106 spectrum decode requires an exact asset")
    return _decode_positive(
        asset.spectrum_codes_qint8, asset.spectrum_scales_fp16, "spectrum"
    )


def _wire_header(asset: D106RDCEAsset) -> dict[str, Any]:
    payload = _asset_payload(
        lineage=asset.lineage,
        deployment_status=asset.deployment_status,
        source_row_count=asset.source_row_count,
        source_class_count=asset.source_class_count,
        basis_codes_qint8=asset.basis_codes_qint8,
        basis_scales_fp16=asset.basis_scales_fp16,
        tau_codes_qint8=asset.tau_codes_qint8,
        tau_scales_fp16=asset.tau_scales_fp16,
        spectrum_codes_qint8=asset.spectrum_codes_qint8,
        spectrum_scales_fp16=asset.spectrum_scales_fp16,
    )
    return {
        "schema": WIRE_SCHEMA,
        "asset": payload,
        "asset_receipt_sha256": asset.asset_receipt_sha256,
    }


def serialize_d106_rdce_asset(asset: D106RDCEAsset) -> bytes:
    if type(asset) is not D106RDCEAsset or not asset.is_formal_deployable:
        raise D106RDCEAssetError(
            "D106 serialization requires a formal loader-authorized asset"
        )
    header = _canonical_bytes(_wire_header(asset))
    arrays = (
        asset.basis_codes_qint8,
        asset.basis_scales_fp16,
        asset.tau_codes_qint8,
        asset.tau_scales_fp16,
        asset.spectrum_codes_qint8,
        asset.spectrum_scales_fp16,
    )
    return WIRE_MAGIC + struct.pack(">I", len(header)) + header + b"".join(
        np.ascontiguousarray(array).tobytes(order="C") for array in arrays
    )


def _parse_wire_array(
    payload: bytes, offset: int, dtype: np.dtype[Any], shape: tuple[int, ...]
) -> tuple[np.ndarray, int]:
    count = int(np.prod(shape, dtype=np.int64))
    length = count * dtype.itemsize
    if offset + length > len(payload):
        raise D106RDCEAssetError("D106 wire is truncated")
    array = (
        np.frombuffer(payload[offset : offset + length], dtype=dtype)
        .copy()
        .reshape(shape)
    )
    return array, offset + length


def deserialize_d106_rdce_asset(
    payload: bytes,
    *,
    expected_wire_sha256: str,
    expected_lineage: D106RDCEAssetLineage,
) -> D106RDCEAsset:
    """Load only a pinned wire into an exactly expected typed lineage."""

    if not isinstance(payload, bytes) or len(payload) <= len(WIRE_MAGIC) + 4:
        raise D106RDCEAssetError("D106 wire must be non-empty bytes")
    expected_wire = _require_sha256(expected_wire_sha256, "expected_wire_sha256")
    if _sha256_bytes(payload) != expected_wire:
        raise D106RDCEAssetError("D106 wire SHA256 trust anchor mismatch")
    if type(expected_lineage) is not D106RDCEAssetLineage:
        raise D106RDCEAssetError("D106 wire requires exact typed expected lineage")
    if not expected_lineage.has_external_tap_authority:
        raise D106RDCEAssetError("D106 wire requires formal external tap lineage")
    if not payload.startswith(WIRE_MAGIC):
        raise D106RDCEAssetError("D106 wire magic drift")
    offset = len(WIRE_MAGIC)
    header_size = struct.unpack(">I", payload[offset : offset + 4])[0]
    offset += 4
    if header_size <= 0 or offset + header_size > len(payload):
        raise D106RDCEAssetError("D106 wire header length drift")
    raw_header = payload[offset : offset + header_size]
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106RDCEAssetError("D106 wire header must be canonical UTF-8 JSON") from error
    if type(header) is not dict or raw_header != _canonical_bytes(header):
        raise D106RDCEAssetError("D106 wire header is not original canonical JSON")
    offset += header_size
    if (
        set(header) != {"schema", "asset", "asset_receipt_sha256"}
        or header["schema"] != WIRE_SCHEMA
        or type(header["asset"]) is not dict
    ):
        raise D106RDCEAssetError("D106 wire header schema drift")
    asset_header = header["asset"]
    expected_keys = {
        "schema",
        "candidate_id",
        "checkpoint_sha256",
        "method_lock_sha256",
        "runtime_sha256",
        "split_id",
        "tap_sha256",
        "construction_code_sha256",
        "content_root_sha256",
        "source_receipt_sha256",
        "tap_receipt_sha256",
        "tap_authority_sha256",
        "deployment_status",
        "source_row_count",
        "source_class_count",
        "receiver_day_cell_count",
        "rank",
        "z_id_geometry_only",
        "z_dom_retained",
        "source_rows_retained",
        "source_names_retained",
        "basis_codes_qint8",
        "basis_scales_fp16",
        "tau_codes_qint8",
        "tau_scales_fp16",
        "spectrum_codes_qint8",
        "spectrum_scales_fp16",
    }
    if set(asset_header) != expected_keys:
        raise D106RDCEAssetError("D106 wire asset header keys drift")
    if asset_header["deployment_status"] != FORMAL_DEPLOYMENT_STATUS:
        raise D106RDCEAssetError("D106 wire may not carry a non-deployable math asset")
    tap_authority = _D106RDCETapAuthority(
        archive_sha256=asset_header["tap_sha256"],
        receipt_sha256=asset_header["tap_receipt_sha256"],
    )
    if asset_header["tap_authority_sha256"] != tap_authority.authority_sha256:
        raise D106RDCEAssetError("D106 wire tap authority digest drift")
    specs = (
        ("basis_codes_qint8", np.dtype(np.int8), (RDCE_RANK, Z_DIM)),
        ("basis_scales_fp16", np.dtype("<f2"), (RDCE_RANK,)),
        ("tau_codes_qint8", np.dtype(np.int8), (RDCE_RANK,)),
        ("tau_scales_fp16", np.dtype("<f2"), (RDCE_RANK,)),
        ("spectrum_codes_qint8", np.dtype(np.int8), (RDCE_RANK,)),
        ("spectrum_scales_fp16", np.dtype("<f2"), (RDCE_RANK,)),
    )
    arrays: dict[str, np.ndarray] = {}
    for name, dtype, shape in specs:
        value, offset = _parse_wire_array(payload, offset, dtype, shape)
        if _array_receipt(value) != asset_header[name]:
            raise D106RDCEAssetError(f"D106 wire {name} receipt drift")
        arrays[name] = value
    if offset != len(payload):
        raise D106RDCEAssetError("D106 wire has trailing bytes")
    asset = D106RDCEAsset(
        checkpoint_sha256=asset_header["checkpoint_sha256"],
        runtime_sha256=asset_header["runtime_sha256"],
        method_lock_sha256=asset_header["method_lock_sha256"],
        split_id=asset_header["split_id"],
        tap_sha256=asset_header["tap_sha256"],
        construction_code_sha256=asset_header["construction_code_sha256"],
        content_root_sha256=asset_header["content_root_sha256"],
        source_receipt_sha256=asset_header["source_receipt_sha256"],
        tap_receipt_sha256=asset_header["tap_receipt_sha256"],
        tap_authority_sha256=asset_header["tap_authority_sha256"],
        source_row_count=asset_header["source_row_count"],
        source_class_count=asset_header["source_class_count"],
        asset_receipt_sha256=header["asset_receipt_sha256"],
        deployment_status=asset_header["deployment_status"],
        _tap_authority=tap_authority,
        _authority_token=_TAP_LOADER_TOKEN,
        **arrays,
    )
    if asset.lineage != expected_lineage:
        raise D106RDCEAssetError("D106 wire expected typed lineage mismatch")
    if _wire_header(asset) != header:
        raise D106RDCEAssetError("D106 wire canonical header drift")
    return asset


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise D106RDCEAssetError(f"refusing to overwrite existing path: {path}")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise D106RDCEAssetError(f"refusing to overwrite existing path: {path}") from error


def _read_regular(path: Path, name: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise D106RDCEAssetError(f"{name} must be a regular non-symlink file")
    return path.read_bytes()


def _atomic_single_file_directory(
    root: Path, file_name: str, payload: bytes, *, kind: str
) -> Path:
    if root.exists() or root.is_symlink():
        raise D106RDCEAssetError(f"D106 {kind} directory must be absent before save")
    parent = root.parent
    if not parent.is_dir() or parent.is_symlink():
        raise D106RDCEAssetError(f"D106 {kind} parent must be an existing regular directory")
    staging = parent / f".{root.name}.staging-{uuid.uuid4().hex}"
    if staging.exists() or staging.is_symlink():
        raise D106RDCEAssetError(f"D106 {kind} staging path collision")
    staging.mkdir()
    try:
        _write_new(staging / file_name, payload)
        if root.exists() or root.is_symlink():
            raise D106RDCEAssetError(f"D106 {kind} directory appeared during save")
        # On the supported Windows deployment surface, rename refuses a
        # pre-existing target and provides the needed same-parent atomic publish.
        os.rename(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return root / file_name


def save_d106_rdce_asset(asset: D106RDCEAsset, directory: str | Path) -> dict[str, str]:
    """Atomically publish one self-authenticating asset wire without sidecars."""

    if type(asset) is not D106RDCEAsset or not asset.is_formal_deployable:
        raise D106RDCEAssetError(
            "D106 save requires a formal loader-authorized asset"
        )
    root = Path(directory)
    wire = serialize_d106_rdce_asset(asset)
    wire_path = _atomic_single_file_directory(
        root, ASSET_WIRE_NAME, wire, kind="asset"
    )
    return {
        "wire": str(wire_path),
        "wire_sha256": _sha256_bytes(wire),
        "asset_receipt_sha256": asset.asset_receipt_sha256,
        "asset_binding_sha256": asset.binding_sha256,
    }


def load_d106_rdce_asset(
    directory: str | Path,
    *,
    expected_wire_sha256: str,
    expected_lineage: D106RDCEAssetLineage,
) -> D106RDCEAsset:
    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise D106RDCEAssetError("D106 asset directory must be a regular directory")
    expected = {ASSET_WIRE_NAME}
    actual = {member.name for member in root.iterdir()}
    if actual != expected:
        raise D106RDCEAssetError("D106 asset directory has a missing or sidecar member")
    wire = _read_regular(root / ASSET_WIRE_NAME, "D106 asset wire")
    return deserialize_d106_rdce_asset(
        wire,
        expected_wire_sha256=expected_wire_sha256,
        expected_lineage=expected_lineage,
    )


def serialize_d106_rdce_scientific_reject(
    receipt: D106RDCEScientificRejectReceipt,
) -> bytes:
    if type(receipt) is not D106RDCEScientificRejectReceipt:
        raise D106RDCEAssetError("D106 reject serialization requires an exact receipt")
    return _canonical_bytes(
        {**_reject_payload(receipt), "receipt_sha256": receipt.receipt_sha256}
    )


def save_d106_rdce_scientific_reject(
    receipt: D106RDCEScientificRejectReceipt, directory: str | Path
) -> dict[str, str]:
    """Atomically publish a no-wire scientific reject receipt."""

    if type(receipt) is not D106RDCEScientificRejectReceipt:
        raise D106RDCEAssetError("D106 reject save requires an exact receipt")
    root = Path(directory)
    payload = serialize_d106_rdce_scientific_reject(receipt)
    receipt_path = _atomic_single_file_directory(
        root, REJECT_NAME, payload, kind="reject"
    )
    return {
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256_bytes(payload),
    }


__all__ = [
    "ASSET_WIRE_NAME",
    "CANDIDATE_ID",
    "D104_CELL_MAX_SAMPLES",
    "D104_CELL_MIN_SAMPLES",
    "D104_DAY_COUNT",
    "D104_RECEIVER_COUNT",
    "D104_RECEIVER_TX_FOUR_DAY_COUNT",
    "D104_SOURCE_CLASS_COUNT",
    "D104_SOURCE_ROW_COUNT",
    "D104_SPLIT_ID",
    "D104_TX_ROW_COUNT",
    "D106RDCEAsset",
    "D106RDCEAssetError",
    "D106RDCEAssetLineage",
    "D106RDCEBuildLock",
    "D106RDCEScientificRejectError",
    "D106RDCEScientificRejectReceipt",
    "EPSILON",
    "FORMAL_DEPLOYMENT_STATUS",
    "NON_DEPLOYABLE_MATH_STATUS",
    "RAW_DECODED_GRAM_ATOL",
    "RDCE_RANK",
    "RECEIVER_DAY_CELL_COUNT",
    "Z_DIM",
    "build_d106_rdce_asset",
    "decode_d106_rdce_basis",
    "decode_d106_rdce_spectrum",
    "decode_d106_rdce_tau",
    "deserialize_d106_rdce_asset",
    "load_d106_rdce_asset",
    "make_d106_rdce_scientific_reject",
    "save_d106_rdce_asset",
    "save_d106_rdce_scientific_reject",
    "serialize_d106_rdce_asset",
    "serialize_d106_rdce_scientific_reject",
    "try_build_d106_rdce_asset",
]
