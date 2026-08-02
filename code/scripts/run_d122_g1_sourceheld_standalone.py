#!/usr/bin/env python3
"""Seal D122's fixed four-arm source-held G1 predictions, then score apart.

``predict`` receives only sealed D104 predictor packages, the frozen D106
RDCE asset and the frozen D112 aggregate inputs.  It has no truth argument.
``score`` first verifies the complete immutable 63-row prediction set and only
then opens the independently held D104 truth package.  This is source-held
evidence, never target Phase2 authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from dataclasses import dataclass
import struct

from cvsrffi.stage2_d112_g0_source_bundle import (  # noqa: E402
    build_d112_g0_source_bundle,
)
from cvsrffi.stage2_d112_seam_bundle import (  # noqa: E402
    build_d112_source_held_g1_bundle,
)
from cvsrffi.stage2_d112_seam_qknn import (  # noqa: E402
    audit_d112_seam_state,
    fit_d112_ground_head_source_held_g1_state,
    score_d112_seam_source_held_g1_logits,
)
from cvsrffi.stage2_d122_rdce_ground_head import (  # noqa: E402
    audit_d122_rdce_ground_head_state,
    fit_d122_rdce_ground_head_source_held_g1_state,
    score_d122_rdce_ground_head_source_held_g1_logits,
    unique_d122_argmax,
)
from cvsrffi.stage2_zid_student_t_qknn import (  # noqa: E402
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    score_zid_student_t_logits,
)


CANDIDATE_ID = "D122_RDCE_STATIC_GROUND_QKNN"
ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")
EFFECT_PAIRS = {
    "DA_AT_BASE": ("M_DA", "M0"),
    "HEAD_AT_ID": ("M_HEAD", "M0"),
    "HEAD_AT_DA": ("M_JOINT", "M_DA"),
}
SPLIT_ID = "d104_source_seed104713_v2"
K_VALUES = (1, 5, 10)
PACKAGE_SCHEMA = "cvs.d104_r1.rxid_angq.held_packages.v2"
PACKAGE_KEYS = {
    "support_pre_relu", "support_zdom", "support_labels",
    "support_physical_ids", "query_pre_relu", "query_physical_ids",
    "registered_classes",
}
D104_CANDIDATE_ID = "D104-R1-ANGQ-RXID-MB4"
D105_LOCK_SHA256 = "7324ff469cf18d34cdc3795e36d053570e60ba341c112167b49d759a150dda08"
PREDICTION_SCHEMA = "cvs.d122.rdce_static_ground_qknn.sourceheld.predictions.v1"
SCORE_SCHEMA = "cvs.d122.rdce_static_ground_qknn.sourceheld.scores.v1"
EFFECT_METRICS = (
    "old_balanced_accuracy",
    "seen_new_accuracy",
    "H_old_new",
    "old_floor",
    "all_class_floor",
    "balanced_accuracy",
    "old_correct_count",
    "seen_new_correct_count",
    "correct_count",
)


class D122G1Error(ValueError):
    """Raised when D122's fixed source-held G1 lifecycle drifts."""



WIRE_MAGIC = b"CVSD106RDCE\x00\x03"
WIRE_SCHEMA = "cvs.phase1.d106.rdce_gtsm_asset_wire.v3"
ASSET_SCHEMA = "cvs.phase1.d106.rdce_gtsm_asset.v3"
LINEAGE_SCHEMA = "cvs.phase1.d106.rdce_gtsm_lineage.v1"
TAP_AUTHORITY_SCHEMA = "cvs.phase1.d106.rdce_gtsm_tap_authority.v1"
CANDIDATE_ASSET_ID = "D106-RDCE/GTSM-r3-SCATTER02"
FORMAL_DEPLOYMENT_STATUS = "FORMAL_DEPLOYABLE"
Z_DIM = 160
RDCE_RANK = 3
SOURCE_ROW_COUNT = 588
SOURCE_CLASS_COUNT = 6
RECEIVER_DAY_CELL_COUNT = 28
INT8_MAX = 127
RAW_DECODED_GRAM_ATOL = 0.125
MIN_BASIS_SINGULAR_VALUE = 1.0e-7
MIN_ATTENUATION_FP16 = np.nextafter(
    np.float16(0.05), np.float16(np.inf), dtype=np.float16
)
MAX_ATTENUATION_FP16 = np.nextafter(
    np.float16(0.95), np.float16(0.0), dtype=np.float16
)


@dataclass(frozen=True, slots=True)
class _StandaloneRDCEAsset:
    checkpoint_sha256: str
    runtime_sha256: str
    method_lock_sha256: str
    split_id: str
    tap_sha256: str
    construction_code_sha256: str
    content_root_sha256: str
    source_receipt_sha256: str
    tap_receipt_sha256: str
    tap_authority_sha256: str
    source_row_count: int
    source_class_count: int
    basis_codes_qint8: np.ndarray
    basis_scales_fp16: np.ndarray
    tau_codes_qint8: np.ndarray
    tau_scales_fp16: np.ndarray
    spectrum_codes_qint8: np.ndarray
    spectrum_scales_fp16: np.ndarray
    asset_receipt_sha256: str


def _require_hash(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise D122G1Error(f"{name} must be a lowercase SHA256")
    return value


def _require_split(value: Any) -> str:
    if type(value) is not str or value != SPLIT_ID:
        raise D122G1Error("D122 RDCE split binding drift")
    return value


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _lineage_dict(asset: _StandaloneRDCEAsset) -> dict[str, Any]:
    return {
        "checkpoint_sha256": asset.checkpoint_sha256,
        "runtime_sha256": asset.runtime_sha256,
        "method_lock_sha256": asset.method_lock_sha256,
        "split_id": asset.split_id,
        "tap_sha256": asset.tap_sha256,
        "construction_code_sha256": asset.construction_code_sha256,
        "content_root_sha256": asset.content_root_sha256,
        "source_receipt_sha256": asset.source_receipt_sha256,
        "tap_receipt_sha256": asset.tap_receipt_sha256,
        "tap_authority_sha256": asset.tap_authority_sha256,
    }


def _asset_payload(asset: _StandaloneRDCEAsset) -> dict[str, Any]:
    return {
        "schema": ASSET_SCHEMA,
        "candidate_id": CANDIDATE_ASSET_ID,
        **_lineage_dict(asset),
        "deployment_status": FORMAL_DEPLOYMENT_STATUS,
        "source_row_count": SOURCE_ROW_COUNT,
        "source_class_count": SOURCE_CLASS_COUNT,
        "receiver_day_cell_count": RECEIVER_DAY_CELL_COUNT,
        "rank": RDCE_RANK,
        "z_id_geometry_only": True,
        "z_dom_retained": False,
        "source_rows_retained": False,
        "source_names_retained": False,
        "basis_codes_qint8": _array_receipt(asset.basis_codes_qint8),
        "basis_scales_fp16": _array_receipt(asset.basis_scales_fp16),
        "tau_codes_qint8": _array_receipt(asset.tau_codes_qint8),
        "tau_scales_fp16": _array_receipt(asset.tau_scales_fp16),
        "spectrum_codes_qint8": _array_receipt(asset.spectrum_codes_qint8),
        "spectrum_scales_fp16": _array_receipt(asset.spectrum_scales_fp16),
    }


def _orthogonal_closure(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    decoded = codes.astype(np.float64) * scales.astype(np.float64)[:, None]
    if decoded.shape != (RDCE_RANK, Z_DIM) or not np.isfinite(decoded).all():
        raise D122G1Error("D122 decoded basis shape/finite drift")
    gram = decoded @ decoded.T
    if not np.allclose(
        gram, np.eye(RDCE_RANK), rtol=0.0, atol=RAW_DECODED_GRAM_ATOL
    ):
        raise D122G1Error("D122 INT8 decoded raw Gram exceeds quantization tolerance")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    if (
        not np.isfinite(eigenvalues).all()
        or float(np.min(eigenvalues)) <= MIN_BASIS_SINGULAR_VALUE**2
    ):
        raise D122G1Error("D122 INT8 decoded basis lost rank")
    inverse_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    closed = inverse_sqrt @ decoded
    if not np.allclose(
        closed @ closed.T, np.eye(RDCE_RANK), rtol=0.0, atol=2.0e-10
    ):
        raise D122G1Error("D122 INT8 decoded basis orthogonal closure drift")
    return np.ascontiguousarray(closed, dtype=np.float64)


def _decode_positive(codes: np.ndarray, scales: np.ndarray, name: str) -> np.ndarray:
    values = codes.astype(np.float64) * scales.astype(np.float64)
    if values.shape != (RDCE_RANK,) or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise D122G1Error(f"D122 {name} decoded values must remain positive")
    return values


def _parse_wire_array(
    payload: bytes, offset: int, dtype: np.dtype[Any], shape: tuple[int, ...]
) -> tuple[np.ndarray, int]:
    count = int(np.prod(shape, dtype=np.int64))
    length = count * dtype.itemsize
    if offset + length > len(payload):
        raise D122G1Error("D122 RDCE wire is truncated")
    array = np.frombuffer(
        payload[offset : offset + length], dtype=dtype
    ).copy().reshape(shape)
    return array, offset + length


def _parse_asset_wire(path: Path, expected_sha256: str) -> _StandaloneRDCEAsset:
    wire = path.read_bytes()
    if hashlib.sha256(wire).hexdigest() != expected_sha256:
        raise D122G1Error("RDCE asset wire SHA256 mismatch")
    if not wire.startswith(WIRE_MAGIC) or len(wire) < len(WIRE_MAGIC) + 4:
        raise D122G1Error("RDCE asset wire framing drift")
    offset = len(WIRE_MAGIC)
    header_size = struct.unpack(">I", wire[offset : offset + 4])[0]
    offset += 4
    if header_size <= 0 or offset + header_size > len(wire):
        raise D122G1Error("D122 RDCE wire header length drift")
    raw_header = wire[offset : offset + header_size]
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D122G1Error("D122 RDCE wire header JSON drift") from error
    if raw_header != _canonical_bytes(header) or type(header) is not dict:
        raise D122G1Error("D122 RDCE wire header is not canonical JSON")
    offset += header_size
    if (
        set(header) != {"schema", "asset", "asset_receipt_sha256"}
        or header["schema"] != WIRE_SCHEMA
        or type(header["asset"]) is not dict
    ):
        raise D122G1Error("D122 RDCE wire header schema drift")
    asset_header = header["asset"]
    expected_keys = {
        "schema", "candidate_id", "checkpoint_sha256", "method_lock_sha256",
        "runtime_sha256", "split_id", "tap_sha256", "construction_code_sha256",
        "content_root_sha256", "source_receipt_sha256", "tap_receipt_sha256",
        "tap_authority_sha256", "deployment_status", "source_row_count",
        "source_class_count", "receiver_day_cell_count", "rank",
        "z_id_geometry_only", "z_dom_retained", "source_rows_retained",
        "source_names_retained", "basis_codes_qint8", "basis_scales_fp16",
        "tau_codes_qint8", "tau_scales_fp16", "spectrum_codes_qint8",
        "spectrum_scales_fp16",
    }
    if set(asset_header) != expected_keys or asset_header["schema"] != ASSET_SCHEMA:
        raise D122G1Error("D122 RDCE wire asset header keys/schema drift")
    if (
        asset_header["candidate_id"] != CANDIDATE_ASSET_ID
        or asset_header["deployment_status"] != FORMAL_DEPLOYMENT_STATUS
        or type(asset_header["source_row_count"]) is not int
        or type(asset_header["source_class_count"]) is not int
        or type(asset_header["receiver_day_cell_count"]) is not int
        or type(asset_header["rank"]) is not int
        or asset_header["source_row_count"] != SOURCE_ROW_COUNT
        or asset_header["source_class_count"] != SOURCE_CLASS_COUNT
        or asset_header["receiver_day_cell_count"] != RECEIVER_DAY_CELL_COUNT
        or asset_header["rank"] != RDCE_RANK
        or asset_header["z_id_geometry_only"] is not True
        or asset_header["z_dom_retained"] is not False
        or asset_header["source_rows_retained"] is not False
        or asset_header["source_names_retained"] is not False
    ):
        raise D122G1Error("D122 RDCE wire scientific header drift")
    lineage_names = (
        "checkpoint_sha256", "runtime_sha256", "method_lock_sha256", "tap_sha256",
        "construction_code_sha256", "content_root_sha256", "source_receipt_sha256",
        "tap_receipt_sha256", "tap_authority_sha256",
    )
    for name in lineage_names:
        _require_hash(asset_header[name], f"asset.{name}")
    _require_hash(header["asset_receipt_sha256"], "asset_receipt_sha256")
    _require_split(asset_header["split_id"])
    authority_payload = {
        "schema": TAP_AUTHORITY_SCHEMA,
        "loader": "load_d106_phase1_ls_tap",
        "archive_sha256": asset_header["tap_sha256"],
        "receipt_sha256": asset_header["tap_receipt_sha256"],
    }
    if hashlib.sha256(_canonical_bytes(authority_payload)).hexdigest() != asset_header["tap_authority_sha256"]:
        raise D122G1Error("D122 RDCE tap authority digest drift")
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
        value, offset = _parse_wire_array(wire, offset, dtype, shape)
        if _array_receipt(value) != asset_header[name]:
            raise D122G1Error(f"D122 RDCE wire {name} receipt drift")
        if not np.isfinite(value).all():
            raise D122G1Error(f"D122 RDCE wire {name} finite drift")
        if name.endswith("codes_qint8"):
            if np.any(value == np.int8(-128)) or (
                name != "basis_codes_qint8" and np.any(value <= 0)
            ):
                raise D122G1Error(f"D122 RDCE wire {name} code range drift")
        elif np.any(value <= 0.0):
            raise D122G1Error(f"D122 RDCE wire {name} scale range drift")
        arrays[name] = value
    if offset != len(wire):
        raise D122G1Error("D122 RDCE wire has trailing bytes")
    asset = _StandaloneRDCEAsset(
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
        **arrays,
    )
    if hashlib.sha256(_canonical_bytes(_asset_payload(asset))).hexdigest() != asset.asset_receipt_sha256:
        raise D122G1Error("D122 RDCE asset payload receipt drift")
    _orthogonal_closure(asset.basis_codes_qint8, asset.basis_scales_fp16)
    _decode_positive(asset.tau_codes_qint8, asset.tau_scales_fp16, "tau")
    _decode_positive(asset.spectrum_codes_qint8, asset.spectrum_scales_fp16, "spectrum")
    if {
        "schema": WIRE_SCHEMA,
        "asset": _asset_payload(asset),
        "asset_receipt_sha256": asset.asset_receipt_sha256,
    } != header:
        raise D122G1Error("D122 RDCE wire canonical header drift")
    return asset


def _normalized(rows: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != Z_DIM or not np.isfinite(values).all():
        raise D122G1Error("D122 feature layout drift")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise D122G1Error("D122 feature contains a zero-norm row")
    return values / norms


def fit_rdce_sourceheld_state(
    asset: _StandaloneRDCEAsset,
    support_zid: np.ndarray,
    labels: Sequence[str],
    k_shot: int,
) -> dict[str, Any]:
    if type(asset) is not _StandaloneRDCEAsset or k_shot not in K_VALUES:
        raise D122G1Error("D122 RDCE asset/K binding drift")
    support = _normalized(support_zid)
    typed_labels = tuple(str(value) for value in labels)
    if len(typed_labels) != len(support):
        raise D122G1Error("D122 support/label cardinality drift")
    basis = _orthogonal_closure(asset.basis_codes_qint8, asset.basis_scales_fp16)
    tau = _decode_positive(asset.tau_codes_qint8, asset.tau_scales_fp16, "tau")
    if k_shot == 1:
        attenuation = np.full(3, np.float16(0.3), dtype=np.float16)
    else:
        scatters = []
        for class_id in sorted(set(typed_labels)):
            group = support[np.asarray(typed_labels) == class_id]
            if len(group) != k_shot:
                raise D122G1Error("support is not exactly K per class")
            projected = (group - np.mean(group, axis=0)) @ basis.T
            scatters.append(np.sum(np.square(projected), axis=0) / float(k_shot - 1))
        scatter = np.mean(np.stack(scatters), axis=0)
        a0 = min(0.95, 1.5 * k_shot / float(k_shot + 4))
        raw = a0 + 0.2 * np.tanh(np.log((scatter + 1.0e-8) / (tau + 1.0e-8)))
        attenuation = np.asarray(
            np.clip(
                raw,
                float(MIN_ATTENUATION_FP16),
                float(MAX_ATTENUATION_FP16),
            ),
            dtype=np.float16,
        )
    payload = {
        "scope": "SOURCE_HELD_NON_TARGET_NO_P2_AUTHORITY",
        "asset_receipt_sha256": asset.asset_receipt_sha256,
        "K": k_shot,
        "attenuation_fp16": [float(value) for value in attenuation],
        "support_root_sha256": hashlib.sha256(
            np.ascontiguousarray(support, dtype=np.float64).tobytes()
        ).hexdigest(),
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
    }
    return {
        "basis": basis,
        "attenuation": attenuation.astype(np.float64),
        "receipt": _sha(payload),
        "payload": payload,
    }


def apply_rdce_state(state: Mapping[str, Any], rows: np.ndarray) -> np.ndarray:
    values = _normalized(rows)
    basis = np.asarray(state["basis"], dtype=np.float64)
    coefficient = 1.0 - np.sqrt(1.0 - np.asarray(state["attenuation"], dtype=np.float64))
    transformed = values - ((values @ basis.T) * coefficient) @ basis
    transformed /= np.linalg.norm(transformed, axis=1, keepdims=True)
    return np.ascontiguousarray(transformed, dtype=np.float32)


def _lock(k_shot: int, package_sha256: str) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        active_k=k_shot,
        phase1_lodo_receipt_sha256=D105_LOCK_SHA256,
        quantization_margin_audit_sha256=package_sha256,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=0.35,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=0.85,
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metric_base(
    truth: np.ndarray,
    predicted: Sequence[str],
    classes: Sequence[str],
    held_class: str | None,
) -> dict[str, Any]:
    prediction = np.asarray(predicted, dtype=str)
    per_class = {}
    for class_id in classes:
        mask = truth == class_id
        if not np.any(mask):
            raise D122G1Error("truth lacks a registered class")
        per_class[class_id] = float(np.mean(prediction[mask] == class_id))
    old_classes = tuple(class_id for class_id in classes if class_id != held_class)
    old_ba = float(np.mean([per_class[class_id] for class_id in old_classes]))
    new_acc = None if held_class is None else per_class[held_class]
    harmonic = (
        None if new_acc is None or old_ba + new_acc == 0.0
        else 2.0 * old_ba * new_acc / (old_ba + new_acc)
    )
    return {
        "old_balanced_accuracy": old_ba,
        "seen_new_accuracy": new_acc,
        "H_old_new": harmonic,
        "old_floor": min(per_class[class_id] for class_id in old_classes),
        "all_class_floor": min(per_class.values()),
        "balanced_accuracy": float(np.mean(list(per_class.values()))),
        "correct_count": int(np.sum(prediction == truth)),
        "query_count": len(truth),
        "per_class_accuracy": per_class,
    }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise D122G1Error(f"expected JSON object: {path}")
    return value


def _write_new(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def _build_d112_g1_bundle(args: argparse.Namespace, truth_input_seal_sha256: str):
    """Build the frozen D112 G1 aggregate without importing its script wrapper."""

    g0 = build_d112_g0_source_bundle(
        args.d106_tap_archive.resolve(strict=True),
        receipt_path=args.d106_tap_receipt.resolve(strict=True),
        checkpoint_sha256=args.checkpoint_sha256,
        expected_tap_sha256=args.d106_tap_archive_sha256,
    )
    return build_d112_source_held_g1_bundle(
        class_registry=g0.class_registry,
        g=g0.g,
        q0=g0.q0,
        U=g0.U,
        sigma0_r=g0.sigma0_r,
        sigma0_amb=g0.sigma0_amb,
        v_g_r=g0.v_g_r,
        v_g_amb=g0.v_g_amb,
        tau_h_r=g0.tau_h_r,
        checkpoint_sha256=str(g0.manifest["checkpoint_sha256"]),
        source_aggregate_sha256=str(g0.manifest["source_aggregate_sha256"]),
        phase1_seal_sha256=_file_sha(args.d106_tap_receipt.resolve(strict=True)),
        source_held_split_sha256=truth_input_seal_sha256,
        global_bundle_valid=bool(g0.manifest["global_bundle_valid"]),
        global_invalid_reason=str(g0.manifest["global_invalid_reason"]),
        g_quantization_l2_error_bound=g0.g_quantization_l2_error_bound,
        q0_quantization_l2_error_bound=g0.q0_quantization_l2_error_bound,
        U_operator_error_upper_bound=g0.U_operator_error_upper_bound,
        endpoint_quantization_chord_mse=g0.endpoint_quantization_chord_mse,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _run_id(value: str) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > 160:
        raise D122G1Error("run ID must be a short non-empty string")
    return value


def fixed_row_specs(
    receivers: Sequence[str], classes: Sequence[str]
) -> tuple[tuple[str, str | None, int], ...]:
    """The frozen complete G1 matrix; there is no partial-row selector."""

    return tuple(
        [(receiver, None, k) for receiver in receivers for k in K_VALUES]
        + [(receiver, class_id, 1) for receiver in receivers for class_id in classes]
    )


def _validate_package_manifest(
    root: Path,
) -> tuple[
    Path,
    tuple[str, ...],
    tuple[str, ...],
    dict[tuple[str, int], dict[str, Any]],
    str,
]:
    manifest_path = root / "package_manifest.json"
    manifest = _read_json(manifest_path)
    receivers = tuple(str(value) for value in manifest.get("receiver_ids", ()))
    classes = tuple(str(value) for value in manifest.get("class_ids", ()))
    packages = manifest.get("packages")
    truth_seal_sha = manifest.get("truth_input_seal_sha256")
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("candidate_id") != D104_CANDIDATE_ID
        or manifest.get("split_id") != SPLIT_ID
        or manifest.get("query_truth_present") is not False
        or manifest.get("target_access") is not False
        or len(receivers) != 7
        or len(set(receivers)) != 7
        or len(classes) != 6
        or len(set(classes)) != 6
        or not isinstance(packages, list)
        or len(packages) != 21
        or type(truth_seal_sha) is not str
        or len(truth_seal_sha) != 64
    ):
        raise D122G1Error("D122 D104 predictor-package manifest closure drift")
    by_key = {
        (str(row.get("held_receiver")), int(row.get("K"))): row for row in packages
    }
    expected = {(receiver, k_shot) for receiver in receivers for k_shot in K_VALUES}
    if len(by_key) != 21 or set(by_key) != expected:
        raise D122G1Error("D122 fixed 21-package matrix drift")
    return manifest_path, receivers, classes, by_key, truth_seal_sha


def _load_package(
    root: Path,
    package_row: Mapping[str, Any],
    *,
    classes: tuple[str, ...],
    k_shot: int,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, tuple[str, ...], str]:
    relative = Path(str(package_row.get("path", "")))
    path = (root / relative).resolve(strict=True)
    if relative.is_absolute() or not path.is_relative_to(root):
        raise D122G1Error("D122 predictor package path escapes package root")
    if _file_sha(path) != package_row.get("sha256"):
        raise D122G1Error("D122 predictor package SHA drift")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != PACKAGE_KEYS:
            raise D122G1Error("D122 predictor package member closure drift")
        support = np.asarray(archive["support_pre_relu"], dtype=np.float32).copy()
        labels = tuple(str(value) for value in archive["support_labels"].astype(str).tolist())
        query = np.asarray(archive["query_pre_relu"], dtype=np.float32).copy()
        query_ids = tuple(
            str(value) for value in archive["query_physical_ids"].astype(str).tolist()
        )
        registry = tuple(
            str(value) for value in archive["registered_classes"].astype(str).tolist()
        )
    if (
        registry != classes
        or support.shape != (len(classes) * k_shot, 160)
        or query.ndim != 2
        or query.shape[1] != 160
        or len(labels) != len(support)
        or not query_ids
        or len(query_ids) != len(query)
        or len(set(query_ids)) != len(query_ids)
        or any(label not in classes for label in labels)
        or any(labels.count(class_id) != k_shot for class_id in classes)
        or not np.isfinite(support).all()
        or not np.isfinite(query).all()
    ):
        raise D122G1Error("D122 predictor package support/query closure drift")
    return support, labels, query, query_ids, str(package_row["sha256"])


def _baseline_logits(bank: Any, query: np.ndarray) -> np.ndarray:
    return score_zid_student_t_logits(
        bank,
        query,
        metric=identity_shared_psd_metric(config=bank.config),
    )


def _assert_query_zero(audits: Mapping[str, Mapping[str, Any]]) -> None:
    for name, audit in audits.items():
        # D112's sealed static-head audit predates the explicit
        # ``query_state_updates``/``query_selection_count`` fields.  Its fit
        # API has no query argument and its audit binds truth/role/quota to
        # zero; D122's new state additionally records all three counters.
        required_zero = ("query_rows_used_for_fit",)
        compatibility_zero = ("truth_role_quota_inputs",)
        optional_zero = ("query_state_updates", "query_selection_count")
        if any(int(audit.get(field, -1)) != 0 for field in required_zero) or any(
            field in audit and int(audit[field]) != 0 for field in compatibility_zero
        ) or any(
            field in audit and int(audit[field]) != 0 for field in optional_zero
        ):
            raise D122G1Error(f"D122 query lifecycle drift in {name}")


def _build_four_arm_predictions(
    *,
    bundle: Any,
    rdce_asset: Any,
    support_signed: np.ndarray,
    labels: tuple[str, ...],
    query_signed: np.ndarray,
    registry: tuple[str, ...],
    k_shot: int,
    package_sha256: str,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Build one frozen 2x2 row package without opening truth."""

    support_plus = np.ascontiguousarray(np.maximum(support_signed, np.float32(0.0)))
    query_plus = np.ascontiguousarray(np.maximum(query_signed, np.float32(0.0)))
    lock = _lock(k_shot, package_sha256)
    identity_bank = build_typed_zid_support_bank(
        support_plus, labels, registry, config=lock
    )
    m0_logits = _baseline_logits(identity_bank, query_plus)
    head_state = fit_d112_ground_head_source_held_g1_state(bundle, identity_bank)
    head_logits = score_d112_seam_source_held_g1_logits(
        head_state, identity_bank, query_plus
    )

    # These are the exact D106 source-held RDCE fitter and transform used by
    # M_DA.  M_JOINT consumes this same state receipt; it does not refit RDCE.
    da_state = fit_rdce_sourceheld_state(rdce_asset, support_plus, labels, k_shot)
    da_support = apply_rdce_state(da_state, support_plus)
    da_query = apply_rdce_state(da_state, query_plus)
    da_bank = build_typed_zid_support_bank(da_support, labels, registry, config=lock)
    da_logits = _baseline_logits(da_bank, da_query)
    joint_state = fit_d122_rdce_ground_head_source_held_g1_state(
        bundle, da_bank, support_plus, labels, da_state
    )
    joint_logits = score_d122_rdce_ground_head_source_held_g1_logits(
        joint_state, da_bank, da_query
    )

    old_indices = np.asarray(joint_state.old_class_indices, dtype=np.int64)
    new_mask = np.ones(len(registry), dtype=bool)
    new_mask[old_indices] = False
    if (
        not np.array_equal(head_logits[:, new_mask], m0_logits[:, new_mask])
        or not np.array_equal(joint_logits[:, new_mask], da_logits[:, new_mask])
    ):
        raise D122G1Error("D122 new-class logit boundary is not bit-exact")
    head_audit = _jsonable(audit_d112_seam_state(head_state))
    joint_audit = _jsonable(audit_d122_rdce_ground_head_state(joint_state))
    _assert_query_zero({"M_HEAD": head_audit, "M_JOINT": joint_audit})
    if (
        joint_audit.get("global_component_valid") is not True
        or joint_audit.get("global_failure_reason") != "NONE"
        or joint_audit.get("rdce_state_receipt_sha256") != str(da_state["receipt"])
    ):
        # A sealed M_JOINT must be a real, globally bound composition.  The
        # core state can expose an audit-only all-M_DA fallback for diagnosis,
        # but G1 must never silently publish that absent joint factor.
        raise D122G1Error("D122 global joint component binding is fail-closed")
    logits = {
        "M0": m0_logits,
        "M_DA": da_logits,
        "M_HEAD": head_logits,
        "M_JOINT": joint_logits,
    }
    predictions = {
        arm: list(unique_d122_argmax(value, registry)) for arm, value in logits.items()
    }
    if set(predictions) != set(ARMS) or any(
        len(values) != len(query_plus) for values in predictions.values()
    ):
        raise D122G1Error("D122 four-arm prediction closure drift")
    return predictions, {
        "student_t_lock_sha256": lock.lock_digest,
        "M_DA_M_JOINT_rdce_state_sha256": str(da_state["receipt"]),
        "M_HEAD_state_audit": head_audit,
        "M_JOINT_state_audit": joint_audit,
        "arm_logits": {arm: _array_receipt(value) for arm, value in logits.items()},
        "new_class_logit_boundary_bit_exact": True,
    }


def predict(args: argparse.Namespace) -> int:
    """Commit every fixed D122 row before any independent truth operation."""

    root = args.package_root.resolve(strict=True)
    output = args.output_dir.resolve()
    run_id = _run_id(args.run_id)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable D122 prediction root exists: {output}")
    manifest_path, receivers, classes, by_key, truth_seal_sha = _validate_package_manifest(root)
    tap_archive_path = args.d106_tap_archive.resolve(strict=True)
    tap_receipt_path = args.d106_tap_receipt.resolve(strict=True)
    actual_tap_archive_sha256 = _file_sha(tap_archive_path)
    actual_tap_receipt_sha256 = _file_sha(tap_receipt_path)
    if actual_tap_archive_sha256 != args.d106_tap_archive_sha256:
        raise D122G1Error("D122 actual D106 tap archive SHA drift")
    rdce_asset = _parse_asset_wire(
        args.rdce_asset_wire.resolve(strict=True), args.rdce_wire_sha256
    )
    if (
        rdce_asset.split_id != SPLIT_ID
        or rdce_asset.checkpoint_sha256 != args.checkpoint_sha256
        or rdce_asset.tap_sha256 != args.d106_tap_archive_sha256
        or rdce_asset.tap_receipt_sha256 != actual_tap_receipt_sha256
    ):
        raise D122G1Error("D122 RDCE asset lineage/source-held binding mismatch")
    bundle = _build_d112_g1_bundle(args, truth_seal_sha)
    if (
        tuple(bundle.class_registry) != classes
        or bundle.manifest.get("checkpoint_sha256") != args.checkpoint_sha256
        or bundle.manifest.get("phase1_seal_sha256") != actual_tap_receipt_sha256
    ):
        raise D122G1Error("D122 D112 ground bundle lineage/source-held binding mismatch")

    output.mkdir(parents=True, exist_ok=False)
    row_root = output / "rows"
    row_root.mkdir()
    cache: dict[tuple[str, int], tuple[tuple[str, ...], str, dict[str, list[str]], dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for receiver, held_class, k_shot in fixed_row_specs(receivers, classes):
        key = (receiver, k_shot)
        package_row = by_key[key]
        if key not in cache:
            support, labels, query, query_ids, package_sha = _load_package(
                root, package_row, classes=classes, k_shot=k_shot
            )
            predictions, receipts = _build_four_arm_predictions(
                bundle=bundle,
                rdce_asset=rdce_asset,
                support_signed=support,
                labels=labels,
                query_signed=query,
                registry=classes,
                k_shot=k_shot,
                package_sha256=package_sha,
            )
            cache[key] = (query_ids, package_sha, predictions, receipts)
        query_ids, package_sha, predictions, receipts = cache[key]
        row = {
            "schema": PREDICTION_SCHEMA + ".row",
            "candidate_id": CANDIDATE_ID,
            "split_id": SPLIT_ID,
            "run_id": run_id,
            "held_receiver": receiver,
            "held_class": held_class,
            "K": k_shot,
            "package_id": str(package_row["package_id"]),
            "registered_classes": list(classes),
            "query_physical_ids": list(query_ids),
            "arm_predictions": {arm: list(predictions[arm]) for arm in ARMS},
            "shared_component_receipts": {
                "package_sha256": package_sha,
                "rdce_asset_wire_sha256": args.rdce_wire_sha256,
                "d112_bundle_content_root_sha256": bundle.manifest["content_root_sha256"],
                **receipts,
            },
            "query_truth_access": False,
            "target_access": False,
            "formal_p2_authority": False,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
        }
        row["prediction_receipt_sha256"] = _sha(row)
        path = row_root / (
            _sha({"receiver": receiver, "held_class": held_class, "K": k_shot}) + ".json"
        )
        _write_new(path, row)
        rows.append(
            {
                "held_receiver": receiver,
                "held_class": held_class,
                "K": k_shot,
                "package_id": str(package_row["package_id"]),
                "path": str(Path("rows") / path.name),
                "sha256": _file_sha(path),
                "prediction_receipt_sha256": row["prediction_receipt_sha256"],
            }
        )
    if len(rows) != 63 or len({row["prediction_receipt_sha256"] for row in rows}) != 63:
        raise D122G1Error("D122 63-row prediction coverage did not close")
    result = {
        "schema": PREDICTION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "split_id": SPLIT_ID,
        "run_id": run_id,
        "arms": list(ARMS),
        "row_count": 63,
        "arm_row_prediction_unit_count": 63 * len(ARMS),
        "rows": rows,
        "package_manifest_sha256": _file_sha(manifest_path),
        "truth_input_seal_sha256": truth_seal_sha,
        "rdce_asset_wire_sha256": args.rdce_wire_sha256,
        "d112_bundle_content_root_sha256": bundle.manifest["content_root_sha256"],
        "query_truth_access": False,
        "target_access": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "sourceheld_non_target": True,
        "formal_p2_authority": False,
        "sealed_at_unix_ns": time.time_ns(),
    }
    result["prediction_set_receipt_sha256"] = _sha(result)
    _write_new(output / "prediction_manifest.json", result)
    print(output / "prediction_manifest.json")
    return 0


def _validate_truth_open_binding(args: argparse.Namespace) -> tuple[Path, dict[str, Any], Path]:
    root = args.prediction_root.resolve(strict=True)
    manifest_path = root / "prediction_manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema") != PREDICTION_SCHEMA
        or manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("split_id") != SPLIT_ID
        or manifest.get("arms") != list(ARMS)
        or manifest.get("row_count") != 63
        or manifest.get("arm_row_prediction_unit_count") != 63 * len(ARMS)
        or manifest.get("query_truth_access") is not False
        or manifest.get("target_access") is not False
        or manifest.get("query_rows_used_for_fit") != 0
        or manifest.get("query_state_updates") != 0
        or manifest.get("query_selection_count") != 0
        or _sha({key: value for key, value in manifest.items() if key != "prediction_set_receipt_sha256"})
        != manifest.get("prediction_set_receipt_sha256")
    ):
        raise D122G1Error("D122 prediction manifest binding drift")
    truth_seal_path = args.truth_input_seal_json.resolve(strict=True)
    if _file_sha(truth_seal_path) != manifest.get("truth_input_seal_sha256"):
        raise D122G1Error("D122 truth-input seal SHA drift")
    package_manifest_path = (truth_seal_path.parent.parent / "package_manifest.json").resolve(
        strict=True
    )
    package_manifest = _read_json(package_manifest_path)
    truth_seal = _read_json(truth_seal_path)
    packages = package_manifest.get("packages")
    if (
        _file_sha(package_manifest_path) != manifest.get("package_manifest_sha256")
        or package_manifest.get("schema") != PACKAGE_SCHEMA
        or package_manifest.get("candidate_id") != D104_CANDIDATE_ID
        or package_manifest.get("split_id") != SPLIT_ID
        or package_manifest.get("query_truth_present") is not False
        or not isinstance(packages, list)
        or len(packages) != 21
        or package_manifest.get("truth_input_seal_sha256") != _file_sha(truth_seal_path)
        or truth_seal.get("split_id") != SPLIT_ID
        or truth_seal.get("package_count") != 21
        or truth_seal.get("predictor_truth_access") is not False
        or set(truth_seal.get("package_ids", ()))
        != {str(row.get("package_id")) for row in packages}
    ):
        raise D122G1Error("D122 D104 package/truth-seal chain drift")
    return manifest_path, manifest, truth_seal_path


def _metric(
    truth: np.ndarray,
    predicted: Sequence[str],
    classes: Sequence[str],
    held_class: str | None,
) -> dict[str, Any]:
    base = _metric_base(truth, predicted, classes, held_class)
    values = np.asarray(predicted, dtype=str)
    old_mask = np.ones(len(truth), dtype=bool) if held_class is None else truth != held_class
    old_correct = int(np.count_nonzero(values[old_mask] == truth[old_mask]))
    if held_class is None:
        new_correct: int | None = None
    else:
        new_mask = truth == held_class
        new_correct = int(np.count_nonzero(values[new_mask] == truth[new_mask]))
    return {
        **base,
        "old_correct_count": old_correct,
        "seen_new_correct_count": new_correct,
    }


def _effect(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float | int | None]:
    return {
        name: None
        if left[name] is None or right[name] is None
        else left[name] - right[name]
        for name in EFFECT_METRICS
    }


def score(args: argparse.Namespace) -> int:
    """Open held truth only after the complete D122 prediction seal is verified."""

    manifest_path, manifest, seal_path = _validate_truth_open_binding(args)
    output = args.output_json.resolve()
    event_path = args.truth_open_event_json.resolve()
    if output.exists() or event_path.exists():
        raise FileExistsError("immutable D122 score/event output exists")
    root = args.prediction_root.resolve(strict=True)
    entries = manifest.get("rows")
    if not isinstance(entries, list) or len(entries) != 63:
        raise D122G1Error("D122 prediction row list drift")
    artifacts: list[dict[str, Any]] = []
    package_ids: set[str] = set()
    package_queries: dict[str, list[str]] = {}
    for entry in entries:
        relative = Path(str(entry.get("path", "")))
        path = (root / relative).resolve(strict=True)
        if relative.is_absolute() or not path.is_relative_to(root) or _file_sha(path) != entry.get("sha256"):
            raise D122G1Error("D122 prediction row seal drift")
        artifact = _read_json(path)
        if _sha({key: value for key, value in artifact.items() if key != "prediction_receipt_sha256"}) != artifact.get("prediction_receipt_sha256"):
            raise D122G1Error("D122 prediction row receipt drift")
        query_ids = artifact.get("query_physical_ids")
        package_id = str(artifact.get("package_id"))
        if (
            artifact.get("schema") != PREDICTION_SCHEMA + ".row"
            or artifact.get("candidate_id") != CANDIDATE_ID
            or artifact.get("split_id") != SPLIT_ID
            or artifact.get("query_truth_access") is not False
            or artifact.get("target_access") is not False
            or artifact.get("formal_p2_authority") is not False
            or any(artifact.get(field) != 0 for field in ("query_rows_used_for_fit", "query_state_updates", "query_selection_count"))
            or set(artifact.get("arm_predictions", {})) != set(ARMS)
            or not isinstance(query_ids, list)
            or not query_ids
            or any(len(artifact["arm_predictions"][arm]) != len(query_ids) for arm in ARMS)
            or any(entry.get(name) != artifact.get(name) for name in ("held_receiver", "held_class", "K", "package_id", "prediction_receipt_sha256"))
        ):
            raise D122G1Error("D122 prediction row lifecycle drift")
        previous = package_queries.setdefault(package_id, query_ids)
        if previous != query_ids:
            raise D122G1Error("D122 package query IDs drift across matched rows")
        package_ids.add(package_id)
        artifacts.append(artifact)
    receivers = tuple(sorted({str(row["held_receiver"]) for row in artifacts}))
    classes = tuple(str(value) for value in artifacts[0]["registered_classes"])
    if (
        len(package_ids) != 21
        or len(receivers) != 7
        or len(classes) != 6
        or any(tuple(row["registered_classes"]) != classes for row in artifacts)
        or {(str(row["held_receiver"]), row["held_class"], int(row["K"])) for row in artifacts}
        != set(fixed_row_specs(receivers, classes))
    ):
        raise D122G1Error("D122 fixed 63-row prediction coverage drift")

    event = {
        "schema": SCORE_SCHEMA + ".truth_open_event",
        "prediction_manifest_sha256": _file_sha(manifest_path),
        "truth_input_seal_sha256": _file_sha(seal_path),
        "prediction_manifest_mtime_ns": manifest_path.stat().st_mtime_ns,
        "truth_opened_after_all_predictions_committed": True,
        "opened_at_unix_ns": time.time_ns(),
    }
    if event["opened_at_unix_ns"] <= event["prediction_manifest_mtime_ns"]:
        raise D122G1Error("D122 truth-open timestamp is not after prediction seal")
    _write_new(event_path, event)
    truth = _read_json(args.truth_json.resolve(strict=True))
    truth_seal = _read_json(seal_path)
    truth_packages = truth.get("packages")
    if (
        truth.get("schema") != "cvs.d104_r1.rxid_angq.held_truth.v2"
        or truth.get("split_id") != SPLIT_ID
        or truth.get("package_count") != 21
        or truth.get("predictor_access") is not False
        or not isinstance(truth_packages, list)
        or len(truth_packages) != 21
        or _canonical_sha256(truth_packages) != truth_seal.get("truth_package_root_sha256")
    ):
        raise D122G1Error("D122 independent truth closure drift")
    truth_by_package = {str(row["package_id"]): row for row in truth_packages}
    if set(truth_by_package) != package_ids:
        raise D122G1Error("D122 truth/prediction package identity drift")

    scored_rows: list[dict[str, Any]] = []
    negative = {name: {metric: 0 for metric in EFFECT_METRICS} for name in EFFECT_PAIRS}
    for artifact in artifacts:
        matching = truth_by_package[str(artifact["package_id"])]
        if matching.get("query_physical_ids") != artifact["query_physical_ids"]:
            raise D122G1Error("D122 truth/prediction physical-ID alignment drift")
        labels = np.asarray(matching["query_truth_labels"], dtype=str)
        metrics = {
            arm: _metric(labels, artifact["arm_predictions"][arm], artifact["registered_classes"], artifact["held_class"])
            for arm in ARMS
        }
        effects = {name: _effect(metrics[left], metrics[right]) for name, (left, right) in EFFECT_PAIRS.items()}
        for name, values in effects.items():
            for metric, delta in values.items():
                if delta is not None and delta < 0:
                    negative[name][metric] += 1
        interaction = {
            metric: None
            if effects["HEAD_AT_DA"][metric] is None or effects["HEAD_AT_ID"][metric] is None
            else effects["HEAD_AT_DA"][metric] - effects["HEAD_AT_ID"][metric]
            for metric in EFFECT_METRICS
        }
        scored_rows.append(
            {
                "held_receiver": artifact["held_receiver"],
                "held_class": artifact["held_class"],
                "K": artifact["K"],
                "package_id": artifact["package_id"],
                "arm_metrics": metrics,
                "same_row_effects": effects,
                "factorial_interaction": interaction,
                "prediction_receipt_sha256": artifact["prediction_receipt_sha256"],
            }
        )
    result = {
        "schema": SCORE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "split_id": SPLIT_ID,
        "arms": list(ARMS),
        "performance_rows": scored_rows,
        "negative_tail_row_counts": negative,
        "interaction_definition": "(M_JOINT-M_DA)-(M_HEAD-M0)",
        "prediction_manifest_sha256": _file_sha(manifest_path),
        "truth_input_seal_sha256": _file_sha(seal_path),
        "truth_sha256": _file_sha(args.truth_json.resolve(strict=True)),
        "truth_open_event_sha256": _file_sha(event_path),
        "prediction_artifact_committed_before_truth": True,
        "target_access": False,
    }
    result["score_set_receipt_sha256"] = _sha(result)
    _write_new(output, result)
    print(output)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    predictor = commands.add_parser("predict")
    predictor.add_argument("--package-root", type=Path, required=True)
    predictor.add_argument("--rdce-asset-wire", type=Path, required=True)
    predictor.add_argument("--rdce-wire-sha256", required=True)
    predictor.add_argument("--d106-tap-archive", type=Path, required=True)
    predictor.add_argument("--d106-tap-receipt", type=Path, required=True)
    predictor.add_argument("--d106-tap-archive-sha256", required=True)
    predictor.add_argument("--checkpoint-sha256", required=True)
    predictor.add_argument("--run-id", required=True)
    predictor.add_argument("--output-dir", type=Path, required=True)
    scorer = commands.add_parser("score")
    scorer.add_argument("--prediction-root", type=Path, required=True)
    scorer.add_argument("--truth-json", type=Path, required=True)
    scorer.add_argument("--truth-input-seal-json", type=Path, required=True)
    scorer.add_argument("--truth-open-event-json", type=Path, required=True)
    scorer.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return predict(args) if args.command == "predict" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
