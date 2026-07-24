"""Support-only ADV3B02 TS-DRQKNN-BCRR/r7-q3support1 primitives.

This module is deliberately a small, typed runtime.  It has no capsule
builder, truth-side scorer, receiver/TX input, query-label input, optimizer,
or cross-query state.  A caller may fit a Stage2-B state from support and may
only extend it through :func:`append_stage2_c`.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_dssc_zdom_jg_qknn_r4_bcrr import (
    canonical_method_lock,
    qknn_lock_from_method_lock,
)
from cvsrffi.stage2_svrn_bcr import (
    BCRR_DENOMINATOR,
    BCRR_MAX_OMEGA,
    _bcr_quant_audit as _existing_bcr_quant_audit,
    _cross_view_loo_scores as _existing_bcr_cross_view_loo,
    _masked_views as _existing_bcr_masked_views,
    _ridge_fit_and_loo as _existing_bcr_ridge_fit_and_loo,
    normalize_score_rows as _normalize_existing_scores,
)
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    TypedSharedPSDMetric,
    _identity_class_scales as _existing_identity_class_scales,
    _quantize_rows as _existing_qknn_quantize_rows,
    identity_shared_psd_metric,
    normalize_zid_rows,
)


CANDIDATE = "ADV3B02-TS-DRQKNN-BCRR/r7-q3support1"
# qzero1 remains query-only; r7-q3support1 changes only the persisted z_id
# support codec and its seals, never support semantics or decision formulae.
PREDICTION_REVISION = "qzero1"
SCHEMA = "cvs.stage2.adv3b02.ts_drqknn_bcrr.r7_q3support1_bcr3_zidtotal1"
Z_DIM = 160
RANK = 2
ARMS = ("M0", "M_DA", "M_OTHER", "M_JOINT")
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
K_VALUES = (1, 5, 10)
MAX_WIRE_BYTES = 256 * 1024
APPEND_RECEIPT_SCHEMA = "cvs.stage2.adv3b02.append_receipt.r7_q3support1_bcr3_zidtotal1"
ACTUAL_BRANCH_SCHEMA = "cvs.stage2.adv3b02.actual_bank_branch.r7_q3support1_bcr3_zidtotal1"
ZID_REPAIR_SCHEMA = "cvs.stage2.adv3b02.zid_repair_receipt.v1"
ZID_REPAIR_RULE = "finite_exact_zero_singleton_class_medoid_v1"
AFFINE_BANK_SCHEMA = "cvs.stage2.adv3b02.affine_int8_bank.r4_q3support1"
AFFINE_WIRE_SCHEMA = "cvs.stage2.adv3b02.affine_int8_wire.r4_q3support1"
SUPPORT_CODEC = "affine_int8_fp16_scale_offset_plus_two_symmetric_int8_fp16_residuals_v1"
SUPPORT_PLANE_ORDER = ("affine_base", "symmetric_residual_q2", "symmetric_residual_q3_after_float32_d2")
SUPPORT_ROUNDING = "numpy_rint_ties_to_even"
SUPPORT_CLIP = (-127, 127)
SUPPORT_RESIDUAL_SCALE_FLOOR = float(np.finfo(np.float16).smallest_subnormal)
CLASS_BANDWIDTH_CODEC = "fp16_hi_plus_fp16_lo_v1"
BCR_WEIGHT_CODEC = "fixed_three_plane_per_class_symmetric_int8_fp16_scale_v1"
BCR_WEIGHT_PLANE_ORDER = (
    "plane1_teacher",
    "plane2_residual_after_plane1_decode",
    "plane3_residual_after_float32_plane1_plus_plane2_decode",
)
BCR_WEIGHT_CODE_DTYPE = "int8"
BCR_WEIGHT_SCALE_DTYPE = "<f2"
BCR_WEIGHT_ROUNDING = "numpy_rint_ties_to_even"
BCR_WEIGHT_CLIP = (-127, 127)
BCR_WEIGHT_SCALE_FLOOR = float(np.finfo(np.float16).smallest_subnormal)


class ADV3B02StateError(ValueError):
    """A frozen typed-state or protocol invariant was violated."""


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def typed_tokens(values: Sequence[Any] | np.ndarray, *, name: str,
                 unique: bool = False) -> tuple[str, ...]:
    """Accept opaque unicode handles without stringifying foreign values."""
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or values.dtype.kind != "U":
            raise ADV3B02StateError(f"{name} must be a one-dimensional unicode ndarray")
        result = tuple(values.tolist())
    elif isinstance(values, (tuple, list)):
        result = tuple(values)
    else:
        raise ADV3B02StateError(f"{name} must be a typed token sequence")
    if not result or any(type(item) is not str or not item for item in result):
        raise ADV3B02StateError(f"{name} contains an empty or non-string token")
    if unique and len(set(result)) != len(result):
        raise ADV3B02StateError(f"{name} must be unique")
    return result


def _rows(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != Z_DIM:
        raise ADV3B02StateError(f"{name} must be float32 [N,{Z_DIM}]")
    if not len(array) or not np.isfinite(array).all():
        raise ADV3B02StateError(f"{name} must be nonempty and finite")
    return np.ascontiguousarray(array)


def _unit(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1.0e-12):
        raise ADV3B02StateError("feature row has zero or non-finite L2 norm")
    return np.asarray(rows / norms, np.float32)


def _quantize_rows(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The existing symmetric z_dom codec; it is not the z_id qKNN codec."""
    rows = np.asarray(value, np.float32)
    scale = np.maximum(np.max(np.abs(rows), axis=1), 1.0e-8) / 127.0
    codes = np.clip(np.rint(rows / scale[:, None]), -127, 127).astype(np.int8)
    return np.ascontiguousarray(codes), np.asarray(scale, np.float16)


def _dequantize_rows(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    if (codes.dtype != np.int8 or codes.ndim != 2 or codes.shape[1] != Z_DIM
            or scales.dtype != np.float16 or scales.shape != (len(codes),)):
        raise ADV3B02StateError("INT8 row state shape/dtype drift")
    return np.asarray(codes.astype(np.float32) * scales.astype(np.float32)[:, None], np.float32)


def _affine_quantize_rows(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frozen r2 affine support codec: int8 code + FP16 scale + FP16 offset."""
    rows = _unit(_rows(value, name="affine z_id support"))
    lo = np.min(rows, axis=1).astype(np.float32)
    hi = np.max(rows, axis=1).astype(np.float32)
    span = hi - lo
    if np.any(~np.isfinite(span)) or np.any(span <= 0.0):
        raise ADV3B02StateError("affine z_id support has zero/nonfinite range")
    scales = np.asarray(span / 254.0, dtype="<f2")
    offsets = np.asarray((hi + lo) / 2.0, dtype="<f2")
    if (np.any(~np.isfinite(scales)) or np.any(scales <= 0.0)
            or np.any(~np.isfinite(offsets))):
        raise ADV3B02StateError("affine FP16 scale/offset closure failed")
    codes = np.clip(
        np.rint((rows - offsets.astype(np.float32)[:, None])
                / scales.astype(np.float32)[:, None]),
        -127,
        127,
    ).astype(np.int8)
    decoded = codes.astype(np.float32) * scales.astype(np.float32)[:, None] + offsets.astype(np.float32)[:, None]
    if not np.isfinite(decoded).all():
        raise ADV3B02StateError("affine INT8 decode became nonfinite")
    return np.ascontiguousarray(codes), np.ascontiguousarray(scales), np.ascontiguousarray(offsets)


def _affine_decode_base_rows(
    codes: np.ndarray, scales: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    if (codes.dtype != np.int8 or codes.ndim != 2 or codes.shape[1] != Z_DIM
            or scales.dtype != np.dtype("<f2") or offsets.dtype != np.dtype("<f2")
            or scales.shape != (len(codes),) or offsets.shape != (len(codes),)
            or np.any(codes == np.int8(-128)) or not np.isfinite(scales).all()
            or not np.isfinite(offsets).all() or np.any(scales <= 0.0)):
        raise ADV3B02StateError("affine INT8 row state shape/dtype drift")
    decoded = codes.astype(np.float32) * scales.astype(np.float32)[:, None] + offsets.astype(np.float32)[:, None]
    if not np.isfinite(decoded).all():
        raise ADV3B02StateError("affine INT8 decoded support is nonfinite")
    return np.ascontiguousarray(decoded)


def _quantize_support_residual(
    residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = _rows(np.asarray(residual), name="affine support residual")
    scales = np.asarray(
        np.maximum(
            np.max(np.abs(value), axis=1) / 127.0,
            SUPPORT_RESIDUAL_SCALE_FLOOR,
        ),
        dtype="<f2",
    )
    if (
        scales.shape != (len(value),)
        or not np.isfinite(scales).all()
        or np.any(scales < np.float16(SUPPORT_RESIDUAL_SCALE_FLOOR))
    ):
        raise ADV3B02StateError("support residual FP16 scale closure failed")
    codes = np.clip(
        np.rint(value / scales.astype(np.float32)[:, None]),
        SUPPORT_CLIP[0],
        SUPPORT_CLIP[1],
    ).astype(np.int8)
    decoded = codes.astype(np.float32) * scales.astype(np.float32)[:, None]
    return (
        np.ascontiguousarray(codes),
        np.ascontiguousarray(scales),
        np.ascontiguousarray(decoded),
    )


def _affine_quantize_rows_two_plane(
    value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed Q1/Q2/Q3 support codec; retained helper name keeps call sites narrow."""
    raw = _rows(value, name="three-plane affine z_id support")
    teacher = _unit(raw)
    codes, scales, offsets = _affine_quantize_rows(raw)
    base = _affine_decode_base_rows(codes, scales, offsets)
    residual_codes, residual_scales, q2 = _quantize_support_residual(
        np.asarray(teacher - base, np.float32)
    )
    d2 = np.asarray(base + q2, np.float32)
    residual2_codes, residual2_scales, _ = _quantize_support_residual(
        np.asarray(teacher - d2, np.float32)
    )
    return codes, scales, offsets, residual_codes, residual_scales, residual2_codes, residual2_scales


def _affine_dequantize_rows(
    codes: np.ndarray,
    scales: np.ndarray,
    offsets: np.ndarray,
    residual_codes: np.ndarray,
    residual_scales: np.ndarray,
    residual2_codes: np.ndarray,
    residual2_scales: np.ndarray,
) -> np.ndarray:
    base = _affine_decode_base_rows(codes, scales, offsets)
    if (
        residual_codes.dtype != np.int8
        or residual_codes.shape != base.shape
        or residual_scales.dtype != np.dtype("<f2")
        or residual_scales.shape != (len(base),)
        or np.any(residual_codes == np.int8(-128))
        or not np.isfinite(residual_scales).all()
        or np.any(residual_scales < np.float16(SUPPORT_RESIDUAL_SCALE_FLOOR))
    ):
        raise ADV3B02StateError("support residual INT8 row state shape/dtype drift")
    d2 = np.asarray(
        base
        + residual_codes.astype(np.float32)
        * residual_scales.astype(np.float32)[:, None],
        np.float32,
    )
    if (
        residual2_codes.dtype != np.int8
        or residual2_codes.shape != base.shape
        or residual2_scales.dtype != np.dtype("<f2")
        or residual2_scales.shape != (len(base),)
        or np.any(residual2_codes == np.int8(-128))
        or not np.isfinite(residual2_scales).all()
        or np.any(residual2_scales < np.float16(SUPPORT_RESIDUAL_SCALE_FLOOR))
    ):
        raise ADV3B02StateError("support Q3 INT8 row state shape/dtype drift")
    deployed = np.asarray(
        d2
        + residual2_codes.astype(np.float32)
        * residual2_scales.astype(np.float32)[:, None],
        np.float32,
    )
    if not np.isfinite(deployed).all():
        raise ADV3B02StateError("three-plane affine INT8 decode became nonfinite")
    return _unit(deployed)


def _split_class_bandwidths(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(value, np.float32)
    if raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all() or np.any(raw <= 0.0):
        raise ADV3B02StateError("raw class bandwidth must be finite positive [C]")
    hi = np.asarray(raw, dtype="<f2")
    lo = np.asarray(raw - hi.astype(np.float32), dtype="<f2")
    _reconstruct_class_bandwidths(hi, lo)
    return np.ascontiguousarray(hi), np.ascontiguousarray(lo)


def _reconstruct_class_bandwidths(
    hi: np.ndarray, lo: np.ndarray
) -> np.ndarray:
    if (
        hi.dtype != np.dtype("<f2")
        or lo.dtype != np.dtype("<f2")
        or hi.ndim != 1
        or lo.shape != hi.shape
        or not len(hi)
        or not np.isfinite(hi).all()
        or not np.isfinite(lo).all()
    ):
        raise ADV3B02StateError("dual-FP16 class bandwidth state drift")
    deployed = np.asarray(
        hi.astype(np.float32) + lo.astype(np.float32), np.float32
    )
    if not np.isfinite(deployed).all() or np.any(deployed <= 0.0):
        raise ADV3B02StateError("dual-FP16 class bandwidth decode failed")
    return np.ascontiguousarray(deployed)


def _quantize_bcr_weight_plane(
    value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize one fixed per-class BCR weight plane."""
    weights = np.asarray(value, np.float64)
    if (
        weights.ndim != 2
        or weights.shape[0] != Z_DIM
        or weights.shape[1] < 2
        or not np.isfinite(weights).all()
    ):
        raise ADV3B02StateError("BCR weight plane must be finite [160,C>=2]")
    scales = np.asarray(
        np.maximum(
            np.max(np.abs(weights), axis=0) / 127.0,
            BCR_WEIGHT_SCALE_FLOOR,
        ),
        dtype="<f2",
    )
    if (
        not np.isfinite(scales).all()
        or np.any(scales <= 0.0)
        or scales.shape != (weights.shape[1],)
    ):
        raise ADV3B02StateError("BCR weight FP16 plane scale closure failed")
    codes = np.clip(
        np.rint(weights / scales.astype(np.float32)[None, :]),
        BCR_WEIGHT_CLIP[0],
        BCR_WEIGHT_CLIP[1],
    ).astype(np.int8)
    decoded = (
        codes.astype(np.float32) * scales.astype(np.float32)[None, :]
    )
    return (
        np.ascontiguousarray(codes),
        np.ascontiguousarray(scales),
        np.ascontiguousarray(decoded),
    )


def _quantize_bcr_weights_three_plane(
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed three-plane INT8 residual codec with frozen float32 accumulation."""
    teacher = np.asarray(weights, np.float64)
    plane1_codes, plane1_scales, plane1 = _quantize_bcr_weight_plane(teacher)
    d1 = np.asarray(plane1, np.float32)
    residual2 = teacher - d1.astype(np.float64)
    plane2_codes, plane2_scales, plane2 = _quantize_bcr_weight_plane(residual2)
    d2 = np.asarray(d1 + plane2.astype(np.float32), np.float32)
    residual3 = teacher - d2.astype(np.float64)
    plane3_codes, plane3_scales, plane3 = _quantize_bcr_weight_plane(residual3)
    deployed = np.asarray(d2 + plane3.astype(np.float32), np.float32)
    if not np.isfinite(deployed).all():
        raise ADV3B02StateError("three-plane BCR weight decode became nonfinite")
    return (
        plane1_codes,
        plane1_scales,
        plane2_codes,
        plane2_scales,
        plane3_codes,
        plane3_scales,
        np.ascontiguousarray(deployed),
    )


def _logsumexp(value: np.ndarray) -> float:
    maximum = float(np.max(value))
    return maximum + math.log(float(np.exp(value - maximum).sum()))


def _balanced_k(labels: tuple[str, ...], classes: tuple[str, ...]) -> int:
    counts = tuple(labels.count(item) for item in classes)
    if len(set(counts)) != 1 or counts[0] not in K_VALUES:
        raise ADV3B02StateError("support must be exact balanced K in {1,5,10}")
    return counts[0]


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(item not in "0123456789abcdef" for item in value)
    ):
        raise ADV3B02StateError(f"{name} SHA256 drift")
    return value


_ZID_REPAIR_RECEIPT_FIELDS = {
    "schema", "rule", "k_shot", "input_support_sha256",
    "output_support_sha256", "unit_output_support_sha256",
    "zero_row_token_root_sha256", "donor_token_root_sha256",
    "class_repair_counts", "normal_rows_bitwise_preserved",
    "repaired_row_count", "query_rows_used_for_fit", "receipt_sha256",
}


def _canonical_support_positions(labels: tuple[str, ...], tokens: tuple[str, ...]) -> list[int]:
    return sorted(range(len(tokens)), key=lambda index: (labels[index], tokens[index]))


def _unit_support_token_binding_sha256(
    support_zid: np.ndarray, tokens: tuple[str, ...]
) -> str:
    """Order-invariant support teacher binding: token plus normalized row."""
    positions = sorted(range(len(tokens)), key=lambda index: tokens[index])
    payload = [
        (tokens[index], _unit(support_zid[index : index + 1]).tobytes().hex())
        for index in positions
    ]
    return sha256_bytes(_canon(payload))


def verify_zid_repair_receipt(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fail closed on the sealed support-only raw-z_id repair evidence."""
    value = dict(receipt) if isinstance(receipt, Mapping) else {}
    if set(value) != _ZID_REPAIR_RECEIPT_FIELDS:
        raise ADV3B02StateError("z_id repair receipt schema drift")
    for name in _ZID_REPAIR_RECEIPT_FIELDS:
        if name.endswith("_sha256"):
            _require_sha256(value[name], name=name)
    counts = value["class_repair_counts"]
    if (
        value["schema"] != ZID_REPAIR_SCHEMA
        or value["rule"] != ZID_REPAIR_RULE
        or type(value["k_shot"]) is not int
        or value["k_shot"] not in K_VALUES
        or type(counts) is not dict
        or not counts
        or any(type(key) is not str or type(item) is not int or item < 0 for key, item in counts.items())
        or type(value["normal_rows_bitwise_preserved"]) is not bool
        or value["normal_rows_bitwise_preserved"] is not True
        or type(value["repaired_row_count"]) is not int
        or value["repaired_row_count"] < 0
        or value["repaired_row_count"] != sum(counts.values())
        or value["query_rows_used_for_fit"] != 0
    ):
        raise ADV3B02StateError("z_id repair receipt value drift")
    body = {key: value[key] for key in value if key != "receipt_sha256"}
    if value["receipt_sha256"] != sha256_bytes(_canon(body)):
        raise ADV3B02StateError("z_id repair receipt SHA drift")
    return value


def repair_finite_exact_zero_singleton_class_medoid(
    support_zid: np.ndarray,
    support_labels: Sequence[Any] | np.ndarray,
    registered_classes: Sequence[Any] | np.ndarray,
    support_physical_tokens: Sequence[Any] | np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repair only the frozen finite exact-zero K5/K10 singleton condition."""
    source = _rows(support_zid, name="raw z_id repair support")
    labels = typed_tokens(support_labels, name="raw z_id repair labels")
    classes = typed_tokens(registered_classes, name="raw z_id repair classes", unique=True)
    tokens = typed_tokens(support_physical_tokens, name="raw z_id repair physical tokens", unique=True)
    if len(source) != len(labels) or len(labels) != len(tokens) or any(item not in classes for item in labels):
        raise ADV3B02StateError("raw z_id repair support layout drift")
    k_shot = _balanced_k(labels, classes)
    zero_rows = np.all(source == np.float32(0.0), axis=1)
    norms = np.linalg.norm(source.astype(np.float64), axis=1)
    if np.any((~zero_rows) & (norms <= 1.0e-12)):
        raise ADV3B02StateError("raw z_id repair rejects tiny nonzero support")
    repaired = np.array(source, dtype=np.float32, copy=True, order="C")
    repaired_counts = {item: 0 for item in classes}
    zero_tokens: list[str] = []
    donor_tokens: list[str] = []
    for class_handle in classes:
        members = [index for index, label in enumerate(labels) if label == class_handle]
        zeros = [index for index in members if bool(zero_rows[index])]
        if not zeros:
            continue
        if k_shot not in (5, 10) or len(zeros) != 1:
            raise ADV3B02StateError("raw z_id repair requires one exact zero in K5/K10 class")
        peers = [index for index in members if index not in zeros]
        if len(peers) != k_shot - 1 or any(norms[index] <= 1.0e-12 for index in peers):
            raise ADV3B02StateError("raw z_id repair lacks finite nonzero same-class peers")
        peers = sorted(peers, key=lambda index: tokens[index])
        unit = source[np.asarray(peers, np.intp)].astype(np.float64)
        unit /= np.linalg.norm(unit, axis=1, keepdims=True)
        cosine_sums = np.sum(unit @ unit.T, axis=1) - 1.0
        donor_local = 0
        for index in range(1, len(peers)):
            if cosine_sums[index] > cosine_sums[donor_local]:
                donor_local = index
        zero_index = zeros[0]
        donor_index = peers[donor_local]
        repaired[zero_index] = source[donor_index]
        repaired_counts[class_handle] = 1
        zero_tokens.append(tokens[zero_index])
        donor_tokens.append(tokens[donor_index])
    normal = ~zero_rows
    body = {
        "schema": ZID_REPAIR_SCHEMA,
        "rule": ZID_REPAIR_RULE,
        "k_shot": k_shot,
        "input_support_sha256": sha256_bytes(np.ascontiguousarray(source).tobytes()),
        "output_support_sha256": sha256_bytes(np.ascontiguousarray(repaired).tobytes()),
        "unit_output_support_sha256": _unit_support_token_binding_sha256(repaired, tokens),
        "zero_row_token_root_sha256": sha256_bytes(_canon(sorted(zero_tokens))),
        "donor_token_root_sha256": sha256_bytes(_canon(sorted(donor_tokens))),
        "class_repair_counts": repaired_counts,
        "normal_rows_bitwise_preserved": bool(
            np.ascontiguousarray(source[normal]).tobytes()
            == np.ascontiguousarray(repaired[normal]).tobytes()
        ),
        "repaired_row_count": int(np.sum(zero_rows)),
        "query_rows_used_for_fit": 0,
    }
    receipt = {**body, "receipt_sha256": sha256_bytes(_canon(body))}
    return repaired, dict(verify_zid_repair_receipt(receipt))


def _validate_repaired_support_for_state(
    receipt: Mapping[str, Any], *, support_zid: np.ndarray,
    labels: tuple[str, ...], classes: tuple[str, ...], tokens: tuple[str, ...],
    require_output_support_sha256: bool = True,
) -> Mapping[str, Any]:
    value = verify_zid_repair_receipt(receipt)
    if (
        value["k_shot"] != _balanced_k(labels, classes)
        or set(value["class_repair_counts"]) != set(classes)
        or (
            require_output_support_sha256
            and value["output_support_sha256"]
            != sha256_bytes(np.ascontiguousarray(support_zid).tobytes())
        )
        or value["unit_output_support_sha256"]
        != _unit_support_token_binding_sha256(support_zid, tokens)
    ):
        raise ADV3B02StateError("z_id repair/state teacher binding drift")
    return value


@dataclass(frozen=True)
class AffineINT8ZIDSupportBank:
    """Candidate-local fixed Q1/Q2/Q3 support bank."""

    classes: tuple[str, ...]
    support_counts: tuple[int, ...]
    codes_qint8: np.ndarray
    scales_fp16: np.ndarray
    offsets_fp16: np.ndarray
    residual_codes_qint8: np.ndarray
    residual_scales_fp16: np.ndarray
    residual2_codes_qint8: np.ndarray
    residual2_scales_fp16: np.ndarray
    class_indices_int16: np.ndarray
    class_scale_hi_fp16: np.ndarray
    class_scale_lo_fp16: np.ndarray
    active_k: int
    config: Phase1ZIDStudentTLock
    config_lock_digest: str
    quantization_audit: Mapping[str, Any]
    bank_receipt_sha256: str
    schema: str = AFFINE_BANK_SCHEMA

    def __post_init__(self) -> None:
        n = len(self.codes_qint8)
        if (self.schema != AFFINE_BANK_SCHEMA or self.active_k not in K_VALUES
                or self.config.active_k != self.active_k
                or self.config.lock_digest != self.config_lock_digest
                or self.codes_qint8.dtype != np.int8
                or self.codes_qint8.shape != (n, Z_DIM)
                or self.scales_fp16.dtype != np.dtype("<f2")
                or self.offsets_fp16.dtype != np.dtype("<f2")
                or self.scales_fp16.shape != (n,) or self.offsets_fp16.shape != (n,)
                or self.residual_codes_qint8.dtype != np.int8
                or self.residual_codes_qint8.shape != (n, Z_DIM)
                or self.residual_scales_fp16.dtype != np.dtype("<f2")
                or self.residual_scales_fp16.shape != (n,)
                or self.residual2_codes_qint8.dtype != np.int8
                or self.residual2_codes_qint8.shape != (n, Z_DIM)
                or self.residual2_scales_fp16.dtype != np.dtype("<f2")
                or self.residual2_scales_fp16.shape != (n,)
                or self.class_indices_int16.dtype != np.dtype("<i2")
                or self.class_indices_int16.shape != (n,)
                or self.class_scale_hi_fp16.dtype != np.dtype("<f2")
                or self.class_scale_lo_fp16.dtype != np.dtype("<f2")
                or self.class_scale_hi_fp16.shape != (len(self.classes),)
                or self.class_scale_lo_fp16.shape != (len(self.classes),)
                or np.any(self.codes_qint8 == np.int8(-128))
                or np.any(self.residual_codes_qint8 == np.int8(-128))
                or np.any(self.residual2_codes_qint8 == np.int8(-128))
                or not np.isfinite(self.scales_fp16).all() or np.any(self.scales_fp16 <= 0.0)
                or not np.isfinite(self.offsets_fp16).all()
                or not np.isfinite(self.residual_scales_fp16).all()
                or np.any(self.residual_scales_fp16 < np.float16(SUPPORT_RESIDUAL_SCALE_FLOOR))
                or not np.isfinite(self.residual2_scales_fp16).all()
                or np.any(self.residual2_scales_fp16 < np.float16(SUPPORT_RESIDUAL_SCALE_FLOOR))):
            raise ADV3B02StateError("affine bank invariant drift")
        _reconstruct_class_bandwidths(
            self.class_scale_hi_fp16, self.class_scale_lo_fp16
        )
        counts = tuple(int(np.sum(self.class_indices_int16 == i)) for i in range(len(self.classes)))
        if (counts != self.support_counts or any(v != self.active_k for v in counts)
                or np.any(self.class_indices_int16 < 0) or np.any(self.class_indices_int16 >= len(self.classes))):
            raise ADV3B02StateError("affine bank class closure drift")
        payload = _affine_bank_payload(self)
        # A zero receipt exists only inside the constructor call that derives
        # the immutable receipt; no Int8QKNNState accepts that provisional bank.
        if (self.bank_receipt_sha256 != "0" * 64
                and self.bank_receipt_sha256 != sha256_bytes(_canon(payload))):
            raise ADV3B02StateError("affine bank receipt drift")

    @property
    def support_row_count(self) -> int:
        return len(self.codes_qint8)

    def deployed_class_scales(self) -> np.ndarray:
        return _reconstruct_class_bandwidths(
            self.class_scale_hi_fp16, self.class_scale_lo_fp16
        )


def _affine_bank_payload(bank: AffineINT8ZIDSupportBank) -> dict[str, Any]:
    return {"schema": AFFINE_BANK_SCHEMA, "support_codec": SUPPORT_CODEC,
            "support_plane_order": list(SUPPORT_PLANE_ORDER),
            "support_rounding": SUPPORT_ROUNDING, "support_clip": list(SUPPORT_CLIP),
            "support_residual_scale_floor": SUPPORT_RESIDUAL_SCALE_FLOOR,
            "class_bandwidth_codec": CLASS_BANDWIDTH_CODEC,
            "classes": list(bank.classes), "support_counts": list(bank.support_counts),
            "codes_sha256": sha256_bytes(np.ascontiguousarray(bank.codes_qint8).tobytes()),
            "scales_sha256": sha256_bytes(np.ascontiguousarray(bank.scales_fp16).tobytes()),
            "offsets_sha256": sha256_bytes(np.ascontiguousarray(bank.offsets_fp16).tobytes()),
            "residual_codes_sha256": sha256_bytes(np.ascontiguousarray(bank.residual_codes_qint8).tobytes()),
            "residual_scales_sha256": sha256_bytes(np.ascontiguousarray(bank.residual_scales_fp16).tobytes()),
            "residual2_codes_sha256": sha256_bytes(np.ascontiguousarray(bank.residual2_codes_qint8).tobytes()),
            "residual2_scales_sha256": sha256_bytes(np.ascontiguousarray(bank.residual2_scales_fp16).tobytes()),
            "class_indices_sha256": sha256_bytes(np.ascontiguousarray(bank.class_indices_int16).tobytes()),
            "class_scale_hi_sha256": sha256_bytes(np.ascontiguousarray(bank.class_scale_hi_fp16).tobytes()),
            "class_scale_lo_sha256": sha256_bytes(np.ascontiguousarray(bank.class_scale_lo_fp16).tobytes()),
            "active_k": bank.active_k, "config_lock_digest": bank.config_lock_digest,
            "quantization_audit": dict(bank.quantization_audit), "query_rows_used_for_fit": 0}


def _affine_wire_array(name: str, value: np.ndarray) -> bytes:
    array = np.ascontiguousarray(value)
    raw = array.tobytes(order="C")
    return struct.pack("<H", len(name)) + name.encode("ascii") + struct.pack("<H", len(array.dtype.str)) + array.dtype.str.encode("ascii") + struct.pack("<H", array.ndim) + struct.pack("<" + "Q" * array.ndim, *array.shape) + struct.pack("<Q", len(raw)) + raw


def _serialize_affine_bank(bank: AffineINT8ZIDSupportBank) -> bytes:
    header = _canon({"schema": AFFINE_WIRE_SCHEMA, "bank_schema": bank.schema, "classes": list(bank.classes),
                     "support_counts": list(bank.support_counts), "active_k": bank.active_k,
                     "config_lock_digest": bank.config_lock_digest, "bank_receipt_sha256": bank.bank_receipt_sha256,
                     "quantization_audit": dict(bank.quantization_audit), "query_state_updates": 0,
                     "endianness": "little"})
    arrays = (
        "codes_qint8", bank.codes_qint8.astype("|i1"),
        "scales_fp16", bank.scales_fp16.astype("<f2"),
        "offsets_fp16", bank.offsets_fp16.astype("<f2"),
        "residual_codes_qint8", bank.residual_codes_qint8.astype("|i1"),
        "residual_scales_fp16", bank.residual_scales_fp16.astype("<f2"),
        "residual2_codes_qint8", bank.residual2_codes_qint8.astype("|i1"),
        "residual2_scales_fp16", bank.residual2_scales_fp16.astype("<f2"),
        "class_indices_int16", bank.class_indices_int16.astype("<i2"),
        "class_scale_hi_fp16", bank.class_scale_hi_fp16.astype("<f2"),
        "class_scale_lo_fp16", bank.class_scale_lo_fp16.astype("<f2"),
    )
    return b"ADV3B02A4\0" + struct.pack("<I", len(header)) + header + b"".join(_affine_wire_array(arrays[i], arrays[i + 1]) for i in range(0, len(arrays), 2))


def _decode_affine_wire(wire: bytes, bank: AffineINT8ZIDSupportBank) -> np.ndarray:
    """Fail-closed little-endian decode used by every deployed qKNN score."""
    if type(wire) is not bytes or not wire.startswith(b"ADV3B02A4\0") or len(wire) < 14:
        raise ADV3B02StateError("affine wire magic/truncation drift")
    header_size = struct.unpack("<I", wire[10:14])[0]
    if 14 + header_size >= len(wire):
        raise ADV3B02StateError("affine wire header length drift")
    try:
        header = json.loads(wire[14:14 + header_size].decode("utf-8"))
    except Exception as exc:
        raise ADV3B02StateError("affine wire header decode failed") from exc
    if (header.get("schema") != AFFINE_WIRE_SCHEMA or header.get("endianness") != "little"
            or header.get("bank_receipt_sha256") != bank.bank_receipt_sha256
            or tuple(header.get("classes", ())) != bank.classes):
        raise ADV3B02StateError("affine wire header binding drift")
    cursor = 14 + header_size; expected = (
        ("codes_qint8", "|i1", bank.codes_qint8.shape),
        ("scales_fp16", "<f2", bank.scales_fp16.shape),
        ("offsets_fp16", "<f2", bank.offsets_fp16.shape),
        ("residual_codes_qint8", "|i1", bank.residual_codes_qint8.shape),
        ("residual_scales_fp16", "<f2", bank.residual_scales_fp16.shape),
        ("residual2_codes_qint8", "|i1", bank.residual2_codes_qint8.shape),
        ("residual2_scales_fp16", "<f2", bank.residual2_scales_fp16.shape),
        ("class_indices_int16", "<i2", bank.class_indices_int16.shape),
        ("class_scale_hi_fp16", "<f2", bank.class_scale_hi_fp16.shape),
        ("class_scale_lo_fp16", "<f2", bank.class_scale_lo_fp16.shape),
    )
    fields: dict[str, np.ndarray] = {}
    for name, dtype, shape in expected:
        if cursor + 2 > len(wire): raise ADV3B02StateError("affine wire field truncation")
        nname = struct.unpack("<H", wire[cursor:cursor + 2])[0]; cursor += 2
        if cursor + nname + 2 > len(wire) or wire[cursor:cursor + nname].decode("ascii") != name: raise ADV3B02StateError("affine wire field order drift")
        cursor += nname; ndtype = struct.unpack("<H", wire[cursor:cursor + 2])[0]; cursor += 2
        if cursor + ndtype + 2 > len(wire) or wire[cursor:cursor + ndtype].decode("ascii") != dtype: raise ADV3B02StateError("affine wire dtype/endianness drift")
        cursor += ndtype; ndim = struct.unpack("<H", wire[cursor:cursor + 2])[0]; cursor += 2
        if ndim != len(shape) or cursor + 8 * ndim + 8 > len(wire): raise ADV3B02StateError("affine wire shape truncation")
        seen = struct.unpack("<" + "Q" * ndim, wire[cursor:cursor + 8 * ndim]); cursor += 8 * ndim
        payload_size = struct.unpack("<Q", wire[cursor:cursor + 8])[0]; cursor += 8
        if tuple(seen) != tuple(shape) or payload_size != int(np.prod(shape)) * np.dtype(dtype).itemsize or cursor + payload_size > len(wire): raise ADV3B02StateError("affine wire array shape/size drift")
        fields[name] = np.frombuffer(wire[cursor:cursor + payload_size], dtype=np.dtype(dtype)).reshape(shape).copy(); cursor += payload_size
    if cursor != len(wire): raise ADV3B02StateError("affine wire trailing bytes drift")
    if (not np.array_equal(fields["codes_qint8"], bank.codes_qint8) or not np.array_equal(fields["scales_fp16"], bank.scales_fp16)
            or not np.array_equal(fields["offsets_fp16"], bank.offsets_fp16)
            or not np.array_equal(fields["residual_codes_qint8"], bank.residual_codes_qint8)
            or not np.array_equal(fields["residual_scales_fp16"], bank.residual_scales_fp16)
            or not np.array_equal(fields["residual2_codes_qint8"], bank.residual2_codes_qint8)
            or not np.array_equal(fields["residual2_scales_fp16"], bank.residual2_scales_fp16)
            or not np.array_equal(fields["class_indices_int16"], bank.class_indices_int16)
            or not np.array_equal(fields["class_scale_hi_fp16"], bank.class_scale_hi_fp16)
            or not np.array_equal(fields["class_scale_lo_fp16"], bank.class_scale_lo_fp16)):
        raise ADV3B02StateError("affine serialized bytes/bank mismatch")
    return _affine_dequantize_rows(
        fields["codes_qint8"],
        fields["scales_fp16"],
        fields["offsets_fp16"],
        fields["residual_codes_qint8"],
        fields["residual_scales_fp16"],
        fields["residual2_codes_qint8"],
        fields["residual2_scales_fp16"],
    )


@dataclass(frozen=True)
class ActualBankBranchState:
    qknn_wire: bytes
    support_physical_ids_canonical: tuple[str, ...]
    bcr_weight_codes_qint8: np.ndarray
    bcr_weight_scales_fp16: np.ndarray
    bcr_weight_residual_codes_qint8: np.ndarray
    bcr_weight_residual_scales_fp16: np.ndarray
    bcr_weight_residual2_codes_qint8: np.ndarray
    bcr_weight_residual2_scales_fp16: np.ndarray
    bcr_lambda: float
    quantization_audit: Mapping[str, Any]
    actual_bank_binding_receipt: Mapping[str, Any]
    support_repair_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        audit = dict(self.quantization_audit)
        if (type(self.qknn_wire) is not bytes or self.bcr_weight_codes_qint8.dtype != np.int8
                or self.bcr_weight_codes_qint8.ndim != 2 or self.bcr_weight_codes_qint8.shape[0] != Z_DIM
                or self.bcr_weight_codes_qint8.shape[1] < 2 or np.any(self.bcr_weight_codes_qint8 == np.int8(-128))
                or self.bcr_weight_scales_fp16.dtype != np.dtype("<f2")
                or self.bcr_weight_scales_fp16.shape != (self.bcr_weight_codes_qint8.shape[1],)
                or not np.isfinite(self.bcr_weight_scales_fp16).all() or np.any(self.bcr_weight_scales_fp16 <= 0.0)
                or self.bcr_weight_residual_codes_qint8.dtype != np.int8
                or self.bcr_weight_residual_codes_qint8.shape != self.bcr_weight_codes_qint8.shape
                or np.any(self.bcr_weight_residual_codes_qint8 == np.int8(-128))
                or self.bcr_weight_residual_scales_fp16.dtype != np.dtype("<f2")
                or self.bcr_weight_residual_scales_fp16.shape != self.bcr_weight_scales_fp16.shape
                or not np.isfinite(self.bcr_weight_residual_scales_fp16).all()
                or np.any(self.bcr_weight_residual_scales_fp16 <= 0.0)
                or self.bcr_weight_residual2_codes_qint8.dtype != np.int8
                or self.bcr_weight_residual2_codes_qint8.shape != self.bcr_weight_codes_qint8.shape
                or np.any(self.bcr_weight_residual2_codes_qint8 == np.int8(-128))
                or self.bcr_weight_residual2_scales_fp16.dtype != np.dtype("<f2")
                or self.bcr_weight_residual2_scales_fp16.shape != self.bcr_weight_scales_fp16.shape
                or not np.isfinite(self.bcr_weight_residual2_scales_fp16).all()
                or np.any(self.bcr_weight_residual2_scales_fp16 <= 0.0)
                or set(audit) != {"qknn", "bcr"} or float(audit["qknn"].get("top1_agreement", -1.0)) < .995
                or int(audit["qknn"].get("large_margin_flip_count", -1)) != 0
                or float(audit["bcr"].get("top1_agreement", -1.0)) != 1.0
                or int(audit["bcr"].get("any_margin_flip_count", -1)) != 0
                or int(audit["bcr"].get("large_margin_flip_count", -1)) != 0):
            raise ADV3B02StateError("affine actual branch audit/state drift")
        receipt = dict(self.actual_bank_binding_receipt)
        repair = verify_zid_repair_receipt(self.support_repair_receipt)
        required = {"schema", "qknn_wire_sha256", "bank_receipt_sha256", "metric_receipt_sha256",
                    "support_token_root_sha256", "teacher_support_sha256", "bcr_weight_codes_sha256",
                    "support_repair_receipt_sha256",
                    "bcr_weight_scales_sha256", "bcr_weight_residual_codes_sha256",
                    "bcr_weight_residual_scales_sha256", "bcr_weight_residual2_codes_sha256",
                    "bcr_weight_residual2_scales_sha256", "bcr_weight_codec", "bcr_weight_plane_count",
                    "bcr_weight_plane_order", "bcr_weight_code_dtype", "bcr_weight_scale_dtype",
                    "bcr_weight_rounding", "bcr_weight_clip", "bcr_weight_scale_floor",
                    "bcr_weight_shape", "bcr_weight_class_order", "bcr_weight_wire_bytes",
                    "bcr_analytic_loo_sha256", "directional_loo_sha256",
                    "qknn_audit_sha256", "bcr_audit_sha256", "query_rows_used_for_fit", "receipt_sha256"}
        body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        if (set(receipt) != required or receipt.get("schema") != ACTUAL_BRANCH_SCHEMA
                or receipt.get("receipt_sha256") != sha256_bytes(_canon(body))
                or receipt.get("qknn_wire_sha256") != sha256_bytes(self.qknn_wire)
                or receipt.get("teacher_support_sha256") != repair["unit_output_support_sha256"]
                or receipt.get("support_repair_receipt_sha256") != repair["receipt_sha256"]
                or receipt.get("support_token_root_sha256") != sha256_bytes(_canon(list(self.support_physical_ids_canonical)))
                or receipt.get("bcr_weight_codes_sha256") != sha256_bytes(np.ascontiguousarray(self.bcr_weight_codes_qint8).tobytes())
                or receipt.get("bcr_weight_scales_sha256") != sha256_bytes(np.ascontiguousarray(self.bcr_weight_scales_fp16).tobytes())
                or receipt.get("bcr_weight_residual_codes_sha256") != sha256_bytes(np.ascontiguousarray(self.bcr_weight_residual_codes_qint8).tobytes())
                or receipt.get("bcr_weight_residual_scales_sha256") != sha256_bytes(np.ascontiguousarray(self.bcr_weight_residual_scales_fp16).tobytes())
                or receipt.get("bcr_weight_residual2_codes_sha256") != sha256_bytes(np.ascontiguousarray(self.bcr_weight_residual2_codes_qint8).tobytes())
                or receipt.get("bcr_weight_residual2_scales_sha256") != sha256_bytes(np.ascontiguousarray(self.bcr_weight_residual2_scales_fp16).tobytes())
                or receipt.get("bcr_weight_codec") != BCR_WEIGHT_CODEC
                or receipt.get("bcr_weight_plane_count") != 3
                or receipt.get("bcr_weight_plane_order") != list(BCR_WEIGHT_PLANE_ORDER)
                or receipt.get("bcr_weight_code_dtype") != BCR_WEIGHT_CODE_DTYPE
                or receipt.get("bcr_weight_scale_dtype") != BCR_WEIGHT_SCALE_DTYPE
                or receipt.get("bcr_weight_rounding") != BCR_WEIGHT_ROUNDING
                or receipt.get("bcr_weight_clip") != list(BCR_WEIGHT_CLIP)
                or receipt.get("bcr_weight_scale_floor") != BCR_WEIGHT_SCALE_FLOOR
                or receipt.get("bcr_weight_shape") != list(self.bcr_weight_codes_qint8.shape)
                or not isinstance(receipt.get("bcr_weight_class_order"), list)
                or len(receipt.get("bcr_weight_class_order", ())) != self.bcr_weight_codes_qint8.shape[1]
                or receipt.get("bcr_weight_wire_bytes") != (
                    self.bcr_weight_codes_qint8.nbytes
                    + self.bcr_weight_scales_fp16.nbytes
                    + self.bcr_weight_residual_codes_qint8.nbytes
                    + self.bcr_weight_residual_scales_fp16.nbytes
                    + self.bcr_weight_residual2_codes_qint8.nbytes
                    + self.bcr_weight_residual2_scales_fp16.nbytes
                )
                or receipt.get("qknn_audit_sha256") != sha256_bytes(_canon(audit["qknn"]))
                or receipt.get("bcr_audit_sha256") != sha256_bytes(_canon(audit["bcr"]))
                or receipt.get("query_rows_used_for_fit") != 0):
            raise ADV3B02StateError("affine actual-branch receipt drift")


@dataclass(frozen=True)
class Int8QKNNState:
    branch_state: ActualBankBranchState
    bank: AffineINT8ZIDSupportBank
    metric: TypedSharedPSDMetric
    qknn_wire: bytes
    registered_classes: tuple[str, ...]
    labels: tuple[str, ...]
    support_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        typed_tokens(self.labels, name="qKNN labels")
        typed_tokens(self.support_tokens, name="qKNN support tokens", unique=True)
        if (self.metric.effective_rank != 0 or self.metric.config_lock_digest != self.bank.config_lock_digest
                or len(self.labels) != self.bank.support_row_count or len(self.support_tokens) != self.bank.support_row_count
                or set(self.bank.classes) != set(self.classes) or self.branch_state.qknn_wire != self.qknn_wire
                or tuple(self.branch_state.support_physical_ids_canonical) != self.support_tokens
                or _serialize_affine_bank(self.bank) != self.qknn_wire):
            raise ADV3B02StateError("affine qKNN wire/state closure drift")
        _decode_affine_wire(self.qknn_wire, self.bank)
        audit = dict(self.branch_state.quantization_audit.get("qknn", {}))
        binding = self.branch_state.actual_bank_binding_receipt
        if (binding.get("qknn_wire_sha256") != sha256_bytes(self.qknn_wire)
                or binding.get("bank_receipt_sha256") != self.bank.bank_receipt_sha256
                or binding.get("metric_receipt_sha256") != self.metric.metric_receipt_sha256
                or binding.get("support_token_root_sha256") != sha256_bytes(_canon(list(self.support_tokens)))
                or binding.get("bcr_weight_codes_sha256") != sha256_bytes(np.ascontiguousarray(self.branch_state.bcr_weight_codes_qint8).tobytes())
                or binding.get("bcr_weight_scales_sha256") != sha256_bytes(np.ascontiguousarray(self.branch_state.bcr_weight_scales_fp16).tobytes())
                or binding.get("bcr_weight_residual_codes_sha256") != sha256_bytes(np.ascontiguousarray(self.branch_state.bcr_weight_residual_codes_qint8).tobytes())
                or binding.get("bcr_weight_residual_scales_sha256") != sha256_bytes(np.ascontiguousarray(self.branch_state.bcr_weight_residual_scales_fp16).tobytes())
                or binding.get("bcr_weight_residual2_codes_sha256") != sha256_bytes(np.ascontiguousarray(self.branch_state.bcr_weight_residual2_codes_qint8).tobytes())
                or binding.get("bcr_weight_residual2_scales_sha256") != sha256_bytes(np.ascontiguousarray(self.branch_state.bcr_weight_residual2_scales_fp16).tobytes())
                or binding.get("bcr_weight_codec") != BCR_WEIGHT_CODEC
                or binding.get("bcr_weight_plane_count") != 3
                or binding.get("bcr_weight_plane_order") != list(BCR_WEIGHT_PLANE_ORDER)
                or binding.get("bcr_weight_code_dtype") != BCR_WEIGHT_CODE_DTYPE
                or binding.get("bcr_weight_scale_dtype") != BCR_WEIGHT_SCALE_DTYPE
                or binding.get("bcr_weight_rounding") != BCR_WEIGHT_ROUNDING
                or binding.get("bcr_weight_clip") != list(BCR_WEIGHT_CLIP)
                or binding.get("bcr_weight_scale_floor") != BCR_WEIGHT_SCALE_FLOOR
                or binding.get("bcr_weight_shape") != list(self.branch_state.bcr_weight_codes_qint8.shape)
                or binding.get("bcr_weight_class_order") != list(self.bank.classes)
                or binding.get("bcr_weight_wire_bytes") != (
                    self.branch_state.bcr_weight_codes_qint8.nbytes
                    + self.branch_state.bcr_weight_scales_fp16.nbytes
                    + self.branch_state.bcr_weight_residual_codes_qint8.nbytes
                    + self.branch_state.bcr_weight_residual_scales_fp16.nbytes
                    + self.branch_state.bcr_weight_residual2_codes_qint8.nbytes
                    + self.branch_state.bcr_weight_residual2_scales_fp16.nbytes
                )
                or binding.get("qknn_audit_sha256") != sha256_bytes(_canon(dict(self.branch_state.quantization_audit["qknn"])))
                or binding.get("bcr_audit_sha256") != sha256_bytes(_canon(dict(self.branch_state.quantization_audit["bcr"])) )):
            raise ADV3B02StateError("affine actual-branch binding field drift")
        if float(audit.get("top1_agreement", -1.0)) < .995 or int(audit.get("large_margin_flip_count", -1)) != 0:
            raise ADV3B02StateError("affine qKNN large-margin audit drift")

    @property
    def classes(self) -> tuple[str, ...]: return self.registered_classes
    @property
    def k_shot(self) -> int: return self.bank.active_k
    def features(self) -> np.ndarray: return _decode_affine_wire(self.qknn_wire, self.bank)
    def wire_bytes(self) -> bytes: return self.qknn_wire
    @property
    def digest(self) -> str: return sha256_bytes(self.qknn_wire)


def _score_support(*, support: np.ndarray, indices: np.ndarray, counts: tuple[int, ...],
                   class_scales: np.ndarray, query: np.ndarray, config: Phase1ZIDStudentTLock) -> np.ndarray:
    q = _unit(_rows(query, name="qKNN query")).astype(np.float64)
    s = _unit(np.asarray(support, np.float32)).astype(np.float64)
    cosine = np.clip(q @ s.T, -1.0, 1.0)
    distance = np.maximum(2.0 * (1.0 - cosine), 0.0)
    columns = []
    for ci, count in enumerate(counts):
        local = distance[:, indices == ci]
        h = float(class_scales[ci])
        kernel = (-config.kernel_volume_gamma * config.kernel_effective_dim * math.log(h)
                  -0.5 * (config.student_nu + config.kernel_effective_dim)
                  * np.log1p(local / (config.student_nu * h * h)))
        maximum = np.max(kernel, axis=1, keepdims=True)
        columns.append(maximum[:, 0] + np.log(np.sum(np.exp(kernel - maximum), axis=1)) - math.log(count))
    result = np.asarray(np.stack(columns, axis=1), np.float32)
    if not np.isfinite(result).all():
        raise ADV3B02StateError("affine Student-t logits nonfinite")
    return result


def _score_affine_bank(bank: AffineINT8ZIDSupportBank, query: np.ndarray) -> np.ndarray:
    return _score_support(support=_affine_dequantize_rows(
                              bank.codes_qint8, bank.scales_fp16, bank.offsets_fp16,
                              bank.residual_codes_qint8, bank.residual_scales_fp16,
                              bank.residual2_codes_qint8, bank.residual2_scales_fp16),
                          indices=bank.class_indices_int16, counts=bank.support_counts,
                          class_scales=bank.deployed_class_scales(), query=query, config=bank.config)


def _affine_margin_audit(bank: AffineINT8ZIDSupportBank, teacher_support: np.ndarray,
                         support_labels: tuple[str, ...], validation: np.ndarray, *,
                         teacher_class_scales: np.ndarray | None = None,
                         teacher_bandwidth_source: str = "complete_unquantized_FP32_all_support") -> dict[str, Any]:
    teacher = _unit(_rows(teacher_support, name="full FP32 affine teacher")).astype(np.float64)
    labels = typed_tokens(support_labels, name="affine teacher labels")
    if len(labels) != len(teacher):
        raise ADV3B02StateError("affine teacher label closure drift")
    indices = np.asarray([bank.classes.index(v) for v in labels], dtype=np.int16)
    counts = tuple(int(np.sum(indices == i)) for i in range(len(bank.classes)))
    if counts != bank.support_counts:
        raise ADV3B02StateError("affine teacher class-count drift")
    if teacher_class_scales is None:
        tscale = _existing_identity_class_scales(
            teacher, indices, len(bank.classes), bank.config
        )
    else:
        tscale = np.asarray(teacher_class_scales, np.float64)
        if (
            tscale.ndim != 1
            or tscale.shape != (len(bank.classes),)
            or not np.isfinite(tscale).all()
            or np.any(tscale <= 0.0)
        ):
            raise ADV3B02StateError("affine teacher class bandwidth drift")
    fp = _score_support(support=teacher, indices=indices, counts=counts, class_scales=tscale,
                        query=validation, config=bank.config).astype(np.float64)
    deployed = _score_affine_bank(bank, validation).astype(np.float64)
    row = np.arange(len(fp))
    winner = np.argmax(fp, axis=1)
    runner_scores = fp.copy()
    runner_scores[row, winner] = -np.inf
    runner = np.argmax(runner_scores, axis=1)
    margin = fp[row, winner] - fp[row, runner]
    deployment_margin = deployed[row, winner] - deployed[row, runner]
    maxerr = np.max(np.abs(fp - deployed), axis=1)
    # Full argmax comparison catches a third class overtaking the teacher,
    # not merely the teacher winner/runner-up pair reversing.
    any_flip = np.argmax(fp, axis=1) != np.argmax(deployed, axis=1)
    large_flip = any_flip & (margin > 2.0 * maxerr)
    bank_scales = bank.deployed_class_scales()
    return {"schema": "cvs.phase2.zid_student_t_qknn.margin_audit.v4_q3support1", "validation_row_count": int(len(fp)),
            "logit_abs_error_mean": float(np.mean(np.abs(fp - deployed))), "logit_abs_error_max": float(np.max(np.abs(fp - deployed))),
            "top1_agreement": float(np.mean(np.argmax(fp, axis=1) == np.argmax(deployed, axis=1))),
            "teacher_margin_mean": float(np.mean(margin)), "quantized_teacher_margin_mean": float(np.mean(deployment_margin)),
            "any_margin_flip_count": int(np.sum(any_flip)), "any_margin_flip_rate": float(np.mean(any_flip)),
            "large_margin_flip_count": int(np.sum(large_flip)), "large_margin_flip_rate": float(np.mean(large_flip)),
            "fp32_teacher_bandwidth_source": teacher_bandwidth_source, "fp32_teacher_support_sha256": sha256_bytes(np.ascontiguousarray(teacher.astype(np.float32)).tobytes()),
            "int8_bank_class_scales_sha256": sha256_bytes(
                np.ascontiguousarray(bank_scales.astype("<f4")).tobytes()
            ),
            "int8_bank_class_scale_count": int(len(bank_scales)),
            "teacher_bank_bandwidth_abs_delta_max": float(np.max(np.abs(tscale - bank_scales.astype(np.float64)))),
            "query_rows_used_for_fit": 0, "state_updates": 0}


def _make_actual_branch(bank: AffineINT8ZIDSupportBank, metric: TypedSharedPSDMetric, wire: bytes,
                        raw_teacher_zid: np.ndarray, labels: tuple[str, ...], tokens: tuple[str, ...], audit: Mapping[str, Any],
                        support_repair_receipt: Mapping[str, Any]) -> ActualBankBranchState:
    raw_teacher = _rows(raw_teacher_zid, name="actual branch raw teacher z_id")
    if len(raw_teacher) != len(labels) or len(labels) != len(tokens):
        raise ADV3B02StateError("actual branch raw teacher layout drift")
    repair = _validate_repaired_support_for_state(
        support_repair_receipt, support_zid=raw_teacher,
        labels=labels, classes=tuple(sorted(bank.classes)), tokens=tokens,
        require_output_support_sha256=False,
    )
    decoded = _affine_dequantize_rows(
        bank.codes_qint8,
        bank.scales_fp16,
        bank.offsets_fp16,
        bank.residual_codes_qint8,
        bank.residual_scales_fp16,
        bank.residual2_codes_qint8,
        bank.residual2_scales_fp16,
    ).astype(np.float64)
    weights, analytic_loo, lam, _ = _existing_bcr_ridge_fit_and_loo(decoded, bank.class_indices_int16, bank.active_k)
    wc, ws, rwc, rws, r2wc, r2ws, dw = _quantize_bcr_weights_three_plane(weights)
    bcr_teacher_logits = _unit(raw_teacher).astype(np.float64) @ weights
    bcr_deployed_logits = (
        _unit(raw_teacher).astype(np.float64) @ dw.astype(np.float64)
    )
    bcr_audit = dict(
        _existing_bcr_quant_audit(bcr_teacher_logits, bcr_deployed_logits)
    )
    bcr_any_flip = np.argmax(bcr_teacher_logits, axis=1) != np.argmax(
        bcr_deployed_logits, axis=1
    )
    bcr_audit.update({
        "codec": BCR_WEIGHT_CODEC,
        "plane_count": 3,
        "plane_order": list(BCR_WEIGHT_PLANE_ORDER),
        "any_margin_flip_count": int(np.sum(bcr_any_flip)),
        "any_margin_flip_rate": float(np.mean(bcr_any_flip)),
        "weight_max_abs_error": float(np.max(np.abs(weights - dw.astype(np.float64)))),
        "weight_wire_bytes": int(wc.nbytes + ws.nbytes + rwc.nbytes + rws.nbytes + r2wc.nbytes + r2ws.nbytes),
    })
    if (float(bcr_audit.get("top1_agreement", 0.0)) != 1.0
            or int(bcr_audit.get("any_margin_flip_count", -1)) != 0
            or int(bcr_audit.get("large_margin_flip_count", -1)) != 0):
        raise ADV3B02StateError("affine BCR INT8 audit gate failed")
    if bank.active_k == 1:
        shape = (bank.support_row_count, len(bank.classes)); qscore = {v: np.zeros(shape) for v in _BCR_DIRECTIONS}; bscore = {v: np.zeros(shape) for v in _BCR_DIRECTIONS}
    else:
        qscore, bscore = _existing_bcr_cross_view_loo(decoded, bank.class_indices_int16.astype(np.intp), bank.classes, bank.config)
    directional = {v: {"qknn_sha256": sha256_bytes(np.ascontiguousarray(qscore[v]).tobytes()), "bcr_sha256": sha256_bytes(np.ascontiguousarray(bscore[v]).tobytes())} for v in _BCR_DIRECTIONS}
    body = {"schema": ACTUAL_BRANCH_SCHEMA, "qknn_wire_sha256": sha256_bytes(wire), "bank_receipt_sha256": bank.bank_receipt_sha256,
            "metric_receipt_sha256": metric.metric_receipt_sha256, "support_token_root_sha256": sha256_bytes(_canon(list(tokens))),
            "teacher_support_sha256": _unit_support_token_binding_sha256(raw_teacher, tokens),
            "support_repair_receipt_sha256": repair["receipt_sha256"], "bcr_weight_codes_sha256": sha256_bytes(np.ascontiguousarray(wc).tobytes()),
            "bcr_weight_scales_sha256": sha256_bytes(np.ascontiguousarray(ws).tobytes()),
            "bcr_weight_residual_codes_sha256": sha256_bytes(np.ascontiguousarray(rwc).tobytes()),
            "bcr_weight_residual_scales_sha256": sha256_bytes(np.ascontiguousarray(rws).tobytes()),
            "bcr_weight_residual2_codes_sha256": sha256_bytes(np.ascontiguousarray(r2wc).tobytes()),
            "bcr_weight_residual2_scales_sha256": sha256_bytes(np.ascontiguousarray(r2ws).tobytes()),
            "bcr_weight_codec": BCR_WEIGHT_CODEC, "bcr_weight_plane_count": 3,
            "bcr_weight_plane_order": list(BCR_WEIGHT_PLANE_ORDER),
            "bcr_weight_code_dtype": BCR_WEIGHT_CODE_DTYPE,
            "bcr_weight_scale_dtype": BCR_WEIGHT_SCALE_DTYPE,
            "bcr_weight_rounding": BCR_WEIGHT_ROUNDING,
            "bcr_weight_clip": list(BCR_WEIGHT_CLIP),
            "bcr_weight_scale_floor": BCR_WEIGHT_SCALE_FLOOR,
            "bcr_weight_shape": list(wc.shape), "bcr_weight_class_order": list(bank.classes),
            "bcr_weight_wire_bytes": int(wc.nbytes + ws.nbytes + rwc.nbytes + rws.nbytes + r2wc.nbytes + r2ws.nbytes),
            "bcr_analytic_loo_sha256": sha256_bytes(np.ascontiguousarray(analytic_loo).tobytes()),
            "directional_loo_sha256": directional, "qknn_audit_sha256": sha256_bytes(_canon(dict(audit))), "bcr_audit_sha256": sha256_bytes(_canon(bcr_audit)), "query_rows_used_for_fit": 0}
    return ActualBankBranchState(
        wire, tokens, wc, ws, rwc, rws, r2wc, r2ws, float(lam),
        {"qknn": dict(audit), "bcr": bcr_audit},
        {**body, "receipt_sha256": sha256_bytes(_canon(body))}, dict(repair),
    )


def phase1_qknn_lock(k_shot: int) -> Phase1ZIDStudentTLock:
    """Use the pre-existing sealed Phase1 lock, never candidate-local qKNN knobs."""
    if type(k_shot) is not int or k_shot not in K_VALUES:
        raise ADV3B02StateError("formal qKNN only permits K={1,5,10}")
    return qknn_lock_from_method_lock(canonical_method_lock(), k_shot=k_shot)


def build_int8_qknn_state(features: np.ndarray, labels: Sequence[Any] | np.ndarray,
                          registered_classes: Sequence[Any] | np.ndarray,
                          physical_tokens: Sequence[Any] | np.ndarray, *,
                          qknn_lock: Phase1ZIDStudentTLock | None = None,
                          support_repair_receipt: Mapping[str, Any] | None = None) -> Int8QKNNState:
    """Build the fixed affine deployment bank from support only."""
    source = _rows(features, name="support z_id")
    classes = typed_tokens(registered_classes, name="registered classes", unique=True)
    labs = typed_tokens(labels, name="support labels")
    tokens = typed_tokens(physical_tokens, name="support physical tokens", unique=True)
    if len(source) != len(labs) or len(labs) != len(tokens) or any(item not in classes for item in labs):
        raise ADV3B02StateError("typed qKNN support labels/features/tokens drift")
    k_shot = _balanced_k(labs, classes)
    if support_repair_receipt is None:
        source, repair = repair_finite_exact_zero_singleton_class_medoid(
            source, labs, classes, tokens
        )
    else:
        repair = _validate_repaired_support_for_state(
            support_repair_receipt, support_zid=source, labels=labs,
            classes=classes, tokens=tokens,
        )
    lock = phase1_qknn_lock(k_shot) if qknn_lock is None else qknn_lock
    if type(lock) is not Phase1ZIDStudentTLock or lock.active_k != k_shot:
        raise ADV3B02StateError("typed qKNN Phase1 lock/K drift")
    canonical_classes = tuple(sorted(classes))
    positions = sorted(range(len(tokens)), key=lambda i: (labs[i], tokens[i]))
    ordered_raw = np.ascontiguousarray(source[np.asarray(positions, np.intp)])
    ordered_unit = _unit(ordered_raw)
    ordered_labels = tuple(labs[i] for i in positions)
    ordered_tokens = tuple(tokens[i] for i in positions)
    indices = np.asarray([canonical_classes.index(v) for v in ordered_labels], dtype="<i2")
    codes, scales, offsets, residual_codes, residual_scales, residual2_codes, residual2_scales = (
        _affine_quantize_rows_two_plane(ordered_raw)
    )
    raw_class_scales = np.asarray(
        _existing_identity_class_scales(
            ordered_unit, indices, len(canonical_classes), lock
        ),
        np.float32,
    )
    class_scale_hi, class_scale_lo = _split_class_bandwidths(raw_class_scales)
    counts = tuple(int(np.sum(indices == i)) for i in range(len(canonical_classes)))
    quantization = {"schema": AFFINE_BANK_SCHEMA, "codec": SUPPORT_CODEC,
                    "plane_order": list(SUPPORT_PLANE_ORDER),
                    "rounding": SUPPORT_ROUNDING, "clip": list(SUPPORT_CLIP),
                    "residual_scale_floor": SUPPORT_RESIDUAL_SCALE_FLOOR,
                    "class_bandwidth_codec": CLASS_BANDWIDTH_CODEC,
                    "support_only": True, "query_rows_used_for_fit": 0,
                    "codes_sha256": sha256_bytes(codes.tobytes()), "scales_sha256": sha256_bytes(scales.tobytes()),
                    "offsets_sha256": sha256_bytes(offsets.tobytes()),
                    "residual_codes_sha256": sha256_bytes(residual_codes.tobytes()),
                    "residual_scales_sha256": sha256_bytes(residual_scales.tobytes()),
                    "residual2_codes_sha256": sha256_bytes(residual2_codes.tobytes()),
                    "residual2_scales_sha256": sha256_bytes(residual2_scales.tobytes()),
                    "class_scale_hi_sha256": sha256_bytes(class_scale_hi.tobytes()),
                    "class_scale_lo_sha256": sha256_bytes(class_scale_lo.tobytes()),
                    "endianness": "little"}
    seed_bank = AffineINT8ZIDSupportBank(
        canonical_classes, counts, codes, scales, offsets, residual_codes,
        residual_scales, residual2_codes, residual2_scales, indices, class_scale_hi, class_scale_lo, k_shot, lock,
        lock.lock_digest, quantization, "0" * 64,
    )
    receipt = sha256_bytes(_canon(_affine_bank_payload(seed_bank)))
    bank = AffineINT8ZIDSupportBank(
        canonical_classes, counts, codes, scales, offsets, residual_codes,
        residual_scales, residual2_codes, residual2_scales, indices, class_scale_hi, class_scale_lo, k_shot, lock,
        lock.lock_digest, quantization, receipt,
    )
    metric = identity_shared_psd_metric(config=lock)
    wire = _serialize_affine_bank(bank)
    audit = _affine_margin_audit(
        bank, ordered_raw, ordered_labels, ordered_unit
    )
    branch = _make_actual_branch(
        bank, metric, wire, ordered_raw, ordered_labels, ordered_tokens, audit, repair
    )
    return Int8QKNNState(branch, bank, metric, wire, classes, ordered_labels, ordered_tokens)


def qknn_logits(state: Int8QKNNState, query_zid: np.ndarray) -> np.ndarray:
    canonical = _score_support(support=state.features(), indices=state.bank.class_indices_int16,
                               counts=state.bank.support_counts, class_scales=state.bank.deployed_class_scales(),
                               query=_rows(query_zid, name="query z_id"), config=state.bank.config)
    return canonical[:, np.asarray([state.bank.classes.index(item) for item in state.classes], np.intp)]


def _query_zid_exact_zero_mask(query_zid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate qzero1's input partition without normalizing exact-zero rows."""
    zid = _rows(query_zid, name="query z_id")
    zero_mask = np.all(zid == np.float32(0.0), axis=1)
    norms = np.linalg.norm(zid.astype(np.float64), axis=1)
    if not np.isfinite(norms).all() or np.any((~zero_mask) & (norms <= 1.0e-12)):
        raise ADV3B02StateError("query z_id rejects tiny nonzero or nonfinite L2 norm")
    return zid, zero_mask


def _zero_query_analytic_logits(state: Int8QKNNState) -> np.ndarray:
    """Student-t qKNN's frozen no-direction score at cos=0 and distance=2."""
    h = state.bank.deployed_class_scales().astype(np.float64)
    config = state.bank.config
    if h.shape != (len(state.bank.classes),) or not np.isfinite(h).all() or np.any(h <= 0.0):
        raise ADV3B02StateError("zero-query class bandwidth state drift")
    canonical = np.asarray(
        -config.kernel_volume_gamma * config.kernel_effective_dim * np.log(h)
        -0.5 * (config.student_nu + config.kernel_effective_dim)
        * np.log1p(2.0 / (config.student_nu * h * h)),
        np.float32,
    )
    if not np.isfinite(canonical).all():
        raise ADV3B02StateError("zero-query Student-t logits nonfinite")
    return np.ascontiguousarray(
        canonical[
            np.asarray(
                [state.bank.classes.index(item) for item in state.classes],
                np.intp,
            )
        ]
    )


def _zero_class_tie_key(state: Int8QKNNState, class_handle: str) -> tuple[bytes, tuple[bytes, ...]]:
    """Canonical label-free qzero hard-tie key for one deployed class payload."""
    canonical_index = state.bank.classes.index(class_handle)
    positions = np.flatnonzero(
        state.bank.class_indices_int16 == canonical_index
    ).astype(np.intp)
    if len(positions) != state.bank.active_k:
        raise ADV3B02StateError("zero-query class payload/cardinality drift")
    positions = np.asarray(
        sorted(positions.tolist(), key=lambda index: state.support_tokens[index].encode("utf-8")),
        np.intp,
    )
    bank = state.bank
    payload = b"".join(
        (
            np.ascontiguousarray(bank.codes_qint8[positions], dtype=np.int8).tobytes(),
            np.ascontiguousarray(bank.scales_fp16[positions], dtype="<f2").tobytes(),
            np.ascontiguousarray(bank.offsets_fp16[positions], dtype="<f2").tobytes(),
            np.ascontiguousarray(bank.residual_codes_qint8[positions], dtype=np.int8).tobytes(),
            np.ascontiguousarray(bank.residual_scales_fp16[positions], dtype="<f2").tobytes(),
            np.ascontiguousarray(bank.residual2_codes_qint8[positions], dtype=np.int8).tobytes(),
            np.ascontiguousarray(bank.residual2_scales_fp16[positions], dtype="<f2").tobytes(),
            np.ascontiguousarray(bank.class_scale_hi_fp16[canonical_index:canonical_index + 1], dtype="<f2").tobytes(),
            np.ascontiguousarray(bank.class_scale_lo_fp16[canonical_index:canonical_index + 1], dtype="<f2").tobytes(),
        )
    )
    tokens = tuple(state.support_tokens[index].encode("utf-8") for index in positions)
    return payload, tokens


def _zero_row_argmax(state: Int8QKNNState, scores: np.ndarray) -> tuple[int, bool]:
    """Resolve only an exact qzero hard tie without labels, axis, or query data."""
    row = np.asarray(scores, np.float32)
    if row.ndim != 1 or row.shape != (len(state.classes),) or not np.isfinite(row).all():
        raise ADV3B02StateError("zero-query tie score layout drift")
    maximum = np.max(row)
    ties = np.flatnonzero(row == maximum).astype(np.intp)
    if not len(ties):
        raise ADV3B02StateError("zero-query hard tie maximum drift")
    if len(ties) == 1:
        return int(ties[0]), False
    keyed = [(_zero_class_tie_key(state, state.classes[index]), int(index)) for index in ties]
    keys = [item[0] for item in keyed]
    if len(set(keys)) != len(keys):
        raise ADV3B02StateError("zero-query hard tie class payload collision")
    return min(keyed, key=lambda item: item[0])[1], True


def int8_audit(state: Int8QKNNState, _original_support_zid: np.ndarray | None = None) -> dict[str, Any]:
    """Return the sealed full-FP32 teacher audit; no query is fitted or opened."""
    audit = dict(state.branch_state.quantization_audit["qknn"])
    if audit.get("top1_agreement", 0.0) < 0.995 or audit.get("large_margin_flip_count") != 0:
        raise ADV3B02StateError("affine qKNN teacher gate failed")
    return audit


@dataclass(frozen=True)
class DomainState:
    """Fixed two-slot target-old nuisance state plus append-only class centres."""

    classes: tuple[str, ...]
    zdom_codes: np.ndarray
    zdom_scales: np.ndarray
    centres: np.ndarray
    q: np.ndarray
    a: np.ndarray
    rho: np.ndarray
    alpha: float
    k_shot: int
    old_class_count: int
    stage: str
    frozen_old_digest: str | None = None

    def __post_init__(self) -> None:
        typed_tokens(self.classes, name="domain class registry", unique=True)
        _dequantize_rows(self.zdom_codes, self.zdom_scales)
        if (self.centres.dtype != np.float32 or self.centres.shape != (len(self.classes), Z_DIM)
                or not np.isfinite(self.centres).all()):
            raise ADV3B02StateError("domain centre state drift")
        if (self.q.dtype != np.float32 or self.q.shape != (Z_DIM, RANK)
                or self.a.dtype != np.float32 or self.a.shape != (RANK, RANK)
                or self.rho.dtype != np.float32 or self.rho.shape != (RANK,)):
            raise ADV3B02StateError("fixed two-slot Q/A/rho state drift")
        if not np.isfinite(self.q).all() or not np.isfinite(self.a).all() or not np.isfinite(self.rho).all():
            raise ADV3B02StateError("domain state must be finite")
        if np.any(self.rho < 0.0) or not np.isfinite(self.alpha) or not 0.0 <= self.alpha < 0.5:
            raise ADV3B02StateError("domain reliability/alpha range drift")
        if type(self.k_shot) is not int or self.k_shot not in K_VALUES or type(self.old_class_count) is not int:
            raise ADV3B02StateError("domain K/old-state cardinality drift")
        if not 0 < self.old_class_count <= len(self.classes) or self.stage not in ("S_B", "S_C"):
            raise ADV3B02StateError("domain lifecycle state drift")
        if self.k_shot == 1 and (self.alpha != 0.0 or np.any(self.q != 0.0) or np.any(self.a != 0.0) or np.any(self.rho != 0.0)):
            raise ADV3B02StateError("K1 domain state must be exact identity")

    def features(self) -> np.ndarray:
        return _unit(_dequantize_rows(self.zdom_codes, self.zdom_scales))

    def wire_bytes(self) -> bytes:
        header = _canon({"schema": SCHEMA, "kind": "fixed3_domain", "classes": list(self.classes),
                         "k": self.k_shot, "old_class_count": self.old_class_count,
                         "stage": self.stage, "alpha": float(self.alpha), "frozen_old_digest": self.frozen_old_digest})
        return b"".join((header, self.zdom_codes.tobytes(), self.zdom_scales.tobytes(), self.centres.tobytes(),
                         self.q.tobytes(), self.a.tobytes(), self.rho.tobytes()))

    @property
    def digest(self) -> str:
        return sha256_bytes(self.wire_bytes())


def _orient(column: np.ndarray) -> np.ndarray:
    index = int(np.argmax(np.abs(column)))
    return -column if column[index] < 0.0 else column


def _class_scatter_matrices(
    zdom: np.ndarray,
    labels: tuple[str, ...],
    classes: tuple[str, ...],
    *,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the frozen unbiased within-class and between-class scatters."""
    if k_shot <= 1:
        raise ADV3B02StateError("scatter matrices require K>1")
    centres = np.stack(
        [
            zdom[np.asarray([item == label for item in labels], bool)].mean(axis=0)
            for label in classes
        ]
    ).astype(np.float32)
    global_centre = centres.mean(axis=0)
    sw = np.zeros((Z_DIM, Z_DIM), np.float64)
    for ci, item in enumerate(classes):
        part = (
            zdom[np.asarray([label == item for label in labels], bool)].astype(
                np.float64
            )
            - centres[ci].astype(np.float64)
        )
        if len(part) != k_shot:
            raise ADV3B02StateError("scatter support is not exact balanced K")
        sw += part.T @ part / float(k_shot - 1)
    sw /= float(len(classes))
    centred = centres.astype(np.float64) - global_centre.astype(np.float64)
    sb = centred.T @ centred / float(len(classes))
    return sw, sb, centres


def _fit_fixed_two_slot(zdom: np.ndarray, labels: tuple[str, ...], classes: tuple[str, ...], *, k_shot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    if k_shot == 1:
        return (np.zeros((Z_DIM, RANK), np.float32), np.zeros((RANK, RANK), np.float32),
                np.zeros((RANK,), np.float32), 0.0, np.zeros((len(classes), Z_DIM), np.float32))
    sw, sb, centres = _class_scatter_matrices(
        zdom, labels, classes, k_shot=k_shot
    )
    trace = float(np.trace(sw))
    eps = max(1.0e-8, 1.0e-6 * trace / float(Z_DIM))
    g = 0.5 * ((sw - sb) + (sw - sb).T)
    try:
        eig, vec = np.linalg.eigh(g)
    except np.linalg.LinAlgError:
        return (np.zeros((Z_DIM, RANK), np.float32), np.zeros((RANK, RANK), np.float32),
                np.zeros((RANK,), np.float32), 0.0, centres)
    order = np.argsort(eig)[::-1][:RANK]
    q = np.zeros((Z_DIM, RANK), np.float64)
    rho = np.zeros((RANK,), np.float64)
    for slot, index in enumerate(order):
        direction = _orient(vec[:, int(index)])
        denominator = float(direction.T @ (sw + sb) @ direction + eps)
        numerator = float(direction.T @ (sw - sb) @ direction)
        if np.isfinite(numerator) and np.isfinite(denominator) and denominator > 0.0 and numerator > 0.0:
            q[:, slot] = direction
            rho[slot] = max(0.0, numerator / denominator)
    metric = q.T @ sw @ q + eps * np.eye(RANK)
    try:
        values, vectors = np.linalg.eigh(0.5 * (metric + metric.T))
        invsqrt = vectors @ np.diag(1.0 / np.sqrt(np.maximum(values, eps))) @ vectors.T
    except np.linalg.LinAlgError:
        invsqrt = np.zeros((RANK, RANK), np.float64)
        rho[:] = 0.0
    a = np.diag(np.sqrt(rho)) @ invsqrt
    alpha = 0.5 * (k_shot - 1) / k_shot * float(np.mean(rho))
    alpha = min(max(alpha, 0.0), float(np.nextafter(0.5, 0.0)))
    if not np.isfinite(a).all() or not np.isfinite(alpha):
        q[:] = 0.0; a[:] = 0.0; rho[:] = 0.0; alpha = 0.0
    return np.asarray(q, np.float32), np.asarray(a, np.float32), np.asarray(rho, np.float32), float(alpha), centres


@dataclass(frozen=True)
class DualQKNNState:
    """One raw z_id bank plus the candidate-class-conditioned domain state."""

    id_bank: Int8QKNNState
    domain: DomainState
    int8_audit_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.id_bank.classes != self.domain.classes or len(self.id_bank.labels) != len(self.domain.zdom_codes):
            raise ADV3B02StateError("dual qKNN raw/domain bank alignment drift")
        if self.id_bank.k_shot != self.domain.k_shot:
            raise ADV3B02StateError("dual qKNN K state drift")
        if self.int8_audit_receipt.get("query_rows_used_for_fit") != 0:
            raise ADV3B02StateError("INT8 receipt must be support-only")

    def wire_bytes(self) -> bytes:
        return self.id_bank.wire_bytes() + self.domain.wire_bytes()

    @property
    def digest(self) -> str:
        return sha256_bytes(self.wire_bytes())


def _reorder_to_bank(rows: np.ndarray, tokens: tuple[str, ...], bank: Int8QKNNState) -> np.ndarray:
    positions = {token: index for index, token in enumerate(tokens)}
    if set(positions) != set(bank.support_tokens):
        raise ADV3B02StateError("support token set differs from the z_id bank")
    return rows[np.asarray([positions[token] for token in bank.support_tokens], np.intp)]


def build_stage2_b_state(*, support_zid: np.ndarray, support_zdom: np.ndarray,
                         support_labels: Sequence[Any] | np.ndarray,
                         registered_classes: Sequence[Any] | np.ndarray,
                         support_physical_tokens: Sequence[Any] | np.ndarray,
                         support_repair_receipt: Mapping[str, Any] | None = None) -> DualQKNNState:
    """Fit the only legal Stage2-B state from target-old support."""
    labels = typed_tokens(support_labels, name="S_B support labels")
    classes = typed_tokens(registered_classes, name="S_B registered classes", unique=True)
    tokens = typed_tokens(support_physical_tokens, name="S_B support physical tokens", unique=True)
    zid = _rows(support_zid, name="S_B z_id")
    zdom = _unit(_rows(support_zdom, name="S_B z_dom"))
    if len(zid) != len(zdom) or len(zid) != len(labels) or len(labels) != len(tokens):
        raise ADV3B02StateError("S_B dual support layout drift")
    bank = build_int8_qknn_state(zid, labels, classes, tokens,
                                 support_repair_receipt=support_repair_receipt)
    ordered_dom = _reorder_to_bank(zdom, tokens, bank)
    q, a, rho, alpha, centres = _fit_fixed_two_slot(ordered_dom, bank.labels, bank.classes, k_shot=bank.k_shot)
    codes, scales = _quantize_rows(ordered_dom)
    domain = DomainState(bank.classes, codes, scales, centres, q, a, rho, alpha, bank.k_shot,
                         len(bank.classes), "S_B")
    ordered_zid = _reorder_to_bank(_unit(zid), tokens, bank)
    return DualQKNNState(bank, domain, int8_audit(bank, ordered_zid))


def _reference_new_suffix_codec(
    new_ordered_raw: np.ndarray,
    new_ordered_labels: tuple[str, ...],
    new_class_order: tuple[str, ...],
    config: Phase1ZIDStudentTLock,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent frozen-formula reference for Stage2-C's new suffix only."""
    raw = np.asarray(new_ordered_raw, np.float32)
    if (
        raw.ndim != 2
        or raw.shape[1] != Z_DIM
        or not len(raw)
        or not np.isfinite(raw).all()
        or set(new_ordered_labels) != set(new_class_order)
    ):
        raise ADV3B02StateError("Stage2-C new suffix reference input drift")
    norms = np.linalg.norm(raw.astype(np.float64), axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1.0e-12):
        raise ADV3B02StateError("Stage2-C new suffix reference norm drift")
    teacher = np.asarray(raw.astype(np.float64) / norms, np.float32)
    lo = np.min(teacher, axis=1).astype(np.float32)
    hi = np.max(teacher, axis=1).astype(np.float32)
    span = hi - lo
    if not np.isfinite(span).all() or np.any(span <= 0.0):
        raise ADV3B02StateError("Stage2-C new suffix reference affine range drift")
    scales = np.asarray(span / 254.0, dtype="<f2")
    offsets = np.asarray((hi + lo) / 2.0, dtype="<f2")
    if (
        not np.isfinite(scales).all()
        or np.any(scales <= 0.0)
        or not np.isfinite(offsets).all()
    ):
        raise ADV3B02StateError("Stage2-C new suffix reference affine closure drift")
    codes = np.clip(
        np.rint(
            (teacher - offsets.astype(np.float32)[:, None])
            / scales.astype(np.float32)[:, None]
        ),
        SUPPORT_CLIP[0],
        SUPPORT_CLIP[1],
    ).astype(np.int8)
    base = np.asarray(
        codes.astype(np.float32) * scales.astype(np.float32)[:, None]
        + offsets.astype(np.float32)[:, None],
        np.float32,
    )
    residual = np.asarray(teacher - base, np.float32)
    residual_scales = np.asarray(
        np.maximum(
            np.max(np.abs(residual), axis=1) / 127.0,
            SUPPORT_RESIDUAL_SCALE_FLOOR,
        ),
        dtype="<f2",
    )
    if (
        not np.isfinite(residual_scales).all()
        or np.any(residual_scales < np.float16(SUPPORT_RESIDUAL_SCALE_FLOOR))
    ):
        raise ADV3B02StateError("Stage2-C new suffix reference residual closure drift")
    residual_codes = np.clip(
        np.rint(residual / residual_scales.astype(np.float32)[:, None]),
        SUPPORT_CLIP[0],
        SUPPORT_CLIP[1],
    ).astype(np.int8)
    d2 = np.asarray(
        base + residual_codes.astype(np.float32)
        * residual_scales.astype(np.float32)[:, None],
        np.float32,
    )
    residual2 = np.asarray(teacher - d2, np.float32)
    residual2_scales = np.asarray(
        np.maximum(
            np.max(np.abs(residual2), axis=1) / 127.0,
            SUPPORT_RESIDUAL_SCALE_FLOOR,
        ),
        dtype="<f2",
    )
    if (
        not np.isfinite(residual2_scales).all()
        or np.any(residual2_scales < np.float16(SUPPORT_RESIDUAL_SCALE_FLOOR))
    ):
        raise ADV3B02StateError("Stage2-C new suffix reference Q3 closure drift")
    residual2_codes = np.clip(
        np.rint(residual2 / residual2_scales.astype(np.float32)[:, None]),
        SUPPORT_CLIP[0],
        SUPPORT_CLIP[1],
    ).astype(np.int8)
    indices = np.asarray(
        [new_class_order.index(label) for label in new_ordered_labels], dtype="<i2"
    )
    counts = tuple(int(np.sum(indices == index)) for index in range(len(new_class_order)))
    if any(count != config.active_k for count in counts):
        raise ADV3B02StateError("Stage2-C new suffix reference class-count drift")
    if config.active_k == 1:
        bandwidth = np.full(len(new_class_order), config.shared_h0, dtype=np.float64)
    else:
        values = []
        for index in range(len(new_class_order)):
            local = teacher[indices == index].astype(np.float64)
            cosine = np.clip(local @ local.T, -1.0, 1.0)
            distance = np.maximum(2.0 * (1.0 - cosine), 0.0)
            empirical = float(np.mean(distance[np.triu_indices(config.active_k, 1)]))
            shrunk = (
                empirical + config.scale_prior_strength * config.shared_h0**2
            ) / (1.0 + config.scale_prior_strength)
            values.append(
                np.clip(
                    math.sqrt(max(shrunk, 1.0e-12)),
                    config.shared_h0 * config.scale_min_ratio,
                    config.shared_h0 * config.scale_max_ratio,
                )
            )
        bandwidth = np.asarray(values, np.float64)
    raw_bandwidth = np.asarray(bandwidth, np.float32)
    bandwidth_hi = np.asarray(raw_bandwidth, dtype="<f2")
    bandwidth_lo = np.asarray(
        raw_bandwidth - bandwidth_hi.astype(np.float32), dtype="<f2"
    )
    return (
        np.ascontiguousarray(codes),
        np.ascontiguousarray(scales),
        np.ascontiguousarray(offsets),
        np.ascontiguousarray(residual_codes),
        np.ascontiguousarray(residual_scales),
        np.ascontiguousarray(residual2_codes),
        np.ascontiguousarray(residual2_scales),
        np.ascontiguousarray(bandwidth_hi),
        np.ascontiguousarray(bandwidth_lo),
    )


def _append_bank(old: Int8QKNNState, new_zid: np.ndarray, new_labels: tuple[str, ...],
                 new_tokens: tuple[str, ...], new_classes: tuple[str, ...], *,
                 complete_teacher_by_token: Mapping[str, np.ndarray],
                 support_repair_receipt: Mapping[str, Any]) -> Int8QKNNState:
    if any(item in old.classes for item in new_classes) or any(item not in new_classes for item in new_labels):
        raise ADV3B02StateError("Stage2-C must append wholly new registered classes")
    if set(old.support_tokens) & set(new_tokens):
        raise ADV3B02StateError("Stage2-C support physical IDs must be disjoint")
    new = _rows(new_zid, name="S_C new z_id")
    if len(new) != len(new_labels) or len(new) != len(new_tokens):
        raise ADV3B02StateError("S_C new z_id/token layout drift")
    if _balanced_k(new_labels, new_classes) != old.k_shot:
        raise ADV3B02StateError("Stage2-C new support K differs from frozen old state")
    classes = old.classes + new_classes
    all_bank_classes = old.bank.classes + tuple(sorted(new_classes))
    old_labels = tuple(old.bank.classes[int(i)] for i in old.bank.class_indices_int16.tolist())
    order = sorted(range(len(new_tokens)), key=lambda i: (new_labels[i], new_tokens[i]))
    new_ordered_raw = np.ascontiguousarray(new[np.asarray(order, np.intp)])
    new_codes, new_scales, new_offsets, new_residual_codes, new_residual_scales, new_residual2_codes, new_residual2_scales = (
        _affine_quantize_rows_two_plane(new_ordered_raw)
    )
    labels = old_labels + tuple(new_labels[i] for i in order)
    tokens = old.support_tokens + tuple(new_tokens[i] for i in order)
    if set(complete_teacher_by_token) != set(tokens):
        raise ADV3B02StateError("Stage2-C complete FP32 teacher/token closure drift")
    teacher = np.stack(
        [complete_teacher_by_token[token] for token in tokens]
    ).astype(np.float32)
    teacher_unit = _unit(teacher)
    indices = np.asarray([all_bank_classes.index(v) for v in labels], dtype="<i2")
    codes = np.concatenate((old.bank.codes_qint8, new_codes), axis=0)
    scales = np.concatenate((old.bank.scales_fp16, new_scales), axis=0).astype("<f2")
    offsets = np.concatenate((old.bank.offsets_fp16, new_offsets), axis=0).astype("<f2")
    residual_codes = np.concatenate(
        (old.bank.residual_codes_qint8, new_residual_codes), axis=0
    ).astype(np.int8)
    residual_scales = np.concatenate(
        (old.bank.residual_scales_fp16, new_residual_scales), axis=0
    ).astype("<f2")
    residual2_codes = np.concatenate(
        (old.bank.residual2_codes_qint8, new_residual2_codes), axis=0
    ).astype(np.int8)
    residual2_scales = np.concatenate(
        (old.bank.residual2_scales_fp16, new_residual2_scales), axis=0
    ).astype("<f2")
    counts = tuple(int(np.sum(indices == i)) for i in range(len(all_bank_classes)))
    raw_class_scales = np.asarray(
        _existing_identity_class_scales(
            teacher_unit, indices, len(all_bank_classes), old.bank.config
        ),
        np.float32,
    )
    class_scale_hi, class_scale_lo = _split_class_bandwidths(raw_class_scales)
    old_class_count = len(old.bank.classes)
    class_scale_hi[:old_class_count] = old.bank.class_scale_hi_fp16
    class_scale_lo[:old_class_count] = old.bank.class_scale_lo_fp16
    quant = {"schema": AFFINE_BANK_SCHEMA, "codec": SUPPORT_CODEC,
             "plane_order": list(SUPPORT_PLANE_ORDER),
             "rounding": SUPPORT_ROUNDING, "clip": list(SUPPORT_CLIP),
             "residual_scale_floor": SUPPORT_RESIDUAL_SCALE_FLOOR,
             "class_bandwidth_codec": CLASS_BANDWIDTH_CODEC, "support_only": True,
             "query_rows_used_for_fit": 0, "codes_sha256": sha256_bytes(codes.tobytes()), "scales_sha256": sha256_bytes(scales.tobytes()),
             "offsets_sha256": sha256_bytes(offsets.tobytes()),
             "residual_codes_sha256": sha256_bytes(residual_codes.tobytes()),
             "residual_scales_sha256": sha256_bytes(residual_scales.tobytes()),
             "residual2_codes_sha256": sha256_bytes(residual2_codes.tobytes()),
             "residual2_scales_sha256": sha256_bytes(residual2_scales.tobytes()),
             "class_scale_hi_sha256": sha256_bytes(class_scale_hi.tobytes()),
             "class_scale_lo_sha256": sha256_bytes(class_scale_lo.tobytes()),
             "endianness": "little"}
    provisional = AffineINT8ZIDSupportBank(
        all_bank_classes, counts, codes, scales, offsets, residual_codes,
        residual_scales, residual2_codes, residual2_scales, indices, class_scale_hi, class_scale_lo, old.k_shot,
        old.bank.config, old.bank.config_lock_digest, quant, "0" * 64,
    )
    bank = replace(
        provisional,
        bank_receipt_sha256=sha256_bytes(
            _canon(_affine_bank_payload(provisional))
        ),
    )
    # Recompute the new suffix from its current legal FP32 support rather than
    # trusting the just-assembled bank payload.  This is intentionally a
    # construction-time closure only: old deployed bytes remain immutable and
    # no FP32 sidecar is persisted or reachable from query inference.
    new_class_order = tuple(sorted(new_classes))
    (
        expected_new_codes,
        expected_new_scales,
        expected_new_offsets,
        expected_new_residual_codes,
        expected_new_residual_scales,
        expected_new_residual2_codes,
        expected_new_residual2_scales,
        expected_new_class_scale_hi,
        expected_new_class_scale_lo,
    ) = _reference_new_suffix_codec(
        new_ordered_raw,
        tuple(new_labels[i] for i in order),
        new_class_order,
        old.bank.config,
    )
    old_rows = old.bank.support_row_count
    if (
        not np.array_equal(bank.codes_qint8[old_rows:], expected_new_codes)
        or not np.array_equal(bank.scales_fp16[old_rows:], expected_new_scales)
        or not np.array_equal(bank.offsets_fp16[old_rows:], expected_new_offsets)
        or not np.array_equal(
            bank.residual_codes_qint8[old_rows:], expected_new_residual_codes
        )
        or not np.array_equal(
            bank.residual_scales_fp16[old_rows:], expected_new_residual_scales
        )
        or not np.array_equal(
            bank.residual2_codes_qint8[old_rows:], expected_new_residual2_codes
        )
        or not np.array_equal(
            bank.residual2_scales_fp16[old_rows:], expected_new_residual2_scales
        )
        or not np.array_equal(
            bank.class_scale_hi_fp16[old_class_count:], expected_new_class_scale_hi
        )
        or not np.array_equal(
            bank.class_scale_lo_fp16[old_class_count:], expected_new_class_scale_lo
        )
    ):
        raise ADV3B02StateError("Stage2-C new suffix codec drift")
    wire = _serialize_affine_bank(bank)
    # The deployed Stage2-B prefix cannot be audited against a later re-extract
    # of those old support rows: their batch-dependent float32 values are not
    # part of the frozen bank lifecycle.  Keep the complete after teacher for
    # token/label/repair binding above, but audit the actual deployed old prefix
    # against its own Q1/Q2/Q3 decode.  New classes remain the current FP32
    # support and therefore still exercise the append codec end-to-end.
    matched_old_teacher = _affine_dequantize_rows(
        old.bank.codes_qint8,
        old.bank.scales_fp16,
        old.bank.offsets_fp16,
        old.bank.residual_codes_qint8,
        old.bank.residual_scales_fp16,
        old.bank.residual2_codes_qint8,
        old.bank.residual2_scales_fp16,
    )
    matched_teacher = np.concatenate(
        (matched_old_teacher, new_ordered_raw), axis=0
    ).astype(np.float32)
    matched_class_scales = np.asarray(
        _existing_identity_class_scales(
            _unit(matched_teacher), indices, len(all_bank_classes), old.bank.config
        ),
        np.float32,
    )
    matched_class_scales[:old_class_count] = old.bank.deployed_class_scales()
    audit = _affine_margin_audit(
        bank,
        matched_teacher,
        labels,
        matched_teacher,
        teacher_class_scales=matched_class_scales,
        teacher_bandwidth_source="matched_frozen_old_bank_plus_new_FP32",
    )
    branch = _make_actual_branch(bank, old.metric, wire, teacher, labels, tokens, audit,
                                 support_repair_receipt)
    result = Int8QKNNState(branch, bank, old.metric, wire, classes, labels, tokens)
    if (not np.array_equal(result.bank.codes_qint8[:old_rows], old.bank.codes_qint8)
            or not np.array_equal(result.bank.scales_fp16[:old_rows], old.bank.scales_fp16)
            or not np.array_equal(result.bank.offsets_fp16[:old_rows], old.bank.offsets_fp16)
            or not np.array_equal(result.bank.residual_codes_qint8[:old_rows], old.bank.residual_codes_qint8)
            or not np.array_equal(result.bank.residual_scales_fp16[:old_rows], old.bank.residual_scales_fp16)
            or not np.array_equal(result.bank.residual2_codes_qint8[:old_rows], old.bank.residual2_codes_qint8)
            or not np.array_equal(result.bank.residual2_scales_fp16[:old_rows], old.bank.residual2_scales_fp16)
            or not np.array_equal(result.bank.class_scale_hi_fp16[:old_class_count], old.bank.class_scale_hi_fp16)
            or not np.array_equal(result.bank.class_scale_lo_fp16[:old_class_count], old.bank.class_scale_lo_fp16)):
        raise ADV3B02StateError("Stage2-C changed frozen Q1/Q2/Q3 old INT8 bytes")
    return result


def _old_domain_prefix_bytes(domain: DomainState) -> bytes:
    """Serialize only the immutable Stage2-B prefix of a domain state."""
    old_rows = domain.old_class_count * domain.k_shot
    header = _canon(
        {
            "schema": SCHEMA,
            "kind": "fixed3_domain_old_prefix",
            "classes": list(domain.classes[: domain.old_class_count]),
            "k": domain.k_shot,
            "old_class_count": domain.old_class_count,
            "alpha": float(domain.alpha),
        }
    )
    return b"".join(
        (
            header,
            np.ascontiguousarray(domain.zdom_codes[:old_rows]).tobytes(),
            np.ascontiguousarray(domain.zdom_scales[:old_rows]).tobytes(),
            np.ascontiguousarray(
                domain.centres[: domain.old_class_count]
            ).tobytes(),
            np.ascontiguousarray(domain.q).tobytes(),
            np.ascontiguousarray(domain.a).tobytes(),
            np.ascontiguousarray(domain.rho).tobytes(),
        )
    )


def _array_sha256(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


_QKNN_MARGIN_AUDIT_FIELDS = {
    "schema",
    "validation_row_count",
    "logit_abs_error_mean",
    "logit_abs_error_max",
    "top1_agreement",
    "teacher_margin_mean",
    "quantized_teacher_margin_mean",
    "any_margin_flip_count",
    "any_margin_flip_rate",
    "large_margin_flip_count",
    "large_margin_flip_rate",
    "fp32_teacher_bandwidth_source",
    "fp32_teacher_support_sha256",
    "int8_bank_class_scales_sha256",
    "int8_bank_class_scale_count",
    "teacher_bank_bandwidth_abs_delta_max",
    "query_rows_used_for_fit",
    "state_updates",
}


_APPEND_RECEIPT_FIELDS = {
    "schema",
    "stage",
    "query_rows_used_for_fit",
    "old_state_sha256",
    "after_state_sha256",
    "old_domain_digest_before",
    "frozen_old_digest_in_after",
    "old_domain_prefix_sha256_before",
    "old_domain_prefix_sha256_after",
    "old_domain_prefix_bytes",
    "old_domain_bytes_preserved",
    "old_int8_codes_sha256_before",
    "old_int8_codes_sha256_after",
    "old_int8_scales_sha256_before",
    "old_int8_scales_sha256_after",
    "old_int8_offsets_sha256_before",
    "old_int8_offsets_sha256_after",
    "old_int8_residual_codes_sha256_before",
    "old_int8_residual_codes_sha256_after",
    "old_int8_residual_scales_sha256_before",
    "old_int8_residual_scales_sha256_after",
    "old_int8_residual2_codes_sha256_before",
    "old_int8_residual2_codes_sha256_after",
    "old_int8_residual2_scales_sha256_before",
    "old_int8_residual2_scales_sha256_after",
    "old_int8_class_scale_hi_sha256_before",
    "old_int8_class_scale_hi_sha256_after",
    "old_int8_class_scale_lo_sha256_before",
    "old_int8_class_scale_lo_sha256_after",
    "old_int8_codes_preserved",
    "old_int8_scales_preserved",
    "old_int8_offsets_preserved",
    "old_int8_residual_codes_preserved",
    "old_int8_residual_scales_preserved",
    "old_int8_residual2_codes_preserved",
    "old_int8_residual2_scales_preserved",
    "old_int8_class_scale_hi_preserved",
    "old_int8_class_scale_lo_preserved",
    "old_q_sha256_before",
    "old_q_sha256_after",
    "old_a_sha256_before",
    "old_a_sha256_after",
    "old_rho_sha256_before",
    "old_rho_sha256_after",
    "old_alpha_before",
    "old_alpha_after",
    "old_q_a_alpha_refit",
    "after_bank_receipt_sha256",
    "after_qknn_wire_sha256",
    "after_metric_receipt_sha256",
    "after_branch_actual_bank_binding_sha256",
    "after_branch_teacher_support_sha256",
    "after_support_repair_receipt_sha256",
    "after_support_repair_unit_output_sha256",
    "after_int8_audit",
    "after_int8_audit_sha256",
    "after_support_row_count",
    "after_class_count",
    "after_support_token_root_sha256",
    "receipt_sha256",
}


def verify_stage2_c_append_receipt(
    receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = dict(receipt) if isinstance(receipt, Mapping) else {}
    if set(value) != _APPEND_RECEIPT_FIELDS:
        raise ADV3B02StateError("Stage2-C append receipt schema drift")
    for name in _APPEND_RECEIPT_FIELDS:
        if name.endswith("_sha256"):
            _require_sha256(value[name], name=name)
    audit = value["after_int8_audit"]
    if (
        type(audit) is not dict
        or set(audit) != _QKNN_MARGIN_AUDIT_FIELDS
        or audit.get("schema")
        != "cvs.phase2.zid_student_t_qknn.margin_audit.v4_q3support1"
        or type(value["after_support_row_count"]) is not int
        or value["after_support_row_count"] <= 0
        or audit.get("validation_row_count")
        != value["after_support_row_count"]
        or audit.get("query_rows_used_for_fit") != 0
        or audit.get("state_updates") != 0
        or float(audit.get("top1_agreement", -1.0)) < 0.995
        or audit.get("large_margin_flip_count") != 0
        or type(value["after_class_count"]) is not int
        or value["after_class_count"] < 2
        or audit.get("int8_bank_class_scale_count") != value["after_class_count"]
        or value["after_int8_audit_sha256"] != sha256_bytes(_canon(audit))
    ):
        raise ADV3B02StateError("Stage2-C actual after-bank audit drift")
    equal_hash_pairs = (
        ("old_domain_prefix_sha256_before", "old_domain_prefix_sha256_after"),
        ("old_int8_codes_sha256_before", "old_int8_codes_sha256_after"),
        ("old_int8_scales_sha256_before", "old_int8_scales_sha256_after"),
        ("old_int8_offsets_sha256_before", "old_int8_offsets_sha256_after"),
        (
            "old_int8_residual_codes_sha256_before",
            "old_int8_residual_codes_sha256_after",
        ),
        (
            "old_int8_residual_scales_sha256_before",
            "old_int8_residual_scales_sha256_after",
        ),
        (
            "old_int8_residual2_codes_sha256_before",
            "old_int8_residual2_codes_sha256_after",
        ),
        (
            "old_int8_residual2_scales_sha256_before",
            "old_int8_residual2_scales_sha256_after",
        ),
        (
            "old_int8_class_scale_hi_sha256_before",
            "old_int8_class_scale_hi_sha256_after",
        ),
        (
            "old_int8_class_scale_lo_sha256_before",
            "old_int8_class_scale_lo_sha256_after",
        ),
        ("old_q_sha256_before", "old_q_sha256_after"),
        ("old_a_sha256_before", "old_a_sha256_after"),
        ("old_rho_sha256_before", "old_rho_sha256_after"),
    )
    if (
        value["schema"] != APPEND_RECEIPT_SCHEMA
        or value["stage"] != "S_C"
        or value["query_rows_used_for_fit"] != 0
        or value["old_domain_prefix_bytes"] <= 0
        or value["old_domain_bytes_preserved"] is not True
        or value["old_int8_codes_preserved"] is not True
        or value["old_int8_scales_preserved"] is not True
        or value["old_int8_offsets_preserved"] is not True
        or value["old_int8_residual_codes_preserved"] is not True
        or value["old_int8_residual_scales_preserved"] is not True
        or value["old_int8_residual2_codes_preserved"] is not True
        or value["old_int8_residual2_scales_preserved"] is not True
        or value["old_int8_class_scale_hi_preserved"] is not True
        or value["old_int8_class_scale_lo_preserved"] is not True
        or value["old_q_a_alpha_refit"] is not False
        or any(value[left] != value[right] for left, right in equal_hash_pairs)
        or value["old_alpha_before"] != value["old_alpha_after"]
        or value["frozen_old_digest_in_after"]
        != value["old_domain_digest_before"]
        or value["after_branch_teacher_support_sha256"]
        != value["after_support_repair_unit_output_sha256"]
    ):
        raise ADV3B02StateError("Stage2-C frozen old-state receipt drift")
    body = {key: value[key] for key in value if key != "receipt_sha256"}
    if value["receipt_sha256"] != sha256_bytes(_canon(body)):
        raise ADV3B02StateError("Stage2-C append receipt SHA drift")
    return value


def append_stage2_c(old_state: DualQKNNState, *, new_support_zid: np.ndarray,
                    new_support_zdom: np.ndarray, new_support_labels: Sequence[Any] | np.ndarray,
                    new_registered_classes: Sequence[Any] | np.ndarray,
                    new_support_physical_tokens: Sequence[Any] | np.ndarray,
                    after_full_teacher_zid: np.ndarray,
                    after_full_teacher_physical_tokens: Sequence[Any] | np.ndarray,
                    after_support_repair_receipt: Mapping[str, Any] | None = None) -> tuple[DualQKNNState, dict[str, Any]]:
    """Append new classes without refitting Q/A/alpha or rewriting old banks."""
    if old_state.domain.stage != "S_B" or old_state.domain.old_class_count != len(old_state.id_bank.classes):
        raise ADV3B02StateError("Stage2-C requires one frozen Stage2-B old state")
    new_classes = typed_tokens(new_registered_classes, name="S_C new registry", unique=True)
    labels = typed_tokens(new_support_labels, name="S_C new labels")
    tokens = typed_tokens(new_support_physical_tokens, name="S_C new physical tokens", unique=True)
    zdom = _unit(_rows(new_support_zdom, name="S_C new z_dom"))
    before = old_state.domain.digest
    full_teacher = _rows(after_full_teacher_zid, name="S_C complete unquantized FP32 all-support teacher")
    full_tokens = typed_tokens(after_full_teacher_physical_tokens, name="S_C complete teacher physical tokens", unique=True)
    if len(full_teacher) != len(full_tokens):
        raise ADV3B02StateError("S_C complete teacher row/token layout drift")
    old_label_by_token = dict(zip(old_state.id_bank.support_tokens, old_state.id_bank.labels))
    new_label_by_token = dict(zip(tokens, labels))
    full_labels_list: list[str] = []
    for token in full_tokens:
        if token in old_label_by_token:
            full_labels_list.append(old_label_by_token[token])
        elif token in new_label_by_token:
            full_labels_list.append(new_label_by_token[token])
        else:
            raise ADV3B02StateError("S_C complete teacher label/token closure drift")
    full_labels = tuple(full_labels_list)
    all_classes = old_state.id_bank.classes + new_classes
    if after_support_repair_receipt is None:
        full_teacher, repair = repair_finite_exact_zero_singleton_class_medoid(
            full_teacher, full_labels, all_classes, full_tokens
        )
    else:
        repair = _validate_repaired_support_for_state(
            after_support_repair_receipt, support_zid=full_teacher,
            labels=full_labels, classes=all_classes, tokens=full_tokens,
        )
    teacher_by_token = {token: full_teacher[i] for i, token in enumerate(full_tokens)}
    expected_teacher_tokens = set(old_state.id_bank.support_tokens) | set(tokens)
    if set(teacher_by_token) != expected_teacher_tokens:
        raise ADV3B02StateError("S_C teacher must contain exactly the full old+new support")
    supplied_new_zid = _rows(new_support_zid, name="S_C supplied new z_id")
    expected_new_zid = np.stack([teacher_by_token[token] for token in tokens]).astype(np.float32)
    if not np.array_equal(supplied_new_zid, expected_new_zid):
        raise ADV3B02StateError("S_C supplied new z_id/full teacher token binding drift")
    new_zid = expected_new_zid
    bank = _append_bank(old_state.id_bank, new_zid, labels, tokens, new_classes,
                        complete_teacher_by_token=teacher_by_token,
                        support_repair_receipt=repair)
    token_positions = {token: index for index, token in enumerate(tokens)}
    appended_tokens = bank.support_tokens[len(old_state.id_bank.support_tokens):]
    if set(appended_tokens) != set(tokens):
        raise ADV3B02StateError("Stage2-C appended domain/token closure drift")
    order = [token_positions[token] for token in appended_tokens]
    ordered_dom = zdom[np.asarray(order, np.intp)]
    ordered_labels = tuple(labels[index] for index in order)
    new_codes, new_scales = _quantize_rows(ordered_dom)
    centres = np.stack([ordered_dom[np.asarray([label == item for label in ordered_labels], bool)].mean(axis=0)
                        for item in new_classes]).astype(np.float32)
    old_domain = old_state.domain
    domain = DomainState(bank.classes, np.concatenate((old_domain.zdom_codes, new_codes), axis=0),
                         np.concatenate((old_domain.zdom_scales, new_scales), axis=0),
                         np.concatenate((old_domain.centres, centres), axis=0), old_domain.q.copy(), old_domain.a.copy(),
                         old_domain.rho.copy(), float(old_domain.alpha), old_domain.k_shot, old_domain.old_class_count,
                         "S_C", before)
    after_int8_audit = int8_audit(bank)
    result = DualQKNNState(bank, domain, after_int8_audit)
    old_prefix_before = _old_domain_prefix_bytes(old_domain)
    old_prefix_after = _old_domain_prefix_bytes(result.domain)
    old_rows = old_state.id_bank.bank.support_row_count
    old_class_count = len(old_state.id_bank.bank.classes)
    after_old_class_scale_hi = np.ascontiguousarray(
        result.id_bank.bank.class_scale_hi_fp16[:old_class_count]
    )
    after_old_class_scale_lo = np.ascontiguousarray(
        result.id_bank.bank.class_scale_lo_fp16[:old_class_count]
    )
    branch = result.id_bank.branch_state
    if type(branch) is not ActualBankBranchState:
        raise ADV3B02StateError("Stage2-C did not close an actual-bank branch")
    body = {
        "schema": APPEND_RECEIPT_SCHEMA,
        "stage": "S_C",
        "query_rows_used_for_fit": 0,
        "old_state_sha256": old_state.digest,
        "after_state_sha256": result.digest,
        "old_domain_digest_before": before,
        "frozen_old_digest_in_after": result.domain.frozen_old_digest,
        "old_domain_prefix_sha256_before": sha256_bytes(old_prefix_before),
        "old_domain_prefix_sha256_after": sha256_bytes(old_prefix_after),
        "old_domain_prefix_bytes": len(old_prefix_before),
        "old_domain_bytes_preserved": old_prefix_before == old_prefix_after,
        "old_int8_codes_sha256_before": _array_sha256(
            old_state.id_bank.bank.codes_qint8
        ),
        "old_int8_codes_sha256_after": _array_sha256(
            result.id_bank.bank.codes_qint8[:old_rows]
        ),
        "old_int8_scales_sha256_before": _array_sha256(
            old_state.id_bank.bank.scales_fp16
        ),
        "old_int8_scales_sha256_after": _array_sha256(
            result.id_bank.bank.scales_fp16[:old_rows]
        ),
        "old_int8_offsets_sha256_before": _array_sha256(
            old_state.id_bank.bank.offsets_fp16
        ),
        "old_int8_offsets_sha256_after": _array_sha256(
            result.id_bank.bank.offsets_fp16[:old_rows]
        ),
        "old_int8_residual_codes_sha256_before": _array_sha256(
            old_state.id_bank.bank.residual_codes_qint8
        ),
        "old_int8_residual_codes_sha256_after": _array_sha256(
            result.id_bank.bank.residual_codes_qint8[:old_rows]
        ),
        "old_int8_residual_scales_sha256_before": _array_sha256(
            old_state.id_bank.bank.residual_scales_fp16
        ),
        "old_int8_residual_scales_sha256_after": _array_sha256(
            result.id_bank.bank.residual_scales_fp16[:old_rows]
        ),
        "old_int8_residual2_codes_sha256_before": _array_sha256(
            old_state.id_bank.bank.residual2_codes_qint8
        ),
        "old_int8_residual2_codes_sha256_after": _array_sha256(
            result.id_bank.bank.residual2_codes_qint8[:old_rows]
        ),
        "old_int8_residual2_scales_sha256_before": _array_sha256(
            old_state.id_bank.bank.residual2_scales_fp16
        ),
        "old_int8_residual2_scales_sha256_after": _array_sha256(
            result.id_bank.bank.residual2_scales_fp16[:old_rows]
        ),
        "old_int8_class_scale_hi_sha256_before": _array_sha256(
            old_state.id_bank.bank.class_scale_hi_fp16
        ),
        "old_int8_class_scale_hi_sha256_after": _array_sha256(
            after_old_class_scale_hi
        ),
        "old_int8_class_scale_lo_sha256_before": _array_sha256(
            old_state.id_bank.bank.class_scale_lo_fp16
        ),
        "old_int8_class_scale_lo_sha256_after": _array_sha256(
            after_old_class_scale_lo
        ),
        "old_int8_codes_preserved": bool(
            np.array_equal(
                old_state.id_bank.bank.codes_qint8,
                result.id_bank.bank.codes_qint8[:old_rows],
            )
        ),
        "old_int8_scales_preserved": bool(
            np.array_equal(
                old_state.id_bank.bank.scales_fp16,
                result.id_bank.bank.scales_fp16[:old_rows],
            )
        ),
        "old_int8_offsets_preserved": bool(
            np.array_equal(
                old_state.id_bank.bank.offsets_fp16,
                result.id_bank.bank.offsets_fp16[:old_rows],
            )
        ),
        "old_int8_residual_codes_preserved": bool(
            np.array_equal(
                old_state.id_bank.bank.residual_codes_qint8,
                result.id_bank.bank.residual_codes_qint8[:old_rows],
            )
        ),
        "old_int8_residual_scales_preserved": bool(
            np.array_equal(
                old_state.id_bank.bank.residual_scales_fp16,
                result.id_bank.bank.residual_scales_fp16[:old_rows],
            )
        ),
        "old_int8_residual2_codes_preserved": bool(
            np.array_equal(
                old_state.id_bank.bank.residual2_codes_qint8,
                result.id_bank.bank.residual2_codes_qint8[:old_rows],
            )
        ),
        "old_int8_residual2_scales_preserved": bool(
            np.array_equal(
                old_state.id_bank.bank.residual2_scales_fp16,
                result.id_bank.bank.residual2_scales_fp16[:old_rows],
            )
        ),
        "old_int8_class_scale_hi_preserved": bool(
            np.array_equal(
                old_state.id_bank.bank.class_scale_hi_fp16,
                after_old_class_scale_hi,
            )
        ),
        "old_int8_class_scale_lo_preserved": bool(
            np.array_equal(
                old_state.id_bank.bank.class_scale_lo_fp16,
                after_old_class_scale_lo,
            )
        ),
        "old_q_sha256_before": _array_sha256(old_domain.q),
        "old_q_sha256_after": _array_sha256(result.domain.q),
        "old_a_sha256_before": _array_sha256(old_domain.a),
        "old_a_sha256_after": _array_sha256(result.domain.a),
        "old_rho_sha256_before": _array_sha256(old_domain.rho),
        "old_rho_sha256_after": _array_sha256(result.domain.rho),
        "old_alpha_before": float(old_domain.alpha),
        "old_alpha_after": float(result.domain.alpha),
        "old_q_a_alpha_refit": False,
        "after_bank_receipt_sha256": result.id_bank.bank.bank_receipt_sha256,
        "after_qknn_wire_sha256": sha256_bytes(result.id_bank.qknn_wire),
        "after_metric_receipt_sha256": result.id_bank.metric.metric_receipt_sha256,
        "after_branch_actual_bank_binding_sha256": branch.actual_bank_binding_receipt[
            "receipt_sha256"
        ],
        "after_branch_teacher_support_sha256": branch.actual_bank_binding_receipt[
            "teacher_support_sha256"
        ],
        "after_support_repair_receipt_sha256": branch.actual_bank_binding_receipt[
            "support_repair_receipt_sha256"
        ],
        "after_support_repair_unit_output_sha256": branch.support_repair_receipt[
            "unit_output_support_sha256"
        ],
        "after_int8_audit": dict(after_int8_audit),
        "after_int8_audit_sha256": sha256_bytes(_canon(after_int8_audit)),
        "after_support_row_count": result.id_bank.bank.support_row_count,
        "after_class_count": len(result.id_bank.bank.classes),
        "after_support_token_root_sha256": sha256_bytes(
            _canon(list(result.id_bank.support_tokens))
        ),
    }
    receipt = {**body, "receipt_sha256": sha256_bytes(_canon(body))}
    verified = verify_stage2_c_append_receipt(receipt)
    return result, dict(verified)


def _domain_weights(state: DualQKNNState, query_dom: np.ndarray, *, class_index: int) -> np.ndarray:
    domain = state.domain
    positions = np.asarray([label == domain.classes[class_index] for label in state.id_bank.labels], bool)
    if domain.alpha == 0.0:
        return np.full((int(positions.sum()),), 1.0 / state.id_bank.k_shot, np.float64)
    support = domain.features()[positions]
    centre = domain.centres[class_index]
    query_coordinate = domain.a @ domain.q.T @ (query_dom - centre)
    support_coordinates = (domain.a @ domain.q.T @ (support - centre).T).T
    logits = support_coordinates @ query_coordinate
    logits -= np.max(logits)
    pi = np.exp(logits); pi /= pi.sum()
    return (1.0 - domain.alpha) / state.id_bank.k_shot + domain.alpha * pi


def _typed_bank_kernel_terms(state: Int8QKNNState, query: np.ndarray, *, class_index: int,
                             positions: np.ndarray) -> np.ndarray:
    """The formal typed-bank Student-t kernel, with no candidate-local h/nu copy."""
    bank = state.bank
    metric = state.metric
    if int(metric.effective_rank) != 0:
        raise ADV3B02StateError("ADV3B02 dual weighting requires the existing identity typed metric")
    support = state.features()[positions].astype(np.float64)
    distance = np.maximum(2.0 * (1.0 - np.clip(support @ query.astype(np.float64), -1.0, 1.0)), 0.0)
    config = bank.config
    h = float(
        bank.deployed_class_scales()[
            bank.classes.index(state.classes[class_index])
        ]
    )
    return np.asarray(
        -config.kernel_volume_gamma * config.kernel_effective_dim * math.log(h)
        -0.5 * (config.student_nu + config.kernel_effective_dim)
        * np.log1p(distance / (config.student_nu * h * h)), np.float64)


def dual_qknn_logits(state: DualQKNNState, query_zid: np.ndarray, query_zdom: np.ndarray) -> np.ndarray:
    """Candidate-centre-conditioned domain weighting over one shared z_id bank."""
    zid = _unit(_rows(query_zid, name="dual query z_id"))
    zdom = _unit(_rows(query_zdom, name="dual query z_dom"))
    if len(zid) != len(zdom):
        raise ADV3B02StateError("dual query z_id/z_dom row count drift")
    if state.domain.alpha == 0.0:
        return qknn_logits(state.id_bank, zid)
    result = np.empty((len(zid), len(state.id_bank.classes)), np.float32)
    for qi, query in enumerate(zid):
        for ci, item in enumerate(state.id_bank.classes):
            positions = np.asarray([label == item for label in state.id_bank.labels], bool)
            kernel = _typed_bank_kernel_terms(state.id_bank, query, class_index=ci, positions=positions)
            weights = _domain_weights(state, zdom[qi], class_index=ci)
            result[qi, ci] = np.float32(_logsumexp(kernel + np.log(weights)))
    return result


def domain_weight_audit(
    state: DualQKNNState, query_zdom: np.ndarray
) -> dict[str, Any]:
    """Summarize frozen domain-weight nonuniformity without query truth."""
    zdom = _unit(_rows(query_zdom, name="domain-weight audit query z_dom"))
    spans: list[float] = []
    for query in zdom:
        for class_index in range(len(state.id_bank.classes)):
            weights = _domain_weights(
                state, query, class_index=class_index
            ).astype(np.float64)
            if (
                not np.isfinite(weights).all()
                or not np.isclose(np.sum(weights), 1.0, atol=1.0e-7)
            ):
                raise ADV3B02StateError("deployment domain weights do not sum to one")
            spans.append(float(np.max(weights) - np.min(weights)))
    return {
        "query_class_rows": len(spans),
        "nonuniform_rows": sum(value > 0.0 for value in spans),
        "max_weight_span": max(spans, default=0.0),
        "mean_weight_span": float(np.mean(spans)) if spans else 0.0,
    }


_BCR_DIRECTIONS = ("0_to_1", "1_to_0")


def _raw_directional_loo(
    state: Int8QKNNState,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Directly reuse the existing two-direction BCR safety probe."""
    canonical_classes = tuple(state.bank.classes)
    shape = (len(state.labels), len(canonical_classes))
    if state.k_shot == 1:
        zero = np.zeros(shape, np.float64)
        return (
            {name: zero.copy() for name in _BCR_DIRECTIONS},
            {name: zero.copy() for name in _BCR_DIRECTIONS},
        )
    try:
        qscore, bscore = _existing_bcr_cross_view_loo(
            state.features().astype(np.float64),
            np.asarray(state.bank.class_indices_int16, np.intp),
            canonical_classes,
            state.bank.config,
        )
    except Exception as exc:
        raise ADV3B02StateError(
            f"existing BCR directional LOO failed: {exc}"
        ) from exc
    return (
        {name: np.asarray(qscore[name], np.float64) for name in _BCR_DIRECTIONS},
        {name: np.asarray(bscore[name], np.float64) for name in _BCR_DIRECTIONS},
    )


def _renormalized_domain_loo_weights(
    domain: DomainState,
    *,
    support_dom: np.ndarray,
    query_dom: np.ndarray,
    centre: np.ndarray,
) -> np.ndarray:
    if len(support_dom) < 1:
        raise ADV3B02StateError("domain LOO support is empty")
    query_coordinate = (
        domain.a.astype(np.float64)
        @ domain.q.astype(np.float64).T
        @ (query_dom - centre)
    )
    support_coordinates = (
        domain.a.astype(np.float64)
        @ domain.q.astype(np.float64).T
        @ (support_dom - centre).T
    ).T
    weight_logits = support_coordinates @ query_coordinate
    weight_logits -= np.max(weight_logits)
    pi = np.exp(weight_logits)
    pi /= np.sum(pi)
    weights = (1.0 - domain.alpha) / len(support_dom) + domain.alpha * pi
    weights /= np.sum(weights)
    if (
        not np.isfinite(weights).all()
        or not np.isclose(np.sum(weights), 1.0, atol=1.0e-12)
    ):
        raise ADV3B02StateError("dual directional LOO weight normalization drift")
    return weights


def _directional_dual_loo(
    state: DualQKNNState,
    raw_qscore: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Recompute dual qKNN on the exact existing masked-view LOO directions."""
    bank = state.id_bank
    canonical_classes = tuple(bank.bank.classes)
    expected_shape = (len(bank.labels), len(canonical_classes))
    if set(raw_qscore) != set(_BCR_DIRECTIONS):
        raise ADV3B02StateError("raw directional qKNN probe schema drift")
    if bank.k_shot == 1 or state.domain.alpha == 0.0:
        result = {
            name: np.asarray(raw_qscore[name], np.float64).copy()
            for name in _BCR_DIRECTIONS
        }
        if any(value.shape != expected_shape for value in result.values()):
            raise ADV3B02StateError("raw directional qKNN probe shape drift")
        return result

    masked = _existing_bcr_masked_views(bank.features().astype(np.float64))
    if any(np.any(np.linalg.norm(view, axis=1) <= 1.0e-12) for view in masked):
        raise ADV3B02StateError("dual BCRR masked-view degeneracy")
    views = tuple(
        normalize_zid_rows(np.asarray(view, np.float32)).astype(np.float64)
        for view in masked
    )
    indices = np.asarray(bank.bank.class_indices_int16, np.intp)
    zdom = state.domain.features().astype(np.float64)
    config = bank.bank.config
    out: dict[str, np.ndarray] = {}
    for source_index, destination_index, name in (
        (0, 1, "0_to_1"),
        (1, 0, "1_to_0"),
    ):
        score = np.empty(expected_shape, np.float64)
        for row_index in range(len(indices)):
            keep = np.arange(len(indices)) != row_index
            _, _, decoded_train = _existing_qknn_quantize_rows(
                np.asarray(views[destination_index][keep], np.float32)
            )
            train = normalize_zid_rows(decoded_train).astype(np.float64)
            train_y = indices[keep]
            source = views[source_index][row_index]
            for canonical_index, class_handle in enumerate(canonical_classes):
                class_train = train[train_y == canonical_index]
                if not len(class_train):
                    raise ADV3B02StateError("dual directional LOO class is empty")
                distance = np.maximum(
                    2.0
                    * (
                        1.0
                        - np.clip(class_train @ source, -1.0, 1.0)
                    ),
                    0.0,
                )
                if len(class_train) < 2:
                    hscale = float(config.shared_h0)
                else:
                    pair = np.maximum(
                        2.0
                        * (
                            1.0
                            - np.clip(
                                class_train @ class_train.T, -1.0, 1.0
                            )
                        ),
                        0.0,
                    )
                    empirical = float(
                        np.mean(pair[np.triu_indices(len(class_train), 1)])
                    )
                    shrunk = (
                        empirical
                        + config.scale_prior_strength * config.shared_h0**2
                    ) / (1.0 + config.scale_prior_strength)
                    hscale = float(
                        np.clip(
                            math.sqrt(max(shrunk, 1.0e-12)),
                            config.shared_h0 * config.scale_min_ratio,
                            config.shared_h0 * config.scale_max_ratio,
                        )
                    )
                kernel = (
                    -config.kernel_volume_gamma
                    * config.kernel_effective_dim
                    * math.log(hscale)
                    - 0.5
                    * (config.student_nu + config.kernel_effective_dim)
                    * np.log1p(
                        distance / (config.student_nu * hscale * hscale)
                    )
                )
                candidate_rows = keep & (indices == canonical_index)
                support_dom = zdom[candidate_rows]
                domain_index = state.domain.classes.index(class_handle)
                centre = state.domain.centres[domain_index].astype(np.float64)
                # The own class has K-1 rows after same-physical-ID LOO.
                # Its uniform component therefore also uses K-1 and the final
                # weights are explicitly renormalized to unit mass.
                weights = _renormalized_domain_loo_weights(
                    state.domain,
                    support_dom=support_dom,
                    query_dom=zdom[row_index],
                    centre=centre,
                )
                if (
                    len(weights) != len(kernel)
                ):
                    raise ADV3B02StateError(
                        "dual directional LOO weight normalization drift"
                    )
                score[row_index, canonical_index] = _logsumexp(
                    kernel + np.log(weights)
                )
        out[name] = score
    return out


@dataclass(frozen=True)
class BCRRState:
    """Branch-local continuous residual correction built only from z_id/LOO values."""

    bank_digest: str
    loo_sha256: str
    omega: float
    k_shot: int
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (type(self.bank_digest) is not str or len(self.bank_digest) != 64
                or type(self.loo_sha256) is not str or len(self.loo_sha256) != 64):
            raise ADV3B02StateError("BCRR branch identity/LOO receipt drift")
        if not np.isfinite(self.omega) or not 0.0 <= self.omega <= 0.5:
            raise ADV3B02StateError("BCRR omega range drift")
        if self.receipt.get("bcrr_reads_z_dom") is not False or self.receipt.get("query_rows_used_for_fit") != 0:
            raise ADV3B02StateError("BCRR must be z_dom-free and support-only")


def fit_bcrr_branch(
    *,
    id_bank: Int8QKNNState,
    directional_qscore: Mapping[str, np.ndarray],
    directional_bscore: Mapping[str, np.ndarray],
) -> BCRRState:
    """Seal omega from two exact same-physical-ID masked-view directions."""
    canonical_classes = tuple(id_bank.bank.classes)
    expected_shape = (len(id_bank.labels), len(canonical_classes))
    if (
        set(directional_qscore) != set(_BCR_DIRECTIONS)
        or set(directional_bscore) != set(_BCR_DIRECTIONS)
    ):
        raise ADV3B02StateError("branch-local BCRR directional schema drift")
    qscore = {
        name: np.asarray(directional_qscore[name], np.float64)
        for name in _BCR_DIRECTIONS
    }
    bscore = {
        name: np.asarray(directional_bscore[name], np.float64)
        for name in _BCR_DIRECTIONS
    }
    if any(
        value.shape != expected_shape or not np.isfinite(value).all()
        for value in tuple(qscore.values()) + tuple(bscore.values())
    ):
        raise ADV3B02StateError("branch-local BCRR directional logits drift")
    indices = np.asarray(id_bank.bank.class_indices_int16, np.intp)
    if id_bank.k_shot == 1:
        omega = 0.0
        fallback = "K1_identity"; qloss = floss = {"0_to_1": {item: 0.0 for item in canonical_classes}, "1_to_0": {item: 0.0 for item in canonical_classes}}
    else:
        try:
            qnorm = {
                name: _normalize_existing_scores(value)
                for name, value in qscore.items()
            }
            bnorm = {name: _normalize_existing_scores(value) for name, value in bscore.items()}
            def losses(weight: float) -> dict[str, dict[str, float]]:
                result: dict[str, dict[str, float]] = {}
                for name in qnorm:
                    fused = (1.0 - weight) * qnorm[name] + weight * bnorm[name]
                    maximum = np.max(fused, axis=1, keepdims=True)
                    probability = np.exp(fused - maximum); probability /= probability.sum(axis=1, keepdims=True)
                    row_loss = -np.log(np.maximum(probability[np.arange(len(indices)), indices], 1.0e-30))
                    result[name] = {label: float(np.mean(row_loss[indices == ci])) for ci, label in enumerate(canonical_classes)}
                return result
            qloss = losses(0.0)
            def safe(weight: float) -> bool:
                candidate = losses(weight)
                return all(candidate[d][label] <= qloss[d][label] + 1.0e-12 for d in qloss for label in canonical_classes)
            def objective_derivative(weight: float) -> tuple[float, float]:
                values: list[float] = []; derivatives: list[float] = []
                for name in qnorm:
                    fused = (1.0 - weight) * qnorm[name] + weight * bnorm[name]
                    delta = bnorm[name] - qnorm[name]
                    maximum = np.max(fused, axis=1, keepdims=True)
                    probability = np.exp(fused - maximum); probability /= probability.sum(axis=1, keepdims=True)
                    row_loss = -np.log(np.maximum(probability[np.arange(len(indices)), indices], 1.0e-30))
                    derivative = np.sum(probability * delta, axis=1) - delta[np.arange(len(indices)), indices]
                    for ci in range(len(canonical_classes)):
                        values.append(float(np.mean(row_loss[indices == ci]))); derivatives.append(float(np.mean(derivative[indices == ci])))
                return float(np.mean(values)), float(np.mean(derivatives))
            low, high = 0.0, BCRR_MAX_OMEGA
            for _ in range(24):
                middle = (low + high) / 2.0
                if safe(middle) and objective_derivative(middle)[1] < 0.0: low = middle
                else: high = middle
            star = low if objective_derivative(low)[0] < objective_derivative(0.0)[0] else 0.0
            omega = math.floor(BCRR_DENOMINATOR * star) / BCRR_DENOMINATOR
            floss = losses(omega)
            if not all(floss[d][label] <= qloss[d][label] + 1.0e-10 for d in qloss for label in canonical_classes):
                omega = 0.0; floss = qloss; fallback = "safety_set_empty"
            else:
                fallback = "none"
        except Exception:
            omega = 0.0; fallback = "score_normalization_degenerate"
            qloss = floss = {"0_to_1": {item: 0.0 for item in canonical_classes}, "1_to_0": {item: 0.0 for item in canonical_classes}}
    directional_sha = {
        name: {
            "qknn_sha256": sha256_bytes(
                np.ascontiguousarray(qscore[name]).tobytes()
            ),
            "bcr_sha256": sha256_bytes(
                np.ascontiguousarray(bscore[name]).tobytes()
            ),
        }
        for name in _BCR_DIRECTIONS
    }
    loo_sha = sha256_bytes(_canon(directional_sha))
    receipt = {"schema": SCHEMA, "branch_bank_sha256": id_bank.digest, "loo_sha256": loo_sha,
               "directional_logits_sha256": directional_sha,
               "omega": float(omega), "bcrr_reads_z_dom": False, "query_rows_used_for_fit": 0,
               "same_physical_id_loo": True,
               "masked_view_directions": list(_BCR_DIRECTIONS),
               "k_shot": id_bank.k_shot,
               "bcr_codes_and_weights": "existing_branch_z_id_support_shared",
               "omega_prelocked_safety": {"normalization": "sqrt(C)*(s-mean(s))/l2(s-mean(s))", "fusion": "F=(1-omega)*N(qKNN)+omega*N(BCR)", "denominator": BCRR_DENOMINATOR, "fallback": fallback, "directional_class_loss_qknn": qloss, "directional_class_loss_bcrr": floss}}
    return BCRRState(id_bank.digest, loo_sha, float(omega), id_bank.k_shot, receipt)


def bcrr_fused_logits(qknn_scores: np.ndarray, query_zid: np.ndarray, state: BCRRState, *, bank: Int8QKNNState) -> np.ndarray:
    scores = np.asarray(qknn_scores, np.float32)
    if state.bank_digest != bank.digest or scores.ndim != 2 or scores.shape[1] != len(bank.classes):
        raise ADV3B02StateError("BCRR must consume its exact matched qKNN branch")
    if state.omega == 0.0:
        return scores.copy()
    features = normalize_zid_rows(_rows(query_zid, name="BCRR query z_id"))
    branch = bank.branch_state
    d1 = np.asarray(
        branch.bcr_weight_codes_qint8.astype(np.float32)
        * branch.bcr_weight_scales_fp16.astype(np.float32)[None, :], np.float32
    )
    d2 = np.asarray(
        d1 + branch.bcr_weight_residual_codes_qint8.astype(np.float32)
        * branch.bcr_weight_residual_scales_fp16.astype(np.float32)[None, :], np.float32
    )
    weights = np.asarray(
        d2 + branch.bcr_weight_residual2_codes_qint8.astype(np.float32)
        * branch.bcr_weight_residual2_scales_fp16.astype(np.float32)[None, :], np.float32
    )
    bcr = np.asarray(features @ weights, np.float32)
    # Existing BCR weight columns are canonical typed-bank order, whereas the
    # qKNN scores are in the caller's registered-class order.
    bcr = bcr[
        :,
        np.asarray(
            [bank.bank.classes.index(item) for item in bank.classes], np.intp
        ),
    ]
    if bcr.shape != scores.shape or not np.isfinite(bcr).all():
        raise ADV3B02StateError("existing BCRR z_id weight geometry drift")
    fused = np.array(scores, dtype=np.float32, copy=True)
    for row in range(len(scores)):
        try:
            nq = _normalize_existing_scores(scores[row:row + 1])[0]
            nb = _normalize_existing_scores(bcr[row:row + 1])[0]
            fused[row] = np.asarray((1.0 - state.omega) * nq + state.omega * nb, np.float32)
        except Exception:
            fused[row] = scores[row]
    return fused


def build_four_arm_states(*, support_zid: np.ndarray, support_zdom: np.ndarray,
                          support_labels: Sequence[Any] | np.ndarray,
                          registered_classes: Sequence[Any] | np.ndarray,
                          support_physical_tokens: Sequence[Any] | np.ndarray,
                          support_repair_receipt: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """Seal four arms; paired arms share the exact raw or dual qKNN object."""
    dual = build_stage2_b_state(support_zid=support_zid, support_zdom=support_zdom,
                                support_labels=support_labels, registered_classes=registered_classes,
                                support_physical_tokens=support_physical_tokens,
                                support_repair_receipt=support_repair_receipt)
    raw = dual.id_bank
    raw_qscore, bscore = _raw_directional_loo(raw)
    dual_qscore = _directional_dual_loo(dual, raw_qscore)
    raw_bcrr = fit_bcrr_branch(
        id_bank=raw,
        directional_qscore=raw_qscore,
        directional_bscore=bscore,
    )
    dual_bcrr = fit_bcrr_branch(
        id_bank=dual.id_bank,
        directional_qscore=dual_qscore,
        directional_bscore=bscore,
    )
    states = {"M0": raw, "M_DA": dual, "M_OTHER": (raw, raw_bcrr),
              "M_JOINT": (dual, dual_bcrr), "query_rows_used_for_fit": 0}
    if states["M_OTHER"][0] is not raw or states["M_JOINT"][0] is not dual:
        raise ADV3B02StateError("four-arm raw/dual state sharing drift")
    return states


def build_four_arm_states_from_dual(dual: DualQKNNState) -> Mapping[str, Any]:
    """Close Stage2-C arms from an append-only dual state without refitting S_B."""
    if dual.domain.stage != "S_C":
        raise ADV3B02StateError("Stage2-C four-arm closure requires an appended dual state")
    raw = dual.id_bank
    raw_qscore, bscore = _raw_directional_loo(raw)
    dual_qscore = _directional_dual_loo(dual, raw_qscore)
    raw_bcrr = fit_bcrr_branch(
        id_bank=raw,
        directional_qscore=raw_qscore,
        directional_bscore=bscore,
    )
    dual_bcrr = fit_bcrr_branch(
        id_bank=dual.id_bank,
        directional_qscore=dual_qscore,
        directional_bscore=bscore,
    )
    return {"M0": raw, "M_DA": dual, "M_OTHER": (raw, raw_bcrr),
            "M_JOINT": (dual, dual_bcrr), "query_rows_used_for_fit": 0}


def predict_four_arms(states: Mapping[str, Any], *, query_zid: np.ndarray,
                      query_zdom: np.ndarray) -> Mapping[str, np.ndarray]:
    """Independent per-query all-class competition; this API has no truth or role."""
    if set(states) != set(ARMS) | {"query_rows_used_for_fit"} or states["query_rows_used_for_fit"] != 0:
        raise ADV3B02StateError("four-arm/query-fit state schema drift")
    raw = states["M0"]; dual = states["M_DA"]; other_bank, other_bcrr = states["M_OTHER"]; joint_dual, joint_bcrr = states["M_JOINT"]
    if not isinstance(raw, Int8QKNNState) or not isinstance(dual, DualQKNNState) or other_bank is not raw or joint_dual is not dual:
        raise ADV3B02StateError("four-arm matched branch sharing drift")
    zid, zero_mask = _query_zid_exact_zero_mask(query_zid)
    zdom = _rows(query_zdom, name="query z_dom")
    if len(zid) != len(zdom):
        raise ADV3B02StateError("four-arm query z_id/z_dom row count drift")
    # Preserve the parent implementation byte-for-byte for its ordinary case.
    if not np.any(zero_mask):
        m0 = qknn_logits(raw, zid)
        mda = dual_qknn_logits(dual, zid, zdom)
        other = bcrr_fused_logits(m0, zid, other_bcrr, bank=raw)
        joint = bcrr_fused_logits(mda, zid, joint_bcrr, bank=dual.id_bank)
        return {"M0": m0, "M_DA": mda, "M_OTHER": other, "M_JOINT": joint}
    if raw.classes != dual.id_bank.classes:
        raise ADV3B02StateError("zero-query four-arm class-axis drift")
    row_count, class_count = len(zid), len(raw.classes)
    result = {
        arm: np.empty((row_count, class_count), np.float32)
        for arm in ARMS
    }
    normal_mask = ~zero_mask
    if np.any(normal_mask):
        normal_zid = zid[normal_mask]
        normal_zdom = zdom[normal_mask]
        m0 = qknn_logits(raw, normal_zid)
        mda = dual_qknn_logits(dual, normal_zid, normal_zdom)
        other = bcrr_fused_logits(m0, normal_zid, other_bcrr, bank=raw)
        joint = bcrr_fused_logits(mda, normal_zid, joint_bcrr, bank=dual.id_bank)
        for arm, scores in (("M0", m0), ("M_DA", mda), ("M_OTHER", other), ("M_JOINT", joint)):
            result[arm][normal_mask] = scores
    zero_scores = _zero_query_analytic_logits(raw)
    for arm in ARMS:
        # Every arm receives an independent byte-for-byte copy of the same
        # M0 analytic extension; no z_dom or BCRR path is evaluated here.
        result[arm][zero_mask] = zero_scores
    if any(not np.array_equal(result["M0"][zero_mask], result[arm][zero_mask]) for arm in ARMS[1:]):
        raise ADV3B02StateError("zero-query four-arm equality drift")
    return {arm: np.ascontiguousarray(result[arm]) for arm in ARMS}


def predict_four_arms_with_predictions(
    states: Mapping[str, Any], *, query_zid: np.ndarray, query_zdom: np.ndarray
) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray], Mapping[str, Any]]:
    """Return scores plus qzero-aware class-axis decisions and audit-only counts."""
    zid, zero_mask = _query_zid_exact_zero_mask(query_zid)
    logits = predict_four_arms(states, query_zid=zid, query_zdom=query_zdom)
    raw = states["M0"]
    if not isinstance(raw, Int8QKNNState):
        raise ADV3B02StateError("zero-query prediction raw-state drift")
    predictions = {
        arm: np.asarray(np.argmax(np.asarray(logits[arm], np.float32), axis=1), np.intp)
        for arm in ARMS
    }
    exact_tie_count = 0
    if np.any(zero_mask):
        for row_index in np.flatnonzero(zero_mask):
            selected, was_tie = _zero_row_argmax(raw, logits["M0"][row_index])
            exact_tie_count += int(was_tie)
            for arm in ARMS:
                predictions[arm][row_index] = selected
        if any(
            not np.array_equal(logits["M0"][zero_mask], logits[arm][zero_mask])
            or not np.array_equal(predictions["M0"][zero_mask], predictions[arm][zero_mask])
            for arm in ARMS[1:]
        ):
            raise ADV3B02StateError("zero-query prediction all-arm equality drift")
    count = int(np.sum(zero_mask))
    return logits, predictions, {
        "query_zid_exact_zero_count": count,
        "query_zid_exact_zero_rate": float(count / len(zid)),
        "query_zid_exact_tie_count": int(exact_tie_count),
        "zero_rows_all_arms_equal": True,
    }


def resource_formula(*, class_count: int, k_shot: int) -> dict[str, int]:
    """Frozen head-only MAC accounting; rank-2 gives 840 at C=26,K=10."""
    if type(class_count) is not int or class_count < 2 or type(k_shot) is not int or k_shot not in K_VALUES:
        raise ADV3B02StateError("resource formula requires a formal class/K slice")
    domain_extra = RANK * Z_DIM + RANK * class_count * k_shot
    return {"rank": RANK, "id_qknn_kernel_mac_per_query": class_count * k_shot * Z_DIM,
            "dual_domain_qknn_extra_mac_per_query": domain_extra,
            "production_joint_head_mac_per_query": class_count * k_shot * Z_DIM + domain_extra + class_count}


def state_receipt(states: Mapping[str, Any]) -> dict[str, Any]:
    """Return byte-sharing, INT8, and branch-local BCRR evidence without predictions."""
    raw = states["M0"]; dual = states["M_DA"]; other_bank, other = states["M_OTHER"]; joint_dual, joint = states["M_JOINT"]
    if other_bank is not raw or joint_dual is not dual:
        raise ADV3B02StateError("state receipt requires exact paired-state sharing")
    bcr_bytes = (
        raw.branch_state.bcr_weight_codes_qint8.nbytes
        + raw.branch_state.bcr_weight_scales_fp16.nbytes
        + raw.branch_state.bcr_weight_residual_codes_qint8.nbytes
        + raw.branch_state.bcr_weight_residual_scales_fp16.nbytes
        + raw.branch_state.bcr_weight_residual2_codes_qint8.nbytes
        + raw.branch_state.bcr_weight_residual2_scales_fp16.nbytes
    )
    support_residual_bytes = (
        raw.bank.residual_codes_qint8.nbytes
        + raw.bank.residual_scales_fp16.nbytes
        + raw.bank.residual2_codes_qint8.nbytes
        + raw.bank.residual2_scales_fp16.nbytes
    )
    class_bandwidth_bytes = (
        raw.bank.class_scale_hi_fp16.nbytes
        + raw.bank.class_scale_lo_fp16.nbytes
    )
    support_codec_extra_bytes = (
        support_residual_bytes + raw.bank.class_scale_lo_fp16.nbytes
    )
    raw_state_bytes = len(raw.wire_bytes()) + bcr_bytes
    dual_domain_bytes = len(dual.domain.wire_bytes())
    branch_binding = getattr(
        raw.branch_state, "actual_bank_binding_receipt", None
    )
    repair_receipt = raw.branch_state.support_repair_receipt
    payload = {"candidate": CANDIDATE, "schema": SCHEMA, "arms": list(ARMS), "query_rows_used_for_fit": 0,
               "raw_qknn_sha256": raw.digest, "dual_qknn_sha256": dual.digest,
               "branch_qknn_wire_sha256": sha256_bytes(raw.branch_state.qknn_wire),
               "branch_actual_bank_binding_sha256": (
                   None
                   if branch_binding is None
                   else branch_binding["receipt_sha256"]
               ),
               "branch_teacher_support_sha256": branch_binding["teacher_support_sha256"],
               "branch_support_repair_receipt_sha256": branch_binding["support_repair_receipt_sha256"],
               "support_repair_receipt": dict(repair_receipt),
               "M0_M_OTHER_raw_state_byte_shared": True, "M_DA_M_JOINT_dual_state_byte_shared": True,
               "raw_bcrr": dict(other.receipt), "dual_bcrr": dict(joint.receipt),
               "int8": dict(dual.int8_audit_receipt),
               "int8_audit_sha256": sha256_bytes(
                   _canon(dict(dual.int8_audit_receipt))
               ),
               "raw_state_bytes": raw_state_bytes,
               "bcr_weight_wire_bytes": bcr_bytes,
               "bcr_weight_codec": BCR_WEIGHT_CODEC,
               "bcr_weight_plane_count": 3,
               "support_codec": SUPPORT_CODEC,
               "support_plane_count": 3,
               "support_residual_wire_bytes": support_residual_bytes,
               "class_bandwidth_codec": CLASS_BANDWIDTH_CODEC,
               "class_bandwidth_wire_bytes": class_bandwidth_bytes,
               "support_codec_extra_state_bytes": support_codec_extra_bytes,
               "dual_domain_state_bytes": dual_domain_bytes,
               "wire_bytes": raw_state_bytes + dual_domain_bytes}
    if payload["wire_bytes"] > MAX_WIRE_BYTES:
        raise ADV3B02StateError("frozen state exceeds 256KiB hard limit")
    return payload


def head_bypass_forward(model: Any, received_iq: Any, *, checkpoint_sha256: str) -> tuple[Any, Any, Mapping[str, Any]]:
    """Use the exact dual-feature bypass and seal evidence that forbidden heads were not called."""
    if type(checkpoint_sha256) is not str or len(checkpoint_sha256) != 64:
        raise ADV3B02StateError("checkpoint SHA must be a 64-character binding")
    from cvsrffi.dual_feature_forward import DOM_FEATURE_KEY, ID_FEATURE_KEY, dual_feature_forward
    z_id, z_dom, _tx_logits = dual_feature_forward(model, received_iq)
    if int(z_id.shape[1]) != Z_DIM or int(z_dom.shape[1]) != Z_DIM:
        raise ADV3B02StateError("head-bypass feature dimension drift")
    return z_id, z_dom, {"schema": SCHEMA, "checkpoint_sha256": checkpoint_sha256,
                         "feature_keys": {"z_id": ID_FEATURE_KEY, "z_dom": DOM_FEATURE_KEY},
                         "feature_dims": {"z_id": Z_DIM, "z_dom": Z_DIM}, "heads_called": 0,
                         "query_rows_used_for_fit": 0}


__all__ = ["ACTUAL_BRANCH_SCHEMA", "ADV3B02StateError", "APPEND_RECEIPT_SCHEMA", "ARMS", "ActualBankBranchState", "BCRRState", "CANDIDATE", "DualQKNNState", "DomainState",
           "Int8QKNNState", "K_VALUES", "MAX_WIRE_BYTES", "RANK", "SCENES", "Z_DIM",
           "append_stage2_c", "bcrr_fused_logits", "build_four_arm_states", "build_four_arm_states_from_dual", "build_int8_qknn_state",
           "build_stage2_b_state", "domain_weight_audit", "dual_qknn_logits", "fit_bcrr_branch", "head_bypass_forward", "int8_audit",
           "predict_four_arms", "predict_four_arms_with_predictions", "qknn_logits", "resource_formula", "sha256_bytes", "state_receipt", "typed_tokens",
           "verify_stage2_c_append_receipt"]
