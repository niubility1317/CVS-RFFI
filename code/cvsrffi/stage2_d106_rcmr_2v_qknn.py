"""D106 RCMR-2V qKNN head with sealed support-only state.

The module implements the frozen ``D106-RCMR-2V-qKNN/r1.1`` design.  It
never reads a ground bank, source rows, query truth, query role, a query batch
quota, or legacy Student-t scores.  The only persistent support representation
is two per-row-scaled INT8 banks plus one FP16 support reliability scalar per
row.  Pairwise support geometry exists only while building a state or inside
one ephemeral scoring context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Sequence

import numpy as np


CANDIDATE_ID = "D106-RCMR-2V-qKNN/r1.1"
Z_DIM = 160
MAX_REGISTERED_CLASSES = 26
MAX_SUPPORT_PER_CLASS = 10
MAX_SUPPORT_ROWS = MAX_REGISTERED_CLASSES * MAX_SUPPORT_PER_CLASS
MAX_REGISTRY_TOKEN_WIRE_BYTES = 64
MAX_ROW_ID_WIRE_BYTES = 64
MAX_CANONICAL_WIRE_BYTES = 90000
PROTOCOL_SCHEMA = "p2_min_v1"
VALIDATED_ONCE = "VALIDATED_ONCE"
METHOD_LOCK_SCHEMA = "cvs.phase2.d106.rcmr_2v_method_lock.v1"
BINDING_SCHEMA = "cvs.phase2.d106.rcmr_2v_binding.v1"
STATE_SCHEMA = "cvs.phase2.d106.rcmr_2v_state.v1"
WIRE_SCHEMA = "cvs.phase2.d106.rcmr_2v_wire.v1"
RESOURCE_RECEIPT_SCHEMA = "cvs.phase2.d106.rcmr_2v_resource_receipt.v1"
G0_RECEIPT_SCHEMA = "cvs.phase2.d106.rcmr_2v_g0_receipt.v1"
WIRE_MAGIC = b"CVSD106RCMR\x00\x01"

_STATE_TOKEN = object()
_CONTEXT_TOKEN = object()
_LOCK_TOKEN = object()


class D106RCMR2VError(ValueError):
    """Raised when a D106 RCMR-2V protocol or numerical invariant drifts."""


class D106RCMRCrossClassTieError(D106RCMR2VError):
    """Raised instead of resolving an exact maximum score by class order."""

    code = "CROSS_CLASS_SCORE_TIE"


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
    if type(value) is not str or len(value) != 64:
        raise D106RCMR2VError(f"{name} must be a lowercase SHA256")
    if any(character not in "0123456789abcdef" for character in value):
        raise D106RCMR2VError(f"{name} must be a lowercase SHA256")
    return value


def _require_text(value: Any, name: str, *, max_wire_bytes: int | None = None) -> str:
    if type(value) is not str or not value:
        raise D106RCMR2VError(f"{name} must be a non-empty exact string")
    if max_wire_bytes is not None:
        encoded = json.dumps(
            value, ensure_ascii=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if len(encoded) > max_wire_bytes:
            raise D106RCMR2VError(f"{name} exceeds the sealed wire-token limit")
    return value


def _immutable_array_root(value: np.ndarray) -> Any:
    root: Any = value
    while isinstance(root, np.ndarray) and root.base is not None:
        root = root.base
    return root


def _is_immutable_bytes_backed(value: np.ndarray) -> bool:
    root = _immutable_array_root(value)
    return isinstance(root, bytes) or (
        isinstance(root, memoryview) and root.readonly
    )


def _freeze_array(value: np.ndarray, dtype: np.dtype[Any], name: str) -> np.ndarray:
    """Freeze one C-contiguous array in immutable bytes without a second ndarray copy."""

    if not isinstance(value, np.ndarray) or value.dtype != dtype or not value.flags.c_contiguous:
        raise D106RCMR2VError(f"{name} has a non-canonical array layout")
    if _is_immutable_bytes_backed(value):
        return value
    payload = value.tobytes(order="C")
    return np.frombuffer(payload, dtype=dtype).reshape(value.shape)


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _array_digest(value: np.ndarray) -> str:
    return _sha256_bytes(_canonical_bytes(_array_receipt(value)))


def _regular_file_bytes(path: str | Path, name: str) -> tuple[Path, bytes, str]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise D106RCMR2VError(f"{name} must be an ordinary non-symlink file")
    raw = source.read_bytes()
    return source, raw, _sha256_bytes(raw)


def _binary64_bits(value: float) -> bytes:
    return struct.pack(">d", 0.0 if value == 0.0 else value)


def _same_binary64(left: float, right: float) -> bool:
    return _binary64_bits(float(left)) == _binary64_bits(float(right))


def _finite_l2_normalized_vector(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise D106RCMR2VError(f"{name} must be an exact float32 numpy vector")
    if value.shape != (Z_DIM,) or not np.isfinite(value).all():
        raise D106RCMR2VError(f"{name} must be finite float32 [{Z_DIM}]")
    result = np.empty(Z_DIM, dtype=np.float64)
    squared_norm = 0.0
    for coordinate in range(Z_DIM):
        number = float(value[coordinate])
        result[coordinate] = number
        squared_norm += number * number
    if not math.isfinite(squared_norm) or squared_norm <= 0.0:
        raise D106RCMR2VError(f"{name} contains a zero-norm row")
    norm = math.sqrt(squared_norm)
    for coordinate in range(Z_DIM):
        result[coordinate] /= norm
    return np.ascontiguousarray(result, dtype=np.float64)


def _finite_l2_normalized_rows(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise D106RCMR2VError(f"{name} must be an exact float32 numpy matrix")
    if (
        value.ndim != 2
        or value.shape[0] < 2
        or value.shape[1] != Z_DIM
        or not np.isfinite(value).all()
    ):
        raise D106RCMR2VError(f"{name} must be finite float32 [N,{Z_DIM}], N>=2")
    result = np.empty(value.shape, dtype=np.float64)
    for row_index in range(len(value)):
        result[row_index] = _finite_l2_normalized_vector(
            value[row_index], f"{name}[{row_index}]"
        )
    return np.ascontiguousarray(result, dtype=np.float64)


def _typed_tokens(value: Any, name: str, rows: int) -> tuple[str, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or len(value) != rows or value.dtype.kind not in {"U", "S"}:
            raise D106RCMR2VError(f"{name} must be a typed one-dimensional token array")
        raw = value.tolist()
    elif isinstance(value, (tuple, list)):
        if len(value) != rows:
            raise D106RCMR2VError(f"{name} length must align to support rows")
        raw = list(value)
    else:
        raise D106RCMR2VError(f"{name} must be an exact token sequence")
    tokens: list[str] = []
    for item in raw:
        if isinstance(item, bytes):
            try:
                item = item.decode("utf-8")
            except UnicodeDecodeError as error:
                raise D106RCMR2VError(f"{name} byte token is not UTF-8") from error
        if type(item) is not str or not item:
            raise D106RCMR2VError(f"{name} contains a blank or non-string token")
        tokens.append(item)
    return tuple(tokens)


def _registry_tokens(value: Sequence[str]) -> tuple[str, ...]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) < 2
        or len(value) > MAX_REGISTERED_CLASSES
    ):
        raise D106RCMR2VError(
            f"registered_classes must contain 2..{MAX_REGISTERED_CLASSES} exact tokens"
        )
    registry = tuple(
        _require_text(
            item,
            "registered class",
            max_wire_bytes=MAX_REGISTRY_TOKEN_WIRE_BYTES,
        )
        for item in value
    )
    if len(set(registry)) != len(registry):
        raise D106RCMR2VError("registered_classes must be unique")
    return registry


def _registry_root(registry: tuple[str, ...]) -> str:
    return _sha256_bytes(_canonical_bytes(list(registry)))


def _support_physical_root(physical_ids: tuple[str, ...]) -> str:
    return _sha256_bytes(_canonical_bytes(sorted(physical_ids)))


def _dot_distance(left: np.ndarray, right: np.ndarray) -> float:
    total = 0.0
    for coordinate in range(Z_DIM):
        total += float(left[coordinate]) * float(right[coordinate])
    distance = 1.0 - total
    if not math.isfinite(distance):
        raise D106RCMR2VError("FP64 cosine distance became non-finite")
    if distance <= 0.0:
        return 0.0
    if distance >= 2.0:
        return 2.0
    return float(distance)


def _quantize_rows(rows: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    codes = np.empty(rows.shape, dtype=np.int8)
    scales = np.empty(len(rows), dtype=np.dtype("<f2"))
    for row_index, row in enumerate(rows):
        scale = float(np.max(np.abs(row))) / 127.0
        encoded_scale = np.float16(scale)
        if not math.isfinite(float(encoded_scale)) or float(encoded_scale) <= 0.0:
            raise D106RCMR2VError(f"{name}[{row_index}] has an invalid FP16 scale")
        quantized = np.rint(row / float(encoded_scale))
        quantized = np.clip(quantized, -127, 127).astype(np.int8, copy=False)
        if not np.any(quantized):
            raise D106RCMR2VError(f"{name}[{row_index}] quantized to a zero vector")
        codes[row_index] = quantized
        scales[row_index] = encoded_scale
    return np.ascontiguousarray(codes), np.ascontiguousarray(scales)


def _decode_rows(codes: np.ndarray, scales: np.ndarray, name: str) -> np.ndarray:
    if (
        codes.dtype != np.dtype(np.int8)
        or codes.ndim != 2
        or codes.shape[1] != Z_DIM
        or scales.dtype != np.dtype("<f2")
        or scales.shape != (len(codes),)
    ):
        raise D106RCMR2VError(f"{name} quantized support layout drift")
    decoded = np.empty(codes.shape, dtype=np.float64)
    for row_index in range(len(codes)):
        scale = float(scales[row_index])
        if not math.isfinite(scale) or scale <= 0.0:
            raise D106RCMR2VError(f"{name}[{row_index}] scale drift")
        raw = np.ascontiguousarray(codes[row_index], dtype=np.float64) * scale
        squared_norm = 0.0
        for coordinate in range(Z_DIM):
            squared_norm += float(raw[coordinate]) * float(raw[coordinate])
        if not math.isfinite(squared_norm) or squared_norm <= 0.0:
            raise D106RCMR2VError(f"{name}[{row_index}] decoded to a zero vector")
        norm = math.sqrt(squared_norm)
        for coordinate in range(Z_DIM):
            decoded[row_index, coordinate] = float(raw[coordinate]) / norm
    return np.ascontiguousarray(decoded, dtype=np.float64)


def _pairwise_distance_matrix(rows: np.ndarray) -> np.ndarray:
    count = len(rows)
    matrix = np.zeros((count, count), dtype=np.float64)
    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            distance = _dot_distance(rows[left_index], rows[right_index])
            matrix[left_index, right_index] = distance
            matrix[right_index, left_index] = distance
    return matrix


def _midranks(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.float64 or values.ndim != 1 or len(values) < 1:
        raise D106RCMR2VError("mid-rank input must be a non-empty FP64 vector")
    if not np.isfinite(values).all():
        raise D106RCMR2VError("mid-rank input contains a non-finite value")
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        exemplar = float(values[order[start]])
        end = start + 1
        while end < len(values) and _same_binary64(float(values[order[end]]), exemplar):
            end += 1
        rank = (start + 0.5 * (end - start) + 0.5) / float(len(values) + 1)
        for position in range(start, end):
            result[order[position]] = rank
        start = end
    return result


def _profiles(matrix: np.ndarray) -> np.ndarray:
    if matrix.dtype != np.float64 or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise D106RCMR2VError("support distance matrix layout drift")
    count = len(matrix)
    output = np.empty((count, count - 1), dtype=np.float64)
    for row_index in range(count):
        values = np.empty(count - 1, dtype=np.float64)
        if row_index:
            values[:row_index] = matrix[row_index, :row_index]
        if row_index + 1 < count:
            values[row_index:] = matrix[row_index, row_index + 1 :]
        values.sort(kind="stable")
        output[row_index] = values
    return output


def _midrank_from_profile(profile: np.ndarray, value: float) -> float:
    if profile.dtype != np.float64 or profile.ndim != 1 or len(profile) < 1:
        raise D106RCMR2VError("support profile layout drift")
    if not math.isfinite(value):
        raise D106RCMR2VError("profile query distance is non-finite")
    left = int(np.searchsorted(profile, value, side="left"))
    right = int(np.searchsorted(profile, value, side="right"))
    return (left + 0.5 * (right - left) + 0.5) / float(len(profile) + 1)


def _support_reliability(plus_rows: np.ndarray, signed_rows: np.ndarray) -> np.ndarray:
    """Fit R_i with row-local pair work only; never retain an NxN builder array."""

    count = len(plus_rows)
    result = np.empty(count, dtype=np.dtype("<f2"))
    for row_index in range(count):
        plus_values = np.empty(count - 1, dtype=np.float64)
        signed_values = np.empty(count - 1, dtype=np.float64)
        position = 0
        for other_index in range(count):
            if other_index == row_index:
                continue
            plus_values[position] = _dot_distance(
                plus_rows[row_index], plus_rows[other_index]
            )
            signed_values[position] = _dot_distance(
                signed_rows[row_index], signed_rows[other_index]
            )
            position += 1
        difference = float(np.mean(np.abs(_midranks(plus_values) - _midranks(signed_values))))
        reliability = math.exp(-difference)
        stored = np.float16(reliability)
        if not math.isfinite(float(stored)) or float(stored) <= 0.0 or float(stored) > 1.0:
            raise D106RCMR2VError("support reliability FP16 closure drift")
        result[row_index] = stored
    return np.ascontiguousarray(result)


@dataclass(frozen=True, slots=True)
class D106RCMR2VBinding:
    """Opaque-row identity the RCMR state must bind before serialization."""

    capsule_id: str
    split_id: str
    validator_receipt_sha256: str
    support_physical_root_sha256: str
    row_id: str
    seed: int
    active_k: int
    da_receipt_sha256: str
    paired_view_receipt_sha256: str
    protocol_schema: str = PROTOCOL_SCHEMA
    phase2_data_status: str = VALIDATED_ONCE
    support_query_disjoint: bool = True
    schema: str = BINDING_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != BINDING_SCHEMA
            or self.protocol_schema != PROTOCOL_SCHEMA
            or self.phase2_data_status != VALIDATED_ONCE
            or self.support_query_disjoint is not True
            or type(self.seed) is not int
            or self.seed < 0
            or type(self.active_k) is not int
            or not 1 <= self.active_k <= MAX_SUPPORT_PER_CLASS
        ):
            raise D106RCMR2VError("RCMR binding lifecycle drift")
        for name in (
            "capsule_id",
            "split_id",
            "validator_receipt_sha256",
            "support_physical_root_sha256",
            "da_receipt_sha256",
            "paired_view_receipt_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        object.__setattr__(
            self,
            "row_id",
            _require_text(
                self.row_id, "row_id", max_wire_bytes=MAX_ROW_ID_WIRE_BYTES
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "split_id": self.split_id,
            "validator_receipt_sha256": self.validator_receipt_sha256,
            "support_physical_root_sha256": self.support_physical_root_sha256,
            "row_id": self.row_id,
            "seed": self.seed,
            "active_k": self.active_k,
            "da_receipt_sha256": self.da_receipt_sha256,
            "paired_view_receipt_sha256": self.paired_view_receipt_sha256,
            "protocol_schema": self.protocol_schema,
            "phase2_data_status": self.phase2_data_status,
            "support_query_disjoint": self.support_query_disjoint,
        }


@dataclass(frozen=True, slots=True)
class _D106RCMR2VMethodLock:
    document_sha256: str
    _loader_token: object | None = field(default=None, repr=False, compare=False)

    @property
    def is_loader_authorized(self) -> bool:
        return self._loader_token is _LOCK_TOKEN


def _validate_method_lock_document(document: dict[str, Any]) -> None:
    expected = {
        "candidate_id",
        "class_aggregation",
        "cross_class_tie",
        "dimension",
        "distance",
        "ephemeral_profiles",
        "fallback",
        "free_numeric_hyperparameters",
        "g0_argmax_change_required",
        "g0_parameter_scan_forbidden",
        "ground_head_access",
        "max_canonical_wire_bytes",
        "max_registered_classes",
        "max_registry_token_wire_bytes",
        "max_row_id_wire_bytes",
        "max_support_per_class",
        "mid_rank",
        "persistent_dense_nxn",
        "protocol_schema",
        "query_batch_count_access",
        "query_class_quota_access",
        "query_fit",
        "query_policy",
        "query_role_access",
        "query_selection",
        "query_truth_access",
        "query_update",
        "quantization",
        "schema",
        "source_runtime_access",
        "support_reliability",
        "views",
    }
    if set(document) != expected:
        raise D106RCMR2VError("RCMR method lock key closure drift")
    if (
        document["schema"] != METHOD_LOCK_SCHEMA
        or document["candidate_id"] != CANDIDATE_ID
        or document["protocol_schema"] != PROTOCOL_SCHEMA
        or document["dimension"] != Z_DIM
        or document["views"] != ["relu_pre_relu", "signed_pre_relu"]
        or document["quantization"] != "int8_per_row_fp16_scale_round_ties_to_even_127"
        or document["distance"] != "fp64_cosine_clipped_0_2"
        or document["mid_rank"] != "binary64_exact_half_offset"
        or document["support_reliability"] != "exp_negative_mean_abs_cross_view_rank"
        or document["class_aggregation"] != "equal_k_mean"
        or document["query_policy"] != "per_sample_all_registered_classes"
        or document["fallback"] != "forbidden"
        or document["cross_class_tie"] != "fail_closed"
        or document["ephemeral_profiles"] != "fp64_once_per_context"
        or document["free_numeric_hyperparameters"] != 0
        or document["g0_argmax_change_required"] is not True
        or document["g0_parameter_scan_forbidden"] is not True
        or document["persistent_dense_nxn"] is not False
        or document["max_canonical_wire_bytes"] != MAX_CANONICAL_WIRE_BYTES
        or document["max_registered_classes"] != MAX_REGISTERED_CLASSES
        or document["max_registry_token_wire_bytes"] != MAX_REGISTRY_TOKEN_WIRE_BYTES
        or document["max_row_id_wire_bytes"] != MAX_ROW_ID_WIRE_BYTES
        or document["max_support_per_class"] != MAX_SUPPORT_PER_CLASS
        or document["ground_head_access"] is not False
        or document["source_runtime_access"] is not False
        or document["query_batch_count_access"] is not False
        or document["query_class_quota_access"] is not False
        or document["query_fit"] is not False
        or document["query_role_access"] is not False
        or document["query_selection"] is not False
        or document["query_truth_access"] is not False
        or document["query_update"] is not False
    ):
        raise D106RCMR2VError("RCMR method lock semantic drift")


def load_d106_rcmr_2v_method_lock(
    path: str | Path, *, expected_sha256: str
) -> _D106RCMR2VMethodLock:
    """Load only the canonical, externally SHA-pinned r1.1 method lock."""

    expected = _require_sha256(expected_sha256, "expected method lock SHA256")
    _source, raw, actual = _regular_file_bytes(path, "RCMR method lock")
    if actual != expected:
        raise D106RCMR2VError("RCMR method lock external SHA256 mismatch")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106RCMR2VError("RCMR method lock must be canonical UTF-8 JSON") from error
    canonical = _canonical_bytes(document)
    # The SHA pins bytes, while a single terminal LF is permitted for a checked-in
    # JSON record.  All interior whitespace and every semantic field remain exact.
    if type(document) is not dict or raw not in {canonical, canonical + b"\n"}:
        raise D106RCMR2VError("RCMR method lock is not original canonical JSON")
    _validate_method_lock_document(document)
    return _D106RCMR2VMethodLock(
        document_sha256=expected,
        _loader_token=_LOCK_TOKEN,
    )


def _require_method_lock(value: Any) -> _D106RCMR2VMethodLock:
    if not isinstance(value, _D106RCMR2VMethodLock) or not value.is_loader_authorized:
        raise D106RCMR2VError("RCMR construction requires a strict-loader method lock")
    return value


def _require_exact_array(
    value: Any, dtype: np.dtype[Any], shape: tuple[int, ...], name: str
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape:
        raise D106RCMR2VError(f"{name} has a non-canonical array layout")
    if not value.flags.c_contiguous:
        raise D106RCMR2VError(f"{name} must be C-contiguous")
    return value


def _require_immutable_exact_array(
    value: Any, dtype: np.dtype[Any], shape: tuple[int, ...], name: str
) -> np.ndarray:
    array = _require_exact_array(value, dtype, shape, name)
    if array.flags.writeable or not _is_immutable_bytes_backed(array):
        raise D106RCMR2VError(f"{name} must be backed by immutable bytes")
    return array


def _state_payload_from_fields(
    *,
    binding: D106RCMR2VBinding,
    method_lock_sha256: str,
    registry: tuple[str, ...],
    codes_plus: np.ndarray,
    codes_signed: np.ndarray,
    scales_plus: np.ndarray,
    scales_signed: np.ndarray,
    reliabilities: np.ndarray,
    class_index: np.ndarray,
) -> dict[str, Any]:
    count = len(class_index)
    return {
        "C": len(registry),
        "D": Z_DIM,
        "K": binding.active_k,
        "N": count,
        "binding": binding.as_dict(),
        "candidate_id": CANDIDATE_ID,
        "class_index": _array_receipt(class_index),
        "codes_plus": _array_receipt(codes_plus),
        "codes_signed": _array_receipt(codes_signed),
        "method_lock_sha256": method_lock_sha256,
        "persistent_dense_nxn_bytes": 0,
        "query_state_updates": 0,
        "raw_support_rows_retained": False,
        "registry": list(registry),
        "registry_root_sha256": _registry_root(registry),
        "reliabilities": _array_receipt(reliabilities),
        "scales_plus": _array_receipt(scales_plus),
        "scales_signed": _array_receipt(scales_signed),
        "schema": STATE_SCHEMA,
    }


def _numeric_state_bytes(count: int) -> int:
    return 2 * count * Z_DIM + 7 * count


@dataclass(frozen=True, slots=True)
class D106RCMR2VState:
    """Formal immutable support state; raw support rows are never retained."""

    binding: D106RCMR2VBinding
    method_lock_sha256: str
    registry: tuple[str, ...]
    codes_plus: np.ndarray
    codes_signed: np.ndarray
    scales_plus: np.ndarray
    scales_signed: np.ndarray
    reliabilities: np.ndarray
    class_index: np.ndarray
    state_receipt_sha256: str
    schema: str = STATE_SCHEMA
    candidate_id: str = CANDIDATE_ID
    raw_support_rows_retained: bool = False
    persistent_dense_nxn_bytes: int = 0
    query_state_updates: int = 0
    _formal_token: object | None = field(default=None, repr=False, compare=False)
    _context_binding_token: object | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.binding, D106RCMR2VBinding)
            or self.schema != STATE_SCHEMA
            or self.candidate_id != CANDIDATE_ID
            or self.raw_support_rows_retained is not False
            or self.persistent_dense_nxn_bytes != 0
            or self.query_state_updates != 0
        ):
            raise D106RCMR2VError("RCMR formal state metadata drift")
        method_lock_sha256 = _require_sha256(
            self.method_lock_sha256, "RCMR method lock SHA256"
        )
        registry = _registry_tokens(list(self.registry))
        count = len(self.class_index) if isinstance(self.class_index, np.ndarray) else -1
        if (
            count < 2
            or count > MAX_SUPPORT_ROWS
            or count != len(registry) * self.binding.active_k
        ):
            raise D106RCMR2VError("RCMR state C/K/N closure drift")

        codes_plus = _require_immutable_exact_array(
            self.codes_plus, np.dtype(np.int8), (count, Z_DIM), "codes_plus"
        )
        codes_signed = _require_immutable_exact_array(
            self.codes_signed, np.dtype(np.int8), (count, Z_DIM), "codes_signed"
        )
        scales_plus = _require_immutable_exact_array(
            self.scales_plus, np.dtype("<f2"), (count,), "scales_plus"
        )
        scales_signed = _require_immutable_exact_array(
            self.scales_signed, np.dtype("<f2"), (count,), "scales_signed"
        )
        reliabilities = _require_immutable_exact_array(
            self.reliabilities, np.dtype("<f2"), (count,), "reliabilities"
        )
        class_index = _require_immutable_exact_array(
            self.class_index, np.dtype(np.uint8), (count,), "class_index"
        )
        if (
            np.any(codes_plus == -128)
            or np.any(codes_signed == -128)
            or np.any(np.all(codes_plus == 0, axis=1))
            or np.any(np.all(codes_signed == 0, axis=1))
            or not np.isfinite(scales_plus).all()
            or not np.isfinite(scales_signed).all()
            or not np.isfinite(reliabilities).all()
            or np.any(scales_plus <= 0.0)
            or np.any(scales_signed <= 0.0)
            or np.any(reliabilities <= 0.0)
            or np.any(reliabilities > 1.0)
            or np.any(class_index >= len(registry))
        ):
            raise D106RCMR2VError("RCMR quantized state contains an invalid value")
        counts = np.bincount(class_index.astype(np.int64), minlength=len(registry))
        if counts.shape != (len(registry),) or not np.all(counts == self.binding.active_k):
            raise D106RCMR2VError("RCMR state does not contain exactly K support rows/class")

        object.__setattr__(self, "method_lock_sha256", method_lock_sha256)
        object.__setattr__(self, "registry", registry)
        if self._formal_token is _STATE_TOKEN and self._context_binding_token is None:
            raise D106RCMR2VError("formal RCMR state lacks a prepare-binding token")

        payload = _state_payload_from_fields(
            binding=self.binding,
            method_lock_sha256=method_lock_sha256,
            registry=registry,
            codes_plus=self.codes_plus,
            codes_signed=self.codes_signed,
            scales_plus=self.scales_plus,
            scales_signed=self.scales_signed,
            reliabilities=self.reliabilities,
            class_index=self.class_index,
        )
        expected_receipt = _sha256_bytes(_canonical_bytes(payload))
        if self.state_receipt_sha256 != expected_receipt:
            raise D106RCMR2VError("RCMR state receipt mismatch")

    @property
    def is_formal(self) -> bool:
        return self._formal_token is _STATE_TOKEN

    @property
    def registry_root_sha256(self) -> str:
        return _registry_root(self.registry)

    @property
    def payload(self) -> dict[str, Any]:
        return _state_payload_from_fields(
            binding=self.binding,
            method_lock_sha256=self.method_lock_sha256,
            registry=self.registry,
            codes_plus=self.codes_plus,
            codes_signed=self.codes_signed,
            scales_plus=self.scales_plus,
            scales_signed=self.scales_signed,
            reliabilities=self.reliabilities,
            class_index=self.class_index,
        )


def _require_formal_state(value: Any) -> D106RCMR2VState:
    if not isinstance(value, D106RCMR2VState) or not value.is_formal:
        raise D106RCMR2VError("RCMR operation requires strict-loader formal state")
    if value._context_binding_token is None:
        raise D106RCMR2VError("RCMR formal state prepare-binding token drift")
    return value


def build_d106_rcmr_2v_state(
    support_plus: np.ndarray,
    support_signed: np.ndarray,
    support_labels: Sequence[str] | np.ndarray,
    support_physical_ids: Sequence[str] | np.ndarray,
    registered_classes: Sequence[str],
    *,
    binding: D106RCMR2VBinding,
    method_lock: _D106RCMR2VMethodLock,
) -> D106RCMR2VState:
    """Seal exactly-K arm-mapped support rows into the permitted compact state."""

    lock = _require_method_lock(method_lock)
    if not isinstance(binding, D106RCMR2VBinding):
        raise D106RCMR2VError("RCMR state requires a validated binding")
    plus_normalized = _finite_l2_normalized_rows(support_plus, "support_plus")
    signed_normalized = _finite_l2_normalized_rows(support_signed, "support_signed")
    if plus_normalized.shape != signed_normalized.shape:
        raise D106RCMR2VError("two support views must contain the same rows")
    count = len(plus_normalized)
    registry = _registry_tokens(registered_classes)
    if count > MAX_SUPPORT_ROWS or count != len(registry) * binding.active_k:
        raise D106RCMR2VError("RCMR support C/K/N closure drift")
    labels = _typed_tokens(support_labels, "support_labels", count)
    physical_ids = _typed_tokens(support_physical_ids, "support_physical_ids", count)
    if len(set(physical_ids)) != count:
        raise D106RCMR2VError("support physical IDs must be unique")
    if _support_physical_root(physical_ids) != binding.support_physical_root_sha256:
        raise D106RCMR2VError("support physical-root receipt mismatch")

    class_by_name = {name: index for index, name in enumerate(registry)}
    try:
        unordered_class_index = np.asarray(
            [class_by_name[label] for label in labels], dtype=np.uint8
        )
    except KeyError as error:
        raise D106RCMR2VError("support label is outside the registered class registry") from error
    counts = np.bincount(unordered_class_index.astype(np.int64), minlength=len(registry))
    if counts.shape != (len(registry),) or not np.all(counts == binding.active_k):
        raise D106RCMR2VError("support labels are not exactly K per registered class")

    # Global slots are canonical physical-ID order, so equal-class summation never
    # depends on caller row order or on a class-ID tie-break.
    order = np.asarray(sorted(range(count), key=lambda index: physical_ids[index]), dtype=np.int64)
    # x_i is normalized before the frozen per-row INT8 quantizer.  The state
    # therefore cannot encode a caller's arbitrary amplitude convention.
    plus_input = np.ascontiguousarray(plus_normalized[order], dtype=np.float32)
    signed_input = np.ascontiguousarray(signed_normalized[order], dtype=np.float32)
    class_index = np.ascontiguousarray(unordered_class_index[order], dtype=np.uint8)
    codes_plus, scales_plus = _quantize_rows(plus_input, "support_plus")
    codes_signed, scales_signed = _quantize_rows(signed_input, "support_signed")
    decoded_plus = _decode_rows(codes_plus, scales_plus, "support_plus")
    decoded_signed = _decode_rows(codes_signed, scales_signed, "support_signed")
    reliabilities = _support_reliability(decoded_plus, decoded_signed)
    del decoded_plus, decoded_signed
    # Each persistent array receives exactly one immutable bytes backing before
    # state construction; __post_init__ validates but never duplicates it.
    codes_plus = _freeze_array(codes_plus, np.dtype(np.int8), "codes_plus")
    codes_signed = _freeze_array(codes_signed, np.dtype(np.int8), "codes_signed")
    scales_plus = _freeze_array(scales_plus, np.dtype("<f2"), "scales_plus")
    scales_signed = _freeze_array(scales_signed, np.dtype("<f2"), "scales_signed")
    reliabilities = _freeze_array(reliabilities, np.dtype("<f2"), "reliabilities")
    class_index = _freeze_array(class_index, np.dtype(np.uint8), "class_index")
    payload = _state_payload_from_fields(
        binding=binding,
        method_lock_sha256=lock.document_sha256,
        registry=registry,
        codes_plus=codes_plus,
        codes_signed=codes_signed,
        scales_plus=scales_plus,
        scales_signed=scales_signed,
        reliabilities=reliabilities,
        class_index=class_index,
    )
    return D106RCMR2VState(
        binding=binding,
        method_lock_sha256=lock.document_sha256,
        registry=registry,
        codes_plus=codes_plus,
        codes_signed=codes_signed,
        scales_plus=scales_plus,
        scales_signed=scales_signed,
        reliabilities=reliabilities,
        class_index=class_index,
        state_receipt_sha256=_sha256_bytes(_canonical_bytes(payload)),
        _formal_token=_STATE_TOKEN,
        _context_binding_token=object(),
    )


def _state_array_receipt(state: D106RCMR2VState) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "class_index": _array_receipt(state.class_index),
                "codes_plus": _array_receipt(state.codes_plus),
                "codes_signed": _array_receipt(state.codes_signed),
                "reliabilities": _array_receipt(state.reliabilities),
                "scales_plus": _array_receipt(state.scales_plus),
                "scales_signed": _array_receipt(state.scales_signed),
            }
        )
    )


def _context_payload(
    *,
    state_receipt_sha256: str,
    state_array_receipt_sha256: str,
    support_count: int,
    class_count: int,
    active_k: int,
    dimension: int,
    decoded_plus: np.ndarray,
    decoded_signed: np.ndarray,
    profiles_plus: np.ndarray,
    profiles_signed: np.ndarray,
) -> dict[str, Any]:
    return {
        "C": class_count,
        "D": dimension,
        "K": active_k,
        "N": support_count,
        "decoded_plus": _array_receipt(decoded_plus),
        "decoded_signed": _array_receipt(decoded_signed),
        "profiles_plus": _array_receipt(profiles_plus),
        "profiles_signed": _array_receipt(profiles_signed),
        "state_array_receipt_sha256": state_array_receipt_sha256,
        "state_receipt_sha256": state_receipt_sha256,
    }


def _validate_sorted_profiles(profiles: np.ndarray, name: str) -> None:
    """Validate profiles with scalar scans, never a second Nx(N-1) temporary."""

    for row_index in range(len(profiles)):
        previous = -math.inf
        for column_index in range(profiles.shape[1]):
            value = float(profiles[row_index, column_index])
            if not math.isfinite(value) or value < previous:
                raise D106RCMR2VError(f"{name} numerical closure drift")
            previous = value


@dataclass(frozen=True, slots=True)
class _D106RCMR2VScoringContext:
    """Private, per-state ephemeral support geometry; it contains no query data."""

    state_receipt_sha256: str
    state_array_receipt_sha256: str
    support_count: int
    class_count: int
    active_k: int
    dimension: int
    decoded_plus: np.ndarray
    decoded_signed: np.ndarray
    profiles_plus: np.ndarray
    profiles_signed: np.ndarray
    context_receipt_sha256: str
    _context_token: object | None = field(default=None, repr=False, compare=False)
    _state_binding_token: object | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        state_receipt = _require_sha256(self.state_receipt_sha256, "context state receipt")
        state_arrays = _require_sha256(
            self.state_array_receipt_sha256, "context state-array receipt"
        )
        count = len(self.decoded_plus) if isinstance(self.decoded_plus, np.ndarray) else -1
        if (
            type(self.support_count) is not int
            or type(self.class_count) is not int
            or type(self.active_k) is not int
            or type(self.dimension) is not int
            or self.support_count != count
            or not 2 <= count <= MAX_SUPPORT_ROWS
            or not 2 <= self.class_count <= MAX_REGISTERED_CLASSES
            or not 1 <= self.active_k <= MAX_SUPPORT_PER_CLASS
            or self.class_count * self.active_k != count
            or self.dimension != Z_DIM
        ):
            raise D106RCMR2VError("RCMR context support count drift")
        decoded_plus = _require_immutable_exact_array(
            self.decoded_plus, np.dtype(np.float64), (count, Z_DIM), "context decoded_plus"
        )
        decoded_signed = _require_immutable_exact_array(
            self.decoded_signed, np.dtype(np.float64), (count, Z_DIM), "context decoded_signed"
        )
        profiles_plus = _require_immutable_exact_array(
            self.profiles_plus, np.dtype(np.float64), (count, count - 1), "context profiles_plus"
        )
        profiles_signed = _require_immutable_exact_array(
            self.profiles_signed, np.dtype(np.float64), (count, count - 1), "context profiles_signed"
        )
        if (
            not np.isfinite(decoded_plus).all()
            or not np.isfinite(decoded_signed).all()
        ):
            raise D106RCMR2VError("RCMR context numerical closure drift")
        _validate_sorted_profiles(profiles_plus, "context profiles_plus")
        _validate_sorted_profiles(profiles_signed, "context profiles_signed")
        if (
            self._context_token is not _CONTEXT_TOKEN
            or self._state_binding_token is None
        ):
            raise D106RCMR2VError("RCMR scoring context was not minted by prepare")
        payload = _context_payload(
            state_receipt_sha256=state_receipt,
            state_array_receipt_sha256=state_arrays,
            support_count=count,
            class_count=self.class_count,
            active_k=self.active_k,
            dimension=self.dimension,
            decoded_plus=decoded_plus,
            decoded_signed=decoded_signed,
            profiles_plus=profiles_plus,
            profiles_signed=profiles_signed,
        )
        if self.context_receipt_sha256 != _sha256_bytes(_canonical_bytes(payload)):
            raise D106RCMR2VError("RCMR scoring-context receipt mismatch")


def prepare_d106_rcmr_2v_scoring_context(
    state: D106RCMR2VState,
) -> _D106RCMR2VScoringContext:
    """Prepare both support-distance profiles exactly once for one sealed state."""

    formal_state = _require_formal_state(state)
    decoded_plus = _decode_rows(
        formal_state.codes_plus, formal_state.scales_plus, "context support_plus"
    )
    plus_matrix = _pairwise_distance_matrix(decoded_plus)
    profiles_plus = _profiles(plus_matrix)
    del plus_matrix
    decoded_plus = _freeze_array(
        decoded_plus, np.dtype(np.float64), "context decoded_plus"
    )
    profiles_plus = _freeze_array(
        profiles_plus, np.dtype(np.float64), "context profiles_plus"
    )
    decoded_signed = _decode_rows(
        formal_state.codes_signed, formal_state.scales_signed, "context support_signed"
    )
    signed_matrix = _pairwise_distance_matrix(decoded_signed)
    profiles_signed = _profiles(signed_matrix)
    del signed_matrix
    decoded_signed = _freeze_array(
        decoded_signed, np.dtype(np.float64), "context decoded_signed"
    )
    profiles_signed = _freeze_array(
        profiles_signed, np.dtype(np.float64), "context profiles_signed"
    )
    array_receipt = _state_array_receipt(formal_state)
    payload = _context_payload(
        state_receipt_sha256=formal_state.state_receipt_sha256,
        state_array_receipt_sha256=array_receipt,
        support_count=len(formal_state.class_index),
        class_count=len(formal_state.registry),
        active_k=formal_state.binding.active_k,
        dimension=Z_DIM,
        decoded_plus=decoded_plus,
        decoded_signed=decoded_signed,
        profiles_plus=profiles_plus,
        profiles_signed=profiles_signed,
    )
    return _D106RCMR2VScoringContext(
        state_receipt_sha256=formal_state.state_receipt_sha256,
        state_array_receipt_sha256=array_receipt,
        support_count=len(formal_state.class_index),
        class_count=len(formal_state.registry),
        active_k=formal_state.binding.active_k,
        dimension=Z_DIM,
        decoded_plus=decoded_plus,
        decoded_signed=decoded_signed,
        profiles_plus=profiles_plus,
        profiles_signed=profiles_signed,
        context_receipt_sha256=_sha256_bytes(_canonical_bytes(payload)),
        _context_token=_CONTEXT_TOKEN,
        _state_binding_token=formal_state._context_binding_token,
    )


def _checked_context(
    state: D106RCMR2VState, value: Any
) -> _D106RCMR2VScoringContext:
    if (
        not isinstance(value, _D106RCMR2VScoringContext)
        or value._context_token is not _CONTEXT_TOKEN
        or value.state_receipt_sha256 != state.state_receipt_sha256
        or value._state_binding_token is not state._context_binding_token
        or value.support_count != len(state.class_index)
        or value.class_count != len(state.registry)
        or value.active_k != state.binding.active_k
        or value.dimension != Z_DIM
    ):
        raise D106RCMR2VError("RCMR scoring context does not bind the sealed state")
    return value


@dataclass(frozen=True, slots=True)
class D106RCMR2VPrediction:
    """Per-query decision, with all class scores kept in registry order."""

    state_receipt_sha256: str
    registry: tuple[str, ...]
    scores: np.ndarray
    predicted_class_index: int
    predicted_class: str
    query_reliability: float

    def __post_init__(self) -> None:
        receipt = _require_sha256(self.state_receipt_sha256, "prediction state receipt")
        registry = _registry_tokens(list(self.registry))
        scores = _require_immutable_exact_array(
            self.scores, np.dtype(np.float64), (len(registry),), "prediction scores"
        )
        if (
            not np.isfinite(scores).all()
            or type(self.predicted_class_index) is not int
            or not 0 <= self.predicted_class_index < len(registry)
            or self.predicted_class != registry[self.predicted_class_index]
            or not math.isfinite(float(self.query_reliability))
            or not 0.0 < float(self.query_reliability) <= 1.0
        ):
            raise D106RCMR2VError("RCMR prediction closure drift")
        object.__setattr__(self, "state_receipt_sha256", receipt)
        object.__setattr__(self, "registry", registry)
        object.__setattr__(self, "query_reliability", float(self.query_reliability))


def score_d106_rcmr_2v_query(
    state: D106RCMR2VState,
    query_plus: np.ndarray,
    query_signed: np.ndarray,
    *,
    da_receipt_sha256: str,
    context: _D106RCMR2VScoringContext,
) -> D106RCMR2VPrediction:
    """Score one query against every class without updating state or context."""

    formal_state = _require_formal_state(state)
    if _require_sha256(da_receipt_sha256, "query DA receipt") != formal_state.binding.da_receipt_sha256:
        raise D106RCMR2VError("query arm-mapping DA receipt mismatch")
    scoring_context = _checked_context(formal_state, context)
    plus_query = _finite_l2_normalized_vector(query_plus, "query_plus")
    signed_query = _finite_l2_normalized_vector(query_signed, "query_signed")
    count = len(formal_state.class_index)
    distances_plus = np.empty(count, dtype=np.float64)
    distances_signed = np.empty(count, dtype=np.float64)
    for slot in range(count):
        distances_plus[slot] = _dot_distance(plus_query, scoring_context.decoded_plus[slot])
        distances_signed[slot] = _dot_distance(signed_query, scoring_context.decoded_signed[slot])
    alpha_plus = _midranks(distances_plus)
    alpha_signed = _midranks(distances_signed)
    beta_plus = np.empty(count, dtype=np.float64)
    beta_signed = np.empty(count, dtype=np.float64)
    for slot in range(count):
        beta_plus[slot] = _midrank_from_profile(
            scoring_context.profiles_plus[slot], float(distances_plus[slot])
        )
        beta_signed[slot] = _midrank_from_profile(
            scoring_context.profiles_signed[slot], float(distances_signed[slot])
        )
    query_reliability = math.exp(
        -float(np.mean(np.abs(alpha_plus - alpha_signed), dtype=np.float64))
    )
    if not math.isfinite(query_reliability) or not 0.0 < query_reliability <= 1.0:
        raise D106RCMR2VError("query reliability became invalid")
    support_reliability = formal_state.reliabilities.astype(np.float64, copy=False)
    weights = query_reliability * support_reliability
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise D106RCMR2VError("RCMR positive reliability closure drift")
    plus_match = (1.0 - alpha_plus) * (1.0 - beta_plus)
    signed_match = (1.0 - alpha_signed) * (1.0 - beta_signed)
    evidence = (plus_match + weights * signed_match) / (1.0 + weights)
    if not np.isfinite(evidence).all():
        raise D106RCMR2VError("RCMR query evidence became non-finite")
    scores = np.zeros(len(formal_state.registry), dtype=np.float64)
    # The loop is the only aggregation route: global slot order is fixed by
    # physical ID at sealing time and no class-order tie break exists below.
    for slot in range(count):
        scores[int(formal_state.class_index[slot])] += float(evidence[slot])
    scores /= float(formal_state.binding.active_k)
    if not np.isfinite(scores).all():
        raise D106RCMR2VError("RCMR class score became non-finite")
    maximum = max(float(score) for score in scores)
    winners = [
        index for index, score in enumerate(scores) if _same_binary64(float(score), maximum)
    ]
    if len(winners) != 1:
        raise D106RCMRCrossClassTieError(
            "CROSS_CLASS_SCORE_TIE: exact cross-class maximum cannot be resolved"
        )
    predicted_index = winners[0]
    return D106RCMR2VPrediction(
        state_receipt_sha256=formal_state.state_receipt_sha256,
        registry=formal_state.registry,
        scores=_freeze_array(scores, np.dtype(np.float64), "prediction scores"),
        predicted_class_index=predicted_index,
        predicted_class=formal_state.registry[predicted_index],
        query_reliability=query_reliability,
    )


def _wire_header(state: D106RCMR2VState) -> dict[str, Any]:
    return {
        "schema": WIRE_SCHEMA,
        "state": state.payload,
        "state_receipt_sha256": state.state_receipt_sha256,
    }


def _wire_array_bytes(state: D106RCMR2VState) -> bytes:
    return b"".join(
        array.tobytes(order="C")
        for array in (
            state.codes_plus,
            state.codes_signed,
            state.scales_plus,
            state.scales_signed,
            state.reliabilities,
            state.class_index,
        )
    )


def serialize_d106_rcmr_2v_state(state: D106RCMR2VState) -> bytes:
    """Emit the one canonical formal-state wire representation."""

    formal_state = _require_formal_state(state)
    header = _canonical_bytes(_wire_header(formal_state))
    if len(header) > 0xFFFFFFFF:
        raise D106RCMR2VError("RCMR wire header exceeds framing limit")
    body = _wire_array_bytes(formal_state)
    if len(body) != _numeric_state_bytes(len(formal_state.class_index)):
        raise D106RCMR2VError("RCMR wire numeric payload layout drift")
    wire = WIRE_MAGIC + struct.pack(">I", len(header)) + header + body
    if len(wire) > MAX_CANONICAL_WIRE_BYTES:
        raise D106RCMR2VError("RCMR canonical wire exceeds the sealed size cap")
    return wire


def _validate_receipt_document(
    value: Any, *, dtype: np.dtype[Any], shape: list[int], name: str
) -> None:
    if type(value) is not dict or set(value) != {"dtype", "shape", "sha256"}:
        raise D106RCMR2VError(f"{name} wire receipt key closure drift")
    if (
        value["dtype"] != dtype.str
        or value["shape"] != shape
        or _require_sha256(value["sha256"], f"{name} wire SHA256") != value["sha256"]
    ):
        raise D106RCMR2VError(f"{name} wire receipt layout drift")


def _binding_from_document(value: Any) -> D106RCMR2VBinding:
    if type(value) is not dict:
        raise D106RCMR2VError("RCMR wire binding must be an object")
    try:
        binding = D106RCMR2VBinding(**value)
    except (TypeError, ValueError) as error:
        raise D106RCMR2VError("RCMR wire binding is invalid") from error
    if binding.as_dict() != value:
        raise D106RCMR2VError("RCMR wire binding canonical closure drift")
    return binding


def _validate_wire_state_document(value: Any) -> tuple[D106RCMR2VBinding, tuple[str, ...], int]:
    expected = {
        "C",
        "D",
        "K",
        "N",
        "binding",
        "candidate_id",
        "class_index",
        "codes_plus",
        "codes_signed",
        "method_lock_sha256",
        "persistent_dense_nxn_bytes",
        "query_state_updates",
        "raw_support_rows_retained",
        "registry",
        "registry_root_sha256",
        "reliabilities",
        "scales_plus",
        "scales_signed",
        "schema",
    }
    if type(value) is not dict or set(value) != expected:
        raise D106RCMR2VError("RCMR wire state key closure drift")
    binding = _binding_from_document(value["binding"])
    registry = _registry_tokens(value["registry"])
    if (
        type(value["C"]) is not int
        or type(value["K"]) is not int
        or type(value["N"]) is not int
        or type(value["D"]) is not int
        or value["C"] != len(registry)
        or value["K"] != binding.active_k
        or value["N"] != len(registry) * binding.active_k
        or not 2 <= value["N"] <= MAX_SUPPORT_ROWS
        or value["D"] != Z_DIM
        or value["schema"] != STATE_SCHEMA
        or value["candidate_id"] != CANDIDATE_ID
        or value["registry_root_sha256"] != _registry_root(registry)
        or value["raw_support_rows_retained"] is not False
        or value["persistent_dense_nxn_bytes"] != 0
        or value["query_state_updates"] != 0
    ):
        raise D106RCMR2VError("RCMR wire state semantic drift")
    _require_sha256(value["method_lock_sha256"], "wire method lock SHA256")
    count = value["N"]
    _validate_receipt_document(
        value["codes_plus"], dtype=np.dtype(np.int8), shape=[count, Z_DIM], name="codes_plus"
    )
    _validate_receipt_document(
        value["codes_signed"], dtype=np.dtype(np.int8), shape=[count, Z_DIM], name="codes_signed"
    )
    _validate_receipt_document(
        value["scales_plus"], dtype=np.dtype("<f2"), shape=[count], name="scales_plus"
    )
    _validate_receipt_document(
        value["scales_signed"], dtype=np.dtype("<f2"), shape=[count], name="scales_signed"
    )
    _validate_receipt_document(
        value["reliabilities"], dtype=np.dtype("<f2"), shape=[count], name="reliabilities"
    )
    _validate_receipt_document(
        value["class_index"], dtype=np.dtype(np.uint8), shape=[count], name="class_index"
    )
    return binding, registry, count


def _read_wire_arrays(raw: bytes, offset: int, count: int) -> tuple[np.ndarray, ...]:
    layouts: tuple[tuple[np.dtype[Any], tuple[int, ...]], ...] = (
        (np.dtype(np.int8), (count, Z_DIM)),
        (np.dtype(np.int8), (count, Z_DIM)),
        (np.dtype("<f2"), (count,)),
        (np.dtype("<f2"), (count,)),
        (np.dtype("<f2"), (count,)),
        (np.dtype(np.uint8), (count,)),
    )
    arrays: list[np.ndarray] = []
    position = offset
    for dtype, shape in layouts:
        byte_count = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if position + byte_count > len(raw):
            raise D106RCMR2VError("RCMR wire is truncated")
        arrays.append(
            np.frombuffer(memoryview(raw)[position : position + byte_count], dtype=dtype)
            .reshape(shape)
        )
        position += byte_count
    if position != len(raw):
        raise D106RCMR2VError("RCMR wire has trailing bytes")
    return tuple(arrays)


def deserialize_d106_rcmr_2v_state(
    payload: bytes,
    *,
    expected_wire_sha256: str,
    expected_binding: D106RCMR2VBinding,
    method_lock: _D106RCMR2VMethodLock,
) -> D106RCMR2VState:
    """Strictly restore a SHA-pinned canonical state wire with no fallbacks."""

    lock = _require_method_lock(method_lock)
    expected_wire = _require_sha256(expected_wire_sha256, "expected RCMR wire SHA256")
    if not isinstance(expected_binding, D106RCMR2VBinding):
        raise D106RCMR2VError("strict wire load requires the expected binding")
    if (
        type(payload) is not bytes
        or len(payload) > MAX_CANONICAL_WIRE_BYTES
        or _sha256_bytes(payload) != expected_wire
    ):
        raise D106RCMR2VError("RCMR wire external SHA256 mismatch")
    prefix_size = len(WIRE_MAGIC) + 4
    if len(payload) <= prefix_size or payload[: len(WIRE_MAGIC)] != WIRE_MAGIC:
        raise D106RCMR2VError("RCMR wire magic mismatch")
    header_size = struct.unpack(">I", payload[len(WIRE_MAGIC) : prefix_size])[0]
    header_start = prefix_size
    header_end = header_start + header_size
    if header_size == 0 or header_end >= len(payload):
        raise D106RCMR2VError("RCMR wire header framing drift")
    header_bytes = payload[header_start:header_end]
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106RCMR2VError("RCMR wire header is not canonical UTF-8 JSON") from error
    if (
        type(header) is not dict
        or set(header) != {"schema", "state", "state_receipt_sha256"}
        or header_bytes != _canonical_bytes(header)
        or header["schema"] != WIRE_SCHEMA
    ):
        raise D106RCMR2VError("RCMR wire header canonical closure drift")
    state_receipt = _require_sha256(
        header["state_receipt_sha256"], "RCMR wire state receipt"
    )
    binding, registry, count = _validate_wire_state_document(header["state"])
    if binding.as_dict() != expected_binding.as_dict():
        raise D106RCMR2VError("RCMR wire binding does not match the requested row")
    if header["state"]["method_lock_sha256"] != lock.document_sha256:
        raise D106RCMR2VError("RCMR wire method lock does not match strict loader")
    arrays = _read_wire_arrays(payload, header_end, count)
    state = D106RCMR2VState(
        binding=binding,
        method_lock_sha256=lock.document_sha256,
        registry=registry,
        codes_plus=arrays[0],
        codes_signed=arrays[1],
        scales_plus=arrays[2],
        scales_signed=arrays[3],
        reliabilities=arrays[4],
        class_index=arrays[5],
        state_receipt_sha256=state_receipt,
        _formal_token=_STATE_TOKEN,
        _context_binding_token=object(),
    )
    if state.payload != header["state"] or _wire_header(state) != header:
        raise D106RCMR2VError("RCMR wire payload receipt does not reproduce exactly")
    return state


def load_d106_rcmr_2v_state(
    path: str | Path,
    *,
    expected_wire_sha256: str,
    expected_binding: D106RCMR2VBinding,
    method_lock: _D106RCMR2VMethodLock,
) -> D106RCMR2VState:
    """Load only an ordinary, SHA-pinned state file through the strict decoder."""

    _source, raw, actual = _regular_file_bytes(path, "RCMR state wire")
    expected = _require_sha256(expected_wire_sha256, "expected RCMR wire SHA256")
    if actual != expected:
        raise D106RCMR2VError("RCMR state-file SHA256 mismatch")
    return deserialize_d106_rcmr_2v_state(
        raw,
        expected_wire_sha256=expected,
        expected_binding=expected_binding,
        method_lock=method_lock,
    )


def judge_d106_rcmr_2v_g0(
    query_ids: Sequence[str] | np.ndarray,
    candidate_argmax: Sequence[str] | np.ndarray,
    baseline_argmax: Sequence[str] | np.ndarray,
    *,
    formal_tap_receipt_sha256: str,
) -> dict[str, Any]:
    """Judge the train-only mechanical G0 gate without reading query truth.

    The caller must supply argmax labels from one formal-tap probe and the
    frozen predecessor qKNN in the same opaque query order.  This helper only
    decides functional non-identity; it never consumes labels, roles, scores,
    or held/Target metrics.
    """

    tap_receipt = _require_sha256(
        formal_tap_receipt_sha256, "formal tap receipt SHA256"
    )
    if not isinstance(query_ids, (tuple, list, np.ndarray)) or len(query_ids) < 1:
        raise D106RCMR2VError("G0 requires one or more opaque query IDs")
    count = len(query_ids)
    ids = _typed_tokens(query_ids, "G0 query_ids", count)
    candidate = _typed_tokens(candidate_argmax, "G0 candidate_argmax", count)
    baseline = _typed_tokens(baseline_argmax, "G0 baseline_argmax", count)
    if len(set(ids)) != count:
        raise D106RCMR2VError("G0 query IDs must be unique and opaque")
    changed = sum(
        1
        for candidate_label, baseline_label in zip(candidate, baseline, strict=True)
        if candidate_label != baseline_label
    )
    return {
        "argmax_changed_count": changed,
        "baseline_argmax_root_sha256": _sha256_bytes(_canonical_bytes(list(baseline))),
        "candidate_argmax_root_sha256": _sha256_bytes(_canonical_bytes(list(candidate))),
        "candidate_id": CANDIDATE_ID,
        "formal_tap_receipt_sha256": tap_receipt,
        "g0_status": (
            "G0_ARGMAX_CHANGED_NO_PERFORMANCE_CLAIM"
            if changed
            else "REJECT_NO_FUNCTION"
        ),
        "probe_scope": "TRAIN_ONLY_MECHANICAL_FORMAL_TAP",
        "protocol_schema": PROTOCOL_SCHEMA,
        "query_count": count,
        "query_ids_root_sha256": _sha256_bytes(_canonical_bytes(list(ids))),
        "query_role_access": False,
        "query_truth_access": False,
        "schema": G0_RECEIPT_SCHEMA,
    }


def audit_d106_rcmr_2v_resources(state: D106RCMR2VState) -> dict[str, Any]:
    """Return deterministic buffer accounting; it is deliberately not an RSS claim."""

    formal_state = _require_formal_state(state)
    count = len(formal_state.class_index)
    current_numeric_state_bytes = _numeric_state_bytes(count)
    decoded_bytes = 2 * count * Z_DIM * np.dtype(np.float64).itemsize
    profiles_bytes = 2 * count * (count - 1) * np.dtype(np.float64).itemsize
    one_distance_matrix_bytes = count * count * np.dtype(np.float64).itemsize
    row_scratch_bytes = count * np.dtype(np.float64).itemsize
    temporary_peak_bytes = (
        decoded_bytes + profiles_bytes + one_distance_matrix_bytes + row_scratch_bytes
    )
    support_profile_comparisons = round(
        2.0 * count * (count - 1) * math.log2(float(count - 1))
    )
    query_rank_comparisons = round(2.0 * count * math.log2(float(count)))
    profile_binary_search_comparisons = 2 * count * math.ceil(math.log2(float(count - 1)))
    wire = serialize_d106_rcmr_2v_state(formal_state)
    wire_header_and_framing_bytes = len(wire) - current_numeric_state_bytes
    if (
        wire_header_and_framing_bytes < len(WIRE_MAGIC) + 4
        or len(wire) != wire_header_and_framing_bytes + current_numeric_state_bytes
    ):
        raise D106RCMR2VError("RCMR actual wire accounting drift")
    return {
        "C": len(formal_state.registry),
        "D": Z_DIM,
        "K": formal_state.binding.active_k,
        "N": count,
        "candidate_id": CANDIDATE_ID,
        "actual_canonical_wire_bytes": len(wire),
        "actual_prepare_plus_canonical_wire_bytes": temporary_peak_bytes + len(wire),
        "actual_prepare_plus_numeric_state_bytes": temporary_peak_bytes
        + current_numeric_state_bytes,
        "actual_wire_header_and_framing_bytes": wire_header_and_framing_bytes,
        "actual_wire_size_cap_bytes": MAX_CANONICAL_WIRE_BYTES,
        "canonical_wire_bytes": len(wire),
        "current_numeric_state_bytes": current_numeric_state_bytes,
        "design_fixed_binary_payload_bytes_at_max_n": 86060,
        "design_fixed_binary_payload_header_allowance_bytes_at_max_n": 1040,
        "design_fixed_total_bytes_at_max_n": 2371980,
        "design_max_n": MAX_SUPPORT_ROWS,
        "design_max_payload_bytes": 86060,
        "design_max_prepare_peak_bytes": 2285920,
        "design_max_total_bytes": 2371980,
        "ephemeral_decoded_bytes": decoded_bytes,
        "ephemeral_profile_bytes": profiles_bytes,
        "ephemeral_row_scratch_bytes": row_scratch_bytes,
        "ephemeral_single_distance_matrix_bytes": one_distance_matrix_bytes,
        "persistent_dense_nxn_bytes": 0,
        "prepare_normalization_squared_sums": 2 * count * Z_DIM,
        "prepare_profile_comparison_estimate": support_profile_comparisons,
        "prepare_support_support_mac": count * (count - 1) * Z_DIM,
        "query_normalization_squared_sums": 2 * Z_DIM,
        "query_profile_binary_search_comparisons": profile_binary_search_comparisons,
        "query_rank_comparison_estimate": query_rank_comparisons,
        "query_state_updates": 0,
        "query_support_mac": 2 * count * Z_DIM,
        "resource_receipt_schema": RESOURCE_RECEIPT_SCHEMA,
        "state_receipt_sha256": formal_state.state_receipt_sha256,
        "temporary_prepare_peak_bytes": temporary_peak_bytes,
        "unaccounted_overhead": (
            "JSON objects, Python objects, allocator behavior, and process RSS are "
            "not measured; only deterministic C-contiguous payload buffers are counted"
        ),
        "wire_bytes_over_design_fixed_binary_payload_at_max_n": len(wire) - 86060,
    }


__all__ = [
    "BINDING_SCHEMA",
    "CANDIDATE_ID",
    "D106RCMRCrossClassTieError",
    "D106RCMR2VBinding",
    "D106RCMR2VError",
    "D106RCMR2VPrediction",
    "D106RCMR2VState",
    "G0_RECEIPT_SCHEMA",
    "MAX_CANONICAL_WIRE_BYTES",
    "MAX_REGISTERED_CLASSES",
    "MAX_SUPPORT_PER_CLASS",
    "MAX_SUPPORT_ROWS",
    "METHOD_LOCK_SCHEMA",
    "PROTOCOL_SCHEMA",
    "RESOURCE_RECEIPT_SCHEMA",
    "STATE_SCHEMA",
    "VALIDATED_ONCE",
    "WIRE_SCHEMA",
    "Z_DIM",
    "audit_d106_rcmr_2v_resources",
    "build_d106_rcmr_2v_state",
    "deserialize_d106_rcmr_2v_state",
    "judge_d106_rcmr_2v_g0",
    "load_d106_rcmr_2v_method_lock",
    "load_d106_rcmr_2v_state",
    "prepare_d106_rcmr_2v_scoring_context",
    "score_d106_rcmr_2v_query",
    "serialize_d106_rcmr_2v_state",
]
