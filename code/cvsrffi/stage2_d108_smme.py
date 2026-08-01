"""D108 support-margin mean equalizing head for D92 LDA logits.

The build surface accepts only a registered support matrix of already-computed
D92 equal-prior LDA logits plus its legal support labels.  It emits a compact,
immutable, class-permutation-equivariant bias state.  Query inference merely
adds that bias to every independently supplied LDA logit row; it has no
query-side fitting, update, truth, role, quota, routing, or batch surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any

import numpy as np


CANDIDATE_ID = "D108-SMME-LDA/r1"
PROTOCOL_SCHEMA = "p2_min_v1"
STATE_SCHEMA = "cvs.phase2.d108.smme_state.v1"
WIRE_SCHEMA = "cvs.phase2.d108.smme_wire.v1"
RESOURCE_SCHEMA = "cvs.phase2.d108.smme_resource.v1"
WIRE_MAGIC = b"CVSD108SMME\x00\x01"
MAX_REGISTERED_CLASSES = 26
MAX_SUPPORT_PER_CLASS = 10
MAX_REGISTRY_TOKEN_WIRE_BYTES = 128
MAX_CANONICAL_WIRE_BYTES = 8192
_ARRAY_FIELDS = ("class_margins_fp64", "delta_fp64")
_FLOAT64_EPSILON = np.finfo(np.float64).eps


class SMMEError(ValueError):
    """Raised when a D108 SMME input, state, wire, or totalization drifts."""


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
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SMMEError(f"{name} must be a lowercase SHA256")
    return value


def _require_text(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise SMMEError(f"{name} must be a non-empty exact string")
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > MAX_REGISTRY_TOKEN_WIRE_BYTES:
        raise SMMEError(f"{name} exceeds the sealed wire-token limit")
    return value


def _registry(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise SMMEError("registered_classes must be a sequence of exact strings")
    result = tuple(
        _require_text(value, f"registered_classes[{index}]")
        for index, value in enumerate(values)
    )
    if not 2 <= len(result) <= MAX_REGISTERED_CLASSES:
        raise SMMEError(
            f"registered_classes must contain 2..{MAX_REGISTERED_CLASSES} classes"
        )
    if len(set(result)) != len(result):
        raise SMMEError("registered_classes must be unique")
    return result


def _labels(values: Sequence[str], rows: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise SMMEError("support_labels must be a sequence of exact strings")
    result = tuple(
        _require_text(value, f"support_labels[{index}]")
        for index, value in enumerate(values)
    )
    if len(result) != rows:
        raise SMMEError("support label/logit row count mismatch")
    return result


def _float32_logit_matrix(
    value: Any, name: str, *, class_count: int
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise SMMEError(f"{name} must be an exact float32 numpy matrix")
    if (
        value.ndim != 2
        or value.shape[0] < 1
        or value.shape[1] != class_count
        or not np.isfinite(value).all()
    ):
        raise SMMEError(f"{name} must be finite float32 [N,{class_count}]")
    return np.ascontiguousarray(value, dtype=np.float64)


def _freeze_array(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(order="C"), dtype=dtype).reshape(array.shape)


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _totalize(values: Any, name: str) -> float:
    """Use sorted FP64 summands so support and registry order cannot steer sums."""

    rows = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(rows) < 1 or not np.isfinite(rows).all():
        raise SMMEError(f"{name} totalization requires finite non-empty values")
    result = math.fsum(sorted(float(value) for value in rows))
    if not math.isfinite(result):
        raise SMMEError(f"{name} totalization became non-finite")
    return result


def _mean_totalized(values: Any, name: str) -> float:
    rows = np.asarray(values, dtype=np.float64).reshape(-1)
    return _totalize(rows, name) / float(len(rows))


def _stable_logsumexp(values: Any, name: str) -> float:
    rows = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(rows) < 1 or not np.isfinite(rows).all():
        raise SMMEError(f"{name} requires finite non-empty competitor logits")
    maximum = float(np.max(rows))
    shifted = np.exp(rows - maximum)
    result = maximum + math.log(_totalize(shifted, name))
    if not math.isfinite(result):
        raise SMMEError(f"{name} logsumexp became non-finite")
    return result


def _zero_sum_tolerance(values: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.abs(values.astype(np.float64)))))
    return 64.0 * _FLOAT64_EPSILON * float(len(values)) * scale


def _class_margins(
    support_logits: np.ndarray,
    support_labels: tuple[str, ...],
    classes: tuple[str, ...],
    k_shot: int,
) -> np.ndarray:
    margins: list[float] = []
    for class_index, class_name in enumerate(classes):
        class_rows = [
            index for index, label in enumerate(support_labels) if label == class_name
        ]
        if len(class_rows) != k_shot:
            raise SMMEError("support K-shot balance drift during margin construction")
        values: list[float] = []
        for row_index in class_rows:
            row = support_logits[row_index]
            competitors = np.concatenate((row[:class_index], row[class_index + 1 :]))
            values.append(
                float(row[class_index])
                - _stable_logsumexp(competitors, "support competitor logit")
            )
        margin = _mean_totalized(values, "per-class support margin")
        if not math.isfinite(margin):
            raise SMMEError("per-class support margin became non-finite")
        margins.append(margin)
    return np.ascontiguousarray(margins, dtype=np.float64)


def _delta_from_margins(class_margins: np.ndarray) -> np.ndarray:
    values = np.asarray(class_margins, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise SMMEError("class margins must be a finite vector with at least two classes")
    mean_margin = _mean_totalized(values, "class margin")
    delta = np.ascontiguousarray(mean_margin - values, dtype=np.float64)
    if not np.isfinite(delta).all():
        raise SMMEError("zero-sum margin bias became non-finite")
    if abs(_totalize(delta, "zero-sum margin bias")) > _zero_sum_tolerance(values):
        raise SMMEError("zero-sum margin-bias totalization drift")
    return delta


@dataclass(frozen=True, slots=True)
class SMMEState:
    """Typed immutable support-only state for the D108 SMME LDA head."""

    registered_classes: tuple[str, ...]
    k_shot: int
    class_margins_fp64: np.ndarray
    delta_fp64: np.ndarray
    state_receipt_sha256: str

    def __post_init__(self) -> None:
        _validate_state(self)


def _resource_payload(state: SMMEState) -> dict[str, Any]:
    class_count = len(state.registered_classes)
    delta_totalization = _totalize(state.delta_fp64, "resource delta")
    return {
        "schema": RESOURCE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "registered_class_count": class_count,
        "k_shot": state.k_shot,
        "support_row_count": class_count * state.k_shot,
        "trainable_parameters": 0,
        "support_only": True,
        "support_logit_value_count": class_count * class_count * state.k_shot,
        "support_margin_comparison_count": class_count
        * state.k_shot
        * (class_count - 1),
        "query_bias_additions_per_row": class_count,
        "query_fit_rows": 0,
        "query_state_updates": 0,
        "numeric_state_bytes": state.class_margins_fp64.nbytes
        + state.delta_fp64.nbytes,
        "delta_totalization_hex": float(delta_totalization).hex(),
        "delta_totalization_tolerance_hex": _zero_sum_tolerance(
            state.class_margins_fp64
        ).hex(),
    }


def _state_payload(state: SMMEState) -> dict[str, Any]:
    return {
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "state_schema": STATE_SCHEMA,
        "registered_classes": list(state.registered_classes),
        "k_shot": state.k_shot,
        "arrays": {
            name: _array_receipt(getattr(state, name)) for name in _ARRAY_FIELDS
        },
        "resource": _resource_payload(state),
    }


def _expected_state_receipt(state: SMMEState) -> str:
    return _sha256_bytes(_canonical_bytes(_state_payload(state)))


def _validate_state(state: SMMEState) -> None:
    if type(state) is not SMMEState:
        raise SMMEError("state must be an exact SMMEState")
    classes = _registry(state.registered_classes)
    if type(state.k_shot) is not int or not 1 <= state.k_shot <= MAX_SUPPORT_PER_CLASS:
        raise SMMEError("state K-shot is outside the frozen support range")
    for name in _ARRAY_FIELDS:
        value = getattr(state, name)
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f8")
            or value.shape != (len(classes),)
            or value.flags.writeable
            or not value.flags.c_contiguous
            or not np.isfinite(value).all()
        ):
            raise SMMEError(f"{name} typed immutable-state contract drift")
    expected_delta = _delta_from_margins(state.class_margins_fp64)
    if not np.array_equal(expected_delta, state.delta_fp64):
        raise SMMEError("state margin/delta equation drift")
    if abs(_totalize(state.delta_fp64, "state delta")) > _zero_sum_tolerance(
        state.class_margins_fp64
    ):
        raise SMMEError("state delta is not numerically zero-sum")
    _require_sha256(state.state_receipt_sha256, "state receipt")
    if state.state_receipt_sha256 != _expected_state_receipt(state):
        raise SMMEError("state receipt verification failed")


def _make_state(
    *,
    registered_classes: tuple[str, ...],
    k_shot: int,
    class_margins_fp64: np.ndarray,
    delta_fp64: np.ndarray,
    expected_receipt: str | None = None,
) -> SMMEState:
    values = {
        "registered_classes": registered_classes,
        "k_shot": int(k_shot),
        "class_margins_fp64": _freeze_array(
            class_margins_fp64, np.dtype("<f8")
        ),
        "delta_fp64": _freeze_array(delta_fp64, np.dtype("<f8")),
    }
    provisional = object.__new__(SMMEState)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    receipt = _expected_state_receipt(provisional)
    if expected_receipt is not None and receipt != expected_receipt:
        raise SMMEError("wire state receipt verification failed")
    values["state_receipt_sha256"] = receipt
    return SMMEState(**values)


def build_smme_state(
    support_logits: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
) -> SMMEState:
    """Build D108 solely from D92 LDA logits of balanced registered support."""

    classes = _registry(registered_classes)
    logits = _float32_logit_matrix(
        support_logits, "support_logits", class_count=len(classes)
    )
    labels = _labels(support_labels, len(logits))
    if any(label not in classes for label in labels):
        raise SMMEError("support labels must belong to registered_classes")
    counts = tuple(labels.count(class_name) for class_name in classes)
    if any(count < 1 for count in counts) or len(set(counts)) != 1:
        raise SMMEError("support must be balanced K-shot over every registered class")
    k_shot = counts[0]
    if k_shot > MAX_SUPPORT_PER_CLASS:
        raise SMMEError("support K exceeds the frozen maximum")
    margins = _class_margins(logits, labels, classes, k_shot)
    delta = _delta_from_margins(margins)
    return _make_state(
        registered_classes=classes,
        k_shot=k_shot,
        class_margins_fp64=margins,
        delta_fp64=delta,
    )


def apply_smme_query(state: SMMEState, query_logits: np.ndarray) -> np.ndarray:
    """Add the sealed support-only bias to each independent D92 LDA logit row."""

    if type(state) is not SMMEState:
        raise SMMEError("state must be an exact SMMEState")
    _validate_state(state)
    logits = _float32_logit_matrix(
        query_logits, "query_logits", class_count=len(state.registered_classes)
    )
    adjusted = np.ascontiguousarray(logits + state.delta_fp64[None, :], dtype=np.float32)
    if not np.isfinite(adjusted).all():
        raise SMMEError("query bias addition became non-finite")
    return adjusted


def _wire_header(state: SMMEState) -> dict[str, Any]:
    return {
        "wire_schema": WIRE_SCHEMA,
        "array_order": list(_ARRAY_FIELDS),
        "state": _state_payload(state),
        "state_receipt_sha256": state.state_receipt_sha256,
    }


def serialize_smme_state(state: SMMEState) -> bytes:
    if type(state) is not SMMEState:
        raise SMMEError("state must be an exact SMMEState")
    _validate_state(state)
    header = _canonical_bytes(_wire_header(state))
    body = b"".join(getattr(state, name).tobytes(order="C") for name in _ARRAY_FIELDS)
    wire = WIRE_MAGIC + struct.pack(">I", len(header)) + header + body
    if len(wire) > MAX_CANONICAL_WIRE_BYTES:
        raise SMMEError("canonical wire exceeds the frozen size bound")
    return wire


def deserialize_smme_state(
    payload: bytes, *, expected_wire_sha256: str | None = None
) -> SMMEState:
    if type(payload) is not bytes:
        raise SMMEError("wire payload must be exact bytes")
    if len(payload) > MAX_CANONICAL_WIRE_BYTES or len(payload) < len(WIRE_MAGIC) + 4:
        raise SMMEError("wire payload size is outside the frozen bound")
    if expected_wire_sha256 is not None:
        expected = _require_sha256(expected_wire_sha256, "expected wire SHA256")
        if _sha256_bytes(payload) != expected:
            raise SMMEError("wire SHA256 verification failed")
    if not payload.startswith(WIRE_MAGIC):
        raise SMMEError("wire magic mismatch")
    offset = len(WIRE_MAGIC)
    header_size = struct.unpack(">I", payload[offset : offset + 4])[0]
    offset += 4
    if header_size < 2 or offset + header_size > len(payload):
        raise SMMEError("wire header length is invalid")
    header_raw = payload[offset : offset + header_size]
    offset += header_size
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SMMEError("wire header is not canonical JSON") from error
    if _canonical_bytes(header) != header_raw:
        raise SMMEError("wire header JSON is not canonical")
    if type(header) is not dict or set(header) != {
        "wire_schema",
        "array_order",
        "state",
        "state_receipt_sha256",
    }:
        raise SMMEError("wire header fields drift")
    if header["wire_schema"] != WIRE_SCHEMA or header["array_order"] != list(
        _ARRAY_FIELDS
    ):
        raise SMMEError("wire schema or array order drift")
    state_data = header["state"]
    if type(state_data) is not dict or set(state_data) != {
        "candidate_id",
        "protocol_schema",
        "state_schema",
        "registered_classes",
        "k_shot",
        "arrays",
        "resource",
    }:
        raise SMMEError("wire state fields drift")
    if (
        state_data["candidate_id"] != CANDIDATE_ID
        or state_data["protocol_schema"] != PROTOCOL_SCHEMA
        or state_data["state_schema"] != STATE_SCHEMA
        or type(state_data["k_shot"]) is not int
    ):
        raise SMMEError("wire state schema drift")
    classes = _registry(state_data["registered_classes"])
    arrays_data = state_data["arrays"]
    if type(arrays_data) is not dict or tuple(arrays_data) != tuple(
        sorted(_ARRAY_FIELDS)
    ):
        raise SMMEError("wire array receipts are not canonical")
    arrays: dict[str, np.ndarray] = {}
    for name in _ARRAY_FIELDS:
        receipt = arrays_data.get(name)
        if type(receipt) is not dict or set(receipt) != {"dtype", "shape", "sha256"}:
            raise SMMEError(f"{name} wire receipt drift")
        if receipt["dtype"] != np.dtype("<f8").str or receipt["shape"] != [
            len(classes)
        ]:
            raise SMMEError(f"{name} wire descriptor drift")
        nbytes = len(classes) * np.dtype("<f8").itemsize
        if offset + nbytes > len(payload):
            raise SMMEError(f"{name} wire body is truncated")
        raw = payload[offset : offset + nbytes]
        offset += nbytes
        if _sha256_bytes(raw) != _require_sha256(receipt["sha256"], f"{name} SHA256"):
            raise SMMEError(f"{name} wire body receipt mismatch")
        arrays[name] = np.frombuffer(raw, dtype=np.dtype("<f8")).reshape(
            (len(classes),)
        )
    if offset != len(payload):
        raise SMMEError("wire has trailing bytes")
    state = _make_state(
        registered_classes=classes,
        k_shot=state_data["k_shot"],
        class_margins_fp64=arrays["class_margins_fp64"],
        delta_fp64=arrays["delta_fp64"],
        expected_receipt=_require_sha256(
            header["state_receipt_sha256"], "wire state receipt"
        ),
    )
    if _state_payload(state) != state_data:
        raise SMMEError("wire state payload is not canonical")
    return state


def smme_resource_receipt(state: SMMEState) -> dict[str, Any]:
    """Return a receipt for the support totalization and singleton query head."""

    if type(state) is not SMMEState:
        raise SMMEError("state must be an exact SMMEState")
    _validate_state(state)
    payload = _resource_payload(state)
    receipt = dict(payload)
    receipt["resource_receipt_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    receipt["canonical_wire_bytes"] = len(serialize_smme_state(state))
    return receipt


__all__ = [
    "CANDIDATE_ID",
    "MAX_CANONICAL_WIRE_BYTES",
    "MAX_REGISTERED_CLASSES",
    "MAX_SUPPORT_PER_CLASS",
    "PROTOCOL_SCHEMA",
    "RESOURCE_SCHEMA",
    "SMMEError",
    "SMMEState",
    "STATE_SCHEMA",
    "WIRE_SCHEMA",
    "apply_smme_query",
    "build_smme_state",
    "deserialize_smme_state",
    "serialize_smme_state",
    "smme_resource_receipt",
]
