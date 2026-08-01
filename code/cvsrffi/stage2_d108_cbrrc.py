"""D108 class-balanced ReLU residual companding.

CB-RRC builds one immutable Stage2-B state from the six-class before-support
set.  It changes only the ReLU ``z_id`` block of the existing D92 288-D
representation.  FFT96 and RF32 are copied byte-for-byte.  The transform has
no query fitting, truth, semantic-role, quota, routing, or batch-statistic
surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any, Sequence

import numpy as np


CANDIDATE_ID = "D108-CB-RRC/r1"
PROTOCOL_SCHEMA = "p2_min_v1"
STATE_SCHEMA = "cvs.phase2.d108.cbrrc_state.v1"
WIRE_SCHEMA = "cvs.phase2.d108.cbrrc_wire.v1"
RESOURCE_SCHEMA = "cvs.phase2.d108.cbrrc_resource.v1"
WIRE_MAGIC = b"CVSD108CBRRC\x00\x01"
FEATURE_DIM = 288
RELU_DIM = 160
FFT_RF_SLICE = slice(RELU_DIM, FEATURE_DIM)
BEFORE_CLASS_COUNT = 6
ALLOWED_K = (1, 5, 10)
MAX_CLASS_TOKEN_WIRE_BYTES = 70
MAX_WIRE_BYTES = 8192
MACHINE_EPSILON = float(np.finfo(np.float64).eps)


class CBRRCError(ValueError):
    """Raised when a D108 protocol, state, wire, or numeric invariant drifts."""


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
        raise CBRRCError(f"{name} must be a lowercase SHA256")
    if any(character not in "0123456789abcdef" for character in value):
        raise CBRRCError(f"{name} must be a lowercase SHA256")
    return value


def _require_class_token(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise CBRRCError(f"{name} must be a non-empty exact string")
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > MAX_CLASS_TOKEN_WIRE_BYTES:
        raise CBRRCError(f"{name} exceeds the canonical wire-token bound")
    return value


def _registry(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise CBRRCError("before_registered_classes must be a sequence")
    result = tuple(
        _require_class_token(value, f"before_registered_classes[{index}]")
        for index, value in enumerate(values)
    )
    if len(result) != BEFORE_CLASS_COUNT or len(set(result)) != len(result):
        raise CBRRCError(
            f"before registry must contain exactly {BEFORE_CLASS_COUNT} unique classes"
        )
    return result


def _feature_rows(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise CBRRCError(f"{name} must be an exact float32 numpy matrix")
    if (
        value.ndim != 2
        or value.shape[0] < 1
        or value.shape[1] != FEATURE_DIM
        or not np.isfinite(value).all()
    ):
        raise CBRRCError(f"{name} must be finite float32 [N,{FEATURE_DIM}]")
    if np.any(value[:, :RELU_DIM] < 0.0):
        raise CBRRCError(f"{name} ReLU block contains a negative coordinate")
    return np.ascontiguousarray(value)


def _unit_relu_rows(rows: np.ndarray) -> np.ndarray:
    relu = np.ascontiguousarray(rows[:, :RELU_DIM], dtype=np.float64)
    norms = np.linalg.norm(relu, axis=1)
    result = np.zeros_like(relu)
    active = norms > MACHINE_EPSILON
    result[active] = relu[active] / norms[active, None]
    return result


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


@dataclass(frozen=True, slots=True)
class CBRRCState:
    before_registered_classes: tuple[str, ...]
    k_shot: int
    energy_fp16: np.ndarray
    energy_mean: float
    state_receipt_sha256: str

    def __post_init__(self) -> None:
        _validate_state(self)


def _state_payload(state: CBRRCState) -> dict[str, Any]:
    return {
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "state_schema": STATE_SCHEMA,
        "before_registered_classes": list(state.before_registered_classes),
        "k_shot": state.k_shot,
        "energy_mean_hex": state.energy_mean.hex(),
        "epsilon_hex": MACHINE_EPSILON.hex(),
        "energy": _array_receipt(state.energy_fp16),
        "query_rows_used_for_fit": 0,
        "state_updates_from_query": 0,
    }


def _expected_receipt(state: CBRRCState) -> str:
    return _sha256_bytes(_canonical_bytes(_state_payload(state)))


def _validate_state(state: CBRRCState) -> None:
    _registry(state.before_registered_classes)
    if type(state.k_shot) is not int or state.k_shot not in ALLOWED_K:
        raise CBRRCError(f"state K must be one of {ALLOWED_K}")
    energy = state.energy_fp16
    if (
        not isinstance(energy, np.ndarray)
        or energy.dtype != np.dtype("<f2")
        or energy.shape != (RELU_DIM,)
        or energy.flags.writeable
        or not energy.flags.c_contiguous
        or not np.isfinite(energy).all()
        or np.any(energy < 0.0)
    ):
        raise CBRRCError("state energy must be immutable nonnegative FP16[160]")
    if (
        type(state.energy_mean) is not float
        or not math.isfinite(state.energy_mean)
        or state.energy_mean < 0.0
    ):
        raise CBRRCError("state energy mean must be finite and nonnegative")
    decoded_mean = float(np.mean(energy.astype(np.float64), dtype=np.float64))
    if decoded_mean.hex() != state.energy_mean.hex():
        raise CBRRCError("state energy mean is not bound to deployed FP16 energy")
    _require_sha256(state.state_receipt_sha256, "state receipt")
    if state.state_receipt_sha256 != _expected_receipt(state):
        raise CBRRCError("state receipt verification failed")


def _make_state(
    *,
    before_registered_classes: tuple[str, ...],
    k_shot: int,
    energy_fp16: np.ndarray,
    expected_receipt: str | None = None,
) -> CBRRCState:
    closed_energy = _freeze_array(energy_fp16, np.dtype("<f2"))
    values = {
        "before_registered_classes": before_registered_classes,
        "k_shot": int(k_shot),
        "energy_fp16": closed_energy,
        "energy_mean": float(
            np.mean(closed_energy.astype(np.float64), dtype=np.float64)
        ),
    }
    provisional = object.__new__(CBRRCState)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    receipt = _sha256_bytes(_canonical_bytes(_state_payload(provisional)))
    if expected_receipt is not None and receipt != expected_receipt:
        raise CBRRCError("wire state receipt verification failed")
    values["state_receipt_sha256"] = receipt
    return CBRRCState(**values)


def build_cbrrc_state(
    before_support_features: np.ndarray,
    before_support_labels: Sequence[str],
    before_registered_classes: Sequence[str],
) -> CBRRCState:
    """Freeze class-balanced ReLU energy from Stage2-B support only."""

    rows = _feature_rows(before_support_features, "before_support_features")
    registry = _registry(before_registered_classes)
    if isinstance(before_support_labels, (str, bytes)):
        raise CBRRCError("before_support_labels must be a sequence")
    labels = tuple(
        _require_class_token(value, f"before_support_labels[{index}]")
        for index, value in enumerate(before_support_labels)
    )
    if len(labels) != len(rows) or any(label not in registry for label in labels):
        raise CBRRCError("before support label/registry alignment drift")
    counts = tuple(labels.count(class_name) for class_name in registry)
    if len(set(counts)) != 1 or counts[0] not in ALLOWED_K:
        raise CBRRCError(
            f"before support must be balanced with K in {ALLOWED_K}"
        )
    unit_relu = _unit_relu_rows(rows)
    class_energy = []
    for class_name in registry:
        indices = [index for index, label in enumerate(labels) if label == class_name]
        class_energy.append(
            np.mean(np.square(unit_relu[indices]), axis=0, dtype=np.float64)
        )
    energy = np.mean(np.stack(class_energy), axis=0, dtype=np.float64)
    energy_fp16 = energy.astype(np.dtype("<f2"))
    if not np.isfinite(energy_fp16).all() or np.any(energy_fp16 < 0.0):
        raise CBRRCError("class-balanced energy cannot be represented in FP16")
    return _make_state(
        before_registered_classes=registry,
        k_shot=counts[0],
        energy_fp16=energy_fp16,
    )


def transform_cbrrc_features(state: CBRRCState, features: np.ndarray) -> np.ndarray:
    """Apply the frozen state independently to every 288-D feature row."""

    if type(state) is not CBRRCState:
        raise CBRRCError("state must be an exact CBRRCState")
    _validate_state(state)
    rows = _feature_rows(features, "features")
    unit_relu = _unit_relu_rows(rows)
    energy = state.energy_fp16.astype(np.float64)
    base = energy + state.energy_mean + MACHINE_EPSILON
    denominator = np.square(unit_relu) + base[None, :]
    residual_gain = np.sqrt(base[None, :] / denominator)
    transformed_relu = unit_relu * (1.0 + residual_gain)
    transformed_norms = np.linalg.norm(transformed_relu, axis=1)
    active = transformed_norms > MACHINE_EPSILON
    transformed_relu[active] /= transformed_norms[active, None]
    original_norms = np.linalg.norm(
        rows[:, :RELU_DIM].astype(np.float64), axis=1
    )
    transformed_relu[active] *= original_norms[active, None]
    transformed_relu[~active] = 0.0
    result = np.array(rows, dtype=np.float32, copy=True, order="C")
    result[:, :RELU_DIM] = transformed_relu.astype(np.float32)
    if not np.array_equal(result[:, FFT_RF_SLICE], rows[:, FFT_RF_SLICE]):
        raise CBRRCError("FFT96/RF32 identity path drift")
    if not np.isfinite(result).all() or np.any(result[:, :RELU_DIM] < 0.0):
        raise CBRRCError("CB-RRC transform produced an invalid feature row")
    result = np.frombuffer(result.tobytes(order="C"), dtype=np.float32).reshape(
        result.shape
    )
    return result


def _wire_header(state: CBRRCState) -> dict[str, Any]:
    return {
        "wire_schema": WIRE_SCHEMA,
        "state": _state_payload(state),
        "state_receipt_sha256": state.state_receipt_sha256,
    }


def serialize_cbrrc_state(state: CBRRCState) -> bytes:
    if type(state) is not CBRRCState:
        raise CBRRCError("state must be an exact CBRRCState")
    _validate_state(state)
    header = _canonical_bytes(_wire_header(state))
    body = state.energy_fp16.tobytes(order="C")
    wire = WIRE_MAGIC + struct.pack(">I", len(header)) + header + body
    if len(wire) > MAX_WIRE_BYTES:
        raise CBRRCError("canonical wire exceeds the frozen bound")
    return wire


def _parse_binary64_hex(value: Any, name: str) -> float:
    if type(value) is not str:
        raise CBRRCError(f"{name} must use canonical binary64 hex")
    try:
        result = float.fromhex(value)
    except ValueError as error:
        raise CBRRCError(f"{name} is not valid binary64 hex") from error
    if not math.isfinite(result) or result.hex() != value:
        raise CBRRCError(f"{name} is not canonical finite binary64 hex")
    return result


def deserialize_cbrrc_state(
    payload: bytes, *, expected_wire_sha256: str | None = None
) -> CBRRCState:
    if type(payload) is not bytes:
        raise CBRRCError("wire payload must be exact bytes")
    if len(payload) > MAX_WIRE_BYTES or len(payload) < len(WIRE_MAGIC) + 4:
        raise CBRRCError("wire payload size is outside the frozen bound")
    if expected_wire_sha256 is not None:
        expected = _require_sha256(expected_wire_sha256, "expected wire SHA256")
        if _sha256_bytes(payload) != expected:
            raise CBRRCError("wire SHA256 verification failed")
    if not payload.startswith(WIRE_MAGIC):
        raise CBRRCError("wire magic mismatch")
    offset = len(WIRE_MAGIC)
    header_size = struct.unpack(">I", payload[offset : offset + 4])[0]
    offset += 4
    if header_size < 2 or offset + header_size > len(payload):
        raise CBRRCError("wire header length drift")
    raw_header = payload[offset : offset + header_size]
    offset += header_size
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CBRRCError("wire header is not canonical UTF-8 JSON") from error
    if _canonical_bytes(header) != raw_header:
        raise CBRRCError("wire header JSON is not canonical")
    if type(header) is not dict or set(header) != {
        "wire_schema",
        "state",
        "state_receipt_sha256",
    }:
        raise CBRRCError("wire header fields drift")
    if header["wire_schema"] != WIRE_SCHEMA:
        raise CBRRCError("wire schema drift")
    state_data = header["state"]
    expected_fields = {
        "candidate_id",
        "protocol_schema",
        "state_schema",
        "before_registered_classes",
        "k_shot",
        "energy_mean_hex",
        "epsilon_hex",
        "energy",
        "query_rows_used_for_fit",
        "state_updates_from_query",
    }
    if type(state_data) is not dict or set(state_data) != expected_fields:
        raise CBRRCError("wire state fields drift")
    if (
        state_data["candidate_id"] != CANDIDATE_ID
        or state_data["protocol_schema"] != PROTOCOL_SCHEMA
        or state_data["state_schema"] != STATE_SCHEMA
        or state_data["query_rows_used_for_fit"] != 0
        or state_data["state_updates_from_query"] != 0
        or _parse_binary64_hex(state_data["epsilon_hex"], "epsilon")
        != MACHINE_EPSILON
    ):
        raise CBRRCError("wire protocol or method identity drift")
    registry = _registry(state_data["before_registered_classes"])
    if type(state_data["k_shot"]) is not int or state_data["k_shot"] not in ALLOWED_K:
        raise CBRRCError("wire K drift")
    energy_receipt = state_data["energy"]
    if type(energy_receipt) is not dict or set(energy_receipt) != {
        "dtype",
        "shape",
        "sha256",
    }:
        raise CBRRCError("wire energy receipt drift")
    if (
        energy_receipt["dtype"] != np.dtype("<f2").str
        or energy_receipt["shape"] != [RELU_DIM]
    ):
        raise CBRRCError("wire energy layout drift")
    body_size = RELU_DIM * np.dtype("<f2").itemsize
    if offset + body_size != len(payload):
        raise CBRRCError("wire energy body length drift")
    body = payload[offset:]
    if _sha256_bytes(body) != _require_sha256(
        energy_receipt["sha256"], "wire energy SHA256"
    ):
        raise CBRRCError("wire energy body receipt mismatch")
    energy = np.frombuffer(body, dtype=np.dtype("<f2"))
    state = _make_state(
        before_registered_classes=registry,
        k_shot=state_data["k_shot"],
        energy_fp16=energy,
        expected_receipt=_require_sha256(
            header["state_receipt_sha256"], "wire state receipt"
        ),
    )
    if _parse_binary64_hex(state_data["energy_mean_hex"], "energy mean").hex() != (
        state.energy_mean.hex()
    ):
        raise CBRRCError("wire energy mean drift")
    if _state_payload(state) != state_data:
        raise CBRRCError("wire state payload is not canonical")
    return state


def cbrrc_resource_receipt(state: CBRRCState) -> dict[str, int | str]:
    if type(state) is not CBRRCState:
        raise CBRRCError("resource receipt requires an exact CBRRCState")
    _validate_state(state)
    support_rows = BEFORE_CLASS_COUNT * state.k_shot
    return {
        "schema": RESOURCE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "support_rows": support_rows,
        "numeric_state_bytes": int(state.energy_fp16.nbytes),
        "canonical_wire_bytes": len(serialize_cbrrc_state(state)),
        "build_elementwise_ops_upper_bound": support_rows * RELU_DIM * 6,
        "query_elementwise_ops_upper_bound": RELU_DIM * 13,
        "query_extra_mac_upper_bound": RELU_DIM * 5,
        "output_feature_dimension": FEATURE_DIM,
    }


__all__ = [
    "ALLOWED_K",
    "BEFORE_CLASS_COUNT",
    "CANDIDATE_ID",
    "CBRRCError",
    "CBRRCState",
    "FEATURE_DIM",
    "RELU_DIM",
    "build_cbrrc_state",
    "cbrrc_resource_receipt",
    "deserialize_cbrrc_state",
    "serialize_cbrrc_state",
    "transform_cbrrc_features",
]
