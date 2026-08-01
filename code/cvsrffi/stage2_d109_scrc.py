"""D109 support-confusion reciprocal calibration for frozen LDA logits.

The build surface consumes only balanced registered support logits and labels.
It freezes a compact, typed state containing the support confusion matrix ``Q``,
its reciprocal transition ``T``, and the support-determined strength ``rho``.
Each query is subsequently processed independently as ``p @ T``; there is no
query fitting, update, truth, role, quota, routing, or batch-assignment API.
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


CANDIDATE_ID = "D109-SCRC/r1"
PROTOCOL_SCHEMA = "p2_min_v1"
STATE_SCHEMA = "cvs.phase2.d109.scrc_state.v1"
WIRE_SCHEMA = "cvs.phase2.d109.scrc_wire.v1"
RESOURCE_SCHEMA = "cvs.phase2.d109.scrc_resource.v1"
WIRE_MAGIC = b"CVSD109SCRC\x00\x01"
MAX_REGISTERED_CLASSES = 26
ALLOWED_K = (1, 5, 10)
MAX_REGISTRY_TOKEN_WIRE_BYTES = 128
MAX_CANONICAL_WIRE_BYTES = 32768
_ARRAY_FIELDS = ("support_confusion_fp32", "transition_fp32")
_FLOAT32_EPSILON = float(np.finfo(np.float32).eps)
_FLOAT32_EXPONENT_FLOOR = float(
    np.nextafter(np.float32(0.0), np.float32(1.0))
)
_LOG_FLOAT32_EXPONENT_FLOOR = math.log(_FLOAT32_EXPONENT_FLOOR)


class SCRCError(ValueError):
    """Raised when a D109 SCRC input, state, wire, or equation drifts."""


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
        raise SCRCError(f"{name} must be a lowercase SHA256")
    return value


def _require_text(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise SCRCError(f"{name} must be a non-empty exact string")
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > MAX_REGISTRY_TOKEN_WIRE_BYTES:
        raise SCRCError(f"{name} exceeds the sealed wire-token limit")
    return value


def _registry(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise SCRCError("registered_classes must be a sequence of exact strings")
    result = tuple(
        _require_text(value, f"registered_classes[{index}]")
        for index, value in enumerate(values)
    )
    if not 2 <= len(result) <= MAX_REGISTERED_CLASSES:
        raise SCRCError(
            f"registered_classes must contain 2..{MAX_REGISTERED_CLASSES} classes"
        )
    if len(set(result)) != len(result):
        raise SCRCError("registered_classes must be unique")
    return result


def _labels(values: Sequence[str], rows: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise SCRCError("support_labels must be a sequence of exact strings")
    result = tuple(
        _require_text(value, f"support_labels[{index}]")
        for index, value in enumerate(values)
    )
    if len(result) != rows:
        raise SCRCError("support label/logit row count mismatch")
    return result


def _float32_logit_matrix(
    value: Any, name: str, *, class_count: int
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise SCRCError(f"{name} must be an exact float32 numpy matrix")
    if (
        value.ndim != 2
        or value.shape[0] < 1
        or value.shape[1] != class_count
        or not np.isfinite(value).all()
    ):
        raise SCRCError(f"{name} must be finite float32 [N,{class_count}]")
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
    """Deterministically totalize finite values without input-order steering."""

    rows = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(rows) < 1 or not np.isfinite(rows).all():
        raise SCRCError(f"{name} totalization requires finite non-empty values")
    result = math.fsum(sorted(float(value) for value in rows))
    if not math.isfinite(result):
        raise SCRCError(f"{name} totalization became non-finite")
    return result


def _row_sum_error(values: np.ndarray, name: str) -> float:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise SCRCError(f"{name} must be a non-empty matrix")
    return max(
        abs(_totalize(matrix[row_index], f"{name} row") - 1.0)
        for row_index in range(matrix.shape[0])
    )


def _stochastic_tolerance(values: np.ndarray) -> float:
    matrix = np.asarray(values, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(matrix))))
    return 128.0 * _FLOAT32_EPSILON * float(matrix.shape[1]) * scale


def _stable_softmax_row(logits: np.ndarray, name: str) -> np.ndarray:
    """Stable softmax with a fixed, audited float32 underflow floor only."""

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if len(values) < 2 or not np.isfinite(values).all():
        raise SCRCError(f"{name} must be finite competitor logits")
    maximum = float(np.max(values))
    shifted = values - maximum
    exponent_terms = np.exp(np.maximum(shifted, _LOG_FLOAT32_EXPONENT_FLOOR))
    exponent_terms = np.maximum(exponent_terms, _FLOAT32_EXPONENT_FLOOR)
    denominator = _totalize(exponent_terms, f"{name} softmax denominator")
    probabilities = np.ascontiguousarray(exponent_terms / denominator, dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any(probabilities <= 0.0):
        raise SCRCError(f"{name} stable softmax lost positive probability")
    return probabilities


def _stable_softmax_matrix(logits: np.ndarray, name: str) -> np.ndarray:
    rows = [_stable_softmax_row(logits[index], name) for index in range(len(logits))]
    return np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float64)


def _logsumexp(values: np.ndarray, name: str) -> float:
    """Return a finite deterministic float64 log-sum-exp."""

    terms = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(terms) < 1 or np.isnan(terms).any() or np.isposinf(terms).any():
        raise SCRCError(f"{name} requires finite or negative-infinite terms")
    maximum = float(np.max(terms))
    if not math.isfinite(maximum):
        raise SCRCError(f"{name} has no finite transition path")
    exponent_terms = np.exp(terms - maximum)
    denominator = _totalize(exponent_terms, f"{name} exponential total")
    result = maximum + math.log(denominator)
    if not math.isfinite(result):
        raise SCRCError(f"{name} became non-finite")
    return result


def _true_log_softmax_row(logits: np.ndarray, name: str) -> np.ndarray:
    """Compute log-softmax without probability-domain underflow flooring."""

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if len(values) < 2 or not np.isfinite(values).all():
        raise SCRCError(f"{name} must be finite competitor logits")
    normalizer = _logsumexp(values, f"{name} log-softmax normalizer")
    result = np.ascontiguousarray(values - normalizer, dtype=np.float64)
    if not np.isfinite(result).all():
        raise SCRCError(f"{name} log-softmax became non-finite")
    return result


def _log_transition_probabilities(
    log_probabilities: np.ndarray, transition: np.ndarray
) -> np.ndarray:
    """Compute log(p @ T) column-wise while preserving exact zero T entries."""

    log_p = np.asarray(log_probabilities, dtype=np.float64).reshape(-1)
    matrix = np.asarray(transition, dtype=np.float64)
    if matrix.shape != (len(log_p), len(log_p)):
        raise SCRCError("query transition/log-probability shape drift")
    log_transition = np.full(matrix.shape, -np.inf, dtype=np.float64)
    positive = matrix > 0.0
    log_transition[positive] = np.log(matrix[positive])
    result = np.asarray(
        [
            _logsumexp(
                log_p + log_transition[:, output_index],
                f"query transition column {output_index}",
            )
            for output_index in range(matrix.shape[1])
        ],
        dtype=np.float64,
    )
    if not np.isfinite(result).all():
        raise SCRCError("query log transition became non-finite")
    return result


def _support_confusion(
    support_probabilities: np.ndarray,
    support_labels: tuple[str, ...],
    classes: tuple[str, ...],
    k_shot: int,
) -> np.ndarray:
    confusion = np.empty((len(classes), len(classes)), dtype=np.float64)
    for class_index, class_name in enumerate(classes):
        row_indices = [
            index for index, label in enumerate(support_labels) if label == class_name
        ]
        if len(row_indices) != k_shot:
            raise SCRCError("support K-shot balance drift during Q construction")
        for logit_index in range(len(classes)):
            confusion[class_index, logit_index] = _totalize(
                support_probabilities[row_indices, logit_index],
                "per-class support posterior",
            ) / float(k_shot)
    frozen = np.ascontiguousarray(confusion, dtype=np.dtype("<f4"))
    if not np.isfinite(frozen).all() or np.any(frozen < 0.0):
        raise SCRCError("support confusion cannot be represented as finite float32")
    return frozen


def _validate_confusion(value: np.ndarray, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype("<f4")
        or value.ndim != 2
        or value.shape[0] != value.shape[1]
        or value.shape[0] < 2
        or not value.flags.c_contiguous
        or not np.isfinite(value).all()
        or np.any(value < 0.0)
        or np.any(value > 1.0)
    ):
        raise SCRCError(f"{name} must be a finite non-negative float32 square matrix")
    error = _row_sum_error(value, name)
    if error > _stochastic_tolerance(value):
        raise SCRCError(f"{name} is not row-stochastic within float32 tolerance")
    return np.ascontiguousarray(value, dtype=np.float64)


def _transition_from_confusion(
    support_confusion: np.ndarray,
) -> tuple[np.float32, np.ndarray, np.ndarray]:
    """Return rho, row-stochastic R[b,a], and T[b,a] from Q[a,b]."""

    confusion = _validate_confusion(support_confusion, "support confusion")
    class_count = confusion.shape[0]
    column_totals = np.asarray(
        [
            _totalize(confusion[:, logit_index], "support confusion column")
            for logit_index in range(class_count)
        ],
        dtype=np.float64,
    )
    if np.any(column_totals <= 0.0) or not np.isfinite(column_totals).all():
        raise SCRCError("support confusion has a non-positive Bayes denominator")
    reciprocal = np.empty((class_count, class_count), dtype=np.float64)
    for observed_index in range(class_count):
        reciprocal[observed_index, :] = (
            confusion[:, observed_index] / column_totals[observed_index]
        )
    if (
        not np.isfinite(reciprocal).all()
        or np.any(reciprocal < 0.0)
        or _row_sum_error(reciprocal, "reciprocal response")
        > _stochastic_tolerance(reciprocal)
    ):
        raise SCRCError("reciprocal response is not finite row-stochastic")
    rho_value = 1.0 - (
        _totalize(np.diag(confusion), "support confusion trace")
        / float(class_count)
    )
    rho_fp32 = np.float32(rho_value)
    if not np.isfinite(rho_fp32) or rho_fp32 < 0.0 or rho_fp32 > 1.0:
        raise SCRCError("support-determined rho is outside [0,1]")
    transition = (
        (1.0 - float(rho_fp32)) * np.eye(class_count, dtype=np.float64)
        + float(rho_fp32) * reciprocal
    )
    frozen_transition = np.ascontiguousarray(transition, dtype=np.dtype("<f4"))
    if (
        not np.isfinite(frozen_transition).all()
        or np.any(frozen_transition < 0.0)
        or _row_sum_error(frozen_transition, "transition")
        > _stochastic_tolerance(frozen_transition)
    ):
        raise SCRCError("transition is not finite row-stochastic")
    return rho_fp32, reciprocal, frozen_transition


@dataclass(frozen=True, slots=True)
class SCRCState:
    """Typed immutable D109 state containing Q[a,b], T[b,a], and rho."""

    registered_classes: tuple[str, ...]
    k_shot: int
    support_confusion_fp32: np.ndarray
    transition_fp32: np.ndarray
    rho_fp32: np.float32
    state_receipt_sha256: str

    def __post_init__(self) -> None:
        _validate_state(self)


def _resource_payload(state: SCRCState) -> dict[str, Any]:
    class_count = len(state.registered_classes)
    _rho, reciprocal, _transition = _transition_from_confusion(
        state.support_confusion_fp32
    )
    return {
        "schema": RESOURCE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "registered_class_count": class_count,
        "k_shot": state.k_shot,
        "support_row_count": class_count * state.k_shot,
        "trainable_parameters": 0,
        "support_only": True,
        "support_logit_value_count": class_count * class_count * state.k_shot,
        "support_softmax_count": class_count * state.k_shot,
        "support_confusion_value_count": class_count * class_count,
        "reciprocal_response_value_count": class_count * class_count,
        "transition_value_count": class_count * class_count,
        "query_transition_multiplications_per_row": class_count * class_count,
        "query_log_corrections_per_row": class_count,
        "query_fit_rows": 0,
        "query_state_updates": 0,
        "query_truth_access": False,
        "query_role_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "clean_sample_access": False,
        "source_sample_access": False,
        "support_confusion_orientation": "Q[a,b]=mean_{i:y_i=a}(softmax(g_i)[b])",
        "reciprocal_response_orientation": "R[b,a]=Q[a,b]/sum_l Q[l,b]",
        "transition_orientation": "T[b,a]=(1-rho)I[b,a]+rhoR[b,a]",
        "query_transform_orientation": "p_tilde[a]=sum_b p[b]T[b,a]",
        "reciprocal_response_reconstructed_from_q": True,
        "numeric_state_bytes": state.support_confusion_fp32.nbytes
        + state.transition_fp32.nbytes
        + np.dtype("<f4").itemsize,
        "rho_fp32_hex": float(state.rho_fp32).hex(),
        "softmax_exponent_floor_fp32_hex": _FLOAT32_EXPONENT_FLOOR.hex(),
        "query_log_probability_floor_fp32_hex": (
            _LOG_FLOAT32_EXPONENT_FLOOR.hex()
        ),
        "support_confusion_row_sum_max_abs_error_hex": _row_sum_error(
            state.support_confusion_fp32, "resource support confusion"
        ).hex(),
        "reciprocal_response_row_sum_max_abs_error_hex": _row_sum_error(
            reciprocal, "resource reciprocal response"
        ).hex(),
        "transition_row_sum_max_abs_error_hex": _row_sum_error(
            state.transition_fp32, "resource transition"
        ).hex(),
    }


def _state_payload(state: SCRCState) -> dict[str, Any]:
    return {
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "state_schema": STATE_SCHEMA,
        "registered_classes": list(state.registered_classes),
        "k_shot": state.k_shot,
        "rho_fp32_hex": float(state.rho_fp32).hex(),
        "arrays": {
            name: _array_receipt(getattr(state, name)) for name in _ARRAY_FIELDS
        },
        "resource": _resource_payload(state),
    }


def _expected_state_receipt(state: SCRCState) -> str:
    return _sha256_bytes(_canonical_bytes(_state_payload(state)))


def _validate_frozen_array(
    value: Any, name: str, shape: tuple[int, int]
) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype("<f4")
        or value.shape != shape
        or value.flags.writeable
        or not value.flags.c_contiguous
        or not np.isfinite(value).all()
    ):
        raise SCRCError(f"{name} typed immutable-state contract drift")
    return value


def _validate_state(state: SCRCState) -> None:
    if type(state) is not SCRCState:
        raise SCRCError("state must be an exact SCRCState")
    classes = _registry(state.registered_classes)
    if type(state.k_shot) is not int or state.k_shot not in ALLOWED_K:
        raise SCRCError("state K-shot is outside the frozen allowed set")
    shape = (len(classes), len(classes))
    confusion = _validate_frozen_array(
        state.support_confusion_fp32, "support_confusion_fp32", shape
    )
    transition = _validate_frozen_array(state.transition_fp32, "transition_fp32", shape)
    _validate_confusion(confusion, "state support confusion")
    expected_rho, _reciprocal, expected_transition = _transition_from_confusion(
        confusion
    )
    if type(state.rho_fp32) is not np.float32 or not np.isfinite(state.rho_fp32):
        raise SCRCError("state rho must be an exact finite float32")
    if state.rho_fp32.tobytes() != expected_rho.tobytes():
        raise SCRCError("state support-confusion/rho equation drift")
    if not np.array_equal(expected_transition, transition):
        raise SCRCError("state support-confusion/transition equation drift")
    _require_sha256(state.state_receipt_sha256, "state receipt")
    if state.state_receipt_sha256 != _expected_state_receipt(state):
        raise SCRCError("state receipt verification failed")


def _make_state(
    *,
    registered_classes: tuple[str, ...],
    k_shot: int,
    support_confusion_fp32: np.ndarray,
    transition_fp32: np.ndarray,
    rho_fp32: np.float32,
    expected_receipt: str | None = None,
) -> SCRCState:
    values = {
        "registered_classes": registered_classes,
        "k_shot": int(k_shot),
        "support_confusion_fp32": _freeze_array(
            support_confusion_fp32, np.dtype("<f4")
        ),
        "transition_fp32": _freeze_array(transition_fp32, np.dtype("<f4")),
        "rho_fp32": np.float32(rho_fp32),
    }
    provisional = object.__new__(SCRCState)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    receipt = _expected_state_receipt(provisional)
    if expected_receipt is not None and receipt != expected_receipt:
        raise SCRCError("wire state receipt verification failed")
    values["state_receipt_sha256"] = receipt
    return SCRCState(**values)


def build_scrc_state(
    support_logits: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
) -> SCRCState:
    """Freeze D109 from balanced legal support logits; query is not accepted."""

    classes = _registry(registered_classes)
    logits = _float32_logit_matrix(
        support_logits, "support_logits", class_count=len(classes)
    )
    labels = _labels(support_labels, len(logits))
    if any(label not in classes for label in labels):
        raise SCRCError("support labels must belong to registered_classes")
    counts = tuple(labels.count(class_name) for class_name in classes)
    if any(count < 1 for count in counts) or len(set(counts)) != 1:
        raise SCRCError("support must be strictly balanced K-shot over every class")
    k_shot = counts[0]
    if k_shot not in ALLOWED_K:
        raise SCRCError(f"support K must be one of {ALLOWED_K}")
    support_probabilities = _stable_softmax_matrix(logits, "support logits")
    confusion = _support_confusion(
        support_probabilities, labels, classes, k_shot
    )
    rho, _reciprocal, transition = _transition_from_confusion(confusion)
    return _make_state(
        registered_classes=classes,
        k_shot=k_shot,
        support_confusion_fp32=confusion,
        transition_fp32=transition,
        rho_fp32=rho,
    )


def apply_scrc_query(state: SCRCState, query_logits: np.ndarray) -> np.ndarray:
    """Apply frozen D109 to each query row independently over all classes."""

    if type(state) is not SCRCState:
        raise SCRCError("state must be an exact SCRCState")
    _validate_state(state)
    logits = _float32_logit_matrix(
        query_logits, "query_logits", class_count=len(state.registered_classes)
    )
    transition = state.transition_fp32.astype(np.float64, copy=False)
    identity = np.eye(len(state.registered_classes), dtype=np.dtype("<f4"))
    if np.array_equal(state.transition_fp32, identity):
        return np.ascontiguousarray(query_logits, dtype=np.float32)
    rows: list[np.ndarray] = []
    for row_index in range(len(logits)):
        log_probabilities = _true_log_softmax_row(
            logits[row_index], "query logits"
        )
        log_transformed = _log_transition_probabilities(
            log_probabilities, transition
        )
        # Logits are identifiable up to a row constant.  Canonicalize the
        # frozen h=g+log(p@T)-log(p) equation to max(h)=0 before float32
        # storage, then apply only the audited float32 probability floor.
        adjusted = np.maximum(
            log_transformed - float(np.max(log_transformed)),
            _LOG_FLOAT32_EXPONENT_FLOOR,
        )
        if not np.isfinite(adjusted).all():
            raise SCRCError("query reciprocal logit correction became non-finite")
        rows.append(adjusted)
    result = np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)
    if not np.isfinite(result).all():
        raise SCRCError("query reciprocal logits cannot be represented as float32")
    return result


def _wire_header(state: SCRCState) -> dict[str, Any]:
    return {
        "wire_schema": WIRE_SCHEMA,
        "array_order": list(_ARRAY_FIELDS),
        "state": _state_payload(state),
        "state_receipt_sha256": state.state_receipt_sha256,
    }


def serialize_scrc_state(state: SCRCState) -> bytes:
    if type(state) is not SCRCState:
        raise SCRCError("state must be an exact SCRCState")
    _validate_state(state)
    header = _canonical_bytes(_wire_header(state))
    body = b"".join(getattr(state, name).tobytes(order="C") for name in _ARRAY_FIELDS)
    wire = WIRE_MAGIC + struct.pack(">I", len(header)) + header + body
    if len(wire) > MAX_CANONICAL_WIRE_BYTES:
        raise SCRCError("canonical wire exceeds the frozen size bound")
    return wire


def _float32_from_hex(value: Any, name: str) -> np.float32:
    if type(value) is not str:
        raise SCRCError(f"{name} must be a canonical float32 hex string")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise SCRCError(f"{name} is not a float hex string") from error
    result = np.float32(parsed)
    if not np.isfinite(result) or float(result).hex() != value:
        raise SCRCError(f"{name} is not canonical finite float32")
    return result


def deserialize_scrc_state(
    payload: bytes, *, expected_wire_sha256: str | None = None
) -> SCRCState:
    if type(payload) is not bytes:
        raise SCRCError("wire payload must be exact bytes")
    if len(payload) > MAX_CANONICAL_WIRE_BYTES or len(payload) < len(WIRE_MAGIC) + 4:
        raise SCRCError("wire payload size is outside the frozen bound")
    if expected_wire_sha256 is not None:
        expected = _require_sha256(expected_wire_sha256, "expected wire SHA256")
        if _sha256_bytes(payload) != expected:
            raise SCRCError("wire SHA256 verification failed")
    if not payload.startswith(WIRE_MAGIC):
        raise SCRCError("wire magic mismatch")
    offset = len(WIRE_MAGIC)
    header_size = struct.unpack(">I", payload[offset : offset + 4])[0]
    offset += 4
    if header_size < 2 or offset + header_size > len(payload):
        raise SCRCError("wire header length is invalid")
    header_raw = payload[offset : offset + header_size]
    offset += header_size
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SCRCError("wire header is not canonical JSON") from error
    if _canonical_bytes(header) != header_raw:
        raise SCRCError("wire header JSON is not canonical")
    if type(header) is not dict or set(header) != {
        "wire_schema",
        "array_order",
        "state",
        "state_receipt_sha256",
    }:
        raise SCRCError("wire header fields drift")
    if header["wire_schema"] != WIRE_SCHEMA or header["array_order"] != list(
        _ARRAY_FIELDS
    ):
        raise SCRCError("wire schema or array order drift")
    state_data = header["state"]
    if type(state_data) is not dict or set(state_data) != {
        "candidate_id",
        "protocol_schema",
        "state_schema",
        "registered_classes",
        "k_shot",
        "rho_fp32_hex",
        "arrays",
        "resource",
    }:
        raise SCRCError("wire state fields drift")
    if (
        state_data["candidate_id"] != CANDIDATE_ID
        or state_data["protocol_schema"] != PROTOCOL_SCHEMA
        or state_data["state_schema"] != STATE_SCHEMA
        or type(state_data["k_shot"]) is not int
    ):
        raise SCRCError("wire state schema drift")
    classes = _registry(state_data["registered_classes"])
    if state_data["k_shot"] not in ALLOWED_K:
        raise SCRCError("wire state K-shot is outside the frozen allowed set")
    rho = _float32_from_hex(state_data["rho_fp32_hex"], "wire rho")
    arrays_data = state_data["arrays"]
    if type(arrays_data) is not dict or tuple(arrays_data) != tuple(
        sorted(_ARRAY_FIELDS)
    ):
        raise SCRCError("wire array receipts are not canonical")
    arrays: dict[str, np.ndarray] = {}
    for name in _ARRAY_FIELDS:
        receipt = arrays_data.get(name)
        if type(receipt) is not dict or set(receipt) != {"dtype", "shape", "sha256"}:
            raise SCRCError(f"{name} wire receipt drift")
        if receipt["dtype"] != np.dtype("<f4").str or receipt["shape"] != [
            len(classes),
            len(classes),
        ]:
            raise SCRCError(f"{name} wire descriptor drift")
        nbytes = len(classes) * len(classes) * np.dtype("<f4").itemsize
        if offset + nbytes > len(payload):
            raise SCRCError(f"{name} wire body is truncated")
        raw = payload[offset : offset + nbytes]
        offset += nbytes
        if _sha256_bytes(raw) != _require_sha256(receipt["sha256"], f"{name} SHA256"):
            raise SCRCError(f"{name} wire body receipt mismatch")
        arrays[name] = np.frombuffer(raw, dtype=np.dtype("<f4")).reshape(
            (len(classes), len(classes))
        )
    if offset != len(payload):
        raise SCRCError("wire has trailing bytes")
    state = _make_state(
        registered_classes=classes,
        k_shot=state_data["k_shot"],
        support_confusion_fp32=arrays["support_confusion_fp32"],
        transition_fp32=arrays["transition_fp32"],
        rho_fp32=rho,
        expected_receipt=_require_sha256(
            header["state_receipt_sha256"], "wire state receipt"
        ),
    )
    if _state_payload(state) != state_data:
        raise SCRCError("wire state payload is not canonical")
    return state


def scrc_resource_receipt(state: SCRCState) -> dict[str, Any]:
    """Return the frozen support/query resource receipt for D109 SCRC."""

    if type(state) is not SCRCState:
        raise SCRCError("state must be an exact SCRCState")
    _validate_state(state)
    payload = _resource_payload(state)
    receipt = dict(payload)
    receipt["resource_receipt_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    receipt["canonical_wire_bytes"] = len(serialize_scrc_state(state))
    return receipt


__all__ = [
    "ALLOWED_K",
    "CANDIDATE_ID",
    "MAX_CANONICAL_WIRE_BYTES",
    "MAX_REGISTERED_CLASSES",
    "PROTOCOL_SCHEMA",
    "RESOURCE_SCHEMA",
    "SCRCError",
    "SCRCState",
    "STATE_SCHEMA",
    "WIRE_SCHEMA",
    "apply_scrc_query",
    "build_scrc_state",
    "deserialize_scrc_state",
    "scrc_resource_receipt",
    "serialize_scrc_state",
]
