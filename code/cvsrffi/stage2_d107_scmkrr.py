"""D107 support-centered mean-embedding kernel ridge core.

The public build surface accepts target support, a frozen Stage2-B anchor, and
the sealed Phase1 ``tau``/``spectrum`` summaries.  The scorer accepts features
only.  It has no truth, semantic-role, quota, fitting, update, routing, or
global-assignment surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any, Sequence

import numpy as np


CANDIDATE_ID = "D107-SCMKRR/r1"
PROTOCOL_SCHEMA = "p2_min_v1"
STATE_SCHEMA = "cvs.phase2.d107.scmkrr_state.v1"
WIRE_SCHEMA = "cvs.phase2.d107.scmkrr_wire.v1"
RESOURCE_SCHEMA = "cvs.phase2.d107.scmkrr_resource.v1"
WIRE_MAGIC = b"CVSD107SCMKRR\x00\x01"
Z_DIM = 160
PHASE1_SUMMARY_DIM = 3
MAX_REGISTERED_CLASSES = 26
MAX_SUPPORT_PER_CLASS = 10
MAX_ANCHOR_ROWS = 60
MAX_REGISTRY_TOKEN_WIRE_BYTES = 70
MAX_CANONICAL_WIRE_BYTES = 32768
ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")
_CENTERED_ARMS = frozenset(("M_DA", "M_JOINT"))
_HEAD_ARMS = frozenset(("M_HEAD", "M_JOINT"))
_FLOAT64_EPSILON = np.finfo(np.float64).eps


class SCMKRRError(ValueError):
    """Raised when a D107 state, input, wire, or numerical invariant drifts."""


class SCMKRRTieError(SCMKRRError):
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
        raise SCMKRRError(f"{name} must be a lowercase SHA256")
    if any(character not in "0123456789abcdef" for character in value):
        raise SCMKRRError(f"{name} must be a lowercase SHA256")
    return value


def _require_text(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise SCMKRRError(f"{name} must be a non-empty exact string")
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > MAX_REGISTRY_TOKEN_WIRE_BYTES:
        raise SCMKRRError(f"{name} exceeds the sealed wire-token limit")
    return value


def _registry(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise SCMKRRError("registered_classes must be a sequence of exact strings")
    result = tuple(
        _require_text(value, f"registered_classes[{index}]")
        for index, value in enumerate(values)
    )
    if not 2 <= len(result) <= MAX_REGISTERED_CLASSES:
        raise SCMKRRError(
            f"registered_classes must contain 2..{MAX_REGISTERED_CLASSES} classes"
        )
    if len(set(result)) != len(result):
        raise SCMKRRError("registered_classes must be unique")
    return result


def _labels(values: Sequence[str], rows: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise SCMKRRError("labels must be a sequence of exact strings")
    result = tuple(
        _require_text(value, f"labels[{index}]") for index, value in enumerate(values)
    )
    if len(result) != rows:
        raise SCMKRRError("labels/support row count mismatch")
    return result


def _normalized_rows(value: Any, name: str, *, maximum_rows: int) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise SCMKRRError(f"{name} must be an exact float32 numpy matrix")
    if (
        value.ndim != 2
        or not 1 <= value.shape[0] <= maximum_rows
        or value.shape[1] != Z_DIM
        or not np.isfinite(value).all()
    ):
        raise SCMKRRError(
            f"{name} must be finite float32 [N,{Z_DIM}], 1<=N<={maximum_rows}"
        )
    rows = np.ascontiguousarray(value, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise SCMKRRError(f"{name} contains a zero-norm row")
    return np.ascontiguousarray(rows / norms[:, None], dtype=np.float64)


def _positive_summary(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != (PHASE1_SUMMARY_DIM,):
        raise SCMKRRError(
            f"{name} must be an exact [{PHASE1_SUMMARY_DIM}] numpy array"
        )
    result = np.ascontiguousarray(value, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise SCMKRRError(f"{name} must contain finite strictly positive values")
    return result


def _freeze_array(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    payload = array.tobytes(order="C")
    return np.frombuffer(payload, dtype=dtype).reshape(array.shape)


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _quantize_rows(rows: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    if rows.ndim != 2 or rows.shape[1] != Z_DIM or not np.isfinite(rows).all():
        raise SCMKRRError(f"{name} must be finite normalized rows")
    codes = np.empty(rows.shape, dtype=np.int8)
    scales = np.empty(len(rows), dtype=np.dtype("<f2"))
    for index, row in enumerate(rows):
        maximum = float(np.max(np.abs(row)))
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise SCMKRRError(f"{name}[{index}] cannot be quantized")
        scale = np.float16(maximum / 127.0)
        if not np.isfinite(scale) or float(scale) <= 0.0:
            raise SCMKRRError(f"{name}[{index}] quantization scale is invalid")
        quantized = np.rint(row / float(scale))
        codes[index] = np.clip(quantized, -127.0, 127.0).astype(np.int8)
        scales[index] = scale
    return _freeze_array(codes, np.dtype(np.int8)), _freeze_array(
        scales, np.dtype("<f2")
    )


def _decode_rows(codes: np.ndarray, scales: np.ndarray, name: str) -> np.ndarray:
    rows = codes.astype(np.float64) * scales.astype(np.float64)[:, None]
    norms = np.linalg.norm(rows, axis=1)
    if not np.isfinite(rows).all() or np.any(norms <= 0.0):
        raise SCMKRRError(f"{name} decode contains a zero or non-finite row")
    return np.ascontiguousarray(rows / norms[:, None], dtype=np.float64)


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(np.square(left), axis=1)[:, None]
        + np.sum(np.square(right), axis=1)[None, :]
        - 2.0 * (left @ right.T)
    )
    return np.maximum(distances, 0.0)


def _kernel(left: np.ndarray, right: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-_squared_distances(left, right) / bandwidth)


def _bandwidth(prototypes: np.ndarray) -> float:
    distances = _squared_distances(prototypes, prototypes)
    upper = distances[np.triu_indices(len(prototypes), 1)]
    result = float(np.median(upper))
    if not math.isfinite(result):
        raise SCMKRRError("prototype bandwidth is non-finite")
    return max(result, _FLOAT64_EPSILON)


def _centered_components(
    prototypes: np.ndarray, anchors: np.ndarray, bandwidth: float
) -> tuple[np.ndarray, np.ndarray, float]:
    prototype_kernel = _kernel(prototypes, prototypes, bandwidth)
    prototype_anchor = _kernel(prototypes, anchors, bandwidth)
    anchor_kernel = _kernel(anchors, anchors, bandwidth)
    prototype_anchor_mean = np.mean(prototype_anchor, axis=1, dtype=np.float64)
    anchor_grand_mean = float(np.mean(anchor_kernel, dtype=np.float64))
    centered = (
        prototype_kernel
        - prototype_anchor_mean[:, None]
        - prototype_anchor_mean[None, :]
        + anchor_grand_mean
    )
    centered = 0.5 * (centered + centered.T)
    return centered, prototype_anchor_mean, anchor_grand_mean


def _phase1_summary_digest(tau: np.ndarray, spectrum: np.ndarray) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "tau": _array_receipt(tau.astype(np.dtype("<f8"))),
                "spectrum": _array_receipt(spectrum.astype(np.dtype("<f8"))),
            }
        )
    )


@dataclass(frozen=True, slots=True)
class SCMKRRState:
    arm: str
    registered_classes: tuple[str, ...]
    anchor_codes_qint8: np.ndarray
    anchor_scales_fp16: np.ndarray
    prototype_codes_qint8: np.ndarray
    prototype_scales_fp16: np.ndarray
    bandwidth: float
    rho: float
    regularization: float
    head_coefficients_fp16: np.ndarray
    prototype_anchor_mean_fp16: np.ndarray
    anchor_grand_mean: float
    phase1_summary_sha256: str
    state_receipt_sha256: str

    def __post_init__(self) -> None:
        _validate_state(self)


_ARRAY_FIELDS = (
    "anchor_codes_qint8",
    "anchor_scales_fp16",
    "prototype_codes_qint8",
    "prototype_scales_fp16",
    "head_coefficients_fp16",
    "prototype_anchor_mean_fp16",
)


def _state_payload(state: SCMKRRState) -> dict[str, Any]:
    return {
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "state_schema": STATE_SCHEMA,
        "arm": state.arm,
        "registered_classes": list(state.registered_classes),
        "bandwidth_hex": float(state.bandwidth).hex(),
        "rho_hex": float(state.rho).hex(),
        "regularization_hex": float(state.regularization).hex(),
        "anchor_grand_mean_hex": float(state.anchor_grand_mean).hex(),
        "phase1_summary_sha256": state.phase1_summary_sha256,
        "arrays": {
            name: _array_receipt(getattr(state, name)) for name in _ARRAY_FIELDS
        },
    }


def _expected_state_receipt(state: SCMKRRState) -> str:
    return _sha256_bytes(_canonical_bytes(_state_payload(state)))


def _validate_state(state: SCMKRRState) -> None:
    if state.arm not in ARMS:
        raise SCMKRRError("state arm is not frozen")
    classes = _registry(state.registered_classes)
    class_count = len(classes)
    expected = {
        "anchor_codes_qint8": (np.dtype(np.int8), (None, Z_DIM)),
        "anchor_scales_fp16": (np.dtype("<f2"), (None,)),
        "prototype_codes_qint8": (np.dtype(np.int8), (class_count, Z_DIM)),
        "prototype_scales_fp16": (np.dtype("<f2"), (class_count,)),
        "prototype_anchor_mean_fp16": (np.dtype("<f2"), (class_count,)),
    }
    for name, (dtype, shape) in expected.items():
        value = getattr(state, name)
        if not isinstance(value, np.ndarray) or value.dtype != dtype:
            raise SCMKRRError(f"{name} dtype drift")
        if shape[0] is None:
            if value.ndim != len(shape) or value.shape[1:] != shape[1:]:
                raise SCMKRRError(f"{name} shape drift")
        elif value.shape != shape:
            raise SCMKRRError(f"{name} shape drift")
        if value.flags.writeable or not value.flags.c_contiguous:
            raise SCMKRRError(f"{name} must be immutable and C-contiguous")
    anchor_rows = len(state.anchor_codes_qint8)
    if not 1 <= anchor_rows <= MAX_ANCHOR_ROWS:
        raise SCMKRRError("anchor row count drift")
    if state.anchor_scales_fp16.shape != (anchor_rows,):
        raise SCMKRRError("anchor code/scale count drift")
    if np.any(state.anchor_codes_qint8 == -128) or np.any(
        state.prototype_codes_qint8 == -128
    ):
        raise SCMKRRError("qint8 codes must remain in [-127,127]")
    if (
        not np.isfinite(state.anchor_scales_fp16).all()
        or np.any(state.anchor_scales_fp16 <= 0)
        or not np.isfinite(state.prototype_scales_fp16).all()
        or np.any(state.prototype_scales_fp16 <= 0)
        or not np.isfinite(state.prototype_anchor_mean_fp16).all()
    ):
        raise SCMKRRError("state FP16 arrays are invalid")
    expected_head_shape = (class_count, class_count) if state.arm in _HEAD_ARMS else (0, 0)
    if (
        not isinstance(state.head_coefficients_fp16, np.ndarray)
        or state.head_coefficients_fp16.dtype != np.dtype("<f2")
        or state.head_coefficients_fp16.shape != expected_head_shape
        or state.head_coefficients_fp16.flags.writeable
        or not state.head_coefficients_fp16.flags.c_contiguous
        or not np.isfinite(state.head_coefficients_fp16).all()
    ):
        raise SCMKRRError("head coefficient state drift")
    for value, name in (
        (state.bandwidth, "bandwidth"),
        (state.rho, "rho"),
        (state.regularization, "regularization"),
        (state.anchor_grand_mean, "anchor_grand_mean"),
    ):
        if type(value) is not float or not math.isfinite(value):
            raise SCMKRRError(f"{name} must be a finite binary64 scalar")
    if state.bandwidth < _FLOAT64_EPSILON:
        raise SCMKRRError("bandwidth is below the machine-epsilon totalization")
    if not 0.0 < state.rho < 1.0 or state.regularization <= 0.0:
        raise SCMKRRError("ridge state is outside its positive domain")
    _require_sha256(state.phase1_summary_sha256, "phase1 summary receipt")
    _require_sha256(state.state_receipt_sha256, "state receipt")
    if state.state_receipt_sha256 != _expected_state_receipt(state):
        raise SCMKRRError("state receipt verification failed")


def _make_state(
    *,
    arm: str,
    registered_classes: tuple[str, ...],
    anchor_codes_qint8: np.ndarray,
    anchor_scales_fp16: np.ndarray,
    prototype_codes_qint8: np.ndarray,
    prototype_scales_fp16: np.ndarray,
    bandwidth: float,
    rho: float,
    regularization: float,
    head_coefficients_fp16: np.ndarray,
    prototype_anchor_mean_fp16: np.ndarray,
    anchor_grand_mean: float,
    phase1_summary_sha256: str,
    expected_receipt: str | None = None,
) -> SCMKRRState:
    values = {
        "arm": arm,
        "registered_classes": registered_classes,
        "anchor_codes_qint8": _freeze_array(anchor_codes_qint8, np.dtype(np.int8)),
        "anchor_scales_fp16": _freeze_array(anchor_scales_fp16, np.dtype("<f2")),
        "prototype_codes_qint8": _freeze_array(
            prototype_codes_qint8, np.dtype(np.int8)
        ),
        "prototype_scales_fp16": _freeze_array(
            prototype_scales_fp16, np.dtype("<f2")
        ),
        "bandwidth": float(bandwidth),
        "rho": float(rho),
        "regularization": float(regularization),
        "head_coefficients_fp16": _freeze_array(
            head_coefficients_fp16, np.dtype("<f2")
        ),
        "prototype_anchor_mean_fp16": _freeze_array(
            prototype_anchor_mean_fp16, np.dtype("<f2")
        ),
        "anchor_grand_mean": float(anchor_grand_mean),
        "phase1_summary_sha256": phase1_summary_sha256,
    }
    provisional = object.__new__(SCMKRRState)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    receipt = _sha256_bytes(_canonical_bytes(_state_payload(provisional)))
    if expected_receipt is not None and receipt != expected_receipt:
        raise SCMKRRError("wire state receipt verification failed")
    values["state_receipt_sha256"] = receipt
    return SCMKRRState(**values)


def build_scmkrr_state(
    support_signed: np.ndarray,
    labels: Sequence[str],
    registered_classes: Sequence[str],
    anchor_signed: np.ndarray,
    tau: np.ndarray,
    spectrum: np.ndarray,
    arm: str,
) -> SCMKRRState:
    """Build one support-only immutable D107 state; no query argument exists."""

    if type(arm) is not str or arm not in ARMS:
        raise SCMKRRError(f"arm must be one of {ARMS}")
    classes = _registry(registered_classes)
    maximum_support_rows = len(classes) * MAX_SUPPORT_PER_CLASS
    support = _normalized_rows(
        support_signed, "support_signed", maximum_rows=maximum_support_rows
    )
    support_labels = _labels(labels, len(support))
    if any(label not in classes for label in support_labels):
        raise SCMKRRError("support labels must belong to registered_classes")
    counts = tuple(support_labels.count(class_name) for class_name in classes)
    if any(count < 1 for count in counts) or len(set(counts)) != 1:
        raise SCMKRRError("formal support must be balanced K-shot over all classes")
    if counts[0] > MAX_SUPPORT_PER_CLASS:
        raise SCMKRRError("support K exceeds the frozen maximum")
    anchors = _normalized_rows(
        anchor_signed, "anchor_signed", maximum_rows=MAX_ANCHOR_ROWS
    )
    tau_values = _positive_summary(tau, "tau")
    spectrum_values = _positive_summary(spectrum, "spectrum")
    if tau_values.shape != spectrum_values.shape:
        raise SCMKRRError("tau and spectrum must have the same shape")

    prototype_rows: list[np.ndarray] = []
    for class_name in classes:
        indices = [
            index for index, label in enumerate(support_labels) if label == class_name
        ]
        mean = np.mean(support[indices], axis=0, dtype=np.float64)
        norm = float(np.linalg.norm(mean))
        if not math.isfinite(norm) or norm <= 0.0:
            raise SCMKRRError("a registered class has a zero-norm mean prototype")
        prototype_rows.append(mean / norm)
    prototypes = np.ascontiguousarray(np.stack(prototype_rows), dtype=np.float64)

    anchor_codes, anchor_scales = _quantize_rows(anchors, "anchor_signed")
    prototype_codes, prototype_scales = _quantize_rows(prototypes, "prototypes")
    closed_anchors = _decode_rows(anchor_codes, anchor_scales, "anchor")
    closed_prototypes = _decode_rows(
        prototype_codes, prototype_scales, "prototypes"
    )
    bandwidth = _bandwidth(closed_prototypes)
    base_kernel = _kernel(closed_prototypes, closed_prototypes, bandwidth)
    _centered_kernel, prototype_anchor_mean, anchor_grand_mean = _centered_components(
        closed_prototypes, closed_anchors, bandwidth
    )
    anchor_mean_fp16 = prototype_anchor_mean.astype(np.dtype("<f2"))
    if not np.isfinite(anchor_mean_fp16).all():
        raise SCMKRRError("prototype anchor means overflow FP16")
    deployed_anchor_mean = anchor_mean_fp16.astype(np.float64)
    centered_kernel = (
        base_kernel
        - deployed_anchor_mean[:, None]
        - deployed_anchor_mean[None, :]
        + anchor_grand_mean
    )
    centered_kernel = 0.5 * (centered_kernel + centered_kernel.T)

    ratios = tau_values / (tau_values + spectrum_values)
    rho = float(np.median(ratios))
    if not 0.0 < rho < 1.0:
        raise SCMKRRError("sealed Phase1 ridge ratio is outside (0,1)")
    active_kernel = centered_kernel if arm in _CENTERED_ARMS else base_kernel
    trace_scale = max(
        float(np.trace(active_kernel)) / float(len(classes)), _FLOAT64_EPSILON
    )
    regularization = (rho / (1.0 - rho)) * trace_scale
    if not math.isfinite(regularization) or regularization <= 0.0:
        raise SCMKRRError("ridge regularization is not finite and positive")

    if arm in _HEAD_ARMS:
        target = np.eye(len(classes), dtype=np.float64) - np.full(
            (len(classes), len(classes)), 1.0 / float(len(classes)), dtype=np.float64
        )
        try:
            coefficients = np.linalg.solve(
                active_kernel
                + regularization * np.eye(len(classes), dtype=np.float64),
                target,
            )
        except np.linalg.LinAlgError as error:
            raise SCMKRRError("simplex kernel-ridge solve failed") from error
        head_coefficients = coefficients.astype(np.dtype("<f2"))
        if not np.isfinite(head_coefficients).all():
            raise SCMKRRError("simplex kernel-ridge coefficients overflow FP16")
    else:
        head_coefficients = np.empty((0, 0), dtype=np.dtype("<f2"))

    return _make_state(
        arm=arm,
        registered_classes=classes,
        anchor_codes_qint8=anchor_codes,
        anchor_scales_fp16=anchor_scales,
        prototype_codes_qint8=prototype_codes,
        prototype_scales_fp16=prototype_scales,
        bandwidth=bandwidth,
        rho=rho,
        regularization=regularization,
        head_coefficients_fp16=head_coefficients,
        prototype_anchor_mean_fp16=anchor_mean_fp16,
        anchor_grand_mean=anchor_grand_mean,
        phase1_summary_sha256=_phase1_summary_digest(tau_values, spectrum_values),
    )


def score_scmkrr_query(state: SCMKRRState, query_signed: np.ndarray) -> np.ndarray:
    """Score independent query rows against every registered class."""

    if type(state) is not SCMKRRState:
        raise SCMKRRError("state must be an exact SCMKRRState")
    _validate_state(state)
    maximum_query_rows = (
        int(query_signed.shape[0])
        if isinstance(query_signed, np.ndarray)
        and query_signed.ndim == 2
        and query_signed.shape[0] >= 1
        else 1
    )
    queries = _normalized_rows(
        query_signed, "query_signed", maximum_rows=maximum_query_rows
    )
    anchors = _decode_rows(
        state.anchor_codes_qint8, state.anchor_scales_fp16, "anchor"
    )
    prototypes = _decode_rows(
        state.prototype_codes_qint8, state.prototype_scales_fp16, "prototypes"
    )
    scores = _kernel(queries, prototypes, state.bandwidth)
    if state.arm in _CENTERED_ARMS:
        query_anchor_mean = np.mean(
            _kernel(queries, anchors, state.bandwidth), axis=1, dtype=np.float64
        )
        scores = (
            scores
            - query_anchor_mean[:, None]
            - state.prototype_anchor_mean_fp16.astype(np.float64)[None, :]
            + state.anchor_grand_mean
        )
    if state.arm in _HEAD_ARMS:
        scores = scores @ state.head_coefficients_fp16.astype(np.float64)
    result = np.ascontiguousarray(scores, dtype=np.float32)
    if not np.isfinite(result).all():
        raise SCMKRRError("query scores are non-finite")
    for row in result:
        maximum = np.max(row)
        if int(np.count_nonzero(row == maximum)) != 1:
            raise SCMKRRTieError(SCMKRRTieError.code)
    return result


def _wire_header(state: SCMKRRState) -> dict[str, Any]:
    return {
        "wire_schema": WIRE_SCHEMA,
        "array_order": list(_ARRAY_FIELDS),
        "state": _state_payload(state),
        "state_receipt_sha256": state.state_receipt_sha256,
    }


def serialize_scmkrr_state(state: SCMKRRState) -> bytes:
    if type(state) is not SCMKRRState:
        raise SCMKRRError("state must be an exact SCMKRRState")
    _validate_state(state)
    header = _canonical_bytes(_wire_header(state))
    body = b"".join(
        getattr(state, name).tobytes(order="C") for name in _ARRAY_FIELDS
    )
    wire = WIRE_MAGIC + struct.pack(">I", len(header)) + header + body
    if len(wire) > MAX_CANONICAL_WIRE_BYTES:
        raise SCMKRRError("canonical wire exceeds the frozen size bound")
    return wire


def _parse_float_hex(value: Any, name: str) -> float:
    if type(value) is not str:
        raise SCMKRRError(f"{name} must use canonical binary64 hex")
    try:
        result = float.fromhex(value)
    except ValueError as error:
        raise SCMKRRError(f"{name} is not canonical binary64 hex") from error
    if not math.isfinite(result) or result.hex() != value:
        raise SCMKRRError(f"{name} is not canonical finite binary64 hex")
    return result


def deserialize_scmkrr_state(
    payload: bytes, *, expected_wire_sha256: str | None = None
) -> SCMKRRState:
    if type(payload) is not bytes:
        raise SCMKRRError("wire payload must be exact bytes")
    if len(payload) > MAX_CANONICAL_WIRE_BYTES or len(payload) < len(WIRE_MAGIC) + 4:
        raise SCMKRRError("wire payload size is outside the frozen bound")
    if expected_wire_sha256 is not None:
        expected = _require_sha256(expected_wire_sha256, "expected wire SHA256")
        if _sha256_bytes(payload) != expected:
            raise SCMKRRError("wire SHA256 verification failed")
    if not payload.startswith(WIRE_MAGIC):
        raise SCMKRRError("wire magic mismatch")
    offset = len(WIRE_MAGIC)
    header_size = struct.unpack(">I", payload[offset : offset + 4])[0]
    offset += 4
    if header_size < 2 or offset + header_size > len(payload):
        raise SCMKRRError("wire header length is invalid")
    header_raw = payload[offset : offset + header_size]
    offset += header_size
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SCMKRRError("wire header is not canonical JSON") from error
    if _canonical_bytes(header) != header_raw:
        raise SCMKRRError("wire header JSON is not canonical")
    if type(header) is not dict or set(header) != {
        "wire_schema",
        "array_order",
        "state",
        "state_receipt_sha256",
    }:
        raise SCMKRRError("wire header fields drift")
    if header["wire_schema"] != WIRE_SCHEMA or header["array_order"] != list(
        _ARRAY_FIELDS
    ):
        raise SCMKRRError("wire schema or array order drift")
    state_data = header["state"]
    if type(state_data) is not dict or set(state_data) != {
        "candidate_id",
        "protocol_schema",
        "state_schema",
        "arm",
        "registered_classes",
        "bandwidth_hex",
        "rho_hex",
        "regularization_hex",
        "anchor_grand_mean_hex",
        "phase1_summary_sha256",
        "arrays",
    }:
        raise SCMKRRError("wire state fields drift")
    if (
        state_data["candidate_id"] != CANDIDATE_ID
        or state_data["protocol_schema"] != PROTOCOL_SCHEMA
        or state_data["state_schema"] != STATE_SCHEMA
    ):
        raise SCMKRRError("wire state schema drift")
    classes = _registry(state_data["registered_classes"])
    arrays_data = state_data["arrays"]
    if type(arrays_data) is not dict or tuple(arrays_data) != tuple(sorted(_ARRAY_FIELDS)):
        raise SCMKRRError("wire array receipts are not canonical")
    arrays: dict[str, np.ndarray] = {}
    for name in _ARRAY_FIELDS:
        receipt = arrays_data.get(name)
        if type(receipt) is not dict or set(receipt) != {"dtype", "shape", "sha256"}:
            raise SCMKRRError(f"{name} wire receipt drift")
        try:
            dtype = np.dtype(receipt["dtype"])
            shape = tuple(receipt["shape"])
        except (TypeError, ValueError) as error:
            raise SCMKRRError(f"{name} wire descriptor is invalid") from error
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise SCMKRRError(f"{name} wire shape is invalid")
        count = math.prod(shape)
        nbytes = count * dtype.itemsize
        if offset + nbytes > len(payload):
            raise SCMKRRError(f"{name} wire body is truncated")
        raw = payload[offset : offset + nbytes]
        offset += nbytes
        if _sha256_bytes(raw) != _require_sha256(receipt["sha256"], f"{name} SHA256"):
            raise SCMKRRError(f"{name} wire body receipt mismatch")
        arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(shape)
    if offset != len(payload):
        raise SCMKRRError("wire has trailing bytes")
    state = _make_state(
        arm=state_data["arm"],
        registered_classes=classes,
        anchor_codes_qint8=arrays["anchor_codes_qint8"],
        anchor_scales_fp16=arrays["anchor_scales_fp16"],
        prototype_codes_qint8=arrays["prototype_codes_qint8"],
        prototype_scales_fp16=arrays["prototype_scales_fp16"],
        bandwidth=_parse_float_hex(state_data["bandwidth_hex"], "bandwidth"),
        rho=_parse_float_hex(state_data["rho_hex"], "rho"),
        regularization=_parse_float_hex(
            state_data["regularization_hex"], "regularization"
        ),
        head_coefficients_fp16=arrays["head_coefficients_fp16"],
        prototype_anchor_mean_fp16=arrays["prototype_anchor_mean_fp16"],
        anchor_grand_mean=_parse_float_hex(
            state_data["anchor_grand_mean_hex"], "anchor_grand_mean"
        ),
        phase1_summary_sha256=state_data["phase1_summary_sha256"],
        expected_receipt=_require_sha256(
            header["state_receipt_sha256"], "wire state receipt"
        ),
    )
    if _state_payload(state) != state_data:
        raise SCMKRRError("wire state payload is not canonical")
    return state


def scmkrr_resource_summary(state: SCMKRRState) -> dict[str, int | str]:
    if type(state) is not SCMKRRState:
        raise SCMKRRError("state must be an exact SCMKRRState")
    _validate_state(state)
    numeric_bytes = sum(getattr(state, name).nbytes for name in _ARRAY_FIELDS)
    anchor_rows = len(state.anchor_codes_qint8)
    class_count = len(state.registered_classes)
    query_mac_upper_bound = (anchor_rows + class_count) * Z_DIM
    if state.arm in _HEAD_ARMS:
        query_mac_upper_bound += class_count * class_count
    return {
        "schema": RESOURCE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "arm": state.arm,
        "anchor_rows": anchor_rows,
        "registered_class_count": class_count,
        "numeric_state_bytes": int(numeric_bytes),
        "canonical_wire_bytes": len(serialize_scmkrr_state(state)),
        "query_mac_upper_bound": int(query_mac_upper_bound),
    }


__all__ = [
    "ARMS",
    "CANDIDATE_ID",
    "MAX_CANONICAL_WIRE_BYTES",
    "PHASE1_SUMMARY_DIM",
    "SCMKRRError",
    "SCMKRRTieError",
    "SCMKRRState",
    "build_scmkrr_state",
    "deserialize_scmkrr_state",
    "score_scmkrr_query",
    "scmkrr_resource_summary",
    "serialize_scmkrr_state",
]
