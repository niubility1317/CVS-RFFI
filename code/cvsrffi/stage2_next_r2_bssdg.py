"""Native D92-Lite Student-t successor head.

This module intentionally contains only the classification head.  It consumes
the current state's canonical signed 160-dimensional support/query features;
the CVFR adapter and the matrix runner own representation construction and
state binding.  The implementation has no query-side fitting or state update,
and an exact prediction tie is never resolved by an external key.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "cvs.phase2.next_r2.bssdg160.state.v1"
BINDING_SCHEMA = "cvs.phase2.next_r2.bssdg160.binding.v1"
PROTOCOL_SCHEMA = "p2_min_v1"
CANDIDATE_ID = "D92-Lite-ST successor / BSSDG-160"
FEATURE_WIDTH = 160
ALLOWED_K = (1, 5)
STATE_LIMIT_BYTES = 6_144
FP16_MIN_NORMAL = float(np.finfo(np.float16).tiny)
FP16_MAX = float(np.finfo(np.float16).max)


class BSSDGError(ValueError):
    """Base error for a fail-closed BSSDG state or prediction."""


class BSSDGWireError(BSSDGError):
    """Raised when the compact FP16/int8 wire state is not representable."""


class BSSDGDuplicateFunctionError(BSSDGError):
    """Raised when two classes compile to exactly the same score function."""


class BSSDGExactTieError(BSSDGError):
    """Raised when a unique prediction cannot be made without a tie key."""


def _canonical_json(value: Any) -> bytes:
    """Encode JSON-compatible values with one stable byte representation."""

    return json.dumps(
        _json_safe(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported JSON value {type(value)!r}")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_sha(value: str, name: str, *, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return value
    if type(value) is not str or len(value) != 64:
        raise BSSDGError(f"{name} must be a SHA256 hex string")
    try:
        int(value, 16)
    except ValueError as error:
        raise BSSDGError(f"{name} must be a SHA256 hex string") from error
    return value


def _readonly(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _registry(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise BSSDGError(f"{name} must be a sequence of class handles")
    result = tuple(str(value) for value in values)
    if not result or len(set(result)) != len(result) or any(not item for item in result):
        raise BSSDGError(f"{name} must contain unique non-empty handles")
    return result


def _rows(value: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise BSSDGError(f"{name} must be numeric float32 rows") from error
    if (
        result.ndim != 2
        or result.shape[1] != FEATURE_WIDTH
        or (not allow_empty and result.shape[0] < 1)
        or not np.isfinite(result).all()
    ):
        raise BSSDGError(f"{name} must be finite [N,{FEATURE_WIDTH}] rows")
    return np.ascontiguousarray(result, dtype=np.float32)


def _labels(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise BSSDGError(f"{name} must be a sequence of labels")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise BSSDGError(f"{name} contains an empty label")
    return result


def validate_signed_fp16(value: Any, name: str = "value") -> np.ndarray:
    """Return a signed FP16 copy; zero is legal and nonzero values are normal."""

    raw = np.asarray(value, dtype=np.float32)
    if not np.isfinite(raw).all():
        raise BSSDGWireError(f"{name} contains a non-finite value")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        stored = raw.astype(np.float16)
    decoded = stored.astype(np.float32)
    if not np.isfinite(decoded).all():
        raise BSSDGWireError(f"{name} overflows FP16")
    nonzero = decoded != np.float32(0.0)
    if np.any(np.abs(decoded[nonzero]) < np.float32(FP16_MIN_NORMAL)):
        raise BSSDGWireError(f"{name} contains an FP16 subnormal")
    if np.any(np.abs(decoded) > np.float32(FP16_MAX)):
        raise BSSDGWireError(f"{name} exceeds the FP16 normal range")
    return np.ascontiguousarray(stored, dtype=np.float16)


def validate_positive_fp16(value: Any, name: str = "value") -> np.ndarray:
    """Return a positive normal FP16 copy; zero, subnormal, and overflow fail."""

    raw = np.asarray(value, dtype=np.float32)
    if not np.isfinite(raw).all() or np.any(raw <= np.float32(0.0)):
        raise BSSDGWireError(f"{name} must be positive and finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        stored = raw.astype(np.float16)
    decoded = stored.astype(np.float32)
    if (
        not np.isfinite(decoded).all()
        or np.any(decoded <= np.float32(0.0))
        or np.any(decoded < np.float32(FP16_MIN_NORMAL))
        or np.any(decoded > np.float32(FP16_MAX))
    ):
        raise BSSDGWireError(f"{name} contains a non-normal FP16 value")
    return np.ascontiguousarray(stored, dtype=np.float16)


@dataclass(frozen=True, slots=True)
class BSSDGBinding:
    """Typed binding for one DA/registration state."""

    state_name: str
    registration_name: str
    canonical_sha256: str = ""
    protocol_schema: str = PROTOCOL_SCHEMA
    schema: str = BINDING_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != BINDING_SCHEMA
            or self.protocol_schema != PROTOCOL_SCHEMA
            or type(self.state_name) is not str
            or not self.state_name
            or type(self.registration_name) is not str
            or not self.registration_name
        ):
            raise BSSDGError("BSSDG binding schema or state name drift")
        _validate_sha(self.canonical_sha256, "canonical_sha256", allow_empty=True)


def _binding(value: BSSDGBinding | Mapping[str, Any] | None) -> BSSDGBinding:
    if value is None:
        return BSSDGBinding(state_name="UNSPECIFIED", registration_name="UNSPECIFIED")
    if isinstance(value, BSSDGBinding):
        value.__post_init__()
        return value
    if isinstance(value, Mapping):
        try:
            result = BSSDGBinding(
                state_name=str(value["state_name"]),
                registration_name=str(value["registration_name"]),
                canonical_sha256=str(value.get("canonical_sha256", "")),
                protocol_schema=str(value.get("protocol_schema", PROTOCOL_SCHEMA)),
                schema=str(value.get("schema", BINDING_SCHEMA)),
            )
        except KeyError as error:
            raise BSSDGError("binding is missing state_name or registration_name") from error
        return result
    raise BSSDGError("binding must be a BSSDGBinding or mapping")


@dataclass(frozen=True, slots=True)
class BSSDGState:
    """Immutable compact BSSDG state."""

    classes: tuple[str, ...]
    active_k: int
    class_means_qint8: np.ndarray
    class_scales_fp16: np.ndarray
    logrho_fp16: np.ndarray
    intercept_fp16: np.ndarray
    v0_fp16: np.ndarray
    binding: BSSDGBinding
    fit_receipt: Mapping[str, Any]
    resource_receipt: Mapping[str, Any]
    state_sha256: str
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        classes = _registry(self.classes, "classes")
        if (
            self.schema != SCHEMA
            or type(self.active_k) is not int
            or self.active_k not in ALLOWED_K
            or type(self.binding) is not BSSDGBinding
            or len(classes) < 2
        ):
            raise BSSDGError("BSSDG state registry/K/binding drift")
        self.binding.__post_init__()
        means = np.asarray(self.class_means_qint8)
        scales = np.asarray(self.class_scales_fp16)
        logrho = np.asarray(self.logrho_fp16)
        intercept = np.asarray(self.intercept_fp16)
        v0 = np.asarray(self.v0_fp16)
        if (
            means.dtype != np.int8
            or means.shape != (len(classes), FEATURE_WIDTH)
            or np.any(means == np.int8(-128))
            or scales.dtype != np.float16
            or scales.shape != (len(classes),)
            or logrho.dtype != np.float16
            or logrho.shape != (len(classes),)
            or intercept.dtype != np.float16
            or intercept.shape != (len(classes),)
            or v0.dtype != np.float16
            or v0.shape != (FEATURE_WIDTH,)
        ):
            raise BSSDGWireError("BSSDG compact array shape or dtype drift")
        validate_positive_fp16(scales, "class_scales_fp16")
        validate_signed_fp16(logrho, "logrho_fp16")
        validate_signed_fp16(intercept, "intercept_fp16")
        validate_positive_fp16(v0, "v0_fp16")
        rho = np.exp(logrho.astype(np.float64))
        if not np.isfinite(rho).all() or np.any(rho <= 0.0):
            raise BSSDGWireError("exp(logrho_fp16) is not finite and positive")
        expected_intercept = (
            -np.float32(0.5 * FEATURE_WIDTH) * logrho.astype(np.float32)
        ).astype(np.float16)
        if not np.array_equal(expected_intercept, intercept):
            raise BSSDGWireError("intercept_fp16 is inconsistent with logrho_fp16")
        if not isinstance(self.fit_receipt, Mapping) or not isinstance(
            self.resource_receipt, Mapping
        ):
            raise BSSDGError("BSSDG receipts must be mappings")
        for key in ("fit_analytic_ops", "query_analytic_ops_per_row", "state_bytes"):
            try:
                if int(self.resource_receipt[key]) < 0:
                    raise BSSDGError(f"resource receipt {key} must be non-negative")
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise BSSDGError(f"resource receipt lacks {key}") from error
        _assert_unique_function_tuples(means, scales, logrho, v0)
        if self.state_sha256:
            _validate_sha(self.state_sha256, "state_sha256")
            if _state_digest(self) != self.state_sha256:
                raise BSSDGError("BSSDG state SHA256 receipt drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "class_means_qint8", _readonly(means, np.dtype(np.int8)))
        object.__setattr__(self, "class_scales_fp16", _readonly(scales, np.dtype(np.float16)))
        object.__setattr__(self, "logrho_fp16", _readonly(logrho, np.dtype(np.float16)))
        object.__setattr__(self, "intercept_fp16", _readonly(intercept, np.dtype(np.float16)))
        object.__setattr__(self, "v0_fp16", _readonly(v0, np.dtype(np.float16)))
        object.__setattr__(self, "fit_receipt", dict(self.fit_receipt))
        object.__setattr__(self, "resource_receipt", dict(self.resource_receipt))

    @property
    def numeric_state_bytes(self) -> int:
        return int(
            self.class_means_qint8.nbytes
            + self.class_scales_fp16.nbytes
            + self.logrho_fp16.nbytes
            + self.intercept_fp16.nbytes
            + self.v0_fp16.nbytes
        )

    @property
    def k_shot(self) -> int:
        """Stable spelling for matrix/report consumers."""

        return self.active_k

    @property
    def full_state_bytes(self) -> int:
        return self.state_bytes

    @property
    def receipt_sha256(self) -> str:
        return self.state_sha256

    @property
    def candidate_id(self) -> str:
        return CANDIDATE_ID

    @property
    def state_bytes(self) -> int:
        return int(self.resource_receipt["state_bytes"])

    @property
    def wire_bytes(self) -> int:
        return len(serialize_bssdg_state(self))

    @property
    def decoded_class_means(self) -> np.ndarray:
        return np.ascontiguousarray(
            self.class_means_qint8.astype(np.float32)
            * self.class_scales_fp16.astype(np.float32)[:, None],
            dtype=np.float32,
        )

    @property
    def decoded_v0(self) -> np.ndarray:
        return self.v0_fp16.astype(np.float32, copy=True)

    @property
    def decoded_rho(self) -> np.ndarray:
        return np.exp(self.logrho_fp16.astype(np.float32)).astype(np.float32)


def _array_wire(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    raw = array.tobytes(order="C")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data_b64": base64.b64encode(raw).decode("ascii"),
    }


def _array_from_wire(value: Mapping[str, Any], name: str) -> np.ndarray:
    try:
        dtype = np.dtype(str(value["dtype"]))
        shape = tuple(int(item) for item in value["shape"])
        raw = base64.b64decode(str(value["data_b64"]).encode("ascii"), validate=True)
    except (KeyError, TypeError, ValueError) as error:
        raise BSSDGWireError(f"{name} wire descriptor is invalid") from error
    if dtype not in (np.dtype(np.int8), np.dtype(np.float16)):
        raise BSSDGWireError(f"{name} wire dtype is not permitted")
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if expected != len(raw):
        raise BSSDGWireError(f"{name} wire byte count drift")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _state_payload(state: BSSDGState, *, include_digest: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": state.schema,
        "candidate_id": CANDIDATE_ID,
        "classes": list(state.classes),
        "active_k": state.active_k,
        "binding": asdict(state.binding),
        "class_means_qint8": _array_wire(state.class_means_qint8),
        "class_scales_fp16": _array_wire(state.class_scales_fp16),
        "logrho_fp16": _array_wire(state.logrho_fp16),
        "intercept_fp16": _array_wire(state.intercept_fp16),
        "v0_fp16": _array_wire(state.v0_fp16),
        "fit_receipt": dict(state.fit_receipt),
        "resource_receipt": dict(state.resource_receipt),
    }
    if include_digest:
        payload["state_sha256"] = state.state_sha256
    return payload


def _state_digest(state: BSSDGState) -> str:
    return _sha256(_canonical_json(_state_payload(state, include_digest=False)))


def serialize_bssdg_state(state: BSSDGState) -> bytes:
    if type(state) is not BSSDGState:
        raise BSSDGError("serialize requires an exact BSSDGState")
    expected = _state_digest(state)
    if state.state_sha256 != expected:
        raise BSSDGError("BSSDG state SHA256 receipt drift")
    return _canonical_json(_state_payload(state, include_digest=True))


def deserialize_bssdg_state(value: bytes | bytearray | str) -> BSSDGState:
    try:
        raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        payload = json.loads(raw.decode("utf-8"))
    except (TypeError, ValueError, UnicodeError) as error:
        raise BSSDGWireError("BSSDG state serialization is invalid") from error
    if not isinstance(payload, Mapping):
        raise BSSDGWireError("BSSDG state serialization must be an object")
    try:
        digest = str(payload["state_sha256"])
        binding = _binding(payload["binding"])
        state = BSSDGState(
            classes=tuple(str(item) for item in payload["classes"]),
            active_k=int(payload["active_k"]),
            class_means_qint8=_array_from_wire(payload["class_means_qint8"], "class_means_qint8"),
            class_scales_fp16=_array_from_wire(payload["class_scales_fp16"], "class_scales_fp16"),
            logrho_fp16=_array_from_wire(payload["logrho_fp16"], "logrho_fp16"),
            intercept_fp16=_array_from_wire(payload["intercept_fp16"], "intercept_fp16"),
            v0_fp16=_array_from_wire(payload["v0_fp16"], "v0_fp16"),
            binding=binding,
            fit_receipt=dict(payload["fit_receipt"]),
            resource_receipt=dict(payload["resource_receipt"]),
            state_sha256=digest,
            schema=str(payload["schema"]),
        )
    except (KeyError, TypeError, ValueError, BSSDGError) as error:
        if isinstance(error, BSSDGError):
            raise
        raise BSSDGWireError("BSSDG state fields are invalid") from error
    if serialize_bssdg_state(state) != _canonical_json(payload):
        raise BSSDGWireError("BSSDG state serialization is not canonical")
    return state


def roundtrip_bssdg_state(state: BSSDGState) -> BSSDGState:
    return deserialize_bssdg_state(serialize_bssdg_state(state))


def _assert_unique_function_tuples(
    means_qint8: np.ndarray,
    scales_fp16: np.ndarray,
    logrho_fp16: np.ndarray,
    v0_fp16: np.ndarray,
) -> None:
    del v0_fp16  # The diagonal prior is shared by every class.
    means = means_qint8.astype(np.float32) * scales_fp16.astype(np.float32)[:, None]
    rho = np.exp(logrho_fp16.astype(np.float32))
    seen: set[bytes] = set()
    for index in range(len(means)):
        key = (
            np.ascontiguousarray(means[index], dtype=np.float32).tobytes(order="C")
            + np.asarray(rho[index], dtype=np.float32).tobytes()
        )
        if key in seen:
            raise BSSDGDuplicateFunctionError(
                "two classes compile to an identical canonical score tuple"
            )
        seen.add(key)


def _balanced_k(labels: tuple[str, ...], classes: tuple[str, ...]) -> int:
    if any(label not in classes for label in labels):
        raise BSSDGError("support contains a class outside registered_classes")
    counts = tuple(labels.count(class_id) for class_id in classes)
    if any(count < 1 for count in counts) or len(set(counts)) != 1:
        raise BSSDGError("support must be balanced over every registered class")
    active_k = counts[0]
    if active_k not in ALLOWED_K:
        raise BSSDGError("BSSDG only permits K1 and K5")
    return active_k


def _quantize_means(means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(means), axis=1).astype(np.float32)
    raw_scale = np.where(
        max_abs > np.float32(0.0),
        max_abs / np.float32(127.0),
        np.float32(FP16_MIN_NORMAL),
    )
    scales = validate_positive_fp16(raw_scale, "class_scales_fp16")
    ratios = means / scales.astype(np.float32)[:, None]
    rounded = np.rint(ratios)
    if (
        not np.isfinite(rounded).all()
        or np.any(rounded < np.float32(-127.0))
        or np.any(rounded > np.float32(127.0))
    ):
        raise BSSDGWireError("class mean quantization would clip an int8 value")
    codes = rounded.astype(np.int8)
    if np.any(codes == np.int8(-128)):
        raise BSSDGWireError("class mean quantization produced reserved -128")
    return np.ascontiguousarray(codes), np.ascontiguousarray(scales)


def _fit_prior(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    count = np.float32(rows.shape[0])
    mean = np.sum(rows, axis=0, dtype=np.float32) / count
    residual = rows - mean[None, :]
    variance = np.sum(residual * residual, axis=0, dtype=np.float32) / count
    sbar = np.sum(variance, dtype=np.float32) / np.float32(FEATURE_WIDTH)
    trace = float(np.sum(variance, dtype=np.float32))
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(variance).all()
        or not math.isfinite(float(sbar))
        or not math.isfinite(trace)
        or trace <= 0.0
    ):
        raise BSSDGError("pooled prior trace is zero or non-finite")
    v0 = (count * variance + sbar) / (count + np.float32(1.0))
    if not np.isfinite(v0).all() or np.any(v0 <= 0.0):
        raise BSSDGError("pooled diagonal prior is not positive and finite")
    return mean, variance, v0, trace


def fit_bssdg(
    support_z: Any,
    labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    k_shot: int | None = None,
    binding: BSSDGBinding | Mapping[str, Any] | None = None,
) -> BSSDGState:
    """Fit one support-only BSSDG state.

    The only inputs are current-state support rows, their class labels, the
    immutable registered class order, and an optional typed state binding.
    Query rows and query metadata never enter this function.
    """

    started_ns = time.perf_counter_ns()
    rows = _rows(support_z, "support_z")
    support_labels = _labels(labels, "labels")
    classes = _registry(registered_classes, "registered_classes")
    if len(rows) != len(support_labels):
        raise BSSDGError("support rows and labels must have equal length")
    active_k = _balanced_k(support_labels, classes)
    if k_shot is not None and (type(k_shot) is not int or k_shot != active_k):
        raise BSSDGError("explicit k_shot does not match balanced support")
    state_binding = _binding(binding)
    m0, variance, v0, prior_trace = _fit_prior(rows)
    inv_v0 = np.float32(1.0) / v0
    means = np.empty((len(classes), FEATURE_WIDTH), dtype=np.float32)
    rho = np.empty(len(classes), dtype=np.float32)
    tau = np.float32(1.0 + active_k)
    nu = np.float32(4.0 + active_k)
    for class_index, class_id in enumerate(classes):
        local = rows[np.asarray([label == class_id for label in support_labels], dtype=bool)]
        zbar = np.sum(local, axis=0, dtype=np.float32) / np.float32(active_k)
        means[class_index] = zbar
        residual = local - zbar[None, :]
        within = np.sum(residual * residual * inv_v0[None, :], dtype=np.float32)
        shift = zbar - m0
        between = (
            np.float32(active_k) / tau
        ) * np.sum(shift * shift * inv_v0, dtype=np.float32)
        energy = np.float32(within + between)
        rho[class_index] = ((np.float32(4.0) + energy) / nu) * (
            (tau + np.float32(1.0)) / tau
        )
    if not np.isfinite(means).all() or not np.isfinite(rho).all() or np.any(rho <= 0.0):
        raise BSSDGError("BSSDG class statistics are not finite and positive")
    means_qint8, scales_fp16 = _quantize_means(means)
    logrho_fp16 = validate_signed_fp16(np.log(rho), "logrho_fp16")
    intercept_fp16 = validate_signed_fp16(
        -np.float32(0.5 * FEATURE_WIDTH) * logrho_fp16.astype(np.float32),
        "intercept_fp16",
    )
    v0_fp16 = validate_positive_fp16(v0, "v0_fp16")
    _assert_unique_function_tuples(means_qint8, scales_fp16, logrho_fp16, v0_fp16)
    fit_ops = int(
        rows.shape[0] * (3 * FEATURE_WIDTH + 2)
        + len(classes) * (5 * active_k * FEATURE_WIDTH + 4 * FEATURE_WIDTH + 10)
    )
    query_ops = int(len(classes) * (4 * FEATURE_WIDTH + 8))
    numeric_bytes = int(
        means_qint8.nbytes
        + scales_fp16.nbytes
        + logrho_fp16.nbytes
        + intercept_fp16.nbytes
        + v0_fp16.nbytes
    )
    registry_bytes = len(_canonical_json({"classes": classes, "binding": asdict(state_binding)}))
    state_bytes = numeric_bytes + registry_bytes + 64
    if state_bytes > STATE_LIMIT_BYTES:
        raise BSSDGWireError("BSSDG state exceeds the 6KiB provisional limit")
    elapsed_ns = int(time.perf_counter_ns() - started_ns)
    fit_receipt = {
        "schema": SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "formula": "pooled_diagonal_student_t_tau0_1_nu0_4_nuv_1",
        "support_row_count": int(rows.shape[0]),
        "registered_class_count": len(classes),
        "active_k": active_k,
        "pooled_prior_trace": prior_trace,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "class_role_access": False,
        "class_permutation_equivariant": True,
    }
    resource_receipt = {
        "numeric_state_bytes": numeric_bytes,
        "registry_bytes": registry_bytes,
        "receipt_bytes": 64,
        "state_bytes": state_bytes,
        "state_limit_bytes": STATE_LIMIT_BYTES,
        "fit_analytic_ops": fit_ops,
        "query_analytic_ops_per_row": query_ops,
        "fit_latency_ns": elapsed_ns,
        "query_latency_ns": 0,
        "query_latency_observed": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
    }
    state = BSSDGState(
        classes=classes,
        active_k=active_k,
        class_means_qint8=means_qint8,
        class_scales_fp16=scales_fp16,
        logrho_fp16=logrho_fp16,
        intercept_fp16=intercept_fp16,
        v0_fp16=v0_fp16,
        binding=state_binding,
        fit_receipt=fit_receipt,
        resource_receipt=resource_receipt,
        state_sha256="",
    )
    object.__setattr__(state, "state_sha256", _state_digest(state))
    state.__post_init__()
    return state


def score_bssdg(state: BSSDGState, query_z: Any) -> np.ndarray:
    """Return float32 Student-t scores; ties remain visible to the caller."""

    if type(state) is not BSSDGState:
        raise BSSDGError("score requires an exact BSSDGState")
    state.__post_init__()
    query = _rows(query_z, "query_z", allow_empty=True)
    if query.shape[0] == 0:
        return np.empty((0, len(state.classes)), dtype=np.float32)
    means = state.decoded_class_means
    v0 = state.decoded_v0
    rho = state.decoded_rho
    inv_v0 = np.float32(1.0) / v0
    nu = np.float32(4.0 + state.active_k)
    delta = query[:, None, :] - means[None, :, :]
    d2 = np.sum(delta * delta * inv_v0[None, None, :], axis=2, dtype=np.float32)
    argument = d2 / (nu * rho[None, :])
    log_term = np.log1p(argument).astype(np.float32)
    scores = (
        -np.float32(0.5 * FEATURE_WIDTH) * state.logrho_fp16.astype(np.float32)[None, :]
        -np.float32(0.5 * (FEATURE_WIDTH + float(nu))) * log_term
    )
    if not np.isfinite(scores).all():
        raise BSSDGError("BSSDG query score became non-finite")
    return np.ascontiguousarray(scores, dtype=np.float32)


def predict_bssdg_unique(state: BSSDGState, query_z: Any) -> np.ndarray:
    """Predict class handles only when every query has one exact winner."""

    scores = score_bssdg(state, query_z)
    if scores.shape[0] == 0:
        return np.empty((0,), dtype=object)
    maxima = np.max(scores, axis=1, keepdims=True)
    tie_rows = np.flatnonzero(np.sum(scores == maxima, axis=1) > 1)
    if len(tie_rows):
        raise BSSDGExactTieError(
            f"exact top tie in query rows {tuple(int(item) for item in tie_rows)}"
        )
    winners = np.argmax(scores, axis=1)
    return np.asarray([state.classes[int(index)] for index in winners], dtype=object)


def verify_bssdg_binding(state: BSSDGState, binding: BSSDGBinding) -> None:
    if type(state) is not BSSDGState or type(binding) is not BSSDGBinding:
        raise BSSDGError("BSSDG binding verification requires typed values")
    state.__post_init__()
    binding.__post_init__()
    if state.binding != binding:
        raise BSSDGError("BSSDG state binding drift")


def bssdg_resource_receipt(state: BSSDGState) -> dict[str, Any]:
    if type(state) is not BSSDGState:
        raise BSSDGError("resource receipt requires an exact BSSDGState")
    state.__post_init__()
    result = dict(state.resource_receipt)
    result["wire_bytes"] = state.wire_bytes
    result["candidate_id"] = CANDIDATE_ID
    result["schema"] = SCHEMA
    return result


__all__ = [
    "ALLOWED_K",
    "BINDING_SCHEMA",
    "BSSDGBinding",
    "BSSDGDuplicateFunctionError",
    "BSSDGError",
    "BSSDGExactTieError",
    "BSSDGState",
    "BSSDGWireError",
    "CANDIDATE_ID",
    "FEATURE_WIDTH",
    "FP16_MAX",
    "FP16_MIN_NORMAL",
    "PROTOCOL_SCHEMA",
    "SCHEMA",
    "STATE_LIMIT_BYTES",
    "bssdg_resource_receipt",
    "deserialize_bssdg_state",
    "fit_bssdg",
    "predict_bssdg_unique",
    "roundtrip_bssdg_state",
    "score_bssdg",
    "serialize_bssdg_state",
    "validate_positive_fp16",
    "validate_signed_fp16",
    "verify_bssdg_binding",
]
