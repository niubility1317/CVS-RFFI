"""Design-frozen NEXT-R2 CVFR domain-adaptation numerical core.

CVFR fits one shared residual from canonical and ``+/- pi/4`` support views at
the frozen ``joint_proj.0`` pre-ReLU 160-dimensional tap.  The module has no
query fitting, scorer, class-role branch, fallback head, or checkpoint update.

Finite exact-zero rows are values, not technical failures.  They totalise to
the zero vector.  A numerically unidentifiable support problem similarly
produces a legal identity state so that a complete experiment can measure a
zero DA effect instead of being stopped before prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "cvs.stage2.next_r2.cvfr.v1"
STATE_SCHEMA = "cvs.phase2.next_r2.cvfr_state.v1"
WIRE_SCHEMA = "cvs.phase2.next_r2.cvfr_wire.v1"
RECEIPT_SCHEMA = "cvs.phase2.next_r2.cvfr_receipt.v1"
RESOURCE_SCHEMA = "cvs.phase2.next_r2.cvfr_resource.v1"

PROTOCOL_SCHEMA = "p2_min_v1"
Z_DIM = 160
SCALE_DIM = Z_DIM - 1
SHIFT_DIM = Z_DIM
PARAM_DIM = SCALE_DIM + SHIFT_DIM
ALLOWED_K = frozenset((1, 5))
ALLOWED_STATE_IDS = frozenset(("DA1_REG0", "DA1_REG1"))

STATUS_APPLIED = "DA_APPLIED_IDENTIFIABLE"
STATUS_IDENTITY_UNIDENTIFIABLE = "DA_IDENTITY_UNIDENTIFIABLE"
STATUS_IDENTITY_ZERO = "DA_IDENTITY_ZERO_SOLUTION"
LEGAL_STATUSES = frozenset(
    (STATUS_APPLIED, STATUS_IDENTITY_UNIDENTIFIABLE, STATUS_IDENTITY_ZERO)
)

TRUST_RADIUS = math.sqrt(2.0)
MAX_CONDITION = 1.0 / math.sqrt(float(np.finfo(np.float32).eps))
GRAM_EPS = float(np.finfo(np.float32).eps)
# A dot product over the analytic Helmert columns may use a different BLAS
# reduction order from construction.  Sixty-four FP64 epsilons is the frozen
# roundoff envelope for both basis validation and runtime sum(u) auditing; it
# is not a learned or performance-dependent tolerance.
ZERO_SUM_AUDIT_TOLERANCE = 64.0 * float(np.finfo(np.float64).eps)
WIRE_MAGIC = b"CVFR319\x00"

_F16 = np.dtype("<f2")
_F32 = np.dtype("<f4")
_F64 = np.dtype("<f8")


class CVFRError(ValueError):
    """A frozen CVFR shape, binding, finite-value, or wire invariant failed."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_json(item) for item in value]
    if isinstance(value, np.generic):
        return _canonical_json(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise CVFRError("CVFR metadata contains a non-canonical or non-finite value")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _canonical_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return _deep_freeze(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise CVFRError("CVFR receipt contains an unsupported value")


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CVFRError(f"{name} must be a nonempty string")
    return value


def _readonly_array(
    value: object,
    *,
    name: str,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype or array.shape != shape or not array.flags.c_contiguous:
        raise CVFRError(
            f"{name} must be C-contiguous {dtype.str} with shape {list(shape)}"
        )
    if dtype.kind == "f" and not np.isfinite(array).all():
        raise CVFRError(f"{name} must be finite")
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _feature_rows(value: object, *, name: str, rows: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != _F32
        or array.ndim != 2
        or array.shape[1] != Z_DIM
        or array.shape[0] < 1
        or (rows is not None and array.shape[0] != rows)
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
    ):
        expected = "N" if rows is None else str(rows)
        raise CVFRError(
            f"{name} must be finite C-contiguous float32 [{expected},{Z_DIM}]"
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def helmert_basis() -> np.ndarray:
    """Return the fixed 160x159 orthonormal zero-sum Helmert basis."""

    basis = np.zeros((Z_DIM, SCALE_DIM), dtype=np.float64)
    for column in range(SCALE_DIM):
        k = column + 1
        denominator = math.sqrt(float(k * (k + 1)))
        basis[:k, column] = 1.0 / denominator
        basis[k, column] = -float(k) / denominator
    # The closed-form columns are exactly zero-sum over the reals.  Repeated
    # float64 addition can nevertheless leave about 4e-15 at d=160.  Reduce
    # that representation-level roundoff using the same all-ones gauge.  The
    # named theoretical tolerance above, not exact floating equality, remains
    # authoritative across different BLAS reduction orders.
    ones = np.ones(Z_DIM, dtype=np.float64)
    for _ in range(2):
        basis -= np.outer(ones, (ones @ basis) / float(Z_DIM))
    basis.setflags(write=False)
    return basis


_HELMERT = helmert_basis()


def totalize_rows(value: object, *, name: str = "features") -> np.ndarray:
    """L2-normalise finite float32 rows and map exact-zero rows to zero."""

    rows = _feature_rows(value, name=name).astype(np.float64, copy=False)
    norms = np.linalg.norm(rows, axis=1)
    result = np.zeros_like(rows, dtype=np.float64)
    nonzero = norms > 0.0
    result[nonzero] = rows[nonzero] / norms[nonzero, None]
    if not np.isfinite(result).all():
        raise CVFRError(f"{name} totalisation produced a non-finite value")
    return np.ascontiguousarray(result, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class CVFRSupportBinding:
    """Exact support-row identity for one DA1 state fit."""

    capsule_id: str
    split_id: str
    outer_key: str
    state_id: str
    k: int
    registered_classes: tuple[str, ...]
    support_physical_ids: tuple[str, ...]
    protocol_schema: str = PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        _require_text(self.capsule_id, name="capsule_id")
        _require_text(self.split_id, name="split_id")
        _require_text(self.outer_key, name="outer_key")
        if self.protocol_schema != PROTOCOL_SCHEMA:
            raise CVFRError("CVFR binding must use protocol_schema=p2_min_v1")
        if self.state_id not in ALLOWED_STATE_IDS:
            raise CVFRError("CVFR fit binding must be DA1_REG0 or DA1_REG1")
        if isinstance(self.k, bool) or int(self.k) not in ALLOWED_K:
            raise CVFRError("CVFR K must be exactly 1 or 5")
        classes = tuple(self.registered_classes)
        if (
            len(classes) < 2
            or any(not isinstance(item, str) or not item for item in classes)
            or len(set(classes)) != len(classes)
        ):
            raise CVFRError("registered_classes must contain unique nonempty strings")
        physical_ids = tuple(self.support_physical_ids)
        if (
            len(physical_ids) != len(classes) * int(self.k)
            or any(not isinstance(item, str) or not item for item in physical_ids)
            or len(set(physical_ids)) != len(physical_ids)
        ):
            raise CVFRError(
                "support_physical_ids must be unique and equal class_count*K"
            )
        object.__setattr__(self, "k", int(self.k))
        object.__setattr__(self, "registered_classes", classes)
        object.__setattr__(self, "support_physical_ids", physical_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_schema": self.protocol_schema,
            "capsule_id": self.capsule_id,
            "split_id": self.split_id,
            "outer_key": self.outer_key,
            "state_id": self.state_id,
            "k": self.k,
            "registered_classes": self.registered_classes,
            "support_physical_ids": self.support_physical_ids,
        }

    @property
    def digest(self) -> str:
        return _sha256(_canonical_bytes(self.as_dict()))


def _class_groups(
    labels: Sequence[str], binding: CVFRSupportBinding
) -> tuple[np.ndarray, ...]:
    if isinstance(labels, (str, bytes)):
        raise CVFRError("support_labels must be a row-aligned sequence")
    try:
        values = tuple(labels)
    except TypeError as exc:
        raise CVFRError("support_labels must be a row-aligned sequence") from exc
    if len(values) != len(binding.support_physical_ids):
        raise CVFRError("support_labels are not row-aligned to the binding")
    if any(not isinstance(item, str) or not item for item in values):
        raise CVFRError("support_labels must be nonempty strings")
    if set(values) != set(binding.registered_classes):
        raise CVFRError("support_labels do not exactly match registered_classes")
    groups: list[tuple[int, np.ndarray]] = []
    for label in binding.registered_classes:
        indices = np.flatnonzero(np.asarray(values, dtype=object) == label)
        if indices.size != binding.k:
            raise CVFRError("every registered class must have exactly K support rows")
        groups.append((int(indices[0]), indices.astype(np.int64, copy=False)))
    groups.sort(key=lambda item: item[0])
    return tuple(indices for _first, indices in groups)


def _row_jacobian(row: np.ndarray, *, rms: float) -> tuple[np.ndarray, np.ndarray]:
    norm = float(np.linalg.norm(row))
    if norm == 0.0:
        return np.zeros(Z_DIM, dtype=np.float64), np.zeros(
            (Z_DIM, PARAM_DIM), dtype=np.float64
        )
    z = row / norm
    projector = (np.eye(Z_DIM, dtype=np.float64) - np.outer(z, z)) / norm
    scale = projector @ (row[:, None] * _HELMERT)
    shift = rms * projector
    jacobian = np.concatenate((scale, shift), axis=1)
    return z, np.ascontiguousarray(jacobian, dtype=np.float64)


def solve_trust_region(
    design: object,
    residual: object,
    *,
    radius: float = TRUST_RADIUS,
) -> tuple[np.ndarray, bool]:
    """Solve the fixed linear least-squares trust problem in FP64.

    This low-level deterministic helper does not decide identifiability.  The
    caller must apply the design-frozen raw-rank and condition rule first.
    """

    matrix = np.asarray(design)
    vector = np.asarray(residual)
    if (
        matrix.dtype != _F64
        or matrix.ndim != 2
        or matrix.shape[1] != PARAM_DIM
        or matrix.shape[0] < PARAM_DIM
        or not matrix.flags.c_contiguous
        or not np.isfinite(matrix).all()
    ):
        raise CVFRError(f"design must be finite C-contiguous float64 [M,{PARAM_DIM}]")
    if (
        vector.dtype != _F64
        or vector.shape != (matrix.shape[0],)
        or not vector.flags.c_contiguous
        or not np.isfinite(vector).all()
    ):
        raise CVFRError("residual must be finite row-aligned C-contiguous float64")
    if not math.isfinite(float(radius)) or float(radius) <= 0.0:
        raise CVFRError("trust radius must be finite and positive")

    u_svd, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    if singular.size != PARAM_DIM or float(singular[-1]) <= 0.0:
        raise CVFRError("trust solver requires a full-column-rank design")
    projected = u_svd.T @ vector
    delta = -(vt.T @ (projected / singular))
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm <= float(radius):
        return np.ascontiguousarray(delta, dtype=np.float64), False

    squared = singular * singular

    def norm_at(lagrange: float) -> float:
        coefficients = singular * projected / (squared + lagrange)
        return float(np.linalg.norm(coefficients))

    low = 0.0
    high = 1.0
    while norm_at(high) > float(radius):
        high *= 2.0
        if not math.isfinite(high):
            raise CVFRError("trust-region multiplier overflow")
    for _ in range(96):
        middle = 0.5 * (low + high)
        if norm_at(middle) > float(radius):
            low = middle
        else:
            high = middle
    lagrange = high
    coefficients = singular * projected / (squared + lagrange)
    delta = -(vt.T @ coefficients)
    if not np.isfinite(delta).all() or float(np.linalg.norm(delta)) > float(radius) * (
        1.0 + 16.0 * np.finfo(np.float64).eps
    ):
        raise CVFRError("trust-region solution is non-finite or outside the frozen cap")
    return np.ascontiguousarray(delta, dtype=np.float64), True


def _quantize_delta(delta: np.ndarray) -> np.ndarray:
    quantized = np.ascontiguousarray(delta, dtype=np.float16)
    if not np.isfinite(quantized).all():
        raise CVFRError("CVFR FP16 state overflow")
    for _ in range(8):
        decoded_norm = float(np.linalg.norm(quantized.astype(np.float64)))
        if decoded_norm <= TRUST_RADIUS:
            quantized.setflags(write=False)
            return quantized
        factor = (TRUST_RADIUS / decoded_norm) * (1.0 - 2.0**-10)
        quantized = np.ascontiguousarray(
            quantized.astype(np.float64) * factor, dtype=np.float16
        )
    raise CVFRError("FP16 CVFR state cannot satisfy the frozen trust cap")


def _transform_f64(rows: np.ndarray, coefficients: np.ndarray, rms: float) -> np.ndarray:
    decoded = coefficients.astype(np.float64, copy=False)
    a = decoded[:SCALE_DIM]
    v = decoded[SCALE_DIM:]
    u = _HELMERT @ a
    scale = np.exp(u)
    if not np.isfinite(scale).all():
        raise CVFRError("CVFR exp(u) overflowed")
    y = rows.astype(np.float64, copy=False) * scale[None, :] + float(rms) * v[None, :]
    if not np.isfinite(y).all():
        raise CVFRError("CVFR transform produced a non-finite pre-totalisation value")
    norms = np.linalg.norm(y, axis=1)
    result = np.zeros_like(y, dtype=np.float64)
    nonzero = norms > 0.0
    result[nonzero] = y[nonzero] / norms[nonzero, None]
    return np.ascontiguousarray(result, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class CVFRState:
    """Immutable deployable CVFR state."""

    status: str
    coefficients_fp16: np.ndarray
    rms_fp32: np.ndarray
    binding_digest: str
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status not in LEGAL_STATUSES:
            raise CVFRError("unknown CVFR state status")
        coefficients = _readonly_array(
            self.coefficients_fp16,
            name="coefficients_fp16",
            dtype=_F16,
            shape=(PARAM_DIM,),
        )
        rms = _readonly_array(
            self.rms_fp32,
            name="rms_fp32",
            dtype=_F32,
            shape=(1,),
        )
        if float(rms[0]) < 0.0:
            raise CVFRError("CVFR RMS cannot be negative")
        if float(np.linalg.norm(coefficients.astype(np.float64))) > TRUST_RADIUS * (
            1.0 + 16.0 * np.finfo(np.float64).eps
        ):
            raise CVFRError("CVFR state exceeds the frozen algorithmic trust cap")
        u = _HELMERT @ coefficients[:SCALE_DIM].astype(np.float64)
        sum_u = float(np.sum(u, dtype=np.float64))
        sum_u_limit = ZERO_SUM_AUDIT_TOLERANCE * max(
            1.0, float(np.linalg.norm(u))
        )
        if abs(sum_u) > sum_u_limit:
            raise CVFRError("CVFR Helmert log-scale contrast lost its zero-sum gauge")
        if not np.isfinite(np.exp(u)).all():
            raise CVFRError("CVFR state exp(u) is non-finite")
        if (
            not isinstance(self.binding_digest, str)
            or len(self.binding_digest) != 64
            or any(char not in "0123456789abcdef" for char in self.binding_digest)
        ):
            raise CVFRError("binding_digest must be a lowercase SHA256")
        if not isinstance(self.receipt, Mapping):
            raise CVFRError("CVFR receipt must be a mapping")
        frozen_receipt = _deep_freeze(dict(self.receipt))
        if frozen_receipt.get("schema") != RECEIPT_SCHEMA:
            raise CVFRError("CVFR receipt schema drift")
        if frozen_receipt.get("status") != self.status:
            raise CVFRError("CVFR receipt/state status drift")
        if frozen_receipt.get("binding_digest") != self.binding_digest:
            raise CVFRError("CVFR receipt/state binding drift")
        object.__setattr__(self, "coefficients_fp16", coefficients)
        object.__setattr__(self, "rms_fp32", rms)
        object.__setattr__(self, "receipt", frozen_receipt)

    def to_wire(self) -> bytes:
        payload = self.coefficients_fp16.astype(_F16, copy=False).tobytes(order="C")
        payload += self.rms_fp32.astype(_F32, copy=False).tobytes(order="C")
        header = {
            "schema": WIRE_SCHEMA,
            "state_schema": STATE_SCHEMA,
            "status": self.status,
            "binding_digest": self.binding_digest,
            "payload_bytes": len(payload),
            "payload_sha256": _sha256(payload),
            "receipt": self.receipt,
        }
        header_bytes = _canonical_bytes(header)
        return WIRE_MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + payload

    @classmethod
    def from_wire(cls, wire: object) -> "CVFRState":
        if not isinstance(wire, bytes) or not wire.startswith(WIRE_MAGIC):
            raise CVFRError("CVFR wire magic mismatch")
        prefix = len(WIRE_MAGIC)
        if len(wire) < prefix + 4:
            raise CVFRError("CVFR wire is truncated")
        header_length = struct.unpack("<I", wire[prefix : prefix + 4])[0]
        header_start = prefix + 4
        header_end = header_start + header_length
        if header_end > len(wire):
            raise CVFRError("CVFR wire header is truncated")
        try:
            header = json.loads(wire[header_start:header_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CVFRError("CVFR wire header is not canonical JSON") from exc
        if _canonical_bytes(header) != wire[header_start:header_end]:
            raise CVFRError("CVFR wire header is not canonical")
        if (
            header.get("schema") != WIRE_SCHEMA
            or header.get("state_schema") != STATE_SCHEMA
        ):
            raise CVFRError("CVFR wire schema drift")
        payload = wire[header_end:]
        expected_payload = PARAM_DIM * _F16.itemsize + _F32.itemsize
        if header.get("payload_bytes") != expected_payload or len(payload) != expected_payload:
            raise CVFRError("CVFR wire payload size drift")
        if header.get("payload_sha256") != _sha256(payload):
            raise CVFRError("CVFR wire payload hash mismatch")
        split = PARAM_DIM * _F16.itemsize
        coefficients = np.frombuffer(payload[:split], dtype=_F16).copy()
        rms = np.frombuffer(payload[split:], dtype=_F32).copy()
        return cls(
            status=header.get("status"),
            coefficients_fp16=np.ascontiguousarray(coefficients, dtype=np.float16),
            rms_fp32=np.ascontiguousarray(rms, dtype=np.float32),
            binding_digest=header.get("binding_digest"),
            receipt=header.get("receipt"),
        )


def transform_cvfr(
    features: object,
    state: CVFRState,
    *,
    expected_binding_digest: str,
) -> np.ndarray:
    """Apply one immutable state; this function has no fit or update path."""

    if not isinstance(state, CVFRState):
        raise CVFRError("transform_cvfr requires a CVFRState")
    if expected_binding_digest != state.binding_digest:
        raise CVFRError("CVFR query/state binding mismatch")
    rows = _feature_rows(features, name="features")
    transformed = _transform_f64(
        rows, state.coefficients_fp16, float(state.rms_fp32[0])
    )
    return np.ascontiguousarray(transformed, dtype=np.float32)


def _displacement_summary(base: np.ndarray, adapted: np.ndarray) -> dict[str, float]:
    distance = np.linalg.norm(adapted - base, axis=1)
    return {
        "mean": float(np.mean(distance)),
        "max": float(np.max(distance)),
    }


def fit_cvfr_support(
    canonical_support: object,
    phase_plus_support: object,
    phase_minus_support: object,
    support_labels: Sequence[str],
    binding: CVFRSupportBinding,
) -> CVFRState:
    """Fit the sole support-only CVFR state for one DA1 protocol state."""

    if not isinstance(binding, CVFRSupportBinding):
        raise CVFRError("fit_cvfr_support requires an explicit support binding")
    rows = len(binding.support_physical_ids)
    canonical = _feature_rows(canonical_support, name="canonical_support", rows=rows)
    plus = _feature_rows(phase_plus_support, name="phase_plus_support", rows=rows)
    minus = _feature_rows(phase_minus_support, name="phase_minus_support", rows=rows)
    groups = _class_groups(support_labels, binding)

    canonical64 = canonical.astype(np.float64, copy=False)
    rms64 = math.sqrt(float(np.sum(canonical64 * canonical64)) / float(rows * Z_DIM))
    if (
        not math.isfinite(rms64)
        or rms64 < 0.0
        or rms64 > float(np.finfo(np.float32).max)
    ):
        raise CVFRError("canonical support RMS must be finite and nonnegative")
    rms_array = np.ascontiguousarray([rms64], dtype=np.float32)
    rms = float(rms_array[0])

    base_z: dict[str, np.ndarray] = {}
    jacobians: dict[str, np.ndarray] = {}
    zero_counts: dict[str, int] = {}
    for name, values in (("canonical", canonical), ("plus", plus), ("minus", minus)):
        z_rows: list[np.ndarray] = []
        j_rows: list[np.ndarray] = []
        zeros = 0
        for row in values.astype(np.float64, copy=False):
            z, jacobian = _row_jacobian(row, rms=rms)
            zeros += int(float(np.linalg.norm(row)) == 0.0)
            z_rows.append(z)
            j_rows.append(jacobian)
        base_z[name] = np.ascontiguousarray(np.stack(z_rows), dtype=np.float64)
        jacobians[name] = np.ascontiguousarray(np.stack(j_rows), dtype=np.float64)
        zero_counts[name] = zeros

    design_parts: list[np.ndarray] = []
    residual_parts: list[np.ndarray] = []
    zero_residual_rows = 0
    zero_residual_l2: list[float] = []
    view_weight = math.sqrt(2.0 * float(rows) * float(Z_DIM))
    for view_name in ("plus", "minus"):
        for index in range(rows):
            if (
                float(np.linalg.norm(canonical64[index])) == 0.0
                or float(
                    np.linalg.norm(
                        (plus if view_name == "plus" else minus)[index].astype(
                            np.float64, copy=False
                        )
                    )
                )
                == 0.0
            ):
                zero_residual_rows += 1
                zero_residual_l2.append(
                    float(
                        np.linalg.norm(
                            base_z["canonical"][index] - base_z[view_name][index]
                        )
                    )
                )
                continue
            design_parts.append(
                (jacobians["canonical"][index] - jacobians[view_name][index])
                / view_weight
            )
            residual_parts.append(
                (base_z["canonical"][index] - base_z[view_name][index])
                / view_weight
            )

    class_centers: list[np.ndarray] = []
    class_jacobians: list[np.ndarray] = []
    for indices in groups:
        class_centers.append(np.mean(base_z["canonical"][indices], axis=0))
        class_jacobians.append(np.mean(jacobians["canonical"][indices], axis=0))
    pair_count = len(groups) * (len(groups) - 1) // 2
    pair_weight = math.sqrt(float(pair_count))
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            center_delta = class_centers[left] - class_centers[right]
            jacobian_delta = class_jacobians[left] - class_jacobians[right]
            q = float(center_delta @ center_delta)
            gradient = 2.0 * (center_delta @ jacobian_delta)
            design_parts.append(
                np.ascontiguousarray(
                    gradient[None, :] / ((q + GRAM_EPS) * pair_weight),
                    dtype=np.float64,
                )
            )
            residual_parts.append(np.zeros(1, dtype=np.float64))

    if design_parts:
        design = np.ascontiguousarray(np.concatenate(design_parts, axis=0), dtype=np.float64)
        residual = np.ascontiguousarray(
            np.concatenate(residual_parts, axis=0), dtype=np.float64
        )
        singular = np.linalg.svd(design, compute_uv=False)
        sigma_max = float(singular[0]) if singular.size else 0.0
        rank_tolerance = (
            float(max(design.shape) * np.finfo(np.float64).eps * sigma_max)
            if sigma_max > 0.0
            else 0.0
        )
        rank = int(np.count_nonzero(singular > rank_tolerance))
        condition = (
            float(singular[0] / singular[PARAM_DIM - 1])
            if rank == PARAM_DIM
            else math.inf
        )
    else:
        design = np.zeros((0, PARAM_DIM), dtype=np.float64)
        residual = np.zeros(0, dtype=np.float64)
        sigma_max = 0.0
        rank_tolerance = 0.0
        rank = 0
        condition = math.inf

    trust_active = False
    if rank < PARAM_DIM or condition > MAX_CONDITION:
        status = STATUS_IDENTITY_UNIDENTIFIABLE
        delta = np.zeros(PARAM_DIM, dtype=np.float64)
    else:
        delta, trust_active = solve_trust_region(design, residual)
        status = STATUS_APPLIED if float(np.linalg.norm(delta)) > 0.0 else STATUS_IDENTITY_ZERO

    quantized = _quantize_delta(delta)
    if not bool(np.any(quantized != np.float16(0.0))):
        status = (
            STATUS_IDENTITY_UNIDENTIFIABLE
            if status == STATUS_IDENTITY_UNIDENTIFIABLE
            else STATUS_IDENTITY_ZERO
        )
    decoded = quantized.astype(np.float64)
    u = _HELMERT @ decoded[:SCALE_DIM]
    sum_u = float(np.sum(u, dtype=np.float64))
    sum_u_limit = ZERO_SUM_AUDIT_TOLERANCE * max(
        1.0, float(np.linalg.norm(u))
    )
    exp_u = np.exp(u)
    v = decoded[SCALE_DIM:]
    adapted = {
        "canonical": _transform_f64(canonical, quantized, rms),
        "plus": _transform_f64(plus, quantized, rms),
        "minus": _transform_f64(minus, quantized, rms),
    }
    displacement = {
        name: _displacement_summary(base_z[name], adapted[name])
        for name in ("canonical", "plus", "minus")
    }
    cross_view = {
        name: _displacement_summary(adapted["canonical"], adapted[name])
        for name in ("plus", "minus")
    }
    condition_value: float | str = condition if math.isfinite(condition) else "inf"
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "method_schema": SCHEMA,
        "state_schema": STATE_SCHEMA,
        "status": status,
        "binding_digest": binding.digest,
        "protocol_schema": binding.protocol_schema,
        "outer_key": binding.outer_key,
        "state_id": binding.state_id,
        "k": binding.k,
        "registered_class_count": len(groups),
        "physical_support_rows": rows,
        "mathematical_views_per_physical_row": 3,
        "view_rows_count_as_additional_k": False,
        "raw_jacobian_rows": int(design.shape[0]),
        "raw_jacobian_rank": rank,
        "raw_jacobian_rank_tolerance": rank_tolerance,
        "raw_jacobian_sigma_max": sigma_max,
        "raw_jacobian_condition": condition_value,
        "condition_limit": MAX_CONDITION,
        "trust_radius": TRUST_RADIUS,
        "trust_active": trust_active,
        "trust_is_algorithmic_not_physical_safety": True,
        "delta_norm_fp16_decoded": float(np.linalg.norm(decoded)),
        "max_abs_u": float(np.max(np.abs(u))),
        "sum_u": sum_u,
        "sum_u_audit_tolerance": sum_u_limit,
        "exp_u_min": float(np.min(exp_u)),
        "exp_u_max": float(np.max(exp_u)),
        "max_abs_v": float(np.max(np.abs(v))),
        "canonical_rms": rms,
        "zero_canonical_rows": zero_counts["canonical"],
        "zero_plus_rows": zero_counts["plus"],
        "zero_minus_rows": zero_counts["minus"],
        "zero_view_residual_count": zero_residual_rows,
        "zero_view_residual_l2_sum": float(sum(zero_residual_l2)),
        "zero_view_residual_l2_max": (
            float(max(zero_residual_l2)) if zero_residual_l2 else 0.0
        ),
        "displacement": displacement,
        "cross_view_displacement": cross_view,
        "canonical_sha256": _sha256(canonical.tobytes(order="C")),
        "plus_sha256": _sha256(plus.tobytes(order="C")),
        "minus_sha256": _sha256(minus.tobytes(order="C")),
        "support_physical_id_root": _sha256(
            _canonical_bytes({"physical_ids": binding.support_physical_ids})
        ),
        "registered_class_root": _sha256(
            _canonical_bytes({"registered_classes": binding.registered_classes})
        ),
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "truth_role_quota_inputs": 0,
        "fallback_calls": 0,
        "resource": {
            "schema": RESOURCE_SCHEMA,
            "numeric_wire_bytes": PARAM_DIM * _F16.itemsize + _F32.itemsize,
            "fp16_parameter_count": PARAM_DIM,
            "fp32_scalar_count": 1,
            "implicit_helmert_values": Z_DIM * SCALE_DIM,
            "per_transform_helmert_matvec_multiply_count": Z_DIM * SCALE_DIM,
            "per_transform_helmert_matvec_add_count": Z_DIM * (SCALE_DIM - 1),
            "per_transform_exp_count": Z_DIM,
            "per_row_scale_multiply_count": Z_DIM,
            "per_row_shift_multiply_count": Z_DIM,
            "per_row_scale_shift_add_count": Z_DIM,
            "per_row_totalization_norm_multiply_count": Z_DIM,
            "per_row_totalization_norm_add_count": Z_DIM - 1,
            "per_row_totalization_sqrt_count": 1,
            "per_row_totalization_divide_count_upper_bound": Z_DIM,
            "per_row_affine_mac_count": 2 * Z_DIM,
            "per_row_affine_mac_scope": (
                "rows*scale plus rms*v only; excludes fixed Helmert matvec, "
                "exp, FP16 decode, norm and totalization divides"
            ),
            "optimizer_steps": 0,
            "checkpoint_parameter_backward_calls": 0,
            "query_updates": 0,
        },
    }
    return CVFRState(
        status=status,
        coefficients_fp16=quantized,
        rms_fp32=rms_array,
        binding_digest=binding.digest,
        receipt=receipt,
    )


__all__ = [
    "ALLOWED_K",
    "CVFRError",
    "CVFRState",
    "CVFRSupportBinding",
    "MAX_CONDITION",
    "PARAM_DIM",
    "PROTOCOL_SCHEMA",
    "SCALE_DIM",
    "SCHEMA",
    "SHIFT_DIM",
    "STATUS_APPLIED",
    "STATUS_IDENTITY_UNIDENTIFIABLE",
    "STATUS_IDENTITY_ZERO",
    "TRUST_RADIUS",
    "ZERO_SUM_AUDIT_TOLERANCE",
    "Z_DIM",
    "fit_cvfr_support",
    "helmert_basis",
    "solve_trust_region",
    "totalize_rows",
    "transform_cvfr",
]
