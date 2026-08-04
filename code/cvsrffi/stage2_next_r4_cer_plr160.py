"""NEXT-R4 CER-PLR160 support-only registration-head numerical core.

The module deliberately owns a small, auditable operation only:

* K=1 is an exact object alias of the frozen qKNN logits;
* K=5 forms a class-equal shared diagonal shape residual from support only;
* the residual is centered across registered classes, bounded by a qKNN-lock
  scale, and deployed as ``int8 W[C,160] + fp16 scale[C] + fp16 intercept[C]``.

It has no query truth, role, old/new partition, leave-one-out, top-k, quota,
or query-selection input.  In particular, incoming R1 vectors are consumed as
the already-produced signed-unit representation: this module never applies a
ReLU or an L2 normalization to either support or query rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import stage2_zid_student_t_qknn as qknn


Z_DIM = 160
K1 = 1
K5 = 5
EPS32 = float(np.finfo(np.float32).eps)
REPRESENTATION_UNIT_ATOL = 2.0e-6
HEAD_ID = "CER-PLR160"
WIRE_SCHEMA = "cvs.phase2.next_r4.cer_plr160.int8_affine_wire.v1"
FIT_SCHEMA = "cvs.phase2.next_r4.cer_plr160.fit.v1"
RESOURCE_SCHEMA = "cvs.phase2.next_r4.cer_plr160.resource.v1"
TIE_RESOLUTION_RULE = "SEALED_CLASS_HANDLE_UTF8_ASC_V1"


class NextR4CERPLR160Error(ValueError):
    """Raised when the frozen CER-PLR160 numerical contract is violated."""


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze_value(value) for key, value in mapping.items()})


def _json_value(value: Any) -> Any:
    """Convert immutable receipt containers back to JSON-native containers."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: str, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise NextR4CERPLR160Error(f"{name} must be a lowercase SHA256")
    return text


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _classes(values: Sequence[str]) -> tuple[str, ...]:
    classes = tuple(str(value) for value in values)
    if len(classes) < 2 or len(set(classes)) != len(classes) or any(not value for value in classes):
        raise NextR4CERPLR160Error("registered classes must be unique non-empty labels")
    return classes


def _representation(value: str) -> str:
    representation = str(value)
    if representation not in {"R0", "R1"}:
        raise NextR4CERPLR160Error("representation must be exact R0 or R1")
    return representation


def _representation_rows(
    value: np.ndarray,
    name: str,
    *,
    representation: str,
) -> np.ndarray:
    rep = _representation(representation)
    rows = np.asarray(value)
    if (
        not isinstance(value, np.ndarray)
        or rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[0] < 1
        or rows.shape[1] != Z_DIM
        or not np.isfinite(rows).all()
    ):
        raise NextR4CERPLR160Error(f"{name} must be finite numpy float32 [N,{Z_DIM}]")
    if rep == "R0" and bool(np.any(rows < np.float32(0.0))):
        raise NextR4CERPLR160Error(f"{name} must be canonical nonnegative R0 rows")
    norms = np.linalg.norm(rows.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=REPRESENTATION_UNIT_ATOL):
        raise NextR4CERPLR160Error(f"{name} must already be unit {rep} rows")
    # Read-only validation: retain the exact incoming values.  No ReLU, L2
    # normalization, sign clipping, or other semantic copy/transform occurs.
    return rows


def _support_contract(
    support_zid160: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    qknn_lock: qknn.Phase1ZIDStudentTLock,
    representation: str,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, int, str]:
    if type(qknn_lock) is not qknn.Phase1ZIDStudentTLock:
        raise NextR4CERPLR160Error("CER-PLR160 requires the exact frozen qKNN lock")
    normalized_representation = _representation(representation)
    rows = _representation_rows(
        support_zid160,
        "support representation",
        representation=normalized_representation,
    )
    classes = _classes(registered_classes)
    labels = tuple(str(value) for value in support_labels)
    if len(labels) != len(rows) or any(label not in classes for label in labels):
        raise NextR4CERPLR160Error("support labels must map to the registered classes")
    class_index = {label: index for index, label in enumerate(classes)}
    indices = np.asarray([class_index[label] for label in labels], dtype=np.int16)
    counts = tuple(int(np.sum(indices == index)) for index in range(len(classes)))
    if len(set(counts)) != 1 or counts[0] < 1:
        raise NextR4CERPLR160Error("CER-PLR160 requires balanced K-shot support")
    active_k = counts[0]
    if active_k not in {K1, K5}:
        raise NextR4CERPLR160Error("CER-PLR160 supports only K1 or K5")
    if qknn_lock.active_k != active_k:
        raise NextR4CERPLR160Error("support K and frozen qKNN lock active K drift")
    return rows, classes, indices, active_k, normalized_representation


def _validate_qknn_logits(qknn_logits: np.ndarray, *, class_count: int) -> np.ndarray:
    logits = np.asarray(qknn_logits)
    if (
        not isinstance(qknn_logits, np.ndarray)
        or logits.dtype != np.float32
        or logits.ndim != 2
        or logits.shape[0] < 1
        or logits.shape[1] != class_count
        or not np.isfinite(logits).all()
    ):
        raise NextR4CERPLR160Error("qKNN logits must be finite numpy float32 [Q,C]")
    return logits


@dataclass(frozen=True, slots=True)
class Float32TopDecision:
    """Per-query final-score decisions without changing the score matrix."""

    predictions: tuple[str, ...]
    tie_query_count: int

    def __post_init__(self) -> None:
        if (
            not self.predictions
            or any(type(handle) is not str or not handle for handle in self.predictions)
            or type(self.tie_query_count) is not int
            or self.tie_query_count < 0
            or self.tie_query_count > len(self.predictions)
        ):
            raise NextR4CERPLR160Error("final float32 top decision drift")


def _sealed_utf8_class_handles(values: Sequence[str]) -> tuple[tuple[str, ...], tuple[bytes, ...]]:
    """Validate the immutable class-handle order without normalizing handles."""

    if isinstance(values, (str, bytes)):
        raise NextR4CERPLR160Error("registered class handles must be a sequence")
    try:
        handles = tuple(values)
    except TypeError as error:
        raise NextR4CERPLR160Error("registered class handles must be a sequence") from error
    if (
        len(handles) < 2
        or any(type(handle) is not str or not handle for handle in handles)
        or len(set(handles)) != len(handles)
    ):
        raise NextR4CERPLR160Error("registered class handles must be unique non-empty strings")
    try:
        utf8 = tuple(handle.encode("utf-8") for handle in handles)
    except UnicodeEncodeError as error:
        raise NextR4CERPLR160Error("registered class handles must be UTF-8 encodable") from error
    return handles, utf8


def resolve_float32_top_handles(
    final_float32_logits: np.ndarray,
    registered_class_handles: Sequence[str],
) -> Float32TopDecision:
    """Resolve every final float32 top score from sealed class handles only.

    A row with one exact maximum keeps that class.  For an exact tie, the
    selected class is the tied handle with lexicographically smallest UTF-8
    bytes.  The helper reads neither query identity nor any query-side role,
    truth, quota, batch count, or cross-query assignment state.  It only reads
    the supplied final scores and sealed registered handles; it never mutates
    or re-quantizes the score matrix.
    """

    handles, utf8_handles = _sealed_utf8_class_handles(registered_class_handles)
    scores = np.asarray(final_float32_logits)
    if (
        not isinstance(final_float32_logits, np.ndarray)
        or scores.dtype != np.float32
        or scores.ndim != 2
        or scores.shape[0] < 1
        or scores.shape[1] != len(handles)
        or not np.isfinite(scores).all()
    ):
        raise NextR4CERPLR160Error("final logits must be finite numpy float32 [Q,C]")

    maxima = np.max(scores, axis=1, keepdims=True)
    tied = scores == maxima
    predictions: list[str] = []
    tie_query_count = 0
    for row_tied in tied:
        tied_columns = np.flatnonzero(row_tied)
        if len(tied_columns) == 1:
            predictions.append(handles[int(tied_columns[0])])
            continue
        tie_query_count += 1
        # The selection key is only the sealed UTF-8 handle bytes.  Column
        # indices locate the candidate handles but never participate in order.
        tied_handles = (
            (utf8_handles[int(column)], handles[int(column)]) for column in tied_columns
        )
        predictions.append(min(tied_handles, key=lambda item: item[0])[1])
    return Float32TopDecision(
        predictions=tuple(predictions),
        tie_query_count=tie_query_count,
    )


def qknn_score_scale_from_lock(lock: qknn.Phase1ZIDStudentTLock) -> float:
    """Derive ``Sq`` from exactly ``nu``, ``d_eff`` and common ``h_c``.

    The expression is the finite Student-t log-kernel excursion at a unit
    cosine-distance displacement.  It intentionally reads no support geometry,
    query value, class label, temperature, or any role metadata.
    """

    if type(lock) is not qknn.Phase1ZIDStudentTLock:
        raise NextR4CERPLR160Error("Sq requires the exact frozen qKNN lock")
    nu = float(lock.student_nu)
    d_eff = float(lock.kernel_effective_dim)
    h_c = float(lock.shared_h0)
    if (
        not math.isfinite(nu)
        or not math.isfinite(d_eff)
        or not math.isfinite(h_c)
        or nu <= 0.0
        or d_eff <= 0.0
        or h_c <= 0.0
    ):
        raise NextR4CERPLR160Error("qKNN nu/d_eff/h_c must be positive and finite")
    value = 0.5 * (nu + d_eff) * math.log1p(2.0 / (nu * h_c * h_c))
    if not math.isfinite(value) or value <= 0.0:
        raise NextR4CERPLR160Error("qKNN lock-derived Sq is not positive and finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class CERPLR160K1QKNNAliasState:
    """K1 has no identifiable 160-D shape state and aliases qKNN exactly."""

    classes: tuple[str, ...]
    qknn_lock_digest: str
    active_k: int = K1

    def __post_init__(self) -> None:
        classes = _classes(self.classes)
        if type(self.active_k) is not int or self.active_k != K1:
            raise NextR4CERPLR160Error("K1 alias state active K drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(
            self, "qknn_lock_digest", _require_sha256(self.qknn_lock_digest, "qKNN lock digest")
        )

    @property
    def numeric_state_bytes(self) -> int:
        return 0


@dataclass(frozen=True, slots=True)
class CERPLR160NoFunctionAliasState:
    """K5 residual is exactly absent after support fit or quantization."""

    classes: tuple[str, ...]
    qknn_lock_digest: str
    reason: str
    active_k: int = K5

    def __post_init__(self) -> None:
        classes = _classes(self.classes)
        if type(self.active_k) is not int or self.active_k != K5:
            raise NextR4CERPLR160Error("NO_HEAD_FUNCTION active K drift")
        if self.reason not in {"Sr_ZERO", "QUANTIZED_RESIDUAL_ZERO"}:
            raise NextR4CERPLR160Error("NO_HEAD_FUNCTION reason drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(
            self, "qknn_lock_digest", _require_sha256(self.qknn_lock_digest, "qKNN lock digest")
        )

    @property
    def numeric_state_bytes(self) -> int:
        return 0


@dataclass(frozen=True, slots=True)
class CERPLR160State:
    """The fixed 164C-byte deployed affine residual wire for a K5 row."""

    classes: tuple[str, ...]
    qknn_lock_digest: str
    weight_qint8: np.ndarray
    scale_fp16: np.ndarray
    intercept_fp16: np.ndarray
    active_k: int = K5
    wire_schema: str = WIRE_SCHEMA

    def __post_init__(self) -> None:
        classes = _classes(self.classes)
        weights = np.asarray(self.weight_qint8)
        scales = np.asarray(self.scale_fp16)
        intercepts = np.asarray(self.intercept_fp16)
        if (
            type(self.active_k) is not int
            or self.active_k != K5
            or self.wire_schema != WIRE_SCHEMA
            or weights.dtype != np.int8
            or weights.shape != (len(classes), Z_DIM)
            or np.any(weights == np.int8(-128))
            or scales.dtype != np.float16
            or scales.shape != (len(classes),)
            or not np.isfinite(scales).all()
            or np.any(scales <= np.float16(0.0))
            or intercepts.dtype != np.float16
            or intercepts.shape != (len(classes),)
            or not np.isfinite(intercepts).all()
        ):
            raise NextR4CERPLR160Error("CER-PLR160 INT8/FP16 affine wire drift")
        if not bool(np.any(weights != 0) or np.any(intercepts != np.float16(0.0))):
            raise NextR4CERPLR160Error("all-zero residual must use NO_HEAD_FUNCTION alias")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(
            self, "qknn_lock_digest", _require_sha256(self.qknn_lock_digest, "qKNN lock digest")
        )
        object.__setattr__(self, "weight_qint8", _readonly(weights, np.int8))
        object.__setattr__(self, "scale_fp16", _readonly(scales, np.float16))
        object.__setattr__(self, "intercept_fp16", _readonly(intercepts, np.float16))

    @property
    def numeric_state_bytes(self) -> int:
        return int(self.weight_qint8.nbytes + self.scale_fp16.nbytes + self.intercept_fp16.nbytes)

    @property
    def numeric_state_formula(self) -> str:
        return "160C+2C+2C=164C_B"

    @property
    def state_sha256(self) -> str:
        header = _canonical_json(
            {
                "wire_schema": self.wire_schema,
                "classes": self.classes,
                "qknn_lock_digest": self.qknn_lock_digest,
                "active_k": self.active_k,
            }
        )
        return _sha256(header + self.to_wire())

    def to_wire(self) -> bytes:
        """Return exactly the numerical W/scale/intercept payload, no metadata."""

        return b"".join(
            (
                np.ascontiguousarray(self.weight_qint8, dtype=np.int8).tobytes(order="C"),
                np.ascontiguousarray(self.scale_fp16, dtype=np.dtype("<f2")).tobytes(order="C"),
                np.ascontiguousarray(self.intercept_fp16, dtype=np.dtype("<f2")).tobytes(order="C"),
            )
        )

    @classmethod
    def from_wire(
        cls,
        classes: Sequence[str],
        qknn_lock_digest: str,
        payload: bytes,
    ) -> "CERPLR160State":
        class_tuple = _classes(classes)
        wire = bytes(payload)
        expected = 164 * len(class_tuple)
        if len(wire) != expected:
            raise NextR4CERPLR160Error("CER-PLR160 wire byte count drift")
        weight_bytes = len(class_tuple) * Z_DIM
        scale_bytes = 2 * len(class_tuple)
        weights = np.frombuffer(wire[:weight_bytes], dtype=np.int8).reshape(len(class_tuple), Z_DIM).copy()
        scales = np.frombuffer(
            wire[weight_bytes : weight_bytes + scale_bytes], dtype=np.dtype("<f2")
        ).astype(np.float16, copy=True)
        intercepts = np.frombuffer(wire[weight_bytes + scale_bytes :], dtype=np.dtype("<f2")).astype(
            np.float16, copy=True
        )
        return cls(
            classes=class_tuple,
            qknn_lock_digest=qknn_lock_digest,
            weight_qint8=weights,
            scale_fp16=scales,
            intercept_fp16=intercepts,
        )

    def decoded_weight(self) -> np.ndarray:
        return _readonly(
            self.weight_qint8.astype(np.float32) * self.scale_fp16.astype(np.float32)[:, None],
            np.float32,
        )


CERPLR160HeadState = CERPLR160State | CERPLR160K1QKNNAliasState | CERPLR160NoFunctionAliasState


@dataclass(frozen=True, slots=True)
class CERPLR160Fit:
    """One support-only fit; aliases are legal normal outcomes, not failures."""

    state: CERPLR160HeadState
    fit_receipt: Mapping[str, Any]
    resource_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.state) not in {
            CERPLR160State,
            CERPLR160K1QKNNAliasState,
            CERPLR160NoFunctionAliasState,
        }:
            raise NextR4CERPLR160Error("CER-PLR160 fit state type drift")
        if not isinstance(self.fit_receipt, Mapping) or not isinstance(self.resource_receipt, Mapping):
            raise NextR4CERPLR160Error("CER-PLR160 receipts must be mappings")
        object.__setattr__(self, "fit_receipt", _freeze(self.fit_receipt))
        object.__setattr__(self, "resource_receipt", _freeze(self.resource_receipt))


def _means(rows: np.ndarray, class_indices: np.ndarray, class_count: int) -> np.ndarray:
    result = np.empty((class_count, Z_DIM), dtype=np.float64)
    for class_index in range(class_count):
        local = rows[class_indices == class_index]
        if len(local) != K5:
            raise NextR4CERPLR160Error("K5 class support closure drift")
        result[class_index] = np.mean(local, axis=0, dtype=np.float64)
    if not np.isfinite(result).all():
        raise NextR4CERPLR160Error("support prototype means became non-finite")
    return result


def _class_equal_variance(
    rows: np.ndarray, means: np.ndarray, class_indices: np.ndarray, class_count: int
) -> np.ndarray:
    per_class = np.empty((class_count, Z_DIM), dtype=np.float64)
    for class_index in range(class_count):
        local = rows[class_indices == class_index]
        residual = local - means[class_index][None, :]
        per_class[class_index] = np.mean(np.square(residual), axis=0, dtype=np.float64)
    variance = np.mean(per_class, axis=0, dtype=np.float64)
    if not np.isfinite(variance).all() or np.any(variance < 0.0):
        raise NextR4CERPLR160Error("class-equal support variance became invalid")
    return variance


def _prototype_shape_residual(
    means: np.ndarray,
    variance: np.ndarray,
    *,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return centered prototype affine residual and its support-only RMS."""

    v_bar = float(np.mean(variance, dtype=np.float64))
    if not math.isfinite(v_bar) or v_bar < 0.0:
        raise NextR4CERPLR160Error("shared diagonal variance mean became invalid")
    shrinkage = float(class_count * (K5 - 1)) / float(class_count * (K5 - 1) + Z_DIM)
    denominator = shrinkage * variance + (1.0 - shrinkage) * v_bar + EPS32
    d = 1.0 / denominator
    d0 = 1.0 / (v_bar + EPS32)
    delta = d - d0
    if not np.isfinite(d).all() or not np.isfinite(delta).all():
        raise NextR4CERPLR160Error("shared diagonal D-D0 became non-finite")
    weight = means * delta[None, :]
    intercept = -0.5 * np.sum(means * weight, axis=1, dtype=np.float64)
    # Center in the class dimension before quantization so no free class-wide
    # logit bias is introduced.  This formula is symmetric in registered labels.
    weight -= np.mean(weight, axis=0, keepdims=True, dtype=np.float64)
    intercept -= float(np.mean(intercept, dtype=np.float64))
    prototype_logits = means @ weight.T + intercept[None, :]
    # ``Sr`` is deliberately the prototype-own-versus-other-class gap RMS,
    # not the RMS of every residual entry.  For each class-c prototype mu_c,
    # compare r_c(mu_c) with every r_a(mu_c), a != c.  This is invariant to a
    # class-wide additive residual and matches the frozen CER scale definition.
    own = np.diag(prototype_logits)
    gaps = own[:, None] - prototype_logits
    off_diagonal = ~np.eye(class_count, dtype=bool)
    sr_squared = float(
        np.sum(np.square(gaps[off_diagonal]), dtype=np.float64)
        / float(class_count * (class_count - 1))
    )
    sr = float(math.sqrt(sr_squared))
    if (
        not np.isfinite(weight).all()
        or not np.isfinite(intercept).all()
        or not math.isfinite(sr)
        or sr < 0.0
    ):
        raise NextR4CERPLR160Error("prototype-logit residual became invalid")
    return weight, intercept, v_bar, shrinkage, sr


def _gamma(*, sq: float, sr: float) -> float:
    if not math.isfinite(sq) or not math.isfinite(sr) or sq <= 0.0 or sr < 0.0:
        raise NextR4CERPLR160Error("Sq/Sr must be finite with Sq positive")
    value = sq / (sq + 4.0 * sr + 64.0 * EPS32 * max(1.0, sq, sr))
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise NextR4CERPLR160Error("CER-PLR160 gamma became invalid")
    if value * sr > sq / 4.0 + 64.0 * EPS32 * max(1.0, sq, sr):
        raise NextR4CERPLR160Error("CER-PLR160 gamma trust bound drift")
    return float(value)


def _quantize_affine_residual(
    weight: np.ndarray, intercept: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        weight.shape[1:] != (Z_DIM,)
        or intercept.shape != (weight.shape[0],)
        or not np.isfinite(weight).all()
        or not np.isfinite(intercept).all()
    ):
        raise NextR4CERPLR160Error("affine residual quantization input drift")
    class_count = len(weight)
    codes = np.zeros((class_count, Z_DIM), dtype=np.int8)
    scales = np.zeros(class_count, dtype=np.float16)
    minimum = float(np.finfo(np.float16).tiny)
    try:
        for class_index in range(class_count):
            maximum = float(np.max(np.abs(weight[class_index])))
            scale_value = max(maximum / 127.0, minimum)
            scale = np.asarray(scale_value, dtype=np.float16).item()
            if not math.isfinite(float(scale)) or float(scale) <= 0.0:
                raise NextR4CERPLR160Error("CER-PLR160 INT8 scale is not representable")
            encoded = np.clip(
                np.rint(weight[class_index] / float(scale)), -127.0, 127.0
            ).astype(np.int8)
            if np.any(encoded == np.int8(-128)):
                raise NextR4CERPLR160Error("CER-PLR160 INT8 code range drift")
            codes[class_index] = encoded
            scales[class_index] = np.float16(scale)
        intercepts = np.asarray(intercept, dtype=np.float16)
    except FloatingPointError as error:
        raise NextR4CERPLR160Error("CER-PLR160 affine wire quantization failed") from error
    if not np.isfinite(intercepts).all():
        raise NextR4CERPLR160Error("CER-PLR160 FP16 intercept is not representable")
    return codes, scales, intercepts


def _resource_receipt(
    *,
    state: CERPLR160HeadState,
    support_rows: int,
    class_count: int,
) -> Mapping[str, Any]:
    if type(state) is CERPLR160State:
        return {
            "schema": RESOURCE_SCHEMA,
            "head": HEAD_ID,
            "active_k": K5,
            "feature_dim": Z_DIM,
            "class_count": class_count,
            "head_status": "FUNCTIONAL",
            "deployed_numeric_state_bytes": state.numeric_state_bytes,
            "deployed_numeric_state_formula": state.numeric_state_formula,
            "deployed_wire": "int8_W[C,160]+fp16_scale[C]+fp16_intercept[C]",
            "incremental_query_head_macs_per_sample": Z_DIM * class_count,
            "query_head_macs_per_sample": Z_DIM * class_count,
            "fit_analytic_mac_equivalent": 4 * support_rows * Z_DIM + 8 * Z_DIM + 2 * class_count * Z_DIM,
            "fit_analytic_mac_formula": "4Nd+8d+2Cd",
            "explicit_dense_matrix_elements_constructed": 0,
            "explicit_spectral_factorization_count": 0,
            "explicit_linear_system_solve_count": 0,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "query_batch_dependency": False,
        }
    return {
        "schema": RESOURCE_SCHEMA,
        "head": HEAD_ID,
        "active_k": state.active_k,
        "feature_dim": Z_DIM,
        "class_count": class_count,
        "head_status": (
            "K1_EXACT_QKNN_ALIAS"
            if type(state) is CERPLR160K1QKNNAliasState
            else "NO_HEAD_FUNCTION"
        ),
        "incremental_deployed_numeric_state_bytes": 0,
        "incremental_query_head_macs_per_sample": 0,
        "underlying_qknn_resource_required": True,
        "underlying_qknn_resource_included": False,
        "fit_analytic_mac_equivalent": 0,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_batch_dependency": False,
    }


def _common_fit_receipt(
    *,
    classes: tuple[str, ...],
    active_k: int,
    representation: str,
    qknn_lock: qknn.Phase1ZIDStudentTLock,
) -> dict[str, Any]:
    return {
        "schema": FIT_SCHEMA,
        "head": HEAD_ID,
        "registered_classes": classes,
        "active_k": active_k,
        "feature_dim": Z_DIM,
        "qknn_lock_digest": qknn_lock.lock_digest,
        "representation": representation,
        "representation_input_consumed_identity": True,
        "r1_post_signed_unit_relu_applied": False,
        "r1_post_signed_unit_l2_normalization_applied": False,
        "support_only": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_top_k_access": False,
        "query_accuracy_access": False,
        "leave_one_out_access": False,
        "old_new_role_access": False,
        "same_formula_all_registered_classes": True,
        "class_label_permutation_equivariant": True,
    }


def _no_function_fit(
    *,
    classes: tuple[str, ...],
    qknn_lock: qknn.Phase1ZIDStudentTLock,
    representation: str,
    reason: str,
    receipt: Mapping[str, Any],
) -> CERPLR160Fit:
    state = CERPLR160NoFunctionAliasState(
        classes=classes,
        qknn_lock_digest=qknn_lock.lock_digest,
        reason=reason,
    )
    updated = dict(receipt)
    updated.update(
        {
            "fit_mode": "exact_qknn_logit_object_alias",
            "head_status": "NO_HEAD_FUNCTION",
            "no_head_function_reason": reason,
            "alias_reason": "support-only diagonal prototype residual is absent after required numerical wire",
            "representation": representation,
            "incremental_deployed_numeric_state_bytes": 0,
        }
    )
    return CERPLR160Fit(
        state=state,
        fit_receipt=updated,
        resource_receipt=_resource_receipt(state=state, support_rows=K5 * len(classes), class_count=len(classes)),
    )


def fit_cer_plr160(
    support_zid160: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    qknn_lock: qknn.Phase1ZIDStudentTLock,
    representation: str = "R0",
) -> CERPLR160Fit:
    """Fit CER-PLR160 only from registered support and a frozen qKNN lock.

    ``support_zid160`` is already the caller-selected R0 canonical or R1
    signed-unit representation.  It is intentionally *not* normalized or
    rectified in this function.
    """

    rows, classes, class_indices, active_k, rep = _support_contract(
        support_zid160,
        support_labels,
        registered_classes,
        qknn_lock=qknn_lock,
        representation=representation,
    )
    receipt = _common_fit_receipt(
        classes=classes,
        active_k=active_k,
        representation=rep,
        qknn_lock=qknn_lock,
    )
    if active_k == K1:
        state = CERPLR160K1QKNNAliasState(
            classes=classes,
            qknn_lock_digest=qknn_lock.lock_digest,
        )
        receipt.update(
            {
                "fit_mode": "exact_qknn_logit_object_alias",
                "head_status": "K1_EXACT_QKNN_ALIAS",
                "alias_reason": "K1 diagonal shape is not identifiable in 160D",
                "historical_k1_equivalence_claim": False,
                "incremental_deployed_numeric_state_bytes": 0,
            }
        )
        return CERPLR160Fit(
            state=state,
            fit_receipt=receipt,
            resource_receipt=_resource_receipt(
                state=state,
                support_rows=len(rows),
                class_count=len(classes),
            ),
        )

    rows64 = rows.astype(np.float64, copy=False)
    means = _means(rows64, class_indices, len(classes))
    variance = _class_equal_variance(rows64, means, class_indices, len(classes))
    raw_weight, raw_intercept, v_bar, shrinkage, sr = _prototype_shape_residual(
        means, variance, class_count=len(classes)
    )
    sq = qknn_score_scale_from_lock(qknn_lock)
    receipt.update(
        {
            "fit_mode": "class_equal_shared_diagonal_shrinkage_centered_prototype_logit_residual",
            "shrinkage_lambda": shrinkage,
            "shrinkage_lambda_formula": "C(K-1)/(C(K-1)+160)",
            "v_bar": v_bar,
            "eps_v": EPS32,
            "sq": sq,
            "sq_formula": "0.5*(nu+d_eff)*log1p(2/(nu*h_c^2))",
            "sq_qknn_lock_fields_only": ("student_nu", "kernel_effective_dim", "shared_h0"),
            "sr": sr,
            "sr_definition": "sqrt(sum_c_sum_a_ne_c[r_c(mu_c)-r_a(mu_c)]^2/(C(C-1)))",
            "raw_class_centered_weight_linf": float(np.max(np.abs(np.mean(raw_weight, axis=0)))),
            "raw_class_centered_intercept_abs": float(abs(np.mean(raw_intercept))),
        }
    )
    if sr == 0.0:
        return _no_function_fit(
            classes=classes,
            qknn_lock=qknn_lock,
            representation=rep,
            reason="Sr_ZERO",
            receipt=receipt,
        )
    gamma = _gamma(sq=sq, sr=sr)
    scaled_weight = gamma * raw_weight
    scaled_intercept = gamma * raw_intercept
    codes, scales, intercepts = _quantize_affine_residual(scaled_weight, scaled_intercept)
    if not bool(np.any(codes != 0) or np.any(intercepts != np.float16(0.0))):
        return _no_function_fit(
            classes=classes,
            qknn_lock=qknn_lock,
            representation=rep,
            reason="QUANTIZED_RESIDUAL_ZERO",
            receipt=receipt,
        )
    state = CERPLR160State(
        classes=classes,
        qknn_lock_digest=qknn_lock.lock_digest,
        weight_qint8=codes,
        scale_fp16=scales,
        intercept_fp16=intercepts,
    )
    receipt.update(
        {
            "head_status": "FUNCTIONAL",
            "gamma": gamma,
            "gamma_formula": "Sq/(Sq+4Sr+64eps32*max(1,Sq,Sr))",
            "gamma_sr_upper_bound": sq / 4.0,
            "gamma_sr": gamma * sr,
            "deployed_wire": "int8_W[C,160]+fp16_scale[C]+fp16_intercept[C]",
            "deployed_numeric_state_bytes": state.numeric_state_bytes,
            "deployed_numeric_state_formula": state.numeric_state_formula,
            "state_sha256": state.state_sha256,
            "quantized_residual_nonzero": True,
        }
    )
    return CERPLR160Fit(
        state=state,
        fit_receipt=receipt,
        resource_receipt=_resource_receipt(
            state=state,
            support_rows=len(rows),
            class_count=len(classes),
        ),
    )


def alias_qknn_logits(fit: CERPLR160Fit, qknn_logits: np.ndarray) -> np.ndarray:
    """Return the exact qKNN object for a legal CER alias state."""

    if type(fit) is not CERPLR160Fit:
        raise NextR4CERPLR160Error("qKNN alias requires an exact CER-PLR160 fit")
    if type(fit.state) not in {CERPLR160K1QKNNAliasState, CERPLR160NoFunctionAliasState}:
        raise NextR4CERPLR160Error("functional K5 CER state is not a qKNN alias")
    logits = _validate_qknn_logits(qknn_logits, class_count=len(fit.state.classes))
    resolve_float32_top_handles(logits, fit.state.classes)
    return qknn_logits


def alias_k1_qknn_logits(fit: CERPLR160Fit, qknn_logits: np.ndarray) -> np.ndarray:
    """K1-only alias entry point for runtime callers that need strict identity."""

    if type(fit) is not CERPLR160Fit or type(fit.state) is not CERPLR160K1QKNNAliasState:
        raise NextR4CERPLR160Error("K1 alias requires a CER-PLR160 K1 alias state")
    return alias_qknn_logits(fit, qknn_logits)


def score_cer_plr160(
    fit: CERPLR160Fit,
    qknn_logits: np.ndarray,
    query_zid160: np.ndarray,
) -> np.ndarray:
    """Add a frozen K5 residual to qKNN, or return its exact legal alias.

    The query is an inference-only input.  It is never included in support
    fitting, scale selection, prototype construction, or any state update.
    """

    if type(fit) is not CERPLR160Fit:
        raise NextR4CERPLR160Error("scoring requires an exact CER-PLR160 fit")
    state = fit.state
    logits = _validate_qknn_logits(qknn_logits, class_count=len(state.classes))
    representation = _representation(str(fit.fit_receipt.get("representation")))
    query = _representation_rows(
        query_zid160,
        "query representation",
        representation=representation,
    )
    if len(query) != len(logits):
        raise NextR4CERPLR160Error("query/logit row count drift")
    if type(state) in {CERPLR160K1QKNNAliasState, CERPLR160NoFunctionAliasState}:
        return alias_qknn_logits(fit, qknn_logits)
    if type(state) is not CERPLR160State:
        raise NextR4CERPLR160Error("CER-PLR160 score state type drift")
    residual = (
        query.astype(np.float64, copy=False) @ state.decoded_weight().astype(np.float64).T
        + state.intercept_fp16.astype(np.float64)[None, :]
    )
    output64 = logits.astype(np.float64, copy=False) + residual
    if not np.isfinite(output64).all():
        raise NextR4CERPLR160Error("CER-PLR160 final logits became non-finite")
    output = _readonly(output64, np.float32)
    resolve_float32_top_handles(output, state.classes)
    return output


__all__ = [
    "CERPLR160Fit",
    "CERPLR160HeadState",
    "CERPLR160K1QKNNAliasState",
    "CERPLR160NoFunctionAliasState",
    "CERPLR160State",
    "EPS32",
    "Float32TopDecision",
    "HEAD_ID",
    "K1",
    "K5",
    "NextR4CERPLR160Error",
    "REPRESENTATION_UNIT_ATOL",
    "TIE_RESOLUTION_RULE",
    "WIRE_SCHEMA",
    "Z_DIM",
    "alias_k1_qknn_logits",
    "alias_qknn_logits",
    "fit_cer_plr160",
    "qknn_score_scale_from_lock",
    "resolve_float32_top_handles",
    "score_cer_plr160",
]
