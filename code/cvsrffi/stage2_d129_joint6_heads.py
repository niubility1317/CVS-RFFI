"""D129 shared-cache six-arm 160-dimensional classification heads.

The module intentionally separates representation and registration-head effects.
It consumes two already-computed, same-row ``z_id160`` representations only:
``R0`` is the base representation and ``R1`` is the caller's support-only DA
representation.  It never calls a backbone, a channel simulator, a scorer, or
any source/clean/query-truth interface.

For each representation it builds three logical heads:

* ``Q``: the frozen Phase1 Student-t qKNN;
* ``F``: a 160-D re-instantiation of historical D92's task-balanced full
  covariance LDA when both task groups contain multiple classes.  The
  5-retained/1-held seen-class LOCO screen necessarily uses an explicitly
  labelled single-class proxy extension because sklearn LDA cannot fit a
  one-class group; and
* ``L``: the corresponding task-balanced diagonal OAS-LDA.

``F`` and ``L`` deliberately share the same compact affine deployment wire.
Consequently their causal comparison is about fit computation and decisions,
not a manufactured deployment-byte advantage.  The historical 288-D D92
replacement byte comparison is emitted in a separate, explicitly non-causal
system receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import stage2_zid_student_t_qknn as qknn


Z_DIM = 160
INT8_MAX = 127.0
ACTIVE_K_VALUES = frozenset({1, 5, 10})
FULL_HEAD = "d92_full160"
LITE_HEAD = "d92_lite160"
QKNN_HEAD = "phase1_locked_student_t_qknn"
R0 = "R0"
R1 = "R1"
ARM_IDS = ("R0Q", "R0F", "R0L", "R1Q", "R1F", "R1L")
SCHEMA = "cvs.phase2.d129.joint6_heads.v1"
AFFINE_SCHEMA = "cvs.phase2.d129.shared_affine160.v1"
ALIAS_SCHEMA = "cvs.phase2.d129.k1_qknn_alias160.v1"
ALIAS_RECEIPT_SCHEMA = "cvs.phase2.d129.k1_qknn_alias_receipt.v1"


class D129Joint6HeadsError(ValueError):
    """Raised when the frozen D129 six-arm head contract drifts."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )
    result.setflags(write=False)
    return result


def _freeze(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _registry(values: Sequence[str]) -> tuple[str, ...]:
    classes = tuple(str(value) for value in values)
    if (
        len(classes) < 4
        or len(set(classes)) != len(classes)
        or any(not value for value in classes)
    ):
        raise D129Joint6HeadsError(
            "registered classes require at least four unique non-empty values"
        )
    return classes


def _opaque_ids(values: Sequence[str], *, expected: int) -> tuple[str, ...]:
    identifiers = tuple(str(value) for value in values)
    if (
        len(identifiers) != expected
        or not identifiers
        or len(set(identifiers)) != len(identifiers)
        or any(not value for value in identifiers)
    ):
        raise D129Joint6HeadsError("opaque query ID closure drift")
    return identifiers


def _raw_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != Z_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D129Joint6HeadsError(f"{name} must be finite float32 [N,{Z_DIM}]")
    return np.ascontiguousarray(rows)


def normalize_zid160_rows(value: np.ndarray, *, name: str = "z_id160") -> np.ndarray:
    """Canonical, representation-level L2 normalization for every head."""

    rows = _raw_rows(value, name=name).astype(np.float64)
    norms = np.sqrt(np.sum(rows * rows, axis=1, keepdims=True))
    if not np.isfinite(norms).all() or bool(np.any(norms <= 0.0)):
        raise D129Joint6HeadsError(f"{name} contains a zero or non-finite row")
    return _readonly(rows / norms, np.float32)


def _normalized_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    rows = _raw_rows(value, name=name)
    norms = np.sqrt(np.sum(rows.astype(np.float64) ** 2, axis=1))
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-6):
        raise D129Joint6HeadsError(
            f"{name} must be the shared canonical L2-normalized representation"
        )
    return np.ascontiguousarray(rows)


def _support_contract(
    support: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    old_class_count: int,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, int, int]:
    rows = _normalized_rows(support, name="support_zid160")
    classes = _registry(registered_classes)
    labels = tuple(str(value) for value in support_labels)
    if len(labels) != len(rows) or any(value not in classes for value in labels):
        raise D129Joint6HeadsError("support labels must close over registered classes")
    index_by_class = {value: index for index, value in enumerate(classes)}
    indices = np.asarray([index_by_class[value] for value in labels], dtype=np.int64)
    counts = np.bincount(indices, minlength=len(classes))
    if np.any(counts < 1) or len(set(int(value) for value in counts)) != 1:
        raise D129Joint6HeadsError("support must be balanced K-shot over all classes")
    active_k = int(counts[0])
    if active_k not in ACTIVE_K_VALUES or len(rows) != len(classes) * active_k:
        raise D129Joint6HeadsError("only balanced K1/K5/K10 support is permitted")
    old_count = int(old_class_count)
    if old_count < 1 or old_count > len(classes) - 1:
        raise D129Joint6HeadsError(
            "old_class_count must leave at least one old and one registered-new class"
        )
    return rows, classes, indices, active_k, old_count


@dataclass(frozen=True, slots=True)
class D129RepresentationCache:
    """One immutable support/query cache for exactly one 160-D representation."""

    representation: str
    support_zid160: np.ndarray
    query_zid160: np.ndarray
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.representation not in {R0, R1}:
            raise D129Joint6HeadsError("representation must be R0 or R1")
        support = _normalized_rows(self.support_zid160, name="cached support")
        query = _normalized_rows(self.query_zid160, name="cached query")
        receipt = dict(self.receipt)
        expected = _canonical_sha256(
            {
                "representation": self.representation,
                "support": _array_receipt(support),
                "query": _array_receipt(query),
            }
        )
        if receipt.get("cache_sha256") != expected:
            raise D129Joint6HeadsError("representation cache receipt drift")
        object.__setattr__(self, "support_zid160", _readonly(support, np.float32))
        object.__setattr__(self, "query_zid160", _readonly(query, np.float32))
        object.__setattr__(self, "receipt", _freeze(receipt))

    @property
    def cache_sha256(self) -> str:
        return str(self.receipt["cache_sha256"])


def _representation_cache(
    representation: str, support: np.ndarray, query: np.ndarray
) -> D129RepresentationCache:
    normalized_support = normalize_zid160_rows(support, name=f"{representation} support")
    normalized_query = normalize_zid160_rows(query, name=f"{representation} query")
    receipt = {
        "schema": "cvs.phase2.d129.representation_cache.v1",
        "representation": representation,
        "feature_dim": Z_DIM,
        "support": _array_receipt(normalized_support),
        "query": _array_receipt(normalized_query),
        "support_query_feature_cache_shared_by_three_heads": True,
        "backbone_forward_calls_in_head_interface": 0,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
    }
    receipt["cache_sha256"] = _canonical_sha256(
        {
            "representation": representation,
            "support": receipt["support"],
            "query": receipt["query"],
        }
    )
    return D129RepresentationCache(
        representation=representation,
        support_zid160=normalized_support,
        query_zid160=normalized_query,
        receipt=receipt,
    )


@dataclass(frozen=True, slots=True)
class D129QKNNState:
    """The exact frozen Phase1 qKNN state used by one representation."""

    bank: qknn.TypedINT8ZIDSupportBank
    metric: qknn.TypedSharedPSDMetric
    resource_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            type(self.bank) is not qknn.TypedINT8ZIDSupportBank
            or type(self.metric) is not qknn.TypedSharedPSDMetric
        ):
            raise D129Joint6HeadsError("Q requires exact typed qKNN states")
        resource = dict(self.resource_receipt)
        if resource.get("query_state_updates") != 0:
            raise D129Joint6HeadsError("qKNN resource query-state closure drift")
        object.__setattr__(self, "resource_receipt", _freeze(resource))


@dataclass(frozen=True, slots=True)
class D129AffineHeadState:
    """The shared compact wire for active Full160 and Lite160 heads."""

    head: str
    classes: tuple[str, ...]
    active_k: int
    weight_qint8: np.ndarray
    scale_fp16: np.ndarray
    intercept_fp16: np.ndarray
    schema: str = AFFINE_SCHEMA

    def __post_init__(self) -> None:
        classes = _registry(self.classes)
        weights = np.asarray(self.weight_qint8)
        scales = np.asarray(self.scale_fp16)
        intercepts = np.asarray(self.intercept_fp16)
        if (
            self.head not in {FULL_HEAD, LITE_HEAD}
            or self.schema != AFFINE_SCHEMA
            or int(self.active_k) not in {5, 10}
            or weights.dtype != np.int8
            or weights.shape != (len(classes), Z_DIM)
            or np.any(weights == np.int8(-128))
            or scales.dtype != np.float16
            or scales.shape != (len(classes),)
            or not np.isfinite(scales).all()
            or np.any(scales <= 0.0)
            or intercepts.dtype != np.float16
            or intercepts.shape != (len(classes),)
            or not np.isfinite(intercepts).all()
        ):
            raise D129Joint6HeadsError("shared affine state shape/dtype/range drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "active_k", int(self.active_k))
        object.__setattr__(self, "weight_qint8", _readonly(weights, np.int8))
        object.__setattr__(self, "scale_fp16", _readonly(scales, np.float16))
        object.__setattr__(self, "intercept_fp16", _readonly(intercepts, np.float16))

    @property
    def numeric_state_bytes(self) -> int:
        return int(
            self.weight_qint8.nbytes
            + self.scale_fp16.nbytes
            + self.intercept_fp16.nbytes
        )

    @property
    def state_sha256(self) -> str:
        return _canonical_sha256(
            {
                "active_k": self.active_k,
                "classes": list(self.classes),
                "head": self.head,
                "intercept": _array_receipt(self.intercept_fp16),
                "scale": _array_receipt(self.scale_fp16),
                "schema": self.schema,
                "weight": _array_receipt(self.weight_qint8),
            }
        )


@dataclass(frozen=True, slots=True)
class D129K1QKNNAliasState:
    """A deliberate 160-D comparator alias, not historical D92/D81 equivalence."""

    head: str
    classes: tuple[str, ...]
    active_k: int = 1
    historical_k1_equivalence_claim: bool = False
    schema: str = ALIAS_SCHEMA

    def __post_init__(self) -> None:
        classes = _registry(self.classes)
        if (
            self.head not in {FULL_HEAD, LITE_HEAD}
            or self.schema != ALIAS_SCHEMA
            or int(self.active_k) != 1
            or self.historical_k1_equivalence_claim is not False
        ):
            raise D129Joint6HeadsError("K1 comparator alias closure drift")
        object.__setattr__(self, "classes", classes)

    @property
    def incremental_numeric_state_bytes(self) -> int:
        return 0


D129RegistrationHeadState = D129AffineHeadState | D129K1QKNNAliasState


@dataclass(frozen=True, slots=True)
class D129HeadFit:
    state: D129RegistrationHeadState
    fit_receipt: Mapping[str, Any]
    resource_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.state) not in {D129AffineHeadState, D129K1QKNNAliasState}:
            raise D129Joint6HeadsError("registration head fit state drift")
        object.__setattr__(self, "fit_receipt", _freeze(self.fit_receipt))
        object.__setattr__(self, "resource_receipt", _freeze(self.resource_receipt))


@dataclass(frozen=True, slots=True)
class D129K1AliasReceipt:
    head: str
    classes: tuple[str, ...]
    opaque_query_ids: tuple[str, ...]
    query_cache_sha256: str
    qknn_logits_sha256: str
    historical_k1_equivalence_claim: bool = False
    schema: str = ALIAS_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        classes = _registry(self.classes)
        ids = _opaque_ids(self.opaque_query_ids, expected=len(self.opaque_query_ids))
        for name, value in (
            ("query cache", self.query_cache_sha256),
            ("qKNN logits", self.qknn_logits_sha256),
        ):
            digest = str(value)
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise D129Joint6HeadsError(f"K1 alias {name} SHA256 drift")
        if (
            self.head not in {FULL_HEAD, LITE_HEAD}
            or self.schema != ALIAS_RECEIPT_SCHEMA
            or self.historical_k1_equivalence_claim is not False
        ):
            raise D129Joint6HeadsError("K1 alias receipt semantics drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "opaque_query_ids", ids)


@dataclass(frozen=True, slots=True)
class D129ArmPrediction:
    arm_id: str
    representation: str
    head: str
    classes: tuple[str, ...]
    logits: np.ndarray
    predictions: tuple[str, ...]
    state: D129QKNNState | D129RegistrationHeadState
    receipt: Mapping[str, Any]
    qknn_alias_receipt: D129K1AliasReceipt | None = None

    def __post_init__(self) -> None:
        classes = _registry(self.classes)
        logits = np.asarray(self.logits)
        if (
            self.arm_id not in ARM_IDS
            or self.representation not in {R0, R1}
            or self.head not in {QKNN_HEAD, FULL_HEAD, LITE_HEAD}
            or logits.dtype != np.float32
            or logits.ndim != 2
            or logits.shape[0] < 1
            or logits.shape[1] != len(classes)
            or not np.isfinite(logits).all()
            or len(self.predictions) != len(logits)
            or any(value not in classes for value in self.predictions)
        ):
            raise D129Joint6HeadsError("arm prediction closure drift")
        if self.head == QKNN_HEAD:
            if type(self.state) is not D129QKNNState or self.qknn_alias_receipt is not None:
                raise D129Joint6HeadsError("Q head state/alias drift")
        elif type(self.state) not in {D129AffineHeadState, D129K1QKNNAliasState}:
            raise D129Joint6HeadsError("registration head arm state drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "predictions", tuple(self.predictions))
        object.__setattr__(self, "receipt", _freeze(self.receipt))


@dataclass(frozen=True, slots=True)
class D129Joint6Result:
    arms: tuple[D129ArmPrediction, ...]
    r0_cache: D129RepresentationCache
    r1_cache: D129RepresentationCache
    row_receipt: Mapping[str, Any]
    head_causal_resource_receipt: Mapping[str, Any]
    system_formal_replacement_resource_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if tuple(arm.arm_id for arm in self.arms) != ARM_IDS:
            raise D129Joint6HeadsError("six logical arm ordering drift")
        if self.r0_cache.representation != R0 or self.r1_cache.representation != R1:
            raise D129Joint6HeadsError("R0/R1 cache identity drift")
        object.__setattr__(self, "row_receipt", _freeze(self.row_receipt))
        object.__setattr__(
            self,
            "head_causal_resource_receipt",
            _freeze(self.head_causal_resource_receipt),
        )
        object.__setattr__(
            self,
            "system_formal_replacement_resource_receipt",
            _freeze(self.system_formal_replacement_resource_receipt),
        )

    def arm(self, arm_id: str) -> D129ArmPrediction:
        for arm in self.arms:
            if arm.arm_id == arm_id:
                return arm
        raise KeyError(arm_id)

    @property
    def r0q(self) -> D129ArmPrediction:
        return self.arm("R0Q")

    @property
    def r0f(self) -> D129ArmPrediction:
        return self.arm("R0F")

    @property
    def r0l(self) -> D129ArmPrediction:
        return self.arm("R0L")

    @property
    def r1q(self) -> D129ArmPrediction:
        return self.arm("R1Q")

    @property
    def r1f(self) -> D129ArmPrediction:
        return self.arm("R1F")

    @property
    def r1l(self) -> D129ArmPrediction:
        return self.arm("R1L")


def _require_sklearn() -> Any:
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    except ImportError as exc:  # pragma: no cover - deployment dependency failure
        raise D129Joint6HeadsError("D92-Full160 requires sklearn") from exc
    return LinearDiscriminantAnalysis


def _group_rows(
    rows: np.ndarray, indices: np.ndarray, class_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(indices, class_indices)
    group_rows = np.asarray(rows[mask], dtype=np.float64)
    local = {int(value): local_index for local_index, value in enumerate(class_indices)}
    group_labels = np.asarray(
        [local[int(value)] for value in indices[mask]], dtype=np.int64
    )
    counts = np.bincount(group_labels, minlength=len(class_indices))
    if (
        group_rows.ndim != 2
        or group_rows.shape[1] != Z_DIM
        or not np.isfinite(group_rows).all()
        or not np.array_equal(np.unique(group_labels), np.arange(len(class_indices)))
        or np.any(counts < 1)
        or len(set(int(value) for value in counts)) != 1
    ):
        raise D129Joint6HeadsError("old/new support task closure drift")
    return group_rows, group_labels


def _auto_shrinkage_full_covariance(
    rows: np.ndarray, indices: np.ndarray, class_indices: np.ndarray
) -> tuple[np.ndarray, str]:
    estimator_type = _require_sklearn()
    group_rows, group_labels = _group_rows(rows, indices, class_indices)
    count = len(class_indices)
    if count == 1:
        # ``LinearDiscriminantAnalysis`` cannot fit a one-class registration
        # group.  This LedoitWolf branch is a directional proxy extension, not
        # a strict reproduction of historical D92's sklearn-LDA path.
        try:
            from sklearn.covariance import LedoitWolf
        except ImportError as exc:  # pragma: no cover - sklearn installation drift
            raise D129Joint6HeadsError("D92-Full160 requires sklearn LedoitWolf") from exc
        estimator = LedoitWolf(assume_centered=False, store_precision=False)
        estimator.fit(group_rows)
        covariance = np.asarray(estimator.covariance_, dtype=np.float64)
        estimator_name = "sklearn_LedoitWolf_single_class_centered_residuals"
    else:
        estimator = estimator_type(
            solver="lsqr",
            shrinkage="auto",
            priors=np.full(count, 1.0 / count, dtype=np.float64),
            store_covariance=True,
        )
        estimator.fit(group_rows, group_labels)
        if not np.array_equal(
            np.asarray(estimator.classes_, dtype=np.int64), np.arange(count, dtype=np.int64)
        ):
            raise D129Joint6HeadsError("D92-Full160 sklearn class order drift")
        covariance = np.asarray(estimator.covariance_, dtype=np.float64)
        estimator_name = "sklearn_LDA_lsqr_auto_shrinkage_equal_prior"
    covariance = 0.5 * (covariance + covariance.T)
    if covariance.shape != (Z_DIM, Z_DIM) or not np.isfinite(covariance).all():
        raise D129Joint6HeadsError("D92-Full160 covariance closure drift")
    return covariance, estimator_name


def _oas_diagonal_covariance(
    rows: np.ndarray, indices: np.ndarray, class_indices: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    group_rows, group_labels = _group_rows(rows, indices, class_indices)
    class_count = len(class_indices)
    means = np.stack(
        [group_rows[group_labels == index].mean(axis=0) for index in range(class_count)]
    )
    residuals = group_rows - means[group_labels]
    n_eff = int(len(group_rows) - class_count)
    if n_eff <= 0:
        raise D129Joint6HeadsError("D92-Lite160 K1 diagonal covariance is unidentifiable")
    scatter = np.sum(residuals * residuals, axis=0, dtype=np.float64) / float(n_eff)
    total = float(np.sum(scatter, dtype=np.float64))
    second_moment = float(np.sum(scatter * scatter, dtype=np.float64))
    tau = total / float(Z_DIM)
    delta = second_moment - total * total / float(Z_DIM)
    if (
        not np.isfinite(scatter).all()
        or not all(math.isfinite(value) for value in (total, second_moment, tau, delta))
        or total <= 0.0
    ):
        raise D129Joint6HeadsError("D92-Lite160 OAS statistics are non-finite")
    if delta <= 0.0:
        shrinkage = 1.0
    else:
        numerator = (1.0 - 2.0 / Z_DIM) * second_moment + total * total
        denominator = (float(n_eff) + 1.0 - 2.0 / Z_DIM) * delta
        shrinkage = min(1.0, numerator / denominator)
    if not math.isfinite(shrinkage) or shrinkage < 0.0:
        raise D129Joint6HeadsError("D92-Lite160 OAS shrinkage is invalid")
    variance_floor = max(
        float(np.finfo(np.float64).tiny),
        float(np.finfo(np.float64).eps) * max(1.0, tau),
    )
    variance = np.maximum(
        (1.0 - shrinkage) * scatter + shrinkage * tau,
        variance_floor,
    )
    if not np.isfinite(variance).all() or np.any(variance <= 0.0):
        raise D129Joint6HeadsError("D92-Lite160 diagonal covariance closure drift")
    return variance, {
        "residual_degrees_of_freedom": float(n_eff),
        "shrinkage": float(shrinkage),
        "variance_floor": float(variance_floor),
        "trace": float(np.sum(variance, dtype=np.float64)),
    }


def _means(rows: np.ndarray, indices: np.ndarray, class_count: int) -> np.ndarray:
    result = np.stack(
        [
            np.asarray(rows[indices == index], dtype=np.float64).mean(axis=0)
            for index in range(class_count)
        ]
    )
    if result.shape != (class_count, Z_DIM) or not np.isfinite(result).all():
        raise D129Joint6HeadsError("class mean closure drift")
    return result


def _quantize_shared_affine(
    *,
    head: str,
    classes: tuple[str, ...],
    active_k: int,
    weights: np.ndarray,
    intercepts: np.ndarray,
) -> tuple[D129AffineHeadState, Mapping[str, Any]]:
    values = np.asarray(weights, dtype=np.float64)
    offset = np.asarray(intercepts, dtype=np.float64)
    if (
        values.shape != (len(classes), Z_DIM)
        or offset.shape != (len(classes),)
        or not np.isfinite(values).all()
        or not np.isfinite(offset).all()
    ):
        raise D129Joint6HeadsError("affine quantization input drift")
    fp16 = np.finfo(np.float16)
    scale_floor = float(fp16.tiny)
    maximum = np.max(np.abs(values), axis=1)
    intercept_peak = float(np.max(np.abs(offset), initial=0.0))
    wire_peak = max(
        intercept_peak,
        float(np.max(maximum, initial=0.0)) / INT8_MAX,
    )
    safe_peak = float(fp16.max)
    if wire_peak <= safe_peak:
        shared_exponent = 0
    else:
        shared_exponent = -int(math.ceil(math.log2(wire_peak / safe_peak)))
    shared_scale = math.ldexp(1.0, shared_exponent)
    if not math.isfinite(shared_scale) or shared_scale <= 0.0:
        raise D129Joint6HeadsError("affine shared logit scale is invalid")
    values = np.ldexp(values, shared_exponent)
    offset = np.ldexp(offset, shared_exponent)
    maximum = np.max(np.abs(values), axis=1)
    nonzero_maximum = maximum[maximum > 0.0]
    if len(nonzero_maximum) and bool(
        np.any(nonzero_maximum / INT8_MAX < scale_floor)
    ):
        raise D129Joint6HeadsError(
            "affine dynamic range cannot fit the FP16 scale wire"
        )
    scales64 = np.maximum(maximum / INT8_MAX, scale_floor)
    if not np.isfinite(scales64).all() or np.any(
        scales64 > float(np.finfo(np.float16).max)
    ):
        raise D129Joint6HeadsError("affine FP16 scale is not representable")
    scales = scales64.astype(np.float16)
    codes = np.clip(
        np.rint(values / scales.astype(np.float64)[:, None]), -INT8_MAX, INT8_MAX
    ).astype(np.int8)
    intercept16 = offset.astype(np.float16)
    if not np.isfinite(intercept16).all():
        raise D129Joint6HeadsError("affine FP16 intercept is not representable")
    nonzero_intercept = offset != 0.0
    intercept_cast_zero_count = int(
        np.count_nonzero(nonzero_intercept & (intercept16 == np.float16(0.0)))
    )
    intercept_subnormal_count = int(
        np.count_nonzero(nonzero_intercept & (np.abs(offset) < scale_floor))
    )
    state = D129AffineHeadState(
        head=head,
        classes=classes,
        active_k=active_k,
        weight_qint8=codes,
        scale_fp16=scales,
        intercept_fp16=intercept16,
    )
    return state, _freeze(
        {
            "schema": "cvs.phase2.d129.shared_affine_logit_scale.v1",
            "policy": "all_class_common_positive_power_of_two_before_quantization",
            "argmax_invariant_in_exact_arithmetic": True,
            "argmax_equivalence_scope": (
                "prequantized_common_positive_scaling_only"
            ),
            "quantized_any_query_argmax_equivalence_claim": False,
            "class_specific_clipping": False,
            "shared_logit_scale": shared_scale,
            "shared_logit_scale_exponent_base2": shared_exponent,
            "pre_scale_intercept_max_abs": intercept_peak,
            "pre_scale_wire_peak": wire_peak,
            "post_scale_intercept_max_abs": float(
                np.max(np.abs(offset), initial=0.0)
            ),
            "post_scale_fp16_scale_max": float(np.max(scales64, initial=0.0)),
            "nonzero_intercept_cast_zero_count": intercept_cast_zero_count,
            "nonzero_intercept_subnormal_count": intercept_subnormal_count,
            "zero_weight_row_count": int(np.count_nonzero(maximum == 0.0)),
            "fp16_safe_peak": safe_peak,
        }
    )


def _affine_logits(state: D129AffineHeadState, query: np.ndarray) -> np.ndarray:
    rows = _normalized_rows(query, name="affine query")
    logits = (
        rows.astype(np.float64) @ state.weight_qint8.astype(np.float64).T
        * state.scale_fp16.astype(np.float64)[None, :]
        + state.intercept_fp16.astype(np.float64)[None, :]
    )
    if not np.isfinite(logits).all():
        raise D129Joint6HeadsError("shared affine score became non-finite")
    return _readonly(logits, np.float32)


def _full_resource(
    *,
    state: D129AffineHeadState,
    support_rows: int,
    class_count: int,
    fit_wall_clock_ns: int,
) -> dict[str, Any]:
    explicit_workspace = int(
        support_rows * Z_DIM * 8
        + class_count * Z_DIM * 8
        + 3 * Z_DIM * Z_DIM * 8
        + class_count * Z_DIM * 8
        + class_count * 8
    )
    mac_equivalent = int(
        support_rows * Z_DIM * Z_DIM
        + 2 * Z_DIM * Z_DIM * Z_DIM
        + class_count * Z_DIM * Z_DIM
    )
    return {
        "schema": "cvs.phase2.d129.head_resource.v1",
        "head": FULL_HEAD,
        "active_k": state.active_k,
        "feature_dim": Z_DIM,
        "class_count": class_count,
        "shared_affine_wire": "int8_W[C,160]+fp16_scale[C]+fp16_intercept[C]",
        "deployed_numeric_state_bytes": state.numeric_state_bytes,
        "deployed_numeric_state_formula": "160C+2C+2C=164C_B",
        "query_head_macs_per_sample": Z_DIM * class_count,
        "query_state_bytes": 0,
        "fit_wall_clock_ns": int(fit_wall_clock_ns),
        "fit_wall_clock_sample_count": 1,
        "fit_wall_clock_statistic": "raw_single_call_not_threshold_evidence",
        "timing_threshold_claim_permitted": False,
        "fit_analytic_mac_equivalent": mac_equivalent,
        "fit_analytic_mac_formula": "N*d*d+2*d*d*d+C*d*d",
        "estimated_peak_explicit_numeric_workspace_bytes": explicit_workspace,
        "actual_peak_workspace_measured": False,
        "explicit_dense_matrix_elements_constructed": 3 * Z_DIM * Z_DIM,
        "explicit_spectral_factorization_count": 1,
        "explicit_linear_system_solve_count": 1,
        "sklearn_auto_shrinkage_group_fit_count": 2,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_batch_dependency": False,
    }


def _lite_resource(
    *,
    state: D129AffineHeadState,
    support_rows: int,
    class_count: int,
    fit_wall_clock_ns: int,
) -> dict[str, Any]:
    explicit_workspace = int(
        support_rows * Z_DIM * 8
        + class_count * Z_DIM * 8
        + 7 * Z_DIM * 8
        + class_count * Z_DIM * 8
        + class_count * 8
    )
    mac_equivalent = int(4 * support_rows * Z_DIM + 8 * Z_DIM + 2 * class_count * Z_DIM)
    return {
        "schema": "cvs.phase2.d129.head_resource.v1",
        "head": LITE_HEAD,
        "active_k": state.active_k,
        "feature_dim": Z_DIM,
        "class_count": class_count,
        "shared_affine_wire": "int8_W[C,160]+fp16_scale[C]+fp16_intercept[C]",
        "deployed_numeric_state_bytes": state.numeric_state_bytes,
        "deployed_numeric_state_formula": "160C+2C+2C=164C_B",
        "query_head_macs_per_sample": Z_DIM * class_count,
        "query_state_bytes": 0,
        "fit_wall_clock_ns": int(fit_wall_clock_ns),
        "fit_wall_clock_sample_count": 1,
        "fit_wall_clock_statistic": "raw_single_call_not_threshold_evidence",
        "timing_threshold_claim_permitted": False,
        "fit_analytic_mac_equivalent": mac_equivalent,
        "fit_analytic_mac_formula": "4*N*d+8*d+2*C*d",
        "estimated_peak_explicit_numeric_workspace_bytes": explicit_workspace,
        "actual_peak_workspace_measured": False,
        "explicit_dense_matrix_elements_constructed": 0,
        "explicit_spectral_factorization_count": 0,
        "explicit_linear_system_solve_count": 0,
        "sklearn_auto_shrinkage_group_fit_count": 0,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_batch_dependency": False,
    }


def _alias_fit(
    *, head: str, classes: tuple[str, ...], support_rows: int
) -> D129HeadFit:
    state = D129K1QKNNAliasState(head=head, classes=classes)
    fit_receipt = {
        "schema": "cvs.phase2.d129.registration_head_fit.v1",
        "head": head,
        "fit_mode": "exact_qknn_alias",
        "feature_dim": Z_DIM,
        "class_count": len(classes),
        "active_k": 1,
        "support_rows": support_rows,
        "support_only": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "historical_k1_equivalence_claim": False,
        "alias_reason": "k1_covariance_unidentifiable_160d_comparator",
        "same_formula_all_registered_classes": True,
        "class_label_permutation_equivariant": True,
    }
    resource = {
        "schema": "cvs.phase2.d129.head_resource.v1",
        "head": head,
        "active_k": 1,
        "feature_dim": Z_DIM,
        "class_count": len(classes),
        "fit_mode": "exact_qknn_alias",
        "incremental_deployed_numeric_state_bytes": 0,
        "incremental_query_head_macs_per_sample": 0,
        "underlying_qknn_resource_required": True,
        "underlying_qknn_resource_included": False,
        "historical_k1_equivalence_claim": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_batch_dependency": False,
    }
    return D129HeadFit(state=state, fit_receipt=fit_receipt, resource_receipt=resource)


def fit_d92_full160(
    support_zid160: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    old_class_count: int,
) -> D129HeadFit:
    """Fit historical D92's full-covariance mechanism on a shared 160-D cache."""

    rows, classes, indices, active_k, old_count = _support_contract(
        support_zid160, support_labels, registered_classes, old_class_count
    )
    if active_k == 1:
        return _alias_fit(head=FULL_HEAD, classes=classes, support_rows=len(rows))
    started = time.perf_counter_ns()
    rows64 = np.asarray(rows, dtype=np.float64)
    means = _means(rows64, indices, len(classes))
    old_indices = np.arange(old_count, dtype=np.int64)
    new_indices = np.arange(old_count, len(classes), dtype=np.int64)
    old_covariance, old_covariance_estimator = _auto_shrinkage_full_covariance(
        rows64, indices, old_indices
    )
    new_covariance, new_covariance_estimator = _auto_shrinkage_full_covariance(
        rows64, indices, new_indices
    )
    covariance = 0.5 * (old_covariance + new_covariance)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.isfinite(eigenvalues).all() or float(np.min(eigenvalues)) <= 0.0:
        raise D129Joint6HeadsError("D92-Full160 balanced covariance is not positive definite")
    coefficients = np.linalg.solve(covariance, means.T).T
    priors = np.full(len(classes), 1.0 / len(classes), dtype=np.float64)
    intercepts = -0.5 * np.diag(means @ coefficients.T) + np.log(priors)
    equation_residual = float(
        np.max(np.abs(covariance @ coefficients.T - means.T))
    )
    coefficients -= coefficients.mean(axis=0, keepdims=True)
    intercepts -= intercepts.mean()
    if not np.isfinite(coefficients).all() or not np.isfinite(intercepts).all():
        raise D129Joint6HeadsError("D92-Full160 affine state became non-finite")
    state, quantization_audit = _quantize_shared_affine(
        head=FULL_HEAD,
        classes=classes,
        active_k=active_k,
        weights=coefficients,
        intercepts=intercepts,
    )
    elapsed = time.perf_counter_ns() - started
    fit_receipt = {
        "schema": "cvs.phase2.d129.registration_head_fit.v1",
        "head": FULL_HEAD,
        "fit_mode": "old_new_task_balanced_auto_shrinkage_full_covariance",
        "feature_dim": Z_DIM,
        "class_count": len(classes),
        "active_k": active_k,
        "support_rows": int(len(rows)),
        "old_class_count": old_count,
        "new_class_count": len(classes) - old_count,
        "old_covariance_weight": 0.5,
        "new_covariance_weight": 0.5,
        "prior_policy": "equal_1_over_registered_class_count",
        "covariance_policy": "per-task_sklearn_auto_shrinkage_full_covariance",
        "old_covariance_estimator": old_covariance_estimator,
        "new_covariance_estimator": new_covariance_estimator,
        "single_class_group_policy": "sklearn_LedoitWolf_centered_residuals_when_needed",
        "class_common_affine_centered_before_quantization": True,
        "shared_logit_scale_audit": dict(quantization_audit),
        "prequantized_weight_class_mean_max_abs": float(
            np.max(np.abs(coefficients.mean(axis=0)))
        ),
        "prequantized_intercept_class_mean_abs": float(abs(intercepts.mean())),
        "balanced_covariance_eigenvalue_min": float(np.min(eigenvalues)),
        "balanced_covariance_eigenvalue_max": float(np.max(eigenvalues)),
        "balanced_covariance_trace": float(np.trace(covariance)),
        "old_covariance_trace": float(np.trace(old_covariance)),
        "new_covariance_trace": float(np.trace(new_covariance)),
        "covariance_equation_residual_max": equation_residual,
        "support_only": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_role_access": False,
        "same_formula_all_registered_classes": True,
        "class_label_permutation_equivariant": True,
        "state_sha256": state.state_sha256,
    }
    return D129HeadFit(
        state=state,
        fit_receipt=fit_receipt,
        resource_receipt=_full_resource(
            state=state,
            support_rows=len(rows),
            class_count=len(classes),
            fit_wall_clock_ns=elapsed,
        ),
    )


def fit_d92_lite160(
    support_zid160: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    old_class_count: int,
) -> D129HeadFit:
    """Fit task-balanced diagonal OAS-LDA on the same shared 160-D cache."""

    rows, classes, indices, active_k, old_count = _support_contract(
        support_zid160, support_labels, registered_classes, old_class_count
    )
    if active_k == 1:
        return _alias_fit(head=LITE_HEAD, classes=classes, support_rows=len(rows))
    started = time.perf_counter_ns()
    rows64 = np.asarray(rows, dtype=np.float64)
    means = _means(rows64, indices, len(classes))
    old_indices = np.arange(old_count, dtype=np.int64)
    new_indices = np.arange(old_count, len(classes), dtype=np.int64)
    old_variance, old_audit = _oas_diagonal_covariance(rows64, indices, old_indices)
    new_variance, new_audit = _oas_diagonal_covariance(rows64, indices, new_indices)
    variance = 0.5 * (old_variance + new_variance)
    if not np.isfinite(variance).all() or np.any(variance <= 0.0):
        raise D129Joint6HeadsError("D92-Lite160 task-balanced variance drift")
    coefficients = means / variance[None, :]
    priors = np.full(len(classes), 1.0 / len(classes), dtype=np.float64)
    intercepts = -0.5 * np.sum(means * coefficients, axis=1) + np.log(priors)
    coefficients -= coefficients.mean(axis=0, keepdims=True)
    intercepts -= intercepts.mean()
    if not np.isfinite(coefficients).all() or not np.isfinite(intercepts).all():
        raise D129Joint6HeadsError("D92-Lite160 affine state became non-finite")
    state, quantization_audit = _quantize_shared_affine(
        head=LITE_HEAD,
        classes=classes,
        active_k=active_k,
        weights=coefficients,
        intercepts=intercepts,
    )
    elapsed = time.perf_counter_ns() - started
    fit_receipt = {
        "schema": "cvs.phase2.d129.registration_head_fit.v1",
        "head": LITE_HEAD,
        "fit_mode": "old_new_task_balanced_diagonal_oas",
        "feature_dim": Z_DIM,
        "class_count": len(classes),
        "active_k": active_k,
        "support_rows": int(len(rows)),
        "old_class_count": old_count,
        "new_class_count": len(classes) - old_count,
        "old_covariance_weight": 0.5,
        "new_covariance_weight": 0.5,
        "prior_policy": "equal_1_over_registered_class_count",
        "covariance_policy": "diagonal_oas_per_registration_task_equal_average",
        "class_common_affine_centered_before_quantization": True,
        "shared_logit_scale_audit": dict(quantization_audit),
        "prequantized_weight_class_mean_max_abs": float(
            np.max(np.abs(coefficients.mean(axis=0)))
        ),
        "prequantized_intercept_class_mean_abs": float(abs(intercepts.mean())),
        "old_diagonal_oas": old_audit,
        "new_diagonal_oas": new_audit,
        "balanced_variance_trace": float(np.sum(variance, dtype=np.float64)),
        "support_only": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_role_access": False,
        "same_formula_all_registered_classes": True,
        "class_label_permutation_equivariant": True,
        "state_sha256": state.state_sha256,
    }
    return D129HeadFit(
        state=state,
        fit_receipt=fit_receipt,
        resource_receipt=_lite_resource(
            state=state,
            support_rows=len(rows),
            class_count=len(classes),
            fit_wall_clock_ns=elapsed,
        ),
    )


def score_d129_affine_head(state: D129AffineHeadState, query_zid160: np.ndarray) -> np.ndarray:
    """Score one active registration head; aliases are intentionally excluded."""

    if type(state) is not D129AffineHeadState:
        raise D129Joint6HeadsError("active affine score requires an affine state")
    return _affine_logits(state, query_zid160)


def _fit_and_score_qknn(
    cache: D129RepresentationCache,
    labels: Sequence[str],
    classes: tuple[str, ...],
    lock: qknn.Phase1ZIDStudentTLock,
) -> tuple[D129QKNNState, np.ndarray, Mapping[str, Any]]:
    started = time.perf_counter_ns()
    bank = qknn.build_typed_zid_support_bank(
        cache.support_zid160, labels, classes, config=lock
    )
    metric = qknn.identity_shared_psd_metric(config=lock)
    fit_elapsed = time.perf_counter_ns() - started
    logits = qknn.score_zid_student_t_logits(
        bank, cache.query_zid160, metric=metric
    )
    resource = qknn.audit_runtime_state(bank, metric)
    state = D129QKNNState(bank=bank, metric=metric, resource_receipt=resource)
    receipt = {
        "schema": "cvs.phase2.d129.qknn_head.v1",
        "head": QKNN_HEAD,
        "representation": cache.representation,
        "cache_sha256": cache.cache_sha256,
        "fit_wall_clock_ns": int(fit_elapsed),
        "support_only": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_batch_dependency": False,
        "all_registered_classes_scored": True,
        "query_role_access": False,
        "source_runtime_access": False,
        "clean_runtime_access": False,
    }
    return state, logits, _freeze(receipt)


def _alias_receipt(
    *,
    head: str,
    classes: tuple[str, ...],
    opaque_query_ids: tuple[str, ...],
    cache: D129RepresentationCache,
    qknn_logits: np.ndarray,
) -> D129K1AliasReceipt:
    return D129K1AliasReceipt(
        head=head,
        classes=classes,
        opaque_query_ids=opaque_query_ids,
        query_cache_sha256=cache.cache_sha256,
        qknn_logits_sha256=_array_receipt(qknn_logits)["sha256"],
    )


def _predictions(logits: np.ndarray, classes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(classes[int(index)] for index in np.argmax(logits, axis=1))


def _qknn_arm(
    *,
    arm_id: str,
    cache: D129RepresentationCache,
    classes: tuple[str, ...],
    state: D129QKNNState,
    logits: np.ndarray,
    q_receipt: Mapping[str, Any],
) -> D129ArmPrediction:
    receipt = dict(q_receipt)
    receipt.update(
        {
            "arm_id": arm_id,
            "same_row_shared_cache": True,
            "head_state_resource_sha256": state.resource_receipt[
                "serialized_state_sha256"
            ],
        }
    )
    return D129ArmPrediction(
        arm_id=arm_id,
        representation=cache.representation,
        head=QKNN_HEAD,
        classes=classes,
        logits=logits,
        predictions=_predictions(logits, classes),
        state=state,
        receipt=receipt,
    )


def _registration_arm(
    *,
    arm_id: str,
    cache: D129RepresentationCache,
    classes: tuple[str, ...],
    fit: D129HeadFit,
    qknn_arm: D129ArmPrediction,
    opaque_query_ids: tuple[str, ...],
) -> D129ArmPrediction:
    state = fit.state
    if type(state) is D129K1QKNNAliasState:
        alias = _alias_receipt(
            head=state.head,
            classes=classes,
            opaque_query_ids=opaque_query_ids,
            cache=cache,
            qknn_logits=qknn_arm.logits,
        )
        logits = qknn_arm.logits
        receipt = {
            "schema": "cvs.phase2.d129.registration_arm.v1",
            "arm_id": arm_id,
            "representation": cache.representation,
            "head": state.head,
            "fit_mode": "exact_qknn_alias",
            "cache_sha256": cache.cache_sha256,
            "underlying_qknn_arm": qknn_arm.arm_id,
            "underlying_qknn_logit_object_reused": True,
            "historical_k1_equivalence_claim": False,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "query_batch_dependency": False,
            "all_registered_classes_scored": True,
            "query_role_access": False,
            "source_runtime_access": False,
            "clean_runtime_access": False,
        }
        return D129ArmPrediction(
            arm_id=arm_id,
            representation=cache.representation,
            head=state.head,
            classes=classes,
            logits=logits,
            predictions=qknn_arm.predictions,
            state=state,
            receipt=receipt,
            qknn_alias_receipt=alias,
        )
    logits = _affine_logits(state, cache.query_zid160)
    receipt = {
        "schema": "cvs.phase2.d129.registration_arm.v1",
        "arm_id": arm_id,
        "representation": cache.representation,
        "head": state.head,
        "fit_mode": fit.fit_receipt["fit_mode"],
        "cache_sha256": cache.cache_sha256,
        "head_state_sha256": state.state_sha256,
        "shared_logit_scale_audit": dict(
            fit.fit_receipt["shared_logit_scale_audit"]
        ),
        "same_row_shared_cache": True,
        "same_affine_wire_as_other_registration_head": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_batch_dependency": False,
        "all_registered_classes_scored": True,
        "query_role_access": False,
        "source_runtime_access": False,
        "clean_runtime_access": False,
    }
    return D129ArmPrediction(
        arm_id=arm_id,
        representation=cache.representation,
        head=state.head,
        classes=classes,
        logits=logits,
        predictions=_predictions(logits, classes),
        state=state,
        receipt=receipt,
    )


def _head_causal_resource_receipt(
    *,
    full_r0: D129HeadFit,
    lite_r0: D129HeadFit,
    full_r1: D129HeadFit,
    lite_r1: D129HeadFit,
    q_r0: D129QKNNState,
    q_r1: D129QKNNState,
) -> Mapping[str, Any]:
    pairs = {
        R0: (full_r0, lite_r0, q_r0),
        R1: (full_r1, lite_r1, q_r1),
    }
    per_representation: dict[str, Any] = {}
    for representation, (full_fit, lite_fit, q_state) in pairs.items():
        full_resource = dict(full_fit.resource_receipt)
        lite_resource = dict(lite_fit.resource_receipt)
        if type(full_fit.state) is D129AffineHeadState:
            if (
                type(lite_fit.state) is not D129AffineHeadState
                or full_resource["deployed_numeric_state_bytes"]
                != lite_resource["deployed_numeric_state_bytes"]
                or full_resource["query_head_macs_per_sample"]
                != lite_resource["query_head_macs_per_sample"]
            ):
                raise D129Joint6HeadsError("Full160/Lite160 shared-wire resource drift")
        per_representation[representation] = {
            "full160": full_resource,
            "lite160": lite_resource,
            "qknn": dict(q_state.resource_receipt),
            "same_160d_cache": True,
            "same_affine_wire_for_full160_lite160": True,
            "head_state_byte_reduction_claimed": False,
            "k1_alias": type(full_fit.state) is D129K1QKNNAliasState,
        }
    return _freeze(
        {
            "schema": "cvs.phase2.d129.head_causal_resource_receipt.v1",
            "comparison_scope": "same_160d_representation_same_affine_wire",
            "not_formal288_replacement_receipt": True,
            "representations": per_representation,
        }
    )


def build_system_formal_replacement_resource_receipt(
    *, class_count: int, da_numeric_state_bytes: int = 0
) -> Mapping[str, Any]:
    """Report the external formal288-to-Lite160 system comparison only."""

    count = int(class_count)
    da_bytes = int(da_numeric_state_bytes)
    if count < 1 or da_bytes < 0:
        raise D129Joint6HeadsError("system resource class/DA byte count drift")
    formal_bytes = 1152 + 590 * count
    lite_bytes = 164 * count
    formal_query_macs = 288 * count
    lite_query_macs = 160 * count
    return _freeze(
        {
            "schema": "cvs.phase2.d129.system_formal_replacement_resource_receipt.v1",
            "formal_reference": "historical_D92_288d_full_pipeline",
            "formal_d92_numeric_state_formula": "1152+590C_B",
            "formal_d92_numeric_state_bytes": formal_bytes,
            "lite160_numeric_state_formula": "164C_B",
            "lite160_numeric_state_bytes": lite_bytes,
            "da_numeric_state_bytes": da_bytes,
            "joint_lite160_da_numeric_state_bytes": lite_bytes + da_bytes,
            "formal_to_lite160_state_reduction_fraction": (
                float(formal_bytes - lite_bytes) / float(formal_bytes)
            ),
            "formal_d92_affine_query_macs_per_sample": formal_query_macs,
            "lite160_affine_query_macs_per_sample": lite_query_macs,
            "formal_to_lite160_affine_query_mac_reduction_fraction": (
                float(formal_query_macs - lite_query_macs)
                / float(formal_query_macs)
            ),
            "timing_median_and_actual_peak_required_for_formal_thresholds": True,
            "formal_efficiency_thresholds_evaluated": False,
            "representation_pipeline_changed": True,
            "not_head_causal_comparator": True,
            "performance_causal_claim_permitted": False,
        }
    )


@dataclass(frozen=True, slots=True)
class D129CommonR0:
    """One base-representation head fit shared by every DA candidate in a row."""

    cache: D129RepresentationCache
    q_state: D129QKNNState
    full_fit: D129HeadFit
    lite_fit: D129HeadFit
    arms: tuple[D129ArmPrediction, D129ArmPrediction, D129ArmPrediction]
    registered_classes: tuple[str, ...]
    support_labels: tuple[str, ...]
    old_class_count: int
    active_k: int
    opaque_query_ids: tuple[str, ...]
    partition_semantics: str
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            self.cache.representation != R0
            or tuple(arm.arm_id for arm in self.arms) != ("R0Q", "R0F", "R0L")
            or self.receipt.get("common_r0_head_fit_count") != 3
            or self.receipt.get("candidate_specific") is not False
        ):
            raise D129Joint6HeadsError("common R0 closure drift")
        object.__setattr__(self, "receipt", _freeze(self.receipt))


def build_d129_common_r0(
    *,
    base_support_zid: np.ndarray,
    base_query_zid: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    old_class_count: int,
    partition_semantics: str,
    opaque_query_ids: Sequence[str],
    qknn_lock: qknn.Phase1ZIDStudentTLock,
) -> D129CommonR0:
    """Fit R0Q/R0F/R0L exactly once for one frozen atomic row."""

    if partition_semantics not in {
        "phase1_seen_class_loco_directional_proxy",
        "formal_stage2c_old_new_registration",
    }:
        raise D129Joint6HeadsError("task partition semantics must be explicit")
    if type(qknn_lock) is not qknn.Phase1ZIDStudentTLock:
        raise D129Joint6HeadsError("Q requires an exact frozen Phase1 qKNN lock")
    cache = _representation_cache(R0, base_support_zid, base_query_zid)
    _support, classes, _indices, active_k, normalized_old_count = _support_contract(
        cache.support_zid160, support_labels, registered_classes, old_class_count
    )
    if qknn_lock.active_k != active_k:
        raise D129Joint6HeadsError("Phase1 qKNN lock K/support K drift")
    query_ids = _opaque_ids(opaque_query_ids, expected=len(cache.query_zid160))
    q_state, q_logits, q_receipt = _fit_and_score_qknn(
        cache, support_labels, classes, qknn_lock
    )
    q_arm = _qknn_arm(
        arm_id="R0Q",
        cache=cache,
        classes=classes,
        state=q_state,
        logits=q_logits,
        q_receipt=q_receipt,
    )
    full_fit = fit_d92_full160(
        cache.support_zid160,
        support_labels,
        classes,
        old_class_count=normalized_old_count,
    )
    lite_fit = fit_d92_lite160(
        cache.support_zid160,
        support_labels,
        classes,
        old_class_count=normalized_old_count,
    )
    full_arm = _registration_arm(
        arm_id="R0F",
        cache=cache,
        classes=classes,
        fit=full_fit,
        qknn_arm=q_arm,
        opaque_query_ids=query_ids,
    )
    lite_arm = _registration_arm(
        arm_id="R0L",
        cache=cache,
        classes=classes,
        fit=lite_fit,
        qknn_arm=q_arm,
        opaque_query_ids=query_ids,
    )
    receipt = {
        "schema": "cvs.phase2.d129.common_r0.v1",
        "r0_cache_sha256": cache.cache_sha256,
        "registered_classes": list(classes),
        "support_labels": [str(value) for value in support_labels],
        "old_class_count": normalized_old_count,
        "active_k": active_k,
        "opaque_query_ids": list(query_ids),
        "partition_semantics": partition_semantics,
        "qknn_lock_digest": qknn_lock.lock_digest,
        "common_r0_head_fit_count": 3,
        "candidate_specific": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
    }
    receipt["common_r0_sha256"] = _canonical_sha256(receipt)
    return D129CommonR0(
        cache=cache,
        q_state=q_state,
        full_fit=full_fit,
        lite_fit=lite_fit,
        arms=(q_arm, full_arm, lite_arm),
        registered_classes=classes,
        support_labels=tuple(str(value) for value in support_labels),
        old_class_count=normalized_old_count,
        active_k=active_k,
        opaque_query_ids=query_ids,
        partition_semantics=partition_semantics,
        receipt=receipt,
    )


def run_d129_joint6_heads(
    *,
    base_support_zid: np.ndarray,
    adapted_support_zid: np.ndarray,
    base_query_zid: np.ndarray,
    adapted_query_zid: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    old_class_count: int,
    partition_semantics: str,
    opaque_query_ids: Sequence[str],
    qknn_lock: qknn.Phase1ZIDStudentTLock,
    common_r0: D129CommonR0 | None = None,
    da_numeric_state_bytes: int = 0,
) -> D129Joint6Result:
    """Build all six logical arms from two same-row 160-D caches.

    The caller owns the support-only DA transform and must supply its output as
    ``adapted_*``.  This interface performs zero backbone calls and accepts no
    query labels, roles, quotas, scorer outputs, or source/clean state.
    """

    if partition_semantics not in {
        "phase1_seen_class_loco_directional_proxy",
        "formal_stage2c_old_new_registration",
    }:
        raise D129Joint6HeadsError("task partition semantics must be explicit")
    if type(qknn_lock) is not qknn.Phase1ZIDStudentTLock:
        raise D129Joint6HeadsError("Q requires an exact frozen Phase1 qKNN lock")
    supplied_common_r0 = common_r0 is not None
    candidate_base_cache = _representation_cache(R0, base_support_zid, base_query_zid)
    if common_r0 is None:
        common_r0 = build_d129_common_r0(
            base_support_zid=base_support_zid,
            base_query_zid=base_query_zid,
            support_labels=support_labels,
            registered_classes=registered_classes,
            old_class_count=old_class_count,
            partition_semantics=partition_semantics,
            opaque_query_ids=opaque_query_ids,
            qknn_lock=qknn_lock,
        )
    if (
        type(common_r0) is not D129CommonR0
        or candidate_base_cache.cache_sha256 != common_r0.cache.cache_sha256
        or tuple(str(value) for value in support_labels) != common_r0.support_labels
        or tuple(str(value) for value in registered_classes)
        != common_r0.registered_classes
        or int(old_class_count) != common_r0.old_class_count
        or tuple(str(value) for value in opaque_query_ids)
        != common_r0.opaque_query_ids
        or partition_semantics != common_r0.partition_semantics
        or qknn_lock.lock_digest != common_r0.receipt.get("qknn_lock_digest")
    ):
        raise D129Joint6HeadsError("supplied common R0 row binding drift")
    r0_cache = common_r0.cache
    r1_cache = _representation_cache(R1, adapted_support_zid, adapted_query_zid)
    if (
        r0_cache.support_zid160.shape != r1_cache.support_zid160.shape
        or r0_cache.query_zid160.shape != r1_cache.query_zid160.shape
    ):
        raise D129Joint6HeadsError("R0/R1 same-row cache shape closure drift")
    support, classes, _indices, active_k, normalized_old_count = _support_contract(
        r0_cache.support_zid160,
        support_labels,
        registered_classes,
        old_class_count,
    )
    _support_contract(
        r1_cache.support_zid160,
        support_labels,
        classes,
        normalized_old_count,
    )
    if qknn_lock.active_k != active_k:
        raise D129Joint6HeadsError("Phase1 qKNN lock K/support K drift")
    query_ids = _opaque_ids(opaque_query_ids, expected=len(r0_cache.query_zid160))

    r1_q_state, r1_q_logits, r1_q_receipt = _fit_and_score_qknn(
        r1_cache, support_labels, classes, qknn_lock
    )
    r0_q, r0_f, r0_l = common_r0.arms
    r0_q_state = common_r0.q_state
    r1_q = _qknn_arm(
        arm_id="R1Q",
        cache=r1_cache,
        classes=classes,
        state=r1_q_state,
        logits=r1_q_logits,
        q_receipt=r1_q_receipt,
    )
    r0_full = common_r0.full_fit
    r0_lite = common_r0.lite_fit
    r1_full = fit_d92_full160(
        r1_cache.support_zid160,
        support_labels,
        classes,
        old_class_count=normalized_old_count,
    )
    r1_lite = fit_d92_lite160(
        r1_cache.support_zid160,
        support_labels,
        classes,
        old_class_count=normalized_old_count,
    )
    r1_f = _registration_arm(
        arm_id="R1F",
        cache=r1_cache,
        classes=classes,
        fit=r1_full,
        qknn_arm=r1_q,
        opaque_query_ids=query_ids,
    )
    r1_l = _registration_arm(
        arm_id="R1L",
        cache=r1_cache,
        classes=classes,
        fit=r1_lite,
        qknn_arm=r1_q,
        opaque_query_ids=query_ids,
    )
    arms = (r0_q, r0_f, r0_l, r1_q, r1_f, r1_l)
    row_receipt = {
        "schema": SCHEMA,
        "arms": list(ARM_IDS),
        "feature_dim": Z_DIM,
        "active_k": active_k,
        "registered_classes": list(classes),
        "old_class_count": normalized_old_count,
        "new_class_count": len(classes) - normalized_old_count,
        "partition_semantics": partition_semantics,
        "formal_new_registration_claim": (
            partition_semantics == "formal_stage2c_old_new_registration"
        ),
        "full160_single_class_proxy_extension": (
            len(classes) - normalized_old_count == 1
        ),
        "full160_strict_historical_d92_group_covariance_path": (
            len(classes) - normalized_old_count > 1
        ),
        "support_labels": [str(value) for value in support_labels],
        "opaque_query_ids": list(query_ids),
        "r0_cache_sha256": r0_cache.cache_sha256,
        "common_r0_sha256": common_r0.receipt["common_r0_sha256"],
        "common_r0_supplied_by_caller": supplied_common_r0,
        "common_r0_head_fit_calls_in_this_candidate_call": (
            0 if supplied_common_r0 else 3
        ),
        "r1_cache_sha256": r1_cache.cache_sha256,
        "same_row_six_arm_binding": True,
        "same_support_labels_across_r0_r1": True,
        "same_query_order_across_r0_r1": True,
        "backbone_forward_calls_in_joint6_interface": 0,
        "representation_cache_count": 2,
        "heads_per_representation": 3,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_batch_dependency": False,
        "all_registered_classes_scored": True,
        "query_role_access": False,
        "source_runtime_access": False,
        "clean_runtime_access": False,
        "truth_input_exists": False,
        "global_reassignment_exists": False,
        "class_quota_input_exists": False,
    }
    causal_resource = _head_causal_resource_receipt(
        full_r0=r0_full,
        lite_r0=r0_lite,
        full_r1=r1_full,
        lite_r1=r1_lite,
        q_r0=r0_q_state,
        q_r1=r1_q_state,
    )
    system_resource = build_system_formal_replacement_resource_receipt(
        class_count=len(classes), da_numeric_state_bytes=da_numeric_state_bytes
    )
    return D129Joint6Result(
        arms=arms,
        r0_cache=r0_cache,
        r1_cache=r1_cache,
        row_receipt=row_receipt,
        head_causal_resource_receipt=causal_resource,
        system_formal_replacement_resource_receipt=system_resource,
    )


__all__ = [
    "ACTIVE_K_VALUES",
    "AFFINE_SCHEMA",
    "ALIAS_RECEIPT_SCHEMA",
    "ALIAS_SCHEMA",
    "ARM_IDS",
    "D129AffineHeadState",
    "D129ArmPrediction",
    "D129CommonR0",
    "D129HeadFit",
    "D129Joint6HeadsError",
    "D129Joint6Result",
    "D129K1AliasReceipt",
    "D129K1QKNNAliasState",
    "D129QKNNState",
    "D129RegistrationHeadState",
    "D129RepresentationCache",
    "FULL_HEAD",
    "INT8_MAX",
    "LITE_HEAD",
    "QKNN_HEAD",
    "R0",
    "R1",
    "SCHEMA",
    "Z_DIM",
    "build_system_formal_replacement_resource_receipt",
    "build_d129_common_r0",
    "fit_d92_full160",
    "fit_d92_lite160",
    "normalize_zid160_rows",
    "run_d129_joint6_heads",
    "score_d129_affine_head",
]
