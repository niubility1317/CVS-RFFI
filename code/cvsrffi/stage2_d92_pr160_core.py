"""D92-Lite-PR160 support-only registration core.

The candidate repairs D131's lossy ``registered_feature[:, :160]`` tap by
consuming the same-forward signed-totalized ``joint_proj.0`` representation.
K1 is an exact qKNN alias.  K5/K10 use one all-class diagonal affine head; no
old/new split or query-side fit is present.  A float32 precision-alias tie may
only be resolved by a unique winner in the same score's pre-cast float64
values; a genuine high-precision tie remains fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from . import stage2_d129_joint6_heads as d129
from .stage2_adv3b02_ts_drqknn_bcrr import phase1_qknn_lock
from .stage2_zid_student_t_qknn import (
    TypedINT8ZIDSupportBank,
    TypedSharedPSDMetric,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    score_zid_student_t_logits_float64,
)
from .stage2_d92_pr160_runtime import (
    EXTRACTOR_RUNTIME_SHA256,
    SOURCE_RUNTIME_SHA256,
)


METHOD_LOCK_SHA256 = "256aacf7b6f790ce213ac27c1bb496be1a964cbf4f21cdd46309630235fb3ca4"
METHOD_LOCK_SCHEMA = "cvs.phase2.d138.d92_lite_pr160.method_lock.v2"
CANDIDATE_ID = "D92-Lite-PR160/r2"
PROTOCOL_SCHEMA = "p2_min_v1"
TRANSPORT_ARM = "M_JOINT"
OLD_CLASS_COUNT = 6
ZID_WIDTH = 160
REGISTERED_FEATURE_WIDTH = ZID_WIDTH


class D92PR160CoreError(ValueError):
    """Raised when the frozen D92-Lite-PR160 core fails closed."""


@dataclass(frozen=True, slots=True)
class SharedDiagAffineState:
    classes: tuple[str, ...]
    active_k: int
    weight_qint8: np.ndarray
    scale_fp16: np.ndarray
    intercept_fp16: np.ndarray
    fit_receipt: Mapping[str, Any]
    resource_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        classes = tuple(str(item) for item in self.classes)
        q = np.asarray(self.weight_qint8)
        scale = np.asarray(self.scale_fp16)
        intercept = np.asarray(self.intercept_fp16)
        if (
            len(classes) < 2
            or len(set(classes)) != len(classes)
            or any(not item for item in classes)
            or int(self.active_k) not in (5, 10)
            or q.dtype != np.int8
            or q.shape != (len(classes), ZID_WIDTH)
            or bool(np.any(q == np.int8(-128)))
            or scale.dtype != np.float16
            or scale.shape != (len(classes),)
            or intercept.dtype != np.float16
            or intercept.shape != (len(classes),)
            or not np.isfinite(scale).all()
            or not np.isfinite(intercept).all()
            or bool(np.any(scale <= 0.0))
        ):
            raise D92PR160CoreError("shared diagonal affine wire drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "active_k", int(self.active_k))
        object.__setattr__(self, "weight_qint8", _readonly(q, np.int8))
        object.__setattr__(self, "scale_fp16", _readonly(scale, np.float16))
        object.__setattr__(self, "intercept_fp16", _readonly(intercept, np.float16))

    @property
    def numeric_state_bytes(self) -> int:
        return int(self.weight_qint8.nbytes + self.scale_fp16.nbytes + self.intercept_fp16.nbytes)

    def dequantized(self) -> tuple[np.ndarray, np.ndarray]:
        weights = self.weight_qint8.astype(np.float64) * self.scale_fp16.astype(
            np.float64
        )[:, None]
        intercepts = self.intercept_fp16.astype(np.float64)
        return weights, intercepts


@dataclass(frozen=True, slots=True)
class D92PR160Pair:
    before_bank: TypedINT8ZIDSupportBank
    before_metric: TypedSharedPSDMetric
    after_bank: TypedINT8ZIDSupportBank
    after_metric: TypedSharedPSDMetric
    after_lite_state: SharedDiagAffineState | None
    old_registered_classes: tuple[str, ...]
    registered_classes: tuple[str, ...]
    active_k: int
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            self.old_registered_classes != self.before_bank.classes
            or self.registered_classes != self.after_bank.classes
            or self.registered_classes[:OLD_CLASS_COUNT] != self.old_registered_classes
            or self.active_k != self.before_bank.active_k
            or self.active_k != self.after_bank.active_k
        ):
            raise D92PR160CoreError("pair registry or K closure drift")
        if self.active_k == 1 and self.after_lite_state is not None:
            raise D92PR160CoreError("K1 must not contain an affine state")
        if self.active_k in (5, 10) and (
            type(self.after_lite_state) is not SharedDiagAffineState
            or self.after_lite_state.classes != self.registered_classes
        ):
            raise D92PR160CoreError("K5/K10 requires the exact shared affine state")


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _unit_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != ZID_WIDTH
        or rows.shape[0] < 1
        or not np.isfinite(rows).all()
    ):
        raise D92PR160CoreError(f"{name} must be finite float32 [N,160]")
    norms = np.sqrt(np.sum(rows.astype(np.float64) ** 2, axis=1))
    if not np.isfinite(norms).all() or not np.allclose(
        norms, 1.0, atol=2.0e-6, rtol=0.0
    ):
        raise D92PR160CoreError(f"{name} must be signed-totalized unit rows")
    return np.ascontiguousarray(rows, dtype=np.float32)


def _texts(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise D92PR160CoreError(f"{name} must be a string sequence")
    result = tuple(str(item) for item in value)
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        raise D92PR160CoreError(f"{name} must contain unique nonempty values")
    return result


def _balanced_k(labels: tuple[str, ...], classes: tuple[str, ...]) -> int:
    counts = tuple(labels.count(item) for item in classes)
    if any(count < 1 for count in counts) or len(set(counts)) != 1:
        raise D92PR160CoreError("support must be balanced over every class")
    if counts[0] not in (1, 5, 10):
        raise D92PR160CoreError("Target125 only permits K1/K5/K10")
    return counts[0]


def _quantize(
    classes: tuple[str, ...], active_k: int, weights: np.ndarray, intercepts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
    try:
        state, audit = d129._quantize_shared_affine(
            head=d129.LITE_HEAD,
            classes=classes,
            active_k=active_k,
            weights=weights,
            intercepts=intercepts,
        )
    except Exception as error:
        raise D92PR160CoreError("shared diagonal affine quantization failed") from error
    return (
        np.array(state.weight_qint8, copy=True),
        np.array(state.scale_fp16, copy=True),
        np.array(state.intercept_fp16, copy=True),
        dict(audit),
    )


def _fit_shared_diag(
    rows: np.ndarray, labels: tuple[str, ...], classes: tuple[str, ...], active_k: int
) -> SharedDiagAffineState:
    started = __import__("time").perf_counter_ns()
    data = rows.astype(np.float64)
    indices = np.asarray([classes.index(label) for label in labels], dtype=np.int64)
    means = np.stack([data[indices == index].mean(axis=0) for index in range(len(classes))])
    residuals = data - means[indices]
    degrees = len(data) - len(classes)
    if degrees <= 0:
        raise D92PR160CoreError("shared diagonal variance is not identifiable")
    scatter = np.sum(residuals * residuals, axis=0, dtype=np.float64) / float(degrees)
    total = float(np.sum(scatter, dtype=np.float64))
    second = float(np.sum(scatter * scatter, dtype=np.float64))
    tau = total / float(ZID_WIDTH)
    delta = second - total * total / float(ZID_WIDTH)
    if not all(np.isfinite(value) for value in (total, second, tau, delta)) or total <= 0.0:
        raise D92PR160CoreError("shared diagonal variance is non-finite")
    if delta <= 0.0:
        shrinkage = 1.0
    else:
        numerator = (1.0 - 2.0 / ZID_WIDTH) * second + total * total
        denominator = (float(degrees) + 1.0 - 2.0 / ZID_WIDTH) * delta
        shrinkage = min(1.0, numerator / denominator)
    if not np.isfinite(shrinkage) or shrinkage < 0.0:
        raise D92PR160CoreError("shared diagonal shrinkage is invalid")
    floor = max(float(np.finfo(np.float64).tiny), float(np.finfo(np.float64).eps) * max(1.0, tau))
    variance = np.maximum((1.0 - shrinkage) * scatter + shrinkage * tau, floor)
    if not np.isfinite(variance).all() or np.any(variance <= 0.0):
        raise D92PR160CoreError("shared diagonal variance closure drift")
    weights = means / variance[None, :]
    intercepts = -0.5 * np.sum(means * weights, axis=1) - np.log(len(classes))
    weights -= weights.mean(axis=0, keepdims=True)
    intercepts -= intercepts.mean()
    if not np.isfinite(weights).all() or not np.isfinite(intercepts).all():
        raise D92PR160CoreError("shared diagonal affine state became non-finite")
    q, scale, intercept, quantization = _quantize(classes, active_k, weights, intercepts)
    fit_receipt = {
        "schema": "cvs.phase2.d138.d92_lite_pr160.fit.v1",
        "head": "shared_all_class_diagonal_affine",
        "fit_mode": "all_class_shared_diagonal_oas",
        "feature_dim": ZID_WIDTH,
        "class_count": len(classes),
        "active_k": active_k,
        "support_rows": len(rows),
        "old_new_role_access": False,
        "class_common_affine_centered_before_quantization": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_role_access": False,
        "pooled_residual_degrees_of_freedom": degrees,
        "diagonal_shrinkage": float(shrinkage),
        "variance_floor": float(floor),
        "variance_trace": float(np.sum(variance, dtype=np.float64)),
        "shared_logit_scale_audit": quantization,
    }
    resource = {
        "schema": "cvs.phase2.d138.d92_lite_pr160.resource.v1",
        "head": "shared_all_class_diagonal_affine",
        "active_k": active_k,
        "feature_dim": ZID_WIDTH,
        "class_count": len(classes),
        "shared_affine_wire": "int8_W[C,160]+fp16_scale[C]+fp16_intercept[C]",
        "deployed_numeric_state_bytes": int(q.nbytes + scale.nbytes + intercept.nbytes),
        "deployed_numeric_state_formula": "160C+2C+2C=164C_B",
        "query_head_macs_per_sample": ZID_WIDTH * len(classes),
        "explicit_dense_matrix_elements_constructed": 0,
        "explicit_spectral_factorization_count": 0,
        "explicit_linear_system_solve_count": 0,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "head_fit_wall_clock_ns": __import__("time").perf_counter_ns() - started,
    }
    return SharedDiagAffineState(classes, active_k, q, scale, intercept, fit_receipt, resource)


def _score_shared(state: SharedDiagAffineState, query: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(_score_shared_float64(state, query), dtype=np.float32)


def _score_shared_float64(state: SharedDiagAffineState, query: np.ndarray) -> np.ndarray:
    rows = _unit_rows(query, name="shared affine query")
    weights, intercepts = state.dequantized()
    logits = rows.astype(np.float64) @ weights.T + intercepts[None, :]
    if not np.isfinite(logits).all():
        raise D92PR160CoreError("shared affine score became non-finite")
    return np.ascontiguousarray(logits, dtype=np.float64)


def _require_unique_top(logits: np.ndarray) -> None:
    values = np.asarray(logits)
    maximum = np.max(values, axis=1, keepdims=True)
    if np.any(np.sum(values == maximum, axis=1) > 1):
        raise D92PR160CoreError("TIE_UNRESOLVED: exact float32 top tie")


def _resolve_float32_precision_alias_ties(
    float64_logits: np.ndarray, float32_logits: np.ndarray
) -> np.ndarray:
    """Resolve only float32 rounding aliases using the same raw score."""

    raw = np.asarray(float64_logits, dtype=np.float64)
    result = np.ascontiguousarray(float32_logits, dtype=np.float32)
    if (
        raw.ndim != 2
        or result.shape != raw.shape
        or not np.isfinite(raw).all()
        or not np.isfinite(result).all()
    ):
        raise D92PR160CoreError("precision-alias tie audit received invalid logits")
    maxima = np.max(result, axis=1, keepdims=True)
    tie_rows = np.flatnonzero(np.sum(result == maxima, axis=1) > 1)
    if len(tie_rows) == 0:
        return result
    resolved = result.copy()
    for row_index in tie_rows:
        top_mask = result[row_index] == maxima[row_index, 0]
        top_indices = np.flatnonzero(top_mask)
        raw_top = raw[row_index, top_mask]
        raw_max = np.max(raw_top)
        raw_winners = top_indices[raw_top == raw_max]
        if len(raw_winners) != 1:
            raise D92PR160CoreError(
                "TIE_UNRESOLVED: exact float32 tie remains tied in float64"
            )
        winner = int(raw_winners[0])
        promoted = np.nextafter(resolved[row_index, winner], np.float32(np.inf))
        if not np.isfinite(promoted):
            raise D92PR160CoreError("precision-alias tie promotion overflow")
        resolved[row_index, winner] = promoted
    return np.ascontiguousarray(resolved, dtype=np.float32)


def build_d92_lite_pair(
    old_support_features160: np.ndarray,
    old_support_labels: Sequence[str],
    old_registered_classes: Sequence[str],
    new_support_features160: np.ndarray,
    new_support_labels: Sequence[str],
    new_registered_classes: Sequence[str],
    *,
    seed: int,
    device: Any,
    d92_fit: Any,
) -> D92PR160Pair:
    del seed, device, d92_fit
    old_classes = _texts(old_registered_classes, "old registered classes")
    new_classes = _texts(new_registered_classes, "new registered classes")
    if len(old_classes) != OLD_CLASS_COUNT or set(old_classes).intersection(new_classes):
        raise D92PR160CoreError("old/new registry partition drift")
    old_labels = tuple(str(item) for item in old_support_labels)
    new_labels = tuple(str(item) for item in new_support_labels)
    if any(label not in old_classes for label in old_labels) or any(
        label not in new_classes for label in new_labels
    ):
        raise D92PR160CoreError("support label registry drift")
    old_rows = _unit_rows(old_support_features160, name="old support PR160")
    new_rows = _unit_rows(new_support_features160, name="new support PR160")
    k_old = _balanced_k(old_labels, old_classes)
    k_new = _balanced_k(new_labels, new_classes)
    if k_old != k_new or len(old_rows) != len(old_labels) or len(new_rows) != len(new_labels):
        raise D92PR160CoreError("old/new support K or row alignment drift")
    lock = phase1_qknn_lock(k_old)
    before_bank = build_typed_zid_support_bank(old_rows, old_labels, old_classes, config=lock)
    before_metric = identity_shared_psd_metric(config=lock)
    registered = old_classes + new_classes
    all_rows = np.ascontiguousarray(np.concatenate([old_rows, new_rows], axis=0))
    all_labels = old_labels + new_labels
    after_bank = build_typed_zid_support_bank(all_rows, all_labels, registered, config=lock)
    after_metric = identity_shared_psd_metric(config=lock)
    lite_state = None if k_old == 1 else _fit_shared_diag(all_rows, all_labels, registered, k_old)
    return D92PR160Pair(
        before_bank=before_bank,
        before_metric=before_metric,
        after_bank=after_bank,
        after_metric=after_metric,
        after_lite_state=lite_state,
        old_registered_classes=old_classes,
        registered_classes=registered,
        active_k=k_old,
        audit={
            "schema": "cvs.phase2.d138.d92_lite_pr160.pair_audit.v1",
            "candidate_id": CANDIDATE_ID,
            "representation": "same_forward_joint_proj_0_signed_prerelu160",
            "head": "shared_all_class_diagonal_affine",
            "active_k": k_old,
            "after_head": "exact_qknn_alias" if k_old == 1 else "shared_all_class_diagonal_affine",
            "old_new_role_access": False,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "query_role_access": False,
            "all_registered_classes_scored": True,
            "qknn_lock_digest": lock.lock_digest,
        },
    )


def score(pair: D92PR160Pair, phase: str, arm: str, query_features160: np.ndarray) -> np.ndarray:
    if type(pair) is not D92PR160Pair or arm != TRANSPORT_ARM:
        raise D92PR160CoreError("invalid D92-Lite-PR160 pair or arm")
    query = _unit_rows(query_features160, name="query PR160")
    if phase == "before":
        raw_logits = score_zid_student_t_logits_float64(
            pair.before_bank, query, metric=pair.before_metric
        )
    elif phase == "after":
        if pair.active_k == 1:
            raw_logits = score_zid_student_t_logits_float64(
                pair.after_bank, query, metric=pair.after_metric
            )
        else:
            assert pair.after_lite_state is not None
            raw_logits = _score_shared_float64(pair.after_lite_state, query)
    else:
        raise D92PR160CoreError("phase must be before or after")
    result = np.ascontiguousarray(np.asarray(raw_logits, dtype=np.float32))
    expected = len(pair.old_registered_classes) if phase == "before" else len(pair.registered_classes)
    if result.shape != (len(query), expected) or not np.isfinite(result).all():
        raise D92PR160CoreError("query logits shape/value drift")
    return _resolve_float32_precision_alias_ties(raw_logits, result)


__all__ = [
    "CANDIDATE_ID",
    "D92PR160CoreError",
    "D92PR160Pair",
    "METHOD_LOCK_SCHEMA",
    "METHOD_LOCK_SHA256",
    "OLD_CLASS_COUNT",
    "PROTOCOL_SCHEMA",
    "REGISTERED_FEATURE_WIDTH",
    "TRANSPORT_ARM",
    "ZID_WIDTH",
    "build_d92_lite_pair",
    "score",
]
