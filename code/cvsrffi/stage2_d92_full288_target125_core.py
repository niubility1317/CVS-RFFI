"""Support-only D92-Lite full-288 Target125 core.

The PR160 route failed on exact score ties after discarding the auxiliary
128-dimensional part of the sealed ``registered_feature``.  This candidate
keeps the complete same-runtime 288-dimensional view.  K1 uses support-only
class-centroid cosine; K5/K10 use one all-class shared diagonal OAS affine
head.  A genuine final-score tie is resolved only by the same support-only
full-288 class centroid and then by a canonical, row-order-independent
support fingerprint.  If two classes have identical support evidence, the
core still fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


METHOD_LOCK_SHA256 = "2bc4384f0a94f3be670a27738ee727db47d937332653bcf3f5ac2a06e02ba728"
METHOD_LOCK_SCHEMA = "cvs.phase2.d138.d92_lite_full288.method_lock.v1"
CANDIDATE_ID = "D92-Lite-FULL288/r1"
PROTOCOL_SCHEMA = "p2_min_v1"
TRANSPORT_ARM = "M_JOINT"
OLD_CLASS_COUNT = 6
FEATURE_WIDTH = 288

SOURCE_RUNTIME_SHA256 = (
    "f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a"
)


class D92Full288CoreError(ValueError):
    """Raised when the full-288 D92-Lite core fails closed."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _unit_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != FEATURE_WIDTH
        or rows.shape[0] < 1
        or not np.isfinite(rows).all()
    ):
        raise D92Full288CoreError(f"{name} must be finite float32 [N,288]")
    norms = np.sqrt(np.sum(rows.astype(np.float64) ** 2, axis=1))
    if not np.isfinite(norms).all() or not np.allclose(
        norms, 1.0, atol=2.0e-5, rtol=0.0
    ):
        raise D92Full288CoreError(f"{name} must retain D92 unit-row normalization")
    return np.ascontiguousarray(rows, dtype=np.float32)


def _texts(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise D92Full288CoreError(f"{name} must be a string sequence")
    result = tuple(str(item) for item in value)
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        raise D92Full288CoreError(f"{name} must contain unique nonempty values")
    return result


def _balanced_k(labels: tuple[str, ...], classes: tuple[str, ...]) -> int:
    counts = tuple(labels.count(item) for item in classes)
    if any(count < 1 for count in counts) or len(set(counts)) != 1:
        raise D92Full288CoreError("support must be balanced over every class")
    if counts[0] not in (1, 5, 10):
        raise D92Full288CoreError("Target125 only permits K1/K5/K10")
    return counts[0]


@dataclass(frozen=True, slots=True)
class Full288AffineState:
    classes: tuple[str, ...]
    active_k: int
    weight_float64: np.ndarray
    intercept_float64: np.ndarray
    fit_receipt: Mapping[str, Any]
    resource_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        classes = tuple(str(item) for item in self.classes)
        weights = np.asarray(self.weight_float64)
        intercepts = np.asarray(self.intercept_float64)
        if (
            len(classes) < 2
            or len(set(classes)) != len(classes)
            or int(self.active_k) not in (5, 10)
            or weights.dtype != np.float64
            or weights.shape != (len(classes), FEATURE_WIDTH)
            or intercepts.dtype != np.float64
            or intercepts.shape != (len(classes),)
            or not np.isfinite(weights).all()
            or not np.isfinite(intercepts).all()
        ):
            raise D92Full288CoreError("full-288 affine state drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "active_k", int(self.active_k))
        object.__setattr__(self, "weight_float64", _readonly(weights, np.float64))
        object.__setattr__(
            self, "intercept_float64", _readonly(intercepts, np.float64)
        )


@dataclass(frozen=True, slots=True)
class D92Full288Pair:
    before_state: Full288AffineState | None
    after_state: Full288AffineState | None
    before_support_features288: np.ndarray
    after_support_features288: np.ndarray
    before_class_indices: np.ndarray
    after_class_indices: np.ndarray
    old_registered_classes: tuple[str, ...]
    registered_classes: tuple[str, ...]
    active_k: int
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        before = np.asarray(self.before_support_features288)
        after = np.asarray(self.after_support_features288)
        before_indices = np.asarray(self.before_class_indices)
        after_indices = np.asarray(self.after_class_indices)
        if (
            before.dtype != np.float32
            or after.dtype != np.float32
            or before.ndim != 2
            or after.ndim != 2
            or before.shape[1] != FEATURE_WIDTH
            or after.shape[1] != FEATURE_WIDTH
            or len(before_indices) != len(before)
            or len(after_indices) != len(after)
            or before_indices.dtype.kind not in "iu"
            or after_indices.dtype.kind not in "iu"
            or not np.isfinite(before).all()
            or not np.isfinite(after).all()
        ):
            raise D92Full288CoreError("full-288 support state drift")
        object.__setattr__(
            self, "before_support_features288", _readonly(before, np.float32)
        )
        object.__setattr__(
            self, "after_support_features288", _readonly(after, np.float32)
        )
        object.__setattr__(
            self, "before_class_indices", _readonly(before_indices, np.int16)
        )
        object.__setattr__(
            self, "after_class_indices", _readonly(after_indices, np.int16)
        )
        if self.active_k not in (1, 5, 10):
            raise D92Full288CoreError("active K drift")
        if self.active_k == 1:
            if self.before_state is not None or self.after_state is not None:
                raise D92Full288CoreError("K1 must use the exact centroid path")
        elif (
            type(self.before_state) is not Full288AffineState
            or type(self.after_state) is not Full288AffineState
            or self.before_state.active_k != self.active_k
            or self.after_state.active_k != self.active_k
        ):
            raise D92Full288CoreError("K5/K10 requires two full-288 affine states")
        if (
            self.old_registered_classes
            != tuple(self.old_registered_classes)
            or self.registered_classes[:OLD_CLASS_COUNT]
            != self.old_registered_classes
        ):
            raise D92Full288CoreError("pair registry drift")


def _fit_shared_diag(
    rows: np.ndarray,
    labels: tuple[str, ...],
    classes: tuple[str, ...],
    active_k: int,
) -> Full288AffineState:
    data = rows.astype(np.float64)
    class_indices = np.asarray([classes.index(label) for label in labels], dtype=np.int64)
    means = np.stack(
        [data[class_indices == index].mean(axis=0) for index in range(len(classes))]
    )
    residuals = data - means[class_indices]
    degrees = len(data) - len(classes)
    if degrees <= 0:
        raise D92Full288CoreError("full-288 variance is not identifiable")
    scatter = np.sum(residuals * residuals, axis=0, dtype=np.float64) / float(degrees)
    total = float(np.sum(scatter, dtype=np.float64))
    second = float(np.sum(scatter * scatter, dtype=np.float64))
    tau = total / float(FEATURE_WIDTH)
    delta = second - total * total / float(FEATURE_WIDTH)
    if not all(np.isfinite(value) for value in (total, second, tau, delta)) or total <= 0.0:
        raise D92Full288CoreError("full-288 variance became non-finite")
    if delta <= 0.0:
        shrinkage = 1.0
    else:
        numerator = (1.0 - 2.0 / FEATURE_WIDTH) * second + total * total
        denominator = (
            float(degrees) + 1.0 - 2.0 / FEATURE_WIDTH
        ) * delta
        shrinkage = min(1.0, numerator / denominator)
    floor = max(
        float(np.finfo(np.float64).tiny),
        float(np.finfo(np.float64).eps) * max(1.0, tau),
    )
    variance = np.maximum((1.0 - shrinkage) * scatter + shrinkage * tau, floor)
    weights = means / variance[None, :]
    intercepts = -0.5 * np.sum(means * weights, axis=1) - np.log(len(classes))
    weights -= weights.mean(axis=0, keepdims=True)
    intercepts -= intercepts.mean()
    if not np.isfinite(weights).all() or not np.isfinite(intercepts).all():
        raise D92Full288CoreError("full-288 affine state became non-finite")
    fit_receipt = {
        "schema": "cvs.phase2.d138.d92_lite_full288.fit.v1",
        "head": "full288_shared_diagonal_oas_float64",
        "fit_mode": "all_class_shared_diagonal_oas",
        "feature_dim": FEATURE_WIDTH,
        "class_count": len(classes),
        "active_k": active_k,
        "support_rows": len(rows),
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_role_access": False,
        "diagonal_shrinkage": float(shrinkage),
        "variance_floor": float(floor),
        "variance_trace": float(np.sum(variance, dtype=np.float64)),
    }
    resource = {
        "schema": "cvs.phase2.d138.d92_lite_full288.resource.v1",
        "feature_dim": FEATURE_WIDTH,
        "class_count": len(classes),
        "active_k": active_k,
        "deployed_numeric_state_bytes": int(weights.nbytes + intercepts.nbytes),
        "query_head_macs_per_sample": FEATURE_WIDTH * len(classes),
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "support_only": True,
    }
    return Full288AffineState(classes, active_k, weights, intercepts, fit_receipt, resource)


def _class_centroids(
    rows: np.ndarray, class_indices: np.ndarray, class_count: int
) -> np.ndarray:
    columns = []
    for class_index in range(class_count):
        local = rows[class_indices == class_index].astype(np.float64)
        if len(local) < 1:
            raise D92Full288CoreError("support class coverage drift")
        centroid = np.mean(local, axis=0)
        norm = float(np.linalg.norm(centroid))
        if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
            raise D92Full288CoreError("support centroid is non-finite or zero")
        columns.append(centroid / norm)
    return np.ascontiguousarray(np.stack(columns), dtype=np.float64)


def _class_fingerprints(
    rows: np.ndarray, class_indices: np.ndarray, class_count: int
) -> tuple[tuple[float, ...], ...]:
    fingerprints: list[tuple[float, ...]] = []
    for class_index in range(class_count):
        local = np.asarray(rows[class_indices == class_index], dtype=np.float32)
        if len(local) < 1:
            raise D92Full288CoreError("support fingerprint class coverage drift")
        # Canonicalize support-row order by feature values, never by registry order.
        ordered = local[np.lexsort(local[:, ::-1].T)]
        fingerprints.append(tuple(float(value) for value in ordered.astype(np.float64).ravel()))
    return tuple(fingerprints)


def _centroid_scores(query: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    result = query.astype(np.float64) @ centroids.T
    if not np.isfinite(result).all():
        raise D92Full288CoreError("full-288 centroid score became non-finite")
    return np.ascontiguousarray(result, dtype=np.float64)


def _affine_scores(state: Full288AffineState, query: np.ndarray) -> np.ndarray:
    result = query.astype(np.float64) @ state.weight_float64.T
    result += state.intercept_float64[None, :]
    if not np.isfinite(result).all():
        raise D92Full288CoreError("full-288 affine score became non-finite")
    return np.ascontiguousarray(result, dtype=np.float64)


def _resolve_ties(
    raw_logits: np.ndarray,
    secondary_logits: np.ndarray,
    fingerprints: tuple[tuple[float, ...], ...],
) -> np.ndarray:
    raw = np.asarray(raw_logits, dtype=np.float64)
    result = np.ascontiguousarray(raw, dtype=np.float32)
    secondary = np.asarray(secondary_logits, dtype=np.float64)
    if (
        raw.ndim != 2
        or secondary.shape != raw.shape
        or not np.isfinite(raw).all()
        or not np.isfinite(result).all()
        or not np.isfinite(secondary).all()
        or len(fingerprints) != raw.shape[1]
    ):
        raise D92Full288CoreError("full-288 tie resolver state drift")
    maxima = np.max(result, axis=1, keepdims=True)
    for row_index in np.flatnonzero(np.sum(result == maxima, axis=1) > 1):
        top_mask = result[row_index] == maxima[row_index, 0]
        top_indices = np.flatnonzero(top_mask)
        raw_top = raw[row_index, top_mask]
        raw_winners = top_indices[raw_top == np.max(raw_top)]
        if len(raw_winners) == 1:
            winner = int(raw_winners[0])
        else:
            secondary_top = secondary[row_index, top_mask]
            secondary_winners = top_indices[secondary_top == np.max(secondary_top)]
            if len(secondary_winners) == 1:
                winner = int(secondary_winners[0])
            else:
                fingerprint_top = [fingerprints[int(index)] for index in secondary_winners]
                max_fingerprint = max(fingerprint_top)
                fingerprint_winners = [
                    int(index)
                    for index, fingerprint in zip(
                        secondary_winners, fingerprint_top, strict=True
                    )
                    if fingerprint == max_fingerprint
                ]
                if len(fingerprint_winners) != 1:
                    raise D92Full288CoreError(
                        "TIE_UNRESOLVED: identical full-288 support evidence"
                    )
                winner = fingerprint_winners[0]
        promoted = np.nextafter(result[row_index, winner], np.float32(np.inf))
        if not np.isfinite(promoted):
            raise D92Full288CoreError("full-288 tie promotion overflow")
        result[row_index, winner] = promoted
    return np.ascontiguousarray(result, dtype=np.float32)


def build_d92_full288_pair(
    old_support_features288: np.ndarray,
    old_support_labels: Sequence[str],
    old_registered_classes: Sequence[str],
    new_support_features288: np.ndarray,
    new_support_labels: Sequence[str],
    new_registered_classes: Sequence[str],
    *,
    seed: int,
    device: Any,
    d92_fit: Any,
) -> D92Full288Pair:
    del seed, device, d92_fit
    old_classes = _texts(old_registered_classes, "old registered classes")
    new_classes = _texts(new_registered_classes, "new registered classes")
    if len(old_classes) != OLD_CLASS_COUNT or set(old_classes).intersection(new_classes):
        raise D92Full288CoreError("old/new registry partition drift")
    old_labels = tuple(str(item) for item in old_support_labels)
    new_labels = tuple(str(item) for item in new_support_labels)
    if any(label not in old_classes for label in old_labels) or any(
        label not in new_classes for label in new_labels
    ):
        raise D92Full288CoreError("support label registry drift")
    old_rows = _unit_rows(old_support_features288, name="old support full288")
    new_rows = _unit_rows(new_support_features288, name="new support full288")
    k_old = _balanced_k(old_labels, old_classes)
    k_new = _balanced_k(new_labels, new_classes)
    if k_old != k_new or len(old_rows) != len(old_labels) or len(new_rows) != len(new_labels):
        raise D92Full288CoreError("old/new support K or row alignment drift")
    registered = old_classes + new_classes
    all_rows = np.ascontiguousarray(np.concatenate([old_rows, new_rows], axis=0))
    all_labels = old_labels + new_labels
    before_indices = np.asarray([old_classes.index(label) for label in old_labels], dtype=np.int16)
    after_indices = np.asarray([registered.index(label) for label in all_labels], dtype=np.int16)
    before_state = None if k_old == 1 else _fit_shared_diag(old_rows, old_labels, old_classes, k_old)
    after_state = None if k_old == 1 else _fit_shared_diag(all_rows, all_labels, registered, k_old)
    return D92Full288Pair(
        before_state=before_state,
        after_state=after_state,
        before_support_features288=old_rows,
        after_support_features288=all_rows,
        before_class_indices=before_indices,
        after_class_indices=after_indices,
        old_registered_classes=old_classes,
        registered_classes=registered,
        active_k=k_old,
        audit={
            "schema": "cvs.phase2.d138.d92_lite_full288.pair_audit.v1",
            "candidate_id": CANDIDATE_ID,
            "representation": "sealed_D92_registered_feature_288",
            "head": "full288_shared_diagonal_oas_float64_or_centroid_cosine_k1",
            "active_k": k_old,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "query_role_access": False,
            "all_registered_classes_scored": True,
            "tie_policy": (
                "float64_unique_then_full288_support_centroid_then_"
                "canonical_sorted_full288_support_fingerprint_else_fail_closed"
            ),
        },
    )


def score(
    pair: D92Full288Pair, phase: str, arm: str, query_features288: np.ndarray
) -> np.ndarray:
    if type(pair) is not D92Full288Pair or arm != TRANSPORT_ARM:
        raise D92Full288CoreError("invalid full-288 pair or arm")
    query = _unit_rows(query_features288, name="query full288")
    if phase == "before":
        state = pair.before_state
        support = pair.before_support_features288
        indices = pair.before_class_indices
        classes = pair.old_registered_classes
    elif phase == "after":
        state = pair.after_state
        support = pair.after_support_features288
        indices = pair.after_class_indices
        classes = pair.registered_classes
    else:
        raise D92Full288CoreError("phase must be before or after")
    class_count = len(classes)
    centroids = _class_centroids(support, indices, class_count)
    raw_logits = _centroid_scores(query, centroids) if state is None else _affine_scores(state, query)
    secondary = _centroid_scores(query, centroids)
    fingerprints = _class_fingerprints(support, indices, class_count)
    result = _resolve_ties(raw_logits, secondary, fingerprints)
    expected = class_count
    if result.shape != (len(query), expected) or not np.isfinite(result).all():
        raise D92Full288CoreError("full-288 query logits shape/value drift")
    return result


__all__ = [
    "CANDIDATE_ID",
    "D92Full288CoreError",
    "D92Full288Pair",
    "FEATURE_WIDTH",
    "METHOD_LOCK_SCHEMA",
    "METHOD_LOCK_SHA256",
    "OLD_CLASS_COUNT",
    "PROTOCOL_SCHEMA",
    "SOURCE_RUNTIME_SHA256",
    "TRANSPORT_ARM",
    "build_d92_full288_pair",
    "score",
]
