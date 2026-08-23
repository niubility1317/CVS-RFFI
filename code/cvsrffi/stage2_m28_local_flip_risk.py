"""Support-only local conformal flip risk for ERBT-IDR M2.8.

The no-RF32 D92 E0 head is the immutable B0 decision and M2.5 B3 is the
immutable performance branch.  M2.8 estimates whether each individual B3
argmax flip is supported by target-support MGD96 geometry.  It never fuses
scores: every emitted query row is copied exactly from B0 or B3.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_m24_compiler import M24InferenceState
from cvsrffi.stage2_m24_features import physical_if256
from cvsrffi.stage2_m25_anchored_residual import (
    B3,
    M25AnchoredResidualState,
    fit_m25_anchored_residual,
)
from cvsrffi.stage2_m26_spectral_anchor import fft_magnitude_geometry


C1 = "M28-C1-B3-MGD-PAIR-POSTERIOR"
C2 = "M28-C2-B3-MGD-LOCAL-CONFORMAL-RECALL"
M28_LOCAL_RISK_ARMS = (C1, C2)

GLOBAL_PRIOR_ALPHA = 1.0
GLOBAL_PRIOR_BETA = 1.0
DESTINATION_PRIOR_STRENGTH = 4.0
PAIR_PRIOR_STRENGTH = 2.0
POSTERIOR_LOWER_Z = 1.0
MIN_RANK_EVENTS = 4
_EPS = 1.0e-12

_POLICY = {
    C1: {
        0: {
            "posterior_mean": 0.55,
            "posterior_lower": 0.20,
            "conformal_p": 0.20,
            "conformal_source_gap": 0.0,
            "radial_p": 0.05,
            "class_loo_accuracy": 0.40,
        }
    },
    C2: {
        0: {
            "posterior_mean": 0.48,
            "posterior_lower": 0.10,
            "conformal_p": "ONE_OVER_K_PLUS_ONE",
            "conformal_source_gap": 0.0,
            "radial_p": 0.025,
            "class_loo_accuracy": 0.20,
        },
        1: {
            "posterior_mean": 0.62,
            "posterior_lower": 0.22,
            "conformal_p": 0.30,
            "conformal_source_gap": "ONE_OVER_K_PLUS_ONE",
            "radial_p": 0.05,
            "class_loo_accuracy": 0.40,
        },
    },
}


class M28LocalFlipRiskError(ValueError):
    """Raised when a local-risk state or query geometry fails closed."""


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _rows(value: Any, *, name: str, width: int | None = None) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if (
        rows.ndim != 2
        or rows.shape[0] <= 0
        or (width is not None and rows.shape[1] != int(width))
        or not np.isfinite(rows).all()
    ):
        expected = "N x D" if width is None else f"N x {int(width)}"
        raise M28LocalFlipRiskError(f"{name} must be finite {expected}")
    return rows


def _unit_rows(value: Any, *, name: str) -> np.ndarray:
    rows = _rows(value, name=name)
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norm <= _EPS):
        raise M28LocalFlipRiskError(f"{name} contains a degenerate row")
    return rows / norm


def _sorted_rows(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    keys = tuple(rows[:, column] for column in reversed(range(rows.shape[1])))
    return rows[np.lexsort(keys)]


def _canonical_digest(payload: Mapping[str, Any], arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _resolved_policy(arm: str, k_shot: int) -> dict[int, dict[str, float]]:
    if arm not in M28_LOCAL_RISK_ARMS:
        raise M28LocalFlipRiskError("unknown M2.8 local-risk arm")
    one_over_k = 1.0 / (int(k_shot) + 1.0)
    result: dict[int, dict[str, float]] = {}
    for rank, raw in _POLICY[arm].items():
        result[int(rank)] = {
            key: (
                one_over_k
                if value == "ONE_OVER_K_PLUS_ONE"
                else float(value)
            )
            for key, value in raw.items()
        }
    return result


def m28_arm_config_hash(arm: str) -> str:
    if arm not in M28_LOCAL_RISK_ARMS:
        raise M28LocalFlipRiskError("unknown M2.8 local-risk arm")
    payload = {
        "schema": "cvs.erbt_idr.m28.local_flip_risk_config.v1",
        "arm": arm,
        "protocol_schema": "p2_min_v1",
        "base": "P2-A1_NO_RF32_R1",
        "performance_branch": B3,
        "representation": "TARGET_CENTERED_MGD96",
        "selection_policy": "QUERY_LOCAL_EXACT_B0_OR_B3",
        "target_shift_source": "CLASS_BALANCED_OLD_SUPPORT",
        "posterior": {
            "global_alpha": GLOBAL_PRIOR_ALPHA,
            "global_beta": GLOBAL_PRIOR_BETA,
            "destination_prior_strength": DESTINATION_PRIOR_STRENGTH,
            "pair_prior_strength": PAIR_PRIOR_STRENGTH,
            "lower_z": POSTERIOR_LOWER_Z,
            "minimum_rank_events": MIN_RANK_EVENTS,
        },
        "policy": _POLICY[arm],
        "k_lt_5": "EXACT_B0",
        "query_state_update": False,
        "query_truth_access": False,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_support(
    support: Any,
    support_labels: Any,
    *,
    base_support_scores: Any,
    classes: Sequence[str],
    old_class_count: int,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    rows = _rows(support, name="MGD support")
    labels = np.asarray(support_labels).astype(str)
    scores = np.asarray(base_support_scores, dtype=np.float64)
    registry = tuple(str(item) for item in classes)
    old_registry = registry[: int(old_class_count)]
    if (
        labels.shape != (len(rows),)
        or scores.shape != (len(rows), len(registry))
        or not np.isfinite(scores).all()
        or len(registry) < 2
        or len(set(registry)) != len(registry)
        or not 0 < len(old_registry) <= len(registry)
        or int(k_shot) < 1
        or set(labels.tolist()) != set(registry)
        or len(rows) != len(registry) * int(k_shot)
        or any(int(np.sum(labels == name)) != int(k_shot) for name in registry)
    ):
        raise M28LocalFlipRiskError(
            "local-risk support must be finite exact class-symmetric K-shot"
        )
    return rows, labels, scores, registry, old_registry


def _fit_geometry(
    rows: np.ndarray,
    labels: np.ndarray,
    registry: tuple[str, ...],
    old_registry: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    class_rows = [_sorted_rows(rows[labels == name]) for name in registry]
    if any(len(item) == 0 for item in class_rows):
        raise M28LocalFlipRiskError("every registered class requires support")
    robust_centres = np.stack([np.median(item, axis=0) for item in class_rows])
    old_indices = [registry.index(name) for name in old_registry]
    shared = np.mean(robust_centres[old_indices], axis=0)
    transformed = _unit_rows(rows - shared[None, :], name="target-centred MGD support")
    prototypes = _unit_rows(
        np.stack(
            [
                np.mean(_sorted_rows(transformed[labels == name]), axis=0)
                for name in registry
            ]
        ),
        name="target-centred MGD prototypes",
    )
    centered_old = robust_centres[old_indices] - shared[None, :]
    old_mask = np.isin(labels, np.asarray(old_registry))
    radial_location = float(
        np.median(np.linalg.norm(rows[old_mask] - shared[None, :], axis=1))
    )
    return shared, centered_old, transformed, prototypes, radial_location


def _beta_mean_lower(alpha: float, beta: float) -> tuple[float, float]:
    total = float(alpha + beta)
    mean = float(alpha / total)
    variance = float(alpha * beta / (total * total * (total + 1.0)))
    lower = max(0.0, mean - POSTERIOR_LOWER_Z * np.sqrt(variance))
    return mean, float(lower)


def _posterior_tables(
    source: np.ndarray,
    candidates: np.ndarray,
    targets: np.ndarray,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.empty((2, class_count, class_count), dtype=np.float64)
    lowers = np.empty_like(means)
    counts = np.zeros_like(means, dtype=np.int64)
    rank_events = np.zeros(2, dtype=np.int64)
    rank_success = np.zeros(2, dtype=np.int64)
    for rank in range(2):
        candidate = candidates[:, rank]
        event = candidate != source
        success = event & (candidate == targets)
        rank_events[rank] = int(np.sum(event))
        rank_success[rank] = int(np.sum(success))
        global_alpha = GLOBAL_PRIOR_ALPHA + rank_success[rank]
        global_beta = GLOBAL_PRIOR_BETA + rank_events[rank] - rank_success[rank]
        global_mean, _global_lower = _beta_mean_lower(global_alpha, global_beta)
        destination_means = np.empty(class_count, dtype=np.float64)
        for destination in range(class_count):
            destination_event = event & (candidate == destination)
            destination_success = destination_event & (targets == destination)
            alpha = (
                DESTINATION_PRIOR_STRENGTH * global_mean
                + int(np.sum(destination_success))
            )
            beta = (
                DESTINATION_PRIOR_STRENGTH * (1.0 - global_mean)
                + int(np.sum(destination_event))
                - int(np.sum(destination_success))
            )
            destination_means[destination], _unused = _beta_mean_lower(alpha, beta)
        for origin in range(class_count):
            for destination in range(class_count):
                pair_event = event & (source == origin) & (candidate == destination)
                pair_success = pair_event & (targets == destination)
                count = int(np.sum(pair_event))
                success_count = int(np.sum(pair_success))
                alpha = PAIR_PRIOR_STRENGTH * destination_means[destination] + success_count
                beta = (
                    PAIR_PRIOR_STRENGTH * (1.0 - destination_means[destination])
                    + count
                    - success_count
                )
                means[rank, origin, destination], lowers[rank, origin, destination] = (
                    _beta_mean_lower(alpha, beta)
                )
                counts[rank, origin, destination] = count
    return means, lowers, counts, rank_events, rank_success


@dataclass(frozen=True)
class LocalFlipRiskModel:
    classes: tuple[str, ...]
    old_classes: tuple[str, ...]
    k_shot: int
    shared_target_center: np.ndarray
    centered_old_class_centres: np.ndarray
    class_prototypes: np.ndarray
    calibration_nonconformity: np.ndarray
    radial_location: float
    radial_nonconformity: np.ndarray
    pair_posterior_mean: np.ndarray
    pair_posterior_lower: np.ndarray
    pair_event_count: np.ndarray
    rank_event_count: np.ndarray
    class_loo_accuracy: np.ndarray
    fallback_policy: str | None
    state_digest: str

    def __post_init__(self) -> None:
        class_count = len(self.classes)
        shared = np.asarray(self.shared_target_center, dtype=np.float32)
        centered = np.asarray(self.centered_old_class_centres, dtype=np.float32)
        prototypes = np.asarray(self.class_prototypes, dtype=np.float32)
        calibration = np.asarray(self.calibration_nonconformity, dtype=np.float32)
        radial = np.asarray(self.radial_nonconformity, dtype=np.float32)
        means = np.asarray(self.pair_posterior_mean, dtype=np.float32)
        lowers = np.asarray(self.pair_posterior_lower, dtype=np.float32)
        pair_counts = np.asarray(self.pair_event_count, dtype=np.int64)
        rank_counts = np.asarray(self.rank_event_count, dtype=np.int64)
        accuracy = np.asarray(self.class_loo_accuracy, dtype=np.float32)
        if (
            shared.ndim != 1
            or centered.shape != (len(self.old_classes), len(shared))
            or prototypes.shape != (class_count, len(shared))
            or calibration.shape != (class_count, int(self.k_shot))
            or radial.shape != (class_count * int(self.k_shot),)
            or means.shape != (2, class_count, class_count)
            or lowers.shape != means.shape
            or pair_counts.shape != means.shape
            or rank_counts.shape != (2,)
            or accuracy.shape != (class_count,)
            or not all(
                np.isfinite(item).all()
                for item in (shared, centered, prototypes, calibration, radial, means, lowers, accuracy)
            )
            or np.any(means < 0.0)
            or np.any(means > 1.0)
            or np.any(lowers < 0.0)
            or np.any(lowers > means)
            or np.any(accuracy < 0.0)
            or np.any(accuracy > 1.0)
            or len(set(self.classes)) != class_count
            or not set(self.old_classes).issubset(self.classes)
            or int(self.k_shot) < 1
            or len(str(self.state_digest)) != 64
        ):
            raise M28LocalFlipRiskError("local flip-risk state drift")
        for name, value, dtype in (
            ("shared_target_center", shared, np.float32),
            ("centered_old_class_centres", centered, np.float32),
            ("class_prototypes", prototypes, np.float32),
            ("calibration_nonconformity", calibration, np.float32),
            ("radial_nonconformity", radial, np.float32),
            ("pair_posterior_mean", means, np.float32),
            ("pair_posterior_lower", lowers, np.float32),
            ("pair_event_count", pair_counts, np.int64),
            ("rank_event_count", rank_counts, np.int64),
            ("class_loo_accuracy", accuracy, np.float32),
        ):
            object.__setattr__(self, name, _readonly(value, dtype))

    @property
    def feature_dim(self) -> int:
        return int(len(self.shared_target_center))

    @property
    def state_bytes(self) -> int:
        return int(
            sum(
                item.nbytes
                for item in (
                    self.shared_target_center,
                    self.centered_old_class_centres,
                    self.class_prototypes,
                    self.calibration_nonconformity,
                    self.radial_nonconformity,
                    self.pair_posterior_mean,
                    self.pair_posterior_lower,
                    self.pair_event_count,
                    self.rank_event_count,
                    self.class_loo_accuracy,
                )
            )
        )

    def transform(self, value: Any) -> np.ndarray:
        rows = _rows(value, name="MGD query", width=self.feature_dim)
        return _unit_rows(
            rows - self.shared_target_center.astype(np.float64)[None, :],
            name="target-centred MGD query",
        ).astype(np.float32)

    def score(self, value: Any) -> np.ndarray:
        transformed = self.transform(value).astype(np.float64)
        return (transformed @ self.class_prototypes.astype(np.float64).T).astype(
            np.float32
        )

    def conformal_pvalues_from_scores(self, scores: Any) -> np.ndarray:
        rows = np.asarray(scores, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(self.classes) or not np.isfinite(rows).all():
            raise M28LocalFlipRiskError("representation score geometry drift")
        nonconformity = 1.0 - rows
        calibration = self.calibration_nonconformity.astype(np.float64)
        result = np.empty_like(rows)
        for class_index in range(len(self.classes)):
            result[:, class_index] = (
                1.0
                + np.sum(
                    calibration[class_index][None, :]
                    >= nonconformity[:, class_index, None] - 1.0e-12,
                    axis=1,
                )
            ) / (int(self.k_shot) + 1.0)
        return result.astype(np.float32)

    def radial_pvalues(self, value: Any) -> np.ndarray:
        rows = _rows(value, name="MGD query", width=self.feature_dim)
        radius = np.linalg.norm(
            rows - self.shared_target_center.astype(np.float64)[None, :], axis=1
        )
        nonconformity = np.abs(radius - float(self.radial_location))
        calibration = self.radial_nonconformity.astype(np.float64)
        return (
            (1.0 + np.sum(calibration[None, :] >= nonconformity[:, None] - 1.0e-12, axis=1))
            / (len(calibration) + 1.0)
        ).astype(np.float32)

    def query_pair_tables(
        self, base_scores: Any, b3_scores: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        base = np.asarray(base_scores)
        branch = np.asarray(b3_scores)
        if base.ndim != 2 or branch.shape != base.shape or base.shape[1] != len(self.classes):
            raise M28LocalFlipRiskError("B0/B3 score geometry drift")
        source = np.argmax(base, axis=1)
        destination = np.argmax(branch, axis=1)
        rows = np.arange(len(base))
        means = np.stack(
            [self.pair_posterior_mean[rank, source, destination] for rank in range(2)],
            axis=1,
        )
        lowers = np.stack(
            [self.pair_posterior_lower[rank, source, destination] for rank in range(2)],
            axis=1,
        )
        rank_events = np.broadcast_to(self.rank_event_count[None, :], (len(base), 2))
        direct = np.stack(
            [self.pair_event_count[rank, source, destination] for rank in range(2)],
            axis=1,
        )
        if len(rows) != len(means):
            raise M28LocalFlipRiskError("query pair lookup drift")
        return means, lowers, rank_events, direct


def fit_local_flip_risk_model(
    support: Any,
    support_labels: Any,
    *,
    base_support_scores: Any,
    classes: Sequence[str],
    old_class_count: int,
    k_shot: int,
) -> tuple[LocalFlipRiskModel, Mapping[str, Any]]:
    rows, labels, base_scores, registry, old_registry = _validate_support(
        support,
        support_labels,
        base_support_scores=base_support_scores,
        classes=classes,
        old_class_count=int(old_class_count),
        k_shot=int(k_shot),
    )
    shared, centered_old, transformed, prototypes, radial_location = _fit_geometry(
        rows, labels, registry, old_registry
    )
    targets = np.asarray([registry.index(name) for name in labels.tolist()], dtype=np.int64)
    class_count = len(registry)
    fallback_policy: str | None = None
    if int(k_shot) < 5:
        full_scores = transformed @ prototypes.T
        calibration = np.stack(
            [
                np.sort(1.0 - full_scores[labels == name, index])
                for index, name in enumerate(registry)
            ]
        )
        radial = np.sort(
            np.abs(np.linalg.norm(rows - shared[None, :], axis=1) - radial_location)
        )
        class_accuracy = np.zeros(class_count, dtype=np.float64)
        source = np.argmax(base_scores, axis=1)
        candidates = np.tile(source[:, None], (1, 2))
        fallback_policy = "K_LT_5_EXACT_B0"
    else:
        loo_scores = np.empty((len(rows), class_count), dtype=np.float64)
        loo_radial = np.empty(len(rows), dtype=np.float64)
        for held in range(len(rows)):
            keep = np.arange(len(rows)) != held
            folded_shared, _folded_old, _folded_rows, folded_prototypes, folded_radial = _fit_geometry(
                rows[keep], labels[keep], registry, old_registry
            )
            held_transformed = _unit_rows(
                rows[held : held + 1] - folded_shared[None, :],
                name="held-out target-centred MGD support",
            )
            loo_scores[held] = held_transformed[0] @ folded_prototypes.T
            held_radius = float(np.linalg.norm(rows[held] - folded_shared))
            loo_radial[held] = abs(held_radius - folded_radial)
        calibration = np.stack(
            [
                np.sort(1.0 - loo_scores[labels == name, index])
                for index, name in enumerate(registry)
            ]
        )
        radial = np.sort(loo_radial)
        loo_prediction = np.argmax(loo_scores, axis=1)
        class_accuracy = np.asarray(
            [
                np.mean(loo_prediction[labels == name] == index)
                for index, name in enumerate(registry)
            ],
            dtype=np.float64,
        )
        source = np.argmax(base_scores, axis=1)
        candidates = np.argsort(-loo_scores, axis=1, kind="mergesort")[:, :2]
    means, lowers, pair_counts, rank_events, rank_success = _posterior_tables(
        source, candidates, targets, class_count
    )
    state_digest = _canonical_digest(
        {
            "schema": "cvs.erbt_idr.m28.local_flip_risk_state.v1",
            "classes": list(registry),
            "old_classes": list(old_registry),
            "k_shot": int(k_shot),
            "radial_location": float(radial_location),
            "fallback_policy": fallback_policy,
        },
        [
            shared.astype(np.float32),
            centered_old.astype(np.float32),
            prototypes.astype(np.float32),
            calibration.astype(np.float32),
            radial.astype(np.float32),
            means.astype(np.float32),
            lowers.astype(np.float32),
            pair_counts.astype(np.int64),
            rank_events.astype(np.int64),
            class_accuracy.astype(np.float32),
        ],
    )
    audit = {
        "schema": "cvs.erbt_idr.m28.local_flip_risk_fit_audit.v1",
        "support_only": True,
        "query_rows_used": 0,
        "query_state_update": False,
        "target_shift_source": "CLASS_BALANCED_OLD_SUPPORT",
        "class_center_estimator": "COMPONENTWISE_MEDIAN",
        "representation": "TARGET_CENTERED_MGD96",
        "old_class_count": len(old_registry),
        "class_count": class_count,
        "k_shot": int(k_shot),
        "feature_dim": int(rows.shape[1]),
        "fallback_policy": fallback_policy,
        "rank_event_count": rank_events.astype(int).tolist(),
        "rank_success_count": rank_success.astype(int).tolist(),
        "class_loo_accuracy": class_accuracy.astype(float).tolist(),
        "nonzero_pair_count": int(np.sum(pair_counts > 0)),
        "radial_location": float(radial_location),
        "state_digest": state_digest,
    }
    model = LocalFlipRiskModel(
        classes=registry,
        old_classes=old_registry,
        k_shot=int(k_shot),
        shared_target_center=shared.astype(np.float32),
        centered_old_class_centres=centered_old.astype(np.float32),
        class_prototypes=prototypes.astype(np.float32),
        calibration_nonconformity=calibration.astype(np.float32),
        radial_location=float(radial_location),
        radial_nonconformity=radial.astype(np.float32),
        pair_posterior_mean=means.astype(np.float32),
        pair_posterior_lower=lowers.astype(np.float32),
        pair_event_count=pair_counts.astype(np.int64),
        rank_event_count=rank_events.astype(np.int64),
        class_loo_accuracy=class_accuracy.astype(np.float32),
        fallback_policy=fallback_policy,
        state_digest=state_digest,
    )
    return model, MappingProxyType(audit)


def apply_local_flip_policy(
    base_scores: Any,
    b3_scores: Any,
    representation_scores: Any,
    conformal_pvalues: Any,
    radial_pvalues: Any,
    posterior_mean_by_rank: Any,
    posterior_lower_by_rank: Any,
    rank_event_count: Any,
    class_stability: Any,
    *,
    arm: str,
    k_shot: int,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    base = np.asarray(base_scores)
    branch = np.asarray(b3_scores)
    representation = np.asarray(representation_scores, dtype=np.float64)
    conformal = np.asarray(conformal_pvalues, dtype=np.float64)
    radial = np.asarray(radial_pvalues, dtype=np.float64)
    means = np.asarray(posterior_mean_by_rank, dtype=np.float64)
    lowers = np.asarray(posterior_lower_by_rank, dtype=np.float64)
    events = np.asarray(rank_event_count, dtype=np.int64)
    stability = np.asarray(class_stability, dtype=np.float64)
    if (
        arm not in M28_LOCAL_RISK_ARMS
        or base.ndim != 2
        or base.shape[1] < 2
        or branch.shape != base.shape
        or representation.shape != base.shape
        or conformal.shape != base.shape
        or radial.shape != (len(base),)
        or means.shape != (len(base), 2)
        or lowers.shape != means.shape
        or events.shape != means.shape
        or stability.shape != base.shape
        or not all(np.isfinite(item).all() for item in (base, branch, representation, conformal, radial, means, lowers, stability))
        or np.any(conformal < 0.0)
        or np.any(conformal > 1.0)
        or np.any(radial < 0.0)
        or np.any(radial > 1.0)
        or int(k_shot) < 1
    ):
        raise M28LocalFlipRiskError("local flip policy geometry drift")
    base_prediction = np.argmax(base, axis=1)
    branch_prediction = np.argmax(branch, axis=1)
    flip = branch_prediction != base_prediction
    if int(k_shot) < 5:
        return np.array(base, copy=True), MappingProxyType(
            {
                "fallback_reason": "K_LT_5_EXACT_B0",
                "query_count": int(len(base)),
                "b3_flip_count": int(np.sum(flip)),
                "selected_b3_count": 0,
                "accepted_b3_flip_count": 0,
                "accepted_rank1_flip_count": 0,
                "accepted_rank2_flip_count": 0,
                "vetoed_b3_flip_count": int(np.sum(flip)),
                "query_state_update": False,
                "row_source_allowlist": ["B0", "B3"],
            }
        )
    top2 = np.argsort(-representation, axis=1, kind="mergesort")[:, :2]
    accepted = np.zeros(len(base), dtype=bool)
    accepted_rank = np.full(len(base), -1, dtype=np.int64)
    policy = _resolved_policy(arm, int(k_shot))
    rows = np.arange(len(base))
    b3_pair_gain = branch[rows, branch_prediction] - branch[rows, base_prediction]
    for index in np.flatnonzero(flip):
        destination = int(branch_prediction[index])
        source = int(base_prediction[index])
        matches = np.flatnonzero(top2[index] == destination)
        if len(matches) == 0:
            continue
        rank = int(matches[0])
        if rank not in policy:
            continue
        threshold = policy[rank]
        if (
            means[index, rank] >= threshold["posterior_mean"]
            and lowers[index, rank] >= threshold["posterior_lower"]
            and events[index, rank] >= MIN_RANK_EVENTS
            and conformal[index, destination] >= threshold["conformal_p"]
            and conformal[index, destination]
            >= conformal[index, source] + threshold["conformal_source_gap"]
            and radial[index] >= threshold["radial_p"]
            and stability[index, destination] >= threshold["class_loo_accuracy"]
            and b3_pair_gain[index] > 0.0
        ):
            accepted[index] = True
            accepted_rank[index] = rank
    select_branch = (~flip) | accepted
    selected = np.where(select_branch[:, None], branch, base)
    return selected, MappingProxyType(
        {
            "fallback_reason": None,
            "query_count": int(len(base)),
            "b3_flip_count": int(np.sum(flip)),
            "selected_b3_count": int(np.sum(select_branch)),
            "accepted_b3_flip_count": int(np.sum(accepted)),
            "accepted_rank1_flip_count": int(np.sum(accepted_rank == 0)),
            "accepted_rank2_flip_count": int(np.sum(accepted_rank == 1)),
            "vetoed_b3_flip_count": int(np.sum(flip & ~accepted)),
            "candidate_policy": {
                str(rank + 1): dict(threshold) for rank, threshold in policy.items()
            },
            "mean_accepted_posterior": float(np.mean(means[rows[accepted], accepted_rank[accepted]])) if np.any(accepted) else None,
            "mean_accepted_conformal_p": float(np.mean(conformal[rows[accepted], branch_prediction[accepted]])) if np.any(accepted) else None,
            "mean_accepted_radial_p": float(np.mean(radial[accepted])) if np.any(accepted) else None,
            "query_state_update": False,
            "row_source_allowlist": ["B0", "B3"],
        }
    )


@dataclass(frozen=True)
class M28LocalFlipRiskState:
    classes: tuple[str, ...]
    arm: str
    base_state: M24InferenceState
    b3_state: M25AnchoredResidualState
    risk_model: LocalFlipRiskModel
    domain_digest: str
    config_hash: str
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            self.arm not in M28_LOCAL_RISK_ARMS
            or tuple(self.base_state.classes) != tuple(self.classes)
            or tuple(self.b3_state.classes) != tuple(self.classes)
            or tuple(self.risk_model.classes) != tuple(self.classes)
        ):
            raise M28LocalFlipRiskError("M2.8 inference state drift")
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))

    @property
    def feature_dim(self) -> int:
        return int(self.base_state.compiled_affine_state.feature_dim)

    @property
    def state_bytes(self) -> int:
        return int(self.b3_state.state_bytes + self.risk_model.state_bytes)

    def representation_features(self, blocks: Any) -> np.ndarray:
        raw = _rows(blocks, name="M2.8 IF blocks")
        if raw.shape[1] < 256:
            raise M28LocalFlipRiskError("M2.8 IF blocks require FFT96")
        return fft_magnitude_geometry(raw[:, 160:256])

    def score_with_audit(self, blocks: Any) -> tuple[np.ndarray, Mapping[str, Any]]:
        physical = physical_if256(blocks)
        base_scores = self.base_state.score(physical)
        b3_scores = self.b3_state.score(blocks)
        representation = self.representation_features(blocks)
        representation_scores = self.risk_model.score(representation)
        conformal = self.risk_model.conformal_pvalues_from_scores(representation_scores)
        radial = self.risk_model.radial_pvalues(representation)
        means, lowers, events, direct = self.risk_model.query_pair_tables(base_scores, b3_scores)
        stability = np.broadcast_to(
            self.risk_model.class_loo_accuracy[None, :], base_scores.shape
        )
        selected, application = apply_local_flip_policy(
            base_scores,
            b3_scores,
            representation_scores,
            conformal,
            radial,
            means,
            lowers,
            events,
            stability,
            arm=self.arm,
            k_shot=self.risk_model.k_shot,
        )
        return selected, MappingProxyType(
            {
                **dict(application),
                "mean_pair_direct_event_count": float(np.mean(direct)),
                "rank_event_count": self.risk_model.rank_event_count.astype(int).tolist(),
            }
        )

    def score(self, blocks: Any) -> np.ndarray:
        scores, _audit = self.score_with_audit(blocks)
        return scores

    def predict(self, blocks: Any) -> np.ndarray:
        return np.asarray(self.classes)[np.argmax(self.score(blocks), axis=1)]


def fit_m28_local_flip_risk(
    *,
    arm: str,
    base_state: M24InferenceState,
    support_blocks: Any,
    support_labels: Any,
    classes: Sequence[str],
    k_shot: int,
    old_class_count: int,
    domain_digest: str,
) -> tuple[M28LocalFlipRiskState, Mapping[str, Any]]:
    if arm not in M28_LOCAL_RISK_ARMS:
        raise M28LocalFlipRiskError("unknown M2.8 local-risk arm")
    blocks = _rows(support_blocks, name="M2.8 support blocks")
    labels = np.asarray(support_labels).astype(str)
    registry = tuple(str(item) for item in classes)
    if blocks.shape[1] < 256 or tuple(base_state.classes) != registry:
        raise M28LocalFlipRiskError("M2.8 support/base registry drift")
    b3_state, b3_audit = fit_m25_anchored_residual(
        arm=B3,
        base_state=base_state,
        support_blocks=blocks,
        support_labels=labels,
        classes=registry,
        k_shot=int(k_shot),
        old_class_count=int(old_class_count),
        domain_digest=str(domain_digest),
    )
    representation = fft_magnitude_geometry(blocks[:, 160:256])
    base_support_scores = base_state.score(physical_if256(blocks))
    risk_model, risk_audit = fit_local_flip_risk_model(
        representation,
        labels,
        base_support_scores=base_support_scores,
        classes=registry,
        old_class_count=int(old_class_count),
        k_shot=int(k_shot),
    )
    quantization = dict(base_state.audit.get("quantization", {}))
    b3_resource = dict(b3_audit["resource"])
    resource = {
        "compiled_inference_state_bytes": int(
            b3_resource["compiled_inference_state_bytes"] + risk_model.state_bytes
        ),
        "persistent_update_state_bytes": 0,
        "transient_registration_workspace_peak_bytes": int(
            max(
                b3_resource["transient_registration_workspace_peak_bytes"],
                representation.nbytes
                + base_support_scores.nbytes
                + risk_model.pair_posterior_mean.nbytes
                + risk_model.pair_posterior_lower.nbytes,
            )
        ),
    }
    audit = {
        "schema": "cvs.erbt_idr.m28.local_flip_risk_fit_audit.v1",
        "arm": arm,
        "k_shot": int(k_shot),
        "feature_dim": int(base_state.compiled_affine_state.feature_dim),
        "support_only": True,
        "query_rows_used": 0,
        "query_state_update": False,
        "base_method": "P2-A1_NO_RF32_R1",
        "performance_branch": B3,
        "representation": "TARGET_CENTERED_MGD96",
        "selection_policy": "QUERY_LOCAL_EXACT_B0_OR_B3",
        "candidate_policy": {
            str(rank + 1): dict(value)
            for rank, value in _resolved_policy(arm, int(k_shot)).items()
        },
        "b3": dict(b3_audit),
        "risk_fit": dict(risk_audit),
        "quantization": quantization,
        "resource": resource,
    }
    state = M28LocalFlipRiskState(
        classes=registry,
        arm=arm,
        base_state=base_state,
        b3_state=b3_state,
        risk_model=risk_model,
        domain_digest=str(domain_digest),
        config_hash=m28_arm_config_hash(arm),
        audit=audit,
    )
    return state, MappingProxyType(audit)


__all__ = [
    "C1",
    "C2",
    "M28_LOCAL_RISK_ARMS",
    "M28LocalFlipRiskError",
    "M28LocalFlipRiskState",
    "LocalFlipRiskModel",
    "apply_local_flip_policy",
    "fit_local_flip_risk_model",
    "fit_m28_local_flip_risk",
    "m28_arm_config_hash",
]
