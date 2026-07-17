"""D33 support-only spherical registration in the locked 288-D B3 space.

Old and new registered classes use exactly the same frozen diagonal transform,
spherical centroid, robust radius, and score ``-d/r - log(r)``.  Radius
hyperparameters are selected from a method-locked 36-point grid using only
class-symmetric support leave-one-support-out (LOSO) evidence.  No fitting API
accepts query rows, roles, quotas, or batch-level assignment information.

K=1 cannot estimate a physical within-class radius.  It therefore uses one
uniform unit radius for every registered class, making the score exactly a
constant-shifted cosine score and avoiding pseudo-radius fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


FEATURE_DIM = 288
MAX_OLD_CLASSES = 6
ALLOWED_NEW_CLASS_COUNTS = (2, 5, 10, 20)
RADIUS_QUANTILES = (0.5, 0.75, 0.9)
RADIUS_SHRINKAGES = (0.25, 0.5, 0.75, 1.0)
RADIUS_RATIO_CAPS = (1.15, 1.25, 1.5)
SELECTION_POLICIES = ("A_overall_first", "B_balanced", "C_floor_first")
MIN_RADIUS = np.float32(1.0e-4)
MAX_ACTIVE_PARAMETERS = 50_000
SCHEMA = "cvs.phase2.d33_spherical_registration.v1"


class D33SphericalRegistrationError(ValueError):
    """Raised when D33 support, state, or method lock drifts."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True).astype(np.float32)
    if bool(np.any(norms <= np.float32(1.0e-12))):
        raise D33SphericalRegistrationError("zero-norm spherical row or centroid")
    return np.asarray(values / norms, dtype=np.float32)


def _validate_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D33SphericalRegistrationError(
            f"{name} must be finite [N,{FEATURE_DIM}]"
        )
    return np.ascontiguousarray(rows, dtype=np.float32)


def _validate_labeled_support(
    features: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    rows = _validate_rows(features, name=f"{name} features")
    label_values = np.asarray(tuple(str(value) for value in labels))
    registry = tuple(str(value) for value in classes)
    if (
        label_values.ndim != 1
        or len(label_values) != len(rows)
        or not registry
        or len(set(registry)) != len(registry)
        or any(not value for value in registry)
        or set(label_values.tolist()) != set(registry)
    ):
        raise D33SphericalRegistrationError(f"{name} class registry drift")
    counts = [int(np.sum(label_values == value)) for value in registry]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise D33SphericalRegistrationError(
            f"{name} must be class-symmetric K-shot"
        )
    return rows, label_values, registry, counts[0]


def _targets(labels: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    mapping = {value: index for index, value in enumerate(classes)}
    return np.asarray(
        [mapping[str(value)] for value in labels.tolist()], dtype=np.int64
    )


def _angular_distance(rows: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    similarities = np.asarray(rows @ centroids.T, dtype=np.float32)
    return np.asarray(
        np.clip(np.float32(1.0) - similarities, 0.0, 2.0), dtype=np.float32
    )


def _quantize_centroids(
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-class int8 quantization with no FP32 resident copy."""

    values = _normalize(np.asarray(centroids, dtype=np.float32))
    peak = np.max(np.abs(values), axis=1).astype(np.float32)
    if bool(np.any(peak <= np.float32(1.0e-12))):
        raise D33SphericalRegistrationError("zero-range centroid quantization")
    scales = np.asarray(peak / np.float32(127.0), dtype=np.float32)
    quantized = np.asarray(
        np.clip(
            np.rint(values / scales[:, None]),
            -127,
            127,
        ),
        dtype=np.int8,
    )
    return quantized, scales


def _dequantized_unit_centroids(
    quantized: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """Construct the temporary deployed centroid surface."""

    restored = np.asarray(quantized, dtype=np.float32) * np.asarray(
        scales, dtype=np.float32
    )[:, None]
    return _normalize(restored)


def _radius_scores(distances: np.ndarray, radii: np.ndarray) -> np.ndarray:
    safe = np.maximum(np.asarray(radii, dtype=np.float32), MIN_RADIUS)
    return np.asarray(
        -distances / safe[None, :] - np.log(safe)[None, :], dtype=np.float32
    )


def _centroids(rows: np.ndarray, targets: np.ndarray, class_count: int) -> np.ndarray:
    return np.stack(
        [
            _normalize(
                np.mean(rows[targets == index], axis=0, keepdims=True, dtype=np.float32)
            )[0]
            for index in range(class_count)
        ]
    ).astype(np.float32)


def _loso_geometry(
    rows: np.ndarray, targets: np.ndarray, class_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row LOSO distances and own-class LOSO distances."""

    sums = np.stack(
        [np.sum(rows[targets == index], axis=0) for index in range(class_count)]
    ).astype(np.float32)
    counts = np.bincount(targets, minlength=class_count).astype(np.int64)
    distances = np.empty((len(rows), class_count), dtype=np.float32)
    for row_index in range(len(rows)):
        held_class = int(targets[row_index])
        fold_sums = sums.copy()
        fold_counts = counts.copy()
        fold_sums[held_class] -= rows[row_index]
        fold_counts[held_class] -= 1
        fold_centroids_fp32 = _normalize(
            fold_sums / fold_counts[:, None].astype(np.float32)
        )
        fold_qint8, fold_scales = _quantize_centroids(fold_centroids_fp32)
        fold_centroids = _dequantized_unit_centroids(fold_qint8, fold_scales)
        distances[row_index] = _angular_distance(
            rows[row_index : row_index + 1], fold_centroids
        )[0]
    own = distances[np.arange(len(rows)), targets]
    return distances, np.asarray(own, dtype=np.float32)


def _robust_radii(
    own_loso_distances: np.ndarray,
    targets: np.ndarray,
    class_count: int,
    *,
    quantile: float,
    shrinkage: float,
    ratio_cap: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    raw = np.asarray(
        [
            np.quantile(
                own_loso_distances[targets == index], quantile, method="linear"
            )
            for index in range(class_count)
        ],
        dtype=np.float32,
    )
    raw = np.maximum(raw, MIN_RADIUS)
    global_median = float(max(float(np.median(raw)), float(MIN_RADIUS)))
    shrunk = (
        np.float32(shrinkage) * raw
        + np.float32(1.0 - shrinkage) * np.float32(global_median)
    )
    lower = np.float32(global_median / ratio_cap)
    upper = np.float32(global_median * ratio_cap)
    clipped = np.asarray(np.clip(shrunk, lower, upper), dtype=np.float32)
    return raw, clipped, global_median


def _selection_rank(
    policy: str,
    *,
    overall: float,
    floor: float,
    min_class_margin: float,
    mean_margin: float,
    quantile: float,
    shrinkage: float,
    ratio_cap: float,
) -> tuple[float, ...]:
    balanced = 0.0 if overall + floor <= 0.0 else 2.0 * overall * floor / (
        overall + floor
    )
    tie = (
        min_class_margin,
        mean_margin,
        -abs(quantile - 0.75),
        -abs(shrinkage - 0.5),
        -ratio_cap,
    )
    if policy == "A_overall_first":
        return (overall, floor, *tie)
    if policy == "B_balanced":
        return (balanced, floor, overall, *tie)
    if policy == "C_floor_first":
        return (floor, overall, *tie)
    raise D33SphericalRegistrationError("unknown D33 selection policy")


@dataclass(frozen=True)
class D33SphericalRegistrationConfig:
    selection_policy: str = "B_balanced"
    radius_quantiles: tuple[float, ...] = RADIUS_QUANTILES
    radius_shrinkages: tuple[float, ...] = RADIUS_SHRINKAGES
    radius_ratio_caps: tuple[float, ...] = RADIUS_RATIO_CAPS

    def __post_init__(self) -> None:
        object.__setattr__(self, "selection_policy", str(self.selection_policy))
        object.__setattr__(
            self, "radius_quantiles", tuple(float(v) for v in self.radius_quantiles)
        )
        object.__setattr__(
            self, "radius_shrinkages", tuple(float(v) for v in self.radius_shrinkages)
        )
        object.__setattr__(
            self, "radius_ratio_caps", tuple(float(v) for v in self.radius_ratio_caps)
        )
        if (
            self.selection_policy not in SELECTION_POLICIES
            or self.radius_quantiles != RADIUS_QUANTILES
            or self.radius_shrinkages != RADIUS_SHRINKAGES
            or self.radius_ratio_caps != RADIUS_RATIO_CAPS
        ):
            raise D33SphericalRegistrationError("D33 fixed radius grid drift")


@dataclass(frozen=True)
class D33SphericalRegistrationState:
    schema: str
    classes: tuple[str, ...]
    old_class_count: int
    log_diag: np.ndarray
    centroids_qint8: np.ndarray
    centroid_scales: np.ndarray
    radii: np.ndarray
    support_count_by_class: np.ndarray
    selected_quantile: float
    selected_shrinkage: float
    selected_ratio_cap: float
    selection_policy: str
    optimizer_steps: int

    def __post_init__(self) -> None:
        class_count = len(self.classes)
        new_count = class_count - int(self.old_class_count)
        log_diag = np.asarray(self.log_diag)
        centroids_qint8 = np.asarray(self.centroids_qint8)
        centroid_scales = np.asarray(self.centroid_scales)
        radii = np.asarray(self.radii)
        counts = np.asarray(self.support_count_by_class)
        if (
            self.schema != SCHEMA
            or not 2 <= int(self.old_class_count) <= MAX_OLD_CLASSES
            or new_count not in ALLOWED_NEW_CLASS_COUNTS
            or len(set(self.classes)) != class_count
            or log_diag.dtype != np.float32
            or log_diag.shape != (FEATURE_DIM,)
            or centroids_qint8.dtype != np.int8
            or centroids_qint8.shape != (class_count, FEATURE_DIM)
            or centroid_scales.dtype != np.float32
            or centroid_scales.shape != (class_count,)
            or radii.dtype != np.float32
            or radii.shape != (class_count,)
            or counts.dtype != np.uint16
            or counts.shape != (class_count,)
            or len(set(int(value) for value in counts.tolist())) != 1
            or bool(np.any(radii < MIN_RADIUS))
            or not np.isfinite(log_diag).all()
            or not np.isfinite(centroid_scales).all()
            or bool(np.any(centroid_scales <= 0.0))
            or not np.isfinite(radii).all()
            or self.selection_policy not in SELECTION_POLICIES
            or int(self.optimizer_steps) != 0
            or self.active_parameters >= MAX_ACTIVE_PARAMETERS
        ):
            raise D33SphericalRegistrationError("D33 spherical state drift")
        object.__setattr__(self, "log_diag", _readonly(log_diag, np.float32))
        object.__setattr__(
            self, "centroids_qint8", _readonly(centroids_qint8, np.int8)
        )
        object.__setattr__(
            self, "centroid_scales", _readonly(centroid_scales, np.float32)
        )
        object.__setattr__(self, "radii", _readonly(radii, np.float32))
        object.__setattr__(
            self, "support_count_by_class", _readonly(counts, np.uint16)
        )

    @property
    def active_parameters(self) -> int:
        return int(
            self.log_diag.size
            + self.centroids_qint8.size
            + self.centroid_scales.size
            + self.radii.size
        )

    def dequantized_centroids(self) -> np.ndarray:
        """Return temporary FP32 unit centroids used by the deployed scorer."""

        return _dequantized_unit_centroids(
            self.centroids_qint8, self.centroid_scales
        )


@dataclass(frozen=True)
class D33SphericalRegistrationResult:
    state: D33SphericalRegistrationState
    selection_trace: tuple[dict[str, Any], ...]
    resource_audit: dict[str, Any]


def fit_d33_spherical_registration(
    old_support_features: np.ndarray,
    old_support_labels: Sequence[str],
    old_registered_classes: Sequence[str],
    new_support_features: np.ndarray,
    new_support_labels: Sequence[str],
    new_registered_classes: Sequence[str],
    old_stage_log_diag: np.ndarray,
    *,
    config: D33SphericalRegistrationConfig | None = None,
) -> D33SphericalRegistrationResult:
    """Register all old/new classes from support using one symmetric rule."""

    locked = config or D33SphericalRegistrationConfig()
    old_rows, old_labels, old_classes, old_k = _validate_labeled_support(
        old_support_features,
        old_support_labels,
        old_registered_classes,
        name="D33 old support",
    )
    new_rows, new_labels, new_classes, new_k = _validate_labeled_support(
        new_support_features,
        new_support_labels,
        new_registered_classes,
        name="D33 new support",
    )
    if (
        len(old_classes) < 2
        or len(old_classes) > MAX_OLD_CLASSES
        or len(new_classes) not in ALLOWED_NEW_CLASS_COUNTS
        or set(old_classes) & set(new_classes)
        or old_k != new_k
    ):
        raise D33SphericalRegistrationError(
            "D33 requires disjoint old/new class-symmetric matched K-shot support"
        )
    log_diag = np.asarray(old_stage_log_diag, dtype=np.float32)
    if (
        log_diag.shape != (FEATURE_DIM,)
        or not np.isfinite(log_diag).all()
    ):
        raise D33SphericalRegistrationError("old Stage2-B log diagonal drift")
    classes = old_classes + new_classes
    labels = np.concatenate((old_labels, new_labels))
    raw_rows = np.concatenate((old_rows, new_rows), axis=0).astype(np.float32)
    rows = _normalize(raw_rows * np.exp(log_diag)[None, :])
    targets = _targets(labels, classes)
    class_count = len(classes)
    trace: list[dict[str, Any]] = []

    if old_k == 1:
        radii = np.ones(class_count, dtype=np.float32)
        selected_quantile = 0.0
        selected_shrinkage = 0.0
        selected_ratio_cap = 1.0
        trace.append(
            {
                "selection_policy": locked.selection_policy,
                "selection_mode": "k1_uniform_radius_pure_cosine_bypass",
                "selected": True,
                "radius_quantile": 0.0,
                "radius_shrinkage": 0.0,
                "radius_ratio_cap": 1.0,
                "optimizer_steps": 0,
                "query_rows_used": 0,
            }
        )
        selection_macs = 0
    else:
        loso_distances, own_loso = _loso_geometry(rows, targets, class_count)
        candidates: list[
            tuple[tuple[float, ...], float, float, float, np.ndarray, dict[str, Any]]
        ] = []
        for quantile in locked.radius_quantiles:
            for shrinkage in locked.radius_shrinkages:
                for ratio_cap in locked.radius_ratio_caps:
                    raw_radii, candidate_radii, global_median = _robust_radii(
                        own_loso,
                        targets,
                        class_count,
                        quantile=quantile,
                        shrinkage=shrinkage,
                        ratio_cap=ratio_cap,
                    )
                    scores = _radius_scores(loso_distances, candidate_radii)
                    predictions = np.argmax(scores, axis=1)
                    correct = predictions == targets
                    true_scores = scores[np.arange(len(rows)), targets]
                    competitors = scores.copy()
                    competitors[np.arange(len(rows)), targets] = -np.inf
                    margins = true_scores - np.max(competitors, axis=1)
                    per_class_accuracy = [
                        float(np.mean(correct[targets == index]))
                        for index in range(class_count)
                    ]
                    per_class_margin = [
                        float(np.mean(margins[targets == index]))
                        for index in range(class_count)
                    ]
                    overall = float(np.mean(correct))
                    floor = float(min(per_class_accuracy))
                    min_class_margin = float(min(per_class_margin))
                    mean_margin = float(np.mean(margins))
                    rank = _selection_rank(
                        locked.selection_policy,
                        overall=overall,
                        floor=floor,
                        min_class_margin=min_class_margin,
                        mean_margin=mean_margin,
                        quantile=quantile,
                        shrinkage=shrinkage,
                        ratio_cap=ratio_cap,
                    )
                    evidence = {
                        "selection_policy": locked.selection_policy,
                        "selection_mode": "fixed_grid_support_loso_robust_radius",
                        "radius_quantile": quantile,
                        "radius_shrinkage": shrinkage,
                        "radius_ratio_cap": ratio_cap,
                        "global_median_radius": global_median,
                        "raw_min_radius": float(np.min(raw_radii)),
                        "raw_max_radius": float(np.max(raw_radii)),
                        "selected_min_radius": float(np.min(candidate_radii)),
                        "selected_max_radius": float(np.max(candidate_radii)),
                        "loso_accuracy": overall,
                        "loso_class_floor": floor,
                        "loso_min_class_mean_margin": min_class_margin,
                        "loso_mean_margin": mean_margin,
                        "optimizer_steps": 0,
                        "query_rows_used": 0,
                    }
                    candidates.append(
                        (
                            rank,
                            quantile,
                            shrinkage,
                            ratio_cap,
                            candidate_radii,
                            evidence,
                        )
                    )
        chosen = max(candidates, key=lambda item: item[0])
        selected_quantile = chosen[1]
        selected_shrinkage = chosen[2]
        selected_ratio_cap = chosen[3]
        radii = chosen[4]
        for _, quantile, shrinkage, ratio_cap, _, evidence in candidates:
            trace.append(
                {
                    **evidence,
                    "selected": (
                        quantile == selected_quantile
                        and shrinkage == selected_shrinkage
                        and ratio_cap == selected_ratio_cap
                    ),
                }
            )
        selection_macs = len(candidates) * len(rows) * class_count

    centroids_fp32 = _centroids(rows, targets, class_count)
    centroids_qint8, centroid_scales = _quantize_centroids(centroids_fp32)
    del centroids_fp32
    counts = np.full(class_count, old_k, dtype=np.uint16)
    state = D33SphericalRegistrationState(
        schema=SCHEMA,
        classes=classes,
        old_class_count=len(old_classes),
        log_diag=log_diag,
        centroids_qint8=centroids_qint8,
        centroid_scales=centroid_scales,
        radii=np.asarray(radii, dtype=np.float32),
        support_count_by_class=counts,
        selected_quantile=selected_quantile,
        selected_shrinkage=selected_shrinkage,
        selected_ratio_cap=selected_ratio_cap,
        selection_policy=locked.selection_policy,
        optimizer_steps=0,
    )
    # Conservative arithmetic count: transform/normalize, LOSO centroid scoring,
    # scalar radius-grid scoring, and final centroid construction.
    transform_macs = 3 * len(rows) * FEATURE_DIM
    loso_geometry_macs = (
        0 if old_k == 1 else len(rows) * class_count * FEATURE_DIM
    )
    final_centroid_macs = 2 * len(rows) * FEATURE_DIM
    estimated_adaptation_macs = int(
        transform_macs
        + loso_geometry_macs
        + selection_macs
        + final_centroid_macs
    )
    audit: dict[str, Any] = {
        "schema": "cvs.phase2.d33_spherical_registration_resource.v1",
        "adaptation_mode": "EVAL_ONLY_CLOSED_FORM_ADAPTATION",
        "optimizer_steps": 0,
        "active_parameters": state.active_parameters,
        "active_parameter_cap": MAX_ACTIVE_PARAMETERS,
        "active_parameter_cap_pass": state.active_parameters < MAX_ACTIVE_PARAMETERS,
        "persistent_state_bytes": int(
            state.log_diag.nbytes
            + state.centroids_qint8.nbytes
            + state.centroid_scales.nbytes
            + state.radii.nbytes
        ),
        "estimated_adaptation_macs": estimated_adaptation_macs,
        "estimated_macs_per_query": int(class_count * (FEATURE_DIM + 4)),
        "dense_query_graph_bytes": 0,
        "old_new_shared_transform": True,
        "old_new_shared_centroid_rule": True,
        "old_new_shared_centroid_quantization": "symmetric_int8_per_class_scale",
        "resident_fp32_centroid_count": 0,
        "centroid_int8_bytes": int(state.centroids_qint8.nbytes),
        "centroid_scale_fp32_bytes": int(state.centroid_scales.nbytes),
        "old_new_shared_radius_rule": True,
        "old_new_shared_score_rule": True,
        "train_deploy_score_surface_identical": True,
        "support_only": True,
        "query_rows_used_for_fit": 0,
        "query_labels_used_for_fit": False,
        "query_features_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "source_sample_access": False,
        "source_derived_signal_access": False,
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "single_received_iq_row_per_support_sample": True,
    }
    return D33SphericalRegistrationResult(
        state=state,
        selection_trace=tuple(trace),
        resource_audit=audit,
    )


def score_d33_spherical_registration(
    state: D33SphericalRegistrationState, features: np.ndarray
) -> np.ndarray:
    """Score each row over all registered classes without role or quota input."""

    rows = _validate_rows(features, name="D33 scoring features")
    transformed = _normalize(rows * np.exp(state.log_diag)[None, :])
    return _readonly(
        _radius_scores(
            _angular_distance(transformed, state.dequantized_centroids()), state.radii
        ),
        np.float32,
    )


def predict_d33_spherical_registration(
    state: D33SphericalRegistrationState, features: np.ndarray
) -> np.ndarray:
    scores = score_d33_spherical_registration(state, features)
    return np.asarray(state.classes)[np.argmax(scores, axis=1)]


__all__ = [
    "ALLOWED_NEW_CLASS_COUNTS",
    "D33SphericalRegistrationConfig",
    "D33SphericalRegistrationError",
    "D33SphericalRegistrationResult",
    "D33SphericalRegistrationState",
    "MAX_ACTIVE_PARAMETERS",
    "RADIUS_QUANTILES",
    "RADIUS_RATIO_CAPS",
    "RADIUS_SHRINKAGES",
    "fit_d33_spherical_registration",
    "predict_d33_spherical_registration",
    "score_d33_spherical_registration",
]
