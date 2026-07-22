"""Independent D4a primitives for one fixed received LEO_weak observation.

The module has no dataset, clean/source, query-label, role-oracle, quota,
global-assignment, scorer, or runner interface. It accepts registered support
features derived from already received IQ, fits one support-only diagonal
equalizer, emits deterministic post-reception representation views, and
computes leave-one-out old-class floor diagnostics.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


EPS = 1.0e-8
DEFAULT_OPERATOR_ID = "d4a_fixed_feature_antithetic_v1"
DEFAULT_VIEW_SEED = 713101
DEFAULT_VIEW_SIGMA = 0.01
DEFAULT_EQUALIZER_SHRINKAGE = 0.25
DEFAULT_MAX_ABS_LOG_SCALE = 0.35
DEFAULT_TEMPERATURE = 18.0
MAX_TRAINABLE_PARAMETERS = 80_000
MAX_PERSISTENT_STATE_BYTES = 256 * 1024
VIEW_NAMES = ("base", "plus", "minus")


class SingleObservationFloorLockError(ValueError):
    """Raised when the independent single-observation primitive fails closed."""


@dataclass(frozen=True)
class ViewLineage:
    """Lineage for one computation view of one fixed received IQ."""

    physical_sample_id: str
    parent_received_iq_sha256: str
    operator_id: str
    view_seed: int
    view_name: str
    counts_as_additional_physical_sample: bool = False
    additional_leo_channel_state_generation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "physical_sample_id": self.physical_sample_id,
            "parent_received_iq_sha256": self.parent_received_iq_sha256,
            "operator_id": self.operator_id,
            "view_seed": self.view_seed,
            "view_name": self.view_name,
            "counts_as_additional_physical_sample": (
                self.counts_as_additional_physical_sample
            ),
            "additional_leo_channel_state_generation": (
                self.additional_leo_channel_state_generation
            ),
        }


@dataclass(frozen=True)
class PostReceptionViews:
    """Representation views that share their parent received-IQ payload."""

    features: np.ndarray
    lineages: tuple[tuple[ViewLineage, ...], ...]
    physical_sample_count: int
    representation_view_count_per_sample: int
    k_increment: int = 0


@dataclass(frozen=True)
class SupportEqualizerState:
    """Immutable support-only diagonal equalizer state."""

    schema: str
    feature_dim: int
    log_scale: np.ndarray
    operator_id: str
    view_seed: int
    view_sigma: float
    equalizer_shrinkage: float
    max_abs_log_scale: float
    trainable_parameters: int
    persistent_state_bytes: int
    support_rows_used_for_fit: int
    support_physical_samples_used_for_fit: int
    query_rows_used_for_fit: int = 0
    query_updates: int = 0

    def resource_audit(self) -> dict[str, Any]:
        return {
            "schema": "cvs.phase2.d4a_single_observation_resource.v1",
            "support_only": True,
            "query_rows_used_for_fit": self.query_rows_used_for_fit,
            "query_updates": self.query_updates,
            "trainable_parameters": self.trainable_parameters,
            "trainable_parameter_limit": MAX_TRAINABLE_PARAMETERS,
            "trainable_parameter_limit_pass": (
                self.trainable_parameters <= MAX_TRAINABLE_PARAMETERS
            ),
            "persistent_state_bytes": self.persistent_state_bytes,
            "persistent_state_limit_bytes": MAX_PERSISTENT_STATE_BYTES,
            "persistent_state_limit_pass": (
                self.persistent_state_bytes <= MAX_PERSISTENT_STATE_BYTES
            ),
            "dense_query_graph_bytes": 0,
            "additional_physical_samples_from_views": 0,
            "additional_leo_channel_states_generated": 0,
            "post_reception_view_from_fixed_received_iq_only": True,
        }


@dataclass(frozen=True)
class SupportEqualizerFit:
    """Support fit result with views and LOO floor diagnostics."""

    state: SupportEqualizerState
    support_views: PostReceptionViews
    loo_floor_statistics: dict[str, Any]


def _readonly_float32(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float32)
    array.setflags(write=False)
    return array


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    return rows / np.maximum(
        np.linalg.norm(rows, axis=-1, keepdims=True), EPS
    )


def _validate_feature_matrix(value: np.ndarray, *, field: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[0] < 1
        or rows.shape[1] < 1
        or not np.isfinite(rows).all()
    ):
        raise SingleObservationFloorLockError(
            f"{field} must be a finite float32 matrix [N,D]"
        )
    return np.ascontiguousarray(rows)


def _validate_sha256(value: str, *, field: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise SingleObservationFloorLockError(
            f"{field} must be lowercase SHA256 hex"
        )
    return text


def _validate_sample_metadata(
    *,
    row_count: int,
    parent_received_iq_sha256: Sequence[str],
    physical_sample_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    hashes = tuple(
        _validate_sha256(value, field="parent_received_iq_sha256")
        for value in parent_received_iq_sha256
    )
    sample_ids = tuple(str(value) for value in physical_sample_ids)
    if len(hashes) != row_count or len(sample_ids) != row_count:
        raise SingleObservationFloorLockError(
            "feature rows and received-IQ lineage metadata are misaligned"
        )
    if any(not value for value in sample_ids):
        raise SingleObservationFloorLockError(
            "physical_sample_id must be a non-empty opaque identifier"
        )
    if len(set(sample_ids)) != len(sample_ids):
        raise SingleObservationFloorLockError(
            "each support/query row must represent one unique physical sample"
        )
    return hashes, sample_ids


def received_iq_sha256(rows: np.ndarray) -> tuple[str, ...]:
    """Hash received IQ rows using the project float32 [N,2,L] convention."""

    iq = np.asarray(rows)
    if (
        iq.dtype != np.float32
        or iq.ndim != 3
        or iq.shape[0] < 1
        or iq.shape[1] != 2
        or iq.shape[2] < 1
        or not np.isfinite(iq).all()
    ):
        raise SingleObservationFloorLockError(
            "received LEO_weak IQ must be finite float32 [N,2,L]"
        )
    return tuple(
        hashlib.sha256(
            np.ascontiguousarray(row, dtype="<f4").tobytes(order="C")
        ).hexdigest()
        for row in iq
    )


def _direction(
    base: np.ndarray,
    *,
    parent_received_iq_sha256: str,
    operator_id: str,
    view_seed: int,
) -> np.ndarray:
    payload = "|".join(
        (
            str(parent_received_iq_sha256),
            str(operator_id),
            str(int(view_seed)),
            str(int(base.shape[0])),
        )
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(base.shape[0]).astype(np.float32)
    direction -= np.dot(direction, base).astype(np.float32) * base
    norm = float(np.linalg.norm(direction))
    if norm <= EPS:
        direction = np.zeros_like(base)
        direction[seed % len(direction)] = 1.0
        direction -= np.dot(direction, base).astype(np.float32) * base
        norm = float(np.linalg.norm(direction))
    if norm <= EPS:
        raise SingleObservationFloorLockError(
            "failed to derive a deterministic post-reception view direction"
        )
    return direction / norm


def derive_post_reception_views(
    features: np.ndarray,
    *,
    parent_received_iq_sha256: Sequence[str],
    physical_sample_ids: Sequence[str],
    operator_id: str = DEFAULT_OPERATOR_ID,
    view_seed: int = DEFAULT_VIEW_SEED,
    sigma: float = DEFAULT_VIEW_SIGMA,
) -> PostReceptionViews:
    """Derive deterministic base/antithetic views from fixed received-IQ features."""

    rows = _validate_feature_matrix(features, field="features")
    hashes, sample_ids = _validate_sample_metadata(
        row_count=len(rows),
        parent_received_iq_sha256=parent_received_iq_sha256,
        physical_sample_ids=physical_sample_ids,
    )
    if not str(operator_id):
        raise SingleObservationFloorLockError("operator_id must be non-empty")
    if int(view_seed) < 0:
        raise SingleObservationFloorLockError("view_seed must be nonnegative")
    if not math.isfinite(float(sigma)) or not 0.0 <= float(sigma) <= 0.10:
        raise SingleObservationFloorLockError("view sigma must be in [0,0.10]")

    base_rows = _normalize_rows(rows)
    result: list[np.ndarray] = []
    lineage_rows: list[tuple[ViewLineage, ...]] = []
    for index, base in enumerate(base_rows):
        direction = _direction(
            base,
            parent_received_iq_sha256=hashes[index],
            operator_id=str(operator_id),
            view_seed=int(view_seed),
        )
        sample_views = np.stack(
            (
                base,
                _normalize_rows((base + float(sigma) * direction)[None, :])[0],
                _normalize_rows((base - float(sigma) * direction)[None, :])[0],
            )
        ).astype(np.float32)
        result.append(sample_views)
        lineage_rows.append(
            tuple(
                ViewLineage(
                    physical_sample_id=sample_ids[index],
                    parent_received_iq_sha256=hashes[index],
                    operator_id=str(operator_id),
                    view_seed=int(view_seed),
                    view_name=view_name,
                )
                for view_name in VIEW_NAMES
            )
        )
    return PostReceptionViews(
        features=_readonly_float32(np.stack(result)),
        lineages=tuple(lineage_rows),
        physical_sample_count=len(rows),
        representation_view_count_per_sample=len(VIEW_NAMES),
        k_increment=0,
    )


def _class_balanced_log_scale(
    support_features: np.ndarray,
    labels: np.ndarray,
    *,
    shrinkage: float,
    max_abs_log_scale: float,
) -> np.ndarray:
    class_energies = np.stack(
        [
            np.mean(
                np.square(support_features[labels == label], dtype=np.float32),
                axis=0,
            )
            for label in sorted(set(labels.tolist()))
        ]
    )
    energy = np.mean(class_energies, axis=0)
    log_energy = np.log(np.maximum(energy, EPS))
    raw = -0.5 * (log_energy - np.mean(log_energy))
    return np.clip(
        float(shrinkage) * raw,
        -float(max_abs_log_scale),
        float(max_abs_log_scale),
    ).astype(np.float32)


def _prototype(rows: np.ndarray) -> np.ndarray:
    return _normalize_rows(np.mean(rows, axis=0, keepdims=True))[0]


def _loo_floor_statistics(
    equalized_views: np.ndarray,
    labels: np.ndarray,
    physical_sample_ids: Sequence[str],
    *,
    temperature: float,
) -> dict[str, Any]:
    classes = tuple(sorted(set(labels.tolist())))
    class_indices = {
        label: np.flatnonzero(labels == label) for label in classes
    }
    records: dict[str, list[tuple[bool, float]]] = {
        label: [] for label in classes
    }
    modes: dict[str, str] = {}

    for true_index, true_label in enumerate(classes):
        indices = class_indices[true_label]
        if len(indices) >= 2:
            modes[true_label] = "leave_one_physical_sample_out"
            evaluations = (
                (int(index), 0, equalized_views[index, 0]) for index in indices
            )
        else:
            modes[true_label] = "leave_one_view_out_k1"
            only = int(indices[0])
            evaluations = (
                (only, view_index, equalized_views[only, view_index])
                for view_index in range(len(VIEW_NAMES))
            )

        for sample_index, view_index, feature in evaluations:
            prototypes: list[np.ndarray] = []
            for candidate_label in classes:
                candidate_indices = class_indices[candidate_label]
                if candidate_label == true_label and len(indices) >= 2:
                    prototype_rows = equalized_views[
                        candidate_indices[candidate_indices != sample_index], 0
                    ]
                elif candidate_label == true_label:
                    remaining_views = [
                        index
                        for index in range(len(VIEW_NAMES))
                        if index != view_index
                    ]
                    prototype_rows = equalized_views[
                        sample_index, remaining_views
                    ]
                elif len(candidate_indices) == 1:
                    prototype_rows = equalized_views[
                        int(candidate_indices[0]), :
                    ]
                else:
                    prototype_rows = equalized_views[candidate_indices, 0]
                prototypes.append(_prototype(prototype_rows))

            scores = float(temperature) * (
                _normalize_rows(feature[None, :])[0]
                @ np.stack(prototypes).T
            )
            other = np.delete(scores, true_index)
            margin = float(scores[true_index] - np.max(other))
            prediction = int(np.argmax(scores))
            records[true_label].append((prediction == true_index, margin))

    per_class: dict[str, dict[str, Any]] = {}
    total_correct = 0
    total_rows = 0
    for label in classes:
        class_records = records[label]
        correct = sum(int(item[0]) for item in class_records)
        margins = np.asarray([item[1] for item in class_records], dtype=np.float64)
        total_correct += correct
        total_rows += len(class_records)
        per_class[label] = {
            "loo_mode": modes[label],
            "unique_physical_sample_count": int(
                len(
                    {
                        str(physical_sample_ids[index])
                        for index in class_indices[label]
                    }
                )
            ),
            "evaluation_rows": len(class_records),
            "correct": correct,
            "accuracy": correct / len(class_records),
            "mean_margin": float(np.mean(margins)),
            "q10_margin": float(np.quantile(margins, 0.10)),
            "worst_margin": float(np.min(margins)),
        }
    return {
        "schema": "cvs.phase2.d4a_loo_floor_statistics.v1",
        "support_only": True,
        "query_rows_used": 0,
        "class_count": len(classes),
        "physical_support_sample_count": len(physical_sample_ids),
        "representation_views_count_as_additional_physical_samples": False,
        "k_increment_from_views": 0,
        "overall_loo_accuracy": total_correct / total_rows,
        "min_class_loo_accuracy": min(
            float(value["accuracy"]) for value in per_class.values()
        ),
        "worst_class_q10_margin": min(
            float(value["q10_margin"]) for value in per_class.values()
        ),
        "worst_class_margin": min(
            float(value["worst_margin"]) for value in per_class.values()
        ),
        "per_class": per_class,
    }


def fit_support_equalizer(
    support_features: np.ndarray,
    support_labels: Sequence[str],
    *,
    parent_received_iq_sha256: Sequence[str],
    physical_sample_ids: Sequence[str],
    operator_id: str = DEFAULT_OPERATOR_ID,
    view_seed: int = DEFAULT_VIEW_SEED,
    view_sigma: float = DEFAULT_VIEW_SIGMA,
    equalizer_shrinkage: float = DEFAULT_EQUALIZER_SHRINKAGE,
    max_abs_log_scale: float = DEFAULT_MAX_ABS_LOG_SCALE,
    temperature: float = DEFAULT_TEMPERATURE,
) -> SupportEqualizerFit:
    """Fit a class-balanced equalizer using registered physical support only."""

    rows = _validate_feature_matrix(
        support_features, field="support_features"
    )
    labels = np.asarray(tuple(str(value) for value in support_labels))
    if len(labels) != len(rows) or any(not value for value in labels.tolist()):
        raise SingleObservationFloorLockError(
            "support feature/label alignment drift"
        )
    if len(set(labels.tolist())) < 2:
        raise SingleObservationFloorLockError(
            "LOO floor statistics require at least two registered classes"
        )
    if not 0.0 <= float(equalizer_shrinkage) <= 1.0:
        raise SingleObservationFloorLockError(
            "equalizer shrinkage must be in [0,1]"
        )
    if not 0.0 < float(max_abs_log_scale) <= 1.5:
        raise SingleObservationFloorLockError(
            "max_abs_log_scale must be in (0,1.5]"
        )
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise SingleObservationFloorLockError("temperature must be positive")

    feature_dim = int(rows.shape[1])
    trainable_parameters = feature_dim
    if trainable_parameters > MAX_TRAINABLE_PARAMETERS:
        raise SingleObservationFloorLockError(
            "single-observation equalizer exceeds 80k trainable parameters"
        )
    views = derive_post_reception_views(
        rows,
        parent_received_iq_sha256=parent_received_iq_sha256,
        physical_sample_ids=physical_sample_ids,
        operator_id=operator_id,
        view_seed=view_seed,
        sigma=view_sigma,
    )
    log_scale = _class_balanced_log_scale(
        _normalize_rows(rows),
        labels,
        shrinkage=float(equalizer_shrinkage),
        max_abs_log_scale=float(max_abs_log_scale),
    )
    metadata = {
        "schema": "cvs.phase2.d4a_single_observation_equalizer.v1",
        "feature_dim": feature_dim,
        "operator_id": str(operator_id),
        "view_seed": int(view_seed),
        "view_sigma": float(view_sigma),
        "equalizer_shrinkage": float(equalizer_shrinkage),
        "max_abs_log_scale": float(max_abs_log_scale),
        "query_rows_used_for_fit": 0,
        "query_updates": 0,
    }
    persistent_state_bytes = len(
        json.dumps(
            metadata,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ) + int(log_scale.nbytes)
    if persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise SingleObservationFloorLockError(
            "single-observation equalizer state exceeds 256KB"
        )
    state = SupportEqualizerState(
        schema=metadata["schema"],
        feature_dim=feature_dim,
        log_scale=_readonly_float32(log_scale),
        operator_id=str(operator_id),
        view_seed=int(view_seed),
        view_sigma=float(view_sigma),
        equalizer_shrinkage=float(equalizer_shrinkage),
        max_abs_log_scale=float(max_abs_log_scale),
        trainable_parameters=trainable_parameters,
        persistent_state_bytes=persistent_state_bytes,
        support_rows_used_for_fit=len(rows),
        support_physical_samples_used_for_fit=len(rows),
    )
    equalized = _normalize_rows(
        views.features * np.exp(state.log_scale)[None, None, :]
    )
    equalized_views = PostReceptionViews(
        features=_readonly_float32(equalized),
        lineages=views.lineages,
        physical_sample_count=views.physical_sample_count,
        representation_view_count_per_sample=(
            views.representation_view_count_per_sample
        ),
        k_increment=0,
    )
    floor_statistics = _loo_floor_statistics(
        equalized_views.features,
        labels,
        tuple(str(value) for value in physical_sample_ids),
        temperature=float(temperature),
    )
    return SupportEqualizerFit(
        state=state,
        support_views=equalized_views,
        loo_floor_statistics=floor_statistics,
    )


def transform_query_features(
    state: SupportEqualizerState,
    query_features: np.ndarray,
    *,
    parent_received_iq_sha256: Sequence[str],
    physical_sample_ids: Sequence[str],
    view_names: Sequence[str] = ("base",),
) -> PostReceptionViews:
    """Inference-only query transform; this function cannot update fit state."""

    if not isinstance(state, SupportEqualizerState):
        raise SingleObservationFloorLockError(
            "query transform requires an immutable support equalizer state"
        )
    rows = _validate_feature_matrix(query_features, field="query_features")
    if rows.shape[1] != state.feature_dim:
        raise SingleObservationFloorLockError("query feature dimension drift")
    requested = tuple(str(value) for value in view_names)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(value not in VIEW_NAMES for value in requested)
    ):
        raise SingleObservationFloorLockError(
            "query view_names must be a unique subset of base/plus/minus"
        )
    derived = derive_post_reception_views(
        rows,
        parent_received_iq_sha256=parent_received_iq_sha256,
        physical_sample_ids=physical_sample_ids,
        operator_id=state.operator_id,
        view_seed=state.view_seed,
        sigma=state.view_sigma,
    )
    positions = [VIEW_NAMES.index(value) for value in requested]
    equalized = _normalize_rows(
        derived.features[:, positions, :]
        * np.exp(state.log_scale)[None, None, :]
    )
    selected_lineages = tuple(
        tuple(row[position] for position in positions)
        for row in derived.lineages
    )
    return PostReceptionViews(
        features=_readonly_float32(equalized),
        lineages=selected_lineages,
        physical_sample_count=len(rows),
        representation_view_count_per_sample=len(positions),
        k_increment=0,
    )


def query_interface_is_inference_only() -> bool:
    """Machine-checkable guard for accidental query-label/update parameters."""

    forbidden = {
        "labels",
        "query_labels",
        "truth",
        "roles",
        "quota",
        "optimizer",
        "update",
        "fit",
    }
    parameters = set(inspect.signature(transform_query_features).parameters)
    return not any(
        any(token in parameter.lower() for token in forbidden)
        for parameter in parameters
    )


__all__ = [
    "DEFAULT_OPERATOR_ID",
    "DEFAULT_VIEW_SEED",
    "MAX_TRAINABLE_PARAMETERS",
    "PostReceptionViews",
    "SingleObservationFloorLockError",
    "SupportEqualizerFit",
    "SupportEqualizerState",
    "ViewLineage",
    "derive_post_reception_views",
    "fit_support_equalizer",
    "query_interface_is_inference_only",
    "received_iq_sha256",
    "transform_query_features",
]
