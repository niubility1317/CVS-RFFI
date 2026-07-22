"""D5 support-only low-rank metric adaptation for sealed Phase2 features.

The primitive is deliberately isolated from datasets, clean/source artifacts,
query truth, old/new role metadata, query quotas, and batch-global assignment.
It fits one closed-form projection per LEO scenario from registered support
labels only.  A before state can later be extended by registering labels that
are absent from its immutable registry; the projection and existing prototypes
remain bitwise unchanged.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np


EPS = 1.0e-8
FORMAL_LEO_WEAK_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
DEFAULT_RANK_CANDIDATES = (8, 16, 32)
DEFAULT_SHRINKAGE_CANDIDATES = (0.10, 0.30, 0.60, 0.90)
SUPERVISED_RESIDUAL_WEIGHT = 0.05
MAX_TRAINABLE_PARAMETERS = 80_000
MAX_PERSISTENT_STATE_BYTES = 256 * 1024


class SupportLowRankMetricError(ValueError):
    """Raised when the closed-form D5 primitive fails closed."""


@dataclass(frozen=True)
class SupportLowRankMetricState:
    """Immutable state for one scenario-specific registered-class metric."""

    schema: str
    scenario: str
    feature_dim: int
    rank: int
    shrinkage: float
    center: np.ndarray
    projection: np.ndarray
    classes: tuple[str, ...]
    prototypes: np.ndarray
    selection_trace: tuple[dict[str, Any], ...]
    selection_protocol: str
    support_rows_used_for_fit: int
    support_fit_solve_count: int
    registration_generation: int
    trainable_parameters: int
    persistent_state_bytes: int
    query_rows_used_for_fit: int = 0
    query_updates: int = 0

    def resource_audit(self) -> dict[str, Any]:
        class_count = len(self.classes)
        return {
            "schema": "cvs.phase2.d5_support_lowrank_resource.v1",
            "candidate": "d5_support_only_lowrank_whitened_fisher",
            "scenario": self.scenario,
            "closed_form_adaptation": True,
            "adaptation_epochs": 0,
            "support_only": True,
            "query_rows_used_for_fit": self.query_rows_used_for_fit,
            "query_updates": self.query_updates,
            "query_decision_policy": "per_sample_all_registered_classes",
            "role_oracle_access": False,
            "class_quota_access": False,
            "batch_global_assignment": False,
            "dense_query_graph_bytes": 0,
            "feature_dim": self.feature_dim,
            "rank": self.rank,
            "registered_class_count": class_count,
            "projection_parameters": self.trainable_parameters,
            "registered_head_parameters": class_count * self.rank,
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
            "projection_macs_per_query": self.feature_dim * self.rank,
            "prototype_macs_per_query": class_count * self.rank,
            "support_fit_solve_count": self.support_fit_solve_count,
            "selection_protocol": self.selection_protocol,
            "old_projection_reused_after_registration": (
                self.registration_generation > 0
            ),
            "old_prototypes_reused_after_registration": (
                self.registration_generation > 0
            ),
        }


@dataclass(frozen=True)
class ScenarioAtomicLowRankState:
    """Three independent scenario states with one support-locked arm."""

    schema: str
    scenarios: tuple[str, ...]
    states: tuple[SupportLowRankMetricState, ...]
    rank: int
    shrinkage: float
    selection_trace: tuple[dict[str, Any], ...]
    trainable_parameters: int
    persistent_state_bytes: int

    def state_for(self, scenario: str) -> SupportLowRankMetricState:
        try:
            index = self.scenarios.index(str(scenario))
        except ValueError as error:
            raise SupportLowRankMetricError(
                "scenario-specific D5 state lookup drift"
            ) from error
        return self.states[index]

    def resource_audit(self) -> dict[str, Any]:
        return {
            "schema": "cvs.phase2.d5_scenario_atomic_resource.v1",
            "candidate": "d5_support_only_lowrank_whitened_fisher",
            "scenario_atomic": True,
            "cross_scenario_support_concat": False,
            "scenario_count": len(self.states),
            "rank": self.rank,
            "shrinkage": self.shrinkage,
            "adaptation_epochs": 0,
            "support_only": True,
            "query_rows_used_for_fit": 0,
            "query_updates": 0,
            "role_oracle_access": False,
            "class_quota_access": False,
            "batch_global_assignment": False,
            "dense_query_graph_bytes": 0,
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
            "per_scenario": {
                scenario: state.resource_audit()
                for scenario, state in zip(self.scenarios, self.states)
            },
        }


@dataclass(frozen=True)
class LowRankPrediction:
    """Inference-only all-registered-class scores and decisions."""

    labels: tuple[str, ...]
    scores: np.ndarray


def _readonly_float32(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float32)
    array.setflags(write=False)
    return array


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), EPS)


def _validate_features(value: np.ndarray, *, field: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[0] < 1
        or rows.shape[1] < 1
        or not np.isfinite(rows).all()
    ):
        raise SupportLowRankMetricError(
            f"{field} must be a finite float32 matrix [N,D]"
        )
    return np.ascontiguousarray(rows)


def _validate_labels(
    values: Sequence[str], *, row_count: int, field: str
) -> np.ndarray:
    labels = np.asarray(tuple(str(value) for value in values))
    if labels.ndim != 1 or len(labels) != row_count or any(not x for x in labels):
        raise SupportLowRankMetricError(f"{field} alignment drift")
    return labels


def _validate_sha256(value: str, *, field: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise SupportLowRankMetricError(f"{field} must be lowercase SHA256 hex")
    return text


def _stable_sign(rows: np.ndarray) -> np.ndarray:
    output = np.ascontiguousarray(rows, dtype=np.float32)
    for column in range(output.shape[1]):
        direction = output[:, column]
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0:
            output[:, column] *= -1.0
    return output


def _class_balanced_center(
    rows: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    classes = tuple(sorted(set(labels.tolist())))
    means = np.stack(
        [rows[labels == label].mean(axis=0) for label in classes], axis=0
    ).astype(np.float32)
    center = means.mean(axis=0, dtype=np.float32)
    return center, classes, means


def _fit_projection(
    rows: np.ndarray,
    labels: np.ndarray,
    *,
    rank: int,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    center, classes, class_means = _class_balanced_center(rows, labels)
    residuals = np.concatenate(
        [rows[labels == label] - class_means[index] for index, label in enumerate(classes)],
        axis=0,
    )
    within_variance = np.mean(np.square(residuals), axis=0, dtype=np.float64)
    positive = within_variance[within_variance > EPS]
    variance_target = float(
        np.median(positive) if len(positive) else np.mean(within_variance) + EPS
    )
    variance_target = max(variance_target, EPS)
    regularized = (
        (1.0 - float(shrinkage)) * within_variance
        + float(shrinkage) * variance_target
    )
    whitening = np.sqrt(
        variance_target / np.maximum(regularized, EPS)
    ).astype(np.float32)
    whitening = np.clip(whitening, 0.25, 4.0)

    between = (class_means - center[None, :]) * whitening[None, :]
    total = (rows - center[None, :]) * whitening[None, :]
    stacked = np.concatenate(
        [
            between / np.sqrt(max(1, len(classes))),
            total
            * np.sqrt(SUPERVISED_RESIDUAL_WEIGHT / max(1, len(total))),
        ],
        axis=0,
    )
    maximum_rank = min(stacked.shape)
    if rank < 1 or rank > maximum_rank:
        raise SupportLowRankMetricError(
            f"rank {rank} exceeds support-derived projection limit {maximum_rank}"
        )
    _u, _singular, vt = np.linalg.svd(stacked, full_matrices=False)
    directions = _stable_sign(vt[:rank].T)
    projection = whitening[:, None] * directions
    return _readonly_float32(center), _readonly_float32(projection)


def _project(
    rows: np.ndarray, *, center: np.ndarray, projection: np.ndarray
) -> np.ndarray:
    return _normalize_rows((rows - center[None, :]) @ projection)


def _prototypes(
    projected: np.ndarray, labels: np.ndarray, classes: tuple[str, ...]
) -> np.ndarray:
    values = np.stack(
        [
            _normalize_rows(projected[labels == label].mean(axis=0, keepdims=True))[0]
            for label in classes
        ],
        axis=0,
    )
    return _readonly_float32(values)


def _deletion_folds(labels: np.ndarray) -> tuple[tuple[int, ...], ...]:
    classes = tuple(sorted(set(labels.tolist())))
    counts = [int(np.sum(labels == label)) for label in classes]
    if min(counts) < 2:
        raise SupportLowRankMetricError(
            "support-only arm selection requires at least two samples per class"
        )
    width = 2 if min(counts) >= 4 else 1
    by_class = [np.flatnonzero(labels == label).tolist() for label in classes]
    folds: list[tuple[int, ...]] = []
    for offset in range(0, max(counts), width):
        held: list[int] = []
        for indices in by_class:
            held.extend(indices[offset : offset + width])
        if held:
            folds.append(tuple(sorted(held)))
    return tuple(folds)


def _candidate_metrics(
    rows: np.ndarray,
    labels: np.ndarray,
    *,
    rank: int,
    shrinkage: float,
) -> dict[str, Any]:
    folds = _deletion_folds(labels)
    predicted: list[str] = []
    truth: list[str] = []
    margins: list[float] = []
    solve_count = 0
    all_indices = np.arange(len(rows))
    for held_tuple in folds:
        held = np.asarray(held_tuple, dtype=np.int64)
        keep = np.ones(len(rows), dtype=bool)
        keep[held] = False
        train_x = rows[keep]
        train_y = labels[keep]
        if len(set(train_y.tolist())) != len(set(labels.tolist())):
            raise SupportLowRankMetricError(
                "support deletion fold removed an entire registered class"
            )
        center, projection = _fit_projection(
            train_x,
            train_y,
            rank=rank,
            shrinkage=shrinkage,
        )
        classes = tuple(sorted(set(train_y.tolist())))
        train_z = _project(train_x, center=center, projection=projection)
        proto = _prototypes(train_z, train_y, classes)
        held_z = _project(rows[held], center=center, projection=projection)
        score = held_z @ proto.T
        order = np.argsort(score, axis=1)
        top = order[:, -1]
        second = order[:, -2] if score.shape[1] > 1 else top
        predicted.extend(classes[int(index)] for index in top)
        truth.extend(labels[held].tolist())
        margins.extend(
            (
                score[np.arange(len(held)), top]
                - score[np.arange(len(held)), second]
            ).astype(float).tolist()
        )
        solve_count += 1
        if len(np.intersect1d(all_indices[keep], held)):
            raise AssertionError("support deletion fold overlap")

    predicted_array = np.asarray(predicted)
    truth_array = np.asarray(truth)
    margin_array = np.asarray(margins, dtype=np.float64)
    classes = tuple(sorted(set(labels.tolist())))
    per_class = {
        label: float(np.mean(predicted_array[truth_array == label] == label))
        for label in classes
    }
    return {
        "rank": int(rank),
        "shrinkage": float(shrinkage),
        "validation_protocol": (
            "support_leave_two_out_per_class"
            if min(int(np.sum(labels == label)) for label in classes) >= 4
            else "support_leave_one_out_per_class"
        ),
        "fold_count": len(folds),
        "solve_count": solve_count,
        "overall_accuracy": float(np.mean(predicted_array == truth_array)),
        "min_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
        "margin_q10": float(np.quantile(margin_array, 0.10)),
        "mean_margin": float(np.mean(margin_array)),
    }


def _candidate_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["min_class_accuracy"]),
        float(row["overall_accuracy"]),
        float(row["margin_q10"]),
        float(row["mean_margin"]),
        -float(row["rank"]),
        float(row["shrinkage"]),
    )


def _selection_maximum_rank(labels: np.ndarray, feature_dim: int) -> int:
    folds = _deletion_folds(labels)
    largest_holdout = max(len(fold) for fold in folds)
    class_count = len(set(labels.tolist()))
    return min(
        int(feature_dim),
        int(len(labels) - largest_holdout + class_count),
    )


def _candidate_grid(
    rows: np.ndarray,
    labels: np.ndarray,
    *,
    rank_candidates: Sequence[int],
    shrinkage_candidates: Sequence[float],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    maximum_rank = _selection_maximum_rank(labels, rows.shape[1])
    ranks = tuple(
        sorted(
            {
                int(rank)
                for rank in rank_candidates
                if int(rank) >= 1 and int(rank) <= maximum_rank
            }
        )
    )
    shrinkages = tuple(
        sorted(
            {
                float(value)
                for value in shrinkage_candidates
                if 0.0 <= float(value) <= 1.0
            }
        )
    )
    if not ranks or not shrinkages:
        raise SupportLowRankMetricError("D5 support selection grid is empty")
    trace = tuple(
        _candidate_metrics(
            rows,
            labels,
            rank=rank,
            shrinkage=shrinkage,
        )
        for rank in ranks
        for shrinkage in shrinkages
    )
    return trace, max(trace, key=_candidate_key)


def _state_bytes(
    *,
    center: np.ndarray,
    projection: np.ndarray,
    prototypes: np.ndarray,
    classes: Sequence[str],
) -> int:
    return int(
        center.nbytes
        + projection.nbytes
        + prototypes.nbytes
        + sum(len(label.encode("utf-8")) for label in classes)
    )


def _fit_locked_state(
    rows: np.ndarray,
    labels: np.ndarray,
    *,
    scenario: str,
    rank: int,
    shrinkage: float,
    selection_trace: tuple[dict[str, Any], ...],
    selection_protocol: str,
    support_fit_solve_count: int,
) -> SupportLowRankMetricState:
    requested_parameters = int(rows.shape[1]) * int(rank)
    if requested_parameters > MAX_TRAINABLE_PARAMETERS:
        raise SupportLowRankMetricError("D5 trainable parameter cap exceeded")
    center, projection = _fit_projection(
        rows, labels, rank=rank, shrinkage=shrinkage
    )
    classes = tuple(sorted(set(labels.tolist())))
    transformed = _project(rows, center=center, projection=projection)
    prototypes = _prototypes(transformed, labels, classes)
    trainable_parameters = int(projection.size)
    persistent_state_bytes = _state_bytes(
        center=center,
        projection=projection,
        prototypes=prototypes,
        classes=classes,
    )
    if trainable_parameters > MAX_TRAINABLE_PARAMETERS:
        raise SupportLowRankMetricError("D5 trainable parameter cap exceeded")
    if persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise SupportLowRankMetricError("D5 persistent state cap exceeded")
    return SupportLowRankMetricState(
        schema="cvs.phase2.d5_support_lowrank_metric_state.v1",
        scenario=str(scenario),
        feature_dim=int(rows.shape[1]),
        rank=int(rank),
        shrinkage=float(shrinkage),
        center=center,
        projection=projection,
        classes=classes,
        prototypes=prototypes,
        selection_trace=selection_trace,
        selection_protocol=selection_protocol,
        support_rows_used_for_fit=int(len(rows)),
        support_fit_solve_count=int(support_fit_solve_count + 1),
        registration_generation=0,
        trainable_parameters=trainable_parameters,
        persistent_state_bytes=persistent_state_bytes,
    )


def fit_support_lowrank_metric(
    support_features: np.ndarray,
    support_labels: Sequence[str],
    *,
    scenario: str,
    rank_candidates: Sequence[int] = DEFAULT_RANK_CANDIDATES,
    shrinkage_candidates: Sequence[float] = DEFAULT_SHRINKAGE_CANDIDATES,
    locked_rank: int | None = None,
    locked_shrinkage: float | None = None,
) -> SupportLowRankMetricState:
    """Fit one scenario from registered support; no query argument exists."""

    rows = _validate_features(support_features, field="support_features")
    labels = _validate_labels(
        support_labels, row_count=len(rows), field="support_labels"
    )
    if len(set(labels.tolist())) < 2:
        raise SupportLowRankMetricError(
            "D5 requires at least two registered support classes"
        )
    if (locked_rank is None) != (locked_shrinkage is None):
        raise SupportLowRankMetricError(
            "locked rank and shrinkage must be provided together"
        )
    if locked_rank is None:
        trace, chosen = _candidate_grid(
            rows,
            labels,
            rank_candidates=rank_candidates,
            shrinkage_candidates=shrinkage_candidates,
        )
        rank = int(chosen["rank"])
        shrinkage = float(chosen["shrinkage"])
        protocol = str(chosen["validation_protocol"])
        solve_count = sum(int(row["solve_count"]) for row in trace)
    else:
        trace = ()
        rank = int(locked_rank)
        shrinkage = float(locked_shrinkage)
        protocol = "locked_from_support_only_development_selection"
        solve_count = 0
    return _fit_locked_state(
        rows,
        labels,
        scenario=scenario,
        rank=rank,
        shrinkage=shrinkage,
        selection_trace=trace,
        selection_protocol=protocol,
        support_fit_solve_count=solve_count,
    )


def register_absent_classes(
    base_state: SupportLowRankMetricState,
    registered_support_features: np.ndarray,
    registered_support_labels: Sequence[str],
) -> SupportLowRankMetricState:
    """Register labels absent from the immutable before registry.

    Existing-label rows are ignored and cannot update the projection, center,
    or existing prototypes.  This is class-registry membership, not query role
    access: the function has no query or old/new role input.
    """

    if not isinstance(base_state, SupportLowRankMetricState):
        raise SupportLowRankMetricError("D5 registration requires a base state")
    rows = _validate_features(
        registered_support_features, field="registered_support_features"
    )
    labels = _validate_labels(
        registered_support_labels,
        row_count=len(rows),
        field="registered_support_labels",
    )
    if rows.shape[1] != base_state.feature_dim:
        raise SupportLowRankMetricError("D5 registration feature dimension drift")
    base_classes = set(base_state.classes)
    new_classes = tuple(sorted(set(labels.tolist()) - base_classes))
    if not new_classes:
        raise SupportLowRankMetricError(
            "D5 registration received no absent registered labels"
        )
    new_rows = np.isin(labels, np.asarray(new_classes))
    transformed = _project(
        rows[new_rows],
        center=base_state.center,
        projection=base_state.projection,
    )
    new_prototypes = _prototypes(
        transformed, labels[new_rows], new_classes
    )
    classes = base_state.classes + new_classes
    prototypes = _readonly_float32(
        np.concatenate([base_state.prototypes, new_prototypes], axis=0)
    )
    persistent_state_bytes = _state_bytes(
        center=base_state.center,
        projection=base_state.projection,
        prototypes=prototypes,
        classes=classes,
    )
    if persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise SupportLowRankMetricError(
            "D5 registered persistent state cap exceeded"
        )
    return replace(
        base_state,
        classes=classes,
        prototypes=prototypes,
        registration_generation=base_state.registration_generation + 1,
        persistent_state_bytes=persistent_state_bytes,
    )


def predict_all_registered(
    state: SupportLowRankMetricState, query_features: np.ndarray
) -> LowRankPrediction:
    """Classify each query independently over every registered class."""

    if not isinstance(state, SupportLowRankMetricState):
        raise SupportLowRankMetricError("D5 prediction requires immutable state")
    rows = _validate_features(query_features, field="query_features")
    if rows.shape[1] != state.feature_dim:
        raise SupportLowRankMetricError("D5 query feature dimension drift")
    transformed = _project(
        rows, center=state.center, projection=state.projection
    )
    scores = _readonly_float32(transformed @ state.prototypes.T)
    indices = np.argmax(scores, axis=1)
    return LowRankPrediction(
        labels=tuple(state.classes[int(index)] for index in indices),
        scores=scores,
    )


def _validate_scenario_lineage(
    support_by_scenario: Mapping[
        str, tuple[np.ndarray, Sequence[str], Sequence[str], Sequence[str]]
    ]
) -> None:
    if set(support_by_scenario) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise SupportLowRankMetricError("D5 formal scenario coverage drift")
    seen_physical: set[str] = set()
    seen_received_iq: set[str] = set()
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        rows, labels, physical_ids, received_hashes = support_by_scenario[scenario]
        row_count = len(np.asarray(rows))
        _validate_labels(labels, row_count=row_count, field="support_labels")
        ids = tuple(str(value) for value in physical_ids)
        hashes = tuple(
            _validate_sha256(value, field="parent_received_iq_sha256")
            for value in received_hashes
        )
        if (
            len(ids) != row_count
            or len(hashes) != row_count
            or any(not value for value in ids)
            or len(set(ids)) != row_count
            or len(set(hashes)) != row_count
        ):
            raise SupportLowRankMetricError(
                "D5 single-observation lineage alignment drift"
            )
        if seen_physical.intersection(ids):
            raise SupportLowRankMetricError(
                "D5 physical support reuse across scenarios is forbidden"
            )
        if seen_received_iq.intersection(hashes):
            raise SupportLowRankMetricError(
                "D5 received-IQ reuse across scenarios is forbidden"
            )
        seen_physical.update(ids)
        seen_received_iq.update(hashes)


def _scenario_candidate_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["worst_scenario_min_class_accuracy"]),
        float(row["mean_scenario_min_class_accuracy"]),
        float(row["mean_overall_accuracy"]),
        float(row["worst_scenario_margin_q10"]),
        -float(row["rank"]),
        float(row["shrinkage"]),
    )


def fit_scenario_atomic_lowrank(
    support_by_scenario: Mapping[
        str, tuple[np.ndarray, Sequence[str], Sequence[str], Sequence[str]]
    ],
    *,
    rank_candidates: Sequence[int] = DEFAULT_RANK_CANDIDATES,
    shrinkage_candidates: Sequence[float] = DEFAULT_SHRINKAGE_CANDIDATES,
) -> ScenarioAtomicLowRankState:
    """Select one arm from support deletion evidence, then fit per scenario."""

    _validate_scenario_lineage(support_by_scenario)
    prepared: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    expected_classes: tuple[str, ...] | None = None
    feature_dim: int | None = None
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        features, labels, _ids, _hashes = support_by_scenario[scenario]
        rows = _validate_features(features, field=f"{scenario}.support_features")
        label_array = _validate_labels(
            labels, row_count=len(rows), field=f"{scenario}.support_labels"
        )
        classes = tuple(sorted(set(label_array.tolist())))
        if expected_classes is None:
            expected_classes = classes
            feature_dim = int(rows.shape[1])
        elif classes != expected_classes or rows.shape[1] != feature_dim:
            raise SupportLowRankMetricError(
                "D5 scenario class registry or feature dimension drift"
            )
        prepared[scenario] = rows, label_array

    candidate_rows: list[dict[str, Any]] = []
    first_rows, first_labels = prepared[FORMAL_LEO_WEAK_SCENARIOS[0]]
    maximum_rank = min(
        _selection_maximum_rank(labels, rows.shape[1])
        for rows, labels in prepared.values()
    )
    ranks = tuple(
        sorted(
            {
                int(rank)
                for rank in rank_candidates
                if 1 <= int(rank) <= maximum_rank
            }
        )
    )
    shrinkages = tuple(
        sorted(
            {
                float(value)
                for value in shrinkage_candidates
                if 0.0 <= float(value) <= 1.0
            }
        )
    )
    if not ranks or not shrinkages:
        raise SupportLowRankMetricError("D5 scenario selection grid is empty")
    for rank in ranks:
        for shrinkage in shrinkages:
            per_scenario = {
                scenario: _candidate_metrics(
                    rows,
                    labels,
                    rank=rank,
                    shrinkage=shrinkage,
                )
                for scenario, (rows, labels) in prepared.items()
            }
            candidate_rows.append(
                {
                    "rank": rank,
                    "shrinkage": shrinkage,
                    "selection_source": "registered_support_only",
                    "per_scenario": per_scenario,
                    "worst_scenario_min_class_accuracy": min(
                        row["min_class_accuracy"]
                        for row in per_scenario.values()
                    ),
                    "mean_scenario_min_class_accuracy": float(
                        np.mean(
                            [
                                row["min_class_accuracy"]
                                for row in per_scenario.values()
                            ]
                        )
                    ),
                    "mean_overall_accuracy": float(
                        np.mean(
                            [
                                row["overall_accuracy"]
                                for row in per_scenario.values()
                            ]
                        )
                    ),
                    "worst_scenario_margin_q10": min(
                        row["margin_q10"] for row in per_scenario.values()
                    ),
                }
            )
    selection_trace = tuple(candidate_rows)
    chosen = max(selection_trace, key=_scenario_candidate_key)
    rank = int(chosen["rank"])
    shrinkage = float(chosen["shrinkage"])
    states = tuple(
        _fit_locked_state(
            rows,
            labels,
            scenario=scenario,
            rank=rank,
            shrinkage=shrinkage,
            selection_trace=(),
            selection_protocol=(
                "three_scenario_registered_support_leave_one_or_two_out"
            ),
            support_fit_solve_count=sum(
                int(candidate["per_scenario"][scenario]["solve_count"])
                for candidate in selection_trace
            ),
        )
        for scenario, (rows, labels) in prepared.items()
    )
    trainable_parameters = sum(state.trainable_parameters for state in states)
    persistent_state_bytes = sum(state.persistent_state_bytes for state in states)
    if trainable_parameters > MAX_TRAINABLE_PARAMETERS:
        raise SupportLowRankMetricError(
            "D5 scenario-atomic trainable parameter cap exceeded"
        )
    if persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise SupportLowRankMetricError(
            "D5 scenario-atomic persistent state cap exceeded"
        )
    return ScenarioAtomicLowRankState(
        schema="cvs.phase2.d5_scenario_atomic_lowrank_state.v1",
        scenarios=FORMAL_LEO_WEAK_SCENARIOS,
        states=states,
        rank=rank,
        shrinkage=shrinkage,
        selection_trace=selection_trace,
        trainable_parameters=trainable_parameters,
        persistent_state_bytes=persistent_state_bytes,
    )


def register_scenario_atomic_absent_classes(
    base_state: ScenarioAtomicLowRankState,
    registered_support_by_scenario: Mapping[
        str, tuple[np.ndarray, Sequence[str], Sequence[str], Sequence[str]]
    ],
) -> ScenarioAtomicLowRankState:
    """Reuse each before state and add only absent registered-class prototypes."""

    if not isinstance(base_state, ScenarioAtomicLowRankState):
        raise SupportLowRankMetricError(
            "D5 scenario registration requires a before state"
        )
    _validate_scenario_lineage(registered_support_by_scenario)
    states = tuple(
        register_absent_classes(
            base_state.state_for(scenario),
            registered_support_by_scenario[scenario][0],
            registered_support_by_scenario[scenario][1],
        )
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    )
    class_sets = {state.classes for state in states}
    if len(class_sets) != 1:
        raise SupportLowRankMetricError(
            "D5 registered class set drifts across scenarios"
        )
    persistent_state_bytes = sum(state.persistent_state_bytes for state in states)
    if persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise SupportLowRankMetricError(
            "D5 scenario registered state cap exceeded"
        )
    return replace(
        base_state,
        states=states,
        persistent_state_bytes=persistent_state_bytes,
    )


def received_iq_sha256(rows: np.ndarray) -> tuple[str, ...]:
    """Hash already received float32 IQ rows for test/runtime lineage binding."""

    iq = np.asarray(rows)
    if (
        iq.dtype != np.float32
        or iq.ndim != 3
        or iq.shape[1] != 2
        or not np.isfinite(iq).all()
    ):
        raise SupportLowRankMetricError(
            "received IQ must be finite float32 [N,2,L]"
        )
    return tuple(
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in iq
    )


__all__ = [
    "DEFAULT_RANK_CANDIDATES",
    "DEFAULT_SHRINKAGE_CANDIDATES",
    "FORMAL_LEO_WEAK_SCENARIOS",
    "LowRankPrediction",
    "MAX_PERSISTENT_STATE_BYTES",
    "MAX_TRAINABLE_PARAMETERS",
    "ScenarioAtomicLowRankState",
    "SupportLowRankMetricError",
    "SupportLowRankMetricState",
    "fit_scenario_atomic_lowrank",
    "fit_support_lowrank_metric",
    "predict_all_registered",
    "received_iq_sha256",
    "register_absent_classes",
    "register_scenario_atomic_absent_classes",
]
