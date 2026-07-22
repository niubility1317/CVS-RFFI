"""D6b identity-primary support-only low-rank residual metric.

Identity cosine is always retained.  A support-selected low-rank branch may
contribute only a bounded score residual.  The module has no clean/source,
query-label, role, quota, scorer, or batch-global assignment interface.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_support_lowrank_metric import (
    DEFAULT_RANK_CANDIDATES,
    DEFAULT_SHRINKAGE_CANDIDATES,
    FORMAL_LEO_WEAK_SCENARIOS,
    MAX_PERSISTENT_STATE_BYTES,
    MAX_TRAINABLE_PARAMETERS,
    SupportLowRankMetricError,
    SupportLowRankMetricState,
    fit_support_lowrank_metric,
    predict_all_registered as predict_lowrank_all_registered,
    register_absent_classes as register_lowrank_absent_classes,
)


EPS = 1.0e-8
DEFAULT_RESIDUAL_WEIGHT_CANDIDATES = (0.0, 0.1, 0.2, 0.3)
DEFAULT_OVERALL_DEGRADATION_TOLERANCE = 0.01


class IdentityLowRankResidualError(ValueError):
    """Raised when the D6b support-only residual route fails closed."""


@dataclass(frozen=True)
class IdentityLowRankResidualState:
    """One scenario-specific identity-primary classifier."""

    schema: str
    scenario: str
    feature_dim: int
    classes: tuple[str, ...]
    identity_prototypes: np.ndarray
    lowrank_state: SupportLowRankMetricState
    residual_weight: float
    selection_trace: tuple[dict[str, Any], ...]
    registration_generation: int
    trainable_parameters: int
    persistent_state_bytes: int
    query_rows_used_for_fit: int = 0
    query_updates: int = 0

    def resource_audit(self) -> dict[str, Any]:
        class_count = len(self.classes)
        return {
            "schema": "cvs.phase2.d6b_identity_lowrank_resource.v1",
            "candidate": "d6b_identity_primary_bounded_lowrank_residual",
            "scenario": self.scenario,
            "identity_primary": True,
            "pure_lowrank_replacement": False,
            "residual_weight": self.residual_weight,
            "residual_weight_upper_bound": max(
                DEFAULT_RESIDUAL_WEIGHT_CANDIDATES
            ),
            "support_only": True,
            "adaptation_epochs": 0,
            "query_rows_used_for_fit": self.query_rows_used_for_fit,
            "query_updates": self.query_updates,
            "query_decision_policy": "per_sample_all_registered_classes",
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
            "identity_macs_per_query": self.feature_dim * class_count,
            "lowrank_macs_per_query": (
                self.feature_dim * self.lowrank_state.rank
                + self.lowrank_state.rank * class_count
            ),
            "old_identity_head_reused_after_registration": (
                self.registration_generation > 0
            ),
            "old_lowrank_state_reused_after_registration": (
                self.registration_generation > 0
            ),
        }


@dataclass(frozen=True)
class ScenarioAtomicIdentityResidualState:
    schema: str
    scenarios: tuple[str, ...]
    states: tuple[IdentityLowRankResidualState, ...]
    rank: int
    shrinkage: float
    residual_weight: float
    selection_trace: tuple[dict[str, Any], ...]
    trainable_parameters: int
    persistent_state_bytes: int

    def state_for(self, scenario: str) -> IdentityLowRankResidualState:
        try:
            return self.states[self.scenarios.index(str(scenario))]
        except ValueError as error:
            raise IdentityLowRankResidualError(
                "D6b scenario state lookup drift"
            ) from error

    def resource_audit(self) -> dict[str, Any]:
        return {
            "schema": "cvs.phase2.d6b_scenario_atomic_resource.v1",
            "candidate": "d6b_identity_primary_bounded_lowrank_residual",
            "scenario_atomic": True,
            "cross_scenario_support_concat": False,
            "identity_primary": True,
            "pure_lowrank_replacement": False,
            "rank": self.rank,
            "shrinkage": self.shrinkage,
            "residual_weight": self.residual_weight,
            "support_only": True,
            "adaptation_epochs": 0,
            "query_rows_used_for_fit": 0,
            "query_updates": 0,
            "role_oracle_access": False,
            "class_quota_access": False,
            "batch_global_assignment": False,
            "dense_query_graph_bytes": 0,
            "trainable_parameters": self.trainable_parameters,
            "trainable_parameter_limit_pass": (
                self.trainable_parameters <= MAX_TRAINABLE_PARAMETERS
            ),
            "persistent_state_bytes": self.persistent_state_bytes,
            "persistent_state_limit_pass": (
                self.persistent_state_bytes <= MAX_PERSISTENT_STATE_BYTES
            ),
            "per_scenario": {
                scenario: state.resource_audit()
                for scenario, state in zip(self.scenarios, self.states)
            },
        }


@dataclass(frozen=True)
class IdentityResidualPrediction:
    labels: tuple[str, ...]
    scores: np.ndarray
    identity_scores: np.ndarray
    lowrank_scores: np.ndarray


def _readonly_float32(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float32)
    array.setflags(write=False)
    return array


def _validate_features(value: np.ndarray, *, field: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[0] < 1
        or rows.shape[1] < 1
        or not np.isfinite(rows).all()
    ):
        raise IdentityLowRankResidualError(
            f"{field} must be a finite float32 matrix [N,D]"
        )
    return np.ascontiguousarray(rows)


def _validate_labels(
    values: Sequence[str], *, row_count: int, field: str
) -> np.ndarray:
    labels = np.asarray(tuple(str(value) for value in values))
    if len(labels) != row_count or any(not label for label in labels):
        raise IdentityLowRankResidualError(f"{field} alignment drift")
    return labels


def _normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), EPS
    )


def _identity_prototypes(
    rows: np.ndarray, labels: np.ndarray, classes: tuple[str, ...]
) -> np.ndarray:
    normalized = _normalize(rows)
    prototypes = np.stack(
        [
            _normalize(
                normalized[labels == label].mean(axis=0, keepdims=True)
            )[0]
            for label in classes
        ],
        axis=0,
    )
    return _readonly_float32(prototypes)


def _folds(labels: np.ndarray) -> tuple[tuple[int, ...], ...]:
    classes = tuple(sorted(set(labels.tolist())))
    counts = [int(np.sum(labels == label)) for label in classes]
    if min(counts) < 2:
        raise IdentityLowRankResidualError(
            "D6b support selection requires at least two samples per class"
        )
    width = 2 if min(counts) >= 4 else 1
    indices = [np.flatnonzero(labels == label).tolist() for label in classes]
    return tuple(
        tuple(
            sorted(
                index
                for class_indices in indices
                for index in class_indices[offset : offset + width]
            )
        )
        for offset in range(0, max(counts), width)
        if any(class_indices[offset : offset + width] for class_indices in indices)
    )


def _metrics(
    predicted: np.ndarray, truth: np.ndarray, margins: np.ndarray
) -> dict[str, Any]:
    classes = tuple(sorted(set(truth.tolist())))
    per_class = {
        label: float(np.mean(predicted[truth == label] == label))
        for label in classes
    }
    return {
        "overall_accuracy": float(np.mean(predicted == truth)),
        "min_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
        "margin_q10": float(np.quantile(margins, 0.10)),
        "mean_margin": float(np.mean(margins)),
    }


def _fold_predictions(
    rows: np.ndarray,
    labels: np.ndarray,
    *,
    rank: int,
    shrinkage: float,
    residual_weight: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity_predicted: list[str] = []
    residual_predicted: list[str] = []
    truth: list[str] = []
    identity_margins: list[float] = []
    residual_margins: list[float] = []
    for held_tuple in _folds(labels):
        held = np.asarray(held_tuple, dtype=np.int64)
        keep = np.ones(len(rows), dtype=bool)
        keep[held] = False
        train_x, train_y = rows[keep], labels[keep]
        classes = tuple(sorted(set(train_y.tolist())))
        identity_proto = _identity_prototypes(train_x, train_y, classes)
        lowrank = fit_support_lowrank_metric(
            train_x,
            train_y,
            scenario="support_deletion_fold",
            locked_rank=rank,
            locked_shrinkage=shrinkage,
        )
        identity_score = _normalize(rows[held]) @ identity_proto.T
        lowrank_score = predict_lowrank_all_registered(
            lowrank, rows[held]
        ).scores
        residual_score = identity_score + float(residual_weight) * (
            lowrank_score - identity_score
        )
        for score, predictions, margins in (
            (identity_score, identity_predicted, identity_margins),
            (residual_score, residual_predicted, residual_margins),
        ):
            order = np.argsort(score, axis=1)
            top = order[:, -1]
            second = order[:, -2]
            predictions.extend(classes[int(index)] for index in top)
            margins.extend(
                (
                    score[np.arange(len(held)), top]
                    - score[np.arange(len(held)), second]
                ).astype(float)
            )
        truth.extend(labels[held].tolist())
    truth_array = np.asarray(truth)
    return (
        _metrics(
            np.asarray(identity_predicted),
            truth_array,
            np.asarray(identity_margins),
        ),
        _metrics(
            np.asarray(residual_predicted),
            truth_array,
            np.asarray(residual_margins),
        ),
    )


def _validate_lineage(
    support_by_scenario: Mapping[
        str, tuple[np.ndarray, Sequence[str], Sequence[str], Sequence[str]]
    ]
) -> None:
    if set(support_by_scenario) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise IdentityLowRankResidualError("D6b formal scenario coverage drift")
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        rows, labels, physical_ids, received_hashes = support_by_scenario[scenario]
        row_count = len(np.asarray(rows))
        _validate_labels(
            labels, row_count=row_count, field=f"{scenario}.support_labels"
        )
        ids = tuple(str(value) for value in physical_ids)
        hashes = tuple(str(value).lower() for value in received_hashes)
        if (
            len(ids) != row_count
            or len(hashes) != row_count
            or len(set(ids)) != row_count
            or len(set(hashes)) != row_count
            or any(not value for value in ids)
            or any(
                len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
                for value in hashes
            )
        ):
            raise IdentityLowRankResidualError(
                "D6b single-observation lineage drift"
            )
        if seen_ids.intersection(ids) or seen_hashes.intersection(hashes):
            raise IdentityLowRankResidualError(
                "D6b cross-scenario support reuse is forbidden"
            )
        seen_ids.update(ids)
        seen_hashes.update(hashes)


def _state_bytes(
    identity_prototypes: np.ndarray,
    lowrank_state: SupportLowRankMetricState,
) -> int:
    return int(
        identity_prototypes.nbytes + lowrank_state.persistent_state_bytes
    )


def _fit_state(
    rows: np.ndarray,
    labels: np.ndarray,
    *,
    scenario: str,
    rank: int,
    shrinkage: float,
    residual_weight: float,
    selection_trace: tuple[dict[str, Any], ...],
) -> IdentityLowRankResidualState:
    classes = tuple(sorted(set(labels.tolist())))
    identity = _identity_prototypes(rows, labels, classes)
    lowrank = fit_support_lowrank_metric(
        rows,
        labels,
        scenario=scenario,
        locked_rank=rank,
        locked_shrinkage=shrinkage,
    )
    if lowrank.classes != classes:
        raise IdentityLowRankResidualError("D6b class registry drift")
    trainable_parameters = lowrank.trainable_parameters
    persistent_state_bytes = _state_bytes(identity, lowrank)
    if trainable_parameters > MAX_TRAINABLE_PARAMETERS:
        raise IdentityLowRankResidualError("D6b parameter cap exceeded")
    if persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise IdentityLowRankResidualError("D6b state cap exceeded")
    return IdentityLowRankResidualState(
        schema="cvs.phase2.d6b_identity_lowrank_residual_state.v1",
        scenario=scenario,
        feature_dim=rows.shape[1],
        classes=classes,
        identity_prototypes=identity,
        lowrank_state=lowrank,
        residual_weight=float(residual_weight),
        selection_trace=selection_trace,
        registration_generation=0,
        trainable_parameters=trainable_parameters,
        persistent_state_bytes=persistent_state_bytes,
    )


def _selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["eligible"]),
        float(row["worst_scenario_min_class_accuracy"]),
        float(row["mean_overall_accuracy"]),
        float(row["worst_scenario_margin_q10"]),
        float(row["residual_weight"]),
        -float(row["rank"]),
        float(row["shrinkage"]),
    )


def fit_scenario_atomic_identity_residual(
    support_by_scenario: Mapping[
        str, tuple[np.ndarray, Sequence[str], Sequence[str], Sequence[str]]
    ],
    *,
    rank_candidates: Sequence[int] = DEFAULT_RANK_CANDIDATES,
    shrinkage_candidates: Sequence[float] = DEFAULT_SHRINKAGE_CANDIDATES,
    residual_weight_candidates: Sequence[float] = (
        DEFAULT_RESIDUAL_WEIGHT_CANDIDATES
    ),
    overall_degradation_tolerance: float = (
        DEFAULT_OVERALL_DEGRADATION_TOLERANCE
    ),
) -> ScenarioAtomicIdentityResidualState:
    """Select one unified arm using support deletion non-degradation gates."""

    _validate_lineage(support_by_scenario)
    prepared: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    classes: tuple[str, ...] | None = None
    dimension: int | None = None
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        features, labels, _ids, _hashes = support_by_scenario[scenario]
        rows = _validate_features(
            features, field=f"{scenario}.support_features"
        )
        label_array = _validate_labels(
            labels, row_count=len(rows), field=f"{scenario}.support_labels"
        )
        current_classes = tuple(sorted(set(label_array.tolist())))
        if classes is None:
            classes, dimension = current_classes, rows.shape[1]
        elif current_classes != classes or rows.shape[1] != dimension:
            raise IdentityLowRankResidualError(
                "D6b scenario registry or dimension drift"
            )
        prepared[scenario] = rows, label_array

    weights = tuple(
        sorted(
            {
                float(value)
                for value in residual_weight_candidates
                if 0.0 <= float(value) <= 0.3
            }
        )
    )
    if 0.0 not in weights:
        raise IdentityLowRankResidualError(
            "D6b residual grid must include identity fallback weight 0"
        )
    trace: list[dict[str, Any]] = []
    for rank in sorted(set(int(value) for value in rank_candidates)):
        for shrinkage in sorted(
            set(float(value) for value in shrinkage_candidates)
        ):
            for weight in weights:
                per_scenario: dict[str, Any] = {}
                feasible = True
                for scenario, (rows, labels) in prepared.items():
                    try:
                        baseline, residual = _fold_predictions(
                            rows,
                            labels,
                            rank=rank,
                            shrinkage=shrinkage,
                            residual_weight=weight,
                        )
                    except (
                        SupportLowRankMetricError,
                        IdentityLowRankResidualError,
                    ):
                        feasible = False
                        break
                    floor_pass = (
                        residual["min_class_accuracy"]
                        >= baseline["min_class_accuracy"]
                    )
                    overall_pass = (
                        residual["overall_accuracy"]
                        + float(overall_degradation_tolerance)
                        >= baseline["overall_accuracy"]
                    )
                    per_scenario[scenario] = {
                        "identity_baseline": baseline,
                        "residual": residual,
                        "floor_non_degradation_pass": floor_pass,
                        "overall_tolerance_pass": overall_pass,
                    }
                if not feasible:
                    continue
                eligible = all(
                    row["floor_non_degradation_pass"]
                    and row["overall_tolerance_pass"]
                    for row in per_scenario.values()
                )
                trace.append(
                    {
                        "rank": rank,
                        "shrinkage": shrinkage,
                        "residual_weight": weight,
                        "eligible": eligible,
                        "selection_source": (
                            "registered_support_leave_one_or_two_out_only"
                        ),
                        "per_scenario": per_scenario,
                        "worst_scenario_min_class_accuracy": min(
                            row["residual"]["min_class_accuracy"]
                            for row in per_scenario.values()
                        ),
                        "mean_overall_accuracy": float(
                            np.mean(
                                [
                                    row["residual"]["overall_accuracy"]
                                    for row in per_scenario.values()
                                ]
                            )
                        ),
                        "worst_scenario_margin_q10": min(
                            row["residual"]["margin_q10"]
                            for row in per_scenario.values()
                        ),
                    }
                )
    if not trace:
        raise IdentityLowRankResidualError("D6b support selection grid is empty")
    eligible_rows = [row for row in trace if row["eligible"]]
    if not eligible_rows:
        raise IdentityLowRankResidualError(
            "D6b identity fallback unexpectedly failed support gates"
        )
    chosen = max(eligible_rows, key=_selection_key)
    rank = int(chosen["rank"])
    shrinkage = float(chosen["shrinkage"])
    weight = float(chosen["residual_weight"])
    states = tuple(
        _fit_state(
            rows,
            labels,
            scenario=scenario,
            rank=rank,
            shrinkage=shrinkage,
            residual_weight=weight,
            selection_trace=(),
        )
        for scenario, (rows, labels) in prepared.items()
    )
    trainable_parameters = sum(state.trainable_parameters for state in states)
    persistent_state_bytes = sum(state.persistent_state_bytes for state in states)
    if trainable_parameters > MAX_TRAINABLE_PARAMETERS:
        raise IdentityLowRankResidualError(
            "D6b scenario-atomic parameter cap exceeded"
        )
    if persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise IdentityLowRankResidualError(
            "D6b scenario-atomic state cap exceeded"
        )
    return ScenarioAtomicIdentityResidualState(
        schema="cvs.phase2.d6b_scenario_atomic_state.v1",
        scenarios=FORMAL_LEO_WEAK_SCENARIOS,
        states=states,
        rank=rank,
        shrinkage=shrinkage,
        residual_weight=weight,
        selection_trace=tuple(trace),
        trainable_parameters=trainable_parameters,
        persistent_state_bytes=persistent_state_bytes,
    )


def register_absent_classes(
    base_state: IdentityLowRankResidualState,
    registered_support_features: np.ndarray,
    registered_support_labels: Sequence[str],
) -> IdentityLowRankResidualState:
    """Add absent registered labels without changing before identity/metric."""

    rows = _validate_features(
        registered_support_features, field="registered_support_features"
    )
    labels = _validate_labels(
        registered_support_labels,
        row_count=len(rows),
        field="registered_support_labels",
    )
    absent = tuple(sorted(set(labels.tolist()) - set(base_state.classes)))
    if not absent:
        raise IdentityLowRankResidualError(
            "D6b registration received no absent labels"
        )
    mask = np.isin(labels, np.asarray(absent))
    new_identity = _identity_prototypes(rows[mask], labels[mask], absent)
    lowrank = register_lowrank_absent_classes(
        base_state.lowrank_state, rows, labels
    )
    identity = _readonly_float32(
        np.concatenate([base_state.identity_prototypes, new_identity], axis=0)
    )
    classes = base_state.classes + absent
    if lowrank.classes != classes:
        raise IdentityLowRankResidualError(
            "D6b identity/low-rank registry drift after registration"
        )
    persistent_state_bytes = _state_bytes(identity, lowrank)
    if persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise IdentityLowRankResidualError(
            "D6b registered state cap exceeded"
        )
    return replace(
        base_state,
        classes=classes,
        identity_prototypes=identity,
        lowrank_state=lowrank,
        registration_generation=base_state.registration_generation + 1,
        persistent_state_bytes=persistent_state_bytes,
    )


def register_scenario_atomic_absent_classes(
    base_state: ScenarioAtomicIdentityResidualState,
    registered_support_by_scenario: Mapping[
        str, tuple[np.ndarray, Sequence[str], Sequence[str], Sequence[str]]
    ],
) -> ScenarioAtomicIdentityResidualState:
    _validate_lineage(registered_support_by_scenario)
    states = tuple(
        register_absent_classes(
            base_state.state_for(scenario),
            registered_support_by_scenario[scenario][0],
            registered_support_by_scenario[scenario][1],
        )
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    )
    persistent_state_bytes = sum(state.persistent_state_bytes for state in states)
    if persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise IdentityLowRankResidualError(
            "D6b registered scenario state cap exceeded"
        )
    return replace(
        base_state,
        states=states,
        persistent_state_bytes=persistent_state_bytes,
    )


def predict_all_registered(
    state: IdentityLowRankResidualState, query_features: np.ndarray
) -> IdentityResidualPrediction:
    """Per-query all-registry prediction with immutable support state."""

    rows = _validate_features(query_features, field="query_features")
    if rows.shape[1] != state.feature_dim:
        raise IdentityLowRankResidualError("D6b query dimension drift")
    identity_scores = _readonly_float32(
        _normalize(rows) @ state.identity_prototypes.T
    )
    lowrank_scores = predict_lowrank_all_registered(
        state.lowrank_state, rows
    ).scores
    scores = _readonly_float32(
        identity_scores
        + state.residual_weight * (lowrank_scores - identity_scores)
    )
    indices = np.argmax(scores, axis=1)
    return IdentityResidualPrediction(
        labels=tuple(state.classes[int(index)] for index in indices),
        scores=scores,
        identity_scores=identity_scores,
        lowrank_scores=lowrank_scores,
    )


__all__ = [
    "DEFAULT_OVERALL_DEGRADATION_TOLERANCE",
    "DEFAULT_RESIDUAL_WEIGHT_CANDIDATES",
    "IdentityLowRankResidualError",
    "IdentityLowRankResidualState",
    "IdentityResidualPrediction",
    "ScenarioAtomicIdentityResidualState",
    "fit_scenario_atomic_identity_residual",
    "predict_all_registered",
    "register_absent_classes",
    "register_scenario_atomic_absent_classes",
]
