"""Support-only near-closed-form solver for the B3 compact diagonal head.

Each input row is one already received LEO_weak observation represented by the
locked 288-D concat feature.  The solver exposes no query-fitting entrypoint.
It estimates a diagonal Fisher ratio from old-class support, shrinks that
direction on a fixed method-locked grid, and selects the shrinkage using only
leave-one-support-out (LOSO) old-class evidence.  K=1 deterministically falls
back to the identity diagonal because no independent LOSO prototype exists.

The fitted surface is exactly one 288-D diagonal plus at most six 288-D class
weights (2,016 scalars).  There is no iterative optimizer state and no dense
support or query graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


FEATURE_DIM = 288
BLOCK_DIMS = (160, 96, 32)
MAX_OLD_CLASSES = 6
MAX_ACTIVE_SCALARS = FEATURE_DIM * (1 + MAX_OLD_CLASSES)
LOG_DIAG_LIMIT = math.log(1.5)
TEMPERATURE = 18.0
SCHEMA = "cvs.phase2.b3_fisher_closed_form.v1"


class B3FisherClosedFormError(ValueError):
    """Raised when the support-only B3 solver contract is violated."""


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
        raise B3FisherClosedFormError("zero-norm B3 feature or prototype")
    return np.asarray(values / norms, dtype=np.float32)


def _block_slices() -> tuple[slice, ...]:
    start = 0
    result: list[slice] = []
    for dimension in BLOCK_DIMS:
        result.append(slice(start, start + dimension))
        start += dimension
    return tuple(result)


def _project_log_diag(values: np.ndarray) -> np.ndarray:
    projected = np.asarray(values, dtype=np.float32).copy()
    for block in _block_slices():
        source = np.asarray(projected[block], dtype=np.float64)
        # Exact projection onto {z: sum(z)=0, |z_i|<=limit}.  Repeated
        # center/clip leaves a small block-mean residual when several entries
        # saturate; the monotone threshold solve removes that free scale.
        lower = float(np.min(source) - LOG_DIAG_LIMIT)
        upper = float(np.max(source) + LOG_DIAG_LIMIT)
        for _ in range(48):
            threshold = 0.5 * (lower + upper)
            total = float(
                np.sum(np.clip(source - threshold, -LOG_DIAG_LIMIT, LOG_DIAG_LIMIT))
            )
            if total > 0.0:
                lower = threshold
            else:
                upper = threshold
        chosen = np.clip(
            source - 0.5 * (lower + upper), -LOG_DIAG_LIMIT, LOG_DIAG_LIMIT
        ).astype(np.float32)
        # Correct the last FP32 rounding residual on a non-saturated entry.
        residual = float(np.sum(chosen, dtype=np.float64))
        free = np.flatnonzero(np.abs(chosen) < np.float32(LOG_DIAG_LIMIT - 1.0e-6))
        if len(free):
            chosen[int(free[0])] -= np.float32(residual)
        projected[block] = chosen
    return np.asarray(projected, dtype=np.float32)


def _validate_support(
    features: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray, int]:
    rows = np.asarray(features, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise B3FisherClosedFormError(
            f"support features must be finite [N,{FEATURE_DIM}]"
        )
    rows = _normalize(rows)
    label_values = np.asarray(tuple(str(value) for value in labels))
    if label_values.ndim != 1 or len(label_values) != len(rows):
        raise B3FisherClosedFormError("support labels do not match rows")
    registry = (
        tuple(str(value) for value in classes)
        if classes is not None
        else tuple(dict.fromkeys(label_values.tolist()))
    )
    if (
        len(registry) < 2
        or len(registry) > MAX_OLD_CLASSES
        or len(set(registry)) != len(registry)
        or set(label_values.tolist()) != set(registry)
    ):
        raise B3FisherClosedFormError("old-class registry drift, <2, or >6 classes")
    counts = [int(np.sum(label_values == value)) for value in registry]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise B3FisherClosedFormError("support must be class-symmetric K-shot")
    mapping = {value: index for index, value in enumerate(registry)}
    targets = np.asarray(
        [mapping[str(value)] for value in label_values.tolist()], dtype=np.int64
    )
    return rows, label_values, registry, targets, counts[0]


@dataclass(frozen=True)
class B3FisherClosedFormConfig:
    shrinkage_strengths: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    variance_ridge: float = 1.0e-4
    fisher_shrinkage: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shrinkage_strengths",
            tuple(float(value) for value in self.shrinkage_strengths),
        )
        if (
            not self.shrinkage_strengths
            or 0.0 not in self.shrinkage_strengths
            or len(set(self.shrinkage_strengths)) != len(self.shrinkage_strengths)
            or not all(0.0 <= value <= 1.0 for value in self.shrinkage_strengths)
            or not math.isfinite(float(self.variance_ridge))
            or float(self.variance_ridge) <= 0.0
            or not math.isfinite(float(self.fisher_shrinkage))
            or not 0.0 <= float(self.fisher_shrinkage) <= 1.0
        ):
            raise B3FisherClosedFormError("invalid fixed B3 Fisher solver lock")


@dataclass(frozen=True)
class B3FisherClosedFormState:
    schema: str
    classes: tuple[str, ...]
    log_diag: np.ndarray
    weights: np.ndarray
    support_count_by_class: np.ndarray
    selected_strength: float
    active_scalars: int
    optimizer_steps: int

    def __post_init__(self) -> None:
        class_count = len(self.classes)
        log_diag = np.asarray(self.log_diag)
        weights = np.asarray(self.weights)
        counts = np.asarray(self.support_count_by_class)
        expected = FEATURE_DIM * (1 + class_count)
        if (
            self.schema != SCHEMA
            or not 2 <= class_count <= MAX_OLD_CLASSES
            or len(set(self.classes)) != class_count
            or log_diag.dtype != np.float32
            or log_diag.shape != (FEATURE_DIM,)
            or weights.dtype != np.float32
            or weights.shape != (class_count, FEATURE_DIM)
            or counts.dtype != np.uint16
            or counts.shape != (class_count,)
            or bool(np.any(counts < 1))
            or not np.isfinite(log_diag).all()
            or not np.isfinite(weights).all()
            or float(np.max(np.abs(log_diag))) > LOG_DIAG_LIMIT + 1.0e-6
            or int(self.active_scalars) != expected
            or expected > MAX_ACTIVE_SCALARS
            or int(self.optimizer_steps) != 0
        ):
            raise B3FisherClosedFormError("B3 Fisher state drift")
        object.__setattr__(self, "log_diag", _readonly(log_diag, np.float32))
        object.__setattr__(self, "weights", _readonly(weights, np.float32))
        object.__setattr__(self, "support_count_by_class", _readonly(counts, np.uint16))


@dataclass(frozen=True)
class B3FisherClosedFormResult:
    state: B3FisherClosedFormState
    solver_trace: tuple[dict[str, Any], ...]
    resource_audit: dict[str, Any]


def _loso_evidence(
    rows: np.ndarray,
    targets: np.ndarray,
    class_count: int,
    log_diag: np.ndarray,
) -> dict[str, Any]:
    scaled = _normalize(rows * np.exp(log_diag, dtype=np.float32)[None, :])
    sums = np.stack(
        [np.sum(scaled[targets == index], axis=0) for index in range(class_count)]
    ).astype(np.float32)
    counts = np.bincount(targets, minlength=class_count).astype(np.int64)
    logits = np.empty((len(rows), class_count), dtype=np.float32)
    for row_index in range(len(rows)):
        prototype_sums = sums.copy()
        prototype_counts = counts.copy()
        truth = int(targets[row_index])
        prototype_sums[truth] -= scaled[row_index]
        prototype_counts[truth] -= 1
        prototypes = _normalize(
            prototype_sums / prototype_counts[:, None].astype(np.float32)
        )
        logits[row_index] = np.float32(TEMPERATURE) * (
            scaled[row_index] @ prototypes.T
        )
    predictions = np.argmax(logits, axis=1)
    correct = predictions == targets
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1))
    ce = -np.mean(shifted[np.arange(len(rows)), targets] - logsumexp)
    masked = logits.copy()
    masked[np.arange(len(rows)), targets] = -np.inf
    margins = logits[np.arange(len(rows)), targets] - np.max(masked, axis=1)
    per_class_acc = [
        float(np.mean(correct[targets == index])) for index in range(class_count)
    ]
    per_class_margin = [
        float(np.mean(margins[targets == index])) for index in range(class_count)
    ]
    return {
        "loso_accuracy": float(np.mean(correct)),
        "loso_class_floor": float(min(per_class_acc)),
        "loso_ce": float(ce),
        "mean_true_margin": float(np.mean(margins)),
        "min_class_mean_true_margin": float(min(per_class_margin)),
    }


def fit_b3_fisher_closed_form(
    support_features: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str] | None = None,
    *,
    config: B3FisherClosedFormConfig | None = None,
) -> B3FisherClosedFormResult:
    """Fit B3 from LEO_weak old support without iterative optimization."""

    locked = config or B3FisherClosedFormConfig()
    rows, _, classes, targets, k_shot = _validate_support(
        support_features, support_labels, registered_classes
    )
    class_count = len(classes)
    means = np.stack(
        [np.mean(rows[targets == index], axis=0) for index in range(class_count)]
    ).astype(np.float32)
    global_mean = np.mean(rows, axis=0, dtype=np.float32)
    between = np.mean((means - global_mean[None, :]) ** 2, axis=0, dtype=np.float32)
    within = np.mean((rows - means[targets]) ** 2, axis=0, dtype=np.float32)
    ridge = np.float32(locked.variance_ridge)
    raw_direction = np.float32(0.5) * np.log(
        (between + ridge) / (within + ridge)
    ).astype(np.float32)
    raw_direction *= np.float32(locked.fisher_shrinkage)
    raw_direction = _project_log_diag(raw_direction)

    trace: list[dict[str, Any]] = []
    if k_shot == 1:
        selected_strength = 0.0
        selected_log_diag = np.zeros(FEATURE_DIM, dtype=np.float32)
        trace.append(
            {
                "solver": "k1_identity_no_loso",
                "strength": 0.0,
                "selected": True,
                "optimizer_steps": 0,
                "query_rows_used": 0,
            }
        )
    else:
        candidates: list[tuple[tuple[float, ...], float, np.ndarray, dict[str, Any]]] = []
        for strength in locked.shrinkage_strengths:
            candidate_diag = _project_log_diag(raw_direction * np.float32(strength))
            evidence = _loso_evidence(rows, targets, class_count, candidate_diag)
            # Stronger geometry must earn improvements; exact ties prefer identity.
            rank = (
                evidence["loso_class_floor"],
                evidence["loso_accuracy"],
                evidence["min_class_mean_true_margin"],
                evidence["mean_true_margin"],
                -evidence["loso_ce"],
                -float(strength),
            )
            candidates.append((rank, float(strength), candidate_diag, evidence))
        chosen = max(candidates, key=lambda item: item[0])
        selected_strength = chosen[1]
        selected_log_diag = chosen[2]
        for _, strength, _, evidence in candidates:
            trace.append(
                {
                    "solver": "diagonal_fisher_closed_form_fixed_strength_loso",
                    "strength": strength,
                    "selected": strength == selected_strength,
                    "optimizer_steps": 0,
                    "query_rows_used": 0,
                    **evidence,
                }
            )

    scaled = _normalize(rows * np.exp(selected_log_diag, dtype=np.float32)[None, :])
    weights = np.stack(
        [
            _normalize(np.mean(scaled[targets == index], axis=0, keepdims=True))[0]
            for index in range(class_count)
        ]
    ).astype(np.float32)
    active_scalars = FEATURE_DIM * (1 + class_count)
    state = B3FisherClosedFormState(
        schema=SCHEMA,
        classes=classes,
        log_diag=np.asarray(selected_log_diag, dtype=np.float32),
        weights=weights,
        support_count_by_class=np.full(class_count, k_shot, dtype=np.uint16),
        selected_strength=selected_strength,
        active_scalars=active_scalars,
        optimizer_steps=0,
    )
    row_count = len(rows)
    strength_count = 0 if k_shot == 1 else len(locked.shrinkage_strengths)
    moment_macs = 2 * row_count * FEATURE_DIM + class_count * FEATURE_DIM
    loso_macs = strength_count * (
        3 * row_count * FEATURE_DIM
        + row_count * class_count * FEATURE_DIM
    )
    finalization_macs = 3 * row_count * FEATURE_DIM
    estimated_macs = int(moment_macs + loso_macs + finalization_macs)
    adam15_reference = int(
        3 * 15 * row_count * FEATURE_DIM * (1 + class_count)
    )
    audit: dict[str, Any] = {
        "schema": "cvs.phase2.b3_fisher_closed_form_resource.v1",
        "adaptation_mode": "EVAL_ONLY_CLOSED_FORM_ADAPTATION",
        "optimizer_steps": 0,
        "closed_form_solve_count": 1,
        "active_scalars": active_scalars,
        "active_scalar_cap": MAX_ACTIVE_SCALARS,
        "active_scalar_cap_pass": active_scalars <= MAX_ACTIVE_SCALARS,
        "estimated_adaptation_macs": estimated_macs,
        "estimated_adam15_reference_macs": adam15_reference,
        "estimated_macs_reduction_fraction_vs_adam15": float(
            1.0 - estimated_macs / adam15_reference
        ),
        "estimated_macs_per_query": FEATURE_DIM * (1 + class_count),
        "persistent_state_bytes": int(state.log_diag.nbytes + state.weights.nbytes),
        "dense_query_graph_bytes": 0,
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
    return B3FisherClosedFormResult(
        state=state,
        solver_trace=tuple(trace),
        resource_audit=audit,
    )


def score_b3_fisher_closed_form(
    state: B3FisherClosedFormState, features: np.ndarray
) -> np.ndarray:
    """Score independent rows; this API cannot fit or select state."""

    rows = np.asarray(features, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != FEATURE_DIM or not np.isfinite(rows).all():
        raise B3FisherClosedFormError("scoring features must be finite [N,288]")
    scaled = _normalize(rows * np.exp(state.log_diag, dtype=np.float32)[None, :])
    scores = np.float32(TEMPERATURE) * (scaled @ state.weights.T)
    return _readonly(scores, np.float32)


__all__ = [
    "B3FisherClosedFormConfig",
    "B3FisherClosedFormError",
    "B3FisherClosedFormResult",
    "B3FisherClosedFormState",
    "MAX_ACTIVE_SCALARS",
    "fit_b3_fisher_closed_form",
    "score_b3_fisher_closed_form",
]
