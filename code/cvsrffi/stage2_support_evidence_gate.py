"""D28 support-only evidence gate over D27 all-class score rows.

The fit boundary accepts registered support labels and explicit within-class
shot ranks.  It never accepts query labels, old/new query roles, quotas, or
batch statistics.  Inference extracts five row-local score contrasts and adds
one bounded scalar to every registered new-class score.  Old scores remain
bitwise unchanged and new-vs-new ordering is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Sequence

import numpy as np


EVIDENCE_DIM = 5
RIDGE_LAMBDA_GRID = (0.1, 1.0, 10.0)
OOF_FOLD_COUNT = 5
MIN_EFFECTIVE_STD = 1.0e-4
MAX_RIDGE_CONDITION_NUMBER = 1.0e6
MAX_COEFFICIENT_L2_NORM = 8.0
MAX_GATE_STATE_BYTES = 256 * 1024
SCHEMA = "cvs.phase2.d28_support_evidence_gate.v1"


class SupportEvidenceGateError(ValueError):
    """Raised when D28 support, state, or row-local inference drifts."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _validate_scores(
    raw_scores: np.ndarray,
    *,
    class_count: int | None = None,
    name: str,
) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=np.float32)
    if (
        scores.ndim != 2
        or len(scores) < 1
        or scores.shape[1] < 2
        or not np.isfinite(scores).all()
        or (class_count is not None and scores.shape[1] != class_count)
    ):
        expected = "[N,C]" if class_count is None else f"[N,{class_count}]"
        raise SupportEvidenceGateError(f"{name} must be finite {expected}")
    return np.ascontiguousarray(scores, dtype=np.float32)


@dataclass(frozen=True)
class SupportEvidenceGateConfig:
    ridge_lambdas: tuple[float, ...] = RIDGE_LAMBDA_GRID
    alpha: float = 1.0
    delta: float = 2.0
    oof_fold_count: int = OOF_FOLD_COUNT

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ridge_lambdas", tuple(float(value) for value in self.ridge_lambdas)
        )
        self.validate()

    def validate(self) -> None:
        if self.ridge_lambdas != RIDGE_LAMBDA_GRID:
            raise SupportEvidenceGateError(
                "D28 ridge lambda grid must remain preregistered as 0.1/1/10"
            )
        if int(self.oof_fold_count) != OOF_FOLD_COUNT:
            raise SupportEvidenceGateError("D28 must use shot-rank 5-fold OOF")
        if not math.isfinite(float(self.alpha)) or float(self.alpha) < 0.0:
            raise SupportEvidenceGateError("D28 alpha must be finite and non-negative")
        if not math.isfinite(float(self.delta)) or float(self.delta) <= 0.0:
            raise SupportEvidenceGateError("D28 delta must be finite and positive")


@dataclass(frozen=True)
class SupportEvidenceGateState:
    schema: str
    registered_classes: tuple[str, ...]
    old_class_count: int
    k_shot: int
    enabled: bool
    selected_lambda: float | None
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    audit_json: str
    config: SupportEvidenceGateConfig

    def __post_init__(self) -> None:
        self.config.validate()
        classes = tuple(str(value) for value in self.registered_classes)
        old_count = int(self.old_class_count)
        k_shot = int(self.k_shot)
        coefficients = np.asarray(self.coefficients)
        feature_mean = np.asarray(self.feature_mean)
        feature_std = np.asarray(self.feature_std)
        if (
            self.schema != SCHEMA
            or len(classes) < 6
            or len(set(classes)) != len(classes)
            or not 3 <= old_count <= len(classes) - 3
            or k_shot < 1
            or coefficients.dtype != np.float32
            or coefficients.ndim != 1
            or not np.isfinite(coefficients).all()
            or feature_mean.dtype != np.float32
            or feature_std.dtype != np.float32
            or feature_mean.ndim != 1
            or feature_std.ndim != 1
            or not np.isfinite(feature_mean).all()
            or not np.isfinite(feature_std).all()
        ):
            raise SupportEvidenceGateError("D28 state drift")
        if self.enabled:
            if (
                k_shot < OOF_FOLD_COUNT
                or self.selected_lambda not in self.config.ridge_lambdas
                or coefficients.shape != (EVIDENCE_DIM + 1,)
                or feature_mean.shape != (EVIDENCE_DIM,)
                or feature_std.shape != (EVIDENCE_DIM,)
                or bool(np.any(feature_std < np.float32(MIN_EFFECTIVE_STD)))
            ):
                raise SupportEvidenceGateError("D28 enabled state drift")
        elif (
            k_shot not in (1,) and k_shot < OOF_FOLD_COUNT
        ):
            raise SupportEvidenceGateError("D28 disabled state has invalid K")
        elif (
            self.selected_lambda is not None
            or coefficients.shape != (0,)
            or feature_mean.shape != (0,)
            or feature_std.shape != (0,)
        ):
            raise SupportEvidenceGateError("D28 disabled state must be exact passthrough")
        try:
            audit = json.loads(str(self.audit_json))
        except json.JSONDecodeError as exc:
            raise SupportEvidenceGateError("D28 audit is invalid JSON") from exc
        if not isinstance(audit, dict):
            raise SupportEvidenceGateError("D28 audit must be one JSON object")
        object.__setattr__(self, "registered_classes", classes)
        object.__setattr__(self, "old_class_count", old_count)
        object.__setattr__(self, "k_shot", k_shot)
        object.__setattr__(self, "coefficients", _readonly(coefficients, np.float32))
        object.__setattr__(self, "feature_mean", _readonly(feature_mean, np.float32))
        object.__setattr__(self, "feature_std", _readonly(feature_std, np.float32))
        object.__setattr__(
            self, "audit_json", _canonical_json_bytes(audit).decode("utf-8")
        )
        if self.persistent_state_bytes > MAX_GATE_STATE_BYTES:
            raise SupportEvidenceGateError("D28 persistent state exceeds 256KB")

    @property
    def persistent_state_bytes(self) -> int:
        metadata = (
            len(self.schema.encode("utf-8"))
            + sum(len(value.encode("utf-8")) for value in self.registered_classes)
            + len(self.audit_json.encode("utf-8"))
            + 32
        )
        return int(
            self.coefficients.nbytes
            + self.feature_mean.nbytes
            + self.feature_std.nbytes
            + metadata
        )

    def resource_audit(self) -> dict[str, Any]:
        new_class_count = len(self.registered_classes) - self.old_class_count
        fitted_parameters = EVIDENCE_DIM + 1 if self.enabled else 0
        audit = json.loads(self.audit_json)
        closed_form_solves = int(audit.get("closed_form_solve_count", 0))
        return {
            "schema": "cvs.phase2.d28_support_evidence_gate_resource.v1",
            "enabled": self.enabled,
            "k_shot": self.k_shot,
            "evidence_dim": EVIDENCE_DIM,
            "fitted_parameter_count": fitted_parameters,
            "ridge_coefficient_count": fitted_parameters,
            "normalization_scalar_count": (
                int(self.feature_mean.size + self.feature_std.size)
            ),
            "total_fitted_state_scalar_count": int(
                fitted_parameters + self.feature_mean.size + self.feature_std.size
            ),
            "gradient_trainable_parameter_count": 0,
            "persistent_state_bytes": self.persistent_state_bytes,
            "persistent_state_cap_bytes": MAX_GATE_STATE_BYTES,
            "persistent_state_cap_pass": (
                self.persistent_state_bytes <= MAX_GATE_STATE_BYTES
            ),
            "normalization_state_bytes": int(
                self.feature_mean.nbytes + self.feature_std.nbytes
            ),
            "ridge_lambda_candidate_count": (
                int(audit.get("ridge_lambda_candidate_count", 0))
            ),
            "closed_form_solve_count": closed_form_solves,
            "maximum_ridge_condition_number": MAX_RIDGE_CONDITION_NUMBER,
            "maximum_coefficient_l2_norm": MAX_COEFFICIENT_L2_NORM,
            "minimum_effective_feature_std": MIN_EFFECTIVE_STD,
            "estimated_gate_macs_per_query": EVIDENCE_DIM + 1 if self.enabled else 0,
            "normalization_subtractions_per_query": (
                EVIDENCE_DIM if self.enabled else 0
            ),
            "normalization_divisions_per_query": EVIDENCE_DIM if self.enabled else 0,
            "clip_scalar_count_per_query": 1 if self.enabled else 0,
            "new_score_additions_per_query": new_class_count if self.enabled else 0,
            "estimated_gate_temporary_bytes": int(
                (
                    EVIDENCE_DIM
                    + 2 * len(self.registered_classes)
                    + 2
                )
                * np.dtype(np.float32).itemsize
                if self.enabled
                else 0
            ),
            "old_score_writes_per_query": 0,
            "dense_query_graph_bytes": 0,
            "query_rows_used_for_fit": 0,
            "query_labels_used_for_fit": False,
            "query_features_used_for_fit": False,
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "query_batch_statistics_used": False,
            "row_local_inference": True,
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "source_sample_access": False,
            "source_derived_signal_access": False,
            "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
            "phase2_query_decision_policy": "per_sample_all_registered_classes",
        }


def extract_e5_per_row(raw_scores: np.ndarray, old_class_count: int) -> np.ndarray:
    """Extract the five D28 contrasts independently for every score row."""

    scores = _validate_scores(raw_scores, name="D28 raw scores")
    old_count = int(old_class_count)
    if not 3 <= old_count <= scores.shape[1] - 3:
        raise SupportEvidenceGateError(
            "D28 E5 requires at least three registered old and new classes"
        )
    old_sorted = np.sort(scores[:, :old_count], axis=1)
    new_sorted = np.sort(scores[:, old_count:], axis=1)
    o1 = old_sorted[:, -1]
    o2 = old_sorted[:, -2]
    n1 = new_sorted[:, -1]
    n2 = new_sorted[:, -2]
    top3_mean_old = np.mean(old_sorted[:, -3:], axis=1, dtype=np.float32)
    top3_mean_new = np.mean(new_sorted[:, -3:], axis=1, dtype=np.float32)
    evidence = np.stack(
        (
            n1 - o1,
            n1 - n2,
            o1 - o2,
            top3_mean_new - top3_mean_old,
            (n1 - top3_mean_new) - (o1 - top3_mean_old),
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    if not np.isfinite(evidence).all():
        raise SupportEvidenceGateError("D28 E5 produced non-finite evidence")
    return np.ascontiguousarray(evidence, dtype=np.float32)


def _validate_support(
    support_scores: np.ndarray,
    support_labels: Sequence[str],
    support_shot_ranks: Sequence[int],
    registered_classes: Sequence[str],
    old_class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], int, int]:
    classes = tuple(str(value) for value in registered_classes)
    old_count = int(old_class_count)
    scores = _validate_scores(
        support_scores, class_count=len(classes), name="D28 support scores"
    )
    labels = np.asarray(tuple(str(value) for value in support_labels))
    ranks = np.asarray(tuple(support_shot_ranks))
    if (
        len(classes) < 6
        or len(set(classes)) != len(classes)
        or not 3 <= old_count <= len(classes) - 3
        or labels.ndim != 1
        or ranks.ndim != 1
        or len(labels) != len(scores)
        or len(ranks) != len(scores)
        or not np.issubdtype(ranks.dtype, np.integer)
        or set(labels.tolist()) != set(classes)
    ):
        raise SupportEvidenceGateError("D28 support registry or row metadata drift")
    ranks = np.asarray(ranks, dtype=np.int64)
    class_counts = [int(np.sum(labels == value)) for value in classes]
    if min(class_counts) < 1 or len(set(class_counts)) != 1:
        raise SupportEvidenceGateError("D28 support must be class-symmetric K-shot")
    k_shot = class_counts[0]
    expected_ranks = np.arange(k_shot, dtype=np.int64)
    for class_name in classes:
        observed = np.sort(ranks[labels == class_name])
        if not np.array_equal(observed, expected_ranks):
            raise SupportEvidenceGateError(
                "D28 each class must expose exact unique shot ranks 0..K-1"
            )
    class_to_index = {value: index for index, value in enumerate(classes)}
    class_indices = np.asarray(
        [class_to_index[str(value)] for value in labels.tolist()], dtype=np.int64
    )
    binary_targets = np.where(class_indices >= old_count, 1.0, -1.0).astype(
        np.float64
    )
    return scores, labels, ranks, classes, old_count, k_shot


def _balanced_weights(binary_targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(binary_targets, dtype=np.float64)
    old_count = int(np.sum(targets < 0.0))
    new_count = int(np.sum(targets > 0.0))
    if old_count < 1 or new_count < 1:
        raise SupportEvidenceGateError("D28 ridge split must contain old and new support")
    weights = np.where(
        targets < 0.0,
        len(targets) / (2.0 * old_count),
        len(targets) / (2.0 * new_count),
    )
    return np.asarray(weights, dtype=np.float64)


def _fit_closed_form_ridge(
    evidence: np.ndarray,
    targets: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    x = np.asarray(evidence, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    feature_mean = np.mean(x, axis=0)
    feature_std = np.std(x, axis=0)
    if (
        feature_mean.shape != (EVIDENCE_DIM,)
        or feature_std.shape != (EVIDENCE_DIM,)
        or not np.isfinite(feature_mean).all()
        or not np.isfinite(feature_std).all()
        or bool(np.any(feature_std < MIN_EFFECTIVE_STD))
    ):
        raise SupportEvidenceGateError(
            "D28 ridge encountered an ineffective evidence feature std<1e-4"
        )
    standardized = (x - feature_mean[None, :]) / feature_std[None, :]
    design = np.concatenate(
        (standardized, np.ones((len(x), 1), dtype=np.float64)), axis=1
    )
    weights = _balanced_weights(y)
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_targets = y * np.sqrt(weights)
    penalty = np.eye(EVIDENCE_DIM + 1, dtype=np.float64)
    penalty[-1, -1] = 0.0
    gram = weighted_design.T @ weighted_design + float(ridge_lambda) * penalty
    rhs = weighted_design.T @ weighted_targets
    condition_number = float(np.linalg.cond(gram))
    if (
        not math.isfinite(condition_number)
        or condition_number > MAX_RIDGE_CONDITION_NUMBER
    ):
        raise SupportEvidenceGateError("D28 ridge condition number exceeds 1e6")
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError as exc:
        raise SupportEvidenceGateError("D28 closed-form ridge solve failed") from exc
    if coefficients.shape != (EVIDENCE_DIM + 1,) or not np.isfinite(coefficients).all():
        raise SupportEvidenceGateError("D28 ridge produced invalid coefficients")
    coefficient_norm = float(np.linalg.norm(coefficients))
    if not math.isfinite(coefficient_norm) or coefficient_norm > MAX_COEFFICIENT_L2_NORM:
        raise SupportEvidenceGateError("D28 ridge coefficient L2 norm exceeds 8")
    diagnostics = {
        "condition_number": condition_number,
        "coefficient_l2_norm": coefficient_norm,
        "minimum_feature_std": float(np.min(feature_std)),
        "maximum_feature_std": float(np.max(feature_std)),
    }
    return (
        np.asarray(coefficients, dtype=np.float64),
        np.asarray(feature_mean, dtype=np.float64),
        np.asarray(feature_std, dtype=np.float64),
        diagnostics,
    )


def _balanced_binary_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    old_mask = truth < 0.0
    new_mask = truth > 0.0
    old_accuracy = float(np.mean(values[old_mask] < 0.0))
    new_accuracy = float(np.mean(values[new_mask] >= 0.0))
    old_mse = float(np.mean((values[old_mask] - truth[old_mask]) ** 2))
    new_mse = float(np.mean((values[new_mask] - truth[new_mask]) ** 2))
    return {
        "old_accuracy": old_accuracy,
        "new_accuracy": new_accuracy,
        "balanced_accuracy": 0.5 * (old_accuracy + new_accuracy),
        "worst_role_accuracy": min(old_accuracy, new_accuracy),
        "balanced_mse": 0.5 * (old_mse + new_mse),
    }


def _identity_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    old_class_count: int,
) -> dict[str, Any]:
    class_to_index = {value: index for index, value in enumerate(classes)}
    truth = np.asarray(
        [class_to_index[str(value)] for value in labels.tolist()], dtype=np.int64
    )
    predictions = np.argmax(scores, axis=1)
    correct = predictions == truth
    old_mask = truth < old_class_count
    new_mask = ~old_mask
    per_old = {
        class_name: float(np.mean(correct[truth == index]))
        for index, class_name in enumerate(classes[:old_class_count])
    }
    per_new = {
        class_name: float(np.mean(correct[truth == old_class_count + index]))
        for index, class_name in enumerate(classes[old_class_count:])
    }
    return {
        "old_overall_accuracy": float(np.mean(correct[old_mask])),
        "new_overall_accuracy": float(np.mean(correct[new_mask])),
        "new_class_floor_accuracy": float(min(per_new.values())),
        "per_old_class_accuracy": per_old,
        "per_new_class_accuracy": per_new,
    }


def _apply_correction_array(
    scores: np.ndarray,
    q: np.ndarray,
    old_class_count: int,
    config: SupportEvidenceGateConfig,
) -> np.ndarray:
    correction = np.clip(
        np.float32(config.alpha) * np.asarray(q, dtype=np.float32),
        -np.float32(config.delta),
        np.float32(config.delta),
    ).astype(np.float32, copy=False)
    adjusted = np.asarray(scores, dtype=np.float32).copy()
    adjusted[:, old_class_count:] += correction[:, None]
    return adjusted


def _oof_identity_safety(
    raw_metrics: dict[str, Any],
    gated_metrics: dict[str, Any],
) -> tuple[bool, bool, dict[str, float]]:
    tolerance = 1.0e-12
    old_class_safe = all(
        gated_metrics["per_old_class_accuracy"][class_name] + tolerance
        >= raw_accuracy
        for class_name, raw_accuracy in raw_metrics["per_old_class_accuracy"].items()
    )
    scalar_names = (
        "old_overall_accuracy",
        "new_overall_accuracy",
        "new_class_floor_accuracy",
    )
    scalar_safe = all(
        gated_metrics[name] + tolerance >= raw_metrics[name] for name in scalar_names
    )
    gains = {
        name: float(gated_metrics[name] - raw_metrics[name]) for name in scalar_names
    }
    per_old_gains = [
        gated_metrics["per_old_class_accuracy"][class_name] - raw_accuracy
        for class_name, raw_accuracy in raw_metrics["per_old_class_accuracy"].items()
    ]
    strict_improvement = bool(
        any(value > tolerance for value in gains.values())
        or any(value > tolerance for value in per_old_gains)
    )
    return bool(old_class_safe and scalar_safe), strict_improvement, gains


def fit_support_evidence_gate(
    support_scores: np.ndarray,
    support_labels: Sequence[str],
    support_shot_ranks: Sequence[int],
    registered_classes: Sequence[str],
    old_class_count: int,
    *,
    config: SupportEvidenceGateConfig | None = None,
) -> SupportEvidenceGateState:
    """Fit D28 from registered support only; K1 returns an exact disabled gate."""

    locked = config or SupportEvidenceGateConfig()
    locked.validate()
    scores, labels, ranks, classes, old_count, k_shot = _validate_support(
        support_scores,
        support_labels,
        support_shot_ranks,
        registered_classes,
        old_class_count,
    )
    if k_shot == 1:
        audit = {
            "schema": "cvs.phase2.d28_support_evidence_gate_fit_audit.v1",
            "selection_policy": "k1_disabled_exact_d27_passthrough",
            "enabled": False,
            "k_shot": 1,
            "ridge_lambda_candidate_count": 0,
            "closed_form_solve_count": 0,
            "query_rows_used": 0,
            "query_batch_statistics_used": False,
        }
        return SupportEvidenceGateState(
            schema=SCHEMA,
            registered_classes=classes,
            old_class_count=old_count,
            k_shot=1,
            enabled=False,
            selected_lambda=None,
            coefficients=np.empty(0, dtype=np.float32),
            feature_mean=np.empty(0, dtype=np.float32),
            feature_std=np.empty(0, dtype=np.float32),
            audit_json=_canonical_json_bytes(audit).decode("utf-8"),
            config=locked,
        )
    if k_shot < OOF_FOLD_COUNT:
        raise SupportEvidenceGateError(
            "D28 requires K>=5 for exact shot-rank 5-fold OOF; only K1 may bypass"
        )

    evidence = extract_e5_per_row(scores, old_count)
    class_to_index = {value: index for index, value in enumerate(classes)}
    class_indices = np.asarray(
        [class_to_index[str(value)] for value in labels.tolist()], dtype=np.int64
    )
    targets = np.where(class_indices >= old_count, 1.0, -1.0).astype(np.float64)
    folds = np.mod(ranks, OOF_FOLD_COUNT)
    raw_identity_metrics = _identity_metrics(scores, labels, classes, old_count)
    lambda_evidence: list[dict[str, Any]] = []
    ranked: list[tuple[tuple[float, ...], float, dict[str, Any]]] = []
    successful_solve_count = 0
    for lambda_index, ridge_lambda in enumerate(locked.ridge_lambdas):
        oof_predictions = np.empty(len(scores), dtype=np.float64)
        fold_evidence: list[dict[str, Any]] = []
        candidate_error: str | None = None
        for fold in range(OOF_FOLD_COUNT):
            holdout = folds == fold
            training = ~holdout
            if not bool(np.any(holdout)) or not bool(np.any(training)):
                raise SupportEvidenceGateError("D28 shot-rank OOF fold is empty")
            for mask_name, mask in (("training", training), ("holdout", holdout)):
                present = set(labels[mask].tolist())
                if present != set(classes):
                    raise SupportEvidenceGateError(
                        f"D28 OOF {mask_name} fold lacks a registered support class"
                    )
            try:
                coefficients, feature_mean, feature_std, diagnostics = (
                    _fit_closed_form_ridge(
                        evidence[training], targets[training], ridge_lambda
                    )
                )
            except SupportEvidenceGateError as exc:
                candidate_error = str(exc)
                break
            successful_solve_count += 1
            standardized_holdout = (
                np.asarray(evidence[holdout], dtype=np.float64)
                - feature_mean[None, :]
            ) / feature_std[None, :]
            holdout_design = np.concatenate(
                (
                    standardized_holdout,
                    np.ones((int(np.sum(holdout)), 1), dtype=np.float64),
                ),
                axis=1,
            )
            fold_predictions = holdout_design @ coefficients
            if not np.isfinite(fold_predictions).all():
                raise SupportEvidenceGateError("D28 OOF produced non-finite predictions")
            oof_predictions[holdout] = fold_predictions
            fold_evidence.append(
                {
                    "fold": fold,
                    "training_rows": int(np.sum(training)),
                    "holdout_rows": int(np.sum(holdout)),
                    "training_feature_mean": feature_mean.tolist(),
                    "training_feature_std": feature_std.tolist(),
                    "ridge_diagnostics": diagnostics,
                    "holdout_metrics": _balanced_binary_metrics(
                        fold_predictions, targets[holdout]
                    ),
                }
            )
        if candidate_error is not None:
            lambda_evidence.append(
                {
                    "ridge_lambda": float(ridge_lambda),
                    "valid": False,
                    "failure_reason": candidate_error,
                    "fold_evidence": fold_evidence,
                }
            )
            continue
        metrics = _balanced_binary_metrics(oof_predictions, targets)
        gated_oof_scores = _apply_correction_array(
            scores, oof_predictions, old_count, locked
        )
        gated_identity_metrics = _identity_metrics(
            gated_oof_scores, labels, classes, old_count
        )
        non_degradation_pass, strict_improvement, identity_gains = (
            _oof_identity_safety(raw_identity_metrics, gated_identity_metrics)
        )
        safety_pass = bool(non_degradation_pass and strict_improvement)
        candidate = {
            "ridge_lambda": float(ridge_lambda),
            "valid": True,
            "binary_role_metrics": metrics,
            "raw_identity_metrics": raw_identity_metrics,
            "gated_identity_metrics": gated_identity_metrics,
            "identity_metric_gains": identity_gains,
            "identity_non_degradation_pass": non_degradation_pass,
            "strict_identity_improvement": strict_improvement,
            "oof_gate_safety_pass": safety_pass,
            "fold_evidence": fold_evidence,
        }
        lambda_evidence.append(candidate)
        if not safety_pass:
            continue
        ranking = (
            identity_gains["new_class_floor_accuracy"],
            identity_gains["new_overall_accuracy"],
            identity_gains["old_overall_accuracy"],
            metrics["balanced_accuracy"],
            metrics["worst_role_accuracy"],
            -metrics["balanced_mse"],
            -float(lambda_index),
        )
        ranked.append((ranking, float(ridge_lambda), candidate))
    if not ranked:
        audit = {
            "schema": "cvs.phase2.d28_support_evidence_gate_fit_audit.v1",
            "selection_policy": "oof_identity_safety_failed_disabled_passthrough",
            "enabled": False,
            "fallback_reason": "no_lambda_passed_oof_identity_non_degradation_and_gain",
            "k_shot": k_shot,
            "fold_assignment_policy": "within_registered_class_shot_rank_mod_5",
            "class_balance_policy": "equal_total_weight_old_vs_new_support_roles",
            "ridge_lambda_grid": list(locked.ridge_lambdas),
            "ridge_lambda_candidate_count": len(locked.ridge_lambdas),
            "closed_form_solve_count": successful_solve_count,
            "raw_identity_metrics": raw_identity_metrics,
            "lambda_evidence": lambda_evidence,
            "query_rows_used": 0,
            "query_batch_statistics_used": False,
        }
        return SupportEvidenceGateState(
            schema=SCHEMA,
            registered_classes=classes,
            old_class_count=old_count,
            k_shot=k_shot,
            enabled=False,
            selected_lambda=None,
            coefficients=np.empty(0, dtype=np.float32),
            feature_mean=np.empty(0, dtype=np.float32),
            feature_std=np.empty(0, dtype=np.float32),
            audit_json=_canonical_json_bytes(audit).decode("utf-8"),
            config=locked,
        )
    selected = max(ranked, key=lambda item: item[0])
    selected_lambda = selected[1]
    try:
        (
            final_coefficients,
            final_feature_mean,
            final_feature_std,
            final_diagnostics,
        ) = _fit_closed_form_ridge(evidence, targets, selected_lambda)
    except SupportEvidenceGateError as exc:
        audit = {
            "schema": "cvs.phase2.d28_support_evidence_gate_fit_audit.v1",
            "selection_policy": "full_fit_safety_failed_disabled_passthrough",
            "enabled": False,
            "fallback_reason": str(exc),
            "k_shot": k_shot,
            "ridge_lambda_grid": list(locked.ridge_lambdas),
            "ridge_lambda_candidate_count": len(locked.ridge_lambdas),
            "closed_form_solve_count": successful_solve_count,
            "raw_identity_metrics": raw_identity_metrics,
            "lambda_evidence": lambda_evidence,
            "query_rows_used": 0,
            "query_batch_statistics_used": False,
        }
        return SupportEvidenceGateState(
            schema=SCHEMA,
            registered_classes=classes,
            old_class_count=old_count,
            k_shot=k_shot,
            enabled=False,
            selected_lambda=None,
            coefficients=np.empty(0, dtype=np.float32),
            feature_mean=np.empty(0, dtype=np.float32),
            feature_std=np.empty(0, dtype=np.float32),
            audit_json=_canonical_json_bytes(audit).decode("utf-8"),
            config=locked,
        )
    successful_solve_count += 1
    audit = {
        "schema": "cvs.phase2.d28_support_evidence_gate_fit_audit.v1",
        "selection_policy": (
            "shot_rank_5fold_oof_identity_safe_lex_new_floor_new_old_then_role"
        ),
        "enabled": True,
        "k_shot": k_shot,
        "fold_assignment_policy": "within_registered_class_shot_rank_mod_5",
        "class_balance_policy": "equal_total_weight_old_vs_new_support_roles",
        "feature_standardization_policy": "per_fold_training_mean_std_then_full_fit",
        "minimum_effective_feature_std": MIN_EFFECTIVE_STD,
        "maximum_ridge_condition_number": MAX_RIDGE_CONDITION_NUMBER,
        "maximum_coefficient_l2_norm": MAX_COEFFICIENT_L2_NORM,
        "ridge_lambda_grid": list(locked.ridge_lambdas),
        "ridge_lambda_candidate_count": len(locked.ridge_lambdas),
        "closed_form_solve_count": successful_solve_count,
        "selected_lambda": selected_lambda,
        "selected_oof_binary_role_metrics": selected[2]["binary_role_metrics"],
        "raw_identity_metrics": raw_identity_metrics,
        "selected_gated_identity_metrics": selected[2]["gated_identity_metrics"],
        "selected_identity_metric_gains": selected[2]["identity_metric_gains"],
        "selected_oof_gate_safety_pass": selected[2]["oof_gate_safety_pass"],
        "full_fit_feature_mean": final_feature_mean.tolist(),
        "full_fit_feature_std": final_feature_std.tolist(),
        "full_fit_ridge_diagnostics": final_diagnostics,
        "lambda_evidence": lambda_evidence,
        "support_rows": len(scores),
        "old_support_rows": int(np.sum(targets < 0.0)),
        "new_support_rows": int(np.sum(targets > 0.0)),
        "query_rows_used": 0,
        "query_batch_statistics_used": False,
    }
    return SupportEvidenceGateState(
        schema=SCHEMA,
        registered_classes=classes,
        old_class_count=old_count,
        k_shot=k_shot,
        enabled=True,
        selected_lambda=selected_lambda,
        coefficients=np.asarray(final_coefficients, dtype=np.float32),
        feature_mean=np.asarray(final_feature_mean, dtype=np.float32),
        feature_std=np.asarray(final_feature_std, dtype=np.float32),
        audit_json=_canonical_json_bytes(audit).decode("utf-8"),
        config=locked,
    )


def apply_support_evidence_gate(
    state: SupportEvidenceGateState,
    raw_scores: np.ndarray,
) -> np.ndarray:
    """Apply one row-local correction while preserving both score prefixes."""

    scores = _validate_scores(
        raw_scores,
        class_count=len(state.registered_classes),
        name="D28 inference scores",
    )
    if not state.enabled:
        return _readonly(scores, np.float32)
    evidence = extract_e5_per_row(scores, state.old_class_count)
    standardized = (
        evidence - state.feature_mean[None, :]
    ) / state.feature_std[None, :]
    # An explicit row loop prevents BLAS from selecting shape-dependent GEMV/GEMM
    # reduction orders.  Five products per row keep the deployment cost bounded.
    q = np.asarray(
        [
            np.float32(
                np.sum(
                    row * state.coefficients[:EVIDENCE_DIM],
                    dtype=np.float32,
                )
                + state.coefficients[EVIDENCE_DIM]
            )
            for row in standardized
        ],
        dtype=np.float32,
    )
    old_prefix = scores[:, : state.old_class_count].copy()
    new_before = scores[:, state.old_class_count :].copy()
    adjusted = _apply_correction_array(
        scores, q, state.old_class_count, state.config
    )
    if not np.array_equal(adjusted[:, : state.old_class_count], old_prefix):
        raise SupportEvidenceGateError("D28 mutated the old score prefix")
    before_order = np.argsort(new_before, axis=1, kind="stable")
    after_order = np.argsort(
        adjusted[:, state.old_class_count :], axis=1, kind="stable"
    )
    if not np.array_equal(before_order, after_order):
        raise SupportEvidenceGateError("D28 FP32 correction changed new-class ordering")
    if not np.isfinite(adjusted).all():
        raise SupportEvidenceGateError("D28 inference produced non-finite scores")
    return _readonly(adjusted, np.float32)


def predict_with_support_evidence_gate(
    state: SupportEvidenceGateState,
    raw_scores: np.ndarray,
) -> np.ndarray:
    """Perform one all-registered-class argmax after row-local correction."""

    adjusted = apply_support_evidence_gate(state, raw_scores)
    classes = np.asarray(state.registered_classes)
    return classes[np.argmax(adjusted, axis=1)]


__all__ = [
    "EVIDENCE_DIM",
    "OOF_FOLD_COUNT",
    "RIDGE_LAMBDA_GRID",
    "SupportEvidenceGateConfig",
    "SupportEvidenceGateError",
    "SupportEvidenceGateState",
    "apply_support_evidence_gate",
    "extract_e5_per_row",
    "fit_support_evidence_gate",
    "predict_with_support_evidence_gate",
]
