"""D29 per-class safe release over frozen D27 all-class scores.

The fitter accepts registered support scores, labels, and within-class shot
ranks only.  It learns no gradient parameters and exposes no query fitting,
role, quota, or batch-statistics surface.  Each new class stores one activation
width and one release amplitude.  Old score columns are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Integral
from typing import Any, Sequence

import numpy as np


SCHEMA = "cvs.phase2.d29_classwise_safe_release.v1"
OOF_FOLD_COUNT = 5
QUANTILE_POLICIES = ("disabled", "q50", "q75", "q90", "max")
OBJECTIVES = ("overall_first", "balance_first", "floor_first")
SAFETY_EPS = 1.0e-4
MIN_WIDTH = 1.0e-6
MAX_PREDICTOR_STATE_BYTES = 256 * 1024


class ClasswiseSafeReleaseError(ValueError):
    """Raised when D29 support, state, or inference drifts."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validate_scores(
    value: np.ndarray, *, class_count: int | None = None, name: str
) -> np.ndarray:
    scores = np.asarray(value, dtype=np.float32)
    if (
        scores.ndim != 2
        or len(scores) < 1
        or scores.shape[1] < 2
        or not np.isfinite(scores).all()
        or (class_count is not None and scores.shape[1] != class_count)
    ):
        raise ClasswiseSafeReleaseError(f"{name} must be finite [N,C]")
    return np.ascontiguousarray(scores, dtype=np.float32)


@dataclass(frozen=True)
class ClasswiseSafeReleaseConfig:
    safety_budget: float = 0.5
    objective: str = "balance_first"
    coordinate_passes: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective", str(self.objective))
        self.validate()

    def validate(self) -> None:
        try:
            safety_budget = float(self.safety_budget)
        except (TypeError, ValueError) as exc:
            raise ClasswiseSafeReleaseError(
                "D29 safety budget must be in (0,1]"
            ) from exc
        if (
            isinstance(self.safety_budget, (bool, np.bool_))
            or not math.isfinite(safety_budget)
            or not 0.0 < safety_budget <= 1.0
        ):
            raise ClasswiseSafeReleaseError("D29 safety budget must be in (0,1]")
        if self.objective not in OBJECTIVES:
            raise ClasswiseSafeReleaseError("D29 objective is not method-locked")
        if (
            not isinstance(self.coordinate_passes, Integral)
            or isinstance(self.coordinate_passes, (bool, np.bool_))
            or int(self.coordinate_passes) not in (1, 2)
        ):
            raise ClasswiseSafeReleaseError("D29 uses one or two coordinate passes")


@dataclass(frozen=True)
class ClasswiseSafeReleaseState:
    schema: str
    registered_classes: tuple[str, ...]
    old_class_count: int
    k_shot: int
    enabled: bool
    widths: np.ndarray
    amplitudes: np.ndarray
    audit_json: str
    config: ClasswiseSafeReleaseConfig

    def __post_init__(self) -> None:
        self.config.validate()
        classes = tuple(str(value) for value in self.registered_classes)
        if (
            not isinstance(self.old_class_count, Integral)
            or isinstance(self.old_class_count, (bool, np.bool_))
            or not isinstance(self.k_shot, Integral)
            or isinstance(self.k_shot, (bool, np.bool_))
        ):
            raise ClasswiseSafeReleaseError("D29 state counts must be integers")
        old_count = int(self.old_class_count)
        k_shot = int(self.k_shot)
        widths = np.asarray(self.widths)
        amplitudes = np.asarray(self.amplitudes)
        new_count = len(classes) - old_count
        if (
            not isinstance(self.enabled, (bool, np.bool_))
            or
            self.schema != SCHEMA
            or len(classes) < 4
            or len(set(classes)) != len(classes)
            or not 2 <= old_count < len(classes)
            or new_count < 2
            or k_shot < 1
            or widths.dtype != np.float32
            or amplitudes.dtype != np.float32
            or widths.shape != (new_count,)
            or amplitudes.shape != (new_count,)
            or not np.isfinite(widths).all()
            or not np.isfinite(amplitudes).all()
            or bool(np.any(widths < 0.0))
            or bool(np.any(amplitudes < 0.0))
        ):
            raise ClasswiseSafeReleaseError("D29 state drift")
        if not self.enabled and (
            bool(np.any(widths != 0.0)) or bool(np.any(amplitudes != 0.0))
        ):
            raise ClasswiseSafeReleaseError("disabled D29 must be exact passthrough")
        try:
            audit = json.loads(str(self.audit_json))
        except json.JSONDecodeError as exc:
            raise ClasswiseSafeReleaseError("D29 audit is invalid JSON") from exc
        if not isinstance(audit, dict):
            raise ClasswiseSafeReleaseError("D29 audit must be one object")
        object.__setattr__(self, "registered_classes", classes)
        object.__setattr__(self, "old_class_count", old_count)
        object.__setattr__(self, "k_shot", k_shot)
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "widths", _readonly(widths, np.float32))
        object.__setattr__(self, "amplitudes", _readonly(amplitudes, np.float32))
        object.__setattr__(self, "audit_json", _canonical_json(audit))
        if self.deployable_predictor_state_bytes > MAX_PREDICTOR_STATE_BYTES:
            raise ClasswiseSafeReleaseError("D29 predictor state exceeds 256KB")

    @property
    def deployable_predictor_state_bytes(self) -> int:
        return int(self.widths.nbytes + self.amplitudes.nbytes + 32)

    @property
    def persistent_evidence_object_bytes(self) -> int:
        return int(
            self.deployable_predictor_state_bytes
            + len(self.audit_json.encode("utf-8"))
            + len(self.schema.encode("utf-8"))
            + sum(len(value.encode("utf-8")) for value in self.registered_classes)
        )

    def resource_audit(self) -> dict[str, Any]:
        new_count = len(self.registered_classes) - self.old_class_count
        scalar_count = 2 * new_count if self.enabled else 0
        return {
            "schema": "cvs.phase2.d29_classwise_safe_release_resource.v1",
            "enabled": self.enabled,
            "k_shot": self.k_shot,
            "new_class_count": new_count,
            "fitted_parameter_count": scalar_count,
            "gradient_trainable_parameter_count": 0,
            "width_scalar_count": new_count if self.enabled else 0,
            "amplitude_scalar_count": new_count if self.enabled else 0,
            "deployable_predictor_state_bytes": self.deployable_predictor_state_bytes,
            "external_evidence_audit_bytes": len(self.audit_json.encode("utf-8")),
            "audit_metadata_excluded_from_deployment_state": True,
            "persistent_state_cap_bytes": MAX_PREDICTOR_STATE_BYTES,
            "persistent_state_cap_pass": (
                self.deployable_predictor_state_bytes
                <= MAX_PREDICTOR_STATE_BYTES
            ),
            "estimated_release_scalar_ops_per_query": (
                4 * new_count if self.enabled else 0
            ),
            "estimated_release_ramp_arithmetic_ops_per_query": (
                4 * new_count if self.enabled else 0
            ),
            "score_reduction_and_ordering_ops_included_in_ramp_count": False,
            "score_reduction_algorithm": "old_max_plus_stable_new_argsort",
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
            "phase2_clean_dataset_reachable": False,
            "phase2_clean_cache_reachable": False,
            "phase2_clean_control_flow_reachable": False,
            "phase2_source_sample_access": False,
            "phase2_source_cache_access": False,
            "phase2_source_label_access": False,
            "phase2_unapproved_source_derived_signal_access": False,
            "phase2_source_replay": False,
            "phase2_external_source_adapter_access": False,
            "phase2_pretrained_artifact_policy": (
                "sealed_phase1_deployment_bundle_with_optional_int8_"
                "domain_class_prototypes_v1"
            ),
            "adaptation_mode": "EVAL_ONLY_CLOSED_FORM_ADAPTATION",
            "gradient_steps": 0,
        }


def _validate_support(
    support_scores: np.ndarray,
    support_labels: Sequence[str],
    support_shot_ranks: Sequence[int],
    registered_classes: Sequence[str],
    old_class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], int, int]:
    classes = tuple(str(value) for value in registered_classes)
    if (
        not isinstance(old_class_count, Integral)
        or isinstance(old_class_count, (bool, np.bool_))
    ):
        raise ClasswiseSafeReleaseError("D29 old class count must be an integer")
    old_count = int(old_class_count)
    scores = _validate_scores(
        support_scores, class_count=len(classes), name="D29 support scores"
    )
    labels = np.asarray(tuple(str(value) for value in support_labels))
    ranks = np.asarray(tuple(support_shot_ranks))
    if (
        len(classes) < 4
        or len(set(classes)) != len(classes)
        or not 2 <= old_count <= len(classes) - 2
        or labels.ndim != 1
        or ranks.ndim != 1
        or len(labels) != len(scores)
        or len(ranks) != len(scores)
        or not np.issubdtype(ranks.dtype, np.integer)
        or set(labels.tolist()) != set(classes)
    ):
        raise ClasswiseSafeReleaseError("D29 support registry drift")
    ranks = np.asarray(ranks, dtype=np.int64)
    counts = [int(np.sum(labels == value)) for value in classes]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise ClasswiseSafeReleaseError("D29 support must be class-symmetric K-shot")
    k_shot = counts[0]
    expected = np.arange(k_shot, dtype=np.int64)
    for class_name in classes:
        if not np.array_equal(np.sort(ranks[labels == class_name]), expected):
            raise ClasswiseSafeReleaseError(
                "D29 each class must expose unique shot ranks 0..K-1"
            )
    return scores, labels, ranks, classes, old_count, k_shot


def _z_matrix(scores: np.ndarray, old_count: int) -> np.ndarray:
    values = _validate_scores(scores, name="D29 score rows")
    new_scores = values[:, old_count:]
    old_top = np.max(values[:, :old_count], axis=1)
    if new_scores.shape[1] < 2:
        raise ClasswiseSafeReleaseError("D29 requires at least two new classes")
    order = np.argsort(new_scores, axis=1, kind="stable")
    top_index = order[:, -1]
    top_value = new_scores[np.arange(len(values)), top_index]
    second_value = new_scores[np.arange(len(values)), order[:, -2]]
    competitor_new = np.where(
        np.arange(new_scores.shape[1])[None, :] == top_index[:, None],
        second_value[:, None],
        top_value[:, None],
    )
    competitor = np.maximum(old_top[:, None], competitor_new)
    return np.asarray(new_scores - competitor, dtype=np.float32)


def _identity_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    old_count: int,
) -> dict[str, Any]:
    predictions = np.asarray(classes)[np.argmax(scores, axis=1)]
    old_labels = classes[:old_count]
    new_labels = classes[old_count:]
    old_mask = np.isin(labels, np.asarray(old_labels))
    new_mask = ~old_mask
    per_old = {
        value: float(np.mean(predictions[labels == value] == value))
        for value in old_labels
    }
    per_new = {
        value: float(np.mean(predictions[labels == value] == value))
        for value in new_labels
    }
    old_acc = float(np.mean(predictions[old_mask] == labels[old_mask]))
    new_acc = float(np.mean(predictions[new_mask] == labels[new_mask]))
    h = 0.0 if old_acc + new_acc <= 0.0 else 2.0 * old_acc * new_acc / (old_acc + new_acc)
    return {
        "old_overall_accuracy": old_acc,
        "new_overall_accuracy": new_acc,
        "new_class_floor_accuracy": min(per_new.values()),
        "H_old_new": h,
        "per_old_class_accuracy": per_old,
        "per_new_class_accuracy": per_new,
        "predictions": predictions,
    }


def _width_from_policy(deficits: np.ndarray, policy: str) -> float:
    if policy == "disabled":
        return 0.0
    values = np.asarray(deficits, dtype=np.float64)
    if len(values) < 1 or not np.isfinite(values).all():
        raise ClasswiseSafeReleaseError("D29 new-class deficits are invalid")
    if policy == "q50":
        base = float(np.quantile(values, 0.50))
    elif policy == "q75":
        base = float(np.quantile(values, 0.75))
    elif policy == "q90":
        base = float(np.quantile(values, 0.90))
    elif policy == "max":
        base = float(np.max(values))
    else:
        raise ClasswiseSafeReleaseError("D29 width policy drift")
    width = 1.05 * base
    return width if width >= MIN_WIDTH else 0.0


def _fit_one_class(
    scores: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    old_count: int,
    new_index: int,
    policy: str,
    config: ClasswiseSafeReleaseConfig,
) -> tuple[float, float, dict[str, Any]]:
    if policy == "disabled":
        return 0.0, 0.0, {"policy": policy, "enabled": False}
    z = _z_matrix(scores, old_count)
    class_name = classes[old_count + new_index]
    deficits = np.maximum(0.0, -z[labels == class_name, new_index])
    width = _width_from_policy(deficits, policy)
    if width <= 0.0:
        return 0.0, 0.0, {"policy": policy, "enabled": False, "reason": "zero_width"}
    p = np.clip((z[:, new_index] + width) / width, 0.0, 1.0)
    raw_predictions = np.asarray(classes)[np.argmax(scores, axis=1)]
    old_mask = np.isin(labels, np.asarray(classes[:old_count]))
    correct_old = old_mask & (raw_predictions == labels)
    class_to_index = {value: index for index, value in enumerate(classes)}
    true_indices = np.asarray([class_to_index[str(value)] for value in labels])
    old_rows = np.flatnonzero(correct_old & (p > 0.0))
    if len(old_rows):
        true_old_scores = scores[old_rows, true_indices[old_rows]]
        new_scores = scores[old_rows, old_count + new_index]
        margin = true_old_scores - new_scores - np.float32(SAFETY_EPS)
        if bool(np.any(margin <= 0.0)):
            return 0.0, 0.0, {"policy": policy, "enabled": False, "reason": "nonpositive_old_margin"}
        safe = float(np.min(margin / p[old_rows]))
    else:
        safe = 2.0 * width
    amplitude = float(config.safety_budget) * min(safe, 2.0 * width)
    if not math.isfinite(amplitude) or amplitude <= 0.0:
        return 0.0, 0.0, {"policy": policy, "enabled": False, "reason": "zero_amplitude"}
    return width, amplitude, {
        "policy": policy,
        "enabled": True,
        "width": width,
        "amplitude": amplitude,
        "safe_amplitude_upper_bound": safe,
        "old_guard_row_count": int(len(old_rows)),
    }


def _apply_arrays(
    scores: np.ndarray, old_count: int, widths: np.ndarray, amplitudes: np.ndarray
) -> np.ndarray:
    values = _validate_scores(scores, name="D29 correction scores")
    if widths.shape != amplitudes.shape or widths.shape != (
        values.shape[1] - old_count,
    ):
        raise ClasswiseSafeReleaseError("D29 release vector drift")
    adjusted = values.copy()
    z = _z_matrix(values, old_count)
    active = widths > 0.0
    if bool(np.any(active)):
        p = np.zeros_like(z, dtype=np.float32)
        p[:, active] = np.clip(
            (z[:, active] + widths[None, active]) / widths[None, active],
            0.0,
            1.0,
        )
        adjusted[:, old_count:] += p * amplitudes[None, :]
    return np.asarray(adjusted, dtype=np.float32)


def _safety_and_rank(
    raw_scores: np.ndarray,
    adjusted_scores: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    old_count: int,
    objective: str,
) -> tuple[bool, tuple[float, ...], dict[str, Any]]:
    raw = _identity_metrics(raw_scores, labels, classes, old_count)
    adjusted = _identity_metrics(adjusted_scores, labels, classes, old_count)
    old_mask = np.isin(labels, np.asarray(classes[:old_count]))
    raw_correct_old = old_mask & (raw["predictions"] == labels)
    correct_rows_preserved = bool(
        np.all(adjusted["predictions"][raw_correct_old] == labels[raw_correct_old])
    )
    old_classwise = all(
        adjusted["per_old_class_accuracy"][value] + 1.0e-12
        >= raw["per_old_class_accuracy"][value]
        for value in classes[:old_count]
    )
    new_nondegrade = bool(
        adjusted["new_overall_accuracy"] + 1.0e-12
        >= raw["new_overall_accuracy"]
        and adjusted["new_class_floor_accuracy"] + 1.0e-12
        >= raw["new_class_floor_accuracy"]
    )
    strict_new = bool(
        adjusted["new_overall_accuracy"]
        > raw["new_overall_accuracy"] + 1.0e-12
        or adjusted["new_class_floor_accuracy"]
        > raw["new_class_floor_accuracy"] + 1.0e-12
    )
    safe = bool(correct_rows_preserved and old_classwise and new_nondegrade)
    if objective == "overall_first":
        rank = (
            adjusted["new_overall_accuracy"],
            adjusted["new_class_floor_accuracy"],
            adjusted["H_old_new"],
        )
    elif objective == "balance_first":
        rank = (
            adjusted["H_old_new"],
            adjusted["new_class_floor_accuracy"],
            adjusted["new_overall_accuracy"],
        )
    else:
        rank = (
            adjusted["new_class_floor_accuracy"],
            adjusted["new_overall_accuracy"],
            adjusted["H_old_new"],
        )
    evidence = {
        "raw": {k: v for k, v in raw.items() if k != "predictions"},
        "adjusted": {k: v for k, v in adjusted.items() if k != "predictions"},
        "raw_correct_old_rows_preserved": correct_rows_preserved,
        "old_classwise_non_degradation": old_classwise,
        "new_overall_and_floor_non_degradation": new_nondegrade,
        "strict_new_improvement": strict_new,
        "safe": safe,
    }
    return safe, rank, evidence


def fit_classwise_safe_release(
    support_scores: np.ndarray,
    support_labels: Sequence[str],
    support_shot_ranks: Sequence[int],
    registered_classes: Sequence[str],
    old_class_count: int,
    *,
    config: ClasswiseSafeReleaseConfig | None = None,
) -> ClasswiseSafeReleaseState:
    """Fit D29 from support only using shot-rank OOF coordinate selection."""

    locked = config or ClasswiseSafeReleaseConfig()
    locked.validate()
    scores, labels, ranks, classes, old_count, k_shot = _validate_support(
        support_scores,
        support_labels,
        support_shot_ranks,
        registered_classes,
        old_class_count,
    )
    new_count = len(classes) - old_count
    zeros = np.zeros(new_count, dtype=np.float32)
    if k_shot == 1:
        audit = {
            "schema": "cvs.phase2.d29_classwise_safe_release_fit_audit.v1",
            "selection_policy": "k1_disabled_exact_d27_passthrough",
            "enabled": False,
            "k_shot": 1,
            "query_rows_used": 0,
        }
        return ClasswiseSafeReleaseState(
            SCHEMA, classes, old_count, 1, False, zeros, zeros, _canonical_json(audit), locked
        )
    if k_shot < OOF_FOLD_COUNT:
        raise ClasswiseSafeReleaseError(
            "D29 requires K>=5 for shot-rank OOF; only K1 may bypass"
        )

    folds = np.mod(ranks, OOF_FOLD_COUNT)
    # Precompute fold-local release arrays for every class and policy.  Each
    # holdout row is corrected only by parameters fitted on other shot ranks.
    cache: dict[tuple[int, str], np.ndarray] = {}
    fit_evidence: dict[str, Any] = {}
    for new_index in range(new_count):
        class_name = classes[old_count + new_index]
        fit_evidence[class_name] = {}
        for policy in QUANTILE_POLICIES:
            widths = np.zeros(OOF_FOLD_COUNT, dtype=np.float32)
            amplitudes = np.zeros(OOF_FOLD_COUNT, dtype=np.float32)
            fold_rows: list[dict[str, Any]] = []
            corrections = np.zeros(len(scores), dtype=np.float32)
            for fold in range(OOF_FOLD_COUNT):
                holdout = folds == fold
                training = ~holdout
                width, amplitude, evidence = _fit_one_class(
                    scores[training],
                    labels[training],
                    classes,
                    old_count,
                    new_index,
                    policy,
                    locked,
                )
                widths[fold] = np.float32(width)
                amplitudes[fold] = np.float32(amplitude)
                if width > 0.0 and amplitude > 0.0:
                    z_holdout = _z_matrix(scores[holdout], old_count)[:, new_index]
                    p = np.clip((z_holdout + width) / width, 0.0, 1.0)
                    corrections[holdout] = np.float32(amplitude) * p.astype(np.float32)
                fold_rows.append({"fold": fold, **evidence})
            cache[(new_index, policy)] = corrections
            fit_evidence[class_name][policy] = {
                "folds": fold_rows,
                "widths": widths.tolist(),
                "amplitudes": amplitudes.tolist(),
            }

    selected = ["disabled"] * new_count
    current_scores = scores.copy()
    safe, current_rank, current_evidence = _safety_and_rank(
        scores, current_scores, labels, classes, old_count, locked.objective
    )
    if not safe:
        raise ClasswiseSafeReleaseError("D29 disabled baseline failed safety")
    coordinate_trace: list[dict[str, Any]] = []
    for pass_index in range(int(locked.coordinate_passes)):
        changed = False
        for new_index in range(new_count):
            best_policy = selected[new_index]
            best_scores = current_scores
            best_rank = current_rank
            best_evidence = current_evidence
            trials: list[dict[str, Any]] = []
            # Remove this class's current correction before trying replacements.
            base_without = current_scores.copy()
            base_without[:, old_count + new_index] -= cache[
                (new_index, selected[new_index])
            ]
            for policy in QUANTILE_POLICIES:
                candidate = base_without.copy()
                candidate[:, old_count + new_index] += cache[(new_index, policy)]
                candidate_safe, candidate_rank, evidence = _safety_and_rank(
                    scores, candidate, labels, classes, old_count, locked.objective
                )
                trials.append(
                    {
                        "policy": policy,
                        "safe": candidate_safe,
                        "rank": list(candidate_rank),
                        "evidence": evidence,
                    }
                )
                if candidate_safe and (
                    candidate_rank > best_rank
                    or (candidate_rank == best_rank and policy < best_policy)
                ):
                    best_policy = policy
                    best_scores = candidate
                    best_rank = candidate_rank
                    best_evidence = evidence
            if best_policy != selected[new_index]:
                changed = True
            selected[new_index] = best_policy
            current_scores = best_scores
            current_rank = best_rank
            current_evidence = best_evidence
            coordinate_trace.append(
                {
                    "pass": pass_index,
                    "new_class": classes[old_count + new_index],
                    "selected_policy": best_policy,
                    "trials": trials,
                }
            )
        if not changed:
            break

    oof_safe, _, oof_evidence = _safety_and_rank(
        scores, current_scores, labels, classes, old_count, locked.objective
    )
    oof_strict = bool(oof_evidence["strict_new_improvement"])
    if not (oof_safe and oof_strict):
        selected = ["disabled"] * new_count

    widths = np.zeros(new_count, dtype=np.float32)
    amplitudes = np.zeros(new_count, dtype=np.float32)
    full_fit: list[dict[str, Any]] = []
    for new_index, policy in enumerate(selected):
        width, amplitude, evidence = _fit_one_class(
            scores, labels, classes, old_count, new_index, policy, locked
        )
        widths[new_index] = np.float32(width)
        amplitudes[new_index] = np.float32(amplitude)
        full_fit.append(
            {
                "new_class": classes[old_count + new_index],
                **evidence,
            }
        )
    enabled = bool(np.any(amplitudes > 0.0))
    full_adjusted = _apply_arrays(scores, old_count, widths, amplitudes)
    full_safe, _, full_evidence = _safety_and_rank(
        scores, full_adjusted, labels, classes, old_count, locked.objective
    )
    full_refit_evidence = full_evidence
    full_fit_fallback_reason: str | None = None
    if enabled and not full_safe:
        full_fit_fallback_reason = "full_fit_safety_failed"
    elif enabled and not bool(full_evidence["strict_new_improvement"]):
        full_fit_fallback_reason = "full_fit_no_strict_new_gain"
    if full_fit_fallback_reason is not None:
        # OOF selection is necessary but not sufficient: the deployable full
        # refit must remain safe and produce a strict support-side new-class
        # gain.  Otherwise seal an exact D27 passthrough instead of aborting
        # the complete experiment matrix.
        selected = ["disabled"] * new_count
        widths.fill(0.0)
        amplitudes.fill(0.0)
        enabled = False
        full_adjusted = _apply_arrays(scores, old_count, widths, amplitudes)
        full_safe, _, full_evidence = _safety_and_rank(
            scores, full_adjusted, labels, classes, old_count, locked.objective
        )
        if not full_safe:
            raise ClasswiseSafeReleaseError("D29 disabled refit failed safety")
    if not enabled:
        widths.fill(0.0)
        amplitudes.fill(0.0)
    audit = {
        "schema": "cvs.phase2.d29_classwise_safe_release_fit_audit.v1",
        "selection_policy": (
            "shot_rank_5fold_per_class_safe_release_coordinate_search"
            if enabled
            else "oof_no_safe_new_gain_disabled_passthrough"
        ),
        "enabled": enabled,
        "k_shot": k_shot,
        "safety_budget": float(locked.safety_budget),
        "objective": locked.objective,
        "quantile_policies": list(QUANTILE_POLICIES),
        "selected_policy_by_new_class": dict(zip(classes[old_count:], selected)),
        "oof_evidence": oof_evidence,
        "full_refit_pre_disable_evidence": full_refit_evidence,
        "full_fit_fallback_reason": full_fit_fallback_reason,
        "full_support_evidence": full_evidence,
        "coordinate_trace": coordinate_trace,
        "fold_fit_evidence": fit_evidence,
        "full_fit": full_fit,
        "query_rows_used": 0,
        "query_batch_statistics_used": False,
    }
    return ClasswiseSafeReleaseState(
        schema=SCHEMA,
        registered_classes=classes,
        old_class_count=old_count,
        k_shot=k_shot,
        enabled=enabled,
        widths=widths,
        amplitudes=amplitudes,
        audit_json=_canonical_json(audit),
        config=locked,
    )


def apply_classwise_safe_release(
    state: ClasswiseSafeReleaseState, raw_scores: np.ndarray
) -> np.ndarray:
    """Apply D29 independently per row and preserve old score bytes."""

    scores = _validate_scores(
        raw_scores,
        class_count=len(state.registered_classes),
        name="D29 inference scores",
    )
    adjusted = _apply_arrays(
        scores, state.old_class_count, state.widths, state.amplitudes
    )
    if not np.array_equal(
        adjusted[:, : state.old_class_count], scores[:, : state.old_class_count]
    ):
        raise ClasswiseSafeReleaseError("D29 mutated old score columns")
    if not np.isfinite(adjusted).all():
        raise ClasswiseSafeReleaseError("D29 inference produced non-finite scores")
    return _readonly(adjusted, np.float32)


def predict_with_classwise_safe_release(
    state: ClasswiseSafeReleaseState, raw_scores: np.ndarray
) -> np.ndarray:
    adjusted = apply_classwise_safe_release(state, raw_scores)
    return np.asarray(state.registered_classes)[np.argmax(adjusted, axis=1)]


__all__ = [
    "ClasswiseSafeReleaseConfig",
    "ClasswiseSafeReleaseError",
    "ClasswiseSafeReleaseState",
    "OBJECTIVES",
    "QUANTILE_POLICIES",
    "apply_classwise_safe_release",
    "fit_classwise_safe_release",
    "predict_with_classwise_safe_release",
]
