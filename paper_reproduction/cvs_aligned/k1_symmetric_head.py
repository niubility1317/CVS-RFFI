"""Role-symmetric, query-free head for one-physical-shot CVS enrollment.

The head treats receive-channel views of one labeled physical support sample as
correlated observations, never as additional shots.  Hyperparameters are
selected by leaving out support views only.  No target-query statistic, old/new
role, class quota, or batch graph enters fitting or scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EPS = 1.0e-8
PROTOTYPE_RULES = ("mean", "trimmed20", "medoid")
DEFAULT_RIDGES: tuple[float | None, ...] = (None, 0.03, 0.1, 0.3, 1.0)


def _normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.maximum(
        np.linalg.norm(values, axis=-1, keepdims=True), EPS
    )


def _validate_support_views(
    support_views: np.ndarray, *, min_views: int = 3
) -> np.ndarray:
    values = np.asarray(support_views, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("support_views must have shape [V,C,D]")
    if (
        values.shape[0] < int(min_views)
        or values.shape[1] < 2
        or values.shape[2] < 2
    ):
        raise ValueError(
            f"support_views require V>={int(min_views)}, C>=2, D>=2"
        )
    if not np.isfinite(values).all():
        raise FloatingPointError("support_views contain non-finite values")
    return values


@dataclass(frozen=True)
class DiagonalAlignment:
    """Global diagonal target-to-source alignment shared by every class."""

    target_mean: np.ndarray
    target_std: np.ndarray
    source_mean: np.ndarray
    source_std: np.ndarray
    variance_floor: float

    @property
    def scale(self) -> np.ndarray:
        return (self.source_std / self.target_std).astype(np.float32)

    @property
    def bias(self) -> np.ndarray:
        return (self.source_mean - self.target_mean * self.scale).astype(np.float32)

    @property
    def persistent_state_bytes_fp16(self) -> int:
        # Deployment stores the already-composed affine only.  Four moment
        # vectors remain calibration provenance, not persistent flight state.
        return int(2 * (self.scale.size + self.bias.size))


@dataclass(frozen=True)
class SymmetricK1Head:
    """Deployment state for independent per-query scoring."""

    prototypes: np.ndarray
    score_transform: np.ndarray
    alignment: DiagonalAlignment | None
    prototype_rule: str
    ridge: float | None
    calibration: dict[str, Any]

    @property
    def class_count(self) -> int:
        return int(self.prototypes.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.prototypes.shape[1])

    @property
    def persistent_state_bytes_fp16(self) -> int:
        alignment_bytes = (
            0
            if self.alignment is None
            else self.alignment.persistent_state_bytes_fp16
        )
        return int(
            2 * (self.prototypes.size + self.score_transform.size)
            + alignment_bytes
        )

    @property
    def extra_macs_per_query(self) -> int:
        # The diagonal affine is counted separately from the frozen-backbone
        # cosine head; Gram correction is a tiny C-by-C matrix multiply.
        alignment_macs = self.feature_dim if self.alignment is not None else 0
        return int(self.class_count * self.class_count + alignment_macs)


def fit_diagonal_alignment(
    support_rows: np.ndarray,
    *,
    source_mean: np.ndarray | None = None,
    source_std: np.ndarray | None = None,
    variance_floor: float = 0.05,
) -> DiagonalAlignment:
    """Fit one global diagonal affine from support and optional source stats."""

    rows = np.asarray(support_rows, dtype=np.float32).reshape(
        -1, np.asarray(support_rows).shape[-1]
    )
    if rows.shape[0] < 2 or rows.shape[1] < 2 or not np.isfinite(rows).all():
        raise ValueError("support_rows must be finite with shape [N>=2,D>=2]")
    floor = float(variance_floor)
    if not 0.0 < floor <= 1.0:
        raise ValueError("variance_floor must be in (0,1]")
    target_mean = rows.mean(axis=0).astype(np.float32)
    target_std = np.maximum(rows.std(axis=0), floor).astype(np.float32)
    if source_mean is None and source_std is None:
        reference_mean = np.zeros_like(target_mean)
        reference_std = np.ones_like(target_std)
    elif source_mean is None or source_std is None:
        raise ValueError("source_mean and source_std must be provided together")
    else:
        reference_mean = np.asarray(source_mean, dtype=np.float32).reshape(-1)
        reference_std = np.asarray(source_std, dtype=np.float32).reshape(-1)
        if reference_mean.shape != target_mean.shape or reference_std.shape != target_std.shape:
            raise ValueError("source statistics do not match support feature dimension")
        if not np.isfinite(reference_mean).all() or not np.isfinite(reference_std).all():
            raise FloatingPointError("source statistics contain non-finite values")
        reference_std = np.maximum(reference_std, floor).astype(np.float32)
    return DiagonalAlignment(
        target_mean=target_mean,
        target_std=target_std,
        source_mean=reference_mean,
        source_std=reference_std,
        variance_floor=floor,
    )


def apply_diagonal_alignment(
    features: np.ndarray, alignment: DiagonalAlignment | None
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if alignment is None:
        return values
    if values.shape[-1] != alignment.target_mean.size:
        raise ValueError("feature dimension does not match diagonal alignment")
    return (values * alignment.scale + alignment.bias).astype(np.float32)


def build_robust_prototypes(
    support_views: np.ndarray, *, rule: str
) -> np.ndarray:
    """Build one spherical prototype per class using the same robust rule."""

    values = _normalize(_validate_support_views(support_views, min_views=1))
    if rule not in PROTOTYPE_RULES:
        raise ValueError(f"unsupported prototype rule: {rule}")
    rows = []
    for class_index in range(values.shape[1]):
        observations = values[:, class_index, :]
        if rule == "mean":
            center = observations.mean(axis=0)
        elif rule == "trimmed20":
            initial = _normalize(observations.mean(axis=0, keepdims=True))[0]
            keep_count = max(2, int(np.ceil(0.8 * len(observations))))
            keep = np.argsort(observations @ initial)[-keep_count:]
            center = observations[keep].mean(axis=0)
        else:
            similarities = observations @ observations.T
            center = observations[int(np.argmax(similarities.mean(axis=1)))]
        rows.append(_normalize(np.asarray(center)[None, :])[0])
    return np.stack(rows).astype(np.float32)


def fit_gram_score_transform(
    prototypes: np.ndarray, *, ridge: float | None
) -> np.ndarray:
    """Return a small class-space transform that suppresses prototype overlap."""

    banks = _normalize(np.asarray(prototypes, dtype=np.float32))
    if banks.ndim != 2 or banks.shape[0] < 2:
        raise ValueError("prototypes must have shape [C>=2,D]")
    if ridge is None:
        return np.eye(banks.shape[0], dtype=np.float32)
    value = float(ridge)
    if value <= 0.0:
        raise ValueError("ridge must be positive or None")
    gram = banks @ banks.T
    transform = np.linalg.solve(
        gram + value * np.eye(len(gram), dtype=np.float32),
        np.eye(len(gram), dtype=np.float32),
    )
    return transform.astype(np.float32)


def score_symmetric_head(features: np.ndarray, head: SymmetricK1Head) -> np.ndarray:
    """Score rows independently; no query batch statistic is computed."""

    rows = apply_diagonal_alignment(features, head.alignment)
    if rows.ndim < 2 or rows.shape[-1] != head.feature_dim:
        raise ValueError("features must end in the head feature dimension")
    flat = _normalize(rows.reshape(-1, rows.shape[-1]))
    raw = flat @ _normalize(head.prototypes).T
    corrected = raw @ np.asarray(head.score_transform, dtype=np.float32)
    return corrected.reshape(*rows.shape[:-1], head.class_count).astype(np.float32)


def leave_one_support_view_out_scores(
    support_views: np.ndarray,
    *,
    use_alignment: bool,
    prototype_rule: str,
    ridge: float | None,
    source_mean: np.ndarray | None = None,
    source_std: np.ndarray | None = None,
    variance_floor: float = 0.05,
) -> np.ndarray:
    """Return legal support calibration scores shaped ``[V,C,C]``."""

    values = _validate_support_views(support_views)
    outputs = []
    for heldout_view in range(values.shape[0]):
        train = np.delete(values, heldout_view, axis=0)
        validation = values[heldout_view]
        alignment = (
            fit_diagonal_alignment(
                train,
                source_mean=source_mean,
                source_std=source_std,
                variance_floor=variance_floor,
            )
            if bool(use_alignment)
            else None
        )
        train_aligned = apply_diagonal_alignment(train, alignment)
        prototypes = build_robust_prototypes(
            train_aligned, rule=str(prototype_rule)
        )
        transform = fit_gram_score_transform(prototypes, ridge=ridge)
        fold_head = SymmetricK1Head(
            prototypes=prototypes,
            score_transform=transform,
            alignment=alignment,
            prototype_rule=str(prototype_rule),
            ridge=ridge,
            calibration={},
        )
        outputs.append(score_symmetric_head(validation, fold_head))
    return np.stack(outputs).astype(np.float32)


def _candidate_key(candidate: dict[str, Any]) -> tuple[float, float, float, int, int]:
    # Accuracy and class floor dominate.  Ties prefer no alignment, no Gram
    # correction, then the simplest prototype rule.
    return (
        float(candidate["accuracy"]),
        float(candidate["min_class_accuracy"]),
        -float(candidate["mean_true_class_rank"]),
        -int(bool(candidate["use_alignment"])),
        -int(candidate["complexity_rank"]),
    )


def calibrate_symmetric_k1_head(
    support_views: np.ndarray,
    *,
    source_mean: np.ndarray | None = None,
    source_std: np.ndarray | None = None,
    prototype_rules: Iterable[str] = PROTOTYPE_RULES,
    ridges: Iterable[float | None] = DEFAULT_RIDGES,
    allow_alignment: bool = True,
    variance_floor: float = 0.05,
) -> dict[str, Any]:
    """Select the head using leave-one-support-view-out predictions only."""

    values = _validate_support_views(support_views)
    rules = tuple(str(rule) for rule in prototype_rules)
    ridge_grid = tuple(ridges)
    if not rules or not ridge_grid:
        raise ValueError("prototype and ridge grids must be non-empty")
    candidates: list[dict[str, Any]] = []
    alignment_grid = (False, True) if allow_alignment else (False,)
    for use_alignment in alignment_grid:
        for rule_index, rule in enumerate(rules):
            if rule not in PROTOTYPE_RULES:
                raise ValueError(f"unsupported prototype rule: {rule}")
            for ridge_index, ridge in enumerate(ridge_grid):
                predictions: list[int] = []
                truths: list[int] = []
                true_ranks: list[int] = []
                for heldout_view in range(values.shape[0]):
                    train = np.delete(values, heldout_view, axis=0)
                    validation = values[heldout_view]
                    alignment = (
                        fit_diagonal_alignment(
                            train,
                            source_mean=source_mean,
                            source_std=source_std,
                            variance_floor=variance_floor,
                        )
                        if use_alignment
                        else None
                    )
                    train_aligned = apply_diagonal_alignment(train, alignment)
                    prototypes = build_robust_prototypes(train_aligned, rule=rule)
                    transform = fit_gram_score_transform(prototypes, ridge=ridge)
                    fold_head = SymmetricK1Head(
                        prototypes=prototypes,
                        score_transform=transform,
                        alignment=alignment,
                        prototype_rule=rule,
                        ridge=ridge,
                        calibration={},
                    )
                    scores = score_symmetric_head(validation, fold_head)
                    fold_truth = np.arange(values.shape[1], dtype=np.int64)
                    fold_prediction = np.argmax(scores, axis=1)
                    predictions.extend(fold_prediction.tolist())
                    truths.extend(fold_truth.tolist())
                    ordering = np.argsort(-scores, axis=1)
                    true_ranks.extend(
                        (
                            np.argmax(ordering == fold_truth[:, None], axis=1) + 1
                        ).tolist()
                    )
                pred = np.asarray(predictions, dtype=np.int64)
                truth = np.asarray(truths, dtype=np.int64)
                class_acc = [
                    float(np.mean(pred[truth == index] == index))
                    for index in range(values.shape[1])
                ]
                candidates.append(
                    {
                        "use_alignment": bool(use_alignment),
                        "prototype_rule": rule,
                        "ridge": None if ridge is None else float(ridge),
                        "accuracy": float(np.mean(pred == truth)),
                        "min_class_accuracy": float(min(class_acc)),
                        "mean_true_class_rank": float(np.mean(true_ranks)),
                        "view_count": int(values.shape[0]),
                        "class_count": int(values.shape[1]),
                        "physical_shots_per_class": 1,
                        "complexity_rank": int(
                            2 * int(use_alignment)
                            + int(ridge is not None)
                            + rule_index
                            + ridge_index
                        ),
                    }
                )
    selected = max(candidates, key=_candidate_key)
    return {
        "selection_source": "support_view_leave_one_out_only",
        "query_rows_used": 0,
        "role_labels_used": False,
        "class_quota_used": False,
        "physical_shots_per_class": 1,
        "selected": dict(selected),
        "candidates": candidates,
    }


def fit_symmetric_k1_head(
    support_views: np.ndarray,
    *,
    source_mean: np.ndarray | None = None,
    source_std: np.ndarray | None = None,
    prototype_rules: Iterable[str] = PROTOTYPE_RULES,
    ridges: Iterable[float | None] = DEFAULT_RIDGES,
    allow_alignment: bool = True,
    variance_floor: float = 0.05,
) -> SymmetricK1Head:
    """Calibrate and fit the final all-view, one-physical-shot head."""

    values = _validate_support_views(support_views)
    calibration = calibrate_symmetric_k1_head(
        values,
        source_mean=source_mean,
        source_std=source_std,
        prototype_rules=prototype_rules,
        ridges=ridges,
        allow_alignment=allow_alignment,
        variance_floor=variance_floor,
    )
    selected = calibration["selected"]
    alignment = (
        fit_diagonal_alignment(
            values,
            source_mean=source_mean,
            source_std=source_std,
            variance_floor=variance_floor,
        )
        if bool(selected["use_alignment"])
        else None
    )
    aligned = apply_diagonal_alignment(values, alignment)
    prototypes = build_robust_prototypes(
        aligned, rule=str(selected["prototype_rule"])
    )
    transform = fit_gram_score_transform(prototypes, ridge=selected["ridge"])
    return SymmetricK1Head(
        prototypes=prototypes,
        score_transform=transform,
        alignment=alignment,
        prototype_rule=str(selected["prototype_rule"]),
        ridge=selected["ridge"],
        calibration=calibration,
    )


def fit_locked_symmetric_support_head(
    support_observations: np.ndarray,
    *,
    physical_shots_per_class: int,
    selected: dict[str, Any],
    source_mean: np.ndarray | None = None,
    source_std: np.ndarray | None = None,
    variance_floor: float = 0.05,
) -> SymmetricK1Head:
    """Fit support statistics with source-locked hyperparameters for any K."""

    values = _validate_support_views(support_observations)
    shots = int(physical_shots_per_class)
    if shots < 1 or values.shape[0] != 3 * shots:
        raise ValueError(
            "formal support observations must be exactly three leo_weak views per physical K"
        )
    required = {"use_alignment", "prototype_rule", "ridge"}
    if not required.issubset(selected):
        raise ValueError(f"locked head config is missing: {sorted(required - set(selected))}")
    use_alignment = bool(selected["use_alignment"])
    alignment = (
        fit_diagonal_alignment(
            values,
            source_mean=source_mean,
            source_std=source_std,
            variance_floor=variance_floor,
        )
        if use_alignment
        else None
    )
    aligned = apply_diagonal_alignment(values, alignment)
    prototypes = build_robust_prototypes(
        aligned, rule=str(selected["prototype_rule"])
    )
    ridge = selected["ridge"]
    transform = fit_gram_score_transform(prototypes, ridge=ridge)
    return SymmetricK1Head(
        prototypes=prototypes,
        score_transform=transform,
        alignment=alignment,
        prototype_rule=str(selected["prototype_rule"]),
        ridge=ridge,
        calibration={
            "selection_source": "source_receiver_holdout_locked",
            "query_rows_used": 0,
            "target_support_used_for_hyperparameter_selection": False,
            "role_labels_used": False,
            "class_quota_used": False,
            "physical_shots_per_class": shots,
            "support_observations_per_class": int(values.shape[0]),
            "selected": dict(selected),
        },
    )


def quantize_symmetric_head_fp16(head: SymmetricK1Head) -> SymmetricK1Head:
    """Round the exact deployment tensors to FP16, then score from that state."""

    alignment = None
    if head.alignment is not None:
        scale = head.alignment.scale.astype(np.float16).astype(np.float32)
        bias = head.alignment.bias.astype(np.float16).astype(np.float32)
        alignment = DiagonalAlignment(
            target_mean=np.zeros_like(scale),
            target_std=np.ones_like(scale),
            source_mean=bias,
            source_std=scale,
            variance_floor=float(head.alignment.variance_floor),
        )
    return SymmetricK1Head(
        prototypes=head.prototypes.astype(np.float16).astype(np.float32),
        score_transform=head.score_transform.astype(np.float16).astype(np.float32),
        alignment=alignment,
        prototype_rule=head.prototype_rule,
        ridge=head.ridge,
        calibration=head.calibration,
    )


def persist_and_reload_symmetric_head_fp16(
    head: SymmetricK1Head, path: str | Path
) -> tuple[SymmetricK1Head, dict[str, Any]]:
    """Persist the exact flight tensors and reconstruct scoring state from disk."""

    quantized = quantize_symmetric_head_fp16(head)
    state: dict[str, np.ndarray] = {
        "prototypes": quantized.prototypes.astype(np.float16),
        "score_transform": quantized.score_transform.astype(np.float16),
    }
    if quantized.alignment is not None:
        state["alignment_scale"] = quantized.alignment.scale.astype(np.float16)
        state["alignment_bias"] = quantized.alignment.bias.astype(np.float16)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **state)
    with np.load(output, allow_pickle=False) as payload:
        prototypes = np.asarray(payload["prototypes"], dtype=np.float32)
        score_transform = np.asarray(payload["score_transform"], dtype=np.float32)
        if "alignment_scale" in payload.files:
            scale = np.asarray(payload["alignment_scale"], dtype=np.float32)
            bias = np.asarray(payload["alignment_bias"], dtype=np.float32)
            alignment = DiagonalAlignment(
                target_mean=np.zeros_like(scale),
                target_std=np.ones_like(scale),
                source_mean=bias,
                source_std=scale,
                variance_floor=(
                    float(quantized.alignment.variance_floor)
                    if quantized.alignment is not None
                    else 0.05
                ),
            )
        else:
            alignment = None
    reloaded = SymmetricK1Head(
        prototypes=prototypes,
        score_transform=score_transform,
        alignment=alignment,
        prototype_rule=head.prototype_rule,
        ridge=head.ridge,
        calibration=head.calibration,
    )
    probe = head.prototypes[: min(8, head.class_count)]
    before = np.argmax(score_symmetric_head(probe, quantized), axis=-1)
    after = np.argmax(score_symmetric_head(probe, reloaded), axis=-1)
    if not np.array_equal(before, after):
        raise RuntimeError("saved/reloaded FP16 head changes probe predictions")
    return reloaded, {
        "path": str(output),
        "tensor_bytes_fp16": int(reloaded.persistent_state_bytes_fp16),
        "prediction_parity_pass": True,
        "storage_dtype": "fp16",
        "scoring_dtype": "fp32_after_fp16_reload",
    }
