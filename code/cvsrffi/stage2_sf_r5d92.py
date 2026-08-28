"""Support-only SF-TAPFT-R5D92-G selection primitives.

The module contains only registered-support operations.  Query IQ, query role,
and truth are deliberately absent from every interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ComplementarySupportSplit:
    fold: int
    fit_indices: tuple[int, ...]
    heldout_indices: tuple[int, ...]


@dataclass(frozen=True)
class R5D92FoldMetrics:
    candidate_id: str
    old_pre: float
    old_post: float
    new_acc: float
    harmonic_old_new: float
    forgetting: float
    min_old: float
    min_new: float
    old_to_new: float
    old_to_wrong_old: float
    fold_h: tuple[float, ...]
    covariance_positive_definite: bool
    covariance_condition_number: float
    identity_covariance_trace: float
    fft_covariance_trace: float
    error: str | None = None

    def __post_init__(self) -> None:
        numeric = (
            self.old_pre,
            self.old_post,
            self.new_acc,
            self.harmonic_old_new,
            self.forgetting,
            self.min_old,
            self.min_new,
            self.old_to_new,
            self.old_to_wrong_old,
            self.covariance_condition_number,
            self.identity_covariance_trace,
            self.fft_covariance_trace,
            *self.fold_h,
        )
        if not self.candidate_id.strip() or not self.fold_h:
            raise ValueError("R5D92 metrics require a candidate and fold H values")
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("R5D92 metrics must be finite")


@dataclass(frozen=True)
class R5D92SelectionDecision:
    selected_candidate_id: str
    fallback_used: bool
    feasible_candidate_ids: tuple[str, ...]
    lcb_h: Mapping[str, float]
    rejections: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lcb_h", MappingProxyType(dict(self.lcb_h)))
        object.__setattr__(self, "rejections", MappingProxyType(dict(self.rejections)))


@dataclass(frozen=True)
class R5D92SupportSelection:
    result: object
    pool: object
    baseline_candidate_id: str
    metrics: Mapping[str, R5D92FoldMetrics]
    decision: R5D92SelectionDecision
    skipped_candidate_ids: tuple[str, ...] = ()
    selection_wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


def complementary_leave_pair_splits(labels: Sequence[int]) -> tuple[ComplementarySupportSplit, ...]:
    """Return five deterministic K10 folds with two held rows per class."""

    targets = np.asarray(labels, dtype=np.int64)
    if targets.ndim != 1 or len(targets) == 0:
        raise ValueError("R5D92 labels must be a non-empty vector")
    classes = np.unique(targets)
    if not np.array_equal(classes, np.arange(len(classes), dtype=np.int64)):
        raise ValueError("R5D92 classes must be contiguous and zero based")
    class_rows = [np.flatnonzero(targets == class_id) for class_id in classes]
    if any(len(rows) != 10 for rows in class_rows):
        raise ValueError("R5D92 complementary folds are locked to K10")
    output = []
    all_indices = set(range(len(targets)))
    for fold in range(5):
        heldout = tuple(
            int(index)
            for rows in class_rows
            for index in rows[2 * fold : 2 * fold + 2]
        )
        fit = tuple(sorted(all_indices.difference(heldout)))
        output.append(
            ComplementarySupportSplit(
                fold=fold,
                fit_indices=fit,
                heldout_indices=tuple(sorted(heldout)),
            )
        )
    return tuple(output)


def _lcb(values: Sequence[float]) -> float:
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 1 or len(rows) == 0 or not np.isfinite(rows).all():
        raise ValueError("LCB requires finite fold values")
    if len(rows) == 1:
        return float(rows[0])
    return float(rows.mean() - 1.96 * rows.std(ddof=1) / math.sqrt(len(rows)))


def choose_r5d92_candidate(
    baseline: R5D92FoldMetrics,
    candidates: Sequence[R5D92FoldMetrics],
    *,
    epsilon_pre: float = 0.005,
    epsilon_old: float = 0.0,
    epsilon_new: float = 0.0,
) -> R5D92SelectionDecision:
    """Apply registration-safety constraints, then maximize fold LCB(H)."""

    tolerances = (float(epsilon_pre), float(epsilon_old), float(epsilon_new))
    if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
        raise ValueError("R5D92 tolerances must be finite and non-negative")
    rejection_rows: dict[str, tuple[str, ...]] = {}
    feasible = []
    lcb_rows = {baseline.candidate_id: _lcb(baseline.fold_h)}
    seen = {baseline.candidate_id}
    for candidate in candidates:
        if candidate.candidate_id in seen:
            raise ValueError("R5D92 candidate IDs must be unique")
        seen.add(candidate.candidate_id)
        reasons = []
        if candidate.error:
            reasons.append("d92_crossfit_error")
        if candidate.old_pre < baseline.old_pre - tolerances[0]:
            reasons.append("old_pre")
        if candidate.old_post < baseline.old_post:
            reasons.append("old_post")
        if candidate.forgetting > baseline.forgetting:
            reasons.append("forgetting")
        if candidate.min_old < baseline.min_old - tolerances[1]:
            reasons.append("min_old")
        if candidate.min_new < baseline.min_new - tolerances[2]:
            reasons.append("min_new")
        if not candidate.covariance_positive_definite:
            reasons.append("covariance_not_positive_definite")
        lcb_rows[candidate.candidate_id] = _lcb(candidate.fold_h)
        if reasons:
            rejection_rows[candidate.candidate_id] = tuple(reasons)
        else:
            feasible.append(candidate)
    if not feasible:
        return R5D92SelectionDecision(
            selected_candidate_id=baseline.candidate_id,
            fallback_used=True,
            feasible_candidate_ids=(),
            lcb_h=lcb_rows,
            rejections=rejection_rows,
        )
    feasible.sort(
        key=lambda item: (
            lcb_rows[item.candidate_id],
            item.min_old,
            item.min_new,
            -item.forgetting,
            item.old_post,
        ),
        reverse=True,
    )
    return R5D92SelectionDecision(
        selected_candidate_id=feasible[0].candidate_id,
        fallback_used=False,
        feasible_candidate_ids=tuple(item.candidate_id for item in candidates if item in feasible),
        lcb_h=lcb_rows,
        rejections=rejection_rows,
    )


def covariance_block_audit(
    covariance: np.ndarray, *, block_dims: Sequence[int] = (160, 96)
) -> dict[str, object]:
    """Audit positive definiteness, conditioning, and feature-block traces."""

    matrix = np.asarray(covariance, dtype=np.float64)
    dims = tuple(int(value) for value in block_dims)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or sum(dims) != matrix.shape[0]
        or any(value <= 0 for value in dims)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("covariance audit geometry is invalid")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    minimum = float(np.min(eigenvalues))
    maximum = float(np.max(eigenvalues))
    positive = minimum > 0.0
    condition = float(maximum / minimum) if positive else float("inf")
    traces = []
    start = 0
    for dimension in dims:
        stop = start + dimension
        traces.append(float(np.trace(symmetric[start:stop, start:stop])))
        start = stop
    return {
        "positive_definite": positive,
        "eigenvalue_min": minimum,
        "eigenvalue_max": maximum,
        "condition_number": condition,
        "block_traces": traces,
    }


def _class_floor(predictions: np.ndarray, truth: np.ndarray, classes: Sequence[int]) -> float:
    values = []
    for class_id in classes:
        mask = truth == int(class_id)
        if not np.any(mask):
            raise ValueError("R5D92 cross-fit is missing a held-out class")
        values.append(float(np.mean(predictions[mask] == truth[mask])))
    return float(min(values))


def evaluate_d92_crossfit_from_fold_features(
    candidate_id: str,
    fold_identity160: Sequence[np.ndarray],
    registered_received_iq: np.ndarray,
    registered_labels: Sequence[int],
    *,
    splits: Sequence[ComplementarySupportSplit],
    old_class_count: int,
    seed: int,
    device: object = "cpu",
) -> R5D92FoldMetrics:
    """Evaluate one adaptation candidate on held-out old/new support through D92 E0."""

    from cvsrffi.stage2_sf_erbt_four_state import fit_erbt_registration_pair
    from cvsrffi.stage2_sf_erbt_oldonly import make_fft96

    labels = np.asarray(registered_labels, dtype=np.int64)
    received = np.asarray(registered_received_iq, dtype=np.float32)
    identities = tuple(np.asarray(value, dtype=np.float32) for value in fold_identity160)
    split_rows = tuple(splits)
    old_count = int(old_class_count)
    classes = np.unique(labels)
    if (
        not candidate_id.strip()
        or old_count != 6
        or labels.ndim != 1
        or received.shape != (len(labels), 2, 256)
        or len(identities) != len(split_rows)
        or len(split_rows) != 5
        or not np.array_equal(classes, np.arange(len(classes), dtype=np.int64))
        or len(classes) <= old_count
        or np.bincount(labels, minlength=len(classes)).tolist() != [10] * len(classes)
        or any(value.shape != (len(labels), 160) or not np.isfinite(value).all() for value in identities)
    ):
        raise ValueError("R5D92 cross-fit input geometry drift")
    fft96 = make_fft96(received)
    old_pre_predictions = []
    old_post_predictions = []
    old_truth_rows = []
    new_predictions = []
    new_truth_rows = []
    fold_h = []
    condition_numbers = []
    identity_traces = []
    fft_traces = []
    positive_flags = []
    for fold, (identity, split) in enumerate(zip(identities, split_rows)):
        fit_index = np.asarray(split.fit_indices, dtype=np.int64)
        heldout_index = np.asarray(split.heldout_indices, dtype=np.int64)
        if (
            split.fold != fold
            or len(fit_index) != 8 * len(classes)
            or len(heldout_index) != 2 * len(classes)
        ):
            raise ValueError("R5D92 complementary split drift")
        old_fit = fit_index[labels[fit_index] < old_count]
        old_heldout = heldout_index[labels[heldout_index] < old_count]
        new_heldout = heldout_index[labels[heldout_index] >= old_count]
        reg0, reg1, _ = fit_erbt_registration_pair(
            identity[old_fit],
            fft96[old_fit],
            labels[old_fit],
            identity[fit_index],
            fft96[fit_index],
            labels[fit_index],
            old_class_ids=tuple(range(old_count)),
            registered_class_ids=tuple(range(len(classes))),
            seed=int(seed) + fold,
            device=device,
        )
        pre = reg0.predict(identity[old_heldout], fft96[old_heldout])
        post = reg1.predict(identity[old_heldout], fft96[old_heldout])
        new = reg1.predict(identity[new_heldout], fft96[new_heldout])
        old_truth = labels[old_heldout]
        new_truth = labels[new_heldout]
        old_pre_predictions.append(pre)
        old_post_predictions.append(post)
        old_truth_rows.append(old_truth)
        new_predictions.append(new)
        new_truth_rows.append(new_truth)
        fold_old = float(np.mean(post == old_truth))
        fold_new = float(np.mean(new == new_truth))
        fold_h.append(
            0.0 if fold_old + fold_new == 0.0 else 2.0 * fold_old * fold_new / (fold_old + fold_new)
        )
        audit = reg1.audit
        minimum = float(audit["d92_balanced_eigenvalue_min"])
        maximum = float(audit["d92_balanced_eigenvalue_max"])
        positive_flags.append(minimum > 0.0)
        condition_numbers.append(maximum / minimum if minimum > 0.0 else float("inf"))
        block_traces = tuple(float(value) for value in audit["d92_covariance_block_traces"])
        if len(block_traces) != 2:
            raise ValueError("R5D92 D92 block trace geometry drift")
        identity_traces.append(block_traces[0])
        fft_traces.append(block_traces[1])
    old_truth = np.concatenate(old_truth_rows)
    new_truth = np.concatenate(new_truth_rows)
    old_pre_prediction = np.concatenate(old_pre_predictions)
    old_post_prediction = np.concatenate(old_post_predictions)
    new_prediction = np.concatenate(new_predictions)
    old_pre = float(np.mean(old_pre_prediction == old_truth))
    old_post = float(np.mean(old_post_prediction == old_truth))
    new_acc = float(np.mean(new_prediction == new_truth))
    harmonic = 0.0 if old_post + new_acc == 0.0 else 2.0 * old_post * new_acc / (old_post + new_acc)
    return R5D92FoldMetrics(
        candidate_id=candidate_id,
        old_pre=old_pre,
        old_post=old_post,
        new_acc=new_acc,
        harmonic_old_new=harmonic,
        forgetting=old_pre - old_post,
        min_old=_class_floor(old_post_prediction, old_truth, range(old_count)),
        min_new=_class_floor(new_prediction, new_truth, range(old_count, len(classes))),
        old_to_new=float(np.mean(old_post_prediction >= old_count)),
        old_to_wrong_old=float(
            np.mean((old_post_prediction < old_count) & (old_post_prediction != old_truth))
        ),
        fold_h=tuple(float(value) for value in fold_h),
        covariance_positive_definite=all(positive_flags),
        covariance_condition_number=float(max(condition_numbers)),
        identity_covariance_trace=float(np.mean(identity_traces)),
        fft_covariance_trace=float(np.mean(fft_traces)),
    )


def run_r5d92_support_selection(
    checkpoint_model: object,
    old_support: object,
    registered_received_iq: np.ndarray,
    registered_labels: Sequence[int],
    config: object,
    *,
    steps: Sequence[int] = (250, 350, 520),
    polish_steps: int = 30,
    seed: int,
    device: object,
    identity_extractor: object | None = None,
    soft_budget_seconds: float = 240.0,
) -> R5D92SupportSelection:
    """Run five support trajectories, D92-check baseline plus Top-2, and commit one model."""

    import copy
    import time

    from cvsrffi.target_only_progressive_adapt import (
        fit_sf_tapft,
        fit_sf_tapft_r5_candidate_pool,
        polish_sf_tapft_r5_candidate,
    )

    if identity_extractor is None:
        from cvsrffi.stage2_sf_erbt_oldonly import _extract_identity160

        identity_extractor = _extract_identity160
    started = time.monotonic()
    if not math.isfinite(float(soft_budget_seconds)) or float(soft_budget_seconds) <= 0:
        raise ValueError("R5D92 soft budget must be positive and finite")
    pool = fit_sf_tapft_r5_candidate_pool(
        checkpoint_model,
        old_support,
        config,
        steps=steps,
        alphas=(0.75, 1.0),
    )
    registered_splits = complementary_leave_pair_splits(registered_labels)
    maximum_step = max(int(value) for value in steps)
    baseline_pool_id = f"S{maximum_step:03d}_A100"
    if baseline_pool_id not in pool.candidates:
        raise RuntimeError("R5D92 baseline trajectory is missing")

    def evaluate(candidate_id: str, fold_results: Sequence[object]) -> R5D92FoldMetrics:
        identities = tuple(
            np.asarray(
                identity_extractor(result.model, registered_received_iq, device),
                dtype=np.float32,
            )
            for result in fold_results
        )
        return evaluate_d92_crossfit_from_fold_features(
            candidate_id,
            identities,
            registered_received_iq,
            registered_labels,
            splits=registered_splits,
            old_class_count=6,
            seed=int(seed),
            device=device,
        )

    baseline_id = "J1_R0_D92"
    baseline_metrics = evaluate(
        baseline_id, pool.candidates[baseline_pool_id].fold_results
    )
    candidate_metrics = []
    metrics: dict[str, R5D92FoldMetrics] = {baseline_id: baseline_metrics}
    skipped = []
    for index, candidate_id in enumerate(pool.top_candidate_ids):
        if index > 0 and time.monotonic() - started >= float(soft_budget_seconds):
            skipped.extend(pool.top_candidate_ids[index:])
            break
        try:
            row = evaluate(candidate_id, pool.candidates[candidate_id].fold_results)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            row = R5D92FoldMetrics(
                candidate_id=candidate_id,
                old_pre=0.0,
                old_post=0.0,
                new_acc=0.0,
                harmonic_old_new=0.0,
                forgetting=1.0,
                min_old=0.0,
                min_new=0.0,
                old_to_new=1.0,
                old_to_wrong_old=1.0,
                fold_h=(0.0,) * 5,
                covariance_positive_definite=False,
                covariance_condition_number=float(np.finfo(np.float64).max),
                identity_covariance_trace=0.0,
                fft_covariance_trace=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        metrics[candidate_id] = row
        candidate_metrics.append(row)
    decision = choose_r5d92_candidate(baseline_metrics, candidate_metrics)
    if decision.fallback_used:
        baseline_config = config.__class__(
            **{**config.__dict__, "validation_steps": (), "rse_snapshot_steps": ()}
        )
        result = fit_sf_tapft(
            copy.deepcopy(checkpoint_model),
            old_support,
            baseline_config,
            checkpoint_selection_mode="final_step",
        )
    else:
        selected = pool.candidates[decision.selected_candidate_id].averaged_result
        result = polish_sf_tapft_r5_candidate(
            checkpoint_model,
            old_support,
            config,
            selected,
            polish_steps=int(polish_steps),
        )
    return R5D92SupportSelection(
        result=result,
        pool=pool,
        baseline_candidate_id=baseline_id,
        metrics=metrics,
        decision=decision,
        skipped_candidate_ids=tuple(skipped),
        selection_wall_seconds=float(time.monotonic() - started),
    )


__all__ = [
    "ComplementarySupportSplit",
    "R5D92FoldMetrics",
    "R5D92SelectionDecision",
    "R5D92SupportSelection",
    "choose_r5d92_candidate",
    "complementary_leave_pair_splits",
    "covariance_block_audit",
    "evaluate_d92_crossfit_from_fold_features",
    "run_r5d92_support_selection",
]
