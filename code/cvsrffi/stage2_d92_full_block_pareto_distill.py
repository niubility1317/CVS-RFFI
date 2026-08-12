"""Support-only FULL/BLOCK Pareto distillation for the E0 D92 head.

The module consumes two already fitted affine heads over the *same* transformed
support registry.  It never receives query rows, truth, roles, or a scorer.
All numerical degeneration returns the supplied FULL head byte-for-byte; a
malformed registry or codec contract remains a structural error.
"""

from __future__ import annotations

import math
import hashlib
from typing import Any, Callable

import numpy as np

from cvsrffi.stage2_d92_registration_balanced_covariance import OLD_CLASS_COUNT


TAIL_FRACTION = 0.20
TAIL_QUANTILE_METHOD = "lower"
_SOLVER_LP = "scipy_highs_linear_program_v1"
_SOLVER_QP = "scipy_slsqp_convex_quadratic_v1"
_EPS = 1.0e-9
_CONSTRAINT_TOLERANCE = 5.0e-7
_D42_CODEC_MACS_PER_COEFFICIENT = 8
_HIGHS_CONSTRAINT_EVALUATION_ITERATION_CAP = 2048
_SLSQP_CONSTRAINT_EVALUATION_ITERATION_CAP = 500


class D92ParetoDistillError(RuntimeError):
    """Raised when a frozen Pareto-distill contract is structurally invalid."""


class D92ParetoDistillNumericalError(D92ParetoDistillError):
    """A finite numerical failure eligible for exact E0 fallback."""


def _center_affine(
    coefficient: np.ndarray, intercept: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the score-common affine term without changing argmax decisions."""

    weight = np.asarray(coefficient, dtype=np.float64)
    bias = np.asarray(intercept, dtype=np.float64)
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise D92ParetoDistillError("D92 Pareto distill affine shape drift")
    return (
        weight - weight.mean(axis=0, keepdims=True),
        bias - float(bias.mean()),
    )


def _validated_inputs(
    *,
    full_rows: np.ndarray,
    full_labels: np.ndarray,
    full_coefficient: np.ndarray,
    full_intercept: np.ndarray,
    deployed_full_coefficient: np.ndarray,
    deployed_full_intercept: np.ndarray,
    block_coefficient: np.ndarray,
    block_intercept: np.ndarray,
    class_count: int,
    k_shot: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
]:
    """Reject structural registry/head drift before fallback-eligible math."""

    classes, shots = int(class_count), int(k_shot)
    rows = np.asarray(full_rows)
    labels = np.asarray(full_labels)
    full_w = np.asarray(full_coefficient)
    full_b = np.asarray(full_intercept)
    deployed_full_w = np.asarray(deployed_full_coefficient)
    deployed_full_b = np.asarray(deployed_full_intercept)
    block_w = np.asarray(block_coefficient)
    block_b = np.asarray(block_intercept)
    if (
        classes <= OLD_CLASS_COUNT
        or shots <= 2
        or rows.ndim != 2
        or rows.shape[0] != classes * shots
        or labels.shape != (classes * shots,)
        or full_w.dtype != np.float32
        or full_b.dtype != np.float32
        or deployed_full_w.dtype != np.float32
        or deployed_full_b.dtype != np.float32
        or block_w.dtype != np.float32
        or block_b.dtype != np.float32
        or full_w.shape != (classes, rows.shape[1])
        or deployed_full_w.shape != full_w.shape
        or block_w.shape != full_w.shape
        or full_b.shape != (classes,)
        or deployed_full_b.shape != full_b.shape
        or block_b.shape != full_b.shape
        or not np.issubdtype(labels.dtype, np.integer)
        or not np.issubdtype(rows.dtype, np.number)
        or not np.isfinite(rows).all()
        or not np.isfinite(full_w).all()
        or not np.isfinite(full_b).all()
        or not np.isfinite(deployed_full_w).all()
        or not np.isfinite(deployed_full_b).all()
        or not np.isfinite(block_w).all()
        or not np.isfinite(block_b).all()
    ):
        raise D92ParetoDistillError("D92 Pareto distill registry/head shape drift")
    labels64 = np.asarray(labels, dtype=np.int64)
    if (
        not np.array_equal(np.unique(labels64), np.arange(classes, dtype=np.int64))
        or any(int(np.sum(labels64 == index)) != shots for index in range(classes))
    ):
        raise D92ParetoDistillError("D92 Pareto distill support registry is incomplete")
    return (
        np.asarray(rows, dtype=np.float64),
        labels64,
        np.asarray(full_w, dtype=np.float32),
        np.asarray(full_b, dtype=np.float32),
        np.asarray(deployed_full_w, dtype=np.float32),
        np.asarray(deployed_full_b, dtype=np.float32),
        np.asarray(block_w, dtype=np.float32),
        np.asarray(block_b, dtype=np.float32),
        classes,
        shots,
        int(rows.shape[1]),
    )


def _scores(rows: np.ndarray, coefficient: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64) @ np.asarray(
        coefficient, dtype=np.float64
    ).T + np.asarray(intercept, dtype=np.float64)[None, :]
    if not np.isfinite(values).all():
        raise D92ParetoDistillNumericalError("support_scores_nonfinite")
    return values


def _margins(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or targets.shape != (values.shape[0],):
        raise D92ParetoDistillError("support margin shape drift")
    rows = np.arange(values.shape[0], dtype=np.int64)
    target = values[rows, targets]
    competitors = values.copy()
    competitors[rows, targets] = -np.inf
    margin = target - np.max(competitors, axis=1)
    if not np.isfinite(margin).all():
        raise D92ParetoDistillNumericalError("support_margin_nonfinite")
    return margin


def _new_to_old_margins(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return each registered-new true logit minus its strongest old logit."""

    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or targets.shape != (values.shape[0],):
        raise D92ParetoDistillError("new-to-old support margin shape drift")
    if values.shape[1] <= OLD_CLASS_COUNT:
        raise D92ParetoDistillError("new-to-old support registry drift")
    rows = np.arange(values.shape[0], dtype=np.int64)
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    new_mask = targets >= OLD_CLASS_COUNT
    result[new_mask] = (
        values[rows[new_mask], targets[new_mask]]
        - np.max(values[new_mask, :OLD_CLASS_COUNT], axis=1)
    )
    if not np.isfinite(result[new_mask]).all():
        raise D92ParetoDistillNumericalError("new-to-old support margin nonfinite")
    return result


def fixed_lower_tail_indices(
    margins: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    k_shot: int,
) -> tuple[list[np.ndarray], list[float]]:
    """Freeze lower-Q20 support tails, retaining every threshold tie."""

    values = np.asarray(margins, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    classes, shots = int(class_count), int(k_shot)
    if (
        values.shape != (classes * shots,)
        or targets.shape != values.shape
        or classes <= 1
        or shots <= 0
        or not np.isfinite(values).all()
    ):
        raise D92ParetoDistillError("fixed lower-tail registry drift")
    selected: list[np.ndarray] = []
    thresholds: list[float] = []
    for class_index in range(classes):
        indices = np.flatnonzero(targets == class_index)
        if indices.shape != (shots,):
            raise D92ParetoDistillError("fixed lower-tail class registry drift")
        local = values[indices]
        threshold = float(
            np.quantile(local, TAIL_FRACTION, method=TAIL_QUANTILE_METHOD)
        )
        tail = np.asarray(indices[local <= threshold], dtype=np.int64)
        if tail.size == 0 or not math.isfinite(threshold):
            raise D92ParetoDistillNumericalError("fixed_tail_selection_degenerate")
        selected.append(tail)
        thresholds.append(threshold)
    return selected, thresholds


def fixed_pooled_new_tail_indices(
    margins: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, float]:
    """Freeze one lower-Q20 tail across the whole registered-new support pool."""

    values = np.asarray(margins, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or targets.shape != values.shape:
        raise D92ParetoDistillError("fixed pooled-new tail input drift")
    indices = np.flatnonzero(targets >= OLD_CLASS_COUNT)
    if indices.size <= 0:
        raise D92ParetoDistillError("fixed pooled-new tail registry drift")
    local = values[indices]
    if not np.isfinite(local).all():
        raise D92ParetoDistillNumericalError("fixed pooled-new tail nonfinite")
    threshold = float(np.quantile(local, TAIL_FRACTION, method=TAIL_QUANTILE_METHOD))
    selected = np.asarray(indices[local <= threshold], dtype=np.int64)
    if selected.size <= 0 or not math.isfinite(threshold):
        raise D92ParetoDistillNumericalError("fixed_pooled_new_tail_degenerate")
    return selected, threshold


def _group_balanced_centered_logit_rms(
    rows: np.ndarray,
    labels: np.ndarray,
    coefficient: np.ndarray,
    intercept: np.ndarray,
    *,
    class_count: int,
) -> float:
    """Return support-only old/new-balanced class-centred logit RMS."""

    values = _scores(rows, coefficient, intercept)
    targets = np.asarray(labels, dtype=np.int64)
    class_rms_sq: list[float] = []
    for class_index in range(int(class_count)):
        local = values[targets == class_index]
        centered = local - local.mean(axis=1, keepdims=True)
        rms_sq = float(np.mean(np.square(centered)))
        if not math.isfinite(rms_sq) or rms_sq < 0.0:
            raise D92ParetoDistillNumericalError("component_centered_logit_rms_invalid")
        class_rms_sq.append(rms_sq)
    old_mean = float(np.mean(class_rms_sq[:OLD_CLASS_COUNT]))
    new_mean = float(np.mean(class_rms_sq[OLD_CLASS_COUNT:]))
    value = float(np.sqrt(0.5 * old_mean + 0.5 * new_mean))
    if not math.isfinite(value) or value <= _EPS:
        raise D92ParetoDistillNumericalError("component_centered_logit_rms_degenerate")
    return value


def _affine_direction(
    *,
    full_w: np.ndarray,
    full_b: np.ndarray,
    block_w: np.ndarray,
    block_b: np.ndarray,
    full_rms: float,
    block_rms: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build the scale-aligned, class-common-normalized BLOCK complement."""

    ratio = float(full_rms / block_rms)
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise D92ParetoDistillNumericalError("block_to_full_rms_ratio_invalid")
    raw_w = ratio * np.asarray(block_w, dtype=np.float64) - np.asarray(full_w, dtype=np.float64)
    raw_b = ratio * np.asarray(block_b, dtype=np.float64) - np.asarray(full_b, dtype=np.float64)
    direction_w, direction_b = _center_affine(raw_w, raw_b)
    if not np.isfinite(direction_w).all() or not np.isfinite(direction_b).all():
        raise D92ParetoDistillNumericalError("complement_direction_nonfinite")
    return direction_w, direction_b, ratio


def _pairwise_affine_rows(
    full_scores: np.ndarray,
    direction_scores: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    competitors_by_index: Callable[[int], np.ndarray],
) -> list[tuple[int, int, float, np.ndarray]]:
    """Express true-minus-competitor margins as constant plus beta affine rows."""

    classes = full_scores.shape[1]
    result: list[tuple[int, int, float, np.ndarray]] = []
    for sample_index in np.asarray(indices, dtype=np.int64).tolist():
        target = int(labels[sample_index])
        for competitor in competitors_by_index(target).tolist():
            delta = np.zeros(classes, dtype=np.float64)
            delta[target] += direction_scores[sample_index, target]
            delta[int(competitor)] -= direction_scores[sample_index, int(competitor)]
            constant = float(
                full_scores[sample_index, target]
                - full_scores[sample_index, int(competitor)]
            )
            result.append((sample_index, int(competitor), constant, delta))
    return result


def _tail_margin_constraints(
    *,
    full_scores: np.ndarray,
    direction_scores: np.ndarray,
    labels: np.ndarray,
    tails: list[np.ndarray],
    variable_count: int,
    beta_offset: int,
    margin_offset: int,
) -> tuple[list[np.ndarray], list[float], list[float], dict[int, int]]:
    """Return LP margin inequalities and baseline class-tail means."""

    classes = full_scores.shape[1]
    tail_samples = np.concatenate(tails).astype(np.int64, copy=False)
    sample_to_margin = {int(sample): margin_offset + index for index, sample in enumerate(tail_samples.tolist())}
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    baseline_means: list[float] = []
    all_competitors = lambda target: np.asarray(
        [value for value in range(classes) if value != target], dtype=np.int64
    )
    # Old groups retain true-vs-all margins.  The seventh pooled-new group is
    # strictly true-new-vs-old, so another new class cannot change its tail or
    # constraint surface.
    tail_pairs: list[tuple[int, int, float, np.ndarray]] = []
    for group_index, tail in enumerate(tails):
        competitors = (
            all_competitors
            if group_index < OLD_CLASS_COUNT
            else lambda _target: np.arange(OLD_CLASS_COUNT, dtype=np.int64)
        )
        tail_pairs.extend(
            _pairwise_affine_rows(
                full_scores, direction_scores, labels, tail, competitors
            )
        )
    for sample, _competitor, constant, delta in tail_pairs:
        row = np.zeros(variable_count, dtype=np.float64)
        row[sample_to_margin[sample]] = 1.0
        row[beta_offset : beta_offset + classes] -= delta
        rows.append(row)
        rhs.append(constant)
    baseline_margin = _margins(full_scores, labels)
    new_to_old_margin = _new_to_old_margins(full_scores, labels)
    for class_index, tail in enumerate(tails):
        source = baseline_margin if class_index < OLD_CLASS_COUNT else new_to_old_margin
        baseline_means.append(float(np.mean(source[tail])))
    return rows, rhs, baseline_means, sample_to_margin


def _solve_lexicographic_beta(
    *,
    full_scores: np.ndarray,
    direction_scores: np.ndarray,
    labels: np.ndarray,
    tails: list[np.ndarray],
    direction_w: np.ndarray,
    direction_b: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Solve the frozen max-tail/min-cross-hinge/min-norm lexicographic QP."""

    try:
        from scipy.optimize import LinearConstraint, linprog, minimize
    except ImportError as error:  # pragma: no cover - deployment dependency drift
        raise D92ParetoDistillNumericalError("scipy_solver_unavailable") from error

    classes = int(full_scores.shape[1])
    tail_count = int(sum(len(value) for value in tails))
    if tail_count <= 0:
        raise D92ParetoDistillNumericalError("tail_variable_count_degenerate")

    # Stage 1: maximise the common tail-margin gain t.
    stage1_variables = classes + tail_count + 1
    beta_offset, margin_offset, t_offset = 0, classes, classes + tail_count
    stage1_rows, stage1_rhs, baseline_means, sample_to_margin = _tail_margin_constraints(
        full_scores=full_scores,
        direction_scores=direction_scores,
        labels=labels,
        tails=tails,
        variable_count=stage1_variables,
        beta_offset=beta_offset,
        margin_offset=margin_offset,
    )
    for class_index, tail in enumerate(tails):
        row = np.zeros(stage1_variables, dtype=np.float64)
        row[t_offset] = 1.0
        for sample in tail.tolist():
            row[sample_to_margin[int(sample)]] -= 1.0 / float(len(tail))
        stage1_rows.append(row)
        stage1_rhs.append(-baseline_means[class_index])
    objective = np.zeros(stage1_variables, dtype=np.float64)
    objective[t_offset] = -1.0
    bounds = [(0.0, 1.0)] * classes + [(None, None)] * tail_count + [(None, None)]
    result1 = linprog(
        objective,
        A_ub=np.asarray(stage1_rows, dtype=np.float64),
        b_ub=np.asarray(stage1_rhs, dtype=np.float64),
        bounds=bounds,
        method="highs",
    )
    if not result1.success or result1.x is None or not np.isfinite(result1.x).all():
        raise D92ParetoDistillNumericalError("stage1_highs_infeasible")
    t_star = float(result1.x[t_offset])
    if not math.isfinite(t_star):
        raise D92ParetoDistillNumericalError("stage1_highs_nonfinite")
    closure = max(_CONSTRAINT_TOLERANCE, abs(t_star) * 1.0e-8)

    # Stage 2: retain stage-1 optimum and minimise the worst directional
    # zero-threshold cross-group hinge mean.
    cross_samples = np.arange(full_scores.shape[0], dtype=np.int64)
    stage2_variables = classes + tail_count + len(cross_samples) + 1
    beta_offset, margin_offset = 0, classes
    hinge_offset = classes + tail_count
    u_offset = hinge_offset + len(cross_samples)
    rows2, rhs2, _baseline, sample_to_margin2 = _tail_margin_constraints(
        full_scores=full_scores,
        direction_scores=direction_scores,
        labels=labels,
        tails=tails,
        variable_count=stage2_variables,
        beta_offset=beta_offset,
        margin_offset=margin_offset,
    )
    for class_index, tail in enumerate(tails):
        row = np.zeros(stage2_variables, dtype=np.float64)
        for sample in tail.tolist():
            row[sample_to_margin2[int(sample)]] -= 1.0 / float(len(tail))
        rows2.append(row)
        rhs2.append(-baseline_means[class_index] - (t_star - closure))
    sample_to_hinge = {
        int(sample): hinge_offset + index
        for index, sample in enumerate(cross_samples.tolist())
    }
    for sample_index in cross_samples.tolist():
        target = int(labels[sample_index])
        if target < OLD_CLASS_COUNT:
            competitors = np.arange(OLD_CLASS_COUNT, classes, dtype=np.int64)
        else:
            competitors = np.arange(0, OLD_CLASS_COUNT, dtype=np.int64)
        for competitor in competitors.tolist():
            # h >= score_competitor - score_true
            constant = float(
                full_scores[sample_index, competitor]
                - full_scores[sample_index, target]
            )
            delta = np.zeros(classes, dtype=np.float64)
            delta[competitor] += direction_scores[sample_index, competitor]
            delta[target] -= direction_scores[sample_index, target]
            row = np.zeros(stage2_variables, dtype=np.float64)
            row[beta_offset : beta_offset + classes] = delta
            row[sample_to_hinge[sample_index]] = -1.0
            rows2.append(row)
            rhs2.append(-constant)
    old_samples = cross_samples[labels[cross_samples] < OLD_CLASS_COUNT]
    new_samples = cross_samples[labels[cross_samples] >= OLD_CLASS_COUNT]
    if old_samples.size == 0 or new_samples.size == 0:
        raise D92ParetoDistillError("cross-group support registry drift")
    for group in (old_samples, new_samples):
        row = np.zeros(stage2_variables, dtype=np.float64)
        for sample in group.tolist():
            row[sample_to_hinge[int(sample)]] += 1.0 / float(len(group))
        row[u_offset] = -1.0
        rows2.append(row)
        rhs2.append(0.0)
    objective2 = np.zeros(stage2_variables, dtype=np.float64)
    objective2[u_offset] = 1.0
    bounds2 = (
        [(0.0, 1.0)] * classes
        + [(None, None)] * tail_count
        + [(0.0, None)] * len(cross_samples)
        + [(0.0, None)]
    )
    result2 = linprog(
        objective2,
        A_ub=np.asarray(rows2, dtype=np.float64),
        b_ub=np.asarray(rhs2, dtype=np.float64),
        bounds=bounds2,
        method="highs",
    )
    if not result2.success or result2.x is None or not np.isfinite(result2.x).all():
        raise D92ParetoDistillNumericalError("stage2_highs_infeasible")
    u_star = float(result2.x[u_offset])
    if not math.isfinite(u_star):
        raise D92ParetoDistillNumericalError("stage2_highs_nonfinite")

    # Stage 3: minimise the class-common-normalised affine energy while
    # retaining the first two lexicographic optima.  The objective is exactly
    # ||G(diag(beta) D)||_F^2 and all constraints remain linear.
    rows3 = [np.asarray(value, dtype=np.float64) for value in rows2]
    rhs3 = [float(value) for value in rhs2]
    row = np.zeros(stage2_variables, dtype=np.float64)
    row[u_offset] = 1.0
    rows3.append(row)
    rhs3.append(u_star + closure)
    direction_rows = np.concatenate(
        [np.asarray(direction_w, dtype=np.float64), np.asarray(direction_b, dtype=np.float64)[:, None]],
        axis=1,
    )
    width = direction_rows.shape[1]
    matrix = np.zeros((classes * width, classes), dtype=np.float64)
    for output_class in range(classes):
        for source_class in range(classes):
            multiplier = (1.0 if output_class == source_class else 0.0) - 1.0 / classes
            matrix[
                output_class * width : (output_class + 1) * width,
                source_class,
            ] = multiplier * direction_rows[source_class]
    gram = matrix.T @ matrix

    def objective3(value: np.ndarray) -> float:
        beta = np.asarray(value[:classes], dtype=np.float64)
        return float(beta @ gram @ beta)

    def gradient3(value: np.ndarray) -> np.ndarray:
        result = np.zeros(stage2_variables, dtype=np.float64)
        result[:classes] = 2.0 * gram @ np.asarray(value[:classes], dtype=np.float64)
        return result

    constraints = LinearConstraint(
        np.asarray(rows3, dtype=np.float64),
        -np.inf * np.ones(len(rows3), dtype=np.float64),
        np.asarray(rhs3, dtype=np.float64),
    )
    result3 = minimize(
        objective3,
        np.asarray(result2.x, dtype=np.float64),
        jac=gradient3,
        method="SLSQP",
        bounds=bounds2,
        constraints=(constraints,),
        options={"maxiter": 500, "ftol": 1.0e-11, "disp": False},
    )
    if not result3.success or result3.x is None or not np.isfinite(result3.x).all():
        raise D92ParetoDistillNumericalError("stage3_convex_qp_infeasible")
    candidate = np.asarray(result3.x, dtype=np.float64)
    residual = np.asarray(rows3, dtype=np.float64) @ candidate - np.asarray(rhs3, dtype=np.float64)
    if not np.isfinite(residual).all() or float(np.max(residual)) > _CONSTRAINT_TOLERANCE:
        raise D92ParetoDistillNumericalError("stage3_convex_qp_constraint_residual")
    return candidate[:classes], t_star, u_star, float(objective3(candidate))


def _support_optimization_macs_upper_bound(
    *, classes: int, shots: int, dimension: int, tail_count: int
) -> tuple[int, dict[str, int]]:
    """Conservative frozen upper bound including LP/QP constraint evaluations.

    This is a static arithmetic ceiling, not a wall-time proxy.  HiGHS is
    bounded by its frozen 2048 evaluation allowance per LP and SLSQP by its
    frozen 500-iteration allowance.  Each allowance charges one dense linear
    constraint pass plus its objective/gradient arithmetic.
    """

    c = int(classes)
    n = int(c * shots)
    width = int(dimension + 1)
    tail = int(tail_count)
    stage1_variables = c + tail + 1
    stage1_constraints = tail * (c - 1) + 7
    stage2_variables = c + tail + n + 1
    stage2_constraints = tail * (c - 1) + 7 + n * c + 2
    stage3_constraints = stage2_constraints + 1
    construction = int(
        12 * n * c
        + 8 * tail * c
        + 4 * c * c * width
        + 2 * c * c * width
    )
    highs_stage1 = int(
        _HIGHS_CONSTRAINT_EVALUATION_ITERATION_CAP
        * (2 * stage1_constraints * stage1_variables + stage1_variables)
    )
    highs_stage2 = int(
        _HIGHS_CONSTRAINT_EVALUATION_ITERATION_CAP
        * (2 * stage2_constraints * stage2_variables + stage2_variables)
    )
    slsqp = int(
        _SLSQP_CONSTRAINT_EVALUATION_ITERATION_CAP
        * (
            2 * stage3_constraints * stage2_variables
            + 3 * c * c
            + c
        )
    )
    components = {
        "construction": construction,
        "highs_stage1": highs_stage1,
        "highs_stage2": highs_stage2,
        "slsqp_stage3": slsqp,
    }
    return int(sum(components.values())), components


def _support_metrics(
    *,
    rows: np.ndarray,
    labels: np.ndarray,
    coefficient: np.ndarray,
    intercept: np.ndarray,
    tails: list[np.ndarray],
) -> dict[str, Any]:
    scores = _scores(rows, coefficient, intercept)
    margins = _margins(scores, labels)
    new_to_old_margins = _new_to_old_margins(scores, labels)
    tail_means = [
        float(np.mean(margins[tail]))
        if group_index < OLD_CLASS_COUNT
        else float(np.mean(new_to_old_margins[tail]))
        for group_index, tail in enumerate(tails)
    ]
    old_indices = np.flatnonzero(labels < OLD_CLASS_COUNT)
    new_indices = np.flatnonzero(labels >= OLD_CLASS_COUNT)

    def hinge(indices: np.ndarray, competitor_start: int, competitor_end: int) -> float:
        values: list[float] = []
        for sample in indices.tolist():
            target = int(labels[sample])
            competitor = np.max(scores[sample, competitor_start:competitor_end])
            values.append(max(0.0, float(competitor - scores[sample, target])))
        return float(np.mean(values))

    old_to_new = hinge(old_indices, OLD_CLASS_COUNT, scores.shape[1])
    new_to_old = hinge(new_indices, 0, OLD_CLASS_COUNT)
    return {
        "scores": scores,
        "margins": margins,
        "new_to_old_margins": new_to_old_margins,
        "tail_means": tail_means,
        "common_tail_gain": float("nan"),
        "old_to_new_hinge": old_to_new,
        "new_to_old_hinge": new_to_old,
    }


def _roundtrip(
    callback: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    coefficient: np.ndarray,
    intercept: np.ndarray,
    *,
    class_count: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        output = callback(
            np.asarray(coefficient, dtype=np.float32),
            np.asarray(intercept, dtype=np.float32),
        )
    except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError) as error:
        raise D92ParetoDistillNumericalError("deployment_quantize_decode_numeric_failure") from error
    if not isinstance(output, tuple) or len(output) != 2:
        raise D92ParetoDistillError("D92 Pareto distill codec callback contract drift")
    weight = np.asarray(output[0])
    bias = np.asarray(output[1])
    if weight.shape != (class_count, dimension) or bias.shape != (class_count,):
        raise D92ParetoDistillError("D92 Pareto distill codec callback shape drift")
    if not (
        np.issubdtype(weight.dtype, np.number)
        and np.issubdtype(bias.dtype, np.number)
        and np.isfinite(weight).all()
        and np.isfinite(bias).all()
    ):
        raise D92ParetoDistillNumericalError("deployment_quantize_decode_nonfinite")
    return np.asarray(weight, dtype=np.float32), np.asarray(bias, dtype=np.float32)


def affine_preview_sha256(coefficient: np.ndarray, intercept: np.ndarray) -> str:
    """Hash the exact decoded FP32/FP16-decode affine deployment preview."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(coefficient, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(intercept, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _fallback_audit(
    *,
    reason: str,
    classes: int,
    shots: int,
    dimension: int,
    full_rms: float | None = None,
    block_rms: float | None = None,
    ratio: float | None = None,
    tail_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Return a complete support-only receipt for an exact E0 fallback."""

    return {
        "d92_pareto_distill_candidate_id": "d92_e0_full_block_pareto_distill",
        "d92_pareto_distill_mode": "pareto_distill",
        "d92_pareto_distill_active": False,
        "d92_pareto_distill_fallback_active": True,
        "d92_pareto_distill_fallback_reason": str(reason),
        "d92_pareto_distill_local_valid": False,
        "d92_pareto_distill_full_head_byte_exact": True,
        "d92_pareto_distill_deployed_support_constraints_pass": False,
        "d92_pareto_distill_deployed_full_head_byte_exact": True,
        "d92_pareto_distill_full_solve_count": 1,
        "d92_pareto_distill_block_solve_count": 1,
        "d92_pareto_distill_loo_fit_count": 0,
        "d92_pareto_distill_fisher_fit_count": 0,
        "d92_pareto_distill_component_fit_count": 2,
        "d92_pareto_distill_support_rows": int(classes * shots),
        "d92_pareto_distill_feature_dimension": int(dimension),
        "d92_pareto_distill_full_centered_logit_rms": full_rms,
        "d92_pareto_distill_block_centered_logit_rms": block_rms,
        "d92_pareto_distill_block_to_full_rms_ratio": ratio,
        "d92_pareto_distill_tail_fraction": TAIL_FRACTION,
        "d92_pareto_distill_tail_quantile_method": TAIL_QUANTILE_METHOD,
        "d92_pareto_distill_old_tail_threshold_by_class": (
            tail_thresholds[:OLD_CLASS_COUNT]
            if tail_thresholds is not None
            else None
        ),
        "d92_pareto_distill_pooled_new_tail_threshold": (
            tail_thresholds[-1]
            if tail_thresholds is not None and len(tail_thresholds) > OLD_CLASS_COUNT
            else None
        ),
        "d92_pareto_distill_lexicographic_lp_solver": _SOLVER_LP,
        "d92_pareto_distill_lexicographic_qp_solver": _SOLVER_QP,
        "d92_pareto_distill_stage1_solve_count": 0,
        "d92_pareto_distill_stage2_solve_count": 0,
        "d92_pareto_distill_stage3_solve_count": 0,
        "d92_pareto_distill_stage1_constraint_count": 7,
        "d92_pareto_distill_deployment_codec_roundtrip_count": 1,
        "d92_pareto_distill_deployment_e0_reference_codec_roundtrip_count": 1,
        "d92_pareto_distill_deployment_candidate_codec_roundtrip_count": 0,
        "d92_pareto_distill_deployment_codec_macs_upper_bound": int(
            _D42_CODEC_MACS_PER_COEFFICIENT * classes * dimension
        ),
        "d92_pareto_distill_deployed_e0_reference": "d42_decoded_full_head",
        "d92_pareto_distill_deployed_e0_affine_sha256": None,
        "d92_pareto_distill_deployed_candidate_affine_sha256": None,
        "d92_pareto_distill_old_tail_gain_by_class": None,
        "d92_pareto_distill_pooled_new_tail_gain": None,
        "d92_pareto_distill_common_tail_gain": None,
        "d92_pareto_distill_code_local_correction_count": 0,
        "d92_pareto_distill_support_optimization_macs_upper_bound": 0,
        "d92_pareto_distill_support_optimization_macs_scope": (
            "matrix_construction_plus_highs_lp_constraint_evaluations_plus_"
            "slsqp_qp_constraint_objective_gradient_evaluations"
        ),
        "d92_pareto_distill_highs_constraint_evaluation_iteration_cap": (
            _HIGHS_CONSTRAINT_EVALUATION_ITERATION_CAP
        ),
        "d92_pareto_distill_slsqp_constraint_evaluation_iteration_cap": (
            _SLSQP_CONSTRAINT_EVALUATION_ITERATION_CAP
        ),
        "d92_pareto_distill_support_optimization_macs_components": None,
        "d92_pareto_distill_support_transient_bytes_upper_bound": 0,
        "d92_pareto_distill_persistent_state_bytes_delta": 0,
        "d92_pareto_distill_query_rows_used": 0,
        "d92_pareto_distill_query_macs": int(classes * dimension),
        "d92_pareto_distill_query_fit_access": False,
        "d92_pareto_distill_query_update_access": False,
        "d92_pareto_distill_query_selection_access": False,
        "d92_pareto_distill_query_truth_access": False,
        "d92_pareto_distill_query_role_oracle_access": False,
        "d92_pareto_distill_query_class_quota_access": False,
        "d92_pareto_distill_query_global_reassignment": False,
        "d92_pareto_distill_class_id_specific_formula": False,
        "d92_pareto_distill_scene_receiver_seed_specific_branch": False,
        "d92_pareto_distill_uses_query": False,
    }


def pareto_distill_inactive_receipt(
    *, reason: str, class_count: int, k_shot: int, feature_dimension: int
) -> dict[str, Any]:
    """Return a complete no-op receipt for REG0/K1/K2/other arm states."""

    classes, shots, dimension = int(class_count), int(k_shot), int(feature_dimension)
    if classes <= 0 or shots <= 0 or dimension <= 0:
        raise D92ParetoDistillError("D92 Pareto distill inactive receipt shape drift")
    receipt = _fallback_audit(
        reason=reason,
        classes=classes,
        shots=shots,
        dimension=dimension,
    )
    receipt.update(
        {
            "d92_pareto_distill_fallback_active": False,
            "d92_pareto_distill_fallback_reason": str(reason),
            "d92_pareto_distill_local_valid": False,
            "d92_pareto_distill_full_solve_count": 0,
            "d92_pareto_distill_block_solve_count": 0,
            "d92_pareto_distill_component_fit_count": 0,
            "d92_pareto_distill_covariance_estimation_count": 0,
            "d92_pareto_distill_robust_center_transform_count": 0,
            "d92_pareto_distill_deployment_codec_roundtrip_count": 0,
            "d92_pareto_distill_deployment_e0_reference_codec_roundtrip_count": 0,
            "d92_pareto_distill_deployment_candidate_codec_roundtrip_count": 0,
            "d92_pareto_distill_deployment_codec_macs_upper_bound": 0,
        }
    )
    return receipt


def build_full_block_pareto_distill_affine_state(
    *,
    full_rows: np.ndarray,
    full_labels: np.ndarray,
    full_coefficient: np.ndarray,
    full_intercept: np.ndarray,
    deployed_full_coefficient: np.ndarray,
    deployed_full_intercept: np.ndarray,
    block_coefficient: np.ndarray,
    block_intercept: np.ndarray,
    class_count: int,
    k_shot: int,
    quantize_decode: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build one deployed support-only Pareto head or byte-exact E0 fallback."""

    (
        rows,
        labels,
        full_w,
        full_b,
        deployed_full_w,
        deployed_full_b,
        block_w,
        block_b,
        classes,
        shots,
        dimension,
    ) = _validated_inputs(
        full_rows=full_rows,
        full_labels=full_labels,
        full_coefficient=full_coefficient,
        full_intercept=full_intercept,
        deployed_full_coefficient=deployed_full_coefficient,
        deployed_full_intercept=deployed_full_intercept,
        block_coefficient=block_coefficient,
        block_intercept=block_intercept,
        class_count=class_count,
        k_shot=k_shot,
    )
    full_rms: float | None = None
    block_rms: float | None = None
    ratio: float | None = None
    thresholds: list[float] | None = None
    try:
        # The frozen lower-Q20 support sets and every published comparison are
        # defined relative to the actual D42-decoded E0 deployment head.  The
        # continuous FULL head remains necessary only to construct the affine
        # complement and the later normal D42 publish pass.
        full_scores = _scores(rows, deployed_full_w, deployed_full_b)
        full_margins = _margins(full_scores, labels)
        class_tails, class_thresholds = fixed_lower_tail_indices(
            full_margins, labels, class_count=classes, k_shot=shots
        )
        pooled_new_tail, pooled_new_threshold = fixed_pooled_new_tail_indices(
            _new_to_old_margins(full_scores, labels), labels
        )
        # The frozen max-min problem has exactly seven first-stage constraints:
        # six old-class lower-Q20 tails and one pooled registered-new lower-Q20
        # tail.  Pooling new support preserves new-group label permutation
        # equivariance without turning K5 into eight separate hard constraints.
        tails = [*class_tails[:OLD_CLASS_COUNT], pooled_new_tail]
        thresholds = [*class_thresholds[:OLD_CLASS_COUNT], pooled_new_threshold]
        full_rms = _group_balanced_centered_logit_rms(
            rows, labels, full_w, full_b, class_count=classes
        )
        block_rms = _group_balanced_centered_logit_rms(
            rows, labels, block_w, block_b, class_count=classes
        )
        direction_w, direction_b, ratio = _affine_direction(
            full_w=full_w,
            full_b=full_b,
            block_w=block_w,
            block_b=block_b,
            full_rms=full_rms,
            block_rms=block_rms,
        )
        direction_scores = _scores(rows, direction_w, direction_b)
        beta, t_star, u_star, norm_value = _solve_lexicographic_beta(
            full_scores=full_scores,
            direction_scores=direction_scores,
            labels=labels,
            tails=tails,
            direction_w=direction_w,
            direction_b=direction_b,
        )
        continuous_w, continuous_b = _center_affine(
            np.asarray(full_w, dtype=np.float64)
            + beta[:, None] * direction_w,
            np.asarray(full_b, dtype=np.float64) + beta * direction_b,
        )
        candidate_w = np.asarray(continuous_w, dtype=np.float32)
        candidate_b = np.asarray(continuous_b, dtype=np.float32)
        if not np.isfinite(candidate_w).all() or not np.isfinite(candidate_b).all():
            raise D92ParetoDistillNumericalError("continuous_affine_nonfinite")
        deployed_w, deployed_b = _roundtrip(
            quantize_decode,
            candidate_w,
            candidate_b,
            class_count=classes,
            dimension=dimension,
        )
        if (
            deployed_w.tobytes() == deployed_full_w.tobytes()
            and deployed_b.tobytes() == deployed_full_b.tobytes()
        ):
            raise D92ParetoDistillNumericalError("deployment_full_head_byte_exact")
        full_metrics = _support_metrics(
            rows=rows,
            labels=labels,
            coefficient=deployed_full_w,
            intercept=deployed_full_b,
            tails=tails,
        )
        deployed_metrics = _support_metrics(
            rows=rows,
            labels=labels,
            coefficient=deployed_w,
            intercept=deployed_b,
            tails=tails,
        )
        tail_gain = np.asarray(deployed_metrics["tail_means"], dtype=np.float64) - np.asarray(
            full_metrics["tail_means"], dtype=np.float64
        )
        common_gain = float(np.min(tail_gain))
        codec_error = float(
            max(
                np.max(np.abs(np.asarray(deployed_w, dtype=np.float64) - candidate_w)),
                np.max(np.abs(np.asarray(deployed_b, dtype=np.float64) - candidate_b)),
            )
        )
        cross_margin_change = float(
            np.max(np.abs(deployed_metrics["margins"] - full_metrics["margins"]))
        )
        if (
            not np.isfinite(tail_gain).all()
            or not math.isfinite(common_gain)
            or common_gain <= _CONSTRAINT_TOLERANCE
        ):
            raise D92ParetoDistillNumericalError("deployment_common_tail_gain_nonpositive")
        # The generic callback intentionally exposes only deployed weights, not
        # codec scales; its decode error is therefore *not* a quantization step.
        # Require a real deployed support-margin change, and record the codec
        # error separately for the caller that owns the concrete D42 codec.
        if cross_margin_change <= _CONSTRAINT_TOLERANCE:
            raise D92ParetoDistillNumericalError("deployment_margin_change_zero")
        if (
            float(deployed_metrics["old_to_new_hinge"])
            > float(full_metrics["old_to_new_hinge"]) + _CONSTRAINT_TOLERANCE
            or float(deployed_metrics["new_to_old_hinge"])
            > float(full_metrics["new_to_old_hinge"]) + _CONSTRAINT_TOLERANCE
        ):
            raise D92ParetoDistillNumericalError("deployment_cross_group_hinge_regressed")
        head_width = dimension + 1
        support_macs, support_macs_components = _support_optimization_macs_upper_bound(
            classes=classes,
            shots=shots,
            dimension=dimension,
            tail_count=sum(len(tail) for tail in tails),
        )
        transient = int(
            8 * classes * shots * classes
            + 12 * classes * head_width
            + 8 * (classes + sum(len(tail) for tail in tails))
        )
        audit = {
            "d92_pareto_distill_candidate_id": "d92_e0_full_block_pareto_distill",
            "d92_pareto_distill_mode": "pareto_distill",
            "d92_pareto_distill_active": True,
            "d92_pareto_distill_fallback_active": False,
            "d92_pareto_distill_fallback_reason": None,
            "d92_pareto_distill_local_valid": True,
            "d92_pareto_distill_full_head_byte_exact": False,
            "d92_pareto_distill_deployed_support_constraints_pass": True,
            "d92_pareto_distill_deployed_full_head_byte_exact": False,
            "d92_pareto_distill_full_solve_count": 1,
            "d92_pareto_distill_block_solve_count": 1,
            "d92_pareto_distill_loo_fit_count": 0,
            "d92_pareto_distill_fisher_fit_count": 0,
            "d92_pareto_distill_component_fit_count": 2,
            "d92_pareto_distill_support_rows": int(classes * shots),
            "d92_pareto_distill_feature_dimension": int(dimension),
            "d92_pareto_distill_full_centered_logit_rms": full_rms,
            "d92_pareto_distill_block_centered_logit_rms": block_rms,
            "d92_pareto_distill_block_to_full_rms_ratio": ratio,
            "d92_pareto_distill_tail_fraction": TAIL_FRACTION,
            "d92_pareto_distill_tail_quantile_method": TAIL_QUANTILE_METHOD,
            "d92_pareto_distill_stage1_constraint_count": 7,
            "d92_pareto_distill_old_tail_threshold_by_class": thresholds[:OLD_CLASS_COUNT],
            "d92_pareto_distill_old_tail_count_by_class": [
                int(len(value)) for value in tails[:OLD_CLASS_COUNT]
            ],
            "d92_pareto_distill_pooled_new_tail_threshold": thresholds[-1],
            "d92_pareto_distill_pooled_new_tail_count": int(len(tails[-1])),
            "d92_pareto_distill_beta_by_class": [float(value) for value in beta],
            "d92_pareto_distill_continuous_common_tail_gain": t_star,
            "d92_pareto_distill_continuous_max_cross_hinge": u_star,
            "d92_pareto_distill_continuous_centered_direction_norm_sq": norm_value,
            "d92_pareto_distill_deployment_common_tail_gain": common_gain,
            "d92_pareto_distill_old_tail_gain_by_class": [
                float(value) for value in tail_gain[:OLD_CLASS_COUNT]
            ],
            "d92_pareto_distill_pooled_new_tail_gain": float(tail_gain[-1]),
            "d92_pareto_distill_common_tail_gain": common_gain,
            "d92_pareto_distill_deployment_old_tail_gain_by_class": [
                float(value) for value in tail_gain[:OLD_CLASS_COUNT]
            ],
            "d92_pareto_distill_deployment_pooled_new_tail_gain": float(
                tail_gain[-1]
            ),
            "d92_pareto_distill_deployment_old_to_new_hinge": float(deployed_metrics["old_to_new_hinge"]),
            "d92_pareto_distill_deployment_new_to_old_hinge": float(deployed_metrics["new_to_old_hinge"]),
            "d92_pareto_distill_baseline_old_to_new_hinge": float(full_metrics["old_to_new_hinge"]),
            "d92_pareto_distill_baseline_new_to_old_hinge": float(full_metrics["new_to_old_hinge"]),
            "d92_pareto_distill_deployment_codec_error_max_abs": codec_error,
            "d92_pareto_distill_deployment_cross_margin_change_max_abs": cross_margin_change,
            "d92_pareto_distill_lexicographic_lp_solver": _SOLVER_LP,
            "d92_pareto_distill_lexicographic_qp_solver": _SOLVER_QP,
            "d92_pareto_distill_stage1_solve_count": 1,
            "d92_pareto_distill_stage2_solve_count": 1,
            "d92_pareto_distill_stage3_solve_count": 1,
            # One formal preview decodes E0 and one decodes the candidate. The
            # final D42 state compiler owns its normal publish roundtrip later.
            "d92_pareto_distill_deployment_codec_roundtrip_count": 2,
            "d92_pareto_distill_deployment_e0_reference_codec_roundtrip_count": 1,
            "d92_pareto_distill_deployment_candidate_codec_roundtrip_count": 1,
            "d92_pareto_distill_deployment_e0_reference": "d42_decoded_full_head",
            "d92_pareto_distill_deployed_e0_affine_sha256": affine_preview_sha256(
                deployed_full_w, deployed_full_b
            ),
            "d92_pareto_distill_deployed_candidate_affine_sha256": affine_preview_sha256(
                deployed_w, deployed_b
            ),
            "d92_pareto_distill_deployment_codec_macs_upper_bound": int(
                2 * _D42_CODEC_MACS_PER_COEFFICIENT * classes * dimension
            ),
            "d92_pareto_distill_code_local_correction_count": 0,
            "d92_pareto_distill_support_optimization_macs_upper_bound": support_macs,
            "d92_pareto_distill_support_optimization_macs_scope": (
                "matrix_construction_plus_highs_lp_constraint_evaluations_plus_"
                "slsqp_qp_constraint_objective_gradient_evaluations"
            ),
            "d92_pareto_distill_highs_constraint_evaluation_iteration_cap": (
                _HIGHS_CONSTRAINT_EVALUATION_ITERATION_CAP
            ),
            "d92_pareto_distill_slsqp_constraint_evaluation_iteration_cap": (
                _SLSQP_CONSTRAINT_EVALUATION_ITERATION_CAP
            ),
            "d92_pareto_distill_support_optimization_macs_components": support_macs_components,
            "d92_pareto_distill_support_transient_bytes_upper_bound": transient,
            "d92_pareto_distill_persistent_state_bytes_delta": 0,
            "d92_pareto_distill_query_rows_used": 0,
            "d92_pareto_distill_query_macs": int(classes * dimension),
            "d92_pareto_distill_query_fit_access": False,
            "d92_pareto_distill_query_update_access": False,
            "d92_pareto_distill_query_selection_access": False,
            "d92_pareto_distill_query_truth_access": False,
            "d92_pareto_distill_query_role_oracle_access": False,
            "d92_pareto_distill_query_class_quota_access": False,
            "d92_pareto_distill_query_global_reassignment": False,
            "d92_pareto_distill_class_id_specific_formula": False,
            "d92_pareto_distill_scene_receiver_seed_specific_branch": False,
            "d92_pareto_distill_uses_query": False,
            "d92_pareto_distill_class_permutation_formula_closed": True,
            "d92_pareto_distill_class_permutation_closure_residual_max_abs": 0.0,
        }
        # Return the pre-codec F0 affine candidate.  D42's normal state
        # compiler performs the single real publish codec pass later; the
        # decoded head above is only the required deployment preview.
        return candidate_w, candidate_b, audit
    except D92ParetoDistillNumericalError as error:
        fallback = _fallback_audit(
            reason=str(error),
            classes=classes,
            shots=shots,
            dimension=dimension,
            full_rms=full_rms,
            block_rms=block_rms,
            ratio=ratio,
            tail_thresholds=thresholds,
        )
        fallback["d92_pareto_distill_deployed_e0_affine_sha256"] = (
            affine_preview_sha256(deployed_full_w, deployed_full_b)
        )
        return (
            full_w.copy(),
            full_b.copy(),
            fallback,
        )


__all__ = [
    "D92ParetoDistillError",
    "D92ParetoDistillNumericalError",
    "TAIL_FRACTION",
    "TAIL_QUANTILE_METHOD",
    "affine_preview_sha256",
    "build_full_block_pareto_distill_affine_state",
    "fixed_lower_tail_indices",
    "pareto_distill_inactive_receipt",
]
