"""Ground-domain tangent smooth-worst top-2 margin correction for D78."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np


OPTIMIZER_STEPS = 20
MAX_BACKTRACKS = 64
Z_DIM = 160


class D78GroundTangentError(RuntimeError):
    """Raised when D78 ground or support evidence is malformed."""


LDAFit = Callable[
    [np.ndarray, np.ndarray, int, int],
    tuple[np.ndarray, np.ndarray, dict[str, Any]],
]


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _within_class_ranks(labels: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.zeros(int(class_count), dtype=np.int64)
    ranks = np.empty(len(labels), dtype=np.int64)
    for row_index, class_index in enumerate(labels):
        ranks[row_index] = counts[int(class_index)]
        counts[int(class_index)] += 1
    return ranks


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def _cross_entropy(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    return np.log(np.sum(np.exp(shifted), axis=1)) - shifted[
        np.arange(len(targets)), targets
    ]


def ground_domain_tangent_basis(
    domain_class_prototypes: np.ndarray,
    domain_class_mask: np.ndarray,
    *,
    feature_dim: int,
    z_dim: int = Z_DIM,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the fixed domain-contrast tangent subspace from immutable centers."""

    prototypes = np.asarray(domain_class_prototypes, dtype=np.float64)
    mask = np.asarray(domain_class_mask, dtype=bool)
    if (
        prototypes.ndim != 3
        or prototypes.shape[:2] != mask.shape
        or prototypes.shape[2] != int(z_dim)
        or int(feature_dim) < int(z_dim)
        or prototypes.shape[0] < 2
        or prototypes.shape[1] < 2
        or not np.isfinite(prototypes).all()
    ):
        raise D78GroundTangentError("D78 requires a finite complete ground grid")
    registry_domains, classes = int(mask.shape[0]), int(mask.shape[1])
    full_domains = np.all(mask, axis=1)
    if np.any(np.any(mask, axis=1) != full_domains) or int(np.sum(full_domains)) < 2:
        raise D78GroundTangentError("D78 ground mask must contain whole class grids")
    active_prototypes = prototypes[full_domains]
    domains = int(active_prototypes.shape[0])
    class_means = np.mean(active_prototypes, axis=0, keepdims=True)
    residual = active_prototypes - class_means
    stacked = residual.reshape(domains * classes, int(z_dim))
    if not np.allclose(
        np.mean(residual, axis=0), 0.0, rtol=0.0, atol=1e-12
    ):
        raise D78GroundTangentError("D78 class-centered ground residual drift")
    _, singular, right = np.linalg.svd(stacked, full_matrices=False)
    tolerance = float(
        max(stacked.shape)
        * np.finfo(np.float64).eps
        * (float(singular[0]) if len(singular) else 0.0)
    )
    numerical_rank = int(np.sum(singular > tolerance))
    tangent_rank = min(domains - 1, numerical_rank)
    if tangent_rank < 1:
        raise D78GroundTangentError("D78 ground tangent rank is zero")
    z_basis = np.ascontiguousarray(right[:tangent_rank].T, dtype=np.float64)
    basis = np.zeros((int(feature_dim), tangent_rank), dtype=np.float64)
    basis[: int(z_dim)] = z_basis
    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(tangent_rank), rtol=0.0, atol=1e-10):
        raise D78GroundTangentError("D78 tangent basis lost orthonormality")
    projector = z_basis @ z_basis.T
    frozen = _readonly(basis, np.float64)
    energy_total = float(np.sum(singular * singular))
    energy_kept = float(np.sum(singular[:tangent_rank] ** 2))
    audit = {
        "schema": "cvs.phase2.d78.ground_domain_tangent_audit.v1",
        "ground_registry_domain_count": registry_domains,
        "ground_domain_count": domains,
        "ground_class_count": classes,
        "ground_component_input_count": int(np.sum(mask)),
        "z_dimension": int(z_dim),
        "feature_dimension": int(feature_dim),
        "numerical_rank": numerical_rank,
        "tangent_rank": tangent_rank,
        "rank_rule": "min(ground_domain_count_minus_one,numerical_rank)",
        "singular_value_max": float(singular[0]),
        "singular_value_min_kept": float(singular[tangent_rank - 1]),
        "retained_energy_fraction": energy_kept / energy_total,
        "basis_orthonormality_max_abs_error": float(
            np.max(np.abs(gram - np.eye(tangent_rank)))
        ),
        "projector_sha256": _sha256(projector.astype(np.float64)),
        "basis_sha256": _sha256(basis.astype(np.float64)),
        # Compatibility alias consumed by the D77 integration scaffold only.
        "preconditioner_sha256": _sha256(projector.astype(np.float64)),
        "ground_class_score_access": False,
        "ground_class_registry_prediction_branch": False,
        "ground_component_update_access": False,
    }
    return frozen, audit


def _validate_support(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    base_coefficient: np.ndarray,
    tangent_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    w = np.asarray(base_coefficient, dtype=np.float64)
    basis = np.asarray(tangent_basis, dtype=np.float64)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or w.shape != (int(class_count), x.shape[1])
        or basis.ndim != 2
        or basis.shape[0] != x.shape[1]
        or basis.shape[1] < 1
        or not np.isfinite(x).all()
        or not np.isfinite(w).all()
        or not np.isfinite(basis).all()
        or int(class_count) < 2
        or int(k_shot) < 1
        or len(x) != int(class_count) * int(k_shot)
        or not np.array_equal(np.unique(y), np.arange(int(class_count)))
        or any(np.sum(y == index) != int(k_shot) for index in range(class_count))
        or not np.allclose(
            basis.T @ basis,
            np.eye(basis.shape[1]),
            rtol=0.0,
            atol=1e-10,
        )
    ):
        raise D78GroundTangentError("D78 requires finite symmetric support and basis")
    return (
        np.ascontiguousarray(x),
        np.ascontiguousarray(y),
        np.ascontiguousarray(w),
        np.ascontiguousarray(basis),
    )


def _top2_objective(
    coefficients: np.ndarray,
    projected: np.ndarray,
    base_margins: np.ndarray,
    targets: np.ndarray,
    rivals: np.ndarray,
    class_count: int,
    temperature: float,
    *,
    need_gradient: bool,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray | None]:
    correction = np.sum(
        (coefficients[targets] - coefficients[rivals]) * projected, axis=1
    )
    margins = base_margins + correction
    losses = np.logaddexp(0.0, -margins)
    per_class = np.asarray(
        [np.mean(losses[targets == index]) for index in range(class_count)],
        dtype=np.float64,
    )
    scaled = per_class / float(temperature)
    shifted = scaled - float(np.max(scaled))
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    objective = float(
        temperature
        * (np.log(np.mean(exp_shifted)) + float(np.max(scaled)))
    )
    if not need_gradient:
        return objective, per_class, margins, None
    gradient = np.zeros_like(coefficients, dtype=np.float64)
    sample_factor = np.empty(len(targets), dtype=np.float64)
    for class_index in range(class_count):
        selected = targets == class_index
        sample_factor[selected] = weights[class_index] / float(np.sum(selected))
    sigmoid_negative_margin = np.exp(-np.logaddexp(0.0, margins))
    scalar = -sample_factor * sigmoid_negative_margin
    for index in range(len(targets)):
        contribution = scalar[index] * projected[index]
        gradient[int(targets[index])] += contribution
        gradient[int(rivals[index])] -= contribution
    gradient -= np.mean(gradient, axis=0, keepdims=True)
    return objective, per_class, margins, gradient


def fit_ground_tangent_worstclass_margin(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    *,
    base_coefficient: np.ndarray,
    tangent_basis: np.ndarray,
    lda_fit: LDAFit,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a D62-row residual learned only in the ground tangent subspace."""

    x, y, base_w, basis = _validate_support(
        rows, labels, class_count, k_shot, base_coefficient, tangent_basis
    )
    classes, dimension = int(class_count), int(x.shape[1])
    tangent_rank = int(basis.shape[1])
    zero = np.zeros((classes, dimension), dtype=np.float32)
    if int(k_shot) == 1:
        return _readonly(zero, np.float32), {
            "schema": "cvs.phase2.d78.ground_tangent_worstclass_margin_audit.v1",
            "status": "k1_exact_d62_fallback",
            "class_count": classes,
            "k_shot": 1,
            "dimension": dimension,
            "tangent_rank": tangent_rank,
            "crossfit_fold_count": 0,
            "crossfit_lda_fit_count": 0,
            "crossfit_held_row_count": 0,
            "optimizer_iterations": 0,
            "frank_wolfe_iterations": 0,
            "optimizer_objective_trace": [],
            "residual_active": False,
            "residual_sha256": _sha256(zero),
            "oof_ce_delta_mean": 0.0,
            "oof_ce_delta_max_class": 0.0,
            "oof_ce_delta_min_class": 0.0,
            "oof_base_correct_count": 0,
            "oof_updated_correct_count": 0,
            "query_rows_used": 0,
            "ground_class_score_access": False,
        }

    ranks = _within_class_ranks(y, classes)
    held_score_rows: list[np.ndarray] = []
    held_feature_rows: list[np.ndarray] = []
    held_target_rows: list[np.ndarray] = []
    fold_coefficient_hashes: set[str] = set()
    for held_rank in range(int(k_shot)):
        held = ranks == held_rank
        train = ~held
        if int(np.sum(held)) != classes:
            raise D78GroundTangentError("D78 physical-rank holdout drift")
        coefficient, intercept, _ = lda_fit(
            x[train], y[train], classes, int(k_shot) - 1
        )
        w = np.asarray(coefficient, dtype=np.float64)
        b = np.asarray(intercept, dtype=np.float64)
        if (
            w.shape != (classes, dimension)
            or b.shape != (classes,)
            or not np.isfinite(w).all()
            or not np.isfinite(b).all()
        ):
            raise D78GroundTangentError("D78 crossfit LDA shape drift")
        held_score_rows.append(x[held] @ w.T + b[None, :])
        held_feature_rows.append(x[held])
        held_target_rows.append(y[held])
        fold_coefficient_hashes.add(_sha256(w.astype(np.float64)))

    flat_scores = np.stack(held_score_rows, axis=0).reshape(-1, classes)
    flat_features = np.stack(held_feature_rows, axis=0).reshape(-1, dimension)
    flat_targets = np.stack(held_target_rows, axis=0).reshape(-1)
    rival_scores = flat_scores.copy()
    rival_scores[np.arange(len(flat_targets)), flat_targets] = -np.inf
    rivals = np.argmax(rival_scores, axis=1).astype(np.int64)
    base_margins = (
        flat_scores[np.arange(len(flat_targets)), flat_targets]
        - flat_scores[np.arange(len(flat_targets)), rivals]
    )
    projected = flat_features @ basis
    coefficients = np.zeros((classes, tangent_rank), dtype=np.float64)
    initial_pair_losses = np.logaddexp(0.0, -base_margins)
    initial_class_losses = np.asarray(
        [
            np.mean(initial_pair_losses[flat_targets == index])
            for index in range(classes)
        ],
        dtype=np.float64,
    )
    temperature = float(np.mean(initial_class_losses))
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise D78GroundTangentError("D78 smooth-worst temperature drift")
    trust_radius = float(np.linalg.norm(base_w) / np.sqrt(classes * dimension))
    max_projected_norm_sq = float(np.max(np.sum(projected * projected, axis=1)))
    step_seed = 1.0 / max(
        0.5 * max_projected_norm_sq, float(np.finfo(np.float64).eps)
    )
    trace: list[dict[str, Any]] = []
    for iteration in range(1, OPTIMIZER_STEPS + 1):
        objective, per_class, margins, gradient = _top2_objective(
            coefficients,
            projected,
            base_margins,
            flat_targets,
            rivals,
            classes,
            temperature,
            need_gradient=True,
        )
        assert gradient is not None
        gradient_norm = float(np.linalg.norm(gradient))
        step = step_seed
        accepted = coefficients
        accepted_objective = objective
        accepted_per_class = per_class
        accepted_margins = margins
        backtracks = 0
        objective_tolerance = float(
            128.0
            * np.finfo(np.float64).eps
            * max(1.0, abs(objective))
        )
        for backtracks in range(MAX_BACKTRACKS + 1):
            proposal = coefficients - step * gradient
            proposal -= np.mean(proposal, axis=0, keepdims=True)
            proposal_norm = float(np.linalg.norm(proposal))
            if proposal_norm > trust_radius:
                proposal *= trust_radius / proposal_norm
            candidate_objective, candidate_class, candidate_margins, _ = (
                _top2_objective(
                    proposal,
                    projected,
                    base_margins,
                    flat_targets,
                    rivals,
                    classes,
                    temperature,
                    need_gradient=False,
                )
            )
            if candidate_objective <= objective + objective_tolerance:
                accepted = proposal
                accepted_objective = candidate_objective
                accepted_per_class = candidate_class
                accepted_margins = candidate_margins
                break
            step *= 0.5
        else:
            raise D78GroundTangentError("D78 deterministic backtracking failed")
        coefficients = accepted
        trace.append(
            {
                "iteration": iteration,
                "objective_before": objective,
                "objective_after": accepted_objective,
                "line_search_gamma": step,
                "accepted_step": step,
                "backtracking_count": int(backtracks),
                "gradient_frobenius": gradient_norm,
                "worst_class_loss_before": float(np.max(per_class)),
                "worst_class_loss_after": float(np.max(accepted_per_class)),
                "mean_margin_before_step": float(np.mean(margins)),
                "mean_margin_after_step": float(np.mean(accepted_margins)),
            }
        )

    delta64 = coefficients @ basis.T
    delta64 -= np.mean(delta64, axis=0, keepdims=True)
    updated_scores = flat_scores + flat_features @ delta64.T
    final_objective, final_class_losses, final_margins, _ = _top2_objective(
        coefficients,
        projected,
        base_margins,
        flat_targets,
        rivals,
        classes,
        temperature,
        need_gradient=False,
    )
    initial_objective, _, _, _ = _top2_objective(
        np.zeros_like(coefficients),
        projected,
        base_margins,
        flat_targets,
        rivals,
        classes,
        temperature,
        need_gradient=False,
    )
    if final_objective > initial_objective + 1e-10:
        raise D78GroundTangentError("D78 smooth-worst objective increased")
    before_ce_rows = _cross_entropy(flat_scores, flat_targets)
    after_ce_rows = _cross_entropy(updated_scores, flat_targets)
    before_ce = np.asarray(
        [np.mean(before_ce_rows[flat_targets == index]) for index in range(classes)]
    )
    after_ce = np.asarray(
        [np.mean(after_ce_rows[flat_targets == index]) for index in range(classes)]
    )
    base_prediction = np.argmax(flat_scores, axis=1)
    updated_prediction = np.argmax(updated_scores, axis=1)
    residual_norm = float(np.linalg.norm(delta64))
    numeric_tolerance = float(
        256.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.linalg.norm(base_w)))
    )
    active = bool(residual_norm > numeric_tolerance)
    delta = _readonly(delta64, np.float32)
    audit = {
        "schema": "cvs.phase2.d78.ground_tangent_worstclass_margin_audit.v1",
        "status": (
            "ground_tangent_worstclass_top2_margin_active"
            if active
            else "zero_projected_gradient_exact_d62_fallback"
        ),
        "class_count": classes,
        "k_shot": int(k_shot),
        "dimension": dimension,
        "tangent_rank": tangent_rank,
        "crossfit_fold_count": int(k_shot),
        "crossfit_lda_fit_count": int(k_shot),
        "crossfit_held_row_count": classes * int(k_shot),
        "crossfit_unique_lda_coefficient_count": len(fold_coefficient_hashes),
        "class_gradient_count": classes,
        "optimizer_iterations": OPTIMIZER_STEPS,
        # Compatibility alias consumed by the D77 integration scaffold only.
        "frank_wolfe_iterations": OPTIMIZER_STEPS,
        "optimizer_objective_trace": trace,
        "temperature_from_initial_mean_class_loss": temperature,
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "objective_delta": final_objective - initial_objective,
        "initial_worst_class_top2_loss": float(np.max(initial_class_losses)),
        "final_worst_class_top2_loss": float(np.max(final_class_losses)),
        "initial_mean_class_top2_loss": float(np.mean(initial_class_losses)),
        "final_mean_class_top2_loss": float(np.mean(final_class_losses)),
        "oof_per_class_top2_loss_before": [
            float(value) for value in initial_class_losses
        ],
        "oof_per_class_top2_loss_after": [
            float(value) for value in final_class_losses
        ],
        "base_margin_min": float(np.min(base_margins)),
        "base_margin_mean": float(np.mean(base_margins)),
        "final_margin_min": float(np.min(final_margins)),
        "final_margin_mean": float(np.mean(final_margins)),
        "nonpositive_margin_count_before": int(np.sum(base_margins <= 0.0)),
        "nonpositive_margin_count_after": int(np.sum(final_margins <= 0.0)),
        "trust_radius_frobenius": trust_radius,
        "tangent_coefficient_frobenius": float(np.linalg.norm(coefficients)),
        "residual_active": active,
        "residual_frobenius": residual_norm,
        "residual_sha256": _sha256(delta.astype(np.float32)),
        "oof_ce_before_mean": float(np.mean(before_ce)),
        "oof_ce_after_mean": float(np.mean(after_ce)),
        "oof_ce_delta_mean": float(np.mean(after_ce - before_ce)),
        "oof_ce_delta_max_class": float(np.max(after_ce - before_ce)),
        "oof_ce_delta_min_class": float(np.min(after_ce - before_ce)),
        "oof_per_class_ce_before": [float(value) for value in before_ce],
        "oof_per_class_ce_after": [float(value) for value in after_ce],
        "oof_per_class_ce_delta": [
            float(value) for value in (after_ce - before_ce)
        ],
        "oof_base_correct_count": int(np.sum(base_prediction == flat_targets)),
        "oof_updated_correct_count": int(
            np.sum(updated_prediction == flat_targets)
        ),
        "oof_correct_delta": int(
            np.sum(updated_prediction == flat_targets)
            - np.sum(base_prediction == flat_targets)
        ),
        "numeric_tolerance": numeric_tolerance,
        "class_id_specific_formula": False,
        "class_permutation_equivariant": True,
        "old_new_role_specific_branch": False,
        "scene_receiver_handle_specific_branch": False,
        "query_rows_used": 0,
        "ground_class_score_access": False,
    }
    return delta, audit


# Names used by the D77 integration scaffold inherited by the D78 probe.
FW_ITERATIONS = OPTIMIZER_STEPS


def fit_ground_preconditioned_common_descent(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    *,
    base_coefficient: np.ndarray,
    preconditioner: np.ndarray,
    lda_fit: LDAFit,
) -> tuple[np.ndarray, dict[str, Any]]:
    return fit_ground_tangent_worstclass_margin(
        rows,
        labels,
        class_count,
        k_shot,
        base_coefficient=base_coefficient,
        tangent_basis=preconditioner,
        lda_fit=lda_fit,
    )


__all__ = [
    "D78GroundTangentError",
    "MAX_BACKTRACKS",
    "OPTIMIZER_STEPS",
    "ground_domain_tangent_basis",
    "fit_ground_tangent_worstclass_margin",
]
