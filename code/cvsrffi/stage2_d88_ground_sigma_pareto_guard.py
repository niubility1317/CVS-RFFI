"""D88 class-symmetric Pareto-safe ground sigma-margin residual.

D88 preserves D87's immutable ground-radius sigma geometry and support-only
smooth-worst objective.  Its only mechanism change is a class-permutation
equivariant projection of every descent direction onto the common clean-OOF
descent cone.  Exact line search then requires every registered class's clean
OOF cross-entropy to be non-increasing.  Ground knowledge proposes domain
directions; target support alone decides whether they are safe to retain.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


_D87_PATH = Path(__file__).with_name(
    "stage2_d87_ground_radius_sigma_margin.py"
)
_SPEC = importlib.util.spec_from_file_location("d88_d87_sigma", _D87_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("D88 could not load the locked D87 sigma geometry")
d87 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = d87
_SPEC.loader.exec_module(d87)


Z_DIM = d87.Z_DIM
OPTIMIZER_STEPS = d87.OPTIMIZER_STEPS
MAX_BACKTRACKS = d87.MAX_BACKTRACKS
CONE_PROJECTION_SWEEPS = 64


class D88GroundSigmaParetoError(RuntimeError):
    """Raised when the D88 support-only Pareto closure drifts."""


LDAFit = Callable[
    [np.ndarray, np.ndarray, int, int],
    tuple[np.ndarray, np.ndarray, dict[str, Any]],
]


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _class_means(values: np.ndarray, targets: np.ndarray, classes: int) -> np.ndarray:
    return np.asarray(
        [np.mean(values[targets == index]) for index in range(classes)],
        dtype=np.float64,
    )


def _clean_class_gradients(
    coefficients: np.ndarray,
    projected: np.ndarray,
    base_scores: np.ndarray,
    targets: np.ndarray,
    classes: int,
) -> np.ndarray:
    """Return centered coefficient gradients for each class clean OOF CE."""

    scores = base_scores + projected @ coefficients.T
    error = d87._softmax_error(scores, targets)
    gradients = np.empty(
        (classes, classes, projected.shape[1]), dtype=np.float64
    )
    for class_index in range(classes):
        selected = targets == class_index
        gradient = np.einsum(
            "nc,nr->cr", error[selected], projected[selected]
        ) / int(np.sum(selected))
        gradient -= np.mean(gradient, axis=0, keepdims=True)
        gradients[class_index] = gradient
    return gradients


def project_common_clean_descent(
    direction: np.ndarray,
    class_gradients: np.ndarray,
    *,
    max_sweeps: int = CONE_PROJECTION_SWEEPS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project a direction onto all class clean-CE descent halfspaces.

    Simultaneous averaged halfspace projection is deterministic and depends
    only on the unordered set of registered-class gradients.  The exact
    nonlinear line-search guard remains authoritative.
    """

    projected = np.asarray(direction, dtype=np.float64).copy()
    gradients = np.asarray(class_gradients, dtype=np.float64)
    if gradients.ndim != 3 or projected.shape != gradients.shape[1:]:
        raise D88GroundSigmaParetoError("D88 cone projection shape drift")
    projected -= np.mean(projected, axis=0, keepdims=True)
    tolerance = float(
        512.0
        * np.finfo(np.float64).eps
        * max(1.0, np.linalg.norm(projected))
        * max(1.0, np.max(np.linalg.norm(gradients.reshape(len(gradients), -1), axis=1)))
    )
    sweep_count = 0
    projection_count = 0
    gradient_norm_sq = np.sum(gradients * gradients, axis=(1, 2))
    for sweep in range(1, int(max_sweeps) + 1):
        violations = np.einsum("kcr,cr->k", gradients, projected)
        active = np.logical_and(
            violations > tolerance,
            gradient_norm_sq > np.finfo(np.float64).eps,
        )
        if np.any(active):
            scales = violations[active] / gradient_norm_sq[active]
            correction = np.einsum(
                "k,kcr->cr", scales, gradients[active]
            ) / int(np.sum(active))
            projected -= correction
            projected -= np.mean(projected, axis=0, keepdims=True)
            projection_count += int(np.sum(active))
        sweep_count = sweep
        violations = np.einsum("kcr,cr->k", gradients, projected)
        if float(np.max(violations)) <= tolerance:
            break
    violations = np.einsum("kcr,cr->k", gradients, projected)
    maximum = float(np.max(violations))
    if maximum > 16.0 * tolerance:
        # The zero vector is always feasible and is the conservative exact
        # fallback if finite cyclic projections do not close numerically.
        projected.fill(0.0)
        violations.fill(0.0)
        maximum = 0.0
    return projected, {
        "projection_sweeps": int(sweep_count),
        "halfspace_projection_count": int(projection_count),
        "maximum_linearized_class_violation": maximum,
        "projection_tolerance": tolerance,
        "projected_direction_frobenius": float(np.linalg.norm(projected)),
    }


def ground_radius_sigma_geometry(
    domain_class_prototypes: np.ndarray,
    domain_class_radius: np.ndarray,
    *,
    feature_dim: int = 288,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reuse the exact D87 v2 geometry with a D88 provenance wrapper."""

    basis, offsets, inherited = d87.ground_radius_sigma_geometry(
        domain_class_prototypes, domain_class_radius, feature_dim=feature_dim
    )
    audit = dict(inherited)
    audit["schema"] = "cvs.phase2.d88.ground_sigma_pareto_geometry.v1"
    audit["d87_geometry_reused_bit_exact"] = True
    return basis, offsets, audit


def fit_ground_sigma_pareto_guard(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    *,
    base_coefficient: np.ndarray,
    tangent_basis: np.ndarray,
    counterfactual_offsets: np.ndarray,
    lda_fit: LDAFit,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit D87 sigma risk under an all-class clean-OOF Pareto guard."""

    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    base_w = np.asarray(base_coefficient, dtype=np.float64)
    basis = np.asarray(tangent_basis, dtype=np.float64)
    offsets = np.asarray(counterfactual_offsets, dtype=np.float64)
    classes, shots = int(class_count), int(k_shot)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or base_w.shape != (classes, x.shape[1])
        or basis.ndim != 2
        or basis.shape[0] != x.shape[1]
        or offsets.ndim != 2
        or offsets.shape[1] != x.shape[1]
        or offsets.shape[0] < 2
        or not np.isfinite(x).all()
        or not np.isfinite(base_w).all()
        or not np.isfinite(basis).all()
        or not np.isfinite(offsets).all()
        or len(x) != classes * shots
        or classes < 2
        or shots < 1
        or not np.array_equal(np.unique(y), np.arange(classes))
        or any(np.sum(y == index) != shots for index in range(classes))
        or not np.allclose(
            basis.T @ basis, np.eye(basis.shape[1]), rtol=0.0, atol=1.0e-10
        )
        or np.any(offsets[:, Z_DIM:] != 0.0)
    ):
        raise D88GroundSigmaParetoError("D88 requires finite symmetric support geometry")
    dimension, rank = int(x.shape[1]), int(basis.shape[1])
    zero_w = np.zeros((classes, dimension), dtype=np.float32)
    zero_b = np.zeros(classes, dtype=np.float32)
    if shots == 1:
        return _readonly(zero_w, np.float32), _readonly(zero_b, np.float32), {
            "schema": "cvs.phase2.d88.ground_sigma_pareto_audit.v1",
            "status": "k1_exact_d62_fallback",
            "class_count": classes,
            "k_shot": shots,
            "effective_rank": rank,
            "counterfactual_domain_count": int(len(offsets)),
            "optimizer_iterations": 0,
            "residual_active": False,
            "query_rows_used": 0,
        }

    support_center = np.mean(x, axis=0)
    centered = np.ascontiguousarray(x - support_center[None, :])
    ranks = d87._within_class_ranks(y, classes)
    score_rows: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    view_delta_rows: list[np.ndarray] = []
    coefficient_hashes: set[str] = set()
    for held_rank in range(shots):
        held = ranks == held_rank
        train = ~held
        coefficient, intercept, _ = lda_fit(
            centered[train], y[train], classes, shots - 1
        )
        fold_w = np.asarray(coefficient, dtype=np.float64)
        fold_b = np.asarray(intercept, dtype=np.float64)
        if fold_w.shape != base_w.shape or fold_b.shape != (classes,):
            raise D88GroundSigmaParetoError("D88 crossfit LDA shape drift")
        score_rows.append(centered[held] @ fold_w.T + fold_b[None, :])
        feature_rows.append(centered[held])
        target_rows.append(y[held])
        view_delta_rows.append(offsets @ fold_w.T)
        coefficient_hashes.add(_sha256(fold_w))
    base_scores = np.stack(score_rows).reshape(-1, classes)
    features = np.stack(feature_rows).reshape(-1, dimension)
    targets = np.stack(target_rows).reshape(-1)
    fold_view_delta = np.stack(view_delta_rows)
    base_view_delta = np.repeat(fold_view_delta, classes, axis=0)
    projected = features @ basis
    offset_projected = offsets @ basis
    coefficients = np.zeros((classes, rank), dtype=np.float64)
    initial0 = d87._sigma_objective(
        coefficients, projected, offset_projected, base_scores,
        base_view_delta, targets, classes, 1.0, need_gradient=False
    )
    temperature = float(np.mean(initial0[1]))
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise D88GroundSigmaParetoError("D88 initial sigma temperature drift")
    initial = d87._sigma_objective(
        coefficients, projected, offset_projected, base_scores,
        base_view_delta, targets, classes, temperature, need_gradient=False
    )
    initial_clean_class = _class_means(initial[2], targets, classes)
    clean_guard_tolerance = float(
        4096.0 * np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(initial_clean_class))))
    )
    trust_radius = float(np.linalg.norm(base_w) / np.sqrt(classes * dimension))
    view_norm_sq = np.sum(
        np.square(projected[:, None, :] + offset_projected[None, :, :]), axis=2
    )
    max_view_norm_sq = float(
        max(np.max(view_norm_sq), np.max(np.sum(projected * projected, axis=1)))
    )
    step_seed = 1.0 / max(
        0.5 * max_view_norm_sq, float(np.finfo(np.float64).eps)
    )
    trace: list[dict[str, Any]] = []
    total_halfspace_projections = 0
    zero_direction_steps = 0
    for iteration in range(1, OPTIMIZER_STEPS + 1):
        current = d87._sigma_objective(
            coefficients, projected, offset_projected, base_scores,
            base_view_delta, targets, classes, temperature, need_gradient=True
        )
        objective, per_class, clean_loss, sigma_loss, gradient = current
        assert gradient is not None
        clean_class = _class_means(clean_loss, targets, classes)
        clean_gradients = _clean_class_gradients(
            coefficients, projected, base_scores, targets, classes
        )
        direction, cone_audit = project_common_clean_descent(
            -gradient, clean_gradients
        )
        total_halfspace_projections += int(cone_audit["halfspace_projection_count"])
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 256.0 * np.finfo(np.float64).eps:
            zero_direction_steps += 1
            candidate = current
            step = 0.0
            backtracks = 0
        else:
            step = step_seed
            tolerance = float(
                128.0 * np.finfo(np.float64).eps * max(1.0, abs(objective))
            )
            for backtracks in range(MAX_BACKTRACKS + 1):
                proposal = coefficients + step * direction
                proposal -= np.mean(proposal, axis=0, keepdims=True)
                norm = float(np.linalg.norm(proposal))
                if norm > trust_radius:
                    proposal *= trust_radius / norm
                candidate = d87._sigma_objective(
                    proposal, projected, offset_projected, base_scores,
                    base_view_delta, targets, classes, temperature,
                    need_gradient=False
                )
                candidate_clean_class = _class_means(
                    candidate[2], targets, classes
                )
                if (
                    candidate[0] <= objective + tolerance
                    and np.all(
                        candidate_clean_class
                        <= initial_clean_class + clean_guard_tolerance
                    )
                ):
                    coefficients = proposal
                    break
                step *= 0.5
            else:
                raise D88GroundSigmaParetoError(
                    "D88 deterministic Pareto backtracking failed"
                )
        candidate_clean_class = _class_means(candidate[2], targets, classes)
        trace.append({
            "iteration": iteration,
            "objective_before": float(objective),
            "objective_after": float(candidate[0]),
            "accepted_step": float(step),
            "line_search_gamma": float(step),
            "backtracking_count": int(backtracks),
            "gradient_frobenius": float(np.linalg.norm(gradient)),
            "projected_direction_frobenius": direction_norm,
            "worst_class_sigma_ce_before": float(np.max(per_class)),
            "worst_class_sigma_ce_after": float(np.max(candidate[1])),
            "clean_ce_mean_before": float(np.mean(clean_loss)),
            "clean_ce_max_class_delta_step": float(
                np.max(candidate_clean_class - clean_class)
            ),
            "clean_ce_max_class_delta_vs_initial": float(
                np.max(candidate_clean_class - initial_clean_class)
            ),
            "sigma_ce_mean_before": float(np.mean(sigma_loss)),
            **cone_audit,
        })

    final = d87._sigma_objective(
        coefficients, projected, offset_projected, base_scores,
        base_view_delta, targets, classes, temperature, need_gradient=False
    )
    final_clean_class = _class_means(final[2], targets, classes)
    if (
        final[0] > initial[0] + 1.0e-10
        or np.any(
            final_clean_class > initial_clean_class + clean_guard_tolerance
        )
    ):
        raise D88GroundSigmaParetoError("D88 final Pareto invariant drift")
    delta_w64 = coefficients @ basis.T
    delta_w64 -= np.mean(delta_w64, axis=0, keepdims=True)
    delta_b64 = -(delta_w64 @ support_center)
    center_logit = delta_w64 @ support_center + delta_b64
    tolerance = float(
        256.0 * np.finfo(np.float64).eps * max(1.0, np.linalg.norm(base_w))
    )
    if float(np.max(np.abs(center_logit))) > tolerance:
        raise D88GroundSigmaParetoError("D88 centered affine compile drift")
    residual_norm = float(np.linalg.norm(delta_w64))
    active = residual_norm > tolerance
    delta_w = _readonly(delta_w64, np.float32)
    delta_b = _readonly(delta_b64, np.float32)
    clean_delta = final_clean_class - initial_clean_class
    audit = {
        "schema": "cvs.phase2.d88.ground_sigma_pareto_audit.v1",
        "status": (
            "ground_sigma_pareto_guard_active"
            if active else "no_common_clean_descent_exact_d62_fallback"
        ),
        "class_count": classes,
        "k_shot": shots,
        "dimension": dimension,
        "effective_rank": rank,
        "tangent_rank": rank,
        "counterfactual_domain_count": int(len(offsets)),
        "crossfit_fold_count": shots,
        "crossfit_lda_fit_count": shots,
        "crossfit_held_row_count": classes * shots,
        "crossfit_unique_lda_coefficient_count": len(coefficient_hashes),
        "sigma_weight_original": 0.5,
        "sigma_weight_plus": 0.25,
        "sigma_weight_minus": 0.25,
        "optimizer_iterations": OPTIMIZER_STEPS,
        "optimizer_objective_trace": trace,
        "temperature_from_initial_mean_class_sigma_ce": temperature,
        "initial_objective": float(initial[0]),
        "final_objective": float(final[0]),
        "objective_delta": float(final[0] - initial[0]),
        "initial_worst_class_sigma_ce": float(np.max(initial[1])),
        "final_worst_class_sigma_ce": float(np.max(final[1])),
        "initial_mean_sigma_ce": float(np.mean(initial[3])),
        "final_mean_sigma_ce": float(np.mean(final[3])),
        "oof_clean_ce_before_mean": float(np.mean(initial_clean_class)),
        "oof_clean_ce_after_mean": float(np.mean(final_clean_class)),
        "oof_clean_ce_delta_max_class": float(np.max(clean_delta)),
        "oof_clean_ce_delta_min_class": float(np.min(clean_delta)),
        "all_class_clean_ce_nonincrease_verified": bool(
            np.all(clean_delta <= clean_guard_tolerance)
        ),
        "clean_pareto_guard_tolerance": clean_guard_tolerance,
        "total_halfspace_projection_count": int(total_halfspace_projections),
        "zero_common_direction_step_count": int(zero_direction_steps),
        "cone_projection_sweeps_per_step_max": CONE_PROJECTION_SWEEPS,
        # Compatibility aliases consumed by the inherited D79 integration.
        "oof_ce_before_mean": float(np.mean(initial_clean_class)),
        "oof_ce_after_mean": float(np.mean(final_clean_class)),
        "oof_ce_delta_mean": float(np.mean(clean_delta)),
        "oof_ce_delta_max_class": float(np.max(clean_delta)),
        "oof_clean_correct_before": int(
            np.sum(np.argmax(base_scores, axis=1) == targets)
        ),
        "oof_clean_correct_after": int(
            np.sum(np.argmax(base_scores + projected @ coefficients.T, axis=1) == targets)
        ),
        "oof_base_correct_count": int(
            np.sum(np.argmax(base_scores, axis=1) == targets)
        ),
        "oof_updated_correct_count": int(
            np.sum(np.argmax(base_scores + projected @ coefficients.T, axis=1) == targets)
        ),
        "trust_radius_frobenius": trust_radius,
        "residual_frobenius": residual_norm,
        "bias_residual_frobenius": float(np.linalg.norm(delta_b64)),
        "residual_active": active,
        "residual_sha256": _sha256(delta_w),
        "bias_residual_sha256": _sha256(delta_b),
        "support_center_sha256": _sha256(support_center),
        "residual_logit_at_support_center_max_abs": float(
            np.max(np.abs(center_logit))
        ),
        "counterfactual_views_count_as_physical_samples": False,
        "physical_group_crossfit_preserved": True,
        "class_permutation_equivariant": True,
        "old_new_role_specific_branch": False,
        "class_id_specific_formula": False,
        "query_rows_used": 0,
        "ground_class_score_access": False,
    }
    return delta_w, delta_b, audit


__all__ = [
    "CONE_PROJECTION_SWEEPS",
    "D88GroundSigmaParetoError",
    "MAX_BACKTRACKS",
    "OPTIMIZER_STEPS",
    "fit_ground_sigma_pareto_guard",
    "ground_radius_sigma_geometry",
    "project_common_clean_descent",
]
