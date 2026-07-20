"""D87 radius-scaled ground sigma-point margin head residual.

The Phase1 component contributes class-agnostic ground-domain directions and
aggregated p90 radii.  Target support labels alone determine a centered affine
residual by minimizing a symmetric non-quadratic all-class margin risk over
physical-rank out-of-fold predictions.  Counterfactual feature views never
become additional physical samples and are not retained for query inference.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


_D86_PATH = Path(__file__).with_name(
    "stage2_d86_ground_radius_counterfactual_center.py"
)
_SPEC = importlib.util.spec_from_file_location("d87_d86_geometry", _D86_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("D87 could not load the locked D86 v2 geometry")
d86 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = d86
_SPEC.loader.exec_module(d86)


Z_DIM = 160
OPTIMIZER_STEPS = 20
MAX_BACKTRACKS = 64


class D87GroundSigmaMarginError(RuntimeError):
    """Raised when the D87 geometry, crossfit, or optimizer closure drifts."""


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


def _within_class_ranks(labels: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.zeros(int(class_count), dtype=np.int64)
    ranks = np.empty(len(labels), dtype=np.int64)
    for row_index, class_index in enumerate(labels):
        ranks[row_index] = counts[int(class_index)]
        counts[int(class_index)] += 1
    return ranks


def _cross_entropy(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    normalizer = np.log(np.sum(np.exp(shifted), axis=-1))
    return normalizer - np.take_along_axis(
        shifted, targets[..., None], axis=-1
    )[..., 0]


def _softmax_error(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    probability = exponential / np.sum(exponential, axis=-1, keepdims=True)
    error = probability
    flat = error.reshape(-1, error.shape[-1])
    flat_targets = np.broadcast_to(targets, error.shape[:-1]).reshape(-1)
    flat[np.arange(len(flat)), flat_targets] -= 1.0
    return error


def ground_radius_sigma_geometry(
    domain_class_prototypes: np.ndarray,
    domain_class_radius: np.ndarray,
    *,
    feature_dim: int = 288,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return orthonormal span and explicit symmetric v2 p90 offsets."""

    if int(feature_dim) < Z_DIM:
        raise D87GroundSigmaMarginError("D87 feature dimension is below z160")
    templates, amplitude, inherited = d86.ground_radius_counterfactual_templates(
        domain_class_prototypes, domain_class_radius
    )
    offsets_z = np.ascontiguousarray(templates.T * amplitude[:, None])
    _, singular, right = np.linalg.svd(offsets_z, full_matrices=False)
    tolerance = float(
        max(offsets_z.shape)
        * np.finfo(np.float64).eps
        * (float(singular[0]) if len(singular) else 0.0)
    )
    rank = int(np.sum(singular > tolerance))
    if rank < 1:
        raise D87GroundSigmaMarginError("D87 counterfactual span has zero rank")
    basis = np.zeros((int(feature_dim), rank), dtype=np.float64)
    basis[:Z_DIM] = right[:rank].T
    offsets = np.zeros((len(offsets_z), int(feature_dim)), dtype=np.float64)
    offsets[:, :Z_DIM] = offsets_z
    gram = basis.T @ basis
    covariance = (offsets.T @ offsets) / (2.0 * len(offsets))
    direct_covariance = np.zeros_like(covariance)
    radius = np.median(np.asarray(domain_class_radius, dtype=np.float64), axis=1)
    for domain_index in range(len(offsets_z)):
        direction = templates[:, domain_index]
        direct_covariance[:Z_DIM, :Z_DIM] += (
            radius[domain_index] * np.outer(direction, direction) / len(offsets_z)
        )
    covariance_error = float(np.max(np.abs(covariance - direct_covariance)))
    if (
        not np.allclose(gram, np.eye(rank), rtol=0.0, atol=1.0e-10)
        or covariance_error > 1.0e-14
        or np.any(offsets[:, Z_DIM:] != 0.0)
    ):
        raise D87GroundSigmaMarginError("D87 sigma geometry invariant drift")
    basis = _readonly(basis, np.float64)
    offsets = _readonly(offsets, np.float64)
    audit = dict(inherited)
    audit.update(
        {
            "schema": "cvs.phase2.d87.ground_radius_sigma_geometry.v1",
            "geometry_policy": (
                "explicit_symmetric_sqrt_two_domain_median_p90_offsets"
            ),
            "counterfactual_domain_count": int(len(offsets)),
            "effective_rank": rank,
            "rank_rule": "numerical_rank_without_scan",
            "basis_orthonormality_max_abs_error": float(
                np.max(np.abs(gram - np.eye(rank)))
            ),
            "sigma_covariance_max_abs_error": covariance_error,
            "basis_sha256": _sha256(basis),
            "offsets_sha256": _sha256(offsets),
            "rank_scan_count": 0,
            "counterfactual_views_materialized_as_support": False,
            "physical_sample_count_multiplier": 1,
            "ground_class_centers_discarded": True,
            "ground_target_identity_mapping_access": False,
        }
    )
    return basis, offsets, audit


def _sigma_objective(
    coefficients: np.ndarray,
    projected: np.ndarray,
    offset_projected: np.ndarray,
    base_scores: np.ndarray,
    base_view_delta: np.ndarray,
    targets: np.ndarray,
    class_count: int,
    temperature: float,
    *,
    need_gradient: bool,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Evaluate 1/2 clean + 1/4 plus + 1/4 minus smooth-worst CE."""

    correction = projected @ coefficients.T
    offset_correction = offset_projected @ coefficients.T
    clean_scores = base_scores + correction
    plus_scores = (
        base_scores[:, None, :]
        + base_view_delta
        + correction[:, None, :]
        + offset_correction[None, :, :]
    )
    minus_scores = (
        base_scores[:, None, :]
        - base_view_delta
        + correction[:, None, :]
        - offset_correction[None, :, :]
    )
    clean_loss = _cross_entropy(clean_scores, targets)
    expanded_targets = np.broadcast_to(targets[:, None], plus_scores.shape[:-1])
    plus_loss = _cross_entropy(plus_scores, expanded_targets)
    minus_loss = _cross_entropy(minus_scores, expanded_targets)
    sigma_loss = (
        0.5 * clean_loss
        + 0.25 * np.mean(plus_loss, axis=1)
        + 0.25 * np.mean(minus_loss, axis=1)
    )
    per_class = np.asarray(
        [np.mean(sigma_loss[targets == index]) for index in range(class_count)],
        dtype=np.float64,
    )
    scaled = per_class / float(temperature)
    maximum = float(np.max(scaled))
    exponential = np.exp(scaled - maximum)
    class_weight = exponential / np.sum(exponential)
    objective = float(
        temperature * (np.log(np.mean(exponential)) + maximum)
    )
    if not need_gradient:
        return objective, per_class, clean_loss, sigma_loss, None

    sample_weight = np.empty(len(targets), dtype=np.float64)
    for class_index in range(class_count):
        selected = targets == class_index
        sample_weight[selected] = class_weight[class_index] / np.sum(selected)
    clean_error = _softmax_error(clean_scores, targets)
    plus_error = _softmax_error(plus_scores, expanded_targets)
    minus_error = _softmax_error(minus_scores, expanded_targets)
    gradient = 0.5 * np.einsum(
        "n,nc,nr->cr", sample_weight, clean_error, projected
    )
    domain_count = int(offset_projected.shape[0])
    plus_features = projected[:, None, :] + offset_projected[None, :, :]
    minus_features = projected[:, None, :] - offset_projected[None, :, :]
    gradient += (0.25 / domain_count) * np.einsum(
        "n,ndc,ndr->cr", sample_weight, plus_error, plus_features
    )
    gradient += (0.25 / domain_count) * np.einsum(
        "n,ndc,ndr->cr", sample_weight, minus_error, minus_features
    )
    gradient -= np.mean(gradient, axis=0, keepdims=True)
    return objective, per_class, clean_loss, sigma_loss, gradient


def fit_ground_radius_sigma_margin(
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
    """Fit a centered head residual from grouped OOF symmetric sigma risk."""

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
        raise D87GroundSigmaMarginError("D87 requires finite symmetric support geometry")
    dimension, rank = int(x.shape[1]), int(basis.shape[1])
    zero_w = np.zeros((classes, dimension), dtype=np.float32)
    zero_b = np.zeros(classes, dtype=np.float32)
    if shots == 1:
        return _readonly(zero_w, np.float32), _readonly(zero_b, np.float32), {
            "schema": "cvs.phase2.d87.ground_radius_sigma_margin_audit.v1",
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
    ranks = _within_class_ranks(y, classes)
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
            raise D87GroundSigmaMarginError("D87 crossfit LDA shape drift")
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
    initial = _sigma_objective(
        coefficients,
        projected,
        offset_projected,
        base_scores,
        base_view_delta,
        targets,
        classes,
        1.0,
        need_gradient=False,
    )
    initial_class = initial[1]
    temperature = float(np.mean(initial_class))
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise D87GroundSigmaMarginError("D87 initial sigma temperature drift")
    initial = _sigma_objective(
        coefficients,
        projected,
        offset_projected,
        base_scores,
        base_view_delta,
        targets,
        classes,
        temperature,
        need_gradient=False,
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
    for iteration in range(1, OPTIMIZER_STEPS + 1):
        current = _sigma_objective(
            coefficients,
            projected,
            offset_projected,
            base_scores,
            base_view_delta,
            targets,
            classes,
            temperature,
            need_gradient=True,
        )
        objective, per_class, clean_loss, sigma_loss, gradient = current
        assert gradient is not None
        step = step_seed
        tolerance = float(
            128.0 * np.finfo(np.float64).eps * max(1.0, abs(objective))
        )
        for backtracks in range(MAX_BACKTRACKS + 1):
            proposal = coefficients - step * gradient
            proposal -= np.mean(proposal, axis=0, keepdims=True)
            norm = float(np.linalg.norm(proposal))
            if norm > trust_radius:
                proposal *= trust_radius / norm
            candidate = _sigma_objective(
                proposal,
                projected,
                offset_projected,
                base_scores,
                base_view_delta,
                targets,
                classes,
                temperature,
                need_gradient=False,
            )
            if candidate[0] <= objective + tolerance:
                coefficients = proposal
                break
            step *= 0.5
        else:
            raise D87GroundSigmaMarginError("D87 deterministic backtracking failed")
        trace.append(
            {
                "iteration": iteration,
                "objective_before": objective,
                "objective_after": candidate[0],
                "accepted_step": step,
                "line_search_gamma": step,
                "backtracking_count": int(backtracks),
                "gradient_frobenius": float(np.linalg.norm(gradient)),
                "worst_class_sigma_ce_before": float(np.max(per_class)),
                "worst_class_sigma_ce_after": float(np.max(candidate[1])),
                "clean_ce_mean_before": float(np.mean(clean_loss)),
                "sigma_ce_mean_before": float(np.mean(sigma_loss)),
            }
        )

    final = _sigma_objective(
        coefficients,
        projected,
        offset_projected,
        base_scores,
        base_view_delta,
        targets,
        classes,
        temperature,
        need_gradient=False,
    )
    if final[0] > initial[0] + 1.0e-10:
        raise D87GroundSigmaMarginError("D87 sigma objective increased")
    delta_w64 = coefficients @ basis.T
    delta_w64 -= np.mean(delta_w64, axis=0, keepdims=True)
    delta_b64 = -(delta_w64 @ support_center)
    center_logit = delta_w64 @ support_center + delta_b64
    tolerance = float(
        256.0 * np.finfo(np.float64).eps * max(1.0, np.linalg.norm(base_w))
    )
    if float(np.max(np.abs(center_logit))) > tolerance:
        raise D87GroundSigmaMarginError("D87 centered affine compile drift")
    # OOF base scores were fitted/evaluated in globally centered coordinates;
    # the compiled bias is only needed when the same residual reads original x.
    updated_clean_scores = base_scores + features @ delta_w64.T
    clean_before = _cross_entropy(base_scores, targets)
    clean_after = _cross_entropy(updated_clean_scores, targets)
    clean_before_class = np.asarray(
        [np.mean(clean_before[targets == index]) for index in range(classes)]
    )
    clean_after_class = np.asarray(
        [np.mean(clean_after[targets == index]) for index in range(classes)]
    )
    residual_norm = float(np.linalg.norm(delta_w64))
    active = residual_norm > tolerance
    delta_w = _readonly(delta_w64, np.float32)
    delta_b = _readonly(delta_b64, np.float32)
    audit = {
        "schema": "cvs.phase2.d87.ground_radius_sigma_margin_audit.v1",
        "status": (
            "ground_radius_sigma_margin_active"
            if active
            else "zero_sigma_gradient_exact_d62_fallback"
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
        "oof_clean_ce_before_mean": float(np.mean(clean_before_class)),
        "oof_clean_ce_after_mean": float(np.mean(clean_after_class)),
        "oof_clean_ce_delta_max_class": float(
            np.max(clean_after_class - clean_before_class)
        ),
        # Compatibility aliases consumed only by the inherited D77/D79
        # integration scaffold; their values retain the D87 clean-OOF meaning.
        "oof_ce_before_mean": float(np.mean(clean_before_class)),
        "oof_ce_after_mean": float(np.mean(clean_after_class)),
        "oof_ce_delta_mean": float(
            np.mean(clean_after_class - clean_before_class)
        ),
        "oof_ce_delta_max_class": float(
            np.max(clean_after_class - clean_before_class)
        ),
        "oof_clean_correct_before": int(np.sum(np.argmax(base_scores, axis=1) == targets)),
        "oof_clean_correct_after": int(
            np.sum(np.argmax(updated_clean_scores, axis=1) == targets)
        ),
        "oof_base_correct_count": int(
            np.sum(np.argmax(base_scores, axis=1) == targets)
        ),
        "oof_updated_correct_count": int(
            np.sum(np.argmax(updated_clean_scores, axis=1) == targets)
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
    "D87GroundSigmaMarginError",
    "MAX_BACKTRACKS",
    "OPTIMIZER_STEPS",
    "fit_ground_radius_sigma_margin",
    "ground_radius_sigma_geometry",
]
