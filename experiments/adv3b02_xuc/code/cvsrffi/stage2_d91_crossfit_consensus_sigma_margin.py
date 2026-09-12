"""D91 crossfit-consensus shrinkage for the D87 ground sigma head.

The immutable compressed ground prototypes still define D87's rank-13
counterfactual sigma geometry.  D91 changes only the support-side confidence
assigned to the fitted residual: the initial sigma-risk gradient is computed
independently in every physical-rank OOF fold, normalized, and the D87
residual is multiplied by the clipped mean off-diagonal cosine agreement.
No threshold, class role, query row, or class quota is used.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


_D87_PATH = Path(__file__).with_name("stage2_d87_ground_radius_sigma_margin.py")
_SPEC = importlib.util.spec_from_file_location("d91_d87_sigma", _D87_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("D91 could not load the locked D87 sigma method")
d87 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = d87
_SPEC.loader.exec_module(d87)


Z_DIM = d87.Z_DIM
OPTIMIZER_STEPS = d87.OPTIMIZER_STEPS


class D91CrossfitConsensusError(RuntimeError):
    """Raised when D91 support-only consensus closure drifts."""


LDAFit = Callable[
    [np.ndarray, np.ndarray, int, int],
    tuple[np.ndarray, np.ndarray, dict[str, Any]],
]


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def ground_radius_sigma_geometry(
    domain_class_prototypes: np.ndarray,
    domain_class_radius: np.ndarray,
    *,
    feature_dim: int = 288,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reuse D87's compressed-v2 ground geometry bit exactly."""

    basis, offsets, inherited = d87.ground_radius_sigma_geometry(
        domain_class_prototypes, domain_class_radius, feature_dim=feature_dim
    )
    audit = dict(inherited)
    audit["schema"] = "cvs.phase2.d91.crossfit_consensus_sigma_geometry.v1"
    audit["d87_geometry_reused_bit_exact"] = True
    return basis, offsets, audit


def _oof_sigma_gradients(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: int,
    shots: int,
    *,
    tangent_basis: np.ndarray,
    counterfactual_offsets: np.ndarray,
    lda_fit: LDAFit,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fold gradients and their exact pooled OOF evaluation tensors."""

    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    basis = np.asarray(tangent_basis, dtype=np.float64)
    offsets = np.asarray(counterfactual_offsets, dtype=np.float64)
    center = np.mean(x, axis=0)
    centered = np.ascontiguousarray(x - center[None, :])
    ranks = d87._within_class_ranks(y, classes)
    offset_projected = offsets @ basis
    gradients: list[np.ndarray] = []
    score_rows: list[np.ndarray] = []
    projected_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    view_delta_rows: list[np.ndarray] = []
    for held_rank in range(shots):
        held = ranks == held_rank
        train = ~held
        coefficient, intercept, _ = lda_fit(
            centered[train], y[train], classes, shots - 1
        )
        fold_w = np.asarray(coefficient, dtype=np.float64)
        fold_b = np.asarray(intercept, dtype=np.float64)
        held_features = centered[held]
        held_targets = y[held]
        if (
            fold_w.shape != (classes, x.shape[1])
            or fold_b.shape != (classes,)
            or len(held_targets) != classes
            or not np.array_equal(np.sort(held_targets), np.arange(classes))
        ):
            raise D91CrossfitConsensusError("D91 physical-rank OOF shape drift")
        base_scores = held_features @ fold_w.T + fold_b[None, :]
        projected = held_features @ basis
        view_delta = np.repeat(
            (offsets @ fold_w.T)[None, :, :], classes, axis=0
        )
        zero = np.zeros((classes, basis.shape[1]), dtype=np.float64)
        result = d87._sigma_objective(
            zero,
            projected,
            offset_projected,
            base_scores,
            view_delta,
            held_targets,
            classes,
            temperature,
            need_gradient=True,
        )
        gradient = result[4]
        if gradient is None or not np.isfinite(gradient).all():
            raise D91CrossfitConsensusError("D91 non-finite fold gradient")
        gradients.append(np.asarray(gradient, dtype=np.float64))
        score_rows.append(base_scores)
        projected_rows.append(projected)
        target_rows.append(held_targets)
        view_delta_rows.append(view_delta)
    return (
        np.stack(gradients),
        np.concatenate(score_rows, axis=0),
        np.concatenate(projected_rows, axis=0),
        np.concatenate(target_rows, axis=0),
        np.concatenate(view_delta_rows, axis=0),
    )


def fit_crossfit_consensus_sigma_margin(
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
    """Fit D87 then shrink it by threshold-free OOF gradient agreement."""

    full_w, full_b, inherited = d87.fit_ground_radius_sigma_margin(
        rows,
        labels,
        class_count,
        k_shot,
        base_coefficient=base_coefficient,
        tangent_basis=tangent_basis,
        counterfactual_offsets=counterfactual_offsets,
        lda_fit=lda_fit,
    )
    audit = dict(inherited)
    audit["schema"] = "cvs.phase2.d91.crossfit_consensus_sigma_margin_audit.v1"
    audit["d87_unshrunk_status"] = inherited["status"]
    if int(k_shot) == 1:
        audit.update(
            {
                "status": "k1_exact_d62_fallback",
                "consensus_factor": 0.0,
                "query_rows_used": 0,
                "class_permutation_equivariant": True,
                "old_new_role_specific_branch": False,
                "class_id_specific_formula": False,
            }
        )
        return full_w, full_b, audit

    gradients, base_scores, projected, targets, base_view_delta = _oof_sigma_gradients(
        rows,
        labels,
        int(class_count),
        int(k_shot),
        tangent_basis=tangent_basis,
        counterfactual_offsets=counterfactual_offsets,
        lda_fit=lda_fit,
        temperature=float(inherited["temperature_from_initial_mean_class_sigma_ce"]),
    )
    flat = gradients.reshape(len(gradients), -1)
    norms = np.linalg.norm(flat, axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise D91CrossfitConsensusError("D91 zero fold gradient")
    unit = flat / norms[:, None]
    gram = unit @ unit.T
    fold_count = len(unit)
    off_diagonal = (float(np.sum(gram)) - fold_count) / (
        fold_count * (fold_count - 1)
    )
    consensus = float(np.clip(off_diagonal, 0.0, 1.0))
    mean_direction_norm = float(np.linalg.norm(np.mean(unit, axis=0)))
    scaled_w64 = np.asarray(full_w, dtype=np.float64) * consensus
    scaled_b64 = np.asarray(full_b, dtype=np.float64) * consensus
    scaled_w = _readonly(scaled_w64, np.float32)
    scaled_b = _readonly(scaled_b64, np.float32)
    center = np.mean(np.asarray(rows, dtype=np.float64), axis=0)
    center_error = float(
        np.max(np.abs(np.asarray(scaled_w, dtype=np.float64) @ center + scaled_b))
    )
    tolerance = float(
        1024.0
        * np.finfo(np.float32).eps
        * max(1.0, np.linalg.norm(np.asarray(full_w, dtype=np.float64)))
    )
    if center_error > tolerance:
        raise D91CrossfitConsensusError("D91 centered affine compile drift")
    basis = np.asarray(tangent_basis, dtype=np.float64)
    coefficients = scaled_w64 @ basis
    offset_projected = np.asarray(counterfactual_offsets, dtype=np.float64) @ basis
    final = d87._sigma_objective(
        coefficients,
        projected,
        offset_projected,
        base_scores,
        base_view_delta,
        targets,
        int(class_count),
        float(inherited["temperature_from_initial_mean_class_sigma_ce"]),
        need_gradient=False,
    )
    updated_clean_scores = base_scores + projected @ coefficients.T
    clean_before = d87._cross_entropy(base_scores, targets)
    clean_after = d87._cross_entropy(updated_clean_scores, targets)
    clean_before_class = np.asarray(
        [np.mean(clean_before[targets == index]) for index in range(int(class_count))]
    )
    clean_after_class = np.asarray(
        [np.mean(clean_after[targets == index]) for index in range(int(class_count))]
    )
    initial_objective = float(inherited["initial_objective"])
    d87_unshrunk = {
        key: inherited[key]
        for key in (
            "final_objective",
            "objective_delta",
            "final_worst_class_sigma_ce",
            "final_mean_sigma_ce",
            "oof_clean_ce_after_mean",
            "oof_clean_ce_delta_max_class",
            "oof_clean_correct_after",
            "residual_sha256",
            "bias_residual_sha256",
        )
    }
    base_prediction = np.argmax(base_scores, axis=1)
    final_prediction = np.argmax(updated_clean_scores, axis=1)
    audit.update(
        {
            "status": (
                "crossfit_consensus_sigma_margin_active"
                if consensus > 0.0 and np.linalg.norm(scaled_w64) > tolerance
                else "zero_consensus_exact_d62_fallback"
            ),
            "consensus_formula": "clip(mean_off_diagonal_cosine(unit_initial_fold_sigma_gradients),0,1)",
            "consensus_factor": consensus,
            "fold_gradient_count": fold_count,
            "fold_gradient_norm_min": float(np.min(norms)),
            "fold_gradient_norm_mean": float(np.mean(norms)),
            "fold_gradient_norm_max": float(np.max(norms)),
            "fold_gradient_cosine_min": float(np.min(gram[np.triu_indices(fold_count, 1)])),
            "fold_gradient_cosine_mean": off_diagonal,
            "fold_gradient_cosine_max": float(np.max(gram[np.triu_indices(fold_count, 1)])),
            "mean_unit_gradient_norm": mean_direction_norm,
            "d87_unshrunk_residual_frobenius": float(np.linalg.norm(full_w)),
            "d87_unshrunk_audit": d87_unshrunk,
            "final_objective": float(final[0]),
            "objective_delta": float(final[0] - initial_objective),
            "final_worst_class_sigma_ce": float(np.max(final[1])),
            "final_mean_sigma_ce": float(np.mean(final[3])),
            "oof_clean_ce_after_mean": float(np.mean(clean_after_class)),
            "oof_clean_ce_delta_max_class": float(
                np.max(clean_after_class - clean_before_class)
            ),
            "oof_ce_after_mean": float(np.mean(clean_after_class)),
            "oof_ce_delta_mean": float(
                np.mean(clean_after_class - clean_before_class)
            ),
            "oof_ce_delta_max_class": float(
                np.max(clean_after_class - clean_before_class)
            ),
            "oof_clean_correct_after": int(np.sum(final_prediction == targets)),
            "oof_updated_correct_count": int(np.sum(final_prediction == targets)),
            "support_prediction_change_count": int(
                np.sum(final_prediction != base_prediction)
            ),
            "residual_frobenius": float(np.linalg.norm(scaled_w64)),
            "bias_residual_frobenius": float(np.linalg.norm(scaled_b64)),
            "residual_sha256": d87._sha256(scaled_w),
            "bias_residual_sha256": d87._sha256(scaled_b),
            "residual_active": bool(consensus > 0.0 and np.linalg.norm(scaled_w64) > tolerance),
            "residual_logit_at_support_center_max_abs": center_error,
            "query_rows_used": 0,
            "class_permutation_equivariant": True,
            "old_new_role_specific_branch": False,
            "class_id_specific_formula": False,
            "ground_class_score_access": False,
            "physical_group_crossfit_preserved": True,
        }
    )
    return scaled_w, scaled_b, audit


# The D87 probe scaffold calls this symbol after loading a replacement core.
fit_ground_radius_sigma_margin = fit_crossfit_consensus_sigma_margin


__all__ = [
    "D91CrossfitConsensusError",
    "fit_crossfit_consensus_sigma_margin",
    "fit_ground_radius_sigma_margin",
    "ground_radius_sigma_geometry",
]
