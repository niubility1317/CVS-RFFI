"""Radius-bounded ground counterfactual support centers for D86.

The immutable ground component contributes only class-agnostic domain directions
and aggregated p90 radii.  Every target class uses the same support-only rule:
downweight observations whose nearest-rival margin is unstable under any sealed
symmetric ground-domain perturbation, then translate the class-common z160 center.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


_D84_PATH = Path(__file__).with_name(
    "stage2_d84_ground_crossclass_consensus_center.py"
)
_SPEC = importlib.util.spec_from_file_location("d86_d84_center_core", _D84_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("D86 could not load the D84 consensus primitive")
d84 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = d84
_SPEC.loader.exec_module(d84)


Z_DIM = 160
RISK_EPSILON = 1.0e-24


class D86GroundCounterfactualError(RuntimeError):
    """Raised when D86 ground geometry or support closure drifts."""


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def ground_radius_counterfactual_templates(
    domain_class_prototypes: np.ndarray,
    domain_class_radius: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return one class-agnostic direction and p90 displacement per domain."""

    prototypes = np.asarray(domain_class_prototypes, dtype=np.float64)
    radius = np.asarray(domain_class_radius, dtype=np.float64)
    if (
        prototypes.ndim != 3
        or prototypes.shape[2] != Z_DIM
        or radius.shape != prototypes.shape[:2]
        or prototypes.shape[0] < 2
        or prototypes.shape[1] < 2
        or not np.isfinite(prototypes).all()
        or not np.isfinite(radius).all()
        or np.any(radius <= 0.0)
    ):
        raise D86GroundCounterfactualError(
            "D86 requires finite positive v2 ground prototype radii"
        )
    mask = np.ones(prototypes.shape[:2], dtype=np.uint8)
    templates, geometry_weight, inherited = (
        d84.ground_crossclass_consensus_templates(prototypes, mask)
    )
    domains = int(prototypes.shape[0])
    if templates.shape != (Z_DIM, domains) or geometry_weight.shape != (domains,):
        raise D86GroundCounterfactualError(
            "D86 requires one retained direction per v2 ground domain"
        )
    domain_radius = np.median(radius, axis=1)
    amplitude = np.sqrt(2.0 * domain_radius)
    offsets = templates.T * amplitude[:, None]
    if (
        not np.isfinite(amplitude).all()
        or np.any(amplitude <= 0.0)
        or not np.isfinite(offsets).all()
        or not np.allclose(
            np.linalg.norm(offsets, axis=1), amplitude, rtol=1.0e-12, atol=1.0e-14
        )
    ):
        raise D86GroundCounterfactualError("D86 counterfactual amplitude drift")
    amplitude = np.ascontiguousarray(amplitude, dtype=np.float64)
    amplitude.setflags(write=False)
    audit = dict(inherited)
    audit.update(
        {
            "schema": "cvs.phase2.d86.ground_radius_counterfactual_templates.v1",
            "template_policy": (
                "class_centered_crossclass_consensus_direction_with_symmetric_v2_p90_radius"
            ),
            "weight_policy": "sqrt_two_times_domain_median_p90_radius_amplitude",
            "ground_radius_definition": (
                "p90_cosine_distance_to_phase1_domain_class_centroid"
            ),
            "ground_radius_overall_min": float(np.min(radius)),
            "ground_radius_overall_median": float(np.median(radius)),
            "ground_radius_overall_mean": float(np.mean(radius)),
            "ground_radius_overall_max": float(np.max(radius)),
            "domain_median_radius_min": float(np.min(domain_radius)),
            "domain_median_radius_reference": float(np.median(domain_radius)),
            "domain_median_radius_max": float(np.max(domain_radius)),
            "counterfactual_amplitude_min": float(np.min(amplitude)),
            "counterfactual_amplitude_mean": float(np.mean(amplitude)),
            "counterfactual_amplitude_max": float(np.max(amplitude)),
            "geometry_weight_min": float(np.min(geometry_weight)),
            "geometry_weight_max": float(np.max(geometry_weight)),
            # Compatibility names consumed by the D84 integration scaffold.
            "spectral_weight_min": float(np.min(amplitude)),
            "spectral_weight_max": float(np.max(amplitude)),
            "weight_sha256": _sha256_array(amplitude),
            "radius_scan_count": 0,
            "radius_hyperparameter_count": 0,
            "counterfactual_sign_count": 2,
            "physical_sample_count_multiplier": 1,
            "ground_class_centers_discarded": True,
            "ground_class_score_access": False,
            "ground_target_identity_mapping_access": False,
            "old_new_role_specific_branch": False,
        }
    )
    return templates, amplitude, audit


def translate_to_counterfactual_robust_centers(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    templates: np.ndarray,
    counterfactual_amplitudes: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Translate class centers using symmetric ground-counterfactual risk."""

    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    u = np.asarray(templates, dtype=np.float64)
    amplitude = np.asarray(counterfactual_amplitudes, dtype=np.float64)
    classes, shots = int(class_count), int(k_shot)
    if (
        x.ndim != 2
        or x.shape[1] < Z_DIM
        or y.shape != (len(x),)
        or len(x) != classes * shots
        or u.ndim != 2
        or u.shape[0] != Z_DIM
        or amplitude.shape != (u.shape[1],)
        or not np.isfinite(x).all()
        or not np.isfinite(u).all()
        or not np.isfinite(amplitude).all()
        or np.any(amplitude <= 0.0)
        or classes <= 1
        or shots <= 0
        or any(int(np.sum(y == index)) != shots for index in range(classes))
        or not np.allclose(
            np.linalg.norm(u, axis=0), 1.0, rtol=1.0e-12, atol=1.0e-12
        )
    ):
        raise D86GroundCounterfactualError(
            "D86 requires finite symmetric support and unit ground directions"
        )
    transformed = x.copy()
    means = np.stack([np.mean(x[y == index, :Z_DIM], axis=0) for index in range(classes)])
    offsets = u.T * amplitude[:, None]
    mean_offset_projection = means @ offsets.T
    shifts: list[np.ndarray] = []
    risks_by_class: list[list[float]] = []
    normalized_weights_by_class: list[list[float]] = []
    effective_samples: list[float] = []
    worst_margins: list[float] = []
    identity = shots <= 2
    for class_index in range(classes):
        indices = np.flatnonzero(y == class_index)
        z = x[indices, :Z_DIM]
        mean = means[class_index]
        if identity:
            risk = np.zeros(shots, dtype=np.float64)
            weight = np.full(shots, 1.0 / shots, dtype=np.float64)
            shift = np.zeros(Z_DIM, dtype=np.float64)
            worst = np.full(shots, np.inf, dtype=np.float64)
        else:
            delta = z[:, None, :] - means[None, :, :]
            distance = np.sum(np.square(delta), axis=2)
            own = distance[:, class_index]
            base_margin = distance - own[:, None]
            sensitivity = 2.0 * np.abs(
                mean_offset_projection[class_index][None, :]
                - mean_offset_projection
            )
            robust_margin = base_margin - np.max(sensitivity, axis=1)[None, :]
            robust_margin[:, class_index] = np.inf
            worst = np.min(robust_margin, axis=1)
            risk = np.logaddexp(0.0, -worst)
            scale = float(np.mean(risk))
            if not np.isfinite(scale) or scale <= RISK_EPSILON:
                raw_weight = np.ones(shots, dtype=np.float64)
            else:
                raw_weight = 1.0 / (1.0 + risk / scale)
            weight = raw_weight / np.sum(raw_weight)
            robust_mean = np.sum(weight[:, None] * z, axis=0)
            shift = robust_mean - mean
            transformed[indices, :Z_DIM] += shift[None, :]
        shifts.append(shift)
        risks_by_class.append(risk.tolist())
        normalized_weights_by_class.append(weight.tolist())
        effective_samples.append(float(1.0 / np.sum(np.square(weight))))
        worst_margins.extend(worst[np.isfinite(worst)].tolist())
    before_means = np.stack(
        [np.mean(x[y == index], axis=0) for index in range(classes)]
    )
    after_means = np.stack(
        [np.mean(transformed[y == index], axis=0) for index in range(classes)]
    )
    before_residual = x - before_means[y]
    after_residual = transformed - after_means[y]
    residual_error = float(np.max(np.abs(before_residual - after_residual)))
    fft_rf_error = float(np.max(np.abs(transformed[:, Z_DIM:] - x[:, Z_DIM:])))
    if (
        not np.isfinite(transformed).all()
        or residual_error > 2.0e-12
        or fft_rf_error != 0.0
    ):
        raise D86GroundCounterfactualError(
            "D86 class-common translation invariant drift"
        )
    shift_array = np.stack(shifts)
    all_weights = np.asarray(normalized_weights_by_class, dtype=np.float64)
    finite_worst = np.asarray(worst_margins, dtype=np.float64)
    audit = {
        "schema": "cvs.phase2.d86.support_counterfactual_center_translation.v1",
        "support_rows": int(len(x)),
        "class_count": classes,
        "k_shot": shots,
        "retained_rank": int(u.shape[1]),
        "center_formula": (
            "one_step_classwise_cauchy_symmetric_ground_counterfactual_margin_risk"
        ),
        "energy_scale_policy": "per_target_class_mean_counterfactual_logistic_risk",
        "translation_scope": "z160_class_common_only",
        "k1_k2_exact_identity": identity,
        "class_id_specific_formula": False,
        "old_new_role_specific_branch": False,
        "scene_receiver_handle_specific_branch": False,
        "uses_outer_held_or_query": False,
        "query_rows_used": 0,
        "hyperparameter_count": 0,
        "weight_scan_count": 0,
        "counterfactual_sign_count": 2,
        "counterfactual_domain_count": int(u.shape[1]),
        "counterfactual_views_count_as_physical_samples": False,
        "counterfactual_amplitude_min": float(np.min(amplitude)),
        "counterfactual_amplitude_max": float(np.max(amplitude)),
        "worst_counterfactual_margin_min": (
            float(np.min(finite_worst)) if len(finite_worst) else None
        ),
        "worst_counterfactual_margin_mean": (
            float(np.mean(finite_worst)) if len(finite_worst) else None
        ),
        "within_class_residual_max_abs_error": residual_error,
        "fft96_rf32_max_abs_error": fft_rf_error,
        "center_shift_l2_by_class": np.linalg.norm(shift_array, axis=1).tolist(),
        "center_shift_l2_max": float(np.max(np.linalg.norm(shift_array, axis=1))),
        "normalized_weight_min": float(np.min(all_weights)),
        "normalized_weight_max": float(np.max(all_weights)),
        "effective_sample_size_by_class": effective_samples,
        "counterfactual_risk_by_class": risks_by_class,
        "normalized_cauchy_weight_by_class": normalized_weights_by_class,
    }
    return transformed, audit


def build_counterfactual_center_component_fit(
    component_fit: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    templates: np.ndarray,
    counterfactual_amplitudes: np.ndarray,
    template_audit: dict[str, Any],
    component_arm: str,
    collector: list[dict[str, Any]],
) -> Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Wrap every D62 full/block scope with D86 counterfactual weighting."""

    def fit(
        rows: np.ndarray,
        labels: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        transformed, transform_audit = translate_to_counterfactual_robust_centers(
            rows,
            labels,
            class_count,
            k_shot,
            templates,
            counterfactual_amplitudes,
        )
        coefficient, intercept, base_audit = component_fit(
            transformed, labels, class_count, k_shot
        )
        collector.append(
            {
                "component_arm": component_arm,
                "class_count": int(class_count),
                "k_shot": int(k_shot),
                "center_shift_l2_max": transform_audit["center_shift_l2_max"],
                "normalized_weight_min": transform_audit["normalized_weight_min"],
            }
        )
        audit = dict(base_audit)
        audit.update(
            {
                "d84_component_arm": component_arm,
                "d84_ground_template_sha256": template_audit["template_sha256"],
                "d84_ground_weight_sha256": template_audit["weight_sha256"],
                "d84_retained_domain_template_count": template_audit[
                    "retained_domain_template_count"
                ],
                "d84_template_policy": template_audit["template_policy"],
                "d84_transform_audit": transform_audit,
            }
        )
        return coefficient, intercept, audit

    return fit


__all__ = [
    "D86GroundCounterfactualError",
    "Z_DIM",
    "build_counterfactual_center_component_fit",
    "ground_radius_counterfactual_templates",
    "translate_to_counterfactual_robust_centers",
]
