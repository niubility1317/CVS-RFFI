"""D89 v2 radius-reliability ground spectrum for D81 Cauchy centers."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np


_D81_PATH = Path(__file__).with_name("stage2_d81_ground_nuisance_cauchy_center.py")
_SPEC = importlib.util.spec_from_file_location("d89_d81_center", _D81_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("D89 could not load the locked D81 center core")
d81 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = d81
_SPEC.loader.exec_module(d81)

Z_DIM = d81.Z_DIM
build_robust_center_component_fit = d81.build_robust_center_component_fit
translate_to_robust_centers = d81.translate_to_robust_centers


class D89V2RadiusCauchyError(RuntimeError):
    """Raised when the D89 v2 radius-weighted spectrum drifts."""


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def radius_reliability_ground_spectrum(
    domain_class_prototypes: np.ndarray,
    domain_class_radius: np.ndarray,
    reconstruction_rmse: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a class-agnostic nuisance spectrum from all reliable v2 cells."""

    prototypes = np.asarray(domain_class_prototypes, dtype=np.float64)
    radius = np.asarray(domain_class_radius, dtype=np.float64)
    rmse = float(reconstruction_rmse)
    if (
        prototypes.ndim != 3
        or prototypes.shape[2] != Z_DIM
        or radius.shape != prototypes.shape[:2]
        or prototypes.shape[0] < 2
        or prototypes.shape[1] < 2
        or not np.isfinite(prototypes).all()
        or not np.isfinite(radius).all()
        or not np.isfinite(rmse)
        or np.any(radius <= 0.0)
        or rmse < 0.0
    ):
        raise D89V2RadiusCauchyError("D89 requires finite v2 cells and radii")
    unweighted_center = np.mean(prototypes, axis=0)
    unweighted_residual = prototypes - unweighted_center[None, :, :]
    cross_domain_signal = np.sum(np.square(unweighted_residual), axis=2)
    chord_variance = 2.0 * radius
    reliability = cross_domain_signal / (cross_domain_signal + chord_variance)
    class_denominator = np.sum(reliability, axis=0)
    if np.any(class_denominator <= 0.0):
        raise D89V2RadiusCauchyError("D89 class reliability denominator vanished")
    cell_weight = reliability / class_denominator[None, :]
    class_center = np.einsum("dc,dcz->cz", cell_weight, prototypes)
    residual = prototypes - class_center[None, :, :]
    covariance = np.einsum(
        "dc,dcz,dcw->zw", cell_weight, residual, residual
    ) / float(prototypes.shape[1])
    covariance = 0.5 * (covariance + covariance.T)
    center_error = float(
        np.max(np.abs(np.einsum("dc,dcz->cz", cell_weight, residual)))
    )
    domain_weight_sum_error = float(
        np.max(np.abs(np.sum(cell_weight, axis=0) - 1.0))
    )
    if (
        not np.isfinite(covariance).all()
        or center_error > 1.0e-12
        or domain_weight_sum_error > 1.0e-14
        or float(np.trace(covariance)) <= 0.0
    ):
        raise D89V2RadiusCauchyError("D89 weighted covariance invariant drift")
    noise_floor = rmse * rmse
    basis, weights, inherited = d81.ground_nuisance_basis(
        covariance, noise_floor
    )
    audit = dict(inherited)
    audit.update({
        "schema": "cvs.phase2.d89.v2_radius_reliability_spectrum.v1",
        "spectrum_policy": (
            "cell_radius_reliability_weighted_ground_class_centered_domain_drift"
        ),
        "radius_definition": "phase1_aggregated_domain_class_p90_cosine_distance",
        "radius_to_chord_variance_formula": "v_dc=2*r_dc",
        "cross_domain_signal_formula": "s_dc=||g_dc-mean_d(g_dc)||_2^2",
        "reliability_formula": "rho_dc=s_dc/(s_dc+2*r_dc)",
        "cell_weight_formula": "q_dc=rho_dc/sum_d(rho_dc)",
        "equal_ground_class_contribution": True,
        "ground_domain_count": int(prototypes.shape[0]),
        "ground_class_count": int(prototypes.shape[1]),
        "ground_component_input_count": int(radius.size),
        "radius_min": float(np.min(radius)),
        "radius_median": float(np.median(radius)),
        "radius_mean": float(np.mean(radius)),
        "radius_max": float(np.max(radius)),
        "reliability_min": float(np.min(reliability)),
        "reliability_mean": float(np.mean(reliability)),
        "reliability_max": float(np.max(reliability)),
        "cross_domain_signal_min": float(np.min(cross_domain_signal)),
        "cross_domain_signal_mean": float(np.mean(cross_domain_signal)),
        "cross_domain_signal_max": float(np.max(cross_domain_signal)),
        "class_reliability_sum_min": float(np.min(class_denominator)),
        "class_reliability_sum_max": float(np.max(class_denominator)),
        "cell_weight_min": float(np.min(cell_weight)),
        "cell_weight_max": float(np.max(cell_weight)),
        "domain_weight_sum_max_abs_error": domain_weight_sum_error,
        "weighted_center_max_abs_error": center_error,
        "weighted_covariance_trace": float(np.trace(covariance)),
        "weighted_covariance_sha256": _sha256(covariance),
        "reliability_sha256": _sha256(reliability),
        "cell_weight_sha256": _sha256(cell_weight),
        "reconstruction_rmse": rmse,
        "quantization_noise_floor_policy": "manifest_reconstruction_rmse_squared",
        "radius_hyperparameter_count": 0,
        "radius_scan_count": 0,
        "ground_class_centers_discarded": True,
        "ground_aggregated_center_access": True,
        "ground_aggregated_p90_radius_access": True,
        "ground_sample_radius_access": False,
        "ground_sample_feature_access": False,
        "ground_target_identity_mapping_access": False,
        "old_new_role_specific_branch": False,
    })
    return basis, weights, audit


__all__ = [
    "D89V2RadiusCauchyError",
    "Z_DIM",
    "build_robust_center_component_fit",
    "radius_reliability_ground_spectrum",
    "translate_to_robust_centers",
]
