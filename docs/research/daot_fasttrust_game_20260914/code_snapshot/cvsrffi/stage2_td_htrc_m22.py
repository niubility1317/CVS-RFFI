"""TD-HTRC M2.2: regularized transport, posterior centres and adaptive spectrum.

M2.2 is an explicit opt-in extension of :mod:`stage2_td_htrc_target_transport`.
It keeps the identifiable shared offset, then adds only low-dimensional
structure that six old-class anchors can support:

* a regularized low-rank correction in the frozen ground nuisance basis;
* optional three-block scales when complete 288-D ground centres are supplied;
* a diagonal Bayesian centre posterior (old classes use ground priors, new
  classes use target-support likelihood only);
* a ground/target residual covariance shrinkage used to rebuild the Cauchy
  nuisance spectrum.

The posterior uncertainty is passed to the D92 covariance component as a
diagonal addition.  No query row, query label, source sample, class quota or
global reassignment is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from cvsrffi.stage2_d81_ground_nuisance_cauchy_center import (
    translate_to_robust_centers,
)
from cvsrffi.stage2_td_htrc_target_transport import (
    FEATURE_DIM,
    OLD_CLASS_COUNT,
    OFFSET_COVARIANCE_FLOOR,
    TDHTRCError,
    _validate_basis,
    _validate_ground_centers,
    _validate_support,
    estimate_shared_target_transport,
)


M22_RIDGE = 1.0e-2
M22_MAX_LOW_RANK_FROBENIUS = 0.5
M22_BLOCK_SCALE_MIN = 0.75
M22_BLOCK_SCALE_MAX = 1.25
M22_PRIOR_MULTIPLIER = 4.0
M22_SPECTRUM_KAPPA = float(OLD_CLASS_COUNT)
M22_SPECTRUM_FLOOR = 1.0e-10


class TDHTRCM22Error(TDHTRCError):
    """Raised when the support-only M2.2 closure is invalid."""


def _readonly(value: np.ndarray, dtype: np.dtype[Any] | type) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(values, dtype=np.float64) * weights[:, None], axis=0)


def _validate_full_ground_centers(
    values: np.ndarray | None, old_class_count: int
) -> tuple[np.ndarray | None, bool]:
    if values is None:
        return None, False
    anchors = np.asarray(values, dtype=np.float64)
    if (
        anchors.shape != (old_class_count, FEATURE_DIM)
        or not np.isfinite(anchors).all()
    ):
        raise TDHTRCM22Error(
            "M2.2 complete ground centres must be [old_class_count,288] finite"
        )
    return anchors, True


@dataclass(frozen=True)
class TDHTRCM22Estimate:
    """Immutable registration-scoped M2.2 transport and centre posterior."""

    shared_offset: np.ndarray
    transport_matrix: np.ndarray
    inverse_transport_matrix: np.ndarray
    adaptive_basis: np.ndarray
    adaptive_weights: np.ndarray
    posterior_centers: np.ndarray
    posterior_variance: np.ndarray
    robust_centers: np.ndarray
    audit: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shared_offset", _readonly(self.shared_offset, np.float64))
        object.__setattr__(
            self, "transport_matrix", _readonly(self.transport_matrix, np.float64)
        )
        object.__setattr__(
            self,
            "inverse_transport_matrix",
            _readonly(self.inverse_transport_matrix, np.float64),
        )
        object.__setattr__(self, "adaptive_basis", _readonly(self.adaptive_basis, np.float64))
        object.__setattr__(self, "adaptive_weights", _readonly(self.adaptive_weights, np.float64))
        object.__setattr__(
            self, "posterior_centers", _readonly(self.posterior_centers, np.float64)
        )
        object.__setattr__(
            self, "posterior_variance", _readonly(self.posterior_variance, np.float64)
        )
        object.__setattr__(self, "robust_centers", _readonly(self.robust_centers, np.float64))
        object.__setattr__(self, "audit", dict(self.audit))


def _fit_low_rank_transport(
    ground: np.ndarray,
    target: np.ndarray,
    offset: np.ndarray,
    basis: np.ndarray,
    weights: np.ndarray,
    *,
    complete_ground: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit block scales and a ridge-regularized low-rank identity correction."""

    old_count = int(len(ground))
    final_target = target - offset[None, :]
    block_slices = (slice(0, 160), slice(160, 256), slice(256, 288))
    block_scales = np.ones(3, dtype=np.float64)
    for index, block in enumerate(block_slices):
        if index > 0 and not complete_ground:
            continue
        ground_centered = ground[:, block] - _weighted_mean(ground[:, block], weights)
        target_centered = final_target[:, block] - _weighted_mean(
            final_target[:, block], weights
        )
        denominator = float(
            np.sum(weights * np.sum(np.square(ground_centered), axis=1))
        )
        numerator = float(
            np.sum(weights * np.sum(np.square(target_centered), axis=1))
        )
        if denominator > OFFSET_COVARIANCE_FLOOR:
            block_scales[index] = float(
                np.clip(
                    np.sqrt(max(numerator, 0.0) / denominator),
                    M22_BLOCK_SCALE_MIN,
                    M22_BLOCK_SCALE_MAX,
                )
            )

    p = ground[:, :160]
    y = final_target[:, :160]
    u = basis
    p_proj = p @ u
    delta_proj = y @ u - block_scales[0] * p_proj
    gram = p_proj.T @ (weights[:, None] * p_proj)
    rhs = p_proj.T @ (weights[:, None] * delta_proj)
    low_rank_transpose = np.linalg.solve(
        gram + M22_RIDGE * np.eye(u.shape[1]), rhs
    )
    r_matrix = low_rank_transpose.T
    frobenius = float(np.linalg.norm(r_matrix, ord="fro"))
    if frobenius > M22_MAX_LOW_RANK_FROBENIUS:
        r_matrix *= M22_MAX_LOW_RANK_FROBENIUS / frobenius

    matrix = np.zeros((FEATURE_DIM, FEATURE_DIM), dtype=np.float64)
    matrix[:160, :160] = block_scales[0] * np.eye(160) + u @ r_matrix @ u.T
    matrix[160:256, 160:256] = block_scales[1] * np.eye(96)
    matrix[256:, 256:] = block_scales[2] * np.eye(32)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if not np.isfinite(eigenvalues).all() or float(np.min(eigenvalues)) <= 0.0:
        raise TDHTRCM22Error("M2.2 transport matrix is not positive definite")
    inverse = np.linalg.inv(matrix)
    audit = {
        "transport_matrix_min_eigenvalue": float(np.min(eigenvalues)),
        "transport_matrix_max_eigenvalue": float(np.max(eigenvalues)),
        "transport_block_scales": block_scales.tolist(),
        "transport_block_scale_estimated": [
            True,
            bool(complete_ground),
            bool(complete_ground),
        ],
        "transport_low_rank_basis_rank": int(u.shape[1]),
        "transport_low_rank_ridge": M22_RIDGE,
        "transport_low_rank_frobenius": float(np.linalg.norm(r_matrix, ord="fro")),
        "transport_low_rank_matrix": r_matrix.tolist(),
        "transport_complete_ground_centers_available": bool(complete_ground),
    }
    return matrix, inverse, block_scales, audit


def _adaptive_spectrum(
    ground_identity: np.ndarray,
    canonical_target_identity: np.ndarray,
    anchor_weights: np.ndarray,
    basis: np.ndarray,
    spectral_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    residual = canonical_target_identity - ground_identity
    residual_covariance = (residual * anchor_weights[:, None]).T @ residual
    residual_covariance /= max(
        1.0 - float(np.sum(np.square(anchor_weights))), 1.0e-6
    )
    ground_covariance = basis @ np.diag(spectral_weights) @ basis.T
    ground_trace = float(np.trace(ground_covariance))
    residual_trace = float(np.trace(residual_covariance))
    if residual_trace > OFFSET_COVARIANCE_FLOOR and ground_trace > 0.0:
        ground_covariance *= residual_trace / ground_trace
    effective_anchor_count = 1.0 / float(np.sum(np.square(anchor_weights)))
    beta = effective_anchor_count / (effective_anchor_count + M22_SPECTRUM_KAPPA)
    posterior = (
        (1.0 - beta) * ground_covariance
        + beta * residual_covariance
        + M22_SPECTRUM_FLOOR * np.eye(160)
    )
    posterior = 0.5 * (posterior + posterior.T)
    eigenvalues, eigenvectors = np.linalg.eigh(posterior)
    positive = eigenvalues > 0.0
    values = eigenvalues[positive]
    vectors = eigenvectors[:, positive]
    if len(values) == 0:
        raise TDHTRCM22Error("M2.2 adaptive spectrum has no positive direction")
    effective_rank = float(np.square(np.sum(values)) / np.sum(np.square(values)))
    rank = min(int(np.ceil(effective_rank)), len(values))
    order = np.argsort(values, kind="stable")[-rank:][::-1]
    selected_values = values[order]
    selected_basis = vectors[:, order]
    selected_weights = selected_values / np.sum(selected_values)
    return selected_basis, selected_weights, {
        "adaptive_spectrum_beta": float(beta),
        "adaptive_spectrum_kappa": M22_SPECTRUM_KAPPA,
        "adaptive_spectrum_residual_trace": residual_trace,
        "adaptive_spectrum_ground_trace_after_match": float(
            np.trace(ground_covariance)
        ),
        "adaptive_spectrum_effective_rank": effective_rank,
        "adaptive_spectrum_retained_rank": int(rank),
        "adaptive_spectrum_weights": selected_weights.tolist(),
        "adaptive_spectrum_floor": M22_SPECTRUM_FLOOR,
    }


def _posterior_centres(
    rows: np.ndarray,
    labels: np.ndarray,
    robust_centers: np.ndarray,
    effective_samples: np.ndarray,
    ground_centers: np.ndarray,
    old_class_count: int,
    complete_ground: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    class_count = int(len(robust_centers))
    class_variance = np.stack(
        [
            np.mean(
                np.square(rows[labels == index] - robust_centers[index][None, :]),
                axis=0,
            )
            for index in range(class_count)
        ]
    )
    likelihood_variance = np.maximum(
        class_variance / np.maximum(effective_samples[:, None], 1.0),
        OFFSET_COVARIANCE_FLOOR,
    )
    prior_scale = np.maximum(
        np.median(likelihood_variance[:old_class_count], axis=0)
        * M22_PRIOR_MULTIPLIER,
        OFFSET_COVARIANCE_FLOOR,
    )
    posterior_centers = robust_centers.copy()
    posterior_variance = likelihood_variance.copy()
    prior_enabled = np.zeros((class_count, FEATURE_DIM), dtype=bool)
    prior_enabled[:old_class_count, :160] = True
    if complete_ground:
        prior_enabled[:old_class_count, 160:] = True
    for index in range(old_class_count):
        mask = prior_enabled[index]
        prior_variance = prior_scale[mask]
        likelihood = np.maximum(likelihood_variance[index, mask], OFFSET_COVARIANCE_FLOOR)
        posterior = 1.0 / (1.0 / prior_variance + 1.0 / likelihood)
        posterior_centers[index, mask] = posterior * (
            ground_centers[index, mask] / prior_variance
            + robust_centers[index, mask] / likelihood
        )
        posterior_variance[index, mask] = posterior
    return posterior_centers, posterior_variance, {
        "posterior_prior_enabled_by_class": np.sum(prior_enabled, axis=1).tolist(),
        "posterior_prior_source": "ground_old_class_centres_with_fixed_4x_likelihood_scale",
        "posterior_prior_multiplier": M22_PRIOR_MULTIPLIER,
        "posterior_is_diagonal": True,
        "posterior_center_shift_l2_by_class": np.linalg.norm(
            posterior_centers - robust_centers, axis=1
        ).tolist(),
        "posterior_variance_trace_by_class": np.sum(posterior_variance, axis=1).tolist(),
    }


def estimate_m22_transport(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    ground_class_centers: np.ndarray,
    basis: np.ndarray,
    spectral_weights: np.ndarray,
    *,
    old_class_count: int = OLD_CLASS_COUNT,
    ground_full_centers: np.ndarray | None = None,
) -> tuple[TDHTRCM22Estimate, np.ndarray, dict[str, Any]]:
    """Estimate M2.2 state and return final canonical support rows."""

    x, y = _validate_support(
        rows, labels, int(class_count), int(k_shot), int(old_class_count)
    )
    anchors160 = _validate_ground_centers(ground_class_centers, int(old_class_count))
    u, rho = _validate_basis(basis, spectral_weights)
    full_ground, complete_ground = _validate_full_ground_centers(
        ground_full_centers, int(old_class_count)
    )
    if complete_ground and not np.allclose(
        full_ground[:, :160], anchors160, rtol=1.0e-5, atol=1.0e-6
    ):
        raise TDHTRCM22Error(
            "M2.2 complete ground centres disagree with the 160-D anchor bundle"
        )
    m21 = estimate_shared_target_transport(
        x,
        y,
        int(class_count),
        int(k_shot),
        anchors160,
        u,
        rho,
        old_class_count=int(old_class_count),
    )
    preliminary, preliminary_center_audit = translate_to_robust_centers(
        x,
        y,
        int(class_count),
        int(k_shot),
        u,
        rho,
    )
    target_full = np.stack(
        [preliminary[y == index].mean(axis=0) for index in range(int(class_count))]
    )
    target_old = target_full[: int(old_class_count)]
    if full_ground is None:
        full_ground = np.zeros((int(old_class_count), FEATURE_DIM), dtype=np.float64)
        full_ground[:, :160] = anchors160
        full_ground[:, 160:] = target_old[:, 160:]
    offset = np.zeros(FEATURE_DIM, dtype=np.float64)
    offset[:160] = m21.shared_offset
    if complete_ground:
        offset[160:] = _weighted_mean(
            target_old[:, 160:] - full_ground[:, 160:], m21.anchor_weights
        )
    matrix, inverse, block_scales, transport_audit = _fit_low_rank_transport(
        full_ground,
        target_old,
        offset,
        u,
        m21.anchor_weights,
        complete_ground=complete_ground,
    )
    canonical = (x - offset[None, :]) @ inverse.T
    canonical_old = (target_old - offset[None, :]) @ inverse.T
    canonical_ground = (full_ground - offset[None, :]) @ inverse.T
    adaptive_basis, adaptive_weights, adaptive_audit = _adaptive_spectrum(
        canonical_ground[:, :160],
        canonical_old[:, :160],
        m21.anchor_weights,
        u,
        rho,
    )
    robust_canonical, robust_audit = translate_to_robust_centers(
        canonical,
        y,
        int(class_count),
        int(k_shot),
        adaptive_basis,
        adaptive_weights,
    )
    robust_centers = np.stack(
        [robust_canonical[y == index].mean(axis=0) for index in range(int(class_count))]
    )
    effective_samples = np.asarray(
        robust_audit["effective_sample_size_by_class"], dtype=np.float64
    )
    posterior_centers, posterior_variance, posterior_audit = _posterior_centres(
        robust_canonical,
        y,
        robust_centers,
        effective_samples,
        canonical_ground,
        int(old_class_count),
        complete_ground,
    )
    final_support = robust_canonical + (
        posterior_centers[y] - robust_centers[y]
    )
    if not np.isfinite(final_support).all():
        raise TDHTRCM22Error("M2.2 final canonical support became non-finite")
    anchor_residual = canonical_old[: int(old_class_count)] - canonical_ground
    residual_energy = float(
        np.sum(m21.anchor_weights * np.sum(np.square(anchor_residual), axis=1))
    )
    estimated_macs = int(
        m21.audit["estimated_registration_macs"]
        + int(class_count) * int(k_shot) * (2 * FEATURE_DIM)
        + int(old_class_count) * FEATURE_DIM * max(1, u.shape[1]) * 4
        + 3 * int(old_class_count) * FEATURE_DIM
        + int(class_count) * FEATURE_DIM * 8
    )
    audit: dict[str, Any] = {
        "schema": "cvs.phase2.td_htrc.m22.transport.v1",
        "method": "TD-HTRC-M2.2",
        "support_rows_used": int(len(x)),
        "query_rows_used": 0,
        "query_truth_used": False,
        "source_sample_used": False,
        "old_class_count": int(old_class_count),
        "registered_class_count": int(class_count),
        "k_shot": int(k_shot),
        "shared_offset_160_norm": float(np.linalg.norm(m21.shared_offset)),
        "shared_offset_288_norm": float(np.linalg.norm(offset)),
        "transport_matrix_frobenius": float(np.linalg.norm(matrix, ord="fro")),
        "transport_inverse_frobenius": float(np.linalg.norm(inverse, ord="fro")),
        "transport_block_scales": block_scales.tolist(),
        "transport_low_rank_enabled": True,
        "transport_complete_ground_centres_available": bool(complete_ground),
        "adaptive_spectrum_enabled": True,
        "adaptive_spectrum_residual_energy": residual_energy,
        "posterior_center_enabled": True,
        "posterior_uncertainty_enabled": True,
        "posterior_uncertainty_representation": "classwise_diagonal_288",
        "posterior_uncertainty_trace": float(np.sum(posterior_variance)),
        "estimated_registration_macs": estimated_macs,
        "transport_estimate_transient_state_bytes": int(
            matrix.nbytes + inverse.nbytes + posterior_variance.nbytes
        ),
        "persistent_transport_state_bytes": 0,
        "query_transport_mode": "compiled_affine_matrix_and_intercept",
        "m21_audit": m21.audit,
        "transport_audit": transport_audit,
        "adaptive_spectrum_audit": adaptive_audit,
        "posterior_audit": posterior_audit,
        "preliminary_center_audit": preliminary_center_audit,
        "final_robust_center_audit": robust_audit,
    }
    return (
        TDHTRCM22Estimate(
            shared_offset=offset,
            transport_matrix=matrix,
            inverse_transport_matrix=inverse,
            adaptive_basis=adaptive_basis,
            adaptive_weights=adaptive_weights,
            posterior_centers=posterior_centers,
            posterior_variance=posterior_variance,
            robust_centers=robust_centers,
            audit=audit,
        ),
        final_support,
        audit,
    )


def build_td_htrc_m22_component_fit(
    component_fit: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    *,
    ground_class_centers: np.ndarray,
    basis: np.ndarray,
    spectral_weights: np.ndarray,
    component_arm: str,
    collector: list[dict[str, Any]],
    center_uncertainty_setter: Callable[[np.ndarray | None], None] | None = None,
    old_class_count: int = OLD_CLASS_COUNT,
    ground_full_centers: np.ndarray | None = None,
    ground_class_registry: Sequence[str] | None = None,
    target_old_class_registry: Sequence[str] | None = None,
) -> Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Wrap a D92 component fit with M2.2 and compile it to raw coordinates."""

    if ground_class_registry is not None and target_old_class_registry is not None:
        if tuple(str(value) for value in ground_class_registry) != tuple(
            str(value) for value in target_old_class_registry
        ):
            raise TDHTRCM22Error("M2.2 ground/target old-class registry mismatch")

    def fit(
        rows: np.ndarray,
        labels: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        estimate, final_support, estimate_audit = estimate_m22_transport(
            rows,
            labels,
            class_count,
            k_shot,
            ground_class_centers,
            basis,
            spectral_weights,
            old_class_count=int(old_class_count),
            ground_full_centers=ground_full_centers,
        )
        if center_uncertainty_setter is not None:
            center_uncertainty_setter(estimate.posterior_variance)
        try:
            coefficient, intercept, base_audit = component_fit(
                final_support, labels, class_count, k_shot
            )
        finally:
            if center_uncertainty_setter is not None:
                center_uncertainty_setter(None)
        coefficient64 = np.asarray(coefficient, dtype=np.float64)
        intercept64 = np.asarray(intercept, dtype=np.float64)
        if (
            coefficient64.shape != (int(class_count), FEATURE_DIM)
            or intercept64.shape != (int(class_count),)
            or not np.isfinite(coefficient64).all()
            or not np.isfinite(intercept64).all()
        ):
            raise TDHTRCM22Error("M2.2 wrapped affine head drift")
        raw_coefficient = coefficient64 @ estimate.inverse_transport_matrix
        raw_intercept = intercept64 - raw_coefficient @ estimate.shared_offset
        if not np.isfinite(raw_coefficient).all() or not np.isfinite(raw_intercept).all():
            raise TDHTRCM22Error("M2.2 raw-query affine head became non-finite")
        collector.append(
            {
                "component_arm": component_arm,
                "class_count": int(class_count),
                "k_shot": int(k_shot),
                "transport_block_scales": estimate_audit["transport_block_scales"],
                "adaptive_spectrum_retained_rank": estimate_audit[
                    "adaptive_spectrum_audit"
                ]["adaptive_spectrum_retained_rank"],
                "posterior_uncertainty_trace": estimate_audit[
                    "posterior_uncertainty_trace"
                ],
                "estimated_registration_macs": estimate_audit[
                    "estimated_registration_macs"
                ],
            }
        )
        audit = dict(base_audit)
        audit.update(
            {
                "td_htrc_method": "TD-HTRC-M2.2",
                "td_htrc_m22_component_arm": component_arm,
                "td_htrc_m22_transport_audit": estimate_audit,
                "td_htrc_m22_support_canonicalized_before_fit": True,
                "td_htrc_m22_query_rows_used": 0,
                "td_htrc_m22_query_transform_fitted": False,
                "td_htrc_m22_query_transform_compiled_into_affine": True,
                "td_htrc_m22_low_rank_affine_enabled": True,
                "td_htrc_m22_adaptive_spectrum_enabled": True,
                "td_htrc_m22_posterior_center_enabled": True,
                "td_htrc_m22_posterior_uncertainty_enabled": True,
            }
        )
        return raw_coefficient.astype(np.float32), raw_intercept.astype(np.float32), audit

    return fit


__all__ = [
    "M22_BLOCK_SCALE_MAX",
    "M22_BLOCK_SCALE_MIN",
    "M22_MAX_LOW_RANK_FROBENIUS",
    "M22_PRIOR_MULTIPLIER",
    "M22_RIDGE",
    "TDHTRCM22Error",
    "TDHTRCM22Estimate",
    "build_td_htrc_m22_component_fit",
    "estimate_m22_transport",
]
