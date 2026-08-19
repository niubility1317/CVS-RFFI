"""Opt-in TD-HTRC M2.1 target-domain transport for module two.

The base D92 E0 path estimates each class centre only inside the current
target-support cloud.  This module adds the smallest identifiable transport
state: a shared 160-dimensional target offset estimated from paired old-class
ground and target centres.  It deliberately does not fit a free affine matrix,
does not read query rows, and keeps the existing classwise Cauchy centre step.

The returned affine head is compiled back to raw target-query coordinates.  A
caller therefore does not need to retain a query-time transform or update any
state after registration:

    score(q - b, W, a) == score(q, W, a - W[:, :160] @ b)

where ``b`` is the shared identity-block offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from cvsrffi.stage2_d81_ground_nuisance_cauchy_center import (
    translate_to_robust_centers,
)


Z_DIM = 160
FEATURE_DIM = 288
OLD_CLASS_COUNT = 6
ENERGY_EPSILON = 1.0e-24
OFFSET_COVARIANCE_FLOOR = 1.0e-10
IRLS_STEPS = 3


class TDHTRCError(ValueError):
    """Raised when the support-only TD-HTRC M2.1 closure is invalid."""


def _readonly(value: np.ndarray, dtype: np.dtype[Any] | type) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _validate_basis(
    basis: np.ndarray, spectral_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(basis, dtype=np.float64)
    rho = np.asarray(spectral_weights, dtype=np.float64)
    if (
        u.ndim != 2
        or u.shape[0] != Z_DIM
        or u.shape[1] <= 0
        or rho.shape != (u.shape[1],)
        or not np.isfinite(u).all()
        or not np.isfinite(rho).all()
        or np.any(rho <= 0.0)
    ):
        raise TDHTRCError("TD-HTRC requires a finite positive ground spectrum")
    return u, rho


def _validate_support(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    old_class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    classes, shots = int(class_count), int(k_shot)
    if (
        x.ndim != 2
        or x.shape[1] < Z_DIM
        or y.shape != (len(x),)
        or len(x) != classes * shots
        or classes < old_class_count
        or shots <= 0
        or not np.isfinite(x).all()
        or not np.array_equal(np.unique(y), np.arange(classes))
        or any(int(np.sum(y == index)) != shots for index in range(classes))
    ):
        raise TDHTRCError("TD-HTRC requires a balanced finite support registry")
    return x, y


def _validate_ground_centers(
    ground_class_centers: np.ndarray, old_class_count: int
) -> np.ndarray:
    anchors = np.asarray(ground_class_centers, dtype=np.float64)
    if (
        anchors.shape != (old_class_count, Z_DIM)
        or not np.isfinite(anchors).all()
    ):
        raise TDHTRCError(
            "TD-HTRC ground anchors must be [old_class_count,160] finite centres"
        )
    return anchors


@dataclass(frozen=True)
class TDHTRCTransportEstimate:
    """Immutable support-only estimate of the shared target transport state."""

    shared_offset: np.ndarray
    offset_covariance: np.ndarray
    ground_centers: np.ndarray
    target_centers: np.ndarray
    anchor_difference: np.ndarray
    anchor_weights: np.ndarray
    target_cauchy_weights: np.ndarray
    effective_samples_by_class: np.ndarray
    audit: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shared_offset", _readonly(self.shared_offset, np.float64))
        object.__setattr__(
            self, "offset_covariance", _readonly(self.offset_covariance, np.float64)
        )
        object.__setattr__(self, "ground_centers", _readonly(self.ground_centers, np.float64))
        object.__setattr__(self, "target_centers", _readonly(self.target_centers, np.float64))
        object.__setattr__(
            self, "anchor_difference", _readonly(self.anchor_difference, np.float64)
        )
        object.__setattr__(self, "anchor_weights", _readonly(self.anchor_weights, np.float64))
        object.__setattr__(
            self,
            "target_cauchy_weights",
            _readonly(self.target_cauchy_weights, np.float64),
        )
        object.__setattr__(
            self,
            "effective_samples_by_class",
            _readonly(self.effective_samples_by_class, np.float64),
        )
        object.__setattr__(self, "audit", dict(self.audit))


def _robust_old_target_centers(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    basis: np.ndarray,
    spectral_weights: np.ndarray,
    old_class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Reuse the locked one-step Cauchy rule to form old-class anchors."""

    old_mask = labels < old_class_count
    old_rows = rows[old_mask]
    old_labels = labels[old_mask]
    transformed, center_audit = translate_to_robust_centers(
        old_rows,
        old_labels,
        old_class_count,
        k_shot,
        basis,
        spectral_weights,
    )
    centers = np.stack(
        [transformed[old_labels == index, :Z_DIM].mean(axis=0) for index in range(old_class_count)]
    )
    weights = np.asarray(
        center_audit["normalized_cauchy_weight_by_class"], dtype=np.float64
    )
    effective = np.asarray(
        center_audit["effective_sample_size_by_class"], dtype=np.float64
    )
    if (
        centers.shape != (old_class_count, Z_DIM)
        or weights.shape != (old_class_count, k_shot)
        or effective.shape != (old_class_count,)
        or not np.isfinite(centers).all()
        or not np.isfinite(weights).all()
        or not np.isfinite(effective).all()
    ):
        raise TDHTRCError("TD-HTRC target anchor closure drift")
    return centers, weights, effective, center_audit


def estimate_shared_target_transport(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    ground_class_centers: np.ndarray,
    basis: np.ndarray,
    spectral_weights: np.ndarray,
    *,
    old_class_count: int = OLD_CLASS_COUNT,
) -> TDHTRCTransportEstimate:
    """Estimate a shared target offset from old-class ground/target pairs.

    Only the first ``old_class_count`` labelled classes are used as anchors.
    The classwise target centres use the existing support-only Cauchy rule;
    three fixed Cauchy IRLS steps then downweight inconsistent class pairs.
    No query or source sample is accepted by this function.
    """

    classes = int(class_count)
    old_count = int(old_class_count)
    x, y = _validate_support(rows, labels, classes, int(k_shot), old_count)
    anchors = _validate_ground_centers(ground_class_centers, old_count)
    u, rho = _validate_basis(basis, spectral_weights)
    target_centers, target_weights, effective, center_audit = _robust_old_target_centers(
        x, y, classes, int(k_shot), u, rho, old_count
    )
    differences = target_centers - anchors
    class_uncertainty = np.asarray(
        [
            np.mean(
                (x[y == index, :Z_DIM] - target_centers[index][None, :]) ** 2
            )
            / max(float(effective[index]), 1.0)
            for index in range(old_count)
        ],
        dtype=np.float64,
    )
    quality = effective / np.maximum(class_uncertainty, OFFSET_COVARIANCE_FLOOR)
    quality = quality / np.sum(quality)

    offset = np.sum(quality[:, None] * differences, axis=0)
    residual_norm = np.linalg.norm(differences - offset[None, :], axis=1)
    scale = float(np.median(residual_norm) / 0.6744897501960817)
    scale = max(scale, float(np.sqrt(OFFSET_COVARIANCE_FLOOR)))
    final_weights = quality.copy()
    for _ in range(IRLS_STEPS):
        residual = differences - offset[None, :]
        robust = 1.0 / (1.0 + np.square(np.linalg.norm(residual, axis=1) / scale))
        final_weights = quality * robust
        final_weights /= np.sum(final_weights)
        offset = np.sum(final_weights[:, None] * differences, axis=0)

    centered = differences - offset[None, :]
    covariance = (centered * final_weights[:, None]).T @ centered
    covariance /= max(1.0 - float(np.sum(np.square(final_weights))), 1.0e-6)
    covariance += OFFSET_COVARIANCE_FLOOR * np.eye(Z_DIM, dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)

    total_energy = float(np.sum(final_weights * np.sum(np.square(differences), axis=1)))
    residual_energy = float(np.sum(final_weights * np.sum(np.square(centered), axis=1)))
    shared_explanation = 1.0 if total_energy <= ENERGY_EPSILON else 1.0 - residual_energy / total_energy
    shared_explanation = float(np.clip(shared_explanation, 0.0, 1.0))
    projected_norm = float(np.sum(np.square(u.T @ offset)))
    offset_norm_sq = float(np.sum(np.square(offset)))
    spectrum_coverage = (
        0.0 if offset_norm_sq <= ENERGY_EPSILON else float(np.clip(projected_norm / offset_norm_sq, 0.0, 1.0))
    )
    rank = int(u.shape[1])
    support_rows = int(len(x))
    estimated_macs = int(
        old_count * int(k_shot) * (2 * Z_DIM * rank + rank)
        + old_count * Z_DIM
        + IRLS_STEPS * old_count * (2 * Z_DIM + 4)
    )
    audit: dict[str, Any] = {
        "schema": "cvs.phase2.td_htrc.m21.transport.v1",
        "method": "TD-HTRC-M2.1",
        "support_rows_used": support_rows,
        "query_rows_used": 0,
        "query_truth_used": False,
        "source_sample_used": False,
        "old_class_count": old_count,
        "registered_class_count": classes,
        "k_shot": int(k_shot),
        "ground_anchor_source": "immutable_phase1_aggregate_old_class_centers",
        "ground_anchor_update_access": False,
        "target_anchor_center_rule": "existing_one_step_classwise_cauchy",
        "target_cauchy_weight_by_class": target_weights.tolist(),
        "effective_samples_by_class": effective.tolist(),
        "shared_offset_norm": float(np.linalg.norm(offset)),
        "shared_offset_covariance_trace": float(np.trace(covariance)),
        "shared_offset_covariance_floor": OFFSET_COVARIANCE_FLOOR,
        "shared_offset_irls_steps": IRLS_STEPS,
        "shared_offset_irls_scale": scale,
        "shared_explanation_ratio": shared_explanation,
        "spectral_coverage_ratio": spectrum_coverage,
        "spectral_perpendicular_ratio": float(1.0 - spectrum_coverage),
        "shared_offset_contains_spectral_perpendicular_component": bool(
            spectrum_coverage < 1.0 - 1.0e-12
        ),
        "ground_spectrum_rank": rank,
        "estimated_registration_macs": estimated_macs,
        # The offset and covariance are registration-time diagnostics only. The
        # offset is compiled into the returned affine intercept, so neither
        # object is part of the persistent query state.
        "transport_estimate_transient_state_bytes": int(
            offset.nbytes + covariance.nbytes
        ),
        "persistent_transport_state_bytes": 0,
        "query_transport_mode": "compiled_into_affine_intercept",
        "center_audit_schema": center_audit.get("schema"),
    }
    return TDHTRCTransportEstimate(
        shared_offset=offset,
        offset_covariance=covariance,
        ground_centers=anchors,
        target_centers=target_centers,
        anchor_difference=differences,
        anchor_weights=final_weights,
        target_cauchy_weights=target_weights,
        effective_samples_by_class=effective,
        audit=audit,
    )


def apply_shared_target_transport(
    rows: np.ndarray, estimate: TDHTRCTransportEstimate
) -> np.ndarray:
    """Map support rows to the canonical space by subtracting the shared offset."""

    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < Z_DIM or not np.isfinite(values).all():
        raise TDHTRCError("TD-HTRC transport rows drift")
    result = values.copy()
    result[:, :Z_DIM] -= estimate.shared_offset[None, :]
    if not np.isfinite(result).all():
        raise TDHTRCError("TD-HTRC canonical support became non-finite")
    return result


def build_td_htrc_component_fit(
    component_fit: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    *,
    ground_class_centers: np.ndarray,
    basis: np.ndarray,
    spectral_weights: np.ndarray,
    component_arm: str,
    collector: list[dict[str, Any]],
    old_class_count: int = OLD_CLASS_COUNT,
    ground_class_registry: Sequence[str] | None = None,
    target_old_class_registry: Sequence[str] | None = None,
) -> Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Wrap a D92 component fit with the support-only M2.1 transport.

    The wrapped component receives canonicalized support.  Its coefficient is
    returned unchanged, while its intercept is shifted so raw target queries
    are scored as if the same shared offset had been subtracted.  The classwise
    Cauchy translation is applied after shared transport and before the wrapped
    component, preserving the existing module-two robustification.
    """

    anchors = _validate_ground_centers(ground_class_centers, int(old_class_count))
    u, rho = _validate_basis(basis, spectral_weights)
    if ground_class_registry is not None and target_old_class_registry is not None:
        if tuple(str(value) for value in ground_class_registry) != tuple(
            str(value) for value in target_old_class_registry
        ):
            raise TDHTRCError("TD-HTRC ground/target old-class registry mismatch")

    def fit(
        rows: np.ndarray,
        labels: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        estimate = estimate_shared_target_transport(
            rows,
            labels,
            class_count,
            k_shot,
            anchors,
            u,
            rho,
            old_class_count=int(old_class_count),
        )
        canonical = apply_shared_target_transport(rows, estimate)
        robust_rows, robust_audit = translate_to_robust_centers(
            canonical,
            labels,
            int(class_count),
            int(k_shot),
            u,
            rho,
        )
        coefficient, intercept, base_audit = component_fit(
            robust_rows, labels, class_count, k_shot
        )
        coefficient64 = np.asarray(coefficient, dtype=np.float64)
        intercept64 = np.asarray(intercept, dtype=np.float64)
        if (
            coefficient64.shape != (int(class_count), FEATURE_DIM)
            or intercept64.shape != (int(class_count),)
            or not np.isfinite(coefficient64).all()
            or not np.isfinite(intercept64).all()
        ):
            raise TDHTRCError("TD-HTRC wrapped affine head drift")
        raw_intercept = intercept64 - coefficient64[:, :Z_DIM] @ estimate.shared_offset
        if not np.isfinite(raw_intercept).all():
            raise TDHTRCError("TD-HTRC raw-query intercept became non-finite")
        record = {
            "component_arm": component_arm,
            "class_count": int(class_count),
            "k_shot": int(k_shot),
            "shared_offset_norm": estimate.audit["shared_offset_norm"],
            "shared_explanation_ratio": estimate.audit["shared_explanation_ratio"],
            "spectral_coverage_ratio": estimate.audit["spectral_coverage_ratio"],
            "estimated_registration_macs": estimate.audit[
                "estimated_registration_macs"
            ],
        }
        collector.append(record)
        audit = dict(base_audit)
        audit.update(
            {
                "td_htrc_method": "TD-HTRC-M2.1",
                "td_htrc_component_arm": component_arm,
                "td_htrc_transport_audit": estimate.audit,
                "td_htrc_robust_center_audit": robust_audit,
                "td_htrc_support_canonicalized_before_fit": True,
                "td_htrc_query_rows_used": 0,
                "td_htrc_query_transform_fitted": False,
                "td_htrc_query_transform_compiled_into_intercept": True,
                "td_htrc_classwise_cauchy_retained": True,
                "td_htrc_low_rank_affine_enabled": False,
                "td_htrc_physical_nuisance_enabled": False,
                "td_htrc_target_adaptive_spectrum_enabled": False,
                "td_htrc_center_posterior_enabled": False,
            }
        )
        return coefficient64.astype(np.float32), raw_intercept.astype(np.float32), audit

    return fit


__all__ = [
    "FEATURE_DIM",
    "IRLS_STEPS",
    "OLD_CLASS_COUNT",
    "TDHTRCError",
    "TDHTRCTransportEstimate",
    "Z_DIM",
    "apply_shared_target_transport",
    "build_td_htrc_component_fit",
    "estimate_shared_target_transport",
]
