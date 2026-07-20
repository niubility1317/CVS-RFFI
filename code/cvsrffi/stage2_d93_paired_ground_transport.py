"""Support-only paired ground-to-target transport for D93.

The immutable ground component is used only to identify a class-common
low-rank domain geometry.  Registered target-old support provides the paired
class supervision.  The fitted inverse transform is applied identically to
target-old, target-new, and query features; ground class centers never produce
query scores.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np


FEATURE_DIM = 288
GROUND_DIM = 160
AUXILIARY_DIM = FEATURE_DIM - GROUND_DIM
AUXILIARY_WEIGHT = 4.0
EPSILON = 1.0e-10
RIDGE_RATIO = 0.10
MAX_UPDATE_SPECTRAL_NORM = 0.50


class D93PairedGroundTransportError(ValueError):
    """Raised when D93 input or numerical closure drifts."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _rows(value: np.ndarray, width: int, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if (
        rows.ndim != 2
        or rows.shape[1] != int(width)
        or len(rows) == 0
        or not np.isfinite(rows).all()
    ):
        raise D93PairedGroundTransportError(f"D93 {name} drift")
    return rows


def _normalize_rows(value: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or bool(np.any(norms <= EPSILON)):
        raise D93PairedGroundTransportError(f"D93 {name} norm drift")
    return rows / norms


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "value_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonical_svd_basis(rows: np.ndarray, maximum_rank: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise D93PairedGroundTransportError("D93 SVD input drift")
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    if len(singular) == 0 or float(singular[0]) <= EPSILON:
        return np.zeros((matrix.shape[1], 0), dtype=np.float64), singular
    threshold = max(EPSILON, float(singular[0]) * 1.0e-10)
    rank = min(int(maximum_rank), int(np.sum(singular > threshold)))
    basis = vh[:rank].T.copy()
    # Canonicalize each sign so hashes are stable across LAPACK implementations.
    for column in range(rank):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1.0
    return basis, singular


def canonical_ground_geometry(
    domain_class_prototypes: np.ndarray,
    domain_class_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return spherical class centers plus identity and nuisance bases."""

    prototypes = np.asarray(domain_class_prototypes, dtype=np.float64)
    mask = np.asarray(domain_class_mask, dtype=bool)
    if (
        prototypes.ndim != 3
        or prototypes.shape[2] != GROUND_DIM
        or mask.shape != prototypes.shape[:2]
        or prototypes.shape[0] < 2
        or prototypes.shape[1] < 2
        or not np.isfinite(prototypes).all()
        or bool(np.any(np.sum(mask, axis=0) < 2))
    ):
        raise D93PairedGroundTransportError("D93 ground prototype tensor drift")

    normalized = np.zeros_like(prototypes)
    canonical: list[np.ndarray] = []
    nuisance_rows: list[np.ndarray] = []
    active_counts: list[int] = []
    class_effective_domain_counts: list[float] = []
    class_stable_ranks: list[float] = []
    class_near_duplicate_pair_fractions: list[float] = []
    for class_index in range(prototypes.shape[1]):
        active = _normalize_rows(
            prototypes[mask[:, class_index], class_index],
            f"ground class {class_index}",
        )
        normalized[mask[:, class_index], class_index] = active
        center = _normalize_rows(active.mean(axis=0, keepdims=True), "ground center")[0]
        canonical.append(center)
        nuisance_rows.extend(active - center[None, :])
        active_counts.append(len(active))
        centered_active = active - active.mean(axis=0, keepdims=True)
        class_singular = np.linalg.svd(centered_active, compute_uv=False)
        class_eigenvalues = class_singular**2
        class_eigenvalue_sum = float(class_eigenvalues.sum())
        if class_eigenvalue_sum <= EPSILON:
            class_effective_domain_counts.append(1.0)
            class_stable_ranks.append(0.0)
        else:
            class_effective_domain_counts.append(
                float(
                    class_eigenvalue_sum**2
                    / max(EPSILON, float(np.sum(class_eigenvalues**2)))
                )
            )
            class_stable_ranks.append(
                float(
                    class_eigenvalue_sum
                    / max(EPSILON, float(class_eigenvalues.max()))
                )
            )
        cosine = np.clip(active @ active.T, -1.0, 1.0)
        upper = cosine[np.triu_indices(len(active), k=1)]
        class_near_duplicate_pair_fractions.append(
            float(np.mean((1.0 - upper) <= 1.0e-4)) if len(upper) else 0.0
        )

    centers = np.stack(canonical)
    centered_identity = centers - centers.mean(axis=0, keepdims=True)
    identity_basis, identity_singular = _canonical_svd_basis(
        centered_identity, centers.shape[0] - 1
    )
    nuisance_matrix = np.stack(nuisance_rows)
    nuisance_basis_all, nuisance_singular = _canonical_svd_basis(
        nuisance_matrix, prototypes.shape[0] - 1
    )
    positive = nuisance_singular[nuisance_singular > EPSILON]
    if len(positive) == 0:
        raise D93PairedGroundTransportError("D93 ground nuisance rank is zero")
    eigenvalues = positive**2
    participation_ratio = float(eigenvalues.sum() ** 2 / np.sum(eigenvalues**2))
    nuisance_rank = min(
        nuisance_basis_all.shape[1],
        max(1, int(np.ceil(participation_ratio))),
    )
    nuisance_basis = nuisance_basis_all[:, :nuisance_rank]
    audit = {
        "ground_domain_count": int(prototypes.shape[0]),
        "ground_class_count": int(prototypes.shape[1]),
        "ground_component_input_count": int(mask.sum()),
        "ground_active_domain_count_by_class": active_counts,
        "ground_effective_domain_count_by_class": class_effective_domain_counts,
        "ground_stable_rank_by_class": class_stable_ranks,
        "ground_near_duplicate_pair_fraction_cosine_eps_1e_4_by_class": class_near_duplicate_pair_fractions,
        "identity_rank": int(identity_basis.shape[1]),
        "nuisance_positive_rank": int(len(positive)),
        "nuisance_participation_ratio": participation_ratio,
        "nuisance_retained_rank": int(nuisance_rank),
        "ground_center_sha256": _sha256_array(centers.astype(np.float32)),
        "identity_basis_sha256": _sha256_array(identity_basis.astype(np.float32)),
        "nuisance_basis_sha256": _sha256_array(nuisance_basis.astype(np.float32)),
        "ground_sample_feature_access": False,
        "ground_member_or_exemplar_access": False,
    }
    return centers, identity_basis, nuisance_basis, audit


def _target_class_centers(
    target_old_z160: np.ndarray,
    target_old_labels: np.ndarray,
    class_count: int,
) -> tuple[np.ndarray, int]:
    rows = _normalize_rows(
        _rows(target_old_z160, GROUND_DIM, "target-old z160"), "target-old z160"
    )
    labels = np.asarray(target_old_labels, dtype=np.int64)
    if labels.shape != (len(rows),) or not np.array_equal(
        np.unique(labels), np.arange(int(class_count), dtype=np.int64)
    ):
        raise D93PairedGroundTransportError("D93 target-old label registry drift")
    counts = np.bincount(labels, minlength=int(class_count))
    if bool(np.any(counts <= 0)) or len(set(counts.tolist())) != 1:
        raise D93PairedGroundTransportError("D93 target-old K-shot balance drift")
    centers = np.stack(
        [
            _normalize_rows(
                rows[labels == index].mean(axis=0, keepdims=True),
                f"target-old center {index}",
            )[0]
            for index in range(int(class_count))
        ]
    )
    return centers, int(counts[0])


def _fit_operator(
    ground_centers: np.ndarray,
    target_centers: np.ndarray,
    identity_basis: np.ndarray,
    nuisance_basis: np.ndarray,
    *,
    include_nuisance_scale: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float],
]:
    ground_mean = ground_centers.mean(axis=0)
    target_mean = target_centers.mean(axis=0)
    ground_centered = ground_centers - ground_mean[None, :]
    target_centered = target_centers - target_mean[None, :]
    residual = target_centered - ground_centered

    identity_coordinates = ground_centered @ identity_basis
    nuisance_residual = residual @ nuisance_basis
    gram = identity_coordinates.T @ identity_coordinates
    ridge = max(
        1.0e-8,
        RIDGE_RATIO * float(np.trace(gram)) / max(1, identity_basis.shape[1]),
    )
    interaction_t = np.linalg.solve(
        gram + ridge * np.eye(identity_basis.shape[1]),
        identity_coordinates.T @ nuisance_residual,
    )
    interaction = interaction_t.T
    update = nuisance_basis @ interaction @ identity_basis.T

    nuisance_scale = np.zeros(nuisance_basis.shape[1], dtype=np.float64)
    if include_nuisance_scale:
        remaining = residual - ground_centered @ update.T
        nuisance_coordinates = ground_centered @ nuisance_basis
        remaining_coordinates = remaining @ nuisance_basis
        denominator = np.sum(nuisance_coordinates**2, axis=0) + ridge
        nuisance_scale = np.sum(
            nuisance_coordinates * remaining_coordinates, axis=0
        ) / denominator
        update += nuisance_basis @ np.diag(nuisance_scale) @ nuisance_basis.T

    raw_update_norm = float(np.linalg.norm(update, ord=2))
    update_scale = min(
        1.0,
        MAX_UPDATE_SPECTRAL_NORM / max(EPSILON, raw_update_norm),
    )
    update *= update_scale
    interaction *= update_scale
    nuisance_scale *= update_scale
    operator = np.eye(GROUND_DIM, dtype=np.float64) + update
    singular = np.linalg.svd(operator, compute_uv=False)
    condition = float(singular.max() / singular.min())
    if not np.isfinite(condition) or float(singular.min()) <= EPSILON:
        raise D93PairedGroundTransportError("D93 transport is singular")
    translation = target_mean - operator @ ground_mean
    inverse_row = np.linalg.inv(operator).T
    predicted = ground_centers @ operator.T + translation[None, :]
    identity_translation = target_mean - ground_mean
    identity_predicted = ground_centers + identity_translation[None, :]
    projected_residual = (residual @ nuisance_basis) @ nuisance_basis.T
    residual_energy = float(np.sum(residual**2))
    nuisance_coverage = float(
        np.sum(projected_residual**2) / max(EPSILON, residual_energy)
    )
    statistics = {
        "ridge": ridge,
        "raw_update_spectral_norm": raw_update_norm,
        "update_scale": float(update_scale),
        "update_spectral_norm": float(np.linalg.norm(update, ord=2)),
        "operator_condition_number": condition,
        "operator_min_singular_value": float(singular.min()),
        "operator_max_singular_value": float(singular.max()),
        "paired_rmse": float(np.sqrt(np.mean((predicted - target_centers) ** 2))),
        "translation_only_rmse": float(
            np.sqrt(np.mean((identity_predicted - target_centers) ** 2))
        ),
        "target_shift_ground_nuisance_coverage": nuisance_coverage,
        "target_shift_out_of_ground_nuisance_energy_ratio": float(
            max(0.0, 1.0 - nuisance_coverage)
        ),
    }
    return operator, inverse_row, translation, interaction, nuisance_scale, statistics


@dataclass(frozen=True)
class D93PairedGroundTransport:
    """Closed-form target state; bases are derived from immutable ground state."""

    inverse_row_fp32: np.ndarray
    translation_fp32: np.ndarray
    identity_basis_fp32: np.ndarray
    nuisance_basis_fp32: np.ndarray
    interaction_fp16: np.ndarray
    nuisance_scale_fp16: np.ndarray
    include_nuisance_scale: bool
    audit: dict[str, Any]

    def __post_init__(self) -> None:
        identity_rank = self.identity_basis_fp32.shape[1]
        nuisance_rank = self.nuisance_basis_fp32.shape[1]
        if (
            self.inverse_row_fp32.shape != (GROUND_DIM, GROUND_DIM)
            or self.translation_fp32.shape != (GROUND_DIM,)
            or self.identity_basis_fp32.shape[0] != GROUND_DIM
            or self.nuisance_basis_fp32.shape[0] != GROUND_DIM
            or self.interaction_fp16.shape != (nuisance_rank, identity_rank)
            or self.nuisance_scale_fp16.shape != (nuisance_rank,)
            or self.inverse_row_fp32.dtype != np.float32
            or self.translation_fp32.dtype != np.float32
            or self.identity_basis_fp32.dtype != np.float32
            or self.nuisance_basis_fp32.dtype != np.float32
            or self.interaction_fp16.dtype != np.float16
            or self.nuisance_scale_fp16.dtype != np.float16
            or not np.isfinite(self.inverse_row_fp32).all()
        ):
            raise D93PairedGroundTransportError("D93 transport state drift")
        for name in (
            "inverse_row_fp32",
            "translation_fp32",
            "identity_basis_fp32",
            "nuisance_basis_fp32",
            "interaction_fp16",
            "nuisance_scale_fp16",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), getattr(self, name).dtype))

    @property
    def incremental_state_bytes(self) -> int:
        # Ground-derived bases are already represented by the immutable component.
        return int(
            self.translation_fp32.astype(np.float16).nbytes
            + self.interaction_fp16.nbytes
            + self.nuisance_scale_fp16.nbytes
        )

    @property
    def parameter_count(self) -> int:
        return int(
            self.translation_fp32.size
            + self.interaction_fp16.size
            + self.nuisance_scale_fp16.size
        )

    @property
    def macs_per_query(self) -> int:
        # Dense value is materialized transiently; deployment may use Woodbury.
        return int(2 * GROUND_DIM * (self.identity_basis_fp32.shape[1] + self.nuisance_basis_fp32.shape[1]))


def fit_paired_ground_transport(
    domain_class_prototypes: np.ndarray,
    domain_class_mask: np.ndarray,
    target_old_z160: np.ndarray,
    target_old_labels: np.ndarray,
    *,
    include_nuisance_scale: bool,
) -> D93PairedGroundTransport:
    """Fit D93 from immutable aggregate ground centers and target-old support."""

    ground_centers, identity_basis, nuisance_basis, ground_audit = (
        canonical_ground_geometry(domain_class_prototypes, domain_class_mask)
    )
    target_centers, k_shot = _target_class_centers(
        target_old_z160, target_old_labels, ground_centers.shape[0]
    )
    (
        operator,
        inverse_row,
        translation,
        interaction,
        nuisance_scale,
        statistics,
    ) = _fit_operator(
        ground_centers,
        target_centers,
        identity_basis,
        nuisance_basis,
        include_nuisance_scale=bool(include_nuisance_scale),
    )
    audit = {
        "schema": "cvs.phase2.d93.paired_ground_transport.v1",
        "mode": "interaction_plus_nuisance_scale" if include_nuisance_scale else "interaction_only",
        "k_shot": int(k_shot),
        "target_old_class_count": int(ground_centers.shape[0]),
        "target_new_support_used_for_transport_fit": False,
        "query_rows_used": 0,
        "query_labels_used": False,
        "query_role_oracle_access": False,
        "query_class_quota_access": False,
        "ground_to_target_identity_pairing_used": True,
        "ground_direct_query_score_access": False,
        "ground_target_prototype_overwrite": False,
        "ground_component_update_access": False,
        "ground_aggregate_prototypes_only": True,
        "ground_member_ids_available": False,
        "target_clean_iq_access": False,
        "target_new_clean_iq_access": False,
        "source_sample_replay_access": False,
        "same_physical_iq_multi_channel_views": False,
        "phase2_channel_simulator_calls": 0,
        "single_received_iq_only": True,
        "operator_sha256": _sha256_array(operator.astype(np.float32)),
        "inverse_sha256": _sha256_array(inverse_row.astype(np.float32)),
        "translation_sha256": _sha256_array(translation.astype(np.float32)),
        **ground_audit,
        **statistics,
    }
    return D93PairedGroundTransport(
        inverse_row_fp32=np.asarray(inverse_row, dtype=np.float32),
        translation_fp32=np.asarray(translation, dtype=np.float32),
        identity_basis_fp32=np.asarray(identity_basis, dtype=np.float32),
        nuisance_basis_fp32=np.asarray(nuisance_basis, dtype=np.float32),
        interaction_fp16=np.asarray(interaction, dtype=np.float16),
        nuisance_scale_fp16=np.asarray(nuisance_scale, dtype=np.float16),
        include_nuisance_scale=bool(include_nuisance_scale),
        audit=audit,
    )


def transform_z160(
    z160: np.ndarray, transport: D93PairedGroundTransport
) -> np.ndarray:
    rows = _normalize_rows(_rows(z160, GROUND_DIM, "z160 transform"), "z160 transform")
    transformed = (
        rows - transport.translation_fp32.astype(np.float64)[None, :]
    ) @ transport.inverse_row_fp32.astype(np.float64)
    return _readonly(
        _normalize_rows(transformed, "transport output").astype(np.float32), np.float32
    )


def transform_registered_features(
    features: np.ndarray, transport: D93PairedGroundTransport
) -> np.ndarray:
    """Transform z160 while preserving the locked same-IQ FFT96/RF32 block."""

    rows = _rows(features, FEATURE_DIM, "registered features")
    primary = _normalize_rows(rows[:, :GROUND_DIM], "registered z160")
    auxiliary = _normalize_rows(rows[:, GROUND_DIM:], "registered auxiliary")
    transformed_primary = transform_z160(primary, transport).astype(np.float64)
    registered = _normalize_rows(
        np.concatenate(
            [transformed_primary, AUXILIARY_WEIGHT * auxiliary], axis=1
        ),
        "registered transport output",
    )
    return _readonly(registered.astype(np.float32), np.float32)


__all__ = [
    "AUXILIARY_DIM",
    "D93PairedGroundTransport",
    "D93PairedGroundTransportError",
    "FEATURE_DIM",
    "GROUND_DIM",
    "canonical_ground_geometry",
    "fit_paired_ground_transport",
    "transform_registered_features",
    "transform_z160",
]
