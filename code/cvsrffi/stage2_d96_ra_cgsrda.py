"""Redundancy-aware, coverage-gated SRDA core for D96.

The immutable ground grid contributes only a class-agnostic low-rank nuisance
covariance.  Target class means are always estimated from registered target
support.  Query rows are scored independently by a compiled affine SRDA head;
no query batch statistic is retained or updated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np


FEATURE_DIM = 160
EPSILON = 1.0e-12


class D96RACGSRDAError(ValueError):
    """Raised when the frozen D96 contract or numerical state drifts."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _normalized_rows(value: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != FEATURE_DIM or not np.isfinite(rows).all():
        raise D96RACGSRDAError(f"{name} must be finite [N,{FEATURE_DIM}]")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if bool(np.any(norms <= EPSILON)):
        raise D96RACGSRDAError(f"{name} contains a zero-norm row")
    return rows / norms


@dataclass(frozen=True)
class Phase1LockedConfig:
    """Parameters selected only by Phase1 evidence before target access."""

    tau: float
    ridge: float
    temp_base: float
    temp_aux: float
    max_rank: int
    phase1_receipt_sha256: str

    def __post_init__(self) -> None:
        values = (self.tau, self.ridge, self.temp_base, self.temp_aux)
        if (
            not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values)
            or isinstance(self.max_rank, bool)
            or not isinstance(self.max_rank, (int, np.integer))
            or not 1 <= int(self.max_rank) <= 4
            or len(self.phase1_receipt_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.phase1_receipt_sha256)
        ):
            raise D96RACGSRDAError(
                "D96 requires positive Phase1-locked tau/ridge/temperatures "
                "and 1 <= max_rank <= 4"
            )

    @property
    def lock_digest(self) -> str:
        payload = {
            "max_rank": int(self.max_rank),
            "phase1_receipt_sha256": self.phase1_receipt_sha256,
            "ridge": float(self.ridge),
            "tau": float(self.tau),
            "temp_aux": float(self.temp_aux),
            "temp_base": float(self.temp_base),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GroundDomainModel:
    """Immutable redundancy-aware ground nuisance summary."""

    class_means_fp32: np.ndarray
    nuisance_basis_fp32: np.ndarray
    nuisance_eigenvalues_fp32: np.ndarray
    density_weights_fp32: np.ndarray
    effective_domain_count: float
    config_lock_digest: str
    audit: dict[str, Any]

    def __post_init__(self) -> None:
        classes = int(self.class_means_fp32.shape[0])
        rank = int(self.nuisance_basis_fp32.shape[1])
        domains = int(self.density_weights_fp32.shape[0])
        if (
            self.class_means_fp32.shape != (classes, FEATURE_DIM)
            or self.nuisance_basis_fp32.shape != (FEATURE_DIM, rank)
            or self.nuisance_eigenvalues_fp32.shape != (rank,)
            or domains < 2
            or rank > 4
            or not np.isfinite(self.class_means_fp32).all()
            or not np.isfinite(self.nuisance_basis_fp32).all()
            or not np.isfinite(self.nuisance_eigenvalues_fp32).all()
            or not np.isfinite(self.density_weights_fp32).all()
            or bool(np.any(self.nuisance_eigenvalues_fp32 <= 0.0))
            or bool(np.any(self.density_weights_fp32 <= 0.0))
            or not np.isclose(float(np.sum(self.density_weights_fp32)), 1.0, atol=2e-6)
            or not math.isfinite(float(self.effective_domain_count))
            or not 1.0 <= float(self.effective_domain_count) <= float(domains)
            or len(self.config_lock_digest) != 64
        ):
            raise D96RACGSRDAError("D96 ground model invariant drift")
        basis = np.asarray(self.nuisance_basis_fp32, dtype=np.float64)
        if rank and not np.allclose(basis.T @ basis, np.eye(rank), atol=2e-5):
            raise D96RACGSRDAError("D96 nuisance basis must be orthonormal")
        if not np.allclose(np.linalg.norm(self.class_means_fp32, axis=1), 1.0, atol=2e-5):
            raise D96RACGSRDAError("D96 class means must be normalized")


@dataclass(frozen=True)
class SRDAState:
    """Compiled target-only class means and affine SRDA scorer."""

    classes: tuple[int, ...]
    coefficient_fp32: np.ndarray
    intercept_fp32: np.ndarray
    target_means_fp32: np.ndarray
    coverage_rho: float
    k_shot: int
    config_lock_digest: str
    audit: dict[str, Any]

    def __post_init__(self) -> None:
        count = len(self.classes)
        if (
            count < 2
            or self.coefficient_fp32.shape != (count, FEATURE_DIM)
            or self.intercept_fp32.shape != (count,)
            or self.target_means_fp32.shape != (count, FEATURE_DIM)
            or not np.isfinite(self.coefficient_fp32).all()
            or not np.isfinite(self.intercept_fp32).all()
            or not np.isfinite(self.target_means_fp32).all()
            or not 0.0 <= float(self.coverage_rho) <= 1.0
            or int(self.k_shot) < 1
            or len(self.config_lock_digest) != 64
        ):
            raise D96RACGSRDAError("D96 SRDA state invariant drift")

    def logits(self, query_rows: np.ndarray) -> np.ndarray:
        query = _normalized_rows(query_rows, "D96 query rows")
        result = query @ self.coefficient_fp32.T + self.intercept_fp32[None, :]
        if not np.isfinite(result).all():
            raise D96RACGSRDAError("D96 produced non-finite query logits")
        return np.asarray(result, dtype=np.float32)


def build_ground_domain_model(
    domain_class_prototypes: np.ndarray,
    domain_class_mask: np.ndarray,
    config: Phase1LockedConfig,
) -> GroundDomainModel:
    """Build density-balanced domain residuals and an adaptive rank<=4 spectrum."""

    prototypes = np.asarray(domain_class_prototypes, dtype=np.float64)
    mask = np.asarray(domain_class_mask, dtype=bool)
    if (
        prototypes.ndim != 3
        or prototypes.shape[2] != FEATURE_DIM
        or mask.shape != prototypes.shape[:2]
        or prototypes.shape[0] < 2
        or prototypes.shape[1] < 2
        or not np.isfinite(prototypes).all()
    ):
        raise D96RACGSRDAError("D96 requires finite [domain,class,160] ground state")
    has_any_class = np.any(mask, axis=1)
    full_domains = np.all(mask, axis=1)
    if bool(np.any(has_any_class != full_domains)) or int(np.sum(full_domains)) < 2:
        raise D96RACGSRDAError("D96 ground mask must contain complete domain grids")

    domains, classes = int(np.sum(full_domains)), int(prototypes.shape[1])
    active = _normalized_rows(
        prototypes[full_domains].reshape(domains * classes, FEATURE_DIM),
        "D96 ground prototypes",
    ).reshape(domains, classes, FEATURE_DIM)
    initial_class_means = active.mean(axis=0)
    residual = active - initial_class_means[None, :, :]

    signatures = residual.reshape(domains, classes * FEATURE_DIM)
    signature_norm = np.linalg.norm(signatures, axis=1, keepdims=True)
    if bool(np.any(signature_norm <= EPSILON)):
        raise D96RACGSRDAError("D96 ground domain signature has zero energy")
    signatures = signatures / signature_norm
    cosine = np.clip(signatures @ signatures.T, -1.0, 1.0)
    density = np.sum(np.exp(-(1.0 - cosine) / float(config.tau)), axis=1)
    if not np.isfinite(density).all() or bool(np.any(density <= EPSILON)):
        raise D96RACGSRDAError("D96 ground density is not finite and positive")
    weights = 1.0 / density
    weights /= np.sum(weights)

    weighted_class_means_raw = np.einsum("d,dcz->cz", weights, active)
    class_means = _normalized_rows(weighted_class_means_raw, "D96 ground class means")
    residual = active - weighted_class_means_raw[None, :, :]

    centered_signatures = signatures - np.sum(weights[:, None] * signatures, axis=0)
    weighted_signatures = np.sqrt(weights)[:, None] * centered_signatures
    diversity_eigenvalues = np.linalg.eigvalsh(weighted_signatures @ weighted_signatures.T)
    diversity_eigenvalues = diversity_eigenvalues[diversity_eigenvalues > EPSILON]
    if len(diversity_eigenvalues) == 0:
        effective_domains = 1.0
    else:
        effective_domains = float(
            np.square(np.sum(diversity_eigenvalues))
            / np.sum(np.square(diversity_eigenvalues))
        )

    covariance = np.einsum("d,dcz,dcw->zw", weights, residual, residual)
    covariance /= float(classes)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    positive = np.flatnonzero(eigenvalues > EPSILON)
    diversity_rank = max(0, int(math.floor(effective_domains)) - 1)
    retained_rank = min(int(config.max_rank), diversity_rank, len(positive), domains - 1)
    if retained_rank:
        order = positive[np.argsort(eigenvalues[positive], kind="stable")[-retained_rank:][::-1]]
        retained_values = eigenvalues[order]
        retained_basis = eigenvectors[:, order]
    else:
        retained_values = np.empty(0, dtype=np.float64)
        retained_basis = np.empty((FEATURE_DIM, 0), dtype=np.float64)
    reconstructed = (
        retained_basis @ np.diag(retained_values) @ retained_basis.T
        if retained_rank
        else np.zeros((FEATURE_DIM, FEATURE_DIM), dtype=np.float64)
    )
    minimum = float(np.min(np.linalg.eigvalsh(0.5 * (reconstructed + reconstructed.T))))
    if not np.isfinite(covariance).all() or minimum < -1.0e-10:
        raise D96RACGSRDAError("D96 low-rank ground covariance is not PSD")

    audit = {
        "schema": "cvs.phase2.d96.ra_cgsrda_ground.v1",
        "phase1_locked_config": True,
        "config_lock_digest": config.lock_digest,
        "phase1_receipt_sha256": config.phase1_receipt_sha256,
        "phase1_locked_tau": float(config.tau),
        "phase1_locked_ridge": float(config.ridge),
        "phase1_locked_temp_base": float(config.temp_base),
        "phase1_locked_temp_aux": float(config.temp_aux),
        "ground_registry_domain_count": int(prototypes.shape[0]),
        "ground_active_domain_count": domains,
        "ground_class_count": classes,
        "ground_component_input_count": int(np.sum(mask)),
        "density_weight_min": float(np.min(weights)),
        "density_weight_max": float(np.max(weights)),
        "effective_domain_count": effective_domains,
        "adaptive_rank_policy": "min(max_rank,floor(D_eff)-1,positive_rank,domain_count-1)",
        "max_rank": int(config.max_rank),
        "retained_rank": retained_rank,
        "low_rank_covariance_trace": float(np.sum(retained_values)),
        "low_rank_covariance_min_eigenvalue": minimum,
        "ground_class_score_access": False,
        "ground_component_update_access": False,
        "ground_sample_or_member_access": False,
        "query_rows_used": 0,
    }
    return GroundDomainModel(
        class_means_fp32=_readonly(class_means, np.float32),
        nuisance_basis_fp32=_readonly(retained_basis, np.float32),
        nuisance_eigenvalues_fp32=_readonly(retained_values, np.float32),
        density_weights_fp32=_readonly(weights, np.float32),
        effective_domain_count=effective_domains,
        config_lock_digest=config.lock_digest,
        audit=audit,
    )


def target_shift_coverage(
    ground: GroundDomainModel,
    support_rows: np.ndarray,
    support_labels: np.ndarray,
    ground_target_labels: np.ndarray,
) -> float:
    """Measure old-support shift energy covered by the ground nuisance span."""

    support = _normalized_rows(support_rows, "D96 old support")
    labels = np.asarray(support_labels, dtype=np.int64)
    mapping = np.asarray(ground_target_labels, dtype=np.int64)
    if labels.shape != (len(support),) or mapping.shape != (len(ground.class_means_fp32),):
        raise D96RACGSRDAError("D96 old-support label mapping shape drift")
    if len(np.unique(mapping)) != len(mapping):
        raise D96RACGSRDAError("D96 ground-to-target label mapping must be one-to-one")
    target_centers = []
    for label in mapping:
        rows = support[labels == int(label)]
        if len(rows) == 0:
            raise D96RACGSRDAError("D96 old support is missing a ground-mapped class")
        target_centers.append(_normalized_rows(rows.mean(axis=0, keepdims=True), "D96 old center")[0])
    delta = np.stack(target_centers) - np.asarray(ground.class_means_fp32, dtype=np.float64)
    delta -= delta.mean(axis=0, keepdims=True)
    total = float(np.sum(delta * delta))
    basis = np.asarray(ground.nuisance_basis_fp32, dtype=np.float64)
    if total <= EPSILON or basis.shape[1] == 0:
        return 0.0
    projected = (delta @ basis) @ basis.T
    return float(np.clip(np.sum(projected * projected) / total, 0.0, 1.0))


def _woodbury_apply(base: float, factor: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Apply (base*I + factor*factor.T)^-1 without a dense inverse."""

    if not math.isfinite(base) or base <= 0.0:
        raise D96RACGSRDAError("D96 Woodbury base must be positive")
    rhs = np.asarray(values, dtype=np.float64)
    if factor.shape[1] == 0:
        return rhs / base
    gram = np.eye(factor.shape[1], dtype=np.float64) + factor.T @ factor / base
    try:
        correction = np.linalg.solve(gram, factor.T @ rhs)
    except np.linalg.LinAlgError as error:
        raise D96RACGSRDAError("D96 Woodbury system is singular") from error
    result = rhs / base - factor @ correction / (base * base)
    if not np.isfinite(result).all():
        raise D96RACGSRDAError("D96 Woodbury precision produced non-finite values")
    return result


def fit_srda(
    ground: GroundDomainModel,
    support_rows: np.ndarray,
    support_labels: np.ndarray,
    ground_target_labels: np.ndarray,
    config: Phase1LockedConfig,
) -> SRDAState:
    """Fit target-only means and a coverage-gated shared SRDA precision."""

    if ground.config_lock_digest != config.lock_digest:
        raise D96RACGSRDAError("D96 Phase1 config differs between ground build and fit")

    support = _normalized_rows(support_rows, "D96 registered support")
    labels = np.asarray(support_labels, dtype=np.int64)
    if labels.shape != (len(support),):
        raise D96RACGSRDAError("D96 registered support label shape drift")
    classes = np.unique(labels)
    if len(classes) < 2:
        raise D96RACGSRDAError("D96 requires at least two registered classes")
    counts = np.asarray([np.sum(labels == label) for label in classes], dtype=np.int64)
    if bool(np.any(counts <= 0)) or len(np.unique(counts)) != 1:
        raise D96RACGSRDAError("D96 requires balanced K-shot registered support")
    k_shot = int(counts[0])
    means = np.stack([support[labels == label].mean(axis=0) for label in classes])
    mean_by_label = {int(label): means[index] for index, label in enumerate(classes)}
    residual = np.stack([row - mean_by_label[int(label)] for row, label in zip(support, labels)])
    class_count = len(classes)
    if k_shot > 1:
        scatter_factor = residual.T / math.sqrt(class_count * (k_shot - 1))
        scatter_trace = float(np.sum(residual * residual) / (class_count * (k_shot - 1)))
        n_residual = class_count * (k_shot - 1)
    else:
        scatter_factor = np.empty((FEATURE_DIM, 0), dtype=np.float64)
        scatter_trace = 0.0
        n_residual = 0
    sigma2 = scatter_trace / FEATURE_DIM if scatter_trace > EPSILON else 1.0 / FEATURE_DIM
    rho = target_shift_coverage(ground, support, labels, ground_target_labels)
    nu_ground = max(0.0, float(ground.effective_domain_count) - 1.0)
    denominator = max(EPSILON, nu_ground + n_residual)
    target_weight = n_residual / denominator
    ground_weight = nu_ground / denominator
    base_variance = float(config.ridge) + ground_weight * (1.0 - rho) * sigma2

    factors = []
    if scatter_factor.shape[1]:
        factors.append(math.sqrt(target_weight) * scatter_factor)
    basis = np.asarray(ground.nuisance_basis_fp32, dtype=np.float64)
    eigenvalues = np.asarray(ground.nuisance_eigenvalues_fp32, dtype=np.float64)
    if basis.shape[1] and rho > 0.0 and ground_weight > 0.0:
        trace_scale = FEATURE_DIM * sigma2 / max(EPSILON, float(np.sum(eigenvalues)))
        ground_scale = ground_weight * rho * trace_scale * eigenvalues
        factors.append(basis * np.sqrt(ground_scale)[None, :])
    factor = np.concatenate(factors, axis=1) if factors else np.empty((FEATURE_DIM, 0))

    precision_means = _woodbury_apply(base_variance, factor, means.T).T
    priors = np.full(class_count, 1.0 / class_count, dtype=np.float64)
    intercept = -0.5 * np.sum(means * precision_means, axis=1) + np.log(priors)
    factor_norm = float(np.linalg.norm(factor, ord=2)) if factor.shape[1] else 0.0
    min_eigenvalue = base_variance
    max_eigenvalue_bound = base_variance + factor_norm * factor_norm
    if min_eigenvalue <= 0.0 or not np.isfinite(intercept).all():
        raise D96RACGSRDAError("D96 posterior covariance/affine head is invalid")

    compiled_fp32_bytes = int(
        precision_means.astype(np.float32).nbytes
        + intercept.astype(np.float32).nbytes
        + means.astype(np.float32).nbytes
    )
    query_macs = class_count * FEATURE_DIM + class_count
    factor_columns = int(factor.shape[1])
    fit_macs = int(
        len(support) * FEATURE_DIM
        + FEATURE_DIM * factor_columns * factor_columns
        + factor_columns**3
        + 2 * class_count * FEATURE_DIM * factor_columns
        + class_count * FEATURE_DIM
    )
    audit = {
        "schema": "cvs.phase2.d96.ra_cgsrda_fit.v1",
        "phase1_locked_config": True,
        "target_class_mean_source": "registered_target_support_only_all_classes_same_formula",
        "ground_class_mean_used_for_score": False,
        "ground_covariance_prior_used": bool(basis.shape[1] and rho > 0.0),
        "class_count": class_count,
        "k_shot": k_shot,
        "coverage_rho": rho,
        "effective_domain_count": float(ground.effective_domain_count),
        "ground_retained_rank": int(basis.shape[1]),
        "support_residual_rank_upper_bound": int(scatter_factor.shape[1]),
        "woodbury_factor_columns": factor_columns,
        "posterior_covariance_min_eigenvalue": min_eigenvalue,
        "posterior_covariance_max_eigenvalue_upper_bound": max_eigenvalue_bound,
        "posterior_covariance_condition_upper_bound": max_eigenvalue_bound / min_eigenvalue,
        "trainable_parameter_count": 0,
        "optimizer_steps": 0,
        "compiled_aux_state_dtype": "fp32",
        "compiled_aux_fp32_state_bytes": compiled_fp32_bytes,
        "query_extra_mac_upper_bound": query_macs,
        "adaptation_mac_upper_bound": fit_macs,
        "persistent_query_batch_state_bytes": 0,
        "query_dependent_batch_optimization": False,
        "query_rows_used_for_fit": 0,
        "old_new_role_specific_scoring_formula": False,
    }
    return SRDAState(
        classes=tuple(int(value) for value in classes),
        coefficient_fp32=_readonly(precision_means, np.float32),
        intercept_fp32=_readonly(intercept, np.float32),
        target_means_fp32=_readonly(means, np.float32),
        coverage_rho=rho,
        k_shot=k_shot,
        config_lock_digest=config.lock_digest,
        audit=audit,
    )


def fuse_base_srda_logits(
    base_logits: np.ndarray,
    srda_logits: np.ndarray,
    *,
    base_classes: tuple[int, ...],
    srda_state: SRDAState,
    support_cv_reliability: float,
    reliability_source: str,
    support_cv_receipt_sha256: str | None,
    config: Phase1LockedConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one row-global residual weight w=rho*q, with exact safe fallback."""

    base = np.asarray(base_logits)
    auxiliary = np.asarray(srda_logits)
    rho, quality = float(srda_state.coverage_rho), float(support_cv_reliability)
    if (
        base.shape != auxiliary.shape
        or base.ndim != 2
        or not np.isfinite(base).all()
        or not np.isfinite(auxiliary).all()
        or not math.isfinite(rho)
        or not math.isfinite(quality)
        or not 0.0 <= rho <= 1.0
        or not 0.0 <= quality <= 1.0
        or tuple(int(value) for value in base_classes) != srda_state.classes
        or base.shape[1] != len(srda_state.classes)
        or srda_state.config_lock_digest != config.lock_digest
        or reliability_source not in {"zero_fallback", "support_crossfit_phase1_smoothed"}
        or (quality > 0.0 and reliability_source != "support_crossfit_phase1_smoothed")
        or (quality == 0.0 and reliability_source != "zero_fallback")
        or (
            quality > 0.0
            and (
                support_cv_receipt_sha256 is None
                or len(support_cv_receipt_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in support_cv_receipt_sha256
                )
            )
        )
        or (quality == 0.0 and support_cv_receipt_sha256 is not None)
    ):
        raise D96RACGSRDAError("D96 fusion input or row-global reliability drift")
    forced_k1 = int(srda_state.k_shot) == 1
    weight = 0.0 if forced_k1 else rho * quality
    audit = {
        "schema": "cvs.phase2.d96.ra_cgsrda_fusion.v1",
        "fusion_formula": "(1-w)*base/temp_base + w*aux/temp_aux",
        "coverage_rho": rho,
        "support_cv_reliability": quality,
        "reliability_source": reliability_source,
        "support_cv_receipt_sha256": support_cv_receipt_sha256,
        "row_global_weight": weight,
        "k1_forced_base_fallback": forced_k1,
        "w0_exact_base_fallback": bool(weight == 0.0),
        "query_batch_state_used": False,
        "class_or_query_specific_weight": False,
        "phase1_locked_temperatures": True,
    }
    if weight == 0.0:
        return base, audit
    fused64 = (
        (1.0 - weight) * np.asarray(base, dtype=np.float64) / float(config.temp_base)
        + weight * np.asarray(auxiliary, dtype=np.float64) / float(config.temp_aux)
    )
    if not np.isfinite(fused64).all():
        raise D96RACGSRDAError("D96 fusion produced non-finite logits")
    return np.asarray(fused64, dtype=np.result_type(base.dtype, np.float32)), audit


__all__ = [
    "D96RACGSRDAError",
    "FEATURE_DIM",
    "GroundDomainModel",
    "Phase1LockedConfig",
    "SRDAState",
    "build_ground_domain_model",
    "fit_srda",
    "fuse_base_srda_logits",
    "target_shift_coverage",
]
