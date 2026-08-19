"""ERBT-IDR Module 2.3 RFGuard support-only adaptation head.

The module deliberately separates three kinds of state:

* identity and FFT features remain the discriminative backbone;
* RF-lite is a ten-dimensional, gain-normalised residual cue;
* RF quality only controls support influence and is never a classifier input.

All fitted state is derived from labelled support and the immutable Phase1
aggregate component.  The fitting surface has no query-side input.  Stage2-B
domain state can therefore be frozen and reused byte-for-byte in Stage2-C.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_ablation_quantization import (
    F3,
    CompiledAffineState,
    compile_affine_state,
    score_affine_state,
)


IDENTITY_DIM = 160
FFT_DIM = 96
IF_DIM = IDENTITY_DIM + FFT_DIM
RF_LITE_DIM = 10
COMPACT_DIM = IF_DIM + RF_LITE_DIM

ARM_RF_QUALITY = "M23-F3-RF-QUALITY"
ARM_RF_LITE_DIAG = "M23-F4-RF-LITE-DIAG"
ARM_RF_LITE_GATED = "M23-F5-RF-LITE-GATED"
SUPPORTED_ARMS = (ARM_RF_QUALITY, ARM_RF_LITE_DIAG, ARM_RF_LITE_GATED)

_EPS = 1.0e-12


class M23RFGuardError(ValueError):
    """Raised when an M2.3 support-only invariant is violated."""


def _readonly(value: Any, dtype: Any = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    if not np.isfinite(result).all():
        raise M23RFGuardError("M2.3 state must be finite")
    result.setflags(write=False)
    return result


def _unit_rows(value: Any) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(rows, axis=-1, keepdims=True)
    return rows / np.maximum(norm, _EPS)


def _canonical_basis(rows: np.ndarray, rank: int) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != IDENTITY_DIM:
        raise M23RFGuardError("basis source must be N x 160")
    if not np.any(np.abs(matrix) > _EPS):
        return np.zeros((IDENTITY_DIM, 0), dtype=np.float64)
    _u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    retained = min(int(rank), int(np.sum(singular > _EPS)))
    if retained <= 0:
        return np.zeros((IDENTITY_DIM, 0), dtype=np.float64)
    basis = vh[:retained].T
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    return basis


def _weighted_center(rows: np.ndarray, quality: np.ndarray) -> np.ndarray:
    weight = np.asarray(quality, dtype=np.float64)
    weight = weight / np.sum(weight)
    center = np.sum(np.asarray(rows, dtype=np.float64) * weight[:, None], axis=0)
    norm = float(np.linalg.norm(center))
    return center / max(norm, _EPS)


def _effective_count(quality: np.ndarray) -> float:
    weight = np.asarray(quality, dtype=np.float64)
    weight = weight / np.sum(weight)
    return float(1.0 / np.sum(np.square(weight)))


def _support_reliability(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    quality: np.ndarray,
) -> np.ndarray:
    """Combine RF quality with a fixed two-pass IF residual reliability."""

    result = np.empty(len(rows), dtype=np.float64)
    for item in classes:
        mask = labels == item
        local_rows = np.asarray(rows[mask, :IF_DIM], dtype=np.float64)
        local_quality = np.asarray(quality[mask], dtype=np.float64)
        initial = _weighted_center(local_rows, local_quality)
        energy = np.sum(np.square(local_rows - initial[None, :]), axis=1)
        positive = energy[energy > _EPS]
        tau = float(np.median(positive)) if len(positive) else 1.0
        tau = max(tau, 1.0e-6)
        result[mask] = local_quality / (1.0 + energy / tau)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise M23RFGuardError("support reliability is invalid")
    return result


def _psd(value: np.ndarray, floor: float = 0.0) -> np.ndarray:
    matrix = 0.5 * (np.asarray(value, dtype=np.float64) + np.asarray(value, dtype=np.float64).T)
    eigenvalue, eigenvector = np.linalg.eigh(matrix)
    clipped = np.maximum(eigenvalue, float(floor))
    result = (eigenvector * clipped[None, :]) @ eigenvector.T
    return 0.5 * (result + result.T)


def _precision(value: np.ndarray) -> tuple[np.ndarray, float]:
    covariance = 0.5 * (value + value.T)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    minimum = float(np.min(eigenvalue))
    if minimum <= 0.0 or not np.isfinite(eigenvalue).all():
        raise M23RFGuardError("M2.3 covariance is not positive definite")
    inverse = (eigenvector * (1.0 / eigenvalue)[None, :]) @ eigenvector.T
    return 0.5 * (inverse + inverse.T), minimum


def _digest_arrays(named: Mapping[str, Any], strings: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(list(strings), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    )
    for name in sorted(named):
        array = np.ascontiguousarray(named[name])
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class M23Config:
    """Fixed conservative controls for the M2.3 head."""

    shared_rank: int = 3
    interaction_rank: int = 2
    interaction_shrinkage: float = 0.35
    out_of_manifold_shrinkage: float = 0.35
    old_prior_strength: float = 2.0
    covariance_shrinkage: float = 0.35
    covariance_floor: float = 2.0e-4
    nuisance_weight: float = 0.25
    identity_weight: float = 1.0
    fft_weight: float = 4.0
    rf_complexity_penalty: float = 0.02
    rf_margin_scale: float = 0.05
    rf_score_weight: float = 0.25
    minimum_quality: float = 0.1

    def __post_init__(self) -> None:
        if self.shared_rank <= 0 or self.interaction_rank <= 0:
            raise M23RFGuardError("manifold ranks must be positive")
        if not 0.0 <= self.interaction_shrinkage <= 1.0:
            raise M23RFGuardError("interaction shrinkage must be in [0, 1]")
        if not 0.0 <= self.out_of_manifold_shrinkage <= 1.0:
            raise M23RFGuardError("out-of-manifold shrinkage must be in [0, 1]")
        if self.old_prior_strength <= 0.0:
            raise M23RFGuardError("old prior strength must be positive")
        if not 0.0 <= self.covariance_shrinkage <= 1.0:
            raise M23RFGuardError("covariance shrinkage must be in [0, 1]")
        if self.covariance_floor <= 0.0 or self.nuisance_weight < 0.0:
            raise M23RFGuardError("covariance controls are invalid")
        if self.identity_weight <= 0.0 or self.fft_weight <= 0.0:
            raise M23RFGuardError("identity/FFT weights must be positive")
        if self.rf_complexity_penalty < 0.0 or self.rf_margin_scale <= 0.0:
            raise M23RFGuardError("RF gate controls are invalid")
        if not 0.0 < self.minimum_quality <= 1.0:
            raise M23RFGuardError("minimum quality must be in (0, 1]")


@dataclass(frozen=True)
class GroundManifold:
    class_registry: tuple[str, ...]
    class_centres: np.ndarray
    shared_basis: np.ndarray
    interaction_basis_by_class: np.ndarray
    loo_source_class_indices: tuple[tuple[int, ...], ...]
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "class_centres", _readonly(self.class_centres))
        object.__setattr__(self, "shared_basis", _readonly(self.shared_basis))
        object.__setattr__(
            self,
            "interaction_basis_by_class",
            _readonly(self.interaction_basis_by_class),
        )
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))


@dataclass(frozen=True)
class Stage2BDomainState:
    class_registry: tuple[str, ...]
    shared_offset: np.ndarray
    class_interaction_offsets: np.ndarray
    target_centres: np.ndarray
    centre_variance: np.ndarray
    nuisance_covariance: np.ndarray
    out_of_manifold_offset_norm: float
    digest: str
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shared_offset", _readonly(self.shared_offset))
        object.__setattr__(
            self,
            "class_interaction_offsets",
            _readonly(self.class_interaction_offsets),
        )
        object.__setattr__(self, "target_centres", _readonly(self.target_centres))
        object.__setattr__(self, "centre_variance", _readonly(self.centre_variance))
        object.__setattr__(
            self, "nuisance_covariance", _readonly(self.nuisance_covariance)
        )
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))


@dataclass(frozen=True)
class M23CenterEstimate:
    """Explicit Module-2 output consumed by the affine head."""

    class_registry: tuple[str, ...]
    centres: np.ndarray
    centre_uncertainty: np.ndarray
    support_weights: np.ndarray
    domain_nuisance_covariance: np.ndarray
    ground_prior_mask: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "centres", _readonly(self.centres))
        object.__setattr__(
            self, "centre_uncertainty", _readonly(self.centre_uncertainty)
        )
        object.__setattr__(self, "support_weights", _readonly(self.support_weights))
        object.__setattr__(
            self,
            "domain_nuisance_covariance",
            _readonly(self.domain_nuisance_covariance),
        )
        mask = np.array(self.ground_prior_mask, dtype=bool, copy=True, order="C")
        mask.setflags(write=False)
        object.__setattr__(self, "ground_prior_mask", mask)


@dataclass(frozen=True)
class M23RFGuardState:
    classes: tuple[str, ...]
    compiled_affine_state: CompiledAffineState
    domain_state: Stage2BDomainState
    center_estimate: M23CenterEstimate
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))

    def _features(self, value: Any) -> np.ndarray:
        rows = np.asarray(value, dtype=np.float32)
        squeezed = rows.ndim == 1
        if squeezed:
            rows = rows[None, :]
        if rows.ndim != 2 or rows.shape[0] <= 0 or rows.shape[1] < self.compiled_affine_state.feature_dim:
            raise M23RFGuardError("scoring blocks have incompatible shape")
        rows = rows[:, : self.compiled_affine_state.feature_dim]
        if not np.isfinite(rows).all():
            raise M23RFGuardError("scoring blocks must be finite")
        return rows[0] if squeezed else rows

    def score(self, value: Any) -> np.ndarray:
        return score_affine_state(self.compiled_affine_state, self._features(value))


def _retained_state_bytes(
    compiled: CompiledAffineState,
    domain_state: Stage2BDomainState,
    center_estimate: M23CenterEstimate,
) -> int:
    """Count every ndarray retained by the fit result without hiding audit state."""

    arrays = (
        domain_state.shared_offset,
        domain_state.class_interaction_offsets,
        domain_state.target_centres,
        domain_state.centre_variance,
        domain_state.nuisance_covariance,
        center_estimate.centres,
        center_estimate.centre_uncertainty,
        center_estimate.support_weights,
        center_estimate.domain_nuisance_covariance,
        center_estimate.ground_prior_mask,
    )
    return int(compiled.state_bytes + sum(value.nbytes for value in arrays))


def extract_rf_lite_quality(received_iq: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract ten scale-free RF-lite coordinates plus fixed quality weights."""

    iq = np.asarray(received_iq, dtype=np.float64)
    if iq.ndim == 2 and iq.shape[0] == 2:
        iq = iq[None, :, :]
    if iq.ndim != 3 or iq.shape[1] != 2 or iq.shape[2] < 16:
        raise M23RFGuardError("received IQ must have shape N x 2 x L with L >= 16")
    if not np.isfinite(iq).all():
        raise M23RFGuardError("received IQ must be finite")

    complex_rows = iq[:, 0] + 1j * iq[:, 1]
    rms = np.sqrt(np.mean(np.abs(complex_rows) ** 2, axis=1, keepdims=True))
    if np.any(rms <= _EPS):
        raise M23RFGuardError("received IQ cannot be all zero")
    scaled = complex_rows / rms
    centred = scaled - np.mean(scaled, axis=1, keepdims=True)
    power = np.mean(np.abs(centred) ** 2, axis=1)
    safe_power = np.maximum(power, _EPS)

    m20 = np.mean(centred**2, axis=1) / safe_power
    c40 = (
        np.mean(centred**4, axis=1)
        - 3.0 * np.square(np.mean(centred**2, axis=1))
    ) / np.square(safe_power)
    c42 = (
        np.mean(np.abs(centred) ** 4, axis=1)
        - np.abs(np.mean(centred**2, axis=1)) ** 2
        - 2.0 * np.square(power)
    ) / np.square(safe_power)
    amplitude = np.abs(scaled)
    q10, q50, q90 = np.quantile(amplitude, (0.10, 0.50, 0.90), axis=1)
    ratio_hi = np.log1p(q90 / np.maximum(q50, _EPS))
    ratio_lo = np.log1p(q50 / np.maximum(q10, _EPS))
    energy = np.mean(np.abs(scaled) ** 2, axis=1)
    r1 = np.abs(np.mean(scaled[:, 1:] * np.conj(scaled[:, :-1]), axis=1))
    r2 = np.abs(np.mean(scaled[:, 2:] * np.conj(scaled[:, :-2]), axis=1))

    lite = np.stack(
        [
            m20.real,
            m20.imag,
            np.abs(m20),
            c40.real,
            c40.imag,
            c42.real,
            ratio_hi,
            ratio_lo,
            r1 / np.maximum(energy, _EPS),
            r2 / np.maximum(energy, _EPS),
        ],
        axis=1,
    ).astype(np.float32)

    dc_ratio = np.abs(np.mean(scaled, axis=1))
    real_var = np.var(scaled.real, axis=1)
    imag_var = np.var(scaled.imag, axis=1)
    imbalance = np.abs(real_var - imag_var) / np.maximum(real_var + imag_var, _EPS)
    peak_ratio = np.quantile(amplitude, 0.995, axis=1) / np.maximum(q50, _EPS)
    tail_penalty = np.maximum(peak_ratio - 1.8, 0.0)
    real_peak = np.max(np.abs(scaled.real), axis=1, keepdims=True)
    imag_peak = np.max(np.abs(scaled.imag), axis=1, keepdims=True)
    rail_fraction = 0.5 * (
        np.mean(np.isclose(np.abs(scaled.real), real_peak, rtol=2e-3, atol=2e-3), axis=1)
        + np.mean(np.isclose(np.abs(scaled.imag), imag_peak, rtol=2e-3, atol=2e-3), axis=1)
    )
    high_order = np.maximum(np.abs(c42.real) - 2.5, 0.0)
    penalty = (
        3.0 * dc_ratio
        + 0.8 * imbalance
        + 0.7 * tail_penalty
        + 3.0 * np.maximum(rail_fraction - 0.03, 0.0)
        + 0.15 * high_order
    )
    quality = np.clip(np.exp(-penalty), 0.1, 1.0).astype(np.float32)
    if not np.isfinite(lite).all() or not np.isfinite(quality).all():
        raise M23RFGuardError("RF-lite extraction produced non-finite state")
    return lite, quality


def build_rfguard_blocks(legacy_features: Any, rf_lite: Any) -> np.ndarray:
    """Recover independent unit identity/FFT blocks and append true RF-lite."""

    legacy = np.asarray(legacy_features, dtype=np.float64)
    lite = np.asarray(rf_lite, dtype=np.float64)
    if legacy.ndim != 2 or legacy.shape[1] != 288 or lite.shape != (len(legacy), RF_LITE_DIM):
        raise M23RFGuardError("legacy/RF-lite feature shape drift")
    if not np.isfinite(legacy).all() or not np.isfinite(lite).all():
        raise M23RFGuardError("feature blocks must be finite")
    identity = _unit_rows(legacy[:, :IDENTITY_DIM])
    fft = _unit_rows(legacy[:, IDENTITY_DIM:IF_DIM])
    rf = _unit_rows(lite)
    if np.any(np.linalg.norm(identity, axis=1) <= _EPS) or np.any(np.linalg.norm(fft, axis=1) <= _EPS) or np.any(np.linalg.norm(rf, axis=1) <= _EPS):
        raise M23RFGuardError("feature block cannot be zero")
    return np.concatenate([identity, fft, rf], axis=1).astype(np.float32)


def build_ground_manifold(component: Any, *, config: M23Config = M23Config()) -> GroundManifold:
    """Reconstruct the aggregate domain-class bank transiently and fit class-LOO bases."""

    domains = tuple(str(value) for value in component.domain_registry)
    classes = tuple(str(value) for value in component.class_registry)
    if len(domains) < 2 or len(classes) < 2 or len(set(domains)) != len(domains) or len(set(classes)) != len(classes):
        raise M23RFGuardError("ground component registry drift")
    dense = np.stack([np.asarray(component.reconstruct_domain(item), dtype=np.float64) for item in domains])
    if dense.shape != (len(domains), len(classes), IDENTITY_DIM) or not np.isfinite(dense).all():
        raise M23RFGuardError("ground domain-class component shape drift")
    dense = _unit_rows(dense)
    class_centres = _unit_rows(np.mean(dense, axis=0))
    domain_delta = dense - class_centres[None, :, :]
    shared_by_domain = np.mean(domain_delta, axis=1)
    shared_basis = _canonical_basis(shared_by_domain, config.shared_rank)
    if shared_basis.shape[1]:
        shared_projection = np.einsum(
            "dcp,pr,qr->dcq",
            domain_delta,
            shared_basis,
            shared_basis,
            optimize=True,
        )
    else:
        shared_projection = np.zeros_like(domain_delta)
    interaction = domain_delta - shared_projection
    loo_indices: list[tuple[int, ...]] = []
    bases: list[np.ndarray] = []
    rank = min(config.interaction_rank, max(1, (len(classes) - 1) * len(domains)))
    for class_index in range(len(classes)):
        source = tuple(index for index in range(len(classes)) if index != class_index)
        loo_indices.append(source)
        basis = _canonical_basis(interaction[:, source, :].reshape(-1, IDENTITY_DIM), rank)
        padded = np.zeros((IDENTITY_DIM, rank), dtype=np.float64)
        padded[:, : basis.shape[1]] = basis
        bases.append(padded)
    resource = dict(component.resource_audit()) if hasattr(component, "resource_audit") else {}
    audit = {
        "schema": "cvs.erbt_idr.m23.ground_manifold.v1",
        "domain_count": len(domains),
        "class_count": len(classes),
        "domain_class_cell_count": len(domains) * len(classes),
        "shared_rank": int(shared_basis.shape[1]),
        "interaction_rank": int(rank),
        "class_loo_interaction_enabled": True,
        "phase1_component_resource_audit": resource,
    }
    return GroundManifold(
        class_registry=classes,
        class_centres=class_centres,
        shared_basis=shared_basis,
        interaction_basis_by_class=np.stack(bases),
        loo_source_class_indices=tuple(loo_indices),
        audit=audit,
    )


def _support_input(
    blocks: Any,
    labels: Any,
    classes: Sequence[str],
    quality: Any,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray, int]:
    rows = np.asarray(blocks, dtype=np.float64)
    y = np.asarray(labels).astype(str)
    registry = tuple(str(value) for value in classes)
    q = np.asarray(quality, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] <= 0 or rows.shape[1] < IF_DIM or not np.isfinite(rows).all():
        raise M23RFGuardError(f"{name} blocks must be finite N x D with D >= 256")
    if y.shape != (len(rows),) or q.shape != (len(rows),):
        raise M23RFGuardError(f"{name} support columns have inconsistent rows")
    if not registry or len(set(registry)) != len(registry) or set(y.tolist()) != set(registry):
        raise M23RFGuardError(f"{name} class registry drift")
    if not np.isfinite(q).all() or np.any(q <= 0.0) or np.any(q > 1.0):
        raise M23RFGuardError(f"{name} quality must be in (0, 1]")
    counts = [int(np.sum(y == item)) for item in registry]
    if min(counts) <= 0 or len(set(counts)) != 1:
        raise M23RFGuardError(f"{name} must be balanced K-shot support")
    return np.array(rows, copy=True), np.array(y, copy=True), registry, np.array(q, copy=True), counts[0]


def estimate_stage2b_domain_state(
    old_support_blocks: Any,
    old_support_labels: Any,
    old_classes: Sequence[str],
    old_support_quality: Any,
    ground_manifold: GroundManifold,
    *,
    config: M23Config = M23Config(),
) -> Stage2BDomainState:
    """Estimate the reusable Stage2-B target-domain state from old support only."""

    rows, labels, classes, quality, k_shot = _support_input(
        old_support_blocks,
        old_support_labels,
        old_classes,
        old_support_quality,
        name="old",
    )
    if classes != ground_manifold.class_registry:
        raise M23RFGuardError("old support and ground component registries differ")
    reliability = _support_reliability(rows, labels, classes, quality)
    observed = np.stack(
        [
            _weighted_center(
                rows[labels == item, :IDENTITY_DIM], reliability[labels == item]
            )
            for item in classes
        ]
    )
    raw_delta = observed - ground_manifold.class_centres
    mean_delta = np.mean(raw_delta, axis=0)
    basis = ground_manifold.shared_basis
    projected = basis @ (basis.T @ mean_delta) if basis.shape[1] else np.zeros_like(mean_delta)
    out_of_manifold = mean_delta - projected
    shared_offset = projected + config.out_of_manifold_shrinkage * out_of_manifold
    class_delta = raw_delta - mean_delta[None, :]
    class_interaction_offsets = []
    for class_index in range(len(classes)):
        class_basis = ground_manifold.interaction_basis_by_class[class_index]
        interaction = class_basis @ (class_basis.T @ class_delta[class_index])
        class_interaction_offsets.append(config.interaction_shrinkage * interaction)
    class_interaction = np.stack(class_interaction_offsets)
    transported = _unit_rows(
        ground_manifold.class_centres
        + shared_offset[None, :]
        + class_interaction
    )

    target_centres: list[np.ndarray] = []
    centre_variance: list[np.ndarray] = []
    for class_index, item in enumerate(classes):
        mask = labels == item
        local_q = reliability[mask]
        effective = _effective_count(local_q)
        support_center = observed[class_index]
        prior = transported[class_index]
        posterior = _unit_rows(
            config.old_prior_strength * prior + effective * support_center
        )
        target_centres.append(posterior)
        residual = rows[mask, :IDENTITY_DIM] - support_center[None, :]
        local_weight = local_q / np.sum(local_q)
        sample_variance = np.sum(local_weight[:, None] * np.square(residual), axis=0)
        mismatch = np.square(support_center - prior)
        centre_variance.append(
            np.maximum(
                (sample_variance + config.old_prior_strength * mismatch)
                / max(effective + config.old_prior_strength, 1.0),
                config.covariance_floor / max(effective, 1.0),
            )
        )
    target = np.stack(target_centres)
    variance = np.stack(centre_variance)

    residual_rows: list[np.ndarray] = []
    residual_weights: list[float] = []
    for class_index, item in enumerate(classes):
        mask = labels == item
        local_q = reliability[mask]
        local_q = local_q / np.sum(local_q) / len(classes)
        residual_rows.extend(rows[mask, :IDENTITY_DIM] - target[class_index][None, :])
        residual_weights.extend(local_q.tolist())
    residual_matrix = np.asarray(residual_rows, dtype=np.float64)
    weight = np.asarray(residual_weights, dtype=np.float64)
    nuisance = (residual_matrix * weight[:, None]).T @ residual_matrix
    diagonal = np.diag(np.diag(nuisance))
    nuisance = _psd(
        (1.0 - config.covariance_shrinkage) * nuisance
        + config.covariance_shrinkage * diagonal,
        floor=0.0,
    )
    digest = _digest_arrays(
        {
            "shared_offset": shared_offset.astype(np.float64),
            "class_interaction_offsets": class_interaction.astype(np.float64),
            "target_centres": target.astype(np.float64),
            "centre_variance": variance.astype(np.float64),
            "nuisance_covariance": nuisance.astype(np.float64),
        },
        classes,
    )
    audit = {
        "schema": "cvs.erbt_idr.m23.stage2b_domain_state.v2",
        "k_shot": int(k_shot),
        "class_loo_interaction_applied": True,
        "class_loo_interaction_shrinkage": float(config.interaction_shrinkage),
        "class_loo_interaction_offset_norm": float(
            np.linalg.norm(class_interaction)
        ),
        "out_of_manifold_offset_norm": float(np.linalg.norm(out_of_manifold)),
        "retained_out_of_manifold_offset_norm": float(
            config.out_of_manifold_shrinkage * np.linalg.norm(out_of_manifold)
        ),
        "nuisance_covariance_min_eigenvalue": float(
            np.min(np.linalg.eigvalsh(nuisance))
        ),
        "digest": digest,
    }
    return Stage2BDomainState(
        class_registry=classes,
        shared_offset=shared_offset,
        class_interaction_offsets=class_interaction,
        target_centres=target,
        centre_variance=variance,
        nuisance_covariance=nuisance,
        out_of_manifold_offset_norm=float(np.linalg.norm(out_of_manifold)),
        digest=digest,
        audit=audit,
    )


def _class_support_statistics(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    quality: np.ndarray,
    domain_state: Stage2BDomainState,
    old_count: int,
    config: M23Config,
) -> tuple[np.ndarray, np.ndarray]:
    centres: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    for class_index, item in enumerate(classes):
        mask = labels == item
        local_rows = rows[mask]
        local_quality = quality[mask]
        effective = _effective_count(local_quality)
        center = _weighted_center(local_rows, local_quality)
        if class_index < old_count:
            center[:IDENTITY_DIM] = domain_state.target_centres[class_index]
        center[:IDENTITY_DIM] = _unit_rows(center[:IDENTITY_DIM])
        center[IDENTITY_DIM:IF_DIM] = _unit_rows(center[IDENTITY_DIM:IF_DIM])
        if rows.shape[1] >= COMPACT_DIM:
            center[IF_DIM:COMPACT_DIM] = _unit_rows(center[IF_DIM:COMPACT_DIM])
        centres.append(center)
        local_weight = local_quality / np.sum(local_quality)
        residual = local_rows - center[None, :]
        variance = np.sum(local_weight[:, None] * np.square(residual), axis=0) / max(effective, 1.0)
        variance = np.maximum(variance, config.covariance_floor / max(effective, 1.0))
        if class_index < old_count:
            variance[:IDENTITY_DIM] = np.maximum(
                variance[:IDENTITY_DIM], domain_state.centre_variance[class_index]
            )
        variances.append(variance)
    return np.stack(centres), np.stack(variances)


def _balanced_weights(
    labels: np.ndarray,
    classes: tuple[str, ...],
    quality: np.ndarray,
    old_count: int,
) -> np.ndarray:
    result = np.zeros(len(labels), dtype=np.float64)
    has_new = old_count < len(classes)
    old_mass = 0.5 if has_new else 1.0
    new_mass = 0.5 if has_new else 0.0
    for index, item in enumerate(classes):
        mask = labels == item
        local = quality[mask]
        if index < old_count:
            class_mass = old_mass / old_count
        else:
            class_mass = new_mass / (len(classes) - old_count)
        result[mask] = class_mass * local / np.sum(local)
    return result


def _loo_rf_gate(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    quality: np.ndarray,
    k_shot: int,
    config: M23Config,
) -> tuple[np.ndarray, int, Mapping[str, Any]]:
    class_index = {item: index for index, item in enumerate(classes)}
    base_margin_delta = np.zeros(len(rows), dtype=np.float64)
    helps = np.zeros(len(classes), dtype=np.int64)
    harms = np.zeros(len(classes), dtype=np.int64)
    for row_index, row in enumerate(rows):
        base_centres = []
        rf_centres = []
        for item in classes:
            mask = labels == item
            if item == labels[row_index]:
                mask = np.array(mask, copy=True)
                mask[row_index] = False
            if not np.any(mask):
                base_centres.append(np.zeros(IF_DIM, dtype=np.float64))
                rf_centres.append(np.zeros(RF_LITE_DIM, dtype=np.float64))
                continue
            base_centres.append(_weighted_center(rows[mask, :IF_DIM], quality[mask]))
            rf_centres.append(_weighted_center(rows[mask, IF_DIM:COMPACT_DIM], quality[mask]))
        base_centres_array = np.stack(base_centres)
        rf_centres_array = np.stack(rf_centres)
        base_score = base_centres_array @ _unit_rows(row[:IF_DIM])
        rf_score = base_score + config.rf_score_weight * (
            rf_centres_array @ _unit_rows(row[IF_DIM:COMPACT_DIM])
        )
        target = class_index[str(labels[row_index])]
        other = np.arange(len(classes)) != target
        base_margin = base_score[target] - np.max(base_score[other])
        augmented_margin = rf_score[target] - np.max(rf_score[other])
        base_margin_delta[row_index] = augmented_margin - base_margin
        base_correct = int(np.argmax(base_score)) == target
        augmented_correct = int(np.argmax(rf_score)) == target
        helps[target] += int((not base_correct) and augmented_correct)
        harms[target] += int(base_correct and (not augmented_correct))

    complexity = config.rf_complexity_penalty * RF_LITE_DIM / IF_DIM
    utility = np.zeros(len(classes), dtype=np.float64)
    for index, item in enumerate(classes):
        mask = labels == item
        local = quality[mask] / np.sum(quality[mask])
        margin_gain = float(np.sum(local * base_margin_delta[mask]))
        correction_gain = float(helps[index] - harms[index]) / max(int(np.sum(mask)), 1)
        utility[index] = margin_gain + 0.1 * correction_gain - complexity
    global_help = int(np.sum(helps))
    global_harm = int(np.sum(harms))
    if k_shot < 10:
        global_weight = quality / np.sum(quality)
        global_margin_gain = float(np.sum(global_weight * base_margin_delta))
        global_correction_gain = float(global_help - global_harm) / max(
            len(rows), 1
        )
        global_utility = (
            global_margin_gain + 0.1 * global_correction_gain - complexity
        )
        utility[:] = global_utility
        gate_value = float(
            np.clip(global_utility / config.rf_margin_scale, 0.0, 1.0)
        )
        fallback = 0
        if global_harm > global_help and gate_value > 0.0:
            gate_value = 0.0
            fallback = 1
        gate = np.full(len(classes), gate_value, dtype=np.float64)
        mode = "global_support_loo"
    else:
        mode = "class_support_loo"
        gate = np.clip(utility / config.rf_margin_scale, 0.0, 1.0)
        fallback = 0
        for index in range(len(classes)):
            if harms[index] > helps[index] and gate[index] > 0.0:
                gate[index] = 0.0
                fallback += 1
    audit = {
        "m23_rf_gate_mode": mode,
        "m23_rf_loo_help_by_class": helps.tolist(),
        "m23_rf_loo_harm_by_class": harms.tolist(),
        "m23_rf_global_help": global_help,
        "m23_rf_global_harm": global_harm,
        "m23_rf_loo_utility_after_complexity_by_class": utility.tolist(),
        "m23_rf_complexity_penalty": float(complexity),
    }
    return gate, fallback, audit


def fit_rfguard_m23(
    old_support_blocks: Any,
    old_support_labels: Any,
    old_classes: Sequence[str],
    old_support_quality: Any,
    *,
    ground_component: Any,
    new_support_blocks: Any | None = None,
    new_support_labels: Any | None = None,
    new_classes: Sequence[str] = (),
    new_support_quality: Any | None = None,
    arm: str = ARM_RF_LITE_GATED,
    frozen_domain_state: Stage2BDomainState | None = None,
    config: M23Config = M23Config(),
) -> M23RFGuardState:
    """Fit one support-only M2.3 head and compile its persistent F3 state."""

    if arm not in SUPPORTED_ARMS:
        raise M23RFGuardError("unsupported M2.3 arm")
    old_x, old_y, old_registry, old_q, old_k = _support_input(
        old_support_blocks,
        old_support_labels,
        old_classes,
        old_support_quality,
        name="old",
    )
    if old_x.shape[1] < COMPACT_DIM:
        raise M23RFGuardError("M2.3 fitting requires compact 266 support blocks")
    old_x = old_x[:, :COMPACT_DIM]
    new_registry = tuple(str(value) for value in new_classes)
    if new_registry:
        if new_support_blocks is None or new_support_labels is None or new_support_quality is None:
            raise M23RFGuardError("new-class support columns are incomplete")
        new_x, new_y, checked_new, new_q, new_k = _support_input(
            new_support_blocks,
            new_support_labels,
            new_registry,
            new_support_quality,
            name="new",
        )
        if new_x.shape[1] < COMPACT_DIM or new_k != old_k or set(new_registry) & set(old_registry):
            raise M23RFGuardError("new-class support geometry drift")
        new_x = new_x[:, :COMPACT_DIM]
        if checked_new != new_registry:
            raise M23RFGuardError("new-class registry order drift")
        rows = np.concatenate([old_x, new_x], axis=0)
        labels = np.concatenate([old_y, new_y])
        quality = np.concatenate([old_q, new_q])
    else:
        if any(value is not None for value in (new_support_blocks, new_support_labels, new_support_quality)):
            raise M23RFGuardError("new-class support was supplied without a registry")
        rows, labels, quality = old_x, old_y, old_q
    classes = old_registry + new_registry

    manifold = build_ground_manifold(ground_component, config=config)
    if manifold.class_registry != old_registry:
        raise M23RFGuardError("ground component does not match old classes")
    reused = frozen_domain_state is not None
    domain_state = frozen_domain_state or estimate_stage2b_domain_state(
        old_x,
        old_y,
        old_registry,
        old_q,
        manifold,
        config=config,
    )
    if domain_state.class_registry != old_registry:
        raise M23RFGuardError("frozen Stage2-B state registry drift")

    reliability = _support_reliability(rows, labels, classes, quality)
    centres, centre_variance = _class_support_statistics(
        rows,
        labels,
        classes,
        reliability,
        domain_state,
        len(old_registry),
        config,
    )
    sample_weight = _balanced_weights(
        labels, classes, reliability, len(old_registry)
    )
    residual = np.stack(
        [rows[index] - centres[classes.index(str(labels[index]))] for index in range(len(rows))]
    )
    if_covariance = (residual[:, :IF_DIM] * sample_weight[:, None]).T @ residual[:, :IF_DIM]
    if_covariance[:IDENTITY_DIM, :IDENTITY_DIM] += (
        config.nuisance_weight * domain_state.nuisance_covariance
    )
    k_regime: str
    if old_k == 1:
        k_regime = "K1_IF_PROTOTYPE_DIAG"
        if_covariance = np.diag(np.diag(if_covariance))
    elif old_k == 2:
        k_regime = "K2_IF_TASK_DIAG"
        if_covariance = np.diag(np.diag(if_covariance))
    elif old_k < 10:
        k_regime = "K5_IF_FULL_RF_GLOBAL_GATE"
        diagonal = np.diag(np.diag(if_covariance))
        if_covariance = (
            (1.0 - config.covariance_shrinkage) * if_covariance
            + config.covariance_shrinkage * diagonal
        )
    else:
        k_regime = "K10_IF_FULL_RF_CLASS_LOO_GATE"
        diagonal = np.diag(np.diag(if_covariance))
        if_covariance = (
            (1.0 - config.covariance_shrinkage) * if_covariance
            + config.covariance_shrinkage * diagonal
        )
    if_covariance = _psd(if_covariance, floor=config.covariance_floor)
    if_precision, if_minimum = _precision(if_covariance)

    base_centres = centres[:, :IF_DIM]
    base_coefficient = base_centres @ if_precision
    if_metric_weight = np.concatenate(
        [
            np.full(IDENTITY_DIM, config.identity_weight, dtype=np.float64),
            np.full(FFT_DIM, config.fft_weight, dtype=np.float64),
        ]
    )
    base_coefficient = base_coefficient * if_metric_weight[None, :]
    if_precision_diagonal = np.diag(if_precision)
    if_penalty = 0.5 * np.sum(
        centre_variance[:, :IF_DIM]
        * if_precision_diagonal[None, :]
        * if_metric_weight[None, :],
        axis=1,
    )
    base_bias = -0.5 * np.sum(base_centres * base_coefficient, axis=1) - if_penalty

    support_loo = False
    rf_fallback = 0
    gate_audit: Mapping[str, Any] = {}
    if arm == ARM_RF_QUALITY or old_k <= 2:
        gates = np.zeros(len(classes), dtype=np.float64)
    elif arm == ARM_RF_LITE_DIAG:
        gates = np.ones(len(classes), dtype=np.float64)
    else:
        support_loo = True
        gates, rf_fallback, gate_audit = _loo_rf_gate(
            rows, labels, classes, reliability, old_k, config
        )

    use_rf = bool(np.any(gates > 0.0)) or arm == ARM_RF_LITE_DIAG
    rf_minimum = 0.0
    rf_penalty = np.zeros(len(classes), dtype=np.float64)
    if use_rf:
        rf_variance = np.sum(
            sample_weight[:, None] * np.square(residual[:, IF_DIM:COMPACT_DIM]), axis=0
        )
        rf_variance = np.maximum(rf_variance, config.covariance_floor)
        rf_minimum = float(np.min(rf_variance))
        rf_coefficient = centres[:, IF_DIM:COMPACT_DIM] / rf_variance[None, :]
        rf_penalty = 0.5 * np.sum(
            centre_variance[:, IF_DIM:COMPACT_DIM] / rf_variance[None, :], axis=1
        )
        coefficient = np.concatenate(
            [base_coefficient, gates[:, None] * rf_coefficient], axis=1
        )
        rf_bias = -0.5 * np.sum(
            centres[:, IF_DIM:COMPACT_DIM] * rf_coefficient, axis=1
        ) - rf_penalty
        bias = base_bias + gates * rf_bias
        feature_dim = COMPACT_DIM
        block_sizes = (IDENTITY_DIM, FFT_DIM, RF_LITE_DIM)
    else:
        coefficient = base_coefficient
        bias = base_bias
        feature_dim = IF_DIM
        block_sizes = (IDENTITY_DIM, FFT_DIM)

    compiled = compile_affine_state(
        coefficient.astype(np.float32),
        bias.astype(np.float32),
        arm_id=F3,
        block_sizes=block_sizes,
    )
    support_features = rows[:, :feature_dim].astype(np.float32, copy=False)
    reference_support_scores = (
        support_features @ coefficient.astype(np.float32).T
        + bias.astype(np.float32)[None, :]
    )
    compiled_support_scores = score_affine_state(compiled, support_features)
    support_error = np.abs(reference_support_scores - compiled_support_scores)
    support_reference_prediction = np.argmax(reference_support_scores, axis=1)
    support_compiled_prediction = np.argmax(compiled_support_scores, axis=1)
    support_flip_count = int(
        np.sum(support_reference_prediction != support_compiled_prediction)
    )
    covariance_structure = (
        "diagonal_if"
        if old_k <= 2
        else ("if_full_plus_rf_diag" if feature_dim == COMPACT_DIM else "if_full")
    )
    all_penalty = if_penalty + gates * rf_penalty
    total_coefficient_norm = float(np.linalg.norm(coefficient, ord="fro"))
    rf_coefficient_norm = (
        float(np.linalg.norm(coefficient[:, IF_DIM:], ord="fro"))
        if feature_dim == COMPACT_DIM
        else 0.0
    )
    rf_coefficient_ratio = rf_coefficient_norm / max(
        total_coefficient_norm, _EPS
    )
    ground_prior_mask = np.zeros((len(classes), COMPACT_DIM), dtype=bool)
    ground_prior_mask[: len(old_registry), :IDENTITY_DIM] = True
    center_estimate = M23CenterEstimate(
        class_registry=classes,
        centres=centres,
        centre_uncertainty=centre_variance,
        support_weights=sample_weight,
        domain_nuisance_covariance=domain_state.nuisance_covariance,
        ground_prior_mask=ground_prior_mask,
    )
    total_retained_state_bytes = _retained_state_bytes(
        compiled,
        domain_state,
        center_estimate,
    )
    task_weight = (
        {"old": 0.5, "new": 0.5}
        if new_registry
        else {"old": 1.0, "new": 0.0}
    )
    audit = {
        "schema": "cvs.erbt_idr.m23.rfguard_state.v1",
        "arm": arm,
        "m23_k_shot": int(old_k),
        "m23_k_regime": k_regime,
        "m23_support_loo_enabled": bool(support_loo),
        "m23_stage2b_domain_state_reused": bool(reused),
        "m23_stage2b_domain_state_digest": domain_state.digest,
        "m23_center_override_enabled": True,
        "m23_old_class_ground_prior_count": len(old_registry),
        "m23_new_class_ground_prior_count": 0,
        "m23_center_uncertainty_intercept_penalty_by_class": all_penalty.tolist(),
        "m23_rf_gate_by_class": gates.tolist(),
        "m23_rf_no_harm_fallback_count": int(rf_fallback),
        "m23_rf_cross_block_frobenius": 0.0,
        "m23_covariance_structure": covariance_structure,
        "m23_covariance_min_eigenvalue": float(
            min(if_minimum, rf_minimum) if feature_dim == COMPACT_DIM else if_minimum
        ),
        "m23_nuisance_covariance_min_eigenvalue": float(
            np.min(np.linalg.eigvalsh(domain_state.nuisance_covariance))
        ),
        "m23_feature_dim": int(feature_dim),
        "m23_feature_block_offsets": list(compiled.block_offsets),
        "m23_compiled_state_bytes": int(compiled.state_bytes),
        "m23_total_retained_state_bytes": total_retained_state_bytes,
        "m23_has_fp32_coefficient_sidecar": False,
        "m23_quantization_audit_scope": "support_fit_rows_transient",
        "m23_quantization_support_row_count": int(len(support_features)),
        "m23_quantization_support_max_logit_abs_error": float(
            np.max(support_error)
        ),
        "m23_quantization_support_mean_logit_abs_error": float(
            np.mean(support_error)
        ),
        "m23_quantization_support_argmax_flip_count": support_flip_count,
        "m23_quantization_support_prediction_agreement_rate": float(
            1.0 - support_flip_count / len(support_features)
        ),
        "m23_query_head_mac": int(len(classes) * feature_dim),
        "m23_rf_quality_classifier_dimension": 0,
        "m23_identity_weight": float(config.identity_weight),
        "m23_fft_weight": float(config.fft_weight),
        "m23_covariance_task_weight": task_weight,
        "m23_support_reliability_rule": (
            "rf_quality_times_inverse_one_plus_if_residual_over_class_tau"
        ),
        "m23_support_reliability_min": float(np.min(reliability)),
        "m23_support_reliability_max": float(np.max(reliability)),
        "m23_rf_coefficient_frobenius_ratio": float(rf_coefficient_ratio),
        "m23_if_coefficient_frobenius": float(
            np.linalg.norm(coefficient[:, :IF_DIM], ord="fro")
        ),
        "m23_rf_coefficient_frobenius": float(rf_coefficient_norm),
        "m23_ground_manifold_audit": dict(manifold.audit),
        **dict(gate_audit),
    }
    return M23RFGuardState(
        classes=classes,
        compiled_affine_state=compiled,
        domain_state=domain_state,
        center_estimate=center_estimate,
        audit=audit,
    )


__all__ = [
    "ARM_RF_LITE_DIAG",
    "ARM_RF_LITE_GATED",
    "ARM_RF_QUALITY",
    "COMPACT_DIM",
    "FFT_DIM",
    "GroundManifold",
    "IDENTITY_DIM",
    "IF_DIM",
    "M23Config",
    "M23CenterEstimate",
    "M23RFGuardError",
    "M23RFGuardState",
    "RF_LITE_DIM",
    "Stage2BDomainState",
    "build_ground_manifold",
    "build_rfguard_blocks",
    "estimate_stage2b_domain_state",
    "extract_rf_lite_quality",
    "fit_rfguard_m23",
]
