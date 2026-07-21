"""Support-only C-id nuisance metric for the Phase2 causal ablation chain.

This module estimates one class-shared, low-rank nuisance subspace from
class-balanced within-class variation in the current-row ``z_id160`` support.
It then closes that subspace through Patch A's existing typed PSD metric.  The
support bank, Student-t kernel, bandwidth, temperature, quantization and query
scoring formula remain owned by :mod:`stage2_zid_student_t_qknn`.

The estimator intentionally returns the exact Patch A identity metric for
K=1: one physical support sample per class cannot identify within-class target
variation.  It has no query, receiver, role, scenario, ground or source input
surface and performs no target-side hyperparameter search.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np

from cvsrffi.stage2_zid_student_t_qknn import (
    ALLOWED_K,
    Z_DIM,
    Phase1ZIDStudentTLock,
    TypedMetricProvenanceReceipt,
    TypedSharedPSDMetric,
    ZIDStudentTQKNNError,
    build_typed_shared_psd_metric,
    identity_shared_psd_metric,
    normalize_zid_rows,
)


C_ID_LOCK_SCHEMA = "cvs.phase1.zid_support_nuisance_metric.lock.v1"
C_ID_AUDIT_SCHEMA = "cvs.phase2.zid_support_nuisance_metric.audit.v1"
MAX_C_ID_RANK = 2


class ZIDSupportNuisanceMetricError(ZIDStudentTQKNNError):
    """Raised when the C-id support-only fit violates its frozen contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: str, name: str) -> str:
    if type(value) is not str:
        raise ZIDSupportNuisanceMetricError(f"{name} must be an exact string SHA256")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ZIDSupportNuisanceMetricError(f"{name} must be a lowercase SHA256")
    return value


def _finite(value: float, name: str) -> float:
    if type(value) is not float:
        raise ZIDSupportNuisanceMetricError(f"{name} must be an exact float")
    result = value
    if not math.isfinite(result):
        raise ZIDSupportNuisanceMetricError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class Phase1ZIDSupportNuisanceLock:
    """K-specific C-id hyperparameters frozen by Phase1 nested LODO."""

    active_k: int
    max_rank: int
    attenuation: float
    between_guard_weight: float
    minimum_nuisance_fraction: float
    minimum_within_energy: float
    qknn_config_lock_digest: str
    qknn_identity_metric_receipt_sha256: str
    phase1_nested_lodo_receipt_sha256: str
    schema: str = C_ID_LOCK_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != C_ID_LOCK_SCHEMA
            or type(self.active_k) is not int
            or self.active_k not in ALLOWED_K
            or type(self.max_rank) is not int
            or self.max_rank < 1
            or self.max_rank > MAX_C_ID_RANK
        ):
            raise ZIDSupportNuisanceMetricError("C-id active K/rank lock drift")
        attenuation = _finite(self.attenuation, "attenuation")
        guard = _finite(self.between_guard_weight, "between_guard_weight")
        fraction = _finite(self.minimum_nuisance_fraction, "minimum_nuisance_fraction")
        energy = _finite(self.minimum_within_energy, "minimum_within_energy")
        if not 0.0 < attenuation <= 0.75:
            raise ZIDSupportNuisanceMetricError("attenuation must be in (0,0.75]")
        if guard <= 0.0:
            raise ZIDSupportNuisanceMetricError("between guard weight must be positive")
        if not 0.5 <= fraction < 1.0:
            raise ZIDSupportNuisanceMetricError(
                "minimum nuisance fraction must be in [0.5,1)"
            )
        if energy <= 0.0:
            raise ZIDSupportNuisanceMetricError("minimum within energy must be positive")
        _require_sha256(
            self.phase1_nested_lodo_receipt_sha256,
            "phase1_nested_lodo_receipt_sha256",
        )
        _require_sha256(self.qknn_config_lock_digest, "qknn_config_lock_digest")
        _require_sha256(
            self.qknn_identity_metric_receipt_sha256,
            "qknn_identity_metric_receipt_sha256",
        )

    @property
    def lock_digest(self) -> str:
        return _sha256(_canonical_bytes(asdict(self)))


@dataclass(frozen=True, slots=True)
class ZIDSupportNuisanceAudit:
    """Small immutable audit for one support-only analytic fit."""

    active_k: int
    class_count: int
    support_rows_used_for_fit: int
    query_rows_used_for_fit: int
    target_optimizer_steps: int
    effective_rank: int
    fallback_reason: str
    selected_within_energy: tuple[float, ...]
    selected_between_energy: tuple[float, ...]
    selected_nuisance_fraction: tuple[float, ...]
    class_balanced: bool
    classifier_formula_unchanged: bool
    phase1_lock_digest: str
    support_receipt_sha256: str
    metric_receipt_sha256: str
    schema: str = C_ID_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        rank = int(self.effective_rank)
        if (
            self.schema != C_ID_AUDIT_SCHEMA
            or type(self.active_k) is not int
            or self.active_k not in ALLOWED_K
            or type(self.class_count) is not int
            or self.class_count < 2
            or type(self.support_rows_used_for_fit) is not int
            or self.support_rows_used_for_fit != self.active_k * self.class_count
            or self.query_rows_used_for_fit != 0
            or type(self.query_rows_used_for_fit) is not int
            or self.target_optimizer_steps != 0
            or type(self.target_optimizer_steps) is not int
            or type(self.effective_rank) is not int
            or rank < 0
            or rank > MAX_C_ID_RANK
            or type(self.class_balanced) is not bool
            or not self.class_balanced
            or type(self.classifier_formula_unchanged) is not bool
            or not self.classifier_formula_unchanged
            or type(self.fallback_reason) is not str
            or self.fallback_reason
            not in ("none", "k1_unidentifiable_identity", "no_guarded_nuisance_direction")
        ):
            raise ZIDSupportNuisanceMetricError("C-id audit invariant drift")
        for values, name in (
            (self.selected_within_energy, "selected_within_energy"),
            (self.selected_between_energy, "selected_between_energy"),
            (self.selected_nuisance_fraction, "selected_nuisance_fraction"),
        ):
            if type(values) is not tuple or len(values) != rank or any(
                not math.isfinite(float(value)) for value in values
            ):
                raise ZIDSupportNuisanceMetricError(f"{name} rank/finite invariant drift")
        if any(value < 0.0 for value in self.selected_within_energy) or any(
            value < 0.0 for value in self.selected_between_energy
        ):
            raise ZIDSupportNuisanceMetricError("C-id selected energy must be nonnegative")
        if any(
            value < 0.0 or value > 1.0 for value in self.selected_nuisance_fraction
        ):
            raise ZIDSupportNuisanceMetricError("C-id nuisance fraction must be in [0,1]")
        if rank == 0 and self.fallback_reason == "none":
            raise ZIDSupportNuisanceMetricError("rank-zero C-id fit requires a fallback reason")
        if rank > 0 and self.fallback_reason != "none":
            raise ZIDSupportNuisanceMetricError("adapted C-id fit cannot claim a fallback")
        _require_sha256(self.phase1_lock_digest, "phase1_lock_digest")
        _require_sha256(self.support_receipt_sha256, "support_receipt_sha256")
        _require_sha256(self.metric_receipt_sha256, "metric_receipt_sha256")

    @property
    def audit_receipt_sha256(self) -> str:
        return _sha256(_canonical_bytes(asdict(self)))


@dataclass(frozen=True, slots=True)
class FittedZIDSupportNuisanceMetric:
    """Typed C-id result containing the Patch A metric and its fit audit."""

    metric: TypedSharedPSDMetric
    audit: ZIDSupportNuisanceAudit

    def __post_init__(self) -> None:
        if (
            type(self.metric) is not TypedSharedPSDMetric
            or type(self.audit) is not ZIDSupportNuisanceAudit
        ):
            raise ZIDSupportNuisanceMetricError("C-id result requires exact typed states")
        if self.metric.metric_receipt_sha256 != self.audit.metric_receipt_sha256:
            raise ZIDSupportNuisanceMetricError("C-id metric/audit receipt drift")
        if self.metric.effective_rank != self.audit.effective_rank:
            raise ZIDSupportNuisanceMetricError("C-id metric/audit rank drift")


def _registry(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if (
        len(result) < 2
        or len(set(result)) != len(result)
        or any(not value for value in result)
    ):
        raise ZIDSupportNuisanceMetricError(
            "registered classes must contain unique non-empty values"
        )
    return result


def _canonical_group(rows: np.ndarray) -> tuple[bytes, np.ndarray]:
    order = sorted(range(len(rows)), key=lambda index: rows[index].tobytes(order="C"))
    ordered = np.ascontiguousarray(rows[np.asarray(order, dtype=np.int64)], dtype=np.float64)
    return _sha256(ordered.tobytes(order="C")).encode("ascii"), ordered


def _canonical_sign(vector: np.ndarray) -> np.ndarray:
    pivot = int(np.argmax(np.abs(vector)))
    return -vector if vector[pivot] < 0.0 else vector


def _audit(
    *,
    lock: Phase1ZIDSupportNuisanceLock,
    class_count: int,
    metric: TypedSharedPSDMetric,
    support_receipt_sha256: str,
    fallback_reason: str,
    within: Sequence[float],
    between: Sequence[float],
    fraction: Sequence[float],
) -> ZIDSupportNuisanceAudit:
    return ZIDSupportNuisanceAudit(
        active_k=lock.active_k,
        class_count=class_count,
        support_rows_used_for_fit=class_count * lock.active_k,
        query_rows_used_for_fit=0,
        target_optimizer_steps=0,
        effective_rank=metric.effective_rank,
        fallback_reason=fallback_reason,
        selected_within_energy=tuple(float(value) for value in within),
        selected_between_energy=tuple(float(value) for value in between),
        selected_nuisance_fraction=tuple(float(value) for value in fraction),
        class_balanced=True,
        classifier_formula_unchanged=True,
        phase1_lock_digest=lock.lock_digest,
        support_receipt_sha256=support_receipt_sha256,
        metric_receipt_sha256=metric.metric_receipt_sha256,
    )


def fit_zid_support_nuisance_metric(
    support_zid: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    qknn_config: Phase1ZIDStudentTLock,
    nuisance_lock: Phase1ZIDSupportNuisanceLock,
    support_receipt_sha256: str,
) -> FittedZIDSupportNuisanceMetric:
    """Fit one analytic class-shared C-id metric from target support only.

    Hyperparameters are supplied exclusively by ``nuisance_lock``.  The
    function has no optimization loop and no query input.  Labels are used only
    to form equal-weight class residuals; their names and registry order do not
    affect the fitted geometry.
    """

    if type(qknn_config) is not Phase1ZIDStudentTLock:
        raise ZIDSupportNuisanceMetricError("C-id requires an exact Patch A lock")
    if type(nuisance_lock) is not Phase1ZIDSupportNuisanceLock:
        raise ZIDSupportNuisanceMetricError("C-id requires an exact nuisance lock")
    if qknn_config.active_k != nuisance_lock.active_k:
        raise ZIDSupportNuisanceMetricError("Patch A/C-id active K drift")
    if qknn_config.lock_digest != nuisance_lock.qknn_config_lock_digest:
        raise ZIDSupportNuisanceMetricError("C-id lock is not bound to this Patch A config")
    identity_receipt = identity_shared_psd_metric(
        config=qknn_config
    ).metric_receipt_sha256
    if identity_receipt != nuisance_lock.qknn_identity_metric_receipt_sha256:
        raise ZIDSupportNuisanceMetricError(
            "C-id lock is not bound to this Patch A identity metric"
        )
    support_receipt = _require_sha256(support_receipt_sha256, "support_receipt_sha256")
    classes = _registry(registered_classes)
    rows = normalize_zid_rows(np.asarray(support_zid)).astype(np.float64)
    labels = np.asarray([str(value) for value in support_labels], dtype=str)
    if labels.ndim != 1 or len(labels) != len(rows):
        raise ZIDSupportNuisanceMetricError("support labels must align with support rows")
    if set(labels.tolist()) != set(classes):
        raise ZIDSupportNuisanceMetricError(
            "support labels must equal the registered class set"
        )

    groups: list[tuple[bytes, np.ndarray]] = []
    for class_name in classes:
        local = rows[labels == class_name]
        if len(local) != nuisance_lock.active_k:
            raise ZIDSupportNuisanceMetricError(
                "every registered class must contain exactly active_k support rows"
            )
        groups.append(_canonical_group(local))
    groups.sort(key=lambda item: item[0])

    if nuisance_lock.active_k == 1:
        metric = identity_shared_psd_metric(config=qknn_config)
        audit = _audit(
            lock=nuisance_lock,
            class_count=len(classes),
            metric=metric,
            support_receipt_sha256=support_receipt,
            fallback_reason="k1_unidentifiable_identity",
            within=(),
            between=(),
            fraction=(),
        )
        return FittedZIDSupportNuisanceMetric(metric=metric, audit=audit)

    class_means = []
    within_scatter = np.zeros((Z_DIM, Z_DIM), dtype=np.float64)
    for _, local in groups:
        center = np.mean(local, axis=0, dtype=np.float64)
        residual = local - center
        within_scatter += (residual.T @ residual) / float(nuisance_lock.active_k - 1)
        class_means.append(center)
    within_scatter /= float(len(groups))

    centers = np.stack(class_means)
    centered = centers - np.mean(centers, axis=0, dtype=np.float64)
    between_scatter = (centered.T @ centered) / float(len(groups))
    guarded = within_scatter - nuisance_lock.between_guard_weight * between_scatter
    guarded = 0.5 * (guarded + guarded.T)
    eigenvalues, eigenvectors = np.linalg.eigh(guarded)

    selected_basis: list[np.ndarray] = []
    selected_within: list[float] = []
    selected_between: list[float] = []
    selected_fraction: list[float] = []
    for index in np.argsort(eigenvalues)[::-1]:
        if len(selected_basis) >= nuisance_lock.max_rank:
            break
        if float(eigenvalues[index]) <= 0.0:
            break
        vector = _canonical_sign(eigenvectors[:, index])
        within_energy = float(vector @ within_scatter @ vector)
        between_energy = max(float(vector @ between_scatter @ vector), 0.0)
        nuisance_fraction = within_energy / (
            within_energy + between_energy + np.finfo(np.float64).eps
        )
        if (
            within_energy < nuisance_lock.minimum_within_energy
            or nuisance_fraction < nuisance_lock.minimum_nuisance_fraction
        ):
            continue
        selected_basis.append(vector)
        selected_within.append(within_energy)
        selected_between.append(between_energy)
        selected_fraction.append(nuisance_fraction)

    if not selected_basis:
        metric = identity_shared_psd_metric(config=qknn_config)
        audit = _audit(
            lock=nuisance_lock,
            class_count=len(classes),
            metric=metric,
            support_receipt_sha256=support_receipt,
            fallback_reason="no_guarded_nuisance_direction",
            within=(),
            between=(),
            fraction=(),
        )
        return FittedZIDSupportNuisanceMetric(metric=metric, audit=audit)

    basis = np.asarray(selected_basis, dtype=np.float32)
    attenuation = np.full(len(basis), nuisance_lock.attenuation, dtype=np.float32)
    provenance = TypedMetricProvenanceReceipt(
        fit_scope="target_support_only",
        source_receipt_sha256=support_receipt,
        query_rows_used_for_fit=0,
    )
    metric = build_typed_shared_psd_metric(
        basis,
        attenuation,
        config=qknn_config,
        source="c_id_support_nuisance_v1",
        provenance=provenance,
    )
    audit = _audit(
        lock=nuisance_lock,
        class_count=len(classes),
        metric=metric,
        support_receipt_sha256=support_receipt,
        fallback_reason="none",
        within=selected_within,
        between=selected_between,
        fraction=selected_fraction,
    )
    return FittedZIDSupportNuisanceMetric(metric=metric, audit=audit)
