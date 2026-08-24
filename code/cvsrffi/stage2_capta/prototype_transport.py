"""Support-only, zero-gradient prototype transport for CAPTA-P0-LS.

The implementation accepts only frozen class prototypes and labelled target
support features.  It deliberately has no source-sample, query, role, quota,
classifier-fit, optimizer, or model interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


A1_SUPPORT_SHRINK = "CAPTA_A1_SUPPORT_SHRINK"
A2_SHARED_SHIFT = "CAPTA_A2_SHARED_SHIFT"
A3_R4_SUPPORT_SHIFT = "CAPTA_A3_R4_SUPPORT_SHIFT"
CANDIDATE_IDS = (A1_SUPPORT_SHRINK, A2_SHARED_SHIFT, A3_R4_SUPPORT_SHIFT)
MAX_DOMAIN_RANK = 4
EPS = 1.0e-8


class CaptaPrototypeError(ValueError):
    """Raised when the CAPTA support-only closure fails closed."""


def _readonly(value: np.ndarray, dtype: Any = np.float32) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _normalized_rows(value: Any, *, field: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if rows.ndim != 2 or min(rows.shape) < 1 or not np.isfinite(rows).all():
        raise CaptaPrototypeError(f"{field} must be a finite matrix [N,D]")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if bool(np.any(norms <= EPS)):
        raise CaptaPrototypeError(f"{field} contains a zero vector")
    return np.ascontiguousarray(rows / norms, dtype=np.float32)


def _stable_basis_sign(basis: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(basis, dtype=np.float32)
    for column in range(result.shape[1]):
        direction = result[:, column]
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0.0:
            result[:, column] *= -1.0
    return result


@dataclass(frozen=True)
class CaptaPrototypeState:
    """Immutable per-row prototype state frozen before query access."""

    candidate_id: str
    source_prototypes: np.ndarray
    support_centers: np.ndarray
    transported_prototypes: np.ndarray
    target_prototypes: np.ndarray
    shared_shift: np.ndarray
    domain_basis: np.ndarray
    domain_code: np.ndarray
    effective_samples: np.ndarray
    audit: dict[str, Any]


def fit_capta_prototypes(
    frozen_prototypes: Any,
    support_features: Any,
    support_class_indices: Any,
    *,
    candidate_id: str,
    rank: int = MAX_DOMAIN_RANK,
    prior_strength: float = 3.0,
) -> CaptaPrototypeState:
    """Fit one immutable CAPTA-P0 state using labelled target support only."""

    if str(candidate_id) not in CANDIDATE_IDS:
        raise CaptaPrototypeError("candidate_id is not preregistered")
    if isinstance(rank, bool) or not 1 <= int(rank) <= MAX_DOMAIN_RANK:
        raise CaptaPrototypeError("domain rank must be an integer in [1,4]")
    prior = float(prior_strength)
    if not math.isfinite(prior) or prior <= 0.0:
        raise CaptaPrototypeError("prior_strength must be finite and positive")

    source = _normalized_rows(frozen_prototypes, field="frozen_prototypes")
    support = _normalized_rows(support_features, field="support_features")
    labels = np.asarray(support_class_indices)
    class_count, feature_dim = source.shape
    if (
        labels.ndim != 1
        or labels.shape != (len(support),)
        or labels.dtype.kind not in "iu"
        or isinstance(support_class_indices, bool)
        or bool(np.any(labels < 0))
        or bool(np.any(labels >= class_count))
        or not np.array_equal(np.unique(labels), np.arange(class_count))
    ):
        raise CaptaPrototypeError(
            "support labels must cover every frozen class with integer indices"
        )
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    counts = np.asarray(
        [int(np.sum(labels == index)) for index in range(class_count)],
        dtype=np.float32,
    )
    centers = np.stack(
        [support[labels == index].mean(axis=0) for index in range(class_count)]
    )
    centers = _normalized_rows(centers, field="support_centers")
    residuals = np.ascontiguousarray(centers - source, dtype=np.float32)
    raw_shared_shift = residuals.mean(axis=0, dtype=np.float32)

    basis = np.empty((feature_dim, 0), dtype=np.float32)
    code = np.empty((0,), dtype=np.float32)
    effective_rank = 0
    basis_source = "none"
    if candidate_id == A1_SUPPORT_SHRINK:
        shared_shift = np.zeros((feature_dim,), dtype=np.float32)
    elif candidate_id == A2_SHARED_SHIFT:
        shared_shift = np.ascontiguousarray(raw_shared_shift, dtype=np.float32)
    else:
        effective_rank = min(int(rank), class_count, feature_dim)
        u, _, _ = np.linalg.svd(
            residuals.astype(np.float64).T, full_matrices=False
        )
        basis = _stable_basis_sign(u[:, :effective_rank])
        code = np.ascontiguousarray(
            basis.T @ raw_shared_shift.astype(np.float32), dtype=np.float32
        )
        shared_shift = np.ascontiguousarray(basis @ code, dtype=np.float32)
        basis_source = "target_support_class_residuals"

    transported = _normalized_rows(
        source + shared_shift[None, :], field="transported_prototypes"
    )
    rho = counts / (counts + np.float32(prior))
    target = _normalized_rows(
        (1.0 - rho[:, None]) * transported + rho[:, None] * centers,
        field="target_prototypes",
    )
    state_arrays = (
        source,
        centers,
        transported,
        target,
        shared_shift,
        basis,
        code,
        counts,
    )
    state_bytes = int(sum(array.nbytes for array in state_arrays))
    audit = {
        "schema": "cvs.phase2.capta_p0.prototype_transport.v1",
        "candidate_id": str(candidate_id),
        "support_only": True,
        "query_rows_used_for_fit": 0,
        "source_sample_rows_used": 0,
        "class_balanced_shift": True,
        "requested_rank": int(rank),
        "effective_rank": effective_rank,
        "basis_source": basis_source,
        "prior_strength": prior,
        "trainable_parameter_count": 0,
        "backward_count": 0,
        "optimizer_state_bytes": 0,
        "target_state_bytes": state_bytes,
        "persistent_classifier_head": False,
    }
    return CaptaPrototypeState(
        candidate_id=str(candidate_id),
        source_prototypes=_readonly(source),
        support_centers=_readonly(centers),
        transported_prototypes=_readonly(transported),
        target_prototypes=_readonly(target),
        shared_shift=_readonly(shared_shift),
        domain_basis=_readonly(basis),
        domain_code=_readonly(code),
        effective_samples=_readonly(counts),
        audit=audit,
    )
