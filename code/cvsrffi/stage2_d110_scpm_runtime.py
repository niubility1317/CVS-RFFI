"""Pure NumPy Phase2 runtime for D110-SCPM.

This module deliberately has no asset, checkpoint, source-row, query-label, or
batch-query dependency.  A caller supplies the already closed D106 basis and
the four sealed conditional prior variances as ordinary NumPy arrays.  The fit
path consumes target support only; the query path accepts one query at a time
and has no state-update surface.

The representation contract is explicit: ``support_z`` and ``query_z`` are
real embedding rows and are L2-normalized here before all SCPM statistics or
scores are calculated.  ``closed_u`` is row-oriented with shape ``[r, d]`` and
must already be orthonormal in that same normalized coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


RELATIVE_VARIANCE_CONDITION_CAP = 20.0
MIN_RELATIVE_VARIANCE = 1.0 / RELATIVE_VARIANCE_CONDITION_CAP
ORTHONORMALITY_ATOL = 2.0e-10
RUNTIME_SCHEMA = "cvs.phase2.d110.scpm_runtime.v1"
SUPPORTED_K = (1, 5, 10)


class D110SCPMRuntimeError(RuntimeError):
    """Raised when the support-only SCPM closure is not well defined."""


def _readonly(value: np.ndarray, dtype: np.dtype | type | None = None) -> np.ndarray:
    copied = np.array(value, dtype=dtype, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _require_float_array(value: object, name: str, *, ndim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != ndim:
        raise D110SCPMRuntimeError(f"{name} must be a {ndim}D NumPy array")
    if value.dtype.kind != "f" or not np.isfinite(value).all():
        raise D110SCPMRuntimeError(f"{name} must contain only finite real floats")
    return np.ascontiguousarray(value, dtype=np.float64)


def _l2_normalize_rows(value: object, name: str) -> np.ndarray:
    rows = _require_float_array(value, name, ndim=2)
    if rows.shape[0] < 1 or rows.shape[1] < 2:
        raise D110SCPMRuntimeError(f"{name} must have shape [N,d] with N>=1 and d>=2")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise D110SCPMRuntimeError(f"{name} contains a zero-norm row")
    return np.ascontiguousarray(rows / norms, dtype=np.float64)


def _l2_normalize_query(value: object, dimension: int) -> np.ndarray:
    query = _require_float_array(value, "query_z", ndim=1)
    if query.shape != (dimension,):
        raise D110SCPMRuntimeError(f"query_z must have shape [{dimension}]")
    norm = float(np.linalg.norm(query))
    if not math.isfinite(norm) or norm <= 0.0:
        raise D110SCPMRuntimeError("query_z must have a finite positive L2 norm")
    return np.ascontiguousarray(query / norm, dtype=np.float64)


def _require_labels(value: object, rows: int) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 1 or len(value) != rows:
        raise D110SCPMRuntimeError(
            "support_labels must be a one-dimensional NumPy array aligned to support_z"
        )
    if value.dtype.kind not in {"i", "u", "U", "S"}:
        raise D110SCPMRuntimeError(
            "support_labels must use an integer, unicode, or bytes NumPy dtype"
        )
    labels = np.ascontiguousarray(value).copy()
    if value.dtype.kind in {"U", "S"} and any(not item for item in labels.tolist()):
        raise D110SCPMRuntimeError("support_labels must not contain an empty class token")
    return labels


def _validate_closed_u(value: object, dimension: int) -> np.ndarray:
    basis = _require_float_array(value, "closed_u", ndim=2)
    rank, basis_dimension = basis.shape
    if rank < 1 or rank >= dimension or basis_dimension != dimension:
        raise D110SCPMRuntimeError(
            "closed_u must have shape [r,d] with 1<=r<d and support dimension d"
        )
    if not np.allclose(
        basis @ basis.T,
        np.eye(rank, dtype=np.float64),
        rtol=0.0,
        atol=ORTHONORMALITY_ATOL,
    ):
        raise D110SCPMRuntimeError("closed_u must already be row-orthonormal")
    return basis


def _validate_prior_variances(value: object, groups: int) -> np.ndarray:
    prior = _require_float_array(value, "prior_variances", ndim=1)
    if prior.shape != (groups,) or np.any(prior < 0.0):
        raise D110SCPMRuntimeError(
            "prior_variances must be finite nonnegative [u_1,...,u_r,perp] values"
        )
    return prior


def _grouped_support(
    support: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    class_labels = np.unique(labels)
    if len(class_labels) < 2:
        raise D110SCPMRuntimeError("SCPM requires support for at least two registered classes")
    groups = [support[labels == label] for label in class_labels]
    counts = [len(group) for group in groups]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise D110SCPMRuntimeError(
            "every registered class must have the same positive K-shot support count"
        )
    return np.stack(groups, axis=0), class_labels, counts[0]


def _class_block_variances(grouped: np.ndarray, closed_u: np.ndarray) -> np.ndarray:
    """Return frozen t_cj/t_cperp for each class, without retaining support rows."""

    class_count, k_shot, dimension = grouped.shape
    rank = closed_u.shape[0]
    residual = grouped - np.mean(grouped, axis=1, keepdims=True, dtype=np.float64)
    projected = residual @ closed_u.T
    t_parallel = np.sum(np.square(projected), axis=1) / float(k_shot - 1)
    residual_energy = np.sum(np.square(residual), axis=(1, 2))
    parallel_energy = np.sum(np.square(projected), axis=(1, 2))
    perpendicular_energy = np.maximum(residual_energy - parallel_energy, 0.0)
    t_perp = perpendicular_energy / float((k_shot - 1) * (dimension - rank))
    result = np.concatenate((t_parallel, t_perp[:, None]), axis=1)
    if result.shape != (class_count, rank + 1) or not np.isfinite(result).all():
        raise D110SCPMRuntimeError("class-block SCPM variance computation drifted")
    return result


def _class_block_lw_alpha(
    class_variances: np.ndarray, prior_variances: np.ndarray, dimension: int, rank: int
) -> tuple[float, np.ndarray]:
    """Apply the frozen class-block Ledoit-Wolf-style moment shrinkage equation."""

    class_count = class_variances.shape[0]
    target_variances = np.mean(class_variances, axis=0, dtype=np.float64)
    variation = np.sum(
        np.square(class_variances - target_variances[None, :]), axis=0
    ) / float(class_count * (class_count - 1))
    group_dimensions = np.concatenate(
        (np.ones(rank, dtype=np.float64), np.asarray([dimension - rank], dtype=np.float64))
    )
    numerator = float(np.dot(group_dimensions, variation))
    denominator = float(
        np.dot(group_dimensions, np.square(target_variances - prior_variances))
    )
    if not math.isfinite(numerator) or not math.isfinite(denominator) or numerator < 0.0:
        raise D110SCPMRuntimeError("class-block Ledoit-Wolf moments became non-finite")
    alpha = 1.0 if denominator == 0.0 else float(np.clip(numerator / denominator, 0.0, 1.0))
    return alpha, target_variances


def _safe_variances(variances: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Apply the fixed relative condition cap, or the all-zero Euclidean fallback."""

    maximum = float(np.max(variances))
    if not math.isfinite(maximum) or maximum < 0.0:
        raise D110SCPMRuntimeError("SCPM variances must be finite and nonnegative")
    if maximum == 0.0:
        groups = len(variances)
        return np.ones(groups, dtype=np.float64), np.ones(groups, dtype=np.float64), True
    relative = np.maximum(variances / maximum, MIN_RELATIVE_VARIANCE)
    safe = relative * maximum
    return relative, safe, False


@dataclass(frozen=True, slots=True)
class D110SCPMRuntimeState:
    """Compact support-only state; it retains class centers but no support/query rows."""

    class_labels: np.ndarray
    centers: np.ndarray
    closed_u: np.ndarray
    prior_variances: np.ndarray
    target_variances: np.ndarray | None
    variances: np.ndarray
    safe_relative_variances: np.ndarray
    predictive_variances: np.ndarray
    alpha: float
    active_k: int
    euclidean_fallback: bool
    query_rows_used_for_fit: int = 0
    query_state_updates: int = 0
    schema: str = RUNTIME_SCHEMA


def fit_d110_scpm_runtime(
    support_z: np.ndarray,
    support_labels: np.ndarray,
    closed_u: np.ndarray,
    prior_variances: np.ndarray,
) -> D110SCPMRuntimeState:
    """Fit one immutable SCPM state from support only.

    ``prior_variances`` follows ``[u_1, ..., u_r, perp]``.  K=1 uses those
    values directly.  For K>1 the support-derived class-block moments are
    combined with the same prior by the frozen shared-alpha rule from D110
    theory §4.3.  No query argument exists in this fit API.
    """

    normalized_support = _l2_normalize_rows(support_z, "support_z")
    labels = _require_labels(support_labels, len(normalized_support))
    basis = _validate_closed_u(closed_u, normalized_support.shape[1])
    prior = _validate_prior_variances(prior_variances, basis.shape[0] + 1)
    grouped, class_labels, k_shot = _grouped_support(normalized_support, labels)
    if k_shot not in SUPPORTED_K:
        raise D110SCPMRuntimeError(
            f"SCPM supports only frozen K values {SUPPORTED_K}, got K={k_shot}"
        )
    centers = np.mean(grouped, axis=1, dtype=np.float64)

    if k_shot == 1:
        alpha = 1.0
        target_variances: np.ndarray | None = None
        variances = prior.copy()
    else:
        class_variances = _class_block_variances(grouped, basis)
        alpha, target_variances = _class_block_lw_alpha(
            class_variances,
            prior,
            normalized_support.shape[1],
            basis.shape[0],
        )
        variances = alpha * prior + (1.0 - alpha) * target_variances
    if not np.isfinite(variances).all() or np.any(variances < 0.0):
        raise D110SCPMRuntimeError("SCPM shrinkage produced an invalid variance")
    safe_relative, safe_variances, euclidean_fallback = _safe_variances(variances)
    predictive = safe_variances * (1.0 + 1.0 / float(k_shot))
    return D110SCPMRuntimeState(
        class_labels=_readonly(class_labels),
        centers=_readonly(centers, np.float64),
        closed_u=_readonly(basis, np.float64),
        prior_variances=_readonly(prior, np.float64),
        target_variances=(
            None if target_variances is None else _readonly(target_variances, np.float64)
        ),
        variances=_readonly(variances, np.float64),
        safe_relative_variances=_readonly(safe_relative, np.float64),
        predictive_variances=_readonly(predictive, np.float64),
        alpha=alpha,
        active_k=k_shot,
        euclidean_fallback=euclidean_fallback,
    )


def _checked_state(state: object) -> D110SCPMRuntimeState:
    if not isinstance(state, D110SCPMRuntimeState):
        raise D110SCPMRuntimeError("SCPM query scoring requires a D110SCPMRuntimeState")
    if (
        state.schema != RUNTIME_SCHEMA
        or state.query_rows_used_for_fit != 0
        or state.query_state_updates != 0
    ):
        raise D110SCPMRuntimeError("SCPM state violates the query-immutable lifecycle")
    return state


def score_d110_scpm_query(
    state: D110SCPMRuntimeState, query_z: np.ndarray
) -> np.ndarray:
    """Score one query against every registered class without mutating ``state``.

    The output is an immutable ``float64[C]`` vector in ``state.class_labels``
    order.  Smaller values are the explicit predictive Mahalanobis distances.
    Query batches are intentionally not accepted so callers cannot introduce a
    cross-query fitting or reassignment surface.
    """

    checked = _checked_state(state)
    query = _l2_normalize_query(query_z, checked.centers.shape[1])
    delta = query[None, :] - checked.centers
    if checked.euclidean_fallback:
        scores = np.sum(np.square(delta), axis=1)
        return _readonly(scores, np.float64)

    projected = delta @ checked.closed_u.T
    parallel = np.sum(
        np.square(projected) / checked.predictive_variances[None, :-1], axis=1
    )
    total_energy = np.sum(np.square(delta), axis=1)
    parallel_energy = np.sum(np.square(projected), axis=1)
    perpendicular = np.maximum(total_energy - parallel_energy, 0.0)
    scores = parallel + perpendicular / checked.predictive_variances[-1]
    if not np.isfinite(scores).all():
        raise D110SCPMRuntimeError("SCPM query score became non-finite")
    return _readonly(scores, np.float64)


def predict_d110_scpm_query(
    state: D110SCPMRuntimeState, query_z: np.ndarray
) -> object:
    """Return the winning registered label for one read-only SCPM query."""

    scores = score_d110_scpm_query(state, query_z)
    return state.class_labels[int(np.argmin(scores))].item()


__all__ = [
    "D110SCPMRuntimeError",
    "D110SCPMRuntimeState",
    "MIN_RELATIVE_VARIANCE",
    "RELATIVE_VARIANCE_CONDITION_CAP",
    "RUNTIME_SCHEMA",
    "SUPPORTED_K",
    "fit_d110_scpm_runtime",
    "predict_d110_scpm_query",
    "score_d110_scpm_query",
]
