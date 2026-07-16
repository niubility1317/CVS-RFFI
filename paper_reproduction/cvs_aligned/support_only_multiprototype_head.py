"""Compressed support-only multi-prototype head for formal CVS Stage2-C.

The head is fitted exclusively from registered labeled support features.  It
never receives target-query labels, roles, batch class counts, class quotas,
query ordering, or query-query edges.  Every query is transformed and scored
independently against the same registered class state.

The design targets K=5/K=10 deployment:

* one global diagonal within-class residual whitening transform;
* at most two deterministic farthest-first prototypes per class;
* role-symmetric class scores blending the closest prototype and the class
  centroid;
* a support-only hubness penalty derived from registered class geometry.

The exact FP16 persistent state and per-query MAC estimate are exposed so the
route can be compared with identity-only single-qKNN without relying on a
self-reported resource manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np


EPS = 1.0e-8
HEAD_SCHEMA = "cvs.phase2.support_only_multiprototype_head.v1"
_DEPLOYMENT_TENSOR_KEYS = (
    "prototypes_fp16",
    "prototype_class_ids_uint16",
    "centroids_fp16",
    "residual_scale_fp16",
    "class_hubness_penalty_fp16",
    "scalars_fp16",
)
PACKED_HEAD_KEYS = (
    "schema_utf8",
    *_DEPLOYMENT_TENSOR_KEYS,
    "support_audit_json_utf8",
)


def _normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), EPS)


def _validate_support(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    rows = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    if rows.ndim != 2 or rows.shape[0] != targets.size or rows.shape[1] < 2:
        raise ValueError("support features must have shape [N,D>=2] and match labels")
    if not np.isfinite(rows).all():
        raise FloatingPointError("support features contain non-finite values")
    if int(class_count) < 2 or targets.size < int(class_count):
        raise ValueError("registered support requires at least two non-empty classes")
    if targets.min(initial=0) < 0 or targets.max(initial=-1) >= int(class_count):
        raise ValueError("support labels are outside the registered class range")
    counts = np.bincount(targets, minlength=int(class_count))
    if np.any(counts == 0):
        raise ValueError("every registered class must have support")
    return rows, targets, int(counts.min())


@dataclass(frozen=True)
class SupportOnlyMultiPrototypeHead:
    """Exact deployment state for independent all-registered-class scoring."""

    prototypes: np.ndarray
    prototype_class_ids: np.ndarray
    centroids: np.ndarray
    residual_scale: np.ndarray
    class_hubness_penalty: np.ndarray
    max_mix: float
    hubness_weight: float
    support_audit: dict[str, Any]

    @property
    def class_count(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.centroids.shape[1])

    @property
    def prototype_count(self) -> int:
        return int(self.prototypes.shape[0])

    @property
    def persistent_state_bytes_fp16(self) -> int:
        # The prototype-to-class map is stored as uint16; all numeric flight
        # tensors use FP16.  Scalars are included as two FP16 values.
        return int(
            2
            * (
                self.prototypes.size
                + self.centroids.size
                + self.residual_scale.size
                + self.class_hubness_penalty.size
                + 2
            )
            + 2 * self.prototype_class_ids.size
        )

    @property
    def extra_macs_per_query(self) -> int:
        # Diagonal transform + prototype cosine + centroid cosine + class
        # pooling/bias.  Normalization reductions are conservatively counted
        # as another 2D operations.
        d = self.feature_dim
        c = self.class_count
        p = self.prototype_count
        return int(3 * d + p * d + c * d + 2 * p + 3 * c)


def _utf8_array(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()


def _decode_utf8_array(value: np.ndarray, *, field: str) -> str:
    rows = np.asarray(value)
    if rows.dtype != np.uint8 or rows.ndim != 1:
        raise ValueError(f"{field} must be a one-dimensional uint8 array")
    try:
        return rows.tobytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field} is not valid UTF-8") from exc


def pack_support_only_multiprototype_head(
    head: SupportOnlyMultiPrototypeHead,
) -> dict[str, np.ndarray]:
    """Return a pickle-free FP16 deployment capsule payload.

    The returned arrays can be written with ``np.savez`` and loaded with
    ``allow_pickle=False``.  Model tensors are quantized to the exact flight
    dtypes counted by ``persistent_state_bytes_fp16``; schema and audit arrays
    are metadata and are deliberately excluded from that resource count.
    """

    if head.class_count > np.iinfo(np.uint16).max:
        raise ValueError("registered class count exceeds uint16 deployment range")
    audit_json = json.dumps(
        head.support_audit,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload = {
        "schema_utf8": _utf8_array(HEAD_SCHEMA),
        "prototypes_fp16": np.asarray(head.prototypes, dtype=np.float16),
        "prototype_class_ids_uint16": np.asarray(
            head.prototype_class_ids, dtype=np.uint16
        ),
        "centroids_fp16": np.asarray(head.centroids, dtype=np.float16),
        "residual_scale_fp16": np.asarray(head.residual_scale, dtype=np.float16),
        "class_hubness_penalty_fp16": np.asarray(
            head.class_hubness_penalty, dtype=np.float16
        ),
        "scalars_fp16": np.asarray(
            [head.max_mix, head.hubness_weight], dtype=np.float16
        ),
        "support_audit_json_utf8": _utf8_array(audit_json),
    }
    tensor_bytes = sum(int(payload[key].nbytes) for key in _DEPLOYMENT_TENSOR_KEYS)
    if tensor_bytes != head.persistent_state_bytes_fp16:
        raise AssertionError("packed deployment state disagrees with resource audit")
    return payload


def unpack_support_only_multiprototype_head(
    payload: dict[str, np.ndarray],
) -> SupportOnlyMultiPrototypeHead:
    """Validate and reconstruct a head from a pickle-free capsule payload."""

    required = set(PACKED_HEAD_KEYS)
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        raise ValueError(f"head capsule members mismatch: missing={missing}, extra={extra}")
    schema = _decode_utf8_array(payload["schema_utf8"], field="schema_utf8")
    if schema != HEAD_SCHEMA:
        raise ValueError(f"unsupported head schema: {schema}")
    try:
        audit = json.loads(
            _decode_utf8_array(
                payload["support_audit_json_utf8"], field="support_audit_json_utf8"
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError("support audit metadata is not valid JSON") from exc
    if not isinstance(audit, dict):
        raise ValueError("support audit metadata must decode to an object")

    prototypes = np.asarray(payload["prototypes_fp16"])
    prototype_ids = np.asarray(payload["prototype_class_ids_uint16"])
    centroids = np.asarray(payload["centroids_fp16"])
    residual_scale = np.asarray(payload["residual_scale_fp16"])
    class_penalty = np.asarray(payload["class_hubness_penalty_fp16"])
    scalars = np.asarray(payload["scalars_fp16"])
    if prototypes.dtype != np.float16 or prototypes.ndim != 2:
        raise ValueError("prototypes_fp16 must be a rank-two float16 array")
    if prototype_ids.dtype != np.uint16 or prototype_ids.shape != (len(prototypes),):
        raise ValueError("prototype class IDs must be uint16 and match prototypes")
    if centroids.dtype != np.float16 or centroids.ndim != 2:
        raise ValueError("centroids_fp16 must be a rank-two float16 array")
    class_count, feature_dim = centroids.shape
    if class_count < 2 or feature_dim < 2 or prototypes.shape[1] != feature_dim:
        raise ValueError("head capsule contains inconsistent class/feature dimensions")
    if residual_scale.dtype != np.float16 or residual_scale.shape != (feature_dim,):
        raise ValueError("residual_scale_fp16 has an invalid shape or dtype")
    if class_penalty.dtype != np.float16 or class_penalty.shape != (class_count,):
        raise ValueError("class_hubness_penalty_fp16 has an invalid shape or dtype")
    if scalars.dtype != np.float16 or scalars.shape != (2,):
        raise ValueError("scalars_fp16 must contain max_mix and hubness_weight")
    numeric = (prototypes, centroids, residual_scale, class_penalty, scalars)
    if not all(np.isfinite(value).all() for value in numeric):
        raise FloatingPointError("head capsule contains non-finite values")
    ids64 = prototype_ids.astype(np.int64)
    if ids64.size == 0 or ids64.max(initial=-1) >= class_count:
        raise ValueError("prototype class IDs are outside the registered class range")
    if np.any(np.bincount(ids64, minlength=class_count) == 0):
        raise ValueError("every registered class must retain a prototype")
    max_mix, hubness_weight = map(float, scalars.astype(np.float32))
    if not 0.0 <= max_mix <= 1.0 or hubness_weight < 0.0:
        raise ValueError("head capsule contains invalid scoring scalars")

    head = SupportOnlyMultiPrototypeHead(
        prototypes=prototypes.astype(np.float32),
        prototype_class_ids=ids64,
        centroids=centroids.astype(np.float32),
        residual_scale=residual_scale.astype(np.float32),
        class_hubness_penalty=class_penalty.astype(np.float32),
        max_mix=max_mix,
        hubness_weight=hubness_weight,
        support_audit=audit,
    )
    packed_bytes = sum(
        int(np.asarray(payload[key]).nbytes) for key in _DEPLOYMENT_TENSOR_KEYS
    )
    if packed_bytes != head.persistent_state_bytes_fp16:
        raise ValueError("head capsule resource bytes do not match reconstructed state")
    return head


def _fit_residual_scale(
    rows: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    shrinkage: float,
    scale_min: float,
    scale_max: float,
) -> np.ndarray:
    value = float(shrinkage)
    if not 0.0 <= value <= 1.0:
        raise ValueError("residual shrinkage must be in [0,1]")
    if not 0.0 < float(scale_min) <= 1.0 <= float(scale_max):
        raise ValueError("residual scale bounds must contain one")
    residuals = []
    for index in range(int(class_count)):
        selected = rows[labels == index]
        residuals.append(selected - selected.mean(axis=0, keepdims=True))
    residual = np.concatenate(residuals, axis=0)
    within_var = np.mean(np.square(residual), axis=0)
    global_var = float(np.mean(within_var))
    shrunk = (1.0 - value) * within_var + value * global_var
    inverse_std = 1.0 / np.sqrt(np.maximum(shrunk, EPS))
    median = float(np.median(inverse_std))
    normalized = inverse_std / max(median, EPS)
    return np.clip(normalized, float(scale_min), float(scale_max)).astype(np.float32)


def _farthest_first_indices(rows: np.ndarray, count: int) -> np.ndarray:
    values = _normalize(rows)
    target = min(max(1, int(count)), len(values))
    centroid = _normalize(values.mean(axis=0, keepdims=True))[0]
    first = int(np.argmax(values @ centroid))
    chosen = [first]
    while len(chosen) < target:
        similarity = values @ values[np.asarray(chosen)].T
        nearest = np.max(similarity, axis=1)
        nearest[np.asarray(chosen)] = np.inf
        chosen.append(int(np.argmin(nearest)))
    return np.asarray(chosen, dtype=np.int64)


def fit_support_only_multiprototype_head(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    *,
    class_count: int,
    max_prototypes_per_class: int = 2,
    residual_shrinkage: float = 0.5,
    residual_scale_min: float = 0.5,
    residual_scale_max: float = 2.0,
    max_mix: float = 0.75,
    hubness_weight: float = 0.25,
) -> SupportOnlyMultiPrototypeHead:
    """Fit a compact head using registered support only."""

    rows, labels, min_k = _validate_support(
        support_features, support_labels, class_count=int(class_count)
    )
    if not 1 <= int(max_prototypes_per_class) <= 4:
        raise ValueError("max_prototypes_per_class must be in [1,4]")
    if not 0.0 <= float(max_mix) <= 1.0:
        raise ValueError("max_mix must be in [0,1]")
    if float(hubness_weight) < 0.0:
        raise ValueError("hubness_weight must be nonnegative")

    scale = _fit_residual_scale(
        rows,
        labels,
        class_count=int(class_count),
        shrinkage=float(residual_shrinkage),
        scale_min=float(residual_scale_min),
        scale_max=float(residual_scale_max),
    )
    transformed = _normalize(rows * scale[None, :])
    centroids = []
    prototypes = []
    prototype_ids = []
    per_class = min(int(max_prototypes_per_class), int(min_k))
    for index in range(int(class_count)):
        selected = transformed[labels == index]
        centroid = _normalize(selected.mean(axis=0, keepdims=True))[0]
        centroids.append(centroid)
        chosen = selected[_farthest_first_indices(selected, per_class)]
        prototypes.extend(chosen)
        prototype_ids.extend([index] * len(chosen))
    centroid_bank = np.stack(centroids).astype(np.float32)
    prototype_bank = np.stack(prototypes).astype(np.float32)
    prototype_class_ids = np.asarray(prototype_ids, dtype=np.int64)

    gram = centroid_bank @ centroid_bank.T
    np.fill_diagonal(gram, -np.inf)
    # A class near many other class centers is a hub and otherwise tends to
    # absorb queries as the registry grows.  The fixed class penalty is fitted
    # from support geometry only and applied symmetrically to old/new classes.
    hubness = np.max(gram, axis=1)
    hubness = hubness - float(np.mean(hubness))
    class_penalty = (float(hubness_weight) * hubness).astype(np.float32)

    return SupportOnlyMultiPrototypeHead(
        prototypes=prototype_bank,
        prototype_class_ids=prototype_class_ids,
        centroids=centroid_bank,
        residual_scale=scale,
        class_hubness_penalty=class_penalty,
        max_mix=float(max_mix),
        hubness_weight=float(hubness_weight),
        support_audit={
            "fit_scope": "registered_support_only",
            "query_rows_used": 0,
            "query_labels_used": False,
            "query_roles_used": False,
            "query_true_batch_class_count_used": False,
            "query_class_quota_used": False,
            "query_global_assignment_used": False,
            "dense_query_graph_used": False,
            "role_symmetric_rule": True,
            "class_count": int(class_count),
            "min_physical_support_per_class": int(min_k),
            "prototypes_per_class": int(per_class),
            "residual_shrinkage": float(residual_shrinkage),
            "residual_scale_min": float(residual_scale_min),
            "residual_scale_max": float(residual_scale_max),
            "max_mix": float(max_mix),
            "hubness_weight": float(hubness_weight),
        },
    )


def score_support_only_multiprototype_head(
    query_features: np.ndarray,
    head: SupportOnlyMultiPrototypeHead,
) -> np.ndarray:
    """Return independent per-query scores over every registered class."""

    rows = np.asarray(query_features, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != head.feature_dim:
        raise ValueError("query features must have shape [N,D] matching the head")
    if not np.isfinite(rows).all():
        raise FloatingPointError("query features contain non-finite values")
    flat = _normalize(rows * head.residual_scale[None, :])
    prototype_scores = flat @ _normalize(head.prototypes).T
    centroid_scores = flat @ _normalize(head.centroids).T
    pooled = np.empty((len(flat), head.class_count), dtype=np.float32)
    for index in range(head.class_count):
        selected = prototype_scores[:, head.prototype_class_ids == index]
        pooled[:, index] = (
            float(head.max_mix) * np.max(selected, axis=1)
            + (1.0 - float(head.max_mix)) * centroid_scores[:, index]
            - head.class_hubness_penalty[index]
        )
    return pooled.astype(np.float32)


def predict_support_only_multiprototype_head(
    query_features: np.ndarray,
    head: SupportOnlyMultiPrototypeHead,
) -> np.ndarray:
    return np.argmax(score_support_only_multiprototype_head(query_features, head), axis=-1)
