"""D34 support-only collision-local new-class registration.

The frozen old cosine-score prefix is supplied by the caller and copied into
the output unchanged.  D34 only registers new support centroids and calibrates
their scores from labeled LEO_weak support.  It has no query fitting, role,
quota, batch-assignment, or query-query graph interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


FEATURE_DIM = 288
TEMPERATURE = np.float32(18.0)
ALLOWED_NEW_CLASS_COUNTS = (2, 5, 10, 20)
SCHEMA = "cvs.phase2.d34_collision_local_registration.v1"
RESOURCE_SCHEMA = "cvs.phase2.d34_collision_local_registration_resource.v1"
EPSILON = np.float32(1.0e-4)
NONEDGE_B = np.float32(2.0)


class D34CollisionLocalRegistrationError(ValueError):
    """Raised when support or the fixed D34 method lock drifts."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _rows(value: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D34CollisionLocalRegistrationError(
            f"{name} must be finite [N,{FEATURE_DIM}]"
        )
    norms = np.linalg.norm(rows, axis=1).astype(np.float32)
    if bool(np.any(np.abs(norms - np.float32(1.0)) > np.float32(2.0e-4))):
        raise D34CollisionLocalRegistrationError(
            f"{name} must already contain FAST-adapted unit rows"
        )
    return np.asarray(rows, dtype=np.float32)


def _support(
    features: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    rows = _rows(features, f"{name} features")
    y = np.asarray(tuple(str(v) for v in labels))
    registry = tuple(str(v) for v in classes)
    if (
        y.ndim != 1
        or len(y) != len(rows)
        or not registry
        or len(set(registry)) != len(registry)
        or any(not v for v in registry)
        or set(y.tolist()) != set(registry)
    ):
        raise D34CollisionLocalRegistrationError(f"{name} registry drift")
    counts = [int(np.sum(y == v)) for v in registry]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise D34CollisionLocalRegistrationError(
            f"{name} must be class-symmetric K-shot"
        )
    return rows, y, registry, counts[0]


def _prefix(value: np.ndarray, n: int, c: int, name: str) -> np.ndarray:
    scores = np.asarray(value)
    if (
        scores.dtype != np.float32
        or scores.shape != (n, c)
        or not np.isfinite(scores).all()
    ):
        raise D34CollisionLocalRegistrationError(
            f"{name} must be finite float32 [{n},{c}] frozen cosine scores"
        )
    return np.ascontiguousarray(scores)


def _unit_mean(rows: np.ndarray) -> np.ndarray:
    mean = np.mean(rows, axis=0, dtype=np.float32)
    norm = np.float32(np.linalg.norm(mean))
    if norm <= np.float32(1.0e-12):
        raise D34CollisionLocalRegistrationError("zero-norm support mean")
    return np.asarray(mean / norm, dtype=np.float32)


def _prototype(rows: np.ndarray, gamma: float) -> np.ndarray:
    mean = _unit_mean(rows)
    similarities = np.asarray(rows @ rows.T, dtype=np.float32)
    medoid = rows[int(np.argmax(np.mean(similarities, axis=1)))]
    mixed = np.asarray((1.0 - gamma) * mean + gamma * medoid, dtype=np.float32)
    return _unit_mean(mixed[None, :])


def _quantize(prototypes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    peak = np.max(np.abs(prototypes), axis=1).astype(np.float32)
    if bool(np.any(peak <= np.float32(1.0e-12))):
        raise D34CollisionLocalRegistrationError("zero-range prototype")
    scales = np.asarray(peak / np.float32(127.0), dtype=np.float32)
    q = np.asarray(
        np.clip(np.rint(prototypes / scales[:, None]), -127, 127), dtype=np.int8
    )
    norms = np.linalg.norm(q.astype(np.float32), axis=1).astype(np.float32)
    inverse_norms = np.asarray(np.float32(1.0) / norms, dtype=np.float32)
    return q, scales, inverse_norms


@dataclass(frozen=True)
class D34CollisionLocalConfig:
    """One of the three fixed support-only D34 arms."""

    arm: str = "B"

    def __post_init__(self) -> None:
        arm = str(self.arm).upper().removeprefix("D34-")
        if arm not in {"A", "B", "C"}:
            raise D34CollisionLocalRegistrationError("D34 arm must be A, B, or C")
        object.__setattr__(self, "arm", arm)

    @property
    def max_edges_per_new(self) -> int:
        return {"A": 1, "B": 2, "C": 3}[self.arm]

    @property
    def gamma_medoid(self) -> float:
        return 0.0 if self.arm == "A" else 0.25

    @property
    def uncertainty_lambda(self) -> float:
        return {"A": 0.0, "B": 0.5, "C": 1.0}[self.arm]


@dataclass(frozen=True)
class D34CollisionLocalRegistrationState:
    schema: str
    classes: tuple[str, ...]
    old_class_count: int
    new_prototypes_qint8: np.ndarray
    new_prototype_scales: np.ndarray
    new_prototype_inverse_norms: np.ndarray
    collision_edge_mask: np.ndarray
    old_anchor_offsets: np.ndarray
    unreachable_new_mask: np.ndarray
    support_count_by_new_class: np.ndarray
    arm: str
    epsilon: float = float(EPSILON)
    nonedge_B: float = float(NONEDGE_B)
    optimizer_steps: int = 0

    def __post_init__(self) -> None:
        c_old = int(self.old_class_count)
        c_new = len(self.classes) - c_old
        if (
            self.schema != SCHEMA
            or not 2 <= c_old <= 6
            or c_new not in ALLOWED_NEW_CLASS_COUNTS
            or self.new_prototypes_qint8.shape != (c_new, FEATURE_DIM)
            or self.new_prototypes_qint8.dtype != np.int8
            or self.new_prototype_scales.shape != (c_new,)
            or self.new_prototype_scales.dtype != np.float32
            or self.new_prototype_inverse_norms.shape != (c_new,)
            or self.new_prototype_inverse_norms.dtype != np.float32
            or self.collision_edge_mask.shape != (c_new, c_old)
            or self.collision_edge_mask.dtype != np.bool_
            or bool(np.any(np.sum(self.collision_edge_mask, axis=1) < 1))
            or self.old_anchor_offsets.shape != (c_old,)
            or self.old_anchor_offsets.dtype != np.float32
            or self.unreachable_new_mask.shape != (c_new,)
            or self.unreachable_new_mask.dtype != np.bool_
            or self.support_count_by_new_class.shape != (c_new,)
            or self.support_count_by_new_class.dtype != np.uint16
            or not np.isfinite(self.new_prototype_scales).all()
            or bool(np.any(self.new_prototype_scales <= 0.0))
            or not np.isfinite(self.new_prototype_inverse_norms).all()
            or bool(np.any(self.new_prototype_inverse_norms <= 0.0))
            or not np.isfinite(self.old_anchor_offsets).all()
            or not np.isfinite(float(self.epsilon))
            or float(self.epsilon) != float(EPSILON)
            or not np.isfinite(float(self.nonedge_B))
            or float(self.nonedge_B) != float(NONEDGE_B)
            or int(self.optimizer_steps) != 0
        ):
            raise D34CollisionLocalRegistrationError("D34 state drift")
        for field, dtype in (
            ("new_prototypes_qint8", np.int8),
            ("new_prototype_scales", np.float32),
            ("new_prototype_inverse_norms", np.float32),
            ("collision_edge_mask", np.bool_),
            ("old_anchor_offsets", np.float32),
            ("unreachable_new_mask", np.bool_),
            ("support_count_by_new_class", np.uint16),
        ):
            object.__setattr__(self, field, _readonly(getattr(self, field), dtype))

    @property
    def old_classes(self) -> tuple[str, ...]:
        return self.classes[: self.old_class_count]

    @property
    def new_classes(self) -> tuple[str, ...]:
        return self.classes[self.old_class_count :]

    @property
    def persistent_state_bytes(self) -> int:
        return int(
            self.new_prototypes_qint8.nbytes
            + self.new_prototype_scales.nbytes
            + self.new_prototype_inverse_norms.nbytes
            + self.collision_edge_mask.nbytes
            + self.old_anchor_offsets.nbytes
            + self.unreachable_new_mask.nbytes
            + self.support_count_by_new_class.nbytes
        )


@dataclass(frozen=True)
class D34CollisionLocalRegistrationResult:
    state: D34CollisionLocalRegistrationState
    geometry_audit: dict[str, Any]
    resource_audit: dict[str, Any]


def _edge_mask(
    new_labels: np.ndarray,
    new_classes: tuple[str, ...],
    new_old_prefix: np.ndarray,
    config: D34CollisionLocalConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    winners = np.argmax(new_old_prefix, axis=1)
    c_old = new_old_prefix.shape[1]
    mask = np.zeros((len(new_classes), c_old), dtype=np.bool_)
    audit: list[dict[str, Any]] = []
    for j, class_name in enumerate(new_classes):
        class_winners = winners[new_labels == class_name]
        counts = np.bincount(class_winners, minlength=c_old)
        order = np.lexsort((np.arange(c_old), -counts))
        if config.arm == "C":
            cumulative = np.cumsum(counts[order]) / float(len(class_winners))
            reached = np.flatnonzero(cumulative >= 0.80)
            degree = min(3, int(reached[0] + 1) if len(reached) else 3)
        else:
            degree = config.max_edges_per_new
        mask[j, order[:degree]] = True
        audit.append(
            {
                "new_class": class_name,
                "old_top_frequency_counts": counts.tolist(),
                "edge_old_indices": order[:degree].tolist(),
                "degree": degree,
                "frequency_coverage": float(np.sum(counts[order[:degree]]) / len(class_winners)),
            }
        )
    return mask, audit


def _old_uncertainty(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    k: int,
    *,
    excluded_row: int | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if k == 1:
        return (
            np.full(len(classes), 0.05, dtype=np.float32),
            [
                {"old_class": name, "mode": "k1_method_lock", "u_i": 0.05}
                for name in classes
            ],
        )
    keep = np.ones(len(rows), dtype=bool)
    if excluded_row is not None:
        keep[int(excluded_row)] = False
    kept_rows = rows[keep]
    kept_labels = labels[keep]
    centroids = np.stack(
        [_unit_mean(kept_rows[kept_labels == name]) for name in classes]
    )
    uncertainties = np.empty(len(classes), dtype=np.float32)
    trace: list[dict[str, Any]] = []
    for i, name in enumerate(classes):
        class_rows = kept_rows[kept_labels == name]
        total = np.sum(class_rows, axis=0, dtype=np.float32)
        margins = []
        for row in class_rows:
            own = (
                _unit_mean((total - row)[None, :])
                if len(class_rows) > 1
                else class_rows[0]
            )
            own_score = float(row @ own)
            other_score = float(np.max(row @ np.delete(centroids, i, axis=0).T))
            margins.append(own_score - other_score)
        spread = float(max(margins) - min(margins)) if len(margins) > 1 else 0.05
        uncertainties[i] = np.float32(spread)
        trace.append(
            {
                "old_class": name,
                "mode": "leave_one_old_out_margin_max_minus_min",
                "margin_min": min(margins),
                "margin_max": max(margins),
                "u_i": spread,
            }
        )
    return uncertainties, trace


def _safe_offsets(
    old_x: np.ndarray,
    old_target: np.ndarray,
    old_prefix: np.ndarray,
    raw_old_support: np.ndarray,
    edge_mask: np.ndarray,
    old_classes: tuple[str, ...],
    config: D34CollisionLocalConfig,
    uncertainty: np.ndarray,
    *,
    excluded_row: int | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    old_winner = np.argmax(old_prefix, axis=1)
    correct = old_winner == old_target
    per_class_acc = np.asarray(
        [np.mean(correct[old_target == i]) for i in range(len(old_classes))],
        dtype=np.float32,
    )
    median_acc = float(np.median(per_class_acc))
    offsets = np.zeros(len(old_classes), dtype=np.float32)
    trace: list[dict[str, Any]] = []
    for i, name in enumerate(old_classes):
        connected_new = np.flatnonzero(edge_mask[:, i])
        protected_rows = np.flatnonzero(correct & (old_target == i))
        if excluded_row is not None:
            protected_rows = protected_rows[protected_rows != int(excluded_row)]
        floor_multiplier = (
            2.0
            if config.arm == "C" and per_class_acc[i] < median_acc
            else 1.0
        )
        if len(connected_new) and len(protected_rows):
            caps = old_prefix[protected_rows, i, None] - raw_old_support[
                np.ix_(protected_rows, connected_new)
            ]
            safe_cap = float(np.min(caps))
            offset = safe_cap - floor_multiplier * (
                float(EPSILON)
                + config.uncertainty_lambda * float(uncertainty[i])
            )
        else:
            safe_cap = 0.0
            offset = 0.0
        offsets[i] = np.float32(offset)
        trace.append(
            {
                "old_class": name,
                "connected_new_count": int(len(connected_new)),
                "protected_correct_support_count": int(len(protected_rows)),
                "safe_cap": safe_cap,
                "u_i": float(uncertainty[i]),
                "uncertainty_lambda": config.uncertainty_lambda,
                "floor_multiplier": floor_multiplier,
                "shared_safe_offset": float(offsets[i]),
                "excluded_old_row": excluded_row,
            }
        )
    return offsets, trace


def _new_raw(state: D34CollisionLocalRegistrationState, rows: np.ndarray) -> np.ndarray:
    return np.asarray(
        TEMPERATURE
        * (rows @ state.new_prototypes_qint8.T)
        * state.new_prototype_inverse_norms[None, :],
        dtype=np.float32,
    )


def _compose(
    state: D34CollisionLocalRegistrationState,
    old_prefix: np.ndarray,
    raw_new: np.ndarray,
) -> np.ndarray:
    winner = np.argmax(old_prefix, axis=1)
    winner_score = old_prefix[np.arange(len(old_prefix)), winner]
    new_scores = np.empty_like(raw_new, dtype=np.float32)
    for j in range(raw_new.shape[1]):
        connected = state.collision_edge_mask[j, winner]
        new_scores[:, j] = np.where(
            connected,
            raw_new[:, j] + state.old_anchor_offsets[winner],
            winner_score - np.float32(state.nonedge_B),
        )
    return _readonly(np.concatenate((old_prefix, new_scores), axis=1), np.float32)


def _compose_sparse(
    state: D34CollisionLocalRegistrationState,
    rows: np.ndarray,
    old_prefix: np.ndarray,
) -> np.ndarray:
    """Score only new classes adjacent to each row's frozen old winner."""

    winner = np.argmax(old_prefix, axis=1)
    winner_score = old_prefix[np.arange(len(old_prefix)), winner]
    c_new = len(state.new_classes)
    new_scores = np.repeat(
        (winner_score - np.float32(state.nonedge_B))[:, None], c_new, axis=1
    ).astype(np.float32)
    for old_index in range(state.old_class_count):
        row_indices = np.flatnonzero(winner == old_index)
        adjacent = np.flatnonzero(state.collision_edge_mask[:, old_index])
        if not len(row_indices) or not len(adjacent):
            continue
        raw = np.asarray(
            rows[row_indices] @ state.new_prototypes_qint8[adjacent].T,
            dtype=np.float32,
        ) * state.new_prototype_inverse_norms[adjacent][None, :]
        raw *= TEMPERATURE
        new_scores[np.ix_(row_indices, adjacent)] = (
            raw + state.old_anchor_offsets[old_index]
        )
    return _readonly(np.concatenate((old_prefix, new_scores), axis=1), np.float32)


def fit_d34_collision_local_registration(
    old_support_features: np.ndarray,
    old_support_labels: Sequence[str],
    old_registered_classes: Sequence[str],
    old_support_score_prefix: np.ndarray,
    new_support_features: np.ndarray,
    new_support_labels: Sequence[str],
    new_registered_classes: Sequence[str],
    new_support_old_score_prefix: np.ndarray,
    *,
    config: D34CollisionLocalConfig | None = None,
) -> D34CollisionLocalRegistrationResult:
    """Fit new registration from support while preserving the frozen old head."""

    locked = config or D34CollisionLocalConfig()
    old_x, old_y, old_classes, old_k = _support(
        old_support_features, old_support_labels, old_registered_classes, "D34 old support"
    )
    new_x, new_y, new_classes, new_k = _support(
        new_support_features, new_support_labels, new_registered_classes, "D34 new support"
    )
    if (
        len(old_classes) < 2
        or len(old_classes) > 6
        or len(new_classes) not in ALLOWED_NEW_CLASS_COUNTS
        or set(old_classes) & set(new_classes)
        or old_k != new_k
    ):
        raise D34CollisionLocalRegistrationError(
            "D34 requires disjoint old/new matched class-symmetric K-shot support"
        )
    old_prefix = _prefix(
        old_support_score_prefix, len(old_x), len(old_classes), "old support prefix"
    )
    new_old_prefix = _prefix(
        new_support_old_score_prefix, len(new_x), len(old_classes), "new support old prefix"
    )
    prototypes = np.stack(
        [_prototype(new_x[new_y == name], locked.gamma_medoid) for name in new_classes]
    ).astype(np.float32)
    q, scales, inverse_norms = _quantize(prototypes)
    edge_mask, edge_trace = _edge_mask(new_y, new_classes, new_old_prefix, locked)
    raw_old_support = np.asarray(
        TEMPERATURE * (old_x @ q.T) * inverse_norms[None, :], dtype=np.float32
    )
    old_target = np.asarray([old_classes.index(str(v)) for v in old_y], dtype=np.int64)
    uncertainty, old_uncertainty_trace = _old_uncertainty(
        old_x, old_y, old_classes, old_k
    )
    offsets, offset_trace = _safe_offsets(
        old_x,
        old_target,
        old_prefix,
        raw_old_support,
        edge_mask,
        old_classes,
        locked,
        uncertainty,
    )
    state0 = D34CollisionLocalRegistrationState(
        schema=SCHEMA,
        classes=old_classes + new_classes,
        old_class_count=len(old_classes),
        new_prototypes_qint8=q,
        new_prototype_scales=scales,
        new_prototype_inverse_norms=inverse_norms,
        collision_edge_mask=edge_mask,
        old_anchor_offsets=offsets,
        unreachable_new_mask=np.zeros(len(new_classes), dtype=np.bool_),
        support_count_by_new_class=np.full(len(new_classes), new_k, dtype=np.uint16),
        arm=locked.arm,
    )
    old_after = _compose(state0, old_prefix, raw_old_support)
    old_before_pred = np.argmax(old_prefix, axis=1)
    correct = old_before_pred == old_target
    old_after_pred = np.argmax(old_after, axis=1)
    intrusions = int(np.sum(correct & (old_after_pred != old_before_pred)))
    if intrusions:
        raise D34CollisionLocalRegistrationError(
            "old correct-support non-degradation hard protection failed"
        )
    old_loso_trace: list[dict[str, Any]] = []
    old_loso_intrusions = 0
    if old_k == 1:
        old_loso_trace.append(
            {
                "mode": "k1_no_pseudo_loso",
                "intrusion_evaluated": False,
                "method_lock_uncertainty": 0.05,
            }
        )
    else:
        for row_index in range(len(old_x)):
            fold_uncertainty, _ = _old_uncertainty(
                old_x,
                old_y,
                old_classes,
                old_k,
                excluded_row=row_index,
            )
            fold_offsets, _ = _safe_offsets(
                old_x,
                old_target,
                old_prefix,
                raw_old_support,
                edge_mask,
                old_classes,
                locked,
                fold_uncertainty,
                excluded_row=row_index,
            )
            fold_state = D34CollisionLocalRegistrationState(
                **{**state0.__dict__, "old_anchor_offsets": fold_offsets}
            )
            held_scores = _compose_sparse(
                fold_state,
                old_x[row_index : row_index + 1],
                old_prefix[row_index : row_index + 1],
            )
            was_correct = bool(correct[row_index])
            prediction = int(np.argmax(held_scores[0]))
            intrusion = bool(was_correct and prediction >= len(old_classes))
            old_loso_intrusions += int(intrusion)
            old_loso_trace.append(
                {
                    "mode": "leave_one_old_out_rebuild_offset",
                    "row_index": row_index,
                    "old_class": str(old_y[row_index]),
                    "was_correct_before_registration": was_correct,
                    "prediction_registered_index": prediction,
                    "new_class_intrusion": intrusion,
                }
            )
    new_loso_trace: list[dict[str, Any]] = []
    unreachable = np.zeros(len(new_classes), dtype=np.bool_)
    if new_k == 1:
        for name in new_classes:
            new_loso_trace.append(
                {"new_class": name, "mode": "k1_no_pseudo_loso", "unreachable": False}
            )
    else:
        for j, name in enumerate(new_classes):
            indices = np.flatnonzero(new_y == name)
            margins = []
            for row_index in indices:
                keep = np.ones(len(new_x), dtype=bool)
                keep[row_index] = False
                fold_prototypes = np.stack(
                    [
                        _prototype(
                            new_x[keep & (new_y == class_name)],
                            locked.gamma_medoid,
                        )
                        for class_name in new_classes
                    ]
                ).astype(np.float32)
                fold_q, fold_scales, fold_inverse = _quantize(fold_prototypes)
                fold_edges, _ = _edge_mask(
                    new_y[keep],
                    new_classes,
                    new_old_prefix[keep],
                    locked,
                )
                fold_raw_old = np.asarray(
                    TEMPERATURE
                    * (old_x @ fold_q.T)
                    * fold_inverse[None, :],
                    dtype=np.float32,
                )
                fold_offsets, _ = _safe_offsets(
                    old_x,
                    old_target,
                    old_prefix,
                    fold_raw_old,
                    fold_edges,
                    old_classes,
                    locked,
                    uncertainty,
                )
                fold_counts = np.asarray(
                    [int(np.sum(new_y[keep] == value)) for value in new_classes],
                    dtype=np.uint16,
                )
                fold_state = D34CollisionLocalRegistrationState(
                    schema=SCHEMA,
                    classes=old_classes + new_classes,
                    old_class_count=len(old_classes),
                    new_prototypes_qint8=fold_q,
                    new_prototype_scales=fold_scales,
                    new_prototype_inverse_norms=fold_inverse,
                    collision_edge_mask=fold_edges,
                    old_anchor_offsets=fold_offsets,
                    unreachable_new_mask=np.zeros(len(new_classes), dtype=np.bool_),
                    support_count_by_new_class=fold_counts,
                    arm=locked.arm,
                )
                scores = _compose_sparse(
                    fold_state,
                    new_x[row_index : row_index + 1],
                    new_old_prefix[row_index : row_index + 1],
                )[0]
                own = float(scores[len(old_classes) + j])
                competitors = np.delete(scores, len(old_classes) + j)
                margins.append(own - float(np.max(competitors)))
            unreachable[j] = bool(min(margins) <= 0.0)
            new_loso_trace.append(
                {
                    "new_class": name,
                    "mode": "physical_support_leave_one_out",
                    "margin_min": min(margins),
                    "margin_max": max(margins),
                    "margin_mean": float(np.mean(margins)),
                    "unreachable": bool(unreachable[j]),
                    "status": "UNREACHABLE_COLLISION_EDGE" if unreachable[j] else "REACHABLE",
                }
            )
    state = D34CollisionLocalRegistrationState(
        **{**state0.__dict__, "unreachable_new_mask": unreachable}
    )
    old_loso_evaluated = old_k > 1
    old_loso_pass = bool(old_loso_evaluated and old_loso_intrusions == 0)
    degrees = np.sum(edge_mask, axis=0).astype(np.int64)
    c_new = len(new_classes)
    adaptation_macs = int(
        (len(old_x) + len(new_x)) * FEATURE_DIM
        + c_new * new_k * FEATURE_DIM
        + (
            c_new * new_k * new_k * FEATURE_DIM
            if locked.gamma_medoid > 0.0 and new_k > 1
            else 0
        )
        + len(old_x) * c_new * FEATURE_DIM
        + len(new_x) * len(old_classes)
    )
    old_loso_evidence_macs = int(
        0
        if old_k == 1
        else len(old_x) * len(old_x) * len(old_classes) * FEATURE_DIM
    )
    new_loso_per_row_macs = int(
        c_new * max(1, new_k - 1) * FEATURE_DIM
        + (
            c_new * max(1, new_k - 1) ** 2 * FEATURE_DIM
            if locked.gamma_medoid > 0.0 and new_k > 1
            else 0
        )
        + c_new * FEATURE_DIM
        + len(old_x) * c_new * FEATURE_DIM
        + int(np.max(degrees)) * (FEATURE_DIM + 2)
    )
    new_loso_evidence_macs = int(
        0 if new_k == 1 else len(new_x) * new_loso_per_row_macs
    )
    development_loso_evidence_macs = int(
        old_loso_evidence_macs + new_loso_evidence_macs
    )
    average_query_macs = float(np.mean(degrees) * (FEATURE_DIM + 2))
    worst_query_macs = int(np.max(degrees) * (FEATURE_DIM + 2))
    active_parameters = int(
        state.new_prototypes_qint8.size
        + state.new_prototype_scales.size
        + state.new_prototype_inverse_norms.size
        + state.collision_edge_mask.size
        + state.old_anchor_offsets.size
    )
    geometry = {
        "schema": "cvs.phase2.d34_collision_local_geometry.v1",
        "arm": locked.arm,
        "collision_edges": edge_trace,
        "collision_edge_count": int(np.sum(edge_mask)),
        "average_degree": float(np.mean(degrees)),
        "worst_degree": int(np.max(degrees)),
        "old_anchor_offset_trace": offset_trace,
        "old_uncertainty_trace": old_uncertainty_trace,
        "old_loso_trace": old_loso_trace,
        "new_loso_trace": new_loso_trace,
        "old_loso_intrusion_count": old_loso_intrusions,
        "old_loso_evaluated": old_loso_evaluated,
        "old_loso_status": "PASS" if old_loso_pass else (
            "FAIL" if old_loso_evaluated else "NOT_EVALUATED_K1"
        ),
        "old_loso_zero_intrusion_pass": old_loso_pass,
        "old_correct_support_intrusion_count": 0,
        "unreachable_edge_count": int(np.sum(unreachable)),
        "unreachable_new_classes": [
            new_classes[i] for i in np.flatnonzero(unreachable).tolist()
        ],
        "old_score_prefix_bitwise_preserved": bool(
            old_after[:, : len(old_classes)].tobytes() == old_prefix.tobytes()
        ),
        "uncertainty_formula": "u_i=max(LOO own-vs-other-old cosine margin)-min(same)",
    }
    resource = {
        "schema": RESOURCE_SCHEMA,
        "adaptation_mode": "EVAL_ONLY_CLOSED_FORM_ADAPTATION",
        "optimizer_steps": 0,
        "active_parameters": active_parameters,
        "active_parameter_cap": 50_000,
        "active_parameter_cap_pass": active_parameters < 50_000,
        "persistent_state_bytes": state.persistent_state_bytes,
        "estimated_adaptation_macs": adaptation_macs,
        "development_loso_evidence_macs_included": False,
        "adaptation_mac_scope": "final_deployable_support_refit_only",
        "estimated_old_loso_evidence_macs": old_loso_evidence_macs,
        "estimated_new_loso_evidence_macs": new_loso_evidence_macs,
        "estimated_development_loso_evidence_macs": development_loso_evidence_macs,
        "estimated_development_screen_total_macs": int(
            adaptation_macs + development_loso_evidence_macs
        ),
        "estimated_macs_per_query": worst_query_macs,
        "estimated_macs_per_query_average_degree": average_query_macs,
        "estimated_macs_per_query_worst_degree": worst_query_macs,
        "query_mac_scope": (
            "winner_adjacent_new_int8_dot_plus_inverse_norm_and_temperature_scale"
        ),
        "average_collision_degree": float(np.mean(degrees)),
        "worst_collision_degree": int(np.max(degrees)),
        "old_loso_intrusion_count": old_loso_intrusions,
        "old_loso_evaluated": old_loso_evaluated,
        "old_loso_zero_intrusion_pass": old_loso_pass,
        "dense_query_graph_bytes": 0,
        "resident_fp32_new_prototype_count": 0,
        "new_prototype_int8_bytes": int(state.new_prototypes_qint8.nbytes),
        "new_prototype_scale_fp32_bytes": int(state.new_prototype_scales.nbytes),
        "new_prototype_inverse_norm_fp32_bytes": int(state.new_prototype_inverse_norms.nbytes),
        "support_only": True,
        "query_rows_used_for_fit": 0,
        "query_features_used_for_fit": False,
        "query_labels_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "source_sample_access": False,
        "source_derived_signal_access": False,
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "single_received_iq_row_per_support_sample": True,
    }
    return D34CollisionLocalRegistrationResult(state, geometry, resource)


def score_d34_collision_local_registration(
    state: D34CollisionLocalRegistrationState,
    features: np.ndarray,
    old_score_prefix: np.ndarray,
) -> np.ndarray:
    """Return finite per-sample scores over all classes with old prefix intact."""

    rows = _rows(features, "D34 scoring features")
    prefix = _prefix(
        old_score_prefix, len(rows), state.old_class_count, "scoring old prefix"
    )
    scores = _compose_sparse(state, rows, prefix)
    if not np.isfinite(scores).all():
        raise D34CollisionLocalRegistrationError("non-finite D34 score")
    return scores


def predict_d34_collision_local_registration(
    state: D34CollisionLocalRegistrationState,
    features: np.ndarray,
    old_score_prefix: np.ndarray,
) -> np.ndarray:
    scores = score_d34_collision_local_registration(state, features, old_score_prefix)
    return np.asarray(state.classes)[np.argmax(scores, axis=1)]


__all__ = [
    "ALLOWED_NEW_CLASS_COUNTS",
    "D34CollisionLocalConfig",
    "D34CollisionLocalRegistrationError",
    "D34CollisionLocalRegistrationResult",
    "D34CollisionLocalRegistrationState",
    "fit_d34_collision_local_registration",
    "predict_d34_collision_local_registration",
    "score_d34_collision_local_registration",
]
