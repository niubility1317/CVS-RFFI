"""D35 dense-safe support-only new-class registration.

Every registered new class is visible for every sample.  The caller supplies
the frozen old cosine-score prefix; D35 appends calibrated new scores without
modifying any old score bit.  Fitting accepts labeled LEO_weak support only and
has no query, role, quota, ordering, or batch-assignment interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


FEATURE_DIM = 288
LOGIT_SCALE = np.float32(18.0)
EPSILON = np.float32(1.0e-4)
ALLOWED_NEW_CLASS_COUNTS = (2, 5, 10, 20)
SCHEMA = "cvs.phase2.d35_dense_safe_registration.v1"


class D35DenseSafeRegistrationError(ValueError):
    """Raised when support, frozen scores, or method lock drift."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _unit_rows(value: np.ndarray, name: str) -> np.ndarray:
    """Validate the upstream FAST-adapted unit surface without changing it."""

    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D35DenseSafeRegistrationError(
            f"{name} must be finite [N,{FEATURE_DIM}]"
        )
    norms = np.linalg.norm(rows, axis=1).astype(np.float32)
    if not bool(np.all(np.abs(norms - np.float32(1.0)) <= np.float32(1.0e-4))):
        raise D35DenseSafeRegistrationError(
            f"{name} must already be FAST-adapted unit rows"
        )
    # Do not renormalize: the frozen old prefix and the D35 geometry must see
    # exactly the same upstream float32 row bytes.
    return np.ascontiguousarray(rows)


def _support(
    features: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    rows = _unit_rows(features, f"{name} features")
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
        raise D35DenseSafeRegistrationError(f"{name} registry drift")
    counts = [int(np.sum(y == name)) for name in registry]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise D35DenseSafeRegistrationError(
            f"{name} must be class-symmetric K-shot"
        )
    return rows, y, registry, counts[0]


def _prefix(value: np.ndarray, rows: int, classes: int, name: str) -> np.ndarray:
    scores = np.asarray(value)
    if (
        scores.dtype != np.float32
        or scores.shape != (rows, classes)
        or not np.isfinite(scores).all()
    ):
        raise D35DenseSafeRegistrationError(
            f"{name} must be finite float32 [{rows},{classes}]"
        )
    return np.ascontiguousarray(scores)


def _unit_mean(rows: np.ndarray) -> np.ndarray:
    mean = np.mean(rows, axis=0, dtype=np.float32)
    norm = np.float32(np.linalg.norm(mean))
    if norm <= np.float32(1.0e-12):
        raise D35DenseSafeRegistrationError("zero-norm prototype mean")
    return np.asarray(mean / norm, dtype=np.float32)


def _class_prototypes(rows: np.ndarray, count: int) -> np.ndarray:
    """Return one mean or two deterministic medoid-seeded cluster means."""

    if count == 1 or len(rows) == 1:
        return _unit_mean(rows)[None, :]
    similarities = np.asarray(rows @ rows.T, dtype=np.float32)
    first = int(np.argmax(np.mean(similarities, axis=1)))
    second = int(np.argmin(similarities[first]))
    assignment = np.argmax(similarities[:, [first, second]], axis=1)
    assignment[first] = 0
    assignment[second] = 1
    return np.stack([_unit_mean(rows[assignment == i]) for i in range(2)])


def _prototype_bank(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    max_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    bank = np.zeros((len(classes), max_count, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((len(classes), max_count), dtype=np.bool_)
    for j, name in enumerate(classes):
        values = _class_prototypes(rows[labels == name], max_count)
        bank[j, : len(values)] = values
        mask[j, : len(values)] = True
    return bank, mask


def _quantize(
    prototypes: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.zeros(prototypes.shape, dtype=np.int8)
    scales = np.ones(mask.shape, dtype=np.float32)
    inverse = np.ones(mask.shape, dtype=np.float32)
    for j, m in np.argwhere(mask):
        value = prototypes[j, m]
        scale = np.float32(np.max(np.abs(value)) / np.float32(127.0))
        quantized = np.asarray(
            np.clip(np.rint(value / scale), -127, 127), dtype=np.int8
        )
        q[j, m] = quantized
        scales[j, m] = scale
        inverse[j, m] = np.float32(
            1.0 / np.linalg.norm(quantized.astype(np.float32))
        )
    return q, scales, inverse


def _uncertainty(
    rows: np.ndarray, labels: np.ndarray, classes: tuple[str, ...]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    centroids = np.stack([_unit_mean(rows[labels == name]) for name in classes])
    result = np.empty(len(classes), dtype=np.float32)
    trace: list[dict[str, Any]] = []
    for i, name in enumerate(classes):
        class_rows = rows[labels == name]
        if len(class_rows) == 1:
            result[i] = np.float32(0.05)
            trace.append(
                {
                    "old_class": name,
                    "mode": "K1_NOT_EVALUATED_method_lock_u_i",
                    "u_i": 0.05,
                }
            )
            continue
        total = np.sum(class_rows, axis=0, dtype=np.float32)
        margins = []
        for row in class_rows:
            own = _unit_mean((total - row)[None, :])
            other = np.delete(centroids, i, axis=0)
            margins.append(float(row @ own) - float(np.max(row @ other.T)))
        spread = float(max(margins) - min(margins))
        result[i] = np.float32(spread)
        trace.append(
            {
                "old_class": name,
                "mode": "leave_one_old_out_margin_max_minus_min",
                "margin_min": min(margins),
                "margin_max": max(margins),
                "u_i": spread,
            }
        )
    return result, trace


@dataclass(frozen=True)
class D35DenseSafeConfig:
    arm: str = "B"

    def __post_init__(self) -> None:
        arm = str(self.arm).upper().removeprefix("D35-")
        if arm not in {"A", "B", "C"}:
            raise D35DenseSafeRegistrationError("D35 arm must be A, B, or C")
        object.__setattr__(self, "arm", arm)

    @property
    def prototypes_per_new_class(self) -> int:
        return 1 if self.arm == "A" else 2

    @property
    def old_buffer_lambda(self) -> float:
        return {"A": 0.0, "B": 0.25, "C": 0.5}[self.arm]


@dataclass(frozen=True)
class D35DenseSafeRegistrationState:
    schema: str
    classes: tuple[str, ...]
    old_class_count: int
    new_prototypes_qint8: np.ndarray
    new_prototype_scales: np.ndarray
    new_prototype_inverse_norms: np.ndarray
    prototype_mask: np.ndarray
    prototype_selector: np.ndarray
    safety_thresholds: np.ndarray
    support_count_by_new_class: np.ndarray
    arm: str
    optimizer_steps: int = 0

    def __post_init__(self) -> None:
        old_count = int(self.old_class_count)
        new_count = len(self.classes) - old_count
        max_proto = 1 if self.arm == "A" else 2
        if (
            self.schema != SCHEMA
            or not 2 <= old_count <= 6
            or new_count not in ALLOWED_NEW_CLASS_COUNTS
            or len(set(self.classes)) != len(self.classes)
            or self.new_prototypes_qint8.shape != (new_count, max_proto, FEATURE_DIM)
            or self.new_prototypes_qint8.dtype != np.int8
            or self.new_prototype_scales.shape != (new_count, max_proto)
            or self.new_prototype_scales.dtype != np.float32
            or self.new_prototype_inverse_norms.shape != (new_count, max_proto)
            or self.new_prototype_inverse_norms.dtype != np.float32
            or self.prototype_mask.shape != (new_count, max_proto)
            or self.prototype_mask.dtype != np.bool_
            or bool(np.any(np.sum(self.prototype_mask, axis=1) < 1))
            or self.prototype_selector.shape != (old_count, new_count)
            or self.prototype_selector.dtype != np.uint8
            or bool(np.any(self.prototype_selector >= max_proto))
            or bool(
                np.any(
                    ~self.prototype_mask[
                        np.arange(new_count)[None, :], self.prototype_selector
                    ]
                )
            )
            or self.safety_thresholds.shape != (old_count, new_count)
            or self.safety_thresholds.dtype != np.float32
            or not np.isfinite(self.safety_thresholds).all()
            or not np.isfinite(self.new_prototype_scales).all()
            or not np.isfinite(self.new_prototype_inverse_norms).all()
            or bool(
                np.any(self.new_prototype_scales[self.prototype_mask] <= 0.0)
            )
            or bool(
                np.any(self.new_prototype_inverse_norms[self.prototype_mask] <= 0.0)
            )
            or bool(
                np.any(
                    np.all(
                        self.new_prototypes_qint8[self.prototype_mask] == 0,
                        axis=1,
                    )
                )
            )
            or bool(np.any(self.new_prototypes_qint8[~self.prototype_mask] != 0))
            or bool(np.any(self.new_prototype_scales[~self.prototype_mask] != 1.0))
            or bool(np.any(self.new_prototype_inverse_norms[~self.prototype_mask] != 1.0))
            or self.support_count_by_new_class.shape != (new_count,)
            or self.support_count_by_new_class.dtype != np.uint16
            or int(self.optimizer_steps) != 0
        ):
            raise D35DenseSafeRegistrationError("D35 state drift")
        for field, dtype in (
            ("new_prototypes_qint8", np.int8),
            ("new_prototype_scales", np.float32),
            ("new_prototype_inverse_norms", np.float32),
            ("prototype_mask", np.bool_),
            ("prototype_selector", np.uint8),
            ("safety_thresholds", np.float32),
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
    def active_prototype_count(self) -> int:
        return int(np.sum(self.prototype_mask))

    @property
    def persistent_state_bytes(self) -> int:
        return int(
            self.new_prototypes_qint8.nbytes
            + self.new_prototype_scales.nbytes
            + self.new_prototype_inverse_norms.nbytes
            + self.prototype_mask.nbytes
            + self.prototype_selector.nbytes
            + self.safety_thresholds.nbytes
            + self.support_count_by_new_class.nbytes
        )


@dataclass(frozen=True)
class D35DenseSafeRegistrationResult:
    state: D35DenseSafeRegistrationState
    geometry_audit: dict[str, Any]
    resource_audit: dict[str, Any]


def _raw_logits(
    rows: np.ndarray,
    q: np.ndarray,
    inverse: np.ndarray,
) -> np.ndarray:
    flat_q = q.reshape(-1, FEATURE_DIM)
    flat_inverse = inverse.reshape(-1)
    values = np.asarray(rows @ flat_q.T, dtype=np.float32)
    values *= flat_inverse[None, :]
    values *= LOGIT_SCALE
    return values.reshape(len(rows), q.shape[0], q.shape[1])


def _thresholds(
    old_rows: np.ndarray,
    old_prefix: np.ndarray,
    q: np.ndarray,
    inverse: np.ndarray,
    mask: np.ndarray,
    selector: np.ndarray,
    uncertainty: np.ndarray,
    per_class_accuracy: np.ndarray,
    config: D35DenseSafeConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    old_count = old_prefix.shape[1]
    winner = np.argmax(old_prefix, axis=1)
    raw = _raw_logits(old_rows, q, inverse)
    threshold = np.zeros((old_count, q.shape[0]), dtype=np.float32)
    median_accuracy = float(np.median(per_class_accuracy))
    trace: list[dict[str, Any]] = []
    for i in range(old_count):
        selected = np.flatnonzero(winner == i)
        fallback = len(selected) == 0
        calibration = selected if len(selected) else np.arange(len(old_rows))
        floor_multiplier = (
            2.0
            if config.arm == "C" and per_class_accuracy[i] < median_accuracy
            else 1.0
        )
        buffer = float(EPSILON) + (
            config.old_buffer_lambda * float(uncertainty[i]) * floor_multiplier
        )
        for j in range(q.shape[0]):
            m = int(selector[i, j])
            threshold[i, j] = np.float32(
                np.max(raw[calibration, j, m] - old_prefix[calibration, i])
                + buffer
            )
        trace.append(
            {
                "old_winner_index": i,
                "calibration_rows": int(len(calibration)),
                "empty_winner_fallback_all_old_rows": fallback,
                "u_i": float(uncertainty[i]),
                "old_buffer_lambda": config.old_buffer_lambda,
                "floor_multiplier": floor_multiplier,
                "buffer": buffer,
            }
        )
    return threshold, trace


def _prototype_selector(
    new_rows: np.ndarray,
    new_labels: np.ndarray,
    new_classes: tuple[str, ...],
    new_old_prefix: np.ndarray,
    q: np.ndarray,
    inverse: np.ndarray,
    mask: np.ndarray,
    config: D35DenseSafeConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select one stored prototype for each old-winner/new-class pair."""

    old_count = new_old_prefix.shape[1]
    winners = np.argmax(new_old_prefix, axis=1)
    selector = np.zeros((old_count, len(new_classes)), dtype=np.uint8)
    trace: list[dict[str, Any]] = []
    for i in range(old_count):
        for j, name in enumerate(new_classes):
            class_rows = np.flatnonzero(new_labels == name)
            grouped = class_rows[winners[class_rows] == i]
            fallback = len(grouped) == 0
            calibration = class_rows if fallback else grouped
            active = np.flatnonzero(mask[j])
            if config.arm == "A" or len(active) == 1:
                chosen = int(active[0])
                mean_margins = [
                    float(
                        np.mean(
                            LOGIT_SCALE
                            * (new_rows[calibration] @ q[j, chosen].astype(np.float32))
                            * inverse[j, chosen]
                            - new_old_prefix[calibration, i]
                        )
                    )
                ]
            else:
                mean_margins = [
                    float(
                        np.mean(
                            LOGIT_SCALE
                            * (new_rows[calibration] @ q[j, m].astype(np.float32))
                            * inverse[j, m]
                            - new_old_prefix[calibration, i]
                        )
                    )
                    for m in active.tolist()
                ]
                chosen = int(active[int(np.argmax(mean_margins))])
            selector[i, j] = np.uint8(chosen)
            trace.append(
                {
                    "old_winner_index": i,
                    "new_class": name,
                    "calibration_rows": int(len(calibration)),
                    "empty_winner_group_fallback_all_new_class_support": fallback,
                    "active_candidate_indices": active.tolist(),
                    "mean_raw_margins": mean_margins,
                    "selected_prototype": chosen,
                }
            )
    return selector, trace


def _compose(
    state: D35DenseSafeRegistrationState,
    rows: np.ndarray,
    old_prefix: np.ndarray,
) -> np.ndarray:
    winners = np.argmax(old_prefix, axis=1)
    new_count = len(state.new_classes)
    new_scores = np.empty((len(rows), new_count), dtype=np.float32)
    class_indices = np.arange(new_count)
    # Winner grouping is only an arithmetic implementation detail.  Every new
    # class remains scored for every row, with exactly one selected dot product.
    for i in range(state.old_class_count):
        group = np.flatnonzero(winners == i)
        if not len(group):
            continue
        selected = state.prototype_selector[i]
        q = state.new_prototypes_qint8[class_indices, selected].astype(np.float32)
        inverse = state.new_prototype_inverse_norms[class_indices, selected]
        raw = np.asarray(rows[group] @ q.T, dtype=np.float32)
        raw *= inverse[None, :]
        raw *= LOGIT_SCALE
        new_scores[group] = raw - state.safety_thresholds[i][None, :]
    return _readonly(np.concatenate((old_prefix, new_scores), axis=1), np.float32)


def _build_state(
    old_x: np.ndarray,
    old_prefix: np.ndarray,
    old_classes: tuple[str, ...],
    old_y: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
    new_classes: tuple[str, ...],
    new_old_prefix: np.ndarray,
    config: D35DenseSafeConfig,
) -> tuple[
    D35DenseSafeRegistrationState,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    prototypes, mask = _prototype_bank(
        new_x, new_y, new_classes, config.prototypes_per_new_class
    )
    q, scales, inverse = _quantize(prototypes, mask)
    selector, selector_trace = _prototype_selector(
        new_x,
        new_y,
        new_classes,
        new_old_prefix,
        q,
        inverse,
        mask,
        config,
    )
    uncertainty, uncertainty_trace = _uncertainty(old_x, old_y, old_classes)
    targets = np.asarray([old_classes.index(str(v)) for v in old_y], dtype=np.int64)
    correct = np.argmax(old_prefix, axis=1) == targets
    accuracy = np.asarray(
        [np.mean(correct[targets == i]) for i in range(len(old_classes))],
        dtype=np.float32,
    )
    thresholds, threshold_trace = _thresholds(
        old_x,
        old_prefix,
        q,
        inverse,
        mask,
        selector,
        uncertainty,
        accuracy,
        config,
    )
    state = D35DenseSafeRegistrationState(
        schema=SCHEMA,
        classes=old_classes + new_classes,
        old_class_count=len(old_classes),
        new_prototypes_qint8=q,
        new_prototype_scales=scales,
        new_prototype_inverse_norms=inverse,
        prototype_mask=mask,
        prototype_selector=selector,
        safety_thresholds=thresholds,
        support_count_by_new_class=np.asarray(
            [np.sum(new_y == name) for name in new_classes], dtype=np.uint16
        ),
        arm=config.arm,
    )
    return state, uncertainty_trace, selector_trace, threshold_trace


def fit_d35_dense_safe_registration(
    old_support_features: np.ndarray,
    old_support_labels: Sequence[str],
    old_registered_classes: Sequence[str],
    old_support_score_prefix: np.ndarray,
    new_support_features: np.ndarray,
    new_support_labels: Sequence[str],
    new_registered_classes: Sequence[str],
    new_support_old_score_prefix: np.ndarray,
    *,
    config: D35DenseSafeConfig | None = None,
) -> D35DenseSafeRegistrationResult:
    """Fit the dense-safe head and support-only development LOO audits."""

    locked = config or D35DenseSafeConfig()
    old_x, old_y, old_classes, old_k = _support(
        old_support_features, old_support_labels, old_registered_classes, "D35 old support"
    )
    new_x, new_y, new_classes, new_k = _support(
        new_support_features, new_support_labels, new_registered_classes, "D35 new support"
    )
    if (
        not 2 <= len(old_classes) <= 6
        or len(new_classes) not in ALLOWED_NEW_CLASS_COUNTS
        or set(old_classes) & set(new_classes)
        or old_k != new_k
    ):
        raise D35DenseSafeRegistrationError(
            "D35 requires disjoint old/new matched class-symmetric K-shot support"
        )
    old_prefix = _prefix(
        old_support_score_prefix, len(old_x), len(old_classes), "old support prefix"
    )
    new_old_prefix = _prefix(
        new_support_old_score_prefix, len(new_x), len(old_classes), "new support old prefix"
    )
    state, uncertainty_trace, selector_trace, threshold_trace = _build_state(
        old_x,
        old_prefix,
        old_classes,
        old_y,
        new_x,
        new_y,
        new_classes,
        new_old_prefix,
        locked,
    )
    before = np.argmax(old_prefix, axis=1)
    targets = np.asarray([old_classes.index(str(v)) for v in old_y], dtype=np.int64)
    correct = before == targets
    old_scores = _compose(state, old_x, old_prefix)
    intrusions = int(np.sum(correct & (np.argmax(old_scores, axis=1) != before)))
    if intrusions:
        raise D35DenseSafeRegistrationError(
            "old correct-support non-degradation hard protection failed"
        )
    old_loso: list[dict[str, Any]] = []
    new_loso: list[dict[str, Any]] = []
    if old_k == 1:
        old_loso.append({"mode": "K1_NOT_EVALUATED", "status": "NOT_EVALUATED"})
        new_loso.append({"mode": "K1_NOT_EVALUATED", "status": "NOT_EVALUATED"})
    else:
        for held in range(len(old_x)):
            keep = np.arange(len(old_x)) != held
            fold_state, _, _, _ = _build_state(
                old_x[keep], old_prefix[keep], old_classes, old_y[keep],
                new_x, new_y, new_classes, new_old_prefix, locked,
            )
            scores = _compose(
                fold_state, old_x[held : held + 1], old_prefix[held : held + 1]
            )[0]
            old_loso.append(
                {
                    "held_index": held,
                    "old_class": str(old_y[held]),
                    "before_correct": bool(correct[held]),
                    "after_correct": bool(np.argmax(scores) == targets[held]),
                    "intruded": bool(correct[held] and np.argmax(scores) != targets[held]),
                }
            )
        for held in range(len(new_x)):
            keep = np.arange(len(new_x)) != held
            fold_state, _, _, _ = _build_state(
                old_x, old_prefix, old_classes, old_y,
                new_x[keep], new_y[keep], new_classes,
                new_old_prefix[keep], locked,
            )
            scores = _compose(
                fold_state, new_x[held : held + 1], new_old_prefix[held : held + 1]
            )[0]
            target = len(old_classes) + new_classes.index(str(new_y[held]))
            competitors = np.delete(scores, target)
            new_loso.append(
                {
                    "held_index": held,
                    "new_class": str(new_y[held]),
                    "correct": bool(np.argmax(scores) == target),
                    "margin": float(scores[target] - np.max(competitors)),
                }
            )
    p = state.active_prototype_count
    n_old, n_new = len(old_x), len(new_x)
    deploy_refit_macs = int(
        (n_old * len(old_classes) + n_new * p + n_old * p) * FEATURE_DIM
        + n_old * p
    )
    old_loso_macs = 0 if old_k == 1 else int(
        n_old * (((n_old - 1) * (len(old_classes) + p)) * FEATURE_DIM)
    )
    new_loso_macs = 0 if new_k == 1 else int(
        n_new * ((n_new - 1) * p + n_old * p) * FEATURE_DIM
    )
    selected_per_query = len(new_classes)
    prototype_dot_macs = int(selected_per_query * FEATURE_DIM)
    inverse_temperature_scalar_ops = int(2 * selected_per_query)
    threshold_subtraction_scalar_ops = int(selected_per_query)
    prototype_max_comparisons = 0
    old_winner_argmax_comparisons = int(len(old_classes) - 1)
    query_macs = prototype_dot_macs
    query_scalar_ops = int(
        inverse_temperature_scalar_ops
        + threshold_subtraction_scalar_ops
        + old_winner_argmax_comparisons
    )
    active_proto_count = int(np.sum(state.prototype_mask))
    active_parameters = int(
        active_proto_count * FEATURE_DIM
        + active_proto_count
        + active_proto_count
        + state.prototype_selector.size
        + state.safety_thresholds.size
        + state.support_count_by_new_class.size
    )
    geometry = {
        "schema": "cvs.phase2.d35_dense_safe_geometry.v1",
        "arm": locked.arm,
        "all_new_classes_global_visible": True,
        "global_visible_not_guaranteed_reachable": True,
        "visibility_gate": False,
        "nonedge_fallback": False,
        "logit_scale": float(LOGIT_SCALE),
        "uncertainty_trace": uncertainty_trace,
        "prototype_selector_trace": selector_trace,
        "threshold_trace": threshold_trace,
        "old_support_intrusion_count": intrusions,
        "old_support_non_degradation_pass": intrusions == 0,
        "old_score_prefix_bitwise_preserved": bool(
            old_scores[:, : len(old_classes)].tobytes() == old_prefix.tobytes()
        ),
        "old_leave_one_out": old_loso,
        "new_physical_leave_one_out": new_loso,
        "old_loso_intrusion_count": int(sum(bool(v.get("intruded", False)) for v in old_loso)),
        "k1_loso_status": "NOT_EVALUATED" if old_k == 1 else "EVALUATED",
    }
    resource = {
        "schema": "cvs.phase2.d35_dense_safe_resource.v1",
        "adaptation_mode": "EVAL_ONLY_CLOSED_FORM_ADAPTATION",
        "optimizer_steps": 0,
        "active_parameters": active_parameters,
        "persistent_state_bytes": state.persistent_state_bytes,
        "active_prototype_count": p,
        "estimated_deploy_refit_macs": deploy_refit_macs,
        "estimated_development_old_loso_macs": old_loso_macs,
        "estimated_development_new_loso_macs": new_loso_macs,
        "estimated_development_total_loso_macs": old_loso_macs + new_loso_macs,
        "estimated_macs_per_query": query_macs,
        "estimated_registration_macs_per_unit_query": query_macs,
        "query_mac_scope": "D35_registration_increment_only_on_FAST_unit_row",
        "query_selected_prototype_count": selected_per_query,
        "query_prototype_dot_macs": prototype_dot_macs,
        "estimated_scalar_ops_per_query": query_scalar_ops,
        "query_inverse_temperature_scalar_ops": inverse_temperature_scalar_ops,
        "query_threshold_subtraction_scalar_ops": threshold_subtraction_scalar_ops,
        "query_prototype_max_comparisons": prototype_max_comparisons,
        "query_old_winner_argmax_comparisons": old_winner_argmax_comparisons,
        "dense_query_graph_bytes": 0,
        "resident_fp32_new_prototype_count": 0,
        "new_prototype_int8_bytes": int(state.new_prototypes_qint8.nbytes),
        "new_prototype_scale_fp32_bytes": int(state.new_prototype_scales.nbytes),
        "new_prototype_inverse_norm_fp32_bytes": int(state.new_prototype_inverse_norms.nbytes),
        "prototype_selector_uint8_bytes": int(state.prototype_selector.nbytes),
        "safety_threshold_fp32_bytes": int(state.safety_thresholds.nbytes),
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
    return D35DenseSafeRegistrationResult(state, geometry, resource)


def score_d35_dense_safe_registration(
    state: D35DenseSafeRegistrationState,
    features: np.ndarray,
    old_score_prefix: np.ndarray,
) -> np.ndarray:
    rows = _unit_rows(features, "D35 scoring features")
    prefix = _prefix(old_score_prefix, len(rows), state.old_class_count, "scoring old prefix")
    scores = _compose(state, rows, prefix)
    if not np.isfinite(scores).all():
        raise D35DenseSafeRegistrationError("non-finite D35 score")
    return scores


def predict_d35_dense_safe_registration(
    state: D35DenseSafeRegistrationState,
    features: np.ndarray,
    old_score_prefix: np.ndarray,
) -> np.ndarray:
    scores = score_d35_dense_safe_registration(state, features, old_score_prefix)
    return np.asarray(state.classes)[np.argmax(scores, axis=1)]


__all__ = [
    "ALLOWED_NEW_CLASS_COUNTS",
    "D35DenseSafeConfig",
    "D35DenseSafeRegistrationError",
    "D35DenseSafeRegistrationResult",
    "D35DenseSafeRegistrationState",
    "fit_d35_dense_safe_registration",
    "predict_d35_dense_safe_registration",
    "score_d35_dense_safe_registration",
]
