"""Support-only D42 tail class-row ascent for the E0 FULL head.

TCRA freezes all tails on the E0 support score.  Its only atom increments one
true-class ``coef2_qint8`` cell by one signed quantum.  Analytic score deltas
rank candidates, while every tentatively accepted prefix and the final state
are materialized and scored through the real D42 decoder.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import Any

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42


QUANTILE = 0.20
QUANTILE_METHOD = "lower"
STATE_POSTPROCESS_MODE = "d42_tcra"
FINAL_GATE_REVISION = "safe_directional_v2"
GUARD_EPSILON_MULTIPLIER = 64.0


class D92D42TCRAError(ValueError):
    """Raised for structural TCRA input or lifecycle drift."""


class _AtomicAscent:
    """One class-row, one-block, one-code residual ascent."""

    __slots__ = (
        "true_class",
        "coordinate",
        "direction",
        "block_index",
        "semantic_handle",
    )

    def __init__(
        self,
        *,
        true_class: int,
        coordinate: int,
        direction: int,
        block_index: int,
        semantic_handle: tuple[Any, ...],
    ) -> None:
        self.true_class = int(true_class)
        self.coordinate = int(coordinate)
        self.direction = int(direction)
        self.block_index = int(block_index)
        self.semantic_handle = tuple(semantic_handle)


def _state_sha256(state: d42.D42UnifiedShrinkageLDAState) -> str:
    digest = hashlib.sha256()
    metadata = {
        "classes": list(state.classes),
        "covariance_policy": state.covariance_policy,
        "old_class_count": int(state.old_class_count),
        "schema": state.schema,
    }
    digest.update(
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name in (
        "log_diag_fp32",
        "coef1_qint8",
        "coef2_qint8",
        "scale1_fp16",
        "scale2_fp16",
        "intercept_fp16",
        "coef_fp32",
        "intercept_fp32",
    ):
        value = np.ascontiguousarray(getattr(state, name))
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def d42_tcra_state_sha256(state: d42.D42UnifiedShrinkageLDAState) -> str:
    """Return the canonical identity used by TCRA deployment receipts."""

    if not isinstance(state, d42.D42UnifiedShrinkageLDAState):
        raise D92D42TCRAError("TCRA state identity type drift")
    return _state_sha256(state)


def _row_bytes(row: np.ndarray) -> bytes:
    return np.ascontiguousarray(row, dtype="<f4").tobytes(order="C")


def _canonical_indices(rows: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1:
        raise D92D42TCRAError("TCRA canonical row index drift")
    ordered = sorted(selected.tolist(), key=lambda index: _row_bytes(rows[index]))
    return np.asarray(ordered, dtype=np.int64)


def _canonical_float64_sum(rows: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Sum rows after canonical row-content ordering in float64."""

    ordered = _canonical_indices(np.asarray(rows, dtype=np.float32), indices)
    if ordered.size == 0:
        raise D92D42TCRAError("TCRA canonical row sum is empty")
    return np.sum(
        np.asarray(rows[ordered], dtype=np.float64), axis=0, dtype=np.float64
    )


def _score(
    state: d42.D42UnifiedShrinkageLDAState, rows: np.ndarray
) -> np.ndarray:
    coefficient = np.asarray(d42.decode_d42_coefficients(state), dtype=np.float32)
    intercept = np.asarray(state.intercept_fp16, dtype=np.float32)
    result = np.asarray(rows, dtype=np.float32) @ coefficient.T
    result += intercept[None, :]
    if not np.isfinite(result).all():
        raise FloatingPointError("TCRA support scores became non-finite")
    return result


def _true_vs_all_margin(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    masked = np.asarray(scores, dtype=np.float64).copy()
    masked[np.arange(len(targets)), targets] = -np.inf
    return scores[np.arange(len(targets)), targets].astype(np.float64) - np.max(
        masked, axis=1
    )


def _true_new_vs_old_margin(
    scores: np.ndarray, targets: np.ndarray, old_class_count: int
) -> np.ndarray:
    return scores[np.arange(len(targets)), targets].astype(np.float64) - np.max(
        np.asarray(scores[:, :old_class_count], dtype=np.float64), axis=1
    )


def _lower_tail_indices(
    rows: np.ndarray, values: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise D92D42TCRAError("TCRA fixed tail is empty")
    threshold = float(
        np.quantile(
            np.asarray(values[selected], dtype=np.float64),
            QUANTILE,
            method=QUANTILE_METHOD,
        )
    )
    tail = selected[np.asarray(values[selected], dtype=np.float64) <= threshold]
    return _canonical_indices(rows, tail)


def _fixed_tails(
    rows: np.ndarray,
    scores: np.ndarray,
    targets: np.ndarray,
    old_class_count: int,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    all_margin = _true_vs_all_margin(scores, targets)
    old_tails = [
        _lower_tail_indices(
            rows, all_margin, np.flatnonzero(targets == class_index)
        )
        for class_index in range(old_class_count)
    ]
    new_indices = np.flatnonzero(targets >= old_class_count)
    new_cross = _true_new_vs_old_margin(scores, targets, old_class_count)
    pooled_new_tail = _lower_tail_indices(rows, new_cross, new_indices)
    class_tails: list[np.ndarray] = [*old_tails]
    class_tails.extend(
        _canonical_indices(
            rows,
            pooled_new_tail[targets[pooled_new_tail] == class_index],
        )
        for class_index in range(old_class_count, scores.shape[1])
    )
    return old_tails, pooled_new_tail, class_tails


def _semantic_class_handle(
    state: d42.D42UnifiedShrinkageLDAState,
    rows: np.ndarray,
    targets: np.ndarray,
    class_index: int,
) -> tuple[str, str]:
    digest = hashlib.sha256()
    digest.update(b"old" if class_index < state.old_class_count else b"new")
    for row_index in _canonical_indices(
        rows, np.flatnonzero(targets == class_index)
    ).tolist():
        digest.update(_row_bytes(rows[row_index]))
    for value in (
        state.coef1_qint8[class_index],
        state.coef2_qint8[class_index],
        state.scale1_fp16[class_index],
        state.scale2_fp16[class_index],
        state.intercept_fp16[class_index : class_index + 1],
    ):
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    # The immutable registry handle resolves the vanishingly rare exact
    # content collision without depending on the transient class-row index.
    return digest.hexdigest(), str(state.classes[class_index])


def _build_atomic_ascent_candidates(
    state: d42.D42UnifiedShrinkageLDAState,
    rows: np.ndarray,
    targets: np.ndarray,
    class_tails: list[np.ndarray],
) -> tuple[list[_AtomicAscent], int]:
    atoms: list[_AtomicAscent] = []
    coordinate_work = 0
    for class_index, tail in enumerate(class_tails):
        if len(tail) == 0:
            continue
        aggregate = _canonical_float64_sum(rows, tail)
        class_handle = _semantic_class_handle(state, rows, targets, class_index)
        for block_index, block in enumerate(d42.BLOCK_SLICES):
            values = np.asarray(aggregate[block], dtype=np.float64)
            coordinate_work += int(len(tail) * len(values) + len(values))
            local_coordinate = int(np.argmax(np.abs(values)))
            maximum = float(values[local_coordinate])
            if not math.isfinite(maximum) or maximum == 0.0:
                continue
            coordinate = int(block.start) + local_coordinate
            direction = 1 if maximum > 0.0 else -1
            atoms.append(
                _AtomicAscent(
                    true_class=class_index,
                    coordinate=coordinate,
                    direction=direction,
                    block_index=block_index,
                    semantic_handle=(
                        "old" if class_index < state.old_class_count else "new",
                        *class_handle,
                        int(block_index),
                        int(coordinate),
                        int(direction),
                    ),
                )
            )
    atoms.sort(key=lambda atom: atom.semantic_handle)
    return atoms, coordinate_work


def _canonical_mean(
    rows: np.ndarray, values: np.ndarray, indices: np.ndarray
) -> float:
    ordered = _canonical_indices(rows, indices)
    if ordered.size == 0:
        raise D92D42TCRAError("TCRA support group is empty")
    return float(np.mean(np.asarray(values[ordered], dtype=np.float64)))


def _cross_group_hinges(
    rows: np.ndarray,
    scores: np.ndarray,
    targets: np.ndarray,
    old_class_count: int,
) -> tuple[float, float]:
    old_rows = np.flatnonzero(targets < old_class_count)
    new_rows = np.flatnonzero(targets >= old_class_count)
    old_true = scores[old_rows, targets[old_rows]].astype(np.float64)
    new_true = scores[new_rows, targets[new_rows]].astype(np.float64)
    old_values = np.maximum(
        0.0,
        np.max(scores[old_rows, old_class_count:], axis=1).astype(np.float64)
        - old_true,
    )
    new_values = np.maximum(
        0.0,
        np.max(scores[new_rows, :old_class_count], axis=1).astype(np.float64)
        - new_true,
    )
    old_by_row = np.zeros(len(rows), dtype=np.float64)
    new_by_row = np.zeros(len(rows), dtype=np.float64)
    old_by_row[old_rows] = old_values
    new_by_row[new_rows] = new_values
    return (
        _canonical_mean(rows, old_by_row, old_rows),
        _canonical_mean(rows, new_by_row, new_rows),
    )


def _support_group_values(
    rows: np.ndarray,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    targets: np.ndarray,
    old_class_count: int,
    old_tails: list[np.ndarray],
    new_tail: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    base_all = _true_vs_all_margin(base_scores, targets)
    candidate_all = _true_vs_all_margin(candidate_scores, targets)
    base_cross = _true_new_vs_old_margin(base_scores, targets, old_class_count)
    candidate_cross = _true_new_vs_old_margin(
        candidate_scores, targets, old_class_count
    )
    all_delta = candidate_all - base_all
    cross_delta = candidate_cross - base_cross
    gains = np.asarray(
        [
            *(
                _canonical_mean(rows, all_delta, tail)
                for tail in old_tails
            ),
            _canonical_mean(rows, cross_delta, new_tail),
        ],
        dtype=np.float64,
    )
    pooled_new_all_gain = _canonical_mean(rows, all_delta, new_tail)
    base_hinges = _cross_group_hinges(
        rows, base_scores, targets, old_class_count
    )
    candidate_hinges = _cross_group_hinges(
        rows, candidate_scores, targets, old_class_count
    )
    return (
        gains,
        pooled_new_all_gain,
        float(candidate_hinges[0] - base_hinges[0]),
        float(candidate_hinges[1] - base_hinges[1]),
    )


def _prefix_guard_pass(
    gains: np.ndarray,
    all_gain: float,
    old_to_new: float,
    new_to_old: float,
    previous_gains: np.ndarray,
    previous_all_gain: float,
    previous_old_to_new: float,
    previous_new_to_old: float,
    tolerance: float,
) -> bool:
    """Require every real prefix to be Pareto-safe versus E0 and its parent."""

    return bool(
        np.all(gains >= previous_gains - tolerance)
        and all_gain >= previous_all_gain - tolerance
        and old_to_new <= previous_old_to_new + tolerance
        and new_to_old <= previous_new_to_old + tolerance
        and np.all(gains >= -tolerance)
        and all_gain >= -tolerance
        and old_to_new <= tolerance
        and new_to_old <= tolerance
    )


def _atom_score_delta(
    state: d42.D42UnifiedShrinkageLDAState,
    rows: np.ndarray,
    atom: _AtomicAscent,
) -> np.ndarray:
    scale = np.float32(
        np.asarray(state.scale2_fp16, dtype=np.float32)[
            atom.true_class, atom.block_index
        ]
    )
    return (
        np.asarray(rows[:, atom.coordinate], dtype=np.float32)
        * np.float32(atom.direction)
        * scale
    )


def _top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    """Return stable descending top indices for a small exact exclusion cache."""

    return np.argsort(-np.asarray(scores, dtype=np.float32), axis=1, kind="stable")[
        :, :count
    ]


def _max_excluding_true_and_candidate(
    scores: np.ndarray,
    targets: np.ndarray,
    candidate_columns: np.ndarray,
) -> np.ndarray:
    """Candidate-by-row max after excluding the true and edited columns."""

    top = _top_indices(scores, 3)
    result = np.full(
        (len(candidate_columns), len(scores)), -np.inf, dtype=np.float32
    )
    candidate_grid = candidate_columns[:, None]
    target_grid = targets[None, :]
    for rank in range(top.shape[1]):
        indices = top[:, rank][None, :]
        values = np.take_along_axis(
            scores, top[:, rank : rank + 1], axis=1
        )[:, 0][None, :]
        usable = (indices != candidate_grid) & (indices != target_grid)
        result = np.where(np.isneginf(result) & usable, values, result)
    if np.isneginf(result).any():
        raise D92D42TCRAError("TCRA analytic all-class max cache drift")
    return result


def _subset_max_excluding_candidate(
    scores: np.ndarray,
    candidate_columns: np.ndarray,
    *,
    start: int,
    stop: int,
) -> np.ndarray:
    """Candidate-by-row subset max after excluding one edited column."""

    subset = np.asarray(scores[:, start:stop], dtype=np.float32)
    top = _top_indices(subset, 2)
    first_index = top[:, 0] + int(start)
    first_value = np.take_along_axis(subset, top[:, :1], axis=1)[:, 0]
    second_value = np.take_along_axis(subset, top[:, 1:2], axis=1)[:, 0]
    return np.where(
        first_index[None, :] != candidate_columns[:, None],
        first_value[None, :],
        second_value[None, :],
    )


def _analytic_candidate_group_values_batch(
    state: d42.D42UnifiedShrinkageLDAState,
    rows: np.ndarray,
    targets: np.ndarray,
    old_class_count: int,
    base_scores: np.ndarray,
    current_scores: np.ndarray,
    old_tails: list[np.ndarray],
    new_tail: np.ndarray,
    atoms: list[_AtomicAscent],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score every one-column atom with O(batch x support rows) storage."""

    if not atoms:
        return (
            np.zeros((0, old_class_count + 1), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    candidate_columns = np.asarray(
        [atom.true_class for atom in atoms], dtype=np.int64
    )
    coordinates = np.asarray([atom.coordinate for atom in atoms], dtype=np.int64)
    directions = np.asarray([atom.direction for atom in atoms], dtype=np.float32)
    block_indices = np.asarray([atom.block_index for atom in atoms], dtype=np.int64)
    scales = np.asarray(state.scale2_fp16, dtype=np.float32)[
        candidate_columns, block_indices
    ]
    score_delta = (
        np.asarray(rows[:, coordinates], dtype=np.float32).T
        * directions[:, None]
        * scales[:, None]
    )
    row_indices = np.arange(len(rows), dtype=np.int64)
    target_match = targets[None, :] == candidate_columns[:, None]
    current_true = current_scores[row_indices, targets].astype(np.float32)
    candidate_true = current_true[None, :] + np.where(
        target_match, score_delta, np.float32(0.0)
    )
    changed_column = current_scores[:, candidate_columns].T + score_delta

    masked = np.asarray(current_scores, dtype=np.float32).copy()
    masked[row_indices, targets] = -np.inf
    current_max_excluding_true = np.max(masked, axis=1)
    unchanged_competitor = _max_excluding_true_and_candidate(
        current_scores, targets, candidate_columns
    )
    candidate_competitor = np.where(
        target_match,
        current_max_excluding_true[None, :],
        np.maximum(unchanged_competitor, changed_column),
    )
    candidate_all_margin = candidate_true.astype(np.float64) - (
        candidate_competitor.astype(np.float64)
    )
    base_all_margin = _true_vs_all_margin(base_scores, targets)
    all_delta = candidate_all_margin - base_all_margin[None, :]

    new_rows = np.flatnonzero(targets >= old_class_count)
    base_cross = _true_new_vs_old_margin(base_scores, targets, old_class_count)
    current_old_max = np.max(current_scores[:, :old_class_count], axis=1)
    candidate_old_max = np.broadcast_to(
        current_old_max[None, :], score_delta.shape
    ).copy()
    old_atom_mask = candidate_columns < old_class_count
    if np.any(old_atom_mask):
        old_columns = candidate_columns[old_atom_mask]
        old_unchanged = _subset_max_excluding_candidate(
            current_scores,
            old_columns,
            start=0,
            stop=old_class_count,
        )
        candidate_old_max[old_atom_mask] = np.maximum(
            old_unchanged, changed_column[old_atom_mask]
        )
    candidate_cross = candidate_true.astype(np.float64) - candidate_old_max.astype(
        np.float64
    )
    cross_delta = candidate_cross - base_cross[None, :]

    gains = np.empty((len(atoms), old_class_count + 1), dtype=np.float64)
    for class_index, tail in enumerate(old_tails):
        gains[:, class_index] = np.mean(
            all_delta[:, tail], axis=1, dtype=np.float64
        )
    gains[:, old_class_count] = np.mean(
        cross_delta[:, new_tail], axis=1, dtype=np.float64
    )
    new_all_gain = np.mean(all_delta[:, new_tail], axis=1, dtype=np.float64)

    old_rows = _canonical_indices(rows, np.flatnonzero(targets < old_class_count))
    canonical_new_rows = _canonical_indices(rows, new_rows)
    current_new_max = np.max(current_scores[:, old_class_count:], axis=1)
    candidate_new_max = np.broadcast_to(
        current_new_max[None, :], score_delta.shape
    ).copy()
    new_atom_mask = ~old_atom_mask
    if np.any(new_atom_mask):
        new_columns = candidate_columns[new_atom_mask]
        new_unchanged = _subset_max_excluding_candidate(
            current_scores,
            new_columns,
            start=old_class_count,
            stop=current_scores.shape[1],
        )
        candidate_new_max[new_atom_mask] = np.maximum(
            new_unchanged, changed_column[new_atom_mask]
        )
    candidate_old_hinge = np.maximum(
        0.0,
        candidate_new_max[:, old_rows].astype(np.float64)
        - candidate_true[:, old_rows].astype(np.float64),
    )
    candidate_new_hinge = np.maximum(
        0.0,
        candidate_old_max[:, canonical_new_rows].astype(np.float64)
        - candidate_true[:, canonical_new_rows].astype(np.float64),
    )
    base_hinges = _cross_group_hinges(
        rows, base_scores, targets, old_class_count
    )
    old_to_new = (
        np.mean(candidate_old_hinge, axis=1, dtype=np.float64) - base_hinges[0]
    )
    new_to_old = (
        np.mean(candidate_new_hinge, axis=1, dtype=np.float64) - base_hinges[1]
    )
    return gains, new_all_gain, old_to_new, new_to_old


def _atom_saturates(
    state: d42.D42UnifiedShrinkageLDAState, atom: _AtomicAscent
) -> bool:
    proposed = int(state.coef2_qint8[atom.true_class, atom.coordinate]) + int(
        atom.direction
    )
    return not -127 <= proposed <= 127


def _final_guard_pass(
    gains: np.ndarray,
    all_gain: float,
    old_to_new: float,
    new_to_old: float,
    tolerance: float,
) -> bool:
    values = np.asarray(gains, dtype=np.float64)
    scalars = np.asarray(
        [all_gain, old_to_new, new_to_old, tolerance], dtype=np.float64
    )
    if (
        values.ndim != 1
        or values.size < 2
        or not np.isfinite(values).all()
        or not np.isfinite(scalars).all()
        or tolerance < 0.0
    ):
        return False
    old_gains = values[:-1]
    return bool(
        np.all(values >= -tolerance)
        and all_gain >= -tolerance
        and old_to_new <= tolerance
        and new_to_old <= tolerance
        and np.max(old_gains) > tolerance
        and np.sum(old_gains, dtype=np.float64) > tolerance
    )


def _strict_search_target_pass(
    gains: np.ndarray,
    all_gain: float,
    old_to_new: float,
    new_to_old: float,
    tolerance: float,
) -> bool:
    """Preserve the frozen v1 greedy stopping target; only final acceptance changed."""

    return bool(
        np.all(np.asarray(gains, dtype=np.float64) > tolerance)
        and all_gain >= -tolerance
        and old_to_new <= tolerance
        and new_to_old <= tolerance
    )


def _base_audit(
    state: d42.D42UnifiedShrinkageLDAState,
    *,
    active: bool,
    fallback_active: bool,
    fallback_reason: str | None,
    final_state_sha256: str,
    modified_state_field_names: list[str],
    changed_code2_count: int,
    generated: int,
    selected: int,
    prefix_guard_rejected: int,
    greedy_steps: int,
    saturation: int,
    old_counts: list[int] | None,
    new_count: int | None,
    tolerance: float | None,
    old_gains: list[float] | None,
    new_cross_gain: float | None,
    new_all_gain: float | None,
    old_to_new_delta: float | None,
    new_to_old_delta: float | None,
    support_guard_pass: bool,
    full_score_evaluations: int,
    analytic_candidate_evaluations: int,
    score_macs: int,
    coordinate_work: int,
    support_macs: int,
    transient_bytes: int,
) -> dict[str, Any]:
    e0_sha = _state_sha256(state)
    rejected = int(generated - selected)
    old_gain_sum = (
        None
        if old_gains is None
        else float(
            np.sum(np.asarray(old_gains, dtype=np.float64), dtype=np.float64)
        )
    )
    old_strict_positive_count = (
        None
        if old_gains is None or tolerance is None
        else int(
            np.sum(
                np.asarray(old_gains, dtype=np.float64) > float(tolerance)
            )
        )
    )
    return {
        "d92_tcra_active": bool(active),
        "d92_tcra_fallback_active": bool(fallback_active),
        "d92_tcra_fallback_reason": fallback_reason,
        "d92_tcra_quantile": QUANTILE,
        "d92_tcra_quantile_method": QUANTILE_METHOD,
        "d92_tcra_state_postprocess_mode": STATE_POSTPROCESS_MODE,
        "d92_tcra_final_gate_revision": FINAL_GATE_REVISION,
        "d92_tcra_direct_state_publish": True,
        "d92_tcra_requantize_call_count": 0,
        "d92_tcra_e0_state_sha256": e0_sha,
        "d92_tcra_final_state_sha256": str(final_state_sha256),
        "d92_tcra_modified_state_field_names": list(modified_state_field_names),
        "d92_tcra_changed_code2_count": int(changed_code2_count),
        "d92_tcra_state_delta_code2_l1": int(changed_code2_count),
        "d92_tcra_requested_atomic_ascent_count": int(selected),
        "d92_tcra_applied_atomic_ascent_count": int(selected if active else 0),
        "d92_tcra_generated_atomic_ascent_count": int(generated),
        "d92_tcra_selected_atomic_ascent_count": int(selected),
        "d92_tcra_rejected_atomic_ascent_count": rejected,
        "d92_tcra_prefix_guard_rejected_count": int(prefix_guard_rejected),
        "d92_tcra_greedy_step_count": int(greedy_steps),
        "d92_tcra_aggregate_saturation_count": int(saturation),
        "d92_tcra_code1_byte_exact": True,
        "d92_tcra_scale1_byte_exact": True,
        "d92_tcra_scale2_byte_exact": True,
        "d92_tcra_intercept_byte_exact": True,
        "d92_tcra_log_diag_byte_exact": True,
        "d92_tcra_coef2_byte_exact": not active,
        "d92_tcra_old_tail_count_by_class": old_counts,
        "d92_tcra_pooled_new_tail_count": new_count,
        "d92_tcra_guard_tolerance": tolerance,
        "d92_tcra_old_tail_gain_by_class": old_gains,
        "d92_tcra_old_tail_min_gain": min(old_gains) if old_gains else None,
        "d92_tcra_old_tail_gain_sum": old_gain_sum,
        "d92_tcra_old_tail_strict_positive_count": old_strict_positive_count,
        "d92_tcra_pooled_new_cross_tail_gain": new_cross_gain,
        "d92_tcra_pooled_new_allclass_tail_gain": new_all_gain,
        "d92_tcra_old_to_new_hinge_delta": old_to_new_delta,
        "d92_tcra_new_to_old_hinge_delta": new_to_old_delta,
        "d92_tcra_support_guard_pass": bool(support_guard_pass),
        "d92_tcra_safe_directional_pass": bool(support_guard_pass),
        "d92_tcra_true_class_row_only": True,
        "d92_tcra_competitor_code_decrement_count": 0,
        "d92_tcra_class_permutation_equivariant": True,
        "d92_tcra_row_permutation_invariant": True,
        "d92_tcra_old_group_uniform_shift": False,
        "d92_tcra_support_full_score_evaluation_count": int(
            full_score_evaluations
        ),
        "d92_tcra_support_analytic_candidate_evaluation_count": int(
            analytic_candidate_evaluations
        ),
        "d92_tcra_support_score_macs_upper_bound": int(score_macs),
        "d92_tcra_support_coordinate_comparisons_upper_bound": int(
            coordinate_work
        ),
        "d92_tcra_support_macs_upper_bound": int(support_macs),
        "d92_tcra_support_transient_bytes_upper_bound": int(transient_bytes),
        "d92_tcra_persistent_state_bytes_delta": 0,
        "d92_tcra_component_fit_count": 0,
        "d92_tcra_query_rows_used": 0,
        "d92_tcra_query_macs": 0,
        "d92_tcra_query_fit_access": False,
        "d92_tcra_query_update_access": False,
        "d92_tcra_query_selection_access": False,
        "d92_tcra_query_truth_access": False,
        "d92_tcra_query_role_oracle_access": False,
        "d92_tcra_query_class_quota_access": False,
        "d92_tcra_query_global_reassignment": False,
    }


def d42_tcra_inactive_receipt(
    state: d42.D42UnifiedShrinkageLDAState,
) -> dict[str, Any]:
    sha = _state_sha256(state)
    return _base_audit(
        state,
        active=False,
        fallback_active=False,
        fallback_reason="K1_K2_EXACT_D92_FULL_ALIAS",
        final_state_sha256=sha,
        modified_state_field_names=[],
        changed_code2_count=0,
        generated=0,
        selected=0,
        prefix_guard_rejected=0,
        greedy_steps=0,
        saturation=0,
        old_counts=None,
        new_count=None,
        tolerance=None,
        old_gains=None,
        new_cross_gain=None,
        new_all_gain=None,
        old_to_new_delta=None,
        new_to_old_delta=None,
        support_guard_pass=False,
        full_score_evaluations=0,
        analytic_candidate_evaluations=0,
        score_macs=0,
        coordinate_work=0,
        support_macs=0,
        transient_bytes=0,
    )


def apply_d42_tail_class_row_ascent(
    state: d42.D42UnifiedShrinkageLDAState,
    transformed_support_rows: np.ndarray,
    targets: np.ndarray,
    *,
    old_class_count: int,
) -> tuple[d42.D42UnifiedShrinkageLDAState, dict[str, Any]]:
    """Return the directly edited D42 state or a byte-exact E0 fallback."""

    if not isinstance(state, d42.D42UnifiedShrinkageLDAState) or not state.is_int8:
        raise D92D42TCRAError("TCRA requires a compiled D42 int8 state")
    rows = np.asarray(transformed_support_rows, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.int64)
    class_count = len(state.classes)
    if (
        rows.ndim != 2
        or rows.shape[1] != d42.FEATURE_DIM
        or target_array.ndim != 1
        or len(rows) != len(target_array)
        or len(rows) == 0
        or int(old_class_count) != int(state.old_class_count)
        or not 2 <= int(old_class_count) < class_count
        or int(np.min(target_array)) < 0
        or int(np.max(target_array)) >= class_count
        or any(int(np.sum(target_array == index)) == 0 for index in range(class_count))
    ):
        raise D92D42TCRAError("TCRA support/state closure drift")

    row_count = len(rows)
    full_score_macs = int(row_count * class_count * d42.FEATURE_DIM)
    transient_bytes = int(
        rows.nbytes
        + 2 * row_count * class_count * np.dtype(np.float32).itemsize
        + class_count * d42.FEATURE_DIM * np.dtype(np.float32).itemsize
        + class_count * d42.FEATURE_DIM * np.dtype(np.int32).itemsize
        + 12 * row_count * np.dtype(np.float64).itemsize
    )
    e0_sha = _state_sha256(state)
    generated = 0
    selected_atoms: list[_AtomicAscent] = []
    prefix_rejected = 0
    saturation = 0
    steps = 0
    analytic_evaluations = 0
    full_score_evaluations = 0
    coordinate_work = 0
    old_counts: list[int] | None = None
    new_count: int | None = None
    tolerance: float | None = None
    current_gains: np.ndarray | None = None
    current_all_gain: float | None = None
    current_old_to_new: float | None = None
    current_new_to_old: float | None = None

    def resource_values() -> tuple[int, int]:
        score_macs = int(full_score_evaluations * full_score_macs)
        support_macs = int(
            score_macs + analytic_evaluations * row_count + coordinate_work
        )
        return score_macs, support_macs

    def fallback(reason: str) -> tuple[d42.D42UnifiedShrinkageLDAState, dict[str, Any]]:
        score_macs, support_macs = resource_values()
        old_gains = (
            None
            if current_gains is None
            else [float(value) for value in current_gains[: int(old_class_count)]]
        )
        return state, _base_audit(
            state,
            active=False,
            fallback_active=True,
            fallback_reason=str(reason),
            final_state_sha256=e0_sha,
            modified_state_field_names=[],
            changed_code2_count=0,
            generated=generated,
            selected=len(selected_atoms),
            prefix_guard_rejected=prefix_rejected,
            greedy_steps=steps,
            saturation=saturation,
            old_counts=old_counts,
            new_count=new_count,
            tolerance=tolerance,
            old_gains=old_gains,
            new_cross_gain=(
                None
                if current_gains is None
                else float(current_gains[int(old_class_count)])
            ),
            new_all_gain=current_all_gain,
            old_to_new_delta=current_old_to_new,
            new_to_old_delta=current_new_to_old,
            support_guard_pass=False,
            full_score_evaluations=full_score_evaluations,
            analytic_candidate_evaluations=analytic_evaluations,
            score_macs=score_macs,
            coordinate_work=coordinate_work,
            support_macs=support_macs,
            transient_bytes=transient_bytes,
        )

    if not np.isfinite(rows).all():
        return fallback("support_nonfinite")
    try:
        base_scores = _score(state, rows)
        full_score_evaluations += 1
        old_tails, new_tail, class_tails = _fixed_tails(
            rows, base_scores, target_array, int(old_class_count)
        )
        atoms, coordinate_work = _build_atomic_ascent_candidates(
            state, rows, target_array, class_tails
        )
    except FloatingPointError:
        return fallback("support_score_nonfinite")
    old_counts = [int(len(tail)) for tail in old_tails]
    new_count = int(len(new_tail))
    generated = int(len(atoms))
    # The batched analytic scorer retains several candidate-by-row float64
    # views at peak.  Count a deliberately conservative sixteen-view envelope.
    transient_bytes = int(
        transient_bytes
        + 16 * generated * row_count * np.dtype(np.float64).itemsize
    )
    if generated == 0:
        return fallback("no_movable_code")

    scale = max(1.0, float(np.max(np.abs(base_scores))))
    tolerance = float(
        GUARD_EPSILON_MULTIPLIER * np.finfo(np.float32).eps * scale
    )
    current_state = state
    current_scores = base_scores
    current_gains = np.zeros(int(old_class_count) + 1, dtype=np.float64)
    current_all_gain = 0.0
    current_old_to_new = 0.0
    current_new_to_old = 0.0
    remaining: list[_AtomicAscent] = []
    for atom in atoms:
        if _atom_saturates(state, atom):
            saturation += 1
        else:
            remaining.append(atom)

    while remaining and not _strict_search_target_pass(
        current_gains,
        current_all_gain,
        current_old_to_new,
        current_new_to_old,
        tolerance,
    ):
        uncovered = current_gains <= tolerance
        ranked: list[tuple[tuple[Any, ...], _AtomicAscent]] = []
        batch_gains, batch_all, batch_old_to_new, batch_new_to_old = (
            _analytic_candidate_group_values_batch(
                state,
                rows,
                target_array,
                int(old_class_count),
                base_scores,
                current_scores,
                old_tails,
                new_tail,
                remaining,
            )
        )
        analytic_evaluations += len(remaining)
        for candidate_index, atom in enumerate(remaining):
            gains = batch_gains[candidate_index]
            all_gain = float(batch_all[candidate_index])
            old_to_new = float(batch_old_to_new[candidate_index])
            new_to_old = float(batch_new_to_old[candidate_index])
            coverage = int(np.sum(uncovered & (gains > tolerance)))
            ranked.append(
                (
                    (
                        -coverage,
                        -float(np.min(gains)),
                        -float(np.sum(gains)),
                        atom.semantic_handle,
                    ),
                    atom,
                )
            )
        ranked.sort(key=lambda value: value[0])
        atom = ranked[0][1]
        remaining.remove(atom)
        steps += 1
        proposed_codes = np.array(current_state.coef2_qint8, dtype=np.int16, copy=True)
        proposed_codes[atom.true_class, atom.coordinate] += atom.direction
        if not -127 <= int(proposed_codes[atom.true_class, atom.coordinate]) <= 127:
            saturation += 1
            continue
        proposed_state = replace(
            state, coef2_qint8=proposed_codes.astype(np.int8, copy=False)
        )
        try:
            proposed_scores = _score(proposed_state, rows)
            full_score_evaluations += 1
        except FloatingPointError:
            prefix_rejected += 1
            continue
        gains, all_gain, old_to_new, new_to_old = _support_group_values(
            rows,
            base_scores,
            proposed_scores,
            target_array,
            int(old_class_count),
            old_tails,
            new_tail,
        )
        if not _prefix_guard_pass(
            gains,
            all_gain,
            old_to_new,
            new_to_old,
            current_gains,
            current_all_gain,
            current_old_to_new,
            current_new_to_old,
            tolerance,
        ):
            prefix_rejected += 1
            continue
        selected_atoms.append(atom)
        current_state = proposed_state
        current_scores = proposed_scores
        current_gains = gains
        current_all_gain = all_gain
        current_old_to_new = old_to_new
        current_new_to_old = new_to_old

    if not selected_atoms:
        reason = "aggregate_saturation" if saturation == generated else "no_pareto_safe_subset"
        return fallback(reason)
    if not _final_guard_pass(
        current_gains,
        current_all_gain,
        current_old_to_new,
        current_new_to_old,
        tolerance,
    ):
        return fallback("support_guard_failed")

    # Re-score the exact state that will be published; analytic values never
    # become deployment evidence.
    try:
        final_scores = _score(current_state, rows)
        full_score_evaluations += 1
    except FloatingPointError:
        return fallback("candidate_score_nonfinite")
    final_gains, final_all, final_old_to_new, final_new_to_old = (
        _support_group_values(
            rows,
            base_scores,
            final_scores,
            target_array,
            int(old_class_count),
            old_tails,
            new_tail,
        )
    )
    current_gains = final_gains
    current_all_gain = final_all
    current_old_to_new = final_old_to_new
    current_new_to_old = final_new_to_old
    if not _final_guard_pass(
        final_gains,
        final_all,
        final_old_to_new,
        final_new_to_old,
        tolerance,
    ):
        return fallback("support_guard_failed")

    candidate_sha = _state_sha256(current_state)
    changed = int(np.sum(current_state.coef2_qint8 != state.coef2_qint8))
    if candidate_sha == e0_sha or changed <= 0:
        return fallback("state_sha_unchanged")
    score_macs, support_macs = resource_values()
    old_gains = [float(value) for value in final_gains[: int(old_class_count)]]
    audit = _base_audit(
        state,
        active=True,
        fallback_active=False,
        fallback_reason=None,
        final_state_sha256=candidate_sha,
        modified_state_field_names=["coef2_qint8"],
        changed_code2_count=changed,
        generated=generated,
        selected=len(selected_atoms),
        prefix_guard_rejected=prefix_rejected,
        greedy_steps=steps,
        saturation=saturation,
        old_counts=old_counts,
        new_count=new_count,
        tolerance=tolerance,
        old_gains=old_gains,
        new_cross_gain=float(final_gains[int(old_class_count)]),
        new_all_gain=float(final_all),
        old_to_new_delta=float(final_old_to_new),
        new_to_old_delta=float(final_new_to_old),
        support_guard_pass=True,
        full_score_evaluations=full_score_evaluations,
        analytic_candidate_evaluations=analytic_evaluations,
        score_macs=score_macs,
        coordinate_work=coordinate_work,
        support_macs=support_macs,
        transient_bytes=transient_bytes,
    )
    return current_state, audit


__all__ = [
    "D92D42TCRAError",
    "FINAL_GATE_REVISION",
    "GUARD_EPSILON_MULTIPLIER",
    "QUANTILE",
    "QUANTILE_METHOD",
    "STATE_POSTPROCESS_MODE",
    "apply_d42_tail_class_row_ascent",
    "d42_tcra_inactive_receipt",
    "d42_tcra_state_sha256",
]
