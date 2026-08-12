"""Support-only D42 tail-pair code exchange for the E0 FULL head.

The method operates on the already compiled D42 deployment state.  It never
re-quantizes a continuous affine head: only the second residual int8 code is
eligible for a synchronous, class-symmetric update.
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
STATE_POSTPROCESS_MODE = "d42_tpce"
GUARD_EPSILON_MULTIPLIER = 64.0
GREEDY_CANDIDATE_BATCH_SIZE = 16


class D92D42TPCEError(ValueError):
    """Raised for structural TPCE input or lifecycle drift."""


class _AtomicExchange:
    """One support-derived, class-symmetric D42 residual-code exchange."""

    __slots__ = (
        "true_class",
        "competitor",
        "coordinate",
        "direction",
        "block_index",
        "stable_handle",
    )

    def __init__(
        self,
        *,
        true_class: int,
        competitor: int,
        coordinate: int,
        direction: int,
        block_index: int,
        stable_handle: tuple[Any, ...],
    ) -> None:
        self.true_class = int(true_class)
        self.competitor = int(competitor)
        self.coordinate = int(coordinate)
        self.direction = int(direction)
        self.block_index = int(block_index)
        self.stable_handle = tuple(stable_handle)


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


def d42_tpce_state_sha256(state: d42.D42UnifiedShrinkageLDAState) -> str:
    """Return the canonical identity used by TPCE deployment receipts."""

    if not isinstance(state, d42.D42UnifiedShrinkageLDAState):
        raise D92D42TPCEError("TPCE state identity type drift")
    return _state_sha256(state)


def _score(
    state: d42.D42UnifiedShrinkageLDAState, rows: np.ndarray
) -> np.ndarray:
    coefficient = np.asarray(d42.decode_d42_coefficients(state), dtype=np.float32)
    intercept = np.asarray(state.intercept_fp16, dtype=np.float32)
    result = np.asarray(rows, dtype=np.float32) @ coefficient.T
    result += intercept[None, :]
    if not np.isfinite(result).all():
        raise FloatingPointError("TPCE support scores became non-finite")
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


def _lower_tail_indices(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise D92D42TPCEError("TPCE fixed tail is empty")
    threshold = float(
        np.quantile(
            np.asarray(values[selected], dtype=np.float64),
            QUANTILE,
            method=QUANTILE_METHOD,
        )
    )
    return selected[np.asarray(values[selected], dtype=np.float64) <= threshold]


def _fixed_tail_and_relations(
    scores: np.ndarray,
    targets: np.ndarray,
    old_class_count: int,
) -> tuple[list[np.ndarray], np.ndarray, list[tuple[int, int, int]], int]:
    class_count = scores.shape[1]
    all_margin = _true_vs_all_margin(scores, targets)
    old_tails: list[np.ndarray] = []
    relations: list[tuple[int, int, int]] = []
    tied_relation_count = 0
    for class_index in range(old_class_count):
        tail = _lower_tail_indices(
            all_margin, np.flatnonzero(targets == class_index)
        )
        old_tails.append(tail)
        for row_index in tail.tolist():
            competitor_scores = np.asarray(scores[row_index], dtype=np.float64).copy()
            competitor_scores[class_index] = -np.inf
            maximum = float(np.max(competitor_scores))
            competitors = np.flatnonzero(competitor_scores == maximum)
            tied_relation_count += int(len(competitors))
            relations.extend(
                (int(row_index), class_index, int(competitor))
                for competitor in competitors.tolist()
            )

    new_indices = np.flatnonzero(targets >= old_class_count)
    new_cross_margin = _true_new_vs_old_margin(scores, targets, old_class_count)
    pooled_new_tail = _lower_tail_indices(new_cross_margin, new_indices)
    for row_index in pooled_new_tail.tolist():
        maximum = float(np.max(scores[row_index, :old_class_count]))
        competitors = np.flatnonzero(
            np.asarray(scores[row_index, :old_class_count], dtype=np.float64)
            == maximum
        )
        tied_relation_count += int(len(competitors))
        relations.extend(
            (int(row_index), int(targets[row_index]), int(competitor))
            for competitor in competitors.tolist()
        )
    if not relations or class_count <= old_class_count:
        raise D92D42TPCEError("TPCE tail relation closure drift")
    return old_tails, pooled_new_tail, relations, tied_relation_count


def _build_atomic_exchange_candidates(
    state: d42.D42UnifiedShrinkageLDAState,
    rows: np.ndarray,
    relations: list[tuple[int, int, int]],
) -> list[_AtomicExchange]:
    q2 = np.asarray(state.coef2_qint8, dtype=np.int16)
    scales = np.asarray(state.scale2_fp16, dtype=np.float32)
    atoms: list[_AtomicExchange] = []
    for row_index, true_class, competitor in relations:
        row_handle = hashlib.sha256(
            np.ascontiguousarray(rows[row_index], dtype=np.float32).tobytes()
        ).hexdigest()
        for block_index, block in enumerate(d42.BLOCK_SLICES):
            values = np.asarray(rows[row_index, block], dtype=np.float32)
            signs = np.sign(values).astype(np.int16)
            true_codes = q2[true_class, block]
            rival_codes = q2[competitor, block]
            movable = (
                (signs != 0)
                & (true_codes + signs <= 127)
                & (true_codes + signs >= -127)
                & (rival_codes - signs <= 127)
                & (rival_codes - signs >= -127)
            )
            if not np.any(movable):
                continue
            gains = np.where(
                movable,
                np.abs(values)
                * (scales[true_class, block_index] + scales[competitor, block_index]),
                -np.inf,
            )
            local_index = int(np.argmax(gains))
            gain = float(gains[local_index])
            if not math.isfinite(gain) or gain <= 0.0:
                continue
            coordinate = int(block.start) + local_index
            direction = int(signs[local_index])
            atoms.append(
                _AtomicExchange(
                    true_class=true_class,
                    competitor=competitor,
                    coordinate=coordinate,
                    direction=direction,
                    block_index=block_index,
                    stable_handle=(
                        str(state.classes[true_class]),
                        str(state.classes[competitor]),
                        int(coordinate),
                        int(direction),
                        int(block_index),
                        row_handle,
                    ),
                )
            )
    return atoms


def _aggregate_atomic_exchanges(
    state: d42.D42UnifiedShrinkageLDAState,
    rows: np.ndarray,
    relations: list[tuple[int, int, int]],
) -> tuple[np.ndarray, int, int]:
    """Retain the original all-atom view for regression/interference receipts."""

    delta = np.zeros(state.coef2_qint8.shape, dtype=np.int32)
    atoms = _build_atomic_exchange_candidates(state, rows, relations)
    for atom in atoms:
        delta[atom.true_class, atom.coordinate] += atom.direction
        delta[atom.competitor, atom.coordinate] -= atom.direction
    return delta, len(atoms), len(relations)


def _atom_is_publishable(
    state: d42.D42UnifiedShrinkageLDAState,
    delta: np.ndarray,
    atom: _AtomicExchange,
) -> tuple[bool, int]:
    """Check the two changed code cells without materializing a full candidate."""

    codes = np.asarray(state.coef2_qint8, dtype=np.int32)
    true_value = int(codes[atom.true_class, atom.coordinate]) + int(
        delta[atom.true_class, atom.coordinate]
    ) + atom.direction
    competitor_value = int(codes[atom.competitor, atom.coordinate]) + int(
        delta[atom.competitor, atom.coordinate]
    ) - atom.direction
    saturation = int(not -127 <= true_value <= 127) + int(
        not -127 <= competitor_value <= 127
    )
    return saturation == 0, saturation


def _apply_atom_in_place(delta: np.ndarray, atom: _AtomicExchange) -> None:
    delta[atom.true_class, atom.coordinate] += atom.direction
    delta[atom.competitor, atom.coordinate] -= atom.direction


def _cross_group_hinges(
    scores: np.ndarray, targets: np.ndarray, old_class_count: int
) -> tuple[float, float]:
    old_mask = targets < old_class_count
    new_mask = ~old_mask
    old_rows = np.flatnonzero(old_mask)
    new_rows = np.flatnonzero(new_mask)
    old_true = scores[old_rows, targets[old_rows]].astype(np.float64)
    new_true = scores[new_rows, targets[new_rows]].astype(np.float64)
    old_to_new = float(
        np.mean(
            np.maximum(
                0.0,
                np.max(scores[old_rows, old_class_count:], axis=1).astype(np.float64)
                - old_true,
            )
        )
    )
    new_to_old = float(
        np.mean(
            np.maximum(
                0.0,
                np.max(scores[new_rows, :old_class_count], axis=1).astype(np.float64)
                - new_true,
            )
        )
    )
    return old_to_new, new_to_old


def _support_group_values(
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
    gains = np.asarray(
        [
            *(
                float(np.mean(candidate_all[tail] - base_all[tail]))
                for tail in old_tails
            ),
            float(np.mean(candidate_cross[new_tail] - base_cross[new_tail])),
        ],
        dtype=np.float64,
    )
    pooled_new_all_gain = float(
        np.mean(candidate_all[new_tail] - base_all[new_tail])
    )
    base_hinges = _cross_group_hinges(base_scores, targets, old_class_count)
    candidate_hinges = _cross_group_hinges(candidate_scores, targets, old_class_count)
    return (
        gains,
        pooled_new_all_gain,
        float(candidate_hinges[0] - base_hinges[0]),
        float(candidate_hinges[1] - base_hinges[1]),
    )


def _candidate_codes(
    state: d42.D42UnifiedShrinkageLDAState, delta: np.ndarray
) -> tuple[np.ndarray | None, int]:
    proposed = np.asarray(state.coef2_qint8, dtype=np.int32) + delta
    saturation = int(np.sum((proposed < -127) | (proposed > 127)))
    if saturation:
        return None, saturation
    return proposed.astype(np.int8), 0


def _analytic_atom_score_delta(
    state: d42.D42UnifiedShrinkageLDAState,
    rows: np.ndarray,
    atom: _AtomicExchange,
) -> np.ndarray:
    """Return the exact linear D42 logit change for one residual-code atom."""

    result = np.zeros((len(rows), len(state.classes)), dtype=np.float32)
    scale = np.float32(
        np.asarray(state.scale2_fp16, dtype=np.float32)[
            atom.true_class, atom.block_index
        ]
    )
    competitor_scale = np.float32(
        np.asarray(state.scale2_fp16, dtype=np.float32)[
            atom.competitor, atom.block_index
        ]
    )
    value = np.asarray(rows[:, atom.coordinate], dtype=np.float32) * np.float32(
        atom.direction
    )
    result[:, atom.true_class] += value * scale
    result[:, atom.competitor] -= value * competitor_scale
    return result


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
    """Enforce the frozen no-regression guard for every published prefix."""

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


def _pareto_safe_greedy_subset_reference(
    state: d42.D42UnifiedShrinkageLDAState,
    rows: np.ndarray,
    targets: np.ndarray,
    old_class_count: int,
    base_scores: np.ndarray,
    old_tails: list[np.ndarray],
    new_tail: np.ndarray,
    atoms: list[_AtomicExchange],
    tolerance: float,
) -> tuple[
    np.ndarray,
    list[_AtomicExchange],
    int,
    np.ndarray,
    float,
    float,
    float,
    bool,
    np.ndarray,
]:
    """Select a deterministic safe subset without ever publishing a bad prefix.

    Candidates are assessed only on the frozen support tails and support hinges.
    The ranking is coverage of not-yet-positive groups, then worst-group and
    total gain; the immutable semantic handle resolves exact ties.
    """

    current_delta = np.zeros(state.coef2_qint8.shape, dtype=np.int32)
    current_gains = np.zeros(old_class_count + 1, dtype=np.float64)
    current_all_gain = 0.0
    current_old_to_new = 0.0
    current_new_to_old = 0.0
    remaining = list(atoms)
    selected: list[_AtomicExchange] = []
    rejected_saturation = 0
    current_scores = np.asarray(base_scores, dtype=np.float32).copy()
    while remaining:
        choices: list[
            tuple[
                int,
                float,
                float,
                tuple[Any, ...],
                _AtomicExchange,
                np.ndarray,
                float,
                float,
                float,
            ]
        ] = []
        for atom in remaining:
            publishable, saturation = _atom_is_publishable(state, current_delta, atom)
            if not publishable:
                rejected_saturation += saturation
                continue
            scores = (
                current_scores
                + _analytic_atom_score_delta(state, rows, atom)
            )
            gains, all_gain, old_to_new, new_to_old = _support_group_values(
                base_scores,
                scores,
                targets,
                old_class_count,
                old_tails,
                new_tail,
            )
            if (
                np.any(gains < current_gains - tolerance)
                or all_gain < current_all_gain - tolerance
                or old_to_new > current_old_to_new + tolerance
                or new_to_old > current_new_to_old + tolerance
                or np.any(gains < -tolerance)
                or all_gain < -tolerance
                or old_to_new > tolerance
                or new_to_old > tolerance
            ):
                continue
            newly_positive = int(
                np.sum((gains > tolerance) & ~(current_gains > tolerance))
            )
            choices.append(
                (
                    newly_positive,
                    float(np.min(gains)),
                    float(np.sum(gains)),
                    atom.stable_handle,
                    atom,
                    gains,
                    all_gain,
                    old_to_new,
                    new_to_old,
                )
            )
        if not choices:
            break
        choices.sort(key=lambda value: (-value[0], -value[1], -value[2], value[3]))
        (
            _coverage,
            _minimum,
            _total,
            _handle,
            chosen,
            current_gains,
            current_all_gain,
            current_old_to_new,
            current_new_to_old,
        ) = choices[0]
        _apply_atom_in_place(current_delta, chosen)
        codes, saturation = _candidate_codes(state, current_delta)
        if codes is None:
            return (
                np.zeros_like(current_delta),
                [],
                rejected_saturation + saturation,
                np.zeros_like(current_gains),
                0.0,
                0.0,
                0.0,
                True,
                np.asarray(base_scores, dtype=np.float32),
            )
        current_scores = _score(
            replace(state, coef2_qint8=codes), rows
        )
        actual_gains, actual_all_gain, actual_old_to_new, actual_new_to_old = (
            _support_group_values(
                base_scores,
                current_scores,
                targets,
                old_class_count,
                old_tails,
                new_tail,
            )
        )
        if not _prefix_guard_pass(
            actual_gains,
            actual_all_gain,
            actual_old_to_new,
            actual_new_to_old,
            current_gains,
            current_all_gain,
            current_old_to_new,
            current_new_to_old,
            tolerance,
        ):
            return (
                np.zeros_like(current_delta),
                [],
                rejected_saturation,
                np.zeros_like(current_gains),
                0.0,
                0.0,
                0.0,
                True,
                np.asarray(base_scores, dtype=np.float32),
            )
        current_gains = actual_gains
        current_all_gain = actual_all_gain
        current_old_to_new = actual_old_to_new
        current_new_to_old = actual_new_to_old
        selected.append(chosen)
        remaining.remove(chosen)
        if (
            np.all(current_gains > tolerance)
            and current_all_gain >= -tolerance
            and current_old_to_new <= tolerance
            and current_new_to_old <= tolerance
        ):
            break
    return (
        current_delta,
        selected,
        rejected_saturation,
        current_gains,
        current_all_gain,
        current_old_to_new,
        current_new_to_old,
        False,
        current_scores,
    )


def _top_score_cache(
    scores: np.ndarray, class_indices: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(class_indices, dtype=np.int64)
    count = min(int(width), int(len(selected)))
    if count <= 0:
        return (
            np.empty((len(scores), 0), dtype=np.float32),
            np.empty((len(scores), 0), dtype=np.int64),
        )
    group_scores = np.asarray(scores[:, selected], dtype=np.float32)
    order = np.argsort(-group_scores, axis=1, kind="stable")[:, :count]
    return (
        np.take_along_axis(group_scores, order, axis=1),
        selected[order],
    )


def _cached_unchanged_max(
    values: np.ndarray,
    indices: np.ndarray,
    *excluded: int | np.ndarray,
) -> np.ndarray:
    result = np.full(len(values), -np.inf, dtype=np.float32)
    unresolved = np.ones(len(values), dtype=bool)
    for column in range(values.shape[1]):
        candidate_index = indices[:, column]
        usable = unresolved.copy()
        for forbidden in excluded:
            usable &= candidate_index != forbidden
        result[usable] = values[usable, column]
        unresolved[usable] = False
        if not np.any(unresolved):
            break
    return result


def _atom_score_columns(
    state: d42.D42UnifiedShrinkageLDAState,
    rows: np.ndarray,
    atom: _AtomicExchange,
) -> tuple[np.ndarray, np.ndarray]:
    signed = np.asarray(rows[:, atom.coordinate], dtype=np.float32) * np.float32(
        atom.direction
    )
    scales = np.asarray(state.scale2_fp16, dtype=np.float32)
    return (
        signed * scales[atom.true_class, atom.block_index],
        -signed * scales[atom.competitor, atom.block_index],
    )


def _batched_cached_unchanged_max(
    values: np.ndarray,
    indices: np.ndarray,
    true_classes: np.ndarray,
    competitor_classes: np.ndarray,
    targets: np.ndarray | None = None,
) -> np.ndarray:
    """Return candidate-by-row maxima after excluding up to three classes."""

    count = len(true_classes)
    row_count = len(indices)
    result = np.full((count, row_count), -np.inf, dtype=np.float32)
    unresolved = np.ones((count, row_count), dtype=bool)
    true_column = np.asarray(true_classes, dtype=np.int64)[:, None]
    competitor_column = np.asarray(competitor_classes, dtype=np.int64)[:, None]
    target_row = None if targets is None else np.asarray(targets, dtype=np.int64)[None]
    for column in range(values.shape[1]):
        candidate_index = indices[None, :, column]
        usable = (candidate_index != true_column) & (
            candidate_index != competitor_column
        )
        if target_row is not None:
            usable &= candidate_index != target_row
        usable &= unresolved
        np.copyto(result, values[None, :, column], where=usable)
        unresolved[usable] = False
        if not np.any(unresolved):
            break
    return result


def _fast_candidate_group_values_batch(
    *,
    state: d42.D42UnifiedShrinkageLDAState,
    rows: np.ndarray,
    atoms: list[_AtomicExchange],
    current_scores: np.ndarray,
    targets: np.ndarray,
    old_class_count: int,
    old_rows: np.ndarray,
    new_rows: np.ndarray,
    old_tails: list[np.ndarray],
    new_tail: np.ndarray,
    base_all: np.ndarray,
    base_cross: np.ndarray,
    base_hinges: tuple[float, float],
    all_cache: tuple[np.ndarray, np.ndarray],
    old_cache: tuple[np.ndarray, np.ndarray],
    new_cache: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a bounded candidate batch with O(batch×N), not batch×N×C."""

    true_classes = np.asarray([atom.true_class for atom in atoms], dtype=np.int64)
    competitor_classes = np.asarray(
        [atom.competitor for atom in atoms], dtype=np.int64
    )
    coordinates = np.asarray([atom.coordinate for atom in atoms], dtype=np.int64)
    directions = np.asarray([atom.direction for atom in atoms], dtype=np.float32)
    blocks = np.asarray([atom.block_index for atom in atoms], dtype=np.int64)
    signed = np.asarray(rows[:, coordinates].T, dtype=np.float32)
    signed *= directions[:, None]
    scales = np.asarray(state.scale2_fp16, dtype=np.float32)
    true_column = np.asarray(
        current_scores[:, true_classes].T
        + signed * scales[true_classes, blocks, None],
        dtype=np.float32,
    )
    competitor_column = np.asarray(
        current_scores[:, competitor_classes].T
        - signed * scales[competitor_classes, blocks, None],
        dtype=np.float32,
    )
    current_true = np.asarray(
        current_scores[np.arange(len(rows)), targets], dtype=np.float32
    )
    true_scores = np.broadcast_to(current_true, true_column.shape).copy()
    target_rows = np.asarray(targets, dtype=np.int64)[None]
    np.copyto(
        true_scores,
        true_column,
        where=target_rows == true_classes[:, None],
    )
    np.copyto(
        true_scores,
        competitor_column,
        where=target_rows == competitor_classes[:, None],
    )

    all_max = _batched_cached_unchanged_max(
        all_cache[0],
        all_cache[1],
        true_classes,
        competitor_classes,
        targets,
    )
    np.maximum(
        all_max,
        true_column,
        out=all_max,
        where=target_rows != true_classes[:, None],
    )
    np.maximum(
        all_max,
        competitor_column,
        out=all_max,
        where=target_rows != competitor_classes[:, None],
    )
    old_max = _batched_cached_unchanged_max(
        old_cache[0], old_cache[1], true_classes, competitor_classes
    )
    new_max = _batched_cached_unchanged_max(
        new_cache[0], new_cache[1], true_classes, competitor_classes
    )
    true_old = true_classes < old_class_count
    competitor_old = competitor_classes < old_class_count
    if np.any(true_old):
        old_max[true_old] = np.maximum(
            old_max[true_old], true_column[true_old]
        )
    if np.any(~true_old):
        new_max[~true_old] = np.maximum(
            new_max[~true_old], true_column[~true_old]
        )
    if np.any(competitor_old):
        old_max[competitor_old] = np.maximum(
            old_max[competitor_old], competitor_column[competitor_old]
        )
    if np.any(~competitor_old):
        new_max[~competitor_old] = np.maximum(
            new_max[~competitor_old], competitor_column[~competitor_old]
        )

    gains = np.empty((len(atoms), old_class_count + 1), dtype=np.float64)
    for class_index, tail in enumerate(old_tails):
        gains[:, class_index] = np.mean(
            true_scores[:, tail].astype(np.float64)
            - all_max[:, tail].astype(np.float64)
            - base_all[tail],
            axis=1,
        )
    gains[:, old_class_count] = np.mean(
        true_scores[:, new_tail].astype(np.float64)
        - old_max[:, new_tail].astype(np.float64)
        - base_cross[new_tail],
        axis=1,
    )
    all_gain = np.mean(
        true_scores[:, new_tail].astype(np.float64)
        - all_max[:, new_tail].astype(np.float64)
        - base_all[new_tail],
        axis=1,
    )
    old_to_new = np.mean(
        np.maximum(
            0.0,
            new_max[:, old_rows].astype(np.float64)
            - true_scores[:, old_rows].astype(np.float64),
        ),
        axis=1,
    ) - base_hinges[0]
    new_to_old = np.mean(
        np.maximum(
            0.0,
            old_max[:, new_rows].astype(np.float64)
            - true_scores[:, new_rows].astype(np.float64),
        ),
        axis=1,
    ) - base_hinges[1]
    return gains, all_gain, old_to_new, new_to_old


def _pareto_safe_greedy_subset(
    state: d42.D42UnifiedShrinkageLDAState,
    rows: np.ndarray,
    targets: np.ndarray,
    old_class_count: int,
    base_scores: np.ndarray,
    old_tails: list[np.ndarray],
    new_tail: np.ndarray,
    atoms: list[_AtomicExchange],
    tolerance: float,
) -> tuple[
    np.ndarray,
    list[_AtomicExchange],
    int,
    np.ndarray,
    float,
    float,
    float,
    bool,
    np.ndarray,
]:
    """Select the reference greedy subset without per-candidate N×C arrays."""

    current_delta = np.zeros(state.coef2_qint8.shape, dtype=np.int32)
    current_scores = np.asarray(base_scores, dtype=np.float32).copy()
    base_all = _true_vs_all_margin(base_scores, targets)
    base_cross = _true_new_vs_old_margin(base_scores, targets, old_class_count)
    base_hinges = _cross_group_hinges(base_scores, targets, old_class_count)
    old_rows = np.flatnonzero(targets < old_class_count)
    new_rows = np.flatnonzero(targets >= old_class_count)
    old_indices = np.arange(old_class_count, dtype=np.int64)
    new_indices = np.arange(old_class_count, len(state.classes), dtype=np.int64)
    current_gains = np.zeros(old_class_count + 1, dtype=np.float64)
    current_all_gain = 0.0
    current_old_to_new = 0.0
    current_new_to_old = 0.0
    remaining = list(atoms)
    selected: list[_AtomicExchange] = []
    rejected_saturation = 0
    residual_codes = np.asarray(state.coef2_qint8, dtype=np.int32)
    while remaining:
        all_cache = _top_score_cache(current_scores, np.arange(len(state.classes)), 4)
        old_cache = _top_score_cache(current_scores, old_indices, 3)
        new_cache = _top_score_cache(current_scores, new_indices, 3)
        best: tuple[
            tuple[Any, ...],
            _AtomicExchange,
            np.ndarray,
            float,
            float,
            float,
        ] | None = None
        for start in range(0, len(remaining), GREEDY_CANDIDATE_BATCH_SIZE):
            chunk = remaining[start : start + GREEDY_CANDIDATE_BATCH_SIZE]
            true_classes = np.asarray(
                [atom.true_class for atom in chunk], dtype=np.int64
            )
            competitor_classes = np.asarray(
                [atom.competitor for atom in chunk], dtype=np.int64
            )
            coordinates = np.asarray([atom.coordinate for atom in chunk], dtype=np.int64)
            directions = np.asarray([atom.direction for atom in chunk], dtype=np.int32)
            saturation = (
                (residual_codes[true_classes, coordinates]
                 + current_delta[true_classes, coordinates]
                 + directions > 127)
                | (residual_codes[true_classes, coordinates]
                   + current_delta[true_classes, coordinates]
                   + directions < -127)
            ).astype(np.int64)
            saturation += (
                (residual_codes[competitor_classes, coordinates]
                 + current_delta[competitor_classes, coordinates]
                 - directions > 127)
                | (residual_codes[competitor_classes, coordinates]
                   + current_delta[competitor_classes, coordinates]
                   - directions < -127)
            ).astype(np.int64)
            rejected_saturation += int(np.sum(saturation))
            valid_indices = np.flatnonzero(saturation == 0)
            if not len(valid_indices):
                continue
            valid_atoms = [chunk[int(index)] for index in valid_indices]
            gains, all_gain, old_to_new, new_to_old = _fast_candidate_group_values_batch(
                state=state,
                rows=rows,
                atoms=valid_atoms,
                current_scores=current_scores,
                targets=targets,
                old_class_count=old_class_count,
                old_rows=old_rows,
                new_rows=new_rows,
                old_tails=old_tails,
                new_tail=new_tail,
                base_all=base_all,
                base_cross=base_cross,
                base_hinges=base_hinges,
                all_cache=all_cache,
                old_cache=old_cache,
                new_cache=new_cache,
            )
            safe = (
                ~np.any(gains < current_gains[None, :] - tolerance, axis=1)
                & (all_gain >= current_all_gain - tolerance)
                & (old_to_new <= current_old_to_new + tolerance)
                & (new_to_old <= current_new_to_old + tolerance)
                & ~np.any(gains < -tolerance, axis=1)
                & (all_gain >= -tolerance)
                & (old_to_new <= tolerance)
                & (new_to_old <= tolerance)
            )
            for local_index in np.flatnonzero(safe):
                index = int(local_index)
                atom = valid_atoms[index]
                coverage = int(
                    np.sum((gains[index] > tolerance) & ~(current_gains > tolerance))
                )
                key: tuple[Any, ...] = (
                    -coverage,
                    -float(np.min(gains[index])),
                    -float(np.sum(gains[index])),
                    atom.stable_handle,
                )
                if best is None or key < best[0]:
                    best = (
                        key,
                        atom,
                        gains[index],
                        float(all_gain[index]),
                        float(old_to_new[index]),
                        float(new_to_old[index]),
                    )
        if best is None:
            break
        (
            _key,
            chosen,
            _analytic_gains,
            _analytic_all_gain,
            _analytic_old_to_new,
            _analytic_new_to_old,
        ) = best
        _apply_atom_in_place(current_delta, chosen)
        codes, saturation = _candidate_codes(state, current_delta)
        if codes is None:
            return (
                np.zeros_like(current_delta),
                [],
                rejected_saturation + saturation,
                np.zeros_like(current_gains),
                0.0,
                0.0,
                0.0,
                True,
                np.asarray(base_scores, dtype=np.float32),
            )
        current_scores = _score(replace(state, coef2_qint8=codes), rows)
        actual_gains, actual_all_gain, actual_old_to_new, actual_new_to_old = (
            _support_group_values(
                base_scores,
                current_scores,
                targets,
                old_class_count,
                old_tails,
                new_tail,
            )
        )
        if not _prefix_guard_pass(
            actual_gains,
            actual_all_gain,
            actual_old_to_new,
            actual_new_to_old,
            current_gains,
            current_all_gain,
            current_old_to_new,
            current_new_to_old,
            tolerance,
        ):
            return (
                np.zeros_like(current_delta),
                [],
                rejected_saturation,
                np.zeros_like(current_gains),
                0.0,
                0.0,
                0.0,
                True,
                np.asarray(base_scores, dtype=np.float32),
            )
        current_gains = actual_gains
        current_all_gain = actual_all_gain
        current_old_to_new = actual_old_to_new
        current_new_to_old = actual_new_to_old
        selected.append(chosen)
        remaining.remove(chosen)
        if (
            np.all(current_gains > tolerance)
            and current_all_gain >= -tolerance
            and current_old_to_new <= tolerance
            and current_new_to_old <= tolerance
        ):
            break
    return (
        current_delta,
        selected,
        rejected_saturation,
        current_gains,
        current_all_gain,
        current_old_to_new,
        current_new_to_old,
        False,
        current_scores,
    )


def _base_audit(
    state: d42.D42UnifiedShrinkageLDAState,
    *,
    active: bool,
    fallback_active: bool,
    fallback_reason: str | None,
    final_state_sha256: str,
    changed_code2_count: int,
    requested_atomic_exchange_count: int,
    applied_atomic_exchange_count: int,
    aggregate_saturation_count: int,
    old_tail_count_by_class: list[int] | None,
    pooled_new_tail_count: int | None,
    tied_competitor_relation_count: int | None,
    guard_tolerance: float | None,
    old_tail_gain_by_class: list[float] | None,
    old_tail_min_gain: float | None,
    pooled_new_cross_tail_gain: float | None,
    pooled_new_allclass_tail_gain: float | None,
    old_to_new_hinge_delta: float | None,
    new_to_old_hinge_delta: float | None,
    support_guard_pass: bool,
    support_score_macs_upper_bound: int,
    support_coordinate_comparisons_upper_bound: int,
    support_transient_bytes_upper_bound: int,
    generated_atomic_exchange_count: int = 0,
    selected_atomic_exchange_count: int = 0,
    rejected_atomic_exchange_count: int = 0,
    greedy_step_count: int = 0,
) -> dict[str, Any]:
    e0_sha = _state_sha256(state)
    return {
        "d92_tpce_active": bool(active),
        "d92_tpce_fallback_active": bool(fallback_active),
        "d92_tpce_fallback_reason": fallback_reason,
        "d92_tpce_quantile": QUANTILE,
        "d92_tpce_quantile_method": QUANTILE_METHOD,
        "d92_tpce_state_postprocess_mode": STATE_POSTPROCESS_MODE,
        "d92_tpce_direct_state_publish": True,
        "d92_tpce_requantize_call_count": 0,
        "d92_tpce_e0_state_sha256": e0_sha,
        "d92_tpce_final_state_sha256": str(final_state_sha256),
        "d92_tpce_changed_code2_count": int(changed_code2_count),
        "d92_tpce_requested_atomic_exchange_count": int(
            requested_atomic_exchange_count
        ),
        "d92_tpce_applied_atomic_exchange_count": int(applied_atomic_exchange_count),
        "d92_tpce_aggregate_saturation_count": int(aggregate_saturation_count),
        "d92_tpce_generated_atomic_exchange_count": int(
            generated_atomic_exchange_count
        ),
        "d92_tpce_selected_atomic_exchange_count": int(
            selected_atomic_exchange_count
        ),
        "d92_tpce_rejected_atomic_exchange_count": int(
            rejected_atomic_exchange_count
        ),
        "d92_tpce_greedy_step_count": int(greedy_step_count),
        "d92_tpce_code1_byte_exact": True,
        "d92_tpce_scale1_byte_exact": True,
        "d92_tpce_scale2_byte_exact": True,
        "d92_tpce_intercept_byte_exact": True,
        "d92_tpce_log_diag_byte_exact": True,
        "d92_tpce_old_tail_count_by_class": old_tail_count_by_class,
        "d92_tpce_pooled_new_tail_count": pooled_new_tail_count,
        "d92_tpce_tied_competitor_relation_count": tied_competitor_relation_count,
        "d92_tpce_guard_tolerance": guard_tolerance,
        "d92_tpce_old_tail_gain_by_class": old_tail_gain_by_class,
        "d92_tpce_old_tail_min_gain": old_tail_min_gain,
        "d92_tpce_pooled_new_cross_tail_gain": pooled_new_cross_tail_gain,
        "d92_tpce_pooled_new_allclass_tail_gain": pooled_new_allclass_tail_gain,
        "d92_tpce_old_to_new_hinge_delta": old_to_new_hinge_delta,
        "d92_tpce_new_to_old_hinge_delta": new_to_old_hinge_delta,
        "d92_tpce_support_guard_pass": bool(support_guard_pass),
        "d92_tpce_class_permutation_equivariant": True,
        "d92_tpce_old_group_uniform_shift": False,
        "d92_tpce_support_score_macs_upper_bound": int(
            support_score_macs_upper_bound
        ),
        "d92_tpce_support_coordinate_comparisons_upper_bound": int(
            support_coordinate_comparisons_upper_bound
        ),
        "d92_tpce_support_macs_upper_bound": int(
            support_score_macs_upper_bound
        ),
        "d92_tpce_support_transient_bytes_upper_bound": int(
            support_transient_bytes_upper_bound
        ),
        "d92_tpce_persistent_state_bytes_delta": 0,
        "d92_tpce_component_fit_count": 0,
    }


def d42_tpce_inactive_receipt(
    state: d42.D42UnifiedShrinkageLDAState,
) -> dict[str, Any]:
    sha = _state_sha256(state)
    return _base_audit(
        state,
        active=False,
        fallback_active=False,
        fallback_reason="K1_K2_EXACT_D92_FULL_ALIAS",
        final_state_sha256=sha,
        changed_code2_count=0,
        requested_atomic_exchange_count=0,
        applied_atomic_exchange_count=0,
        aggregate_saturation_count=0,
        old_tail_count_by_class=None,
        pooled_new_tail_count=None,
        tied_competitor_relation_count=None,
        guard_tolerance=None,
        old_tail_gain_by_class=None,
        old_tail_min_gain=None,
        pooled_new_cross_tail_gain=None,
        pooled_new_allclass_tail_gain=None,
        old_to_new_hinge_delta=None,
        new_to_old_hinge_delta=None,
        support_guard_pass=False,
        support_score_macs_upper_bound=0,
        support_coordinate_comparisons_upper_bound=0,
        support_transient_bytes_upper_bound=0,
    )


def apply_d42_tail_pair_code_exchange(
    state: d42.D42UnifiedShrinkageLDAState,
    transformed_support_rows: np.ndarray,
    targets: np.ndarray,
    *,
    old_class_count: int,
) -> tuple[d42.D42UnifiedShrinkageLDAState, dict[str, Any]]:
    """Return the directly edited D42 state or an exact E0 fallback."""

    if not isinstance(state, d42.D42UnifiedShrinkageLDAState) or not state.is_int8:
        raise D92D42TPCEError("TPCE requires a compiled D42 int8 state")
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
        raise D92D42TPCEError("TPCE support/state closure drift")

    score_macs = int(2 * len(rows) * class_count * d42.FEATURE_DIM)
    # The inner greedy pass keeps E0 scores, its accumulated two-class score
    # delta, one analytic candidate delta, and temporary float64 margin views.
    # This deliberately counts a six-array float64-equivalent envelope rather
    # than the smaller retained arrays only.
    transient = int(
        6 * len(rows) * class_count * np.dtype(np.float64).itemsize
        + class_count * d42.FEATURE_DIM * np.dtype(np.int32).itemsize
        + class_count * d42.FEATURE_DIM * np.dtype(np.float32).itemsize
    )
    e0_sha = _state_sha256(state)

    def fallback(
        reason: str,
        *,
        requested: int = 0,
        saturation: int = 0,
        old_counts: list[int] | None = None,
        new_count: int | None = None,
        tied_count: int | None = None,
        coordinate_comparisons: int = 0,
        tolerance: float | None = None,
        old_gains: list[float] | None = None,
        new_cross_gain: float | None = None,
        new_all_gain: float | None = None,
        old_to_new_delta: float | None = None,
        new_to_old_delta: float | None = None,
        generated: int = 0,
        selected: int = 0,
        rejected: int = 0,
        greedy_steps: int = 0,
    ) -> tuple[d42.D42UnifiedShrinkageLDAState, dict[str, Any]]:
        return state, _base_audit(
            state,
            active=False,
            fallback_active=True,
            fallback_reason=str(reason),
            final_state_sha256=e0_sha,
            changed_code2_count=0,
            requested_atomic_exchange_count=requested,
            applied_atomic_exchange_count=0,
            aggregate_saturation_count=saturation,
            old_tail_count_by_class=old_counts,
            pooled_new_tail_count=new_count,
            tied_competitor_relation_count=tied_count,
            guard_tolerance=tolerance,
            old_tail_gain_by_class=old_gains,
            old_tail_min_gain=(min(old_gains) if old_gains else None),
            pooled_new_cross_tail_gain=new_cross_gain,
            pooled_new_allclass_tail_gain=new_all_gain,
            old_to_new_hinge_delta=old_to_new_delta,
            new_to_old_hinge_delta=new_to_old_delta,
            support_guard_pass=False,
            support_score_macs_upper_bound=score_macs,
            support_coordinate_comparisons_upper_bound=coordinate_comparisons,
            support_transient_bytes_upper_bound=transient,
            generated_atomic_exchange_count=generated,
            selected_atomic_exchange_count=selected,
            rejected_atomic_exchange_count=rejected,
            greedy_step_count=greedy_steps,
        )

    if not np.isfinite(rows).all():
        return fallback("support_nonfinite")
    try:
        base_scores = _score(state, rows)
        old_tails, new_tail, relations, tied_count = _fixed_tail_and_relations(
            base_scores, target_array, int(old_class_count)
        )
        atoms = _build_atomic_exchange_candidates(state, rows, relations)
    except FloatingPointError:
        return fallback("support_score_nonfinite")
    old_counts = [int(len(value)) for value in old_tails]
    coordinate_comparisons = int(len(relations) * d42.FEATURE_DIM)
    generated = int(len(atoms))
    transient = int(
        transient
        + generated * 128
    )
    # E0, each accepted exact prefix and the final state each use a decoded
    # full-head support score.  Candidate ranking uses exact two-column
    # analytic deltas in bounded batches; margin/hinge comparisons are not MACs.
    score_macs = int(
        score_macs
        + len(rows) * 2 * (generated * (generated + 1) // 2)
    )
    if generated <= 0:
        return fallback(
            "no_movable_code",
            old_counts=old_counts,
            new_count=int(len(new_tail)),
            tied_count=tied_count,
            coordinate_comparisons=coordinate_comparisons,
        )
    scale = max(1.0, float(np.max(np.abs(base_scores))))
    tolerance = float(GUARD_EPSILON_MULTIPLIER * np.finfo(np.float32).eps * scale)
    (
        delta,
        selected_atoms,
        rejected_saturation,
        _analytic_gains,
        _analytic_all_gain,
        _analytic_old_to_new,
        _analytic_new_to_old,
        prefix_guard_failed,
        _prefix_scores,
    ) = _pareto_safe_greedy_subset(
        state,
        rows,
        target_array,
        int(old_class_count),
        base_scores,
        old_tails,
        new_tail,
        atoms,
        tolerance,
    )
    selected = int(len(selected_atoms))
    rejected = int(generated - selected)
    score_macs = int(score_macs + selected * len(rows) * class_count * d42.FEATURE_DIM)
    if prefix_guard_failed:
        return fallback(
            "prefix_support_guard_failed",
            old_counts=old_counts,
            new_count=int(len(new_tail)),
            tied_count=tied_count,
            coordinate_comparisons=coordinate_comparisons,
            generated=generated,
            rejected=rejected,
        )
    if selected <= 0 or not np.any(delta):
        return fallback(
            "aggregate_saturation" if rejected_saturation else "no_pareto_safe_subset",
            saturation=rejected_saturation,
            old_counts=old_counts,
            new_count=int(len(new_tail)),
            tied_count=tied_count,
            coordinate_comparisons=coordinate_comparisons,
            generated=generated,
            rejected=rejected,
        )
    codes, saturation = _candidate_codes(state, delta)
    if codes is None:
        return fallback(
            "aggregate_saturation",
            saturation=saturation,
            old_counts=old_counts,
            new_count=int(len(new_tail)),
            tied_count=tied_count,
            coordinate_comparisons=coordinate_comparisons,
            generated=generated,
            selected=selected,
            rejected=rejected,
            greedy_steps=selected,
        )
    candidate = replace(state, coef2_qint8=codes)
    changed = int(np.sum(candidate.coef2_qint8 != state.coef2_qint8))
    if changed <= 0:
        return fallback(
            "zero_code_delta",
            old_counts=old_counts,
            new_count=int(len(new_tail)),
            tied_count=tied_count,
            coordinate_comparisons=coordinate_comparisons,
            generated=generated,
            selected=selected,
            rejected=rejected,
            greedy_steps=selected,
        )
    try:
        candidate_scores = _score(candidate, rows)
    except FloatingPointError:
        return fallback(
            "candidate_score_nonfinite",
            old_counts=old_counts,
            new_count=int(len(new_tail)),
            tied_count=tied_count,
            coordinate_comparisons=coordinate_comparisons,
            generated=generated,
            selected=selected,
            rejected=rejected,
            greedy_steps=selected,
        )
    gains, new_all_gain, old_to_new_delta, new_to_old_delta = _support_group_values(
        base_scores,
        candidate_scores,
        target_array,
        int(old_class_count),
        old_tails,
        new_tail,
    )
    old_gains = [float(value) for value in gains[:old_class_count]]
    new_cross_gain = float(gains[old_class_count])
    scale = max(scale, float(np.max(np.abs(candidate_scores))))
    tolerance = float(GUARD_EPSILON_MULTIPLIER * np.finfo(np.float32).eps * scale)
    guard = bool(
        all(value > tolerance for value in old_gains)
        and new_cross_gain > tolerance
        and new_all_gain >= -tolerance
        and old_to_new_delta <= tolerance
        and new_to_old_delta <= tolerance
    )
    if not guard:
        return fallback(
            "support_guard_failed",
            old_counts=old_counts,
            new_count=int(len(new_tail)),
            tied_count=tied_count,
            coordinate_comparisons=coordinate_comparisons,
            tolerance=tolerance,
            old_gains=old_gains,
            new_cross_gain=new_cross_gain,
            new_all_gain=new_all_gain,
            old_to_new_delta=old_to_new_delta,
            new_to_old_delta=new_to_old_delta,
            generated=generated,
            selected=selected,
            rejected=rejected,
            greedy_steps=selected,
        )
    candidate_sha = _state_sha256(candidate)
    if candidate_sha == e0_sha:
        return fallback(
            "state_sha_unchanged",
            old_counts=old_counts,
            new_count=int(len(new_tail)),
            tied_count=tied_count,
            coordinate_comparisons=coordinate_comparisons,
            tolerance=tolerance,
            old_gains=old_gains,
            new_cross_gain=new_cross_gain,
            new_all_gain=new_all_gain,
            old_to_new_delta=old_to_new_delta,
            new_to_old_delta=new_to_old_delta,
            generated=generated,
            selected=selected,
            rejected=rejected,
            greedy_steps=selected,
        )
    audit = _base_audit(
        state,
        active=True,
        fallback_active=False,
        fallback_reason=None,
        final_state_sha256=candidate_sha,
        changed_code2_count=changed,
        requested_atomic_exchange_count=selected,
        applied_atomic_exchange_count=selected,
        aggregate_saturation_count=0,
        old_tail_count_by_class=old_counts,
        pooled_new_tail_count=int(len(new_tail)),
        tied_competitor_relation_count=tied_count,
        guard_tolerance=tolerance,
        old_tail_gain_by_class=old_gains,
        old_tail_min_gain=float(min(old_gains)),
        pooled_new_cross_tail_gain=new_cross_gain,
        pooled_new_allclass_tail_gain=new_all_gain,
        old_to_new_hinge_delta=old_to_new_delta,
        new_to_old_hinge_delta=new_to_old_delta,
        support_guard_pass=True,
        support_score_macs_upper_bound=score_macs,
        support_coordinate_comparisons_upper_bound=coordinate_comparisons,
        support_transient_bytes_upper_bound=transient,
        generated_atomic_exchange_count=generated,
        selected_atomic_exchange_count=selected,
        rejected_atomic_exchange_count=rejected,
        greedy_step_count=selected,
    )
    return candidate, audit


__all__ = [
    "D92D42TPCEError",
    "GUARD_EPSILON_MULTIPLIER",
    "QUANTILE",
    "QUANTILE_METHOD",
    "STATE_POSTPROCESS_MODE",
    "apply_d42_tail_pair_code_exchange",
    "d42_tpce_inactive_receipt",
    "d42_tpce_state_sha256",
]
