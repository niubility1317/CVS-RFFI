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


class D92D42TPCEError(ValueError):
    """Raised for structural TPCE input or lifecycle drift."""


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


def _aggregate_atomic_exchanges(
    state: d42.D42UnifiedShrinkageLDAState,
    rows: np.ndarray,
    relations: list[tuple[int, int, int]],
) -> tuple[np.ndarray, int, int]:
    q2 = np.asarray(state.coef2_qint8, dtype=np.int16)
    scales = np.asarray(state.scale2_fp16, dtype=np.float32)
    delta = np.zeros(q2.shape, dtype=np.int32)
    requested = 0
    used_relations = 0
    for row_index, true_class, competitor in relations:
        row_used = False
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
            delta[true_class, coordinate] += direction
            delta[competitor, coordinate] -= direction
            requested += 1
            row_used = True
        used_relations += int(row_used)
    return delta, requested, used_relations


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
    transient = int(
        2 * len(rows) * class_count * np.dtype(np.float32).itemsize
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
        )

    if not np.isfinite(rows).all():
        return fallback("support_nonfinite")
    try:
        base_scores = _score(state, rows)
        old_tails, new_tail, relations, tied_count = _fixed_tail_and_relations(
            base_scores, target_array, int(old_class_count)
        )
        delta, requested, _ = _aggregate_atomic_exchanges(state, rows, relations)
    except FloatingPointError:
        return fallback("support_score_nonfinite")
    old_counts = [int(len(value)) for value in old_tails]
    coordinate_comparisons = int(len(relations) * d42.FEATURE_DIM)
    if requested <= 0 or not np.any(delta):
        return fallback(
            "no_movable_code",
            requested=requested,
            old_counts=old_counts,
            new_count=int(len(new_tail)),
            tied_count=tied_count,
            coordinate_comparisons=coordinate_comparisons,
        )
    proposed = np.asarray(state.coef2_qint8, dtype=np.int32) + delta
    saturation = int(np.sum((proposed < -127) | (proposed > 127)))
    if saturation:
        return fallback(
            "aggregate_saturation",
            requested=requested,
            saturation=saturation,
            old_counts=old_counts,
            new_count=int(len(new_tail)),
            tied_count=tied_count,
            coordinate_comparisons=coordinate_comparisons,
        )
    candidate = replace(state, coef2_qint8=proposed.astype(np.int8))
    changed = int(np.sum(candidate.coef2_qint8 != state.coef2_qint8))
    if changed <= 0:
        return fallback(
            "zero_code_delta",
            requested=requested,
            old_counts=old_counts,
            new_count=int(len(new_tail)),
            tied_count=tied_count,
            coordinate_comparisons=coordinate_comparisons,
        )
    try:
        candidate_scores = _score(candidate, rows)
    except FloatingPointError:
        return fallback(
            "candidate_score_nonfinite",
            requested=requested,
            old_counts=old_counts,
            new_count=int(len(new_tail)),
            tied_count=tied_count,
            coordinate_comparisons=coordinate_comparisons,
        )

    base_all = _true_vs_all_margin(base_scores, target_array)
    candidate_all = _true_vs_all_margin(candidate_scores, target_array)
    base_cross = _true_new_vs_old_margin(base_scores, target_array, old_class_count)
    candidate_cross = _true_new_vs_old_margin(
        candidate_scores, target_array, old_class_count
    )
    old_gains = [
        float(np.mean(candidate_all[tail] - base_all[tail])) for tail in old_tails
    ]
    new_cross_gain = float(
        np.mean(candidate_cross[new_tail] - base_cross[new_tail])
    )
    new_all_gain = float(np.mean(candidate_all[new_tail] - base_all[new_tail]))
    base_hinges = _cross_group_hinges(base_scores, target_array, old_class_count)
    candidate_hinges = _cross_group_hinges(
        candidate_scores, target_array, old_class_count
    )
    old_to_new_delta = float(candidate_hinges[0] - base_hinges[0])
    new_to_old_delta = float(candidate_hinges[1] - base_hinges[1])
    scale = max(
        1.0,
        float(np.max(np.abs(base_scores))),
        float(np.max(np.abs(candidate_scores))),
    )
    tolerance = float(
        GUARD_EPSILON_MULTIPLIER * np.finfo(np.float32).eps * scale
    )
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
            requested=requested,
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
        )
    candidate_sha = _state_sha256(candidate)
    if candidate_sha == e0_sha:
        return fallback(
            "state_sha_unchanged",
            requested=requested,
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
        )
    audit = _base_audit(
        state,
        active=True,
        fallback_active=False,
        fallback_reason=None,
        final_state_sha256=candidate_sha,
        changed_code2_count=changed,
        requested_atomic_exchange_count=requested,
        applied_atomic_exchange_count=requested,
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
