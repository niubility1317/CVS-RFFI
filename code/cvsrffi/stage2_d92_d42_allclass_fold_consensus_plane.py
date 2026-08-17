"""Support-only all-class fold-consensus D42 residual-code plane."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42


STATE_POSTPROCESS_MODE = "d42_allclass_fold_consensus_plane"
FORMULA_REVISION = "allclass_fold_consensus_plane_v1"
SUPPORT_ROW_CANONICALIZATION = (
    "float32_row_bytes_then_float64_row_bytes_duplicate_class_handle_fail_closed"
)
FOLD_RULE = "per_class_canonical_alternating_twofold"
FOLD_TIE_POLICY = "coordinate_or_selection_boundary_fail_closed"


class D92D42AFCPError(ValueError):
    """Raised when the frozen AFCP state or support closure drifts."""


def d42_afcp_state_sha256(state: d42.D42UnifiedShrinkageLDAState) -> str:
    """Return the canonical deployment-state identity used by AFCP receipts."""

    if not isinstance(state, d42.D42UnifiedShrinkageLDAState):
        raise D92D42AFCPError("AFCP state identity type drift")
    digest = hashlib.sha256()
    metadata = {
        "classes": list(state.classes),
        "covariance_policy": state.covariance_policy,
        "old_class_count": int(state.old_class_count),
        "schema": state.schema,
    }
    digest.update(
        json.dumps(
            metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
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
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def d42_afcp_inactive_receipt(
    state: d42.D42UnifiedShrinkageLDAState,
    *,
    k_shot: int,
    old_class_count: int,
) -> dict[str, Any]:
    """Return the exact-E0 AFCP alias receipt without reading support rows."""

    if not isinstance(state, d42.D42UnifiedShrinkageLDAState) or not state.is_int8:
        raise D92D42AFCPError("AFCP inactive state identity drift")
    if int(k_shot) < 1:
        raise D92D42AFCPError("AFCP inactive K-shot drift")
    if int(old_class_count) < 1 or int(old_class_count) > len(state.classes):
        raise D92D42AFCPError("AFCP inactive old-class count drift")
    if int(old_class_count) == len(state.classes):
        reason = "NOT_REGISTERED_STATE"
    elif int(old_class_count) == int(state.old_class_count):
        reason = "K1_K2_EXACT_D92_FULL_ALIAS"
    else:
        raise D92D42AFCPError("AFCP inactive old-class registry drift")
    return _base_receipt(
        state,
        state,
        active=False,
        fallback_active=False,
        fallback_reason=reason,
        k_shot=int(k_shot),
        old_class_count=int(old_class_count),
    )


def _state_byte_equal(
    e0_state: d42.D42UnifiedShrinkageLDAState,
    candidate_state: d42.D42UnifiedShrinkageLDAState,
    name: str,
) -> bool:
    return bool(
        np.ascontiguousarray(np.asarray(getattr(e0_state, name))).tobytes()
        == np.ascontiguousarray(np.asarray(getattr(candidate_state, name))).tobytes()
    )


def _static_resource_values(
    state: d42.D42UnifiedShrinkageLDAState, *, k_shot: int
) -> dict[str, int]:
    """Return the fixed streaming work bound; no 288-square buffer is live."""

    class_count = int(len(state.classes))
    feature_dim = int(d42.FEATURE_DIM)
    row_count = class_count * int(k_shot)
    rows_bytes = row_count * feature_dim * np.dtype(np.float32).itemsize
    score_bytes = row_count * class_count * np.dtype(np.float32).itemsize
    direction_bytes = 4 * class_count * feature_dim * np.dtype(np.float64).itemsize
    graph_bytes = 2 * class_count * class_count * np.dtype(np.float64).itemsize
    code_bytes = class_count * feature_dim * np.dtype(np.int16).itemsize
    decoded_bytes = class_count * feature_dim * np.dtype(np.float32).itemsize
    return {
        "support_macs_upper_bound": 2
        * row_count
        * class_count
        * feature_dim,
        "support_rows_bytes_upper_bound": rows_bytes,
        "support_direction_bytes_upper_bound": direction_bytes,
        "support_graph_bytes_upper_bound": graph_bytes,
        "support_288_square_matrix_bytes": 0,
        "support_transient_bytes_upper_bound": 2 * rows_bytes
        + 2 * score_bytes
        + direction_bytes
        + graph_bytes
        + code_bytes
        + 2 * decoded_bytes,
    }


def _base_receipt(
    e0_state: d42.D42UnifiedShrinkageLDAState,
    candidate_state: d42.D42UnifiedShrinkageLDAState,
    *,
    active: bool,
    fallback_active: bool,
    fallback_reason: str | None,
    k_shot: int,
    old_class_count: int,
    block_coordinates: Sequence[int | None] | None = None,
    block_changed_counts: Sequence[int] | None = None,
    fold_class_deltas: Sequence[Sequence[float]] | None = None,
    fold_old_to_new_deltas: Sequence[float] | None = None,
    fold_new_to_old_deltas: Sequence[float] | None = None,
    class_guard_pass: bool = False,
    cross_guard_pass: bool = False,
    support_guard_pass: bool = False,
    support_margin_delta_max_abs: float | None = None,
) -> dict[str, Any]:
    """Create one complete raw AFCP receipt for active, fallback, or alias paths."""

    class_count = int(len(e0_state.classes))
    resources = _static_resource_values(e0_state, k_shot=int(k_shot))
    e0_sha = d42_afcp_state_sha256(e0_state)
    final_sha = d42_afcp_state_sha256(candidate_state)
    changed_mask = np.asarray(candidate_state.coef2_qint8, dtype=np.int16) - np.asarray(
        e0_state.coef2_qint8, dtype=np.int16
    )
    changed = int(np.count_nonzero(changed_mask))
    state_delta_l1 = int(np.abs(changed_mask).sum(dtype=np.int64))
    coordinates = list(block_coordinates or (None,) * len(d42.BLOCK_SLICES))
    counts = [int(value) for value in (block_changed_counts or (0,) * len(d42.BLOCK_SLICES))]
    if len(coordinates) != len(d42.BLOCK_SLICES) or len(counts) != len(
        d42.BLOCK_SLICES
    ):
        raise D92D42AFCPError("AFCP block receipt shape drift")
    all_three_blocks_changed = bool(all(value > 0 for value in counts))
    modified = ["coef2_qint8"] if changed > 0 else []
    return {
        "d92_afcp_active": bool(active),
        "d92_afcp_fallback_active": bool(fallback_active),
        "d92_afcp_fallback_reason": fallback_reason,
        "d92_afcp_formula_revision": FORMULA_REVISION,
        "d92_afcp_state_postprocess_mode": STATE_POSTPROCESS_MODE,
        "d92_afcp_direct_state_publish": True,
        "d92_afcp_support_only": True,
        "d92_afcp_all_class_symmetric": True,
        "d92_afcp_class_permutation_equivariant": True,
        "d92_afcp_row_permutation_invariant": True,
        "d92_afcp_task_swap_equivariant": True,
        "d92_afcp_support_row_canonicalization": SUPPORT_ROW_CANONICALIZATION,
        "d92_afcp_fold_rule": FOLD_RULE,
        "d92_afcp_fold_tie_policy": FOLD_TIE_POLICY,
        "d92_afcp_e0_state_sha256": e0_sha,
        "d92_afcp_final_state_sha256": final_sha,
        "d92_afcp_modified_state_field_names": modified,
        "d92_afcp_coef1_byte_exact": _state_byte_equal(
            e0_state, candidate_state, "coef1_qint8"
        ),
        "d92_afcp_coef2_byte_exact": _state_byte_equal(
            e0_state, candidate_state, "coef2_qint8"
        ),
        "d92_afcp_scale1_byte_exact": _state_byte_equal(
            e0_state, candidate_state, "scale1_fp16"
        ),
        "d92_afcp_scale2_byte_exact": _state_byte_equal(
            e0_state, candidate_state, "scale2_fp16"
        ),
        "d92_afcp_intercept_byte_exact": _state_byte_equal(
            e0_state, candidate_state, "intercept_fp16"
        ),
        "d92_afcp_log_diag_byte_exact": _state_byte_equal(
            e0_state, candidate_state, "log_diag_fp32"
        ),
        "d92_afcp_coef_fp32_byte_exact": _state_byte_equal(
            e0_state, candidate_state, "coef_fp32"
        ),
        "d92_afcp_intercept_fp32_byte_exact": _state_byte_equal(
            e0_state, candidate_state, "intercept_fp32"
        ),
        "d92_afcp_class_registry_byte_exact": tuple(e0_state.classes)
        == tuple(candidate_state.classes),
        "d92_afcp_state_shape_byte_exact": bool(
            e0_state.coef2_qint8.shape == candidate_state.coef2_qint8.shape
        ),
        "d92_afcp_class_count": class_count,
        "d92_afcp_old_class_count": int(old_class_count),
        "d92_afcp_k_shot": int(k_shot),
        "d92_afcp_block_coordinate_indices": coordinates,
        "d92_afcp_block_changed_code2_counts": counts,
        "d92_afcp_changed_code2_count": changed,
        "d92_afcp_state_delta_code2_l1": state_delta_l1,
        "d92_afcp_all_three_blocks_changed": all_three_blocks_changed,
        "d92_afcp_final_state_non_e0": final_sha != e0_sha,
        "d92_afcp_support_margin_delta_max_abs": support_margin_delta_max_abs,
        "d92_afcp_support_margin_quantum_pass": bool(
            support_margin_delta_max_abs is not None
            and float(support_margin_delta_max_abs) > 0.0
        ),
        "d92_afcp_fold_class_all_margin_delta_mean": (
            None
            if fold_class_deltas is None
            else [[float(value) for value in fold] for fold in fold_class_deltas]
        ),
        "d92_afcp_fold_old_to_new_cross_margin_delta_mean": (
            None
            if fold_old_to_new_deltas is None
            else [float(value) for value in fold_old_to_new_deltas]
        ),
        "d92_afcp_fold_new_to_old_cross_margin_delta_mean": (
            None
            if fold_new_to_old_deltas is None
            else [float(value) for value in fold_new_to_old_deltas]
        ),
        "d92_afcp_twofold_class_guard_pass": bool(class_guard_pass),
        "d92_afcp_twofold_cross_guard_pass": bool(cross_guard_pass),
        "d92_afcp_support_guard_pass": bool(support_guard_pass),
        "d92_afcp_requantize_call_count": 0,
        "d92_afcp_additional_full_fit_count": 0,
        "d92_afcp_block_fit_count": 0,
        "d92_afcp_loo_fit_count": 0,
        "d92_afcp_fisher_fit_count": 0,
        "d92_afcp_tail_selection_count": 0,
        "d92_afcp_rival_pair_selection_count": 0,
        "d92_afcp_atomic_candidate_count": 0,
        "d92_afcp_prefix_evaluation_count": 0,
        "d92_afcp_candidate_scan_count": 0,
        "d92_afcp_persistent_state_bytes_delta": 0,
        "d92_afcp_query_macs_delta": 0,
        "d92_afcp_query_rows_used": 0,
        "d92_afcp_clean_sample_access": False,
        "d92_afcp_source_sample_access": False,
        "d92_afcp_query_fit_access": False,
        "d92_afcp_query_update_access": False,
        "d92_afcp_query_selection_access": False,
        "d92_afcp_query_truth_access": False,
        "d92_afcp_query_role_oracle_access": False,
        "d92_afcp_query_class_quota_access": False,
        "d92_afcp_query_global_reassignment": False,
        **{f"d92_afcp_{name}": value for name, value in resources.items()},
    }


def _canonical_folds(
    rows: np.ndarray, targets: np.ndarray, *, class_count: int
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    fold_zero: list[np.ndarray] = []
    fold_one: list[np.ndarray] = []
    for class_index in range(class_count):
        indices = np.flatnonzero(targets == class_index)
        keys: list[tuple[bytes, bytes, int]] = []
        for index in indices.tolist():
            row32 = np.ascontiguousarray(rows[index], dtype=np.float32).tobytes()
            row64 = np.ascontiguousarray(rows[index], dtype=np.float64).tobytes()
            keys.append((row32, row64, int(index)))
        canonical_keys = [(row32, row64) for row32, row64, _ in keys]
        if len(set(canonical_keys)) != len(canonical_keys):
            raise D92D42AFCPError("duplicate_canonical_support_row")
        ordered = [index for _, _, index in sorted(keys, key=lambda value: value[:2])]
        first = np.asarray(ordered[0::2], dtype=np.int64)
        second = np.asarray(ordered[1::2], dtype=np.int64)
        if len(first) == 0 or len(second) == 0:
            raise D92D42AFCPError("empty_fold_side")
        fold_zero.append(first)
        fold_one.append(second)
    return tuple(fold_zero), tuple(fold_one)


def _score_transformed(
    state: d42.D42UnifiedShrinkageLDAState, rows: np.ndarray
) -> np.ndarray:
    coefficient = np.asarray(d42.decode_d42_coefficients(state), dtype=np.float32)
    scores = np.asarray(rows, dtype=np.float32) @ coefficient.T
    scores += np.asarray(state.intercept_fp16, dtype=np.float32)[None, :]
    if not np.isfinite(scores).all():
        raise D92D42AFCPError("support_score_nonfinite")
    return np.asarray(scores, dtype=np.float32)


def _competitor_probabilities(score_row: np.ndarray, true_class: int) -> np.ndarray:
    values = np.asarray(score_row, dtype=np.float64).copy()
    values[int(true_class)] = -np.inf
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        raise D92D42AFCPError("competitor_score_nonfinite")
    unnormalized = np.exp(values - maximum)
    unnormalized[int(true_class)] = 0.0
    denominator = float(np.sum(unnormalized, dtype=np.float64))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise D92D42AFCPError("competitor_probability_degenerate")
    return unnormalized / denominator


def _fold_direction(
    rows: np.ndarray,
    targets: np.ndarray,
    scores: np.ndarray,
    fold: Sequence[np.ndarray],
    *,
    class_count: int,
) -> np.ndarray:
    means = np.zeros((class_count, d42.FEATURE_DIM), dtype=np.float64)
    directed = np.zeros((class_count, class_count), dtype=np.float64)
    for class_index, indices in enumerate(fold):
        class_rows = np.asarray(rows[indices], dtype=np.float64)
        means[class_index] = np.mean(class_rows, axis=0, dtype=np.float64)
        for index in indices.tolist():
            if int(targets[index]) != class_index:
                raise D92D42AFCPError("fold_target_closure_drift")
            directed[class_index] += _competitor_probabilities(
                scores[index], class_index
            ) / float(len(indices))
    symmetric = 0.5 * (directed + directed.T)
    laplacian = np.diag(np.sum(symmetric, axis=1, dtype=np.float64)) - symmetric
    direction = laplacian @ means
    if not np.isfinite(direction).all():
        raise D92D42AFCPError("fold_direction_nonfinite")
    return direction


def _consensus_delta(
    direction_zero: np.ndarray, direction_one: np.ndarray
) -> tuple[np.ndarray, list[int], list[int]]:
    product = np.asarray(direction_zero, dtype=np.float64) * np.asarray(
        direction_one, dtype=np.float64
    )
    consensus = np.zeros_like(product, dtype=np.float64)
    same_sign = product > 0.0
    consensus[same_sign] = np.sign(direction_zero[same_sign]) * np.sqrt(
        np.abs(product[same_sign])
    )
    centered = consensus - np.mean(consensus, axis=0, dtype=np.float64)[None, :]
    if not np.isfinite(centered).all():
        raise D92D42AFCPError("consensus_nonfinite")
    class_count = int(centered.shape[0])
    delta = np.zeros_like(centered, dtype=np.int16)
    coordinates: list[int] = []
    changed_counts: list[int] = []
    for block in d42.BLOCK_SLICES:
        energy = np.sum(np.square(centered[:, block]), axis=0, dtype=np.float64)
        maximum = float(np.max(energy))
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise D92D42AFCPError("empty_coordinate_energy")
        winners = np.flatnonzero(energy == maximum)
        if len(winners) != 1:
            raise D92D42AFCPError("coordinate_tie")
        coordinate = int(block.start) + int(winners[0])
        values = np.asarray(centered[:, coordinate], dtype=np.float64)
        positive = np.flatnonzero(values > 0.0)
        negative = np.flatnonzero(values < 0.0)
        if len(positive) == 0 or len(negative) == 0:
            raise D92D42AFCPError("empty_signed_side")
        count = int(min(len(positive), len(negative)))
        positive_order = positive[np.argsort(-values[positive], kind="stable")]
        negative_order = negative[np.argsort(values[negative], kind="stable")]
        if (
            len(positive_order) > count
            and values[positive_order[count - 1]] == values[positive_order[count]]
        ):
            raise D92D42AFCPError("positive_selection_boundary_tie")
        if (
            len(negative_order) > count
            and values[negative_order[count - 1]] == values[negative_order[count]]
        ):
            raise D92D42AFCPError("negative_selection_boundary_tie")
        delta[positive_order[:count], coordinate] = 1
        delta[negative_order[:count], coordinate] = -1
        coordinates.append(coordinate)
        changed_counts.append(2 * count)
    if delta.shape != (class_count, d42.FEATURE_DIM):
        raise D92D42AFCPError("delta_shape_drift")
    return delta, coordinates, changed_counts


def _logsumexp(values: np.ndarray) -> float:
    numeric = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(numeric))
    if not np.isfinite(maximum):
        raise D92D42AFCPError("margin_competitor_nonfinite")
    result = maximum + float(np.log(np.sum(np.exp(numeric - maximum))))
    if not np.isfinite(result):
        raise D92D42AFCPError("margin_logsumexp_nonfinite")
    return result


def _mean_margin(
    scores: np.ndarray,
    indices: np.ndarray,
    targets: np.ndarray,
    *,
    competitors: np.ndarray | None = None,
) -> float:
    values: list[float] = []
    for index in indices.tolist():
        true_class = int(targets[index])
        if competitors is None:
            eligible = np.concatenate(
                [
                    np.arange(true_class, dtype=np.int64),
                    np.arange(true_class + 1, scores.shape[1], dtype=np.int64),
                ]
            )
        else:
            eligible = np.asarray(competitors, dtype=np.int64)
        if len(eligible) == 0:
            raise D92D42AFCPError("empty_margin_competitor_set")
        values.append(float(scores[index, true_class]) - _logsumexp(scores[index, eligible]))
    result = float(np.mean(np.asarray(values, dtype=np.float64), dtype=np.float64))
    if not np.isfinite(result):
        raise D92D42AFCPError("margin_mean_nonfinite")
    return result


def _support_guard(
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    targets: np.ndarray,
    folds: tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]],
    *,
    old_class_count: int,
) -> tuple[list[list[float]], list[float], list[float], bool, bool, float]:
    class_deltas: list[list[float]] = []
    old_cross_deltas: list[float] = []
    new_cross_deltas: list[float] = []
    all_margin_deltas: list[float] = []
    old_classes = np.arange(int(old_class_count), dtype=np.int64)
    new_classes = np.arange(int(old_class_count), base_scores.shape[1], dtype=np.int64)
    for fold in folds:
        per_class: list[float] = []
        fold_old: list[np.ndarray] = []
        fold_new: list[np.ndarray] = []
        for class_index, indices in enumerate(fold):
            base_mean = _mean_margin(base_scores, indices, targets)
            candidate_mean = _mean_margin(candidate_scores, indices, targets)
            delta = float(candidate_mean - base_mean)
            per_class.append(delta)
            fold_old.extend([indices] if class_index < old_class_count else [])
            fold_new.extend([indices] if class_index >= old_class_count else [])
            for index in indices.tolist():
                all_margin_deltas.append(
                    float(
                        _mean_margin(
                            candidate_scores,
                            np.asarray([index], dtype=np.int64),
                            targets,
                        )
                        - _mean_margin(
                            base_scores,
                            np.asarray([index], dtype=np.int64),
                            targets,
                        )
                    )
                )
        old_indices = np.concatenate(fold_old)
        new_indices = np.concatenate(fold_new)
        old_cross_deltas.append(
            float(
                _mean_margin(
                    candidate_scores,
                    old_indices,
                    targets,
                    competitors=new_classes,
                )
                - _mean_margin(
                    base_scores, old_indices, targets, competitors=new_classes
                )
            )
        )
        new_cross_deltas.append(
            float(
                _mean_margin(
                    candidate_scores,
                    new_indices,
                    targets,
                    competitors=old_classes,
                )
                - _mean_margin(
                    base_scores, new_indices, targets, competitors=old_classes
                )
            )
        )
        class_deltas.append(per_class)
    values = np.asarray(
        [*class_deltas[0], *class_deltas[1], *old_cross_deltas, *new_cross_deltas],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise D92D42AFCPError("support_guard_nonfinite")
    class_pass = bool(np.all(np.asarray(class_deltas, dtype=np.float64) >= 0.0))
    cross_pass = bool(
        np.all(np.asarray(old_cross_deltas, dtype=np.float64) >= 0.0)
        and np.all(np.asarray(new_cross_deltas, dtype=np.float64) >= 0.0)
    )
    max_abs = float(np.max(np.abs(np.asarray(all_margin_deltas, dtype=np.float64))))
    return class_deltas, old_cross_deltas, new_cross_deltas, class_pass, cross_pass, max_abs


def apply_d42_allclass_fold_consensus_plane(
    state: d42.D42UnifiedShrinkageLDAState,
    transformed_rows: np.ndarray,
    targets: np.ndarray,
    *,
    old_class_count: int,
) -> tuple[d42.D42UnifiedShrinkageLDAState, dict[str, Any]]:
    """Publish one guarded AFCP residual-code plane or exact-E0 fallback."""

    if not isinstance(state, d42.D42UnifiedShrinkageLDAState) or not state.is_int8:
        raise D92D42AFCPError("AFCP requires a compiled D42 int8 state")
    rows = np.asarray(transformed_rows, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != d42.FEATURE_DIM:
        raise D92D42AFCPError("AFCP support feature shape drift")
    if target_array.shape != (len(rows),):
        raise D92D42AFCPError("AFCP support target shape drift")
    if not np.isfinite(rows).all():
        raise D92D42AFCPError("AFCP support feature nonfinite")
    if (
        int(old_class_count) != int(state.old_class_count)
        or int(old_class_count) < 1
        or int(old_class_count) >= len(state.classes)
    ):
        if int(old_class_count) >= len(state.classes):
            counts = np.bincount(target_array, minlength=len(state.classes))
            if len(counts) == len(state.classes) and np.all(counts > 0):
                return state, _base_receipt(
                    state,
                    state,
                    active=False,
                    fallback_active=False,
                    fallback_reason="NOT_REGISTERED_STATE",
                    k_shot=int(counts[0]),
                    old_class_count=int(old_class_count),
                )
        raise D92D42AFCPError("AFCP old-class registry drift")
    if np.any(target_array < 0) or np.any(target_array >= len(state.classes)):
        raise D92D42AFCPError("AFCP support target range drift")
    counts = np.bincount(target_array, minlength=len(state.classes))
    if len(counts) != len(state.classes) or np.any(counts <= 0):
        raise D92D42AFCPError("AFCP support class closure drift")
    if len(set(int(value) for value in counts.tolist())) != 1:
        raise D92D42AFCPError("AFCP support K closure drift")
    k_shot = int(counts[0])
    if k_shot <= 2:
        return state, _base_receipt(
            state,
            state,
            active=False,
            fallback_active=False,
            fallback_reason="K1_K2_EXACT_D92_FULL_ALIAS",
            k_shot=k_shot,
            old_class_count=int(old_class_count),
        )
    try:
        folds = _canonical_folds(rows, target_array, class_count=len(state.classes))
        base_scores = _score_transformed(state, rows)
        direction_zero = _fold_direction(
            rows,
            target_array,
            base_scores,
            folds[0],
            class_count=len(state.classes),
        )
        direction_one = _fold_direction(
            rows,
            target_array,
            base_scores,
            folds[1],
            class_count=len(state.classes),
        )
        delta, coordinates, block_counts = _consensus_delta(
            direction_zero, direction_one
        )
        candidate_codes = np.asarray(state.coef2_qint8, dtype=np.int16) + delta
        if np.any(candidate_codes < -127) or np.any(candidate_codes > 127):
            raise D92D42AFCPError("code_saturation")
        candidate_state = replace(
            state, coef2_qint8=candidate_codes.astype(np.int8, copy=False)
        )
        candidate_scores = _score_transformed(candidate_state, rows)
        (
            class_deltas,
            old_cross_deltas,
            new_cross_deltas,
            class_guard,
            cross_guard,
            margin_delta_max_abs,
        ) = _support_guard(
            base_scores,
            candidate_scores,
            target_array,
            folds,
            old_class_count=int(old_class_count),
        )
        provisional = _base_receipt(
            state,
            candidate_state,
            active=True,
            fallback_active=False,
            fallback_reason=None,
            k_shot=k_shot,
            old_class_count=int(old_class_count),
            block_coordinates=coordinates,
            block_changed_counts=block_counts,
            fold_class_deltas=class_deltas,
            fold_old_to_new_deltas=old_cross_deltas,
            fold_new_to_old_deltas=new_cross_deltas,
            class_guard_pass=class_guard,
            cross_guard_pass=cross_guard,
            support_guard_pass=bool(class_guard and cross_guard),
            support_margin_delta_max_abs=margin_delta_max_abs,
        )
        if (
            not provisional["d92_afcp_all_three_blocks_changed"]
            or not provisional["d92_afcp_final_state_non_e0"]
            or not provisional["d92_afcp_support_margin_quantum_pass"]
            or not provisional["d92_afcp_support_guard_pass"]
            or provisional["d92_afcp_changed_code2_count"] > 3 * len(state.classes)
        ):
            raise D92D42AFCPError("support_guard_failed")
        return candidate_state, provisional
    except (D92D42AFCPError, FloatingPointError) as error:
        return state, _base_receipt(
            state,
            state,
            active=False,
            fallback_active=True,
            fallback_reason=str(error),
            k_shot=k_shot,
            old_class_count=int(old_class_count),
        )


__all__ = [
    "D92D42AFCPError",
    "STATE_POSTPROCESS_MODE",
    "apply_d42_allclass_fold_consensus_plane",
    "d42_afcp_inactive_receipt",
    "d42_afcp_state_sha256",
]
