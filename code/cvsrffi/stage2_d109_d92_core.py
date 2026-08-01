"""D109 four-arm D92/CB-RRC core with a support-confusion SCRC head.

D109 reuses D108 only to construct the exact four formal D42 int8 score
states and its frozen CB-RRC transform.  The D108 SMME states exist only
inside that temporary construction and are deliberately neither extracted nor
persisted here.  D109 then freezes one SCRC state for each matching
base/DA-by-before/after support surface.  Query scoring is row-independent
and has no fitting or state-update surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi import stage2_d108_cbrrc as cbrrc
from cvsrffi import stage2_d108_d92_core as d108
from cvsrffi import stage2_d109_scrc as scrc


CANDIDATE_ID = "D109-SCRC/r1"
PROTOCOL_SCHEMA = "p2_min_v1"
ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")
PHASES = ("before", "after")
_DA_ARMS = frozenset(("M_DA", "M_JOINT"))
_HEAD_ARMS = frozenset(("M_HEAD", "M_JOINT"))


class D109D92CoreError(ValueError):
    """Raised when the D109 pair closure or fixed score path drifts."""


def _strict_text_sequence(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise D109D92CoreError(f"{name} must be a sequence of exact strings")
    result = tuple(value)
    if any(type(item) is not str or not item for item in result):
        raise D109D92CoreError(f"{name} must contain non-empty exact strings")
    return result


def _formal_int8_snapshot(state: d42.D42UnifiedShrinkageLDAState) -> tuple[bytes, ...]:
    if type(state) is not d42.D42UnifiedShrinkageLDAState or not state.is_int8:
        raise D109D92CoreError("formal D42 state must be an exact int8 state")
    return d42._state_snapshot(state)


def _validate_formal_state(
    state: d42.D42UnifiedShrinkageLDAState,
    classes: tuple[str, ...],
    old_class_count: int,
    name: str,
) -> None:
    if (
        type(state) is not d42.D42UnifiedShrinkageLDAState
        or not state.is_int8
        or state.classes != classes
        or state.old_class_count != old_class_count
    ):
        raise D109D92CoreError(f"{name} formal class/state closure drift")
    _formal_int8_snapshot(state)


def _validate_scrc_state(
    state: scrc.SCRCState,
    classes: tuple[str, ...],
    k_shot: int,
    name: str,
) -> None:
    if (
        type(state) is not scrc.SCRCState
        or state.registered_classes != classes
    ):
        raise D109D92CoreError(f"{name} SCRC registry closure drift")
    receipt = scrc.scrc_resource_receipt(state)
    if (
        receipt.get("registered_class_count") != len(classes)
        or receipt.get("k_shot") != k_shot
        or receipt.get("support_row_count") != len(classes) * k_shot
        or receipt.get("support_only") is not True
        or receipt.get("query_fit_rows") != 0
        or receipt.get("query_state_updates") != 0
        or receipt.get("query_truth_access") is not False
        or receipt.get("query_role_access") is not False
        or receipt.get("query_class_quota_access") is not False
        or receipt.get("query_batch_global_assignment") is not False
        or not isinstance(receipt.get("numeric_state_bytes"), int)
        or receipt["numeric_state_bytes"] < 1
    ):
        raise D109D92CoreError(f"{name} SCRC resource/state closure drift")


@dataclass(frozen=True, slots=True)
class D109D92Pair:
    """Typed frozen D109 state for the fixed M0/M_DA/M_HEAD/M_JOINT arms."""

    base_before_state: d42.D42UnifiedShrinkageLDAState
    base_after_state: d42.D42UnifiedShrinkageLDAState
    da_before_state: d42.D42UnifiedShrinkageLDAState
    da_after_state: d42.D42UnifiedShrinkageLDAState
    cbrrc_state: cbrrc.CBRRCState
    base_before_scrc: scrc.SCRCState
    base_after_scrc: scrc.SCRCState
    da_before_scrc: scrc.SCRCState
    da_after_scrc: scrc.SCRCState
    old_registered_classes: tuple[str, ...]
    registered_classes: tuple[str, ...]
    k_shot: int
    d42_trainable_parameters: int
    d42_optimizer_steps_per_fit: int

    def __post_init__(self) -> None:
        _validate_pair(self)


def _validate_pair(pair: D109D92Pair) -> None:
    if type(pair) is not D109D92Pair:
        raise D109D92CoreError("pair must be an exact D109D92Pair")
    old_classes = _strict_text_sequence(
        pair.old_registered_classes, "old_registered_classes"
    )
    classes = _strict_text_sequence(pair.registered_classes, "registered_classes")
    if (
        len(old_classes) != cbrrc.BEFORE_CLASS_COUNT
        or len(classes) <= len(old_classes)
        or classes[: len(old_classes)] != old_classes
        or len(set(classes)) != len(classes)
        or type(pair.k_shot) is not int
        or pair.k_shot not in cbrrc.ALLOWED_K
        or type(pair.d42_trainable_parameters) is not int
        or pair.d42_trainable_parameters < 0
        or type(pair.d42_optimizer_steps_per_fit) is not int
        or pair.d42_optimizer_steps_per_fit < 0
    ):
        raise D109D92CoreError("pair registry, K-shot, or resource closure drift")
    _validate_formal_state(
        pair.base_before_state, old_classes, len(old_classes), "base before"
    )
    _validate_formal_state(
        pair.base_after_state, classes, len(old_classes), "base after"
    )
    _validate_formal_state(
        pair.da_before_state, old_classes, len(old_classes), "DA before"
    )
    _validate_formal_state(pair.da_after_state, classes, len(old_classes), "DA after")
    if (
        type(pair.cbrrc_state) is not cbrrc.CBRRCState
        or pair.cbrrc_state.before_registered_classes != old_classes
        or pair.cbrrc_state.k_shot != pair.k_shot
    ):
        raise D109D92CoreError("CB-RRC old-support state closure drift")
    for state, state_classes, name in (
        (pair.base_before_scrc, old_classes, "base before"),
        (pair.base_after_scrc, classes, "base after"),
        (pair.da_before_scrc, old_classes, "DA before"),
        (pair.da_after_scrc, classes, "DA after"),
    ):
        _validate_scrc_state(state, state_classes, pair.k_shot, name)


def _concat_support(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate((left, right), axis=0), dtype=np.float32)


def _assert_d42_states_unchanged(
    snapshots: tuple[tuple[d42.D42UnifiedShrinkageLDAState, tuple[bytes, ...]], ...]
) -> None:
    if any(_formal_int8_snapshot(state) != expected for state, expected in snapshots):
        raise D109D92CoreError("formal D42 state mutated during SCRC construction")


def _build_scrc_state(
    state: d42.D42UnifiedShrinkageLDAState,
    support_features: np.ndarray,
    support_labels: tuple[str, ...],
    registered_classes: tuple[str, ...],
) -> scrc.SCRCState:
    support_logits = d42.score_d42_unified_shrinkage_lda(state, support_features)
    return scrc.build_scrc_state(
        support_logits, support_labels, registered_classes
    )


def build_d109_d92_pair(
    old_support_features: np.ndarray,
    old_support_labels: Sequence[str],
    old_registered_classes: Sequence[str],
    new_support_features: np.ndarray,
    new_support_labels: Sequence[str],
    new_registered_classes: Sequence[str],
    *,
    seed: int,
    device: d42.torch.device | str,
    d92_fit: Callable[..., Any],
) -> D109D92Pair:
    """Build D109 from support only, retaining no D108 SMME state."""

    old_labels = _strict_text_sequence(old_support_labels, "old_support_labels")
    new_labels = _strict_text_sequence(new_support_labels, "new_support_labels")
    old_classes = _strict_text_sequence(
        old_registered_classes, "old_registered_classes"
    )
    new_classes = _strict_text_sequence(
        new_registered_classes, "new_registered_classes"
    )
    registered_classes = old_classes + new_classes

    # This is deliberately temporary: it establishes exact D108 D42/CB-RRC
    # parity, then all SMME-bearing state is dropped before D109 states exist.
    temporary_d108_pair = d108.build_d108_d92_pair(
        old_support_features,
        old_labels,
        old_classes,
        new_support_features,
        new_labels,
        new_classes,
        seed=seed,
        device=device,
        d92_fit=d92_fit,
    )
    base_before_state = temporary_d108_pair.base_before_state
    base_after_state = temporary_d108_pair.base_after_state
    da_before_state = temporary_d108_pair.da_before_state
    da_after_state = temporary_d108_pair.da_after_state
    cbrrc_state = temporary_d108_pair.cbrrc_state
    k_shot = temporary_d108_pair.k_shot
    d42_trainable_parameters = temporary_d108_pair.d42_trainable_parameters
    d42_optimizer_steps_per_fit = temporary_d108_pair.d42_optimizer_steps_per_fit
    del temporary_d108_pair

    base_before_snapshot = _formal_int8_snapshot(base_before_state)
    base_after_snapshot = _formal_int8_snapshot(base_after_state)
    da_before_snapshot = _formal_int8_snapshot(da_before_state)
    da_after_snapshot = _formal_int8_snapshot(da_after_state)
    all_labels = old_labels + new_labels
    base_after_support = _concat_support(old_support_features, new_support_features)
    da_old_support = cbrrc.transform_cbrrc_features(cbrrc_state, old_support_features)
    da_new_support = cbrrc.transform_cbrrc_features(cbrrc_state, new_support_features)
    da_after_support = _concat_support(da_old_support, da_new_support)

    base_before_scrc = _build_scrc_state(
        base_before_state, old_support_features, old_labels, old_classes
    )
    base_after_scrc = _build_scrc_state(
        base_after_state, base_after_support, all_labels, registered_classes
    )
    da_before_scrc = _build_scrc_state(
        da_before_state, da_old_support, old_labels, old_classes
    )
    da_after_scrc = _build_scrc_state(
        da_after_state, da_after_support, all_labels, registered_classes
    )
    _assert_d42_states_unchanged(
        (
            (base_before_state, base_before_snapshot),
            (base_after_state, base_after_snapshot),
            (da_before_state, da_before_snapshot),
            (da_after_state, da_after_snapshot),
        )
    )
    return D109D92Pair(
        base_before_state=base_before_state,
        base_after_state=base_after_state,
        da_before_state=da_before_state,
        da_after_state=da_after_state,
        cbrrc_state=cbrrc_state,
        base_before_scrc=base_before_scrc,
        base_after_scrc=base_after_scrc,
        da_before_scrc=da_before_scrc,
        da_after_scrc=da_after_scrc,
        old_registered_classes=old_classes,
        registered_classes=registered_classes,
        k_shot=k_shot,
        d42_trainable_parameters=d42_trainable_parameters,
        d42_optimizer_steps_per_fit=d42_optimizer_steps_per_fit,
    )


def score(
    pair: D109D92Pair,
    phase: str,
    arm: str,
    query_features: np.ndarray,
) -> np.ndarray:
    """Score the four fixed arms without any query-side fitting or update."""

    _validate_pair(pair)
    if type(phase) is not str or phase not in PHASES:
        raise D109D92CoreError(f"phase must be one of {PHASES}")
    if type(arm) is not str or arm not in ARMS:
        raise D109D92CoreError(f"arm must be one of {ARMS}")
    use_da = arm in _DA_ARMS
    features = (
        cbrrc.transform_cbrrc_features(pair.cbrrc_state, query_features)
        if use_da
        else query_features
    )
    if use_da:
        d42_state = pair.da_before_state if phase == "before" else pair.da_after_state
    else:
        d42_state = (
            pair.base_before_state if phase == "before" else pair.base_after_state
        )
    logits = d42.score_d42_unified_shrinkage_lda(d42_state, features)
    if arm not in _HEAD_ARMS:
        return logits
    if use_da:
        scrc_state = pair.da_before_scrc if phase == "before" else pair.da_after_scrc
    else:
        scrc_state = (
            pair.base_before_scrc if phase == "before" else pair.base_after_scrc
        )
    return scrc.apply_scrc_query(scrc_state, logits)


def _scrc_numeric_state_bytes(state: scrc.SCRCState) -> int:
    receipt = scrc.scrc_resource_receipt(state)
    value = receipt.get("numeric_state_bytes")
    if type(value) is not int or value < 1:
        raise D109D92CoreError("SCRC resource numeric-state receipt drift")
    return value


def resource_summary(pair: D109D92Pair) -> dict[str, int | bool | str]:
    """Return the frozen D42/CB-RRC/SCRC state and query-safety summary."""

    _validate_pair(pair)
    d42_states = (
        pair.base_before_state,
        pair.base_after_state,
        pair.da_before_state,
        pair.da_after_state,
    )
    scrc_states = (
        pair.base_before_scrc,
        pair.base_after_scrc,
        pair.da_before_scrc,
        pair.da_after_scrc,
    )
    return {
        "candidate_id": CANDIDATE_ID,
        "formal_int8_score_state_count": len(d42_states),
        "formal_int8_score_states_only": True,
        "scrc_state_count": len(scrc_states),
        "d108_smme_state_persisted": False,
        "formal_score_state_bytes": int(
            sum(state.persistent_state_bytes for state in d42_states)
            + sum(_scrc_numeric_state_bytes(state) for state in scrc_states)
            + pair.cbrrc_state.energy_fp16.nbytes
        ),
        "scrc_numeric_state_bytes": int(
            sum(_scrc_numeric_state_bytes(state) for state in scrc_states)
        ),
        "d42_trainable_parameters_per_fit": pair.d42_trainable_parameters,
        "aggregate_d42_optimizer_steps": 2 * pair.d42_optimizer_steps_per_fit,
        "support_only": True,
        "cbrrc_state_source_old_before_support_only": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_truth_access": False,
        "query_role_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "clean_sample_access": False,
        "source_sample_access": False,
    }


__all__ = [
    "ARMS",
    "CANDIDATE_ID",
    "D109D92CoreError",
    "D109D92Pair",
    "PHASES",
    "PROTOCOL_SCHEMA",
    "build_d109_d92_pair",
    "resource_summary",
    "score",
]
