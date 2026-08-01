"""Minimal D108 four-arm D92 core: CB-RRC plus the SMME LDA head.

The caller supplies a prebuilt D92 fit callable.  Construction accepts only
old/new support, freezes CB-RRC from old support alone, and retains only the
four formal D42 int8 score states plus fixed resource scalars.  Scoring takes
features only and decides every row independently over its registered classes.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Callable, Sequence

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi import stage2_d108_cbrrc as cbrrc
from cvsrffi import stage2_d108_smme as smme


CANDIDATE_ID = "D108-CB-RRC-SMME/r1"
PROTOCOL_SCHEMA = "p2_min_v1"
ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")
PHASES = ("before", "after")
_DA_ARMS = frozenset(("M_DA", "M_JOINT"))
_HEAD_ARMS = frozenset(("M_HEAD", "M_JOINT"))
_D42_FIT_INJECTION_LOCK = threading.RLock()


class D108D92CoreError(ValueError):
    """Raised when the D108 pair contract or fixed score path drifts."""


def _strict_text_sequence(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise D108D92CoreError(f"{name} must be a sequence of exact strings")
    result = tuple(value)
    if any(type(item) is not str or not item for item in result):
        raise D108D92CoreError(f"{name} must contain non-empty exact strings")
    return result


def _validate_seed_and_device(seed: int, device: d42.torch.device | str) -> None:
    if type(seed) is not int or seed < 0:
        raise D108D92CoreError("seed must be a non-negative exact int")
    if type(device) is str and not device:
        raise D108D92CoreError("device must not be an empty string")
    try:
        d42.torch.device(device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise D108D92CoreError("device is not a valid torch device") from error


def _formal_int8_snapshot(state: d42.D42UnifiedShrinkageLDAState) -> tuple[bytes, ...]:
    if type(state) is not d42.D42UnifiedShrinkageLDAState or not state.is_int8:
        raise D108D92CoreError("formal D42 state must be an exact int8 state")
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
        raise D108D92CoreError(f"{name} formal class/state closure drift")
    _formal_int8_snapshot(state)


def _validate_d42_result(
    result: d42.D42UnifiedShrinkageLDAResult,
    old_registered_classes: tuple[str, ...],
    registered_classes: tuple[str, ...],
    name: str,
) -> None:
    if type(result) is not d42.D42UnifiedShrinkageLDAResult:
        raise D108D92CoreError(f"{name} must be an exact D42 result")
    _validate_formal_state(
        result.before_state,
        old_registered_classes,
        len(old_registered_classes),
        f"{name} before",
    )
    _validate_formal_state(
        result.state,
        registered_classes,
        len(old_registered_classes),
        f"{name} after",
    )


def _resource_scalars(
    result: d42.D42UnifiedShrinkageLDAResult, name: str
) -> tuple[int, int]:
    resource = result.resource_audit
    try:
        trainable_parameters = int(resource["trainable_parameters"])
        optimizer_steps = int(resource["optimizer_steps"])
    except (KeyError, TypeError, ValueError) as error:
        raise D108D92CoreError(f"{name} D42 resource scalar drift") from error
    if trainable_parameters < 0 or optimizer_steps < 0:
        raise D108D92CoreError(f"{name} D42 resource scalar is negative")
    return trainable_parameters, optimizer_steps


@dataclass(frozen=True, slots=True)
class D108D92Pair:
    """Typed frozen state for M0, M_DA, M_HEAD, and M_JOINT."""

    base_before_state: d42.D42UnifiedShrinkageLDAState
    base_after_state: d42.D42UnifiedShrinkageLDAState
    da_before_state: d42.D42UnifiedShrinkageLDAState
    da_after_state: d42.D42UnifiedShrinkageLDAState
    cbrrc_state: cbrrc.CBRRCState
    base_before_smme: smme.SMMEState
    base_after_smme: smme.SMMEState
    da_before_smme: smme.SMMEState
    da_after_smme: smme.SMMEState
    old_registered_classes: tuple[str, ...]
    registered_classes: tuple[str, ...]
    k_shot: int
    d42_trainable_parameters: int
    d42_optimizer_steps_per_fit: int

    def __post_init__(self) -> None:
        _validate_pair(self)


def _validate_pair(pair: D108D92Pair) -> None:
    if type(pair) is not D108D92Pair:
        raise D108D92CoreError("pair must be an exact D108D92Pair")
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
        raise D108D92CoreError("pair registry, K-shot, or resource closure drift")
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
        raise D108D92CoreError("CB-RRC old-support state closure drift")
    for state, state_classes in (
        (pair.base_before_smme, old_classes),
        (pair.base_after_smme, classes),
        (pair.da_before_smme, old_classes),
        (pair.da_after_smme, classes),
    ):
        if (
            type(state) is not smme.SMMEState
            or state.registered_classes != state_classes
            or state.k_shot != pair.k_shot
        ):
            raise D108D92CoreError("SMME registry or K-shot closure drift")


def _fit_d42_with_temporary_d92(
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
) -> d42.D42UnifiedShrinkageLDAResult:
    """Run one D42 fit under a narrow, exception-safe D92 hook injection."""

    if not callable(d92_fit):
        raise D108D92CoreError("d92_fit must be a prebuilt callable")
    with _D42_FIT_INJECTION_LOCK:
        original_fit = d42._fit_equal_prior_lda
        if not callable(original_fit):
            raise D108D92CoreError("D42 baseline LDA hook is not callable")
        try:
            d42._fit_equal_prior_lda = d92_fit
            result = d42.fit_d42_unified_shrinkage_lda(
                old_support_features,
                old_support_labels,
                old_registered_classes,
                new_support_features,
                new_support_labels,
                new_registered_classes,
                seed=seed,
                device=device,
            )
            if d42._fit_equal_prior_lda is not d92_fit:
                raise D108D92CoreError("D42 D92 injection was altered during fit")
        finally:
            d42._fit_equal_prior_lda = original_fit
    if type(result) is not d42.D42UnifiedShrinkageLDAResult:
        raise D108D92CoreError("D42 injection did not return an exact result")
    return result


def _concat_support(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate((left, right), axis=0), dtype=np.float32)


def _assert_before_states_unchanged(
    snapshots: tuple[tuple[d42.D42UnifiedShrinkageLDAState, tuple[bytes, ...]], ...]
) -> None:
    if any(_formal_int8_snapshot(state) != expected for state, expected in snapshots):
        raise D108D92CoreError("before formal state mutated during after construction")


def build_d108_d92_pair(
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
) -> D108D92Pair:
    """Build all four arms from support only; no query argument exists."""

    _validate_seed_and_device(seed, device)
    old_labels = _strict_text_sequence(old_support_labels, "old_support_labels")
    new_labels = _strict_text_sequence(new_support_labels, "new_support_labels")
    old_classes = _strict_text_sequence(
        old_registered_classes, "old_registered_classes"
    )
    new_classes = _strict_text_sequence(
        new_registered_classes, "new_registered_classes"
    )
    if set(old_classes) & set(new_classes):
        raise D108D92CoreError("old and new registered classes must be disjoint")

    # The sole CB-RRC build uses old/before support.  New support is transform-only.
    cbrrc_state = cbrrc.build_cbrrc_state(
        old_support_features, old_labels, old_classes
    )
    da_old_support = cbrrc.transform_cbrrc_features(cbrrc_state, old_support_features)
    da_new_support = cbrrc.transform_cbrrc_features(cbrrc_state, new_support_features)
    registered_classes = old_classes + new_classes

    base_result = _fit_d42_with_temporary_d92(
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
    _validate_d42_result(base_result, old_classes, registered_classes, "base result")
    base_before_state = base_result.before_state
    base_after_state = base_result.state
    base_before_snapshot = _formal_int8_snapshot(base_before_state)
    base_resource = _resource_scalars(base_result, "base")

    da_result = _fit_d42_with_temporary_d92(
        da_old_support,
        old_labels,
        old_classes,
        da_new_support,
        new_labels,
        new_classes,
        seed=seed,
        device=device,
        d92_fit=d92_fit,
    )
    _validate_d42_result(da_result, old_classes, registered_classes, "DA result")
    da_before_state = da_result.before_state
    da_after_state = da_result.state
    da_before_snapshot = _formal_int8_snapshot(da_before_state)
    da_resource = _resource_scalars(da_result, "DA")
    if base_resource != da_resource:
        raise D108D92CoreError("base/DA D42 resource scalar drift")
    del base_result
    del da_result
    _assert_before_states_unchanged(((base_before_state, base_before_snapshot),))

    all_labels = old_labels + new_labels
    base_before_smme = smme.build_smme_state(
        d42.score_d42_unified_shrinkage_lda(base_before_state, old_support_features),
        old_labels,
        old_classes,
    )
    base_after_smme = smme.build_smme_state(
        d42.score_d42_unified_shrinkage_lda(
            base_after_state, _concat_support(old_support_features, new_support_features)
        ),
        all_labels,
        registered_classes,
    )
    da_before_smme = smme.build_smme_state(
        d42.score_d42_unified_shrinkage_lda(da_before_state, da_old_support),
        old_labels,
        old_classes,
    )
    da_after_smme = smme.build_smme_state(
        d42.score_d42_unified_shrinkage_lda(
            da_after_state, _concat_support(da_old_support, da_new_support)
        ),
        all_labels,
        registered_classes,
    )
    _assert_before_states_unchanged(
        (
            (base_before_state, base_before_snapshot),
            (da_before_state, da_before_snapshot),
        )
    )
    return D108D92Pair(
        base_before_state=base_before_state,
        base_after_state=base_after_state,
        da_before_state=da_before_state,
        da_after_state=da_after_state,
        cbrrc_state=cbrrc_state,
        base_before_smme=base_before_smme,
        base_after_smme=base_after_smme,
        da_before_smme=da_before_smme,
        da_after_smme=da_after_smme,
        old_registered_classes=old_classes,
        registered_classes=registered_classes,
        k_shot=cbrrc_state.k_shot,
        d42_trainable_parameters=base_resource[0],
        d42_optimizer_steps_per_fit=base_resource[1],
    )


def score(
    pair: D108D92Pair,
    phase: str,
    arm: str,
    query_features: np.ndarray,
) -> np.ndarray:
    """Score M0/M_DA/M_HEAD/M_JOINT without query-side fitting or updates."""

    _validate_pair(pair)
    if type(phase) is not str or phase not in PHASES:
        raise D108D92CoreError(f"phase must be one of {PHASES}")
    if type(arm) is not str or arm not in ARMS:
        raise D108D92CoreError(f"arm must be one of {ARMS}")
    use_da = arm in _DA_ARMS
    features = (
        cbrrc.transform_cbrrc_features(pair.cbrrc_state, query_features)
        if use_da
        else query_features
    )
    if use_da:
        state = pair.da_before_state if phase == "before" else pair.da_after_state
    else:
        state = pair.base_before_state if phase == "before" else pair.base_after_state
    logits = d42.score_d42_unified_shrinkage_lda(state, features)
    if arm not in _HEAD_ARMS:
        return logits
    if use_da:
        smme_state = pair.da_before_smme if phase == "before" else pair.da_after_smme
    else:
        smme_state = (
            pair.base_before_smme if phase == "before" else pair.base_after_smme
        )
    return smme.apply_smme_query(smme_state, logits)


def resource_summary(pair: D108D92Pair) -> dict[str, int | bool | str]:
    """Return the small resource summary needed by the runner/report path."""

    _validate_pair(pair)
    d42_states = (
        pair.base_before_state,
        pair.base_after_state,
        pair.da_before_state,
        pair.da_after_state,
    )
    smme_bytes = sum(
        state.class_margins_fp64.nbytes + state.delta_fp64.nbytes
        for state in (
            pair.base_before_smme,
            pair.base_after_smme,
            pair.da_before_smme,
            pair.da_after_smme,
        )
    )
    return {
        "candidate_id": CANDIDATE_ID,
        "formal_int8_score_state_count": len(d42_states),
        "formal_int8_score_states_only": True,
        "formal_score_state_bytes": int(
            sum(state.persistent_state_bytes for state in d42_states)
            + smme_bytes
            + pair.cbrrc_state.energy_fp16.nbytes
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
    "D108D92CoreError",
    "D108D92Pair",
    "PHASES",
    "PROTOCOL_SCHEMA",
    "build_d108_d92_pair",
    "resource_summary",
    "score",
]
