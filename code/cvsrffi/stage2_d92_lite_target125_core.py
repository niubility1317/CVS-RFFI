"""D92-Lite160 core for the formal Target125 execution plane.

``M_JOINT`` is retained only as the existing D108 transport key.  This module
does not execute any D108 joint mechanism: before registration it uses the
frozen Phase1 qKNN head, while after registration K5/K10 use D92-Lite160 and
K1 is an exact qKNN-logit alias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import stage2_d129_joint6_heads as d129
from .stage2_adv3b02_ts_drqknn_bcrr import phase1_qknn_lock
from .stage2_zid_student_t_qknn import (
    TypedINT8ZIDSupportBank,
    TypedSharedPSDMetric,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    score_zid_student_t_logits,
)


METHOD_LOCK_SHA256 = "e0f7f8623b4d53002206aca8575f8eadd2bca4150a7c5aed3d017b4827fa5dac"
METHOD_LOCK_SCHEMA = "cvs.phase2.d131.d92_lite160_qtie_target125.method_lock.v2"
CANDIDATE_ID = f"D131-D92-LITE160-QTIE/r2@ml-{METHOD_LOCK_SHA256}"
PROTOCOL_SCHEMA = "p2_min_v1"
TRANSPORT_ARM = "M_JOINT"
OLD_CLASS_COUNT = 6
REGISTERED_FEATURE_WIDTH = 288
ZID_WIDTH = 160


class D92LiteTarget125CoreError(ValueError):
    """Raised when the frozen D92-Lite160 Target125 core drifts."""


def normalized_zid160_from_registered_feature(value: np.ndarray) -> np.ndarray:
    """Recover D129's canonical normalized z_id160 view without a new forward."""

    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != REGISTERED_FEATURE_WIDTH
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D92LiteTarget125CoreError(
            "registered features must be finite float32 [N,288]"
        )
    try:
        return d129.normalize_zid160_rows(
            np.ascontiguousarray(rows[:, :ZID_WIDTH], dtype=np.float32),
            name="registered_feature_primary160",
        )
    except d129.D129Joint6HeadsError as error:
        raise D92LiteTarget125CoreError(
            "registered feature primary160 normalization failed"
        ) from error


def _texts(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise D92LiteTarget125CoreError(f"{name} must be a string sequence")
    result = tuple(str(item) for item in value)
    if not result or any(not item for item in result):
        raise D92LiteTarget125CoreError(f"{name} contains an empty value")
    return result


def _balanced_k(labels: tuple[str, ...], classes: tuple[str, ...]) -> int:
    if len(set(classes)) != len(classes):
        raise D92LiteTarget125CoreError("registered classes must be unique")
    counts = tuple(labels.count(item) for item in classes)
    if any(count < 1 for count in counts) or len(set(counts)) != 1:
        raise D92LiteTarget125CoreError("support must be balanced over every class")
    if counts[0] not in (1, 5, 10):
        raise D92LiteTarget125CoreError("Target125 only permits K1/K5/K10")
    return counts[0]


@dataclass(frozen=True, slots=True)
class D92LiteTarget125Pair:
    before_bank: TypedINT8ZIDSupportBank
    before_metric: TypedSharedPSDMetric
    after_bank: TypedINT8ZIDSupportBank
    after_metric: TypedSharedPSDMetric
    after_lite_state: d129.D129AffineHeadState | None
    old_registered_classes: tuple[str, ...]
    registered_classes: tuple[str, ...]
    active_k: int
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            self.old_registered_classes != self.before_bank.classes
            or self.registered_classes != self.after_bank.classes
            or self.registered_classes[:OLD_CLASS_COUNT]
            != self.old_registered_classes
            or self.active_k != self.before_bank.active_k
            or self.active_k != self.after_bank.active_k
        ):
            raise D92LiteTarget125CoreError("pair registry or K closure drift")
        if self.active_k == 1:
            if self.after_lite_state is not None:
                raise D92LiteTarget125CoreError("K1 must not contain a Lite state")
        elif (
            type(self.after_lite_state) is not d129.D129AffineHeadState
            or self.after_lite_state.classes != self.registered_classes
        ):
            raise D92LiteTarget125CoreError("K5/K10 requires the exact Lite state")


def build_d92_lite_pair(
    old_support_features288: np.ndarray,
    old_support_labels: Sequence[str],
    old_registered_classes: Sequence[str],
    new_support_features288: np.ndarray,
    new_support_labels: Sequence[str],
    new_registered_classes: Sequence[str],
    *,
    seed: int,
    device: Any,
    d92_fit: Callable[..., Any],
) -> D92LiteTarget125Pair:
    """Build support-only before/after states using the D108 PairBuilder API."""

    if type(seed) is not int or seed < 0:
        raise D92LiteTarget125CoreError("seed must be a non-negative exact int")
    del device, d92_fit  # Transport compatibility; neither is used by this head.
    old_classes = _texts(old_registered_classes, "old registered classes")
    new_classes = _texts(new_registered_classes, "new registered classes")
    if (
        len(old_classes) != OLD_CLASS_COUNT
        or set(old_classes).intersection(new_classes)
        or len(set(new_classes)) != len(new_classes)
    ):
        raise D92LiteTarget125CoreError("old/new registry partition drift")
    old_labels = _texts(old_support_labels, "old support labels")
    new_labels = _texts(new_support_labels, "new support labels")
    if any(label not in old_classes for label in old_labels) or any(
        label not in new_classes for label in new_labels
    ):
        raise D92LiteTarget125CoreError("support label role partition drift")
    old_zid = normalized_zid160_from_registered_feature(old_support_features288)
    new_zid = normalized_zid160_from_registered_feature(new_support_features288)
    k_old = _balanced_k(old_labels, old_classes)
    k_new = _balanced_k(new_labels, new_classes)
    if k_old != k_new or len(old_zid) != len(old_labels) or len(new_zid) != len(new_labels):
        raise D92LiteTarget125CoreError("old/new support K or row alignment drift")
    active_k = k_old
    lock = phase1_qknn_lock(active_k)
    before_bank = build_typed_zid_support_bank(
        old_zid, old_labels, old_classes, config=lock
    )
    before_metric = identity_shared_psd_metric(config=lock)
    registered = old_classes + new_classes
    all_zid = np.ascontiguousarray(np.concatenate([old_zid, new_zid], axis=0))
    all_labels = old_labels + new_labels
    after_bank = build_typed_zid_support_bank(
        all_zid, all_labels, registered, config=lock
    )
    after_metric = identity_shared_psd_metric(config=lock)
    lite_state: d129.D129AffineHeadState | None = None
    fit_receipt: Mapping[str, Any] = {
        "head": d129.LITE_HEAD,
        "fit_mode": "exact_qknn_alias_no_Lite_fit",
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
    }
    if active_k != 1:
        fit = d129.fit_d92_lite160(
            all_zid,
            all_labels,
            registered,
            old_class_count=OLD_CLASS_COUNT,
        )
        if type(fit.state) is not d129.D129AffineHeadState:
            raise D92LiteTarget125CoreError("K5/K10 Lite fit returned an alias")
        lite_state = fit.state
        fit_receipt = dict(fit.fit_receipt)
    audit = {
        "schema": "cvs.phase2.d131.d92_lite160.target125.pair_audit.v1",
        "candidate_id": CANDIDATE_ID,
        "transport_arm": TRANSPORT_ARM,
        "transport_arm_is_D108_joint_mechanism": False,
        "active_k": active_k,
        "before_head": "phase1_locked_student_t_qknn",
        "after_head": (
            "exact_same_qknn_logits_alias" if active_k == 1 else d129.LITE_HEAD
        ),
        "after_exact_top_tie_policy": (
            "not_applicable_exact_qknn_alias"
            if active_k == 1
            else "Lite_top_set_same_after_qknn_unique_winner_plus_one_float32_ULP"
        ),
        "after_secondary_exact_tie": "fail_closed",
        "qknn_lock_digest": lock.lock_digest,
        "after_fit_receipt": fit_receipt,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_role_access": False,
        "all_registered_classes_scored": True,
    }
    return D92LiteTarget125Pair(
        before_bank=before_bank,
        before_metric=before_metric,
        after_bank=after_bank,
        after_metric=after_metric,
        after_lite_state=lite_state,
        old_registered_classes=old_classes,
        registered_classes=registered,
        active_k=active_k,
        audit=audit,
    )


def _resolve_exact_lite_top_ties(
    pair: D92LiteTarget125Pair,
    query: np.ndarray,
    lite_logits: np.ndarray,
) -> np.ndarray:
    """Resolve only exact Lite top ties with the same after-qKNN state."""

    result = np.ascontiguousarray(lite_logits, dtype=np.float32)
    maxima = np.max(result, axis=1, keepdims=True)
    tie_rows = np.flatnonzero(np.sum(result == maxima, axis=1) > 1)
    if len(tie_rows) == 0:
        return result

    qknn_logits = np.asarray(
        score_zid_student_t_logits(
            pair.after_bank,
            np.ascontiguousarray(query[tie_rows], dtype=np.float32),
            metric=pair.after_metric,
        ),
        dtype=np.float32,
    )
    if qknn_logits.shape != (len(tie_rows), len(pair.registered_classes)) or not np.isfinite(
        qknn_logits
    ).all():
        raise D92LiteTarget125CoreError("exact-top-tie qKNN score drift")

    resolved = result.copy()
    positive_infinity = np.float32(np.inf)
    for local_index, row_index in enumerate(tie_rows):
        top_mask = result[row_index] == maxima[row_index, 0]
        secondary = qknn_logits[local_index, top_mask]
        secondary_max = np.max(secondary)
        secondary_winners = np.flatnonzero(top_mask)[secondary == secondary_max]
        if len(secondary_winners) != 1:
            raise D92LiteTarget125CoreError(
                "exact Lite top tie remains tied under same-after qKNN"
            )
        winner = int(secondary_winners[0])
        promoted = np.nextafter(result[row_index, winner], positive_infinity)
        if not np.isfinite(promoted):
            raise D92LiteTarget125CoreError("exact-top-tie one-ULP promotion overflow")
        resolved[row_index, winner] = promoted
    return np.ascontiguousarray(resolved)


def score(
    pair: D92LiteTarget125Pair,
    phase: str,
    arm: str,
    query_features288: np.ndarray,
) -> np.ndarray:
    """Score each query independently over the phase registry."""

    if type(pair) is not D92LiteTarget125Pair:
        raise D92LiteTarget125CoreError("pair must be an exact D92-Lite pair")
    if arm != TRANSPORT_ARM:
        raise D92LiteTarget125CoreError("only the frozen transport arm is permitted")
    query = normalized_zid160_from_registered_feature(query_features288)
    if phase == "before":
        logits = score_zid_student_t_logits(
            pair.before_bank, query, metric=pair.before_metric
        )
    elif phase == "after":
        if pair.active_k == 1:
            logits = score_zid_student_t_logits(
                pair.after_bank, query, metric=pair.after_metric
            )
        else:
            assert pair.after_lite_state is not None
            logits = d129.score_d129_affine_head(pair.after_lite_state, query)
            logits = _resolve_exact_lite_top_ties(pair, query, logits)
    else:
        raise D92LiteTarget125CoreError("phase must be before or after")
    result = np.asarray(logits, dtype=np.float32)
    expected_classes = (
        pair.old_registered_classes if phase == "before" else pair.registered_classes
    )
    if result.shape != (len(query), len(expected_classes)) or not np.isfinite(result).all():
        raise D92LiteTarget125CoreError("query logits shape/value drift")
    return np.ascontiguousarray(result)


__all__ = [
    "CANDIDATE_ID",
    "D92LiteTarget125CoreError",
    "D92LiteTarget125Pair",
    "METHOD_LOCK_SCHEMA",
    "METHOD_LOCK_SHA256",
    "OLD_CLASS_COUNT",
    "PROTOCOL_SCHEMA",
    "TRANSPORT_ARM",
    "build_d92_lite_pair",
    "normalized_zid160_from_registered_feature",
    "score",
]
