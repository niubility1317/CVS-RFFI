"""Frozen receiver-held x seen-class-LOCO proxy matrix for D129 Joint6.

This module contains no feature extractor, method state, prediction values, or
truth.  It fixes the 7 receiver x 6 held-class x {K1,K5} directional screen and
binds the Phase1-fit/support/query physical-ID facts consumed by the runner.

All six classes were already visible to the sealed Phase1 checkpoint.  The
held class is therefore a *seen-class proxy group*, never a registered-new TX.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence


MATRIX_SCHEMA = "cvs.stage2.d129.joint6.seen_class_loco_proxy.v2"
ROW_BINDING_SCHEMA = "cvs.stage2.d129.joint6.row_binding.v2"
PHASE1_FOLD_SEAL_SCHEMA = "cvs.phase1.d129.joint6.fold_asset_seal.v1"
CANDIDATE_IDS = ("CSPAR-2", "SRDH-2")
ARM_IDS = ("R0Q", "R0F", "R0L", "R1Q", "R1F", "R1L")
COMMON_ARM_IDS = ("R0Q", "R0F", "R0L")
ADAPTED_ARM_IDS = ("R1Q", "R1F", "R1L")
K_VALUES = (1, 5)
RECEIVER_COUNT = 7
CLASS_COUNT = 6
QUERY_PER_CLASS = 9
K1_SUPPORT_COUNT = CLASS_COUNT
K5_SUPPORT_COUNT = CLASS_COUNT * 5
QUERY_COUNT = CLASS_COUNT * QUERY_PER_CLASS
PHASE1_FIT_COUNT = (RECEIVER_COUNT - 1) * (CLASS_COUNT - 1) * 14
FOLD_COUNT = RECEIVER_COUNT * CLASS_COUNT
ROW_COUNT_PER_CANDIDATE = FOLD_COUNT * len(K_VALUES)


class D129Joint6MatrixError(ValueError):
    """Raised when the frozen Joint6 matrix or physical-ID binding drifts."""


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _registry(values: Sequence[str], *, expected: int, name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if (
        len(result) != expected
        or len(set(result)) != expected
        or any(not value for value in result)
    ):
        raise D129Joint6MatrixError(
            f"{name} must contain exactly {expected} unique non-empty values"
        )
    return tuple(sorted(result))


@dataclass(frozen=True)
class Joint6LocoRow:
    """Truth-free identity of one held receiver/class/K atomic row."""

    row_id: str
    held_receiver: str
    held_class: str
    active_k: int
    retained_classes: tuple[str, ...]
    registered_classes: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.row_id
            or not self.held_receiver
            or not self.held_class
            or self.active_k not in K_VALUES
            or len(self.registered_classes) != CLASS_COUNT
            or len(set(self.registered_classes)) != CLASS_COUNT
            or self.held_class not in self.registered_classes
            or self.registered_classes != self.retained_classes + (self.held_class,)
        ):
            raise D129Joint6MatrixError("Joint6 LOCO row identity drift")

    @property
    def held_proxy_classes(self) -> tuple[str, ...]:
        return (self.held_class,)


def build_joint6_loco_plan(
    receiver_ids: Sequence[str], class_ids: Sequence[str]
) -> Mapping[str, Any]:
    """Build the complete deterministic 84-row plan without data or truth."""

    receivers = _registry(
        receiver_ids, expected=RECEIVER_COUNT, name="receiver_ids"
    )
    classes = _registry(class_ids, expected=CLASS_COUNT, name="class_ids")
    rows: list[Joint6LocoRow] = []
    for receiver in receivers:
        for held_class in classes:
            retained_classes = tuple(value for value in classes if value != held_class)
            registered_classes = retained_classes + (held_class,)
            for active_k in K_VALUES:
                rows.append(
                    Joint6LocoRow(
                        row_id=f"rx={receiver}|held={held_class}|K={active_k}",
                        held_receiver=receiver,
                        held_class=held_class,
                        active_k=active_k,
                        retained_classes=retained_classes,
                        registered_classes=registered_classes,
                    )
                )
    payload: dict[str, Any] = {
        "schema": MATRIX_SCHEMA,
        "receiver_ids": list(receivers),
        "class_ids": list(classes),
        "candidate_ids": list(CANDIDATE_IDS),
        "arm_ids": list(ARM_IDS),
        "k_values": list(K_VALUES),
        "fold_count": FOLD_COUNT,
        "row_count_per_candidate": ROW_COUNT_PER_CANDIDATE,
        "rows": [
            {
                "row_id": row.row_id,
                "held_receiver": row.held_receiver,
                "held_class": row.held_class,
                "active_k": row.active_k,
                "retained_classes": list(row.retained_classes),
                "held_proxy_classes": list(row.held_proxy_classes),
                "registered_classes": list(row.registered_classes),
            }
            for row in rows
        ],
        "common_arm_cache_shared_across_candidates": True,
        "six_logical_arms_per_candidate": True,
        "evaluation_semantics": "phase1_seen_class_loco_directional_proxy",
        "formal_new_registration_claim": False,
        "truth_loaded": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
    }
    payload["matrix_sha256"] = _canonical_sha256(payload)
    return MappingProxyType(payload)


def _physical_map(
    value: Mapping[str, Sequence[str]],
    *,
    classes: tuple[str, ...],
    expected_per_class: int,
    name: str,
) -> dict[str, tuple[str, ...]]:
    if set(value) != set(classes):
        raise D129Joint6MatrixError(f"{name} class registry drift")
    result: dict[str, tuple[str, ...]] = {}
    for class_id in classes:
        rows = tuple(str(item) for item in value[class_id])
        if (
            len(rows) != expected_per_class
            or len(set(rows)) != expected_per_class
            or any(not item for item in rows)
        ):
            raise D129Joint6MatrixError(
                f"{name}[{class_id}] must contain {expected_per_class} unique IDs"
            )
        result[class_id] = rows
    return result


def bind_joint6_physical_ids(
    *,
    row_k1: Joint6LocoRow,
    row_k5: Joint6LocoRow,
    loco_fold_receipt: Mapping[str, Any],
    phase1_fit_ids: Sequence[str],
    k1_support_ids_by_class: Mapping[str, Sequence[str]],
    k5_support_ids_by_class: Mapping[str, Sequence[str]],
    query_ids_by_class: Mapping[str, Sequence[str]],
) -> Mapping[str, Any]:
    """Prove the K1 prefix and support/query physical-ID disjointness.

    The caller supplies already frozen received-IQ selections.  This function
    validates their identity only; it never selects or revalidates data.  The
    returned fold seal must be supplied to the Phase1 asset builder and checked
    again by the runtime, preventing an asset from a different held fold from
    being substituted.
    """

    if (
        row_k1.active_k != 1
        or row_k5.active_k != 5
        or row_k1.held_receiver != row_k5.held_receiver
        or row_k1.held_class != row_k5.held_class
        or row_k1.registered_classes != row_k5.registered_classes
    ):
        raise D129Joint6MatrixError("K1/K5 row pairing drift")
    classes = row_k1.registered_classes
    support1 = _physical_map(
        k1_support_ids_by_class,
        classes=classes,
        expected_per_class=1,
        name="k1_support_ids_by_class",
    )
    support5 = _physical_map(
        k5_support_ids_by_class,
        classes=classes,
        expected_per_class=5,
        name="k5_support_ids_by_class",
    )
    if any(support1[class_id] != support5[class_id][:1] for class_id in classes):
        raise D129Joint6MatrixError("K1 support must be the exact K5 prefix")
    if not isinstance(query_ids_by_class, Mapping) or set(query_ids_by_class) != set(classes):
        raise D129Joint6MatrixError("query_ids_by_class class registry drift")
    query: dict[str, tuple[str, ...]] = {}
    for class_id in classes:
        rows = tuple(str(item) for item in query_ids_by_class[class_id])
        if (
            len(rows) != QUERY_PER_CLASS
            or len(set(rows)) != len(rows)
            or any(not item for item in rows)
        ):
            raise D129Joint6MatrixError(
                f"query_ids_by_class[{class_id}] must contain {QUERY_PER_CLASS} unique IDs"
            )
        query[class_id] = rows
    support_union = {item for rows in support5.values() for item in rows}
    query_union = {item for rows in query.values() for item in rows}
    phase1_fit = tuple(str(item) for item in phase1_fit_ids)
    if (
        len(phase1_fit) != PHASE1_FIT_COUNT
        or len(set(phase1_fit)) != PHASE1_FIT_COUNT
        or any(not item for item in phase1_fit)
    ):
        raise D129Joint6MatrixError(
            f"phase1_fit_ids must contain exactly {PHASE1_FIT_COUNT} unique IDs"
        )
    if len(support_union) != CLASS_COUNT * 5:
        raise D129Joint6MatrixError("K5 support physical IDs overlap across classes")
    if len(query_union) != sum(len(rows) for rows in query.values()):
        raise D129Joint6MatrixError("query physical IDs overlap across classes")
    if support_union & query_union:
        raise D129Joint6MatrixError("support/query physical IDs overlap")
    if set(phase1_fit) & (support_union | query_union):
        raise D129Joint6MatrixError("Phase1-fit/support/query physical IDs overlap")
    phase1_root = hashlib.sha256("\n".join(phase1_fit).encode("utf-8")).hexdigest()
    support1_ordered = tuple(
        item for class_id in classes for item in support1[class_id]
    )
    support5_ordered = tuple(
        item for class_id in classes for item in support5[class_id]
    )
    query_ordered = tuple(item for class_id in classes for item in query[class_id])
    support1_root = hashlib.sha256(
        "\n".join(support1_ordered).encode("utf-8")
    ).hexdigest()
    support5_root = hashlib.sha256(
        "\n".join(support5_ordered).encode("utf-8")
    ).hexdigest()
    query_root = hashlib.sha256("\n".join(query_ordered).encode("utf-8")).hexdigest()
    expected_fold = dict(loco_fold_receipt)
    if (
        expected_fold.get("held_receiver") != row_k1.held_receiver
        or expected_fold.get("held_class") != row_k1.held_class
        or expected_fold.get("phase1_fit_count") != PHASE1_FIT_COUNT
        or expected_fold.get("phase1_fit_physical_root_sha256") != phase1_root
        or expected_fold.get("support_k1_count") != K1_SUPPORT_COUNT
        or expected_fold.get("support_k1_physical_root_sha256") != support1_root
        or expected_fold.get("support_k5_count") != K5_SUPPORT_COUNT
        or expected_fold.get("support_k5_physical_root_sha256") != support5_root
        or expected_fold.get("outer_query_count") != QUERY_COUNT
        or expected_fold.get("outer_query_physical_root_sha256") != query_root
        or expected_fold.get("k1_is_k5_prefix") is not True
    ):
        raise D129Joint6MatrixError("full LOCO plan/fold physical binding drift")
    seal_payload = {
        "schema": PHASE1_FOLD_SEAL_SCHEMA,
        "held_receiver": row_k1.held_receiver,
        "held_class": row_k1.held_class,
        "phase1_fit_count": PHASE1_FIT_COUNT,
        "phase1_fit_physical_root_sha256": phase1_root,
        "support_k1_physical_root_sha256": support1_root,
        "support_k5_physical_root_sha256": support5_root,
        "query_physical_root_sha256": query_root,
    }
    phase1_seal_sha256 = _canonical_sha256(seal_payload)
    payload: dict[str, Any] = {
        "schema": ROW_BINDING_SCHEMA,
        "held_receiver": row_k1.held_receiver,
        "held_class": row_k1.held_class,
        "k1_row_id": row_k1.row_id,
        "k5_row_id": row_k5.row_id,
        "registered_classes": list(classes),
        "evaluation_semantics": "phase1_seen_class_loco_directional_proxy",
        "formal_new_registration_claim": False,
        "phase1_fit_count": PHASE1_FIT_COUNT,
        "phase1_fit_physical_root_sha256": phase1_root,
        "support_k1_physical_root_sha256": support1_root,
        "support_k5_physical_root_sha256": support5_root,
        "query_physical_root_sha256": query_root,
        "phase1_fold_seal_schema": PHASE1_FOLD_SEAL_SCHEMA,
        "phase1_seal_sha256": phase1_seal_sha256,
        "k1_support_ids_by_class": {key: list(value) for key, value in support1.items()},
        "k5_support_ids_by_class": {key: list(value) for key, value in support5.items()},
        "query_ids_by_class": {key: list(value) for key, value in query.items()},
        "k1_is_exact_k5_prefix": True,
        "support_query_physical_ids_disjoint": True,
        "k1_support_count": K1_SUPPORT_COUNT,
        "k5_support_count": K5_SUPPORT_COUNT,
        "query_count": QUERY_COUNT,
        "data_revalidated": False,
    }
    payload["binding_sha256"] = _canonical_sha256(payload)
    return MappingProxyType(payload)


def validate_joint6_binding(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recompute the immutable row-binding digest before every runtime use."""

    if not isinstance(value, Mapping):
        raise D129Joint6MatrixError("row binding must be a mapping")
    payload = dict(value)
    observed = payload.pop("binding_sha256", None)
    if (
        value.get("schema") != ROW_BINDING_SCHEMA
        or not isinstance(observed, str)
        or observed != _canonical_sha256(payload)
    ):
        raise D129Joint6MatrixError("row binding SHA256 drift")
    return MappingProxyType(dict(value))


__all__ = [
    "ADAPTED_ARM_IDS",
    "ARM_IDS",
    "CANDIDATE_IDS",
    "CLASS_COUNT",
    "COMMON_ARM_IDS",
    "D129Joint6MatrixError",
    "FOLD_COUNT",
    "Joint6LocoRow",
    "K_VALUES",
    "K1_SUPPORT_COUNT",
    "K5_SUPPORT_COUNT",
    "MATRIX_SCHEMA",
    "PHASE1_FIT_COUNT",
    "PHASE1_FOLD_SEAL_SCHEMA",
    "RECEIVER_COUNT",
    "QUERY_COUNT",
    "QUERY_PER_CLASS",
    "ROW_BINDING_SCHEMA",
    "ROW_COUNT_PER_CANDIDATE",
    "bind_joint6_physical_ids",
    "build_joint6_loco_plan",
    "validate_joint6_binding",
]
