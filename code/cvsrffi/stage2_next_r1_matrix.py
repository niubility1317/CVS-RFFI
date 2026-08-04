"""Frozen NEXT-R1 FABR-TSL source-held LOCO matrix contract.

The matrix is deliberately a small, truth-free input contract.  It describes
one candidate over the seven receiver-held by six seen-class LOCO folds and
the two frozen K values.  Physical IDs are accepted only as an already frozen
binding; this module never selects samples, opens truth, or performs scoring.

The implementation is intentionally independent of the historical D129
schemas.  Only the compatible matrix ideas (six logical arms, a K1 prefix of
K5, and disjoint physical-ID roots) are retained.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence


MATRIX_SCHEMA = "cvs.stage2.next_r1.fabr_tsl.seen_class_loco.v1"
ROW_BINDING_SCHEMA = "cvs.stage2.next_r1.fabr_tsl.row_binding.v1"
PHASE1_FOLD_SEAL_SCHEMA = "cvs.phase1.next_r1.fabr_tsl.fold_asset_seal.v1"
CANDIDATE_ID = "NEXT-R1"
CANDIDATE_IDS = (CANDIDATE_ID,)
METHOD_LOCK = "NEXT-R1 FABR-TSL"
ARM_IDS = ("R0Q", "R0F", "R0L", "R1Q", "R1F", "R1L")
COMMON_ARM_IDS = ("R0Q", "R0F", "R0L")
ADAPTED_ARM_IDS = ("R1Q", "R1F", "R1L")
K_VALUES = (1, 5)
RECEIVER_COUNT = 7
CLASS_COUNT = 6
FOLD_COUNT = RECEIVER_COUNT * CLASS_COUNT
PHYSICAL_PER_CELL = 14
QUERY_PER_CLASS = 9
K1_SUPPORT_COUNT = CLASS_COUNT
K5_SUPPORT_COUNT = CLASS_COUNT * 5
QUERY_COUNT = CLASS_COUNT * QUERY_PER_CLASS
PHASE1_FIT_COUNT = (RECEIVER_COUNT - 1) * (CLASS_COUNT - 1) * PHYSICAL_PER_CELL
ROW_COUNT = FOLD_COUNT * len(K_VALUES)
ROW_COUNT_PER_CANDIDATE = ROW_COUNT

# These are field names that would make the matrix depend on an unavailable
# oracle or a cross-query assignment policy.  Values (for example, an opaque
# class token containing ``role``) are not inspected.
_FORBIDDEN_FIELD_FRAGMENTS = ("truth", "role", "quota", "assignment", "global")


class NextR1MatrixError(ValueError):
    """Raised when the frozen NEXT-R1 matrix or physical binding drifts."""


# A descriptive alias is useful to callers that use the older ``MatrixError``
# naming convention while keeping the new schema and candidate closure clear.
NEXT_R1MatrixError = NextR1MatrixError


def _json_ready(value: Any) -> Any:
    """Convert immutable containers to JSON-compatible canonical values."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported canonical value type: {type(value)!r}")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    """Public helper for constructing matching external fold receipts."""

    return _canonical_sha256(value)


def _assert_no_forbidden_fields(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).lower()
            if any(fragment in name for fragment in _FORBIDDEN_FIELD_FRAGMENTS):
                raise NextR1MatrixError(f"forbidden matrix field: {path}.{key}")
            _assert_no_forbidden_fields(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_no_forbidden_fields(item, path=f"{path}[{index}]")


def _registry(values: Sequence[str], *, expected: int, name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if (
        len(result) != expected
        or len(set(result)) != expected
        or any(not value for value in result)
    ):
        raise NextR1MatrixError(
            f"{name} must contain exactly {expected} unique non-empty values"
        )
    return tuple(sorted(result))


@dataclass(frozen=True)
class NextR1LocoRow:
    """Truth-free identity of one receiver-held/class-held/K atomic row."""

    row_id: str
    held_receiver: str
    held_class: str
    active_k: int
    retained_classes: tuple[str, ...]
    registered_classes: tuple[str, ...]
    candidate_id: str = CANDIDATE_ID

    def __post_init__(self) -> None:
        if (
            self.candidate_id != CANDIDATE_ID
            or not self.row_id
            or not self.held_receiver
            or not self.held_class
            or self.active_k not in K_VALUES
            or len(self.retained_classes) != CLASS_COUNT - 1
            or len(set(self.retained_classes)) != CLASS_COUNT - 1
            or len(self.registered_classes) != CLASS_COUNT
            or len(set(self.registered_classes)) != CLASS_COUNT
            or self.held_class not in self.registered_classes
            or self.registered_classes != self.retained_classes + (self.held_class,)
        ):
            raise NextR1MatrixError("NEXT-R1 LOCO row identity drift")

    @property
    def held_proxy_classes(self) -> tuple[str, ...]:
        """The held class is a seen-class proxy, never a new-class claim."""

        return (self.held_class,)


# Alternate concise names keep the contract easy to discover without creating
# a second schema or a second implementation.
NextR1Row = NextR1LocoRow


def _row_to_dict(row: NextR1LocoRow) -> dict[str, Any]:
    return {
        "candidate_id": row.candidate_id,
        "row_id": row.row_id,
        "held_receiver": row.held_receiver,
        "held_class": row.held_class,
        "active_k": row.active_k,
        "retained_classes": list(row.retained_classes),
        "held_proxy_classes": list(row.held_proxy_classes),
        "registered_classes": list(row.registered_classes),
        "arm_ids": list(ARM_IDS),
    }


def build_next_r1_loco_plan(
    receiver_ids: Sequence[str], class_ids: Sequence[str]
) -> Mapping[str, Any]:
    """Build the complete deterministic 84-row NEXT-R1 plan.

    Receiver and class input order is ignored.  The returned digest covers the
    full matrix, candidate closure, arm closure, and row identities.
    """

    receivers = _registry(receiver_ids, expected=RECEIVER_COUNT, name="receiver_ids")
    classes = _registry(class_ids, expected=CLASS_COUNT, name="class_ids")
    rows: list[NextR1LocoRow] = []
    for receiver in receivers:
        for held_class in classes:
            retained_classes = tuple(value for value in classes if value != held_class)
            registered_classes = retained_classes + (held_class,)
            for active_k in K_VALUES:
                rows.append(
                    NextR1LocoRow(
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
        "method_lock": METHOD_LOCK,
        "candidate_id": CANDIDATE_ID,
        "candidate_ids": list(CANDIDATE_IDS),
        "receiver_ids": list(receivers),
        "class_ids": list(classes),
        "arm_ids": list(ARM_IDS),
        "common_arm_ids": list(COMMON_ARM_IDS),
        "adapted_arm_ids": list(ADAPTED_ARM_IDS),
        "k_values": list(K_VALUES),
        "receiver_count": RECEIVER_COUNT,
        "class_count": CLASS_COUNT,
        "fold_count": FOLD_COUNT,
        "row_count": ROW_COUNT,
        "row_count_per_candidate": ROW_COUNT_PER_CANDIDATE,
        "rows": [_row_to_dict(row) for row in rows],
        "common_arm_cache_shared": True,
        "six_logical_arms_per_row": True,
        "evaluation_semantics": "phase1_seen_class_loco_directional_proxy",
        "query_policy": "independent_per_sample_all_registered_classes",
    }
    _assert_no_forbidden_fields(payload)
    payload["matrix_sha256"] = _canonical_sha256(payload)
    return MappingProxyType(payload)


# Matrix-oriented aliases; all point to the same implementation and schema.
build_next_r1_matrix = build_next_r1_loco_plan
build_next_r1_plan = build_next_r1_loco_plan


def validate_next_r1_plan(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the matrix digest and frozen coverage invariants."""

    if not isinstance(value, Mapping):
        raise NextR1MatrixError("NEXT-R1 matrix must be a mapping")
    payload = dict(value)
    observed = payload.pop("matrix_sha256", None)
    if not isinstance(observed, str) or observed != _canonical_sha256(payload):
        raise NextR1MatrixError("NEXT-R1 matrix SHA256 drift")
    _assert_no_forbidden_fields(payload)
    if (
        payload.get("schema") != MATRIX_SCHEMA
        or payload.get("method_lock") != METHOD_LOCK
        or payload.get("candidate_id") != CANDIDATE_ID
        or payload.get("candidate_ids") != list(CANDIDATE_IDS)
        or payload.get("arm_ids") != list(ARM_IDS)
        or payload.get("common_arm_ids") != list(COMMON_ARM_IDS)
        or payload.get("adapted_arm_ids") != list(ADAPTED_ARM_IDS)
        or payload.get("k_values") != list(K_VALUES)
        or payload.get("fold_count") != FOLD_COUNT
        or payload.get("row_count") != ROW_COUNT
        or payload.get("row_count_per_candidate") != ROW_COUNT_PER_CANDIDATE
        or payload.get("common_arm_cache_shared") is not True
        or payload.get("six_logical_arms_per_row") is not True
    ):
        raise NextR1MatrixError("NEXT-R1 frozen matrix coverage drift")
    receivers = _registry(payload.get("receiver_ids", ()), expected=RECEIVER_COUNT, name="receiver_ids")
    classes = _registry(payload.get("class_ids", ()), expected=CLASS_COUNT, name="class_ids")
    rows = payload.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != ROW_COUNT:
        raise NextR1MatrixError("NEXT-R1 row coverage drift")
    seen: set[tuple[str, str, int]] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise NextR1MatrixError("NEXT-R1 row must be a mapping")
        if raw.get("candidate_id") != CANDIDATE_ID or raw.get("arm_ids") != list(ARM_IDS):
            raise NextR1MatrixError("NEXT-R1 row candidate/arm drift")
        key = (str(raw.get("held_receiver")), str(raw.get("held_class")), raw.get("active_k"))
        if key in seen or key[0] not in receivers or key[1] not in classes or key[2] not in K_VALUES:
            raise NextR1MatrixError("NEXT-R1 row coverage drift")
        seen.add(key)
        retained = tuple(raw.get("retained_classes", ()))
        registered = tuple(raw.get("registered_classes", ()))
        expected_retained = tuple(item for item in classes if item != key[1])
        if retained != expected_retained or registered != expected_retained + (key[1],):
            raise NextR1MatrixError("NEXT-R1 row class binding drift")
    if len(seen) != ROW_COUNT:
        raise NextR1MatrixError("NEXT-R1 row coverage is incomplete")
    return MappingProxyType(dict(value))


def _physical_map(
    value: Mapping[str, Sequence[str]],
    *,
    classes: tuple[str, ...],
    expected_per_class: int,
    name: str,
) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or set(value) != set(classes):
        raise NextR1MatrixError(f"{name} class registry drift")
    result: dict[str, tuple[str, ...]] = {}
    for class_id in classes:
        rows = tuple(str(item) for item in value[class_id])
        if (
            len(rows) != expected_per_class
            or len(set(rows)) != expected_per_class
            or any(not item for item in rows)
        ):
            raise NextR1MatrixError(
                f"{name}[{class_id}] must contain {expected_per_class} unique IDs"
            )
        result[class_id] = rows
    return result


def _id_root(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _receipt_value(receipt: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in receipt:
            return receipt[name]
    return None


def bind_next_r1_physical_ids(
    *,
    row_k1: NextR1LocoRow,
    row_k5: NextR1LocoRow,
    loco_fold_receipt: Mapping[str, Any],
    phase1_fit_ids: Sequence[str],
    k1_support_ids_by_class: Mapping[str, Sequence[str]],
    k5_support_ids_by_class: Mapping[str, Sequence[str]],
    query_ids_by_class: Mapping[str, Sequence[str]],
) -> Mapping[str, Any]:
    """Bind frozen physical IDs for one K1/K5 fold pair.

    The function proves only identity, prefix, root, and disjointness facts;
    it does not select IDs or access received IQ.  An external fold receipt is
    checked before a new immutable binding digest is returned.
    """

    if not isinstance(row_k1, NextR1LocoRow) or not isinstance(row_k5, NextR1LocoRow):
        raise NextR1MatrixError("NEXT-R1 row binding requires NextR1LocoRow values")
    if (
        row_k1.active_k != 1
        or row_k5.active_k != 5
        or row_k1.candidate_id != CANDIDATE_ID
        or row_k5.candidate_id != CANDIDATE_ID
        or row_k1.held_receiver != row_k5.held_receiver
        or row_k1.held_class != row_k5.held_class
        or row_k1.registered_classes != row_k5.registered_classes
    ):
        raise NextR1MatrixError("NEXT-R1 K1/K5 row pairing drift")
    if not isinstance(loco_fold_receipt, Mapping):
        raise NextR1MatrixError("NEXT-R1 LOCO fold receipt must be a mapping")
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
        raise NextR1MatrixError("K1 support must be the exact K5 prefix")
    query = _physical_map(
        query_ids_by_class,
        classes=classes,
        expected_per_class=QUERY_PER_CLASS,
        name="query_ids_by_class",
    )
    phase1_fit = tuple(str(item) for item in phase1_fit_ids)
    if (
        len(phase1_fit) != PHASE1_FIT_COUNT
        or len(set(phase1_fit)) != PHASE1_FIT_COUNT
        or any(not item for item in phase1_fit)
    ):
        raise NextR1MatrixError(
            f"phase1_fit_ids must contain exactly {PHASE1_FIT_COUNT} unique IDs"
        )
    support_union = {item for rows in support5.values() for item in rows}
    query_union = {item for rows in query.values() for item in rows}
    if len(support_union) != K5_SUPPORT_COUNT:
        raise NextR1MatrixError("K5 support physical IDs overlap across classes")
    if len(query_union) != QUERY_COUNT:
        raise NextR1MatrixError("query physical IDs overlap across classes")
    if support_union & query_union:
        raise NextR1MatrixError("support/query physical IDs overlap")
    if set(phase1_fit) & (support_union | query_union):
        raise NextR1MatrixError("Phase1-fit/support/query physical IDs overlap")

    phase1_root = _id_root(phase1_fit)
    support1_ordered = tuple(item for class_id in classes for item in support1[class_id])
    support5_ordered = tuple(item for class_id in classes for item in support5[class_id])
    query_ordered = tuple(item for class_id in classes for item in query[class_id])
    support1_root = _id_root(support1_ordered)
    support5_root = _id_root(support5_ordered)
    query_root = _id_root(query_ordered)

    expected_fold = loco_fold_receipt
    outer_query_count = _receipt_value(expected_fold, "outer_query_count", "query_count")
    outer_query_root = _receipt_value(
        expected_fold, "outer_query_physical_root_sha256", "query_physical_root_sha256"
    )
    prefix_flag = _receipt_value(
        expected_fold, "k1_is_k5_prefix", "k1_support_is_k5_prefix"
    )
    if (
        expected_fold.get("held_receiver") != row_k1.held_receiver
        or expected_fold.get("held_class") != row_k1.held_class
        or expected_fold.get("phase1_fit_count") != PHASE1_FIT_COUNT
        or expected_fold.get("phase1_fit_physical_root_sha256") != phase1_root
        or expected_fold.get("support_k1_count") != K1_SUPPORT_COUNT
        or expected_fold.get("support_k1_physical_root_sha256") != support1_root
        or expected_fold.get("support_k5_count") != K5_SUPPORT_COUNT
        or expected_fold.get("support_k5_physical_root_sha256") != support5_root
        or outer_query_count != QUERY_COUNT
        or outer_query_root != query_root
        or prefix_flag is not True
    ):
        raise NextR1MatrixError("NEXT-R1 LOCO plan/fold physical binding drift")

    seal_payload: dict[str, Any] = {
        "schema": PHASE1_FOLD_SEAL_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "held_receiver": row_k1.held_receiver,
        "held_class": row_k1.held_class,
        "phase1_fit_count": PHASE1_FIT_COUNT,
        "phase1_fit_physical_root_sha256": phase1_root,
        "support_k1_count": K1_SUPPORT_COUNT,
        "support_k1_physical_root_sha256": support1_root,
        "support_k5_count": K5_SUPPORT_COUNT,
        "support_k5_physical_root_sha256": support5_root,
        "query_count": QUERY_COUNT,
        "query_physical_root_sha256": query_root,
        "k1_is_exact_k5_prefix": True,
        "support_query_phase1_physical_ids_disjoint": True,
    }
    phase1_seal_sha256 = _canonical_sha256(seal_payload)
    payload: dict[str, Any] = {
        "schema": ROW_BINDING_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "held_receiver": row_k1.held_receiver,
        "held_class": row_k1.held_class,
        "k1_row_id": row_k1.row_id,
        "k5_row_id": row_k5.row_id,
        "registered_classes": list(classes),
        "phase1_fit_count": PHASE1_FIT_COUNT,
        "phase1_fit_physical_root_sha256": phase1_root,
        "support_k1_count": K1_SUPPORT_COUNT,
        "support_k1_physical_root_sha256": support1_root,
        "support_k5_count": K5_SUPPORT_COUNT,
        "support_k5_physical_root_sha256": support5_root,
        "query_count": QUERY_COUNT,
        "query_physical_root_sha256": query_root,
        "phase1_fold_seal_schema": PHASE1_FOLD_SEAL_SCHEMA,
        "phase1_seal_sha256": phase1_seal_sha256,
        "k1_support_ids_by_class": {key: list(value) for key, value in support1.items()},
        "k5_support_ids_by_class": {key: list(value) for key, value in support5.items()},
        "query_ids_by_class": {key: list(value) for key, value in query.items()},
        "k1_is_exact_k5_prefix": True,
        "support_query_phase1_physical_ids_disjoint": True,
        "validated_once_reused": True,
    }
    _assert_no_forbidden_fields(payload)
    payload["binding_sha256"] = _canonical_sha256(payload)
    return MappingProxyType(payload)


bind_next_r1_physical_id_binding = bind_next_r1_physical_ids
bind_next_r1_row_physical_ids = bind_next_r1_physical_ids


def validate_next_r1_binding(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recompute and structurally validate an immutable row-binding digest."""

    if not isinstance(value, Mapping):
        raise NextR1MatrixError("NEXT-R1 row binding must be a mapping")
    payload = dict(value)
    observed = payload.pop("binding_sha256", None)
    if (
        payload.get("schema") != ROW_BINDING_SCHEMA
        or payload.get("candidate_id") != CANDIDATE_ID
        or not isinstance(observed, str)
        or observed != _canonical_sha256(payload)
    ):
        raise NextR1MatrixError("NEXT-R1 row binding SHA256 drift")
    _assert_no_forbidden_fields(payload)
    classes = tuple(payload.get("registered_classes", ()))
    if (
        len(classes) != CLASS_COUNT
        or len(set(classes)) != CLASS_COUNT
        or payload.get("phase1_fit_count") != PHASE1_FIT_COUNT
        or payload.get("support_k1_count") != K1_SUPPORT_COUNT
        or payload.get("support_k5_count") != K5_SUPPORT_COUNT
        or payload.get("query_count") != QUERY_COUNT
        or payload.get("k1_is_exact_k5_prefix") is not True
        or payload.get("support_query_phase1_physical_ids_disjoint") is not True
        or payload.get("validated_once_reused") is not True
    ):
        raise NextR1MatrixError("NEXT-R1 row binding invariant drift")
    support1 = _physical_map(
        payload.get("k1_support_ids_by_class", {}),
        classes=classes,
        expected_per_class=1,
        name="k1_support_ids_by_class",
    )
    support5 = _physical_map(
        payload.get("k5_support_ids_by_class", {}),
        classes=classes,
        expected_per_class=5,
        name="k5_support_ids_by_class",
    )
    query = _physical_map(
        payload.get("query_ids_by_class", {}),
        classes=classes,
        expected_per_class=QUERY_PER_CLASS,
        name="query_ids_by_class",
    )
    if any(support1[key] != support5[key][:1] for key in classes):
        raise NextR1MatrixError("NEXT-R1 binding K1/K5 prefix drift")
    support_union = {item for rows in support5.values() for item in rows}
    query_union = {item for rows in query.values() for item in rows}
    if len(support_union) != K5_SUPPORT_COUNT or len(query_union) != QUERY_COUNT:
        raise NextR1MatrixError("NEXT-R1 binding physical-ID duplication")
    if support_union & query_union:
        raise NextR1MatrixError("NEXT-R1 binding support/query overlap")
    if payload.get("support_k1_physical_root_sha256") != _id_root(
        tuple(item for key in classes for item in support1[key])
    ):
        raise NextR1MatrixError("NEXT-R1 binding K1 root drift")
    if payload.get("support_k5_physical_root_sha256") != _id_root(
        tuple(item for key in classes for item in support5[key])
    ):
        raise NextR1MatrixError("NEXT-R1 binding K5 root drift")
    if payload.get("query_physical_root_sha256") != _id_root(
        tuple(item for key in classes for item in query[key])
    ):
        raise NextR1MatrixError("NEXT-R1 binding query root drift")
    return MappingProxyType(dict(value))


validate_next_r1_row_binding = validate_next_r1_binding


__all__ = [
    "ADAPTED_ARM_IDS",
    "ARM_IDS",
    "CANDIDATE_ID",
    "CANDIDATE_IDS",
    "CLASS_COUNT",
    "COMMON_ARM_IDS",
    "FOLD_COUNT",
    "K_VALUES",
    "K1_SUPPORT_COUNT",
    "K5_SUPPORT_COUNT",
    "MATRIX_SCHEMA",
    "METHOD_LOCK",
    "NextR1LocoRow",
    "NextR1MatrixError",
    "NEXT_R1MatrixError",
    "NextR1Row",
    "PHASE1_FIT_COUNT",
    "PHASE1_FOLD_SEAL_SCHEMA",
    "PHYSICAL_PER_CELL",
    "QUERY_COUNT",
    "QUERY_PER_CLASS",
    "RECEIVER_COUNT",
    "ROW_BINDING_SCHEMA",
    "ROW_COUNT",
    "ROW_COUNT_PER_CANDIDATE",
    "bind_next_r1_physical_id_binding",
    "bind_next_r1_physical_ids",
    "bind_next_r1_row_physical_ids",
    "build_next_r1_loco_plan",
    "build_next_r1_matrix",
    "build_next_r1_plan",
    "canonical_sha256",
    "validate_next_r1_binding",
    "validate_next_r1_plan",
    "validate_next_r1_row_binding",
]
