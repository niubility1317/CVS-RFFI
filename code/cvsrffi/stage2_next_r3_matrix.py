"""Frozen NEXT-R3 RDCExTSL-160 source-held proxy matrix.

The matrix is intentionally mechanical.  It fixes two preregistered held
receivers (``1-1`` and ``18-2``), accepts the six held classes from the call
site, and emits one candidate with 24 atomic rows (receiver x class x K).
Truth and prediction values are not part of this module.  Physical-ID
binding helpers are kept small and are compatible with the D129 row-binding
shape so an existing runner can reuse its disjointness checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence


SCHEMA = "cvs.stage2.next_r3.rdce_tsl160.proxy24.plan.v1"
ROW_BINDING_SCHEMA = "cvs.stage2.next_r3.rdce_tsl160.row_binding.v1"
PROTOCOL_SCHEMA = "p2_min_v1"
CANDIDATE_ID = "NEXT-R3-RDCE-TSL160-R1"
REPRESENTATION_RULE = "d106_canonical_normalized_relu_zid160"
PROXY_SEMANTICS = "SOURCE_HELD_PROXY"

HELD_RECEIVERS = ("1-1", "18-2")
STATE_IDS = ("DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1")
REGISTRATION_IDS = ("REG0", "REG1")
ARM_IDS = ("R0Q", "R0F", "R0L", "R1Q", "R1F", "R1L")
COMMON_ARM_IDS = ("R0Q", "R0F", "R0L")
ADAPTED_ARM_IDS = ("R1Q", "R1F", "R1L")
DA1_STATES = frozenset(("DA1_REG0", "DA1_REG1"))
REG1_STATES = frozenset(("DA0_REG1", "DA1_REG1"))
REG0_STATES = frozenset(("DA0_REG0", "DA1_REG0"))
K_VALUES = (1, 5)
CLASS_COUNT = 6
SELECTED_RECEIVER_COUNT = 2
ROW_COUNT = SELECTED_RECEIVER_COUNT * CLASS_COUNT * len(K_VALUES)
OUTER_KEY_COUNT = ROW_COUNT
STATE_COUNT = len(STATE_IDS)
STATE_PREDICTION_COUNT = OUTER_KEY_COUNT * STATE_COUNT
ARM_PREDICTION_COUNT = STATE_PREDICTION_COUNT * len(ARM_IDS)
QUERY_PER_CLASS = 9
PHYSICAL_PER_CLASS = 14
MAX_SUPPORT_K = 5
K1_SUPPORT_COUNT = CLASS_COUNT
K5_SUPPORT_COUNT = CLASS_COUNT * MAX_SUPPORT_K
PHASE1_RECEIVER_COUNT = 7
PHASE1_FIT_COUNT = (PHASE1_RECEIVER_COUNT - 1) * (CLASS_COUNT - 1) * PHYSICAL_PER_CLASS


class NextR3MatrixError(ValueError):
    """Raised when the frozen NEXT-R3 matrix or row binding drifts."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise NextR3MatrixError("matrix metadata contains an unsupported value")


def canonical_bytes(value: Any) -> bytes:
    """Canonical JSON bytes used by matrix, row and scorer receipts."""

    return json.dumps(
        _json_ready(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise NextR3MatrixError(f"{name} must be a lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise NextR3MatrixError(f"{name} must be a lowercase SHA256") from error
    return value


def _registry(
    values: Sequence[str], *, expected: int, name: str, sort_values: bool = True
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise NextR3MatrixError(f"{name} must be an ordered string registry")
    result = tuple(values)
    if (
        len(result) != expected
        or any(not isinstance(item, str) or not item for item in result)
        or len(set(result)) != expected
    ):
        raise NextR3MatrixError(
            f"{name} must contain exactly {expected} unique nonempty handles"
        )
    return tuple(sorted(result)) if sort_values else result


def _exact_held_receivers(values: Sequence[str]) -> tuple[str, str]:
    result = tuple(values)
    if result != HELD_RECEIVERS:
        raise NextR3MatrixError(
            "held_receivers are frozen to the ordered pair ('1-1', '18-2')"
        )
    return result


@dataclass(frozen=True, slots=True)
class NextR3ProxyRow:
    """Identity of one receiver-held/class-LOCO/K atomic row."""

    row_id: str
    held_receiver: str
    held_class: str
    active_k: int
    retained_classes: tuple[str, ...]
    all_registered_classes: tuple[str, ...]

    def __post_init__(self) -> None:
        classes = _registry(
            self.all_registered_classes,
            expected=CLASS_COUNT,
            name="all_registered_classes",
            sort_values=False,
        )
        retained = _registry(
            self.retained_classes,
            expected=CLASS_COUNT - 1,
            name="retained_classes",
            sort_values=False,
        )
        if (
            not isinstance(self.row_id, str)
            or not self.row_id.startswith("r3-")
            or self.held_receiver not in HELD_RECEIVERS
            or not isinstance(self.held_class, str)
            or self.held_class not in classes
            or self.active_k not in K_VALUES
            or retained != tuple(item for item in classes if item != self.held_class)
        ):
            raise NextR3MatrixError("NEXT-R3 row identity drift")
        object.__setattr__(self, "retained_classes", retained)
        object.__setattr__(self, "all_registered_classes", classes)

    @property
    def registered_classes(self) -> tuple[str, ...]:
        """D129-compatible alias for the complete source class registry."""

        return self.all_registered_classes

    @property
    def row_key(self) -> tuple[str, str, int]:
        return self.held_receiver, self.held_class, self.active_k

    def registration_classes(self, registration_id: str) -> tuple[str, ...]:
        if registration_id == "REG0":
            return self.retained_classes
        if registration_id == "REG1":
            return self.all_registered_classes
        raise NextR3MatrixError(f"unknown registration ID {registration_id}")

    def as_dict(self) -> dict[str, Any]:
        registrations = {
            registration_id: {
                "registration_id": registration_id,
                "registered_classes": list(self.registration_classes(registration_id)),
                "state_ids": list(STATE_IDS),
                "arm_ids": list(ARM_IDS),
                "query_per_class": QUERY_PER_CLASS,
                "query_count": len(self.registration_classes(registration_id))
                * QUERY_PER_CLASS,
            }
            for registration_id in REGISTRATION_IDS
        }
        return {
            "row_id": self.row_id,
            "held_receiver": self.held_receiver,
            "held_class": self.held_class,
            "active_k": self.active_k,
            "retained_classes": list(self.retained_classes),
            "all_registered_classes": list(self.all_registered_classes),
            "registered_classes": list(self.all_registered_classes),
            "state_ids": list(STATE_IDS),
            "arm_ids": list(ARM_IDS),
            "registrations": registrations,
            "evaluation_semantics": PROXY_SEMANTICS,
            "artifact_semantics": PROXY_SEMANTICS,
            "formal_new_registration_claim": False,
        }


# Friendly names used by D129-shaped callers.
NextR3OuterKey = NextR3ProxyRow
NextR3LocoRow = NextR3ProxyRow


def _row_id(receiver: str, held_class: str, active_k: int) -> str:
    digest = hashlib.sha256(
        f"{SCHEMA}|{receiver}|{held_class}|K{active_k}".encode("utf-8")
    ).hexdigest()
    return f"r3-{digest[:24]}"


def _resolve_build_inputs(
    class_registry: Sequence[str] | None,
    held_classes: Sequence[str] | None,
    held_receivers: Sequence[str],
) -> tuple[tuple[str, str], tuple[str, ...]]:
    """Accept both ``(classes, *, held_receivers=...)`` and D129-style pairs."""

    receivers = _exact_held_receivers(held_receivers)
    if class_registry is None and held_classes is None:
        raise NextR3MatrixError("six held classes must be supplied explicitly")
    if held_classes is None:
        classes_input = class_registry
    elif class_registry is None:
        classes_input = held_classes
    else:
        first = tuple(class_registry)
        # Compatibility with build_plan(held_receivers, held_classes).
        if first != receivers:
            raise NextR3MatrixError(
                "two positional registries must be (held_receivers, held_classes)"
            )
        classes_input = held_classes
    classes = _registry(classes_input, expected=CLASS_COUNT, name="held_classes")
    return receivers, classes


def build_next_r3_proxy24_plan(
    class_registry: Sequence[str] | None = None,
    held_classes: Sequence[str] | None = None,
    *,
    held_receivers: Sequence[str] = HELD_RECEIVERS,
    class_ids: Sequence[str] | None = None,
    receiver_ids: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    """Build the immutable 24-row plan without target data or truth.

    ``class_registry`` is the preferred first argument.  For callers that
    mirror D129, ``build_next_r3_proxy24_plan(held_receivers, held_classes)``
    is also accepted; the receiver pair remains exact and cannot be chosen by
    performance.  ``class_ids``/``receiver_ids`` are spelling aliases only.
    """

    if class_ids is not None:
        if class_registry is not None:
            raise NextR3MatrixError("class registry supplied twice")
        class_registry = class_ids
    if receiver_ids is not None:
        if tuple(held_receivers) != HELD_RECEIVERS:
            raise NextR3MatrixError("held receiver pair supplied twice")
        held_receivers = receiver_ids
    receivers, classes = _resolve_build_inputs(
        class_registry, held_classes, held_receivers
    )
    rows = tuple(
        NextR3ProxyRow(
            row_id=_row_id(receiver, held_class, active_k),
            held_receiver=receiver,
            held_class=held_class,
            active_k=active_k,
            retained_classes=tuple(item for item in classes if item != held_class),
            all_registered_classes=classes,
        )
        for receiver in receivers
        for held_class in classes
        for active_k in K_VALUES
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "representation_rule": REPRESENTATION_RULE,
        "proxy_artifact_state_prefix": PROXY_SEMANTICS,
        "evaluation_semantics": PROXY_SEMANTICS,
        "artifact_semantics": PROXY_SEMANTICS,
        "formal_new_registration_claim": False,
        "held_receivers": list(receivers),
        "held_receiver_count": SELECTED_RECEIVER_COUNT,
        "held_classes": list(classes),
        "held_class_count": CLASS_COUNT,
        "k_values": list(K_VALUES),
        "state_ids": list(STATE_IDS),
        "registration_ids": list(REGISTRATION_IDS),
        "arm_ids": list(ARM_IDS),
        "row_count": ROW_COUNT,
        "outer_key_count": OUTER_KEY_COUNT,
        "state_prediction_count": STATE_PREDICTION_COUNT,
        "arm_prediction_count": ARM_PREDICTION_COUNT,
        "query_per_class": QUERY_PER_CLASS,
        "rows": [row.as_dict() for row in rows],
        "single_candidate": True,
        "candidate_count": 1,
        "truth_loaded": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_truth_access": False,
        "query_role_access": False,
        "class_quota_access": False,
        "true_batch_class_count_access": False,
        "global_reassignment": False,
        "source_runtime_access": False,
        "clean_runtime_access": False,
        "output_overwrite_allowed": False,
    }
    payload["matrix_sha256"] = canonical_sha256(payload)
    return MappingProxyType(_json_ready(payload))


def outer_key_from_mapping(value: Mapping[str, Any]) -> NextR3ProxyRow:
    if not isinstance(value, Mapping):
        raise NextR3MatrixError("row mapping must be an object")
    try:
        raw_classes = value.get("all_registered_classes")
        if raw_classes is None:
            raw_classes = value["registered_classes"]
        classes = tuple(raw_classes)
        retained = tuple(value["retained_classes"])
        return NextR3ProxyRow(
            row_id=str(value["row_id"]),
            held_receiver=str(value["held_receiver"]),
            held_class=str(value["held_class"]),
            active_k=int(value["active_k"]),
            retained_classes=retained,
            all_registered_classes=classes,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise NextR3MatrixError("row mapping is incomplete") from error


def row_from_mapping(value: Mapping[str, Any]) -> NextR3ProxyRow:
    return outer_key_from_mapping(value)


def registered_classes_for_state(
    row: NextR3ProxyRow, state_id: str
) -> tuple[str, ...]:
    if state_id not in STATE_IDS:
        raise NextR3MatrixError(f"unknown NEXT-R3 four-state ID {state_id}")
    return row.registration_classes("REG1" if state_id in REG1_STATES else "REG0")


def registration_for_state(state_id: str) -> str:
    if state_id in REG0_STATES:
        return "REG0"
    if state_id in REG1_STATES:
        return "REG1"
    raise NextR3MatrixError(f"unknown NEXT-R3 state {state_id}")


def query_count_for_state(row: NextR3ProxyRow, state_id: str) -> int:
    return len(registered_classes_for_state(row, state_id)) * QUERY_PER_CLASS


def _physical_map(
    value: Mapping[str, Sequence[str]],
    *,
    classes: tuple[str, ...],
    expected_per_class: int,
    name: str,
) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or set(value) != set(classes):
        raise NextR3MatrixError(f"{name} class registry drift")
    result: dict[str, tuple[str, ...]] = {}
    for class_id in classes:
        ids = tuple(value[class_id])
        if (
            len(ids) != expected_per_class
            or len(set(ids)) != expected_per_class
            or any(not isinstance(item, str) or not item for item in ids)
        ):
            raise NextR3MatrixError(
                f"{name}[{class_id}] must contain {expected_per_class} unique IDs"
            )
        result[class_id] = ids
    return result


def _root(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def bind_next_r3_physical_ids(
    *,
    row_k1: NextR3ProxyRow,
    row_k5: NextR3ProxyRow,
    loco_fold_receipt: Mapping[str, Any],
    phase1_fit_ids: Sequence[str],
    k1_support_ids_by_class: Mapping[str, Sequence[str]],
    k5_support_ids_by_class: Mapping[str, Sequence[str]],
    query_ids_by_class: Mapping[str, Sequence[str]],
) -> Mapping[str, Any]:
    """Bind K1/K5 prefixes and support/query/Phase1 physical-ID disjointness."""

    if not isinstance(row_k1, NextR3ProxyRow) or not isinstance(row_k5, NextR3ProxyRow):
        raise NextR3MatrixError("row_k1/row_k5 must be exact NEXT-R3 rows")
    if (
        row_k1.active_k != 1
        or row_k5.active_k != 5
        or row_k1.held_receiver != row_k5.held_receiver
        or row_k1.held_class != row_k5.held_class
        or row_k1.all_registered_classes != row_k5.all_registered_classes
    ):
        raise NextR3MatrixError("K1/K5 row pairing drift")
    classes = row_k1.all_registered_classes
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
    query = _physical_map(
        query_ids_by_class,
        classes=classes,
        expected_per_class=QUERY_PER_CLASS,
        name="query_ids_by_class",
    )
    if any(support1[c] != support5[c][:1] for c in classes):
        raise NextR3MatrixError("K1 support must be the exact K5 prefix")
    support_union = {item for values in support5.values() for item in values}
    query_union = {item for values in query.values() for item in values}
    phase1 = tuple(phase1_fit_ids)
    if (
        len(phase1) != PHASE1_FIT_COUNT
        or len(set(phase1)) != PHASE1_FIT_COUNT
        or any(not isinstance(item, str) or not item for item in phase1)
    ):
        raise NextR3MatrixError(
            f"phase1_fit_ids must contain exactly {PHASE1_FIT_COUNT} unique IDs"
        )
    if len(support_union) != K5_SUPPORT_COUNT:
        raise NextR3MatrixError("K5 support physical IDs overlap across classes")
    if len(query_union) != CLASS_COUNT * QUERY_PER_CLASS:
        raise NextR3MatrixError("query physical IDs overlap across classes")
    if support_union & query_union:
        raise NextR3MatrixError("support/query physical IDs overlap")
    if set(phase1) & (support_union | query_union):
        raise NextR3MatrixError("Phase1-fit/support/query physical IDs overlap")
    fold = dict(loco_fold_receipt)
    phase1_root = _root(phase1)
    support1_ordered = tuple(item for c in classes for item in support1[c])
    support5_ordered = tuple(item for c in classes for item in support5[c])
    query_ordered = tuple(item for c in classes for item in query[c])
    support1_root = _root(support1_ordered)
    support5_root = _root(support5_ordered)
    query_root = _root(query_ordered)
    # D129 fold receipts use these names.  Check only fields that are present
    # so a compact NEXT-R3 receipt remains usable, but never silently accept a
    # conflicting value.
    expected_fold = {
        "held_receiver": row_k1.held_receiver,
        "held_class": row_k1.held_class,
        "phase1_fit_count": PHASE1_FIT_COUNT,
        "phase1_fit_physical_root_sha256": phase1_root,
        "support_k1_count": K1_SUPPORT_COUNT,
        "support_k1_physical_root_sha256": support1_root,
        "support_k5_count": K5_SUPPORT_COUNT,
        "support_k5_physical_root_sha256": support5_root,
        "outer_query_count": CLASS_COUNT * QUERY_PER_CLASS,
        "outer_query_physical_root_sha256": query_root,
        "k1_is_k5_prefix": True,
    }
    for field, expected in expected_fold.items():
        if field in fold and fold[field] != expected:
            raise NextR3MatrixError("full LOCO plan/fold physical binding drift")
    seal_payload = {
        "schema": "cvs.phase1.next_r3.fold_asset_seal.v1",
        "held_receiver": row_k1.held_receiver,
        "held_class": row_k1.held_class,
        "phase1_fit_count": PHASE1_FIT_COUNT,
        "phase1_fit_physical_root_sha256": phase1_root,
        "support_k1_physical_root_sha256": support1_root,
        "support_k5_physical_root_sha256": support5_root,
        "query_physical_root_sha256": query_root,
    }
    phase1_seal = canonical_sha256(seal_payload)
    payload: dict[str, Any] = {
        "schema": ROW_BINDING_SCHEMA,
        "held_receiver": row_k1.held_receiver,
        "held_class": row_k1.held_class,
        "k1_row_id": row_k1.row_id,
        "k5_row_id": row_k5.row_id,
        "registered_classes": list(classes),
        "all_registered_classes": list(classes),
        "evaluation_semantics": PROXY_SEMANTICS,
        "formal_new_registration_claim": False,
        "phase1_fit_count": PHASE1_FIT_COUNT,
        "phase1_fit_physical_root_sha256": phase1_root,
        "support_k1_physical_root_sha256": support1_root,
        "support_k5_physical_root_sha256": support5_root,
        "query_physical_root_sha256": query_root,
        "phase1_fold_seal_schema": seal_payload["schema"],
        "phase1_seal_sha256": phase1_seal,
        "k1_support_ids_by_class": {key: list(value) for key, value in support1.items()},
        "k5_support_ids_by_class": {key: list(value) for key, value in support5.items()},
        "query_ids_by_class": {key: list(value) for key, value in query.items()},
        "k1_is_exact_k5_prefix": True,
        "support_query_physical_ids_disjoint": True,
        "k1_support_count": K1_SUPPORT_COUNT,
        "k5_support_count": K5_SUPPORT_COUNT,
        "query_count": CLASS_COUNT * QUERY_PER_CLASS,
        "data_revalidated": False,
    }
    payload["binding_sha256"] = canonical_sha256(payload)
    return MappingProxyType(_json_ready(payload))


# D129-shaped aliases used by runners that only change the module import.
bind_joint6_physical_ids = bind_next_r3_physical_ids


def validate_next_r3_binding(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NextR3MatrixError("row binding must be a mapping")
    payload = dict(value)
    observed = payload.pop("binding_sha256", None)
    if payload.get("schema") != ROW_BINDING_SCHEMA or observed != canonical_sha256(payload):
        raise NextR3MatrixError("row binding SHA256 drift")
    if (
        payload.get("evaluation_semantics") != PROXY_SEMANTICS
        or payload.get("formal_new_registration_claim") is not False
        or payload.get("k1_is_exact_k5_prefix") is not True
        or payload.get("support_query_physical_ids_disjoint") is not True
    ):
        raise NextR3MatrixError("row binding legality drift")
    return MappingProxyType(_json_ready(value))


validate_joint6_binding = validate_next_r3_binding


def validate_next_r3_proxy24_plan(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NextR3MatrixError("NEXT-R3 plan must be a mapping")
    payload = dict(value)
    observed = payload.pop("matrix_sha256", None)
    if observed != canonical_sha256(payload):
        raise NextR3MatrixError("NEXT-R3 matrix SHA256 drift")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("candidate_id") != CANDIDATE_ID
        or payload.get("protocol_schema") != PROTOCOL_SCHEMA
        or payload.get("representation_rule") != REPRESENTATION_RULE
        or payload.get("proxy_artifact_state_prefix") != PROXY_SEMANTICS
        or payload.get("evaluation_semantics") != PROXY_SEMANTICS
        or payload.get("artifact_semantics") != PROXY_SEMANTICS
        or payload.get("formal_new_registration_claim") is not False
        or tuple(payload.get("held_receivers", ())) != HELD_RECEIVERS
        or tuple(payload.get("k_values", ())) != K_VALUES
        or tuple(payload.get("state_ids", ())) != STATE_IDS
        or tuple(payload.get("registration_ids", ())) != REGISTRATION_IDS
        or tuple(payload.get("arm_ids", ())) != ARM_IDS
        or payload.get("held_receiver_count") != SELECTED_RECEIVER_COUNT
        or payload.get("held_class_count") != CLASS_COUNT
        or payload.get("row_count") != ROW_COUNT
        or payload.get("outer_key_count") != OUTER_KEY_COUNT
        or payload.get("state_prediction_count") != STATE_PREDICTION_COUNT
        or payload.get("arm_prediction_count") != ARM_PREDICTION_COUNT
        or payload.get("single_candidate") is not True
        or payload.get("candidate_count") != 1
        or payload.get("truth_loaded") is not False
        or any(payload.get(field) != 0 for field in ("query_rows_used_for_fit", "query_state_updates", "query_selection_count"))
        or any(payload.get(field) is not False for field in ("query_truth_access", "query_role_access", "class_quota_access", "true_batch_class_count_access", "global_reassignment", "source_runtime_access", "clean_runtime_access", "output_overwrite_allowed"))
    ):
        raise NextR3MatrixError("NEXT-R3 plan frozen constants drift")
    classes = _registry(payload.get("held_classes", ()), expected=CLASS_COUNT, name="held_classes")
    rows = payload.get("rows")
    if not isinstance(rows, (tuple, list)) or len(rows) != ROW_COUNT:
        raise NextR3MatrixError("NEXT-R3 row coverage drift")
    parsed = tuple(outer_key_from_mapping(item) for item in rows)
    expected_keys = tuple(
        (receiver, held_class, active_k)
        for receiver in HELD_RECEIVERS
        for held_class in classes
        for active_k in K_VALUES
    )
    if tuple(row.row_key for row in parsed) != expected_keys:
        raise NextR3MatrixError("NEXT-R3 row order/coverage drift")
    if any(row.all_registered_classes != classes for row in parsed):
        raise NextR3MatrixError("NEXT-R3 row class registry drift")
    rebuilt = build_next_r3_proxy24_plan(classes, held_receivers=HELD_RECEIVERS)
    if _json_ready(rebuilt) != _json_ready(value):
        raise NextR3MatrixError("NEXT-R3 plan does not match deterministic rebuild")
    return MappingProxyType(_json_ready(value))


validate_plan = validate_next_r3_proxy24_plan


__all__ = [
    "ADAPTED_ARM_IDS",
    "ARM_IDS",
    "ARM_PREDICTION_COUNT",
    "CANDIDATE_ID",
    "CLASS_COUNT",
    "COMMON_ARM_IDS",
    "DA1_STATES",
    "HELD_RECEIVERS",
    "K1_SUPPORT_COUNT",
    "K5_SUPPORT_COUNT",
    "K_VALUES",
    "NextR3LocoRow",
    "NextR3MatrixError",
    "NextR3OuterKey",
    "NextR3ProxyRow",
    "OUTER_KEY_COUNT",
    "PHASE1_FIT_COUNT",
    "PROTOCOL_SCHEMA",
    "PROXY_SEMANTICS",
    "QUERY_PER_CLASS",
    "REG0_STATES",
    "REG1_STATES",
    "REGISTRATION_IDS",
    "REPRESENTATION_RULE",
    "ROW_BINDING_SCHEMA",
    "ROW_COUNT",
    "SCHEMA",
    "STATE_COUNT",
    "STATE_IDS",
    "STATE_PREDICTION_COUNT",
    "bind_joint6_physical_ids",
    "bind_next_r3_physical_ids",
    "build_next_r3_proxy24_plan",
    "canonical_bytes",
    "canonical_sha256",
    "outer_key_from_mapping",
    "query_count_for_state",
    "registered_classes_for_state",
    "registration_for_state",
    "row_from_mapping",
    "validate_joint6_binding",
    "validate_next_r3_binding",
    "validate_next_r3_proxy24_plan",
    "validate_plan",
]
