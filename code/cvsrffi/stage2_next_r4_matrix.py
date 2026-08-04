"""Frozen NEXT-R4 FA-RDCE3 x CER-PLR160 proxy matrix.

This module is deliberately mechanical.  It creates the 24 logical
receiver/class/K rows and validates the physical-ID/receipt bindings used by
the runtime.  It does not implement FA, CER, prediction, scoring, or any
performance decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence


SCHEMA = "cvs.stage2.next_r4.fa_rdce3_cer_plr160.proxy24.plan.v1"
ROW_BINDING_SCHEMA = "cvs.stage2.next_r4.fa_rdce3_cer_plr160.row_binding.v2"
QUERY_PAIR_ORDER_POLICY = "opaque_physical_id_lexicographic_v1"
PROTOCOL_SCHEMA = "p2_min_v1"
CANDIDATE_ID = "NEXT-R4-FA-RDCE3-CER-PLR160"
REPRESENTATION_RULE = "d106_canonical_normalized_relu_zid160"
PROXY_SEMANTICS = "SOURCE_HELD_PROXY"

HELD_RECEIVERS = ("1-1", "18-2")
STATE_IDS = ("DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1")
QUERY_VIEW_IDS = ("K1", "K5", *STATE_IDS)
STATE_NAMES_ZH = {
    "DA0_REG0": "域适应前/新类注册前",
    "DA1_REG0": "域适应后/新类注册前",
    "DA0_REG1": "域适应前/新类注册后",
    "DA1_REG1": "域适应后/新类注册后",
}
STATE_NAME_ZH = STATE_NAMES_ZH
REGISTRATION_IDS = ("REG0", "REG1")
REG0_STATES = frozenset(("DA0_REG0", "DA1_REG0"))
REG1_STATES = frozenset(("DA0_REG1", "DA1_REG1"))
DA0_STATES = frozenset(("DA0_REG0", "DA0_REG1"))
DA1_STATES = frozenset(("DA1_REG0", "DA1_REG1"))
ARM_IDS = ("Q", "H")
K1_UNIQUE_ARM_IDS = ("Q",)
K5_UNIQUE_ARM_IDS = ("Q", "H")
K_VALUES = (1, 5)
CLASS_COUNT = 6
SELECTED_RECEIVER_COUNT = 2
ROW_COUNT = SELECTED_RECEIVER_COUNT * CLASS_COUNT * len(K_VALUES)
K1_ROW_COUNT = SELECTED_RECEIVER_COUNT * CLASS_COUNT
K5_ROW_COUNT = K1_ROW_COUNT
STATE_COUNT = len(STATE_IDS)
K1_UNIQUE_PREDICTION_COUNT = K1_ROW_COUNT * STATE_COUNT
K1_ARTIFACT_COUNT = K1_ROW_COUNT * STATE_COUNT * len(ARM_IDS)
K5_UNIQUE_PREDICTION_COUNT = K5_ROW_COUNT * STATE_COUNT * len(K5_UNIQUE_ARM_IDS)
K5_ARTIFACT_COUNT = K5_ROW_COUNT * STATE_COUNT * len(ARM_IDS)
UNIQUE_PREDICTION_COUNT = K1_UNIQUE_PREDICTION_COUNT + K5_UNIQUE_PREDICTION_COUNT
ARTIFACT_ARM_COUNT = K1_ARTIFACT_COUNT + K5_ARTIFACT_COUNT

# Only support K values are frozen by NEXT-R4. Query and Phase1 asset
# cardinalities are runtime facts and must not inherit an R3-specific gate.
MAX_SUPPORT_K = 5
K1_SUPPORT_COUNT = CLASS_COUNT
K5_SUPPORT_COUNT = CLASS_COUNT * MAX_SUPPORT_K

PRIMARY_COMPARISONS = (
    "DA1_REG0-DA0_REG0",
    "DA1_REG1-DA0_REG1",
    "DA0_REG1-DA0_REG0",
    "DA1_REG1-DA1_REG0",
    "K5_H-K5_Q",
    "(DA1_H-DA1_Q)-(DA0_H-DA0_Q)",
)


class NextR4MatrixError(ValueError):
    """Raised when a frozen NEXT-R4 matrix or receipt drifts."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise NextR4MatrixError("matrix metadata contains an unsupported value")


def canonical_bytes(value: Any) -> bytes:
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
        raise NextR4MatrixError(f"{name} must be a lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise NextR4MatrixError(f"{name} must be a lowercase SHA256") from error
    return value


def _registry(values: Sequence[str], *, expected: int, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise NextR4MatrixError(f"{name} must be an ordered string registry")
    result = tuple(values)
    if (
        len(result) != expected
        or any(not isinstance(item, str) or not item for item in result)
        or len(set(result)) != expected
    ):
        raise NextR4MatrixError(
            f"{name} must contain exactly {expected} unique nonempty handles"
        )
    # Runtime registry order is part of the receipt.  Always canonicalize it
    # here rather than trusting caller order.
    return tuple(sorted(result))


def _exact_held_receivers(values: Sequence[str]) -> tuple[str, str]:
    result = tuple(values)
    if result != HELD_RECEIVERS:
        raise NextR4MatrixError(
            "held_receivers are frozen to the ordered pair ('1-1', '18-2')"
        )
    return result


def _registration_for_state(state_id: str) -> str:
    if state_id in REG0_STATES:
        return "REG0"
    if state_id in REG1_STATES:
        return "REG1"
    raise NextR4MatrixError(f"unknown NEXT-R4 four-state ID {state_id}")


@dataclass(frozen=True, slots=True)
class NextR4ProxyRow:
    """Identity of one receiver-held-class-K logical row."""

    row_id: str
    held_receiver: str
    held_class: str
    active_k: int
    retained_classes: tuple[str, ...]
    all_registered_classes: tuple[str, ...]

    def __post_init__(self) -> None:
        classes = _registry(
            self.all_registered_classes, expected=CLASS_COUNT, name="all_registered_classes"
        )
        retained = _registry(
            self.retained_classes, expected=CLASS_COUNT - 1, name="retained_classes"
        )
        if (
            not isinstance(self.row_id, str)
            or not self.row_id.startswith("r4-")
            or self.held_receiver not in HELD_RECEIVERS
            or self.held_class not in classes
            or self.active_k not in K_VALUES
            or retained != tuple(item for item in classes if item != self.held_class)
        ):
            raise NextR4MatrixError("NEXT-R4 row identity drift")
        object.__setattr__(self, "retained_classes", retained)
        object.__setattr__(self, "all_registered_classes", classes)

    @property
    def registered_classes(self) -> tuple[str, ...]:
        return self.all_registered_classes

    @property
    def row_key(self) -> tuple[str, str, int]:
        return self.held_receiver, self.held_class, self.active_k

    def registration_classes(self, registration_id: str) -> tuple[str, ...]:
        if registration_id == "REG0":
            return self.retained_classes
        if registration_id == "REG1":
            return self.all_registered_classes
        raise NextR4MatrixError(f"unknown registration ID {registration_id}")

    def as_dict(self) -> dict[str, Any]:
        state_specs = {
            state_id: {
                "state_id": state_id,
                "state_name_zh": STATE_NAMES_ZH[state_id],
                "registration_id": _registration_for_state(state_id),
                "adaptation_id": "DA1" if state_id in DA1_STATES else "DA0",
                "fa_state_source": "DA1_REG0" if state_id == "DA1_REG1" else None,
                "k1_unique_arm_ids": list(K1_UNIQUE_ARM_IDS),
                "k1_artifact_arm_ids": list(ARM_IDS),
                "k1_h_is_alias_receipt": True,
                "k5_unique_arm_ids": list(K5_UNIQUE_ARM_IDS),
                "k5_artifact_arm_ids": list(ARM_IDS),
                "k5_h_is_alias_receipt": False,
            }
            for state_id in STATE_IDS
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
            "state_names_zh": dict(STATE_NAMES_ZH),
            "state_specs": state_specs,
            "arm_ids": list(ARM_IDS),
            "evaluation_semantics": PROXY_SEMANTICS,
            "artifact_semantics": PROXY_SEMANTICS,
            "formal_new_registration_claim": False,
        }


def _row_id(receiver: str, held_class: str, active_k: int) -> str:
    digest = hashlib.sha256(
        f"{SCHEMA}|{receiver}|{held_class}|K{active_k}".encode("utf-8")
    ).hexdigest()
    return f"r4-{digest[:24]}"


def _resolve_build_inputs(
    class_registry: Sequence[str] | None,
    held_classes: Sequence[str] | None,
    held_receivers: Sequence[str],
) -> tuple[tuple[str, str], tuple[str, ...]]:
    receivers = _exact_held_receivers(held_receivers)
    if class_registry is None and held_classes is None:
        raise NextR4MatrixError("six runtime classes must be supplied explicitly")
    if held_classes is None:
        classes_input = class_registry
    elif class_registry is None:
        classes_input = held_classes
    else:
        if tuple(class_registry) != receivers:
            raise NextR4MatrixError(
                "two positional registries must be (held_receivers, held_classes)"
            )
        classes_input = held_classes
    return receivers, _registry(classes_input, expected=CLASS_COUNT, name="class_registry")


def build_next_r4_proxy24_plan(
    class_registry: Sequence[str] | None = None,
    held_classes: Sequence[str] | None = None,
    *,
    held_receivers: Sequence[str] = HELD_RECEIVERS,
) -> Mapping[str, Any]:
    """Build the immutable 24-row plan without target data or truth."""

    receivers, classes = _resolve_build_inputs(
        class_registry, held_classes, held_receivers
    )
    rows = tuple(
        NextR4ProxyRow(
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
        "class_registry_policy": "runtime_supplied_unique_sorted",
        "held_classes": list(classes),
        "held_class_count": CLASS_COUNT,
        "k_values": list(K_VALUES),
        "state_ids": list(STATE_IDS),
        "state_names_zh": dict(STATE_NAMES_ZH),
        "registration_ids": list(REGISTRATION_IDS),
        "arm_ids": list(ARM_IDS),
        "k1_unique_arm_ids": list(K1_UNIQUE_ARM_IDS),
        "k5_unique_arm_ids": list(K5_UNIQUE_ARM_IDS),
        "k1_h_semantics": "per_logit_alias_receipt",
        "k5_h_semantics": "unique_prediction",
        "row_count": ROW_COUNT,
        "logical_row_count": ROW_COUNT,
        "outer_key_count": ROW_COUNT,
        "state_count": STATE_COUNT,
        "k1_row_count": K1_ROW_COUNT,
        "k5_row_count": K5_ROW_COUNT,
        "k1_unique_prediction_count": K1_UNIQUE_PREDICTION_COUNT,
        "k1_artifact_count": K1_ARTIFACT_COUNT,
        "k5_unique_prediction_count": K5_UNIQUE_PREDICTION_COUNT,
        "k5_artifact_count": K5_ARTIFACT_COUNT,
        "unique_prediction_count": UNIQUE_PREDICTION_COUNT,
        "artifact_arm_count": ARTIFACT_ARM_COUNT,
        "prediction_count": UNIQUE_PREDICTION_COUNT,
        "primary_comparisons": list(PRIMARY_COMPARISONS),
        "common_query_binding": {
            "physical_ids": True,
            "observation_ids": True,
            "byte_exact": True,
            "order_exact": True,
            "reuse_across_k": True,
            "reuse_across_states": True,
            "required_view_ids": list(QUERY_VIEW_IDS),
        },
        "da1_reg1_fa_state_reuse": {
            "required": True,
            "source_state": "DA1_REG0",
            "target_state": "DA1_REG1",
            "same_state_sha256": True,
        },
        "rows": [row.as_dict() for row in rows],
        "single_candidate": True,
        "candidate_count": 1,
        "parameter_search_allowed": False,
        "seed_search_allowed": False,
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


def outer_key_from_mapping(value: Mapping[str, Any]) -> NextR4ProxyRow:
    if not isinstance(value, Mapping):
        raise NextR4MatrixError("row mapping must be an object")
    try:
        raw_classes = value.get("all_registered_classes")
        if raw_classes is None:
            raw_classes = value["registered_classes"]
        classes = tuple(raw_classes)
        return NextR4ProxyRow(
            row_id=str(value["row_id"]),
            held_receiver=str(value["held_receiver"]),
            held_class=str(value["held_class"]),
            active_k=int(value["active_k"]),
            retained_classes=tuple(value["retained_classes"]),
            all_registered_classes=classes,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise NextR4MatrixError("row mapping is incomplete") from error


def registered_classes_for_state(row: NextR4ProxyRow, state_id: str) -> tuple[str, ...]:
    return row.registration_classes(_registration_for_state(state_id))


def registration_for_state(state_id: str) -> str:
    return _registration_for_state(state_id)


def _physical_map(
    value: Mapping[str, Sequence[str]],
    *,
    classes: tuple[str, ...],
    expected_per_class: int,
    name: str,
) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or tuple(sorted(value)) != classes:
        raise NextR4MatrixError(f"{name} class registry drift")
    result: dict[str, tuple[str, ...]] = {}
    for class_id in classes:
        ids = tuple(value[class_id])
        if (
            len(ids) != expected_per_class
            or len(set(ids)) != expected_per_class
            or any(not isinstance(item, str) or not item for item in ids)
        ):
            raise NextR4MatrixError(
                f"{name}[{class_id}] must contain {expected_per_class} unique IDs"
            )
        result[class_id] = ids
    return result


def _query_map(
    value: Mapping[str, Sequence[str]],
    *,
    classes: tuple[str, ...],
    name: str,
) -> dict[str, tuple[str, ...]]:
    """Normalize runtime query IDs without imposing a frozen class count."""

    if not isinstance(value, Mapping) or tuple(sorted(value)) != classes:
        raise NextR4MatrixError(f"{name} class registry drift")
    result: dict[str, tuple[str, ...]] = {}
    for class_id in classes:
        ids = tuple(value[class_id])
        if (
            not ids
            or len(ids) != len(set(ids))
            or any(not isinstance(item, str) or not item for item in ids)
        ):
            raise NextR4MatrixError(
                f"{name}[{class_id}] must contain unique nonempty runtime IDs"
            )
        result[class_id] = ids
    return result


def _root(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _check_common_view_maps(
    value: Mapping[str, Mapping[str, Sequence[str]]] | None,
    *,
    base: Mapping[str, Sequence[str]],
    classes: tuple[str, ...],
    name: str,
) -> tuple[str, ...]:
    if value is None:
        raise NextR4MatrixError(f"{name} is required for common-query receipt")
    views = tuple(value)
    if tuple(sorted(views)) != tuple(sorted(QUERY_VIEW_IDS)):
        raise NextR4MatrixError(
            f"{name} must contain exactly K1/K5 and all four state views"
        )
    base_norm = _query_map(base, classes=classes, name=f"{name}.base")
    for view_id in views:
        current = _query_map(value[view_id], classes=classes, name=f"{name}[{view_id}]")
        if current != base_norm:
            raise NextR4MatrixError(
                f"common query {name} drift across K/state views"
            )
    return views


def validate_fa_state_reuse(
    fa_state_sha256_by_state: Mapping[str, str],
) -> Mapping[str, Any]:
    """Validate the required DA1_REG0 -> DA1_REG1 FA-state byte reuse."""

    if not isinstance(fa_state_sha256_by_state, Mapping):
        raise NextR4MatrixError("FA state SHA receipt must be a mapping")
    missing = {"DA1_REG0", "DA1_REG1"} - set(fa_state_sha256_by_state)
    if missing:
        raise NextR4MatrixError("FA state SHA receipt is missing DA1_REG0/DA1_REG1")
    source = _require_sha256(
        fa_state_sha256_by_state["DA1_REG0"], name="DA1_REG0 state SHA256"
    )
    target = _require_sha256(
        fa_state_sha256_by_state["DA1_REG1"], name="DA1_REG1 state SHA256"
    )
    if source != target:
        raise NextR4MatrixError("DA1_REG1 must reuse the DA1_REG0 FA state SHA256")
    payload = {
        "schema": "cvs.stage2.next_r4.fa_state_reuse_receipt.v1",
        "source_state": "DA1_REG0",
        "target_state": "DA1_REG1",
        "source_sha256": source,
        "target_sha256": target,
        "same_state_sha256": True,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return MappingProxyType(_json_ready(payload))


def bind_next_r4_physical_ids(
    *,
    row_k1: NextR4ProxyRow,
    row_k5: NextR4ProxyRow,
    phase1_fit_ids: Sequence[str],
    k1_support_ids_by_class: Mapping[str, Sequence[str]],
    k5_support_ids_by_class: Mapping[str, Sequence[str]],
    query_ids_by_class: Mapping[str, Sequence[str]],
    query_observation_ids_by_class: Mapping[str, Sequence[str]],
    query_ids_by_view: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    query_observation_ids_by_view: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> Mapping[str, Any]:
    """Bind support/query physical IDs shared by the paired K1/K5 rows.

    FA state reuse is intentionally row-local because K1 and K5 consume
    different support sets and therefore need not produce the same FA state.
    Each runtime row carries its own canonical ``fa_state_reuse_receipt``.
    """

    if not isinstance(row_k1, NextR4ProxyRow) or not isinstance(row_k5, NextR4ProxyRow):
        raise NextR4MatrixError("row_k1/row_k5 must be NEXT-R4 rows")
    if (
        row_k1.active_k != 1
        or row_k5.active_k != 5
        or row_k1.held_receiver != row_k5.held_receiver
        or row_k1.held_class != row_k5.held_class
        or row_k1.all_registered_classes != row_k5.all_registered_classes
    ):
        raise NextR4MatrixError("K1/K5 row pairing drift")
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
    query = _query_map(query_ids_by_class, classes=classes, name="query_ids_by_class")
    observations = _query_map(
        query_observation_ids_by_class,
        classes=classes,
        name="query_observation_ids_by_class",
    )
    if any(len(query[c]) != len(observations[c]) for c in classes):
        raise NextR4MatrixError(
            "query physical/observation IDs must have the same per-class length"
        )
    if any(support1[c] != support5[c][:1] for c in classes):
        raise NextR4MatrixError("K1 support must be the exact K5 prefix")
    support_union = {item for values in support5.values() for item in values}
    query_union = {item for values in query.values() for item in values}
    observation_union = {item for values in observations.values() for item in values}
    phase1 = tuple(phase1_fit_ids)
    if (
        not phase1
        or len(set(phase1)) != len(phase1)
        or any(not isinstance(item, str) or not item for item in phase1)
    ):
        raise NextR4MatrixError("phase1_fit_ids must contain unique nonempty IDs")
    if len(support_union) != K5_SUPPORT_COUNT:
        raise NextR4MatrixError("K5 support physical IDs overlap across classes")
    if len(query_union) != sum(len(query[c]) for c in classes):
        raise NextR4MatrixError("query physical IDs overlap across classes")
    if len(observation_union) != sum(len(observations[c]) for c in classes):
        raise NextR4MatrixError("query observation IDs overlap across classes")
    if support_union & query_union:
        raise NextR4MatrixError("support/query physical IDs overlap")
    if support_union & observation_union:
        raise NextR4MatrixError("support/query observation IDs overlap")
    if set(phase1) & (support_union | query_union | observation_union):
        raise NextR4MatrixError("Phase1-fit/support/query IDs overlap")
    id_views = _check_common_view_maps(
        query_ids_by_view,
        base=query,
        classes=classes,
        name="query_ids_by_view",
    )
    observation_views = _check_common_view_maps(
        query_observation_ids_by_view,
        base=observations,
        classes=classes,
        name="query_observation_ids_by_view",
    )
    support1_ordered = tuple(item for c in classes for item in support1[c])
    support5_ordered = tuple(item for c in classes for item in support5[c])
    # The builder needs class-grouped inputs for the one-time legality checks,
    # but the predictor-visible receipt must not retain their truth grouping.
    # Sort opaque physical IDs globally and carry the paired observation ID
    # with each item.  Physical IDs are unique, so this order is deterministic
    # without a class-dependent tie-breaker.
    query_pairs = sorted(
        (
            (physical_id, observations[class_id][index])
            for class_id in classes
            for index, physical_id in enumerate(query[class_id])
        ),
        key=lambda pair: pair[0],
    )
    query_ordered = tuple(pair[0] for pair in query_pairs)
    observation_ordered = tuple(pair[1] for pair in query_pairs)
    payload: dict[str, Any] = {
        "schema": ROW_BINDING_SCHEMA,
        "held_receiver": row_k1.held_receiver,
        "held_class": row_k1.held_class,
        "k1_row_id": row_k1.row_id,
        "k5_row_id": row_k5.row_id,
        "registered_classes": list(classes),
        "phase1_fit_count": len(phase1),
        "phase1_fit_physical_root_sha256": _root(phase1),
        "support_k1_physical_root_sha256": _root(support1_ordered),
        "support_k5_physical_root_sha256": _root(support5_ordered),
        "query_physical_root_sha256": _root(query_ordered),
        "query_observation_root_sha256": _root(observation_ordered),
        "k1_support_ids_by_class": {key: list(value) for key, value in support1.items()},
        "k5_support_ids_by_class": {key: list(value) for key, value in support5.items()},
        "query_physical_ids": list(query_ordered),
        "query_observation_ids": list(observation_ordered),
        "query_pair_order_policy": QUERY_PAIR_ORDER_POLICY,
        "k1_is_exact_k5_prefix": True,
        "support_query_physical_ids_disjoint": True,
        "support_query_observation_ids_disjoint": True,
        "common_query_physical_ids_across_k_states": True,
        "common_query_observation_ids_across_k_states": True,
        "query_view_ids_checked": list(id_views),
        "query_observation_view_ids_checked": list(observation_views),
        "k1_support_count": K1_SUPPORT_COUNT,
        "k5_support_count": K5_SUPPORT_COUNT,
        "query_count": len(query_ordered),
    }
    payload["binding_sha256"] = canonical_sha256(payload)
    return MappingProxyType(_json_ready(payload))


def validate_next_r4_binding(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NextR4MatrixError("row binding must be a mapping")
    payload = dict(value)
    observed = payload.pop("binding_sha256", None)
    forbidden = {
        "query_ids_by_class",
        "query_observation_ids_by_class",
        "query_count_by_class",
    }

    def exposes_grouped_query(item: Any) -> bool:
        if isinstance(item, Mapping):
            return any(
                str(key).strip().lower().replace("-", "_") in forbidden
                or exposes_grouped_query(child)
                for key, child in item.items()
            )
        if isinstance(item, (tuple, list)):
            return any(exposes_grouped_query(child) for child in item)
        return False

    if exposes_grouped_query(payload):
        raise NextR4MatrixError("row binding exposes class-grouped query metadata")
    if payload.get("schema") != ROW_BINDING_SCHEMA or observed != canonical_sha256(payload):
        raise NextR4MatrixError("row binding SHA256 drift")
    query_ids = payload.get("query_physical_ids")
    observation_ids = payload.get("query_observation_ids")
    if not isinstance(query_ids, (tuple, list)) or not isinstance(
        observation_ids, (tuple, list)
    ):
        raise NextR4MatrixError("row binding requires flattened query ID pairs")
    query_ids = tuple(query_ids)
    observation_ids = tuple(observation_ids)
    if (
        not query_ids
        or len(query_ids) != len(observation_ids)
        or len(set(query_ids)) != len(query_ids)
        or len(set(observation_ids)) != len(observation_ids)
        or any(not isinstance(item, str) or not item for item in query_ids + observation_ids)
    ):
        raise NextR4MatrixError("row binding flattened query ID pairs drift")
    if (
        query_ids != tuple(sorted(query_ids))
        or payload.get("query_pair_order_policy") != QUERY_PAIR_ORDER_POLICY
        or payload.get("query_count") != len(query_ids)
        or payload.get("query_physical_root_sha256") != _root(query_ids)
        or payload.get("query_observation_root_sha256") != _root(observation_ids)
    ):
        raise NextR4MatrixError("row binding flattened query order/root drift")
    if (
        payload.get("k1_is_exact_k5_prefix") is not True
        or payload.get("support_query_physical_ids_disjoint") is not True
        or payload.get("support_query_observation_ids_disjoint") is not True
        or payload.get("common_query_physical_ids_across_k_states") is not True
        or payload.get("common_query_observation_ids_across_k_states") is not True
    ):
        raise NextR4MatrixError("NEXT-R4 row binding legality drift")
    return MappingProxyType(_json_ready(value))


def validate_next_r4_proxy24_plan(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NextR4MatrixError("NEXT-R4 plan must be a mapping")
    payload = dict(value)
    observed = payload.pop("matrix_sha256", None)
    if observed != canonical_sha256(payload):
        raise NextR4MatrixError("NEXT-R4 matrix SHA256 drift")
    fixed = (
        payload.get("schema") == SCHEMA
        and payload.get("candidate_id") == CANDIDATE_ID
        and payload.get("protocol_schema") == PROTOCOL_SCHEMA
        and payload.get("representation_rule") == REPRESENTATION_RULE
        and payload.get("proxy_artifact_state_prefix") == PROXY_SEMANTICS
        and payload.get("evaluation_semantics") == PROXY_SEMANTICS
        and payload.get("artifact_semantics") == PROXY_SEMANTICS
        and payload.get("formal_new_registration_claim") is False
        and tuple(payload.get("held_receivers", ())) == HELD_RECEIVERS
        and tuple(payload.get("k_values", ())) == K_VALUES
        and tuple(payload.get("state_ids", ())) == STATE_IDS
        and payload.get("state_names_zh") == STATE_NAMES_ZH
        and tuple(payload.get("registration_ids", ())) == REGISTRATION_IDS
        and tuple(payload.get("arm_ids", ())) == ARM_IDS
        and tuple(payload.get("k1_unique_arm_ids", ())) == K1_UNIQUE_ARM_IDS
        and tuple(payload.get("k5_unique_arm_ids", ())) == K5_UNIQUE_ARM_IDS
        and payload.get("k1_h_semantics") == "per_logit_alias_receipt"
        and payload.get("k5_h_semantics") == "unique_prediction"
        and payload.get("held_receiver_count") == SELECTED_RECEIVER_COUNT
        and payload.get("held_class_count") == CLASS_COUNT
        and payload.get("row_count") == ROW_COUNT
        and payload.get("logical_row_count") == ROW_COUNT
        and payload.get("outer_key_count") == ROW_COUNT
        and payload.get("state_count") == STATE_COUNT
        and payload.get("k1_row_count") == K1_ROW_COUNT
        and payload.get("k5_row_count") == K5_ROW_COUNT
        and payload.get("k1_unique_prediction_count") == K1_UNIQUE_PREDICTION_COUNT
        and payload.get("k1_artifact_count") == K1_ARTIFACT_COUNT
        and payload.get("k5_unique_prediction_count") == K5_UNIQUE_PREDICTION_COUNT
        and payload.get("k5_artifact_count") == K5_ARTIFACT_COUNT
        and payload.get("unique_prediction_count") == UNIQUE_PREDICTION_COUNT
        and payload.get("prediction_count") == UNIQUE_PREDICTION_COUNT
        and payload.get("artifact_arm_count") == ARTIFACT_ARM_COUNT
        and tuple(payload.get("primary_comparisons", ())) == PRIMARY_COMPARISONS
        and payload.get("common_query_binding") == {
            "physical_ids": True,
            "observation_ids": True,
            "byte_exact": True,
            "order_exact": True,
            "reuse_across_k": True,
            "reuse_across_states": True,
            "required_view_ids": list(QUERY_VIEW_IDS),
        }
        and payload.get("da1_reg1_fa_state_reuse") == {
            "required": True,
            "source_state": "DA1_REG0",
            "target_state": "DA1_REG1",
            "same_state_sha256": True,
        }
        and payload.get("single_candidate") is True
        and payload.get("candidate_count") == 1
        and payload.get("parameter_search_allowed") is False
        and payload.get("seed_search_allowed") is False
        and payload.get("truth_loaded") is False
        and all(payload.get(field) == 0 for field in ("query_rows_used_for_fit", "query_state_updates", "query_selection_count"))
        and all(payload.get(field) is False for field in ("query_truth_access", "query_role_access", "class_quota_access", "true_batch_class_count_access", "global_reassignment", "source_runtime_access", "clean_runtime_access", "output_overwrite_allowed"))
    )
    if not fixed:
        raise NextR4MatrixError("NEXT-R4 plan frozen constants drift")
    classes = _registry(payload.get("held_classes", ()), expected=CLASS_COUNT, name="class_registry")
    rows = payload.get("rows")
    if not isinstance(rows, (tuple, list)) or len(rows) != ROW_COUNT:
        raise NextR4MatrixError("NEXT-R4 row coverage drift")
    parsed = tuple(outer_key_from_mapping(item) for item in rows)
    expected_keys = tuple(
        (receiver, held_class, active_k)
        for receiver in HELD_RECEIVERS
        for held_class in classes
        for active_k in K_VALUES
    )
    if tuple(row.row_key for row in parsed) != expected_keys:
        raise NextR4MatrixError("NEXT-R4 row order/coverage drift")
    if any(row.all_registered_classes != classes for row in parsed):
        raise NextR4MatrixError("NEXT-R4 row class registry drift")
    rebuilt = build_next_r4_proxy24_plan(classes)
    if _json_ready(rebuilt) != _json_ready(value):
        raise NextR4MatrixError("NEXT-R4 plan does not match deterministic rebuild")
    return MappingProxyType(_json_ready(value))


__all__ = [
    "ARM_IDS",
    "ARTIFACT_ARM_COUNT",
    "CANDIDATE_ID",
    "CLASS_COUNT",
    "DA0_STATES",
    "DA1_STATES",
    "HELD_RECEIVERS",
    "K1_ARTIFACT_COUNT",
    "K1_ROW_COUNT",
    "K1_SUPPORT_COUNT",
    "K1_UNIQUE_ARM_IDS",
    "K1_UNIQUE_PREDICTION_COUNT",
    "K5_ARTIFACT_COUNT",
    "K5_ROW_COUNT",
    "K5_SUPPORT_COUNT",
    "K5_UNIQUE_ARM_IDS",
    "K5_UNIQUE_PREDICTION_COUNT",
    "K_VALUES",
    "NextR4MatrixError",
    "NextR4ProxyRow",
    "PRIMARY_COMPARISONS",
    "PROTOCOL_SCHEMA",
    "PROXY_SEMANTICS",
    "QUERY_VIEW_IDS",
    "REG0_STATES",
    "REG1_STATES",
    "REGISTRATION_IDS",
    "REPRESENTATION_RULE",
    "ROW_BINDING_SCHEMA",
    "QUERY_PAIR_ORDER_POLICY",
    "ROW_COUNT",
    "SCHEMA",
    "STATE_COUNT",
    "STATE_IDS",
    "STATE_NAME_ZH",
    "STATE_NAMES_ZH",
    "UNIQUE_PREDICTION_COUNT",
    "bind_next_r4_physical_ids",
    "build_next_r4_proxy24_plan",
    "canonical_bytes",
    "canonical_sha256",
    "outer_key_from_mapping",
    "registered_classes_for_state",
    "registration_for_state",
    "validate_fa_state_reuse",
    "validate_next_r4_binding",
    "validate_next_r4_proxy24_plan",
]
