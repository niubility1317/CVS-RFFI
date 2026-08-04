"""Frozen 24-key/96-state source-held plan for NEXT-R2 CVFR-BSSDG."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence


SCHEMA = "cvs.stage2.next_r2.proxy24.plan.v1"
CANDIDATE_ID = "CVFR-BSSDG/r1"
PROTOCOL_SCHEMA = "p2_min_v1"

STATE_IDS = ("DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1")
DA1_STATES = frozenset(("DA1_REG0", "DA1_REG1"))
REG1_STATES = frozenset(("DA0_REG1", "DA1_REG1"))
K_VALUES = (1, 5)

SOURCE_RECEIVER_COUNT = 7
SELECTED_RECEIVER_COUNT = 2
CLASS_COUNT = 6
PHYSICAL_PER_CELL = 14
MAX_SUPPORT_K = 5
QUERY_PER_CLASS = PHYSICAL_PER_CELL - MAX_SUPPORT_K
OUTER_KEY_COUNT = SELECTED_RECEIVER_COUNT * CLASS_COUNT * len(K_VALUES)
STATE_PREDICTION_COUNT = OUTER_KEY_COUNT * len(STATE_IDS)

RECEIVER_SELECTION_SALT = "NEXT-R2-CVFR-BSSDG-SOURCE-RECEIVER-HASH-v1"
CELL_ORDER_SALT = "NEXT-R2-CVFR-BSSDG-PHYSICAL-CELL-HASH-v1"


class NextR2MatrixError(ValueError):
    """The frozen source-held plan or state identity drifted."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise NextR2MatrixError("matrix metadata contains an unsupported value")


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise NextR2MatrixError(f"{name} must be a lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise NextR2MatrixError(f"{name} must be a lowercase SHA256") from error
    return value


def _registry(values: Sequence[str], *, expected: int, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise NextR2MatrixError(f"{name} must be an ordered string registry")
    registry = tuple(values)
    if (
        len(registry) != expected
        or any(not isinstance(item, str) or not item for item in registry)
        or len(set(registry)) != expected
    ):
        raise NextR2MatrixError(
            f"{name} must contain exactly {expected} unique nonempty handles"
        )
    return registry


def select_source_receivers(
    receiver_registry: Sequence[str], *, source_identity_sha256: str
) -> tuple[str, str]:
    """Select two receivers using only a preregistered source-identity hash."""

    receivers = _registry(
        receiver_registry, expected=SOURCE_RECEIVER_COUNT, name="receiver_registry"
    )
    source_root = _require_sha256(source_identity_sha256, name="source_identity_sha256")
    ranked = sorted(
        receivers,
        key=lambda receiver: (
            hashlib.sha256(
                f"{RECEIVER_SELECTION_SALT}|{source_root}|{receiver}".encode("utf-8")
            ).hexdigest(),
            receiver,
        ),
    )
    return ranked[0], ranked[1]


@dataclass(frozen=True, slots=True)
class NextR2OuterKey:
    outer_key_id: str
    held_receiver: str
    held_class: str
    active_k: int
    retained_classes: tuple[str, ...]
    all_registered_classes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.outer_key_id, str) or not self.outer_key_id.startswith(
            "n2-"
        ):
            raise NextR2MatrixError("outer_key_id must use the frozen n2 hash form")
        if not isinstance(self.held_receiver, str) or not self.held_receiver:
            raise NextR2MatrixError("held_receiver must be nonempty")
        if self.active_k not in K_VALUES:
            raise NextR2MatrixError("active_k must be exactly 1 or 5")
        all_classes = _registry(
            self.all_registered_classes, expected=CLASS_COUNT, name="all_registered_classes"
        )
        retained = _registry(
            self.retained_classes, expected=CLASS_COUNT - 1, name="retained_classes"
        )
        if self.held_class not in all_classes:
            raise NextR2MatrixError("held_class is outside the all-class registry")
        if retained != tuple(item for item in all_classes if item != self.held_class):
            raise NextR2MatrixError("retained_classes are not all classes except held_class")
        object.__setattr__(self, "retained_classes", retained)
        object.__setattr__(self, "all_registered_classes", all_classes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "outer_key_id": self.outer_key_id,
            "held_receiver": self.held_receiver,
            "held_class": self.held_class,
            "active_k": self.active_k,
            "retained_classes": self.retained_classes,
            "all_registered_classes": self.all_registered_classes,
            "state_ids": STATE_IDS,
        }


def registered_classes_for_state(
    outer_key: NextR2OuterKey, state_id: str
) -> tuple[str, ...]:
    if state_id not in STATE_IDS:
        raise NextR2MatrixError("unknown NEXT-R2 four-state ID")
    return (
        outer_key.all_registered_classes
        if state_id in REG1_STATES
        else outer_key.retained_classes
    )


def query_count_for_state(outer_key: NextR2OuterKey, state_id: str) -> int:
    return len(registered_classes_for_state(outer_key, state_id)) * QUERY_PER_CLASS


def _key_id(receiver: str, held_class: str, active_k: int) -> str:
    digest = hashlib.sha256(
        f"{SCHEMA}|{receiver}|{held_class}|K{active_k}".encode("utf-8")
    ).hexdigest()
    return f"n2-{digest[:24]}"


def build_next_r2_proxy24_plan(
    receiver_registry: Sequence[str],
    class_registry: Sequence[str],
    *,
    source_identity_sha256: str,
) -> Mapping[str, Any]:
    receivers = _registry(
        receiver_registry, expected=SOURCE_RECEIVER_COUNT, name="receiver_registry"
    )
    classes = _registry(class_registry, expected=CLASS_COUNT, name="class_registry")
    source_root = _require_sha256(source_identity_sha256, name="source_identity_sha256")
    selected = select_source_receivers(
        receivers, source_identity_sha256=source_root
    )
    keys = tuple(
        NextR2OuterKey(
            outer_key_id=_key_id(receiver, held_class, active_k),
            held_receiver=receiver,
            held_class=held_class,
            active_k=active_k,
            retained_classes=tuple(item for item in classes if item != held_class),
            all_registered_classes=classes,
        )
        for receiver in selected
        for held_class in classes
        for active_k in K_VALUES
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "source_identity_sha256": source_root,
        "receiver_selection": {
            "rule": "ascending_sha256_of_salt_source_identity_receiver",
            "salt": RECEIVER_SELECTION_SALT,
            "performance_inputs": 0,
            "selected_receivers": selected,
        },
        "receiver_registry": receivers,
        "class_registry": classes,
        "k_values": K_VALUES,
        "state_ids": STATE_IDS,
        "outer_key_count": OUTER_KEY_COUNT,
        "state_prediction_count": STATE_PREDICTION_COUNT,
        "query_per_class": QUERY_PER_CLASS,
        "keys": tuple(item.as_dict() for item in keys),
        "target_receiver_count": 0,
        "target_cn20": False,
        "k10": False,
        "truth_or_accuracy_receiver_selection": False,
    }
    payload["matrix_sha256"] = canonical_sha256(payload)
    return MappingProxyType(_json_ready(payload))


def outer_key_from_mapping(value: Mapping[str, Any]) -> NextR2OuterKey:
    try:
        return NextR2OuterKey(
            outer_key_id=str(value["outer_key_id"]),
            held_receiver=str(value["held_receiver"]),
            held_class=str(value["held_class"]),
            active_k=int(value["active_k"]),
            retained_classes=tuple(value["retained_classes"]),
            all_registered_classes=tuple(value["all_registered_classes"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise NextR2MatrixError("outer-key mapping is incomplete") from error


def validate_next_r2_proxy24_plan(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NextR2MatrixError("NEXT-R2 plan must be a mapping")
    payload = dict(value)
    observed_sha = payload.pop("matrix_sha256", None)
    if observed_sha != canonical_sha256(payload):
        raise NextR2MatrixError("NEXT-R2 matrix SHA256 drift")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("candidate_id") != CANDIDATE_ID
        or payload.get("protocol_schema") != PROTOCOL_SCHEMA
        or payload.get("outer_key_count") != OUTER_KEY_COUNT
        or payload.get("state_prediction_count") != STATE_PREDICTION_COUNT
        or tuple(payload.get("state_ids", ())) != STATE_IDS
        or tuple(payload.get("k_values", ())) != K_VALUES
        or payload.get("target_receiver_count") != 0
        or payload.get("target_cn20") is not False
        or payload.get("k10") is not False
        or payload.get("truth_or_accuracy_receiver_selection") is not False
    ):
        raise NextR2MatrixError("NEXT-R2 plan frozen constants drift")
    rebuilt = build_next_r2_proxy24_plan(
        tuple(payload.get("receiver_registry", ())),
        tuple(payload.get("class_registry", ())),
        source_identity_sha256=str(payload.get("source_identity_sha256", "")),
    )
    if _json_ready(rebuilt) != _json_ready(value):
        raise NextR2MatrixError("NEXT-R2 plan does not match deterministic rebuild")
    keys = tuple(outer_key_from_mapping(item) for item in payload.get("keys", ()))
    if len(keys) != OUTER_KEY_COUNT or len({item.outer_key_id for item in keys}) != len(keys):
        raise NextR2MatrixError("NEXT-R2 outer-key coverage drift")
    return MappingProxyType(_json_ready(value))


__all__ = [
    "CANDIDATE_ID",
    "CELL_ORDER_SALT",
    "CLASS_COUNT",
    "DA1_STATES",
    "K_VALUES",
    "MAX_SUPPORT_K",
    "NextR2MatrixError",
    "NextR2OuterKey",
    "OUTER_KEY_COUNT",
    "PHYSICAL_PER_CELL",
    "PROTOCOL_SCHEMA",
    "QUERY_PER_CLASS",
    "RECEIVER_SELECTION_SALT",
    "REG1_STATES",
    "SCHEMA",
    "SELECTED_RECEIVER_COUNT",
    "SOURCE_RECEIVER_COUNT",
    "STATE_IDS",
    "STATE_PREDICTION_COUNT",
    "build_next_r2_proxy24_plan",
    "canonical_bytes",
    "canonical_sha256",
    "outer_key_from_mapping",
    "query_count_for_state",
    "registered_classes_for_state",
    "select_source_receivers",
    "validate_next_r2_proxy24_plan",
]
