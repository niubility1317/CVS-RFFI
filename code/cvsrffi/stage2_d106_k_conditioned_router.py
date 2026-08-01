"""Frozen K-only prediction router for D106-KCR/r1.

The router does not fit, score, inspect query contents, or choose from
receiver/scene/class/truth/metric information.  Its only routing decision is
the preregistered ``active_k`` lookup.  All four same-row arm predictions must
already exist before one arm can be selected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any


CANDIDATE_ID = "D106-KCR/r1"
SCHEMA = "cvs.phase2.d106.k_conditioned_prediction.v1"
G1_ROW_SCHEMA = "cvs.d106.g1.sourceheld.predictions.v1.row"
TARGET25_ROW_SCHEMA = "cvs.phase2.d106.target25.prediction_row.v1"
ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")
ROUTE_BY_K = MappingProxyType({1: "M_DA", 5: "M0", 10: "M_HEAD"})
G1_ROW_KEYS = frozenset(
    {
        "K",
        "arm_predictions",
        "formal_p2_authority",
        "held_class",
        "held_receiver",
        "package_id",
        "prediction_receipt_sha256",
        "query_physical_ids",
        "query_state_updates",
        "query_truth_access",
        "registered_classes",
        "schema",
        "shared_component_receipts",
        "target_access",
    }
)
TARGET25_ROW_KEYS = frozenset(
    {
        "K",
        "arm_predictions",
        "prediction_receipt_sha256",
        "query_physical_ids",
        "query_role_access",
        "query_selection",
        "query_state_updates",
        "query_truth_access",
        "receiver",
        "registered_classes",
        "row_id",
        "scene",
        "schema",
        "shared_component_receipts",
    }
)
ROW_KEYS_BY_SCHEMA = MappingProxyType(
    {
        G1_ROW_SCHEMA: G1_ROW_KEYS,
        TARGET25_ROW_SCHEMA: TARGET25_ROW_KEYS,
    }
)


class D106KConditionedRouterError(ValueError):
    """Raised when the frozen K route or same-row prediction closure drifts."""


def _require_text_sequence(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise D106KConditionedRouterError(f"{name} must be an ordered sequence")
    result = tuple(value)
    if not result or any(type(item) is not str or not item for item in result):
        raise D106KConditionedRouterError(
            f"{name} must contain non-empty builtin strings"
        )
    return result


def _require_sha256(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise D106KConditionedRouterError(f"{name} must be a lowercase SHA256")
    return value


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise D106KConditionedRouterError(
            "row_prediction canonical JSON payload is invalid"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _require_non_empty_text(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise D106KConditionedRouterError(f"{name} must be non-empty builtin text")
    return value


@dataclass(frozen=True, slots=True)
class D106KConditionedPrediction:
    """Immutable selected prediction plus its unchanged row bindings."""

    active_k: int
    selected_arm: str
    predictions: tuple[str, ...]
    query_order: tuple[str, ...]
    registered_classes: tuple[str, ...]
    source_prediction_receipt_sha256: str
    shared_component_receipts: tuple[tuple[str, str], ...]
    query_state_updates: int = 0
    candidate_id: str = CANDIDATE_ID
    schema: str = SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "active_k": self.active_k,
            "selected_arm": self.selected_arm,
            "predictions": list(self.predictions),
            "query_order": list(self.query_order),
            "registered_classes": list(self.registered_classes),
            "source_prediction_receipt_sha256": (
                self.source_prediction_receipt_sha256
            ),
            "shared_component_receipts": dict(self.shared_component_receipts),
            "query_state_updates": self.query_state_updates,
        }


def route_d106_k_conditioned_prediction(
    *,
    active_k: int,
    row_prediction: Mapping[str, Any],
) -> D106KConditionedPrediction:
    """Select the frozen arm using only ``active_k`` after row validation."""

    if type(active_k) is not int or active_k not in ROUTE_BY_K:
        raise D106KConditionedRouterError("active_k must be exactly 1, 5, or 10")
    if not isinstance(row_prediction, Mapping):
        raise D106KConditionedRouterError("row_prediction must be a mapping")
    schema = row_prediction.get("schema")
    if type(schema) is not str or schema not in ROW_KEYS_BY_SCHEMA:
        raise D106KConditionedRouterError("row_prediction schema is unsupported")
    if set(row_prediction) != ROW_KEYS_BY_SCHEMA[schema]:
        raise D106KConditionedRouterError(
            f"{schema} row_prediction field closure drift"
        )
    if row_prediction["query_truth_access"] is not False:
        raise D106KConditionedRouterError("query_truth_access must remain false")
    if type(row_prediction.get("K")) is not int or row_prediction["K"] != active_k:
        raise D106KConditionedRouterError("row_prediction K binding drift")

    arm_predictions = row_prediction.get("arm_predictions")
    if not isinstance(arm_predictions, Mapping) or set(arm_predictions) != set(ARMS):
        raise D106KConditionedRouterError("same-row predictions require exact four-arm closure")
    query_state_updates = row_prediction.get("query_state_updates")
    if type(query_state_updates) is not int or query_state_updates != 0:
        raise D106KConditionedRouterError("query_state_updates must remain exactly zero")
    if schema == G1_ROW_SCHEMA:
        if (
            row_prediction["formal_p2_authority"] is not False
            or row_prediction["target_access"] is not False
        ):
            raise D106KConditionedRouterError(
                "G1 row authority/access binding drift"
            )
        _require_non_empty_text(row_prediction["held_receiver"], "held_receiver")
        held_class = row_prediction["held_class"]
        if held_class is not None and (type(held_class) is not str or not held_class):
            raise D106KConditionedRouterError("held_class binding drift")
        _require_non_empty_text(row_prediction["package_id"], "package_id")
    else:
        if (
            row_prediction["query_role_access"] is not False
            or row_prediction["query_selection"] is not False
        ):
            raise D106KConditionedRouterError(
                "Target25 query role/selection binding drift"
            )
        for name in ("row_id", "receiver", "scene"):
            _require_non_empty_text(row_prediction[name], name)

    query_ids = _require_text_sequence(
        row_prediction.get("query_physical_ids"), "query_physical_ids"
    )
    if len(set(query_ids)) != len(query_ids):
        raise D106KConditionedRouterError("query_order must contain unique IDs")
    registry = _require_text_sequence(
        row_prediction.get("registered_classes"), "registered_classes"
    )
    if len(set(registry)) != len(registry):
        raise D106KConditionedRouterError("registered_classes must be unique")

    closed_predictions: dict[str, tuple[str, ...]] = {}
    for arm in ARMS:
        predictions = _require_text_sequence(arm_predictions[arm], f"{arm} predictions")
        if len(predictions) != len(query_ids):
            raise D106KConditionedRouterError(
                "all four arms must preserve the complete query order"
            )
        if any(prediction not in registry for prediction in predictions):
            raise D106KConditionedRouterError(
                f"{arm} prediction falls outside the ordered registry"
            )
        closed_predictions[arm] = predictions

    source_receipt = _require_sha256(
        row_prediction.get("prediction_receipt_sha256"),
        "prediction_receipt_sha256",
    )
    shared_component_receipts = row_prediction.get("shared_component_receipts")
    if not isinstance(shared_component_receipts, Mapping) or not shared_component_receipts:
        raise D106KConditionedRouterError(
            "shared_component_receipts must be a non-empty mapping"
        )
    receipts: list[tuple[str, str]] = []
    for name, value in shared_component_receipts.items():
        if type(name) is not str or not name:
            raise D106KConditionedRouterError(
                "shared component receipt names must be non-empty builtin strings"
            )
        receipts.append((name, _require_sha256(value, name)))

    payload_without_receipt = {
        key: value
        for key, value in row_prediction.items()
        if key != "prediction_receipt_sha256"
    }
    if source_receipt != _canonical_sha256(payload_without_receipt):
        raise D106KConditionedRouterError("prediction_receipt_sha256 drift")

    selected_arm = ROUTE_BY_K[active_k]
    return D106KConditionedPrediction(
        active_k=active_k,
        selected_arm=selected_arm,
        predictions=closed_predictions[selected_arm],
        query_order=query_ids,
        registered_classes=registry,
        source_prediction_receipt_sha256=source_receipt,
        shared_component_receipts=tuple(receipts),
    )


__all__ = [
    "ARMS",
    "CANDIDATE_ID",
    "D106KConditionedPrediction",
    "D106KConditionedRouterError",
    "G1_ROW_KEYS",
    "G1_ROW_SCHEMA",
    "ROUTE_BY_K",
    "ROW_KEYS_BY_SCHEMA",
    "SCHEMA",
    "TARGET25_ROW_KEYS",
    "TARGET25_ROW_SCHEMA",
    "route_d106_k_conditioned_prediction",
]
