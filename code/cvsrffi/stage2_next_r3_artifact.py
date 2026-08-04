"""Mechanical truth-free prediction artifact for the frozen NEXT-R3 proxy.

The runtime owns the actual prediction heads.  This module only projects the
runtime's public four-state view into the prediction mapping consumed by
``stage2_next_r3_score``.  It deliberately does not fit, score, or join
truth.  A runtime result may be the real typed ``NextR3RuntimeResult`` or a
small object/mapping exposing the same public fields, which keeps the helper
usable by runners and focused tests without constructing D106 state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from . import stage2_next_r3_matrix as matrix
from . import stage2_next_r3_score as score


ARTIFACT_SCHEMA = score.PREDICTION_SCHEMA
ARTIFACT_VERSION = "v1"

_FORBIDDEN_KEYS = frozenset(
    {
        "truth",
        "truth_label",
        "query_truth",
        "query_label",
        "query_labels",
        "query_role",
        "query_roles",
        "class_quota",
        "batch_class_count",
        "true_batch_class_count",
        "true_batch_class_counts",
        "global_reassignment",
        "hungarian",
        "optimal_transport",
    }
)
_REGISTRATION_STATES = {
    "REG0": ("DA0_REG0", "DA1_REG0"),
    "REG1": ("DA0_REG1", "DA1_REG1"),
}
_HEADS = ("Q", "F", "L")


class NextR3ArtifactError(ValueError):
    """Raised when a runtime-to-artifact binding is incomplete or polluted."""


def _key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _reject_forbidden(value: Any, *, name: str) -> None:
    """Reject truth/role/quota fields before they can enter the artifact."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if _key(key) in _FORBIDDEN_KEYS:
                raise NextR3ArtifactError(f"{name} contains forbidden field {key}")
            _reject_forbidden(item, name=f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_forbidden(item, name=f"{name}[{index}]")


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _required(value: Any, name: str) -> Any:
    if value is None:
        raise NextR3ArtifactError(f"missing runtime field {name}")
    return value


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    """Make a shallow public mapping from a mapping or an ``as_dict`` object."""

    if isinstance(value, Mapping):
        result = {str(key): item for key, item in value.items()}
    else:
        as_dict = getattr(value, "as_dict", None)
        if callable(as_dict):
            candidate = as_dict()
            if not isinstance(candidate, Mapping):
                raise NextR3ArtifactError(f"{name}.as_dict() must return a mapping")
            result = {str(key): item for key, item in candidate.items()}
        else:
            result = {}
    _reject_forbidden(result, name=name)
    return result


def _plain(value: Any, *, name: str) -> Any:
    """Copy JSON-like receipt values while never copying arbitrary runtime state."""

    if isinstance(value, Mapping):
        _reject_forbidden(value, name=name)
        return {str(key): _plain(item, name=f"{name}.{key}") for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item, name=f"{name}[{index}]") for index, item in enumerate(value)]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    # Numpy scalar values occur in a few receipts but importing numpy here is
    # unnecessary.  ``item`` is the standard scalar conversion protocol.
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if converted is not value:
            return _plain(converted, name=name)
    # Arrays/features/logits are intentionally not part of a prediction
    # artifact.  Callers should expose their hashes and predictions instead.
    raise NextR3ArtifactError(f"{name} contains an unsupported runtime value")


def _strings(value: Any, *, name: str, unique: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise NextR3ArtifactError(f"{name} must be a sequence")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise NextR3ArtifactError(f"{name} contains an empty value")
    if unique and len(set(result)) != len(result):
        raise NextR3ArtifactError(f"{name} must contain unique values")
    return result


def _cache_public(cache: Any, *, name: str) -> dict[str, Any]:
    """Copy the physical/query/class/cache receipt surface of one cache."""

    receipt = _get(cache, "receipt")
    receipt_payload = _plain(receipt, name=f"{name}.receipt") if receipt is not None else {}
    registered = _strings(
        _required(_get(cache, "registered_classes"), f"{name}.registered_classes"),
        name=f"{name}.registered_classes",
    )
    support_ids = _strings(
        _required(_get(cache, "support_physical_ids"), f"{name}.support_physical_ids"),
        name=f"{name}.support_physical_ids",
    )
    query_ids = _strings(
        _required(_get(cache, "query_physical_ids"), f"{name}.query_physical_ids"),
        name=f"{name}.query_physical_ids",
    )
    labels = _strings(
        _required(_get(cache, "support_labels"), f"{name}.support_labels"),
        name=f"{name}.support_labels",
        unique=False,
    )
    if len(labels) != len(support_ids):
        raise NextR3ArtifactError(f"{name}.support_labels/support_physical_ids length drift")
    if set(support_ids) & set(query_ids):
        raise NextR3ArtifactError(f"{name} support/query physical IDs overlap")
    representation = _required(_get(cache, "representation"), f"{name}.representation")
    registration_state = _required(
        _get(cache, "registration_state"), f"{name}.registration_state"
    )
    payload: dict[str, Any] = {
        "representation": str(representation),
        "registration_state": str(registration_state),
        "registered_classes": list(registered),
        "support_labels": list(labels),
        "support_physical_ids": list(support_ids),
        "query_physical_ids": list(query_ids),
        "support_physical_root_sha256": _get(cache, "support_physical_root_sha256"),
        "query_physical_root_sha256": _get(cache, "query_physical_root_sha256"),
        "ordered_support_physical_root_sha256": _get(
            cache, "ordered_support_physical_root_sha256"
        ),
        "ordered_query_physical_root_sha256": _get(
            cache, "ordered_query_physical_root_sha256"
        ),
        "cache_sha256": _get(cache, "cache_sha256", receipt_payload.get("cache_sha256")),
        "receipt": receipt_payload,
    }
    # Omit absent optional hash properties, but keep every actual receipt value.
    return {key: value for key, value in payload.items() if value is not None}


def _arm_predictions(arm: Any, *, name: str, expected_classes: Sequence[str]) -> tuple[str, ...]:
    predictions = _get(arm, "predictions")
    if predictions is None and isinstance(arm, Mapping):
        predictions = arm.get("values")
    values = _strings(_required(predictions, f"{name}.predictions"), name=f"{name}.predictions", unique=False)
    if any(value not in expected_classes for value in values):
        raise NextR3ArtifactError(f"{name}.predictions contains an unregistered class")
    return values


def _arm_public(arm: Any, *, name: str, expected_classes: Sequence[str]) -> dict[str, Any]:
    cache = _required(_get(arm, "cache"), f"{name}.cache")
    predictions = _arm_predictions(arm, name=name, expected_classes=expected_classes)
    receipt = _get(arm, "receipt")
    payload: dict[str, Any] = {"predictions": list(predictions), "cache": _cache_public(cache, name=f"{name}.cache")}
    if receipt is not None:
        payload["receipt"] = _plain(receipt, name=f"{name}.receipt")
    return payload


def _bridge_public(result: Any, *, expected_row_id: str) -> dict[str, Any]:
    bridge = _required(_get(result, "bridge"), "runtime.bridge")
    bridge_payload = _as_mapping(bridge, name="runtime.bridge")
    row_id = _get(bridge, "row_id", bridge_payload.get("row_id"))
    if row_id != expected_row_id:
        raise NextR3ArtifactError("runtime bridge row_id does not match plan row")
    binding_sha = _get(bridge, "binding_sha256", bridge_payload.get("binding_sha256"))
    if binding_sha is not None:
        bridge_payload["binding_sha256"] = str(binding_sha)
    bridge_payload["row_id"] = expected_row_id
    return _plain(bridge_payload, name="runtime.bridge")


def _registration_public(
    result: Any,
    *,
    registration_id: str,
    planned: Mapping[str, Any],
    state_ids: Sequence[str],
) -> dict[str, Any]:
    registration = _get(result, registration_id.lower())
    if registration is None:
        registration = _get(result, registration_id)
    registration = _required(registration, f"runtime.{registration_id.lower()}")
    registration_state = _get(registration, "registration_state")
    if registration_state != registration_id:
        raise NextR3ArtifactError(f"runtime {registration_id} registration_state drift")
    caches = _required(_get(registration, "caches"), f"runtime.{registration_id}.caches")
    if not isinstance(caches, Mapping) or set(caches) != {"R0", "R1"}:
        raise NextR3ArtifactError(f"runtime.{registration_id}.caches must contain R0/R1")
    expected_classes = tuple(
        planned["retained_classes"] if registration_id == "REG0" else planned["all_registered_classes"]
    )
    cache_payloads: dict[str, dict[str, Any]] = {}
    query_ids: tuple[str, ...] | None = None
    support_ids: tuple[str, ...] | None = None
    support_labels: tuple[str, ...] | None = None
    for representation in ("R0", "R1"):
        cache = caches[representation]
        payload = _cache_public(cache, name=f"runtime.{registration_id}.caches.{representation}")
        if tuple(payload["registered_classes"]) != expected_classes:
            raise NextR3ArtifactError(f"runtime {registration_id} class registry drift")
        current_query = tuple(payload["query_physical_ids"])
        current_support = tuple(payload["support_physical_ids"])
        current_labels = tuple(payload["support_labels"])
        if query_ids is None:
            query_ids, support_ids, support_labels = current_query, current_support, current_labels
        elif current_query != query_ids or current_support != support_ids or current_labels != support_labels:
            raise NextR3ArtifactError(f"runtime {registration_id} R0/R1 cache binding drift")
        cache_payloads[representation] = payload
    assert query_ids is not None and support_ids is not None and support_labels is not None
    arms = _get(registration, "arms")
    if not isinstance(arms, Mapping):
        raise NextR3ArtifactError(f"runtime.{registration_id}.arms must be a mapping")
    arm_payloads = {
        arm_id: _arm_public(
            _required(arms.get(arm_id), f"runtime.{registration_id}.arms.{arm_id}"),
            name=f"runtime.{registration_id}.arms.{arm_id}",
            expected_classes=expected_classes,
        )
        for arm_id in matrix.ARM_IDS
    }
    # Verify the full registration arm bundle uses the same query stream as
    # its caches.  The state view below independently selects only Q/F/L.
    for arm_id, arm_payload in arm_payloads.items():
        if tuple(arm_payload["cache"]["query_physical_ids"]) != query_ids:
            raise NextR3ArtifactError(f"runtime {registration_id}/{arm_id} query binding drift")
    receipt = _get(registration, "receipt")
    output: dict[str, Any] = {
        "registration_id": registration_id,
        "registered_classes": list(expected_classes),
        "support_physical_ids": list(support_ids),
        "support_labels": list(support_labels),
        "query_physical_ids": list(query_ids),
        "states": {},
        "cache_receipts": cache_payloads,
        "arm_receipts": {arm_id: arm_payload["receipt"] for arm_id, arm_payload in arm_payloads.items() if "receipt" in arm_payload},
    }
    if receipt is not None:
        output["receipt"] = _plain(receipt, name=f"runtime.{registration_id}.receipt")

    four_state = _required(_get(result, "four_state"), "runtime.four_state")
    if not isinstance(four_state, Mapping):
        raise NextR3ArtifactError("runtime.four_state must be a mapping")
    four_receipt = _get(result, "four_state_receipt")
    states_receipt = four_receipt.get("states", {}) if isinstance(four_receipt, Mapping) else {}
    for state_id in state_ids:
        state = _required(four_state.get(state_id), f"runtime.four_state.{state_id}")
        if not isinstance(state, Mapping):
            raise NextR3ArtifactError(f"runtime.four_state.{state_id} must be a mapping")
        prefix = "R0" if state_id.startswith("DA0") else "R1"
        state_arms: dict[str, Any] = {}
        state_query_ids: tuple[str, ...] | None = None
        for head in _HEADS:
            arm = state.get(head)
            if arm is None:
                arm = state.get(f"{prefix}{head}")
            arm = _required(arm, f"runtime.four_state.{state_id}.{head}")
            full_arm_id = f"{prefix}{head}"
            arm_payload = _arm_public(
                arm,
                name=f"runtime.four_state.{state_id}.{head}",
                expected_classes=expected_classes,
            )
            arm_query_ids = tuple(arm_payload["cache"]["query_physical_ids"])
            if state_query_ids is None:
                state_query_ids = arm_query_ids
            elif state_query_ids != arm_query_ids:
                raise NextR3ArtifactError(f"runtime {state_id} Q/F/L query binding drift")
            if state_query_ids != query_ids:
                raise NextR3ArtifactError(f"runtime {state_id} query stream differs from registration cache")
            state_arms[full_arm_id] = arm_payload
        state_output: dict[str, Any] = {
            "registration_id": registration_id,
            "state_id": state_id,
            "query_physical_ids": list(state_query_ids or ()),
            "arms": state_arms,
        }
        if isinstance(states_receipt, Mapping) and state_id in states_receipt:
            state_output["receipt"] = _plain(states_receipt[state_id], name=f"runtime.four_state_receipt.states.{state_id}")
        output["states"][state_id] = state_output
    return output


def build_next_r3_prediction_artifact(
    plan: Mapping[str, Any], row_results: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Build the complete 24-row, 96-state, 288-arm truth-free artifact."""

    try:
        frozen = matrix.validate_next_r3_proxy24_plan(plan)
    except Exception as error:
        raise NextR3ArtifactError("invalid NEXT-R3 plan") from error
    if not isinstance(row_results, Mapping):
        raise NextR3ArtifactError("row_results must be a mapping keyed by plan row_id")
    expected_ids = tuple(str(row["row_id"]) for row in frozen["rows"])
    if set(row_results) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(row_results))
        extra = sorted(set(row_results) - set(expected_ids))
        raise NextR3ArtifactError(f"row_results coverage drift; missing={missing}, extra={extra}")

    rows: list[dict[str, Any]] = []
    for planned in frozen["rows"]:
        row_id = str(planned["row_id"])
        result = row_results[row_id]
        # Mapping fakes are checked for forbidden fields; typed runtime objects
        # expose only the explicit public fields copied below.
        if isinstance(result, Mapping):
            _reject_forbidden(result, name=f"row_results[{row_id}]")
        bridge = _bridge_public(result, expected_row_id=row_id)
        registrations: dict[str, Any] = {}
        for registration_id, state_ids in _REGISTRATION_STATES.items():
            registrations[registration_id] = _registration_public(
                result,
                registration_id=registration_id,
                planned=planned,
                state_ids=state_ids,
            )
        row: dict[str, Any] = {
            "row_id": row_id,
            "held_receiver": planned["held_receiver"],
            "held_class": planned["held_class"],
            "active_k": planned["active_k"],
            "retained_classes": list(planned["retained_classes"]),
            "all_registered_classes": list(planned["all_registered_classes"]),
            "evaluation_semantics": matrix.PROXY_SEMANTICS,
            "artifact_semantics": matrix.PROXY_SEMANTICS,
            "formal_new_registration_claim": False,
            "binding": bridge,
            "bridge": bridge,
            "registrations": registrations,
        }
        runtime_receipt = _get(result, "runtime_receipt")
        resource_receipt = _get(result, "resource_receipt")
        if runtime_receipt is not None:
            row["runtime_receipt"] = _plain(runtime_receipt, name=f"runtime[{row_id}].runtime_receipt")
        if resource_receipt is not None:
            row["resource_receipt"] = _plain(resource_receipt, name=f"runtime[{row_id}].resource_receipt")
        four_state_receipt = _get(result, "four_state_receipt")
        if four_state_receipt is not None:
            row["four_state_receipt"] = _plain(four_state_receipt, name=f"runtime[{row_id}].four_state_receipt")
        da0 = _get(result, "da1_reg0_state_sha256")
        da1 = _get(result, "da1_reg1_state_sha256")
        if da0 is not None:
            row["da1_reg0_state_sha256"] = str(da0)
        if da1 is not None:
            row["da1_reg1_state_sha256"] = str(da1)
        rows.append(row)

    artifact: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "artifact_version": ARTIFACT_VERSION,
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "matrix_sha256": frozen["matrix_sha256"],
        "evaluation_semantics": matrix.PROXY_SEMANTICS,
        "artifact_semantics": matrix.PROXY_SEMANTICS,
        "formal_new_registration_claim": False,
        "formal_target_claim": False,
        "truth_loaded": False,
        "rows_complete": True,
        "single_candidate": True,
        "candidate_count": 1,
        "row_count": matrix.ROW_COUNT,
        "state_prediction_count": matrix.STATE_PREDICTION_COUNT,
        "arm_prediction_count": matrix.ARM_PREDICTION_COUNT,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "rows": rows,
    }
    # A final local shape check catches accidental truth pollution and keeps
    # the builder's contract aligned with the independent scorer.
    try:
        score._validate_prediction(artifact, frozen)
    except Exception as error:
        raise NextR3ArtifactError("built NEXT-R3 artifact does not satisfy scorer schema") from error
    return artifact


__all__ = ["ARTIFACT_SCHEMA", "NextR3ArtifactError", "build_next_r3_prediction_artifact"]
