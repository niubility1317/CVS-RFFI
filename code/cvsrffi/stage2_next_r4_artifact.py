"""Truth-free prediction artifact builder for the frozen NEXT-R4 matrix.

The runtime deliberately returns ordinary mappings.  This module is the only
projection from those row results to the public prediction schema consumed by
``stage2_next_r4_score``.  It does not import the runtime, read truth, or carry
features/logits.  Before invoking the scorer's structural validator it proves
the complete 24-row/144-prediction/192-arm closure itself.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from . import stage2_next_r4_matrix as matrix
from . import stage2_next_r4_score as scorer


PREDICTION_SCHEMA = scorer.PREDICTION_SCHEMA
ARTIFACT_SCHEMA = PREDICTION_SCHEMA


class NextR4ArtifactError(ValueError):
    """Raised when a runtime result cannot be projected safely."""


# These names are rejected recursively.  The first group mirrors the scorer;
# the second group prevents accidental leakage of model internals through a
# resource or receipt mapping.
_FORBIDDEN_KEYS = frozenset(
    {
        "truth",
        "truth_label",
        "query_truth",
        "query_label",
        "query_labels",
        "query_role",
        "query_roles",
        "role",
        "roles",
        "class_quota",
        "batch_class_count",
        "true_batch_class_count",
        "true_batch_class_counts",
        "global_reassignment",
        "hungarian",
        "optimal_transport",
        "feature",
        "features",
        "logit",
        "logits",
        "query_features",
        "query_logits",
        "source_feature",
        "source_features",
        "source_cache",
        "clean",
        "clean_iq",
        "raw_iq",
        "query_fit",
        "query_fitted",
        "query_update",
    }
)

_REQUIRED_ROW_KEYS = frozenset(
    {
        "row_id",
        "held_receiver",
        "held_class",
        "active_k",
        "binding_receipt",
        "fa_state_reuse_receipt",
        "registrations",
        "resource_receipt",
        "query_isolation_receipt",
    }
)
_OPTIONAL_ROW_KEYS = frozenset()
_ROW_KEYS = _REQUIRED_ROW_KEYS | _OPTIONAL_ROW_KEYS
_REGISTRATION_KEYS = frozenset({"registered_classes", "states"})
_STATE_KEYS = frozenset(
    {
        "state_id",
        "state_name_zh",
        "registered_classes",
        "query_physical_ids",
        "query_observation_ids",
        "arms",
    }
)
_ARM_KEYS = frozenset({"predictions", "receipt"})


def _key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _reject_forbidden(value: Any, *, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _key(key) in _FORBIDDEN_KEYS:
                raise NextR4ArtifactError(f"{name} contains forbidden field {key}")
            _reject_forbidden(item, name=f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_forbidden(item, name=f"{name}[{index}]")


def _plain(value: Any, *, name: str) -> Any:
    """Copy only JSON-like values; do not serialize model objects."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item, name=f"{name}.{key}") for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item, name=f"{name}[{index}]") for index, item in enumerate(value)]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NextR4ArtifactError(f"{name} contains a non-finite number")
        return value
    raise NextR4ArtifactError(f"{name} contains a non-JSON runtime value")


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, name: str) -> None:
    observed = {str(key) for key in value}
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        extra = sorted(observed - set(expected))
        raise NextR4ArtifactError(f"{name} keys drift (missing={missing}, extra={extra})")


def _strings(value: Any, *, name: str, unique: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise NextR4ArtifactError(f"{name} must be a sequence")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise NextR4ArtifactError(f"{name} contains an empty/non-string value")
    if unique and len(set(result)) != len(result):
        raise NextR4ArtifactError(f"{name} must be unique")
    return result


def _binding_query_ids(binding: Mapping[str, Any], *, observation: bool = False) -> tuple[str, ...]:
    field = "query_observation_ids_by_class" if observation else "query_ids_by_class"
    classes = _strings(binding.get("registered_classes"), name="binding.registered_classes")
    values = binding.get(field)
    if not isinstance(values, Mapping) or set(values) != set(classes):
        raise NextR4ArtifactError(f"binding.{field} class registry drift")
    flattened: list[str] = []
    for class_id in classes:
        flattened.extend(_strings(values[class_id], name=f"binding.{field}.{class_id}"))
    if len(flattened) != len(set(flattened)):
        raise NextR4ArtifactError(f"binding.{field} contains duplicate IDs")
    return tuple(flattened)


def _validate_fa_receipt(receipt: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        checked = matrix.validate_fa_state_reuse(
            {"DA1_REG0": receipt.get("source_sha256"), "DA1_REG1": receipt.get("target_sha256")}
        )
    except Exception as error:
        raise NextR4ArtifactError(f"{name} is not a canonical FA reuse receipt") from error
    result = _plain(checked, name=name)
    if dict(receipt) != result:
        raise NextR4ArtifactError(f"{name} is not canonical")
    return result


def _validate_query_isolation(receipt: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    result = _plain(receipt, name=name)
    if not isinstance(result, dict):  # defensive; _plain(mapping) is a dict
        raise NextR4ArtifactError(f"{name} must be a mapping")
    # A receipt may include additional resource counters, but every query
    # access counter it declares must be zero/false.
    zero_fields = (
        "query_rows_used_for_fit",
        "query_state_updates",
        "query_selection_count",
        "global_reassignment_calls",
    )
    false_fields = (
        "query_truth_access",
        "query_role_access",
        "class_quota_access",
        "true_batch_class_count_access",
        "query_batch_dependency",
    )
    for field in zero_fields:
        if field not in result:
            raise NextR4ArtifactError(f"{name}.{field} is required")
        if result[field] != 0:
            raise NextR4ArtifactError(f"{name}.{field} must be zero")
    for field in false_fields:
        if field not in result:
            raise NextR4ArtifactError(f"{name}.{field} is required")
        if result[field] is not False:
            raise NextR4ArtifactError(f"{name}.{field} must be false")
    return result


def _validate_arm(
    arm: Mapping[str, Any], *, name: str, registered_classes: tuple[str, ...], query_count: int
) -> dict[str, Any]:
    _exact_keys(arm, _ARM_KEYS, name=name)
    predictions = _strings(arm.get("predictions"), name=f"{name}.predictions", unique=False)
    if len(predictions) != query_count:
        raise NextR4ArtifactError(f"{name}.predictions length drift")
    if any(value not in registered_classes for value in predictions):
        raise NextR4ArtifactError(f"{name}.predictions contains an unregistered class")
    receipt = arm.get("receipt")
    if not isinstance(receipt, Mapping):
        raise NextR4ArtifactError(f"{name}.receipt must be a mapping")
    _reject_forbidden(receipt, name=f"{name}.receipt")
    # Keep arm receipts public and JSON-like.  The scorer performs the
    # semantic alias/function checks below; this builder repeats them so no
    # malformed object reaches the scorer.
    receipt_out = _plain(receipt, name=f"{name}.receipt")
    return {"predictions": list(predictions), "receipt": receipt_out}


def _validate_state(
    state: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    registration_id: str,
    state_id: str,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    name = f"row {row['row_id']} {registration_id}.{state_id}"
    _exact_keys(state, _STATE_KEYS, name=name)
    if state.get("state_id") != state_id or state.get("state_name_zh") != matrix.STATE_NAMES_ZH[state_id]:
        raise NextR4ArtifactError(f"{name} state identity drift")
    expected_classes = tuple(row["retained_classes"] if registration_id == "REG0" else row["all_registered_classes"])
    registered = _strings(state.get("registered_classes"), name=f"{name}.registered_classes")
    if registered != expected_classes:
        raise NextR4ArtifactError(f"{name}.registered_classes drift")
    query_ids = _strings(state.get("query_physical_ids"), name=f"{name}.query_physical_ids")
    observation_ids = _strings(state.get("query_observation_ids"), name=f"{name}.query_observation_ids")
    expected_query = _binding_query_ids(binding)
    expected_observation = _binding_query_ids(binding, observation=True)
    if query_ids != expected_query or observation_ids != expected_observation:
        raise NextR4ArtifactError(f"{name} query physical/observation binding drift")
    arms = state.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(matrix.ARM_IDS):
        raise NextR4ArtifactError(f"{name}.arms must contain exactly Q/H")
    parsed = {
        arm_id: _validate_arm(
            arms[arm_id],
            name=f"{name}.arms.{arm_id}",
            registered_classes=registered,
            query_count=len(query_ids),
        )
        for arm_id in matrix.ARM_IDS
    }
    q_predictions = parsed["Q"]["predictions"]
    h_predictions = parsed["H"]["predictions"]
    h_receipt = parsed["H"]["receipt"]
    active_k = int(row["active_k"])
    if active_k == 1:
        if h_predictions != q_predictions or h_receipt.get("exact_qknn_alias") is not True:
            raise NextR4ArtifactError(f"{name} K1 H must be an exact Q alias")
        if h_receipt.get("alias_target_arm") not in (None, "Q"):
            raise NextR4ArtifactError(f"{name} K1 alias target drift")
    elif active_k == 5:
        if h_receipt.get("head_status") == "NO_HEAD_FUNCTION":
            if (
                h_receipt.get("no_head_function_reason") not in {"Sr_ZERO", "QUANTIZED_RESIDUAL_ZERO"}
                or h_receipt.get("exact_qknn_alias") is not True
                or h_receipt.get("unique_prediction") is not False
                or h_receipt.get("alias_target_arm") not in (None, "Q")
                or h_predictions != q_predictions
            ):
                raise NextR4ArtifactError(f"{name} K5 NO_HEAD_FUNCTION alias receipt drift")
        elif (
            h_receipt.get("head_status") != "FUNCTIONAL"
            or h_receipt.get("exact_qknn_alias") is True
            or h_receipt.get("unique_prediction") is not True
        ):
            raise NextR4ArtifactError(f"{name} K5 functional H receipt drift")
    else:  # matrix validation should already reject this, keep a local invariant.
        raise NextR4ArtifactError(f"{name} active_k is not frozen")
    return {
        "state_id": state_id,
        "state_name_zh": matrix.STATE_NAMES_ZH[state_id],
        "registered_classes": list(registered),
        "query_physical_ids": list(query_ids),
        "query_observation_ids": list(observation_ids),
        "arms": parsed,
    }


def _validate_registration(
    registration: Mapping[str, Any], *, row: Mapping[str, Any], registration_id: str, binding: Mapping[str, Any]
) -> dict[str, Any]:
    name = f"row {row['row_id']} {registration_id}"
    _exact_keys(registration, _REGISTRATION_KEYS, name=name)
    expected_classes = tuple(row["retained_classes"] if registration_id == "REG0" else row["all_registered_classes"])
    classes = _strings(registration.get("registered_classes"), name=f"{name}.registered_classes")
    if classes != expected_classes:
        raise NextR4ArtifactError(f"{name}.registered_classes drift")
    states = registration.get("states")
    expected_states = ("DA0_REG0", "DA1_REG0") if registration_id == "REG0" else ("DA0_REG1", "DA1_REG1")
    if not isinstance(states, Mapping) or set(states) != set(expected_states):
        raise NextR4ArtifactError(f"{name}.states must contain exactly {expected_states}")
    parsed = {
        state_id: _validate_state(
            states[state_id], row=row, registration_id=registration_id, state_id=state_id, binding=binding
        )
        for state_id in expected_states
    }
    return {"registered_classes": list(classes), "states": parsed}


def _validate_row(raw: Mapping[str, Any], planned: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise NextR4ArtifactError(f"row {planned['row_id']} must be a mapping")
    _reject_forbidden(raw, name=f"row {planned['row_id']}")
    _exact_keys(raw, _ROW_KEYS, name=f"row {planned['row_id']}")
    for field in ("row_id", "held_receiver", "held_class", "active_k"):
        if raw.get(field) != planned[field]:
            raise NextR4ArtifactError(f"row {planned['row_id']} identity drift in {field}")
    binding = raw.get("binding_receipt")
    if not isinstance(binding, Mapping):
        raise NextR4ArtifactError(f"row {planned['row_id']} binding_receipt must be a mapping")
    try:
        validated_binding = matrix.validate_next_r4_binding(binding)
    except Exception as error:
        raise NextR4ArtifactError(f"row {planned['row_id']} binding receipt drift") from error
    binding_out = _plain(validated_binding, name=f"row {planned['row_id']}.binding_receipt")
    expected_key = "k1_row_id" if int(planned["active_k"]) == 1 else "k5_row_id"
    if binding_out.get(expected_key) != planned["row_id"]:
        raise NextR4ArtifactError(f"row {planned['row_id']} binding row id drift")
    if tuple(binding_out.get("registered_classes", ())) != tuple(planned["all_registered_classes"]):
        raise NextR4ArtifactError(f"row {planned['row_id']} binding class registry drift")
    # Ensure the public state query lists have a well-defined expected order
    # before any state is copied.
    _binding_query_ids(binding_out)
    _binding_query_ids(binding_out, observation=True)
    fa = raw.get("fa_state_reuse_receipt")
    if not isinstance(fa, Mapping):
        raise NextR4ArtifactError(f"row {planned['row_id']} fa_state_reuse_receipt must be a mapping")
    fa_out = _validate_fa_receipt(fa, name=f"row {planned['row_id']}.fa_state_reuse_receipt")
    registrations = raw.get("registrations")
    if not isinstance(registrations, Mapping) or set(registrations) != set(matrix.REGISTRATION_IDS):
        raise NextR4ArtifactError(f"row {planned['row_id']} registrations must contain REG0/REG1")
    result: dict[str, Any] = {
        "row_id": planned["row_id"],
        "held_receiver": planned["held_receiver"],
        "held_class": planned["held_class"],
        "active_k": planned["active_k"],
        "evaluation_semantics": matrix.PROXY_SEMANTICS,
        "formal_new_registration_claim": False,
        "binding_receipt": binding_out,
        "fa_state_reuse_receipt": fa_out,
        "registrations": {
            registration_id: _validate_registration(
                registrations[registration_id], row=planned, registration_id=registration_id, binding=binding_out
            )
            for registration_id in matrix.REGISTRATION_IDS
        },
    }
    resource = raw.get("resource_receipt")
    if not isinstance(resource, Mapping):
        raise NextR4ArtifactError(f"row {planned['row_id']}.resource_receipt must be a mapping")
    result["resource_receipt"] = _plain(resource, name=f"row {planned['row_id']}.resource_receipt")
    isolation = raw.get("query_isolation_receipt")
    if not isinstance(isolation, Mapping):
        raise NextR4ArtifactError(f"row {planned['row_id']}.query_isolation_receipt must be a mapping")
    result["query_isolation_receipt"] = _validate_query_isolation(
        isolation, name=f"row {planned['row_id']}.query_isolation_receipt"
    )
    return result


def _assert_structural_closure(artifact: Mapping[str, Any]) -> None:
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != matrix.ROW_COUNT:
        raise NextR4ArtifactError("prediction artifact must contain exactly 24 rows")
    arm_count = 0
    unique_count = 0
    for row in rows:
        registrations = row["registrations"]
        for registration_id in matrix.REGISTRATION_IDS:
            states = registrations[registration_id]["states"]
            for state_id in ("DA0_REG0", "DA1_REG0") if registration_id == "REG0" else ("DA0_REG1", "DA1_REG1"):
                arms = states[state_id]["arms"]
                arm_count += 2
                unique_count += 1 if int(row["active_k"]) == 1 else 2
                if set(arms) != set(matrix.ARM_IDS):
                    raise NextR4ArtifactError("prediction artifact arm closure drift")
    if arm_count != matrix.ARTIFACT_ARM_COUNT or unique_count != matrix.UNIQUE_PREDICTION_COUNT:
        raise NextR4ArtifactError(
            f"prediction closure drift: unique={unique_count}, arms={arm_count}; expected {matrix.UNIQUE_PREDICTION_COUNT}/{matrix.ARTIFACT_ARM_COUNT}"
        )


def build_next_r4_prediction_artifact(*, plan: Mapping[str, Any], row_results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Project complete runtime rows into the strict truth-free artifact.

    ``row_results`` must use the exact runtime mapping contract; no aliases,
    runtime objects, logits, feature payloads or truth-bearing fields are
    accepted.  The result is ordinary JSON-compatible mappings and can be
    handed directly to ``stage2_next_r4_score``.
    """

    try:
        frozen = matrix.validate_next_r4_proxy24_plan(plan)
    except Exception as error:
        raise NextR4ArtifactError("invalid NEXT-R4 plan") from error
    if isinstance(row_results, (str, bytes)) or not isinstance(row_results, Sequence):
        raise NextR4ArtifactError("row_results must be an ordered sequence")
    if len(row_results) != matrix.ROW_COUNT:
        raise NextR4ArtifactError("row_results must contain exactly 24 rows")
    rows = [_validate_row(raw, planned) for raw, planned in zip(row_results, frozen["rows"], strict=True)]
    if [row["row_id"] for row in rows] != [row["row_id"] for row in frozen["rows"]]:
        raise NextR4ArtifactError("row_results must follow the frozen matrix order")
    artifact: dict[str, Any] = {
        "schema": PREDICTION_SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "matrix_sha256": frozen["matrix_sha256"],
        "evaluation_semantics": matrix.PROXY_SEMANTICS,
        "formal_new_registration_claim": False,
        "truth_loaded": False,
        "rows_complete": True,
        "row_count": matrix.ROW_COUNT,
        "unique_prediction_count": matrix.UNIQUE_PREDICTION_COUNT,
        "prediction_count": matrix.UNIQUE_PREDICTION_COUNT,
        "artifact_arm_count": matrix.ARTIFACT_ARM_COUNT,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "rows": rows,
    }
    _assert_structural_closure(artifact)
    # The scorer's private validator is structural and does not touch truth.
    # Calling it here is safe only after the closure proof above.
    try:
        scorer._validate_prediction(artifact, frozen)
    except Exception as error:
        raise NextR4ArtifactError("scorer structural validation rejected artifact") from error
    return artifact


build_artifact = build_next_r4_prediction_artifact


__all__ = [
    "ARTIFACT_SCHEMA",
    "NextR4ArtifactError",
    "PREDICTION_SCHEMA",
    "build_artifact",
    "build_next_r4_prediction_artifact",
]
