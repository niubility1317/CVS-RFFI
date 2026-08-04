"""Frozen, truth-free NEXT-R5 Target5 four-state release plan.

The plan deliberately contains only the five D106 Target25 receivers at
K=5/new=20, their three approved LEO weak scenarios, and the four explicit
domain-adaptation/registration states.  It has no dataset, predictor, score,
truth, filesystem, or N607 dependency.  Runtime code can use the companion
binding receipt validator to prove that all four states of one scenario reuse
the same opaque query-ID root and that the DA1 state is reused across
registration.
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .stage2_d106_matrix_protocol import (
    LEO_SCENARIOS as D106_TARGET25_LEO_SCENARIOS,
    PROTOCOL_SCHEMA as D106_PROTOCOL_SCHEMA,
    RECEIVERS as D106_TARGET25_RECEIVERS,
)


SCHEMA = "cvs.stage2.next_r5.k5_fa_rdce3_q.target5.plan.v1"
BINDING_RECEIPT_SCHEMA = "cvs.stage2.next_r5.k5_fa_rdce3_q.target5.binding_receipt.v1"
CANDIDATE_ID = "NEXT-R5-K5-FA-RDCE3-Q"
PROTOCOL_SCHEMA = D106_PROTOCOL_SCHEMA

# The receiver and scenario tuples are aliases, rather than copied literals:
# Target5 must stay tied to the pre-existing D106 Target25 definition.
RECEIVERS = D106_TARGET25_RECEIVERS
LEO_SCENARIOS = D106_TARGET25_LEO_SCENARIOS

PREFERRED_SEED = 713103
K_SHOT = 5
NEW_CLASS_COUNT = 20
ARM_IDS = ("Q",)
STATE_IDS = ("DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1")
STATE_NAMES_ZH = {
    "DA0_REG0": "域适应前/新类注册前",
    "DA1_REG0": "域适应后/新类注册前",
    "DA0_REG1": "域适应前/新类注册后",
    "DA1_REG1": "域适应后/新类注册后",
}
REG0_STATES = frozenset(("DA0_REG0", "DA1_REG0"))
REG1_STATES = frozenset(("DA0_REG1", "DA1_REG1"))
DA1_STATES = frozenset(("DA1_REG0", "DA1_REG1"))

JOB_COUNT = len(RECEIVERS)
SCENARIO_ROW_COUNT = JOB_COUNT * len(LEO_SCENARIOS)
STATE_SURFACE_COUNT = SCENARIO_ROW_COUNT * len(STATE_IDS)

PROHIBITED_COMPONENTS = ("K1", "CER", "H", "D92-Lite", "K10")
NEGATIVE_PROTOCOL_FLAGS = (
    "clean_runtime_access",
    "source_runtime_access",
    "query_truth_access",
    "query_role_access",
    "query_fit_access",
    "query_update_access",
    "query_selection",
    "query_class_quota_access",
    "query_true_batch_class_count_access",
    "query_global_reassignment",
    "output_overwrite_allowed",
    "parameter_search_allowed",
    "seed_search_allowed",
    "performance_selection_allowed",
)

_PLAN_FIELDS = frozenset(
    (
        "schema",
        "candidate_id",
        "protocol_schema",
        "receivers",
        "leo_scenarios",
        "state_ids",
        "state_names_zh",
        "k_shot",
        "new_class_count",
        "arm_ids",
        "prohibited_components",
        "seed_policy",
        "negative_protocol_flags",
        "metric_availability_by_state",
        "query_id_root_policy",
        "fa_state_reuse",
        "job_count",
        "scenario_row_count",
        "state_surface_count",
        "jobs",
        "scenario_rows",
        "state_surfaces",
        "plan_receipt_sha256",
    )
)
_JOB_FIELDS = frozenset(("job_id", "receiver", "seed", "k_shot", "new_class_count"))
_SCENARIO_ROW_FIELDS = frozenset(
    ("scenario_row_id", "job_id", "receiver", "scenario", "seed", "k_shot", "new_class_count")
)
_STATE_SURFACE_FIELDS = frozenset(
    (
        "surface_id",
        "scenario_row_id",
        "job_id",
        "receiver",
        "scenario",
        "seed",
        "k_shot",
        "new_class_count",
        "arm_id",
        "state",
        "seen_new_acc",
        "H_old_new",
    )
)
_SEED_POLICY = {
    "preferred_seed": PREFERRED_SEED,
    "allowed_seeds": [PREFERRED_SEED],
    "fallback_allowed": False,
    "same_key_failure_action": "technical_stop_new_immutable_revision_required",
    "performance_or_result_selection": False,
}
_METRIC_AVAILABILITY_BY_STATE = {
    "DA0_REG0": {"seen_new_acc": "N/A", "H_old_new": "N/A"},
    "DA1_REG0": {"seen_new_acc": "N/A", "H_old_new": "N/A"},
    "DA0_REG1": {"seen_new_acc": "REQUIRED", "H_old_new": "REQUIRED"},
    "DA1_REG1": {"seen_new_acc": "REQUIRED", "H_old_new": "REQUIRED"},
}
_QUERY_ID_ROOT_POLICY = {
    "old_field": "old_query_id_root_sha256",
    "old_scope": "same_job_same_scenario_all_four_states",
    "old_requirement": "exact_same_opaque_old_query_id_root",
    "new_field": "new_query_id_root_sha256",
    "new_scope": "same_job_same_scenario_reg1_states_only",
    "new_requirement": "N/A_in_REG0_and_exact_same_opaque_new_query_id_root_in_REG1",
}
_FA_STATE_REUSE = {
    "required": True,
    "source_state": "DA1_REG0",
    "target_state": "DA1_REG1",
    "scope": "same_job_same_scenario",
    "field": "fa_state_binding_sha256",
    "requirement": "exact_same_binding",
}
_BINDING_RECEIPT_FIELDS = frozenset(
    ("schema", "plan_receipt_sha256", "state_bindings", "binding_receipt_sha256")
)
_STATE_BINDING_FIELDS = frozenset(
    (
        "surface_id",
        "effective_seed",
        "old_query_id_root_sha256",
        "new_query_id_root_sha256",
        "fa_state_binding_sha256",
    )
)


class NextR5Target5PlanError(ValueError):
    """Raised when frozen Target5 metadata or a runtime binding drifts."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    raise NextR5Target5PlanError("Target5 canonical payload has an unsupported value")


def canonical_bytes(value: Any) -> bytes:
    """Return the one canonical JSON representation used by Target5 receipts."""

    return json.dumps(
        _json_ready(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return a lowercase SHA256 over :func:`canonical_bytes`."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    raise NextR5Target5PlanError("Target5 immutable payload has an unsupported value")


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _job_id(receiver: str) -> str:
    return (
        f"next-r5-rx-{receiver.replace('-', '_')}__seed-{PREFERRED_SEED}"
        f"__k-{K_SHOT}__new-{NEW_CLASS_COUNT}"
    )


def _scenario_row_id(job_id: str, scenario: str) -> str:
    return f"{job_id}__scenario-{scenario}"


def _surface_id(scenario_row_id: str, state: str) -> str:
    return f"{scenario_row_id}__arm-Q__state-{state}"


def _frozen_jobs() -> list[dict[str, Any]]:
    return [
        {
            "job_id": _job_id(receiver),
            "receiver": receiver,
            "seed": PREFERRED_SEED,
            "k_shot": K_SHOT,
            "new_class_count": NEW_CLASS_COUNT,
        }
        for receiver in RECEIVERS
    ]


def _frozen_scenario_rows(jobs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        for scenario in LEO_SCENARIOS:
            rows.append(
                {
                    "scenario_row_id": _scenario_row_id(str(job["job_id"]), scenario),
                    "job_id": job["job_id"],
                    "receiver": job["receiver"],
                    "scenario": scenario,
                    "seed": PREFERRED_SEED,
                    "k_shot": K_SHOT,
                    "new_class_count": NEW_CLASS_COUNT,
                }
            )
    return rows


def _frozen_state_surfaces(
    scenario_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    for row in scenario_rows:
        for state in STATE_IDS:
            metric_policy = _METRIC_AVAILABILITY_BY_STATE[state]
            surfaces.append(
                {
                    "surface_id": _surface_id(str(row["scenario_row_id"]), state),
                    "scenario_row_id": row["scenario_row_id"],
                    "job_id": row["job_id"],
                    "receiver": row["receiver"],
                    "scenario": row["scenario"],
                    "seed": PREFERRED_SEED,
                    "k_shot": K_SHOT,
                    "new_class_count": NEW_CLASS_COUNT,
                    "arm_id": "Q",
                    "state": state,
                    "seen_new_acc": metric_policy["seen_new_acc"],
                    "H_old_new": metric_policy["H_old_new"],
                }
            )
    return surfaces


def build_next_r5_target5_plan() -> Mapping[str, Any]:
    """Build the only permitted NEXT-R5 Target5 structural plan."""

    jobs = _frozen_jobs()
    scenario_rows = _frozen_scenario_rows(jobs)
    state_surfaces = _frozen_state_surfaces(scenario_rows)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "receivers": list(RECEIVERS),
        "leo_scenarios": list(LEO_SCENARIOS),
        "state_ids": list(STATE_IDS),
        "state_names_zh": _json_ready(STATE_NAMES_ZH),
        "k_shot": K_SHOT,
        "new_class_count": NEW_CLASS_COUNT,
        "arm_ids": list(ARM_IDS),
        "prohibited_components": list(PROHIBITED_COMPONENTS),
        "seed_policy": _json_ready(_SEED_POLICY),
        "negative_protocol_flags": {
            name: False for name in NEGATIVE_PROTOCOL_FLAGS
        },
        "metric_availability_by_state": _json_ready(_METRIC_AVAILABILITY_BY_STATE),
        "query_id_root_policy": _json_ready(_QUERY_ID_ROOT_POLICY),
        "fa_state_reuse": _json_ready(_FA_STATE_REUSE),
        "job_count": JOB_COUNT,
        "scenario_row_count": SCENARIO_ROW_COUNT,
        "state_surface_count": STATE_SURFACE_COUNT,
        "jobs": jobs,
        "scenario_rows": scenario_rows,
        "state_surfaces": state_surfaces,
    }
    payload["plan_receipt_sha256"] = canonical_sha256(payload)
    return _deep_freeze(_json_ready(payload))


def _require_exact_mapping_keys(
    value: Any, expected: frozenset[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise NextR5Target5PlanError(f"{name} field closure drift")
    return value


def _validate_frozen_plan_shape(payload: Mapping[str, Any]) -> None:
    _require_exact_mapping_keys(payload, _PLAN_FIELDS, "Target5 plan")
    if (
        payload["schema"] != SCHEMA
        or payload["candidate_id"] != CANDIDATE_ID
        or payload["protocol_schema"] != PROTOCOL_SCHEMA
        or tuple(payload["receivers"]) != RECEIVERS
        or tuple(payload["leo_scenarios"]) != LEO_SCENARIOS
        or tuple(payload["state_ids"]) != STATE_IDS
        or payload["state_names_zh"] != STATE_NAMES_ZH
        or payload["k_shot"] != K_SHOT
        or payload["new_class_count"] != NEW_CLASS_COUNT
        or tuple(payload["arm_ids"]) != ARM_IDS
        or tuple(payload["prohibited_components"]) != PROHIBITED_COMPONENTS
        or _json_ready(payload["seed_policy"]) != _SEED_POLICY
        or _json_ready(payload["metric_availability_by_state"])
        != _METRIC_AVAILABILITY_BY_STATE
        or _json_ready(payload["query_id_root_policy"]) != _QUERY_ID_ROOT_POLICY
        or _json_ready(payload["fa_state_reuse"]) != _FA_STATE_REUSE
        or payload["job_count"] != JOB_COUNT
        or payload["scenario_row_count"] != SCENARIO_ROW_COUNT
        or payload["state_surface_count"] != STATE_SURFACE_COUNT
    ):
        raise NextR5Target5PlanError("Target5 frozen constants drift")
    negative_flags = payload["negative_protocol_flags"]
    if (
        not isinstance(negative_flags, Mapping)
        or tuple(negative_flags) != NEGATIVE_PROTOCOL_FLAGS
        or any(negative_flags[name] is not False for name in NEGATIVE_PROTOCOL_FLAGS)
    ):
        raise NextR5Target5PlanError("Target5 negative protocol flags drift")
    for state in REG0_STATES:
        if payload["metric_availability_by_state"][state] != {
            "seen_new_acc": "N/A",
            "H_old_new": "N/A",
        }:
            raise NextR5Target5PlanError("REG0 new/H metric availability drift")


def validate_next_r5_target5_plan(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fail closed unless *value* is the exact deterministic Target5 plan."""

    if not isinstance(value, Mapping):
        raise NextR5Target5PlanError("Target5 plan must be a mapping")
    payload = dict(value)
    _validate_frozen_plan_shape(payload)
    receipt = payload.pop("plan_receipt_sha256")
    if not _is_sha256(receipt) or receipt != canonical_sha256(payload):
        raise NextR5Target5PlanError("Target5 plan canonical receipt drift")

    expected = _json_ready(build_next_r5_target5_plan())
    actual = _json_ready(value)
    if actual != expected:
        raise NextR5Target5PlanError("Target5 plan does not match deterministic rebuild")

    jobs = value["jobs"]
    scenario_rows = value["scenario_rows"]
    surfaces = value["state_surfaces"]
    if (
        not isinstance(jobs, Sequence)
        or isinstance(jobs, (str, bytes))
        or len(jobs) != JOB_COUNT
        or not isinstance(scenario_rows, Sequence)
        or isinstance(scenario_rows, (str, bytes))
        or len(scenario_rows) != SCENARIO_ROW_COUNT
        or not isinstance(surfaces, Sequence)
        or isinstance(surfaces, (str, bytes))
        or len(surfaces) != STATE_SURFACE_COUNT
    ):
        raise NextR5Target5PlanError("Target5 cardinality drift")
    for job in jobs:
        _require_exact_mapping_keys(job, _JOB_FIELDS, "Target5 job")
    for row in scenario_rows:
        _require_exact_mapping_keys(row, _SCENARIO_ROW_FIELDS, "Target5 scenario row")
    for surface in surfaces:
        _require_exact_mapping_keys(surface, _STATE_SURFACE_FIELDS, "Target5 surface")
    return _deep_freeze(_json_ready(value))


def _state_binding_payload(
    plan: Mapping[str, Any], state_bindings: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    validate_next_r5_target5_plan(plan)
    if isinstance(state_bindings, (str, bytes)) or not isinstance(
        state_bindings, Sequence
    ):
        raise NextR5Target5PlanError("Target5 state bindings must be a sequence")
    if len(state_bindings) != STATE_SURFACE_COUNT:
        raise NextR5Target5PlanError("Target5 binding state-surface coverage drift")
    records: list[dict[str, Any]] = []
    expected_surfaces = plan["state_surfaces"]
    for expected, binding in zip(expected_surfaces, state_bindings, strict=True):
        _require_exact_mapping_keys(binding, _STATE_BINDING_FIELDS, "Target5 state binding")
        record = dict(binding)
        if record["surface_id"] != expected["surface_id"]:
            raise NextR5Target5PlanError("Target5 binding surface order/identity drift")
        if type(record["effective_seed"]) is not int or record["effective_seed"] != expected["seed"]:
            raise NextR5Target5PlanError("Target5 effective seed binding drift")
        if not _is_sha256(record["old_query_id_root_sha256"]):
            raise NextR5Target5PlanError("Target5 old-query-ID root must be a SHA256")
        state = expected["state"]
        new_query_root = record["new_query_id_root_sha256"]
        if state in REG0_STATES:
            if new_query_root != "N/A":
                raise NextR5Target5PlanError("REG0 must not carry a new-query-ID root")
        elif not _is_sha256(new_query_root):
            raise NextR5Target5PlanError("REG1 new-query-ID root must be a SHA256")
        binding_sha = record["fa_state_binding_sha256"]
        if state in DA1_STATES:
            if not _is_sha256(binding_sha):
                raise NextR5Target5PlanError("DA1 FA state binding must be a SHA256")
        elif binding_sha != "N/A":
            raise NextR5Target5PlanError("DA0 must not carry an FA state binding")
        records.append(record)
    return {
        "schema": BINDING_RECEIPT_SCHEMA,
        "plan_receipt_sha256": plan["plan_receipt_sha256"],
        "state_bindings": records,
    }


def _validate_binding_constraints(
    plan: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    by_scenario: dict[str, dict[str, Mapping[str, Any]]] = {}
    for surface, record in zip(plan["state_surfaces"], records, strict=True):
        group = by_scenario.setdefault(surface["scenario_row_id"], {})
        group[surface["state"]] = record
    if len(by_scenario) != SCENARIO_ROW_COUNT:
        raise NextR5Target5PlanError("Target5 binding scenario coverage drift")
    for scenario_id, group in by_scenario.items():
        if tuple(group) != STATE_IDS:
            raise NextR5Target5PlanError(
                f"Target5 binding four-state coverage drift for {scenario_id}"
            )
        old_query_roots = {
            group[state]["old_query_id_root_sha256"] for state in STATE_IDS
        }
        if len(old_query_roots) != 1:
            raise NextR5Target5PlanError(
                f"Target5 old-query-ID root reuse drift for {scenario_id}"
            )
        new_query_roots = {
            group[state]["new_query_id_root_sha256"] for state in REG1_STATES
        }
        if len(new_query_roots) != 1:
            raise NextR5Target5PlanError(
                f"Target5 new-query-ID root reuse drift for {scenario_id}"
            )
        if (
            group["DA1_REG0"]["fa_state_binding_sha256"]
            != group["DA1_REG1"]["fa_state_binding_sha256"]
        ):
            raise NextR5Target5PlanError(
                f"Target5 DA1 FA state reuse drift for {scenario_id}"
            )


def build_next_r5_target5_binding_receipt(
    plan: Mapping[str, Any], state_bindings: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    """Seal truth-free per-surface query-root and FA-state binding evidence."""

    payload = _state_binding_payload(plan, state_bindings)
    _validate_binding_constraints(plan, payload["state_bindings"])
    payload["binding_receipt_sha256"] = canonical_sha256(payload)
    return _deep_freeze(_json_ready(payload))


def validate_next_r5_target5_binding_receipt(
    plan: Mapping[str, Any], value: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Validate complete same-scene query and DA1 state reuse evidence."""

    validate_next_r5_target5_plan(plan)
    _require_exact_mapping_keys(value, _BINDING_RECEIPT_FIELDS, "Target5 binding receipt")
    if (
        value["schema"] != BINDING_RECEIPT_SCHEMA
        or value["plan_receipt_sha256"] != plan["plan_receipt_sha256"]
    ):
        raise NextR5Target5PlanError("Target5 binding receipt identity drift")
    payload = dict(value)
    receipt = payload.pop("binding_receipt_sha256")
    if not _is_sha256(receipt) or receipt != canonical_sha256(payload):
        raise NextR5Target5PlanError("Target5 binding canonical receipt drift")
    canonical_payload = _state_binding_payload(plan, value["state_bindings"])
    _validate_binding_constraints(plan, canonical_payload["state_bindings"])
    expected = build_next_r5_target5_binding_receipt(plan, value["state_bindings"])
    if _json_ready(value) != _json_ready(expected):
        raise NextR5Target5PlanError("Target5 binding receipt deterministic rebuild drift")
    return _deep_freeze(_json_ready(value))


__all__ = [
    "ARM_IDS",
    "BINDING_RECEIPT_SCHEMA",
    "CANDIDATE_ID",
    "JOB_COUNT",
    "K_SHOT",
    "LEO_SCENARIOS",
    "NEW_CLASS_COUNT",
    "NEGATIVE_PROTOCOL_FLAGS",
    "NextR5Target5PlanError",
    "PREFERRED_SEED",
    "PROHIBITED_COMPONENTS",
    "PROTOCOL_SCHEMA",
    "RECEIVERS",
    "SCENARIO_ROW_COUNT",
    "SCHEMA",
    "STATE_IDS",
    "STATE_NAMES_ZH",
    "STATE_SURFACE_COUNT",
    "build_next_r5_target5_binding_receipt",
    "build_next_r5_target5_plan",
    "canonical_bytes",
    "canonical_sha256",
    "validate_next_r5_target5_binding_receipt",
    "validate_next_r5_target5_plan",
]
