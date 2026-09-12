"""Formal Stage2-B/C SOMP-H matrix and coverage contract.

This module belongs to the offline controller boundary.  It may name legal
receivers and TXs for auditability, but its output must never be mounted in a
Phase2 predictor.  Predictor packages retain only opaque class/sample handles.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from cvsrffi.leo_weak_cache import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT


SCHEMA = "cvs.phase2.somph_formal_matrix.v3"
ARTIFACT_BOUNDARY = "offline_matrix_controller_only_never_mounted_in_phase2"
ADV3B02_CHECKPOINT_SHA256 = (
    "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
)
FORMAL_RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
DEVELOPMENT_SEED = 713101
CONFIRMATION_SEEDS = (713102, 713103, 713104, 713105, 713106)
FORMAL_K_VALUES = (1, 5, 10, 20)
PRIMARY_K_VALUES = (5, 10)
FORMAL_NEW_CLASS_COUNTS = (5, 10, 20)
OFFLINE_AUTHORITY_SUPPORT_POOL_MAX_K = 20
# Backward-compatible import alias.  In this module it never describes
# Phase2-reachable support; each formal row exposes exactly its own k_shot.
SUPPORT_POOL_MAX_K = OFFLINE_AUTHORITY_SUPPORT_POOL_MAX_K
SUPPORT_POOL_MAX_K_SEMANTICS = (
    "offline_authority_candidate_pool_only_not_phase2_reachable"
)
PHASE2_REACHABLE_SUPPORT_POOL_POLICY = "sealed_exact_k_per_registered_class"
NESTED_K_SUPPORT_PACKAGE_POLICY = (
    "k1_k5_are_k10_ordered_prefixes_k20_is_separate_exact_package"
)
KSHOT_REACHABILITY_VIOLATION_STATUS = "PROTOCOL_INVALID_KSHOT_REACHABILITY"
_PHASE2_CONTRACT_ITEMS = tuple(PHASE2_FULL_CONTRACT.items())


def _phase2_contract() -> dict[str, Any]:
    return dict(_PHASE2_CONTRACT_ITEMS)


def _success_criteria() -> dict[str, Any]:
    return {
        "k10_target_old_overall_accuracy_min": 0.92,
        "k10_each_old_class_accuracy_min": 0.88,
        "k10_seen_new_accuracy_min_by_count": {
            "5": 0.92,
            "10": 0.90,
            "20": 0.86,
        },
        "k5_max_drop_from_matched_k10_pp": 3.0,
    }


def _resource_limits() -> dict[str, Any]:
    return {
        "adapter_parameters_max": 80_000,
        "adaptation_epochs_max": 30,
        "dense_query_graph_allowed": False,
        "per_sample_query_decision_required": True,
    }


def _development_lock() -> dict[str, Any]:
    return {
        "receiver": "20-1",
        "seed": DEVELOPMENT_SEED,
        "k_shot": 10,
    }
OLD_TX_IDS = ("14-10", "14-7", "20-15", "20-19", "6-15", "8-20")
NEW_TX_IDS = (
    "1-16",
    "1-18",
    "18-10",
    "14-11",
    "8-3",
    "18-8",
    "10-10",
    "16-19",
    "20-12",
    "4-10",
    "13-14",
    "2-5",
    "1-8",
    "19-13",
    "19-9",
    "3-8",
    "19-8",
    "11-19",
    "2-16",
    "19-6",
)


class SomphFormalMatrixError(ValueError):
    """Raised when a formal Stage2-B/C matrix is incomplete or ambiguous."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _row_id(
    *,
    stage: str,
    registration_state: str,
    receiver: str,
    seed: int,
    k_shot: int,
    new_class_count: int,
) -> str:
    payload = {
        "stage": stage,
        "registration_state": registration_state,
        "receiver": receiver,
        "seed": seed,
        "k_shot": k_shot,
        "new_class_count": new_class_count,
    }
    return "row_" + _sha256_json(payload)


def _pair_id(*, receiver: str, seed: int, k_shot: int, new_class_count: int) -> str:
    payload = {
        "receiver": receiver,
        "seed": seed,
        "k_shot": k_shot,
        "new_class_count": new_class_count,
    }
    return "pair_" + _sha256_json(payload)


def _row(
    *,
    stage: str,
    registration_state: str,
    receiver: str,
    seed: int,
    k_shot: int,
    new_class_count: int,
) -> dict[str, Any]:
    if stage == "stage2b":
        registered_tx_ids = list(OLD_TX_IDS)
    elif registration_state == "before":
        registered_tx_ids = list(OLD_TX_IDS)
    else:
        registered_tx_ids = list(OLD_TX_IDS + NEW_TX_IDS[:new_class_count])
    development_lock = _development_lock()
    selection_eligible = (
        receiver == development_lock["receiver"]
        and seed == development_lock["seed"]
        and k_shot == development_lock["k_shot"]
    )
    row = {
        "row_id": _row_id(
            stage=stage,
            registration_state=registration_state,
            receiver=receiver,
            seed=seed,
            k_shot=k_shot,
            new_class_count=new_class_count,
        ),
        "pair_id": (
            None
            if stage == "stage2b"
            else _pair_id(
                receiver=receiver,
                seed=seed,
                k_shot=k_shot,
                new_class_count=new_class_count,
            )
        ),
        "stage": stage,
        "registration_state": registration_state,
        "receiver": receiver,
        "seed": seed,
        "seed_role": (
            "development" if seed == DEVELOPMENT_SEED else "independent_confirmation"
        ),
        "k_shot": k_shot,
        "k_role": "primary" if k_shot in PRIMARY_K_VALUES else "stress",
        "selection_eligible": selection_eligible,
        "confirmation_aggregate_eligible": seed in CONFIRMATION_SEEDS,
        "new_class_count": new_class_count,
        "registered_class_count": len(registered_tx_ids),
        "registered_tx_ids": registered_tx_ids,
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "offline_authority_support_pool_max_k": (
            OFFLINE_AUTHORITY_SUPPORT_POOL_MAX_K
        ),
        "support_pool_max_k": SUPPORT_POOL_MAX_K,
        "support_pool_max_k_semantics": SUPPORT_POOL_MAX_K_SEMANTICS,
        "reachable_support_pool_max_k": k_shot,
        "reachable_support_count_per_registered_class": k_shot,
        "phase2_reachable_support_pool_policy": (
            PHASE2_REACHABLE_SUPPORT_POOL_POLICY
        ),
        "support_member_allowlist_required": True,
        "support_preopen_per_class_count_validation_required": True,
        "nested_k_support_package_policy": NESTED_K_SUPPORT_PACKAGE_POLICY,
        "nested_k_support_prefix_validation_required": True,
        "kshot_reachability_violation_status": (
            KSHOT_REACHABILITY_VIOLATION_STATUS
        ),
        "support_selection_rule": (
            "offline_rank_then_seal_exact_k_ordered_prefix_per_class"
        ),
        "query_excludes_full_support_pool_max_k": True,
        "cross_scenario_physical_sample_reuse": False,
        "scenario_physical_sample_assignment_policy": (
            "disjoint_preoverlay_tx_day_stratified_v1"
        ),
        "scenario_support_query_physical_id_sets_pairwise_disjoint": True,
        "data_binding_status": "UNBOUND_REQUIREMENT_TEMPLATE",
        "row_manifest_sha256": None,
        "query_physical_roots_sha256_by_scenario": None,
        "support_pool_physical_roots_sha256_by_scenario": None,
        "physical_sample_scenario_assignment_sha256": None,
        "overlay_lineage_receipt_sha256": None,
        "formal_launch_authority": False,
        **_phase2_contract(),
    }
    unsigned = dict(row)
    row["structural_row_sha256"] = _sha256_json(unsigned)
    return row


def build_formal_matrix() -> dict[str, Any]:
    """Build the immutable required row universe before any data binding."""

    rows: list[dict[str, Any]] = []
    seeds = (DEVELOPMENT_SEED, *CONFIRMATION_SEEDS)
    for receiver in FORMAL_RECEIVERS:
        for seed in seeds:
            for k_shot in FORMAL_K_VALUES:
                rows.append(
                    _row(
                        stage="stage2b",
                        registration_state="before",
                        receiver=receiver,
                        seed=seed,
                        k_shot=k_shot,
                        new_class_count=0,
                    )
                )
                for new_class_count in FORMAL_NEW_CLASS_COUNTS:
                    for registration_state in ("before", "after"):
                        rows.append(
                            _row(
                                stage="stage2c",
                                registration_state=registration_state,
                                receiver=receiver,
                                seed=seed,
                                k_shot=k_shot,
                                new_class_count=new_class_count,
                            )
                        )
    payload = {
        "schema": SCHEMA,
        "artifact_boundary": ARTIFACT_BOUNDARY,
        "formal_launch_authority": False,
        "base_model": "ADV3B02",
        "base_checkpoint_sha256": ADV3B02_CHECKPOINT_SHA256,
        "success_criteria": _success_criteria(),
        "resource_limits": _resource_limits(),
        "development_lock": _development_lock(),
        "development_seed": DEVELOPMENT_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "receivers": list(FORMAL_RECEIVERS),
        "k_values": list(FORMAL_K_VALUES),
        "primary_k_values": list(PRIMARY_K_VALUES),
        "new_class_counts": list(FORMAL_NEW_CLASS_COUNTS),
        "offline_authority_support_pool_max_k": (
            OFFLINE_AUTHORITY_SUPPORT_POOL_MAX_K
        ),
        "support_pool_max_k": SUPPORT_POOL_MAX_K,
        "support_pool_max_k_semantics": SUPPORT_POOL_MAX_K_SEMANTICS,
        "phase2_reachable_support_pool_policy": (
            PHASE2_REACHABLE_SUPPORT_POOL_POLICY
        ),
        "support_member_allowlist_required": True,
        "support_preopen_per_class_count_validation_required": True,
        "nested_k_support_package_policy": NESTED_K_SUPPORT_PACKAGE_POLICY,
        "nested_k_support_prefix_validation_required": True,
        "kshot_reachability_violation_status": (
            KSHOT_REACHABILITY_VIOLATION_STATUS
        ),
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "old_tx_ids": list(OLD_TX_IDS),
        "nested_new_tx_ids": list(NEW_TX_IDS),
        "stage2b_row_count": 5 * 6 * 4,
        "stage2c_structural_pair_cell_count": 5 * 6 * 4 * 3,
        "stage2c_state_row_count": 5 * 6 * 4 * 3 * 2,
        "confirmation_stage2b_row_count": 5 * 5 * 4,
        "confirmation_stage2c_structural_pair_cell_count": 5 * 5 * 4 * 3,
        "data_bound_stage2c_pair_count": 0,
        "rows": rows,
        **_phase2_contract(),
    }
    payload["matrix_sha256"] = _sha256_json(payload)
    validate_formal_matrix(payload)
    return payload


TOP_LEVEL_KEYS = {
    "schema",
    "artifact_boundary",
    "formal_launch_authority",
    "base_model",
    "base_checkpoint_sha256",
    "success_criteria",
    "resource_limits",
    "development_lock",
    "development_seed",
    "confirmation_seeds",
    "receivers",
    "k_values",
    "primary_k_values",
    "new_class_counts",
    "offline_authority_support_pool_max_k",
    "support_pool_max_k",
    "support_pool_max_k_semantics",
    "phase2_reachable_support_pool_policy",
    "support_member_allowlist_required",
    "support_preopen_per_class_count_validation_required",
    "nested_k_support_package_policy",
    "nested_k_support_prefix_validation_required",
    "kshot_reachability_violation_status",
    "target_channel_scenarios",
    "old_tx_ids",
    "nested_new_tx_ids",
    "stage2b_row_count",
    "stage2c_structural_pair_cell_count",
    "stage2c_state_row_count",
    "confirmation_stage2b_row_count",
    "confirmation_stage2c_structural_pair_cell_count",
    "data_bound_stage2c_pair_count",
    "rows",
    "matrix_sha256",
    *dict(_PHASE2_CONTRACT_ITEMS).keys(),
}
ROW_KEYS = {
    "row_id",
    "pair_id",
    "stage",
    "registration_state",
    "receiver",
    "seed",
    "seed_role",
    "k_shot",
    "k_role",
    "selection_eligible",
    "confirmation_aggregate_eligible",
    "new_class_count",
    "registered_class_count",
    "registered_tx_ids",
    "target_channel_scenarios",
    "offline_authority_support_pool_max_k",
    "support_pool_max_k",
    "support_pool_max_k_semantics",
    "reachable_support_pool_max_k",
    "reachable_support_count_per_registered_class",
    "phase2_reachable_support_pool_policy",
    "support_member_allowlist_required",
    "support_preopen_per_class_count_validation_required",
    "nested_k_support_package_policy",
    "nested_k_support_prefix_validation_required",
    "kshot_reachability_violation_status",
    "support_selection_rule",
    "query_excludes_full_support_pool_max_k",
    "cross_scenario_physical_sample_reuse",
    "scenario_physical_sample_assignment_policy",
    "scenario_support_query_physical_id_sets_pairwise_disjoint",
    "data_binding_status",
    "row_manifest_sha256",
    "query_physical_roots_sha256_by_scenario",
    "support_pool_physical_roots_sha256_by_scenario",
    "physical_sample_scenario_assignment_sha256",
    "overlay_lineage_receipt_sha256",
    "formal_launch_authority",
    "structural_row_sha256",
    *dict(_PHASE2_CONTRACT_ITEMS).keys(),
}
FORBIDDEN_AUTHORITY_ALIASES = {
    "launch_authority",
    "protocol_status",
    "protocol_valid",
    "status",
    "pass",
    "formal_pass",
}


def _exact_sequence(value: Any, expected: Sequence[Any], *, field: str) -> None:
    if not isinstance(value, list) or value != list(expected):
        raise SomphFormalMatrixError(f"{field} must equal {list(expected)!r}")


def validate_formal_matrix(payload: Mapping[str, Any]) -> None:
    """Reject missing, duplicate, non-nested, or protocol-invalid rows."""

    if set(payload) != TOP_LEVEL_KEYS:
        raise SomphFormalMatrixError("formal matrix exact schema drift")
    if FORBIDDEN_AUTHORITY_ALIASES.intersection(payload):
        raise SomphFormalMatrixError("formal matrix contains authority alias")
    if payload.get("schema") != SCHEMA:
        raise SomphFormalMatrixError("formal matrix schema drift")
    if payload.get("artifact_boundary") != ARTIFACT_BOUNDARY:
        raise SomphFormalMatrixError("formal matrix boundary drift")
    if payload.get("formal_launch_authority") is not False:
        raise SomphFormalMatrixError("unbound formal matrix cannot authorize launch")
    if payload.get("base_model") != "ADV3B02":
        raise SomphFormalMatrixError("formal matrix base model drift")
    if payload.get("base_checkpoint_sha256") != ADV3B02_CHECKPOINT_SHA256:
        raise SomphFormalMatrixError("formal matrix checkpoint drift")
    if payload.get("success_criteria") != _success_criteria():
        raise SomphFormalMatrixError("formal matrix success criteria drift")
    if payload.get("resource_limits") != _resource_limits():
        raise SomphFormalMatrixError("formal matrix resource limits drift")
    if payload.get("development_lock") != _development_lock():
        raise SomphFormalMatrixError("formal matrix development lock drift")
    _exact_sequence(payload.get("receivers"), FORMAL_RECEIVERS, field="receivers")
    _exact_sequence(
        payload.get("confirmation_seeds"),
        CONFIRMATION_SEEDS,
        field="confirmation_seeds",
    )
    _exact_sequence(payload.get("k_values"), FORMAL_K_VALUES, field="k_values")
    _exact_sequence(
        payload.get("primary_k_values"), PRIMARY_K_VALUES, field="primary_k_values"
    )
    _exact_sequence(
        payload.get("new_class_counts"),
        FORMAL_NEW_CLASS_COUNTS,
        field="new_class_counts",
    )
    _exact_sequence(
        payload.get("target_channel_scenarios"),
        FORMAL_LEO_WEAK_SCENARIOS,
        field="target_channel_scenarios",
    )
    _exact_sequence(payload.get("old_tx_ids"), OLD_TX_IDS, field="old_tx_ids")
    _exact_sequence(
        payload.get("nested_new_tx_ids"), NEW_TX_IDS, field="nested_new_tx_ids"
    )
    if payload.get("development_seed") != DEVELOPMENT_SEED:
        raise SomphFormalMatrixError("development seed drift")
    if (
        payload.get("offline_authority_support_pool_max_k")
        != OFFLINE_AUTHORITY_SUPPORT_POOL_MAX_K
        or payload.get("support_pool_max_k")
        != OFFLINE_AUTHORITY_SUPPORT_POOL_MAX_K
    ):
        raise SomphFormalMatrixError(
            "offline authority support candidate pool must be maxK20"
        )
    if payload.get("support_pool_max_k_semantics") != SUPPORT_POOL_MAX_K_SEMANTICS:
        raise SomphFormalMatrixError("support pool maxK semantics drift")
    if (
        payload.get("phase2_reachable_support_pool_policy")
        != PHASE2_REACHABLE_SUPPORT_POOL_POLICY
        or payload.get("support_member_allowlist_required") is not True
        or payload.get("support_preopen_per_class_count_validation_required")
        is not True
        or payload.get("nested_k_support_package_policy")
        != NESTED_K_SUPPORT_PACKAGE_POLICY
        or payload.get("nested_k_support_prefix_validation_required") is not True
        or payload.get("kshot_reachability_violation_status")
        != KSHOT_REACHABILITY_VIOLATION_STATUS
    ):
        raise SomphFormalMatrixError(
            f"{KSHOT_REACHABILITY_VIOLATION_STATUS}: top-level exact-K policy drift"
        )
    if len(set(OLD_TX_IDS)) != len(OLD_TX_IDS):
        raise SomphFormalMatrixError("old TX registry contains duplicates")
    if len(set(NEW_TX_IDS)) != len(NEW_TX_IDS):
        raise SomphFormalMatrixError("new TX registry contains duplicates")
    if set(OLD_TX_IDS).intersection(NEW_TX_IDS):
        raise SomphFormalMatrixError("old/new TX registries overlap")
    for key, expected in _PHASE2_CONTRACT_ITEMS:
        if payload.get(key) != expected:
            raise SomphFormalMatrixError(f"Phase2 contract drift: {key}")

    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise SomphFormalMatrixError("rows must be a list")
    expected = build_required_row_keys()
    observed: dict[tuple[str, str, str, int, int, int], Mapping[str, Any]] = {}
    row_ids: set[str] = set()
    pair_states: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise SomphFormalMatrixError("row must be an object")
        if set(row) != ROW_KEYS:
            raise SomphFormalMatrixError("formal matrix row exact schema drift")
        if FORBIDDEN_AUTHORITY_ALIASES.intersection(row):
            raise SomphFormalMatrixError("formal matrix row contains authority alias")
        for field in (
            "seed",
            "k_shot",
            "new_class_count",
            "registered_class_count",
            "offline_authority_support_pool_max_k",
            "support_pool_max_k",
            "reachable_support_pool_max_k",
            "reachable_support_count_per_registered_class",
        ):
            if (
                not isinstance(row.get(field), int)
                or isinstance(row.get(field), bool)
            ):
                raise SomphFormalMatrixError(f"row integer type drift: {field}")
        key = (
            str(row.get("stage")),
            str(row.get("registration_state")),
            str(row.get("receiver")),
            row["seed"],
            row["k_shot"],
            row["new_class_count"],
        )
        if key in observed:
            raise SomphFormalMatrixError(f"duplicate row: {key}")
        observed[key] = row
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or row_id in row_ids:
            raise SomphFormalMatrixError("row_id missing or duplicate")
        row_ids.add(row_id)
        if row_id != _row_id(
            stage=key[0],
            registration_state=key[1],
            receiver=key[2],
            seed=key[3],
            k_shot=key[4],
            new_class_count=key[5],
        ):
            raise SomphFormalMatrixError("row_id content binding drift")
        expected_pair_id = (
            None
            if key[0] == "stage2b"
            else _pair_id(
                receiver=key[2],
                seed=key[3],
                k_shot=key[4],
                new_class_count=key[5],
            )
        )
        if row.get("pair_id") != expected_pair_id:
            raise SomphFormalMatrixError("row pair_id content binding drift")
        if expected_pair_id is not None:
            pair_states.setdefault(expected_pair_id, set()).add(key[1])
        if row.get("formal_launch_authority") is not False:
            raise SomphFormalMatrixError("unbound row cannot authorize launch")
        if row.get("data_binding_status") != "UNBOUND_REQUIREMENT_TEMPLATE":
            raise SomphFormalMatrixError("requirement row data binding status drift")
        for field in (
            "row_manifest_sha256",
            "query_physical_roots_sha256_by_scenario",
            "support_pool_physical_roots_sha256_by_scenario",
            "physical_sample_scenario_assignment_sha256",
            "overlay_lineage_receipt_sha256",
        ):
            if row.get(field) is not None:
                raise SomphFormalMatrixError(
                    f"unbound requirement row must not claim {field}"
                )
        for contract_key, contract_value in _PHASE2_CONTRACT_ITEMS:
            if row.get(contract_key) != contract_value:
                raise SomphFormalMatrixError(
                    f"row Phase2 contract drift: {contract_key}"
                )
        _exact_sequence(
            row.get("target_channel_scenarios"),
            FORMAL_LEO_WEAK_SCENARIOS,
            field="row.target_channel_scenarios",
        )
        if (
            row.get("offline_authority_support_pool_max_k")
            != OFFLINE_AUTHORITY_SUPPORT_POOL_MAX_K
            or row.get("support_pool_max_k")
            != OFFLINE_AUTHORITY_SUPPORT_POOL_MAX_K
        ):
            raise SomphFormalMatrixError(
                "row offline authority support candidate pool is not maxK20"
            )
        if row.get("support_pool_max_k_semantics") != SUPPORT_POOL_MAX_K_SEMANTICS:
            raise SomphFormalMatrixError("row support pool maxK semantics drift")
        reachable_k = row.get("reachable_support_pool_max_k")
        reachable_count = row.get("reachable_support_count_per_registered_class")
        if reachable_k != key[4] or reachable_count != key[4]:
            raise SomphFormalMatrixError(
                f"{KSHOT_REACHABILITY_VIOLATION_STATUS}: "
                f"row K={key[4]} must expose exactly K support per class, "
                f"got reachable_max={reachable_k}, count={reachable_count}"
            )
        if (
            row.get("phase2_reachable_support_pool_policy")
            != PHASE2_REACHABLE_SUPPORT_POOL_POLICY
            or row.get("support_member_allowlist_required") is not True
            or row.get("support_preopen_per_class_count_validation_required")
            is not True
            or row.get("nested_k_support_package_policy")
            != NESTED_K_SUPPORT_PACKAGE_POLICY
            or row.get("nested_k_support_prefix_validation_required") is not True
            or row.get("kshot_reachability_violation_status")
            != KSHOT_REACHABILITY_VIOLATION_STATUS
        ):
            raise SomphFormalMatrixError(
                f"{KSHOT_REACHABILITY_VIOLATION_STATUS}: row exact-K guard drift"
            )
        if (
            row.get("support_selection_rule")
            != "offline_rank_then_seal_exact_k_ordered_prefix_per_class"
        ):
            raise SomphFormalMatrixError("row support selection rule drift")
        if row.get("query_excludes_full_support_pool_max_k") is not True:
            raise SomphFormalMatrixError("query/support exclusion is not explicit")
        if row.get("cross_scenario_physical_sample_reuse") is not False:
            raise SomphFormalMatrixError("cross-scenario physical sample reuse is enabled")
        if (
            row.get("scenario_physical_sample_assignment_policy")
            != "disjoint_preoverlay_tx_day_stratified_v1"
        ):
            raise SomphFormalMatrixError("scenario physical assignment policy drift")
        if (
            row.get("scenario_support_query_physical_id_sets_pairwise_disjoint")
            is not True
        ):
            raise SomphFormalMatrixError("scenario physical ID disjointness is absent")
        expected_seed_role = (
            "development"
            if key[3] == DEVELOPMENT_SEED
            else "independent_confirmation"
        )
        if row.get("seed_role") != expected_seed_role:
            raise SomphFormalMatrixError("row seed role drift")
        expected_k_role = "primary" if key[4] in PRIMARY_K_VALUES else "stress"
        if row.get("k_role") != expected_k_role:
            raise SomphFormalMatrixError("row K role drift")
        development_lock = _development_lock()
        expected_selection = (
            key[2] == development_lock["receiver"]
            and key[3] == development_lock["seed"]
            and key[4] == development_lock["k_shot"]
        )
        if row.get("selection_eligible") is not expected_selection:
            raise SomphFormalMatrixError("row selection eligibility drift")
        if row.get("confirmation_aggregate_eligible") is not (
            key[3] in CONFIRMATION_SEEDS
        ):
            raise SomphFormalMatrixError("row confirmation eligibility drift")
        registered = row.get("registered_tx_ids")
        if key[0] == "stage2b" or key[1] == "before":
            expected_registry = list(OLD_TX_IDS)
        else:
            expected_registry = list(OLD_TX_IDS + NEW_TX_IDS[: key[5]])
        if registered != expected_registry:
            raise SomphFormalMatrixError("registered TX registry is not nested/exact")
        if row.get("registered_class_count") != len(expected_registry):
            raise SomphFormalMatrixError("registered class count drift")
        unsigned_row = dict(row)
        claimed_row_sha256 = unsigned_row.pop("structural_row_sha256")
        if claimed_row_sha256 != _sha256_json(unsigned_row):
            raise SomphFormalMatrixError("structural row digest mismatch")
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing or extra:
        raise SomphFormalMatrixError(
            f"formal matrix coverage mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
    if any(states != {"before", "after"} for states in pair_states.values()):
        raise SomphFormalMatrixError("Stage2-C structural pair is incomplete")
    expected_counts = {
        "stage2b_row_count": sum(key[0] == "stage2b" for key in observed),
        "stage2c_structural_pair_cell_count": len(pair_states),
        "stage2c_state_row_count": sum(key[0] == "stage2c" for key in observed),
        "confirmation_stage2b_row_count": sum(
            key[0] == "stage2b" and key[3] in CONFIRMATION_SEEDS
            for key in observed
        ),
        "confirmation_stage2c_structural_pair_cell_count": len(
            {
                row["pair_id"]
                for row in rows
                if row["stage"] == "stage2c"
                and row["confirmation_aggregate_eligible"] is True
            }
        ),
        "data_bound_stage2c_pair_count": 0,
    }
    for field, value in expected_counts.items():
        if payload.get(field) != value:
            raise SomphFormalMatrixError(f"formal matrix count drift: {field}")
    unsigned = dict(payload)
    claimed_sha256 = unsigned.pop("matrix_sha256", None)
    if claimed_sha256 != _sha256_json(unsigned):
        raise SomphFormalMatrixError("formal matrix digest mismatch")


def build_required_row_keys() -> set[tuple[str, str, str, int, int, int]]:
    seeds = (DEVELOPMENT_SEED, *CONFIRMATION_SEEDS)
    keys: set[tuple[str, str, str, int, int, int]] = set()
    for receiver in FORMAL_RECEIVERS:
        for seed in seeds:
            for k_shot in FORMAL_K_VALUES:
                keys.add(("stage2b", "before", receiver, seed, k_shot, 0))
                for new_class_count in FORMAL_NEW_CLASS_COUNTS:
                    keys.add(
                        (
                            "stage2c",
                            "before",
                            receiver,
                            seed,
                            k_shot,
                            new_class_count,
                        )
                    )
                    keys.add(
                        (
                            "stage2c",
                            "after",
                            receiver,
                            seed,
                            k_shot,
                            new_class_count,
                        )
                    )
    return keys
