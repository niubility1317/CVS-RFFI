"""Truth-free D92 locator projection for the frozen D106 Target25 matrix.

This module does not open received IQ, rebuild a dataset, validate predictions,
or read truth/score material.  It binds already sealed D92 package and split
locators to the structural identities returned by
``freeze_d106_matrix_protocol`` and publishes an immutable D106 plan/context.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping
from .stage2_d106_matrix_protocol import (
    LEO_SCENARIOS,
    MATCHED_ARM_PAIR_COUNT,
    OUTER_JOB_COUNT,
    PROTOCOL_SCHEMA,
    RECEIVERS,
    SCENARIO_ROW_COUNT,
    STATE_SURFACE_COUNT,
    TARGET25_SEED,
    TARGET25_SLICES,
    canonical_sha256,
    freeze_d106_matrix_protocol,
)


D106_INDEX_SCHEMA = "cvs.phase2.d106.target25_d92_matrix_locator.v1"
LEGACY_D92_INDEX_SCHEMA = "cvs.phase2.d105.target25_d92_matrix_index.v1"
D106_SPLIT_LOCATOR_SCHEMA = "cvs.phase2.d106.target25_d92_split_locator.v1"
LEGACY_D92_SPLIT_SCHEMA = "cvs.phase2.d105.target25.v1.plan_manifest"
PLAN_SCHEMA = "cvs.phase2.d106.target25_input_plan.v1"
CONTEXT_SCHEMA = "cvs.phase2.d106.target25_input_context.v1"
PREPARE_RECEIPT_SCHEMA = "cvs.phase2.d106.target25_input_receipt.v1"
KCR_ROUTE_LOCK_SCHEMA = "cvs.phase2.d106.k_conditioned_route_lock.v1"

_PACKAGE_NAMES = (
    "before_enrollment",
    "before_apply",
    "after_enrollment",
    "after_apply",
)
_PACKAGE_REF_FIELDS = {
    "package_root",
    "detached_seal_path",
    "expected_seal_sha256",
    "formal_policy_path",
    "formal_policy_authorization_path",
    "signed_policy_authorization_envelope_path",
    "expected_signed_policy_authorization_envelope_sha256",
}
_AUTHORITY_FIELDS = {
    "receiver",
    "authority_bundle_root",
    "expected_authority_commit_sha256",
    "cache_set_manifest_path",
}
_D106_INDEX_FIELDS = {
    "schema",
    "seed",
    "claim_scope",
    "formal_launch_authority",
    "authorities",
    "rows",
}
_LEGACY_INDEX_FIELDS = _D106_INDEX_FIELDS | {
    "phase1_bundle_dir",
    "checkpoint_path",
    "candidate_runtime_manifest_path",
    "candidate_method_lock_path",
    "feature_batch_size",
    "score_chunk_size",
}
_D106_ROW_FIELDS = {"receiver", "k_shot", "new_count", *_PACKAGE_NAMES}
_LEGACY_ROW_FIELDS = _D106_ROW_FIELDS | {"qknn_lock"}
_D106_SPLIT_FIELDS = {
    "schema",
    "protocol_schema",
    "phase2_data_status",
    "seed",
    "rows",
    "locator_receipt_sha256",
}
_NATIVE_SPLIT_ROW_FIELDS = {"receiver", "k_shot", "new_count", "scenarios"}
_LEGACY_SPLIT_ROW_FIELDS = {
    "row_id",
    "receiver",
    "k_shot",
    "new_count",
    "scenarios",
}
_SCENARIO_FIELDS = {"scenario", "before", "after"}
_NATIVE_STATE_FIELDS = {
    "protocol_schema",
    "phase2_data_status",
    "capsule_id",
    "split_id",
    "authority_receipt_sha256",
    "authority_envelope_sha256",
    "support_physical_ids",
    "query_physical_ids",
    "registered_classes",
    "old_classes",
    "new_classes",
    "prediction_context_sha256",
}
_LEGACY_STATE_FIELDS = _NATIVE_STATE_FIELDS | {
    "stage",
    "registration_state",
    "data_feature_runtime_sha256",
    "data_materialization_lock_sha256",
    "d105_candidate_runtime_manifest_sha256",
    "d105_candidate_method_lock_sha256",
    "single_leo_observation",
    "clean_source_runtime_access",
    "query_fit_access",
    "query_decision_policy",
    "support_physical_root_sha256",
    "query_physical_root_sha256",
}
_LEGACY_PLAN_PAYLOAD_FIELDS = {
    "schema",
    "seed",
    "claim_scope",
    "formal_launch_authority",
    "authority_envelope_root_sha256",
    "data_feature_runtime_sha256",
    "data_materialization_lock_sha256",
    "d105_candidate_runtime_manifest_sha256",
    "d105_candidate_method_lock_sha256",
    "arms",
    "leo_scenarios",
    "target25_slices",
    "rows",
}
_KCR_ROUTE_LOCK_FIELDS = {
    "schema",
    "candidate_id",
    "route_by_k",
    "query_truth_access",
    "query_role_access",
    "query_fit_access",
    "query_update_access",
    "query_selection",
}


class D106Target25InputError(ValueError):
    """Raised when a locator cannot close on the frozen D106 matrix."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise D106Target25InputError(f"{name} must be a lowercase SHA256")
    return text


def _regular_path(value: Any, name: str, *, directory: bool = False) -> Path:
    source = Path(str(value))
    if not source.is_absolute() or source.is_symlink() or not source.exists():
        raise D106Target25InputError(
            f"{name} must be an existing absolute non-symlink path"
        )
    source = source.resolve(strict=True)
    if (directory and not source.is_dir()) or (
        not directory and not source.is_file()
    ):
        raise D106Target25InputError(f"{name} has the wrong file type")
    return source


def _read_json(path: Path, name: str, expected_sha256: str) -> dict[str, Any]:
    source = _regular_path(path, name)
    if _sha256_file(source) != _sha(expected_sha256, f"expected {name} SHA256"):
        raise D106Target25InputError(f"{name} SHA mismatch")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise D106Target25InputError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise D106Target25InputError(f"{name} must contain an object")
    return value


def _unique_texts(value: Any, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise D106Target25InputError(f"{name} must be a list")
    result = tuple(str(item) for item in value)
    if any(not item or item.strip() != item for item in result):
        raise D106Target25InputError(f"{name} contains invalid text")
    if len(set(result)) != len(result):
        raise D106Target25InputError(f"{name} contains duplicates")
    if not allow_empty and not result:
        raise D106Target25InputError(f"{name} must not be empty")
    return result


def _asset(path: Path, expected_sha256: str, name: str) -> dict[str, str]:
    source = _regular_path(path, name)
    expected = _sha(expected_sha256, f"expected {name} SHA256")
    if _sha256_file(source) != expected:
        raise D106Target25InputError(f"{name} SHA mismatch")
    return {"path": str(source), "sha256": expected}


def _kcr_route_lock(path: Path, expected_sha256: str) -> dict[str, Any]:
    source = _regular_path(path, "KCR route lock")
    expected = _sha(expected_sha256, "expected KCR route-lock SHA256")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise D106Target25InputError("KCR route-lock SHA mismatch")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106Target25InputError("KCR route lock is not valid UTF-8 JSON") from error
    if (
        not isinstance(document, dict)
        or raw not in {_canonical_bytes(document), _canonical_bytes(document) + b"\n"}
        or set(document) != _KCR_ROUTE_LOCK_FIELDS
        or document.get("schema") != KCR_ROUTE_LOCK_SCHEMA
        or document.get("candidate_id") != "D106-KCR/r1"
        or document.get("route_by_k")
        != {"1": "M_DA", "5": "M0", "10": "M_HEAD"}
        or any(
            document.get(name) is not False
            for name in (
                "query_truth_access",
                "query_role_access",
                "query_fit_access",
                "query_update_access",
                "query_selection",
            )
        )
    ):
        raise D106Target25InputError("KCR route-lock canonical schema/route drift")
    return {
        "path": str(source),
        "sha256": expected,
        "candidate_id": document["candidate_id"],
        "route_by_k": document["route_by_k"],
    }


def _sealed_package_ref(value: Any, name: str) -> dict[str, str]:
    """Validate only the sealed locator plane; never open package data."""

    if not isinstance(value, Mapping) or set(value) != _PACKAGE_REF_FIELDS:
        raise D106Target25InputError(f"{name} package reference closure drift")
    root = _regular_path(value["package_root"], f"{name}.package_root", directory=True)
    seal = _regular_path(value["detached_seal_path"], f"{name}.detached_seal_path")
    expected_seal = _sha(
        value["expected_seal_sha256"], f"{name}.expected_seal_sha256"
    )
    if _sha256_file(seal) != expected_seal:
        raise D106Target25InputError(f"{name} detached seal SHA mismatch")
    policy = _regular_path(value["formal_policy_path"], f"{name}.formal_policy_path")
    authorization = _regular_path(
        value["formal_policy_authorization_path"],
        f"{name}.formal_policy_authorization_path",
    )
    envelope = _regular_path(
        value["signed_policy_authorization_envelope_path"],
        f"{name}.signed_policy_authorization_envelope_path",
    )
    expected_envelope = _sha(
        value["expected_signed_policy_authorization_envelope_sha256"],
        f"{name}.expected_signed_policy_authorization_envelope_sha256",
    )
    if _sha256_file(envelope) != expected_envelope:
        raise D106Target25InputError(f"{name} signed envelope SHA mismatch")
    return {
        "package_root": str(root),
        "detached_seal_path": str(seal),
        "expected_seal_sha256": expected_seal,
        "formal_policy_path": str(policy),
        "formal_policy_authorization_path": str(authorization),
        "signed_policy_authorization_envelope_path": str(envelope),
        "expected_signed_policy_authorization_envelope_sha256": expected_envelope,
    }


def _authority_locator(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _AUTHORITY_FIELDS:
        raise D106Target25InputError("D92 authority locator closure drift")
    return {
        "receiver": str(value["receiver"]),
        "authority_bundle_root": str(
            _regular_path(
                value["authority_bundle_root"],
                "authority_bundle_root",
                directory=True,
            )
        ),
        "expected_authority_commit_sha256": _sha(
            value["expected_authority_commit_sha256"],
            "expected_authority_commit_sha256",
        ),
        "cache_set_manifest_path": str(
            _regular_path(value["cache_set_manifest_path"], "cache_set_manifest_path")
        ),
    }


def _project_matrix_index(index: Mapping[str, Any]) -> tuple[
    str,
    bool,
    dict[str, dict[str, str]],
    dict[tuple[str, int, int], dict[str, dict[str, str]]],
]:
    schema = index.get("schema")
    expected_fields = (
        _D106_INDEX_FIELDS if schema == D106_INDEX_SCHEMA else _LEGACY_INDEX_FIELDS
    )
    if schema not in (D106_INDEX_SCHEMA, LEGACY_D92_INDEX_SCHEMA) or set(index) != expected_fields:
        raise D106Target25InputError("D92 matrix locator schema closure drift")
    if index.get("seed") != TARGET25_SEED:
        raise D106Target25InputError("D92 matrix locator seed drift")
    claim_scope = str(index.get("claim_scope"))
    formal = index.get("formal_launch_authority")
    if not claim_scope or type(formal) is not bool:
        raise D106Target25InputError("D92 matrix locator claim binding drift")
    authority_values = index.get("authorities")
    if not isinstance(authority_values, list) or len(authority_values) != len(RECEIVERS):
        raise D106Target25InputError("D92 matrix locator must contain five authorities")
    authorities: dict[str, dict[str, str]] = {}
    for value in authority_values:
        locator = _authority_locator(value)
        receiver = locator["receiver"]
        if receiver in authorities:
            raise D106Target25InputError("duplicate D92 authority receiver")
        authorities[receiver] = locator
    if tuple(authorities) != RECEIVERS:
        raise D106Target25InputError("D92 authority receiver order/coverage drift")

    rows = index.get("rows")
    if not isinstance(rows, list) or len(rows) != OUTER_JOB_COUNT:
        raise D106Target25InputError("D92 matrix locator must contain 25 rows")
    expected_keys = [
        (receiver, k_shot, new_count)
        for receiver in RECEIVERS
        for k_shot, new_count in TARGET25_SLICES
    ]
    projected: dict[tuple[str, int, int], dict[str, dict[str, str]]] = {}
    actual_keys: list[tuple[str, int, int]] = []
    row_fields = _D106_ROW_FIELDS if schema == D106_INDEX_SCHEMA else _LEGACY_ROW_FIELDS
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != row_fields:
            raise D106Target25InputError("D92 package row locator closure drift")
        key = (str(row["receiver"]), row["k_shot"], row["new_count"])
        actual_keys.append(key)
        try:
            projected[key] = {
                package_name: _sealed_package_ref(row[package_name], package_name)
                for package_name in _PACKAGE_NAMES
            }
        except Exception as error:
            raise D106Target25InputError("D92 sealed package locator drift") from error
    if actual_keys != expected_keys or len(projected) != OUTER_JOB_COUNT:
        raise D106Target25InputError("D92 matrix row order/coverage drift")
    return claim_scope, formal, authorities, projected


def _project_state(value: Any, state: str, *, legacy: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise D106Target25InputError("D92 split state must be an object")
    expected_fields = _LEGACY_STATE_FIELDS if legacy else _NATIVE_STATE_FIELDS
    if set(value) != expected_fields:
        raise D106Target25InputError("D92 split-state locator closure drift")
    protocol = value.get("protocol_schema")
    status = value.get("phase2_data_status")
    if protocol != PROTOCOL_SCHEMA or status != "VALIDATED_ONCE":
        raise D106Target25InputError("D92 split-state protocol/status drift")
    support = _unique_texts(value.get("support_physical_ids"), "support physical IDs")
    query = _unique_texts(value.get("query_physical_ids"), "query physical IDs")
    if set(support).intersection(query):
        raise D106Target25InputError("support/query physical IDs are not disjoint")
    registered = _unique_texts(value.get("registered_classes"), "registered classes")
    old = _unique_texts(value.get("old_classes"), "old classes")
    new = _unique_texts(value.get("new_classes"), "new classes", allow_empty=True)
    if registered != old + new or set(old).intersection(new):
        raise D106Target25InputError("registered old/new lifecycle union drift")
    if (state == "before" and new) or (state == "after" and not new):
        raise D106Target25InputError("before/after registry lifecycle drift")
    expected_registration = (
        "BEFORE_REGISTRATION" if state == "before" else "AFTER_REGISTRATION"
    )
    if "registration_state" in value and value["registration_state"] != expected_registration:
        raise D106Target25InputError("registration-state locator drift")
    support_root = canonical_sha256(sorted(support))
    query_root = canonical_sha256(sorted(query))
    if legacy:
        expected_stage = "S_B" if state == "before" else "S_C"
        if (
            value.get("stage") != expected_stage
            or value.get("single_leo_observation") is not True
            or value.get("clean_source_runtime_access") is not False
            or value.get("query_fit_access") is not False
            or value.get("query_decision_policy")
            != "per_sample_all_registered_classes"
            or value.get("support_physical_root_sha256") != support_root
            or value.get("query_physical_root_sha256") != query_root
        ):
            raise D106Target25InputError("legacy split-state lifecycle/root drift")
        for name in (
            "data_feature_runtime_sha256",
            "data_materialization_lock_sha256",
            "d105_candidate_runtime_manifest_sha256",
            "d105_candidate_method_lock_sha256",
        ):
            _sha(value.get(name), name)
    return {
        "state": state,
        "registration_state": expected_registration,
        "protocol_schema": protocol,
        "phase2_data_status": status,
        "capsule_id": _sha(value.get("capsule_id"), "capsule_id"),
        "split_id": _sha(value.get("split_id"), "split_id"),
        "authority_receipt_sha256": _sha(
            value.get("authority_receipt_sha256"), "authority_receipt_sha256"
        ),
        "authority_envelope_sha256": _sha(
            value.get("authority_envelope_sha256"), "authority_envelope_sha256"
        ),
        "support_physical_ids": support,
        "support_physical_root_sha256": support_root,
        "query_physical_ids": query,
        "query_physical_root_sha256": query_root,
        "registered_classes": registered,
        "old_classes": old,
        "new_classes": new,
        "prediction_context_sha256": _sha(
            value.get("prediction_context_sha256"), "prediction_context_sha256"
        ),
    }


def _split_rows(locator: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], bool]:
    schema = locator.get("schema")
    if schema == D106_SPLIT_LOCATOR_SCHEMA:
        if set(locator) != _D106_SPLIT_FIELDS:
            raise D106Target25InputError("D106 split locator field closure drift")
        payload_without_receipt = {
            key: value for key, value in locator.items() if key != "locator_receipt_sha256"
        }
        if locator.get("locator_receipt_sha256") != canonical_sha256(payload_without_receipt):
            raise D106Target25InputError("D106 split locator receipt drift")
        if (
            locator.get("protocol_schema") != PROTOCOL_SCHEMA
            or locator.get("phase2_data_status") != "VALIDATED_ONCE"
            or locator.get("seed") != TARGET25_SEED
        ):
            raise D106Target25InputError("D106 split locator protocol/status/seed drift")
        rows = locator.get("rows")
        legacy = False
    elif schema == LEGACY_D92_SPLIT_SCHEMA:
        expected = {
            "schema",
            "plan_payload",
            "candidate_identity_sources",
            "plan_receipt_sha256",
            "plan_manifest_receipt_sha256",
        }
        if set(locator) != expected:
            raise D106Target25InputError("legacy D92 split locator closure drift")
        without_receipt = {
            key: value
            for key, value in locator.items()
            if key != "plan_manifest_receipt_sha256"
        }
        if locator.get("plan_manifest_receipt_sha256") != canonical_sha256(without_receipt):
            raise D106Target25InputError("legacy D92 split locator receipt drift")
        payload = locator.get("plan_payload")
        if (
            not isinstance(payload, Mapping)
            or set(payload) != _LEGACY_PLAN_PAYLOAD_FIELDS
            or payload.get("schema") != "cvs.phase2.d105.target25.v1.plan"
            or payload.get("seed") != TARGET25_SEED
            or payload.get("arms") != ["M0", "M_DA", "M_HEAD", "M_JOINT"]
            or payload.get("leo_scenarios") != list(LEO_SCENARIOS)
            or payload.get("target25_slices")
            != [list(value) for value in TARGET25_SLICES]
            or not isinstance(locator.get("candidate_identity_sources"), Mapping)
            or set(locator["candidate_identity_sources"])
            != {
                "candidate_runtime_manifest_path",
                "candidate_method_lock_path",
            }
        ):
            raise D106Target25InputError("legacy D92 split locator seed drift")
        rows = payload.get("rows")
        legacy = True
    else:
        raise D106Target25InputError("unsupported D92 split locator schema")
    if not isinstance(rows, list) or len(rows) != OUTER_JOB_COUNT:
        raise D106Target25InputError("D92 split locator must contain 25 rows")
    return rows, legacy


def _project_split_locator(locator: Mapping[str, Any]) -> dict[
    tuple[str, int, int], dict[str, dict[str, dict[str, Any]]]
]:
    rows, legacy = _split_rows(locator)
    result: dict[tuple[str, int, int], dict[str, dict[str, dict[str, Any]]]] = {}
    expected_keys = [
        (receiver, k_shot, new_count)
        for receiver in RECEIVERS
        for k_shot, new_count in TARGET25_SLICES
    ]
    actual_keys: list[tuple[str, int, int]] = []
    expected_row_fields = (
        _LEGACY_SPLIT_ROW_FIELDS if legacy else _NATIVE_SPLIT_ROW_FIELDS
    )
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected_row_fields:
            raise D106Target25InputError("D92 split-row locator closure drift")
        key = (str(row["receiver"]), row["k_shot"], row["new_count"])
        actual_keys.append(key)
        scenarios = row.get("scenarios")
        if not isinstance(scenarios, list) or [
            item.get("scenario") if isinstance(item, Mapping) else None
            for item in scenarios
        ] != list(LEO_SCENARIOS):
            raise D106Target25InputError("D92 split scenario order/coverage drift")
        projected_scenarios: dict[str, dict[str, dict[str, Any]]] = {}
        for item in scenarios:
            if not isinstance(item, Mapping) or set(item) != _SCENARIO_FIELDS:
                raise D106Target25InputError("D92 split scenario closure drift")
            scenario = str(item["scenario"])
            projected_scenarios[scenario] = {
                "before": _project_state(item["before"], "before", legacy=legacy),
                "after": _project_state(item["after"], "after", legacy=legacy),
            }
        result[key] = projected_scenarios
    if actual_keys != expected_keys or len(result) != OUTER_JOB_COUNT:
        raise D106Target25InputError("D92 split row order/coverage drift")
    _validate_split_semantics(result)
    return result


def _validate_split_semantics(
    rows: Mapping[tuple[str, int, int], Mapping[str, Mapping[str, Mapping[str, Any]]]]
) -> None:
    for (receiver, k_shot, new_count), scenarios in rows.items():
        for scenario in LEO_SCENARIOS:
            before = scenarios[scenario]["before"]
            after = scenarios[scenario]["after"]
            if (
                before["old_classes"] != after["old_classes"]
                or before["registered_classes"] != before["old_classes"]
                or len(after["new_classes"]) != new_count
            ):
                raise D106Target25InputError("D92 before/after registry binding drift")
            for state in (before, after):
                if len(state["support_physical_ids"]) != k_shot * len(
                    state["registered_classes"]
                ):
                    raise D106Target25InputError("support does not have exact K-shot coverage")
        if set(scenarios) != set(LEO_SCENARIOS):
            raise D106Target25InputError("D92 scenario coverage drift")
    for receiver in RECEIVERS:
        short = rows[(receiver, 5, 20)]
        long = rows[(receiver, 10, 20)]
        for scenario in LEO_SCENARIOS:
            for state_name in ("before", "after"):
                left = short[scenario][state_name]
                right = long[scenario][state_name]
                if (
                    left["capsule_id"] != right["capsule_id"]
                    or left["registered_classes"] != right["registered_classes"]
                    or not set(left["support_physical_ids"]).issubset(
                        right["support_physical_ids"]
                    )
                    or left["query_physical_root_sha256"]
                    != right["query_physical_root_sha256"]
                ):
                    raise D106Target25InputError(
                        "K5 support must be a K10 subset with the same query root"
                    )


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output already exists: {path}")
    raw = _canonical_bytes(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, stat.S_IREAD)
    return hashlib.sha256(raw).hexdigest()


def prepare_d106_target25_inputs(
    *,
    matrix_index_path: Path,
    expected_matrix_index_sha256: str,
    split_locator_path: Path,
    expected_split_locator_sha256: str,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    rdce_wire_path: Path,
    expected_rdce_wire_sha256: str,
    rdce_lock_path: Path,
    expected_rdce_lock_sha256: str,
    rcmr_lock_path: Path,
    expected_rcmr_lock_sha256: str,
    kcr_route_lock_path: Path,
    expected_kcr_route_lock_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Publish one immutable D106 Target25 input plan/context/receipt."""

    matrix_source = _regular_path(matrix_index_path, "D92 matrix locator")
    split_source = _regular_path(split_locator_path, "D92 split locator")
    matrix_sha = _sha(expected_matrix_index_sha256, "expected matrix locator SHA256")
    split_sha = _sha(expected_split_locator_sha256, "expected split locator SHA256")
    index = _read_json(matrix_source, "D92 matrix locator", matrix_sha)
    split_locator = _read_json(split_source, "D92 split locator", split_sha)
    claim_scope, formal, authorities, package_rows = _project_matrix_index(index)
    split_rows = _project_split_locator(split_locator)
    assets = {
        "checkpoint": _asset(checkpoint_path, expected_checkpoint_sha256, "checkpoint"),
        "rdce_wire": _asset(rdce_wire_path, expected_rdce_wire_sha256, "RDCE wire"),
        "rdce_lock": _asset(rdce_lock_path, expected_rdce_lock_sha256, "RDCE lock"),
        "rcmr_lock": _asset(rcmr_lock_path, expected_rcmr_lock_sha256, "RCMR lock"),
        "kcr_route_lock": _kcr_route_lock(
            kcr_route_lock_path, expected_kcr_route_lock_sha256
        ),
    }
    matrix = freeze_d106_matrix_protocol()
    plan_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    for job in matrix.jobs:
        key = (job.receiver, job.k_shot, job.new_count)
        scenario_plan_rows: list[dict[str, Any]] = []
        scenario_context_rows: list[dict[str, Any]] = []
        for scenario_row in (
            item for item in matrix.scenario_rows if item.job_id == job.job_id
        ):
            state_plans: list[dict[str, Any]] = []
            state_contexts: list[dict[str, Any]] = []
            for state_name in ("before", "after"):
                source = split_rows[key][scenario_row.scenario][state_name]
                support_package = (
                    "before_enrollment" if state_name == "before" else "after_enrollment"
                )
                query_package = "before_apply" if state_name == "before" else "after_apply"
                state_identity = {
                    key_name: source[key_name]
                    for key_name in (
                        "state",
                        "registration_state",
                        "protocol_schema",
                        "phase2_data_status",
                        "capsule_id",
                        "split_id",
                        "authority_receipt_sha256",
                        "authority_envelope_sha256",
                        "support_physical_root_sha256",
                        "query_physical_root_sha256",
                        "registered_classes",
                        "old_classes",
                        "new_classes",
                        "prediction_context_sha256",
                    )
                }
                state_identity.update(
                    {
                        "scenario": scenario_row.scenario,
                        "k_shot": job.k_shot,
                        "new_count": job.new_count,
                        "support_received_iq_seal_sha256": package_rows[key][support_package][
                            "expected_seal_sha256"
                        ],
                        "query_received_iq_seal_sha256": package_rows[key][query_package][
                            "expected_seal_sha256"
                        ],
                    }
                )
                state_identity["state_input_receipt_sha256"] = canonical_sha256(
                    state_identity
                )
                state_plans.append(state_identity)
                state_contexts.append(
                    {
                        **state_identity,
                        "support_received_iq_ref": package_rows[key][support_package],
                        "query_received_iq_ref": package_rows[key][query_package],
                    }
                )
            scenario_plan_rows.append(
                {
                    "scenario_row_id": scenario_row.scenario_row_id,
                    "scenario": scenario_row.scenario,
                    "states": state_plans,
                }
            )
            scenario_context_rows.append(
                {
                    "scenario_row_id": scenario_row.scenario_row_id,
                    "scenario": scenario_row.scenario,
                    "states": state_contexts,
                }
            )
        row_base = {
            "job_id": job.job_id,
            "receiver": job.receiver,
            "seed": job.seed,
            "k_shot": job.k_shot,
            "new_count": job.new_count,
        }
        plan_rows.append({**row_base, "scenarios": scenario_plan_rows})
        context_rows.append(
            {
                **row_base,
                "authority_locator": authorities[job.receiver],
                "scenarios": scenario_context_rows,
            }
        )

    identity = {
        "matrix_receipt_sha256": matrix.matrix_receipt_sha256,
        "matrix_index_sha256": matrix_sha,
        "split_locator_sha256": split_sha,
        "assets": assets,
    }
    plan_document: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "claim_scope": claim_scope,
        "formal_launch_authority": formal,
        "identity": identity,
        "matrix_protocol": matrix.receipt_payload(),
        "rows": plan_rows,
    }
    plan_document["plan_receipt_sha256"] = canonical_sha256(plan_document)
    context_document: dict[str, Any] = {
        "schema": CONTEXT_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "plan_receipt_sha256": plan_document["plan_receipt_sha256"],
        "identity": identity,
        "rows": context_rows,
    }
    context_document["context_receipt_sha256"] = canonical_sha256(context_document)

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable prepare output already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise D106Target25InputError("unsafe prepare output parent")
    destination.mkdir()
    plan_path = destination / "target25_plan.json"
    context_path = destination / "target25_context.json"
    plan_file_sha = _write_json_new(plan_path, plan_document)
    context_file_sha = _write_json_new(context_path, context_document)
    receipt: dict[str, Any] = {
        "schema": PREPARE_RECEIPT_SCHEMA,
        "status": "TARGET25_INPUTS_PREPARED",
        "promotable": False,
        "matrix_receipt_sha256": matrix.matrix_receipt_sha256,
        "matrix_index_sha256": matrix_sha,
        "split_locator_sha256": split_sha,
        "plan_receipt_sha256": plan_document["plan_receipt_sha256"],
        "context_receipt_sha256": context_document["context_receipt_sha256"],
        "plan_file_sha256": plan_file_sha,
        "context_file_sha256": context_file_sha,
        "outer_job_count": OUTER_JOB_COUNT,
        "scenario_row_count": SCENARIO_ROW_COUNT,
        "matched_arm_pair_count": MATCHED_ARM_PAIR_COUNT,
        "state_surface_count": STATE_SURFACE_COUNT,
    }
    receipt["prepare_receipt_sha256"] = canonical_sha256(receipt)
    receipt_path = destination / "prepare_receipt.json"
    receipt_file_sha = _write_json_new(receipt_path, receipt)
    return {
        **receipt,
        "plan_manifest": str(plan_path),
        "context_manifest": str(context_path),
        "prepare_receipt": str(receipt_path),
        "prepare_receipt_file_sha256": receipt_file_sha,
    }


__all__ = [
    "CONTEXT_SCHEMA",
    "D106_INDEX_SCHEMA",
    "D106_SPLIT_LOCATOR_SCHEMA",
    "D106Target25InputError",
    "KCR_ROUTE_LOCK_SCHEMA",
    "PLAN_SCHEMA",
    "PREPARE_RECEIPT_SCHEMA",
    "prepare_d106_target25_inputs",
]
