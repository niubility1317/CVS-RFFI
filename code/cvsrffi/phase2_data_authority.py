"""Method-free, metadata-only Phase2 data authority producer.

The producer deliberately does not open support or query payload members.  It
binds an already-built predictor manifest/seal, its offline build audit, and a
separately pinned validation-COMMIT metadata record.  Missing upstream facts
fail closed: this module never reconstructs IQ, selects samples, or upgrades a
report assertion into ``VALIDATED_ONCE`` evidence.

The emitted artifact is intentionally unsigned.  Supplying an expected SHA256
only pins bytes; it cannot create formal authority.  A future external SOMP-H
authority service may wrap the exact canonical artifact bytes and SHA without
this producer reading a signing key.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cvsrffi.phase2_runtime_contract import (
    PHASE2_FULL_CONTRACT,
    PHASE2_SINGLE_OBSERVATION_CONTRACT,
)
from cvsrffi.stage2_predictor_bundle import (
    FORMAL_LEO_WEAK_SCENARIOS,
    MANIFEST_REQUIRED_KEYS,
    MEMBER_DESCRIPTOR_REQUIRED_KEYS,
    PREDICTOR_INPUT_STAGE,
    PREDICTOR_PACKAGE_MANIFEST_SCHEMA,
    PREDICTOR_PACKAGE_SEAL_SCHEMA,
    QUERY_NPZ_MEMBERS,
    QUERY_SCHEMA,
    SEAL_REQUIRED_KEYS,
    SUPPORT_NPZ_MEMBERS,
    SUPPORT_SCHEMA,
    canonical_json_bytes,
    package_root_sha256,
)


DATA_AUTHORITY_SCHEMA = "cvs.phase2.data_authority.v1"
DATA_AUTHORITY_STATUS = "UPSTREAM_COMMIT_BLOCKED_UNSIGNED_NOT_FORMAL"
DATA_VALIDATION_COMMIT_SCHEMA = (
    "cvs.phase2.data_validation_commit_metadata.v1"
)
DATA_VALIDATION_COMMIT_STATUS = "COMMITTED_VALIDATED_ONCE_METADATA"
OFFLINE_AUDIT_SCHEMA = "cvs.phase2.predictor_package_offline_build_audit.v2"
PROTOCOL_SCHEMA = "p2_min_v1"

EXTERNAL_SIGNATURE_PROFILE = "somph_authority_external_envelope"
EXTERNAL_SIGNATURE_ENVELOPE_SCHEMA = (
    "cvs.phase2.data_authority_signed_envelope.v1"
)
EXTERNAL_SIGNATURE_DOMAIN = "cvs.phase2.data_authority.ed25519.v1"

CAPSULE_IDENTITY_SCHEMA = "cvs.phase2.data_capsule_identity.v1"
SPLIT_IDENTITY_SCHEMA = "cvs.phase2.data_split_identity.v1"
SOURCE_BINDING_SCHEMA = "cvs.phase2.data_authority_source_binding.v1"
ACCESS_AUDIT_SCHEMA = "cvs.phase2.data_authority_producer_access_audit.v1"

SHA256_RE = re.compile(r"[0-9a-f]{64}")
OPAQUE_CLASS_RE = re.compile(r"cls_[0-9a-f]{32,64}")
MAX_CONTROL_BYTES = 32 * 1024 * 1024

OFFLINE_AUDIT_KEYS = {
    "schema",
    "status",
    "target_cache_manifest",
    "target_cache_audit",
    "predictor_package_root_sha256",
    "predictor_package_seal_sha256",
    "predictor_scorer_roots_distinct",
    "opaque_token_secret_persisted",
    "same_scenario_support_query_physical_disjointness",
    "cross_scenario_selected_physical_disjointness",
    "cross_scenario_opaque_token_disjointness",
    "registered_class_rank_structure_consistent",
}

DATA_VALIDATION_COMMIT_KEYS = {
    "schema",
    "status",
    "protocol_schema",
    "phase2_data_status",
    "predictor_package_root_sha256",
    "predictor_package_seal_sha256",
    "predictor_package_manifest_sha256",
    "offline_build_audit_sha256",
    "offline_build_audit_canonical_sha256",
    "target_cache_manifest_file_sha256",
    "target_cache_manifest_canonical_sha256",
    "target_cache_audit_canonical_sha256",
    "data_member_descriptors_root_sha256",
    "receiver",
    "seed",
    "stage",
    "scenarios",
    "k_shot",
    "old_registry",
    "final_registry",
    "old_registry_identity_root_sha256",
    "final_registry_identity_root_sha256",
    "support_count_by_scenario",
    "support_count_by_class_by_scenario",
    "support_physical_ids_root_by_class_by_scenario",
    "query_count_by_scenario",
    "ordered_support_opaque_token_root_sha256_by_scenario",
    "ordered_query_opaque_token_root_sha256_by_scenario",
    "ordered_support_physical_ids_root_sha256_by_scenario",
    "ordered_query_physical_ids_root_sha256_by_scenario",
    "support_post_channel_iq_sha256_root_by_scenario",
    "query_post_channel_iq_sha256_root_by_scenario",
    "all_selected_physical_ids_root_sha256_by_scenario",
    "all_post_channel_iq_sha256_root_by_scenario",
    "same_scenario_support_query_physical_disjointness",
    "cross_scenario_selected_physical_disjointness",
    "cross_scenario_opaque_token_disjointness",
    "single_leo_observation",
    "clean_source_runtime_access",
    "query_fit_access",
    "query_decision_policy",
    "query_truth_in_predictor",
    "query_role_in_predictor",
}

DATA_MEMBER_DESCRIPTOR_FIELDS = (
    "artifact_role",
    "scenario",
    "schema",
    "sha256",
    "size_bytes",
    "npz_members",
)


class Phase2DataAuthorityError(ValueError):
    """Raised when metadata cannot prove the unsigned data authority facts."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise Phase2DataAuthorityError(f"{field} must be lowercase SHA256")
    return value


def _require_exact_mapping(
    value: Any, *, keys: set[str], field: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise Phase2DataAuthorityError(
            f"{field} exact schema drift: "
            f"missing={sorted(keys-actual)}, unexpected={sorted(actual-keys)}"
        )
    return dict(value)


def _read_regular_bytes(
    path: str | Path, *, expected_sha256: str, field: str
) -> bytes:
    expected = _require_sha256(expected_sha256, field=f"expected {field} SHA256")
    source = Path(path).absolute()
    try:
        before = source.lstat()
    except OSError as exc:
        raise Phase2DataAuthorityError(f"{field} is unreadable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise Phase2DataAuthorityError(f"{field} must be a regular non-symlink file")
    if before.st_size > MAX_CONTROL_BYTES:
        raise Phase2DataAuthorityError(f"{field} exceeds control-file size limit")
    try:
        with source.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise Phase2DataAuthorityError(f"{field} opened object is not regular")
            raw = handle.read(MAX_CONTROL_BYTES + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise Phase2DataAuthorityError(f"{field} could not be read") from exc
    if len(raw) > MAX_CONTROL_BYTES:
        raise Phase2DataAuthorityError(f"{field} exceeds control-file size limit")
    if (
        before.st_size != opened.st_size
        or opened.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise Phase2DataAuthorityError(f"{field} changed during read")
    if sha256_bytes(raw) != expected:
        raise Phase2DataAuthorityError(f"{field} SHA256 mismatch")
    return raw


def _decode_json(raw: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase2DataAuthorityError(f"{field} is not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise Phase2DataAuthorityError(f"{field} root must be an object")
    return value


def _scenario_sha_map(value: Any, *, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or list(value) != list(FORMAL_LEO_WEAK_SCENARIOS):
        raise Phase2DataAuthorityError(f"{field} must use exact ordered scenarios")
    return {
        scenario: _require_sha256(value[scenario], field=f"{field}.{scenario}")
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }


def _scenario_count_map(value: Any, *, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or list(value) != list(FORMAL_LEO_WEAK_SCENARIOS):
        raise Phase2DataAuthorityError(f"{field} must use exact ordered scenarios")
    result: dict[str, int] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        count = value[scenario]
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise Phase2DataAuthorityError(f"{field}.{scenario} must be positive int")
        result[scenario] = count
    return result


def _scenario_class_count_map(
    value: Any, *, classes: Sequence[str], field: str
) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping) or list(value) != list(FORMAL_LEO_WEAK_SCENARIOS):
        raise Phase2DataAuthorityError(f"{field} must use exact ordered scenarios")
    result: dict[str, dict[str, int]] = {}
    expected_classes = list(classes)
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        rows = value[scenario]
        if not isinstance(rows, Mapping) or set(rows) != set(expected_classes):
            raise Phase2DataAuthorityError(
                f"{field}.{scenario} must follow ordered final registry"
            )
        checked: dict[str, int] = {}
        for class_handle in expected_classes:
            count = rows[class_handle]
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise Phase2DataAuthorityError(
                    f"{field}.{scenario}.{class_handle} must be positive int"
                )
            checked[class_handle] = count
        result[scenario] = checked
    return result


def _scenario_class_sha_map(
    value: Any, *, classes: Sequence[str], field: str
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or list(value) != list(FORMAL_LEO_WEAK_SCENARIOS):
        raise Phase2DataAuthorityError(f"{field} must use exact ordered scenarios")
    result: dict[str, dict[str, str]] = {}
    expected_classes = list(classes)
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        rows = value[scenario]
        if not isinstance(rows, Mapping) or set(rows) != set(expected_classes):
            raise Phase2DataAuthorityError(
                f"{field}.{scenario} must follow ordered final registry"
            )
        result[scenario] = {
            class_handle: _require_sha256(
                rows[class_handle],
                field=f"{field}.{scenario}.{class_handle}",
            )
            for class_handle in expected_classes
        }
    return result


def _registered_classes(value: Any, *, expected_count: int) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise Phase2DataAuthorityError("registered_classes must be an ordered list")
    handles: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"class_index", "class_handle"}:
            raise Phase2DataAuthorityError("registered class exact schema drift")
        handle = item.get("class_handle")
        if item.get("class_index") != index or not isinstance(handle, str):
            raise Phase2DataAuthorityError("registered class order drift")
        if OPAQUE_CLASS_RE.fullmatch(handle) is None:
            raise Phase2DataAuthorityError("registered class handle is not opaque")
        handles.append(handle)
    if len(handles) != expected_count or len(set(handles)) != len(handles):
        raise Phase2DataAuthorityError("registered class count/uniqueness drift")
    return handles


def _validate_member_descriptors(manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    raw_members = manifest.get("members")
    if not isinstance(raw_members, list):
        raise Phase2DataAuthorityError("predictor manifest members missing")
    members: list[dict[str, Any]] = []
    roles: set[str] = set()
    paths: set[str] = set()
    for raw in raw_members:
        item = _require_exact_mapping(
            raw, keys=MEMBER_DESCRIPTOR_REQUIRED_KEYS, field="predictor member"
        )
        role = item.get("artifact_role")
        relative = item.get("relative_path")
        if not isinstance(role, str) or not isinstance(relative, str):
            raise Phase2DataAuthorityError("predictor member role/path invalid")
        if role in roles or relative in paths:
            raise Phase2DataAuthorityError("predictor member role/path duplicate")
        roles.add(role)
        paths.add(relative)
        _require_sha256(item.get("sha256"), field=f"member {role} SHA256")
        size = item.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise Phase2DataAuthorityError(f"member {role} size invalid")
        members.append(item)
    required_roles = {"checkpoint", "adapter", "head", "tta_policy"}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        required_roles.update({f"support:{scenario}", f"query:{scenario}"})
    if roles != required_roles:
        raise Phase2DataAuthorityError("predictor member role closure drift")
    if manifest.get("package_root_sha256") != package_root_sha256(members):
        raise Phase2DataAuthorityError("predictor package root descriptor drift")
    data_descriptors: list[dict[str, Any]] = []
    by_role = {item["artifact_role"]: item for item in members}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        for prefix, schema, npz_members in (
            ("support", SUPPORT_SCHEMA, SUPPORT_NPZ_MEMBERS),
            ("query", QUERY_SCHEMA, QUERY_NPZ_MEMBERS),
        ):
            item = by_role[f"{prefix}:{scenario}"]
            if (
                item.get("scenario") != scenario
                or item.get("schema") != schema
                or item.get("npz_members") != list(npz_members)
            ):
                raise Phase2DataAuthorityError(
                    f"predictor {prefix} descriptor drift: {scenario}"
                )
            data_descriptors.append(
                {field: item[field] for field in DATA_MEMBER_DESCRIPTOR_FIELDS}
            )
    return members, canonical_sha256(data_descriptors)


def _validate_manifest_and_seal(
    manifest: Mapping[str, Any],
    seal: Mapping[str, Any],
    *,
    manifest_raw: bytes,
) -> tuple[list[str], str]:
    _require_exact_mapping(
        manifest, keys=MANIFEST_REQUIRED_KEYS, field="predictor manifest"
    )
    _require_exact_mapping(seal, keys=SEAL_REQUIRED_KEYS, field="predictor seal")
    if (
        manifest.get("schema") != PREDICTOR_PACKAGE_MANIFEST_SCHEMA
        or manifest.get("artifact_stage") != PREDICTOR_INPUT_STAGE
        or seal.get("schema") != PREDICTOR_PACKAGE_SEAL_SCHEMA
        or seal.get("manifest_relative_path") != "package_manifest.json"
    ):
        raise Phase2DataAuthorityError("predictor manifest/seal identity drift")
    for field, expected in PHASE2_FULL_CONTRACT.items():
        if manifest.get(field) != expected:
            raise Phase2DataAuthorityError(f"predictor contract drift: {field}")
    if manifest.get("target_channel_view") != "leo_weak_only":
        raise Phase2DataAuthorityError("predictor target channel view drift")
    if manifest.get("target_channel_scenarios") != list(FORMAL_LEO_WEAK_SCENARIOS):
        raise Phase2DataAuthorityError("predictor scenario order drift")
    if manifest.get("stage") not in {"stage2b", "stage2c"}:
        raise Phase2DataAuthorityError("predictor stage drift")
    if not isinstance(manifest.get("receiver"), str) or not manifest["receiver"]:
        raise Phase2DataAuthorityError("predictor receiver missing")
    if not isinstance(manifest.get("seed"), int) or isinstance(manifest["seed"], bool):
        raise Phase2DataAuthorityError("predictor seed invalid")
    class_count = manifest.get("registered_class_count")
    new_count = manifest.get("new_class_count")
    max_k = manifest.get("support_pool_max_k")
    if (
        not isinstance(class_count, int)
        or isinstance(class_count, bool)
        or class_count < 1
        or not isinstance(new_count, int)
        or isinstance(new_count, bool)
        or new_count < 0
        or new_count >= class_count
        or not isinstance(max_k, int)
        or isinstance(max_k, bool)
        or max_k < 1
    ):
        raise Phase2DataAuthorityError("predictor registry/K metadata invalid")
    if manifest["stage"] == "stage2b" and new_count != 0:
        raise Phase2DataAuthorityError("Stage2-B cannot register new classes")
    if manifest["stage"] == "stage2c" and new_count < 1:
        raise Phase2DataAuthorityError("Stage2-C must register new classes")
    handles = _registered_classes(
        manifest.get("registered_classes"), expected_count=class_count
    )
    _members, data_root = _validate_member_descriptors(manifest)
    manifest_sha = sha256_bytes(manifest_raw)
    if (
        seal.get("manifest_sha256") != manifest_sha
        or seal.get("manifest_size_bytes") != len(manifest_raw)
        or seal.get("package_root_sha256") != manifest.get("package_root_sha256")
        or seal.get("artifact_member_allowlist_sha256")
        != manifest.get("package_root_sha256")
    ):
        raise Phase2DataAuthorityError("predictor manifest/seal binding drift")
    return handles, data_root


def _validate_offline_audit(
    audit: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    seal_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _require_exact_mapping(
        audit, keys=OFFLINE_AUDIT_KEYS, field="offline build audit"
    )
    if (
        value.get("schema") != OFFLINE_AUDIT_SCHEMA
        or value.get("status") != "PASS"
        or value.get("predictor_package_root_sha256")
        != manifest.get("package_root_sha256")
        or value.get("predictor_package_seal_sha256") != seal_sha256
        or value.get("predictor_scorer_roots_distinct") is not True
        or value.get("opaque_token_secret_persisted") is not False
        or value.get("same_scenario_support_query_physical_disjointness") != "PASS"
        or value.get("cross_scenario_selected_physical_disjointness") != "PASS"
        or value.get("cross_scenario_opaque_token_disjointness") != "PASS"
        or value.get("registered_class_rank_structure_consistent") != "PASS"
    ):
        raise Phase2DataAuthorityError("offline build audit did not pass")
    cache_manifest = value.get("target_cache_manifest")
    cache_audit = value.get("target_cache_audit")
    if not isinstance(cache_manifest, Mapping) or not isinstance(cache_audit, Mapping):
        raise Phase2DataAuthorityError("offline build cache evidence missing")
    required_cache_manifest = {
        "cache_scope": "stage2_registered",
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
        **PHASE2_SINGLE_OBSERVATION_CONTRACT,
    }
    for field, expected in required_cache_manifest.items():
        if cache_manifest.get(field) != expected:
            raise Phase2DataAuthorityError(f"cache manifest contract drift: {field}")
    _scenario_sha_map(
        cache_manifest.get("physical_sample_ids_sha256_by_scenario"),
        field="cache manifest physical roots",
    )
    _require_sha256(
        cache_manifest.get("physical_sample_scenario_assignment_sha256"),
        field="cache manifest scenario assignment",
    )
    if (
        cache_audit.get("scope") != "stage2_registered"
        or cache_audit.get("scenario_order") != list(FORMAL_LEO_WEAK_SCENARIOS)
        or cache_audit.get("clean_sample_access") is not False
        or cache_audit.get("phase2_sample_view_policy")
        != "leo_weak_only_no_clean_access"
        or cache_audit.get("phase2_cross_scenario_physical_sample_reuse") is not False
        or cache_audit.get("phase2_single_observation_compliant") is not True
        or cache_audit.get("phase2_physical_sample_observation_policy")
        != PHASE2_SINGLE_OBSERVATION_CONTRACT[
            "phase2_physical_sample_observation_policy"
        ]
    ):
        raise Phase2DataAuthorityError("cache audit single-observation evidence failed")
    audit_physical = _scenario_sha_map(
        cache_audit.get("physical_sample_ids_sha256_by_scenario"),
        field="cache audit physical roots",
    )
    if (
        audit_physical
        != dict(cache_manifest["physical_sample_ids_sha256_by_scenario"])
        or cache_audit.get("physical_sample_scenario_assignment_sha256")
        != cache_manifest.get("physical_sample_scenario_assignment_sha256")
    ):
        raise Phase2DataAuthorityError("cache manifest/audit physical roots drift")
    cache_audits = cache_audit.get("cache_audits")
    if not isinstance(cache_audits, Mapping) or list(cache_audits) != list(
        FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise Phase2DataAuthorityError("cache per-scenario audit missing")
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        item = cache_audits[scenario]
        if (
            not isinstance(item, Mapping)
            or item.get("scenario") != scenario
            or item.get("clean_sample_access") is not False
            or item.get("phase2_sample_view_policy")
            != "leo_weak_only_no_clean_access"
        ):
            raise Phase2DataAuthorityError(
                f"cache per-scenario audit drift: {scenario}"
            )
        for field in (
            "sha256",
            "physical_sample_ids_sha256",
            "post_channel_iq_sha256_root",
            "overlay_ids_sha256",
            "manifest_sha256",
        ):
            _require_sha256(item.get(field), field=f"cache audit {scenario}.{field}")
    return dict(cache_manifest), dict(cache_audit)


def _validate_commit(
    commit: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    seal_sha256: str,
    offline_audit_raw_sha256: str,
    offline_audit: Mapping[str, Any],
    cache_manifest: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
    data_descriptor_root: str,
    manifest_registry: list[str],
) -> dict[str, Any]:
    value = _require_exact_mapping(
        commit, keys=DATA_VALIDATION_COMMIT_KEYS, field="data validation COMMIT"
    )
    expected_bindings = {
        "schema": DATA_VALIDATION_COMMIT_SCHEMA,
        "status": DATA_VALIDATION_COMMIT_STATUS,
        "protocol_schema": PROTOCOL_SCHEMA,
        "phase2_data_status": "VALIDATED_ONCE",
        "predictor_package_root_sha256": manifest["package_root_sha256"],
        "predictor_package_seal_sha256": seal_sha256,
        "predictor_package_manifest_sha256": manifest_sha256,
        "offline_build_audit_sha256": offline_audit_raw_sha256,
        "offline_build_audit_canonical_sha256": canonical_sha256(offline_audit),
        "target_cache_manifest_file_sha256": cache_audit.get("sha256"),
        "target_cache_manifest_canonical_sha256": canonical_sha256(cache_manifest),
        "target_cache_audit_canonical_sha256": canonical_sha256(cache_audit),
        "data_member_descriptors_root_sha256": data_descriptor_root,
        "receiver": manifest["receiver"],
        "seed": manifest["seed"],
        "stage": manifest["stage"],
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "same_scenario_support_query_physical_disjointness": "PASS",
        "cross_scenario_selected_physical_disjointness": "PASS",
        "cross_scenario_opaque_token_disjointness": "PASS",
        "single_leo_observation": True,
        "clean_source_runtime_access": False,
        "query_fit_access": False,
        "query_decision_policy": "per_sample_all_registered_classes",
        "query_truth_in_predictor": False,
        "query_role_in_predictor": False,
    }
    for field, expected in expected_bindings.items():
        if value.get(field) != expected:
            raise Phase2DataAuthorityError(f"data validation COMMIT drift: {field}")
    k_shot = value.get("k_shot")
    if (
        not isinstance(k_shot, int)
        or isinstance(k_shot, bool)
        or k_shot < 1
        or k_shot > manifest["support_pool_max_k"]
    ):
        raise Phase2DataAuthorityError("data validation COMMIT K invalid")
    old_registry = value.get("old_registry")
    final_registry = value.get("final_registry")
    if not isinstance(old_registry, list) or not isinstance(final_registry, list):
        raise Phase2DataAuthorityError("data validation COMMIT registry missing")
    new_count = manifest["new_class_count"]
    old_count = manifest["registered_class_count"] - new_count
    expected_old = manifest_registry[:old_count]
    if (
        final_registry != manifest_registry
        or old_registry != expected_old
        or final_registry[: len(old_registry)] != old_registry
        or len(set(final_registry)) != len(final_registry)
    ):
        raise Phase2DataAuthorityError("old/final registry prefix drift")
    if manifest["stage"] == "stage2c" and len(final_registry) <= len(old_registry):
        raise Phase2DataAuthorityError("Stage2-C final registry must extend old prefix")
    old_registry_identity_root = _require_sha256(
        value.get("old_registry_identity_root_sha256"),
        field="old registry identity root",
    )
    final_registry_identity_root = _require_sha256(
        value.get("final_registry_identity_root_sha256"),
        field="final registry identity root",
    )
    if manifest["stage"] == "stage2b":
        if old_registry != final_registry or old_registry_identity_root != final_registry_identity_root:
            raise Phase2DataAuthorityError(
                "Stage2-B old/final registry identities must be equal"
            )
    elif old_registry_identity_root == final_registry_identity_root:
        raise Phase2DataAuthorityError(
            "Stage2-C old/final registry identity roots must differ"
        )
    support_counts = _scenario_count_map(
        value.get("support_count_by_scenario"), field="support counts"
    )
    support_class_counts = _scenario_class_count_map(
        value.get("support_count_by_class_by_scenario"),
        classes=final_registry,
        field="support class counts",
    )
    support_class_roots = _scenario_class_sha_map(
        value.get("support_physical_ids_root_by_class_by_scenario"),
        classes=final_registry,
        field="support class physical roots",
    )
    query_counts = _scenario_count_map(
        value.get("query_count_by_scenario"), field="query counts"
    )
    expected_support_count = len(final_registry) * k_shot
    if any(count != expected_support_count for count in support_counts.values()):
        raise Phase2DataAuthorityError("support count does not prove exact K per class")
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        per_class = support_class_counts[scenario]
        if any(count != k_shot for count in per_class.values()):
            raise Phase2DataAuthorityError("support class count does not prove exact K")
        if sum(per_class.values()) != support_counts[scenario]:
            raise Phase2DataAuthorityError("support class/total count drift")
    if len(set(query_counts.values())) != 1:
        raise Phase2DataAuthorityError("query count drifts across scenarios")
    for field in (
        "ordered_support_opaque_token_root_sha256_by_scenario",
        "ordered_query_opaque_token_root_sha256_by_scenario",
        "ordered_support_physical_ids_root_sha256_by_scenario",
        "ordered_query_physical_ids_root_sha256_by_scenario",
        "support_post_channel_iq_sha256_root_by_scenario",
        "query_post_channel_iq_sha256_root_by_scenario",
        "all_selected_physical_ids_root_sha256_by_scenario",
        "all_post_channel_iq_sha256_root_by_scenario",
    ):
        value[field] = _scenario_sha_map(value.get(field), field=field)
    value["support_count_by_scenario"] = support_counts
    value["support_count_by_class_by_scenario"] = support_class_counts
    value["support_physical_ids_root_by_class_by_scenario"] = support_class_roots
    value["query_count_by_scenario"] = query_counts
    return value


def build_phase2_data_authority_payload(
    *,
    predictor_manifest: Mapping[str, Any],
    predictor_manifest_raw: bytes,
    predictor_seal: Mapping[str, Any],
    predictor_seal_sha256: str,
    offline_build_audit: Mapping[str, Any],
    offline_build_audit_raw_sha256: str,
    data_validation_commit: Mapping[str, Any],
    data_validation_commit_sha256: str,
) -> dict[str, Any]:
    """Build one unsigned authority from already-validated control metadata."""

    seal_sha = _require_sha256(
        predictor_seal_sha256, field="predictor seal SHA256"
    )
    audit_sha = _require_sha256(
        offline_build_audit_raw_sha256, field="offline audit SHA256"
    )
    commit_sha = _require_sha256(
        data_validation_commit_sha256, field="validation COMMIT SHA256"
    )
    manifest_registry, data_descriptor_root = _validate_manifest_and_seal(
        predictor_manifest,
        predictor_seal,
        manifest_raw=predictor_manifest_raw,
    )
    cache_manifest, cache_audit = _validate_offline_audit(
        offline_build_audit,
        manifest=predictor_manifest,
        seal_sha256=seal_sha,
    )
    commit = _validate_commit(
        data_validation_commit,
        manifest=predictor_manifest,
        manifest_sha256=sha256_bytes(predictor_manifest_raw),
        seal_sha256=seal_sha,
        offline_audit_raw_sha256=audit_sha,
        offline_audit=offline_build_audit,
        cache_manifest=cache_manifest,
        cache_audit=cache_audit,
        data_descriptor_root=data_descriptor_root,
        manifest_registry=manifest_registry,
    )

    split_roots = {
        field: commit[field]
        for field in (
            "support_count_by_scenario",
            "query_count_by_scenario",
            "ordered_support_physical_ids_root_sha256_by_scenario",
            "ordered_query_physical_ids_root_sha256_by_scenario",
        )
    }
    capsule_identity = {
        "schema": CAPSULE_IDENTITY_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "receiver": commit["receiver"],
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "all_selected_physical_ids_root_sha256_by_scenario": commit[
            "all_selected_physical_ids_root_sha256_by_scenario"
        ],
        "all_post_channel_iq_sha256_root_by_scenario": commit[
            "all_post_channel_iq_sha256_root_by_scenario"
        ],
        "single_leo_observation": True,
        "physical_sample_observation_policy": PHASE2_SINGLE_OBSERVATION_CONTRACT[
            "phase2_physical_sample_observation_policy"
        ],
        "post_reception_view_from_fixed_received_iq_only": True,
    }
    capsule_id = canonical_sha256(capsule_identity)
    split_identity = {
        "schema": SPLIT_IDENTITY_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "capsule_id": capsule_id,
        "stage": commit["stage"],
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "k_shot": commit["k_shot"],
        "old_registry_identity_root_sha256": commit[
            "old_registry_identity_root_sha256"
        ],
        "final_registry_identity_root_sha256": commit[
            "final_registry_identity_root_sha256"
        ],
        "old_registry_is_exact_final_prefix": True,
        **split_roots,
    }
    split_id = canonical_sha256(split_identity)
    source_binding = {
        "schema": SOURCE_BINDING_SCHEMA,
        "predictor_package_root_sha256": predictor_manifest[
            "package_root_sha256"
        ],
        "predictor_package_manifest_sha256": sha256_bytes(predictor_manifest_raw),
        "predictor_package_seal_sha256": seal_sha,
        "offline_build_audit_sha256": audit_sha,
        "data_validation_commit_sha256": commit_sha,
        "target_cache_manifest_file_sha256": cache_audit["sha256"],
        "data_member_descriptors_root_sha256": data_descriptor_root,
        "excluded_from_capsule_id": True,
        "excluded_from_split_id": True,
    }
    data_facts = {
        "receiver": commit["receiver"],
        "seed": commit["seed"],
        "stage": commit["stage"],
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "k_shot": commit["k_shot"],
        "old_registry": list(commit["old_registry"]),
        "final_registry": list(commit["final_registry"]),
        "old_registry_identity_root_sha256": commit[
            "old_registry_identity_root_sha256"
        ],
        "final_registry_identity_root_sha256": commit[
            "final_registry_identity_root_sha256"
        ],
        "old_registry_is_exact_final_prefix": True,
        **{
            field: commit[field]
            for field in (
                "support_count_by_scenario",
                "support_count_by_class_by_scenario",
                "support_physical_ids_root_by_class_by_scenario",
                "query_count_by_scenario",
                "ordered_support_opaque_token_root_sha256_by_scenario",
                "ordered_query_opaque_token_root_sha256_by_scenario",
                "ordered_support_physical_ids_root_sha256_by_scenario",
                "ordered_query_physical_ids_root_sha256_by_scenario",
                "support_post_channel_iq_sha256_root_by_scenario",
                "query_post_channel_iq_sha256_root_by_scenario",
                "all_selected_physical_ids_root_sha256_by_scenario",
                "all_post_channel_iq_sha256_root_by_scenario",
            )
        },
        "same_scenario_support_query_physical_disjointness": "PASS",
        "cross_scenario_selected_physical_disjointness": "PASS",
        "cross_scenario_opaque_token_disjointness": "PASS",
        "single_observation_contract": dict(PHASE2_SINGLE_OBSERVATION_CONTRACT),
    }
    return {
        "schema": DATA_AUTHORITY_SCHEMA,
        "status": DATA_AUTHORITY_STATUS,
        "formal_data_authority": False,
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "external_signature_present": False,
        "external_signature_required_for_formal": True,
        "external_signature_profile": EXTERNAL_SIGNATURE_PROFILE,
        "external_signature_envelope_schema": EXTERNAL_SIGNATURE_ENVELOPE_SCHEMA,
        "external_signature_domain": EXTERNAL_SIGNATURE_DOMAIN,
        "protocol_schema": PROTOCOL_SCHEMA,
        "phase2_data_status": "UPSTREAM_COMMIT_BLOCKED",
        "upstream_validated_once_claim_present": True,
        "upstream_validated_once_claim_trusted": False,
        "upstream_commit_authority_status": "UNVERIFIED_UNSIGNED",
        "single_leo_observation": True,
        "clean_source_runtime_access": False,
        "query_fit_access": False,
        "query_decision_policy": "per_sample_all_registered_classes",
        "receiver": commit["receiver"],
        "seed": commit["seed"],
        "stage": commit["stage"],
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "k_shot": commit["k_shot"],
        "old_registry": list(commit["old_registry"]),
        "final_registry": list(commit["final_registry"]),
        "capsule_id": capsule_id,
        "split_id": split_id,
        "capsule_identity": capsule_identity,
        "split_identity": split_identity,
        "data_facts": data_facts,
        "source_binding": source_binding,
        "producer_access_audit": {
            "schema": ACCESS_AUDIT_SCHEMA,
            "control_files_opened": [
                "predictor_manifest",
                "predictor_detached_seal",
                "offline_build_audit",
                "data_validation_commit_metadata",
            ],
            "support_payload_open_count": 0,
            "query_payload_open_count": 0,
            "iq_payload_materialized": False,
            "query_truth_open_count": 0,
            "cache_payload_open_count": 0,
            "data_revalidation_performed": False,
        },
    }


def write_phase2_data_authority(
    *,
    predictor_manifest_path: str | Path,
    predictor_seal_path: str | Path,
    expected_predictor_seal_sha256: str,
    offline_build_audit_path: str | Path,
    expected_offline_build_audit_sha256: str,
    data_validation_commit_path: str | Path,
    expected_data_validation_commit_sha256: str,
    output_path: str | Path,
    expected_output_sha256: str | None = None,
) -> dict[str, Any]:
    """Read four control files and publish one canonical unsigned authority."""

    seal_raw = _read_regular_bytes(
        predictor_seal_path,
        expected_sha256=expected_predictor_seal_sha256,
        field="predictor detached seal",
    )
    seal = _decode_json(seal_raw, field="predictor detached seal")
    manifest_expected = _require_sha256(
        seal.get("manifest_sha256"), field="seal manifest SHA256"
    )
    manifest_raw = _read_regular_bytes(
        predictor_manifest_path,
        expected_sha256=manifest_expected,
        field="predictor manifest",
    )
    audit_raw = _read_regular_bytes(
        offline_build_audit_path,
        expected_sha256=expected_offline_build_audit_sha256,
        field="offline build audit",
    )
    commit_raw = _read_regular_bytes(
        data_validation_commit_path,
        expected_sha256=expected_data_validation_commit_sha256,
        field="data validation COMMIT metadata",
    )
    payload = build_phase2_data_authority_payload(
        predictor_manifest=_decode_json(manifest_raw, field="predictor manifest"),
        predictor_manifest_raw=manifest_raw,
        predictor_seal=seal,
        predictor_seal_sha256=sha256_bytes(seal_raw),
        offline_build_audit=_decode_json(audit_raw, field="offline build audit"),
        offline_build_audit_raw_sha256=sha256_bytes(audit_raw),
        data_validation_commit=_decode_json(
            commit_raw, field="data validation COMMIT metadata"
        ),
        data_validation_commit_sha256=sha256_bytes(commit_raw),
    )
    output_raw = canonical_json_bytes(payload) + b"\n"
    output_sha = sha256_bytes(output_raw)
    if expected_output_sha256 is not None:
        expected = _require_sha256(
            expected_output_sha256, field="expected output SHA256"
        )
        if output_sha != expected:
            raise Phase2DataAuthorityError("unsigned authority output SHA256 mismatch")
    destination = Path(output_path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise Phase2DataAuthorityError("output parent must be a non-symlink directory")
    try:
        with destination.open("xb") as handle:
            handle.write(output_raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise
    return {
        "schema": DATA_AUTHORITY_SCHEMA,
        "status": DATA_AUTHORITY_STATUS,
        "output_path": str(destination),
        "output_sha256": output_sha,
        "output_size_bytes": len(output_raw),
        "capsule_id": payload["capsule_id"],
        "split_id": payload["split_id"],
        "formal_data_authority": False,
        "formal_launch_authority": False,
        "external_signature_present": False,
        "external_signature_required_for_formal": True,
    }


__all__ = [
    "ACCESS_AUDIT_SCHEMA",
    "CAPSULE_IDENTITY_SCHEMA",
    "DATA_AUTHORITY_SCHEMA",
    "DATA_AUTHORITY_STATUS",
    "DATA_VALIDATION_COMMIT_KEYS",
    "DATA_VALIDATION_COMMIT_SCHEMA",
    "DATA_VALIDATION_COMMIT_STATUS",
    "EXTERNAL_SIGNATURE_DOMAIN",
    "EXTERNAL_SIGNATURE_ENVELOPE_SCHEMA",
    "EXTERNAL_SIGNATURE_PROFILE",
    "Phase2DataAuthorityError",
    "PROTOCOL_SCHEMA",
    "SOURCE_BINDING_SCHEMA",
    "SPLIT_IDENTITY_SCHEMA",
    "build_phase2_data_authority_payload",
    "canonical_sha256",
    "sha256_bytes",
    "write_phase2_data_authority",
]
