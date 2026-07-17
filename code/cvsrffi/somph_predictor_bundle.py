"""SOMP-H-specific sealed enrollment/apply bundles.

The two profiles are physically distinct:

``enrollment_only``
    checkpoint, method lock, overlay provenance, and three registered-support
    LEO_weak IQ archives.

``apply_only``
    checkpoint, method lock, frozen head capsule, overlay provenance, and
    three unlabeled-query LEO_weak IQ archives.

All package-member and provenance checks happen before any IQ archive is
materialized with NumPy.  The module deliberately exposes no dataset path,
cache builder, truth sidecar, scorer, or legacy loader entry point.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import weakref
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Callable, Mapping

import numpy as np

from cvsrffi import leo_weak_cache
from cvsrffi import somph_formal_matrix as formal_matrix
from cvsrffi import somph_lineage_authority as lineage_authority
from cvsrffi import stage2_predictor_bundle as stage2_bundle_module
from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT
from cvsrffi.somph_formal_matrix import (
    FORMAL_NEW_CLASS_COUNTS,
    FORMAL_RECEIVERS,
    NEW_TX_IDS,
    OLD_TX_IDS,
)
from cvsrffi.somph_predictor_runtime import (
    SOMPH_HEAD_CAPSULE_SCHEMA,
    expected_somph_method_lock,
    validate_somph_head_capsule,
)
from cvsrffi.stage2_predictor_bundle import (
    FORMAL_LEO_WEAK_SCENARIOS,
    OPAQUE_TOKEN_RE,
    PredictorPackageError,
    _ensure_root as _base_ensure_root,
    _hash_handle,
    _json_from_handle,
    _parse_embedded_manifest,
    _validate_package_root_exact_allowlist,
    _zip_members_from_handle,
    canonical_json_bytes,
    iq_row_sha256,
    open_regular_member_same_fd,
    package_root_sha256,
    sha256_bytes,
    sha256_file,
    validate_relative_member_path,
)


ADV3B02_PHASE1_CHECKPOINT_SHA256 = (
    "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
)
# Compatibility alias for callers that still import the historical name.  The
# bytes identified here are Phase1 state-dict lineage, never a Phase2 runtime.
ADV3B02_CHECKPOINT_SHA256 = ADV3B02_PHASE1_CHECKPOINT_SHA256
ADV3B02_FEATURE_SCHEMA = "adv3b02_z_id160_fp32"
SOMPH_METHOD_LOCK_SCHEMA = "cvs.phase2.somph_method_lock.v1"
SOMPH_ENROLLMENT_BINDING_SCHEMA = "cvs.phase2.somph_enrollment_binding.v1"

SOMPH_BUNDLE_MANIFEST_SCHEMA = "cvs.phase2.somph_predictor_bundle.v1"
SOMPH_BUNDLE_SEAL_SCHEMA = "cvs.phase2.somph_predictor_bundle_seal.v1"
SOMPH_OVERLAY_PROVENANCE_SCHEMA = "cvs.phase2.somph_overlay_provenance.v1"
SOMPH_SUPPORT_IQ_SCHEMA = "cvs.phase2.somph_registered_support_iq.v1"
SOMPH_QUERY_IQ_SCHEMA = "cvs.phase2.somph_unlabeled_query_iq.v1"
SOMPH_FORMAL_POLICY_AUTHORIZATION_SCHEMA = (
    "cvs.phase2.somph_formal_row_policy_authorization.v1"
)
SOMPH_FORMAL_POLICY_AUTHORIZATION_STATUS = "FORMAL_ROW_POLICY_AUTHORIZED"
SOMPH_FORMAL_POLICY_SCHEMA = "cvs.phase2.somph_formal_execution_policy.v1"
SOMPH_FORMAL_POLICY_STATUS = "FORMAL_EXECUTION_POLICY_LOCKED"
SOMPH_SIGNED_POLICY_ENVELOPE_SCHEMA = (
    "cvs.phase2.somph_signed_policy_authorization_envelope.v1"
)
SOMPH_SIGNED_POLICY_ENVELOPE_DOMAIN = (
    "cvs.somph.formal_policy_authorization.ed25519.v1"
)

ENROLLMENT_ONLY = "enrollment_only"
APPLY_ONLY = "apply_only"
PROFILE_VALUES = frozenset({ENROLLMENT_ONLY, APPLY_ONLY})
REGISTRATION_STATES = frozenset({"before", "after"})
SUPPORT_POOL_MAX_K = 20

FEATURE_RUNTIME_RELATIVE_PATH = "sealed_feature_runtime.pt"
METHOD_LOCK_RELATIVE_PATH = "method_lock.json"
HEAD_CAPSULE_RELATIVE_PATH = "head_capsule.npz"
OVERLAY_PROVENANCE_RELATIVE_PATH = "overlay_provenance.json"
MANIFEST_RELATIVE_PATH = "package_manifest.json"

SUPPORT_NPZ_MEMBERS = (
    "support_leo_weak_iq",
    "support_class_indices",
    "support_rank_within_class",
    "support_tokens",
    "support_overlay_tokens",
    "support_satellite_seeds",
    "support_post_channel_iq_sha256",
    "manifest_json",
)
QUERY_NPZ_MEMBERS = (
    "query_leo_weak_iq",
    "query_tokens",
    "query_overlay_tokens",
    "query_satellite_seeds",
    "query_post_channel_iq_sha256",
    "manifest_json",
)
_RUNTIME_HEAD_TENSOR_KEYS = (
    "prototypes_fp16",
    "prototype_class_ids_uint16",
    "centroids_fp16",
    "residual_scale_fp16",
    "class_hubness_penalty_fp16",
    "scalars_fp16",
)
HEAD_CAPSULE_NPZ_MEMBERS = (
    "schema_utf8",
    "method_lock_sha256_utf8",
    "enrollment_binding_json_utf8",
    "class_count_uint16",
    "feature_dim_uint16",
    "k_shot_uint16",
    *tuple(
        member
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
        for member in tuple(
            f"{scenario}__{key}" for key in _RUNTIME_HEAD_TENSOR_KEYS
        )
    ),
)

MEMBER_DESCRIPTOR_KEYS = {
    "relative_path",
    "sha256",
    "size_bytes",
    "kind",
    "schema",
    "scenario",
    "npz_members",
}
MANIFEST_KEYS = {
    "schema",
    "profile",
    "stage",
    "registration_state",
    "receiver",
    "seed",
    "k_shot",
    "support_pool_max_k",
    "target_channel_scenarios",
    "registered_class_count",
    "registered_classes",
    "phase1_checkpoint_sha256",
    "feature_runtime_sha256",
    "method_lock_sha256",
    "overlay_provenance_sha256",
    "head_capsule_sha256",
    "head_enrollment_binding_sha256",
    "row_handle",
    "row_manifest_sha256",
    "members",
    "package_root_sha256",
    *PHASE2_FULL_CONTRACT.keys(),
}
SEAL_KEYS = {
    "schema",
    "manifest_relative_path",
    "manifest_sha256",
    "manifest_size_bytes",
    "package_root_sha256",
    "artifact_member_allowlist_sha256",
}
PROVENANCE_KEYS = {
    "schema",
    "profile",
    "receiver",
    "seed",
    "samples",
}
PROVENANCE_SAMPLE_KEYS = {
    "sample_token",
    "scenario",
    "overlay_token",
    "satellite_seed",
    "post_channel_iq_sha256",
    "source_leo_cache_sha256",
    "source_leo_provenance_sha256",
}
FORMAL_POLICY_AUTHORIZATION_KEYS = {
    "schema",
    "status",
    "formal_launch_authority",
    "formal_metric_claim_allowed",
    "package_root_sha256",
    "package_detached_seal_sha256",
    "artifact_member_allowlist_sha256",
    "manifest_sha256",
    "overlay_provenance_sha256",
    "authority_commit_sha256",
    "authority_lock_sha256",
    "authority_attestation_sha256",
    "receiver",
    "seed",
    "stage",
    "registration_state",
    "k_shot",
    "cache_scope",
    "old_tx_ids",
    "new_tx_ids",
    "dataset_authority_root_sha256",
    "cache_role_inputs_root_sha256",
    "physical_sample_ids_sha256_by_scenario",
    "physical_sample_scenario_assignment_sha256",
    "post_channel_iq_sha256_root_by_scenario",
    "overlay_ids_sha256_by_scenario",
    "preflight_code_sha256",
    "formal_policy_sha256",
    "code_closure_sha256",
    "package_class_registry_sha256",
    "package_role_registry_sha256",
    "package_physical_sample_ids_sha256_by_scenario",
    "package_overlay_ids_sha256_by_scenario",
    "package_post_channel_iq_sha256_by_scenario",
    "package_satellite_seed_sha256_by_scenario",
    "package_materialized_assignment_sha256_by_scenario",
    "package_sample_assignment_sha256",
}
FORMAL_POLICY_KEYS = {
    "schema",
    "status",
    "formal_receivers",
    "old_tx_ids",
    "nested_new_tx_ids",
    "cache_scope",
    "old_dataset_basename",
    "new_dataset_basename",
    "physical_sample_scenario_assignment_policy",
    "single_observation_contract",
    "required_code_closure_members",
}
SIGNED_POLICY_ENVELOPE_KEYS = {
    "schema",
    "domain",
    "issuer",
    "key_id",
    "authorization_canonical_sha256",
    "formal_policy_sha256",
    "package_root_sha256",
    "package_detached_seal_sha256",
    "authority_commit_sha256",
    "receiver",
    "seed",
    "stage",
    "registration_state",
    "k_shot",
    "code_closure_sha256",
    "signature_ed25519_hex",
}
CODE_CLOSURE_LOGICAL_MEMBERS = (
    "somph_predictor_bundle.py",
    "somph_lineage_authority.py",
    "stage2_predictor_bundle.py",
    "somph_formal_matrix.py",
    "leo_weak_cache.py",
)
RUNTIME_BINDING_KEYS = {
    "package_root_sha256",
    "feature_runtime_sha256",
    "phase1_checkpoint_sha256",
    "method_lock_sha256",
    "preflight_anchor_sha256",
    "code_closure_sha256",
}
MATERIALIZATION_RECEIPT_KEYS = {
    "schema",
    "status",
    "iq_payload_materialized",
    "materialized_scenarios",
    "package_root_sha256",
    "preflight_anchor_sha256",
    "runtime_binding",
}
SOMPH_MATERIALIZATION_RECEIPT_SCHEMA = (
    "cvs.phase2.somph_iq_materialization_receipt.v1"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ROW_HANDLE_RE = re.compile(r"row_[0-9a-f]{64}")
ENROLLMENT_BINDING_KEYS = {
    "schema",
    "stage",
    "registration_state",
    "receiver",
    "seed",
    "k_shot",
    "registered_class_handles",
    "enrollment_package_root_sha256",
    "enrollment_package_seal_sha256",
    "phase1_checkpoint_sha256",
    "feature_runtime_sha256",
    "method_lock_sha256",
    "support_token_sha256_by_scenario",
    "support_feature_sha256_by_scenario",
}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value.lower()) is not None


def _is_row_handle(value: Any) -> bool:
    return isinstance(value, str) and ROW_HANDLE_RE.fullmatch(value) is not None


def _is_reparse_or_symlink(path: Path) -> bool:
    metadata = path.lstat()
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _reject_reparse_ancestor_chain(path: Path, *, context: str) -> Path:
    """Reject a root reached through any symlink/junction/reparse ancestor."""

    absolute = Path(os.path.abspath(path))
    anchor = Path(absolute.anchor)
    for candidate in (absolute, *absolute.parents):
        if candidate == anchor:
            break
        try:
            if _is_reparse_or_symlink(candidate):
                raise PredictorPackageError(
                    f"{context} parent symlink/reparse path rejected: {candidate}"
                )
        except FileNotFoundError as exc:
            raise PredictorPackageError(
                f"missing {context} ancestor: {candidate}"
            ) from exc
    return absolute


def _ensure_root(root: Path) -> Path:
    return _base_ensure_root(
        _reject_reparse_ancestor_chain(root, context="SOMP-H package root")
    )


def _profile_kinds(profile: str) -> tuple[str, ...]:
    common = ("feature_runtime", "method_lock", "overlay_provenance")
    if profile == ENROLLMENT_ONLY:
        return common + tuple(f"support:{value}" for value in FORMAL_LEO_WEAK_SCENARIOS)
    if profile == APPLY_ONLY:
        return common[:2] + ("head_capsule", common[2]) + tuple(
            f"query:{value}" for value in FORMAL_LEO_WEAK_SCENARIOS
        )
    raise PredictorPackageError("SOMP-H bundle profile invalid")


def _relative_path_for_kind(kind: str) -> str:
    fixed = {
        "feature_runtime": FEATURE_RUNTIME_RELATIVE_PATH,
        "method_lock": METHOD_LOCK_RELATIVE_PATH,
        "head_capsule": HEAD_CAPSULE_RELATIVE_PATH,
        "overlay_provenance": OVERLAY_PROVENANCE_RELATIVE_PATH,
    }
    if kind in fixed:
        return fixed[kind]
    prefix, separator, scenario = kind.partition(":")
    if separator != ":" or scenario not in FORMAL_LEO_WEAK_SCENARIOS:
        raise PredictorPackageError(f"SOMP-H bundle member kind invalid: {kind}")
    if prefix == "support":
        return f"support_{scenario}.npz"
    if prefix == "query":
        return f"query_{scenario}.npz"
    raise PredictorPackageError(f"SOMP-H bundle member kind invalid: {kind}")


def _schema_for_kind(kind: str) -> str:
    if kind == "feature_runtime":
        return "adv3b02.torchscript_identity_runtime.v1"
    if kind == "method_lock":
        return SOMPH_METHOD_LOCK_SCHEMA
    if kind == "head_capsule":
        return SOMPH_HEAD_CAPSULE_SCHEMA
    if kind == "overlay_provenance":
        return SOMPH_OVERLAY_PROVENANCE_SCHEMA
    if kind.startswith("support:"):
        return SOMPH_SUPPORT_IQ_SCHEMA
    if kind.startswith("query:"):
        return SOMPH_QUERY_IQ_SCHEMA
    raise PredictorPackageError(f"SOMP-H bundle member kind invalid: {kind}")


def _npz_members_for_kind(kind: str) -> tuple[str, ...]:
    if kind == "head_capsule":
        return HEAD_CAPSULE_NPZ_MEMBERS
    if kind.startswith("support:"):
        return SUPPORT_NPZ_MEMBERS
    if kind.startswith("query:"):
        return QUERY_NPZ_MEMBERS
    return ()


def _scenario_for_kind(kind: str) -> str | None:
    if ":" not in kind:
        return None
    return kind.split(":", 1)[1]


def _descriptor(root: Path, kind: str) -> dict[str, Any]:
    relative = _relative_path_for_kind(kind)
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise PredictorPackageError(f"missing regular SOMP-H member: {relative}")
    return {
        "relative_path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "kind": kind,
        "schema": _schema_for_kind(kind),
        "scenario": _scenario_for_kind(kind),
        "npz_members": list(_npz_members_for_kind(kind)),
    }


def _validate_registry(value: Any, count: int) -> None:
    if not isinstance(value, list) or len(value) != count or count < 1:
        raise PredictorPackageError("SOMP-H registered class registry size mismatch")
    handles: set[str] = set()
    for expected_index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"class_index", "class_handle"}:
            raise PredictorPackageError("SOMP-H registered class registry schema mismatch")
        if item["class_index"] != expected_index:
            raise PredictorPackageError("SOMP-H registered class registry index drift")
        handle = item["class_handle"]
        if (
            not isinstance(handle, str)
            or not handle.startswith("cls_")
            or OPAQUE_TOKEN_RE.fullmatch(handle) is None
            or handle in handles
        ):
            raise PredictorPackageError("SOMP-H registered class handle is not unique opaque")
        handles.add(handle)


def _validate_stage_state(stage: Any, registration_state: Any) -> None:
    if stage not in {"stage2b", "stage2c"}:
        raise PredictorPackageError("SOMP-H stage must be stage2b or stage2c")
    if registration_state not in REGISTRATION_STATES:
        raise PredictorPackageError("SOMP-H registration state invalid")
    if stage == "stage2b" and registration_state != "before":
        raise PredictorPackageError("Stage2-B permits only the before registry")


def _validate_descriptors(value: Any, *, profile: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise PredictorPackageError("SOMP-H members must be a list")
    expected_kinds = _profile_kinds(profile)
    if len(value) != len(expected_kinds):
        raise PredictorPackageError("SOMP-H physical profile member count mismatch")
    checked: list[dict[str, Any]] = []
    paths: set[str] = set()
    for expected_kind, raw in zip(expected_kinds, value):
        if not isinstance(raw, dict) or set(raw) != MEMBER_DESCRIPTOR_KEYS:
            raise PredictorPackageError("SOMP-H member descriptor exact schema mismatch")
        item = dict(raw)
        item["relative_path"] = validate_relative_member_path(item["relative_path"])
        if item["kind"] != expected_kind:
            raise PredictorPackageError("SOMP-H physical profile kind/order mismatch")
        if item["relative_path"] != _relative_path_for_kind(expected_kind):
            raise PredictorPackageError("SOMP-H physical profile path mismatch")
        if item["schema"] != _schema_for_kind(expected_kind):
            raise PredictorPackageError("SOMP-H member schema mismatch")
        if item["scenario"] != _scenario_for_kind(expected_kind):
            raise PredictorPackageError("SOMP-H member scenario mismatch")
        if item["npz_members"] != list(_npz_members_for_kind(expected_kind)):
            raise PredictorPackageError("SOMP-H NPZ allowlist descriptor mismatch")
        if item["relative_path"] in paths:
            raise PredictorPackageError("duplicate SOMP-H member path")
        if not _is_sha256(item["sha256"]):
            raise PredictorPackageError("invalid SOMP-H member SHA256")
        if not isinstance(item["size_bytes"], int) or item["size_bytes"] < 0:
            raise PredictorPackageError("invalid SOMP-H member size")
        paths.add(item["relative_path"])
        checked.append(item)
    return checked


def _validate_method_lock(payload: dict[str, Any], *, checkpoint_sha256: str) -> None:
    expected = expected_somph_method_lock()
    expected["checkpoint_sha256"] = checkpoint_sha256
    if set(payload) != set(expected):
        raise PredictorPackageError("SOMP-H method lock exact schema drift")
    failed = [
        key for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
    if failed:
        raise PredictorPackageError(f"SOMP-H method lock contract failed: {failed}")


def _validate_enrollment_binding(
    payload: dict[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> str:
    if set(payload) != ENROLLMENT_BINDING_KEYS:
        raise PredictorPackageError("SOMP-H enrollment binding exact schema drift")
    if payload.get("schema") != SOMPH_ENROLLMENT_BINDING_SCHEMA:
        raise PredictorPackageError("SOMP-H enrollment binding schema drift")
    if payload.get("stage") not in {"stage2b", "stage2c"}:
        raise PredictorPackageError("SOMP-H enrollment binding stage drift")
    if payload.get("registration_state") not in REGISTRATION_STATES:
        raise PredictorPackageError("SOMP-H enrollment binding state drift")
    if payload["stage"] == "stage2b" and payload["registration_state"] != "before":
        raise PredictorPackageError("Stage2-B head binding must be before")
    if not isinstance(payload.get("receiver"), str) or not payload["receiver"]:
        raise PredictorPackageError("SOMP-H enrollment binding receiver drift")
    for field in ("seed", "k_shot"):
        if (
            not isinstance(payload.get(field), int)
            or isinstance(payload[field], bool)
            or payload[field] < 1
        ):
            raise PredictorPackageError(
                f"SOMP-H enrollment binding integer drift: {field}"
            )
    handles = payload.get("registered_class_handles")
    if (
        not isinstance(handles, list)
        or len(handles) < 1
        or len(handles) != len(set(handles))
        or any(
            not isinstance(value, str)
            or not value.startswith("cls_")
            or OPAQUE_TOKEN_RE.fullmatch(value) is None
            for value in handles
        )
    ):
        raise PredictorPackageError("SOMP-H enrollment binding registry drift")
    if (
        payload.get("phase1_checkpoint_sha256")
        != ADV3B02_PHASE1_CHECKPOINT_SHA256
    ):
        raise PredictorPackageError(
            "SOMP-H enrollment binding Phase1 checkpoint lineage drift"
        )
    if not _is_sha256(payload.get("feature_runtime_sha256")):
        raise PredictorPackageError(
            "SOMP-H enrollment binding feature runtime SHA256 drift"
        )
    for field in (
        "enrollment_package_root_sha256",
        "enrollment_package_seal_sha256",
        "method_lock_sha256",
    ):
        if not _is_sha256(payload.get(field)):
            raise PredictorPackageError(
                f"SOMP-H enrollment binding SHA256 drift: {field}"
            )
    for field in (
        "support_token_sha256_by_scenario",
        "support_feature_sha256_by_scenario",
    ):
        values = payload.get(field)
        if (
            not isinstance(values, dict)
            or tuple(values) != FORMAL_LEO_WEAK_SCENARIOS
            or any(not _is_sha256(values[scenario]) for scenario in values)
        ):
            raise PredictorPackageError(
                f"SOMP-H enrollment binding scenario digest drift: {field}"
            )
    digest = sha256_bytes(canonical_json_bytes(payload))
    if manifest is not None:
        expected_handles = [
            item["class_handle"] for item in manifest["registered_classes"]
        ]
        for field in ("stage", "registration_state", "receiver", "seed", "k_shot"):
            if payload[field] != manifest[field]:
                raise PredictorPackageError(
                    f"SOMP-H head/manifest binding mismatch: {field}"
                )
        if handles != expected_handles:
            raise PredictorPackageError(
                "SOMP-H head/manifest class registry mismatch"
            )
        if payload["method_lock_sha256"] != manifest["method_lock_sha256"]:
            raise PredictorPackageError(
                "SOMP-H head/manifest method lock mismatch"
            )
        for field in ("phase1_checkpoint_sha256", "feature_runtime_sha256"):
            if payload[field] != manifest[field]:
                raise PredictorPackageError(
                    f"SOMP-H head/manifest runtime lineage mismatch: {field}"
                )
        if digest != manifest["head_enrollment_binding_sha256"]:
            raise PredictorPackageError(
                "SOMP-H head enrollment binding digest mismatch"
            )
    return digest


def _validate_provenance(
    payload: dict[str, Any],
    *,
    profile: str,
    receiver: str,
    seed: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    if set(payload) != PROVENANCE_KEYS:
        raise PredictorPackageError("SOMP-H overlay provenance exact schema mismatch")
    expected = {
        "schema": SOMPH_OVERLAY_PROVENANCE_SCHEMA,
        "profile": profile,
        "receiver": receiver,
        "seed": seed,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise PredictorPackageError("SOMP-H overlay provenance package binding mismatch")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise PredictorPackageError("SOMP-H overlay provenance samples missing")
    expected_prefix = "sid_" if profile == ENROLLMENT_ONLY else "qid_"
    by_scenario: dict[str, dict[str, dict[str, Any]]] = {
        value: {} for value in FORMAL_LEO_WEAK_SCENARIOS
    }
    overlay_tokens: set[str] = set()
    for raw in samples:
        if not isinstance(raw, dict) or set(raw) != PROVENANCE_SAMPLE_KEYS:
            raise PredictorPackageError("SOMP-H sample provenance exact schema mismatch")
        token = raw["sample_token"]
        overlay = raw["overlay_token"]
        scenario = raw["scenario"]
        if (
            not isinstance(token, str)
            or not token.startswith(expected_prefix)
            or OPAQUE_TOKEN_RE.fullmatch(token) is None
        ):
            raise PredictorPackageError("SOMP-H provenance sample token is not opaque")
        if (
            not isinstance(overlay, str)
            or not overlay.startswith("oid_")
            or OPAQUE_TOKEN_RE.fullmatch(overlay) is None
            or overlay in overlay_tokens
        ):
            raise PredictorPackageError("SOMP-H provenance overlay token is not unique opaque")
        if scenario not in FORMAL_LEO_WEAK_SCENARIOS:
            raise PredictorPackageError("SOMP-H provenance scenario is not formal LEO_weak")
        if token in by_scenario[scenario]:
            raise PredictorPackageError("duplicate SOMP-H provenance sample token")
        if not isinstance(raw["satellite_seed"], int):
            raise PredictorPackageError("SOMP-H provenance satellite seed invalid")
        for key in (
            "post_channel_iq_sha256",
            "source_leo_cache_sha256",
            "source_leo_provenance_sha256",
        ):
            if not _is_sha256(raw[key]):
                raise PredictorPackageError(f"SOMP-H provenance SHA256 invalid: {key}")
        overlay_tokens.add(overlay)
        by_scenario[scenario][token] = dict(raw)
    if any(not by_scenario[value] for value in FORMAL_LEO_WEAK_SCENARIOS):
        raise PredictorPackageError("SOMP-H provenance lacks one or more LEO_weak scenarios")
    return by_scenario


def _validate_manifest(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if set(payload) != MANIFEST_KEYS:
        raise PredictorPackageError("SOMP-H bundle manifest exact schema mismatch")
    expected = {
        "schema": SOMPH_BUNDLE_MANIFEST_SCHEMA,
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "phase1_checkpoint_sha256": ADV3B02_PHASE1_CHECKPOINT_SHA256,
        **PHASE2_FULL_CONTRACT,
    }
    failed = [key for key, value in expected.items() if payload.get(key) != value]
    if failed:
        raise PredictorPackageError(f"SOMP-H bundle contract failed before IQ: {failed}")
    profile = payload.get("profile")
    if profile not in PROFILE_VALUES:
        raise PredictorPackageError("SOMP-H bundle profile invalid")
    _validate_stage_state(payload.get("stage"), payload.get("registration_state"))
    if not isinstance(payload.get("receiver"), str) or not payload["receiver"]:
        raise PredictorPackageError("SOMP-H receiver invalid")
    if not isinstance(payload.get("seed"), int):
        raise PredictorPackageError("SOMP-H seed invalid")
    if (
        not isinstance(payload.get("k_shot"), int)
        or isinstance(payload["k_shot"], bool)
        or payload["k_shot"] < 1
        or payload["k_shot"] > SUPPORT_POOL_MAX_K
    ):
        raise PredictorPackageError("SOMP-H K is outside the sealed K20 pool")
    support_pool_k = payload.get("support_pool_max_k")
    if (
        not isinstance(support_pool_k, int)
        or isinstance(support_pool_k, bool)
        or support_pool_k != payload["k_shot"]
    ):
        raise PredictorPackageError(
            "SOMP-H reachable support pool must equal the declared K"
        )
    count = payload.get("registered_class_count")
    if not isinstance(count, int):
        raise PredictorPackageError("SOMP-H registered class count invalid")
    _validate_registry(payload.get("registered_classes"), count)
    if not _is_sha256(payload.get("method_lock_sha256")):
        raise PredictorPackageError("SOMP-H method lock SHA256 invalid")
    if not _is_sha256(payload.get("feature_runtime_sha256")):
        raise PredictorPackageError("SOMP-H feature runtime SHA256 invalid")
    if payload["feature_runtime_sha256"] == payload["phase1_checkpoint_sha256"]:
        raise PredictorPackageError(
            "Phase1 state-dict checkpoint cannot be the sealed feature runtime"
        )
    if not _is_sha256(payload.get("overlay_provenance_sha256")):
        raise PredictorPackageError("SOMP-H overlay provenance SHA256 invalid")
    head_binding = payload.get("head_enrollment_binding_sha256")
    head_capsule = payload.get("head_capsule_sha256")
    row_handle = payload.get("row_handle")
    row_manifest_sha256 = payload.get("row_manifest_sha256")
    if payload["profile"] == ENROLLMENT_ONLY:
        if (
            head_binding is not None
            or head_capsule is not None
            or row_handle is not None
            or row_manifest_sha256 is not None
        ):
            raise PredictorPackageError(
                "SOMP-H enrollment package must not claim apply-only bindings"
            )
    else:
        if not _is_sha256(head_binding):
            raise PredictorPackageError(
                "SOMP-H apply package lacks a head enrollment binding"
            )
        if not _is_sha256(head_capsule):
            raise PredictorPackageError(
                "SOMP-H apply package lacks an enrollment head trust root"
            )
        if not _is_row_handle(row_handle):
            raise PredictorPackageError(
                "SOMP-H apply package lacks an opaque row handle"
            )
        if (
            not isinstance(row_manifest_sha256, str)
            or SHA256_RE.fullmatch(row_manifest_sha256) is None
        ):
            raise PredictorPackageError(
                "SOMP-H apply package lacks a strict row manifest SHA256"
            )
    members = _validate_descriptors(payload.get("members"), profile=profile)
    if payload["package_root_sha256"] != package_root_sha256(members):
        raise PredictorPackageError("SOMP-H package root digest mismatch")
    return members


def _open_detached_seal(path: Path) -> tuple[dict[str, Any], str, int]:
    if path.is_symlink() or not path.is_file():
        raise PredictorPackageError("SOMP-H detached seal must be a regular file")
    before = path.stat(follow_symlinks=False)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise PredictorPackageError("SOMP-H detached seal is not regular")
        if (before.st_dev, before.st_ino, before.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise PredictorPackageError("SOMP-H detached seal identity changed")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            digest, size = _hash_handle(handle)
            payload = _json_from_handle(handle, context="SOMP-H detached seal")
    finally:
        os.close(descriptor)
    return payload, digest, size


def _verify_regular_member(
    root: Path,
    descriptor: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    with open_regular_member_same_fd(root, descriptor["relative_path"]) as handle:
        digest, size = _hash_handle(handle)
        if digest != descriptor["sha256"] or size != descriptor["size_bytes"]:
            raise PredictorPackageError(
                f"SOMP-H member digest mismatch: {descriptor['relative_path']}"
            )
        payload = None
        if descriptor["schema"] in {
            SOMPH_METHOD_LOCK_SCHEMA,
            SOMPH_OVERLAY_PROVENANCE_SCHEMA,
        }:
            payload = _json_from_handle(handle, context=descriptor["relative_path"])
    return payload, {
        "relative_path": descriptor["relative_path"],
        "sha256": digest,
        "size_bytes": size,
        "status": "PASS",
    }


def _inspect_iq_member(root: Path, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Hash and inspect the ZIP allowlist without NumPy IQ materialization."""

    with open_regular_member_same_fd(root, descriptor["relative_path"]) as handle:
        digest, size = _hash_handle(handle)
        if digest != descriptor["sha256"] or size != descriptor["size_bytes"]:
            raise PredictorPackageError(
                f"SOMP-H member digest mismatch: {descriptor['relative_path']}"
            )
        actual = _zip_members_from_handle(handle, context=descriptor["relative_path"])
    if actual != tuple(descriptor["npz_members"]):
        raise PredictorPackageError(
            f"SOMP-H NPZ member allowlist mismatch: {descriptor['relative_path']}"
        )
    return {
        "relative_path": descriptor["relative_path"],
        "sha256": digest,
        "size_bytes": size,
        "status": "PASS",
    }


def _load_head_capsule_member(
    root: Path,
    descriptor: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    with open_regular_member_same_fd(root, descriptor["relative_path"]) as handle:
        digest, size = _hash_handle(handle)
        if digest != descriptor["sha256"] or size != descriptor["size_bytes"]:
            raise PredictorPackageError("SOMP-H head capsule digest mismatch")
        actual = _zip_members_from_handle(
            handle, context=descriptor["relative_path"]
        )
        if actual != tuple(descriptor["npz_members"]):
            raise PredictorPackageError("SOMP-H head capsule member drift")
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in actual}
    raw_binding = np.asarray(arrays["enrollment_binding_json_utf8"])
    if raw_binding.dtype != np.uint8 or raw_binding.ndim != 1:
        raise PredictorPackageError(
            "SOMP-H head enrollment binding must be a uint8 vector"
        )
    try:
        binding = json.loads(raw_binding.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictorPackageError(
            "SOMP-H head enrollment binding JSON invalid"
        ) from exc
    if not isinstance(binding, dict):
        raise PredictorPackageError("SOMP-H head enrollment binding is not an object")
    binding_sha256 = _validate_enrollment_binding(binding)
    return arrays, binding, binding_sha256


def write_somph_predictor_bundle(
    package_root: str | Path,
    *,
    profile: str,
    stage: str,
    registration_state: str,
    receiver: str,
    seed: int,
    k_shot: int,
    registered_classes: list[Mapping[str, Any]],
    expected_method_lock_sha256: str,
    expected_overlay_provenance_sha256: str,
    detached_seal_path: str | Path,
    expected_head_enrollment_binding_sha256: str | None = None,
    expected_head_capsule_sha256: str | None = None,
    expected_row_handle: str | None = None,
    expected_row_manifest_sha256: str | None = None,
    support_pool_max_k: int | None = None,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    """Seal one physically isolated SOMP-H predictor profile."""

    root = _ensure_root(Path(package_root))
    _validate_stage_state(stage, registration_state)
    if profile not in PROFILE_VALUES:
        raise PredictorPackageError("SOMP-H bundle profile invalid")
    if not _is_sha256(expected_method_lock_sha256):
        raise PredictorPackageError("expected SOMP-H method lock SHA256 invalid")
    if not _is_sha256(expected_overlay_provenance_sha256):
        raise PredictorPackageError(
            "expected SOMP-H overlay provenance SHA256 invalid"
        )
    if (
        not isinstance(k_shot, int)
        or isinstance(k_shot, bool)
        or k_shot < 1
        or k_shot > SUPPORT_POOL_MAX_K
    ):
        raise PredictorPackageError("SOMP-H K is outside the sealed K20 pool")
    if support_pool_max_k is None:
        support_pool_max_k = k_shot
    if (
        not isinstance(support_pool_max_k, int)
        or isinstance(support_pool_max_k, bool)
        or support_pool_max_k != k_shot
    ):
        raise PredictorPackageError(
            "SOMP-H reachable support pool must equal the declared K"
        )
    if profile == ENROLLMENT_ONLY:
        if (
            expected_head_enrollment_binding_sha256 is not None
            or expected_head_capsule_sha256 is not None
            or expected_row_handle is not None
            or expected_row_manifest_sha256 is not None
        ):
            raise PredictorPackageError(
                "enrollment package must not accept apply-only bindings"
            )
    else:
        if not _is_sha256(expected_head_enrollment_binding_sha256):
            raise PredictorPackageError("apply package requires a head binding SHA256")
        if not _is_sha256(expected_head_capsule_sha256):
            raise PredictorPackageError(
                "apply package requires an enrollment head capsule SHA256"
            )
        if not _is_row_handle(expected_row_handle):
            raise PredictorPackageError(
                "apply package requires an opaque row handle"
            )
        if (
            not isinstance(expected_row_manifest_sha256, str)
            or SHA256_RE.fullmatch(expected_row_manifest_sha256) is None
        ):
            raise PredictorPackageError(
                "apply package requires a strict row manifest SHA256"
            )
    seal_path = Path(detached_seal_path).resolve()
    try:
        seal_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise PredictorPackageError("SOMP-H detached seal must be outside package root")

    kinds = _profile_kinds(profile)
    expected_paths = {_relative_path_for_kind(value) for value in kinds}
    _validate_package_root_exact_allowlist(root, allowed_files=expected_paths)
    members = [_descriptor(root, value) for value in kinds]
    by_kind = {item["kind"]: item for item in members}
    feature_runtime_sha256 = by_kind["feature_runtime"]["sha256"]
    if feature_runtime_sha256 == ADV3B02_PHASE1_CHECKPOINT_SHA256:
        raise PredictorPackageError(
            "Phase1 state-dict checkpoint cannot be packaged as TorchScript runtime"
        )
    if by_kind["method_lock"]["sha256"] != expected_method_lock_sha256:
        raise PredictorPackageError("SOMP-H method lock SHA256 mismatch")

    method_lock, _ = _verify_regular_member(root, by_kind["method_lock"])
    assert method_lock is not None
    _validate_method_lock(
        method_lock, checkpoint_sha256=ADV3B02_PHASE1_CHECKPOINT_SHA256
    )
    if (
        by_kind["method_lock"]["sha256"]
        != sha256_bytes(canonical_json_bytes(method_lock))
    ):
        raise PredictorPackageError(
            "SOMP-H method lock must use canonical JSON bytes"
        )
    provenance, _ = _verify_regular_member(root, by_kind["overlay_provenance"])
    assert provenance is not None
    if (
        by_kind["overlay_provenance"]["sha256"]
        != expected_overlay_provenance_sha256
    ):
        raise PredictorPackageError(
            "SOMP-H overlay provenance does not match external trust root"
        )
    provenance_index = _validate_provenance(
        provenance, profile=profile, receiver=receiver, seed=seed
    )
    if profile == APPLY_ONLY:
        _arrays, head_binding, head_binding_sha256 = _load_head_capsule_member(
            root, by_kind["head_capsule"]
        )
        if by_kind["head_capsule"]["sha256"] != expected_head_capsule_sha256:
            raise PredictorPackageError(
                "SOMP-H apply head capsule does not match enrollment trust root"
            )
        if head_binding_sha256 != expected_head_enrollment_binding_sha256:
            raise PredictorPackageError(
                "SOMP-H apply head binding does not match the external trust root"
            )
        try:
            validate_somph_head_capsule(
                _arrays,
                method_lock=method_lock,
                expected_enrollment_binding_sha256=head_binding_sha256,
            )
        except ValueError as exc:
            raise PredictorPackageError(
                "SOMP-H apply head semantic validation failed"
            ) from exc
    for kind in kinds:
        if kind.startswith(("support:", "query:")):
            _inspect_iq_member(root, by_kind[kind])

    count = len(registered_classes)
    registry = [dict(value) for value in registered_classes]
    _validate_registry(registry, count)
    if profile == ENROLLMENT_ONLY:
        support_manifest_context = {
            "registration_state": registration_state,
            "registered_class_count": count,
            "support_pool_max_k": support_pool_max_k,
        }
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            arrays, embedded = _materialize_iq(
                root, by_kind[f"support:{scenario}"]
            )
            _validate_support_payload(
                arrays,
                embedded,
                manifest=support_manifest_context,
                scenario=scenario,
                provenance=provenance_index[scenario],
            )
    root_digest = package_root_sha256(members)
    manifest = {
        "schema": SOMPH_BUNDLE_MANIFEST_SCHEMA,
        "profile": profile,
        "stage": stage,
        "registration_state": registration_state,
        "receiver": receiver,
        "seed": seed,
        "k_shot": k_shot,
        "support_pool_max_k": support_pool_max_k,
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "registered_class_count": count,
        "registered_classes": registry,
        "phase1_checkpoint_sha256": ADV3B02_PHASE1_CHECKPOINT_SHA256,
        "feature_runtime_sha256": feature_runtime_sha256,
        "method_lock_sha256": expected_method_lock_sha256,
        "overlay_provenance_sha256": expected_overlay_provenance_sha256,
        "head_capsule_sha256": (
            expected_head_capsule_sha256 if profile == APPLY_ONLY else None
        ),
        "head_enrollment_binding_sha256": (
            expected_head_enrollment_binding_sha256
            if profile == APPLY_ONLY
            else None
        ),
        "row_handle": expected_row_handle if profile == APPLY_ONLY else None,
        "row_manifest_sha256": (
            expected_row_manifest_sha256 if profile == APPLY_ONLY else None
        ),
        "members": members,
        "package_root_sha256": root_digest,
        **PHASE2_FULL_CONTRACT,
    }
    _validate_manifest(manifest)
    if profile == APPLY_ONLY:
        _validate_enrollment_binding(head_binding, manifest=manifest)
    manifest_path = root / MANIFEST_RELATIVE_PATH
    if manifest_path.exists() or seal_path.exists():
        raise FileExistsError("refusing to overwrite SOMP-H manifest or detached seal")
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    seal = {
        "schema": SOMPH_BUNDLE_SEAL_SCHEMA,
        "manifest_relative_path": MANIFEST_RELATIVE_PATH,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "manifest_size_bytes": len(manifest_bytes),
        "package_root_sha256": root_digest,
        "artifact_member_allowlist_sha256": root_digest,
    }
    manifest_created = False
    seal_created = False
    try:
        with manifest_path.open("xb") as handle:
            manifest_created = True
            handle.write(manifest_bytes)
        _validate_package_root_exact_allowlist(
            root, allowed_files=expected_paths | {MANIFEST_RELATIVE_PATH}
        )
        seal_path.parent.mkdir(parents=True, exist_ok=True)
        with seal_path.open("xb") as handle:
            seal_created = True
            handle.write(canonical_json_bytes(seal) + b"\n")
    except Exception:
        for owned_path, created in (
            (seal_path, seal_created),
            (manifest_path, manifest_created),
        ):
            if created and owned_path.exists():
                os.chmod(owned_path, 0o600)
                owned_path.unlink()
        raise
    return manifest_path, seal_path, manifest, seal


def _preflight(
    package_root: str | Path,
    *,
    detached_seal_path: str | Path,
    expected_seal_sha256: str,
    inspect_iq_members: bool = True,
    load_npz_control_members: bool = True,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, dict[str, Any]]],
]:
    root = _ensure_root(Path(package_root))
    seal, seal_digest, _seal_size = _open_detached_seal(Path(detached_seal_path))
    if not _is_sha256(expected_seal_sha256) or seal_digest != expected_seal_sha256:
        raise PredictorPackageError("SOMP-H detached seal digest mismatch")
    if set(seal) != SEAL_KEYS or seal.get("schema") != SOMPH_BUNDLE_SEAL_SCHEMA:
        raise PredictorPackageError("SOMP-H detached seal exact schema mismatch")
    if seal.get("manifest_relative_path") != MANIFEST_RELATIVE_PATH:
        raise PredictorPackageError("SOMP-H manifest path drift")

    with open_regular_member_same_fd(root, MANIFEST_RELATIVE_PATH) as handle:
        manifest_digest, manifest_size = _hash_handle(handle)
        if (
            manifest_digest != seal["manifest_sha256"]
            or manifest_size != seal["manifest_size_bytes"]
        ):
            raise PredictorPackageError("SOMP-H manifest detached digest mismatch")
        manifest = _json_from_handle(handle, context="SOMP-H package manifest")
    members = _validate_manifest(manifest)
    if manifest["package_root_sha256"] != seal["package_root_sha256"]:
        raise PredictorPackageError("SOMP-H manifest/seal package root mismatch")
    if seal["artifact_member_allowlist_sha256"] != seal["package_root_sha256"]:
        raise PredictorPackageError("SOMP-H package allowlist digest mismatch")
    _validate_package_root_exact_allowlist(
        root,
        allowed_files={item["relative_path"] for item in members}
        | {MANIFEST_RELATIVE_PATH},
    )
    by_kind = {item["kind"]: item for item in members}

    opened: list[dict[str, Any]] = []
    runtime_payload, receipt = _verify_regular_member(
        root, by_kind["feature_runtime"]
    )
    assert runtime_payload is None
    opened.append(receipt)
    if receipt["sha256"] != manifest["feature_runtime_sha256"]:
        raise PredictorPackageError(
            "sealed ADV3B02 feature runtime SHA256 mismatch before IQ"
        )
    if receipt["sha256"] == ADV3B02_PHASE1_CHECKPOINT_SHA256:
        raise PredictorPackageError(
            "Phase1 state-dict checkpoint reached the Phase2 runtime package"
        )

    method_lock, receipt = _verify_regular_member(root, by_kind["method_lock"])
    opened.append(receipt)
    assert method_lock is not None
    if receipt["sha256"] != manifest["method_lock_sha256"]:
        raise PredictorPackageError("SOMP-H method lock digest binding mismatch")
    _validate_method_lock(
        method_lock, checkpoint_sha256=ADV3B02_PHASE1_CHECKPOINT_SHA256
    )
    if receipt["sha256"] != sha256_bytes(canonical_json_bytes(method_lock)):
        raise PredictorPackageError(
            "SOMP-H method lock is not the canonical JSON trust root"
        )

    provenance, receipt = _verify_regular_member(root, by_kind["overlay_provenance"])
    opened.append(receipt)
    assert provenance is not None
    if receipt["sha256"] != manifest["overlay_provenance_sha256"]:
        raise PredictorPackageError(
            "SOMP-H overlay provenance trust root mismatch"
        )
    provenance_index = _validate_provenance(
        provenance,
        profile=manifest["profile"],
        receiver=manifest["receiver"],
        seed=manifest["seed"],
    )

    if manifest["profile"] == APPLY_ONLY:
        if by_kind["head_capsule"]["sha256"] != manifest["head_capsule_sha256"]:
            raise PredictorPackageError(
                "SOMP-H apply head capsule trust root mismatch"
            )
        if load_npz_control_members:
            _head_arrays, head_binding, head_binding_sha256 = (
                _load_head_capsule_member(root, by_kind["head_capsule"])
            )
            _validate_enrollment_binding(head_binding, manifest=manifest)
            if head_binding_sha256 != manifest["head_enrollment_binding_sha256"]:
                raise PredictorPackageError(
                    "SOMP-H apply head binding trust root mismatch"
                )
            try:
                validate_somph_head_capsule(
                    _head_arrays,
                    method_lock=method_lock,
                    expected_enrollment_binding_sha256=head_binding_sha256,
                )
            except ValueError as exc:
                raise PredictorPackageError(
                    "SOMP-H apply head semantic validation failed before query IQ"
                ) from exc
            opened.append(
                {
                    "relative_path": by_kind["head_capsule"]["relative_path"],
                    "sha256": by_kind["head_capsule"]["sha256"],
                    "size_bytes": by_kind["head_capsule"]["size_bytes"],
                    "status": "PASS",
                    "enrollment_binding_sha256": head_binding_sha256,
                }
            )
        else:
            _unused, receipt = _verify_regular_member(
                root, by_kind["head_capsule"]
            )
            assert _unused is None
            opened.append(receipt)

    if inspect_iq_members:
        # This is intentionally after checkpoint, method-lock, and provenance
        # validation.  It decompresses/CRC-checks the archives but does not
        # np.load.  The authority-aware entry point disables even this open.
        prefix = (
            "support:" if manifest["profile"] == ENROLLMENT_ONLY else "query:"
        )
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            opened.append(_inspect_iq_member(root, by_kind[prefix + scenario]))

    audit = {
        "schema": "cvs.phase2.somph_preopen_audit.v1",
        "status": "STRUCTURAL_SELF_CONSISTENCY_PASS",
        "profile": manifest["profile"],
        "package_root_sha256": seal["package_root_sha256"],
        "artifact_member_allowlist_sha256": seal[
            "artifact_member_allowlist_sha256"
        ],
        "manifest_sha256": manifest_digest,
        "opened_members": opened,
        "overlay_provenance_status": "STRUCTURAL_SELF_CONSISTENCY_PASS",
        "iq_payload_materialized": False,
        "hash_and_member_audit_same_file_descriptor": True,
        "phase2_protocol_evidence_status": (
            "STRUCTURAL_ONLY_REAL_INPUT_RECOMPUTE_REQUIRED"
        ),
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "control_state": "LOCAL_PROTOCOL_REPAIR_REQUIRED",
    }
    return manifest, seal, audit, provenance_index


def _external_json_with_expected_sha256(
    path: str | Path,
    *,
    expected_sha256: str,
    context: str,
) -> tuple[dict[str, Any], str]:
    candidate = Path(path).resolve(strict=True)
    if Path(path).is_symlink() or not candidate.is_file():
        raise PredictorPackageError(f"{context} must be a regular non-symlink file")
    if not _is_sha256(expected_sha256):
        raise PredictorPackageError(f"{context} external SHA256 is invalid")
    with open_regular_member_same_fd(candidate.parent, candidate.name) as handle:
        digest, _size = _hash_handle(handle)
        if digest != expected_sha256:
            raise PredictorPackageError(f"{context} external SHA256 mismatch")
        payload = _json_from_handle(handle, context=context)
    return payload, digest


def _external_json_actual(path: str | Path, *, context: str) -> tuple[dict[str, Any], str]:
    candidate = Path(path).resolve(strict=True)
    if Path(path).is_symlink() or not candidate.is_file():
        raise PredictorPackageError(f"{context} must be a regular non-symlink file")
    with open_regular_member_same_fd(candidate.parent, candidate.name) as handle:
        digest, _size = _hash_handle(handle)
        payload = _json_from_handle(handle, context=context)
    return payload, digest


def _code_closure() -> tuple[list[dict[str, str]], str]:
    paths = {
        "somph_predictor_bundle.py": Path(__file__).resolve(),
        "somph_lineage_authority.py": Path(lineage_authority.__file__).resolve(),
        "stage2_predictor_bundle.py": Path(stage2_bundle_module.__file__).resolve(),
        "somph_formal_matrix.py": Path(formal_matrix.__file__).resolve(),
        "leo_weak_cache.py": Path(leo_weak_cache.__file__).resolve(),
    }
    if tuple(paths) != CODE_CLOSURE_LOGICAL_MEMBERS:
        raise PredictorPackageError("SOMP-H authority code closure order drift")
    members = [
        {"logical_name": name, "sha256": sha256_file(path)}
        for name, path in paths.items()
    ]
    return members, sha256_bytes(canonical_json_bytes(members))


def _validate_formal_policy(policy: Mapping[str, Any]) -> None:
    expected = {
        "schema": SOMPH_FORMAL_POLICY_SCHEMA,
        "status": SOMPH_FORMAL_POLICY_STATUS,
        "formal_receivers": list(FORMAL_RECEIVERS),
        "old_tx_ids": list(OLD_TX_IDS),
        "nested_new_tx_ids": [
            list(NEW_TX_IDS[:count]) for count in FORMAL_NEW_CLASS_COUNTS
        ],
        "cache_scope": "stage2_registered",
        "old_dataset_basename": "ManySig.pkl",
        "new_dataset_basename": "ManyTx.pkl",
        "physical_sample_scenario_assignment_policy": (
            lineage_authority.PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY
        ),
        "single_observation_contract": (
            lineage_authority.PHASE2_SINGLE_OBSERVATION_CONTRACT
        ),
        "required_code_closure_members": list(CODE_CLOSURE_LOGICAL_MEMBERS),
    }
    if set(policy) != FORMAL_POLICY_KEYS or policy != expected:
        raise PredictorPackageError("SOMP-H actual formal policy contract drift")


def _package_control_roots(
    manifest: Mapping[str, Any],
    provenance_index: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    new_tx_ids: list[str],
) -> dict[str, Any]:
    class_root = sha256_bytes(
        canonical_json_bytes(manifest["registered_classes"])
    )
    roles = {
        "target_old_class_count": len(OLD_TX_IDS),
        "target_new_class_count": (
            0 if manifest["registration_state"] == "before" else len(new_tx_ids)
        ),
        "registration_state": manifest["registration_state"],
    }
    role_root = sha256_bytes(canonical_json_bytes(roles))
    physical_roots: dict[str, str] = {}
    overlay_roots: dict[str, str] = {}
    iq_roots: dict[str, str] = {}
    seed_roots: dict[str, str] = {}
    materialized_assignment_roots: dict[str, str] = {}
    assignment_rows: list[dict[str, Any]] = []
    expected_rows = manifest["registered_class_count"] * manifest["k_shot"]
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        rows = provenance_index.get(scenario)
        if not isinstance(rows, dict) or len(rows) != expected_rows:
            raise PredictorPackageError(
                "SOMP-H provenance row count cannot prove exact-K before IQ"
            )
        ordered = [rows[token] for token in sorted(rows)]
        physical_roots[scenario] = sha256_bytes(
            canonical_json_bytes([item["sample_token"] for item in ordered])
        )
        overlay_roots[scenario] = sha256_bytes(
            canonical_json_bytes([item["overlay_token"] for item in ordered])
        )
        iq_roots[scenario] = sha256_bytes(
            canonical_json_bytes(
                [item["post_channel_iq_sha256"] for item in ordered]
            )
        )
        seed_roots[scenario] = sha256_bytes(
            canonical_json_bytes([int(item["satellite_seed"]) for item in ordered])
        )
        materialized_assignment_roots[scenario] = sha256_bytes(
            canonical_json_bytes(
                [
                    {
                        "sample_token": item["sample_token"],
                        "overlay_token": item["overlay_token"],
                        "satellite_seed": int(item["satellite_seed"]),
                        "post_channel_iq_sha256": item[
                            "post_channel_iq_sha256"
                        ],
                    }
                    for item in ordered
                ]
            )
        )
        assignment_rows.extend(
            {
                key: item[key]
                for key in (
                    "sample_token",
                    "scenario",
                    "overlay_token",
                    "satellite_seed",
                    "post_channel_iq_sha256",
                    "source_leo_cache_sha256",
                    "source_leo_provenance_sha256",
                )
            }
            for item in ordered
        )
    return {
        "package_class_registry_sha256": class_root,
        "package_role_registry_sha256": role_root,
        "package_physical_sample_ids_sha256_by_scenario": physical_roots,
        "package_overlay_ids_sha256_by_scenario": overlay_roots,
        "package_post_channel_iq_sha256_by_scenario": iq_roots,
        "package_satellite_seed_sha256_by_scenario": seed_roots,
        "package_materialized_assignment_sha256_by_scenario": (
            materialized_assignment_roots
        ),
        "package_sample_assignment_sha256": sha256_bytes(
            canonical_json_bytes(assignment_rows)
        ),
    }


def _policy_signature_message(envelope: Mapping[str, Any]) -> bytes:
    signed = {key: envelope[key] for key in SIGNED_POLICY_ENVELOPE_KEYS if key != "signature_ed25519_hex"}
    return (
        SOMPH_SIGNED_POLICY_ENVELOPE_DOMAIN.encode("ascii")
        + b"\x00"
        + canonical_json_bytes(signed)
    )


def _make_signed_policy_envelope_verifier(
    public_key: bytes,
    *,
    expected_public_key_sha256: str,
) -> Callable[[Mapping[str, Any], Mapping[str, Any]], None]:
    pinned_public_key = bytes(public_key)
    pinned_public_key_sha256 = str(expected_public_key_sha256)
    pinned_issuer = str(lineage_authority.PINNED_AUTHORITY_ISSUER)
    pinned_key_id = str(lineage_authority.PINNED_AUTHORITY_KEY_ID)
    if (
        len(pinned_public_key) != 32
        or hashlib.sha256(pinned_public_key).hexdigest()
        != pinned_public_key_sha256
    ):
        raise PredictorPackageError("SOMP-H pinned policy public key SHA drift")

    def verify(
        envelope: Mapping[str, Any], expected: Mapping[str, Any]
    ) -> None:
        if set(envelope) != SIGNED_POLICY_ENVELOPE_KEYS:
            raise PredictorPackageError(
                "SOMP-H signed policy envelope exact schema drift"
            )
        pinned = {
            "schema": SOMPH_SIGNED_POLICY_ENVELOPE_SCHEMA,
            "domain": SOMPH_SIGNED_POLICY_ENVELOPE_DOMAIN,
            "issuer": pinned_issuer,
            "key_id": pinned_key_id,
            **dict(expected),
        }
        if any(envelope.get(key) != value for key, value in pinned.items()):
            raise PredictorPackageError(
                "SOMP-H signed policy envelope binding drift"
            )
        try:
            signature = bytes.fromhex(str(envelope["signature_ed25519_hex"]))
        except ValueError as exc:
            raise PredictorPackageError(
                "SOMP-H signed policy envelope hex invalid"
            ) from exc
        try:
            lineage_authority._verify_ed25519(
                pinned_public_key,
                _policy_signature_message(envelope),
                signature,
            )
        except Exception as exc:
            raise PredictorPackageError(
                "SOMP-H signed policy authorization invalid"
            ) from exc

    return verify


_DEFAULT_SIGNED_POLICY_ENVELOPE_VERIFIER = _make_signed_policy_envelope_verifier(
    bytes.fromhex(lineage_authority.PINNED_AUTHORITY_PUBLIC_KEY_HEX),
    expected_public_key_sha256=(
        lineage_authority.PINNED_AUTHORITY_PUBLIC_KEY_SHA256
    ),
)


def _make_test_signed_policy_envelope_verifier(
    public_key: bytes,
) -> Callable[[Mapping[str, Any], Mapping[str, Any]], None]:
    """Build an explicit synthetic-key verifier without mutating trust globals."""

    return _make_signed_policy_envelope_verifier(
        bytes(public_key),
        expected_public_key_sha256=hashlib.sha256(bytes(public_key)).hexdigest(),
    )


def _scenario_sha_map(value: Any, *, field: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise PredictorPackageError(f"SOMP-H authority {field} scenario schema drift")
    result = {str(key): str(item) for key, item in value.items()}
    if any(not _is_sha256(item) for item in result.values()):
        raise PredictorPackageError(f"SOMP-H authority {field} SHA256 drift")
    return result


def _dataset_basename(value: Any) -> str:
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def _authority_member_sha256(commit: Mapping[str, Any], name: str) -> str:
    members = commit.get("members")
    if not isinstance(members, list):
        raise PredictorPackageError("SOMP-H authority commit member registry missing")
    matches = [
        item
        for item in members
        if isinstance(item, dict) and item.get("name") == name
    ]
    if len(matches) != 1 or not _is_sha256(matches[0].get("sha256")):
        raise PredictorPackageError("SOMP-H authority commit member binding missing")
    return str(matches[0]["sha256"])


def _validate_formal_lineage_authority(
    *,
    manifest: Mapping[str, Any],
    seal: Mapping[str, Any],
    provenance_index: Mapping[str, Mapping[str, Mapping[str, Any]]],
    authority_lock: Mapping[str, Any],
    authority_attestation: Mapping[str, Any],
    authority_commit: Mapping[str, Any],
    expected_authority_commit_sha256: str,
    expected_package_detached_seal_sha256: str,
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    actual_formal_policy_sha256: str,
    code_closure_sha256: str,
    package_control_roots: Mapping[str, Any],
) -> dict[str, Any]:
    receiver = manifest.get("receiver")
    seed = manifest.get("seed")
    if receiver not in FORMAL_RECEIVERS:
        raise PredictorPackageError(
            "SOMP-H formal authority requires a formal receiver"
        )
    if (
        authority_lock.get("receiver") != receiver
        or authority_lock.get("seed") != seed
        or authority_lock.get("cache_scope") != "stage2_registered"
    ):
        raise PredictorPackageError("SOMP-H package/authority row binding mismatch")
    old_tx_ids = authority_lock.get("old_tx_ids")
    new_tx_ids = authority_lock.get("new_tx_ids")
    if old_tx_ids != list(OLD_TX_IDS):
        raise PredictorPackageError("SOMP-H authority target-old TX set is not formal")
    if (
        not isinstance(new_tx_ids, list)
        or len(new_tx_ids) not in FORMAL_NEW_CLASS_COUNTS
        or new_tx_ids != list(NEW_TX_IDS[: len(new_tx_ids)])
    ):
        raise PredictorPackageError("SOMP-H authority target-new TX set is not formal")
    registration_state = manifest.get("registration_state")
    expected_stage = "stage2b" if registration_state == "before" else "stage2c"
    expected_class_count = len(old_tx_ids) + (
        0 if registration_state == "before" else len(new_tx_ids)
    )
    if (
        manifest.get("stage") != expected_stage
        or manifest.get("registered_class_count") != expected_class_count
    ):
        raise PredictorPackageError(
            "SOMP-H formal registration state/class-count binding drift"
        )
    datasets = authority_lock.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 2:
        raise PredictorPackageError("SOMP-H authority dataset registry missing")
    dataset_by_role = {
        item.get("role"): item for item in datasets if isinstance(item, dict)
    }
    old_dataset = dataset_by_role.get("target_old")
    new_dataset = dataset_by_role.get("target_new")
    if (
        not isinstance(old_dataset, dict)
        or not isinstance(new_dataset, dict)
        or _dataset_basename(old_dataset.get("path")) != "ManySig.pkl"
        or _dataset_basename(new_dataset.get("path")) != "ManyTx.pkl"
        or old_dataset.get("tx_ids") != old_tx_ids
        or new_dataset.get("tx_ids") != new_tx_ids
    ):
        raise PredictorPackageError(
            "SOMP-H formal authority requires ManySig-old and ManyTx-new"
        )
    for field, expected in lineage_authority.PHASE2_SINGLE_OBSERVATION_CONTRACT.items():
        if authority_lock.get(field) != expected or manifest.get(field) != expected:
            raise PredictorPackageError(
                f"SOMP-H single-observation authority drift: {field}"
            )
    if (
        authority_lock.get("physical_sample_scenario_assignment_policy")
        != lineage_authority.PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY
        or authority_lock.get("cross_scenario_physical_disjointness_audit") != "PASS"
        or authority_lock.get("single_observation_contract_audit") != "PASS"
    ):
        raise PredictorPackageError(
            "SOMP-H single-observation assignment authority failed"
        )
    cache_sha_by_scenario = _scenario_sha_map(
        authority_lock.get("cache_sha256_by_scenario"),
        field="cache_sha256_by_scenario",
    )
    for field in (
        "physical_sample_ids_sha256_by_scenario",
        "post_channel_iq_sha256_root_by_scenario",
        "overlay_ids_sha256_by_scenario",
    ):
        _scenario_sha_map(authority_lock.get(field), field=field)
    if not _is_sha256(
        authority_lock.get("physical_sample_scenario_assignment_sha256")
    ):
        raise PredictorPackageError("SOMP-H physical assignment SHA256 missing")
    structural_receipt_sha256 = authority_attestation.get(
        "structural_receipt_sha256"
    )
    if not _is_sha256(structural_receipt_sha256):
        raise PredictorPackageError("SOMP-H structural receipt authority missing")
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        rows = provenance_index.get(scenario)
        if not isinstance(rows, dict) or not rows:
            raise PredictorPackageError("SOMP-H package provenance scenario missing")
        for item in rows.values():
            if (
                item.get("source_leo_cache_sha256") != cache_sha_by_scenario[scenario]
                or item.get("source_leo_provenance_sha256")
                != structural_receipt_sha256
            ):
                raise PredictorPackageError(
                    "SOMP-H package provenance is detached from formal authority"
                )
    if set(authorization) != FORMAL_POLICY_AUTHORIZATION_KEYS:
        raise PredictorPackageError(
            "SOMP-H formal policy authorization exact schema drift"
        )
    attestation_sha256 = _authority_member_sha256(
        authority_commit, lineage_authority.AUTHORITY_ATTESTATION_NAME
    )
    for field in (
        "dataset_authority_root_sha256",
        "cache_role_inputs_root_sha256",
    ):
        source = (
            authority_attestation
            if field.startswith("dataset_")
            else authority_lock
        )
        if not _is_sha256(source.get(field)):
            raise PredictorPackageError(f"SOMP-H authority {field} missing")
    preflight_code_sha256 = sha256_file(Path(__file__).resolve())
    if not _is_sha256(actual_formal_policy_sha256):
        raise PredictorPackageError("SOMP-H actual formal policy SHA256 is invalid")
    expected = {
        "schema": SOMPH_FORMAL_POLICY_AUTHORIZATION_SCHEMA,
        "status": SOMPH_FORMAL_POLICY_AUTHORIZATION_STATUS,
        "formal_launch_authority": True,
        "formal_metric_claim_allowed": True,
        "package_root_sha256": manifest["package_root_sha256"],
        "package_detached_seal_sha256": expected_package_detached_seal_sha256,
        "artifact_member_allowlist_sha256": seal[
            "artifact_member_allowlist_sha256"
        ],
        "manifest_sha256": seal["manifest_sha256"],
        "overlay_provenance_sha256": manifest["overlay_provenance_sha256"],
        "authority_commit_sha256": expected_authority_commit_sha256,
        "authority_lock_sha256": authority_commit["authority_lock_sha256"],
        "authority_attestation_sha256": attestation_sha256,
        "receiver": receiver,
        "seed": seed,
        "stage": manifest["stage"],
        "registration_state": manifest["registration_state"],
        "k_shot": manifest["k_shot"],
        "cache_scope": "stage2_registered",
        "old_tx_ids": old_tx_ids,
        "new_tx_ids": new_tx_ids,
        "dataset_authority_root_sha256": authority_attestation[
            "dataset_authority_root_sha256"
        ],
        "cache_role_inputs_root_sha256": authority_lock[
            "cache_role_inputs_root_sha256"
        ],
        "physical_sample_ids_sha256_by_scenario": authority_lock[
            "physical_sample_ids_sha256_by_scenario"
        ],
        "physical_sample_scenario_assignment_sha256": authority_lock[
            "physical_sample_scenario_assignment_sha256"
        ],
        "post_channel_iq_sha256_root_by_scenario": authority_lock[
            "post_channel_iq_sha256_root_by_scenario"
        ],
        "overlay_ids_sha256_by_scenario": authority_lock[
            "overlay_ids_sha256_by_scenario"
        ],
        "preflight_code_sha256": preflight_code_sha256,
        "formal_policy_sha256": actual_formal_policy_sha256,
        "code_closure_sha256": code_closure_sha256,
        **dict(package_control_roots),
    }
    if authorization != expected:
        raise PredictorPackageError("SOMP-H formal policy authorization binding drift")
    return {
        "formal_policy_authorization_sha256": authorization_sha256,
        "preflight_code_sha256": preflight_code_sha256,
        "authority_attestation_sha256": attestation_sha256,
    }


def preflight_somph_predictor_bundle(
    package_root: str | Path,
    *,
    detached_seal_path: str | Path,
    expected_seal_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the complete package without NumPy IQ materialization."""

    manifest, seal, audit, _provenance = _preflight(
        package_root,
        detached_seal_path=detached_seal_path,
        expected_seal_sha256=expected_seal_sha256,
    )
    return manifest, seal, audit


def _preflight_somph_predictor_bundle_with_authority_impl(
    package_root: str | Path,
    *,
    detached_seal_path: str | Path,
    expected_seal_sha256: str,
    authority_bundle_root: str | Path,
    expected_authority_commit_sha256: str,
    formal_policy_path: str | Path,
    formal_policy_authorization_path: str | Path,
    signed_policy_authorization_envelope_path: str | Path,
    expected_signed_policy_authorization_envelope_sha256: str,
    signed_policy_verifier: Callable[
        [Mapping[str, Any], Mapping[str, Any]], None
    ],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authorize one formal package without opening any IQ archive.

    The authority bundle proves the signed pre-overlay/cache lineage but, by
    design, does not grant launch authority.  A separately expected policy
    authorization must bind that lineage, this exact package/seal, the current
    preflight code, and the formal row policy.  Every failure occurs before an
    IQ archive is opened, CRC-inspected, decompressed, or materialized.
    """

    manifest, seal, structural_audit, provenance = _preflight(
        package_root,
        detached_seal_path=detached_seal_path,
        expected_seal_sha256=expected_seal_sha256,
        inspect_iq_members=False,
        load_npz_control_members=False,
    )
    try:
        authority_lock, authority_attestation, authority_commit = (
            lineage_authority.verify_somph_lineage_authority_bundle(
                authority_bundle_root,
                expected_commit_sha256=expected_authority_commit_sha256,
            )
        )
    except Exception as exc:
        raise PredictorPackageError(
            "SOMP-H external lineage authority verification failed"
        ) from exc
    policy, actual_policy_sha256 = _external_json_actual(
        formal_policy_path, context="SOMP-H actual formal execution policy"
    )
    _validate_formal_policy(policy)
    code_members, code_closure_sha256 = _code_closure()
    authorization, _authorization_file_sha256 = _external_json_actual(
        formal_policy_authorization_path,
        context="SOMP-H formal row policy authorization",
    )
    authorization_canonical_sha256 = sha256_bytes(
        canonical_json_bytes(authorization)
    )
    package_control_roots = _package_control_roots(
        manifest,
        provenance,
        new_tx_ids=list(authority_lock.get("new_tx_ids", [])),
    )
    bindings = _validate_formal_lineage_authority(
        manifest=manifest,
        seal=seal,
        provenance_index=provenance,
        authority_lock=authority_lock,
        authority_attestation=authority_attestation,
        authority_commit=authority_commit,
        expected_authority_commit_sha256=expected_authority_commit_sha256,
        expected_package_detached_seal_sha256=expected_seal_sha256,
        authorization=authorization,
        authorization_sha256=authorization_canonical_sha256,
        actual_formal_policy_sha256=actual_policy_sha256,
        code_closure_sha256=code_closure_sha256,
        package_control_roots=package_control_roots,
    )
    envelope, envelope_sha256 = _external_json_with_expected_sha256(
        signed_policy_authorization_envelope_path,
        expected_sha256=expected_signed_policy_authorization_envelope_sha256,
        context="SOMP-H signed policy authorization envelope",
    )
    signed_policy_verifier(
        envelope,
        {
            "authorization_canonical_sha256": authorization_canonical_sha256,
            "formal_policy_sha256": actual_policy_sha256,
            "package_root_sha256": manifest["package_root_sha256"],
            "package_detached_seal_sha256": expected_seal_sha256,
            "authority_commit_sha256": expected_authority_commit_sha256,
            "receiver": manifest["receiver"],
            "seed": manifest["seed"],
            "stage": manifest["stage"],
            "registration_state": manifest["registration_state"],
            "k_shot": manifest["k_shot"],
            "code_closure_sha256": code_closure_sha256,
        },
    )
    audit = dict(structural_audit)
    audit.update(
        {
            "status": "AUTHORITY_PREFLIGHT_PASS_IQ_OPEN_AUTHORIZED",
            "iq_archive_opened": False,
            "np_load_invoked": False,
            "iq_payload_materialized": False,
            "iq_open_authorized": True,
            "phase2_protocol_evidence_status": (
                "AUTHORITY_PREFLIGHT_PASS_IQ_OPEN_AUTHORIZED"
            ),
            "external_authority_lock_verified": True,
            "authority_commit_sha256": expected_authority_commit_sha256,
            "formal_policy_sha256": actual_policy_sha256,
            "signed_policy_authorization_envelope_sha256": envelope_sha256,
            "package_detached_seal_sha256": expected_seal_sha256,
            "code_closure_members": code_members,
            "code_closure_sha256": code_closure_sha256,
            **package_control_roots,
            **bindings,
            "formal_launch_authority": False,
            "formal_metric_claim_allowed": False,
            "control_state": "AUTHORITY_PREFLIGHT_PASS_IQ_OPEN_AUTHORIZED",
        }
    )
    audit["preflight_anchor_sha256"] = sha256_bytes(canonical_json_bytes(audit))
    return manifest, seal, audit


def preflight_somph_predictor_bundle_with_authority(
    package_root: str | Path,
    *,
    detached_seal_path: str | Path,
    expected_seal_sha256: str,
    authority_bundle_root: str | Path,
    expected_authority_commit_sha256: str,
    formal_policy_path: str | Path,
    formal_policy_authorization_path: str | Path,
    signed_policy_authorization_envelope_path: str | Path,
    expected_signed_policy_authorization_envelope_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return _preflight_somph_predictor_bundle_with_authority_impl(
        package_root,
        detached_seal_path=detached_seal_path,
        expected_seal_sha256=expected_seal_sha256,
        authority_bundle_root=authority_bundle_root,
        expected_authority_commit_sha256=expected_authority_commit_sha256,
        formal_policy_path=formal_policy_path,
        formal_policy_authorization_path=formal_policy_authorization_path,
        signed_policy_authorization_envelope_path=(
            signed_policy_authorization_envelope_path
        ),
        expected_signed_policy_authorization_envelope_sha256=(
            expected_signed_policy_authorization_envelope_sha256
        ),
        signed_policy_verifier=_DEFAULT_SIGNED_POLICY_ENVELOPE_VERIFIER,
    )


def _make_test_authority_preflight(
    public_key: bytes,
) -> Callable[..., tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Return the real preflight bound to one explicit synthetic test key."""

    verifier = _make_test_signed_policy_envelope_verifier(public_key)

    def test_preflight(package_root: str | Path, **kwargs: Any):
        return _preflight_somph_predictor_bundle_with_authority_impl(
            package_root,
            signed_policy_verifier=verifier,
            **kwargs,
        )

    return test_preflight


def _materialize_iq(
    root: Path, descriptor: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with open_regular_member_same_fd(root, descriptor["relative_path"]) as handle:
        digest, size = _hash_handle(handle)
        if digest != descriptor["sha256"] or size != descriptor["size_bytes"]:
            raise PredictorPackageError("SOMP-H IQ changed after preflight")
        actual = _zip_members_from_handle(handle, context=descriptor["relative_path"])
        if actual != tuple(descriptor["npz_members"]):
            raise PredictorPackageError("SOMP-H IQ allowlist changed after preflight")
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in actual}
    embedded = _parse_embedded_manifest(
        arrays.pop("manifest_json"), context=descriptor["relative_path"]
    )
    return arrays, embedded


def _tokens(values: np.ndarray, *, prefix: str, context: str) -> list[str]:
    result = np.asarray(values).astype(str).tolist()
    if (
        not result
        or len(result) != len(set(result))
        or any(
            not value.startswith(prefix) or OPAQUE_TOKEN_RE.fullmatch(value) is None
            for value in result
        )
    ):
        raise PredictorPackageError(f"SOMP-H {context} tokens are not unique opaque")
    return result


def _validate_iq(values: np.ndarray, hashes: np.ndarray, *, context: str) -> None:
    iq = np.asarray(values)
    declared = np.asarray(hashes).astype(str).tolist()
    if iq.dtype != np.float32 or iq.ndim != 3 or iq.shape[1] != 2:
        raise PredictorPackageError(f"SOMP-H LEO IQ shape/dtype drift: {context}")
    if len(declared) != iq.shape[0] or any(not _is_sha256(value) for value in declared):
        raise PredictorPackageError(f"SOMP-H IQ digest array drift: {context}")
    actual = [iq_row_sha256(iq[index]) for index in range(iq.shape[0])]
    if declared != actual:
        raise PredictorPackageError(f"SOMP-H post-channel IQ digest mismatch: {context}")


def _cross_provenance(
    *,
    tokens: list[str],
    overlays: list[str],
    satellite_seeds: np.ndarray,
    iq_hashes: np.ndarray,
    provenance: Mapping[str, Mapping[str, Any]],
    context: str,
) -> None:
    seeds = np.asarray(satellite_seeds)
    hashes = np.asarray(iq_hashes).astype(str).tolist()
    if seeds.shape != (len(tokens),) or len(overlays) != len(tokens):
        raise PredictorPackageError(f"SOMP-H sample metadata count drift: {context}")
    if set(tokens) != set(provenance):
        raise PredictorPackageError(f"SOMP-H provenance/sample token set mismatch: {context}")
    for index, token in enumerate(tokens):
        expected = provenance[token]
        if (
            overlays[index] != expected["overlay_token"]
            or int(seeds[index]) != expected["satellite_seed"]
            or hashes[index] != expected["post_channel_iq_sha256"]
        ):
            raise PredictorPackageError(
                f"SOMP-H sample-level provenance cross-check failed: {context}"
            )


def _validate_support_payload(
    arrays: dict[str, np.ndarray],
    embedded: dict[str, Any],
    *,
    manifest: Mapping[str, Any],
    scenario: str,
    provenance: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_embedded = {
        "schema": SOMPH_SUPPORT_IQ_SCHEMA,
        "scenario": scenario,
        "registration_state": manifest["registration_state"],
        "registered_class_count": manifest["registered_class_count"],
        "support_pool_max_k": manifest["support_pool_max_k"],
        "token_scheme": "hmac_sha256_opaque_v1",
    }
    if embedded != expected_embedded:
        raise PredictorPackageError("SOMP-H support embedded manifest drift")
    tokens = _tokens(arrays["support_tokens"], prefix="sid_", context="support")
    overlays = _tokens(
        arrays["support_overlay_tokens"], prefix="oid_", context="support overlay"
    )
    labels = np.asarray(arrays["support_class_indices"])
    ranks = np.asarray(arrays["support_rank_within_class"])
    if (
        labels.dtype != np.int64
        or ranks.dtype != np.int64
        or labels.shape != ranks.shape
        or labels.shape != (len(tokens),)
    ):
        raise PredictorPackageError("SOMP-H support class/rank schema drift")
    expected_pairs = [
        (class_index, rank)
        for class_index in range(manifest["registered_class_count"])
        for rank in range(manifest["support_pool_max_k"])
    ]
    if list(zip(labels.tolist(), ranks.tolist())) != expected_pairs:
        raise PredictorPackageError(
            "SOMP-H support does not expose exactly K ordered samples per class"
        )
    _validate_iq(
        arrays["support_leo_weak_iq"],
        arrays["support_post_channel_iq_sha256"],
        context=f"support:{scenario}",
    )
    _cross_provenance(
        tokens=tokens,
        overlays=overlays,
        satellite_seeds=arrays["support_satellite_seeds"],
        iq_hashes=arrays["support_post_channel_iq_sha256"],
        provenance=provenance,
        context=f"support:{scenario}",
    )


def _validate_query_payload(
    arrays: dict[str, np.ndarray],
    embedded: dict[str, Any],
    *,
    manifest: Mapping[str, Any],
    scenario: str,
    provenance: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_embedded = {
        "schema": SOMPH_QUERY_IQ_SCHEMA,
        "scenario": scenario,
        "registration_state": manifest["registration_state"],
        "token_scheme": "hmac_sha256_opaque_v1",
    }
    if embedded != expected_embedded:
        raise PredictorPackageError("SOMP-H query embedded manifest drift")
    tokens = _tokens(arrays["query_tokens"], prefix="qid_", context="query")
    overlays = _tokens(
        arrays["query_overlay_tokens"], prefix="oid_", context="query overlay"
    )
    _validate_iq(
        arrays["query_leo_weak_iq"],
        arrays["query_post_channel_iq_sha256"],
        context=f"query:{scenario}",
    )
    _cross_provenance(
        tokens=tokens,
        overlays=overlays,
        satellite_seeds=arrays["query_satellite_seeds"],
        iq_hashes=arrays["query_post_channel_iq_sha256"],
        provenance=provenance,
        context=f"query:{scenario}",
    )


def _immutable_array(value: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(value)
    return np.frombuffer(source.tobytes(), dtype=source.dtype).reshape(source.shape)


def _immutable_payloads(
    payloads: Mapping[str, Mapping[str, np.ndarray]],
) -> Mapping[str, Mapping[str, np.ndarray]]:
    return MappingProxyType(
        {
            scenario: MappingProxyType(
                {key: _immutable_array(value) for key, value in arrays.items()}
            )
            for scenario, arrays in payloads.items()
        }
    )


def _validated_preflight_anchor(
    authority_preflight_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    preflight = json.loads(canonical_json_bytes(authority_preflight_audit))
    anchor = preflight.pop("preflight_anchor_sha256", None)
    if (
        not _is_sha256(anchor)
        or sha256_bytes(canonical_json_bytes(preflight)) != anchor
        or preflight.get("status")
        != "AUTHORITY_PREFLIGHT_PASS_IQ_OPEN_AUTHORIZED"
        or preflight.get("control_state")
        != "AUTHORITY_PREFLIGHT_PASS_IQ_OPEN_AUTHORIZED"
        or preflight.get("iq_open_authorized") is not True
        or preflight.get("iq_archive_opened") is not False
        or preflight.get("np_load_invoked") is not False
        or preflight.get("iq_payload_materialized") is not False
        or preflight.get("formal_launch_authority") is not False
        or preflight.get("formal_metric_claim_allowed") is not False
    ):
        raise PredictorPackageError("SOMP-H authority preflight anchor invalid")
    preflight["preflight_anchor_sha256"] = anchor
    return preflight, anchor


def _actual_materialized_roots(
    payloads: Mapping[str, Mapping[str, np.ndarray]],
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    if set(payloads) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise PredictorPackageError("SOMP-H materialized scenario set drift")
    class_count = int(manifest["registered_class_count"])
    k_shot = int(manifest["k_shot"])
    expected_rows = class_count * k_shot
    observed_tokens: set[str] = set()
    token_roots: dict[str, str] = {}
    overlay_roots: dict[str, str] = {}
    iq_roots: dict[str, str] = {}
    seed_roots: dict[str, str] = {}
    assignment_roots: dict[str, str] = {}
    exact_k_roots: dict[str, str] = {}
    required_arrays = set(SUPPORT_NPZ_MEMBERS) - {"manifest_json"}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        arrays = payloads[scenario]
        if set(arrays) != required_arrays:
            raise PredictorPackageError("SOMP-H materialized support schema drift")
        _validate_iq(
            arrays["support_leo_weak_iq"],
            arrays["support_post_channel_iq_sha256"],
            context=f"verified-materializer:{scenario}",
        )
        labels = np.asarray(arrays["support_class_indices"])
        ranks = np.asarray(arrays["support_rank_within_class"])
        if (
            labels.shape != (expected_rows,)
            or ranks.shape != (expected_rows,)
            or not np.issubdtype(labels.dtype, np.integer)
            or not np.issubdtype(ranks.dtype, np.integer)
        ):
            raise PredictorPackageError("SOMP-H verified exact-K shape drift")
        expected_pairs = {
            (class_index, rank)
            for class_index in range(class_count)
            for rank in range(k_shot)
        }
        actual_pairs = [
            (int(label), int(rank)) for label, rank in zip(labels, ranks)
        ]
        if set(actual_pairs) != expected_pairs or len(set(actual_pairs)) != expected_rows:
            raise PredictorPackageError("SOMP-H verified exact-K assignment drift")
        tokens = _tokens(
            arrays["support_tokens"],
            prefix="sid_",
            context=f"verified-materializer:{scenario}",
        )
        overlays = _tokens(
            arrays["support_overlay_tokens"],
            prefix="oid_",
            context=f"verified-materializer-overlay:{scenario}",
        )
        iq_hashes = np.asarray(
            arrays["support_post_channel_iq_sha256"]
        ).astype(str).tolist()
        seeds = np.asarray(arrays["support_satellite_seeds"])
        if (
            len(tokens) != expected_rows
            or len(overlays) != expected_rows
            or seeds.shape != (expected_rows,)
            or not np.issubdtype(seeds.dtype, np.integer)
        ):
            raise PredictorPackageError("SOMP-H verified support metadata drift")
        if observed_tokens & set(tokens):
            raise PredictorPackageError(
                "SOMP-H verified cross-scenario physical reuse"
            )
        observed_tokens.update(tokens)
        ordered = sorted(
            zip(tokens, overlays, seeds.tolist(), iq_hashes, actual_pairs),
            key=lambda item: item[0],
        )
        token_roots[scenario] = sha256_bytes(
            canonical_json_bytes([item[0] for item in ordered])
        )
        overlay_roots[scenario] = sha256_bytes(
            canonical_json_bytes([item[1] for item in ordered])
        )
        seed_roots[scenario] = sha256_bytes(
            canonical_json_bytes([int(item[2]) for item in ordered])
        )
        iq_roots[scenario] = sha256_bytes(
            canonical_json_bytes([item[3] for item in ordered])
        )
        assignment_roots[scenario] = sha256_bytes(
            canonical_json_bytes(
                [
                    {
                        "sample_token": item[0],
                        "overlay_token": item[1],
                        "satellite_seed": int(item[2]),
                        "post_channel_iq_sha256": item[3],
                    }
                    for item in ordered
                ]
            )
        )
        exact_k_roots[scenario] = sha256_bytes(
            canonical_json_bytes(
                [
                    {
                        "sample_token": item[0],
                        "class_index": item[4][0],
                        "rank_within_class": item[4][1],
                    }
                    for item in ordered
                ]
            )
        )
    return {
        "actual_support_token_sha256_by_scenario": token_roots,
        "actual_overlay_ids_sha256_by_scenario": overlay_roots,
        "actual_iq_sha256_root_by_scenario": iq_roots,
        "actual_satellite_seed_sha256_by_scenario": seed_roots,
        "actual_materialized_assignment_sha256_by_scenario": assignment_roots,
        "actual_exact_k_assignment_sha256_by_scenario": exact_k_roots,
    }


def _require_actual_roots_match_preflight(
    actual: Mapping[str, Mapping[str, str]],
    preflight: Mapping[str, Any],
) -> None:
    pairs = (
        (
            "actual_support_token_sha256_by_scenario",
            "package_physical_sample_ids_sha256_by_scenario",
        ),
        (
            "actual_overlay_ids_sha256_by_scenario",
            "package_overlay_ids_sha256_by_scenario",
        ),
        (
            "actual_iq_sha256_root_by_scenario",
            "package_post_channel_iq_sha256_by_scenario",
        ),
        (
            "actual_satellite_seed_sha256_by_scenario",
            "package_satellite_seed_sha256_by_scenario",
        ),
        (
            "actual_materialized_assignment_sha256_by_scenario",
            "package_materialized_assignment_sha256_by_scenario",
        ),
    )
    if any(dict(actual[left]) != dict(preflight.get(right, {})) for left, right in pairs):
        raise PredictorPackageError(
            "SOMP-H actual materialized roots differ from preauthorized roots"
        )


def _materialization_runtime_binding(
    manifest: Mapping[str, Any], preflight: Mapping[str, Any]
) -> dict[str, str]:
    return {
        "package_root_sha256": manifest["package_root_sha256"],
        "feature_runtime_sha256": manifest["feature_runtime_sha256"],
        "phase1_checkpoint_sha256": manifest["phase1_checkpoint_sha256"],
        "method_lock_sha256": manifest["method_lock_sha256"],
        "preflight_anchor_sha256": preflight["preflight_anchor_sha256"],
        "code_closure_sha256": preflight["code_closure_sha256"],
    }


def _make_verified_materialization_api():
    capability = object()
    issued: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()

    class SomphMaterializedEnrollmentEvidence:
        __slots__ = (
            "_manifest_json",
            "_seal_json",
            "_preflight_json",
            "_payloads",
            "_binding_json",
            "_evidence_sha256",
            "__weakref__",
        )

        def __init__(
            self,
            *,
            manifest: Mapping[str, Any],
            seal: Mapping[str, Any],
            preflight: Mapping[str, Any],
            payloads: Mapping[str, Mapping[str, np.ndarray]],
            binding: Mapping[str, Any],
            _capability: object,
        ) -> None:
            if _capability is not capability:
                raise PredictorPackageError(
                    "SOMP-H materialized evidence requires verified materializer"
                )
            object.__setattr__(self, "_manifest_json", canonical_json_bytes(manifest))
            object.__setattr__(self, "_seal_json", canonical_json_bytes(seal))
            object.__setattr__(self, "_preflight_json", canonical_json_bytes(preflight))
            object.__setattr__(self, "_payloads", _immutable_payloads(payloads))
            object.__setattr__(self, "_binding_json", canonical_json_bytes(binding))
            digest = sha256_bytes(
                canonical_json_bytes(
                    {
                        "manifest_sha256": sha256_bytes(self._manifest_json),
                        "seal_sha256": sha256_bytes(self._seal_json),
                        "preflight_sha256": sha256_bytes(self._preflight_json),
                        "binding_sha256": sha256_bytes(self._binding_json),
                    }
                )
            )
            object.__setattr__(self, "_evidence_sha256", digest)

        @property
        def manifest(self) -> dict[str, Any]:
            return json.loads(self._manifest_json)

        @property
        def materialized_payloads(self) -> Mapping[str, Mapping[str, np.ndarray]]:
            return self._payloads

        @property
        def evidence_sha256(self) -> str:
            return self._evidence_sha256

    def materialize(
        package_root: str | Path,
        *,
        detached_seal_path: str | Path,
        expected_seal_sha256: str,
        authority_preflight_audit: Mapping[str, Any],
    ) -> SomphMaterializedEnrollmentEvidence:
        preflight, _anchor = _validated_preflight_anchor(
            authority_preflight_audit
        )
        if expected_seal_sha256 != preflight.get(
            "package_detached_seal_sha256"
        ):
            raise PredictorPackageError(
                "SOMP-H materializer detached seal/preflight drift"
            )
        manifest, seal, _structural, provenance = _preflight(
            package_root,
            detached_seal_path=detached_seal_path,
            expected_seal_sha256=expected_seal_sha256,
            inspect_iq_members=False,
            load_npz_control_members=False,
        )
        if (
            manifest.get("profile") != ENROLLMENT_ONLY
            or manifest.get("package_root_sha256")
            != preflight.get("package_root_sha256")
            or seal.get("manifest_sha256") != preflight.get("manifest_sha256")
            or seal.get("package_root_sha256")
            != manifest.get("package_root_sha256")
        ):
            raise PredictorPackageError("SOMP-H materializer package binding drift")
        _members, current_code_closure = _code_closure()
        if current_code_closure != preflight.get("code_closure_sha256"):
            raise PredictorPackageError("SOMP-H materializer code closure drift")
        by_kind = {item["kind"]: item for item in manifest["members"]}
        payloads: dict[str, dict[str, np.ndarray]] = {}
        iq_member_roots: dict[str, str] = {}
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            descriptor = by_kind[f"support:{scenario}"]
            arrays, embedded = _materialize_iq(Path(package_root), descriptor)
            _validate_support_payload(
                arrays,
                embedded,
                manifest=manifest,
                scenario=scenario,
                provenance=provenance[scenario],
            )
            payloads[scenario] = arrays
            iq_member_roots[scenario] = descriptor["sha256"]
        immutable_payloads = _immutable_payloads(payloads)
        actual = _actual_materialized_roots(immutable_payloads, manifest)
        _require_actual_roots_match_preflight(actual, preflight)
        runtime = _materialization_runtime_binding(manifest, preflight)
        binding = {
            "schema": "cvs.phase2.somph_verified_materialization_evidence.v1",
            "package_detached_seal_sha256": expected_seal_sha256,
            "manifest_sha256": seal["manifest_sha256"],
            "artifact_member_allowlist_sha256": seal[
                "artifact_member_allowlist_sha256"
            ],
            "iq_member_sha256_by_scenario": iq_member_roots,
            "runtime_binding": runtime,
            **actual,
        }
        evidence = SomphMaterializedEnrollmentEvidence(
            manifest=manifest,
            seal=seal,
            preflight=preflight,
            payloads=immutable_payloads,
            binding=binding,
            _capability=capability,
        )
        issued[evidence] = evidence.evidence_sha256
        return evidence

    def finalize(evidence: Any) -> dict[str, Any]:
        registered_digest = (
            issued.pop(evidence, None)
            if isinstance(evidence, SomphMaterializedEnrollmentEvidence)
            else None
        )
        if registered_digest is None or registered_digest != getattr(
            evidence, "_evidence_sha256", None
        ):
            raise PredictorPackageError(
                "SOMP-H finalizer requires fresh token-sealed materialized evidence"
            )
        manifest = json.loads(evidence._manifest_json)
        seal = json.loads(evidence._seal_json)
        preflight, _anchor = _validated_preflight_anchor(
            json.loads(evidence._preflight_json)
        )
        binding = json.loads(evidence._binding_json)
        recomputed_digest = sha256_bytes(
            canonical_json_bytes(
                {
                    "manifest_sha256": sha256_bytes(evidence._manifest_json),
                    "seal_sha256": sha256_bytes(evidence._seal_json),
                    "preflight_sha256": sha256_bytes(evidence._preflight_json),
                    "binding_sha256": sha256_bytes(evidence._binding_json),
                }
            )
        )
        if recomputed_digest != evidence.evidence_sha256:
            raise PredictorPackageError("SOMP-H materialized evidence digest drift")
        if (
            manifest.get("profile") != ENROLLMENT_ONLY
            or manifest.get("package_root_sha256")
            != preflight.get("package_root_sha256")
            or seal.get("package_root_sha256")
            != manifest.get("package_root_sha256")
            or seal.get("manifest_sha256") != preflight.get("manifest_sha256")
            or binding.get("package_detached_seal_sha256")
            != preflight.get("package_detached_seal_sha256")
            or binding.get("artifact_member_allowlist_sha256")
            != preflight.get("artifact_member_allowlist_sha256")
        ):
            raise PredictorPackageError("SOMP-H finalizer package/seal binding drift")
        _members, current_code_closure = _code_closure()
        runtime = _materialization_runtime_binding(manifest, preflight)
        if (
            current_code_closure != preflight.get("code_closure_sha256")
            or binding.get("runtime_binding") != runtime
        ):
            raise PredictorPackageError("SOMP-H finalizer code/runtime binding drift")
        actual = _actual_materialized_roots(evidence._payloads, manifest)
        _require_actual_roots_match_preflight(actual, preflight)
        if any(binding.get(key) != value for key, value in actual.items()):
            raise PredictorPackageError("SOMP-H finalizer evidence root drift")
        final = dict(preflight)
        final.update(
            {
                "status": "CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS",
                "control_state": "CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS",
                "phase2_protocol_evidence_status": (
                    "CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS"
                ),
                "iq_archive_opened": True,
                "np_load_invoked": True,
                "iq_payload_materialized": True,
                "verified_materialization_evidence_sha256": (
                    evidence.evidence_sha256
                ),
                **actual,
                "runtime_binding": runtime,
                "formal_launch_authority": True,
                "formal_metric_claim_allowed": True,
            }
        )
        final["post_materialization_audit_sha256"] = sha256_bytes(
            canonical_json_bytes(final)
        )
        return final

    return SomphMaterializedEnrollmentEvidence, materialize, finalize


(
    SomphMaterializedEnrollmentEvidence,
    materialize_somph_enrollment_with_authority,
    finalize_somph_enrollment_authority_after_materialization,
) = _make_verified_materialization_api()


def load_verified_somph_predictor_bundle(
    package_root: str | Path,
    *,
    detached_seal_path: str | Path,
    expected_seal_sha256: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]]:
    """Preflight, then materialize exactly one profile's three LEO_weak views."""

    manifest, _seal, audit, provenance = _preflight(
        package_root,
        detached_seal_path=detached_seal_path,
        expected_seal_sha256=expected_seal_sha256,
    )
    root = Path(package_root)
    by_kind = {item["kind"]: item for item in manifest["members"]}
    prefix = "support:" if manifest["profile"] == ENROLLMENT_ONLY else "query:"
    payloads: dict[str, dict[str, np.ndarray]] = {}
    observed_sample_tokens: set[str] = set()
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        arrays, embedded = _materialize_iq(root, by_kind[prefix + scenario])
        if manifest["profile"] == ENROLLMENT_ONLY:
            _validate_support_payload(
                arrays,
                embedded,
                manifest=manifest,
                scenario=scenario,
                provenance=provenance[scenario],
            )
            sample_tokens = set(
                np.asarray(arrays["support_tokens"]).astype(str).tolist()
            )
        else:
            _validate_query_payload(
                arrays,
                embedded,
                manifest=manifest,
                scenario=scenario,
                provenance=provenance[scenario],
            )
            sample_tokens = set(
                np.asarray(arrays["query_tokens"]).astype(str).tolist()
            )
        reused_tokens = observed_sample_tokens & sample_tokens
        if reused_tokens:
            raise PredictorPackageError(
                "SOMP-H physical sample-token reuse across LEO_weak scenarios"
            )
        observed_sample_tokens.update(sample_tokens)
        payloads[scenario] = arrays
    return payloads, manifest, {
        **audit,
        "iq_payload_materialized": True,
        "materialized_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "sample_level_overlay_provenance_crosscheck": "PASS",
        "cross_scenario_physical_sample_token_disjointness_check": "PASS",
        "per_scenario_unified_support_pool_check": (
            "PASS" if manifest["profile"] == ENROLLMENT_ONLY else "NOT_APPLICABLE"
        ),
    }


def load_verified_somph_head_capsule(
    package_root: str | Path,
    *,
    detached_seal_path: str | Path,
    expected_seal_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    """Load the frozen apply-only head after full non-IQ preflight."""

    manifest, _seal, _audit, _provenance = _preflight(
        package_root,
        detached_seal_path=detached_seal_path,
        expected_seal_sha256=expected_seal_sha256,
    )
    if manifest["profile"] != APPLY_ONLY:
        raise PredictorPackageError(
            "SOMP-H head capsule is reachable only from apply_only"
        )
    by_kind = {item["kind"]: item for item in manifest["members"]}
    arrays, binding, binding_sha256 = _load_head_capsule_member(
        Path(package_root), by_kind["head_capsule"]
    )
    _validate_enrollment_binding(binding, manifest=manifest)
    return arrays, binding, binding_sha256
