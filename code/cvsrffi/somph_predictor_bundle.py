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

import json
import os
import re
import stat
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import numpy as np

from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT
from cvsrffi.somph_predictor_runtime import (
    SOMPH_HEAD_CAPSULE_SCHEMA,
    expected_somph_method_lock,
    validate_somph_head_capsule,
)
from cvsrffi.stage2_predictor_bundle import (
    FORMAL_LEO_WEAK_SCENARIOS,
    OPAQUE_TOKEN_RE,
    PredictorPackageError,
    _ensure_root,
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


ADV3B02_CHECKPOINT_SHA256 = (
    "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
)
ADV3B02_FEATURE_SCHEMA = "adv3b02_z_id160_fp32"
SOMPH_METHOD_LOCK_SCHEMA = "cvs.phase2.somph_method_lock.v1"
SOMPH_ENROLLMENT_BINDING_SCHEMA = "cvs.phase2.somph_enrollment_binding.v1"

SOMPH_BUNDLE_MANIFEST_SCHEMA = "cvs.phase2.somph_predictor_bundle.v1"
SOMPH_BUNDLE_SEAL_SCHEMA = "cvs.phase2.somph_predictor_bundle_seal.v1"
SOMPH_OVERLAY_PROVENANCE_SCHEMA = "cvs.phase2.somph_overlay_provenance.v1"
SOMPH_SUPPORT_IQ_SCHEMA = "cvs.phase2.somph_registered_support_iq.v1"
SOMPH_QUERY_IQ_SCHEMA = "cvs.phase2.somph_unlabeled_query_iq.v1"

ENROLLMENT_ONLY = "enrollment_only"
APPLY_ONLY = "apply_only"
PROFILE_VALUES = frozenset({ENROLLMENT_ONLY, APPLY_ONLY})
REGISTRATION_STATES = frozenset({"before", "after"})
SUPPORT_POOL_MAX_K = 20

CHECKPOINT_RELATIVE_PATH = "checkpoint.pt"
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
    "checkpoint_sha256",
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
    "checkpoint_sha256",
    "method_lock_sha256",
    "support_token_sha256_by_scenario",
    "support_feature_sha256_by_scenario",
}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value.lower()) is not None


def _is_row_handle(value: Any) -> bool:
    return isinstance(value, str) and ROW_HANDLE_RE.fullmatch(value) is not None


def _profile_kinds(profile: str) -> tuple[str, ...]:
    common = ("checkpoint", "method_lock", "overlay_provenance")
    if profile == ENROLLMENT_ONLY:
        return common + tuple(f"support:{value}" for value in FORMAL_LEO_WEAK_SCENARIOS)
    if profile == APPLY_ONLY:
        return common[:2] + ("head_capsule", common[2]) + tuple(
            f"query:{value}" for value in FORMAL_LEO_WEAK_SCENARIOS
        )
    raise PredictorPackageError("SOMP-H bundle profile invalid")


def _relative_path_for_kind(kind: str) -> str:
    fixed = {
        "checkpoint": CHECKPOINT_RELATIVE_PATH,
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
    if kind == "checkpoint":
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
    if payload.get("checkpoint_sha256") != ADV3B02_CHECKPOINT_SHA256:
        raise PredictorPackageError("SOMP-H enrollment binding checkpoint drift")
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
        "support_pool_max_k": SUPPORT_POOL_MAX_K,
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "checkpoint_sha256": ADV3B02_CHECKPOINT_SHA256,
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
    count = payload.get("registered_class_count")
    if not isinstance(count, int):
        raise PredictorPackageError("SOMP-H registered class count invalid")
    _validate_registry(payload.get("registered_classes"), count)
    if not _is_sha256(payload.get("method_lock_sha256")):
        raise PredictorPackageError("SOMP-H method lock SHA256 invalid")
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
    if by_kind["checkpoint"]["sha256"] != ADV3B02_CHECKPOINT_SHA256:
        raise PredictorPackageError("formal ADV3B02 checkpoint SHA256 mismatch")
    if by_kind["method_lock"]["sha256"] != expected_method_lock_sha256:
        raise PredictorPackageError("SOMP-H method lock SHA256 mismatch")

    method_lock, _ = _verify_regular_member(root, by_kind["method_lock"])
    assert method_lock is not None
    _validate_method_lock(method_lock, checkpoint_sha256=ADV3B02_CHECKPOINT_SHA256)
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
    _validate_provenance(
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
    root_digest = package_root_sha256(members)
    manifest = {
        "schema": SOMPH_BUNDLE_MANIFEST_SCHEMA,
        "profile": profile,
        "stage": stage,
        "registration_state": registration_state,
        "receiver": receiver,
        "seed": seed,
        "k_shot": k_shot,
        "support_pool_max_k": SUPPORT_POOL_MAX_K,
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "registered_class_count": count,
        "registered_classes": registry,
        "checkpoint_sha256": ADV3B02_CHECKPOINT_SHA256,
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
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    _validate_package_root_exact_allowlist(
        root, allowed_files=expected_paths | {MANIFEST_RELATIVE_PATH}
    )
    seal = {
        "schema": SOMPH_BUNDLE_SEAL_SCHEMA,
        "manifest_relative_path": MANIFEST_RELATIVE_PATH,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "manifest_size_bytes": len(manifest_bytes),
        "package_root_sha256": root_digest,
        "artifact_member_allowlist_sha256": root_digest,
    }
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    with seal_path.open("xb") as handle:
        handle.write(canonical_json_bytes(seal) + b"\n")
    return manifest_path, seal_path, manifest, seal


def _preflight(
    package_root: str | Path,
    *,
    detached_seal_path: str | Path,
    expected_seal_sha256: str,
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
    checkpoint_payload, receipt = _verify_regular_member(root, by_kind["checkpoint"])
    assert checkpoint_payload is None
    opened.append(receipt)
    if receipt["sha256"] != ADV3B02_CHECKPOINT_SHA256:
        raise PredictorPackageError("formal ADV3B02 checkpoint SHA256 mismatch before IQ")

    method_lock, receipt = _verify_regular_member(root, by_kind["method_lock"])
    opened.append(receipt)
    assert method_lock is not None
    if receipt["sha256"] != manifest["method_lock_sha256"]:
        raise PredictorPackageError("SOMP-H method lock digest binding mismatch")
    _validate_method_lock(method_lock, checkpoint_sha256=ADV3B02_CHECKPOINT_SHA256)
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
        _head_arrays, head_binding, head_binding_sha256 = _load_head_capsule_member(
            root, by_kind["head_capsule"]
        )
        if by_kind["head_capsule"]["sha256"] != manifest["head_capsule_sha256"]:
            raise PredictorPackageError(
                "SOMP-H apply head capsule trust root mismatch"
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

    # This is intentionally after checkpoint, method-lock, and provenance
    # validation.  It decompresses/CRC-checks the archives but does not np.load.
    prefix = "support:" if manifest["profile"] == ENROLLMENT_ONLY else "query:"
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
        "support_pool_max_k": SUPPORT_POOL_MAX_K,
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
        for rank in range(SUPPORT_POOL_MAX_K)
    ]
    if list(zip(labels.tolist(), ranks.tolist())) != expected_pairs:
        raise PredictorPackageError("SOMP-H support is not the unified K20 pool")
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
    reference_sample_tokens: set[str] | None = None
    reference_support_assignment: dict[str, tuple[int, int]] | None = None
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
            support_assignment = {
                token: (int(class_index), int(rank))
                for token, class_index, rank in zip(
                    np.asarray(arrays["support_tokens"]).astype(str).tolist(),
                    np.asarray(arrays["support_class_indices"]).tolist(),
                    np.asarray(arrays["support_rank_within_class"]).tolist(),
                )
            }
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
        if reference_sample_tokens is None:
            reference_sample_tokens = sample_tokens
        elif sample_tokens != reference_sample_tokens:
            raise PredictorPackageError(
                "SOMP-H physical sample-token set drifts across LEO_weak scenarios"
            )
        if manifest["profile"] == ENROLLMENT_ONLY:
            if reference_support_assignment is None:
                reference_support_assignment = support_assignment
            elif support_assignment != reference_support_assignment:
                raise PredictorPackageError(
                    "SOMP-H support token/class/rank mapping drifts across "
                    "LEO_weak scenarios"
                )
        payloads[scenario] = arrays
    return payloads, manifest, {
        **audit,
        "iq_payload_materialized": True,
        "materialized_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "sample_level_overlay_provenance_crosscheck": "PASS",
        "cross_scenario_physical_sample_token_set_check": "PASS",
        "cross_scenario_support_assignment_check": (
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
