"""Strict jointly sealed ADV3B02 runtime plus compact-prototype bundle.

The standalone compact component deliberately remains formally ineligible.  A
formal context is returned only after this outer package, its detached seal,
and an external signature envelope have all been verified.  The package is a
deployment surface: it never accepts a training checkpoint, dataset locator,
sample feature, or cache/build specification.
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import shutil
import weakref
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from cvsrffi import somph_runtime_trust as runtime_trust

from cvsrffi.phase1_center_lowrank_prototype_bundle import (
    ALLOWED_NPZ_MEMBERS,
    FEATURE_DIM,
    FINAL_MANIFEST_FIELDS,
    MANIFEST_NAME as COMPONENT_MANIFEST_NAME,
    MANIFEST_SHA_NAME as COMPONENT_MANIFEST_SHA_NAME,
    NPZ_NAME as COMPONENT_NPZ_NAME,
    PENDING_OUTER_JOINT_SEAL,
    RESIDUAL_RANK,
    SCHEMA as COMPONENT_SCHEMA,
    CenterLowRankPrototypeComponent,
    _pre_sign_content_root as _component_pre_sign_content_root,
    _scalar_string,
    _string_registry,
    _validate_payload as _validate_component_payload,
)
from cvsrffi.phase1_grb_jp4_bundle import (
    ALLOWED_NPZ_MEMBERS as GRB_ALLOWED_NPZ_MEMBERS,
    CLASS_COUNT as GRB_CLASS_COUNT,
    COMPONENT_PROFILE as GRB_COMPONENT_PROFILE,
    FEATURE_DIM as GRB_FEATURE_DIM,
    FEATURE_SCHEMA as GRB_FEATURE_SCHEMA,
    FINAL_MANIFEST_FIELDS as GRB_FINAL_MANIFEST_FIELDS,
    HIDDEN_DIM as GRB_HIDDEN_DIM,
    NPZ_NAME as GRB_COMPONENT_NPZ_NAME,
    PENDING_OUTER_JOINT_SEAL as GRB_PENDING_OUTER_JOINT_SEAL,
    RANK as GRB_RANK,
    SCHEMA as GRB_COMPONENT_SCHEMA,
    GRBJP4CompactComponent,
    _class_binding_digest as _grb_class_binding_digest,
    _pre_sign_content_root as _grb_pre_sign_content_root,
    _registry_sha256 as _grb_registry_sha256,
    _validate_payload as _validate_grb_payload,
)
from cvsrffi.stage2_predictor_bundle import (
    PredictorPackageError,
    _ensure_root,
    _hash_handle,
    _is_sha256,
    _json_from_handle,
    _validate_package_root_exact_allowlist,
    _zip_members_from_handle,
    canonical_json_bytes,
    open_regular_member_same_fd,
    sha256_bytes,
    sha256_file,
)


BUNDLE_MANIFEST_SCHEMA = "cvs.phase1.adv3b02_deployment_bundle_manifest.v1"
DETACHED_SEAL_SCHEMA = "cvs.phase1.adv3b02_deployment_bundle_detached_seal.v1"
SIGNING_REQUEST_SCHEMA = "cvs.phase1.adv3b02_deployment_bundle_signing_request.v1"
SIGNATURE_ENVELOPE_SCHEMA = "cvs.phase1.adv3b02_deployment_bundle_signature_envelope.v1"
SIGNATURE_DOMAIN = "cvs.phase1.adv3b02_deployment_bundle.ed25519.v1"
FORMAL_CONTEXT_SCHEMA = "cvs.phase2.adv3b02_joint_formal_context.v1"
CLASS_BINDING_SCHEMA = "phase1_tx_class_handle_binding_v1"
COMPONENT_PROFILE_SCHEMA = "cvs.phase1.adv3b02_component_profile.v1"
COMPONENT_PROFILE_CENTER_LOWRANK_V2 = "center_lowrank_residual_radius_v2"
COMPONENT_PROFILE_GRB_JP4_Q4 = "grb_jp4_q4_int8_v1"

# This is deliberately a profile resolver, not a second container, authority,
# validator, receipt, or signature path.  The outer package keeps its exact
# eight-member allowlist and the existing authority/method-lock contract.
_COMPONENT_PROFILES = {
    COMPONENT_SCHEMA: COMPONENT_PROFILE_CENTER_LOWRANK_V2,
    GRB_COMPONENT_SCHEMA: COMPONENT_PROFILE_GRB_JP4_Q4,
}

MANIFEST_RELATIVE_PATH = "deployment_manifest.json"
RUNTIME_RELATIVE_PATH = "runtime/adv3b02_runtime.torchscript.pt"
COMPONENT_NPZ_RELATIVE_PATH = f"component/{COMPONENT_NPZ_NAME}"
COMPONENT_MANIFEST_RELATIVE_PATH = f"component/{COMPONENT_MANIFEST_NAME}"
COMPONENT_MANIFEST_SHA_RELATIVE_PATH = (
    f"component/{COMPONENT_MANIFEST_SHA_NAME}"
)
CLASS_BINDING_RELATIVE_PATH = "locks/class_binding.json"
PARITY_RECEIPT_RELATIVE_PATH = "locks/runtime_checkpoint_parity_receipt.json"
GENERATION_LOCK_RELATIVE_PATH = "locks/generation_lock.json"
METHOD_LOCK_RELATIVE_PATH = "locks/method_lock.json"

ROLE_TO_PATH = {
    "sealed_torchscript_runtime": RUNTIME_RELATIVE_PATH,
    "v2_component_npz": COMPONENT_NPZ_RELATIVE_PATH,
    "v2_component_manifest": COMPONENT_MANIFEST_RELATIVE_PATH,
    "v2_component_manifest_sha256": COMPONENT_MANIFEST_SHA_RELATIVE_PATH,
    "class_handle_binding": CLASS_BINDING_RELATIVE_PATH,
    "runtime_checkpoint_parity_receipt": PARITY_RECEIPT_RELATIVE_PATH,
    "generation_lock": GENERATION_LOCK_RELATIVE_PATH,
    "method_lock": METHOD_LOCK_RELATIVE_PATH,
}

_MEMBER_KEYS = {"relative_path", "sha256", "size_bytes", "artifact_role"}
_BINDING_KEYS = {
    "checkpoint_lineage_sha256",
    "runtime_sha256",
    "component_pre_sign_content_root_sha256",
    "class_handle_binding_sha256",
    "parity_receipt_sha256",
    "generation_lock_sha256",
    "method_lock_sha256",
    "generation_config_sha256",
    "generation_code_sha256",
}
_MANIFEST_KEYS = {
    "schema",
    "artifact_stage",
    "members",
    "outer_content_root_sha256",
    *_BINDING_KEYS,
}
_SEAL_KEYS = {
    "schema",
    "manifest_relative_path",
    "manifest_sha256",
    "manifest_size_bytes",
    "outer_content_root_sha256",
    *_BINDING_KEYS,
}
_ENVELOPE_KEYS = {
    "schema",
    "domain",
    "issuer",
    "key_id",
    "detached_seal_sha256",
    "signature_ed25519_hex",
}
_FORBIDDEN_KEY_FRAGMENTS = (
    "dataset",
    "raw_iq",
    "clean",
    "cache",
    "loader",
    "source_path",
    "sample_feature",
    "sample_count",
    "member_count",
    "build_spec",
    "checkpoint_path",
)
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_EXTERNAL_JSON_BYTES = 256 * 1024
MAX_RUNTIME_FILE_BYTES = 64 * 1024 * 1024
MAX_RUNTIME_ARCHIVE_MEMBER_BYTES = 32 * 1024 * 1024
MAX_RUNTIME_ARCHIVE_TOTAL_BYTES = 128 * 1024 * 1024
MAX_RUNTIME_STATE_BYTES = 64 * 1024 * 1024
MAX_COMPONENT_NPZ_BYTES = 8 * 1024 * 1024
MAX_LOCK_JSON_BYTES = 1024 * 1024
_RUNTIME_STRUCTURE_KEYS = {
    "runtime_archive_member_root_sha256",
    "runtime_state_schema_root_sha256",
    "runtime_state_bytes",
    "runtime_structure_sha256",
}


class ADV3B02DeploymentBundleError(PredictorPackageError):
    """Raised before an untrusted deployment member is materialized."""


@dataclass(frozen=True, init=False)
class VerifiedADV3B02DeploymentBundle:
    """Loader-issued materialization of one production-signed outer bundle.

    This is not a public data container. Formal consumers re-verify the
    retained external chain and consume only that fresh materialization, never
    a caller-supplied ``runtime`` attribute.
    """

    runtime: Any
    runtime_member_path: str
    component: CenterLowRankPrototypeComponent | GRBJP4CompactComponent
    class_binding: Mapping[str, Any]
    parity_receipt: Mapping[str, Any]
    generation_lock: Mapping[str, Any]
    method_lock: Mapping[str, Any]
    formal_phase2_context: Mapping[str, Any]
    verification_receipt: Mapping[str, Any]
    audit: Mapping[str, Any]
    _formal_reverify_package_root: str
    _formal_reverify_kwargs: Mapping[str, Any]
    _issued_coordinator_sha256: str
    _issued_runtime_identity: int
    _formal_runtime_lifecycle: Mapping[str, Any]

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise ADV3B02DeploymentBundleError(
            "VerifiedADV3B02DeploymentBundle is production-factory issued only"
        )


def _verified_bundle_production_factory() -> tuple[Any, Any]:
    """Keep issuance capability private to the verified production loader.

    Python cannot make a module implementation detail cryptographic authority;
    the authority remains the external signature chain.  This closure prevents
    callers from obtaining a reusable issuer token or passing one to the public
    constructor, while retaining a weak issuance registry for formal consumers.
    """

    issued: dict[int, weakref.ReferenceType[VerifiedADV3B02DeploymentBundle]] = {}

    def issue(**values: Any) -> VerifiedADV3B02DeploymentBundle:
        bundle = object.__new__(VerifiedADV3B02DeploymentBundle)
        for name, value in values.items():
            if name in {"_formal_reverify_kwargs", "_formal_runtime_lifecycle"}:
                value = MappingProxyType(dict(value))
            object.__setattr__(bundle, name, value)
        issued[id(bundle)] = weakref.ref(bundle)
        return bundle

    def was_issued(bundle: Any) -> bool:
        reference = issued.get(id(bundle))
        return (
            type(bundle) is VerifiedADV3B02DeploymentBundle
            and reference is not None
            and reference() is bundle
        )

    return issue, was_issued


_issue_verified_bundle_after_external_verification, _was_verified_bundle_issued = (
    _verified_bundle_production_factory()
)


def _verified_coordinator_sha256(
    *,
    runtime_member_path: str,
    component: CenterLowRankPrototypeComponent | GRBJP4CompactComponent,
    class_binding: Mapping[str, Any],
    parity_receipt: Mapping[str, Any],
    generation_lock: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    formal_phase2_context: Mapping[str, Any],
    verification_receipt: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> str:
    """Bind loader-issued coordination fields without treating them as authority."""

    return sha256_bytes(
        canonical_json_bytes(
            {
                "runtime_member_path": runtime_member_path,
                "component_type": type(component).__name__,
                "component_manifest": dict(getattr(component, "manifest", {})),
                "class_binding": dict(class_binding),
                "parity_receipt": dict(parity_receipt),
                "generation_lock": dict(generation_lock),
                "method_lock": dict(method_lock),
                "formal_phase2_context": dict(formal_phase2_context),
                "verification_receipt": dict(verification_receipt),
                "audit": dict(audit),
            }
        )
    )


def reverify_formal_adv3b02_deployment_bundle(
    bundle: VerifiedADV3B02DeploymentBundle,
) -> VerifiedADV3B02DeploymentBundle:
    """Re-run the existing external seal/signature/member chain for formal use.

    The initial loader object is only a capability to retain the pinned loader
    arguments.  The caller's materialized runtime and component are never
    returned to a formal consumer; a new same-file-descriptor-verified runtime
    is loaded from the signed outer package instead.
    """

    if (
        not _was_verified_bundle_issued(bundle)
        or bundle._issued_runtime_identity != id(bundle.runtime)
    ):
        raise ADV3B02DeploymentBundleError(
            "formal bundle loader issuance/runtime identity drift"
        )
    current_coordinator_sha = _verified_coordinator_sha256(
        runtime_member_path=bundle.runtime_member_path,
        component=bundle.component,
        class_binding=bundle.class_binding,
        parity_receipt=bundle.parity_receipt,
        generation_lock=bundle.generation_lock,
        method_lock=bundle.method_lock,
        formal_phase2_context=bundle.formal_phase2_context,
        verification_receipt=bundle.verification_receipt,
        audit=bundle.audit,
    )
    if current_coordinator_sha != bundle._issued_coordinator_sha256:
        raise ADV3B02DeploymentBundleError(
            "formal bundle loader coordination field drift"
        )
    package_root = bundle._formal_reverify_package_root
    reverify_kwargs = dict(bundle._formal_reverify_kwargs)
    # Consume the sole loader-owned materialization before opening a second
    # runtime.  This makes formal re-verification a transfer of ownership,
    # rather than a copy that briefly leaves two full model instances live.
    try:
        source_runtime_ref = weakref.ref(bundle.runtime)
    except TypeError as exc:
        raise ADV3B02DeploymentBundleError(
            "formal source runtime does not support weak lifecycle observation"
        ) from exc
    object.__setattr__(bundle, "runtime", None)
    gc.collect()
    if source_runtime_ref() is not None:
        raise ADV3B02DeploymentBundleError(
            "formal source runtime weakref remained live after ownership transfer"
        )
    object.__setattr__(
        bundle,
        "_formal_runtime_lifecycle",
        MappingProxyType(
            {
                "schema": "cvs.phase1.adv3b02_runtime_ownership.v2",
                "source_bundle_runtime_consumed": True,
                "source_runtime_weakref_released_before_reload": True,
            }
        ),
    )
    fresh = load_formal_adv3b02_deployment_bundle(
        package_root,
        **reverify_kwargs,
    )
    if fresh._issued_coordinator_sha256 != bundle._issued_coordinator_sha256:
        raise ADV3B02DeploymentBundleError(
            "formal bundle reverified coordinator binding drift"
        )
    object.__setattr__(
        fresh,
        "_formal_runtime_lifecycle",
        MappingProxyType(
            {
                "schema": "cvs.phase1.adv3b02_runtime_ownership.v2",
                "source_bundle_runtime_consumed": True,
                "source_runtime_weakref_released_before_reload": True,
                "formal_runtime_weakref_live_after_materialization": True,
            }
        ),
    )
    return fresh


def component_profile_for_schema(component_schema: str) -> Mapping[str, Any]:
    """Return the fixed profile recognized by the existing joint-seal surface.

    Resolving a GRB profile does not make it formal: only a production-signed
    eight-member outer package can do that.  Keeping the resolver separate
    avoids inventing an authority, validator, receipt, or sidecar format.
    """

    profile = _COMPONENT_PROFILES.get(str(component_schema))
    if profile is None:
        raise ADV3B02DeploymentBundleError("unrecognized joint-seal component profile")
    return {
        "schema": COMPONENT_PROFILE_SCHEMA,
        "component_schema": str(component_schema),
        "component_profile": profile,
        "container_member_count": len(ROLE_TO_PATH),
        "signature_domain": SIGNATURE_DOMAIN,
        "method_lock_schema": "cvs.phase1.adv3b02_method_lock.v1",
    }


def _component_input_profile(component_root: Path) -> tuple[str, str, set[str]]:
    """Select one sealed component profile without changing outer members."""

    manifest = _load_json_regular(
        component_root / COMPONENT_MANIFEST_NAME, context="component manifest"
    )
    schema = str(manifest.get("schema", ""))
    profile = component_profile_for_schema(schema)["component_profile"]
    if profile == COMPONENT_PROFILE_CENTER_LOWRANK_V2:
        return schema, COMPONENT_NPZ_NAME, {
            COMPONENT_NPZ_NAME,
            COMPONENT_MANIFEST_NAME,
            COMPONENT_MANIFEST_SHA_NAME,
        }
    if profile == COMPONENT_PROFILE_GRB_JP4_Q4:
        if GRB_COMPONENT_PROFILE != COMPONENT_PROFILE_GRB_JP4_Q4:
            raise ADV3B02DeploymentBundleError("GRB component profile constant drift")
        if manifest.get("component_profile") != COMPONENT_PROFILE_GRB_JP4_Q4:
            raise ADV3B02DeploymentBundleError("GRB component manifest profile drift")
        return schema, GRB_COMPONENT_NPZ_NAME, {
            GRB_COMPONENT_NPZ_NAME,
            COMPONENT_MANIFEST_NAME,
            COMPONENT_MANIFEST_SHA_NAME,
        }
    raise ADV3B02DeploymentBundleError("unsupported joint-seal component profile")


def _validate_sha(value: Any, field: str) -> str:
    if type(value) is not str or value.lower() != value or not _is_sha256(value):
        raise ADV3B02DeploymentBundleError(f"invalid SHA256 for {field}")
    return value


def _validate_opaque_token(value: Any, *, field: str) -> str:
    token = str(value)
    lower = token.lower()
    if (
        not token
        or any(ord(character) < 32 or ord(character) == 127 for character in token)
        or any(character in token for character in ("/", "\\", ":"))
        or any(suffix in lower for suffix in (".pth", ".pkl", ".pt"))
        or token in {".", ".."}
    ):
        raise ADV3B02DeploymentBundleError(f"non-opaque deployment token for {field}")
    return token


def _validate_domain_handle(value: Any, *, field: str) -> str:
    handle = str(value)
    if handle.count(":") != 1:
        raise ADV3B02DeploymentBundleError(
            f"domain handle must contain exactly one separator for {field}"
        )
    prefix, suffix = handle.split(":", 1)
    _validate_opaque_token(prefix, field=f"{field} prefix")
    _validate_opaque_token(suffix, field=f"{field} suffix")
    return handle


def class_handle_binding_sha256(class_registry: Any) -> str:
    handles = tuple(
        _validate_opaque_token(value, field=f"class_handle[{index}]")
        for index, value in enumerate(class_registry)
    )
    if not handles or len(set(handles)) != len(handles) or any(not item for item in handles):
        raise ADV3B02DeploymentBundleError(
            "class registry handles must be non-empty and unique"
        )
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema": CLASS_BINDING_SCHEMA,
                "class_id_to_handle": [
                    {"class_index": index, "class_handle": handle}
                    for index, handle in enumerate(handles)
                ],
            }
        )
    )


def _reject_forbidden_keys(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).lower()
            if any(fragment in key for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                raise ADV3B02DeploymentBundleError(
                    f"forbidden deployment metadata key in {context}: {raw_key}"
                )
            _reject_forbidden_keys(nested, context=context)
    elif isinstance(value, list):
        for nested in value:
            _reject_forbidden_keys(nested, context=context)


def _json_exact(
    payload: Mapping[str, Any], required: set[str], *, context: str
) -> dict[str, Any]:
    result = dict(payload)
    if set(result) != required:
        raise ADV3B02DeploymentBundleError(f"{context} exact schema mismatch")
    _reject_forbidden_keys(result, context=context)
    return result


def _member_descriptor(path: Path, *, relative_path: str, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ADV3B02DeploymentBundleError(f"member must be a regular file: {role}")
    return {
        "relative_path": relative_path,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "artifact_role": role,
    }


def _content_root(members: list[Mapping[str, Any]]) -> str:
    ordered = sorted((dict(item) for item in members), key=lambda item: item["relative_path"])
    return sha256_bytes(canonical_json_bytes(ordered))


def _validate_root_allowlist(root: Path, *, allowed_files: set[str]) -> None:
    try:
        _validate_package_root_exact_allowlist(root, allowed_files=allowed_files)
    except PredictorPackageError as exc:
        raise ADV3B02DeploymentBundleError(str(exc)) from exc


def _validate_members(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(ROLE_TO_PATH):
        raise ADV3B02DeploymentBundleError("deployment member list size mismatch")
    checked: list[dict[str, Any]] = []
    roles: set[str] = set()
    paths: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != _MEMBER_KEYS:
            raise ADV3B02DeploymentBundleError("deployment member descriptor schema mismatch")
        item = dict(raw)
        role = str(item["artifact_role"])
        path = str(item["relative_path"])
        if ROLE_TO_PATH.get(role) != path or role in roles or path in paths:
            raise ADV3B02DeploymentBundleError("deployment member role/path allowlist drift")
        _validate_sha(item["sha256"], f"member:{role}")
        if not isinstance(item["size_bytes"], int) or item["size_bytes"] <= 0:
            raise ADV3B02DeploymentBundleError("deployment member size invalid")
        roles.add(role)
        paths.add(path)
        checked.append(item)
    if roles != set(ROLE_TO_PATH):
        raise ADV3B02DeploymentBundleError("deployment member role set mismatch")
    return checked


def _manifest_bindings(payload: Mapping[str, Any]) -> dict[str, str]:
    return {key: _validate_sha(payload[key], key) for key in _BINDING_KEYS}


def _validate_manifest(payload: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = _json_exact(payload, _MANIFEST_KEYS, context="deployment manifest")
    if result["schema"] != BUNDLE_MANIFEST_SCHEMA:
        raise ADV3B02DeploymentBundleError("deployment manifest schema drift")
    if result["artifact_stage"] != "phase1_offline_joint_deployment_bundle":
        raise ADV3B02DeploymentBundleError("deployment artifact stage drift")
    members = _validate_members(result["members"])
    if result["outer_content_root_sha256"] != _content_root(members):
        raise ADV3B02DeploymentBundleError("outer content root mismatch")
    _manifest_bindings(result)
    return result, members


def _signature_message(envelope: Mapping[str, Any]) -> bytes:
    unsigned = {key: envelope[key] for key in sorted(_ENVELOPE_KEYS - {"signature_ed25519_hex"})}
    return SIGNATURE_DOMAIN.encode("ascii") + b"\x00" + canonical_json_bytes(unsigned)


def _load_json_regular(path: Path, *, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ADV3B02DeploymentBundleError(f"{context} must be a regular file")
    with path.open("rb") as handle:
        raw = handle.read()
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ADV3B02DeploymentBundleError(f"invalid JSON for {context}") from exc
    if not isinstance(value, dict):
        raise ADV3B02DeploymentBundleError(f"{context} must be a JSON object")
    return value


def _load_external_json_same_handle(
    path: Path, *, expected_sha256: str, context: str
) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ADV3B02DeploymentBundleError(f"{context} must be a regular file")
    if path.stat().st_size > MAX_EXTERNAL_JSON_BYTES:
        raise ADV3B02DeploymentBundleError(f"{context} exceeds size limit")
    try:
        with path.open("rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise ADV3B02DeploymentBundleError(f"failed to read {context}") from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ADV3B02DeploymentBundleError(f"{context} external trust-root mismatch")
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ADV3B02DeploymentBundleError(f"invalid JSON for {context}") from exc
    if not isinstance(value, dict):
        raise ADV3B02DeploymentBundleError(f"{context} must be a JSON object")
    return value, raw


def _runtime_structure_from_bytes(runtime_bytes: bytes) -> tuple[dict[str, Any], Any]:
    if len(runtime_bytes) > MAX_RUNTIME_FILE_BYTES:
        raise ADV3B02DeploymentBundleError("TorchScript runtime exceeds file-size limit")
    archive_rows: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(runtime_bytes), mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if not infos or len(names) != len(set(names)):
                raise ADV3B02DeploymentBundleError("TorchScript archive member set invalid")
            total = 0
            for info in infos:
                path = info.filename.replace("\\", "/")
                parts = path.split("/")
                if (
                    info.is_dir()
                    or path.startswith("/")
                    or any(part in {"", ".", ".."} for part in parts)
                    or any(part.lower() == "extra" for part in parts)
                ):
                    raise ADV3B02DeploymentBundleError(
                        "TorchScript archive contains unsafe or extra-file member"
                    )
                if info.file_size > MAX_RUNTIME_ARCHIVE_MEMBER_BYTES:
                    raise ADV3B02DeploymentBundleError(
                        "TorchScript archive member exceeds size limit"
                    )
                total += info.file_size
                if total > MAX_RUNTIME_ARCHIVE_TOTAL_BYTES:
                    raise ADV3B02DeploymentBundleError(
                        "TorchScript archive total exceeds size limit"
                    )
                archive_rows.append(
                    {
                        "name": path,
                        "size_bytes": int(info.file_size),
                        "crc32": f"{int(info.CRC):08x}",
                    }
                )
    except ADV3B02DeploymentBundleError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ADV3B02DeploymentBundleError("invalid TorchScript archive") from exc
    try:
        import torch

        runtime = torch.jit.load(io.BytesIO(runtime_bytes), map_location="cpu")
        runtime.eval()
    except Exception as exc:
        raise ADV3B02DeploymentBundleError(
            "sealed runtime is not a loadable TorchScript module"
        ) from exc
    state_rows: list[dict[str, Any]] = []
    state_bytes = 0
    for kind, values in (
        ("parameter", runtime.named_parameters()),
        ("buffer", runtime.named_buffers()),
    ):
        for name, tensor in values:
            opaque_name = _validate_opaque_token(name, field=f"runtime {kind} name")
            size = int(tensor.numel()) * int(tensor.element_size())
            state_bytes += size
            state_rows.append(
                {
                    "kind": kind,
                    "name": opaque_name,
                    "shape": [int(value) for value in tensor.shape],
                    "dtype": str(tensor.dtype),
                    "size_bytes": size,
                }
            )
    if state_bytes > MAX_RUNTIME_STATE_BYTES:
        raise ADV3B02DeploymentBundleError("TorchScript runtime state exceeds size limit")
    archive_root = sha256_bytes(
        canonical_json_bytes(sorted(archive_rows, key=lambda item: item["name"]))
    )
    state_root = sha256_bytes(
        canonical_json_bytes(
            sorted(state_rows, key=lambda item: (item["kind"], item["name"]))
        )
    )
    structure = {
        "runtime_archive_member_root_sha256": archive_root,
        "runtime_state_schema_root_sha256": state_root,
        "runtime_state_bytes": state_bytes,
    }
    structure["runtime_structure_sha256"] = sha256_bytes(
        canonical_json_bytes(structure)
    )
    return structure, runtime


def runtime_structure_receipt(runtime_path: str | Path) -> dict[str, Any]:
    path = Path(runtime_path)
    if path.is_symlink() or not path.is_file():
        raise ADV3B02DeploymentBundleError("TorchScript runtime must be a regular file")
    if path.stat().st_size > MAX_RUNTIME_FILE_BYTES:
        raise ADV3B02DeploymentBundleError("TorchScript runtime exceeds file-size limit")
    return _runtime_structure_from_bytes(path.read_bytes())[0]


def _copy_regular(source: Path, destination: Path, *, role: str) -> None:
    if source.is_symlink() or not source.is_file():
        raise ADV3B02DeploymentBundleError(f"{role} source must be regular")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _validate_lock_documents(
    *,
    class_binding: Mapping[str, Any],
    parity: Mapping[str, Any],
    generation: Mapping[str, Any],
    method: Mapping[str, Any],
    bindings: Mapping[str, str],
) -> None:
    class_binding = _json_exact(
        class_binding,
        {
            "schema",
            "checkpoint_lineage_sha256",
            "class_id_to_handle",
            "class_handle_binding_sha256",
        },
        context="class binding",
    )
    if class_binding["schema"] != CLASS_BINDING_SCHEMA:
        raise ADV3B02DeploymentBundleError("class binding schema drift")
    rows = class_binding["class_id_to_handle"]
    if not isinstance(rows, list) or not rows:
        raise ADV3B02DeploymentBundleError("class handle registry invalid")
    handles: list[str] = []
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"class_index", "class_handle"}
            or row["class_index"] != index
            or not isinstance(row["class_handle"], str)
            or not row["class_handle"]
        ):
            raise ADV3B02DeploymentBundleError("class handle registry invalid")
        handles.append(
            _validate_opaque_token(
                row["class_handle"], field=f"class_handle[{index}]"
            )
        )
    if len(handles) != len(set(handles)):
        raise ADV3B02DeploymentBundleError("class handle registry invalid")
    semantic_sha = class_handle_binding_sha256(handles)
    if class_binding["class_handle_binding_sha256"] != semantic_sha:
        raise ADV3B02DeploymentBundleError("class binding semantic SHA drift")
    if semantic_sha != bindings["class_handle_binding_sha256"]:
        raise ADV3B02DeploymentBundleError("class binding outer semantic SHA drift")
    if class_binding["checkpoint_lineage_sha256"] != bindings["checkpoint_lineage_sha256"]:
        raise ADV3B02DeploymentBundleError("class binding checkpoint lineage drift")

    parity = _json_exact(
        parity,
        {
            "schema",
            "checkpoint_lineage_sha256",
            "runtime_sha256",
            "parity_status",
            "max_abs_output_delta",
            "parity_vector_root_sha256",
            *_RUNTIME_STRUCTURE_KEYS,
        },
        context="runtime parity receipt",
    )
    if parity["schema"] != "cvs.phase1.runtime_checkpoint_parity_receipt.v1":
        raise ADV3B02DeploymentBundleError("runtime parity receipt schema drift")
    if parity["parity_status"] != "PASS":
        raise ADV3B02DeploymentBundleError("runtime/checkpoint parity did not pass")
    try:
        delta = float(parity["max_abs_output_delta"])
    except (TypeError, ValueError) as exc:
        raise ADV3B02DeploymentBundleError("runtime parity delta invalid") from exc
    if not np.isfinite(delta) or delta < 0.0 or delta > 1.0e-5:
        raise ADV3B02DeploymentBundleError("runtime parity delta exceeds fixed tolerance")
    _validate_sha(parity["parity_vector_root_sha256"], "parity_vector_root_sha256")
    for key in _RUNTIME_STRUCTURE_KEYS - {"runtime_state_bytes"}:
        _validate_sha(parity[key], key)
    if (
        not isinstance(parity["runtime_state_bytes"], int)
        or parity["runtime_state_bytes"] < 0
        or parity["runtime_state_bytes"] > MAX_RUNTIME_STATE_BYTES
    ):
        raise ADV3B02DeploymentBundleError("runtime state byte receipt invalid")
    for key in ("checkpoint_lineage_sha256", "runtime_sha256"):
        if parity[key] != bindings[key]:
            raise ADV3B02DeploymentBundleError(f"runtime parity binding drift for {key}")

    generation = _json_exact(
        generation,
        {
            "schema",
            "checkpoint_lineage_sha256",
            "component_pre_sign_content_root_sha256",
            "class_handle_binding_sha256",
            "generation_config_sha256",
            "generation_code_sha256",
            "phase1_stream_sha256",
            "radius_generation_proof_sha256",
        },
        context="generation lock",
    )
    if generation["schema"] != "cvs.phase1.prototype_generation_lock.v1":
        raise ADV3B02DeploymentBundleError("generation lock schema drift")
    for key in (
        "checkpoint_lineage_sha256",
        "component_pre_sign_content_root_sha256",
        "class_handle_binding_sha256",
        "generation_config_sha256",
        "generation_code_sha256",
    ):
        if generation[key] != bindings[key]:
            raise ADV3B02DeploymentBundleError(f"generation lock binding drift for {key}")
    _validate_sha(generation["phase1_stream_sha256"], "phase1_stream_sha256")
    _validate_sha(generation["radius_generation_proof_sha256"], "radius_generation_proof_sha256")

    method = _json_exact(
        method,
        {
            "schema",
            "method_id",
            "checkpoint_lineage_sha256",
            "runtime_sha256",
            "component_pre_sign_content_root_sha256",
            "class_handle_binding_sha256",
            "parity_receipt_sha256",
            "generation_lock_sha256",
            "generation_config_sha256",
            "generation_code_sha256",
        },
        context="method lock",
    )
    if method["schema"] != "cvs.phase1.adv3b02_method_lock.v1" or not isinstance(
        method["method_id"], str
    ) or not method["method_id"]:
        raise ADV3B02DeploymentBundleError("method lock identity drift")
    _validate_opaque_token(method["method_id"], field="method_id")
    for key in set(method) - {"schema", "method_id"}:
        if method[key] != bindings[key]:
            raise ADV3B02DeploymentBundleError(f"method lock binding drift for {key}")


def build_unsigned_adv3b02_deployment_bundle(
    output_dir: str | Path,
    *,
    torchscript_runtime_path: str | Path,
    component_dir: str | Path,
    class_binding_path: str | Path,
    parity_receipt_path: str | Path,
    generation_lock_path: str | Path,
    method_lock_path: str | Path,
    detached_seal_path: str | Path,
    signing_request_path: str | Path,
) -> dict[str, Any]:
    """Build immutable unsigned content and an external signing request.

    No signature or private key is generated.  The returned request describes
    the exact signature envelope that an external authority must issue.
    """

    root = Path(output_dir)
    if root.exists():
        raise FileExistsError("refusing to reuse deployment bundle output directory")
    runtime_source = Path(torchscript_runtime_path)
    if runtime_source.suffix.lower() == ".pth":
        raise ADV3B02DeploymentBundleError("raw training checkpoint is forbidden")
    component_root = _ensure_root(Path(component_dir))
    _component_schema, component_npz_name, expected_component_names = _component_input_profile(
        component_root
    )
    if {item.name for item in component_root.iterdir()} != expected_component_names:
        raise ADV3B02DeploymentBundleError("component exact member allowlist mismatch")

    seal_path = Path(detached_seal_path).resolve()
    request_path = Path(signing_request_path).resolve()
    for external in (seal_path, request_path):
        if external.exists():
            raise FileExistsError("refusing to overwrite detached signing artifact")
        try:
            external.relative_to(root.resolve())
        except ValueError:
            pass
        else:
            raise ADV3B02DeploymentBundleError("signing artifacts must remain outside package root")

    sources = {
        "sealed_torchscript_runtime": runtime_source,
        "v2_component_npz": component_root / component_npz_name,
        "v2_component_manifest": component_root / COMPONENT_MANIFEST_NAME,
        "v2_component_manifest_sha256": component_root / COMPONENT_MANIFEST_SHA_NAME,
        "class_handle_binding": Path(class_binding_path),
        "runtime_checkpoint_parity_receipt": Path(parity_receipt_path),
        "generation_lock": Path(generation_lock_path),
        "method_lock": Path(method_lock_path),
    }
    role_size_limits = {
        "sealed_torchscript_runtime": MAX_RUNTIME_FILE_BYTES,
        "v2_component_npz": MAX_COMPONENT_NPZ_BYTES,
        "v2_component_manifest": MAX_LOCK_JSON_BYTES,
        "v2_component_manifest_sha256": MAX_EXTERNAL_JSON_BYTES,
        "class_handle_binding": MAX_LOCK_JSON_BYTES,
        "runtime_checkpoint_parity_receipt": MAX_LOCK_JSON_BYTES,
        "generation_lock": MAX_LOCK_JSON_BYTES,
        "method_lock": MAX_LOCK_JSON_BYTES,
    }
    for role, source in sources.items():
        if source.stat().st_size > role_size_limits[role]:
            raise ADV3B02DeploymentBundleError(f"deployment source exceeds size limit: {role}")
    root.mkdir(parents=True)
    for role, source in sources.items():
        _copy_regular(source, root / ROLE_TO_PATH[role], role=role)
    members = [
        _member_descriptor(root / relative, relative_path=relative, role=role)
        for role, relative in ROLE_TO_PATH.items()
    ]
    by_role = {item["artifact_role"]: item for item in members}
    component_manifest = _load_json_regular(
        root / COMPONENT_MANIFEST_RELATIVE_PATH, context="component manifest"
    )
    class_binding = _load_json_regular(root / CLASS_BINDING_RELATIVE_PATH, context="class binding")
    parity = _load_json_regular(root / PARITY_RECEIPT_RELATIVE_PATH, context="parity receipt")
    generation = _load_json_regular(root / GENERATION_LOCK_RELATIVE_PATH, context="generation lock")
    method = _load_json_regular(root / METHOD_LOCK_RELATIVE_PATH, context="method lock")
    bindings = {
        "checkpoint_lineage_sha256": _validate_sha(component_manifest.get("checkpoint_sha256"), "checkpoint lineage"),
        "runtime_sha256": by_role["sealed_torchscript_runtime"]["sha256"],
        "component_pre_sign_content_root_sha256": _validate_sha(
            component_manifest.get("pre_sign_content_root_sha256"), "component root"
        ),
        "class_handle_binding_sha256": _validate_sha(
            class_binding.get("class_handle_binding_sha256"),
            "class handle semantic binding",
        ),
        "parity_receipt_sha256": by_role["runtime_checkpoint_parity_receipt"]["sha256"],
        "generation_lock_sha256": by_role["generation_lock"]["sha256"],
        "method_lock_sha256": by_role["method_lock"]["sha256"],
        "generation_config_sha256": _validate_sha(component_manifest.get("generation_config_sha256"), "generation config"),
        "generation_code_sha256": _validate_sha(component_manifest.get("generation_code_sha256"), "generation code"),
    }
    if component_manifest.get("class_handle_binding_sha256") != bindings["class_handle_binding_sha256"]:
        raise ADV3B02DeploymentBundleError("component/class binding digest drift")
    _validate_lock_documents(
        class_binding=class_binding,
        parity=parity,
        generation=generation,
        method=method,
        bindings=bindings,
    )
    actual_runtime_structure, _ = _runtime_structure_from_bytes(runtime_source.read_bytes())
    if any(parity.get(key) != value for key, value in actual_runtime_structure.items()):
        raise ADV3B02DeploymentBundleError("runtime structure receipt drift")
    manifest = {
        "schema": BUNDLE_MANIFEST_SCHEMA,
        "artifact_stage": "phase1_offline_joint_deployment_bundle",
        "members": members,
        "outer_content_root_sha256": _content_root(members),
        **bindings,
    }
    _validate_manifest(manifest)
    manifest_path = root / MANIFEST_RELATIVE_PATH
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    _validate_root_allowlist(
        root, allowed_files=set(ROLE_TO_PATH.values()) | {MANIFEST_RELATIVE_PATH}
    )
    seal = {
        "schema": DETACHED_SEAL_SCHEMA,
        "manifest_relative_path": MANIFEST_RELATIVE_PATH,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "manifest_size_bytes": len(manifest_bytes),
        "outer_content_root_sha256": manifest["outer_content_root_sha256"],
        **bindings,
    }
    seal_bytes = canonical_json_bytes(seal) + b"\n"
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    with seal_path.open("xb") as handle:
        handle.write(seal_bytes)
    unsigned_envelope = {
        "schema": SIGNATURE_ENVELOPE_SCHEMA,
        "domain": SIGNATURE_DOMAIN,
        "issuer": runtime_trust.PINNED_AUTHORITY_ISSUER,
        "key_id": runtime_trust.PINNED_AUTHORITY_KEY_ID,
        "detached_seal_sha256": sha256_bytes(seal_bytes),
    }
    signing_request = {
        "schema": SIGNING_REQUEST_SCHEMA,
        "signature_message_sha256": sha256_bytes(
            SIGNATURE_DOMAIN.encode("ascii")
            + b"\x00"
            + canonical_json_bytes(unsigned_envelope)
        ),
        "unsigned_signature_envelope": unsigned_envelope,
        "outer_content_root_sha256": manifest["outer_content_root_sha256"],
    }
    request_path.parent.mkdir(parents=True, exist_ok=True)
    with request_path.open("xb") as handle:
        handle.write(canonical_json_bytes(signing_request) + b"\n")
    return {
        "manifest_path": str(manifest_path),
        "detached_seal_path": str(seal_path),
        "signing_request_path": str(request_path),
        "outer_content_root_sha256": manifest["outer_content_root_sha256"],
        "detached_seal_sha256": sha256_bytes(seal_bytes),
        **bindings,
    }


def _validate_component_from_opened(
    *, manifest: Mapping[str, Any], manifest_raw: bytes, sha_raw: bytes,
    npz_sha256: str, npz_size: int | None = None, arrays: Mapping[str, np.ndarray], bindings: Mapping[str, str]
) -> CenterLowRankPrototypeComponent:
    if set(manifest) != FINAL_MANIFEST_FIELDS:
        raise ADV3B02DeploymentBundleError("component manifest field set mismatch")
    if manifest.get("schema") != COMPONENT_SCHEMA or int(manifest.get("residual_rank", -1)) != RESIDUAL_RANK:
        raise ADV3B02DeploymentBundleError("component schema/rank drift")
    if manifest.get("formal_phase2_eligible") is not False or manifest.get("component_state") != PENDING_OUTER_JOINT_SEAL:
        raise ADV3B02DeploymentBundleError("standalone component formal-state drift")
    if manifest.get("outer_bundle_signature_required") is not True:
        raise ADV3B02DeploymentBundleError("component outer-signature requirement drift")
    if manifest.get("member_allowlist") != [COMPONENT_NPZ_NAME] or manifest.get("npz_member_allowlist") != sorted(ALLOWED_NPZ_MEMBERS):
        raise ADV3B02DeploymentBundleError("component member allowlist drift")
    expected_sidecar = f"{hashlib.sha256(manifest_raw).hexdigest()}  {COMPONENT_MANIFEST_NAME}\n"
    try:
        normalized_sidecar = sha_raw.decode("ascii").replace("\r\n", "\n")
    except UnicodeDecodeError as exc:
        raise ADV3B02DeploymentBundleError("component manifest SHA sidecar invalid") from exc
    if normalized_sidecar != expected_sidecar:
        raise ADV3B02DeploymentBundleError("component manifest SHA sidecar mismatch")
    if manifest.get("component_npz_sha256") != npz_sha256:
        raise ADV3B02DeploymentBundleError("component NPZ SHA mismatch")
    if manifest.get("pre_sign_content_root_sha256") != _component_pre_sign_content_root(manifest, npz_sha256):
        raise ADV3B02DeploymentBundleError("component pre-sign root mismatch")
    component_expected = {
        "checkpoint_sha256": bindings["checkpoint_lineage_sha256"],
        "class_handle_binding_sha256": bindings["class_handle_binding_sha256"],
        "pre_sign_content_root_sha256": bindings["component_pre_sign_content_root_sha256"],
        "generation_config_sha256": bindings["generation_config_sha256"],
        "generation_code_sha256": bindings["generation_code_sha256"],
    }
    for key, expected in component_expected.items():
        if manifest.get(key) != expected:
            raise ADV3B02DeploymentBundleError(f"component binding drift for {key}")
    if set(arrays) != ALLOWED_NPZ_MEMBERS:
        raise ADV3B02DeploymentBundleError("component NPZ allowlist drift")
    details = _validate_component_payload(arrays)
    for index, handle in enumerate(details["classes"]):
        _validate_opaque_token(handle, field=f"component class_handle[{index}]")
    for index, handle in enumerate(details["domains"]):
        _validate_domain_handle(handle, field=f"component domain_handle[{index}]")
    _validate_domain_handle(details["center"], field="component center_domain_handle")
    if int(manifest.get("feature_dim", -1)) != FEATURE_DIM:
        raise ADV3B02DeploymentBundleError("component feature dimension drift")
    if manifest.get("center_domain_handle") != details["center"]:
        raise ADV3B02DeploymentBundleError("component center handle drift")
    kwargs = {
        "core_q": np.array(arrays["core_q"], copy=True),
        "core_scale": np.array(arrays["core_scale"], copy=True),
        "residual_basis_q": np.array(arrays["residual_basis_q"], copy=True),
        "residual_basis_scale": np.array(arrays["residual_basis_scale"], copy=True),
        "residual_coeff_q": np.array(arrays["residual_coeff_q"], copy=True),
        "residual_coeff_scale": np.array(arrays["residual_coeff_scale"], copy=True),
        "radius_q": np.array(arrays["radius_q"], copy=True),
        "radius_scale": np.array(arrays["radius_scale"], copy=True),
        "domain_registry": _string_registry(arrays["domain_registry"], "domain_registry"),
        "residual_domain_registry": _string_registry(arrays["residual_domain_registry"], "residual_domain_registry"),
        "class_registry": _string_registry(arrays["class_registry"], "class_registry"),
        "center_domain_handle": _scalar_string(arrays["center_domain_handle"], "center_domain_handle"),
        "manifest": dict(manifest),
    }
    return CenterLowRankPrototypeComponent(**kwargs)


def _validate_grb_component_from_opened(
    *,
    manifest: Mapping[str, Any],
    manifest_raw: bytes,
    sha_raw: bytes,
    npz_sha256: str,
    npz_size: int,
    arrays: Mapping[str, np.ndarray],
    bindings: Mapping[str, str],
) -> GRBJP4CompactComponent:
    """Validate the GRB profile inside the unchanged eight-member package."""

    if set(manifest) != GRB_FINAL_MANIFEST_FIELDS:
        raise ADV3B02DeploymentBundleError("GRB component manifest field set mismatch")
    if (
        manifest.get("schema") != GRB_COMPONENT_SCHEMA
        or manifest.get("component_profile") != COMPONENT_PROFILE_GRB_JP4_Q4
        or manifest.get("feature_schema") != GRB_FEATURE_SCHEMA
        or int(manifest.get("feature_dim", -1)) != GRB_FEATURE_DIM
        or int(manifest.get("hidden_dim", -1)) != GRB_HIDDEN_DIM
        or int(manifest.get("class_count", -1)) != GRB_CLASS_COUNT
        or int(manifest.get("rank", -1)) != GRB_RANK
    ):
        raise ADV3B02DeploymentBundleError("GRB component schema/profile drift")
    if (
        manifest.get("formal_phase2_eligible") is not False
        or manifest.get("component_state") != GRB_PENDING_OUTER_JOINT_SEAL
        or manifest.get("outer_bundle_signature_required") is not True
    ):
        raise ADV3B02DeploymentBundleError("GRB standalone component formal-state drift")
    if (
        manifest.get("member_allowlist") != [GRB_COMPONENT_NPZ_NAME]
        or manifest.get("npz_member_allowlist") != sorted(GRB_ALLOWED_NPZ_MEMBERS)
    ):
        raise ADV3B02DeploymentBundleError("GRB component member allowlist drift")
    if manifest.get("svd_sign_canonicalization") != "largest_abs_basis_entry_positive_lowest_index_tie_v1" or manifest.get("rounding_rule") != "numpy_rint_ties_to_even_v1":
        raise ADV3B02DeploymentBundleError("GRB component canonicalization drift")
    if manifest.get("quantization") != {
        "dtype": "int8",
        "scale_dtype": "float16",
        "mode": "symmetric_per_vector",
        "qmin": -127,
        "qmax": 127,
    }:
        raise ADV3B02DeploymentBundleError("GRB component quantization drift")
    expected_policy = {
        "phase2_authorized_phase1_model_knowledge_policy": GRB_COMPONENT_SCHEMA,
        "phase2_phase1_component_generation_stage": "phase1_offline_before_target_access",
        "phase2_phase1_component_payload": "int8_p_g_l_g_r_fp16_scales_plus_kappa_g_and_class_registry_only",
        "phase2_phase1_component_immutable": True,
        "phase2_phase1_component_update_access": False,
        "phase2_phase1_component_member_or_exemplar_access": False,
        "phase2_phase1_component_sample_reconstruction_access": False,
        "phase2_nonbundle_source_artifact_access": False,
    }
    if any(manifest.get(key) != value for key, value in expected_policy.items()):
        raise ADV3B02DeploymentBundleError("GRB component protocol manifest drift")
    for field in (
        "checkpoint_sha256",
        "class_handle_binding_sha256",
        "source_aggregate_generation_digest_sha256",
        "generation_code_sha256",
        "generation_config_sha256",
        "registry_sha256",
        "component_npz_sha256",
        "pre_sign_content_root_sha256",
    ):
        _validate_sha(manifest.get(field), f"GRB component {field}")
    expected_sidecar = f"{hashlib.sha256(manifest_raw).hexdigest()}  {COMPONENT_MANIFEST_NAME}\n"
    try:
        normalized_sidecar = sha_raw.decode("ascii").replace("\r\n", "\n")
    except UnicodeDecodeError as exc:
        raise ADV3B02DeploymentBundleError("GRB component manifest SHA sidecar invalid") from exc
    if normalized_sidecar != expected_sidecar:
        raise ADV3B02DeploymentBundleError("GRB component manifest SHA sidecar mismatch")
    if manifest.get("component_npz_sha256") != npz_sha256:
        raise ADV3B02DeploymentBundleError("GRB component NPZ SHA mismatch")
    if manifest.get("pre_sign_content_root_sha256") != _grb_pre_sign_content_root(
        manifest, npz_sha256
    ):
        raise ADV3B02DeploymentBundleError("GRB component pre-sign root mismatch")
    component_expected = {
        "checkpoint_sha256": bindings["checkpoint_lineage_sha256"],
        "class_handle_binding_sha256": bindings["class_handle_binding_sha256"],
        "pre_sign_content_root_sha256": bindings["component_pre_sign_content_root_sha256"],
        "generation_config_sha256": bindings["generation_config_sha256"],
        "generation_code_sha256": bindings["generation_code_sha256"],
    }
    for key, expected in component_expected.items():
        if manifest.get(key) != expected:
            raise ADV3B02DeploymentBundleError(f"GRB component binding drift for {key}")
    if set(arrays) != GRB_ALLOWED_NPZ_MEMBERS:
        raise ADV3B02DeploymentBundleError("GRB component NPZ allowlist drift")
    try:
        details = _validate_grb_payload(arrays)
        registry_digest = _grb_class_binding_digest(details["class_registry"])
    except ValueError as exc:
        raise ADV3B02DeploymentBundleError(f"GRB component payload invalid: {exc}") from exc
    if registry_digest != bindings["class_handle_binding_sha256"]:
        raise ADV3B02DeploymentBundleError("GRB component/class binding digest drift")
    if int(manifest.get("class_count", -1)) != len(details["class_registry"]):
        raise ADV3B02DeploymentBundleError("GRB component class count drift")
    if manifest.get("registry_sha256") != _grb_registry_sha256(details["class_registry"]):
        raise ADV3B02DeploymentBundleError("GRB component registry digest drift")
    recorded_audit = manifest.get("resource_audit")
    if not isinstance(recorded_audit, Mapping) or any(
        recorded_audit.get(key) != value
        for key, value in details["resource_audit"].items()
    ):
        raise ADV3B02DeploymentBundleError("GRB component resource audit drift")
    if (
        int(manifest.get("serialized_component_bytes", -1))
        != int(recorded_audit.get("serialized_component_bytes", -2))
        or int(manifest.get("serialized_component_bytes", -1))
        != int(npz_size)
    ):
        raise ADV3B02DeploymentBundleError("GRB component serialized byte receipt drift")
    try:
        kappa_g = float(manifest.get("kappa_g"))
    except (TypeError, ValueError) as exc:
        raise ADV3B02DeploymentBundleError("GRB component kappa scalar invalid") from exc
    if not np.isfinite(kappa_g) or kappa_g < 1.0:
        raise ADV3B02DeploymentBundleError("GRB component kappa scalar invalid")
    return GRBJP4CompactComponent(
        p_g_q=GRBJP4CompactComponent._readonly(arrays["p_g_q"], np.int8),
        p_g_scale=GRBJP4CompactComponent._readonly(arrays["p_g_scale"], np.float16),
        l_g_q=GRBJP4CompactComponent._readonly(arrays["l_g_q"], np.int8),
        l_g_scale=GRBJP4CompactComponent._readonly(arrays["l_g_scale"], np.float16),
        r_q=GRBJP4CompactComponent._readonly(arrays["r_q"], np.int8),
        r_scale=GRBJP4CompactComponent._readonly(arrays["r_scale"], np.float16),
        kappa_g=kappa_g,
        class_registry=details["class_registry"],
        manifest=dict(manifest),
    )


def load_formal_adv3b02_deployment_bundle(
    package_root: str | Path,
    *,
    detached_seal_path: str | Path,
    expected_detached_seal_sha256: str,
    signature_envelope_path: str | Path,
    expected_signature_envelope_sha256: str,
    expected_checkpoint_lineage_sha256: str,
    expected_runtime_sha256: str,
    expected_component_pre_sign_content_root_sha256: str,
    expected_class_handle_binding_sha256: str,
    expected_parity_receipt_sha256: str,
    expected_generation_lock_sha256: str,
    expected_method_lock_sha256: str,
    expected_generation_config_sha256: str,
    expected_generation_code_sha256: str,
    expected_outer_content_root_sha256: str,
) -> VerifiedADV3B02DeploymentBundle:
    """Verify the complete external trust chain, then materialize runtime state."""

    expected_bindings = {
        "checkpoint_lineage_sha256": expected_checkpoint_lineage_sha256,
        "runtime_sha256": expected_runtime_sha256,
        "component_pre_sign_content_root_sha256": expected_component_pre_sign_content_root_sha256,
        "class_handle_binding_sha256": expected_class_handle_binding_sha256,
        "parity_receipt_sha256": expected_parity_receipt_sha256,
        "generation_lock_sha256": expected_generation_lock_sha256,
        "method_lock_sha256": expected_method_lock_sha256,
        "generation_config_sha256": expected_generation_config_sha256,
        "generation_code_sha256": expected_generation_code_sha256,
    }
    expected_bindings = {key: _validate_sha(value, key) for key, value in expected_bindings.items()}
    expected_outer = _validate_sha(expected_outer_content_root_sha256, "outer content root")
    expected_seal = _validate_sha(expected_detached_seal_sha256, "detached seal")
    expected_envelope = _validate_sha(expected_signature_envelope_sha256, "signature envelope")

    seal_path = Path(detached_seal_path)
    envelope_path = Path(signature_envelope_path)
    seal_payload, _ = _load_external_json_same_handle(
        seal_path, expected_sha256=expected_seal, context="detached seal"
    )
    envelope_payload, _ = _load_external_json_same_handle(
        envelope_path,
        expected_sha256=expected_envelope,
        context="signature envelope",
    )
    seal = _json_exact(seal_payload, _SEAL_KEYS, context="detached seal")
    if seal["schema"] != DETACHED_SEAL_SCHEMA or seal["manifest_relative_path"] != MANIFEST_RELATIVE_PATH:
        raise ADV3B02DeploymentBundleError("detached seal schema/path drift")
    envelope = _json_exact(envelope_payload, _ENVELOPE_KEYS, context="signature envelope")
    pinned_envelope = {
        "schema": SIGNATURE_ENVELOPE_SCHEMA,
        "domain": SIGNATURE_DOMAIN,
        "issuer": runtime_trust.PINNED_AUTHORITY_ISSUER,
        "key_id": runtime_trust.PINNED_AUTHORITY_KEY_ID,
        "detached_seal_sha256": expected_seal,
    }
    if any(envelope.get(key) != value for key, value in pinned_envelope.items()):
        raise ADV3B02DeploymentBundleError("signature envelope binding drift")
    try:
        signature = bytes.fromhex(str(envelope["signature_ed25519_hex"]))
    except ValueError as exc:
        raise ADV3B02DeploymentBundleError("signature envelope hex invalid") from exc
    message = _signature_message(envelope)
    public_key = bytes.fromhex(runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_HEX)
    if (
        len(public_key) != 32
        or hashlib.sha256(public_key).hexdigest()
        != runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_SHA256
    ):
        raise ADV3B02DeploymentBundleError("pinned authority public-key hash drift")
    try:
        runtime_trust.verify_ed25519(public_key, message, signature)
    except Exception as exc:
        raise ADV3B02DeploymentBundleError(
            "deployment bundle authority signature invalid"
        ) from exc

    root = _ensure_root(Path(package_root))
    with open_regular_member_same_fd(root, MANIFEST_RELATIVE_PATH) as handle:
        if handle.seek(0, io.SEEK_END) > MAX_MANIFEST_BYTES:
            raise ADV3B02DeploymentBundleError("deployment manifest exceeds size limit")
        handle.seek(0)
        manifest_sha, manifest_size = _hash_handle(handle)
        manifest = _json_from_handle(handle, context="deployment manifest")
    manifest, members = _validate_manifest(manifest)
    if manifest_sha != seal["manifest_sha256"] or manifest_size != seal["manifest_size_bytes"]:
        raise ADV3B02DeploymentBundleError("manifest detached-seal digest drift")
    if manifest["outer_content_root_sha256"] != expected_outer or seal["outer_content_root_sha256"] != expected_outer:
        raise ADV3B02DeploymentBundleError("outer root external binding drift")
    for key, expected in expected_bindings.items():
        if manifest[key] != expected or seal[key] != expected:
            raise ADV3B02DeploymentBundleError(f"external binding drift for {key}")
    _validate_root_allowlist(root, allowed_files=set(ROLE_TO_PATH.values()) | {MANIFEST_RELATIVE_PATH})

    opened: dict[str, bytes] = {}
    by_role = {item["artifact_role"]: item for item in members}
    for role, descriptor in by_role.items():
        limit = {
            "sealed_torchscript_runtime": MAX_RUNTIME_FILE_BYTES,
            "v2_component_npz": MAX_COMPONENT_NPZ_BYTES,
        }.get(role, MAX_LOCK_JSON_BYTES)
        if descriptor["size_bytes"] > limit:
            raise ADV3B02DeploymentBundleError(
                f"deployment member exceeds size limit: {role}"
            )
        with open_regular_member_same_fd(root, descriptor["relative_path"]) as handle:
            digest, size = _hash_handle(handle)
            if digest != descriptor["sha256"] or size != descriptor["size_bytes"]:
                raise ADV3B02DeploymentBundleError(f"deployment member digest drift: {role}")
            opened[role] = handle.read()

    json_roles = {
        "v2_component_manifest",
        "class_handle_binding",
        "runtime_checkpoint_parity_receipt",
        "generation_lock",
        "method_lock",
    }
    documents: dict[str, dict[str, Any]] = {}
    for role in json_roles:
        try:
            value = json.loads(opened[role].decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ADV3B02DeploymentBundleError(f"invalid JSON deployment member: {role}") from exc
        if not isinstance(value, dict):
            raise ADV3B02DeploymentBundleError(f"JSON member must be object: {role}")
        documents[role] = value
    _validate_lock_documents(
        class_binding=documents["class_handle_binding"],
        parity=documents["runtime_checkpoint_parity_receipt"],
        generation=documents["generation_lock"],
        method=documents["method_lock"],
        bindings=expected_bindings,
    )
    npz_handle = io.BytesIO(opened["v2_component_npz"])
    component_schema = str(documents["v2_component_manifest"].get("schema", ""))
    component_profile = component_profile_for_schema(component_schema)["component_profile"]
    actual_npz_members = _zip_members_from_handle(
        npz_handle, context="joint-sealed component"
    )
    expected_npz_members = (
        ALLOWED_NPZ_MEMBERS
        if component_profile == COMPONENT_PROFILE_CENTER_LOWRANK_V2
        else GRB_ALLOWED_NPZ_MEMBERS
    )
    if set(actual_npz_members) != expected_npz_members:
        raise ADV3B02DeploymentBundleError("component NPZ ZIP member allowlist drift")
    with np.load(npz_handle, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in actual_npz_members}
    component_validator = (
        _validate_component_from_opened
        if component_profile == COMPONENT_PROFILE_CENTER_LOWRANK_V2
        else _validate_grb_component_from_opened
    )
    component = component_validator(
        manifest=documents["v2_component_manifest"],
        manifest_raw=opened["v2_component_manifest"],
        sha_raw=opened["v2_component_manifest_sha256"],
        npz_sha256=by_role["v2_component_npz"]["sha256"],
        npz_size=by_role["v2_component_npz"]["size_bytes"],
        arrays=arrays,
        bindings=expected_bindings,
    )
    bound_handles = [
        row["class_handle"]
        for row in documents["class_handle_binding"]["class_id_to_handle"]
    ]
    if list(component.class_registry) != bound_handles:
        raise ADV3B02DeploymentBundleError("component/class registry ordered binding drift")
    actual_runtime_structure, runtime = _runtime_structure_from_bytes(
        opened["sealed_torchscript_runtime"]
    )
    if any(
        documents["runtime_checkpoint_parity_receipt"].get(key) != value
        for key, value in actual_runtime_structure.items()
    ):
        raise ADV3B02DeploymentBundleError("runtime structure receipt drift")
    inner_component_filename = documents["v2_component_manifest"]["member_allowlist"][0]
    outer_component_slot_relative_path = by_role["v2_component_npz"]["relative_path"]
    verification_receipt = {
        "schema": "cvs.phase1.adv3b02_verified_bundle_receipt.v1",
        "outer_content_root_sha256": expected_outer,
        "member_table_sha256": sha256_bytes(
            canonical_json_bytes(
                sorted(
                    (
                        {
                            "artifact_role": item["artifact_role"],
                            "relative_path": item["relative_path"],
                            "sha256": item["sha256"],
                            "size_bytes": item["size_bytes"],
                        }
                        for item in members
                    ),
                    key=lambda item: item["artifact_role"],
                )
            )
        ),
        "component_profile": component_profile,
        "component_outer_slot_sha256": by_role["v2_component_npz"]["sha256"],
        "component_manifest_sha256": by_role["v2_component_manifest"]["sha256"],
        "component_manifest_semantic_sha256": sha256_bytes(
            canonical_json_bytes(documents["v2_component_manifest"])
        ),
        "checkpoint_lineage_sha256": expected_bindings["checkpoint_lineage_sha256"],
        "runtime_sha256": by_role["sealed_torchscript_runtime"]["sha256"],
        "runtime_structure_sha256": actual_runtime_structure["runtime_structure_sha256"],
        "parity_receipt_sha256": by_role["runtime_checkpoint_parity_receipt"]["sha256"],
        "parity_receipt_semantic_sha256": sha256_bytes(
            canonical_json_bytes(documents["runtime_checkpoint_parity_receipt"])
        ),
        "class_binding_semantic_sha256": sha256_bytes(
            canonical_json_bytes(documents["class_handle_binding"])
        ),
        "generation_lock_semantic_sha256": sha256_bytes(
            canonical_json_bytes(documents["generation_lock"])
        ),
        "method_lock_sha256": by_role["method_lock"]["sha256"],
        "method_lock_semantic_sha256": sha256_bytes(
            canonical_json_bytes(documents["method_lock"])
        ),
        "external_signature_envelope_sha256": expected_envelope,
        "detached_seal_sha256": expected_seal,
    }
    verification_receipt_sha256 = sha256_bytes(
        canonical_json_bytes(verification_receipt)
    )
    formal_context = {
        "schema": FORMAL_CONTEXT_SCHEMA,
        "formal_phase2_eligible": True,
        "standalone_component_formal_phase2_eligible": False,
        "component_profile": component_profile,
        # The inner artifact retains its profile-specific filename while the
        # established eight-member container uses one immutable slot.  Both
        # facts are independently committed by the pre-sign and outer roots.
        "component_inner_filename": inner_component_filename,
        "component_outer_slot_relative_path": outer_component_slot_relative_path,
        "component_outer_slot_sha256": by_role["v2_component_npz"]["sha256"],
        "outer_signature_verified": True,
        "detached_seal_verified": True,
        "runtime_checkpoint_parity_verified": True,
        "outer_content_root_sha256": expected_outer,
        "verified_bundle_receipt_sha256": verification_receipt_sha256,
        # Stage2 consumes a checkpoint SHA under its layer-binding name;
        # retain the existing lineage name as the external formal contract.
        "checkpoint_sha256": expected_bindings["checkpoint_lineage_sha256"],
        **expected_bindings,
    }
    audit = {
        "schema": "cvs.phase1.adv3b02_deployment_bundle_load_audit.v1",
        "status": "PASS",
        "hash_and_materialization_same_file_descriptor": True,
        "exact_root_member_allowlist": True,
        "raw_training_checkpoint_present": False,
        "external_signature_envelope_sha256": expected_envelope,
        "detached_seal_sha256": expected_seal,
        "outer_content_root_sha256": expected_outer,
        "verified_bundle_receipt_sha256": verification_receipt_sha256,
        "component_inner_filename": inner_component_filename,
        "component_outer_slot_relative_path": outer_component_slot_relative_path,
        "component_inner_filename_bound_by_pre_sign_root": True,
        "component_outer_slot_bound_by_outer_content_root": True,
    }
    _validate_root_allowlist(
        root,
        allowed_files=set(ROLE_TO_PATH.values()) | {MANIFEST_RELATIVE_PATH},
    )
    runtime_member_path = str(
        (root / by_role["sealed_torchscript_runtime"]["relative_path"]).resolve()
    )
    formal_reverify_kwargs = {
        "detached_seal_path": str(seal_path.resolve()),
        "expected_detached_seal_sha256": expected_seal,
        "signature_envelope_path": str(envelope_path.resolve()),
        "expected_signature_envelope_sha256": expected_envelope,
        "expected_checkpoint_lineage_sha256": expected_bindings[
            "checkpoint_lineage_sha256"
        ],
        "expected_runtime_sha256": expected_bindings["runtime_sha256"],
        "expected_component_pre_sign_content_root_sha256": expected_bindings[
            "component_pre_sign_content_root_sha256"
        ],
        "expected_class_handle_binding_sha256": expected_bindings[
            "class_handle_binding_sha256"
        ],
        "expected_parity_receipt_sha256": expected_bindings[
            "parity_receipt_sha256"
        ],
        "expected_generation_lock_sha256": expected_bindings[
            "generation_lock_sha256"
        ],
        "expected_method_lock_sha256": expected_bindings["method_lock_sha256"],
        "expected_generation_config_sha256": expected_bindings[
            "generation_config_sha256"
        ],
        "expected_generation_code_sha256": expected_bindings[
            "generation_code_sha256"
        ],
        "expected_outer_content_root_sha256": expected_outer,
    }
    coordinator_sha = _verified_coordinator_sha256(
        runtime_member_path=runtime_member_path,
        component=component,
        class_binding=documents["class_handle_binding"],
        parity_receipt=documents["runtime_checkpoint_parity_receipt"],
        generation_lock=documents["generation_lock"],
        method_lock=documents["method_lock"],
        formal_phase2_context=formal_context,
        verification_receipt=verification_receipt,
        audit=audit,
    )
    return _issue_verified_bundle_after_external_verification(
        runtime=runtime,
        runtime_member_path=runtime_member_path,
        component=component,
        class_binding=documents["class_handle_binding"],
        parity_receipt=documents["runtime_checkpoint_parity_receipt"],
        generation_lock=documents["generation_lock"],
        method_lock=documents["method_lock"],
        formal_phase2_context=formal_context,
        verification_receipt=verification_receipt,
        audit=audit,
        _formal_reverify_package_root=str(root.resolve()),
        _formal_reverify_kwargs=formal_reverify_kwargs,
        _issued_coordinator_sha256=coordinator_sha,
        _issued_runtime_identity=id(runtime),
        _formal_runtime_lifecycle={
            "schema": "cvs.phase1.adv3b02_runtime_ownership.v2",
            "source_bundle_runtime_consumed": False,
            "source_runtime_weakref_released_before_reload": False,
            "formal_runtime_weakref_live_after_materialization": True,
        },
    )


__all__ = [
    "ADV3B02DeploymentBundleError",
    "BUNDLE_MANIFEST_SCHEMA",
    "COMPONENT_PROFILE_CENTER_LOWRANK_V2",
    "COMPONENT_PROFILE_GRB_JP4_Q4",
    "COMPONENT_PROFILE_SCHEMA",
    "DETACHED_SEAL_SCHEMA",
    "FORMAL_CONTEXT_SCHEMA",
    "SIGNATURE_DOMAIN",
    "SIGNATURE_ENVELOPE_SCHEMA",
    "SIGNING_REQUEST_SCHEMA",
    "VerifiedADV3B02DeploymentBundle",
    "build_unsigned_adv3b02_deployment_bundle",
    "class_handle_binding_sha256",
    "component_profile_for_schema",
    "load_formal_adv3b02_deployment_bundle",
    "reverify_formal_adv3b02_deployment_bundle",
    "runtime_structure_receipt",
]
