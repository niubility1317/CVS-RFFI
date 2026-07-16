"""Offline-only SOMP-H Stage2-C row-pair package production.

This module is outside the Phase2 predictor boundary.  It may read the
byte-verified target cache's transmitter labels and roles, but it emits only
opaque HMAC handles into predictor roots.  Scorer truth stays in a physically
separate root and is never referenced by an enrollment/apply bundle.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from cvsrffi import somph_leo_weak_lineage_seal as lineage
from cvsrffi import somph_lineage_authority as authority
from cvsrffi import somph_predictor_bundle as bundle
from cvsrffi.somph_formal_matrix import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEED,
    FORMAL_K_VALUES,
    FORMAL_NEW_CLASS_COUNTS,
    FORMAL_RECEIVERS,
)
from cvsrffi.somph_metric_scorer import (
    FORMAL_NEW20_TX_LABELS,
    FORMAL_OLD_TX_LABELS,
    REGISTRATION_PAIR_SCHEMA,
)
from cvsrffi.stage2_metric_scorer import TRUTH_SIDECAR_SCHEMA
from cvsrffi.stage2_predictor_bundle import (
    _hash_handle,
    _json_from_handle,
    _zip_members_from_handle,
    canonical_json_bytes,
    open_regular_member_same_fd,
    sha256_bytes,
    sha256_file,
)


OFFLINE_BUILD_SCHEMA = "cvs.phase2.somph_offline_row_pair_build.v2"
PAIR_STAGING_SCHEMA = "cvs.phase2.somph_registration_pair_staging.v2"
ROW_MANIFEST_SCHEMA = "cvs.phase2.somph_row_manifest.v2"
APPLY_FINALIZATION_SCHEMA = "cvs.phase2.somph_apply_finalization.v1"
FORMAL_APPLY_STAGING_AUTHORITY_SCHEMA = (
    "cvs.phase2.somph_formal_apply_staging_authority.v2"
)
FORMAL_APPLY_STAGING_AUTHORITY_SEAL_SCHEMA = (
    "cvs.phase2.somph_formal_apply_staging_authority_seal.v2"
)
DIAGNOSTIC_APPLY_STAGING_AUTHORITY_SCHEMA = (
    "cvs.phase2.somph_diagnostic_apply_staging_authority.v2"
)
DIAGNOSTIC_APPLY_STAGING_AUTHORITY_SEAL_SCHEMA = (
    "cvs.phase2.somph_diagnostic_apply_staging_authority_seal.v2"
)
VERIFIED_LINEAGE_CONTEXT_SCHEMA = (
    "cvs.phase2.somph_verified_lineage_context.v2"
)
SUPPORT_POOL_MAX_K = 20

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_LINEAGE_RECEIPT_KEYS = {
    "schema",
    "status",
    "cache_scope",
    "scenario_order",
    "cache_set_manifest_sha256",
    "cache_set_manifest_size_bytes",
    "exporter_sha256",
    "exporter_size_bytes",
    "build_spec_sha256",
    "build_spec_size_bytes",
    "channel_code_closure_sha256",
    "channel_code_members",
    "physical_sample_ids_sha256_by_scenario",
    "physical_sample_scenario_assignment_sha256",
    "scenario_receipts",
    "same_fd_nofollow_read",
    "npz_member_crc_size_ratio_audit",
    "cross_scenario_physical_disjointness_audit",
    "single_observation_contract_audit",
    "sample_level_overlay_recompute",
    "manifest_hex_self_declaration_sufficient",
    "external_authority_lock_verified",
    "contains_build_spec_or_dataset_paths",
    "formal_launch_authority",
}
_LINEAGE_SEAL_KEYS = {
    "schema",
    "receipt_sha256",
    "receipt_size_bytes",
    "lineage_root_sha256",
}
_SCENARIO_RECEIPT_KEYS = {
    "cache_sha256",
    "cache_size_bytes",
    "cache_manifest_sha256",
    "channel_config_sha256",
    "physical_sample_ids_sha256",
    "post_channel_iq_sha256_root",
    "overlay_ids_sha256",
    "row_count",
    "zip_member_crc_and_bounds_check",
    "sample_level_overlay_recompute",
}
_CACHE_SET_KEYS = {
    "schema",
    "artifact_stage",
    "cache_set_id",
    "cache_scope",
    "phase2_sample_view_policy",
    "clean_sample_access",
    "clean_derived_signal_access",
    "target_channel_view",
    "target_channel_scenarios",
    "output_roles",
    "cache_npz_by_scenario",
    "cache_sha256_by_scenario",
    "cache_audits",
    "phase2_physical_sample_observation_policy",
    "phase2_cross_scenario_physical_sample_reuse",
    "phase2_additional_leo_channel_state_generation",
    "phase2_post_reception_equalization_augmentation_transform_allowed",
    "phase2_post_reception_view_from_fixed_received_iq_only",
    "phase2_post_reception_view_counts_as_additional_physical_sample",
    "phase2_physical_sample_root_id_policy",
    "phase2_query_post_reception_view_fit_access",
    "physical_sample_scenario_assignment_policy",
    "physical_sample_ids_sha256_by_scenario",
    "physical_sample_scenario_assignment_sha256",
    "builder_sha256",
    "build_spec_sha256",
    "build_spec_path_exposed_to_phase2",
}
_VERIFIED_CONTEXT_KEYS = {
    "schema",
    "status",
    "receipt",
    "lineage_seal",
    "cache_set",
    "cache_set_manifest_path",
    "lineage_receipt_sha256",
    "lineage_seal_sha256",
    "cache_set_manifest_sha256",
    "authority_lock",
    "authority_attestation",
    "authority_commit",
    "authority_commit_sha256",
    "authority_attestation_sha256",
    "external_authority_lock_verified",
    "formal_launch_authority",
}
_APPLY_STAGING_AUTHORITY_KEYS = {
    "schema",
    "status",
    "profile",
    "stage",
    "registration_state",
    "receiver",
    "seed",
    "k_shot",
    "row_handle",
    "row_manifest_sha256",
    "registered_classes",
    "registered_classes_sha256",
    "enrollment_package_root_sha256",
    "enrollment_package_seal_sha256",
    "phase1_checkpoint_sha256",
    "feature_runtime_sha256",
    "method_lock_sha256",
    "apply_staging_root",
    "apply_overlay_provenance_sha256",
    "cache_set_manifest_sha256",
    "cache_physical_sample_ids_sha256_by_scenario",
    "cache_physical_sample_scenario_assignment_sha256",
    "lineage_receipt_sha256",
    "lineage_seal_sha256",
    "authority_commit_sha256",
    "authority_attestation_sha256",
    "external_authority_lock_verified",
    "formal_launch_authority",
}
_APPLY_STAGING_AUTHORITY_SEAL_KEYS = {
    "schema",
    "authority_file_name",
    "authority_sha256",
    "authority_size_bytes",
}


class SomphOfflinePackageError(ValueError):
    """Raised when offline row-pair construction cannot remain protocol-safe."""


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SomphOfflinePackageError(f"{field} must be a lowercase SHA256")
    return value


def _canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _require_scenario_sha_map(
    value: Any,
    *,
    field: str,
) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or tuple(value) != bundle.FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise SomphOfflinePackageError(
            f"{field} must use the exact formal scenario order"
        )
    return {
        scenario: _require_sha256(
            value[scenario],
            field=f"{field}.{scenario}",
        )
        for scenario in bundle.FORMAL_LEO_WEAK_SCENARIOS
    }


def _read_json_same_fd(path: str | Path, *, context: str) -> tuple[dict[str, Any], str, int]:
    candidate = Path(path)
    try:
        with open_regular_member_same_fd(
            candidate.parent.resolve(), candidate.name
        ) as handle:
            digest, size = _hash_handle(handle)
            payload = _json_from_handle(handle, context=context)
    except Exception as exc:
        if isinstance(exc, SomphOfflinePackageError):
            raise
        raise SomphOfflinePackageError(f"{context} could not be read safely") from exc
    return payload, digest, size


def _write_new(path: Path, payload: bytes, *, readonly: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (
        f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, 0o600)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite output: {path}")
        temporary.unlink()
        if readonly:
            os.chmod(path, 0o444)
        published = True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            try:
                os.chmod(temporary, 0o600)
                temporary.unlink()
            except OSError:
                if not published:
                    raise


def _write_json_new(path: Path, payload: Mapping[str, Any], *, readonly: bool = True) -> None:
    _write_new(path, canonical_json_bytes(dict(payload)) + b"\n", readonly=readonly)


def _copy_regular_new(
    source: str | Path,
    destination: Path,
    *,
    expected_sha256: str | None = None,
) -> str:
    source_path = Path(source)
    temporary = destination.parent / (
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    )
    try:
        with open_regular_member_same_fd(
            source_path.parent.resolve(), source_path.name
        ) as input_handle:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            output_fd = os.open(temporary, flags, 0o600)
            digest = hashlib.sha256()
            try:
                with os.fdopen(output_fd, "wb", closefd=False) as output_handle:
                    for chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                        output_handle.write(chunk)
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
            finally:
                os.close(output_fd)
            if (
                expected_sha256 is not None
                and digest.hexdigest() != expected_sha256
            ):
                raise SomphOfflinePackageError(
                    f"sealed input SHA mismatch: {source_path.name}"
                )
            try:
                os.link(temporary, destination)
            except FileExistsError:
                raise FileExistsError(
                    f"refusing to overwrite sealed input: {destination}"
                )
    except SomphOfflinePackageError:
        raise
    except Exception as exc:
        raise SomphOfflinePackageError(
            f"could not copy sealed input: {source_path.name}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def _remove_owned_file(path: Path) -> None:
    if not path.exists():
        return
    os.chmod(path, 0o600)
    path.unlink()


def _write_apply_staging_authority(
    *,
    path: Path,
    seal_path: Path,
    stage: str,
    registration_state: str,
    receiver: str,
    seed: int,
    k_shot: int,
    row_handle: str,
    row_manifest_sha256: str,
    registered_classes: list[Mapping[str, Any]],
    enrollment_package_root_sha256: str,
    enrollment_package_seal_sha256: str,
    phase1_checkpoint_sha256: str,
    feature_runtime_sha256: str,
    method_lock_sha256: str,
    apply_staging_root: Path,
    apply_overlay_provenance_sha256: str,
    cache_set_manifest_sha256: str,
    cache_physical_sample_ids_sha256_by_scenario: Mapping[str, str],
    cache_physical_sample_scenario_assignment_sha256: str,
    lineage_receipt_sha256: str,
    lineage_seal_sha256: str,
    authority_commit_sha256: str | None,
    authority_attestation_sha256: str | None,
    external_authority_lock_verified: bool,
    formal_profile: bool,
) -> tuple[str, str]:
    registry = [dict(value) for value in registered_classes]
    bundle._validate_registry(registry, len(registry))
    if formal_profile:
        staging_schema = FORMAL_APPLY_STAGING_AUTHORITY_SCHEMA
        seal_schema = FORMAL_APPLY_STAGING_AUTHORITY_SEAL_SCHEMA
        staging_status = "SEALED_FORMAL_APPLY_STAGING_AUTHORITY"
        profile = "formal_external_authority"
        if (
            external_authority_lock_verified is not True
            or authority_commit_sha256 is None
            or authority_attestation_sha256 is None
        ):
            raise SomphOfflinePackageError(
                "formal staging requires verified external authority"
            )
    else:
        staging_schema = DIAGNOSTIC_APPLY_STAGING_AUTHORITY_SCHEMA
        seal_schema = DIAGNOSTIC_APPLY_STAGING_AUTHORITY_SEAL_SCHEMA
        staging_status = "SEALED_DIAGNOSTIC_APPLY_STAGING_AUTHORITY"
        profile = "diagnostic_structural_only"
        external_authority_lock_verified = False
        authority_commit_sha256 = None
        authority_attestation_sha256 = None
    cache_physical_roots = _require_scenario_sha_map(
        cache_physical_sample_ids_sha256_by_scenario,
        field="cache_physical_sample_ids_sha256_by_scenario",
    )
    payload = {
        "schema": staging_schema,
        "status": staging_status,
        "profile": profile,
        "stage": stage,
        "registration_state": registration_state,
        "receiver": receiver,
        "seed": seed,
        "k_shot": k_shot,
        "row_handle": row_handle,
        "row_manifest_sha256": row_manifest_sha256,
        "registered_classes": registry,
        "registered_classes_sha256": _canonical_sha256(registry),
        "enrollment_package_root_sha256": enrollment_package_root_sha256,
        "enrollment_package_seal_sha256": enrollment_package_seal_sha256,
        "phase1_checkpoint_sha256": phase1_checkpoint_sha256,
        "feature_runtime_sha256": feature_runtime_sha256,
        "method_lock_sha256": method_lock_sha256,
        "apply_staging_root": str(apply_staging_root.resolve()),
        "apply_overlay_provenance_sha256": (
            apply_overlay_provenance_sha256
        ),
        "cache_set_manifest_sha256": cache_set_manifest_sha256,
        "cache_physical_sample_ids_sha256_by_scenario": (
            cache_physical_roots
        ),
        "cache_physical_sample_scenario_assignment_sha256": _require_sha256(
            cache_physical_sample_scenario_assignment_sha256,
            field="cache_physical_sample_scenario_assignment_sha256",
        ),
        "lineage_receipt_sha256": lineage_receipt_sha256,
        "lineage_seal_sha256": lineage_seal_sha256,
        "authority_commit_sha256": authority_commit_sha256,
        "authority_attestation_sha256": authority_attestation_sha256,
        "external_authority_lock_verified": (
            external_authority_lock_verified
        ),
        "formal_launch_authority": False,
    }
    _write_json_new(path, payload)
    authority_sha = sha256_file(path)
    seal = {
        "schema": seal_schema,
        "authority_file_name": path.name,
        "authority_sha256": authority_sha,
        "authority_size_bytes": path.stat().st_size,
    }
    try:
        _write_json_new(seal_path, seal)
    except Exception:
        _remove_owned_file(path)
        raise
    return authority_sha, sha256_file(seal_path)


def _load_apply_staging_authority(
    *,
    authority_path: str | Path,
    authority_seal_path: str | Path,
    expected_authority_seal_sha256: str,
    require_external_authority: bool,
) -> dict[str, Any]:
    expected_authority_schema = (
        FORMAL_APPLY_STAGING_AUTHORITY_SCHEMA
        if require_external_authority
        else DIAGNOSTIC_APPLY_STAGING_AUTHORITY_SCHEMA
    )
    expected_seal_schema = (
        FORMAL_APPLY_STAGING_AUTHORITY_SEAL_SCHEMA
        if require_external_authority
        else DIAGNOSTIC_APPLY_STAGING_AUTHORITY_SEAL_SCHEMA
    )
    expected_status = (
        "SEALED_FORMAL_APPLY_STAGING_AUTHORITY"
        if require_external_authority
        else "SEALED_DIAGNOSTIC_APPLY_STAGING_AUTHORITY"
    )
    expected_profile = (
        "formal_external_authority"
        if require_external_authority
        else "diagnostic_structural_only"
    )
    expected_seal = _require_sha256(
        expected_authority_seal_sha256,
        field="expected_staging_authority_seal_sha256",
    )
    seal, seal_sha, _seal_size = _read_json_same_fd(
        authority_seal_path,
        context="SOMP-H apply staging authority seal",
    )
    if seal_sha != expected_seal:
        raise SomphOfflinePackageError(
            "apply staging authority external seal SHA mismatch"
        )
    authority_file = Path(authority_path)
    if (
        set(seal) != _APPLY_STAGING_AUTHORITY_SEAL_KEYS
        or seal.get("schema") != expected_seal_schema
        or seal.get("authority_file_name") != authority_file.name
        or not isinstance(seal.get("authority_size_bytes"), int)
        or seal["authority_size_bytes"] < 1
    ):
        raise SomphOfflinePackageError(
            "apply staging authority seal exact schema drift"
        )
    authority_payload, authority_sha, authority_size = _read_json_same_fd(
        authority_file,
        context="SOMP-H apply staging authority",
    )
    if (
        authority_sha
        != _require_sha256(
            seal.get("authority_sha256"),
            field="apply_staging_authority_seal.authority_sha256",
        )
        or authority_size != seal["authority_size_bytes"]
    ):
        raise SomphOfflinePackageError(
            "apply staging authority detached binding mismatch"
        )
    if (
        set(authority_payload) != _APPLY_STAGING_AUTHORITY_KEYS
        or authority_payload.get("schema") != expected_authority_schema
        or authority_payload.get("status") != expected_status
        or authority_payload.get("profile") != expected_profile
        or authority_payload.get("stage") not in {"stage2b", "stage2c"}
        or authority_payload.get("registration_state")
        not in bundle.REGISTRATION_STATES
        or (
            authority_payload.get("stage") == "stage2b"
            and authority_payload.get("registration_state") != "before"
        )
        or not isinstance(authority_payload.get("receiver"), str)
        or not authority_payload["receiver"]
        or not isinstance(authority_payload.get("seed"), int)
        or isinstance(authority_payload["seed"], bool)
        or authority_payload["seed"] < 1
        or authority_payload.get("k_shot") not in FORMAL_K_VALUES
        or authority_payload.get("formal_launch_authority") is not False
        or not isinstance(
            authority_payload.get("external_authority_lock_verified"), bool
        )
    ):
        raise SomphOfflinePackageError(
            "apply staging authority exact schema drift"
        )
    for field in (
        "row_manifest_sha256",
        "registered_classes_sha256",
        "enrollment_package_root_sha256",
        "enrollment_package_seal_sha256",
        "phase1_checkpoint_sha256",
        "feature_runtime_sha256",
        "method_lock_sha256",
        "apply_overlay_provenance_sha256",
        "cache_set_manifest_sha256",
        "cache_physical_sample_scenario_assignment_sha256",
        "lineage_receipt_sha256",
        "lineage_seal_sha256",
    ):
        _require_sha256(
            authority_payload.get(field),
            field=f"apply_staging_authority.{field}",
        )
    _require_scenario_sha_map(
        authority_payload.get(
            "cache_physical_sample_ids_sha256_by_scenario"
        ),
        field=(
            "apply_staging_authority."
            "cache_physical_sample_ids_sha256_by_scenario"
        ),
    )
    if (
        not isinstance(authority_payload.get("row_handle"), str)
        or re.fullmatch(
            r"row_[0-9a-f]{64}", authority_payload["row_handle"]
        )
        is None
    ):
        raise SomphOfflinePackageError(
            "apply staging authority row handle drift"
        )
    registry = authority_payload.get("registered_classes")
    try:
        bundle._validate_registry(registry, len(registry))
    except (TypeError, bundle.PredictorPackageError) as exc:
        raise SomphOfflinePackageError(
            "apply staging authority registry drift"
        ) from exc
    if _canonical_sha256(registry) != authority_payload[
        "registered_classes_sha256"
    ]:
        raise SomphOfflinePackageError(
            "apply staging authority registry digest mismatch"
        )
    for field in ("authority_commit_sha256", "authority_attestation_sha256"):
        value = authority_payload.get(field)
        if value is not None:
            _require_sha256(
                value, field=f"apply_staging_authority.{field}"
            )
    if require_external_authority:
        if (
            authority_payload["external_authority_lock_verified"] is not True
            or authority_payload["authority_commit_sha256"] is None
            or authority_payload["authority_attestation_sha256"] is None
        ):
            raise SomphOfflinePackageError(
                "formal apply finalization requires external lineage authority"
            )
    elif (
        authority_payload["external_authority_lock_verified"] is False
        and (
            authority_payload["authority_commit_sha256"] is not None
            or authority_payload["authority_attestation_sha256"] is not None
        )
    ):
        raise SomphOfflinePackageError(
            "diagnostic apply authority cannot carry authority artifacts"
        )
    return authority_payload


def _prevalidate_external_head(
    *,
    head_capsule_path: str | Path,
    expected_head_capsule_sha256: str,
    expected_head_enrollment_binding_sha256: str,
    staging_authority: Mapping[str, Any],
) -> None:
    head_path = Path(head_capsule_path)
    expected_head = _require_sha256(
        expected_head_capsule_sha256,
        field="expected_head_capsule_sha256",
    )
    expected_binding = _require_sha256(
        expected_head_enrollment_binding_sha256,
        field="expected_head_enrollment_binding_sha256",
    )
    try:
        size = head_path.stat().st_size
        descriptor = {
            "relative_path": head_path.name,
            "sha256": expected_head,
            "size_bytes": size,
            "kind": "head_capsule",
            "schema": bundle.SOMPH_HEAD_CAPSULE_SCHEMA,
            "scenario": None,
            "npz_members": list(bundle.HEAD_CAPSULE_NPZ_MEMBERS),
        }
        arrays, binding, binding_sha = bundle._load_head_capsule_member(
            head_path.parent.resolve(), descriptor
        )
        method_path = (
            Path(staging_authority["apply_staging_root"])
            / bundle.METHOD_LOCK_RELATIVE_PATH
        )
        method_lock, method_sha, _method_size = _read_json_same_fd(
            method_path,
            context="SOMP-H apply staging method lock",
        )
        if (
            method_sha != staging_authority["method_lock_sha256"]
            or sha256_bytes(canonical_json_bytes(method_lock)) != method_sha
        ):
            raise SomphOfflinePackageError(
                "apply staging method lock byte binding drift"
            )
        bundle.validate_somph_head_capsule(
            arrays,
            method_lock=method_lock,
            expected_enrollment_binding_sha256=expected_binding,
        )
    except SomphOfflinePackageError:
        raise
    except Exception as exc:
        raise SomphOfflinePackageError(
            "external head capsule failed complete prevalidation"
        ) from exc
    expected_binding_fields = {
        "stage": staging_authority["stage"],
        "registration_state": staging_authority["registration_state"],
        "receiver": staging_authority["receiver"],
        "seed": staging_authority["seed"],
        "k_shot": staging_authority["k_shot"],
        "registered_class_handles": [
            value["class_handle"]
            for value in staging_authority["registered_classes"]
        ],
        "enrollment_package_root_sha256": staging_authority[
            "enrollment_package_root_sha256"
        ],
        "enrollment_package_seal_sha256": staging_authority[
            "enrollment_package_seal_sha256"
        ],
        "phase1_checkpoint_sha256": staging_authority[
            "phase1_checkpoint_sha256"
        ],
        "feature_runtime_sha256": staging_authority[
            "feature_runtime_sha256"
        ],
        "method_lock_sha256": staging_authority["method_lock_sha256"],
    }
    failed = [
        field
        for field, expected in expected_binding_fields.items()
        if binding.get(field) != expected
    ]
    if binding_sha != expected_binding or failed:
        raise SomphOfflinePackageError(
            "head capsule does not match sealed apply staging authority"
        )


def load_verified_lineage_context_from_receipt_seal(
    *,
    cache_set_manifest_path: str | Path,
    lineage_receipt_path: str | Path,
    lineage_seal_path: str | Path,
    expected_lineage_receipt_sha256: str,
    expected_lineage_seal_sha256: str,
) -> dict[str, Any]:
    """Diagnostic compatibility adapter for structural self-consistency only.

    The row-pair producer consumes the verified-context interface below rather
    than depending on the lineage writer's argument surface.  This loader never
    claims an external authority lock and cannot authorize a formal launch.
    """
    expected_receipt = _require_sha256(
        expected_lineage_receipt_sha256,
        field="expected_lineage_receipt_sha256",
    )
    expected_seal = _require_sha256(
        expected_lineage_seal_sha256,
        field="expected_lineage_seal_sha256",
    )
    try:
        receipt, seal = lineage.verify_somph_leo_weak_lineage_seal(
            lineage_receipt_path,
            lineage_seal_path,
            expected_detached_seal_sha256=expected_seal,
        )
    except Exception as exc:
        raise SomphOfflinePackageError(
            "lineage receipt/seal consumer verification failed"
        ) from exc
    _receipt_again, receipt_sha, receipt_size = _read_json_same_fd(
        lineage_receipt_path, context="SOMP-H lineage receipt"
    )
    _seal_again, seal_sha, _seal_size = _read_json_same_fd(
        lineage_seal_path, context="SOMP-H lineage detached seal"
    )
    if receipt_sha != expected_receipt or seal_sha != expected_seal:
        raise SomphOfflinePackageError("external lineage receipt/seal SHA mismatch")
    if set(receipt) != _LINEAGE_RECEIPT_KEYS:
        raise SomphOfflinePackageError("lineage receipt exact schema drift")
    if set(seal) != _LINEAGE_SEAL_KEYS:
        raise SomphOfflinePackageError("lineage seal exact schema drift")
    if (
        receipt.get("schema") != lineage.LINEAGE_RECEIPT_SCHEMA
        or receipt.get("status") != "BYTE_GROUNDED_SELF_CONSISTENCY_PASS"
        or receipt.get("cache_scope") != "stage2_registered"
        or receipt.get("scenario_order")
        != list(bundle.FORMAL_LEO_WEAK_SCENARIOS)
        or receipt.get("same_fd_nofollow_read") is not True
        or receipt.get("cross_scenario_physical_disjointness_audit") != "PASS"
        or receipt.get("single_observation_contract_audit") != "PASS"
        or receipt.get("sample_level_overlay_recompute") != "PASS"
        or receipt.get("manifest_hex_self_declaration_sufficient") is not False
        or receipt.get("external_authority_lock_verified") is not False
        or receipt.get("contains_build_spec_or_dataset_paths") is not False
        or receipt.get("formal_launch_authority") is not False
    ):
        raise SomphOfflinePackageError("lineage receipt protocol contract drift")
    if (
        seal.get("schema") != lineage.LINEAGE_SEAL_SCHEMA
        or seal.get("receipt_sha256") != receipt_sha
        or seal.get("receipt_size_bytes") != receipt_size
        or seal.get("lineage_root_sha256") != _canonical_sha256(receipt)
    ):
        raise SomphOfflinePackageError("lineage detached seal binding failed")

    cache_set, cache_set_sha, _cache_set_size = _read_json_same_fd(
        cache_set_manifest_path, context="SOMP-H target cache-set manifest"
    )
    if (
        cache_set_sha != receipt["cache_set_manifest_sha256"]
        or set(cache_set) != _CACHE_SET_KEYS
        or cache_set.get("cache_scope") != "stage2_registered"
        or cache_set.get("target_channel_scenarios")
        != list(bundle.FORMAL_LEO_WEAK_SCENARIOS)
        or set(cache_set.get("output_roles", []))
        != {"target_old", "target_new"}
        or cache_set.get("clean_sample_access") is not False
        or cache_set.get("clean_derived_signal_access") is not False
        or cache_set.get("build_spec_path_exposed_to_phase2") is not False
        or cache_set.get("phase2_physical_sample_observation_policy")
        != "single_leo_weak_observation_per_physical_sample"
        or cache_set.get("phase2_cross_scenario_physical_sample_reuse")
        is not False
        or cache_set.get("phase2_additional_leo_channel_state_generation")
        is not False
        or cache_set.get(
            "phase2_post_reception_equalization_augmentation_transform_allowed"
        )
        is not True
        or cache_set.get(
            "phase2_post_reception_view_from_fixed_received_iq_only"
        )
        is not True
        or cache_set.get(
            "phase2_post_reception_view_counts_as_additional_physical_sample"
        )
        is not False
        or cache_set.get("phase2_physical_sample_root_id_policy")
        != "immutable_preoverlay_lineage_token"
        or cache_set.get("phase2_query_post_reception_view_fit_access")
        is not False
        or cache_set.get("physical_sample_scenario_assignment_policy")
        != "disjoint_preoverlay_tx_day_stratified_v1"
    ):
        raise SomphOfflinePackageError("cache-set/lineage binding failed")
    receipt_physical_roots = _require_scenario_sha_map(
        receipt.get("physical_sample_ids_sha256_by_scenario"),
        field="lineage_receipt.physical_sample_ids_sha256_by_scenario",
    )
    cache_physical_roots = _require_scenario_sha_map(
        cache_set.get("physical_sample_ids_sha256_by_scenario"),
        field="cache_set.physical_sample_ids_sha256_by_scenario",
    )
    receipt_assignment_root = _require_sha256(
        receipt.get("physical_sample_scenario_assignment_sha256"),
        field="lineage_receipt.physical_sample_scenario_assignment_sha256",
    )
    cache_assignment_root = _require_sha256(
        cache_set.get("physical_sample_scenario_assignment_sha256"),
        field="cache_set.physical_sample_scenario_assignment_sha256",
    )
    if (
        receipt_physical_roots != cache_physical_roots
        or receipt_assignment_root != cache_assignment_root
    ):
        raise SomphOfflinePackageError(
            "cache-set/lineage physical scenario assignment binding failed"
        )
    scenario_receipts = receipt.get("scenario_receipts")
    if (
        not isinstance(scenario_receipts, dict)
        or tuple(scenario_receipts) != bundle.FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise SomphOfflinePackageError("lineage scenario receipt registry drift")
    for scenario in bundle.FORMAL_LEO_WEAK_SCENARIOS:
        item = scenario_receipts[scenario]
        if (
            not isinstance(item, dict)
            or set(item) != _SCENARIO_RECEIPT_KEYS
            or item["cache_sha256"]
            != cache_set["cache_sha256_by_scenario"][scenario]
            or item["physical_sample_ids_sha256"]
            != receipt_physical_roots[scenario]
            or item["zip_member_crc_and_bounds_check"] != "PASS"
            or item["sample_level_overlay_recompute"] != "PASS"
        ):
            raise SomphOfflinePackageError(
                f"lineage scenario receipt drift: {scenario}"
            )
    return {
        "schema": VERIFIED_LINEAGE_CONTEXT_SCHEMA,
        "status": "VERIFIED_LINEAGE_CONTEXT",
        "receipt": receipt,
        "lineage_seal": seal,
        "cache_set": cache_set,
        "cache_set_manifest_path": str(Path(cache_set_manifest_path).resolve()),
        "lineage_receipt_sha256": receipt_sha,
        "lineage_seal_sha256": seal_sha,
        "cache_set_manifest_sha256": cache_set_sha,
        "authority_lock": None,
        "authority_attestation": None,
        "authority_commit": None,
        "authority_commit_sha256": None,
        "authority_attestation_sha256": None,
        "external_authority_lock_verified": False,
        "formal_launch_authority": False,
    }


def load_verified_lineage_context_from_authority_commit(
    *,
    cache_set_manifest_path: str | Path,
    authority_bundle_root: str | Path,
    expected_authority_commit_sha256: str,
) -> dict[str, Any]:
    """Load formal offline lineage from one externally expected authority commit."""

    if authority.AUTHORITY_LOCK_SCHEMA.endswith(".v1"):
        raise SomphOfflinePackageError(
            "formal single-observation authority v2 is pending; "
            "v1 authority cannot bind per-scenario physical roots and assignment SHA"
        )
    expected_commit = _require_sha256(
        expected_authority_commit_sha256,
        field="expected_authority_commit_sha256",
    )
    root = Path(authority_bundle_root)
    try:
        lock, attestation, commit = (
            authority.verify_somph_lineage_authority_bundle(
                root,
                expected_commit_sha256=expected_commit,
            )
        )
    except Exception as exc:
        raise SomphOfflinePackageError(
            "authority commit consumer verification failed"
        ) from exc
    locked_cache = lock.get("cache_set_manifest")
    if (
        not isinstance(locked_cache, dict)
        or set(locked_cache) != {"path", "sha256", "size_bytes"}
    ):
        raise SomphOfflinePackageError(
            "authority lock cache-set descriptor schema drift"
        )
    caller_cache_path = Path(cache_set_manifest_path).resolve()
    locked_cache_path = Path(str(locked_cache.get("path", ""))).resolve()
    if caller_cache_path != locked_cache_path:
        raise SomphOfflinePackageError(
            "caller cache-set path differs from authority-locked path"
        )
    structural_receipt_path = root / authority.STRUCTURAL_RECEIPT_NAME
    structural_seal_path = root / authority.STRUCTURAL_SEAL_NAME
    member_map = {
        item.get("name"): item
        for item in commit.get("members", [])
        if isinstance(item, dict)
    }
    receipt_member = member_map.get(authority.STRUCTURAL_RECEIPT_NAME)
    seal_member = member_map.get(authority.STRUCTURAL_SEAL_NAME)
    attestation_member = member_map.get(authority.AUTHORITY_ATTESTATION_NAME)
    if not all(
        isinstance(value, dict)
        for value in (receipt_member, seal_member, attestation_member)
    ):
        raise SomphOfflinePackageError(
            "authority commit structural member registry drift"
        )
    diagnostic = load_verified_lineage_context_from_receipt_seal(
        cache_set_manifest_path=caller_cache_path,
        lineage_receipt_path=structural_receipt_path,
        lineage_seal_path=structural_seal_path,
        expected_lineage_receipt_sha256=str(receipt_member["sha256"]),
        expected_lineage_seal_sha256=str(seal_member["sha256"]),
    )
    if (
        diagnostic["cache_set_manifest_sha256"]
        != _require_sha256(
            locked_cache.get("sha256"),
            field="authority_lock.cache_set_manifest.sha256",
        )
        or int(locked_cache.get("size_bytes", -1))
        != caller_cache_path.stat().st_size
        or attestation.get("structural_receipt_sha256")
        != diagnostic["lineage_receipt_sha256"]
        or attestation.get("structural_detached_seal_sha256")
        != diagnostic["lineage_seal_sha256"]
        or attestation.get("authority_lock_sha256")
        != commit.get("authority_lock_sha256")
        or attestation.get("external_authority_lock_verified") is not True
        or attestation.get("formal_launch_authority") is not False
        or commit.get("external_authority_lock_verified") is not True
        or commit.get("formal_launch_authority") is not False
    ):
        raise SomphOfflinePackageError(
            "authority lock/attestation/structural lineage binding failed"
        )
    context = dict(diagnostic)
    context.update(
        {
            "authority_lock": lock,
            "authority_attestation": attestation,
            "authority_commit": commit,
            "authority_commit_sha256": expected_commit,
            "authority_attestation_sha256": _require_sha256(
                attestation_member.get("sha256"),
                field="authority_attestation_sha256",
            ),
            "external_authority_lock_verified": True,
            "formal_launch_authority": False,
        }
    )
    return context


def _resolve_cache(cache_set_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise SomphOfflinePackageError("cache-set contains an invalid cache path")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else cache_set_path.parent / candidate


def _load_cache(
    path: Path,
    *,
    scenario: str,
    expected_sha256: str,
    scenario_receipt: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    try:
        with lineage._open_external_same_fd(path) as handle:
            digest, _size = _hash_handle(handle)
            if digest != expected_sha256:
                raise SomphOfflinePackageError(
                    f"target cache SHA mismatch: {scenario}"
                )
            members = _zip_members_from_handle(handle, context=f"target cache:{scenario}")
            if members not in {
                lineage._CACHE_MEMBERS,
                lineage._CACHE_MEMBERS + lineage._OPTIONAL_SPLIT_MEMBERS,
            }:
                raise SomphOfflinePackageError(
                    f"target cache member allowlist drift: {scenario}"
                )
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in members
                    if name != "manifest_json"
                }
    except SomphOfflinePackageError:
        raise
    except Exception as exc:
        raise SomphOfflinePackageError(
            f"target cache could not be loaded safely: {scenario}"
        ) from exc
    ids = np.asarray(arrays["sample_ids"]).astype(str).tolist()
    if (
        _canonical_sha256(ids) != scenario_receipt["physical_sample_ids_sha256"]
        and lineage.ids_sha256(ids)
        != scenario_receipt["physical_sample_ids_sha256"]
    ):
        raise SomphOfflinePackageError(
            f"target cache physical sample root drift: {scenario}"
        )
    iq = np.asarray(arrays["leo_weak_iq"])
    stored_hashes = np.asarray(arrays["post_channel_iq_sha256"]).astype(str).tolist()
    actual_hashes = [bundle.iq_row_sha256(row) for row in iq]
    if (
        iq.dtype != np.float32
        or iq.ndim != 3
        or iq.shape[1] != 2
        or stored_hashes != actual_hashes
        or not bool(np.all(np.asarray(arrays["overlay_applied"])))
        or not bool(
            np.all(np.asarray(arrays["sat_scenarios"]).astype(str) == scenario)
        )
    ):
        raise SomphOfflinePackageError(
            f"target cache row-level LEO_weak contract drift: {scenario}"
        )
    return arrays


def _load_scenario_caches(
    cache_set_manifest_path: str | Path,
    *,
    cache_set: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    manifest_path = Path(cache_set_manifest_path)
    result: dict[str, dict[str, np.ndarray]] = {}
    expected_roots = _require_scenario_sha_map(
        cache_set.get("physical_sample_ids_sha256_by_scenario"),
        field="cache_set.physical_sample_ids_sha256_by_scenario",
    )
    physical_ids_by_scenario: dict[str, list[str]] = {}
    for scenario in bundle.FORMAL_LEO_WEAK_SCENARIOS:
        arrays = _load_cache(
            _resolve_cache(
                manifest_path, cache_set["cache_npz_by_scenario"][scenario]
            ),
            scenario=scenario,
            expected_sha256=receipt["scenario_receipts"][scenario][
                "cache_sha256"
            ],
            scenario_receipt=receipt["scenario_receipts"][scenario],
        )
        physical_ids = np.asarray(arrays["sample_ids"]).astype(str).tolist()
        if len(set(physical_ids)) != len(physical_ids):
            raise SomphOfflinePackageError(
                f"duplicate physical sample ID within scenario: {scenario}"
            )
        if lineage.ids_sha256(physical_ids) != expected_roots[scenario]:
            raise SomphOfflinePackageError(
                f"cache-set physical root drift: {scenario}"
            )
        physical_ids_by_scenario[scenario] = physical_ids
        result[scenario] = arrays
    observed: set[str] = set()
    for scenario in bundle.FORMAL_LEO_WEAK_SCENARIOS:
        overlap = observed.intersection(physical_ids_by_scenario[scenario])
        if overlap:
            raise SomphOfflinePackageError(
                "physical samples are reused across LEO_weak scenarios"
            )
        observed.update(physical_ids_by_scenario[scenario])
    assignment_root = _canonical_sha256(physical_ids_by_scenario)
    if assignment_root != _require_sha256(
        cache_set.get("physical_sample_scenario_assignment_sha256"),
        field="cache_set.physical_sample_scenario_assignment_sha256",
    ):
        raise SomphOfflinePackageError(
            "cache-set physical scenario assignment root mismatch"
        )
    return result


def _opaque(secret: bytes, prefix: str, *parts: Any) -> str:
    message = json.dumps(
        [str(value) for value in parts],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}_" + hmac.new(secret, message, hashlib.sha256).hexdigest()


def _selection_order(
    sample_ids: np.ndarray,
    indices: np.ndarray,
    *,
    receiver: str,
    seed: int,
    role: str,
    tx_label: str,
) -> list[int]:
    return sorted(
        (int(value) for value in indices.tolist()),
        key=lambda index: (
            hashlib.sha256(
                (
                    f"somph-offline-split-v1|{receiver}|{seed}|{role}|"
                    f"{tx_label}|{sample_ids[index]}"
                ).encode("utf-8")
            ).hexdigest(),
            str(sample_ids[index]),
        ),
    )


def _select_class_rows(
    reference: Mapping[str, np.ndarray],
    *,
    receiver: str,
    seed: int,
    role: str,
    tx_label: str,
    support_k: int,
    query_per_tx: int,
) -> tuple[list[int], list[int]]:
    if (
        not isinstance(support_k, int)
        or isinstance(support_k, bool)
        or support_k < 1
        or support_k > SUPPORT_POOL_MAX_K
    ):
        raise SomphOfflinePackageError("support_k is outside 1..20")
    roles = np.asarray(reference["dataset_role"]).astype(str)
    tx_ids = np.asarray(reference["tx_ids"]).astype(str)
    rx_ids = np.asarray(reference["rx_ids"]).astype(str)
    sample_ids = np.asarray(reference["sample_ids"]).astype(str)
    candidates = np.flatnonzero(
        (roles == role) & (tx_ids == tx_label) & (rx_ids == receiver)
    )
    if "split_partition" in reference:
        partitions = np.asarray(reference["split_partition"]).astype(str)
        ranks = np.asarray(reference["split_rank"])
        if ranks.dtype.kind not in {"i", "u"}:
            raise SomphOfflinePackageError("offline split rank dtype drift")

        def ranked(partition: str) -> list[int]:
            selected = [int(value) for value in candidates if partitions[value] == partition]
            selected.sort(key=lambda index: (int(ranks[index]), sample_ids[index]))
            if len({int(ranks[index]) for index in selected}) != len(selected):
                raise SomphOfflinePackageError(
                    f"duplicate offline split rank: {role}/{tx_label}/{partition}"
                )
            return selected

        support = ranked("support_pool")[:support_k]
        query = ranked("query")
    else:
        ordered = _selection_order(
            sample_ids,
            candidates,
            receiver=receiver,
            seed=seed,
            role=role,
            tx_label=tx_label,
        )
        support = ordered[:support_k]
        query = ordered[SUPPORT_POOL_MAX_K : SUPPORT_POOL_MAX_K + query_per_tx]
    if len(support) < support_k or len(query) < query_per_tx:
        raise SomphOfflinePackageError(
            f"insufficient target rows for {role}/{tx_label}: "
            f"support={len(support)},query={len(query)}"
        )
    support = support[:support_k]
    query = query[:query_per_tx]
    if set(support) & set(query):
        raise SomphOfflinePackageError(
            f"support/query overlap for {role}/{tx_label}"
        )
    return support, query


def _write_npz_new(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (
        f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite NPZ output: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _registry(secret: bytes, labels: list[tuple[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "class_index": index,
            "class_handle": _opaque(
                secret, "cls", "somph-class-v1", role, tx_label
            ),
        }
        for index, (role, tx_label) in enumerate(labels)
    ]


def _write_profile_payloads(
    root: Path,
    *,
    profile: str,
    registration_state: str,
    receiver: str,
    seed: int,
    labels: list[tuple[str, str]],
    selected_by_scenario: Mapping[
        str,
        Mapping[tuple[str, str], tuple[list[int], list[int]]],
    ],
    arrays_by_scenario: Mapping[str, Mapping[str, np.ndarray]],
    secret: bytes,
    lineage_receipt_sha256: str,
    cache_sha256_by_scenario: Mapping[str, str],
    support_pool_k: int = SUPPORT_POOL_MAX_K,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (
        not isinstance(support_pool_k, int)
        or isinstance(support_pool_k, bool)
        or support_pool_k < 1
        or support_pool_k > SUPPORT_POOL_MAX_K
    ):
        raise SomphOfflinePackageError("support_pool_k is outside 1..20")
    support_class_indices = np.repeat(
        np.arange(len(labels), dtype=np.int64), support_pool_k
    )
    support_ranks = np.tile(
        np.arange(support_pool_k, dtype=np.int64), len(labels)
    )
    provenance_samples: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    registry = _registry(secret, labels)
    for scenario in bundle.FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario[scenario]
        scenario_selected = selected_by_scenario[scenario]
        sample_ids = np.asarray(arrays["sample_ids"]).astype(str)
        support_indices = [
            index
            for label in labels
            for index in scenario_selected[label][0]
        ]
        query_indices = [
            index
            for label in labels
            for index in scenario_selected[label][1]
        ]
        query_indices.sort(
            key=lambda index: (
                hmac.new(
                    secret,
                    (
                        f"somph-query-global-order-v2|{scenario}|"
                        f"{receiver}|{seed}|{sample_ids[index]}"
                    ).encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest(),
                sample_ids[index],
            )
        )
        support_tokens = np.asarray(
            [
                _opaque(
                    secret,
                    "sid",
                    "somph-support-v2",
                    scenario,
                    receiver,
                    seed,
                    sample_ids[index],
                )
                for index in support_indices
            ]
        )
        query_tokens = np.asarray(
            [
                _opaque(
                    secret,
                    "qid",
                    "somph-query-v2",
                    scenario,
                    receiver,
                    seed,
                    sample_ids[index],
                )
                for index in query_indices
            ]
        )
        iq = np.asarray(arrays["leo_weak_iq"])
        overlays = np.asarray(arrays["overlay_ids"]).astype(str)
        seeds = np.asarray(arrays["satellite_seeds"])
        iq_hashes = np.asarray(arrays["post_channel_iq_sha256"]).astype(str)
        if profile == bundle.ENROLLMENT_ONLY:
            selected = support_indices
            tokens = support_tokens
            embedded = {
                "schema": bundle.SOMPH_SUPPORT_IQ_SCHEMA,
                "scenario": scenario,
                "registration_state": registration_state,
                "registered_class_count": len(labels),
                "support_pool_max_k": support_pool_k,
                "token_scheme": "hmac_sha256_opaque_v1",
            }
            overlay_tokens = np.asarray(
                [
                    _opaque(
                        secret,
                        "oid",
                        "somph-support-overlay-v2",
                        scenario,
                        overlays[index],
                    )
                    for index in selected
                ]
            )
            _write_npz_new(
                root / f"support_{scenario}.npz",
                support_leo_weak_iq=iq[selected],
                support_class_indices=support_class_indices,
                support_rank_within_class=support_ranks,
                support_tokens=tokens,
                support_overlay_tokens=overlay_tokens,
                support_satellite_seeds=seeds[selected].astype(np.int64),
                support_post_channel_iq_sha256=iq_hashes[selected],
                manifest_json=np.asarray(
                    json.dumps(embedded, sort_keys=True, separators=(",", ":"))
                ),
            )
        else:
            selected = query_indices
            tokens = query_tokens
            embedded = {
                "schema": bundle.SOMPH_QUERY_IQ_SCHEMA,
                "scenario": scenario,
                "registration_state": registration_state,
                "token_scheme": "hmac_sha256_opaque_v1",
            }
            overlay_tokens = np.asarray(
                [
                    _opaque(
                        secret,
                        "oid",
                        "somph-query-overlay-v2",
                        scenario,
                        overlays[index],
                    )
                    for index in selected
                ]
            )
            _write_npz_new(
                root / f"query_{scenario}.npz",
                query_leo_weak_iq=iq[selected],
                query_tokens=tokens,
                query_overlay_tokens=overlay_tokens,
                query_satellite_seeds=seeds[selected].astype(np.int64),
                query_post_channel_iq_sha256=iq_hashes[selected],
                manifest_json=np.asarray(
                    json.dumps(embedded, sort_keys=True, separators=(",", ":"))
                ),
            )
        for token, overlay_token, index in zip(tokens, overlay_tokens, selected):
            provenance_samples.append(
                {
                    "sample_token": str(token),
                    "scenario": scenario,
                    "overlay_token": str(overlay_token),
                    "satellite_seed": int(seeds[index]),
                    "post_channel_iq_sha256": str(iq_hashes[index]),
                    "source_leo_cache_sha256": cache_sha256_by_scenario[scenario],
                    "source_leo_provenance_sha256": lineage_receipt_sha256,
                }
            )
        if profile == bundle.APPLY_ONLY:
            tx_ids = np.asarray(arrays["tx_ids"]).astype(str)
            roles = np.asarray(arrays["dataset_role"]).astype(str)
            rx_ids = np.asarray(arrays["rx_ids"]).astype(str)
            day_ids = np.asarray(arrays["day_ids"]).astype(str)
            sig_ids = np.asarray(arrays["sig_ids"]).astype(str)
            label_to_class = {
                label: index for index, label in enumerate(labels)
            }
            for token, index in zip(query_tokens.tolist(), query_indices):
                key = (str(roles[index]), str(tx_ids[index]))
                class_index = label_to_class[key]
                truth_rows.append(
                    {
                        "query_token": token,
                        "true_class_index": class_index,
                        "true_class_handle": registry[class_index][
                            "class_handle"
                        ],
                        "transmitter_label": str(tx_ids[index]),
                        "evaluation_role": str(roles[index]),
                        "receiver_label": str(rx_ids[index]),
                        "day_label": str(day_ids[index]),
                        "signal_label": str(sig_ids[index]),
                        "physical_sample_id": str(sample_ids[index]),
                    }
                )
    provenance = {
        "schema": bundle.SOMPH_OVERLAY_PROVENANCE_SCHEMA,
        "profile": profile,
        "receiver": receiver,
        "seed": seed,
        "samples": provenance_samples,
    }
    _write_json_new(root / "overlay_provenance.json", provenance, readonly=False)
    return registry, truth_rows


def _prepare_root(
    root: Path,
    *,
    sealed_feature_runtime_path: str | Path,
    method_lock_path: str | Path,
) -> tuple[str, str]:
    root.mkdir(parents=True, exist_ok=False)
    runtime_sha = _copy_regular_new(
        sealed_feature_runtime_path,
        root / bundle.FEATURE_RUNTIME_RELATIVE_PATH,
    )
    method_sha = _copy_regular_new(method_lock_path, root / "method_lock.json")
    return runtime_sha, method_sha


def _physical_root(rows: list[tuple[str, str]]) -> str:
    ordered = [physical_id for _token, physical_id in sorted(rows)]
    return _canonical_sha256(ordered)


def _build_somph_offline_row_pair_from_context(
    *,
    cache_set_manifest_path: str | Path,
    verified_lineage_context: Mapping[str, Any],
    formal_staging_authority: bool,
    phase1_checkpoint_path: str | Path,
    sealed_feature_runtime_path: str | Path,
    method_lock_path: str | Path,
    output_root: str | Path,
    receiver: str,
    seed: int,
    k_shot: int,
    new_class_count: int,
    query_per_tx: int,
    token_secret: bytes | None = None,
) -> dict[str, Any]:
    """Build one row pair from an already resolved lineage context."""

    if receiver not in FORMAL_RECEIVERS:
        raise SomphOfflinePackageError("receiver is outside the formal target set")
    if seed not in {DEVELOPMENT_SEED, *CONFIRMATION_SEEDS}:
        raise SomphOfflinePackageError("seed is outside development/confirmation lock")
    if k_shot not in FORMAL_K_VALUES:
        raise SomphOfflinePackageError("K is outside the formal nested K20 anchors")
    if new_class_count not in FORMAL_NEW_CLASS_COUNTS:
        raise SomphOfflinePackageError("new_class_count must be 5, 10, or 20")
    if not isinstance(query_per_tx, int) or isinstance(query_per_tx, bool) or query_per_tx < 1:
        raise SomphOfflinePackageError("query_per_tx must be positive")
    if token_secret is None:
        token_secret = os.urandom(32)
    if not isinstance(token_secret, bytes) or len(token_secret) < 32:
        raise SomphOfflinePackageError("token_secret must contain at least 256 bits")

    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output}")
    context = dict(verified_lineage_context)
    if (
        set(context) != _VERIFIED_CONTEXT_KEYS
        or context.get("schema") != VERIFIED_LINEAGE_CONTEXT_SCHEMA
        or context.get("status") != "VERIFIED_LINEAGE_CONTEXT"
        or not isinstance(context.get("external_authority_lock_verified"), bool)
        or context.get("formal_launch_authority") is not False
    ):
        raise SomphOfflinePackageError(
            "verified lineage context exact schema drift"
        )
    if bool(context["external_authority_lock_verified"]) != bool(
        formal_staging_authority
    ):
        raise SomphOfflinePackageError(
            "lineage context does not match formal/diagnostic build profile"
        )
    receipt = context["receipt"]
    lineage_seal = context["lineage_seal"]
    cache_set = context["cache_set"]
    if (
        not isinstance(receipt, dict)
        or not isinstance(lineage_seal, dict)
        or not isinstance(cache_set, dict)
    ):
        raise SomphOfflinePackageError("verified lineage context payload drift")
    receipt_sha = _require_sha256(
        context["lineage_receipt_sha256"],
        field="verified_lineage_context.lineage_receipt_sha256",
    )
    lineage_seal_sha = _require_sha256(
        context["lineage_seal_sha256"],
        field="verified_lineage_context.lineage_seal_sha256",
    )
    actual_cache_set_sha = sha256_file(cache_set_manifest_path)
    context_cache_path = Path(
        str(context.get("cache_set_manifest_path", ""))
    ).resolve()
    if (
        context_cache_path != Path(cache_set_manifest_path).resolve()
        or
        actual_cache_set_sha
        != _require_sha256(
            context["cache_set_manifest_sha256"],
            field="verified_lineage_context.cache_set_manifest_sha256",
        )
        or actual_cache_set_sha != receipt.get("cache_set_manifest_sha256")
    ):
        raise SomphOfflinePackageError(
            "verified lineage context/cache-set byte binding failed"
        )
    if context["external_authority_lock_verified"]:
        authority_lock = context.get("authority_lock")
        authority_attestation = context.get("authority_attestation")
        authority_commit = context.get("authority_commit")
        if (
            not isinstance(authority_lock, dict)
            or not isinstance(authority_attestation, dict)
            or not isinstance(authority_commit, dict)
            or authority_lock.get("receiver") != receiver
            or authority_lock.get("seed") != seed
            or authority_lock.get("cache_scope") != "stage2_registered"
            or authority_lock.get("old_tx_ids") != list(FORMAL_OLD_TX_LABELS)
            or authority_lock.get("new_tx_ids")
            != list(FORMAL_NEW20_TX_LABELS)
            or authority_attestation.get("receiver") != receiver
            or authority_attestation.get("seed") != seed
            or authority_attestation.get("cache_scope") != "stage2_registered"
            or authority_attestation.get("external_authority_lock_verified")
            is not True
            or authority_attestation.get("formal_launch_authority") is not False
            or authority_commit.get("external_authority_lock_verified") is not True
            or authority_commit.get("formal_launch_authority") is not False
        ):
            raise SomphOfflinePackageError(
                "authority lineage context does not match the requested formal row"
            )
        locked_cache = authority_lock.get("cache_set_manifest")
        if (
            not isinstance(locked_cache, dict)
            or Path(str(locked_cache.get("path", ""))).resolve()
            != Path(cache_set_manifest_path).resolve()
            or locked_cache.get("sha256") != actual_cache_set_sha
            or authority_attestation.get("structural_receipt_sha256")
            != receipt_sha
            or authority_attestation.get("structural_detached_seal_sha256")
            != lineage_seal_sha
        ):
            raise SomphOfflinePackageError(
                "authority lineage context cache-set/structural binding drift"
            )
        _require_sha256(
            context.get("authority_commit_sha256"),
            field="verified_lineage_context.authority_commit_sha256",
        )
        _require_sha256(
            context.get("authority_attestation_sha256"),
            field="verified_lineage_context.authority_attestation_sha256",
        )
    elif any(
        context.get(field) is not None
        for field in (
            "authority_lock",
            "authority_attestation",
            "authority_commit",
            "authority_commit_sha256",
            "authority_attestation_sha256",
        )
    ):
        raise SomphOfflinePackageError(
            "diagnostic lineage context cannot carry authority artifacts"
        )
    arrays_by_scenario = _load_scenario_caches(
        cache_set_manifest_path,
        cache_set=cache_set,
        receipt=receipt,
    )
    if context["external_authority_lock_verified"]:
        authority_lock = context["authority_lock"]
        expected_tx_by_role = {
            "target_old": set(authority_lock["old_tx_ids"]),
            "target_new": set(authority_lock["new_tx_ids"]),
        }
        for scenario in bundle.FORMAL_LEO_WEAK_SCENARIOS:
            arrays = arrays_by_scenario[scenario]
            roles = np.asarray(arrays["dataset_role"]).astype(str)
            tx_ids = np.asarray(arrays["tx_ids"]).astype(str)
            if set(roles.tolist()) != set(expected_tx_by_role):
                raise SomphOfflinePackageError(
                    f"authority-locked cache role coverage drift: {scenario}"
                )
            for role, expected_tx in expected_tx_by_role.items():
                if set(tx_ids[roles == role].tolist()) != expected_tx:
                    raise SomphOfflinePackageError(
                        "authority-locked cache TX coverage drift: "
                        f"{scenario}/{role}"
                    )
    old_labels = [("target_old", value) for value in FORMAL_OLD_TX_LABELS]
    new_labels = [
        ("target_new", value)
        for value in FORMAL_NEW20_TX_LABELS[:new_class_count]
    ]
    selected_by_scenario: dict[
        str,
        dict[tuple[str, str], tuple[list[int], list[int]]],
    ] = {}
    selected_physical_ids_by_scenario: dict[str, set[str]] = {}
    support_physical_roots_by_scenario: dict[str, str] = {}
    query_physical_roots_by_scenario: dict[str, str] = {}
    for scenario in bundle.FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario[scenario]
        scenario_selected: dict[
            tuple[str, str], tuple[list[int], list[int]]
        ] = {}
        for role, label in old_labels + new_labels:
            scenario_selected[(role, label)] = _select_class_rows(
                arrays,
                receiver=receiver,
                seed=seed,
                role=role,
                tx_label=label,
                support_k=k_shot,
                query_per_tx=query_per_tx,
            )
        selected_by_scenario[scenario] = scenario_selected
        sample_ids = np.asarray(arrays["sample_ids"]).astype(str)
        support_ids = [
            str(sample_ids[index])
            for label in old_labels + new_labels
            for index in scenario_selected[label][0]
        ]
        query_ids = [
            str(sample_ids[index])
            for label in old_labels + new_labels
            for index in scenario_selected[label][1]
        ]
        if set(support_ids).intersection(query_ids):
            raise SomphOfflinePackageError(
                f"same-scenario support/query physical overlap: {scenario}"
            )
        selected_physical_ids_by_scenario[scenario] = set(
            support_ids + query_ids
        )
        support_physical_roots_by_scenario[scenario] = _canonical_sha256(
            sorted(support_ids)
        )
        query_physical_roots_by_scenario[scenario] = _canonical_sha256(
            sorted(query_ids)
        )
    observed_selected: set[str] = set()
    for scenario in bundle.FORMAL_LEO_WEAK_SCENARIOS:
        if observed_selected.intersection(
            selected_physical_ids_by_scenario[scenario]
        ):
            raise SomphOfflinePackageError(
                "selected physical samples overlap across scenarios"
            )
        observed_selected.update(selected_physical_ids_by_scenario[scenario])

    output.mkdir(parents=True, exist_ok=False)
    predictor_root = output / "predictor"
    scorer_root = output / "scorer"
    seals_root = output / "seals"
    predictor_root.mkdir()
    scorer_root.mkdir()
    seals_root.mkdir()
    method_sha = sha256_file(method_lock_path)
    phase1_checkpoint = Path(phase1_checkpoint_path).resolve()
    sealed_feature_runtime = Path(sealed_feature_runtime_path).resolve()
    phase1_checkpoint_sha = sha256_file(phase1_checkpoint)
    feature_runtime_sha = sha256_file(sealed_feature_runtime)
    if phase1_checkpoint_sha != bundle.ADV3B02_PHASE1_CHECKPOINT_SHA256:
        raise SomphOfflinePackageError(
            "Phase1 checkpoint lineage SHA256 does not match ADV3B02"
        )
    if phase1_checkpoint == sealed_feature_runtime:
        raise SomphOfflinePackageError(
            "Phase1 checkpoint and sealed feature runtime paths must differ"
        )
    if feature_runtime_sha == phase1_checkpoint_sha:
        raise SomphOfflinePackageError(
            "Phase1 state-dict checkpoint bytes cannot be the sealed runtime"
        )
    row_handle = _opaque(
        token_secret,
        "row",
        "somph-row-v1",
        receiver,
        seed,
        k_shot,
        new_class_count,
    )
    row_manifest = {
        "schema": ROW_MANIFEST_SCHEMA,
        "method_lock_sha256": method_sha,
        "phase1_checkpoint_sha256": phase1_checkpoint_sha,
        "feature_runtime_sha256": feature_runtime_sha,
        "split_role": (
            "development"
            if (receiver, seed, k_shot) == ("20-1", DEVELOPMENT_SEED, 10)
            else "confirmation"
        ),
        "receiver": receiver,
        "seed": seed,
        "k_shot": k_shot,
        "new_class_count": new_class_count,
        "support_pool_max_k": k_shot,
        "scenarios": list(bundle.FORMAL_LEO_WEAK_SCENARIOS),
        "physical_sample_ids_sha256_by_scenario": dict(
            cache_set["physical_sample_ids_sha256_by_scenario"]
        ),
        "physical_sample_scenario_assignment_sha256": cache_set[
            "physical_sample_scenario_assignment_sha256"
        ],
        "selected_support_physical_ids_sha256_by_scenario": (
            support_physical_roots_by_scenario
        ),
        "selected_query_physical_ids_sha256_by_scenario": (
            query_physical_roots_by_scenario
        ),
        "same_scenario_support_query_disjointness_audit": "PASS",
        "cross_scenario_selected_physical_disjointness_audit": "PASS",
    }
    row_manifest_sha = _canonical_sha256(row_manifest)
    _write_json_new(scorer_root / "row_manifest.json", row_manifest)

    state_results: dict[str, Any] = {}
    truth_by_state: dict[str, list[dict[str, Any]]] = {}
    for state, labels in (("before", old_labels), ("after", old_labels + new_labels)):
        stage = "stage2b" if state == "before" else "stage2c"
        enrollment_root = predictor_root / state / "enrollment_only"
        apply_staging_root = predictor_root / state / "apply_only_staging"
        enrollment_runtime, enrollment_method = _prepare_root(
            enrollment_root,
            sealed_feature_runtime_path=sealed_feature_runtime,
            method_lock_path=method_lock_path,
        )
        apply_runtime, apply_method = _prepare_root(
            apply_staging_root,
            sealed_feature_runtime_path=sealed_feature_runtime,
            method_lock_path=method_lock_path,
        )
        if {
            enrollment_runtime,
            apply_runtime,
            feature_runtime_sha,
        } != {feature_runtime_sha} or {
            enrollment_method,
            apply_method,
            method_sha,
        } != {method_sha}:
            raise SomphOfflinePackageError("runtime/method copy digest drift")
        registry, _unused_truth = _write_profile_payloads(
            enrollment_root,
            profile=bundle.ENROLLMENT_ONLY,
            registration_state=state,
            receiver=receiver,
            seed=seed,
            labels=labels,
            selected_by_scenario=selected_by_scenario,
            arrays_by_scenario=arrays_by_scenario,
            secret=token_secret,
            lineage_receipt_sha256=receipt_sha,
            cache_sha256_by_scenario=cache_set["cache_sha256_by_scenario"],
            support_pool_k=k_shot,
        )
        apply_registry, truth_rows = _write_profile_payloads(
            apply_staging_root,
            profile=bundle.APPLY_ONLY,
            registration_state=state,
            receiver=receiver,
            seed=seed,
            labels=labels,
            selected_by_scenario=selected_by_scenario,
            arrays_by_scenario=arrays_by_scenario,
            secret=token_secret,
            lineage_receipt_sha256=receipt_sha,
            cache_sha256_by_scenario=cache_set["cache_sha256_by_scenario"],
            support_pool_k=k_shot,
        )
        if apply_registry != registry:
            raise SomphOfflinePackageError("enrollment/apply registry drift")
        enrollment_overlay_sha = sha256_file(
            enrollment_root / "overlay_provenance.json"
        )
        seal_path = seals_root / f"{state}_enrollment.seal.json"
        _manifest_path, _seal_path, manifest, _seal = (
            bundle.write_somph_predictor_bundle(
                enrollment_root,
                profile=bundle.ENROLLMENT_ONLY,
                stage=stage,
                registration_state=state,
                receiver=receiver,
                seed=seed,
                k_shot=k_shot,
                registered_classes=registry,
                expected_method_lock_sha256=method_sha,
                expected_overlay_provenance_sha256=enrollment_overlay_sha,
                detached_seal_path=seal_path,
                support_pool_max_k=k_shot,
            )
        )
        enrollment_seal_sha = sha256_file(seal_path)
        apply_overlay_sha = sha256_file(
            apply_staging_root / "overlay_provenance.json"
        )
        staging_authority_path = (
            seals_root / f"{state}_apply_staging_authority.json"
        )
        staging_authority_seal_path = (
            seals_root / f"{state}_apply_staging_authority.seal.json"
        )
        staging_authority_sha, staging_authority_seal_sha = (
            _write_apply_staging_authority(
                path=staging_authority_path,
                seal_path=staging_authority_seal_path,
                stage=stage,
                registration_state=state,
                receiver=receiver,
                seed=seed,
                k_shot=k_shot,
                row_handle=row_handle,
                row_manifest_sha256=row_manifest_sha,
                registered_classes=registry,
                enrollment_package_root_sha256=manifest[
                    "package_root_sha256"
                ],
                enrollment_package_seal_sha256=enrollment_seal_sha,
                phase1_checkpoint_sha256=phase1_checkpoint_sha,
                feature_runtime_sha256=feature_runtime_sha,
                method_lock_sha256=method_sha,
                apply_staging_root=apply_staging_root,
                apply_overlay_provenance_sha256=apply_overlay_sha,
                cache_set_manifest_sha256=context[
                    "cache_set_manifest_sha256"
                ],
                cache_physical_sample_ids_sha256_by_scenario=cache_set[
                    "physical_sample_ids_sha256_by_scenario"
                ],
                cache_physical_sample_scenario_assignment_sha256=cache_set[
                    "physical_sample_scenario_assignment_sha256"
                ],
                lineage_receipt_sha256=receipt_sha,
                lineage_seal_sha256=lineage_seal_sha,
                authority_commit_sha256=context[
                    "authority_commit_sha256"
                ],
                authority_attestation_sha256=context[
                    "authority_attestation_sha256"
                ],
                external_authority_lock_verified=context[
                    "external_authority_lock_verified"
                ],
                formal_profile=formal_staging_authority,
            )
        )
        state_results[state] = {
            "stage": stage,
            "phase1_checkpoint_sha256": phase1_checkpoint_sha,
            "feature_runtime_sha256": feature_runtime_sha,
            "registered_classes": registry,
            "enrollment_package_root": str(enrollment_root),
            "enrollment_package_root_sha256": manifest["package_root_sha256"],
            "enrollment_package_seal": str(seal_path),
            "enrollment_package_seal_sha256": enrollment_seal_sha,
            "apply_staging_root": str(apply_staging_root),
            "apply_overlay_provenance_sha256": apply_overlay_sha,
            "apply_staging_authority": str(staging_authority_path),
            "apply_staging_authority_sha256": staging_authority_sha,
            "apply_staging_authority_seal": str(
                staging_authority_seal_path
            ),
            "apply_staging_authority_seal_sha256": (
                staging_authority_seal_sha
            ),
        }
        truth_by_state[state] = truth_rows

    before_truth = truth_by_state["before"]
    after_truth = truth_by_state["after"]
    before_old = {
        row["query_token"]: row["physical_sample_id"] for row in before_truth
    }
    after_old = {
        row["query_token"]: row["physical_sample_id"]
        for row in after_truth
        if row["evaluation_role"] == "target_old"
    }
    if before_old != after_old:
        raise SomphOfflinePackageError("before/after old query physical mapping drift")
    if (
        len(before_old) != len(before_truth)
        or len({row["physical_sample_id"] for row in after_truth})
        != len(after_truth)
    ):
        raise SomphOfflinePackageError(
            "truth token or physical sample uniqueness drift"
        )
    old_support_rows = []
    for scenario in bundle.FORMAL_LEO_WEAK_SCENARIOS:
        sample_ids = np.asarray(
            arrays_by_scenario[scenario]["sample_ids"]
        ).astype(str)
        for label in old_labels:
            for index in selected_by_scenario[scenario][label][0]:
                token = _opaque(
                    token_secret,
                    "sid",
                    "somph-support-v2",
                    scenario,
                    receiver,
                    seed,
                    sample_ids[index],
                )
                old_support_rows.append((token, str(sample_ids[index])))
    old_support_root = _physical_root(old_support_rows)
    old_query_root = _physical_root(list(before_old.items()))

    truth_sidecar = {
        "schema": TRUTH_SIDECAR_SCHEMA,
        "stage": "stage2c",
        "receiver": receiver,
        "seed": seed,
        "rows": after_truth,
    }
    truth_path = scorer_root / "truth_sidecar.json"
    _write_json_new(truth_path, truth_sidecar)
    pair_staging = {
        "schema": PAIR_STAGING_SCHEMA,
        "row_handle": row_handle,
        "row_manifest_sha256": row_manifest_sha,
        "before_enrollment_package_root_sha256": state_results["before"][
            "enrollment_package_root_sha256"
        ],
        "after_enrollment_package_root_sha256": state_results["after"][
            "enrollment_package_root_sha256"
        ],
        "before_apply_staging_root": state_results["before"]["apply_staging_root"],
        "after_apply_staging_root": state_results["after"]["apply_staging_root"],
        "old_support_physical_ids_sha256_before": old_support_root,
        "old_support_physical_ids_sha256_after": old_support_root,
        "old_query_physical_ids_sha256_before": old_query_root,
        "old_query_physical_ids_sha256_after": old_query_root,
        "before_after_old_support_query_reuse_audit": "PASS",
        "same_scenario_support_query_disjointness_audit": "PASS",
        "cross_scenario_selected_physical_disjointness_audit": "PASS",
        "before_binding_sha256": None,
        "after_binding_sha256": None,
        "finalization_required": True,
    }
    pair_staging_path = scorer_root / "registration_pair_manifest.json"
    _write_json_new(pair_staging_path, pair_staging)
    build_receipt = {
        "schema": OFFLINE_BUILD_SCHEMA,
        "status": "STAGE1_ENROLLMENT_PACKAGES_AND_APPLY_STAGING_READY",
        "formal_launch_authority": False,
        "lineage_receipt_sha256": receipt_sha,
        "lineage_seal_sha256": lineage_seal_sha,
        "authority_commit_sha256": context["authority_commit_sha256"],
        "authority_attestation_sha256": context[
            "authority_attestation_sha256"
        ],
        "external_authority_lock_verified": context[
            "external_authority_lock_verified"
        ],
        "row_handle": row_handle,
        "row_manifest_sha256": row_manifest_sha,
        "receiver": receiver,
        "seed": seed,
        "k_shot": k_shot,
        "new_class_count": new_class_count,
        "query_per_tx": query_per_tx,
        "old_tx_labels": list(FORMAL_OLD_TX_LABELS),
        "new_tx_labels": list(FORMAL_NEW20_TX_LABELS[:new_class_count]),
        "physical_sample_ids_sha256_by_scenario": dict(
            cache_set["physical_sample_ids_sha256_by_scenario"]
        ),
        "physical_sample_scenario_assignment_sha256": cache_set[
            "physical_sample_scenario_assignment_sha256"
        ],
        "selected_support_physical_ids_sha256_by_scenario": (
            support_physical_roots_by_scenario
        ),
        "selected_query_physical_ids_sha256_by_scenario": (
            query_physical_roots_by_scenario
        ),
        "same_scenario_support_query_disjointness_audit": "PASS",
        "cross_scenario_selected_physical_disjointness_audit": "PASS",
        "before_after_old_support_query_reuse_audit": "PASS",
        "formal_authority_state": (
            "VERIFIED_SINGLE_OBSERVATION_AUTHORITY_V2"
            if context["external_authority_lock_verified"]
            else "DIAGNOSTIC_STRUCTURAL_ONLY_NO_FORMAL_AUTHORITY"
        ),
        "phase1_checkpoint_lineage": {
            "path": str(phase1_checkpoint),
            "sha256": phase1_checkpoint_sha,
            "copied_into_phase2_predictor_package": False,
        },
        "sealed_feature_runtime": {
            "source_path": str(sealed_feature_runtime),
            "sha256": feature_runtime_sha,
            "package_relative_path": bundle.FEATURE_RUNTIME_RELATIVE_PATH,
        },
        "states": state_results,
        "truth_sidecar": str(truth_path),
        "truth_sidecar_sha256": sha256_file(truth_path),
        "registration_pair_manifest": str(pair_staging_path),
        "registration_pair_manifest_sha256": sha256_file(pair_staging_path),
        "predictor_scorer_roots_physically_distinct": True,
        "token_secret_persisted": False,
    }
    _write_json_new(output / "offline_build_receipt.json", build_receipt)
    return build_receipt


def build_somph_offline_row_pair_diagnostic(
    *,
    cache_set_manifest_path: str | Path,
    verified_lineage_loader: Callable[..., Mapping[str, Any]],
    verified_lineage_loader_kwargs: Mapping[str, Any],
    phase1_checkpoint_path: str | Path,
    sealed_feature_runtime_path: str | Path,
    method_lock_path: str | Path,
    output_root: str | Path,
    receiver: str,
    seed: int,
    k_shot: int,
    new_class_count: int,
    query_per_tx: int,
    token_secret: bytes | None = None,
) -> dict[str, Any]:
    """Build an irrevocably diagnostic row pair from a supplied verifier."""

    if not callable(verified_lineage_loader):
        raise SomphOfflinePackageError(
            "verified_lineage_loader must be a diagnostic verifier"
        )
    context = dict(
        verified_lineage_loader(
            cache_set_manifest_path=cache_set_manifest_path,
            **dict(verified_lineage_loader_kwargs),
        )
    )
    # An arbitrary diagnostic callback can never mint formal authority, even
    # when it copies or fabricates the formal-looking context fields.
    context.update(
        {
            "authority_lock": None,
            "authority_attestation": None,
            "authority_commit": None,
            "authority_commit_sha256": None,
            "authority_attestation_sha256": None,
            "external_authority_lock_verified": False,
            "formal_launch_authority": False,
        }
    )
    return _build_somph_offline_row_pair_from_context(
        cache_set_manifest_path=cache_set_manifest_path,
        verified_lineage_context=context,
        formal_staging_authority=False,
        phase1_checkpoint_path=phase1_checkpoint_path,
        sealed_feature_runtime_path=sealed_feature_runtime_path,
        method_lock_path=method_lock_path,
        output_root=output_root,
        receiver=receiver,
        seed=seed,
        k_shot=k_shot,
        new_class_count=new_class_count,
        query_per_tx=query_per_tx,
        token_secret=token_secret,
    )


def build_somph_offline_row_pair(
    *,
    cache_set_manifest_path: str | Path,
    authority_bundle_root: str | Path,
    expected_authority_commit_sha256: str,
    phase1_checkpoint_path: str | Path,
    sealed_feature_runtime_path: str | Path,
    method_lock_path: str | Path,
    output_root: str | Path,
    receiver: str,
    seed: int,
    k_shot: int,
    new_class_count: int,
    query_per_tx: int,
    token_secret: bytes | None = None,
) -> dict[str, Any]:
    """Build one authority-verified formal Stage2-C row pair offline."""

    if (
        not isinstance(query_per_tx, int)
        or isinstance(query_per_tx, bool)
        or query_per_tx != 20
    ):
        raise SomphOfflinePackageError(
            "formal Stage2-C row requires exactly 20 query samples per TX"
        )
    context = load_verified_lineage_context_from_authority_commit(
        cache_set_manifest_path=cache_set_manifest_path,
        authority_bundle_root=authority_bundle_root,
        expected_authority_commit_sha256=expected_authority_commit_sha256,
    )
    return _build_somph_offline_row_pair_from_context(
        cache_set_manifest_path=cache_set_manifest_path,
        verified_lineage_context=context,
        formal_staging_authority=True,
        phase1_checkpoint_path=phase1_checkpoint_path,
        sealed_feature_runtime_path=sealed_feature_runtime_path,
        method_lock_path=method_lock_path,
        output_root=output_root,
        receiver=receiver,
        seed=seed,
        k_shot=k_shot,
        new_class_count=new_class_count,
        query_per_tx=query_per_tx,
        token_secret=token_secret,
    )


def _finalize_somph_apply_package(
    *,
    apply_staging_root: str | Path,
    detached_seal_path: str | Path,
    staging_authority_path: str | Path,
    staging_authority_seal_path: str | Path,
    expected_staging_authority_seal_sha256: str,
    head_capsule_path: str | Path,
    expected_head_capsule_sha256: str,
    expected_head_enrollment_binding_sha256: str,
    require_external_authority: bool,
    authority_bundle_root: str | Path | None,
    expected_authority_commit_sha256: str | None,
) -> dict[str, Any]:
    """Validate all trust roots before publishing one apply-only head."""

    root = Path(apply_staging_root).resolve()
    staging_authority = _load_apply_staging_authority(
        authority_path=staging_authority_path,
        authority_seal_path=staging_authority_seal_path,
        expected_authority_seal_sha256=(
            expected_staging_authority_seal_sha256
        ),
        require_external_authority=require_external_authority,
    )
    if require_external_authority:
        if authority.AUTHORITY_LOCK_SCHEMA.endswith(".v1"):
            raise SomphOfflinePackageError(
                "formal single-observation authority v2 is pending; "
                "diagnostic finalization remains available"
            )
        if (
            authority_bundle_root is None
            or expected_authority_commit_sha256 is None
        ):
            raise SomphOfflinePackageError(
                "formal apply finalization requires authority bundle inputs"
            )
        expected_commit = _require_sha256(
            expected_authority_commit_sha256,
            field="expected_authority_commit_sha256",
        )
        try:
            authority_lock, authority_attestation, authority_commit = (
                authority.verify_somph_lineage_authority_bundle(
                    authority_bundle_root,
                    expected_commit_sha256=expected_commit,
                )
            )
        except authority.SomphLineageAuthorityError as exc:
            raise SomphOfflinePackageError(
                "formal apply authority bundle verification failed"
            ) from exc
        attestation_sha = next(
            item["sha256"]
            for item in authority_commit["members"]
            if item["name"] == authority.AUTHORITY_ATTESTATION_NAME
        )
        failed_authority_bindings = []
        expected_bindings = {
            "authority_commit_sha256": expected_commit,
            "authority_attestation_sha256": attestation_sha,
            "cache_set_manifest_sha256": authority_lock[
                "cache_set_manifest"
            ]["sha256"],
            "cache_physical_sample_ids_sha256_by_scenario": authority_lock[
                "physical_sample_ids_sha256_by_scenario"
            ],
            "cache_physical_sample_scenario_assignment_sha256": authority_lock[
                "physical_sample_scenario_assignment_sha256"
            ],
            "lineage_receipt_sha256": authority_attestation[
                "structural_receipt_sha256"
            ],
            "lineage_seal_sha256": authority_attestation[
                "structural_detached_seal_sha256"
            ],
            "receiver": authority_lock["receiver"],
            "seed": authority_lock["seed"],
        }
        for field, expected in expected_bindings.items():
            if staging_authority.get(field) != expected:
                failed_authority_bindings.append(field)
        if failed_authority_bindings:
            raise SomphOfflinePackageError(
                "formal staging/authority bundle binding drift: "
                f"{failed_authority_bindings}"
            )
    elif (
        authority_bundle_root is not None
        or expected_authority_commit_sha256 is not None
    ):
        raise SomphOfflinePackageError(
            "diagnostic finalization must not accept authority bundle inputs"
        )
    if Path(staging_authority["apply_staging_root"]).resolve() != root:
        raise SomphOfflinePackageError(
            "apply staging root differs from sealed staging authority"
        )
    expected_head = _require_sha256(
        expected_head_capsule_sha256, field="expected_head_capsule_sha256"
    )
    expected_binding = _require_sha256(
        expected_head_enrollment_binding_sha256,
        field="expected_head_enrollment_binding_sha256",
    )
    if sha256_file(root / "overlay_provenance.json") != staging_authority[
        "apply_overlay_provenance_sha256"
    ]:
        raise SomphOfflinePackageError(
            "apply overlay differs from sealed staging authority"
        )
    _prevalidate_external_head(
        head_capsule_path=head_capsule_path,
        expected_head_capsule_sha256=expected_head,
        expected_head_enrollment_binding_sha256=expected_binding,
        staging_authority=staging_authority,
    )
    head_destination = root / bundle.HEAD_CAPSULE_RELATIVE_PATH
    manifest_destination = root / bundle.MANIFEST_RELATIVE_PATH
    seal_destination = Path(detached_seal_path).resolve()
    if (
        head_destination.exists()
        or manifest_destination.exists()
        or seal_destination.exists()
    ):
        raise FileExistsError(
            "refusing to overwrite apply finalization artifacts"
        )
    head_created = False
    try:
        copied = _copy_regular_new(
            head_capsule_path,
            head_destination,
            expected_sha256=expected_head,
        )
        head_created = True
        if copied != expected_head:
            raise SomphOfflinePackageError(
                "head capsule external SHA mismatch"
            )
        os.chmod(head_destination, 0o444)
        _manifest_path, seal_path, manifest, _seal = (
            bundle.write_somph_predictor_bundle(
                root,
                profile=bundle.APPLY_ONLY,
                stage=staging_authority["stage"],
                registration_state=staging_authority[
                    "registration_state"
                ],
                receiver=staging_authority["receiver"],
                seed=staging_authority["seed"],
                k_shot=staging_authority["k_shot"],
                registered_classes=staging_authority[
                    "registered_classes"
                ],
                expected_method_lock_sha256=staging_authority[
                    "method_lock_sha256"
                ],
                expected_overlay_provenance_sha256=staging_authority[
                    "apply_overlay_provenance_sha256"
                ],
                detached_seal_path=seal_destination,
                expected_head_enrollment_binding_sha256=expected_binding,
                expected_head_capsule_sha256=expected_head,
                expected_row_handle=staging_authority["row_handle"],
                expected_row_manifest_sha256=staging_authority[
                    "row_manifest_sha256"
                ],
            )
        )
    except Exception:
        if head_created:
            _remove_owned_file(head_destination)
        raise
    return {
        "schema": APPLY_FINALIZATION_SCHEMA,
        "profile": bundle.APPLY_ONLY,
        "registration_state": staging_authority["registration_state"],
        "package_root_sha256": manifest["package_root_sha256"],
        "package_seal_sha256": sha256_file(seal_path),
        "head_capsule_sha256": expected_head,
        "head_enrollment_binding_sha256": expected_binding,
        "staging_authority_seal_sha256": (
            expected_staging_authority_seal_sha256
        ),
        "authority_commit_sha256": staging_authority[
            "authority_commit_sha256"
        ],
        "external_authority_lock_verified": staging_authority[
            "external_authority_lock_verified"
        ],
        "formal_launch_authority": False,
    }


def finalize_somph_apply_package(
    *,
    apply_staging_root: str | Path,
    detached_seal_path: str | Path,
    staging_authority_path: str | Path,
    staging_authority_seal_path: str | Path,
    expected_staging_authority_seal_sha256: str,
    head_capsule_path: str | Path,
    expected_head_capsule_sha256: str,
    expected_head_enrollment_binding_sha256: str,
    authority_bundle_root: str | Path,
    expected_authority_commit_sha256: str,
) -> dict[str, Any]:
    """Finalize an authority-verified formal apply-only package."""

    return _finalize_somph_apply_package(
        apply_staging_root=apply_staging_root,
        detached_seal_path=detached_seal_path,
        staging_authority_path=staging_authority_path,
        staging_authority_seal_path=staging_authority_seal_path,
        expected_staging_authority_seal_sha256=(
            expected_staging_authority_seal_sha256
        ),
        head_capsule_path=head_capsule_path,
        expected_head_capsule_sha256=expected_head_capsule_sha256,
        expected_head_enrollment_binding_sha256=(
            expected_head_enrollment_binding_sha256
        ),
        require_external_authority=True,
        authority_bundle_root=authority_bundle_root,
        expected_authority_commit_sha256=(
            expected_authority_commit_sha256
        ),
    )


def finalize_somph_apply_package_diagnostic(
    *,
    apply_staging_root: str | Path,
    detached_seal_path: str | Path,
    staging_authority_path: str | Path,
    staging_authority_seal_path: str | Path,
    expected_staging_authority_seal_sha256: str,
    head_capsule_path: str | Path,
    expected_head_capsule_sha256: str,
    expected_head_enrollment_binding_sha256: str,
) -> dict[str, Any]:
    """Finalize an explicitly diagnostic apply-only package."""

    return _finalize_somph_apply_package(
        apply_staging_root=apply_staging_root,
        detached_seal_path=detached_seal_path,
        staging_authority_path=staging_authority_path,
        staging_authority_seal_path=staging_authority_seal_path,
        expected_staging_authority_seal_sha256=(
            expected_staging_authority_seal_sha256
        ),
        head_capsule_path=head_capsule_path,
        expected_head_capsule_sha256=expected_head_capsule_sha256,
        expected_head_enrollment_binding_sha256=(
            expected_head_enrollment_binding_sha256
        ),
        require_external_authority=False,
        authority_bundle_root=None,
        expected_authority_commit_sha256=None,
    )


def finalize_registration_pair_manifest(
    *,
    staging_manifest_path: str | Path,
    output_path: str | Path,
    before_binding_sha256: str,
    after_binding_sha256: str,
) -> dict[str, Any]:
    """Bind post-enrollment stage-input hashes into the scorer pair entity."""

    before = _require_sha256(
        before_binding_sha256, field="before_binding_sha256"
    )
    after = _require_sha256(after_binding_sha256, field="after_binding_sha256")
    staging, _digest, _size = _read_json_same_fd(
        staging_manifest_path, context="SOMP-H registration pair staging"
    )
    expected_keys = {
        "schema",
        "row_handle",
        "row_manifest_sha256",
        "before_enrollment_package_root_sha256",
        "after_enrollment_package_root_sha256",
        "before_apply_staging_root",
        "after_apply_staging_root",
        "old_support_physical_ids_sha256_before",
        "old_support_physical_ids_sha256_after",
        "old_query_physical_ids_sha256_before",
        "old_query_physical_ids_sha256_after",
        "before_after_old_support_query_reuse_audit",
        "same_scenario_support_query_disjointness_audit",
        "cross_scenario_selected_physical_disjointness_audit",
        "before_binding_sha256",
        "after_binding_sha256",
        "finalization_required",
    }
    if (
        set(staging) != expected_keys
        or staging.get("schema") != PAIR_STAGING_SCHEMA
        or staging.get("before_binding_sha256") is not None
        or staging.get("after_binding_sha256") is not None
        or staging.get("finalization_required") is not True
        or staging.get("before_after_old_support_query_reuse_audit") != "PASS"
        or staging.get("same_scenario_support_query_disjointness_audit")
        != "PASS"
        or staging.get("cross_scenario_selected_physical_disjointness_audit")
        != "PASS"
    ):
        raise SomphOfflinePackageError(
            "registration pair staging exact schema drift"
        )
    pair = {
        "schema": REGISTRATION_PAIR_SCHEMA,
        "row_manifest_sha256": staging["row_manifest_sha256"],
        "before_binding_sha256": before,
        "after_binding_sha256": after,
        "old_support_physical_ids_sha256_before": staging[
            "old_support_physical_ids_sha256_before"
        ],
        "old_support_physical_ids_sha256_after": staging[
            "old_support_physical_ids_sha256_after"
        ],
        "old_query_physical_ids_sha256_before": staging[
            "old_query_physical_ids_sha256_before"
        ],
        "old_query_physical_ids_sha256_after": staging[
            "old_query_physical_ids_sha256_after"
        ],
    }
    _write_json_new(Path(output_path), pair)
    return pair


__all__ = [
    "APPLY_FINALIZATION_SCHEMA",
    "DIAGNOSTIC_APPLY_STAGING_AUTHORITY_SCHEMA",
    "DIAGNOSTIC_APPLY_STAGING_AUTHORITY_SEAL_SCHEMA",
    "FORMAL_APPLY_STAGING_AUTHORITY_SCHEMA",
    "FORMAL_APPLY_STAGING_AUTHORITY_SEAL_SCHEMA",
    "OFFLINE_BUILD_SCHEMA",
    "PAIR_STAGING_SCHEMA",
    "VERIFIED_LINEAGE_CONTEXT_SCHEMA",
    "SomphOfflinePackageError",
    "build_somph_offline_row_pair",
    "build_somph_offline_row_pair_diagnostic",
    "finalize_registration_pair_manifest",
    "finalize_somph_apply_package",
    "finalize_somph_apply_package_diagnostic",
    "load_verified_lineage_context_from_authority_commit",
    "load_verified_lineage_context_from_receipt_seal",
]
