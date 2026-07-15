"""Fail-closed ADV3B02/effective8 candidate capsule validation.

The capsule is produced before Phase2 and contains no target sample, target
label, or query-derived value.  Its detached file digest is the candidate
trust root supplied by the controller.  The strict predictor package may carry
a byte-for-byte copy, but it must never be allowed to choose that digest.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


CANDIDATE_CAPSULE_SCHEMA = "cvs.phase2.adv3b02_effective8_candidate_capsule.v1"
BASE_MODEL_ID = "ADV3B02_CORE90_SOFT_E200"
BASE_CHECKPOINT_SHA256 = (
    "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
)
RESOURCE_PROFILE_ID = "effective8_r16_e12_preferred_v1"
EFFECTIVE8_TARGET_MODULES = (
    "id_backbone.t_proj",
    "id_backbone.f_proj",
    "id_backbone.pa_proj.0",
    "id_backbone.fuse.0",
    "id_backbone.cls_head.id_proj.0",
    "id_backbone.cls_head.pa_proj.0",
    "id_backbone.cls_head.id_gate.0",
    "id_backbone.cls_head.joint_proj.0",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")

CAPSULE_REQUIRED_KEYS = {
    "schema",
    "candidate_id",
    "resource_profile_id",
    "base_model_id",
    "base_checkpoint_sha256",
    "candidate_lock_sha256",
    "base_runtime",
    "candidate_runtime",
    "adapter_state",
    "adapter_manifest",
    "source_feature_stats",
    "head_lock",
    "tta_policy",
    "merge_parity",
    "resource_accounting",
    "permissions",
    "deployment_install_contract",
    "payload_sha256",
}
ARTIFACT_REQUIRED_KEYS = {"sha256", "size_bytes", "schema"}
RESOURCE_REQUIRED_KEYS = {
    "adapter_trainable_parameters",
    "adapter_state_bytes_fp16",
    "adapter_serialized_file_bytes",
    "ground_adapter_train_epochs",
    "on_orbit_gradient_epochs",
    "on_orbit_optimizer_steps",
    "on_orbit_trainable_parameters",
    "head_state_bytes_fp16",
    "tta_threshold_state_bytes",
    "deployment_tensor_payload_bytes",
    "deployment_incremental_persistent_bytes",
    "adapter_parameter_cap",
    "ground_epoch_cap",
    "deployment_incremental_state_cap_bytes",
    "fft_descriptor_dim",
    "fft_descriptor_persistent_bytes",
}
PERMISSION_REQUIRED_KEYS = {
    "phase2_sample_view_policy",
    "clean_sample_access",
    "clean_derived_signal_access",
    "target_support_used_for_selection",
    "target_query_features_used_for_selection",
    "target_query_labels_used_for_selection",
    "old_new_role_oracle_used",
    "query_true_batch_class_count_used",
    "class_quota_used",
    "query_batch_global_assignment_used",
    "query_fit_used",
}
INSTALL_REQUIRED_KEYS = {
    "mode",
    "base_runtime_preinstalled",
    "candidate_runtime_is_reproducible_execution_image",
    "candidate_runtime_not_counted_as_incremental_state",
    "delta_rebuild_and_parity_required",
    "full_merged_runtime_persisted_as_extra_copy",
    "deployment_install_evidence_status",
}
PARITY_REQUIRED_KEYS = {
    "schema",
    "receipt_sha256",
    "base_runtime_sha256",
    "candidate_runtime_sha256",
    "adapter_state_sha256",
    "target_modules",
    "lora_tensor_keys",
    "max_abs_injected_vs_merged_feature",
    "max_abs_injected_vs_merged_logit",
    "max_abs_merged_vs_torchscript_feature",
    "max_abs_merged_vs_torchscript_logit",
    "status",
}


class CandidateCapsuleError(ValueError):
    """Raised before a candidate artifact is accepted into a Phase2 package."""


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value.lower()) is not None


def _load_json_file(path: Path, *, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CandidateCapsuleError(f"{context} must be a regular non-symlink file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateCapsuleError(f"{context} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise CandidateCapsuleError(f"{context} JSON root must be an object")
    return payload


def artifact_descriptor(path: str | Path, *, schema: str) -> dict[str, Any]:
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise CandidateCapsuleError(f"candidate artifact must be a regular file: {value}")
    return {
        "sha256": sha256_file(value),
        "size_bytes": int(value.stat().st_size),
        "schema": str(schema),
    }


def _validate_artifact(
    value: Any,
    *,
    field: str,
    expected_path: Path | None = None,
    expected_schema: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != ARTIFACT_REQUIRED_KEYS:
        raise CandidateCapsuleError(f"candidate artifact schema mismatch: {field}")
    item = dict(value)
    if not _is_sha256(item["sha256"]):
        raise CandidateCapsuleError(f"candidate artifact SHA256 invalid: {field}")
    if not isinstance(item["size_bytes"], int) or item["size_bytes"] < 1:
        raise CandidateCapsuleError(f"candidate artifact size invalid: {field}")
    if not isinstance(item["schema"], str) or not item["schema"]:
        raise CandidateCapsuleError(f"candidate artifact type invalid: {field}")
    if expected_schema is not None and item["schema"] != expected_schema:
        raise CandidateCapsuleError(f"candidate artifact type drift: {field}")
    if expected_path is not None:
        actual = artifact_descriptor(expected_path, schema=item["schema"])
        if item != actual:
            raise CandidateCapsuleError(f"candidate artifact content drift: {field}")
    return item


def _validate_candidate_lock(
    path: Path,
    expected_sha256: str,
    *,
    candidate_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> None:
    if sha256_file(path) != expected_sha256:
        raise CandidateCapsuleError("candidate lock file digest mismatch")
    lock = _load_json_file(path, context="candidate lock")
    if lock.get("schema") != "cvs_stage2c_source_candidate_lock_v2":
        raise CandidateCapsuleError("candidate lock schema mismatch")
    candidate = lock.get("locked_candidate")
    if not isinstance(candidate, dict) or canonical_sha256(candidate) != lock.get(
        "locked_candidate_sha256"
    ):
        raise CandidateCapsuleError("candidate lock self-hash mismatch")
    checkpoint = dict(candidate.get("checkpoint", {}))
    if checkpoint.get("sha256") != BASE_CHECKPOINT_SHA256:
        raise CandidateCapsuleError("candidate lock does not bind the strict ADV3B02 checkpoint")
    if candidate.get("candidate_id") != candidate_id:
        raise CandidateCapsuleError("candidate lock/capsule candidate id mismatch")
    immutable_fields = (
        "adapter_state",
        "promotion_manifest",
        "training_manifest",
        "source_validation",
        "source_feature_statistics",
        "class_split",
        "execution_plan",
    )
    for field in immutable_fields:
        entry = dict(candidate.get(field, {}))
        artifact_path = Path(str(entry.get("path", "")))
        if (
            not artifact_path.is_file()
            or artifact_path.is_symlink()
            or not _is_sha256(entry.get("sha256"))
            or sha256_file(artifact_path) != entry["sha256"]
        ):
            raise CandidateCapsuleError(f"candidate lock immutable artifact drift: {field}")
    if candidate["adapter_state"]["sha256"] != artifacts["adapter_state"]["sha256"]:
        raise CandidateCapsuleError("candidate lock/capsule adapter-state mismatch")
    if candidate["training_manifest"]["sha256"] != artifacts["adapter_manifest"]["sha256"]:
        raise CandidateCapsuleError("candidate lock/capsule adapter-manifest mismatch")
    if (
        candidate["source_feature_statistics"]["sha256"]
        != artifacts["source_feature_stats"]["sha256"]
    ):
        raise CandidateCapsuleError("candidate lock/capsule feature-statistics mismatch")
    for field, entry in dict(candidate.get("source_leo_weak_cache_sets", {})).items():
        if field in {
            "phase2_sample_view_policy",
            "clean_sample_access",
            "clean_derived_signal_access",
        }:
            continue
        cache = dict(entry)
        cache_path = Path(str(cache.get("path", "")))
        if (
            not cache_path.is_file()
            or cache_path.is_symlink()
            or not _is_sha256(cache.get("sha256"))
            or sha256_file(cache_path) != cache["sha256"]
        ):
            raise CandidateCapsuleError(f"candidate lock immutable source cache drift: {field}")
    repo_root = Path(__file__).resolve().parents[2]
    for relative, expected in dict(candidate.get("code_artifacts_sha256", {})).items():
        code_path = repo_root / str(relative)
        if not _is_sha256(expected) or not code_path.is_file() or sha256_file(code_path) != expected:
            raise CandidateCapsuleError(f"candidate lock code artifact drift: {relative}")
    permissions = dict(candidate.get("permissions", {}))
    forbidden_true = (
        "target_support_used_for_selection",
        "target_query_features_used_for_selection",
        "target_query_labels_used_for_selection",
        "old_new_role_oracle_used",
        "class_quota_used",
        "clean_samples_used",
        "clean_derived_signals_used",
        "query_fit_used",
    )
    if any(permissions.get(key) is not False for key in forbidden_true):
        raise CandidateCapsuleError("candidate lock permission contract is not strict")


def validate_candidate_capsule(
    payload: Mapping[str, Any],
    *,
    expected_capsule_sha256: str | None = None,
    capsule_path: Path | None = None,
    candidate_lock_path: Path | None = None,
    expected_artifact_paths: Mapping[str, Path] | None = None,
    allow_unsealed_build: bool = False,
) -> dict[str, Any]:
    """Validate exact schema, provenance, parity, and deployment resource claims."""

    if set(payload) != CAPSULE_REQUIRED_KEYS:
        raise CandidateCapsuleError("candidate capsule exact schema mismatch")
    capsule = dict(payload)
    if capsule.get("schema") != CANDIDATE_CAPSULE_SCHEMA:
        raise CandidateCapsuleError("candidate capsule schema version drift")
    content = {key: value for key, value in capsule.items() if key != "payload_sha256"}
    if not _is_sha256(capsule.get("payload_sha256")) or canonical_sha256(
        content
    ) != capsule["payload_sha256"]:
        raise CandidateCapsuleError("candidate capsule payload self-hash mismatch")
    if capsule_path is not None:
        actual_file_sha = sha256_file(capsule_path)
        if expected_capsule_sha256 is not None and actual_file_sha != expected_capsule_sha256:
            raise CandidateCapsuleError("external candidate capsule trust root mismatch")
    elif expected_capsule_sha256 is not None:
        raise CandidateCapsuleError("capsule path is required for external trust-root verification")
    elif not allow_unsealed_build:
        raise CandidateCapsuleError(
            "external candidate capsule trust root is required outside the builder"
        )
    if not isinstance(capsule.get("candidate_id"), str) or not capsule["candidate_id"]:
        raise CandidateCapsuleError("candidate id must be nonempty")
    if capsule.get("resource_profile_id") != RESOURCE_PROFILE_ID:
        raise CandidateCapsuleError("candidate resource profile drift")
    if capsule.get("base_model_id") != BASE_MODEL_ID:
        raise CandidateCapsuleError("candidate base model is not ADV3B02")
    if capsule.get("base_checkpoint_sha256") != BASE_CHECKPOINT_SHA256:
        raise CandidateCapsuleError("candidate base checkpoint digest drift")
    if not _is_sha256(capsule.get("candidate_lock_sha256")):
        raise CandidateCapsuleError("candidate lock digest invalid")
    expected_paths = dict(expected_artifact_paths or {})
    schemas = {
        "base_runtime": "adv3b02.torchscript_identity_runtime.v1",
        "candidate_runtime": "adv3b02.torchscript_effective8_merged_runtime.v1",
        "adapter_state": "cvs.effective8_lora_state.fp16.v1",
        "adapter_manifest": "cvs.ground_effective8_adapter_manifest.v1",
        "source_feature_stats": "cvs.source_feature_stats.npz.v1",
        "head_lock": "cvs.symmetric_head_lock.v1",
        "tta_policy": "cvs.phase2.adaptive_rxlight_tta.v1",
    }
    artifacts = {
        field: _validate_artifact(
            capsule[field],
            field=field,
            expected_path=expected_paths.get(field),
            expected_schema=schema,
        )
        for field, schema in schemas.items()
    }
    if candidate_lock_path is not None:
        _validate_candidate_lock(
            candidate_lock_path,
            capsule["candidate_lock_sha256"],
            candidate_id=capsule["candidate_id"],
            artifacts=artifacts,
        )

    parity = capsule.get("merge_parity")
    if not isinstance(parity, Mapping) or set(parity) != PARITY_REQUIRED_KEYS:
        raise CandidateCapsuleError("candidate merge parity schema mismatch")
    parity = dict(parity)
    if parity.get("schema") != "cvs.adv3b02_effective8_torchscript_parity.v1":
        raise CandidateCapsuleError("candidate merge parity version drift")
    if parity.get("status") != "PASS" or not _is_sha256(parity.get("receipt_sha256")):
        raise CandidateCapsuleError("candidate merge parity is not PASS")
    if parity.get("base_runtime_sha256") != artifacts["base_runtime"]["sha256"]:
        raise CandidateCapsuleError("base runtime/parity digest mismatch")
    if parity.get("candidate_runtime_sha256") != artifacts["candidate_runtime"]["sha256"]:
        raise CandidateCapsuleError("candidate runtime/parity digest mismatch")
    if parity.get("adapter_state_sha256") != artifacts["adapter_state"]["sha256"]:
        raise CandidateCapsuleError("adapter/parity digest mismatch")
    if tuple(parity.get("target_modules", ())) != EFFECTIVE8_TARGET_MODULES:
        raise CandidateCapsuleError("effective8 target-module set/order drift")
    keys = parity.get("lora_tensor_keys")
    if not isinstance(keys, list) or len(keys) != 16 or len(set(keys)) != 16:
        raise CandidateCapsuleError("effective8 LoRA tensor-key contract drift")
    for field in (
        "max_abs_injected_vs_merged_feature",
        "max_abs_injected_vs_merged_logit",
        "max_abs_merged_vs_torchscript_feature",
        "max_abs_merged_vs_torchscript_logit",
    ):
        value = parity.get(field)
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0e-4:
            raise CandidateCapsuleError(f"candidate parity tolerance failed: {field}")

    resource = capsule.get("resource_accounting")
    if not isinstance(resource, Mapping) or set(resource) != RESOURCE_REQUIRED_KEYS:
        raise CandidateCapsuleError("candidate resource accounting schema mismatch")
    resource = dict(resource)
    exact = {
        "adapter_trainable_parameters": 44_048,
        "adapter_state_bytes_fp16": 88_096,
        "ground_adapter_train_epochs": 12,
        "on_orbit_gradient_epochs": 0,
        "on_orbit_optimizer_steps": 0,
        "on_orbit_trainable_parameters": 0,
        "adapter_parameter_cap": 50_000,
        "ground_epoch_cap": 20,
        "deployment_incremental_state_cap_bytes": 256 * 1024,
        "fft_descriptor_dim": 96,
        "fft_descriptor_persistent_bytes": 0,
    }
    if any(resource.get(key) != value for key, value in exact.items()):
        raise CandidateCapsuleError("candidate effective8 resource claim drift")
    for field in (
        "adapter_serialized_file_bytes",
        "head_state_bytes_fp16",
        "tta_threshold_state_bytes",
        "deployment_tensor_payload_bytes",
        "deployment_incremental_persistent_bytes",
    ):
        if not isinstance(resource.get(field), int) or resource[field] < 0:
            raise CandidateCapsuleError(f"candidate resource field invalid: {field}")
    if resource["tta_threshold_state_bytes"] != 24:
        raise CandidateCapsuleError("adaptive TTA must persist exactly six float32 thresholds")
    if resource["adapter_serialized_file_bytes"] != artifacts["adapter_state"]["size_bytes"]:
        raise CandidateCapsuleError("serialized adapter byte accounting mismatch")
    expected_tensor_payload = (
        resource["adapter_state_bytes_fp16"]
        + resource["head_state_bytes_fp16"]
        + resource["tta_threshold_state_bytes"]
    )
    if resource["deployment_tensor_payload_bytes"] != expected_tensor_payload:
        raise CandidateCapsuleError("candidate tensor-payload byte accounting mismatch")
    expected_incremental = (
        resource["adapter_serialized_file_bytes"]
        + resource["head_state_bytes_fp16"]
        + resource["tta_threshold_state_bytes"]
    )
    if resource["deployment_incremental_persistent_bytes"] != expected_incremental:
        raise CandidateCapsuleError("candidate incremental-state byte accounting mismatch")
    if expected_incremental > resource["deployment_incremental_state_cap_bytes"]:
        raise CandidateCapsuleError("candidate incremental state exceeds 256KiB")

    permissions = capsule.get("permissions")
    if not isinstance(permissions, Mapping) or set(permissions) != PERMISSION_REQUIRED_KEYS:
        raise CandidateCapsuleError("candidate permission schema mismatch")
    permissions = dict(permissions)
    if permissions.get("phase2_sample_view_policy") != "leo_weak_only_no_clean_access":
        raise CandidateCapsuleError("candidate sample-view policy drift")
    if any(value is not False for key, value in permissions.items() if key != "phase2_sample_view_policy"):
        raise CandidateCapsuleError("candidate contains forbidden data or Oracle permission")

    install = capsule.get("deployment_install_contract")
    if not isinstance(install, Mapping) or set(install) != INSTALL_REQUIRED_KEYS:
        raise CandidateCapsuleError("candidate deployment-install schema mismatch")
    expected_install = {
        "mode": "rebuild_merged_runtime_from_preinstalled_base_plus_delta",
        "base_runtime_preinstalled": True,
        "candidate_runtime_is_reproducible_execution_image": True,
        "candidate_runtime_not_counted_as_incremental_state": True,
        "delta_rebuild_and_parity_required": True,
        "full_merged_runtime_persisted_as_extra_copy": False,
        "deployment_install_evidence_status": "PASS",
    }
    if dict(install) != expected_install:
        raise CandidateCapsuleError("candidate deployment resource boundary is not auditable")
    return capsule


def load_and_validate_candidate_capsule(
    capsule_path: str | Path,
    *,
    expected_capsule_sha256: str,
    candidate_lock_path: Path | None = None,
    expected_artifact_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    path = Path(capsule_path)
    payload = _load_json_file(path, context="candidate capsule")
    return validate_candidate_capsule(
        payload,
        expected_capsule_sha256=expected_capsule_sha256,
        capsule_path=path,
        candidate_lock_path=candidate_lock_path,
        expected_artifact_paths=expected_artifact_paths,
        allow_unsealed_build=False,
    )
