from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cvsrffi.phase2_candidate_capsule import (
    BASE_CHECKPOINT_SHA256,
    BASE_MODEL_ID,
    CANDIDATE_CAPSULE_SCHEMA,
    EFFECTIVE8_TARGET_MODULES,
    RESOURCE_PROFILE_ID,
    CandidateCapsuleError,
    artifact_descriptor,
    canonical_sha256,
    sha256_file,
    validate_candidate_capsule,
)


def _candidate_lock(
    path: Path,
    *,
    artifacts: dict[str, Path],
    forbidden: bool = False,
) -> Path:
    extras = {}
    for field in ("promotion_manifest", "source_validation", "class_split", "execution_plan"):
        value = path.parent / f"locked_{field}.json"
        if not value.exists():
            value.write_text(json.dumps({"field": field}), encoding="utf-8")
        extras[field] = value
    source_cache = path.parent / "locked_source_cache.json"
    if not source_cache.exists():
        source_cache.write_text('{"cache_scope":"source_train"}', encoding="utf-8")

    def entry(value: Path):
        return {"path": str(value), "sha256": sha256_file(value)}

    candidate = {
        "candidate_id": "ADV3B02_EFFECTIVE8_R16_E12",
        "checkpoint": {
            "path": "/offline/ADV3B02.pth",
            "sha256": BASE_CHECKPOINT_SHA256,
        },
        "adapter_state": entry(artifacts["adapter_state"]),
        "promotion_manifest": entry(extras["promotion_manifest"]),
        "training_manifest": entry(artifacts["adapter_manifest"]),
        "source_validation": entry(extras["source_validation"]),
        "source_feature_statistics": entry(artifacts["source_feature_stats"]),
        "class_split": entry(extras["class_split"]),
        "execution_plan": entry(extras["execution_plan"]),
        "source_leo_weak_cache_sets": {
            "source_train": entry(source_cache),
            "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
        },
        "permissions": {
            "target_support_used_for_selection": forbidden,
            "target_query_features_used_for_selection": False,
            "target_query_labels_used_for_selection": False,
            "old_new_role_oracle_used": False,
            "class_quota_used": False,
            "clean_samples_used": False,
            "clean_derived_signals_used": False,
            "query_fit_used": False,
        },
        "code_artifacts_sha256": {},
    }
    path.write_text(
        json.dumps(
            {
                "schema": "cvs_stage2c_source_candidate_lock_v2",
                "locked_candidate": candidate,
                "locked_candidate_sha256": canonical_sha256(candidate),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _fixture(tmp_path: Path):
    artifact_paths = {}
    schemas = {
        "base_runtime": "adv3b02.torchscript_identity_runtime.v1",
        "candidate_runtime": "adv3b02.torchscript_effective8_merged_runtime.v1",
        "adapter_state": "cvs.effective8_lora_state.fp16.v1",
        "adapter_manifest": "cvs.ground_effective8_adapter_manifest.v1",
        "source_feature_stats": "cvs.source_feature_stats.npz.v1",
        "head_lock": "cvs.symmetric_head_lock.v1",
        "tta_policy": "cvs.phase2.adaptive_rxlight_tta.v1",
    }
    for index, (field, schema) in enumerate(schemas.items()):
        path = tmp_path / f"{field}.bin"
        path.write_bytes(f"artifact-{index}".encode("ascii"))
        artifact_paths[field] = path
    lock = _candidate_lock(
        tmp_path / "candidate_lock.json", artifacts=artifact_paths
    )
    descriptors = {
        field: artifact_descriptor(path, schema=schemas[field])
        for field, path in artifact_paths.items()
    }
    payload = {
        "schema": CANDIDATE_CAPSULE_SCHEMA,
        "candidate_id": "ADV3B02_EFFECTIVE8_R16_E12",
        "resource_profile_id": RESOURCE_PROFILE_ID,
        "base_model_id": BASE_MODEL_ID,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "candidate_lock_sha256": sha256_file(lock),
        **descriptors,
        "merge_parity": {
            "schema": "cvs.adv3b02_effective8_torchscript_parity.v1",
            "receipt_sha256": "1" * 64,
            "base_runtime_sha256": descriptors["base_runtime"]["sha256"],
            "candidate_runtime_sha256": descriptors["candidate_runtime"]["sha256"],
            "adapter_state_sha256": descriptors["adapter_state"]["sha256"],
            "target_modules": list(EFFECTIVE8_TARGET_MODULES),
            "lora_tensor_keys": [f"tensor_{index}" for index in range(16)],
            "max_abs_injected_vs_merged_feature": 1.0e-6,
            "max_abs_injected_vs_merged_logit": 2.0e-6,
            "max_abs_merged_vs_torchscript_feature": 3.0e-6,
            "max_abs_merged_vs_torchscript_logit": 4.0e-6,
            "status": "PASS",
        },
        "resource_accounting": {
            "adapter_trainable_parameters": 44_048,
            "adapter_state_bytes_fp16": 88_096,
            "adapter_serialized_file_bytes": descriptors["adapter_state"]["size_bytes"],
            "ground_adapter_train_epochs": 12,
            "on_orbit_gradient_epochs": 0,
            "on_orbit_optimizer_steps": 0,
            "on_orbit_trainable_parameters": 0,
            "head_state_bytes_fp16": 15_740,
            "tta_threshold_state_bytes": 24,
            "deployment_tensor_payload_bytes": 103_860,
            "deployment_incremental_persistent_bytes": (
                descriptors["adapter_state"]["size_bytes"] + 15_740 + 24
            ),
            "adapter_parameter_cap": 50_000,
            "ground_epoch_cap": 20,
            "deployment_incremental_state_cap_bytes": 256 * 1024,
            "fft_descriptor_dim": 96,
            "fft_descriptor_persistent_bytes": 0,
        },
        "permissions": {
            "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "target_support_used_for_selection": False,
            "target_query_features_used_for_selection": False,
            "target_query_labels_used_for_selection": False,
            "old_new_role_oracle_used": False,
            "query_true_batch_class_count_used": False,
            "class_quota_used": False,
            "query_batch_global_assignment_used": False,
            "query_fit_used": False,
        },
        "deployment_install_contract": {
            "mode": "rebuild_merged_runtime_from_preinstalled_base_plus_delta",
            "base_runtime_preinstalled": True,
            "candidate_runtime_is_reproducible_execution_image": True,
            "candidate_runtime_not_counted_as_incremental_state": True,
            "delta_rebuild_and_parity_required": True,
            "full_merged_runtime_persisted_as_extra_copy": False,
            "deployment_install_evidence_status": "UNVERIFIED",
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    capsule = tmp_path / "candidate_capsule.json"
    capsule.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return payload, capsule, lock, artifact_paths


def _rehash(payload: dict) -> dict:
    value = copy.deepcopy(payload)
    value.pop("payload_sha256", None)
    value["payload_sha256"] = canonical_sha256(value)
    return value


def test_candidate_capsule_accepts_strict_adv3b02_effective8(tmp_path: Path) -> None:
    payload, capsule, lock, artifacts = _fixture(tmp_path)
    verified = validate_candidate_capsule(
        payload,
        capsule_path=capsule,
        expected_capsule_sha256=sha256_file(capsule),
        candidate_lock_path=lock,
        expected_artifact_paths=artifacts,
    )
    assert verified["base_model_id"] == BASE_MODEL_ID
    resource = verified["resource_accounting"]
    assert resource["deployment_tensor_payload_bytes"] == 103_860
    assert resource["deployment_incremental_persistent_bytes"] == (
        payload["adapter_state"]["size_bytes"] + 15_740 + 24
    )


def test_candidate_capsule_rejects_external_trust_root_mismatch(tmp_path: Path) -> None:
    payload, capsule, lock, artifacts = _fixture(tmp_path)
    with pytest.raises(CandidateCapsuleError, match="trust root"):
        validate_candidate_capsule(
            payload,
            capsule_path=capsule,
            expected_capsule_sha256="f" * 64,
            candidate_lock_path=lock,
            expected_artifact_paths=artifacts,
        )


def test_candidate_capsule_structure_cannot_be_mistaken_for_external_trust(
    tmp_path: Path,
) -> None:
    payload, _capsule, lock, artifacts = _fixture(tmp_path)
    with pytest.raises(CandidateCapsuleError, match="external candidate capsule"):
        validate_candidate_capsule(
            payload, candidate_lock_path=lock, expected_artifact_paths=artifacts
        )


def test_candidate_capsule_rejects_effective8_layer_drift(tmp_path: Path) -> None:
    payload, _capsule, lock, artifacts = _fixture(tmp_path)
    payload["merge_parity"]["target_modules"] = list(reversed(EFFECTIVE8_TARGET_MODULES))
    payload = _rehash(payload)
    with pytest.raises(CandidateCapsuleError, match="target-module"):
        validate_candidate_capsule(
            payload,
            candidate_lock_path=lock,
            expected_artifact_paths=artifacts,
            allow_unsealed_build=True,
        )


def test_candidate_capsule_rejects_incremental_byte_misaccounting(tmp_path: Path) -> None:
    payload, _capsule, lock, artifacts = _fixture(tmp_path)
    payload["resource_accounting"]["deployment_incremental_persistent_bytes"] -= 1
    payload = _rehash(payload)
    with pytest.raises(CandidateCapsuleError, match="byte accounting"):
        validate_candidate_capsule(
            payload,
            candidate_lock_path=lock,
            expected_artifact_paths=artifacts,
            allow_unsealed_build=True,
        )


def test_candidate_capsule_rejects_forbidden_candidate_selection_permission(
    tmp_path: Path,
) -> None:
    payload, _capsule, _lock, artifacts = _fixture(tmp_path)
    forbidden_lock = _candidate_lock(
        tmp_path / "forbidden_lock.json", artifacts=artifacts, forbidden=True
    )
    payload["candidate_lock_sha256"] = sha256_file(forbidden_lock)
    payload = _rehash(payload)
    with pytest.raises(CandidateCapsuleError, match="permission"):
        validate_candidate_capsule(
            payload,
            candidate_lock_path=forbidden_lock,
            expected_artifact_paths=artifacts,
            allow_unsealed_build=True,
        )


def test_candidate_capsule_rejects_full_merged_runtime_as_unaccounted_copy(
    tmp_path: Path,
) -> None:
    payload, _capsule, lock, artifacts = _fixture(tmp_path)
    payload["deployment_install_contract"][
        "full_merged_runtime_persisted_as_extra_copy"
    ] = True
    payload = _rehash(payload)
    with pytest.raises(CandidateCapsuleError, match="resource boundary"):
        validate_candidate_capsule(
            payload,
            candidate_lock_path=lock,
            expected_artifact_paths=artifacts,
            allow_unsealed_build=True,
        )
