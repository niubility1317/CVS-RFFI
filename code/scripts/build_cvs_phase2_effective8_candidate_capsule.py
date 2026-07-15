#!/usr/bin/env python
"""Build one externally hashable ADV3B02/effective8 Phase2 candidate capsule."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

from cvsrffi.phase2_candidate_capsule import (  # noqa: E402
    BASE_CHECKPOINT_SHA256,
    BASE_MODEL_ID,
    CANDIDATE_CAPSULE_SCHEMA,
    EFFECTIVE8_TARGET_MODULES,
    RESOURCE_PROFILE_ID,
    artifact_descriptor,
    canonical_sha256,
    sha256_file,
    validate_candidate_capsule,
)


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{context} root must be an object")
    return payload


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _audit_adapter_state(path: Path) -> list[str]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only was introduced
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not state:
        raise ValueError("effective8 adapter state must be a nonempty tensor dictionary")
    keys = sorted(str(key) for key in state)
    if len(keys) != 16 or len(set(keys)) != 16:
        raise ValueError("effective8 adapter state must contain exactly 16 LoRA tensors")
    expected_suffixes = {"lora_a.weight", "lora_b.weight"}
    module_hits: dict[str, set[str]] = {name: set() for name in EFFECTIVE8_TARGET_MODULES}
    elements = 0
    for raw_key, value in state.items():
        key = str(raw_key)
        if not torch.is_tensor(value):
            raise TypeError(f"effective8 adapter state member is not a tensor: {key}")
        if value.dtype != torch.float16 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"effective8 adapter tensor must be finite FP16: {key}")
        matched = False
        for module in EFFECTIVE8_TARGET_MODULES:
            prefix = module + "."
            if key.startswith(prefix) and key[len(prefix) :] in expected_suffixes:
                module_hits[module].add(key[len(prefix) :])
                matched = True
                break
        if not matched:
            raise ValueError(f"unexpected effective8 adapter tensor key: {key}")
        elements += int(value.numel())
    if any(suffixes != expected_suffixes for suffixes in module_hits.values()):
        raise ValueError("effective8 adapter does not cover both LoRA tensors for all 8 layers")
    if elements != 44_048:
        raise ValueError(f"effective8 adapter parameter count drift: {elements}")
    return keys


def _audit_adapter_manifest(path: Path, *, adapter_state: Path) -> dict[str, Any]:
    manifest = _read_json(path, context="adapter manifest")
    resources = dict(manifest.get("resources", {}))
    hyper = dict(manifest.get("hyperparameters", {}))
    checks = {
        "method": manifest.get("method") == "ground_source_effective_feature_lora_v1",
        "state_sha256": manifest.get("adapter_state_sha256") == sha256_file(adapter_state),
        "scope": hyper.get("scope") == "effective_feature",
        "rank": int(hyper.get("rank", -1)) == 16,
        "alpha": float(hyper.get("alpha", -1.0)) == 16.0,
        "epochs": int(hyper.get("epochs", manifest.get("epochs", -1))) == 12,
        "parameters": int(resources.get("trainable_parameters", -1)) == 44_048,
        "state_bytes": int(resources.get("adapter_state_bytes_fp16", -1)) == 88_096,
        "query_update_forbidden": manifest.get("query_update_forbidden") is True,
        "query_labels_unused": manifest.get("query_labels_used_for_training") is False,
        "role_oracle_unused": manifest.get("old_new_role_used_by_optimizer") is False,
        "class_quota_unused": manifest.get("class_quota_used_at_inference") is False,
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"effective8 adapter manifest audit failed: {failed}")
    return manifest


def _parity_summary(
    path: Path,
    *,
    base_runtime: Path,
    candidate_runtime: Path,
    adapter_state: Path,
    lora_tensor_keys: list[str],
) -> dict[str, Any]:
    receipt = _read_json(path, context="TorchScript parity receipt")
    required = {
        "schema",
        "status",
        "base_runtime_sha256",
        "candidate_runtime_sha256",
        "adapter_state_sha256",
        "target_modules",
        "lora_tensor_keys",
        "max_abs_injected_vs_merged_feature",
        "max_abs_injected_vs_merged_logit",
        "max_abs_merged_vs_torchscript_feature",
        "max_abs_merged_vs_torchscript_logit",
    }
    if set(receipt) != required:
        raise ValueError("TorchScript parity receipt exact schema mismatch")
    expected = {
        "schema": "cvs.adv3b02_effective8_torchscript_parity.v1",
        "status": "PASS",
        "base_runtime_sha256": sha256_file(base_runtime),
        "candidate_runtime_sha256": sha256_file(candidate_runtime),
        "adapter_state_sha256": sha256_file(adapter_state),
        "target_modules": list(EFFECTIVE8_TARGET_MODULES),
        "lora_tensor_keys": lora_tensor_keys,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"TorchScript parity receipt drift: {key}")
    return {**receipt, "receipt_sha256": sha256_file(path)}


def _audit_head_and_tta_policy(
    candidate_lock: dict[str, Any], *, head_lock: Path, tta_policy: Path
) -> None:
    candidate = dict(candidate_lock.get("locked_candidate", {}))
    expected_head = {
        "schema": "cvs.symmetric_head_lock.v1",
        **dict(candidate.get("head", {})),
    }
    if _read_json(head_lock, context="symmetric head lock") != expected_head:
        raise ValueError("symmetric head lock differs from the source candidate lock")
    adaptive = dict(candidate.get("adaptive_tta", {}))
    thresholds = dict(adaptive.get("thresholds", {}))
    expected_tta = {
        "schema": "cvs.phase2.adaptive_rxlight_tta.v1",
        "mode": "adaptive_1_3_5",
        "base_views": 1,
        "max_views": 5,
        "base_stop_margin": thresholds.get("base_stop_margin"),
        "shift3_stop_margin": thresholds.get("shift3_stop_margin"),
        "shift3_max_disagreement": thresholds.get("shift3_max_disagreement"),
        "base_stop_min_score": thresholds.get("base_stop_min_score"),
        "shift3_stop_min_score": thresholds.get("shift3_stop_min_score"),
        "fusion_std_penalty": thresholds.get("fusion_std_penalty"),
        "calibration_scope": "source_validation",
        "uses_query_labels": False,
        "uses_query_role": False,
        "uses_class_quota": False,
    }
    if _read_json(tta_policy, context="adaptive TTA policy") != expected_tta:
        raise ValueError("adaptive TTA policy differs from the source candidate lock")


def build(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "base_runtime": Path(args.base_runtime),
        "candidate_runtime": Path(args.candidate_runtime),
        "adapter_state": Path(args.adapter_state),
        "adapter_manifest": Path(args.adapter_manifest),
        "source_feature_stats": Path(args.source_feature_stats),
        "head_lock": Path(args.head_lock),
        "tta_policy": Path(args.tta_policy),
    }
    lora_tensor_keys = _audit_adapter_state(paths["adapter_state"])
    _audit_adapter_manifest(paths["adapter_manifest"], adapter_state=paths["adapter_state"])
    candidate_lock = _read_json(Path(args.candidate_lock), context="candidate lock")
    if candidate_lock.get("schema") != "cvs_stage2c_source_candidate_lock_v2":
        raise ValueError("candidate lock schema mismatch")
    _audit_head_and_tta_policy(
        candidate_lock,
        head_lock=paths["head_lock"],
        tta_policy=paths["tta_policy"],
    )
    parity = _parity_summary(
        Path(args.parity_receipt),
        base_runtime=paths["base_runtime"],
        candidate_runtime=paths["candidate_runtime"],
        adapter_state=paths["adapter_state"],
        lora_tensor_keys=lora_tensor_keys,
    )
    head_bytes = 15_740
    payload: dict[str, Any] = {
        "schema": CANDIDATE_CAPSULE_SCHEMA,
        "candidate_id": str(args.candidate_id),
        "resource_profile_id": RESOURCE_PROFILE_ID,
        "base_model_id": BASE_MODEL_ID,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "candidate_lock_sha256": sha256_file(Path(args.candidate_lock)),
        "base_runtime": artifact_descriptor(
            paths["base_runtime"], schema="adv3b02.torchscript_identity_runtime.v1"
        ),
        "candidate_runtime": artifact_descriptor(
            paths["candidate_runtime"],
            schema="adv3b02.torchscript_effective8_merged_runtime.v1",
        ),
        "adapter_state": artifact_descriptor(
            paths["adapter_state"], schema="cvs.effective8_lora_state.fp16.v1"
        ),
        "adapter_manifest": artifact_descriptor(
            paths["adapter_manifest"], schema="cvs.ground_effective8_adapter_manifest.v1"
        ),
        "source_feature_stats": artifact_descriptor(
            paths["source_feature_stats"], schema="cvs.source_feature_stats.npz.v1"
        ),
        "head_lock": artifact_descriptor(
            paths["head_lock"], schema="cvs.symmetric_head_lock.v1"
        ),
        "tta_policy": artifact_descriptor(
            paths["tta_policy"], schema="cvs.phase2.adaptive_rxlight_tta.v1"
        ),
        "merge_parity": parity,
        "resource_accounting": {
            "adapter_trainable_parameters": 44_048,
            "adapter_state_bytes_fp16": 88_096,
            "adapter_serialized_file_bytes": int(paths["adapter_state"].stat().st_size),
            "ground_adapter_train_epochs": 12,
            "on_orbit_gradient_epochs": 0,
            "on_orbit_optimizer_steps": 0,
            "on_orbit_trainable_parameters": 0,
            "head_state_bytes_fp16": head_bytes,
            "tta_threshold_state_bytes": 24,
            "deployment_tensor_payload_bytes": 88_096 + head_bytes + 24,
            "deployment_incremental_persistent_bytes": int(
                paths["adapter_state"].stat().st_size
            )
            + head_bytes
            + 24,
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
            "deployment_install_evidence_status": "PASS",
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    validate_candidate_capsule(
        payload,
        candidate_lock_path=Path(args.candidate_lock),
        expected_artifact_paths=paths,
        allow_unsealed_build=True,
    )
    output = Path(args.out_json)
    _write_json_new(output, payload)
    return {
        "candidate_capsule": str(output),
        "candidate_capsule_sha256": sha256_file(output),
        "candidate_id": payload["candidate_id"],
        "resource_profile_id": RESOURCE_PROFILE_ID,
        "deployment_incremental_persistent_bytes": payload["resource_accounting"][
            "deployment_incremental_persistent_bytes"
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument("--base-runtime", type=Path, required=True)
    parser.add_argument("--candidate-runtime", type=Path, required=True)
    parser.add_argument("--adapter-state", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--source-feature-stats", type=Path, required=True)
    parser.add_argument("--head-lock", type=Path, required=True)
    parser.add_argument("--tta-policy", type=Path, required=True)
    parser.add_argument("--parity-receipt", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(build(parse_args()), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
