#!/usr/bin/env python
"""Fit locked JG_R8_LR020 from a sealed support-only package.

There is deliberately no query CLI argument.  The package profile has an exact
role allowlist which contains registered support and excludes every query/truth
member before any IQ payload is materialised.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
CODE_SCRIPT_ROOT = CODE_ROOT / "scripts"
for value in (str(REPO_ROOT), str(CODE_ROOT), str(CODE_SCRIPT_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT), str(CODE_SCRIPT_ROOT)):
    sys.path.insert(0, value)

from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint  # noqa: E402
from cvsrffi.identity_only_forward import identity_only_feature_forward  # noqa: E402
from cvsrffi.stage2_predictor_bundle import iq_row_sha256  # noqa: E402
from cvsrffi.stage2_predictor_runtime import select_nested_support_prefix  # noqa: E402
from paper_reproduction.cvs_aligned.jg020_stage2c import (  # noqa: E402
    ENROLLMENT_PROFILE,
    FORMAL_SCENARIOS,
    HEAD_SCHEMA,
    RECEIPT_SCHEMA,
    RUNTIME_FIXED_BATCH_SIZE,
    build_head_state,
    descriptor_by_role,
    head_npz_members,
    load_npz_member,
    numpy_from_torch_compat,
    open_regular_member_same_fd,
    prepare_preincrement_adaptation_support,
    preflight_package,
    sha256_file,
    torch_tensor_from_numpy_compat,
    train_support_only_bp_jg_cached,
    validate_direct_class_mapping,
    validate_locked_candidate,
)
from export_adv3b02_effective8_torchscript import (  # noqa: E402
    ADV3B02IdentityRuntime,
)
from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import (  # noqa: E402
    inject_feat_joint_lora,
    load_and_merge_ground_lora,
    merge_feat_joint_lora,
)


SUPPORT_FIELDS = {
    "support_pool_leo_weak_iq",
    "support_pool_class_indices",
    "support_pool_rank_within_class",
    "support_pool_tokens",
    "support_pool_overlay_tokens",
    "support_pool_satellite_seeds",
    "support_pool_post_channel_iq_sha256",
    "manifest_json",
}


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_json_member(root: Path, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    with open_regular_member_same_fd(root, descriptor["relative_path"]) as handle:
        value = json.loads(handle.read().decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("JG020 JSON package member must be an object")
    return value


def _load_checkpoint_member(root: Path, descriptor: Mapping[str, Any]) -> Any:
    with open_regular_member_same_fd(root, descriptor["relative_path"]) as handle:
        try:
            return torch.load(handle, map_location="cpu", weights_only=False)
        except TypeError:
            handle.seek(0)
            return torch.load(handle, map_location="cpu")


def _validate_support_arrays(arrays: Mapping[str, np.ndarray], *, scenario: str, class_count: int) -> None:
    if set(arrays) != SUPPORT_FIELDS:
        raise ValueError("JG020 support NPZ exact field drift")
    manifest = json.loads(str(np.asarray(arrays["manifest_json"]).item()))
    if manifest.get("schema") != "cvs.phase2.registered_support_pool.v2" or manifest.get("scenario") != scenario:
        raise ValueError("JG020 support embedded manifest drift")
    if manifest.get("registered_class_count") != class_count or manifest.get("support_pool_max_k") < 10:
        raise ValueError("JG020 support class/K contract drift")
    iq = np.asarray(arrays["support_pool_leo_weak_iq"], dtype=np.float32)
    hashes = np.asarray(arrays["support_pool_post_channel_iq_sha256"]).astype(str)
    if iq.ndim != 3 or iq.shape[1] != 2 or len(iq) != len(hashes):
        raise ValueError("JG020 support IQ layout drift")
    observed = np.asarray([iq_row_sha256(row) for row in iq])
    if not np.array_equal(observed, hashes):
        raise ValueError("JG020 support sample-level post-channel hash drift")


@torch.no_grad()
def _forward_features(model: torch.nn.Module, rows: np.ndarray, *, device: torch.device, batch_size: int) -> np.ndarray:
    output = []
    model.eval()
    for start in range(0, len(rows), int(batch_size)):
        batch = torch_tensor_from_numpy_compat(
            np.asarray(rows[start : start + batch_size], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        result = identity_only_feature_forward(model, batch, "z_id")
        if result is None:
            raise RuntimeError("ADV3B02 identity-only feature path unavailable")
        output.append(numpy_from_torch_compat(result[0].float(), dtype=np.dtype(np.float32)))
    return np.concatenate(output).astype(np.float32)


@torch.no_grad()
def _runtime_parity(wrapper: torch.nn.Module, runtime: torch.jit.ScriptModule, rows: torch.Tensor) -> dict[str, float]:
    eager_feature, eager_logit = wrapper(rows)
    script_feature, script_logit = runtime(rows)
    return {
        "feature_max_abs": float(torch.max(torch.abs(eager_feature - script_feature)).item()),
        "logit_max_abs": float(torch.max(torch.abs(eager_logit - script_logit)).item()),
    }


def _trace_runtime(wrapper: torch.nn.Module, example: torch.Tensor, output: Path) -> torch.jit.ScriptModule:
    if output.exists():
        raise FileExistsError(output)
    wrapper.eval()
    # Torch 2.1's repeated trace checker compares an internal complex tensor
    # against a real tensor on this model and aborts before publication.  The
    # loaded runtime is checked numerically against eager output immediately
    # below for every exported wrapper, so keep only that explicit parity gate.
    traced = torch.jit.trace(wrapper, example, strict=False, check_trace=False)
    torch.jit.save(traced, output)
    return torch.jit.load(str(output), map_location=example.device).eval()


def _write_trace(output_root: Path, trace: list[dict[str, Any]]) -> None:
    _write_json_new(output_root / "loss_trace.json", {"schema": "cvs.phase2.jg020_loss_trace.v1", "rows": trace})
    columns = list(trace[0]) if trace else []
    with (output_root / "loss_trace.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(trace)


def enroll(args: argparse.Namespace) -> dict[str, Any]:
    package_root = Path(args.package_root).resolve(strict=True)
    document, preopen = preflight_package(
        package_root,
        detached_seal=args.detached_seal,
        expected_seal_sha256=args.expected_seal_sha256,
        expected_profile=ENROLLMENT_PROFILE,
    )
    roles = descriptor_by_role(document)
    lock = validate_locked_candidate(_load_json_member(package_root, roles["candidate_lock"]))
    if document["candidate_lock_sha256"] != roles["candidate_lock"]["sha256"]:
        raise ValueError("JG020 package/candidate lock digest drift")
    checkpoint = _load_checkpoint_member(package_root, roles["checkpoint_full"])
    if roles["checkpoint_full"]["sha256"] != lock["checkpoint_sha256"]:
        raise ValueError("JG020 sealed checkpoint binding drift")
    ground_path = package_root / roles["ground_adapter"]["relative_path"]
    if roles["ground_adapter"]["sha256"] != lock["ground_adapter_sha256"]:
        raise ValueError("JG020 sealed ground adapter binding drift")
    if roles["direct_class_mapping"]["sha256"] != lock["direct_class_mapping_sha256"]:
        raise ValueError("JG020 direct class mapping artifact binding drift")
    direct_mapping = _load_json_member(package_root, roles["direct_class_mapping"])
    direct_mapping_audit = validate_direct_class_mapping(direct_mapping, lock=lock)
    support_all: dict[str, dict[str, np.ndarray]] = {}
    support_iq_by_scenario: dict[str, np.ndarray] = {}
    reference_labels: np.ndarray | None = None
    reference_tokens: np.ndarray | None = None
    for scenario in FORMAL_SCENARIOS:
        arrays = load_npz_member(package_root, roles[f"support:{scenario}"])
        _validate_support_arrays(arrays, scenario=scenario, class_count=document["registered_class_count"])
        selected_iq, selected_y, selected_tokens = select_nested_support_prefix(
            arrays, k_shot=10, class_count=document["registered_class_count"]
        )
        if reference_labels is None:
            reference_labels = selected_y
            reference_tokens = selected_tokens
        elif not np.array_equal(reference_labels, selected_y) or not np.array_equal(reference_tokens, selected_tokens):
            raise ValueError("JG020 physical support ordering drifts across LEO scenarios")
        support_all[scenario] = arrays
        support_iq_by_scenario[scenario] = selected_iq
    assert reference_labels is not None and reference_tokens is not None
    all_physical_ids = reference_tokens.astype(str).tolist()
    if len(set(all_physical_ids)) != len(all_physical_ids):
        raise ValueError("JG020 physical support token collision")
    # Strict temporal chain: adapt once on the pre-increment six-class registry,
    # freeze the runtime, then append prototypes from registered new support.
    # The cached trainer therefore receives no target-new row and contains no
    # old/new branch; registered support class indices are ordinary fit labels.
    adapt_rows, adapt_labels, adapt_row_ids, adapt_physical_ids, adapt_input_audit = (
        prepare_preincrement_adaptation_support(
            support_iq_by_scenario,
            reference_labels,
            reference_tokens,
            old_class_count=6,
            k_shot=lock["k_shot"],
        )
    )

    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    torch.manual_seed(lock["seed"])
    np.random.seed(lock["seed"] % (2**32))
    direct_model, direct_load_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=int(adapt_rows.shape[-1]), device=device
    )
    identity_model, identity_load_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=int(adapt_rows.shape[-1]), device=device
    )
    candidate_model, candidate_load_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=int(adapt_rows.shape[-1]), device=device
    )
    if not (direct_load_audit == identity_load_audit == candidate_load_audit):
        raise ValueError("JG020 three-model checkpoint reconstruction drift")
    direct_model.to(device).eval()
    identity_model.to(device).eval()
    candidate_model.to(device).eval()
    identity_ground_audit = load_and_merge_ground_lora(
        identity_model, ground_path,
        scope=lock["ground_adapter_scope"], rank=lock["ground_adapter_rank"],
        alpha=lock["ground_adapter_alpha"], expected_sha256=lock["ground_adapter_sha256"],
    )
    candidate_ground_audit = load_and_merge_ground_lora(
        candidate_model, ground_path,
        scope=lock["ground_adapter_scope"], rank=lock["ground_adapter_rank"],
        alpha=lock["ground_adapter_alpha"], expected_sha256=lock["ground_adapter_sha256"],
    )
    resources = inject_feat_joint_lora(
        candidate_model, rank=lock["rank"], alpha=lock["alpha"], scope=lock["scope"]
    )
    if resources["trainable_parameters"] != 6_400:
        raise ValueError("JG020 trainable parameter count is not the locked 6,400")
    candidate_model.to(device).eval()
    trace, training_runtime = train_support_only_bp_jg_cached(
        candidate_model,
        adapt_rows,
        adapt_labels,
        physical_support_ids=adapt_physical_ids,
        support_row_physical_ids=adapt_row_ids,
        epochs=lock["epochs"],
        learning_rate=lock["learning_rate"],
        weight_decay=lock["weight_decay"],
        temperature=lock["temperature"],
        support_view_count=lock["support_view_count"],
        batch_size=int(args.batch_size),
        max_optimizer_steps=lock["max_optimizer_steps"],
        grad_clip=lock["grad_clip"],
        seed=lock["seed"],
        device=device,
    )
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    _write_trace(output_root, trace)
    target_state = {
        name: parameter.detach().cpu().half()
        for name, parameter in candidate_model.named_parameters()
        if parameter.requires_grad
    }
    if sum(tensor.numel() for tensor in target_state.values()) != 6_400:
        raise ValueError("JG020 persisted delta element count drift")
    delta_path = output_root / "target_delta_fp16.pt"
    torch.save(target_state, delta_path)
    merge_audit = merge_feat_joint_lora(candidate_model)
    candidate_model.eval()

    candidate_features: dict[str, np.ndarray] = {}
    identity_features: dict[str, np.ndarray] = {}
    for scenario in FORMAL_SCENARIOS:
        candidate_features[scenario] = _forward_features(
            candidate_model, support_iq_by_scenario[scenario], device=device, batch_size=args.batch_size
        )
        identity_features[scenario] = _forward_features(
            identity_model, support_iq_by_scenario[scenario], device=device, batch_size=args.batch_size
        )
    handles = [item["class_handle"] for item in document["registered_classes"]]
    head = build_head_state(
        class_handles=handles,
        old_class_count=6,
        candidate_features_by_scenario=candidate_features,
        identity_features_by_scenario=identity_features,
        support_labels=reference_labels,
        temperature=lock["temperature"],
    )
    for name in list(head):
        if "prototypes__" in name:
            head[name] = np.asarray(head[name], dtype=np.float16)
    head_path = output_root / "prototype_head.npz"
    with head_path.open("xb") as handle:
        np.savez(handle, **head)
    with np.load(head_path, allow_pickle=False) as archive:
        if list(archive.files) != head_npz_members():
            raise ValueError("JG020 persisted head member order drift")
    head_tensor_bytes = int(sum(value.nbytes for name, value in head.items() if "prototypes__" in name))

    probe = torch_tensor_from_numpy_compat(
        np.asarray(adapt_rows[: min(8, len(adapt_rows))], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    if len(probe) < RUNTIME_FIXED_BATCH_SIZE:
        raise ValueError("JG020 runtime parity probe is smaller than the fixed trace batch")
    example = probe[:RUNTIME_FIXED_BATCH_SIZE]
    runtime_specs = {
        "direct_runtime.ts": ADV3B02IdentityRuntime(direct_model).to(device).eval(),
        "identity_runtime.ts": ADV3B02IdentityRuntime(identity_model).to(device).eval(),
        "candidate_runtime.ts": ADV3B02IdentityRuntime(candidate_model).to(device).eval(),
    }
    runtime_audit: dict[str, Any] = {}
    for filename, wrapper in runtime_specs.items():
        path = output_root / filename
        runtime = _trace_runtime(wrapper, example, path)
        parity = _runtime_parity(wrapper, runtime, example)
        if max(parity.values()) > 1.0e-4:
            raise ValueError(f"JG020 TorchScript parity failed: {filename}: {parity}")
        runtime_audit[filename] = {"sha256": sha256_file(path), **parity}

    target_state_bytes = int(sum(tensor.numel() * tensor.element_size() for tensor in target_state.values()))
    ground_state_bytes = int(identity_ground_audit["resources"]["adapter_state_bytes_fp16"])
    persistent_state_bytes = ground_state_bytes + target_state_bytes + head_tensor_bytes
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS",
        "candidate_id": lock["candidate_id"],
        "receiver": lock["receiver"],
        "seed": lock["seed"],
        "k_shot": lock["k_shot"],
        "new_class_count": lock["new_class_count"],
        "trainable_parameters": resources["trainable_parameters"],
        "adapter_macs_per_support_forward_before_merge": resources["adapter_macs_per_query"],
        "deployment_added_macs_per_query_after_merge": merge_audit["deployment_added_macs_per_view_after_merge"],
        "adapt_epochs": lock["epochs"],
        "optimizer_steps": training_runtime["optimizer_steps"],
        "adaptation_wall_seconds": training_runtime["adaptation_wall_seconds"],
        "peak_cuda_memory_bytes": training_runtime["peak_cuda_memory_bytes"],
        "support_forward_sample_equivalents": training_runtime["support_forward_sample_equivalents"],
        "training_compute_mode": training_runtime["training_compute_mode"],
        "full_backbone_forward_call_count": training_runtime["full_backbone_forward_call_count"],
        "full_backbone_forward_sample_equivalents": training_runtime[
            "full_backbone_forward_sample_equivalents"
        ],
        "cached_small_path_forward_count": training_runtime[
            "cached_small_path_forward_count"
        ],
        "cached_small_path_forward_sample_equivalents": training_runtime[
            "cached_small_path_forward_sample_equivalents"
        ],
        "legacy_uncached_full_backbone_forward_sample_equivalents": training_runtime[
            "legacy_uncached_full_backbone_forward_sample_equivalents"
        ],
        "full_backbone_forward_avoided_sample_equivalents": training_runtime[
            "full_backbone_forward_avoided_sample_equivalents"
        ],
        "support_physical_count": len(adapt_physical_ids),
        "adapt_fit_class_count": 6,
        "prototype_fit_class_count": int(document["registered_class_count"]),
        "optimizer_input_stage": "preincrement_registered_old_only",
        "registered_support_labels_used": True,
        "new_support_gradient_used": False,
        "adapter_retrained_at_registration": False,
        "query_role_used_by_optimizer": False,
        "per_sample_old_new_role_branch_used": False,
        "adapt_input_audit": adapt_input_audit,
        "support_view_count": lock["support_view_count"],
        "query_view_count": lock["query_view_count"],
        "ground_state_bytes_fp16": ground_state_bytes,
        "target_delta_tensor_bytes_fp16": target_state_bytes,
        "prototype_head_tensor_bytes_fp16": head_tensor_bytes,
        "persistent_state_bytes": persistent_state_bytes,
        "persistent_state_cap_bytes": 256 * 1024,
        "persistent_state_within_cap": persistent_state_bytes <= 256 * 1024,
        "adapter_alpha": lock["adapter_alpha"],
        "trust_decision": lock["trust_decision"],
        "k1_trust_gate_enabled": lock["k1_trust_gate_enabled"],
        "query_rows_used_for_training": 0,
        "query_path_argument_exists": False,
        "old_new_query_role_used": False,
        "query_class_quota_used": False,
        "dense_query_graph_used": False,
        "package_root_sha256": document["package_root_sha256"],
        "package_seal_sha256": args.expected_seal_sha256,
        "candidate_lock_sha256": document["candidate_lock_sha256"],
        "target_delta_sha256": sha256_file(delta_path),
        "prototype_head_sha256": sha256_file(head_path),
        "loss_trace_sha256": sha256_file(output_root / "loss_trace.json"),
        "runtime_audit": runtime_audit,
        "runtime_fixed_batch_size": RUNTIME_FIXED_BATCH_SIZE,
        "direct_class_mapping_audit": direct_mapping_audit,
        "preopen_audit": preopen,
        "checkpoint_load_audit": candidate_load_audit,
        "identity_ground_audit": identity_ground_audit,
        "candidate_ground_audit": candidate_ground_audit,
        "target_merge_audit": merge_audit,
    }
    if not receipt["persistent_state_within_cap"] or receipt["optimizer_steps"] > 50:
        raise ValueError("JG020 enrollment resource contract failed")
    receipt_path = output_root / "enrollment_receipt.json"
    _write_json_new(receipt_path, receipt)
    return {
        "status": "PASS",
        "output_root": str(output_root),
        "enrollment_receipt": str(receipt_path),
        "enrollment_receipt_sha256": sha256_file(receipt_path),
        "target_delta": str(delta_path),
        "prototype_head": str(head_path),
        "runtime_audit": runtime_audit,
        "resource_summary": {
            key: receipt[key]
            for key in (
                "trainable_parameters", "adapt_epochs", "optimizer_steps",
                "adaptation_wall_seconds", "peak_cuda_memory_bytes", "persistent_state_bytes"
            )
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--detached-seal", type=Path, required=True)
    parser.add_argument("--expected-seal-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(enroll(parse_args()), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
