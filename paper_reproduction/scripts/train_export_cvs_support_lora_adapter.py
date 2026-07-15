#!/usr/bin/env python
"""Train a support-only lightweight adapter on the ADV3B02 identity path.

All original checkpoint parameters remain frozen.  The trainer supports either
identity-initialized LoRA branches or a 1,280-parameter late channel-wise FiLM
adapter.  Training consumes registered target support labels and preregistered
LEO support views; target query rows never enter fitting or model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    while candidate in sys.path:
        sys.path.remove(candidate)
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, candidate)

from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
from export_spaceborne_features import _satellite_tta_views
from paper_reproduction.cvs_aligned.cvs_method_runner import SCENARIOS
from paper_reproduction.scripts.train_export_cvs_micro_iq_adapter import (
    _batched_feature_forward,
    _class_prototypes,
    _feature_forward,
    _json_safe,
    _load_npz,
    _norm_rows,
    _numpy_to_tensor_compat,
    _sha256_file,
    _tensor_to_numpy_compat,
    _write_trace,
    assemble_support_views,
    export_adapted_cache,
)


LORA_TARGETS = (
    "id_backbone.cls_head.id_proj.0",
    "id_backbone.cls_head.pa_proj.0",
    "id_backbone.cls_head.id_gate.0",
    "id_backbone.cls_head.joint_proj.0",
)

LATE_LORA_TARGETS = (
    "id_backbone.t_proj",
    "id_backbone.f_proj",
    "id_backbone.pa_proj.0",
    "id_backbone.fuse.0",
    *LORA_TARGETS,
)

PROJECTION_LORA_TARGETS = (
    "id_backbone.t_proj",
    "id_backbone.f_proj",
    "id_backbone.pa_proj.0",
    "id_backbone.fuse.0",
)

JOINT_PROJECTION_LORA_TARGETS = (
    "id_backbone.cls_head.joint_proj.0",
)

JOINT_GATE_LORA_TARGETS = (
    "id_backbone.cls_head.id_gate.0",
    "id_backbone.cls_head.joint_proj.0",
)

FULL_FEATURE_LORA_TARGETS = (
    "id_backbone.t_proj",
    "id_backbone.f_proj",
    "id_backbone.pa_proj.0",
    "id_backbone.fuse.0",
    "id_backbone.con_proj.0",
    *LORA_TARGETS,
    "id_backbone.cls_head.imp_merge.0",
)

# The two extra modules in FULL_FEATURE_LORA_TARGETS are kept for exact
# compatibility with historical states.  Neither contributes to the deployed
# qKNN feature (feat_joint), so new extreme-light ground adapters use this
# effective eight-layer path instead.
EFFECTIVE_FEATURE_LORA_TARGETS = LATE_LORA_TARGETS

LATE_FILM_TARGETS = (
    "id_backbone.t_proj",
    "id_backbone.f_proj",
    "id_backbone.pa_proj.0",
    "id_backbone.fuse.0",
)

LATE_KEY_FT_TARGETS = (
    "id_backbone.t_proj",
    "id_backbone.f_proj",
    "id_backbone.pa_proj.0",
)

RX_SHIFT_PAIR_SLOT_COUNT = 5
RX_SHIFT_PAIR_VIEW_NAMES = ("rx_base", "rx_shift_m2", "rx_shift_p2")


def build_rx_shift_pair_cycle(
    support_rows: np.ndarray,
    support_labels: np.ndarray,
    *,
    input_view_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build five two-view epoch slots from three legal support scenarios.

    Each slot contains the base view plus exactly one +/-2 sample shift.  The
    scenario rotates over the three formal LEO caches while the shift sign
    alternates.  Across the whole adaptation there are only three registered
    receive views (base, shift -2, shift +2), preserving the project cap.
    """

    rows = np.asarray(support_rows, dtype=np.float32)
    labels = np.asarray(support_labels, dtype=np.int64)
    if int(input_view_count) != len(SCENARIOS):
        raise ValueError(
            "rx_shift_pair_cycle requires exactly three formal scenario views"
        )
    if len(rows) == 0 or len(rows) % int(input_view_count) != 0:
        raise ValueError("support rows cannot be grouped by formal scenario")
    physical_count = int(len(rows) // int(input_view_count))
    base_labels = labels[:physical_count]
    scenario_views: list[dict[str, np.ndarray]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        start = int(scenario_index * physical_count)
        stop = int(start + physical_count)
        if not np.array_equal(labels[start:stop], base_labels):
            raise ValueError(f"support label alignment drift for {scenario}")
        tensor = _numpy_to_tensor_compat(
            rows[start:stop],
            numpy_dtype=np.dtype(np.float32),
            torch_dtype=torch.float32,
        )
        generated = _satellite_tta_views(tensor, "rx_shift3")
        names = tuple(name for name, _ in generated)
        if names != RX_SHIFT_PAIR_VIEW_NAMES:
            raise ValueError(
                f"rx_shift3 definition drift: {names} != {RX_SHIFT_PAIR_VIEW_NAMES}"
            )
        scenario_views.append(
            {
                name: _tensor_to_numpy_compat(
                    value, dtype=np.dtype(np.float32)
                )
                for name, value in generated
            }
        )
    slot_rows: list[np.ndarray] = []
    slot_labels: list[np.ndarray] = []
    schedule: list[dict[str, Any]] = []
    for slot_index in range(RX_SHIFT_PAIR_SLOT_COUNT):
        scenario_index = int(slot_index % len(SCENARIOS))
        perturbation = RX_SHIFT_PAIR_VIEW_NAMES[1 + (slot_index % 2)]
        views = scenario_views[scenario_index]
        slot_rows.append(
            np.concatenate([views["rx_base"], views[perturbation]], axis=0)
        )
        slot_labels.append(np.concatenate([base_labels, base_labels], axis=0))
        schedule.append(
            {
                "epoch_slot": int(slot_index + 1),
                "scenario": str(SCENARIOS[scenario_index]),
                "receive_views": ["rx_base", perturbation],
            }
        )
    expanded_rows = np.concatenate(slot_rows, axis=0).astype(np.float32)
    expanded_labels = np.concatenate(slot_labels, axis=0).astype(np.int64)
    expected = int(RX_SHIFT_PAIR_SLOT_COUNT * 2 * physical_count)
    if int(len(expanded_rows)) != expected or int(len(expanded_labels)) != expected:
        raise RuntimeError("rx_shift_pair_cycle expansion size mismatch")
    return expanded_rows, expanded_labels, {
        "policy": "rx_shift_pair_cycle_v1",
        "input_formal_scenario_count": int(input_view_count),
        "physical_support_count": int(physical_count),
        "epoch_slot_count": int(RX_SHIFT_PAIR_SLOT_COUNT),
        "receive_views_per_physical_sample_per_epoch": 2,
        "unique_receive_view_names": list(RX_SHIFT_PAIR_VIEW_NAMES),
        "unique_receive_view_count": len(RX_SHIFT_PAIR_VIEW_NAMES),
        "support_receive_view_cap": 3,
        "schedule": schedule,
        "expanded_training_rows": int(expected),
    }


class LoRALinear(nn.Module):
    """Frozen Linear plus an identity-initialized low-rank residual branch."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        if int(rank) <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False).to(
            device=base.weight.device, dtype=base.weight.dtype
        )
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False).to(
            device=base.weight.device, dtype=base.weight.dtype
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.base(rows) + self.scaling * self.lora_b(self.lora_a(rows))

    @property
    def trainable_parameter_count(self) -> int:
        return int(self.lora_a.weight.numel() + self.lora_b.weight.numel())

    @property
    def added_macs_per_sample(self) -> int:
        return self.trainable_parameter_count


class ChannelAffineLinear(nn.Module):
    """Frozen Linear followed by identity-initialized channel-wise FiLM."""

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.film_scale = nn.Parameter(torch.zeros(base.out_features))
        self.film_bias = nn.Parameter(torch.zeros(base.out_features))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        base_rows = self.base(rows)
        return base_rows * (1.0 + self.film_scale) + self.film_bias

    @property
    def trainable_parameter_count(self) -> int:
        return int(self.film_scale.numel() + self.film_bias.numel())

    @property
    def added_macs_per_sample(self) -> int:
        return self.trainable_parameter_count


def _resolve_parent(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def inject_feat_joint_lora(
    model: nn.Module, *, rank: int, alpha: float, scope: str = "feat_joint"
) -> dict[str, Any]:
    scope_norm = str(scope).strip().lower()
    if scope_norm == "feat_joint":
        target_names = LORA_TARGETS
    elif scope_norm == "late_feat_joint":
        target_names = LATE_LORA_TARGETS
    elif scope_norm == "full_feature":
        target_names = FULL_FEATURE_LORA_TARGETS
    elif scope_norm == "effective_feature":
        target_names = EFFECTIVE_FEATURE_LORA_TARGETS
    elif scope_norm == "projection_feature":
        target_names = PROJECTION_LORA_TARGETS
    elif scope_norm == "joint_projection":
        target_names = JOINT_PROJECTION_LORA_TARGETS
    elif scope_norm == "joint_gate":
        target_names = JOINT_GATE_LORA_TARGETS
    else:
        raise ValueError(f"unsupported LoRA scope: {scope}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = dict(model.named_modules())
    injected: list[dict[str, Any]] = []
    for name in target_names:
        original = modules.get(name)
        if not isinstance(original, nn.Linear):
            raise TypeError(f"required feat_joint Linear is missing: {name}")
        replacement = LoRALinear(original, rank=int(rank), alpha=float(alpha))
        parent, leaf = _resolve_parent(model, name)
        if leaf.isdigit():
            parent[int(leaf)] = replacement
        else:
            setattr(parent, leaf, replacement)
        injected.append(
            {
                "module": name,
                "in_features": int(original.in_features),
                "out_features": int(original.out_features),
                "rank": int(rank),
                "trainable_parameters": replacement.trainable_parameter_count,
                "added_macs_per_query": replacement.added_macs_per_sample,
            }
        )
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    unexpected = [name for name, _ in trainable if ".lora_" not in name]
    if unexpected:
        raise RuntimeError(f"non-LoRA checkpoint parameters became trainable: {unexpected}")
    parameter_count = int(sum(parameter.numel() for _, parameter in trainable))
    fp16_bytes = int(parameter_count * 2)
    macs = int(sum(row["added_macs_per_query"] for row in injected))
    audit = {
        "adapter_type": f"{scope_norm}_lora",
        "scope": scope_norm,
        "target_modules": injected,
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_parameters": parameter_count,
        "adapter_state_bytes_fp16": fp16_bytes,
        "adapter_state_bytes_fp32": int(parameter_count * 4),
        "adapter_macs_per_query": macs,
        "query_view_count": 1,
        "original_checkpoint_trainable_parameters": 0,
        "original_checkpoint_gradient_updates": 0,
        "full_model_finetune": False,
    }
    parameter_cap = 100_000 if scope_norm == "full_feature" else 50_000
    audit["resource_tier"] = (
        "performance_relaxed" if scope_norm == "full_feature" else "preferred"
    )
    audit["trainable_parameter_cap"] = int(parameter_cap)
    audit["persistent_state_cap_bytes"] = int(256 * 1024)
    if parameter_count > parameter_cap:
        raise ValueError(
            f"LoRA exceeds {parameter_cap} parameter cap for {scope_norm}: {audit}"
        )
    if fp16_bytes > 256 * 1024:
        raise ValueError(f"LoRA exceeds 256KB state cap: {audit}")
    return audit


@torch.no_grad()
def merge_feat_joint_lora(model: nn.Module) -> dict[str, Any]:
    """Merge every LoRA residual into its frozen Linear and remove wrappers."""

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]
    if not targets:
        raise ValueError("model contains no LoRALinear modules to merge")
    parity: list[dict[str, Any]] = []
    for name, module in targets:
        base = module.base
        probe = torch.linspace(
            -0.5,
            0.5,
            steps=max(2, 3 * int(base.in_features)),
            device=base.weight.device,
            dtype=base.weight.dtype,
        ).reshape(3, int(base.in_features))
        expected = module(probe)
        merged = nn.Linear(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        delta = module.scaling * (module.lora_b.weight @ module.lora_a.weight)
        merged.weight.copy_(base.weight + delta.to(base.weight.dtype))
        if base.bias is not None:
            merged.bias.copy_(base.bias)
        for parameter in merged.parameters():
            parameter.requires_grad_(False)
        parent, leaf = _resolve_parent(model, name)
        if leaf.isdigit():
            parent[int(leaf)] = merged
        else:
            setattr(parent, leaf, merged)
        actual = merged(probe)
        max_abs = float(torch.max(torch.abs(expected - actual)).item())
        parity.append({"module": name, "max_absolute_difference": max_abs})
    remaining = [
        name for name, module in model.named_modules() if isinstance(module, LoRALinear)
    ]
    max_difference = max(row["max_absolute_difference"] for row in parity)
    if remaining or not math.isfinite(max_difference) or max_difference > 1.0e-5:
        raise RuntimeError(
            f"LoRA merge parity failed: remaining={remaining}, max_diff={max_difference}"
        )
    return {
        "merged_module_count": int(len(parity)),
        "merged_modules": parity,
        "remaining_lora_wrappers": remaining,
        "post_merge_trainable_parameters": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "max_absolute_difference": max_difference,
        "algebraic_probe_parity_pass": True,
        "deployment_added_macs_per_view_after_merge": 0,
    }


def inject_late_channel_film(model: nn.Module) -> dict[str, Any]:
    """Attach four late pooled/projection FiLM blocks and freeze the checkpoint."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = dict(model.named_modules())
    injected: list[dict[str, Any]] = []
    for name in LATE_FILM_TARGETS:
        original = modules.get(name)
        if not isinstance(original, nn.Linear):
            raise TypeError(f"required late FiLM Linear is missing: {name}")
        replacement = ChannelAffineLinear(original)
        parent, leaf = _resolve_parent(model, name)
        if leaf.isdigit():
            parent[int(leaf)] = replacement
        else:
            setattr(parent, leaf, replacement)
        injected.append(
            {
                "module": name,
                "in_features": int(original.in_features),
                "out_features": int(original.out_features),
                "trainable_parameters": replacement.trainable_parameter_count,
                "added_macs_per_query": replacement.added_macs_per_sample,
            }
        )
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    unexpected = [
        name
        for name, _ in trainable
        if not (name.endswith(".film_scale") or name.endswith(".film_bias"))
    ]
    if unexpected:
        raise RuntimeError(
            f"non-FiLM checkpoint parameters became trainable: {unexpected}"
        )
    parameter_count = int(sum(parameter.numel() for _, parameter in trainable))
    fp16_bytes = int(parameter_count * 2)
    macs = int(sum(row["added_macs_per_query"] for row in injected))
    audit = {
        "adapter_type": "late_channel_film",
        "scope": "late_pooled_projection",
        "target_modules": injected,
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_parameters": parameter_count,
        "adapter_state_bytes_fp16": fp16_bytes,
        "adapter_state_bytes_fp32": int(parameter_count * 4),
        "adapter_macs_per_query": macs,
        "query_view_count": 1,
        "original_checkpoint_trainable_parameters": 0,
        "original_checkpoint_gradient_updates": 0,
    }
    if parameter_count > 4_096:
        raise ValueError(f"late FiLM exceeds 4,096-parameter preferred cap: {audit}")
    if fp16_bytes > 64 * 1024:
        raise ValueError(f"late FiLM exceeds 64KiB preferred state cap: {audit}")
    return audit


def enable_late_key_layer_finetune(model: nn.Module) -> dict[str, Any]:
    """Enable the exact <=50k ADV3B02 late-layer update whitelist."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = dict(model.named_modules())
    enabled: list[dict[str, Any]] = []
    allowed_parameter_names: set[str] = set()
    existing_macs = 0
    for name in LATE_KEY_FT_TARGETS:
        layer = modules.get(name)
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"required late key Linear is missing: {name}")
        layer.weight.requires_grad_(True)
        allowed_parameter_names.add(f"{name}.weight")
        parameter_count = int(layer.weight.numel())
        if layer.bias is not None:
            layer.bias.requires_grad_(True)
            allowed_parameter_names.add(f"{name}.bias")
            parameter_count += int(layer.bias.numel())
        layer_macs = int(layer.in_features * layer.out_features)
        existing_macs += layer_macs
        enabled.append(
            {
                "module": name,
                "in_features": int(layer.in_features),
                "out_features": int(layer.out_features),
                "updated_checkpoint_parameters": int(parameter_count),
                "existing_layer_macs_per_query": int(layer_macs),
                "added_macs_per_query_after_merge": 0,
            }
        )
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(trainable) != allowed_parameter_names:
        raise RuntimeError(
            "late key-layer trainable whitelist mismatch: "
            f"observed={sorted(trainable)}, expected={sorted(allowed_parameter_names)}"
        )
    parameter_count = int(sum(parameter.numel() for parameter in trainable.values()))
    if parameter_count != 31_200 or parameter_count > 50_000:
        raise ValueError(f"late key-layer parameter budget mismatch: {parameter_count}")
    fp16_bytes = int(parameter_count * 2)
    return {
        "adapter_type": "late_key_layer_delta",
        "scope": "preregistered_sparse_checkpoint_delta",
        "target_modules": enabled,
        "checkpoint_update_target_modules": list(LATE_KEY_FT_TARGETS),
        "trainable_parameter_names": sorted(trainable),
        "trainable_parameters": int(parameter_count),
        "updated_checkpoint_parameters": int(parameter_count),
        "adapter_state_bytes_fp16": int(fp16_bytes),
        "adapter_state_bytes_fp32": int(parameter_count * 4),
        "delta_patch_state_bytes_fp16": int(fp16_bytes),
        "adapter_macs_per_query": 0,
        "updated_existing_layer_macs_per_query": int(existing_macs),
        "deployment_added_macs_per_query_after_merge": 0,
        "query_view_count": 1,
        "original_checkpoint_trainable_parameters": int(parameter_count),
        "original_checkpoint_gradient_updates": 0,
        "full_model_finetune": False,
        "exact_layer_whitelist_enforced": True,
    }


def load_trainable_adapter_state(
    model: nn.Module, state_path: Path
) -> dict[str, Any]:
    """Strictly load a ground-trained state into the currently injected adapter."""

    payload = torch.load(state_path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError("adapter state must be a tensor mapping")
    expected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    observed = {str(name): value for name, value in payload.items()}
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "adapter state key mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    tensor_count = 0
    element_count = 0
    l2_sq = 0.0
    with torch.no_grad():
        for name, parameter in expected.items():
            value = observed[name]
            if not torch.is_tensor(value):
                raise TypeError(f"adapter state value is not a tensor: {name}")
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"adapter state shape mismatch for {name}: "
                    f"{tuple(value.shape)} != {tuple(parameter.shape)}"
                )
            value_float = value.detach().float()
            if not bool(torch.isfinite(value_float).all()):
                raise FloatingPointError(f"non-finite adapter state: {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            tensor_count += 1
            element_count += int(value.numel())
            l2_sq += float(torch.sum(value_float.square()))
    return {
        "mode": "ground_source_pretrained",
        "path": str(state_path),
        "sha256": _sha256_file(state_path),
        "tensor_count": int(tensor_count),
        "element_count": int(element_count),
        "l2_norm": float(math.sqrt(l2_sq)),
        "strict_key_match": True,
        "finite": True,
    }


def load_and_merge_ground_lora(
    model: nn.Module,
    state_path: Path,
    *,
    scope: str,
    rank: int,
    alpha: float,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Load a ground LoRA exactly, merge it, then refreeze the base model."""

    state_path = Path(state_path)
    observed_sha256 = _sha256_file(state_path)
    if expected_sha256 is not None:
        expected_sha256 = str(expected_sha256).strip().lower()
        if len(expected_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in expected_sha256
        ):
            raise ValueError("ground adapter expected SHA256 must be 64 hex characters")
        if observed_sha256 != expected_sha256:
            raise ValueError(
                "ground adapter SHA256 mismatch before load: "
                f"{observed_sha256} != {expected_sha256}"
            )
    resources = inject_feat_joint_lora(
        model, rank=int(rank), alpha=float(alpha), scope=str(scope)
    )
    initialization = load_trainable_adapter_state(model, state_path)
    merge = merge_feat_joint_lora(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {
        "mode": "ground_lora_loaded_and_merged",
        "scope": str(scope),
        "rank": int(rank),
        "alpha": float(alpha),
        "resources": resources,
        "initialization": initialization,
        "merge": merge,
        "serialized_file_bytes": int(state_path.stat().st_size),
        "expected_sha256": expected_sha256,
        "observed_sha256_before_load": observed_sha256,
        "sha256_preload_match": expected_sha256 is None
        or observed_sha256 == expected_sha256,
        "deployment_added_macs_per_query_after_merge": 0,
    }


def roundtrip_fp16_target_lora_and_merge(
    model: nn.Module, state_path: Path
) -> dict[str, Any]:
    """Reload the persisted FP16 target patch before deployment merge/export."""

    state_path = Path(state_path)
    roundtrip = load_trainable_adapter_state(model, state_path)
    merge = merge_feat_joint_lora(model)
    return {
        "mode": "fp16_artifact_roundtrip_then_merge",
        "state_roundtrip": roundtrip,
        "merge": merge,
        "artifact_sha256": _sha256_file(state_path),
        "deployment_added_macs_per_query_after_merge": 0,
    }


def _prototype_banks_from_matched_views(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_count: int,
    view_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all-view and leave-one-view-out class prototypes."""
    if int(view_count) <= 1 or int(features.shape[0]) % int(view_count) != 0:
        raise ValueError("support features cannot be grouped into matched views")
    physical_count = int(features.shape[0]) // int(view_count)
    for view_index in range(1, int(view_count)):
        if not torch.equal(
            labels[:physical_count],
            labels[view_index * physical_count : (view_index + 1) * physical_count],
        ):
            raise ValueError("support labels are not matched across views")
    normalized = _norm_rows(features)
    view_ids = torch.arange(len(labels), device=labels.device) // physical_count
    all_view: list[torch.Tensor] = []
    leave_one_out: list[torch.Tensor] = []
    for class_index in range(int(class_count)):
        class_mask = labels == class_index
        if not bool(class_mask.any()):
            raise ValueError(f"support class {class_index} is empty")
        all_view.append(_norm_rows(normalized[class_mask].mean(dim=0, keepdim=True))[0])
    for held_out_view in range(int(view_count)):
        per_class: list[torch.Tensor] = []
        for class_index in range(int(class_count)):
            mask = (labels == class_index) & (view_ids != held_out_view)
            if not bool(mask.any()):
                raise ValueError(
                    f"support class {class_index} is empty outside view {held_out_view}"
                )
            per_class.append(
                _norm_rows(normalized[mask].mean(dim=0, keepdim=True))[0]
            )
        leave_one_out.append(torch.stack(per_class, dim=0))
    return torch.stack(all_view, dim=0).detach(), torch.stack(leave_one_out, dim=0).detach()


def view_score_distillation_loss(
    scores: torch.Tensor, *, view_count: int, temperature: float
) -> torch.Tensor:
    """Distill the detached multiView score ensemble into every support View."""

    if int(view_count) <= 1:
        raise ValueError("view score distillation requires at least two views")
    if scores.ndim != 2 or int(scores.shape[0]) % int(view_count) != 0:
        raise ValueError("scores cannot be grouped into matched views")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("view score distillation temperature must be positive")
    grouped = scores.reshape(int(view_count), -1, int(scores.shape[1]))
    softened_teacher = torch.softmax(
        grouped.detach().mean(dim=0) / float(temperature), dim=-1
    )
    student_log_probs = torch.log_softmax(grouped / float(temperature), dim=-1)
    targets = softened_teacher.unsqueeze(0).expand_as(student_log_probs)
    return F.kl_div(
        student_log_probs, targets, reduction="batchmean"
    ) * float(temperature) ** 2


def _matched_view_support_layout(
    labels: torch.Tensor, *, view_count: int
) -> tuple[int, int, int, list[torch.Tensor]]:
    """Validate view-major support rows and return physical class positions."""

    if labels.ndim != 1 or int(view_count) <= 1:
        raise ValueError("BP-JG requires one-dimensional labels and at least two views")
    if int(labels.numel()) % int(view_count) != 0:
        raise ValueError("support labels cannot be grouped into matched views")
    physical_count = int(labels.numel()) // int(view_count)
    physical_labels = labels[:physical_count]
    for view_index in range(1, int(view_count)):
        current = labels[
            view_index * physical_count : (view_index + 1) * physical_count
        ]
        if not torch.equal(physical_labels, current):
            raise ValueError("support labels are not matched across views")
    unique = torch.unique(physical_labels, sorted=True)
    if unique.numel() < 2 or not torch.equal(
        unique, torch.arange(int(unique.numel()), device=labels.device)
    ):
        raise ValueError("support classes must be contiguous and include at least two classes")
    class_positions = [
        torch.nonzero(physical_labels == class_index, as_tuple=False).flatten()
        for class_index in range(int(unique.numel()))
    ]
    counts = {int(positions.numel()) for positions in class_positions}
    if len(counts) != 1 or min(counts) <= 0:
        raise ValueError("each registered class must have the same positive K-shot count")
    k_shot = int(next(iter(counts)))
    return physical_count, int(unique.numel()), k_shot, class_positions


def build_shot_index_episode_positions(
    labels: torch.Tensor,
    *,
    view_count: int,
    max_episodes_per_epoch: int = 10,
) -> list[torch.Tensor]:
    """Group all registered classes by shot index without role or class quotas."""

    if int(max_episodes_per_epoch) <= 0:
        raise ValueError("max_episodes_per_epoch must be positive")
    physical_count, class_count, k_shot, class_positions = (
        _matched_view_support_layout(labels, view_count=int(view_count))
    )
    episode_count = min(int(k_shot), int(max_episodes_per_epoch))
    shot_groups = np.array_split(np.arange(k_shot, dtype=np.int64), episode_count)
    episodes: list[torch.Tensor] = []
    for shot_group in shot_groups:
        shot_tensor = torch.as_tensor(
            shot_group, device=labels.device, dtype=torch.long
        )
        physical = torch.cat(
            [class_positions[class_index][shot_tensor] for class_index in range(class_count)]
        )
        episodes.append(
            torch.cat(
                [
                    physical + view_index * physical_count
                    for view_index in range(int(view_count))
                ]
            )
        )
    covered = torch.sort(
        torch.cat([episode[: int(episode.numel()) // int(view_count)] for episode in episodes])
    ).values
    if not torch.equal(
        covered, torch.arange(physical_count, device=labels.device)
    ):
        raise RuntimeError("shot-index episodes do not cover every physical support exactly once")
    return episodes


def bp_jg_episode_loss(
    features: torch.Tensor,
    base_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    view_count: int,
    temperature: float = 18.0,
    boundary_margin_delta: float = 0.02,
    boundary_weight: float = 2.0,
    anchor_weight: float = 0.5,
    gram_weight: float = 0.5,
    separation_weight: float = 0.25,
    view_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Boundary-preserving, class-symmetric loss on registered support only."""

    if features.shape != base_features.shape or features.ndim != 2:
        raise ValueError("adapted and base episode features must have the same [N,D] shape")
    if int(features.shape[0]) != int(labels.numel()):
        raise ValueError("episode feature and label counts differ")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be positive and finite")
    if not 0.0 <= float(boundary_margin_delta) <= 0.5:
        raise ValueError("boundary_margin_delta must be in [0,0.5]")
    physical_count, class_count, _, _ = _matched_view_support_layout(
        labels, view_count=int(view_count)
    )
    z = _norm_rows(features).reshape(int(view_count), physical_count, -1)
    z0 = _norm_rows(base_features.detach()).reshape(
        int(view_count), physical_count, -1
    )
    physical_labels = labels[:physical_count]

    def class_prototypes(rows: torch.Tensor, row_labels: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                _norm_rows(
                    rows[row_labels == class_index].mean(dim=0, keepdim=True)
                )[0]
                for class_index in range(class_count)
            ],
            dim=0,
        )

    cross_scores: list[torch.Tensor] = []
    base_cross_scores: list[torch.Tensor] = []
    for held_out_view in range(int(view_count)):
        kept = [index for index in range(int(view_count)) if index != held_out_view]
        prototype_rows = torch.cat([z[index] for index in kept], dim=0)
        base_prototype_rows = torch.cat([z0[index] for index in kept], dim=0)
        prototype_labels = physical_labels.repeat(len(kept))
        prototypes = class_prototypes(prototype_rows, prototype_labels)
        base_prototypes = class_prototypes(base_prototype_rows, prototype_labels)
        cross_scores.append(z[held_out_view] @ prototypes.T)
        base_cross_scores.append(z0[held_out_view] @ base_prototypes.T)
    cosine_scores = torch.cat(cross_scores, dim=0)
    base_cosine_scores = torch.cat(base_cross_scores, dim=0).detach()
    targets = physical_labels.repeat(int(view_count))
    xview_ce = F.cross_entropy(float(temperature) * cosine_scores, targets)

    row_index = torch.arange(int(targets.numel()), device=targets.device)
    true_scores = cosine_scores[row_index, targets]
    base_true_scores = base_cosine_scores[row_index, targets]
    class_mask = F.one_hot(targets, num_classes=class_count).bool()
    negative_scores = cosine_scores.masked_fill(class_mask, float("-inf")).max(dim=1).values
    base_negative_scores = base_cosine_scores.masked_fill(
        class_mask, float("-inf")
    ).max(dim=1).values
    margin = true_scores - negative_scores
    base_margin = base_true_scores - base_negative_scores
    boundary = F.relu(base_margin + float(boundary_margin_delta) - margin).mean()
    anchor = (1.0 - torch.sum(z * z0, dim=-1)).mean()

    all_rows = z.reshape(-1, int(z.shape[-1]))
    base_all_rows = z0.reshape(-1, int(z0.shape[-1]))
    all_labels = physical_labels.repeat(int(view_count))
    prototypes = class_prototypes(all_rows, all_labels)
    base_prototypes = class_prototypes(base_all_rows, all_labels).detach()
    gram = prototypes @ prototypes.T
    base_gram = base_prototypes @ base_prototypes.T
    off_diagonal = ~torch.eye(class_count, device=gram.device, dtype=torch.bool)
    gram_preservation = F.smooth_l1_loss(
        gram[off_diagonal], base_gram[off_diagonal]
    )
    separation_cap = torch.minimum(base_gram, torch.full_like(base_gram, 0.65))
    separation = F.relu(gram[off_diagonal] - separation_cap[off_diagonal]).square().mean()
    view_center = _norm_rows(z.mean(dim=0))
    view_consistency = (1.0 - torch.sum(z * view_center.unsqueeze(0), dim=-1)).mean()
    loss = (
        xview_ce
        + float(boundary_weight) * boundary
        + float(anchor_weight) * anchor
        + float(gram_weight) * gram_preservation
        + float(separation_weight) * separation
        + float(view_weight) * view_consistency
    )
    return {
        "loss": loss,
        "xview_prototype_ce": xview_ce,
        "boundary_margin_loss": boundary,
        "feature_anchor_loss": anchor,
        "prototype_gram_loss": gram_preservation,
        "prototype_separation_loss": separation,
        "view_consistency_loss": view_consistency,
        "mean_margin": margin.mean(),
        "mean_base_margin": base_margin.mean(),
        "correct": (cosine_scores.argmax(dim=1) == targets).sum(),
        "sample_count": torch.as_tensor(
            int(targets.numel()), device=features.device, dtype=torch.long
        ),
    }


def train_support_only_bp_jg(
    model: nn.Module,
    support_rows: np.ndarray,
    support_labels: np.ndarray,
    *,
    physical_support_ids: Sequence[str],
    support_row_physical_ids: Sequence[str],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    support_view_count: int,
    batch_size: int,
    max_optimizer_steps: int,
    grad_clip: float,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train only the target JG LoRA with at most ten shot episodes per epoch."""

    if not 1 <= int(epochs) <= 5:
        raise ValueError("BP-JG adaptation epochs must be in [1,5]")
    if not 1 <= int(max_optimizer_steps) <= 50:
        raise ValueError("BP-JG max_optimizer_steps must be in [1,50]")
    if float(grad_clip) <= 0.0:
        raise ValueError("grad_clip must be positive")
    rows = _numpy_to_tensor_compat(
        support_rows, numpy_dtype=np.dtype(np.float32), torch_dtype=torch.float32
    ).to(device)
    labels = _numpy_to_tensor_compat(
        support_labels, numpy_dtype=np.dtype(np.int64), torch_dtype=torch.int64
    ).to(device)
    physical_count, class_count, k_shot, _ = _matched_view_support_layout(
        labels, view_count=int(support_view_count)
    )
    support_ids = [str(value) for value in physical_support_ids]
    if len(support_ids) != int(physical_count):
        raise ValueError(
            "physical support ID count mismatch: "
            f"{len(support_ids)} != {physical_count}"
        )
    if any(not value for value in support_ids) or len(set(support_ids)) != len(
        support_ids
    ):
        raise ValueError("physical support IDs must be non-empty and unique")
    support_ids_sha256 = hashlib.sha256(
        "\n".join(support_ids).encode("utf-8")
    ).hexdigest()
    support_row_ids = [str(value) for value in support_row_physical_ids]
    if len(support_row_ids) != int(labels.numel()):
        raise ValueError(
            "support row physical ID count mismatch: "
            f"{len(support_row_ids)} != {int(labels.numel())}"
        )
    for view_index in range(int(support_view_count)):
        start = int(view_index * physical_count)
        stop = int(start + physical_count)
        if support_row_ids[start:stop] != support_ids:
            raise ValueError(
                f"support row physical ID alignment drift for view {view_index}"
            )
    support_row_ids_sha256 = hashlib.sha256(
        "\n".join(support_row_ids).encode("utf-8")
    ).hexdigest()
    max_episodes_per_epoch = max(1, int(max_optimizer_steps) // int(epochs))
    episodes = build_shot_index_episode_positions(
        labels,
        view_count=int(support_view_count),
        max_episodes_per_epoch=max_episodes_per_epoch,
    )
    identity = nn.Identity()
    model.eval()
    with torch.no_grad():
        base_features, _, _ = _batched_feature_forward(
            model, identity, rows, batch_size=int(batch_size), require_grad=False
        )
        base_features = _norm_rows(base_features).detach()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("BP-JG injection selected no trainable parameters")
    optimizer = torch.optim.SGD(
        parameters,
        lr=float(learning_rate),
        momentum=0.0,
        weight_decay=float(weight_decay),
    )
    rng = np.random.default_rng(int(seed))
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    optimizer_steps = 0
    forward_sample_equivalents = int(rows.shape[0])
    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        totals = {
            "loss": 0.0,
            "xview_prototype_ce": 0.0,
            "boundary_margin_loss": 0.0,
            "feature_anchor_loss": 0.0,
            "prototype_gram_loss": 0.0,
            "prototype_separation_loss": 0.0,
            "view_consistency_loss": 0.0,
            "mean_margin": 0.0,
            "mean_base_margin": 0.0,
            "correct": 0.0,
            "grad": 0.0,
        }
        seen = 0
        batches = 0
        for episode_index in rng.permutation(len(episodes)):
            if optimizer_steps >= int(max_optimizer_steps):
                break
            positions = episodes[int(episode_index)]
            optimizer.zero_grad(set_to_none=True)
            z, _ = _feature_forward(model, rows[positions])
            losses = bp_jg_episode_loss(
                z,
                base_features[positions],
                labels[positions],
                view_count=int(support_view_count),
                temperature=float(temperature),
            )
            losses["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(grad_clip))
            optimizer.step()
            optimizer_steps += 1
            count = int(losses["sample_count"].detach())
            forward_sample_equivalents += int(positions.numel())
            seen += count
            batches += 1
            for key in (
                "loss",
                "xview_prototype_ce",
                "boundary_margin_loss",
                "feature_anchor_loss",
                "prototype_gram_loss",
                "prototype_separation_loss",
                "view_consistency_loss",
                "mean_margin",
                "mean_base_margin",
            ):
                totals[key] += float(losses[key].detach()) * count
            totals["correct"] += float(losses["correct"].detach())
            totals["grad"] += float(grad_norm.detach())
        row = {
            "epoch": int(epoch),
            **{
                key: totals[key] / max(1, seen)
                for key in (
                    "loss",
                    "xview_prototype_ce",
                    "boundary_margin_loss",
                    "feature_anchor_loss",
                    "prototype_gram_loss",
                    "prototype_separation_loss",
                    "view_consistency_loss",
                    "mean_margin",
                    "mean_base_margin",
                )
            },
            "support_train_acc": totals["correct"] / max(1, seen),
            "gradient_norm": totals["grad"] / max(1, batches),
            "optimizer_steps": int(optimizer_steps),
            "episode_count": int(batches),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"non-finite BP-JG trace: {row}")
        trace.append(row)
        print("[BP-JG-EPOCH] " + json.dumps(row, sort_keys=True), flush=True)
        if optimizer_steps >= int(max_optimizer_steps):
            break
    expected_natural_steps = int(epochs) * len(episodes)
    runtime = {
        "adaptation_wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "optimizer_state_deployment_required": False,
        "optimizer": "sgd",
        "optimizer_momentum": 0.0,
        "optimizer_steps": int(optimizer_steps),
        "max_optimizer_steps": int(max_optimizer_steps),
        "optimizer_training_state_bytes_estimate": 0,
        "support_view_count": int(support_view_count),
        "physical_support_count": int(physical_count),
        "physical_support_ids_sha256": support_ids_sha256,
        "physical_support_ids_unique": True,
        "support_row_physical_ids_sha256": support_row_ids_sha256,
        "matched_view_physical_id_order": True,
        "physical_ids_used_for_layout_validation_only": True,
        "train_receive_views_per_physical_sample_per_epoch": int(
            support_view_count
        ),
        "registered_class_count": int(class_count),
        "k_shot_inferred": int(k_shot),
        "episodes_per_epoch": int(len(episodes)),
        "max_episodes_per_epoch": int(max_episodes_per_epoch),
        "shot_indices_consumed_per_epoch": int(k_shot),
        "support_forward_sample_equivalents": int(forward_sample_equivalents),
        "terminated_by_step_cap": int(optimizer_steps) < int(expected_natural_steps),
        "query_rows_used_for_training": 0,
        "old_new_role_used_by_optimizer": False,
        "class_quota_used_by_optimizer": False,
        "dense_query_graph_used": False,
    }
    return trace, runtime


def train_support_only_lora(
    model: nn.Module,
    support_rows: np.ndarray,
    support_labels: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    feature_anchor_weight: float,
    view_consistency_weight: float,
    cross_view_prototype_weight: float,
    view_score_distill_weight: float,
    view_score_distill_temperature: float,
    cosine_margin: float,
    class_dro_temperature: float,
    support_view_count: int,
    batch_size: int,
    optimizer_name: str = "adamw",
    max_optimizer_steps: int = 0,
    grad_clip: float = 5.0,
    view_sampling_mode: str = "stacked",
    matched_view_teacher_weight: float = 0.0,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if int(epochs) <= 0 or int(epochs) > 20:
        raise ValueError("formal extreme-light adaptation epochs must be in [1,20]")
    if not 0.0 <= float(cosine_margin) <= 0.5:
        raise ValueError("cosine_margin must be in [0,0.5]")
    if float(class_dro_temperature) < 0.0:
        raise ValueError("class_dro_temperature must be nonnegative")
    if not 0.0 <= float(cross_view_prototype_weight) <= 1.0:
        raise ValueError("cross_view_prototype_weight must be in [0,1]")
    if float(view_score_distill_weight) < 0.0:
        raise ValueError("view_score_distill_weight must be nonnegative")
    if (
        not math.isfinite(float(view_score_distill_temperature))
        or float(view_score_distill_temperature) <= 0.0
    ):
        raise ValueError("view_score_distill_temperature must be positive")
    if float(matched_view_teacher_weight) < 0.0:
        raise ValueError("matched_view_teacher_weight must be nonnegative")
    if float(grad_clip) <= 0.0:
        raise ValueError("grad_clip must be positive")
    optimizer_name_norm = str(optimizer_name).strip().lower()
    if optimizer_name_norm not in {"adamw", "sgd"}:
        raise ValueError("optimizer_name must be adamw or sgd")
    view_sampling_mode_norm = str(view_sampling_mode).strip().lower()
    if view_sampling_mode_norm not in {"stacked", "rotating_single"}:
        raise ValueError("view_sampling_mode must be stacked or rotating_single")
    if int(max_optimizer_steps) < 0:
        raise ValueError("max_optimizer_steps must be nonnegative")
    if view_sampling_mode_norm == "rotating_single" and (
        float(view_consistency_weight) > 0.0
        or float(cross_view_prototype_weight) > 0.0
        or float(view_score_distill_weight) > 0.0
    ):
        raise ValueError(
            "rotating_single uses cached matched-view teacher targets; "
            "stacked view losses must be disabled"
        )
    rows = _numpy_to_tensor_compat(
        support_rows, numpy_dtype=np.dtype(np.float32), torch_dtype=torch.float32
    ).to(device)
    labels = _numpy_to_tensor_compat(
        support_labels, numpy_dtype=np.dtype(np.int64), torch_dtype=torch.int64
    ).to(device)
    class_count = int(labels.max().item()) + 1
    identity = nn.Identity()
    model.eval()
    with torch.no_grad():
        base_features, _, _ = _batched_feature_forward(
            model, identity, rows, batch_size=int(batch_size), require_grad=False
        )
        base_features = _norm_rows(base_features).detach()
    view_count = int(support_view_count)
    physical_count = 0
    matched_view_teacher = None
    if view_sampling_mode_norm == "rotating_single" or float(
        matched_view_teacher_weight
    ) > 0.0:
        if view_count <= 1 or int(rows.shape[0]) % view_count != 0:
            raise ValueError("support rows cannot be grouped into matched views")
        physical_count = int(rows.shape[0]) // view_count
        for view_index in range(1, view_count):
            if not torch.equal(
                labels[:physical_count],
                labels[
                    view_index * physical_count : (view_index + 1) * physical_count
                ],
            ):
                raise ValueError("support labels are not matched across views")
        matched_view_teacher = _norm_rows(
            base_features.reshape(view_count, physical_count, -1).mean(dim=0)
        ).detach()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("adapter injection selected no trainable parameters")
    if optimizer_name_norm == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=float(learning_rate),
            momentum=0.0,
            weight_decay=float(weight_decay),
        )
        optimizer_training_state_bytes = 0
    else:
        optimizer = torch.optim.AdamW(
            parameters, lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        optimizer_training_state_bytes = int(
            2 * sum(parameter.numel() for parameter in parameters) * 4
        )
    rng = np.random.default_rng(int(seed))
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    optimizer_steps = 0
    forward_sample_equivalents = int(rows.shape[0])
    terminated_by_step_cap = False
    for epoch in range(1, int(epochs) + 1):
        if int(max_optimizer_steps) > 0 and optimizer_steps >= int(
            max_optimizer_steps
        ):
            terminated_by_step_cap = True
            break
        epoch_started = time.perf_counter()
        cross_view_prototypes = None
        train_view_index = None
        if view_sampling_mode_norm == "rotating_single":
            train_view_index = int((epoch - 1) % view_count)
            epoch_start = train_view_index * physical_count
            epoch_stop = epoch_start + physical_count
            prototype_rows = rows[epoch_start:epoch_stop]
            prototype_labels = labels[epoch_start:epoch_stop]
            prototypes = _class_prototypes(
                model,
                identity,
                prototype_rows,
                prototype_labels,
                class_count=class_count,
                batch_size=int(batch_size),
            )
            forward_sample_equivalents += physical_count
        elif float(cross_view_prototype_weight) > 0.0:
            with torch.no_grad():
                prototype_features, _, _ = _batched_feature_forward(
                    model,
                    identity,
                    rows,
                    batch_size=int(batch_size),
                    require_grad=False,
                )
            prototypes, cross_view_prototypes = _prototype_banks_from_matched_views(
                prototype_features,
                labels,
                class_count=class_count,
                view_count=view_count,
            )
            forward_sample_equivalents += int(rows.shape[0])
        else:
            prototypes = _class_prototypes(
                model,
                identity,
                rows,
                labels,
                class_count=class_count,
                batch_size=int(batch_size),
            )
            forward_sample_equivalents += int(rows.shape[0])
        use_view_groups = (
            float(view_consistency_weight) > 0.0
            or float(cross_view_prototype_weight) > 0.0
            or float(view_score_distill_weight) > 0.0
        )
        if view_sampling_mode_norm == "rotating_single":
            physical_order = rng.permutation(physical_count)
            batches_index = [
                physical_order[offset : offset + int(batch_size)]
                + train_view_index * physical_count
                for offset in range(0, physical_count, int(batch_size))
            ]
        elif use_view_groups:
            if view_count <= 1 or int(rows.shape[0]) % view_count != 0:
                raise ValueError("support rows cannot be grouped into matched views")
            physical_count = int(rows.shape[0]) // view_count
            for view_index in range(1, view_count):
                if not torch.equal(
                    labels[:physical_count],
                    labels[view_index * physical_count : (view_index + 1) * physical_count],
                ):
                    raise ValueError("support labels are not matched across views")
            physical_order = rng.permutation(physical_count)
            physical_batch_size = max(1, int(batch_size) // view_count)
            batches_index = []
            for offset in range(0, physical_count, physical_batch_size):
                physical = physical_order[offset : offset + physical_batch_size]
                batches_index.append(
                    np.concatenate(
                        [physical + view_index * physical_count for view_index in range(view_count)]
                    )
                )
        else:
            order = rng.permutation(int(rows.shape[0]))
            batches_index = [
                order[offset : offset + int(batch_size)]
                for offset in range(0, len(order), int(batch_size))
            ]
        totals = {
            "loss": 0.0,
            "ce": 0.0,
            "cross_view_ce": 0.0,
            "dro": 0.0,
            "anchor": 0.0,
            "matched_view_teacher": 0.0,
            "consistency": 0.0,
            "view_score_distill": 0.0,
            "correct": 0.0,
            "grad": 0.0,
        }
        seen = 0
        batches = 0
        for batch_positions in batches_index:
            if int(max_optimizer_steps) > 0 and optimizer_steps >= int(
                max_optimizer_steps
            ):
                terminated_by_step_cap = True
                break
            positions = torch.as_tensor(
                batch_positions, device=device, dtype=torch.long
            )
            optimizer.zero_grad(set_to_none=True)
            z, _ = _feature_forward(model, rows[positions])
            z = _norm_rows(z)
            scores = float(temperature) * (z @ prototypes.T)
            targets = labels[positions]
            margin_scores = scores.clone()
            if float(cosine_margin) > 0.0:
                margin_scores[
                    torch.arange(len(positions), device=device), targets
                ] -= float(temperature) * float(cosine_margin)
            per_sample_all_view_ce = F.cross_entropy(
                margin_scores, targets, reduction="none"
            )
            cross_view_ce = z.new_zeros(())
            decision_scores = scores
            if cross_view_prototypes is not None:
                sample_view_ids = torch.div(
                    positions, physical_count, rounding_mode="floor"
                )
                sample_prototypes = cross_view_prototypes[sample_view_ids]
                cross_scores = float(temperature) * torch.einsum(
                    "bd,bcd->bc", z, sample_prototypes
                )
                cross_margin_scores = cross_scores.clone()
                if float(cosine_margin) > 0.0:
                    cross_margin_scores[
                        torch.arange(len(positions), device=device), targets
                    ] -= float(temperature) * float(cosine_margin)
                per_sample_cross_view_ce = F.cross_entropy(
                    cross_margin_scores, targets, reduction="none"
                )
                cross_view_ce = per_sample_cross_view_ce.mean()
                blend = float(cross_view_prototype_weight)
                per_sample_ce = (
                    (1.0 - blend) * per_sample_all_view_ce
                    + blend * per_sample_cross_view_ce
                )
                decision_scores = (1.0 - blend) * scores + blend * cross_scores
            else:
                per_sample_ce = per_sample_all_view_ce
            ce = per_sample_ce.mean()
            if float(class_dro_temperature) > 0.0:
                class_losses = []
                for class_index in range(class_count):
                    class_mask = targets == class_index
                    if bool(class_mask.any()):
                        class_losses.append(per_sample_ce[class_mask].mean())
                stacked_class_losses = torch.stack(class_losses)
                dro_weights = torch.softmax(
                    float(class_dro_temperature) * stacked_class_losses.detach(), dim=0
                )
                dro = torch.sum(dro_weights * stacked_class_losses)
            else:
                dro = ce
            anchor = (1.0 - torch.sum(z * base_features[positions], dim=1)).mean()
            if matched_view_teacher is not None:
                teacher_positions = torch.remainder(positions, physical_count)
                matched_teacher = (
                    1.0
                    - torch.sum(
                        z * matched_view_teacher[teacher_positions], dim=1
                    )
                ).mean()
            else:
                matched_teacher = z.new_zeros(())
            if use_view_groups:
                per_view = z.reshape(view_count, -1, z.shape[-1])
                view_center = _norm_rows(per_view.mean(dim=0))
                consistency = (
                    1.0 - torch.sum(per_view * view_center.unsqueeze(0), dim=-1)
                ).mean()
            else:
                consistency = z.new_zeros(())
            if float(view_score_distill_weight) > 0.0:
                score_distill = view_score_distillation_loss(
                    decision_scores,
                    view_count=view_count,
                    temperature=float(view_score_distill_temperature),
                )
            else:
                score_distill = z.new_zeros(())
            loss = (
                dro
                + float(feature_anchor_weight) * anchor
                + float(matched_view_teacher_weight) * matched_teacher
                + float(view_consistency_weight) * consistency
                + float(view_score_distill_weight) * score_distill
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(grad_clip))
            optimizer.step()
            optimizer_steps += 1
            count = int(positions.numel())
            forward_sample_equivalents += count
            seen += count
            batches += 1
            totals["loss"] += float(loss.detach()) * count
            totals["ce"] += float(ce.detach()) * count
            totals["cross_view_ce"] += float(cross_view_ce.detach()) * count
            totals["dro"] += float(dro.detach()) * count
            totals["anchor"] += float(anchor.detach()) * count
            totals["matched_view_teacher"] += float(matched_teacher.detach()) * count
            totals["consistency"] += float(consistency.detach()) * count
            totals["view_score_distill"] += float(score_distill.detach()) * count
            totals["correct"] += float(
                (decision_scores.argmax(dim=1) == targets).sum().detach()
            )
            totals["grad"] += float(grad_norm.detach())
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(1, seen),
            "prototype_ce": totals["ce"] / max(1, seen),
            "leave_one_view_out_ce": totals["cross_view_ce"] / max(1, seen),
            "class_dro_loss": totals["dro"] / max(1, seen),
            "feature_anchor": totals["anchor"] / max(1, seen),
            "matched_view_teacher_loss": totals["matched_view_teacher"]
            / max(1, seen),
            "view_consistency": totals["consistency"] / max(1, seen),
            "view_score_distillation_loss": totals["view_score_distill"]
            / max(1, seen),
            "support_train_acc": totals["correct"] / max(1, seen),
            "gradient_norm": totals["grad"] / max(1, batches),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimizer_steps": int(optimizer_steps),
            "train_view_index": (
                int(train_view_index) if train_view_index is not None else -1
            ),
            "train_views_per_physical_sample": (
                1 if view_sampling_mode_norm == "rotating_single" else view_count
            ),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"non-finite support-adapter trace: {row}")
        trace.append(row)
        print("[SUPPORT-ADAPTER-EPOCH] " + json.dumps(row, sort_keys=True), flush=True)
        if terminated_by_step_cap:
            break
    runtime = {
        "adaptation_wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "optimizer_state_deployment_required": False,
        "optimizer": optimizer_name_norm,
        "optimizer_steps": int(optimizer_steps),
        "max_optimizer_steps": int(max_optimizer_steps),
        "optimizer_training_state_bytes_estimate": int(
            optimizer_training_state_bytes
        ),
        "view_sampling_mode": view_sampling_mode_norm,
        "teacher_precompute_view_count": (
            view_count if matched_view_teacher is not None else 0
        ),
        "train_views_per_physical_sample_per_epoch": (
            1 if view_sampling_mode_norm == "rotating_single" else view_count
        ),
        "support_forward_sample_equivalents": int(forward_sample_equivalents),
        "terminated_by_step_cap": bool(terminated_by_step_cap),
    }
    return trace, runtime


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--new_count", type=int, choices=(5, 10, 20), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k_shot", type=int, choices=(1, 5, 10, 20), default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--adapter_type",
        choices=("lora", "late_film", "late_key_ft"),
        default="lora",
    )
    parser.add_argument("--rank", type=int, choices=(2, 4, 8, 16, 24), default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument(
        "--scope",
        choices=(
            "feat_joint",
            "late_feat_joint",
            "full_feature",
            "joint_projection",
            "joint_gate",
        ),
        default="feat_joint",
    )
    parser.add_argument(
        "--adapt_objective",
        choices=("legacy", "bp_jg", "p4_identity"),
        default="legacy",
    )
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=18.0)
    parser.add_argument("--feature_anchor_weight", type=float, default=0.05)
    parser.add_argument("--view_consistency_weight", type=float, default=0.0)
    parser.add_argument("--cross_view_prototype_weight", type=float, default=0.0)
    parser.add_argument("--view_score_distill_weight", type=float, default=0.0)
    parser.add_argument("--view_score_distill_temperature", type=float, default=2.0)
    parser.add_argument("--cosine_margin", type=float, default=0.0)
    parser.add_argument("--class_dro_temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--max_optimizer_steps", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument(
        "--view_sampling_mode",
        choices=("stacked", "rotating_single", "shot_index"),
        default="stacked",
    )
    parser.add_argument("--matched_view_teacher_weight", type=float, default=0.0)
    parser.add_argument(
        "--support_view_policy",
        choices=("formal_scenario_cycle", "rx_shift_pair_cycle"),
        default="formal_scenario_cycle",
    )
    parser.add_argument("--init_adapter_state", type=Path, default=None)
    parser.add_argument("--ground_adapter_state", type=Path, default=None)
    parser.add_argument("--ground_adapter_sha256", default=None)
    parser.add_argument(
        "--ground_adapter_scope",
        choices=("projection_feature", "effective_feature"),
        default="projection_feature",
    )
    parser.add_argument("--ground_adapter_rank", type=int, choices=(8, 16), default=16)
    parser.add_argument("--ground_adapter_alpha", type=float, default=16.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def _validate_deployment_controls(args: argparse.Namespace) -> None:
    if str(args.adapt_objective) == "bp_jg":
        checks = {
            "bp_jg_lora_only": str(args.adapter_type) == "lora",
            "bp_jg_target_scope": str(args.scope)
            in {"joint_projection", "joint_gate"},
            "bp_jg_rank": int(args.rank) in {8, 16},
            "bp_jg_alpha_matches_rank": math.isclose(
                float(args.alpha), float(args.rank), rel_tol=0.0, abs_tol=1.0e-12
            ),
            "bp_jg_exactly_five_epochs": int(args.epochs) == 5,
            "bp_jg_sgd_without_momentum_state": str(args.optimizer) == "sgd",
            "bp_jg_exactly_fifty_step_cap": int(args.max_optimizer_steps) == 50,
            "bp_jg_grad_clip_one": math.isclose(
                float(args.grad_clip), 1.0, rel_tol=0.0, abs_tol=1.0e-12
            ),
            "bp_jg_shot_index_episodes": str(args.view_sampling_mode)
            == "shot_index",
            "bp_jg_formal_scenario_views": str(args.support_view_policy)
            == "formal_scenario_cycle",
            "bp_jg_ground_state_required": args.ground_adapter_state is not None,
            "bp_jg_ground_sha256_required": args.ground_adapter_sha256 is not None
            and len(str(args.ground_adapter_sha256).strip()) == 64
            and all(
                value in "0123456789abcdef"
                for value in str(args.ground_adapter_sha256).strip().lower()
            ),
            "bp_jg_ground_p4_scope": str(args.ground_adapter_scope)
            == "projection_feature",
            "bp_jg_ground_rank16": int(args.ground_adapter_rank) == 16,
            "bp_jg_ground_alpha16": math.isclose(
                float(args.ground_adapter_alpha), 16.0, rel_tol=0.0, abs_tol=1.0e-12
            ),
            "bp_jg_no_legacy_init_state": args.init_adapter_state is None,
            "bp_jg_learning_rate": math.isclose(
                float(args.learning_rate), 5.0e-3, rel_tol=0.0, abs_tol=1.0e-12
            ),
            "bp_jg_weight_decay": math.isclose(
                float(args.weight_decay), 1.0e-4, rel_tol=0.0, abs_tol=1.0e-12
            ),
            "bp_jg_temperature": math.isclose(
                float(args.temperature), 18.0, rel_tol=0.0, abs_tol=1.0e-12
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"invalid BP-JG deployment controls: {failed}")
        return
    if str(args.adapt_objective) == "p4_identity":
        checks = {
            "p4_identity_lora_cli": str(args.adapter_type) == "lora",
            "p4_identity_zero_epochs": int(args.epochs) == 0,
            "p4_identity_zero_steps": int(args.max_optimizer_steps) == 0,
            "p4_identity_ground_state_required": args.ground_adapter_state
            is not None,
            "p4_identity_ground_sha256_required": args.ground_adapter_sha256
            is not None
            and len(str(args.ground_adapter_sha256).strip()) == 64
            and all(
                value in "0123456789abcdef"
                for value in str(args.ground_adapter_sha256).strip().lower()
            ),
            "p4_identity_ground_scope": str(args.ground_adapter_scope)
            == "projection_feature",
            "p4_identity_ground_rank16": int(args.ground_adapter_rank) == 16,
            "p4_identity_ground_alpha16": math.isclose(
                float(args.ground_adapter_alpha),
                16.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            "p4_identity_no_legacy_init": args.init_adapter_state is None,
            "p4_identity_formal_views": str(args.support_view_policy)
            == "formal_scenario_cycle",
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"invalid P4 identity controls: {failed}")
        return
    if str(args.adapter_type) not in {"late_film", "late_key_ft"}:
        return
    checks = {
        "epochs_1_to_5": 1 <= int(args.epochs) <= 5,
        "sgd_without_moment_state": str(args.optimizer) == "sgd",
        "optimizer_steps_1_to_50": 1 <= int(args.max_optimizer_steps) <= 50,
        "rotating_single_view": str(args.view_sampling_mode)
        == "rotating_single",
        "matched_view_teacher_enabled": float(args.matched_view_teacher_weight)
        > 0.0,
        "no_stacked_view_consistency": float(args.view_consistency_weight) == 0.0,
        "no_stacked_cross_view_prototype": float(
            args.cross_view_prototype_weight
        )
        == 0.0,
        "no_stacked_view_score_distillation": float(
            args.view_score_distill_weight
        )
        == 0.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if str(args.support_view_policy) == "rx_shift_pair_cycle":
        pair_checks = {
            "rx_pair_sparse_key_only": str(args.adapter_type) == "late_key_ft",
            "rx_pair_ground_initialization_required": args.init_adapter_state
            is not None,
            "rx_pair_exactly_five_epochs": int(args.epochs) == 5,
            "rx_pair_rotating_single_slots": str(args.view_sampling_mode)
            == "rotating_single",
            "rx_pair_teacher_enabled": float(args.matched_view_teacher_weight)
            > 0.0,
        }
        failed.extend(
            name for name, passed in pair_checks.items() if not passed
        )
    if failed:
        raise ValueError(
            f"invalid extreme-light {args.adapter_type} controls: {failed}"
        )


def build_support_run_id(args: argparse.Namespace) -> str:
    if str(args.adapt_objective) == "bp_jg":
        ground_tag = str(args.ground_adapter_sha256).strip().lower()[:12]
        run_prefix = (
            f"support_p4_{ground_tag}_bp_jg_{args.scope}_r{int(args.rank)}"
        )
    elif str(args.adapt_objective) == "p4_identity":
        ground_tag = str(args.ground_adapter_sha256).strip().lower()[:12]
        run_prefix = f"support_p4_{ground_tag}_identity"
    else:
        run_prefix = (
            "support_late_film"
            if str(args.adapter_type) == "late_film"
            else (
                "support_late_key_ft"
                if str(args.adapter_type) == "late_key_ft"
                else f"support_lora_{args.scope}"
            )
        )
    return (
        f"{run_prefix}_rx_{args.receiver}_new_{args.new_count}"
        f"_seed_{args.seed}_k_{args.k_shot}"
    )


def validate_bp_jg_qknn_config(config: dict[str, Any]) -> None:
    """Lock the post-adaptation classifier to class-symmetric prototype qKNN."""

    forbidden = [
        key
        for key in ("primary_k_shot", "sensitivity_k_shot")
        if key in config
    ]
    if forbidden:
        raise ValueError(f"BP-JG config retains legacy K10/K5 fields: {forbidden}")
    expected = {
        "support_pool_max_k": 20,
        "qknnv42_head_mode": "qknn",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_decision_mode": "per_sample_argmax",
        "qknnv42_old_anchor_bias": 0.0,
    }
    failed = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if failed:
        raise ValueError(f"invalid BP-JG class-symmetric qKNN config: {failed}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_deployment_controls(args)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    support_pool_max_k = int(config.get("support_pool_max_k", 10))
    if str(args.adapt_objective) in {"bp_jg", "p4_identity"} and support_pool_max_k != 20:
        raise ValueError(
            "BP-JG requires support_pool_max_k=20 so K1/5/10/20 share one "
            "disjoint query set"
        )
    if str(args.adapt_objective) in {"bp_jg", "p4_identity"}:
        validate_bp_jg_qknn_config(config)
    old_labels = [str(value) for value in config["target_old_tx_labels"]]
    new_labels = [str(value) for value in config["target_new_tx_labels"]][: int(args.new_count)]
    mapping = config.get("feature_npz_by_scenario", {})
    if set(mapping) != set(SCENARIOS):
        raise ValueError(f"config must map exactly the formal scenarios: {SCENARIOS}")
    caches: dict[str, dict[str, np.ndarray]] = {}
    source_manifests: dict[str, dict[str, Any]] = {}
    cache_hashes: dict[str, str] = {}
    for scenario in SCENARIOS:
        path = Path(mapping[scenario])
        caches[scenario], source_manifests[scenario] = _load_npz(path)
        cache_hashes[scenario] = _sha256_file(path)
        roles = caches[scenario]["dataset_role"].astype(str)
        target_mask = np.isin(roles, ["target_old", "target_new"])
        observed = set(caches[scenario]["sat_scenarios"][target_mask].astype(str).tolist())
        if observed != {scenario}:
            raise ValueError(f"cache scenario mismatch for {scenario}: {sorted(observed)}")
    support_rows, support_labels, split_manifest = assemble_support_views(
        caches,
        receiver=str(args.receiver),
        old_labels=old_labels,
        new_labels=new_labels,
        seed=int(args.seed),
        k_shot=int(args.k_shot),
        support_pool_max_k=support_pool_max_k,
        query_per_tx=int(config.get("query_per_tx", 20)),
    )
    physical_support_ids = [str(value) for value in split_manifest["physical_support_ids"]]
    physical_query_ids = [str(value) for value in split_manifest["physical_query_ids"]]
    support_query_overlap = sorted(set(physical_support_ids) & set(physical_query_ids))
    if support_query_overlap:
        raise ValueError(
            "physical support/query overlap detected: "
            f"{support_query_overlap[:5]}"
        )
    split_manifest["support_pool_max_k"] = support_pool_max_k
    split_manifest["support_query_overlap_count"] = 0
    support_view_policy_audit: dict[str, Any] = {
        "policy": "formal_scenario_cycle",
        "input_formal_scenario_count": int(split_manifest["support_view_count"]),
        "physical_support_count": int(
            len(support_rows) // int(split_manifest["support_view_count"])
        ),
        "receive_views_per_physical_sample_per_epoch": (
            int(split_manifest["support_view_count"])
            if str(args.adapt_objective) == "bp_jg"
            else 1
        ),
        "unique_receive_view_count": (
            int(split_manifest["support_view_count"])
            if str(args.adapt_objective) == "bp_jg"
            else 1
        ),
    }
    training_support_view_count = int(split_manifest["support_view_count"])
    if str(args.support_view_policy) == "rx_shift_pair_cycle":
        support_rows, support_labels, support_view_policy_audit = (
            build_rx_shift_pair_cycle(
                support_rows,
                support_labels,
                input_view_count=int(split_manifest["support_view_count"]),
            )
        )
        training_support_view_count = int(
            support_view_policy_audit["epoch_slot_count"]
        )
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32))
    # Project checkpoints contain the trusted SatViewStage enum in addition to tensors.
    # PyTorch 2.6 defaults weights_only=True, which rejects that local metadata.
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, checkpoint_load_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=int(support_rows.shape[-1]), device=device
    )
    if str(getattr(model, "id_feature_key", "")) != "feat_joint":
        raise ValueError(
            "lightweight adapter targets are preregistered for feat_joint, got "
            f"{getattr(model, 'id_feature_key', None)!r}"
        )
    ground_adapter_audit: dict[str, Any] = {"mode": "none"}
    if args.ground_adapter_state is not None:
        if str(args.adapt_objective) not in {"bp_jg", "p4_identity"}:
            raise ValueError(
                "ground_adapter_state is reserved for the BP-JG two-stage route"
            )
        ground_adapter_audit = load_and_merge_ground_lora(
            model,
            Path(args.ground_adapter_state),
            scope=str(args.ground_adapter_scope),
            rank=int(args.ground_adapter_rank),
            alpha=float(args.ground_adapter_alpha),
            expected_sha256=str(args.ground_adapter_sha256),
        )
    if str(args.adapt_objective) == "p4_identity":
        resources = {
            "adapter_type": "target_identity",
            "scope": "none",
            "target_modules": [],
            "trainable_parameter_names": [],
            "trainable_parameters": 0,
            "adapter_state_bytes_fp16": 0,
            "adapter_state_bytes_fp32": 0,
            "adapter_macs_per_query": 0,
            "query_view_count": 1,
            "original_checkpoint_trainable_parameters": 0,
            "original_checkpoint_gradient_updates": 0,
            "full_model_finetune": False,
            "resource_tier": "preferred_identity_control",
            "trainable_parameter_cap": 50_000,
            "persistent_state_cap_bytes": 256 * 1024,
        }
        method = "support_only_p4_identity_control_v1"
    elif str(args.adapter_type) == "late_film":
        resources = inject_late_channel_film(model)
        method = (
            "support_only_late_channel_film_source_init_v1"
            if args.init_adapter_state is not None
            else "support_only_late_channel_film_v1"
        )
    elif str(args.adapter_type) == "late_key_ft":
        resources = enable_late_key_layer_finetune(model)
        method = (
            (
                "support_only_late_key_ft_source_init_rx_shift_pair_v1"
                if str(args.support_view_policy) == "rx_shift_pair_cycle"
                else "support_only_late_key_ft_source_init_v1"
            )
            if args.init_adapter_state is not None
            else "support_only_late_key_ft_v1"
        )
    else:
        resources = inject_feat_joint_lora(
            model, rank=int(args.rank), alpha=float(args.alpha), scope=str(args.scope)
        )
        method = (
            f"support_only_p4_bp_jg_{args.scope}_lora_v1"
            if str(args.adapt_objective) == "bp_jg"
            else f"support_only_{args.scope}_lora_v1"
        )
    strict_checkpoint_trainable_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    initialization_audit = {"mode": "identity_zero", "strict_key_match": True}
    if args.init_adapter_state is not None:
        if str(args.adapter_type) not in {"late_film", "late_key_ft"}:
            raise ValueError(
                "ground initialization is restricted to late_film or late_key_ft"
            )
        initialization_audit = load_trainable_adapter_state(
            model, Path(args.init_adapter_state)
        )
    model.to(device).eval()
    if str(args.adapt_objective) == "bp_jg":
        trace, runtime = train_support_only_bp_jg(
            model,
            support_rows,
            support_labels,
            physical_support_ids=split_manifest["physical_support_ids"],
            support_row_physical_ids=split_manifest["support_row_physical_ids"],
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            temperature=float(args.temperature),
            support_view_count=int(training_support_view_count),
            batch_size=int(args.batch_size),
            max_optimizer_steps=int(args.max_optimizer_steps),
            grad_clip=float(args.grad_clip),
            seed=int(args.seed),
            device=device,
        )
    elif str(args.adapt_objective) == "p4_identity":
        trace = [
            {
                "epoch": 0,
                "loss": 0.0,
                "support_train_acc": 0.0,
                "gradient_norm": 0.0,
                "optimizer_steps": 0,
                "episode_count": 0,
                "learning_rate": 0.0,
                "epoch_seconds": 0.0,
            }
        ]
        runtime = {
            "adaptation_wall_seconds": 0.0,
            "peak_cuda_memory_bytes": 0,
            "optimizer_state_deployment_required": False,
            "optimizer": "none",
            "optimizer_momentum": 0.0,
            "optimizer_steps": 0,
            "max_optimizer_steps": 0,
            "optimizer_training_state_bytes_estimate": 0,
            "support_view_count": int(training_support_view_count),
            "physical_support_count": len(physical_support_ids),
            "registered_class_count": len(old_labels) + len(new_labels),
            "k_shot_inferred": int(args.k_shot),
            "episodes_per_epoch": 0,
            "support_forward_sample_equivalents": 0,
            "query_rows_used_for_training": 0,
            "old_new_role_used_by_optimizer": False,
            "class_quota_used_by_optimizer": False,
            "dense_query_graph_used": False,
        }
    else:
        trace, runtime = train_support_only_lora(
            model,
            support_rows,
            support_labels,
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            temperature=float(args.temperature),
            feature_anchor_weight=float(args.feature_anchor_weight),
            view_consistency_weight=float(args.view_consistency_weight),
            cross_view_prototype_weight=float(args.cross_view_prototype_weight),
            view_score_distill_weight=float(args.view_score_distill_weight),
            view_score_distill_temperature=float(
                args.view_score_distill_temperature
            ),
            cosine_margin=float(args.cosine_margin),
            class_dro_temperature=float(args.class_dro_temperature),
            support_view_count=int(training_support_view_count),
            batch_size=int(args.batch_size),
            optimizer_name=str(args.optimizer),
            max_optimizer_steps=int(args.max_optimizer_steps),
            grad_clip=float(args.grad_clip),
            view_sampling_mode=str(args.view_sampling_mode),
            matched_view_teacher_weight=float(args.matched_view_teacher_weight),
            seed=int(args.seed),
            device=device,
        )
    if str(args.adapter_type) == "late_key_ft":
        resources["original_checkpoint_gradient_updates"] = int(
            runtime["optimizer_steps"]
        )
    runtime["support_view_policy"] = str(args.support_view_policy)
    runtime["support_view_policy_audit"] = support_view_policy_audit
    if str(args.support_view_policy) == "rx_shift_pair_cycle":
        runtime["train_receive_views_per_physical_sample_per_epoch"] = 2
        runtime["unique_receive_view_count"] = 3
    run_id = build_support_run_id(args)
    run_dir = args.out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_trace(run_dir / "loss_trace.json", trace)
    if str(args.adapter_type) == "late_key_ft":
        fp16_state = {
            name: (
                parameter.detach().cpu() - strict_checkpoint_trainable_state[name]
            ).half()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        adapter_state_format = "fp16_delta_from_strict_checkpoint"
    else:
        fp16_state = {
            name: parameter.detach().cpu().half()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        adapter_state_format = "fp16_trainable_state"
    adapter_state_path = run_dir / "adapter_state_fp16.pt"
    torch.save(fp16_state, adapter_state_path)
    resources["adapter_state_file_bytes_fp16_pt"] = int(
        adapter_state_path.stat().st_size
    )
    target_merge_audit: dict[str, Any] = {"mode": "not_merged"}
    if str(args.adapt_objective) == "bp_jg":
        target_merge_audit = roundtrip_fp16_target_lora_and_merge(
            model, adapter_state_path
        )
        resources["deployment_added_macs_per_query_after_merge"] = 0
        resources["target_lora_merge_audit"] = target_merge_audit
    registered_class_count = int(len(old_labels) + len(new_labels))
    registered_feature_dim = int(
        160
        + (
            int(config.get("qknnv42_aux_feature_dim", 0))
            if str(config.get("qknnv42_aux_feature_key", ""))
            else 0
        )
    )
    prototype_state_bytes = int(
        5 * registered_class_count * registered_feature_dim * 2
    )
    threshold_state_bytes = 24
    ground_adapter_state_bytes = int(
        ground_adapter_audit.get("resources", {}).get(
            "adapter_state_bytes_fp16", 0
        )
    )
    combined_state_bytes = int(
        ground_adapter_state_bytes
        + resources["adapter_state_bytes_fp16"]
        + prototype_state_bytes
        + threshold_state_bytes
    )
    persistent_state_cap = int(
        config.get("extreme_light_max_persistent_state_bytes", 256 * 1024)
    )
    resources.update(
        {
            "ground_adapter_state_bytes_fp16": ground_adapter_state_bytes,
            "target_adapter_state_bytes_fp16": int(
                resources["adapter_state_bytes_fp16"]
            ),
            "ground_plus_target_adapter_state_bytes_fp16": int(
                ground_adapter_state_bytes + resources["adapter_state_bytes_fp16"]
            ),
            "ground_adapter_serialized_file_bytes": int(
                ground_adapter_audit.get("serialized_file_bytes", 0)
            ),
            "target_adapter_serialized_file_bytes": int(
                resources.get("adapter_state_file_bytes_fp16_pt", 0)
            ),
            "runner_persistent_state_bytes": None,
            "runner_persistent_state_requires_post_evaluation_audit": True,
            "combined_adaptive_tta_state_is_pre_evaluation_estimate": True,
            "five_view_prototype_state_bytes_fp16": prototype_state_bytes,
            "adaptive_tta_threshold_state_bytes_fp32": threshold_state_bytes,
            "combined_adaptive_tta_state_bytes": combined_state_bytes,
            "combined_persistent_state_cap_bytes": persistent_state_cap,
            "combined_persistent_state_within_cap": combined_state_bytes
            <= persistent_state_cap,
            "parameter_increase_vs_50k_percent": float(
                max(
                    0.0,
                    (
                        float(resources["trainable_parameters"]) / 50_000.0
                        - 1.0
                    )
                    * 100.0,
                )
            ),
        }
    )
    if not bool(resources["combined_persistent_state_within_cap"]):
        raise ValueError(
            "adapter plus five-view prototype state exceeds configured cap: "
            f"{resources}"
        )
    adaptation_manifest = {
        "method": method,
        "adapt_objective": str(args.adapt_objective),
        "resource_tier": str(resources.get("resource_tier", "preferred")),
        "receiver": str(args.receiver),
        "new_count": int(args.new_count),
        "seed": int(args.seed),
        "k_shot": int(args.k_shot),
        "support_view_count": int(split_manifest["support_view_count"]),
        "training_epoch_slot_count": int(training_support_view_count),
        "support_view_policy": str(args.support_view_policy),
        "support_view_policy_audit": support_view_policy_audit,
        "query_view_count": 1,
        "support_only": True,
        "query_update_forbidden": True,
        "query_features_used_for_training": False,
        "query_labels_used_for_training": False,
        "old_new_role_used_by_optimizer": False,
        "class_quota_used_at_inference": False,
        "epochs": int(args.epochs),
        "hyperparameters": {
            "rank": int(args.rank),
            "alpha": float(args.alpha),
            "scope": str(args.scope),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "temperature": float(args.temperature),
            "feature_anchor_weight": float(args.feature_anchor_weight),
            "view_consistency_weight": float(args.view_consistency_weight),
            "cross_view_prototype_weight": float(args.cross_view_prototype_weight),
            "view_score_distill_weight": float(args.view_score_distill_weight),
            "view_score_distill_temperature": float(
                args.view_score_distill_temperature
            ),
            "cosine_margin": float(args.cosine_margin),
            "class_dro_temperature": float(args.class_dro_temperature),
            "batch_size": int(args.batch_size),
            "adapter_type": str(args.adapter_type),
            "optimizer": str(args.optimizer),
            "max_optimizer_steps": int(args.max_optimizer_steps),
            "grad_clip": float(args.grad_clip),
            "view_sampling_mode": str(args.view_sampling_mode),
            "matched_view_teacher_weight": float(
                args.matched_view_teacher_weight
            ),
            "support_view_policy": str(args.support_view_policy),
            "bp_jg_loss_weights": (
                {
                    "xview_prototype_ce": 1.0,
                    "boundary_margin": 2.0,
                    "feature_anchor": 0.5,
                    "prototype_gram": 0.5,
                    "prototype_separation": 0.25,
                    "view_consistency": 0.1,
                    "boundary_margin_delta": 0.02,
                }
                if str(args.adapt_objective) == "bp_jg"
                else None
            ),
        },
        "resources": resources,
        "initialization": initialization_audit,
        "ground_adapter": ground_adapter_audit,
        "target_merge": target_merge_audit,
        "adapter_state_format": adapter_state_format,
        "adapter_state": str(adapter_state_path),
        "adapter_state_sha256": _sha256_file(adapter_state_path),
        "runtime": runtime,
        "split": split_manifest,
        "input_cache_sha256": cache_hashes,
        "checkpoint": str(args.ckpt),
        "checkpoint_sha256": _sha256_file(args.ckpt),
        "checkpoint_load_audit": checkpoint_load_audit,
    }
    identity = nn.Identity()
    export_audit: dict[str, Any] = {}
    output_mapping: dict[str, str] = {}
    for scenario in SCENARIOS:
        out_path = run_dir / f"{scenario}.npz"
        export_audit[scenario] = export_adapted_cache(
            caches[scenario],
            source_manifests[scenario],
            model=model,
            adapter=identity,
            receiver=str(args.receiver),
            old_labels=old_labels,
            new_labels=new_labels,
            scenario=scenario,
            batch_size=int(args.batch_size),
            device=device,
            out_path=out_path,
            adaptation_manifest=adaptation_manifest,
            payload_source=f"cvs_stage2c_{method}",
        )
        output_mapping[scenario] = str(out_path)
    resolved = dict(config)
    resolved.update(
        {
            "experiment_id": run_id,
            "feature_npz_by_scenario": output_mapping,
            "target_receiver_labels": [str(args.receiver)],
            "target_new_tx_labels": new_labels,
            "split_seed": int(args.seed),
            "seed": int(args.seed),
            "k_shot": int(args.k_shot),
            "qknnv42_expected_tta_view_count": 1,
            "input_adapter_method": method,
            "input_adapter_manifest": str(run_dir / "training_manifest.json"),
        }
    )
    resolved_path = run_dir / "resolved_qknn_config.json"
    resolved_path.write_text(
        json.dumps(_json_safe(resolved), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    training_manifest = {
        **adaptation_manifest,
        "loss_trace_json": str(run_dir / "loss_trace.json"),
        "loss_trace_csv": str(run_dir / "loss_trace.csv"),
        "adapter_state": str(adapter_state_path),
        "adapter_state_sha256": _sha256_file(adapter_state_path),
        "export_audit": export_audit,
        "resolved_qknn_config": str(resolved_path),
    }
    (run_dir / "training_manifest.json").write_text(
        json.dumps(_json_safe(training_manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "resolved_qknn_config": str(resolved_path),
                "resources": resources,
                "last_epoch": trace[-1],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
