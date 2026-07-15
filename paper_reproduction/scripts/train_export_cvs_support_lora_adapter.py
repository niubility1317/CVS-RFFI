#!/usr/bin/env python
"""Train a support-only lightweight adapter on the ADV3B02 identity path.

All original checkpoint parameters remain frozen.  The trainer supports either
identity-initialized LoRA branches or a 1,280-parameter late channel-wise FiLM
adapter.  Training consumes registered target support labels and preregistered
LEO support views; target query rows never enter fitting or model selection.
"""

from __future__ import annotations

import argparse
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

FULL_FEATURE_LORA_TARGETS = (
    "id_backbone.t_proj",
    "id_backbone.f_proj",
    "id_backbone.pa_proj.0",
    "id_backbone.fuse.0",
    "id_backbone.con_proj.0",
    *LORA_TARGETS,
    "id_backbone.cls_head.imp_merge.0",
)

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
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False)
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
    parser.add_argument("--k_shot", type=int, choices=(5, 10), default=10)
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
        choices=("feat_joint", "late_feat_joint", "full_feature"),
        default="feat_joint",
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
        choices=("stacked", "rotating_single"),
        default="stacked",
    )
    parser.add_argument("--matched_view_teacher_weight", type=float, default=0.0)
    parser.add_argument(
        "--support_view_policy",
        choices=("formal_scenario_cycle", "rx_shift_pair_cycle"),
        default="formal_scenario_cycle",
    )
    parser.add_argument("--init_adapter_state", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def _validate_deployment_controls(args: argparse.Namespace) -> None:
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_deployment_controls(args)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
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
        support_pool_max_k=int(config.get("support_pool_max_k", 10)),
        query_per_tx=int(config.get("query_per_tx", 20)),
    )
    support_view_policy_audit: dict[str, Any] = {
        "policy": "formal_scenario_cycle",
        "input_formal_scenario_count": int(split_manifest["support_view_count"]),
        "physical_support_count": int(
            len(support_rows) // int(split_manifest["support_view_count"])
        ),
        "receive_views_per_physical_sample_per_epoch": 1,
        "unique_receive_view_count": 1,
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
    if str(args.adapter_type) == "late_film":
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
        method = f"support_only_{args.scope}_lora_v1"
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
    run_prefix = (
        "support_late_film"
        if str(args.adapter_type) == "late_film"
        else (
            "support_late_key_ft"
            if str(args.adapter_type) == "late_key_ft"
            else f"support_lora_{args.scope}"
        )
    )
    run_id = (
        f"{run_prefix}_rx_{args.receiver}_new_{args.new_count}"
        f"_seed_{args.seed}_k_{args.k_shot}"
    )
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
    threshold_state_bytes = 12
    combined_state_bytes = int(
        resources["adapter_state_bytes_fp16"]
        + prototype_state_bytes
        + threshold_state_bytes
    )
    persistent_state_cap = int(
        config.get("extreme_light_max_persistent_state_bytes", 256 * 1024)
    )
    resources.update(
        {
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
        },
        "resources": resources,
        "initialization": initialization_audit,
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
