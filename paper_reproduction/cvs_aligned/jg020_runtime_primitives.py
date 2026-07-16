"""Minimal sealed-runtime primitives for the JG_R8_LR020 adapter.

This module intentionally contains no dataset loader, cache builder, CLI, clean
sample path, or query-scoring API.  It is safe to place on the Phase2 runtime
allowlist together with a sealed checkpoint, adapter, and LEO_weak bundle.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCOPES = {
    "projection_feature": (
        "id_backbone.t_proj",
        "id_backbone.f_proj",
        "id_backbone.pa_proj.0",
        "id_backbone.fuse.0",
    ),
    "joint_gate": (
        "id_backbone.cls_head.id_gate.0",
        "id_backbone.cls_head.joint_proj.0",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        if int(rank) <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in base.parameters():
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


def _resolve_parent(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def inject_feat_joint_lora(
    model: nn.Module, *, rank: int, alpha: float, scope: str
) -> dict[str, Any]:
    scope = str(scope).strip().lower()
    if scope not in SCOPES:
        raise ValueError(f"sealed JG runtime does not allow LoRA scope={scope}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = dict(model.named_modules())
    injected: list[dict[str, Any]] = []
    for name in SCOPES[scope]:
        original = modules.get(name)
        if not isinstance(original, nn.Linear):
            raise TypeError(f"required JG Linear is missing: {name}")
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
            }
        )
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if any(".lora_" not in name for name, _ in trainable):
        raise RuntimeError("sealed JG runtime exposed non-LoRA trainable parameters")
    count = int(sum(parameter.numel() for _, parameter in trainable))
    if count > 50_000 or count * 2 > 256 * 1024:
        raise ValueError("sealed JG runtime adapter exceeds resource cap")
    return {
        "adapter_type": f"{scope}_lora",
        "scope": scope,
        "target_modules": injected,
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_parameters": count,
        "adapter_state_bytes_fp16": count * 2,
        "adapter_state_bytes_fp32": count * 4,
        "original_checkpoint_gradient_updates": 0,
        "full_model_finetune": False,
    }


@torch.no_grad()
def merge_feat_joint_lora(model: nn.Module) -> dict[str, Any]:
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]
    if not targets:
        raise ValueError("model contains no LoRA module to merge")
    parity: list[dict[str, Any]] = []
    for name, module in targets:
        base = module.base
        probe = torch.linspace(
            -0.5,
            0.5,
            steps=3 * int(base.in_features),
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
        merged.weight.copy_(
            base.weight
            + (module.scaling * (module.lora_b.weight @ module.lora_a.weight)).to(
                base.weight.dtype
            )
        )
        if base.bias is not None:
            merged.bias.copy_(base.bias)
        for parameter in merged.parameters():
            parameter.requires_grad_(False)
        parent, leaf = _resolve_parent(model, name)
        if leaf.isdigit():
            parent[int(leaf)] = merged
        else:
            setattr(parent, leaf, merged)
        max_abs = float(torch.max(torch.abs(expected - merged(probe))).item())
        parity.append({"module": name, "max_absolute_difference": max_abs})
    max_difference = max(row["max_absolute_difference"] for row in parity)
    if max_difference > 1.0e-5:
        raise RuntimeError(f"LoRA merge parity failed: {max_difference}")
    return {
        "merged_module_count": len(parity),
        "merged_modules": parity,
        "remaining_lora_wrappers": [],
        "post_merge_trainable_parameters": 0,
        "max_absolute_difference": max_difference,
        "algebraic_probe_parity_pass": True,
        "deployment_added_macs_per_view_after_merge": 0,
    }


def load_and_merge_ground_lora(
    model: nn.Module,
    state_path: Path,
    *,
    scope: str,
    rank: int,
    alpha: float,
    expected_sha256: str,
) -> dict[str, Any]:
    state_path = Path(state_path)
    observed = _sha256(state_path)
    if observed != str(expected_sha256).lower():
        raise ValueError("ground adapter SHA256 mismatch")
    resources = inject_feat_joint_lora(
        model, rank=int(rank), alpha=float(alpha), scope=str(scope)
    )
    payload = torch.load(state_path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError("ground adapter state must be a tensor mapping")
    expected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(payload) != set(expected):
        raise ValueError("ground adapter state key mismatch")
    l2_sq = 0.0
    with torch.no_grad():
        for name, parameter in expected.items():
            value = payload[name]
            if not torch.is_tensor(value) or tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"ground adapter tensor drift: {name}")
            value_float = value.detach().float()
            if not bool(torch.isfinite(value_float).all()):
                raise FloatingPointError(f"non-finite ground adapter tensor: {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            l2_sq += float(torch.sum(value_float.square()))
    merge = merge_feat_joint_lora(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {
        "mode": "ground_lora_loaded_and_merged",
        "scope": str(scope),
        "rank": int(rank),
        "alpha": float(alpha),
        "resources": resources,
        "tensor_count": len(expected),
        "element_count": int(sum(value.numel() for value in payload.values())),
        "l2_norm": math.sqrt(l2_sq),
        "strict_key_match": True,
        "merge": merge,
        "serialized_file_bytes": int(state_path.stat().st_size),
        "expected_sha256": str(expected_sha256).lower(),
        "observed_sha256_before_load": observed,
        "sha256_preload_match": True,
        "deployment_added_macs_per_query_after_merge": 0,
    }


def _norm_rows(rows: torch.Tensor) -> torch.Tensor:
    return F.normalize(rows, dim=-1)


def _matched_view_support_layout(
    labels: torch.Tensor, *, view_count: int
) -> tuple[int, int, int, list[torch.Tensor]]:
    if labels.ndim != 1 or int(view_count) <= 1:
        raise ValueError("matched-view support labels/view count invalid")
    if int(labels.numel()) % int(view_count) != 0:
        raise ValueError("matched-view support row count drift")
    physical_count = int(labels.numel()) // int(view_count)
    physical_labels = labels[:physical_count]
    for view_index in range(1, int(view_count)):
        start = view_index * physical_count
        if not torch.equal(labels[start : start + physical_count], physical_labels):
            raise ValueError("matched-view label order drift")
    classes = torch.unique(physical_labels, sorted=True)
    if int(classes.numel()) < 2 or not torch.equal(
        classes, torch.arange(int(classes.numel()), device=labels.device)
    ):
        raise ValueError("registered class indices must be contiguous")
    positions = [
        torch.nonzero(physical_labels == class_index, as_tuple=False).flatten()
        for class_index in range(int(classes.numel()))
    ]
    counts = {int(value.numel()) for value in positions}
    if len(counts) != 1 or not counts or min(counts) < 1:
        raise ValueError("registered classes need equal non-empty K-shot support")
    return physical_count, int(classes.numel()), counts.pop(), positions


def build_shot_index_episode_positions(
    labels: torch.Tensor,
    *,
    view_count: int,
    max_episodes_per_epoch: int = 10,
    pair_physical_shots: bool = False,
) -> list[torch.Tensor]:
    if int(max_episodes_per_epoch) <= 0:
        raise ValueError("max_episodes_per_epoch must be positive")
    physical_count, class_count, k_shot, positions = _matched_view_support_layout(
        labels, view_count=int(view_count)
    )
    episode_count = min(k_shot, int(max_episodes_per_epoch))
    if pair_physical_shots and 2 <= k_shot <= int(max_episodes_per_epoch):
        groups = [np.asarray([index, (index + 1) % k_shot]) for index in range(k_shot)]
        expected_repeats = 2
    else:
        groups = list(np.array_split(np.arange(k_shot), episode_count))
        expected_repeats = 1
    episodes: list[torch.Tensor] = []
    for group in groups:
        shot = torch.as_tensor(group, device=labels.device, dtype=torch.long)
        physical = torch.cat([positions[index][shot] for index in range(class_count)])
        episodes.append(
            torch.cat(
                [physical + view * physical_count for view in range(int(view_count))]
            )
        )
    covered = torch.cat(
        [episode[: int(episode.numel()) // int(view_count)] for episode in episodes]
    )
    if not torch.equal(
        torch.bincount(covered, minlength=physical_count),
        torch.full((physical_count,), expected_repeats, device=labels.device),
    ):
        raise RuntimeError("JG episode coverage drift")
    return episodes


def _leave_one_physical_prototype_banks(
    rows: torch.Tensor, physical_labels: torch.Tensor, *, class_count: int
) -> torch.Tensor:
    one_hot = F.one_hot(physical_labels, num_classes=class_count).to(rows.dtype)
    counts = one_hot.sum(dim=0) * float(rows.shape[0])
    sums = torch.einsum("pc,vpd->cd", one_hot, rows)
    banks = sums.unsqueeze(0).expand(int(rows.shape[1]), -1, -1).clone()
    bank_counts = counts.unsqueeze(0).expand(int(rows.shape[1]), -1).clone()
    indices = torch.arange(int(rows.shape[1]), device=rows.device)
    banks[indices, physical_labels] -= rows.sum(dim=0)
    bank_counts[indices, physical_labels] -= float(rows.shape[0])
    return _norm_rows(banks / bank_counts.unsqueeze(-1))


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
    leave_one_physical_shot: bool = False,
) -> dict[str, torch.Tensor]:
    physical_count, class_count, episode_k, _ = _matched_view_support_layout(
        labels, view_count=int(view_count)
    )
    z = _norm_rows(features).reshape(int(view_count), physical_count, -1)
    z0 = _norm_rows(base_features.detach()).reshape(int(view_count), physical_count, -1)
    physical_labels = labels[:physical_count]

    def prototypes(rows: torch.Tensor, row_labels: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                _norm_rows(rows[row_labels == index].mean(dim=0, keepdim=True))[0]
                for index in range(class_count)
            ]
        )

    if leave_one_physical_shot and episode_k >= 2:
        banks = _leave_one_physical_prototype_banks(
            z, physical_labels, class_count=class_count
        )
        base_banks = _leave_one_physical_prototype_banks(
            z0, physical_labels, class_count=class_count
        )
        scores = [torch.einsum("pd,pcd->pc", z[v], banks) for v in range(view_count)]
        base_scores = [
            torch.einsum("pd,pcd->pc", z0[v], base_banks) for v in range(view_count)
        ]
    else:
        scores, base_scores = [], []
        for held in range(view_count):
            kept = [value for value in range(view_count) if value != held]
            proto_labels = physical_labels.repeat(len(kept))
            scores.append(z[held] @ prototypes(torch.cat([z[v] for v in kept]), proto_labels).T)
            base_scores.append(
                z0[held] @ prototypes(torch.cat([z0[v] for v in kept]), proto_labels).T
            )
    cosine = torch.cat(scores)
    base_cosine = torch.cat(base_scores).detach()
    targets = physical_labels.repeat(view_count)
    xview_ce = F.cross_entropy(float(temperature) * cosine, targets)
    row = torch.arange(int(targets.numel()), device=targets.device)
    mask = F.one_hot(targets, num_classes=class_count).bool()
    margin = cosine[row, targets] - cosine.masked_fill(mask, float("-inf")).max(1).values
    base_margin = (
        base_cosine[row, targets]
        - base_cosine.masked_fill(mask, float("-inf")).max(1).values
    )
    boundary = F.relu(base_margin + float(boundary_margin_delta) - margin).mean()
    anchor = (1.0 - torch.sum(z * z0, dim=-1)).mean()
    all_rows = z.reshape(-1, int(z.shape[-1]))
    base_rows = z0.reshape(-1, int(z0.shape[-1]))
    all_labels = physical_labels.repeat(view_count)
    proto = prototypes(all_rows, all_labels)
    base_proto = prototypes(base_rows, all_labels).detach()
    gram = proto @ proto.T
    base_gram = base_proto @ base_proto.T
    off_diag = ~torch.eye(class_count, device=gram.device, dtype=torch.bool)
    gram_loss = F.smooth_l1_loss(gram[off_diag], base_gram[off_diag])
    cap = torch.minimum(base_gram, torch.full_like(base_gram, 0.65))
    separation = F.relu(gram[off_diag] - cap[off_diag]).square().mean()
    center = _norm_rows(z.mean(dim=0))
    consistency = (1.0 - torch.sum(z * center.unsqueeze(0), dim=-1)).mean()
    loss = (
        xview_ce
        + boundary_weight * boundary
        + anchor_weight * anchor
        + gram_weight * gram_loss
        + separation_weight * separation
        + view_weight * consistency
    )
    return {
        "loss": loss,
        "xview_prototype_ce": xview_ce,
        "boundary_margin_loss": boundary,
        "feature_anchor_loss": anchor,
        "prototype_gram_loss": gram_loss,
        "prototype_separation_loss": separation,
        "view_consistency_loss": consistency,
        "mean_margin": margin.mean(),
        "mean_base_margin": base_margin.mean(),
        "correct": (cosine.argmax(dim=1) == targets).sum(),
        "sample_count": torch.as_tensor(targets.numel(), device=features.device),
    }
