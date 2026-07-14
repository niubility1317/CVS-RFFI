#!/usr/bin/env python
"""Train a support-only LoRA adapter on the ADV3B02 identity feature head.

All original checkpoint parameters remain frozen.  Identity-initialized LoRA
branches are attached only to the four Linear modules that produce ``feat_joint``.
Training consumes registered target support labels and three preregistered LEO
support views; target query rows never enter fitting or model selection.
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


def _resolve_parent(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def inject_feat_joint_lora(
    model: nn.Module, *, rank: int, alpha: float
) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = dict(model.named_modules())
    injected: list[dict[str, Any]] = []
    for name in LORA_TARGETS:
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
        "adapter_type": "feat_joint_lora",
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
    if parameter_count > 50_000:
        raise ValueError(f"LoRA exceeds 50k parameter cap: {audit}")
    if fp16_bytes > 128 * 1024:
        raise ValueError(f"LoRA exceeds 128KB state cap: {audit}")
    return audit


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
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if int(epochs) <= 0 or int(epochs) > 20:
        raise ValueError("formal extreme-light adaptation epochs must be in [1,20]")
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
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("LoRA injection selected no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters, lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    rng = np.random.default_rng(int(seed))
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        prototypes = _class_prototypes(
            model,
            identity,
            rows,
            labels,
            class_count=class_count,
            batch_size=int(batch_size),
        )
        order = rng.permutation(int(rows.shape[0]))
        totals = {"loss": 0.0, "ce": 0.0, "anchor": 0.0, "correct": 0.0, "grad": 0.0}
        seen = 0
        batches = 0
        for start in range(0, len(order), int(batch_size)):
            positions = torch.as_tensor(
                order[start : start + int(batch_size)], device=device, dtype=torch.long
            )
            optimizer.zero_grad(set_to_none=True)
            z, _ = _feature_forward(model, rows[positions])
            z = _norm_rows(z)
            scores = float(temperature) * (z @ prototypes.T)
            ce = F.cross_entropy(scores, labels[positions])
            anchor = (1.0 - torch.sum(z * base_features[positions], dim=1)).mean()
            loss = ce + float(feature_anchor_weight) * anchor
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            count = int(positions.numel())
            seen += count
            batches += 1
            totals["loss"] += float(loss.detach()) * count
            totals["ce"] += float(ce.detach()) * count
            totals["anchor"] += float(anchor.detach()) * count
            totals["correct"] += float((scores.argmax(dim=1) == labels[positions]).sum().detach())
            totals["grad"] += float(grad_norm.detach())
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(1, seen),
            "prototype_ce": totals["ce"] / max(1, seen),
            "feature_anchor": totals["anchor"] / max(1, seen),
            "support_train_acc": totals["correct"] / max(1, seen),
            "gradient_norm": totals["grad"] / max(1, batches),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"non-finite LoRA trace: {row}")
        trace.append(row)
        print("[SUPPORT-LORA-EPOCH] " + json.dumps(row, sort_keys=True), flush=True)
    runtime = {
        "adaptation_wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "optimizer_state_deployment_required": False,
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
    parser.add_argument("--rank", type=int, choices=(2, 4, 8, 16), default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=18.0)
    parser.add_argument("--feature_anchor_weight", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32))
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model, checkpoint_load_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=int(support_rows.shape[-1]), device=device
    )
    if str(getattr(model, "id_feature_key", "")) != "feat_joint":
        raise ValueError(
            f"LoRA targets are preregistered for feat_joint, got {getattr(model, 'id_feature_key', None)!r}"
        )
    resources = inject_feat_joint_lora(
        model, rank=int(args.rank), alpha=float(args.alpha)
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
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=device,
    )
    run_id = f"support_lora_rx_{args.receiver}_new_{args.new_count}_seed_{args.seed}_k_{args.k_shot}"
    run_dir = args.out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_trace(run_dir / "loss_trace.json", trace)
    fp16_state = {
        name: parameter.detach().cpu().half()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(fp16_state, run_dir / "adapter_state_fp16.pt")
    resources["adapter_state_file_bytes_fp16_pt"] = int(
        (run_dir / "adapter_state_fp16.pt").stat().st_size
    )
    adaptation_manifest = {
        "method": "support_only_feat_joint_lora_v1",
        "receiver": str(args.receiver),
        "new_count": int(args.new_count),
        "seed": int(args.seed),
        "k_shot": int(args.k_shot),
        "support_view_count": 3,
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
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "temperature": float(args.temperature),
            "feature_anchor_weight": float(args.feature_anchor_weight),
            "batch_size": int(args.batch_size),
        },
        "resources": resources,
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
            payload_source="cvs_stage2c_support_only_feat_joint_lora_v1",
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
            "input_adapter_method": "support_only_feat_joint_lora_v1",
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
        "adapter_state": str(run_dir / "adapter_state_fp16.pt"),
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
