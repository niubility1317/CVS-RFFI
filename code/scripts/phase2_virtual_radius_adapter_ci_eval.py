#!/usr/bin/env python
"""VRA-CI virtual-radius adapter evaluation for Stage2-C qknn8.

VRA-CI trains a small low-rank residual adapter on ADV3B02 features using only
source old rows, source-side proxy_unknown rows, and target old/seen-new support.
It adds virtual boundary negatives sampled between known prototypes and proxy
unknown rows, then evaluates the adapted feature file with the existing qknn8
collaborative ENPC/SLEV backends. Target unknown rows remain evaluation-only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from phase2_collaborative_open_set_qknn_eval import _normalize_rows, load_feature_npz  # noqa: E402
from phase2_proxy_adapter_ci_eval import (  # noqa: E402
    AdapterTrainingPlan,
    LowRankResidualAdapter,
    _build_prototypes,
    _proto_logits,
    _state_bytes,
    apply_adapter,
    build_training_plan,
    run_backends,
    save_adapted_npz,
)


def _known_margin_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.view(-1, 1), -1.0e9)
    second = masked.max(dim=1).values
    return F.relu(float(margin) - (true_logits - second)).mean()


def _virtual_boundary_samples(
    prototypes: torch.Tensor,
    proxy_features: torch.Tensor,
    *,
    count: int,
    mix_low: float,
    mix_high: float,
    noise_scale: float,
) -> torch.Tensor:
    n = max(1, int(count))
    class_count = int(prototypes.shape[0])
    device = prototypes.device
    if class_count <= 1:
        base = prototypes[torch.randint(0, class_count, (n,), device=device)]
        other = proxy_features[torch.randint(0, int(proxy_features.shape[0]), (n,), device=device)]
    else:
        first = torch.randint(0, class_count, (n,), device=device)
        second = torch.randint(0, class_count - 1, (n,), device=device)
        second = second + (second >= first).long()
        known_mix = 0.5 * prototypes.index_select(0, first) + 0.5 * prototypes.index_select(0, second)
        proxy_take = proxy_features[torch.randint(0, int(proxy_features.shape[0]), (n,), device=device)]
        base = F.normalize(known_mix, dim=1)
        other = proxy_take
    alpha = torch.empty((n, 1), device=device).uniform_(float(mix_low), float(mix_high))
    virtual = (1.0 - alpha) * base + alpha * other
    if float(noise_scale) > 0.0:
        noise = torch.randn_like(virtual) * float(noise_scale)
        virtual = virtual + noise
    return F.normalize(virtual, dim=1)


def train_vra_adapter(
    payload: Mapping[str, Any],
    plan: AdapterTrainingPlan,
    args: argparse.Namespace,
) -> tuple[LowRankResidualAdapter, dict[str, Any]]:
    requested_device = str(args.device)
    device = torch.device(requested_device if torch.cuda.is_available() or not requested_device.startswith("cuda") else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    features_np = _normalize_rows(np.asarray(payload["features"], dtype=np.float32))
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    dim = int(features_np.shape[1])
    source_labels = [str(tx_ids[i]) for i in plan.source_old_indices]
    support_labels = [str(v) for v in plan.support_labels]
    proto_labels, proto_np = _build_prototypes(
        features_np[np.asarray(plan.source_old_indices + plan.support_indices, dtype=int)],
        source_labels + support_labels,
    )
    label_to_pos = {label: i for i, label in enumerate(proto_labels)}
    source_y = np.asarray([label_to_pos[str(tx_ids[i])] for i in plan.source_old_indices], dtype=np.int64)
    support_y = np.asarray([label_to_pos[str(v)] for v in support_labels], dtype=np.int64)

    x = torch.as_tensor(features_np, dtype=torch.float32, device=device)
    source_idx = torch.as_tensor(plan.source_old_indices, dtype=torch.long, device=device)
    support_idx = torch.as_tensor(plan.support_indices, dtype=torch.long, device=device)
    proxy_idx_all = torch.as_tensor(plan.proxy_unknown_indices, dtype=torch.long, device=device)
    y_by_global = torch.full((int(features_np.shape[0]),), -1, dtype=torch.long, device=device)
    y_by_global.index_copy_(0, source_idx, torch.as_tensor(source_y, dtype=torch.long, device=device))
    y_by_global.index_copy_(0, support_idx, torch.as_tensor(support_y, dtype=torch.long, device=device))
    prototypes = torch.as_tensor(proto_np, dtype=torch.float32, device=device)

    adapter = LowRankResidualAdapter(dim, int(args.adapter_rank), float(args.adapter_alpha), float(args.dropout)).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(args.seed))
    losses: list[float] = []
    start_time = time.perf_counter()
    source_count = int(source_idx.numel())
    support_count = int(support_idx.numel())
    proxy_count = int(proxy_idx_all.numel())
    batch = max(1, int(args.batch_size))

    for _epoch in range(int(args.adapter_epochs)):
        adapter.train()
        source_order = source_idx[torch.randperm(source_count, device=device)]
        support_order = support_idx[torch.randperm(support_count, device=device)]
        epoch_loss = 0.0
        steps = 0
        max_steps = max(math.ceil(source_count / batch), math.ceil(support_count / batch), 1)
        for step in range(max_steps):
            src_batch = source_order[(step * batch) % source_count : min((step * batch) % source_count + batch, source_count)]
            if src_batch.numel() == 0:
                src_batch = source_order[: min(batch, source_count)]
            sup_batch = support_order[(step * batch) % support_count : min((step * batch) % support_count + batch, support_count)]
            if sup_batch.numel() == 0:
                sup_batch = support_order[: min(batch, support_count)]
            proxy_take = min(max(int(src_batch.numel()), int(sup_batch.numel()), int(args.virtual_count)), proxy_count)
            proxy_np = rng.choice(proxy_count, size=proxy_take, replace=proxy_take > proxy_count)
            proxy_batch = proxy_idx_all[torch.as_tensor(proxy_np, dtype=torch.long, device=device)]

            z_src0 = x.index_select(0, src_batch)
            z_sup0 = x.index_select(0, sup_batch)
            z_proxy0 = x.index_select(0, proxy_batch)
            virtual0 = _virtual_boundary_samples(
                prototypes,
                z_proxy0,
                count=int(args.virtual_count),
                mix_low=float(args.virtual_mix_low),
                mix_high=float(args.virtual_mix_high),
                noise_scale=float(args.virtual_noise_scale),
            )

            z_src = adapter(z_src0)
            z_sup = adapter(z_sup0)
            z_proxy = adapter(z_proxy0)
            z_virtual = adapter(virtual0)

            src_y = y_by_global.index_select(0, src_batch)
            sup_y = y_by_global.index_select(0, sup_batch)
            src_logits = _proto_logits(z_src, prototypes.detach(), float(args.proto_temperature))
            sup_logits = _proto_logits(z_sup, prototypes.detach(), float(args.proto_temperature))
            proxy_logits = _proto_logits(z_proxy, prototypes.detach(), float(args.proto_temperature))
            virtual_logits = _proto_logits(z_virtual, prototypes.detach(), float(args.proto_temperature))
            base_src_logits = _proto_logits(z_src0, prototypes.detach(), float(args.proto_temperature)).detach()

            src_ce = F.cross_entropy(src_logits, src_y)
            sup_ce = F.cross_entropy(sup_logits, sup_y)
            old_preserve = F.kl_div(F.log_softmax(src_logits, dim=1), F.softmax(base_src_logits, dim=1), reduction="batchmean")
            source_margin = _known_margin_loss(src_logits, src_y, float(args.known_margin))
            support_margin = _known_margin_loss(sup_logits, sup_y, float(args.known_margin))
            proxy_open = F.softplus(proxy_logits.max(dim=1).values - float(args.proxy_open_margin)).mean()
            virtual_open = F.softplus(virtual_logits.max(dim=1).values - float(args.virtual_open_margin)).mean()
            support_compact = (1.0 - F.cosine_similarity(z_sup, prototypes.index_select(0, sup_y), dim=1)).mean()
            residual = ((z_src - z_src0) ** 2).mean() + ((z_sup - z_sup0) ** 2).mean() + ((z_proxy - z_proxy0) ** 2).mean()
            loss = (
                float(args.source_cls_weight) * src_ce
                + float(args.support_cls_weight) * sup_ce
                + float(args.old_preserve_weight) * old_preserve
                + float(args.known_margin_weight) * (source_margin + support_margin)
                + float(args.proxy_open_weight) * proxy_open
                + float(args.virtual_open_weight) * virtual_open
                + float(args.support_compact_weight) * support_compact
                + float(args.residual_weight) * residual
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
            opt.step()
            epoch_loss += float(loss.detach().item())
            steps += 1
        losses.append(epoch_loss / max(steps, 1))

    adapter.eval()
    with torch.no_grad():
        z_source = adapter(x.index_select(0, source_idx))
        z_support = adapter(x.index_select(0, support_idx))
        z_proxy = adapter(x.index_select(0, proxy_idx_all))
        source_y_t = y_by_global.index_select(0, source_idx)
        support_y_t = y_by_global.index_select(0, support_idx)
        source_pred_before = _proto_logits(x.index_select(0, source_idx), prototypes, float(args.proto_temperature)).argmax(dim=1)
        source_pred_after = _proto_logits(z_source, prototypes, float(args.proto_temperature)).argmax(dim=1)
        support_pred_before = _proto_logits(x.index_select(0, support_idx), prototypes, float(args.proto_temperature)).argmax(dim=1)
        support_pred_after = _proto_logits(z_support, prototypes, float(args.proto_temperature)).argmax(dim=1)
        source_logits_after = _proto_logits(z_source, prototypes, float(args.proto_temperature))
        support_logits_after = _proto_logits(z_support, prototypes, float(args.proto_temperature))
        proxy_max_before = _proto_logits(x.index_select(0, proxy_idx_all), prototypes, float(args.proto_temperature)).max(dim=1).values
        proxy_max_after = _proto_logits(z_proxy, prototypes, float(args.proto_temperature)).max(dim=1).values
        metrics = {
            "adapter_train_seconds": time.perf_counter() - start_time,
            "device": str(device),
            "final_loss": losses[-1] if losses else None,
            "loss_trace_tail": losses[-5:],
            "source_proto_acc_before": float((source_pred_before == source_y_t).float().mean().item()),
            "source_proto_acc_after": float((source_pred_after == source_y_t).float().mean().item()),
            "support_proto_acc_before": float((support_pred_before == support_y_t).float().mean().item()),
            "support_proto_acc_after": float((support_pred_after == support_y_t).float().mean().item()),
            "source_margin_after_mean": float((source_logits_after.gather(1, source_y_t.view(-1, 1)).squeeze(1) - source_logits_after.scatter(1, source_y_t.view(-1, 1), -1.0e9).max(dim=1).values).mean().item()),
            "support_margin_after_mean": float((support_logits_after.gather(1, support_y_t.view(-1, 1)).squeeze(1) - support_logits_after.scatter(1, support_y_t.view(-1, 1), -1.0e9).max(dim=1).values).mean().item()),
            "proxy_max_logit_before_mean": float(proxy_max_before.mean().item()),
            "proxy_max_logit_after_mean": float(proxy_max_after.mean().item()),
            "mean_source_residual_norm": float((z_source - x.index_select(0, source_idx)).norm(dim=1).mean().item()),
            "mean_support_residual_norm": float((z_support - x.index_select(0, support_idx)).norm(dim=1).mean().item()),
            "mean_proxy_residual_norm": float((z_proxy - x.index_select(0, proxy_idx_all)).norm(dim=1).mean().item()),
            "prototype_labels": proto_labels,
            "training_counts": {
                "source_old": len(plan.source_old_indices),
                "proxy_unknown": len(plan.proxy_unknown_indices),
                "target_support": len(plan.support_indices),
                "target_unknown_eval_only": len(plan.target_unknown_indices),
                "target_unknown_training_count": 0,
            },
            "state_bytes": _state_bytes(dim, int(args.adapter_rank), len(proto_labels)),
        }
    return adapter, metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--adapter_npz", type=Path, default=None)
    p.add_argument("--backend", default="both")
    p.add_argument("--collab_counts", default="all")
    p.add_argument("--collab_group_policy", default="same_max_budget", choices=["exact_k", "available_up_to_k", "same_max_budget"])
    p.add_argument("--partial_collab_min_receivers", type=int, default=1)
    p.add_argument("--k_shot", type=int, default=8)
    p.add_argument("--query_per_class", type=int, default=20)
    p.add_argument("--qknn_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=4070607)
    p.add_argument("--support_selection_policy", default="stable_first", choices=["stable_first", "centroid", "scenario_diverse", "strict_event_query_preserve"])
    p.add_argument("--event_alignment_policy", default="receiver_domain_ranked", choices=["strict_event_key", "receiver_domain_ranked"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--adapter_epochs", type=int, default=80)
    p.add_argument("--adapter_rank", type=int, default=8)
    p.add_argument("--adapter_alpha", type=float, default=0.35)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--proto_temperature", type=float, default=0.08)
    p.add_argument("--known_margin", type=float, default=2.5)
    p.add_argument("--proxy_open_margin", type=float, default=0.05)
    p.add_argument("--virtual_open_margin", type=float, default=0.02)
    p.add_argument("--virtual_count", type=int, default=128)
    p.add_argument("--virtual_mix_low", type=float, default=0.35)
    p.add_argument("--virtual_mix_high", type=float, default=0.65)
    p.add_argument("--virtual_noise_scale", type=float, default=0.015)
    p.add_argument("--source_cls_weight", type=float, default=1.0)
    p.add_argument("--support_cls_weight", type=float, default=1.2)
    p.add_argument("--old_preserve_weight", type=float, default=5.0)
    p.add_argument("--known_margin_weight", type=float, default=0.20)
    p.add_argument("--proxy_open_weight", type=float, default=1.4)
    p.add_argument("--virtual_open_weight", type=float, default=0.8)
    p.add_argument("--support_compact_weight", type=float, default=0.7)
    p.add_argument("--residual_weight", type=float, default=0.4)
    p.add_argument("--enpc_profiles", default="all")
    p.add_argument("--slev_profiles", default="all")
    p.add_argument("--slev_energy_support_quantile", type=float, default=0.90)
    p.add_argument("--slev_logit_temperature", type=float, default=1.0)
    p.add_argument("--slev_energy_risk_temperature", type=float, default=0.75)
    p.add_argument("--slev_energy_risk_margin", type=float, default=0.0)
    p.add_argument("--max_event_bytes", type=float, default=1152.0)
    p.add_argument("--max_event_latency_ms", type=float, default=20.0)
    p.add_argument("--target_old_acc", type=float, default=0.99)
    p.add_argument("--target_min_old", type=float, default=0.95)
    p.add_argument("--target_seen_new_acc", type=float, default=0.97)
    p.add_argument("--target_min_seen", type=float, default=0.93)
    p.add_argument("--target_unknown_reject", type=float, default=0.99)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_feature_npz(args.feature_npz)
    plan = build_training_plan(
        payload,
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
        support_selection_policy=str(args.support_selection_policy),
    )
    adapter, train_metrics = train_vra_adapter(payload, plan, args)
    adapted_features = apply_adapter(payload, adapter, str(args.device))
    adapter_npz = args.adapter_npz or (args.output_dir / "vra_ci_adapter_features.npz")
    metadata = {
        "algorithm": "VRA-CI",
        "adapter": "virtual_radius_low_rank_residual",
        "target_unknown_eval_only": True,
        "training_roles": ["source", "proxy_unknown", "target_old_support", "target_new_support", "virtual_boundary_negative"],
        "forbidden_roles": ["target_unknown"],
        "train_metrics": train_metrics,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    save_adapted_npz(args.feature_npz, adapter_npz, adapted_features, metadata)
    backend_results = run_backends(args, adapter_npz, args.output_dir)
    summary = {
        "algorithm": "VRA-CI",
        "feature_npz": str(args.feature_npz),
        "adapter_feature_npz": str(adapter_npz),
        "target_unknown_eval_only": True,
        "training_counts": train_metrics["training_counts"],
        "state_bytes": train_metrics["state_bytes"],
        "target_receivers": plan.target_receivers,
        "collab_counts_requested": str(args.collab_counts),
        "backend_results": {
            key: {
                "best_joint_row": value.get("best_joint_row"),
                "summary_row_count": len(value.get("summary_rows", [])),
                "target_gates": value.get("target_gates", {}),
            }
            for key, value in backend_results.items()
        },
        "adapter_metadata": metadata,
    }
    (args.output_dir / "vra_ci_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"adapter_feature_npz": str(adapter_npz), "target_receivers": plan.target_receivers}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
