#!/usr/bin/env python
"""SAGE-OSR feature-level open-set repair for Stage2-C collaborative inference.

SAGE-OSR trains a lightweight low-rank residual adapter on frozen ADV3B02
features. Training uses source-old rows, target old/seen-new K-shot support and
source-side proxy_unknown hard negatives. target_unknown rows remain evaluation
only. The adapted feature package is evaluated by AWARE-CI/qknn8 collaborative
inference over collab_count=1..N.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
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

from phase2_aware_ci_eval import _write_csv as _write_aware_csv  # noqa: E402
from phase2_aware_ci_eval import parse_args as parse_aware_args  # noqa: E402
from phase2_aware_ci_eval import run_aware_ci  # noqa: E402
from phase2_collaborative_open_set_qknn_eval import PROXY_UNKNOWN_ROLE, UNKNOWN_ROLE, _normalize_rows, load_feature_npz  # noqa: E402
from phase2_proxy_adapter_ci_eval import (  # noqa: E402
    LowRankResidualAdapter,
    _build_prototypes,
    _proto_logits,
    _state_bytes,
    apply_adapter,
    build_training_plan,
    save_adapted_npz,
)


def _resolve_device(requested: str) -> torch.device:
    return torch.device(str(requested) if torch.cuda.is_available() or not str(requested).startswith("cuda") else "cpu")


def _label_positions(labels: Sequence[str]) -> dict[str, int]:
    return {str(label): int(i) for i, label in enumerate(labels)}


def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * torch.log(torch.clamp(probs, min=1e-8))).sum(dim=1)
    return -entropy.mean()


def train_sage_adapter(payload: Mapping[str, Any], plan: Any, args: argparse.Namespace) -> tuple[LowRankResidualAdapter, dict[str, Any]]:
    device = _resolve_device(str(args.device))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    features_np = _normalize_rows(np.asarray(payload["features"], dtype=np.float32))
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    dim = int(features_np.shape[1])
    source_labels = [str(tx_ids[i]) for i in plan.source_old_indices]
    support_labels = [str(v) for v in plan.support_labels]
    proto_labels, proto_np = _build_prototypes(
        features_np[np.asarray(plan.source_old_indices + plan.support_indices, dtype=int)],
        source_labels + support_labels,
    )
    label_to_pos = _label_positions(proto_labels)
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
    batch = max(1, int(args.batch_size))
    source_count = int(source_idx.numel())
    support_count = int(support_idx.numel())
    proxy_count = int(proxy_idx_all.numel())
    losses: list[float] = []
    component_tail: list[dict[str, float]] = []
    start_time = time.perf_counter()

    for _epoch in range(int(args.adapter_epochs)):
        adapter.train()
        source_order = source_idx[torch.randperm(source_count, device=device)]
        support_order = support_idx[torch.randperm(support_count, device=device)]
        max_steps = max(math.ceil(source_count / batch), math.ceil(support_count / batch), 1)
        total_loss = 0.0
        total_components: dict[str, float] = {
            "src_ce": 0.0,
            "sup_ce": 0.0,
            "old_distill": 0.0,
            "proxy_hard_open": 0.0,
            "proxy_entropy": 0.0,
            "support_compact": 0.0,
            "support_margin": 0.0,
            "residual": 0.0,
        }
        for step in range(max_steps):
            src_batch = source_order[(step * batch) % source_count : min((step * batch) % source_count + batch, source_count)]
            if src_batch.numel() == 0:
                src_batch = source_order[: min(batch, source_count)]
            sup_batch = support_order[(step * batch) % support_count : min((step * batch) % support_count + batch, support_count)]
            if sup_batch.numel() == 0:
                sup_batch = support_order[: min(batch, support_count)]
            proxy_take = min(max(int(src_batch.numel()), int(sup_batch.numel())), proxy_count)
            proxy_np = rng.choice(proxy_count, size=proxy_take, replace=proxy_take > proxy_count)
            proxy_batch = proxy_idx_all[torch.as_tensor(proxy_np, dtype=torch.long, device=device)]

            z_src0 = x.index_select(0, src_batch)
            z_sup0 = x.index_select(0, sup_batch)
            z_proxy0 = x.index_select(0, proxy_batch)
            z_src = adapter(z_src0)
            z_sup = adapter(z_sup0)
            z_proxy = adapter(z_proxy0)
            src_y = y_by_global.index_select(0, src_batch)
            sup_y = y_by_global.index_select(0, sup_batch)

            src_logits = _proto_logits(z_src, prototypes.detach(), float(args.proto_temperature))
            sup_logits = _proto_logits(z_sup, prototypes.detach(), float(args.proto_temperature))
            proxy_logits = _proto_logits(z_proxy, prototypes.detach(), float(args.proto_temperature))
            base_src_logits = _proto_logits(z_src0, prototypes.detach(), float(args.proto_temperature)).detach()
            base_sup_logits = _proto_logits(z_sup0, prototypes.detach(), float(args.proto_temperature)).detach()

            src_ce = F.cross_entropy(src_logits, src_y)
            sup_ce = F.cross_entropy(sup_logits, sup_y)
            old_distill = F.kl_div(
                F.log_softmax(src_logits, dim=1),
                F.softmax(base_src_logits, dim=1),
                reduction="batchmean",
            )
            old_distill = old_distill + 0.5 * F.kl_div(
                F.log_softmax(sup_logits, dim=1),
                F.softmax(base_sup_logits, dim=1),
                reduction="batchmean",
            )
            proxy_max = proxy_logits.max(dim=1).values
            hard_k = max(1, int(math.ceil(float(args.proxy_hard_fraction) * int(proxy_max.numel()))))
            hard_proxy = torch.topk(proxy_max, k=min(hard_k, int(proxy_max.numel()))).values
            proxy_hard_open = F.softplus(hard_proxy - float(args.proxy_open_margin)).mean()
            proxy_entropy = _entropy_loss(proxy_logits)
            support_compact = (1.0 - F.cosine_similarity(z_sup, prototypes.index_select(0, sup_y), dim=1)).mean()
            true_sup = sup_logits.gather(1, sup_y.view(-1, 1)).squeeze(1)
            masked = sup_logits.scatter(1, sup_y.view(-1, 1), -1e6)
            support_margin = F.relu(float(args.support_margin_target) - (true_sup - masked.max(dim=1).values)).mean()
            residual = ((z_src - z_src0) ** 2).mean() + ((z_sup - z_sup0) ** 2).mean() + ((z_proxy - z_proxy0) ** 2).mean()
            loss = (
                float(args.source_cls_weight) * src_ce
                + float(args.support_cls_weight) * sup_ce
                + float(args.old_distill_weight) * old_distill
                + float(args.proxy_open_weight) * proxy_hard_open
                + float(args.proxy_entropy_weight) * proxy_entropy
                + float(args.support_compact_weight) * support_compact
                + float(args.support_margin_weight) * support_margin
                + float(args.residual_weight) * residual
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
            opt.step()
            total_loss += float(loss.detach().item())
            for key, value in (
                ("src_ce", src_ce),
                ("sup_ce", sup_ce),
                ("old_distill", old_distill),
                ("proxy_hard_open", proxy_hard_open),
                ("proxy_entropy", proxy_entropy),
                ("support_compact", support_compact),
                ("support_margin", support_margin),
                ("residual", residual),
            ):
                total_components[key] += float(value.detach().item())
        losses.append(total_loss / max_steps)
        component_tail.append({key: value / max_steps for key, value in total_components.items()})

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
        proxy_max_before = _proto_logits(x.index_select(0, proxy_idx_all), prototypes, float(args.proto_temperature)).max(dim=1).values
        proxy_max_after = _proto_logits(z_proxy, prototypes, float(args.proto_temperature)).max(dim=1).values
        metrics = {
            "algorithm": "SAGE-OSR",
            "adapter_train_seconds": float(time.perf_counter() - start_time),
            "device": str(device),
            "final_loss": losses[-1] if losses else None,
            "loss_trace_tail": losses[-5:],
            "component_trace_tail": component_tail[-5:],
            "source_proto_acc_before": float((source_pred_before == source_y_t).float().mean().item()),
            "source_proto_acc_after": float((source_pred_after == source_y_t).float().mean().item()),
            "support_proto_acc_before": float((support_pred_before == support_y_t).float().mean().item()),
            "support_proto_acc_after": float((support_pred_after == support_y_t).float().mean().item()),
            "proxy_max_logit_before_mean": float(proxy_max_before.mean().item()),
            "proxy_max_logit_after_mean": float(proxy_max_after.mean().item()),
            "proxy_max_logit_before_q90": float(torch.quantile(proxy_max_before, 0.90).item()),
            "proxy_max_logit_after_q90": float(torch.quantile(proxy_max_after, 0.90).item()),
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


def _aware_argv(args: argparse.Namespace, feature_npz: Path, output_dir: Path) -> list[str]:
    return [
        "--feature_npz",
        str(feature_npz),
        "--output_json",
        str(output_dir / "sage_osr_aware.json"),
        "--output_summary_csv",
        str(output_dir / "sage_osr_aware_summary.csv"),
        "--output_evidence_csv",
        str(output_dir / "sage_osr_aware_evidence.csv"),
        "--profiles",
        str(args.aware_profiles),
        "--collab_counts",
        str(args.collab_counts),
        "--collab_group_policy",
        str(args.collab_group_policy),
        "--k_shot",
        str(args.k_shot),
        "--query_per_class",
        str(args.query_per_class),
        "--qknn_k",
        str(args.qknn_k),
        "--seed",
        str(args.seed),
        "--support_selection_policy",
        str(args.support_selection_policy),
        "--component_known_quantile",
        str(args.component_known_quantile),
        "--proto_weight",
        str(args.aware_proto_weight),
        "--knn_weight",
        str(args.aware_knn_weight),
        "--maha_weight",
        str(args.aware_maha_weight),
        "--entropy_weight",
        str(args.aware_entropy_weight),
        "--proxy_weight",
        str(args.aware_proxy_weight),
        "--maha_var_floor",
        str(args.maha_var_floor),
        "--evidence_packet_bytes",
        str(args.evidence_packet_bytes),
        "--receiver_latency_ms",
        str(args.receiver_latency_ms),
        "--max_event_bytes",
        str(args.max_event_bytes),
        "--max_event_latency_ms",
        str(args.max_event_latency_ms),
        "--target_old_acc",
        str(args.target_old_acc),
        "--target_min_old",
        str(args.target_min_old),
        "--target_seen_new_acc",
        str(args.target_seen_new_acc),
        "--target_min_seen",
        str(args.target_min_seen),
        "--target_unknown_reject",
        str(args.target_unknown_reject),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--adapter_npz", type=Path, default=None)
    p.add_argument("--collab_counts", default="all")
    p.add_argument("--collab_group_policy", default="same_max_budget", choices=["exact_k", "available_up_to_k", "same_max_budget"])
    p.add_argument("--k_shot", type=int, default=8)
    p.add_argument("--query_per_class", type=int, default=12)
    p.add_argument("--qknn_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=4070706)
    p.add_argument("--support_selection_policy", default="stable_first", choices=["stable_first", "centroid", "scenario_diverse", "strict_event_query_preserve"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--adapter_epochs", type=int, default=60)
    p.add_argument("--adapter_rank", type=int, default=12)
    p.add_argument("--adapter_alpha", type=float, default=0.10)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--proto_temperature", type=float, default=0.08)
    p.add_argument("--proxy_open_margin", type=float, default=0.05)
    p.add_argument("--proxy_hard_fraction", type=float, default=0.35)
    p.add_argument("--support_margin_target", type=float, default=1.0)
    p.add_argument("--source_cls_weight", type=float, default=1.0)
    p.add_argument("--support_cls_weight", type=float, default=1.5)
    p.add_argument("--old_distill_weight", type=float, default=5.0)
    p.add_argument("--proxy_open_weight", type=float, default=2.0)
    p.add_argument("--proxy_entropy_weight", type=float, default=0.25)
    p.add_argument("--support_compact_weight", type=float, default=0.75)
    p.add_argument("--support_margin_weight", type=float, default=0.25)
    p.add_argument("--residual_weight", type=float, default=0.50)
    p.add_argument("--aware_profiles", default="all")
    p.add_argument("--component_known_quantile", type=float, default=0.95)
    p.add_argument("--aware_proto_weight", type=float, default=0.25)
    p.add_argument("--aware_knn_weight", type=float, default=0.25)
    p.add_argument("--aware_maha_weight", type=float, default=0.30)
    p.add_argument("--aware_entropy_weight", type=float, default=0.10)
    p.add_argument("--aware_proxy_weight", type=float, default=0.10)
    p.add_argument("--maha_var_floor", type=float, default=1e-3)
    p.add_argument("--evidence_packet_bytes", type=float, default=128.0)
    p.add_argument("--receiver_latency_ms", type=float, default=0.45)
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
    adapter, train_metrics = train_sage_adapter(payload, plan, args)
    adapted = apply_adapter(payload, adapter, str(args.device))
    adapted_npz = args.adapter_npz or (args.output_dir / "sage_osr_adapted_features.npz")
    metadata = {
        "algorithm": "SAGE-OSR",
        "adapter": "low_rank_residual_hard_proxy_open_set_adapter",
        "target_unknown_eval_only": True,
        "training_roles": ["source", PROXY_UNKNOWN_ROLE, "target_old_support", "target_new_support"],
        "forbidden_roles": [UNKNOWN_ROLE],
        "plan": asdict(plan),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_metrics": train_metrics,
    }
    save_adapted_npz(args.feature_npz, adapted_npz, adapted, metadata)
    aware_args = parse_aware_args(_aware_argv(args, adapted_npz, args.output_dir))
    aware_result = run_aware_ci(aware_args)
    Path(aware_args.output_json).write_text(json.dumps(aware_result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if aware_args.output_summary_csv:
        _write_aware_csv(aware_args.output_summary_csv, aware_result["summary_rows"])
    summary = {
        "algorithm": "SAGE-OSR",
        "feature_npz": str(args.feature_npz),
        "adapted_feature_npz": str(adapted_npz),
        "target_unknown_eval_only": True,
        "training_counts": train_metrics["training_counts"],
        "state_bytes": train_metrics["state_bytes"],
        "target_receivers": plan.target_receivers,
        "collab_counts_requested": str(args.collab_counts),
        "adapter_metadata": metadata,
        "aware_best_joint_row": aware_result.get("best_joint_row"),
        "aware_summary_row_count": len(aware_result.get("summary_rows", [])),
        "aware_target_gates": aware_result.get("target_gates", {}),
    }
    (args.output_dir / "sage_osr_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"adapted_feature_npz": str(adapted_npz), "aware_best_joint_row": summary["aware_best_joint_row"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
