#!/usr/bin/env python
"""SOM-CI source-only open metric evaluation for Stage2-C qknn8.

SOM-CI approximates a ground-side open-set representation repair on exported
ADV3B02 features. It learns a compact diagonal metric using only source old
features and source-side proxy_unknown features. Target old/seen-new support is
used later only by the Stage2-C qknn8 backends, and target_unknown rows remain
evaluation-only.
"""

from __future__ import annotations

import argparse
import json
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
from phase2_proxy_adapter_ci_eval import build_training_plan, run_backends, save_adapted_npz  # noqa: E402

SOURCE_ROLE = "source"
PROXY_UNKNOWN_ROLE = "proxy_unknown"
TARGET_UNKNOWN_ROLE = "target_unknown"


def _build_prototypes(features: np.ndarray, labels: Sequence[str]) -> tuple[list[str], np.ndarray]:
    label_arr = np.asarray([str(v) for v in labels], dtype=object)
    label_values = sorted({str(v) for v in label_arr.tolist()})
    protos = []
    for label in label_values:
        protos.append(_normalize_rows(features[label_arr == label].mean(axis=0, keepdims=True))[0])
    return label_values, _normalize_rows(np.vstack(protos).astype(np.float32))


def _metric_transform_t(x: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    scale = torch.exp(0.5 * torch.clamp(log_scale, -4.0, 4.0))
    return F.normalize(x * scale.view(1, -1), dim=1)


def _metric_transform_np(features: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
    scale = np.exp(0.5 * np.clip(np.asarray(log_scale, dtype=np.float32), -4.0, 4.0))
    return _normalize_rows(np.asarray(features, dtype=np.float32) * scale[None, :])


def _proto_logits(x: torch.Tensor, prototypes: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.normalize(x.float(), dim=1) @ F.normalize(prototypes.float(), dim=1).t() / max(float(temperature), 1e-6)


def _known_margin_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.view(-1, 1), -1.0e9)
    second = masked.max(dim=1).values
    return F.relu(float(margin) - (true_logits - second)).mean()


def _virtual_source_negatives(
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
    first = torch.randint(0, class_count, (n,), device=device)
    if class_count > 1:
        second = torch.randint(0, class_count - 1, (n,), device=device)
        second = second + (second >= first).long()
        known = F.normalize(0.5 * prototypes.index_select(0, first) + 0.5 * prototypes.index_select(0, second), dim=1)
    else:
        known = prototypes.index_select(0, first)
    proxy = proxy_features[torch.randint(0, int(proxy_features.shape[0]), (n,), device=device)]
    alpha = torch.empty((n, 1), device=device).uniform_(float(mix_low), float(mix_high))
    x = (1.0 - alpha) * known + alpha * proxy
    if float(noise_scale) > 0.0:
        x = x + torch.randn_like(x) * float(noise_scale)
    return F.normalize(x, dim=1)


def train_source_metric(payload: Mapping[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    requested_device = str(args.device)
    device = torch.device(requested_device if torch.cuda.is_available() or not requested_device.startswith("cuda") else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    features_np = _normalize_rows(np.asarray(payload["features"], dtype=np.float32))
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    source_idx_np = np.where(roles == SOURCE_ROLE)[0].astype(int)
    proxy_idx_np = np.where(roles == PROXY_UNKNOWN_ROLE)[0].astype(int)
    target_unknown_idx_np = np.where(roles == TARGET_UNKNOWN_ROLE)[0].astype(int)
    if source_idx_np.size == 0 or proxy_idx_np.size == 0:
        raise RuntimeError("LOCAL_DATASET_EXTENSION_REQUIRED: SOM-CI requires source and proxy_unknown rows")
    source_labels = [str(tx_ids[i]) for i in source_idx_np.tolist()]
    proto_labels, proto_np = _build_prototypes(features_np[source_idx_np], source_labels)
    label_to_pos = {label: i for i, label in enumerate(proto_labels)}
    source_y_np = np.asarray([label_to_pos[str(tx_ids[i])] for i in source_idx_np.tolist()], dtype=np.int64)
    x = torch.as_tensor(features_np, dtype=torch.float32, device=device)
    source_idx = torch.as_tensor(source_idx_np, dtype=torch.long, device=device)
    proxy_idx = torch.as_tensor(proxy_idx_np, dtype=torch.long, device=device)
    source_y = torch.as_tensor(source_y_np, dtype=torch.long, device=device)
    prototypes_base = torch.as_tensor(proto_np, dtype=torch.float32, device=device)
    log_scale = torch.zeros((features_np.shape[1],), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.AdamW([log_scale], lr=float(args.lr), weight_decay=float(args.weight_decay))
    start = time.perf_counter()
    losses: list[float] = []
    for _epoch in range(int(args.metric_epochs)):
        z_source = _metric_transform_t(x.index_select(0, source_idx), log_scale)
        z_proxy = _metric_transform_t(x.index_select(0, proxy_idx), log_scale)
        z_proto = _metric_transform_t(prototypes_base, log_scale)
        virtual = _virtual_source_negatives(
            z_proto.detach(),
            z_proxy.detach(),
            count=int(args.virtual_count),
            mix_low=float(args.virtual_mix_low),
            mix_high=float(args.virtual_mix_high),
            noise_scale=float(args.virtual_noise_scale),
        )
        source_logits = _proto_logits(z_source, z_proto.detach(), float(args.proto_temperature))
        proxy_logits = _proto_logits(z_proxy, z_proto.detach(), float(args.proto_temperature))
        virtual_logits = _proto_logits(virtual, z_proto.detach(), float(args.proto_temperature))
        base_source_logits = _proto_logits(x.index_select(0, source_idx), prototypes_base.detach(), float(args.proto_temperature)).detach()
        source_ce = F.cross_entropy(source_logits, source_y)
        source_margin = _known_margin_loss(source_logits, source_y, float(args.known_margin))
        old_preserve = F.kl_div(F.log_softmax(source_logits, dim=1), F.softmax(base_source_logits, dim=1), reduction="batchmean")
        source_compact = (1.0 - F.cosine_similarity(z_source, z_proto.index_select(0, source_y), dim=1)).mean()
        proxy_open = F.softplus(proxy_logits.max(dim=1).values - float(args.proxy_open_margin)).mean()
        virtual_open = F.softplus(virtual_logits.max(dim=1).values - float(args.virtual_open_margin)).mean()
        metric_reg = (log_scale**2).mean()
        loss = (
            float(args.source_cls_weight) * source_ce
            + float(args.known_margin_weight) * source_margin
            + float(args.old_preserve_weight) * old_preserve
            + float(args.source_compact_weight) * source_compact
            + float(args.proxy_open_weight) * proxy_open
            + float(args.virtual_open_weight) * virtual_open
            + float(args.metric_reg_weight) * metric_reg
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([log_scale], float(args.grad_clip))
        opt.step()
        losses.append(float(loss.detach().item()))
    with torch.no_grad():
        z_source = _metric_transform_t(x.index_select(0, source_idx), log_scale)
        z_proxy = _metric_transform_t(x.index_select(0, proxy_idx), log_scale)
        z_proto = _metric_transform_t(prototypes_base, log_scale)
        source_pred_before = _proto_logits(x.index_select(0, source_idx), prototypes_base, float(args.proto_temperature)).argmax(dim=1)
        source_pred_after = _proto_logits(z_source, z_proto, float(args.proto_temperature)).argmax(dim=1)
        proxy_max_before = _proto_logits(x.index_select(0, proxy_idx), prototypes_base, float(args.proto_temperature)).max(dim=1).values
        proxy_max_after = _proto_logits(z_proxy, z_proto, float(args.proto_temperature)).max(dim=1).values
        log_scale_cpu = log_scale.detach().cpu()
        log_scale_np = log_scale_cpu.numpy().astype(np.float32)
        metrics = {
            "metric_train_seconds": time.perf_counter() - start,
            "device": str(device),
            "final_loss": losses[-1] if losses else None,
            "loss_trace_tail": losses[-5:],
            "source_proto_acc_before": float((source_pred_before == source_y).float().mean().item()),
            "source_proto_acc_after": float((source_pred_after == source_y).float().mean().item()),
            "proxy_max_logit_before_mean": float(proxy_max_before.mean().item()),
            "proxy_max_logit_after_mean": float(proxy_max_after.mean().item()),
            "log_scale_min": float(log_scale_cpu.min().item()),
            "log_scale_max": float(log_scale_cpu.max().item()),
            "log_scale_std": float(log_scale_cpu.std(unbiased=False).item()),
            "prototype_labels": proto_labels,
            "training_counts": {
                "source_old": int(source_idx_np.size),
                "proxy_unknown": int(proxy_idx_np.size),
                "target_support": 0,
                "target_unknown_eval_only": int(target_unknown_idx_np.size),
                "target_unknown_training_count": 0,
            },
            "state_bytes": {
                "metric_dim": int(features_np.shape[1]),
                "metric_fp16_state_bytes": int(features_np.shape[1]) * 2,
                "source_prototype_fp16_bytes": int(len(proto_labels) * features_np.shape[1] * 2),
                "total_fp16_state_bytes": int(features_np.shape[1] * 2 + len(proto_labels) * features_np.shape[1] * 2),
            },
        }
    return log_scale_np, metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--metric_npz", type=Path, default=None)
    p.add_argument("--backend", default="both")
    p.add_argument("--collab_counts", default="all")
    p.add_argument("--collab_group_policy", default="same_max_budget", choices=["exact_k", "available_up_to_k", "same_max_budget"])
    p.add_argument("--partial_collab_min_receivers", type=int, default=1)
    p.add_argument("--k_shot", type=int, default=8)
    p.add_argument("--query_per_class", type=int, default=20)
    p.add_argument("--qknn_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=4070701)
    p.add_argument("--support_selection_policy", default="stable_first", choices=["stable_first", "centroid", "scenario_diverse", "strict_event_query_preserve"])
    p.add_argument("--event_alignment_policy", default="receiver_domain_ranked", choices=["strict_event_key", "receiver_domain_ranked"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--metric_epochs", type=int, default=160)
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--proto_temperature", type=float, default=0.08)
    p.add_argument("--known_margin", type=float, default=2.0)
    p.add_argument("--proxy_open_margin", type=float, default=0.15)
    p.add_argument("--virtual_open_margin", type=float, default=0.10)
    p.add_argument("--virtual_count", type=int, default=512)
    p.add_argument("--virtual_mix_low", type=float, default=0.35)
    p.add_argument("--virtual_mix_high", type=float, default=0.65)
    p.add_argument("--virtual_noise_scale", type=float, default=0.01)
    p.add_argument("--source_cls_weight", type=float, default=1.0)
    p.add_argument("--known_margin_weight", type=float, default=0.1)
    p.add_argument("--old_preserve_weight", type=float, default=5.0)
    p.add_argument("--source_compact_weight", type=float, default=0.5)
    p.add_argument("--proxy_open_weight", type=float, default=1.0)
    p.add_argument("--virtual_open_weight", type=float, default=0.5)
    p.add_argument("--metric_reg_weight", type=float, default=0.05)
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
    log_scale, train_metrics = train_source_metric(payload, args)
    adapted_features = _metric_transform_np(np.asarray(payload["features"], dtype=np.float32), log_scale)
    metric_npz = args.metric_npz or (args.output_dir / "som_ci_metric_features.npz")
    metadata = {
        "algorithm": "SOM-CI",
        "adapter": "source_only_open_metric",
        "target_unknown_eval_only": True,
        "training_roles": ["source", "proxy_unknown", "virtual_source_negative"],
        "forbidden_roles": ["target_old", "target_new", "target_unknown"],
        "log_scale": log_scale.tolist(),
        "train_metrics": train_metrics,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    save_adapted_npz(args.feature_npz, metric_npz, adapted_features, metadata)
    backend_results = run_backends(args, metric_npz, args.output_dir)
    summary = {
        "algorithm": "SOM-CI",
        "feature_npz": str(args.feature_npz),
        "metric_feature_npz": str(metric_npz),
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
        "metric_metadata": metadata,
    }
    (args.output_dir / "som_ci_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"metric_feature_npz": str(metric_npz), "target_receivers": plan.target_receivers}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
