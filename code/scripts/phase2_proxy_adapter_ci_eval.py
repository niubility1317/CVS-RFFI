#!/usr/bin/env python
"""OPR-CI feature-level proxy adapter for Stage2-C collaborative qknn8.

OPR-CI keeps the ADV3B02 feature extractor and qknn8 evidence stack frozen.
It trains a small residual adapter using only source old rows, source-side
proxy_unknown rows, and target old/seen-new K-shot support. target_unknown rows
remain evaluation-only and are never used for adapter fitting or threshold
selection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from phase2_collaborative_open_set_qknn_eval import (  # noqa: E402
    PROXY_UNKNOWN_ROLE,
    UNKNOWN_ROLE,
    _normalize_rows,
    _split_support_query_selected,
    canonical_tx_id,
    load_feature_npz,
    validate_required_roles,
)
from phase2_orbit_enpc_ci_eval import (  # noqa: E402
    _write_csv as _write_enpc_csv,
    parse_args as parse_enpc_args,
    run_enpc_ci,
)
from phase2_orbit_slev_ci_eval import (  # noqa: E402
    _write_csv as _write_slev_csv,
    parse_args as parse_slev_args,
    run_slev_ci,
)


@dataclass(frozen=True)
class AdapterTrainingPlan:
    source_old_indices: list[int]
    proxy_unknown_indices: list[int]
    support_indices: list[int]
    support_labels: list[str]
    target_unknown_indices: list[int]
    target_receivers: list[str]
    old_labels: list[str]
    seen_new_labels: list[str]
    unknown_labels: list[str]


class LowRankResidualAdapter(nn.Module):
    def __init__(self, dim: int, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, int(rank), bias=False)
        self.up = nn.Linear(int(rank), dim, bias=False)
        self.drop = nn.Dropout(float(dropout))
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.drop(F.gelu(self.down(self.norm(x)))))
        return F.normalize(x + self.alpha * delta, dim=1)


def _as_list(values: np.ndarray) -> list[str]:
    return [str(v) for v in np.asarray(values).astype(str).tolist()]


def build_training_plan(
    payload: Mapping[str, Any],
    *,
    k_shot: int,
    query_per_class: int,
    seed: int,
    support_selection_policy: str,
) -> AdapterTrainingPlan:
    validate_required_roles(payload)
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    features = np.asarray(payload["features"], dtype=np.float32)

    old_labels = sorted({str(tx_ids[i]) for i in np.where(roles == "target_old")[0].tolist()})
    seen_new_labels = sorted({str(tx_ids[i]) for i in np.where(roles == "target_new")[0].tolist()})
    unknown_labels = sorted({str(tx_ids[i]) for i in np.where(roles == UNKNOWN_ROLE)[0].tolist()})
    source_labels = sorted({str(tx_ids[i]) for i in np.where(roles == "source")[0].tolist()})
    label_overlaps = {
        "old_new": sorted(set(old_labels) & set(seen_new_labels)),
        "old_unknown": sorted(set(old_labels) & set(unknown_labels)),
        "new_unknown": sorted(set(seen_new_labels) & set(unknown_labels)),
    }
    if any(label_overlaps.values()):
        raise RuntimeError(f"LOCAL_PROTOCOL_REPAIR_REQUIRED: target TX sets overlap: {label_overlaps}")
    old_not_source = sorted(set(old_labels) - set(source_labels))
    non_old_in_source = sorted((set(seen_new_labels) | set(unknown_labels)) & set(source_labels))
    if old_not_source or non_old_in_source:
        raise RuntimeError(
            "LOCAL_PROTOCOL_REPAIR_REQUIRED: Stage2-C TX split violates source/target semantics; "
            f"old_not_in_source={old_not_source}, non_old_in_source={non_old_in_source}"
        )

    target_mask = np.isin(roles, ["target_old", "target_new", UNKNOWN_ROLE])
    target_receivers = sorted({str(rx_ids[i]) for i in np.where(target_mask)[0].tolist()})
    source_receivers = sorted({str(rx_ids[i]) for i in np.where(roles == "source")[0].tolist()})
    overlap_receivers = sorted(set(target_receivers) & set(source_receivers))
    if overlap_receivers:
        raise RuntimeError(f"LOCAL_PROTOCOL_REPAIR_REQUIRED: R_s and R_t overlap: {overlap_receivers}")

    proxy_indices = np.where(roles == PROXY_UNKNOWN_ROLE)[0].astype(int).tolist()
    if not proxy_indices:
        raise RuntimeError("LOCAL_DATASET_EXTENSION_REQUIRED: OPR-CI requires source-side proxy_unknown rows")
    proxy_labels = sorted({str(tx_ids[i]) for i in proxy_indices})
    proxy_receivers = sorted({str(rx_ids[i]) for i in proxy_indices})
    proxy_label_overlap = sorted(set(proxy_labels) & (set(old_labels) | set(seen_new_labels) | set(unknown_labels)))
    proxy_receiver_overlap = sorted(set(proxy_receivers) & set(target_receivers))
    if proxy_label_overlap or proxy_receiver_overlap:
        raise RuntimeError(
            "LOCAL_PROTOCOL_REPAIR_REQUIRED: proxy_unknown rows must be source-side and label-disjoint; "
            f"label_overlap={proxy_label_overlap}, receiver_overlap={proxy_receiver_overlap}"
        )

    support_indices: list[int] = []
    support_labels: list[str] = []
    for rx in target_receivers:
        for role, labels in (("target_old", old_labels), ("target_new", seen_new_labels)):
            for label in labels:
                support, query = _split_support_query_selected(
                    payload,
                    features=features,
                    role=role,
                    tx_id=label,
                    rx_id=rx,
                    k_shot=int(k_shot),
                    query_per_class=int(query_per_class),
                    seed=int(seed),
                    support_selection_policy=support_selection_policy,
                )
                if len(support) < int(k_shot) or len(query) < int(query_per_class):
                    raise RuntimeError(
                        "LOCAL_DATASET_EXTENSION_REQUIRED: incomplete Stage2-C known support/query; "
                        f"receiver={rx}, role={role}, tx_id={label}, "
                        f"support={len(support)}/{int(k_shot)}, query={len(query)}/{int(query_per_class)}"
                    )
                support_indices.extend(int(i) for i in support)
                support_labels.extend([label] * len(support))

    source_old_indices = [
        int(i)
        for i in np.where(roles == "source")[0].tolist()
        if str(tx_ids[int(i)]) in set(old_labels)
    ]
    target_unknown_indices = np.where(roles == UNKNOWN_ROLE)[0].astype(int).tolist()
    leak = sorted(set(target_unknown_indices) & (set(source_old_indices) | set(proxy_indices) | set(support_indices)))
    if leak:
        raise RuntimeError(f"LOCAL_PROTOCOL_REPAIR_REQUIRED: target_unknown leaked into adapter training: {leak[:5]}")

    return AdapterTrainingPlan(
        source_old_indices=source_old_indices,
        proxy_unknown_indices=proxy_indices,
        support_indices=support_indices,
        support_labels=support_labels,
        target_unknown_indices=target_unknown_indices,
        target_receivers=target_receivers,
        old_labels=old_labels,
        seen_new_labels=seen_new_labels,
        unknown_labels=unknown_labels,
    )


def _build_prototypes(features: np.ndarray, labels: Sequence[str]) -> tuple[list[str], np.ndarray]:
    label_values = sorted({str(v) for v in labels})
    protos = []
    arr_labels = np.asarray([str(v) for v in labels])
    for label in label_values:
        protos.append(_normalize_rows(features[arr_labels == label].mean(axis=0, keepdims=True))[0])
    return label_values, _normalize_rows(np.vstack(protos).astype(np.float32))


def _proto_logits(x: torch.Tensor, prototypes: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.normalize(x.float(), dim=1) @ F.normalize(prototypes.float(), dim=1).t() / max(float(temperature), 1e-6)


def _state_bytes(dim: int, rank: int, class_count: int) -> dict[str, int]:
    adapter_params = int(dim) * int(rank) * 2
    adapter_fp16_bytes = adapter_params * 2
    verifier_fp16_bytes = (int(dim) + 1) * 2
    prototype_fp16_bytes = int(class_count) * int(dim) * 2
    return {
        "adapter_params": adapter_params,
        "adapter_fp16_bytes": adapter_fp16_bytes,
        "verifier_fp16_bytes": verifier_fp16_bytes,
        "prototype_fp16_bytes": prototype_fp16_bytes,
        "total_fp16_state_bytes": adapter_fp16_bytes + verifier_fp16_bytes + prototype_fp16_bytes,
    }


def train_adapter(
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

            src_ce = F.cross_entropy(src_logits, src_y)
            sup_ce = F.cross_entropy(sup_logits, sup_y)
            old_preserve = F.kl_div(
                F.log_softmax(src_logits, dim=1),
                F.softmax(base_src_logits, dim=1),
                reduction="batchmean",
            )
            proxy_open = F.softplus(proxy_logits.max(dim=1).values - float(args.proxy_open_margin)).mean()
            support_compact = (1.0 - F.cosine_similarity(z_sup, prototypes.index_select(0, sup_y), dim=1)).mean()
            residual = ((z_src - z_src0) ** 2).mean() + ((z_sup - z_sup0) ** 2).mean() + ((z_proxy - z_proxy0) ** 2).mean()
            loss = (
                float(args.source_cls_weight) * src_ce
                + float(args.support_cls_weight) * sup_ce
                + float(args.old_preserve_weight) * old_preserve
                + float(args.proxy_open_weight) * proxy_open
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


@torch.no_grad()
def apply_adapter(payload: Mapping[str, Any], adapter: nn.Module, device: str) -> np.ndarray:
    requested_device = str(device)
    torch_device = torch.device(requested_device if torch.cuda.is_available() or not requested_device.startswith("cuda") else "cpu")
    adapter = adapter.to(torch_device)
    adapter.eval()
    features = torch.as_tensor(np.asarray(payload["features"], dtype=np.float32), dtype=torch.float32, device=torch_device)
    adapted = []
    for start in range(0, int(features.shape[0]), 2048):
        adapted.append(adapter(features[start : start + 2048]).detach().cpu().numpy())
    return _normalize_rows(np.vstack(adapted).astype(np.float32))


def save_adapted_npz(src_path: Path, dst_path: Path, adapted_features: np.ndarray, metadata: Mapping[str, Any]) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(src_path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files if key != "features"}
    manifest = {}
    if "manifest_json" in arrays:
        try:
            manifest = json.loads(str(np.asarray(arrays["manifest_json"]).item()))
        except Exception:
            manifest = {}
    manifest["opr_ci_adapter"] = dict(metadata)
    arrays["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=False), dtype=object)
    arrays["features"] = np.asarray(adapted_features, dtype=np.float32)
    np.savez_compressed(dst_path, **arrays)


def _backend_common_argv(args: argparse.Namespace, feature_npz: Path, output_dir: Path, name: str) -> list[str]:
    return [
        "--feature_npz",
        str(feature_npz),
        "--output_json",
        str(output_dir / f"{name}.json"),
        "--output_summary_csv",
        str(output_dir / f"{name}_summary.csv"),
        "--output_evidence_csv",
        str(output_dir / f"{name}_evidence.csv"),
        "--collab_counts",
        str(args.collab_counts),
        "--collab_group_policy",
        str(args.collab_group_policy),
        "--partial_collab_min_receivers",
        str(args.partial_collab_min_receivers),
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
        "--event_alignment_policy",
        str(args.event_alignment_policy),
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


def run_backends(args: argparse.Namespace, adapted_npz: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = {part.strip().lower() for part in str(args.backend).split(",") if part.strip()}
    results: dict[str, Any] = {}
    if "both" in requested:
        requested.update({"enpc", "slev"})
    if "enpc" in requested:
        argv = ["--profiles", str(args.enpc_profiles), *_backend_common_argv(args, adapted_npz, output_dir, "opr_ci_enpc")]
        enpc_args = parse_enpc_args(argv)
        result = run_enpc_ci(enpc_args)
        evidence = result.pop("_evidence_rows")
        Path(enpc_args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        _write_enpc_csv(enpc_args.output_summary_csv, result["summary_rows"])
        _write_enpc_csv(enpc_args.output_evidence_csv, evidence)
        results["enpc"] = result
    if "slev" in requested:
        argv = [
            "--profiles",
            str(args.slev_profiles),
            "--slev_energy_support_quantile",
            str(args.slev_energy_support_quantile),
            "--slev_logit_temperature",
            str(args.slev_logit_temperature),
            "--slev_energy_risk_temperature",
            str(args.slev_energy_risk_temperature),
            "--slev_energy_risk_margin",
            str(args.slev_energy_risk_margin),
            *_backend_common_argv(args, adapted_npz, output_dir, "opr_ci_slev"),
        ]
        slev_args = parse_slev_args(argv)
        result = run_slev_ci(slev_args)
        evidence = result.pop("_evidence_rows")
        Path(slev_args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        _write_slev_csv(slev_args.output_summary_csv, result["summary_rows"])
        _write_slev_csv(slev_args.output_evidence_csv, evidence)
        results["slev"] = result
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--adapter_npz", type=Path, default=None)
    p.add_argument("--backend", default="both", help="Comma list: enpc,slev,both")
    p.add_argument("--collab_counts", default="all")
    p.add_argument("--collab_group_policy", default="same_max_budget", choices=["exact_k", "available_up_to_k", "same_max_budget"])
    p.add_argument("--partial_collab_min_receivers", type=int, default=1)
    p.add_argument("--k_shot", type=int, default=8)
    p.add_argument("--query_per_class", type=int, default=20)
    p.add_argument("--qknn_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=4070404)
    p.add_argument("--support_selection_policy", default="stable_first", choices=["stable_first", "centroid", "scenario_diverse", "strict_event_query_preserve"])
    p.add_argument("--event_alignment_policy", default="receiver_domain_ranked", choices=["strict_event_key", "receiver_domain_ranked"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--adapter_epochs", type=int, default=40)
    p.add_argument("--adapter_rank", type=int, default=16)
    p.add_argument("--adapter_alpha", type=float, default=0.20)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--proto_temperature", type=float, default=0.08)
    p.add_argument("--proxy_open_margin", type=float, default=0.20)
    p.add_argument("--source_cls_weight", type=float, default=1.0)
    p.add_argument("--support_cls_weight", type=float, default=1.0)
    p.add_argument("--old_preserve_weight", type=float, default=2.0)
    p.add_argument("--proxy_open_weight", type=float, default=1.0)
    p.add_argument("--support_compact_weight", type=float, default=0.5)
    p.add_argument("--residual_weight", type=float, default=0.10)
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
    adapter, train_metrics = train_adapter(payload, plan, args)
    adapted = apply_adapter(payload, adapter, str(args.device))
    adapted_npz = args.adapter_npz or (args.output_dir / "opr_ci_adapted_features.npz")
    adapter_metadata = {
        "algorithm": "OPR-CI",
        "adapter": "low_rank_residual_feature_adapter",
        "target_unknown_eval_only": True,
        "training_roles": ["source", PROXY_UNKNOWN_ROLE, "target_old_support", "target_new_support"],
        "forbidden_roles": [UNKNOWN_ROLE],
        "plan": asdict(plan),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_metrics": train_metrics,
    }
    save_adapted_npz(args.feature_npz, adapted_npz, adapted, adapter_metadata)
    backend_results = run_backends(args, adapted_npz, args.output_dir)
    summary = {
        "algorithm": "OPR-CI",
        "feature_npz": str(args.feature_npz),
        "adapted_feature_npz": str(adapted_npz),
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
        "adapter_metadata": adapter_metadata,
    }
    (args.output_dir / "opr_ci_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"adapted_feature_npz": str(adapted_npz), "target_receivers": plan.target_receivers}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
