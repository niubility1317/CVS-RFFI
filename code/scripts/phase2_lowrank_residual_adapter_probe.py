#!/usr/bin/env python3
"""Probe a support-only low-rank residual adapter for Phase2-C many-new.

The exported backbone features are frozen. The adapter is a small residual
metric layer fitted only from target support labels with a support leave-one-out
prototype loss; unused enrollment-pool labels may be used for model selection.
The deployed state stores adapter parameters and quantized support codes, not
raw support IQ.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import phase2_qknn_active_support_select as active
import phase2_source_guarded_qknn_sweep as qknn


def _rank_score(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    joint_ratio = min(
        float(row["old_acc"]) / 0.80,
        float(row["min_old_class_acc"]) / 0.75,
        float(row["seen_new_acc"]) / 0.75,
        float(row["min_seen_new_class_acc"]) / 0.75,
    )
    return (
        float(bool(row["passes_joint_target"])),
        joint_ratio,
        float(row["min_seen_new_class_acc"]),
        float(row["seen_new_acc"]),
        float(row["min_old_class_acc"]),
    )


class LowRankResidualAdapter(torch.nn.Module):
    def __init__(self, dim: int, rank: int, residual_scale: float) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.log_diag = torch.nn.Parameter(torch.zeros(dim))
        if int(rank) > 0:
            self.left = torch.nn.Parameter(torch.randn(dim, int(rank)) * 0.01)
            self.right = torch.nn.Parameter(torch.zeros(dim, int(rank)))
        else:
            self.left = None
            self.right = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diag = torch.clamp(self.log_diag, -0.25, 0.25).exp()
        out = x * diag
        if self.left is not None and self.right is not None:
            low = (x @ self.left) @ self.right.T
            out = out + self.residual_scale * low
        return F.normalize(out, dim=1)

    def reg_loss(self) -> torch.Tensor:
        reg = torch.mean(self.log_diag.square())
        if self.left is not None and self.right is not None:
            reg = reg + torch.mean(self.left.square()) + torch.mean(self.right.square())
        return reg


def _class_prototypes(z: torch.Tensor, y: torch.Tensor, class_count: int) -> torch.Tensor:
    protos = []
    for cls in range(class_count):
        mask = y == cls
        protos.append(F.normalize(z[mask].mean(dim=0, keepdim=True), dim=1)[0])
    return torch.stack(protos, dim=0)


def _support_loo_logits(z: torch.Tensor, y: torch.Tensor, class_count: int, temperature: float) -> torch.Tensor:
    sums = []
    counts = []
    for cls in range(class_count):
        mask = y == cls
        sums.append(z[mask].sum(dim=0))
        counts.append(torch.sum(mask).clamp_min(1))
    sums_t = torch.stack(sums, dim=0)
    counts_t = torch.stack(counts, dim=0).to(z.dtype)
    mean_t = F.normalize(sums_t / counts_t[:, None], dim=1)
    logits = z @ mean_t.T
    own_counts = counts_t[y].clamp_min(2.0) - 1.0
    own_proto = F.normalize((sums_t[y] - z) / own_counts[:, None], dim=1)
    own_logits = torch.sum(z * own_proto, dim=1)
    logits = logits.clone()
    logits[torch.arange(z.shape[0], device=z.device), y] = own_logits
    return logits * float(temperature)


def _collect_indices(
    old_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    new_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    old_labels: list[str],
    new_labels: list[str],
) -> tuple[list[int], list[str], list[int], list[int]]:
    support_indices: list[int] = []
    support_labels: list[str] = []
    old_query: list[int] = []
    new_query: list[int] = []
    for label in old_labels:
        support, query = old_splits[label]
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
        old_query.extend(query.tolist())
    for label in new_labels:
        support, query = new_splits[label]
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
        new_query.extend(query.tolist())
    return support_indices, support_labels, old_query, new_query


def _evaluate_adapter(
    *,
    features: np.ndarray,
    tx_ids: np.ndarray,
    scenarios: np.ndarray,
    adapter: LowRankResidualAdapter,
    support_indices: list[int],
    support_labels: list[str],
    old_query: list[int],
    new_query: list[int],
    old_labels: list[str],
    new_labels: list[str],
    topk: int,
    radius_norm: float,
    scenario_aware: bool,
    old_target: float,
    old_floor: float,
    new_target: float,
    new_floor: float,
) -> dict[str, Any]:
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32)
        adapted = adapter(x).cpu().numpy()
    bank = qknn._build_support_bank(
        adapted,
        support_indices,
        support_labels,
        set(old_labels),
        support_scenarios=scenarios[np.asarray(support_indices, dtype=int)] if scenario_aware else None,
    )
    query_idx = np.asarray(old_query + new_query, dtype=int)
    pred = qknn._predict_from_bank(
        bank,
        adapted[query_idx],
        topk=int(topk),
        old_bias=0.0,
        radius_norm=float(radius_norm),
        query_scenarios=scenarios[query_idx] if scenario_aware else None,
        scenario_aware=bool(scenario_aware),
    )
    truth = tx_ids[query_idx]
    old_count = len(old_query)
    old_pred = pred[:old_count]
    old_truth = truth[:old_count]
    new_pred = pred[old_count:]
    new_truth = truth[old_count:]
    per_old = {label: qknn._accuracy(old_pred[old_truth == label], old_truth[old_truth == label]) for label in old_labels}
    per_new = {label: qknn._accuracy(new_pred[new_truth == label], new_truth[new_truth == label]) for label in new_labels}
    old_acc = qknn._accuracy(old_pred, old_truth)
    new_acc = qknn._accuracy(new_pred, new_truth)
    min_old = min(per_old.values()) if per_old else 0.0
    min_new = min(per_new.values()) if per_new else 0.0
    return {
        "old_acc": old_acc,
        "min_old_class_acc": min_old,
        "seen_new_acc": new_acc,
        "min_seen_new_class_acc": min_new,
        "per_old_acc": per_old,
        "per_new_acc": per_new,
        "passes_goal_floor75": min_new >= float(new_floor),
        "passes_joint_target": old_acc >= float(old_target)
        and min_old >= float(old_floor)
        and new_acc >= float(new_target)
        and min_new >= float(new_floor),
        "old_query_count": int(len(old_query)),
        "new_query_count": int(len(new_query)),
    }


def _fit_adapter(
    *,
    features: np.ndarray,
    support_indices: list[int],
    support_labels: list[str],
    class_labels: list[str],
    rank: int,
    residual_scale: float,
    steps: int,
    lr: float,
    weight_decay: float,
    reg_weight: float,
    temperature: float,
    seed: int,
) -> tuple[LowRankResidualAdapter, list[dict[str, float]]]:
    torch.manual_seed(int(seed))
    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    support_x = torch.as_tensor(features[np.asarray(support_indices, dtype=int)], dtype=torch.float32)
    support_y = torch.as_tensor([label_to_idx[label] for label in support_labels], dtype=torch.long)
    adapter = LowRankResidualAdapter(dim=support_x.shape[1], rank=int(rank), residual_scale=float(residual_scale))
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    trace: list[dict[str, float]] = []
    for step in range(max(1, int(steps))):
        opt.zero_grad(set_to_none=True)
        z = adapter(support_x)
        logits = _support_loo_logits(z, support_y, len(class_labels), temperature=float(temperature))
        ce = F.cross_entropy(logits, support_y)
        loss = ce + float(reg_weight) * adapter.reg_loss()
        loss.backward()
        opt.step()
        if step == 0 or (step + 1) % max(1, int(steps) // 5) == 0 or step + 1 == int(steps):
            pred = torch.argmax(logits.detach(), dim=1)
            trace.append(
                {
                    "step": float(step + 1),
                    "support_loo_loss": float(loss.detach().cpu().item()),
                    "support_loo_acc": float(torch.mean((pred == support_y).float()).detach().cpu().item()),
                }
            )
    return adapter, trace


def _prefixed(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in row.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--new_tx_ids", required=True)
    parser.add_argument("--policies", default="stable_first")
    parser.add_argument("--ranks", default="0,4,8,16")
    parser.add_argument("--residual_scales", default="0.05,0.1,0.2")
    parser.add_argument("--steps_grid", default="100,200")
    parser.add_argument("--lr_grid", default="0.005,0.001")
    parser.add_argument("--reg_weights", default="0.001,0.01,0.1")
    parser.add_argument("--temperature", type=float, default=24.0)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--radius_norm", type=float, default=0.2)
    parser.add_argument("--seed_start", type=int, default=422947)
    parser.add_argument("--seed_count", type=int, default=1)
    parser.add_argument("--k_old", type=int, default=20)
    parser.add_argument("--k_new", type=int, default=20)
    parser.add_argument("--query_per_old", type=int, default=60)
    parser.add_argument("--query_per_new", type=int, default=60)
    parser.add_argument("--pool_per_old", type=int, default=50)
    parser.add_argument("--pool_per_new", type=int, default=50)
    parser.add_argument("--scenario_aware", action="store_true")
    parser.add_argument("--exclude_pool_from_query", action="store_true")
    parser.add_argument("--old_target", type=float, default=0.80)
    parser.add_argument("--old_floor", type=float, default=0.75)
    parser.add_argument("--seen_new_target", type=float, default=0.75)
    parser.add_argument("--seen_new_floor", type=float, default=0.75)
    args = parser.parse_args()

    data = np.load(Path(args.feature_npz), allow_pickle=True)
    features = qknn._normalize_rows(data["features"])
    tx_ids = np.asarray(data["tx_ids"], dtype=object).astype(str)
    roles = np.asarray(data["dataset_role"], dtype=object).astype(str)
    logits = np.asarray(data["tx_logits"], dtype=np.float64)
    scenarios = np.asarray(data["sat_scenarios"], dtype=object).astype(str)
    old_labels = qknn._parse_csv(args.old_tx_ids)
    new_labels = qknn._parse_csv(args.new_tx_ids)
    class_labels = old_labels + new_labels
    source_probs = active._softmax(logits)
    source_label_to_idx = {label: idx for idx, label in enumerate(old_labels)}
    source_prototypes: dict[str, np.ndarray] = {}
    for label in old_labels:
        source_idx = np.where((tx_ids == label) & (roles == "source"))[0].astype(int)
        if source_idx.size:
            source_prototypes[label] = qknn._normalize_rows(features[source_idx].mean(axis=0, keepdims=True))[0]

    rows: list[dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        for policy in qknn._parse_csv(args.policies):
            old_raw = active._build_active_splits(
                tx_ids=tx_ids,
                roles=roles,
                features=features,
                scenarios=scenarios,
                source_probs=source_probs,
                source_label_to_idx=source_label_to_idx,
                source_prototypes=source_prototypes,
                labels=old_labels,
                role="target_old",
                k=args.k_old,
                query_per_class=args.query_per_old,
                pool_per_class=args.pool_per_old,
                policy=policy,
                seed=seed,
                exclude_pool_from_query=bool(args.exclude_pool_from_query),
            )
            new_raw = active._build_active_splits(
                tx_ids=tx_ids,
                roles=roles,
                features=features,
                scenarios=scenarios,
                source_probs=source_probs,
                source_label_to_idx=source_label_to_idx,
                source_prototypes=source_prototypes,
                labels=new_labels,
                role="target_new",
                k=args.k_new,
                query_per_class=args.query_per_new,
                pool_per_class=args.pool_per_new,
                policy=policy,
                seed=seed,
                exclude_pool_from_query=bool(args.exclude_pool_from_query),
            )
            if set(old_raw) != set(old_labels) or set(new_raw) != set(new_labels):
                continue
            old_eval = active._as_eval_splits(old_raw)
            new_eval = active._as_eval_splits(new_raw)
            old_enroll = active._as_eval_splits(old_raw, use_enrollment_val=True)
            new_enroll = active._as_eval_splits(new_raw, use_enrollment_val=True)
            support_indices, support_labels, old_query, new_query = _collect_indices(old_eval, new_eval, old_labels, new_labels)
            _, _, old_enroll_query, new_enroll_query = _collect_indices(old_enroll, new_enroll, old_labels, new_labels)
            for rank in qknn._parse_int_csv(args.ranks):
                for residual_scale in qknn._parse_float_csv(args.residual_scales):
                    if int(rank) == 0 and float(residual_scale) != qknn._parse_float_csv(args.residual_scales)[0]:
                        continue
                    for steps in qknn._parse_int_csv(args.steps_grid):
                        for lr in qknn._parse_float_csv(args.lr_grid):
                            for reg_weight in qknn._parse_float_csv(args.reg_weights):
                                fit_seed = qknn._stable_seed(seed, f"{policy}:{rank}:{residual_scale}:{steps}:{lr}:{reg_weight}")
                                adapter, trace = _fit_adapter(
                                    features=features,
                                    support_indices=support_indices,
                                    support_labels=support_labels,
                                    class_labels=class_labels,
                                    rank=int(rank),
                                    residual_scale=float(residual_scale),
                                    steps=int(steps),
                                    lr=float(lr),
                                    weight_decay=0.0,
                                    reg_weight=float(reg_weight),
                                    temperature=float(args.temperature),
                                    seed=int(fit_seed),
                                )
                                enroll_row = _evaluate_adapter(
                                    features=features,
                                    tx_ids=tx_ids,
                                    scenarios=scenarios,
                                    adapter=adapter,
                                    support_indices=support_indices,
                                    support_labels=support_labels,
                                    old_query=old_enroll_query,
                                    new_query=new_enroll_query,
                                    old_labels=old_labels,
                                    new_labels=new_labels,
                                    topk=args.topk,
                                    radius_norm=args.radius_norm,
                                    scenario_aware=bool(args.scenario_aware),
                                    old_target=args.old_target,
                                    old_floor=args.old_floor,
                                    new_target=args.seen_new_target,
                                    new_floor=args.seen_new_floor,
                                )
                                query_row = _evaluate_adapter(
                                    features=features,
                                    tx_ids=tx_ids,
                                    scenarios=scenarios,
                                    adapter=adapter,
                                    support_indices=support_indices,
                                    support_labels=support_labels,
                                    old_query=old_query,
                                    new_query=new_query,
                                    old_labels=old_labels,
                                    new_labels=new_labels,
                                    topk=args.topk,
                                    radius_norm=args.radius_norm,
                                    scenario_aware=bool(args.scenario_aware),
                                    old_target=args.old_target,
                                    old_floor=args.old_floor,
                                    new_target=args.seen_new_target,
                                    new_floor=args.seen_new_floor,
                                )
                                row: dict[str, Any] = {
                                    "seed": int(seed),
                                    "support_selection_policy": policy,
                                    "rank": int(rank),
                                    "residual_scale": float(residual_scale),
                                    "steps": int(steps),
                                    "lr": float(lr),
                                    "reg_weight": float(reg_weight),
                                    "temperature": float(args.temperature),
                                    "topk": int(args.topk),
                                    "radius_norm": float(args.radius_norm),
                                    "scenario_aware": bool(args.scenario_aware),
                                    "k_old": int(args.k_old),
                                    "k_new": int(args.k_new),
                                    "pool_per_old": int(args.pool_per_old),
                                    "pool_per_new": int(args.pool_per_new),
                                    "adapter_param_scalars": int(features.shape[1] + 2 * features.shape[1] * int(rank)),
                                    "stored_quantized_support_code_count": int(len(support_indices)),
                                    "stored_raw_support_count": 0,
                                    "loss_trace": trace,
                                }
                                row.update(_prefixed("enroll_val", enroll_row))
                                row.update(_prefixed("query", query_row))
                                row["enroll_val_rank_score"] = _rank_score(enroll_row)
                                row["query_rank_score"] = _rank_score(query_row)
                                rows.append(row)

    rows.sort(
        key=lambda row: (
            row["query_min_seen_new_class_acc"],
            row["query_seen_new_acc"],
            row["query_min_old_class_acc"],
            row["query_old_acc"],
        ),
        reverse=True,
    )
    best_by_enroll = sorted(rows, key=lambda row: tuple(row["enroll_val_rank_score"]), reverse=True)
    summary = {
        "diagnostic_scope": "SUPPORT_ONLY_LOWRANK_RESIDUAL_ADAPTER_NO_QUERY_LABEL_FIT",
        "selection_note": "best_by_enrollment uses unused labeled enrollment-pool rows only; best_by_query is audit.",
        "feature_npz": str(args.feature_npz),
        "old_tx_ids": old_labels,
        "new_tx_ids": new_labels,
        "rows": rows,
        "best_by_query": rows[:20],
        "best_by_enrollment": best_by_enroll[:20],
    }
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "seed",
        "support_selection_policy",
        "rank",
        "residual_scale",
        "steps",
        "lr",
        "reg_weight",
        "topk",
        "radius_norm",
        "query_old_acc",
        "query_min_old_class_acc",
        "query_seen_new_acc",
        "query_min_seen_new_class_acc",
        "query_passes_goal_floor75",
        "query_passes_joint_target",
        "query_per_old_acc",
        "query_per_new_acc",
        "enroll_val_old_acc",
        "enroll_val_min_old_class_acc",
        "enroll_val_seen_new_acc",
        "enroll_val_min_seen_new_class_acc",
        "enroll_val_per_old_acc",
        "enroll_val_per_new_acc",
        "adapter_param_scalars",
        "stored_quantized_support_code_count",
        "stored_raw_support_count",
        "loss_trace",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fields}
            for key in ("query_per_old_acc", "query_per_new_acc", "enroll_val_per_old_acc", "enroll_val_per_new_acc", "loss_trace"):
                out[key] = json.dumps(row[key], ensure_ascii=False, sort_keys=True)
            writer.writerow(out)
    print(
        json.dumps(
            {
                "best_by_enrollment": best_by_enroll[:3],
                "best_by_query": rows[:3],
                "output_json": str(output_json),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
