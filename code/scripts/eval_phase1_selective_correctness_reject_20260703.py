#!/usr/bin/env python
"""Source-only selective-correctness gate for Phase1 sat-only rejection.

This V14 diagnostic keeps the Phase1 backbone frozen and reuses source-only LEO
repair adapters. Unlike an oldness gate, the training target is whether a source
LEO sample is correctly classified by the repaired old-class prototype ensemble.
Proxy unknown rows and source misclassified rows are negatives. Target labels are
used only for final metric accounting.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_phase1_repair_ensemble_manifold_reject_20260703 import (
    canonical_tx_id,
    fit_adapter_model,
    grouped_rows,
    load_npz,
    metrics,
    parse_csv_list,
    parse_tx_ids,
    rank_scale,
    score_one,
)


class TinyGate(nn.Module):
    def __init__(self, dim: int, kind: str, hidden: int) -> None:
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(dim, 1)
        elif kind == "mlp":
            self.net = nn.Sequential(
                nn.Linear(dim, int(hidden)),
                nn.ReLU(),
                nn.Linear(int(hidden), int(hidden)),
                nn.ReLU(),
                nn.Linear(int(hidden), 1),
            )
        else:
            raise ValueError(kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


def train_gate(x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, kind: str, hidden: int, epochs: int, seed: int) -> np.ndarray:
    torch.manual_seed(int(seed))
    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y[train_mask], dtype=torch.float32)
    idx = torch.as_tensor(np.where(train_mask)[0], dtype=torch.long)
    model = TinyGate(x.shape[1], kind, hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01 if kind == "linear" else 0.004, weight_decay=1.0e-4)
    pos = float((y_t == 1).sum().item())
    neg = float((y_t == 0).sum().item())
    pos_weight = torch.tensor(max(1.0, neg / max(pos, 1.0)), dtype=torch.float32)
    for _ in range(int(epochs)):
        logits = model(x_t.index_select(0, idx))
        loss = F.binary_cross_entropy_with_logits(logits, y_t, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    with torch.no_grad():
        return np.asarray(torch.sigmoid(model(x_t)).cpu().tolist(), dtype=np.float64)


def build_records(run_dir: Path, adapters: list[str], args: argparse.Namespace, source_tx_ids: list[str]) -> list[dict]:
    per_adapter = {}
    models = {}
    common: set[tuple[str, str, str, str, str]] | None = None
    for adapter in adapters:
        npz_path = run_dir / adapter / str(args.feature_relpath)
        if not npz_path.is_file():
            return []
        rows = grouped_rows(load_npz(npz_path))
        per_adapter[adapter] = rows
        models[adapter] = fit_adapter_model(rows, source_tx_ids, float(args.var_eps))
        keys = set(rows)
        common = keys if common is None else common & keys
    if not common:
        return []

    records = []
    for key in sorted(common):
        base = per_adapter[adapters[0]][key]
        role = base["role"]
        tx = canonical_tx_id(base["tx_id"])
        preds = []
        preds_mah = []
        max_sims = []
        sim_margins = []
        min_mahs = []
        mah_margins = []
        for adapter in adapters:
            scored = score_one(per_adapter[adapter][key]["feature"], models[adapter])
            preds.append(scored["pred_tx"])
            preds_mah.append(scored["pred_tx_mah"])
            max_sims.append(scored["max_sim"])
            sim_margins.append(scored["sim_margin"])
            min_mahs.append(scored["min_mah"])
            mah_margins.append(scored["mah_margin"])
        pred_tx, pred_count = Counter(preds).most_common(1)[0]
        pred_mah_tx, pred_mah_count = Counter(preds_mah).most_common(1)[0]
        records.append({
            "role": role,
            "tx_id": tx,
            "rx_id": base["rx_id"],
            "day_id": base["day_id"],
            "sig_id": base["sig_id"],
            "pred_tx": pred_tx,
            "pred_tx_mah": pred_mah_tx,
            "vote_agreement": pred_count / float(len(adapters)),
            "mah_vote_agreement": pred_mah_count / float(len(adapters)),
            "mean_max_sim": float(np.mean(max_sims)),
            "mean_sim_margin": float(np.mean(sim_margins)),
            "mean_min_mah": float(np.mean(min_mahs)),
            "mean_mah_margin": float(np.mean(mah_margins)),
            "std_max_sim": float(np.std(max_sims)),
            "std_min_mah": float(np.std(min_mahs)),
            "is_known_query": role == "target_old" and tx in source_tx_ids,
            "is_unknown_query": role == "target_unknown",
            "closed_correct_known": role == "target_old" and tx == pred_tx,
            "source_correct": role == "source" and tx == pred_tx,
        })
    return records


def feature_matrix(records: list[dict], train_fit_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    raw = {
        "mean_max_sim": np.asarray([r["mean_max_sim"] for r in records], dtype=np.float64),
        "mean_sim_margin": np.asarray([r["mean_sim_margin"] for r in records], dtype=np.float64),
        "neg_mean_min_mah": -np.asarray([r["mean_min_mah"] for r in records], dtype=np.float64),
        "mean_mah_margin": np.asarray([r["mean_mah_margin"] for r in records], dtype=np.float64),
        "vote_agreement": np.asarray([r["vote_agreement"] for r in records], dtype=np.float64),
        "mah_vote_agreement": np.asarray([r["mah_vote_agreement"] for r in records], dtype=np.float64),
        "neg_std_max_sim": -np.asarray([r["std_max_sim"] for r in records], dtype=np.float64),
        "neg_std_min_mah": -np.asarray([r["std_min_mah"] for r in records], dtype=np.float64),
    }
    scaled = {name: rank_scale(values, train_fit_mask) for name, values in raw.items()}
    scaled["repair_ensemble_oldness"] = (
        0.25 * scaled["mean_max_sim"]
        + 0.20 * scaled["mean_sim_margin"]
        + 0.25 * scaled["neg_mean_min_mah"]
        + 0.10 * scaled["mean_mah_margin"]
        + 0.10 * scaled["vote_agreement"]
        + 0.10 * scaled["mah_vote_agreement"]
    )
    names = [
        "mean_max_sim",
        "mean_sim_margin",
        "neg_mean_min_mah",
        "mean_mah_margin",
        "vote_agreement",
        "mah_vote_agreement",
        "neg_std_max_sim",
        "neg_std_min_mah",
        "repair_ensemble_oldness",
    ]
    return np.stack([scaled[name] for name in names], axis=1).astype(np.float32), names


def evaluate_run(run_dir: Path, adapters: list[str], args: argparse.Namespace, source_tx_ids: list[str]) -> list[dict]:
    records = build_records(run_dir, adapters, args, source_tx_ids)
    if not records:
        return []
    roles = np.asarray([r["role"] for r in records])
    source = roles == "source"
    proxy = roles == "proxy_unknown"
    known = np.asarray([bool(r["is_known_query"]) for r in records], dtype=bool)
    unknown = np.asarray([bool(r["is_unknown_query"]) for r in records], dtype=bool)
    closed = np.asarray([bool(r["closed_correct_known"]) for r in records], dtype=bool)
    source_correct = np.asarray([bool(r["source_correct"]) for r in records], dtype=bool)
    if not source.any() or not proxy.any() or not known.any() or not unknown.any() or not source_correct.any():
        return []

    train_fit = source | proxy
    x, feature_names = feature_matrix(records, train_fit)
    y = np.zeros(len(records), dtype=np.float32)
    y[source_correct] = 1.0
    train_mask = source | proxy
    target_rxs = sorted({r["rx_id"] for r in records if r["role"] in {"target_old", "target_unknown"}})
    out = []
    for model_kind in parse_csv_list(args.model_kinds):
        for seed_offset in range(int(args.seed_count)):
            scores = train_gate(
                x,
                y,
                train_mask,
                kind=model_kind,
                hidden=int(args.hidden_dim),
                epochs=int(args.epochs),
                seed=int(args.seed) + 1009 * seed_offset + (0 if model_kind == "linear" else 17),
            )
            for policy in ["source_correct_accept", "proxy_far", "max_source_proxy", "mean_source_proxy"]:
                for source_q in [0.001, 0.005, 0.010, 0.020, 0.050]:
                    proxy_qs = [0.95] if policy == "source_correct_accept" else [0.85, 0.90, 0.95, 0.97, 0.99]
                    for proxy_q in proxy_qs:
                        source_t = float(np.quantile(scores[source_correct], source_q))
                        proxy_t = float(np.quantile(scores[proxy], proxy_q))
                        if policy == "source_correct_accept":
                            threshold = source_t
                        elif policy == "proxy_far":
                            threshold = proxy_t
                        elif policy == "max_source_proxy":
                            threshold = max(source_t, proxy_t)
                        elif policy == "mean_source_proxy":
                            threshold = 0.5 * (source_t + proxy_t)
                        else:
                            raise ValueError(policy)
                        accept = scores >= threshold
                        m = metrics(accept, known, unknown, closed)
                        m.update({
                            "run_id": run_dir.name,
                            "target_rx": ",".join(target_rxs),
                            "mode": "selective_correctness_gate",
                            "adapter_set": ",".join(adapters),
                            "adapter_count": len(adapters),
                            "model_kind": model_kind,
                            "seed_offset": seed_offset,
                            "feature_names": ",".join(feature_names),
                            "score_name": "correctness_prob",
                            "threshold_policy": policy,
                            "source_accept_quantile": source_q,
                            "proxy_far_quantile": proxy_q,
                            "threshold": threshold,
                            "source_threshold": source_t,
                            "proxy_threshold": proxy_t,
                            "source_group_count": int(source.sum()),
                            "source_correct_group_count": int(source_correct.sum()),
                            "proxy_group_count": int(proxy.sum()),
                        })
                        out.append(m)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--metrics_json", type=Path, default=None)
    p.add_argument("--run_glob", default="phase1_adv3b02_multiview_keepold_*_20260702")
    p.add_argument("--adapters", default="LEOADAPT3_LINR_COS,LEOADAPT3_MLP_ID")
    p.add_argument("--feature_relpath", default="features_leo_repaired.npz")
    p.add_argument("--source_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    p.add_argument("--model_kinds", default="linear,mlp")
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=220)
    p.add_argument("--seed_count", type=int, default=2)
    p.add_argument("--seed", type=int, default=4070367)
    p.add_argument("--var_eps", type=float, default=1.0e-4)
    p.add_argument("--run_tag", default="V14_selective_correctness")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    adapters = parse_csv_list(args.adapters)
    source_tx_ids = parse_tx_ids(args.source_tx_ids)
    rows = []
    for run_dir in sorted(args.runs_root.glob(str(args.run_glob))):
        rows.extend(evaluate_run(run_dir, adapters, args, source_tx_ids))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ["run_id"]
    leading = ["run_id", "target_rx", "mode", "model_kind", "score_name", "threshold_policy"]
    fields = leading + [f for f in fields if f not in leading]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "phase": "phase1_selective_correctness_reject_v14",
        "run_tag": str(args.run_tag),
        "rows": len(rows),
        "dual_pass": int(sum(1 for r in rows if bool(r.get("passes_dual_target")))),
        "out_csv": str(args.out_csv),
        "adapters": adapters,
        "model_kinds": parse_csv_list(args.model_kinds),
        "seed_count": int(args.seed_count),
        "uses_target_clean": False,
        "uses_target_labels_for_threshold": False,
        "uses_unknown_query_for_threshold": False,
    }
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
