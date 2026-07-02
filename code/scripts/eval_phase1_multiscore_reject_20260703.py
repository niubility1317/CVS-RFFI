#!/usr/bin/env python
"""Evaluate deployable multi-score rejection from existing V3 score tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


KEY_FIELDS = ["group", "role", "tx_id", "rx_id", "day_id", "sig_id"]
COMPONENT_PATHS = {
    "lin": "ADAPT3_LIN_SRC9999/score_table.csv",
    "mlp": "ADAPT3_MLP64_SRC9999/score_table.csv",
    "pcos": "ADAPT3_PROTO_COS_SRC9999/score_table.csv",
    "pmah": "ADAPT3_PROTO_MAH_MIN05/score_table.csv",
}
COMPONENT_SETS = {
    "lin_mlp": ["lin", "mlp"],
    "lin_pmah": ["lin", "pmah"],
    "mlp_pmah": ["mlp", "pmah"],
    "lin_mlp_pmah": ["lin", "mlp", "pmah"],
    "all4": ["lin", "mlp", "pcos", "pmah"],
}


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_rate(num: int, den: int) -> float:
    return float("nan") if den <= 0 else float(num) / float(den)


def _load_score_table(path: Path, component: str) -> dict[tuple[str, ...], dict]:
    rows: dict[tuple[str, ...], dict] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = tuple(str(row[field]) for field in KEY_FIELDS)
            rows[key] = {
                **{field: str(row[field]) for field in KEY_FIELDS},
                "is_known_query": row.get("is_known_query", "0"),
                "is_unknown_query": row.get("is_unknown_query", "0"),
                "closed_correct_known": row.get("closed_correct_known", "0"),
                f"score_{component}": float(row["unknown_score"]),
            }
    return rows


def _merge(adapter_dir: Path, components: list[str]) -> list[dict]:
    merged: dict[tuple[str, ...], dict] | None = None
    for comp in components:
        rows = _load_score_table(adapter_dir / COMPONENT_PATHS[comp], comp)
        if merged is None:
            merged = rows
        else:
            common = sorted(set(merged) & set(rows))
            merged = {
                key: {**merged[key], f"score_{comp}": rows[key][f"score_{comp}"]}
                for key in common
            }
    if not merged:
        raise ValueError(f"empty merged score table for {adapter_dir}")
    return [merged[key] for key in sorted(merged)]


def _rank_features(values: np.ndarray, source_mask: np.ndarray) -> np.ndarray:
    out = []
    for j in range(values.shape[1]):
        src = np.sort(values[source_mask, j].astype(np.float64))
        if src.size == 0:
            raise ValueError("empty source mask for rank normalization")
        out.append(np.searchsorted(src, values[:, j], side="right") / float(src.size))
    return np.stack(out, axis=1).astype(np.float32)


class TinyRejector(nn.Module):
    def __init__(self, dim: int, mode: str, hidden: int = 16) -> None:
        super().__init__()
        if mode == "linear":
            self.net = nn.Linear(dim, 1)
        elif mode == "mlp":
            self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:
            raise ValueError(f"unknown model={mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


def _train_scores(x: np.ndarray, train_mask: np.ndarray, y: np.ndarray, *, model_kind: str, seed: int, epochs: int, lr: float) -> np.ndarray:
    torch.manual_seed(int(seed))
    x_t = torch.as_tensor(x, dtype=torch.float32)
    train_idx = torch.as_tensor(np.where(train_mask)[0], dtype=torch.long)
    y_t = torch.as_tensor(y[train_mask], dtype=torch.float32)
    model = TinyRejector(x.shape[1], model_kind)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    pos = float((y_t == 1).sum().item())
    neg = float((y_t == 0).sum().item())
    pos_weight = torch.tensor(max(1.0, neg / max(pos, 1.0)), dtype=torch.float32)
    for _ in range(int(epochs)):
        logits = model(x_t.index_select(0, train_idx))
        loss = F.binary_cross_entropy_with_logits(logits, y_t, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        values = torch.sigmoid(model(x_t)).detach().cpu().tolist()
        return np.asarray(values, dtype=np.float64)


def _threshold(scores: np.ndarray, source_mask: np.ndarray, proxy_mask: np.ndarray, policy: str, source_q: float, proxy_q: float) -> tuple[float, float, float]:
    source_t = float(np.quantile(scores[source_mask], float(source_q)))
    proxy_t = float(np.quantile(scores[proxy_mask], float(proxy_q)))
    if policy == "source_accept":
        t = source_t
    elif policy == "min_source_proxy":
        t = min(source_t, proxy_t)
    elif policy == "mean_source_proxy":
        t = 0.5 * (source_t + proxy_t)
    else:
        raise ValueError(f"unknown threshold policy={policy}")
    return float(t), source_t, proxy_t


def _metrics(scores: np.ndarray, threshold: float, known: np.ndarray, unknown: np.ndarray, closed_correct: np.ndarray) -> dict:
    accepted = scores <= float(threshold)
    known_total = int(known.sum())
    unknown_total = int(unknown.sum())
    known_closed_correct = int((known & closed_correct).sum())
    known_correct_after = int((known & closed_correct & accepted).sum())
    known_accepted = int((known & accepted).sum())
    unknown_accepted = int((unknown & accepted).sum())
    closed_acc = _safe_rate(known_closed_correct, known_total)
    full_acc = _safe_rate(known_correct_after, known_total)
    old_drop = 100.0 * (closed_acc - full_acc)
    far = _safe_rate(unknown_accepted, unknown_total)
    return {
        "unknown_FAR": far,
        "known_closed_accuracy_no_reject": closed_acc,
        "known_full_accuracy_after_reject": full_acc,
        "old_drop_pp_vs_closed": old_drop,
        "passes_unknown_far_target": bool(far <= 0.05),
        "passes_old_drop_target": bool(old_drop <= 2.0),
        "passes_dual_target": bool(far <= 0.05 and old_drop <= 2.0),
        "known_coverage": _safe_rate(known_accepted, known_total),
        "known_accepted_accuracy": _safe_rate(known_correct_after, known_accepted),
        "known_query_count": known_total,
        "unknown_query_count": unknown_total,
    }


def _oracle_threshold(scores: np.ndarray, known: np.ndarray, unknown: np.ndarray, closed_correct: np.ndarray) -> dict:
    finite = np.unique(scores[np.isfinite(scores)])
    thresholds = np.concatenate(([np.nextafter(float(finite[0]), -np.inf)], finite, [np.nextafter(float(finite[-1]), np.inf)]))
    best = None
    for t in thresholds:
        m = _metrics(scores, float(t), known, unknown, closed_correct)
        m["threshold"] = float(t)
        score = max(0.0, m["unknown_FAR"] - 0.05) * 100.0 + max(0.0, m["old_drop_pp_vs_closed"] - 2.0)
        m["oracle_joint_score"] = score
        if best is None or (score, m["unknown_FAR"], m["old_drop_pp_vs_closed"]) < (best["oracle_joint_score"], best["unknown_FAR"], best["old_drop_pp_vs_closed"]):
            best = m
    return best or {}


def _run_one(run_dir: Path, adapter: str, component_set: str, model_kind: str, seed: int, epochs: int) -> list[dict]:
    components = COMPONENT_SETS[component_set]
    rows = _merge(run_dir / adapter, components)
    values = np.asarray([[float(row[f"score_{c}"]) for c in components] for row in rows], dtype=np.float32)
    roles = np.asarray([str(row["role"]) for row in rows])
    source_mask = roles == "source"
    proxy_mask = roles == "proxy_unknown"
    known = np.asarray([_truthy(row["is_known_query"]) for row in rows], dtype=bool)
    unknown = np.asarray([_truthy(row["is_unknown_query"]) for row in rows], dtype=bool)
    closed = np.asarray([_truthy(row["closed_correct_known"]) for row in rows], dtype=bool)
    x = _rank_features(values, source_mask)

    y_source_proxy = np.zeros(len(rows), dtype=np.float32)
    y_source_proxy[proxy_mask] = 1.0
    train_source_proxy = source_mask | proxy_mask
    source_scores = _train_scores(x, train_source_proxy, y_source_proxy, model_kind=model_kind, seed=seed, epochs=epochs, lr=0.02)

    y_oracle = np.ones(len(rows), dtype=np.float32)
    oracle_accept = known & closed
    oracle_train = (known | unknown)
    y_oracle[oracle_accept] = 0.0
    oracle_scores = _train_scores(x, oracle_train, y_oracle, model_kind=model_kind, seed=seed + 1009, epochs=epochs, lr=0.02)

    out = []
    for policy in ["source_accept", "min_source_proxy", "mean_source_proxy"]:
        for source_q in [0.99, 0.995, 0.999, 0.9999]:
            proxy_qs = [0.05] if policy == "source_accept" else [0.03, 0.05, 0.10, 0.15]
            for proxy_q in proxy_qs:
                t, source_t, proxy_t = _threshold(source_scores, source_mask, proxy_mask, policy, source_q, proxy_q)
                m = _metrics(source_scores, t, known, unknown, closed)
                m.update({
                    "run_id": run_dir.name,
                    "adapter": adapter,
                    "component_set": component_set,
                    "model_kind": model_kind,
                    "mode": "source_proxy_train",
                    "threshold_policy": policy,
                    "source_accept_quantile": source_q,
                    "proxy_far_quantile": proxy_q,
                    "threshold": t,
                    "source_threshold": source_t,
                    "proxy_threshold": proxy_t,
                })
                out.append(m)
    oracle = _oracle_threshold(oracle_scores, known, unknown, closed)
    oracle.update({
        "run_id": run_dir.name,
        "adapter": adapter,
        "component_set": component_set,
        "model_kind": model_kind,
        "mode": "target_label_oracle_multiscore",
        "threshold_policy": "target_label_oracle",
        "source_accept_quantile": "",
        "proxy_far_quantile": "",
        "source_threshold": "",
        "proxy_threshold": "",
    })
    out.append(oracle)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--run_glob", default="phase1_adv3b02_multiview_keepold_*_20260702")
    parser.add_argument("--adapters", default="LEOADAPT3_LINR_COS,LEOADAPT3_MLP_ID")
    parser.add_argument("--component_sets", default="lin_mlp_pmah,all4")
    parser.add_argument("--model_kinds", default="linear,mlp")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=4070323)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapters = [x.strip() for x in str(args.adapters).split(",") if x.strip()]
    component_sets = [x.strip() for x in str(args.component_sets).split(",") if x.strip()]
    model_kinds = [x.strip() for x in str(args.model_kinds).split(",") if x.strip()]
    rows = []
    for run_dir in sorted(args.runs_root.glob(str(args.run_glob))):
        for adapter in adapters:
            if not (run_dir / adapter).is_dir():
                continue
            for component_set in component_sets:
                for model_kind in model_kinds:
                    rows.extend(_run_one(run_dir, adapter, component_set, model_kind, int(args.seed), int(args.epochs)))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["run_id", "adapter"]
    leading = ["run_id", "adapter", "component_set", "model_kind", "mode", "threshold_policy"]
    fields = leading + [f for f in fields if f not in leading]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    dual = sum(1 for row in rows if bool(row.get("passes_dual_target")))
    print({"rows": len(rows), "dual_pass": dual, "out_csv": str(args.out_csv)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
