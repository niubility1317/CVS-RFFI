#!/usr/bin/env python
"""Train a source-only K+1 open-set head on repaired Phase1 features."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def canonical_tx_id(value: object) -> str:
    text = str(value)
    if text.startswith("tx"):
        text = text[2:]
    return text.replace("_", "-")


def parse_tx_ids(text: str) -> list[str]:
    return [canonical_tx_id(x.strip()) for x in str(text).split(",") if x.strip()]


def _as_str(data, key: str, n: int) -> np.ndarray:
    if key not in data.files:
        return np.asarray([""] * n, dtype=str)
    arr = np.asarray(data[key])
    if arr.shape[0] != n:
        raise ValueError(f"{key} length mismatch")
    return arr.astype(str)


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        logits = np.asarray(data["tx_logits"], dtype=np.float32)
        n = int(features.shape[0])
        return {
            "features": features,
            "tx_logits": logits,
            "dataset_role": _as_str(data, "dataset_role", n),
            "tx_ids": _as_str(data, "tx_ids", n),
            "rx_ids": _as_str(data, "rx_ids", n),
            "day_ids": _as_str(data, "day_ids", n),
            "sig_ids": _as_str(data, "sig_ids", n),
        }


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)


def _group_payload(payload: dict, source_tx_ids: list[str]) -> tuple[list[dict], np.ndarray]:
    grouped: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)
    for i in range(payload["features"].shape[0]):
        key = (
            str(payload["dataset_role"][i]),
            canonical_tx_id(payload["tx_ids"][i]),
            str(payload["rx_ids"][i]),
            str(payload["day_ids"][i]),
            str(payload["sig_ids"][i]),
        )
        grouped[key].append(int(i))
    rows = []
    feats = []
    class_to_tx = {i: tx for i, tx in enumerate(source_tx_ids)}
    for key, idx in sorted(grouped.items()):
        role, tx, rx, day, sig = key
        z = payload["features"][idx]
        lo = payload["tx_logits"][idx]
        mean_z = z.mean(axis=0)
        std_z = z.std(axis=0)
        mean_lo = lo.mean(axis=0)
        probs = _softmax(mean_lo.reshape(1, -1))[0]
        pred = int(mean_lo.argmax())
        pred_tx = class_to_tx.get(pred, str(pred))
        top2 = np.partition(probs, -2)[-2:] if probs.size >= 2 else np.asarray([0.0, probs.max()])
        feat = np.concatenate([
            mean_z,
            std_z,
            mean_lo,
            probs,
            np.asarray([
                float(probs.max()),
                float(top2[-1] - top2[-2]),
                float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum()),
                float(len(idx)),
            ], dtype=np.float32),
        ]).astype(np.float32)
        rows.append({
            "role": role,
            "tx_id": tx,
            "pred_tx_id": pred_tx,
            "is_known_query": role == "target_old" and tx in set(source_tx_ids),
            "is_unknown_query": role == "target_unknown",
            "closed_correct_known": role == "target_old" and tx == pred_tx,
        })
        feats.append(feat)
    return rows, np.asarray(feats, dtype=np.float32)


class KPlusOne(nn.Module):
    def __init__(self, dim: int, classes: int, kind: str) -> None:
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(dim, classes)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Linear(dim, 96), nn.ReLU(), nn.Dropout(0.05), nn.Linear(96, classes))
        else:
            raise ValueError(kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _standardize(x: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    mu = x[fit_mask].mean(axis=0, keepdims=True)
    sd = x[fit_mask].std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32)


def _train_predict(x: np.ndarray, labels: np.ndarray, train_mask: np.ndarray, classes: int, kind: str, seed: int, epochs: int) -> np.ndarray:
    torch.manual_seed(seed)
    model = KPlusOne(x.shape[1], classes, kind)
    xt = torch.as_tensor(x, dtype=torch.float32)
    idx = torch.as_tensor(np.where(train_mask)[0], dtype=torch.long)
    yt = torch.as_tensor(labels[train_mask], dtype=torch.long)
    counts = np.bincount(labels[train_mask], minlength=classes).astype(np.float32)
    weights = counts.sum() / np.clip(counts, 1.0, None)
    weights = weights / weights.mean()
    wt = torch.as_tensor(weights, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(epochs):
        logits = model(xt.index_select(0, idx))
        loss = F.cross_entropy(logits, yt, weight=wt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        probs = F.softmax(model(xt), dim=1).detach().cpu().tolist()
    return np.asarray(probs, dtype=np.float64)


def _safe_rate(num: int, den: int) -> float:
    return float("nan") if den <= 0 else float(num) / float(den)


def _metrics(accept: np.ndarray, known: np.ndarray, unknown: np.ndarray, closed: np.ndarray) -> dict:
    known_total = int(known.sum())
    unknown_total = int(unknown.sum())
    known_closed = int((known & closed).sum())
    known_correct_after = int((known & closed & accept).sum())
    known_accepted = int((known & accept).sum())
    unknown_accepted = int((unknown & accept).sum())
    closed_acc = _safe_rate(known_closed, known_total)
    full_acc = _safe_rate(known_correct_after, known_total)
    old_drop = 100.0 * (closed_acc - full_acc)
    far = _safe_rate(unknown_accepted, unknown_total)
    return {
        "unknown_FAR": far,
        "known_closed_accuracy_no_reject": closed_acc,
        "known_full_accuracy_after_reject": full_acc,
        "old_drop_pp_vs_closed": old_drop,
        "known_coverage": _safe_rate(known_accepted, known_total),
        "known_accepted_accuracy": _safe_rate(known_correct_after, known_accepted),
        "passes_unknown_far_target": bool(far <= 0.05),
        "passes_old_drop_target": bool(old_drop <= 2.0),
        "passes_dual_target": bool(far <= 0.05 and old_drop <= 2.0),
        "known_query_count": known_total,
        "unknown_query_count": unknown_total,
    }


def _run_one(run_dir: Path, adapter: str, source_tx_ids: list[str], kind: str, seed: int, epochs: int) -> list[dict]:
    npz = run_dir / adapter / "features_leo_repaired.npz"
    payload = _load_npz(npz)
    rows, x_raw = _group_payload(payload, source_tx_ids)
    roles = np.asarray([r["role"] for r in rows])
    tx_ids = [r["tx_id"] for r in rows]
    old_map = {tx: i for i, tx in enumerate(source_tx_ids)}
    unknown_class = len(source_tx_ids)
    labels = np.full(len(rows), -1, dtype=np.int64)
    for i, (role, tx) in enumerate(zip(roles, tx_ids)):
        if role == "source" and tx in old_map:
            labels[i] = old_map[tx]
        elif role == "proxy_unknown":
            labels[i] = unknown_class
    train_mask = labels >= 0
    source_mask = roles == "source"
    proxy_mask = roles == "proxy_unknown"
    x = _standardize(x_raw, source_mask)
    probs = _train_predict(x, labels, train_mask, unknown_class + 1, kind, seed, epochs)
    unknown_prob = probs[:, unknown_class]
    pred_k1 = probs.argmax(axis=1)
    known = np.asarray([bool(r["is_known_query"]) for r in rows], dtype=bool)
    unknown = np.asarray([bool(r["is_unknown_query"]) for r in rows], dtype=bool)
    closed = np.asarray([bool(r["closed_correct_known"]) for r in rows], dtype=bool)

    out = []
    for policy in ["source_accept", "min_source_proxy", "mean_source_proxy"]:
        for source_q in [0.99, 0.995, 0.999, 0.9999]:
            pqs = [0.05] if policy == "source_accept" else [0.03, 0.05, 0.10, 0.15]
            for proxy_q in pqs:
                source_t = float(np.quantile(unknown_prob[source_mask], source_q))
                proxy_t = float(np.quantile(unknown_prob[proxy_mask], proxy_q))
                if policy == "source_accept":
                    threshold = source_t
                elif policy == "min_source_proxy":
                    threshold = min(source_t, proxy_t)
                else:
                    threshold = 0.5 * (source_t + proxy_t)
                accept = (unknown_prob <= threshold) & (pred_k1 != unknown_class)
                m = _metrics(accept, known, unknown, closed)
                m.update({
                    "run_id": run_dir.name,
                    "adapter": adapter,
                    "model_kind": kind,
                    "mode": "source_proxy_kplus1",
                    "threshold_policy": policy,
                    "source_accept_quantile": source_q,
                    "proxy_far_quantile": proxy_q,
                    "threshold": threshold,
                    "source_threshold": source_t,
                    "proxy_threshold": proxy_t,
                    "source_train_count": int(source_mask.sum()),
                    "proxy_train_count": int(proxy_mask.sum()),
                })
                out.append(m)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--run_glob", default="phase1_adv3b02_multiview_keepold_*_20260702")
    p.add_argument("--adapters", default="LEOADAPT3_LINR_COS,LEOADAPT3_MLP_ID")
    p.add_argument("--source_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    p.add_argument("--model_kinds", default="linear,mlp")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=4070341)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source_tx_ids = parse_tx_ids(args.source_tx_ids)
    rows = []
    for run_dir in sorted(args.runs_root.glob(args.run_glob)):
        for adapter in [x.strip() for x in args.adapters.split(",") if x.strip()]:
            if not (run_dir / adapter / "features_leo_repaired.npz").is_file():
                continue
            for kind in [x.strip() for x in args.model_kinds.split(",") if x.strip()]:
                rows.extend(_run_one(run_dir, adapter, source_tx_ids, kind, int(args.seed), int(args.epochs)))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ["run_id", "adapter"]
    leading = ["run_id", "adapter", "model_kind", "mode", "threshold_policy"]
    fields = leading + [f for f in fields if f not in leading]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print({"rows": len(rows), "dual_pass": sum(1 for r in rows if bool(r.get("passes_dual_target"))), "out_csv": str(args.out_csv)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
