#!/usr/bin/env python
"""Reject unknowns using before/after LEO feature-repair delta signals."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


KEY_FIELDS = ["dataset_role", "tx_ids", "rx_ids", "day_ids", "sig_ids"]


def canonical_tx_id(value: object) -> str:
    text = str(value)
    if text.startswith("tx"):
        text = text[2:]
    return text.replace("_", "-")


def parse_tx_ids(text: str) -> list[str]:
    return [canonical_tx_id(x.strip()) for x in str(text).split(",") if x.strip()]


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_rate(num: int, den: int) -> float:
    return float("nan") if den <= 0 else float(num) / float(den)


def _str_array(data, key: str, n: int) -> np.ndarray:
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
        manifest = {}
        if "manifest_json" in data.files:
            try:
                manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
            except Exception:
                manifest = {}
        return {
            "features": features,
            "tx_logits": logits,
            "dataset_role": _str_array(data, "dataset_role", n),
            "tx_ids": _str_array(data, "tx_ids", n),
            "rx_ids": _str_array(data, "rx_ids", n),
            "day_ids": _str_array(data, "day_ids", n),
            "sig_ids": _str_array(data, "sig_ids", n),
            "manifest": manifest,
        }


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def _entropy(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)


def _margin(probs: np.ndarray) -> np.ndarray:
    if probs.shape[1] < 2:
        return np.zeros(probs.shape[0], dtype=np.float32)
    part = np.partition(probs, -2, axis=1)[:, -2:]
    return part[:, 1] - part[:, 0]


def _group_indices(payload: dict) -> dict[tuple[str, str, str, str, str], list[int]]:
    groups: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)
    for i in range(len(payload["dataset_role"])):
        key = (
            str(payload["dataset_role"][i]),
            canonical_tx_id(payload["tx_ids"][i]),
            str(payload["rx_ids"][i]),
            str(payload["day_ids"][i]),
            str(payload["sig_ids"][i]),
        )
        groups[key].append(i)
    return groups


def _make_groups(raw: dict, repaired: dict, source_tx_ids: list[str]) -> tuple[list[dict], np.ndarray]:
    raw_groups = _group_indices(raw)
    rep_groups = _group_indices(repaired)
    common = sorted(set(raw_groups) & set(rep_groups))
    class_to_tx = {i: tx for i, tx in enumerate(source_tx_ids)}

    source_proto_raw = {}
    source_proto_rep = {}
    for tx in source_tx_ids:
        raw_parts = []
        rep_parts = []
        for key in common:
            if key[0] == "source" and key[1] == tx:
                raw_parts.append(raw["features"][raw_groups[key]].mean(axis=0))
                rep_parts.append(repaired["features"][rep_groups[key]].mean(axis=0))
        if raw_parts:
            source_proto_raw[tx] = np.asarray(raw_parts, dtype=np.float32).mean(axis=0)
            source_proto_rep[tx] = np.asarray(rep_parts, dtype=np.float32).mean(axis=0)
    fallback_raw = raw["features"].mean(axis=0)
    fallback_rep = repaired["features"].mean(axis=0)

    rows = []
    feats = []
    for key in common:
        ridx = raw_groups[key]
        aidx = rep_groups[key]
        z0 = raw["features"][ridx].mean(axis=0)
        z1 = repaired["features"][aidx].mean(axis=0)
        l0 = raw["tx_logits"][ridx].mean(axis=0, keepdims=True)
        l1 = repaired["tx_logits"][aidx].mean(axis=0, keepdims=True)
        p0 = _softmax_np(l0)[0]
        p1 = _softmax_np(l1)[0]
        pred0 = int(l0[0].argmax())
        pred1 = int(l1[0].argmax())
        pred_tx = class_to_tx.get(pred1, str(pred1))
        proto0 = source_proto_raw.get(pred_tx, fallback_raw)
        proto1 = source_proto_rep.get(pred_tx, fallback_rep)
        residual = z1 - z0
        cos01 = float(np.dot(z0, z1) / (np.linalg.norm(z0) * np.linalg.norm(z1) + 1e-12))
        cos_proto0 = float(np.dot(z0, proto0) / (np.linalg.norm(z0) * np.linalg.norm(proto0) + 1e-12))
        cos_proto1 = float(np.dot(z1, proto1) / (np.linalg.norm(z1) * np.linalg.norm(proto1) + 1e-12))
        feat = [
            float(np.linalg.norm(residual)),
            float(np.linalg.norm(residual) / (np.linalg.norm(z0) + 1e-12)),
            cos01,
            float(np.linalg.norm(z0)),
            float(np.linalg.norm(z1)),
            float(np.linalg.norm(z0 - proto0)),
            float(np.linalg.norm(z1 - proto1)),
            cos_proto0,
            cos_proto1,
            float(p0.max()),
            float(p1.max()),
            float(_entropy(p0.reshape(1, -1))[0]),
            float(_entropy(p1.reshape(1, -1))[0]),
            float(_margin(p0.reshape(1, -1))[0]),
            float(_margin(p1.reshape(1, -1))[0]),
            float(np.linalg.norm(p1 - p0)),
            float(pred0 != pred1),
        ]
        role, tx, rx, day, sig = key
        rows.append({
            "role": role,
            "tx_id": tx,
            "rx_id": rx,
            "day_id": day,
            "sig_id": sig,
            "pred_tx_id": pred_tx,
            "is_known_query": role == "target_old" and tx in set(source_tx_ids),
            "is_unknown_query": role == "target_unknown",
            "closed_correct_known": role == "target_old" and tx == pred_tx,
        })
        feats.append(feat)
    return rows, np.asarray(feats, dtype=np.float32)


class Rejector(nn.Module):
    def __init__(self, dim: int, kind: str) -> None:
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(dim, 1)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 1))
        else:
            raise ValueError(kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


def _standardize(x: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    mu = x[fit_mask].mean(axis=0, keepdims=True)
    sd = x[fit_mask].std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32)


def _train_scores(x: np.ndarray, train_mask: np.ndarray, y: np.ndarray, kind: str, seed: int, epochs: int) -> np.ndarray:
    torch.manual_seed(seed)
    model = Rejector(x.shape[1], kind)
    xt = torch.as_tensor(x, dtype=torch.float32)
    idx = torch.as_tensor(np.where(train_mask)[0], dtype=torch.long)
    yt = torch.as_tensor(y[train_mask], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    pos = float((yt == 1).sum().item())
    neg = float((yt == 0).sum().item())
    pos_weight = torch.tensor(max(1.0, neg / max(pos, 1.0)), dtype=torch.float32)
    for _ in range(epochs):
        logits = model(xt.index_select(0, idx))
        loss = F.binary_cross_entropy_with_logits(logits, yt, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return np.asarray(torch.sigmoid(model(xt)).detach().cpu().tolist(), dtype=np.float64)


def _metrics(scores: np.ndarray, threshold: float, known: np.ndarray, unknown: np.ndarray, closed: np.ndarray) -> dict:
    accepted = scores <= threshold
    known_total = int(known.sum())
    unknown_total = int(unknown.sum())
    known_closed = int((known & closed).sum())
    known_correct_after = int((known & closed & accepted).sum())
    known_accepted = int((known & accepted).sum())
    unknown_accepted = int((unknown & accepted).sum())
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


def _oracle(scores: np.ndarray, known: np.ndarray, unknown: np.ndarray, closed: np.ndarray) -> dict:
    vals = np.unique(scores[np.isfinite(scores)])
    thresholds = np.concatenate(([np.nextafter(float(vals[0]), -np.inf)], vals, [np.nextafter(float(vals[-1]), np.inf)]))
    best = None
    for t in thresholds:
        m = _metrics(scores, float(t), known, unknown, closed)
        m["threshold"] = float(t)
        m["joint_score"] = max(0.0, m["unknown_FAR"] - 0.05) * 100.0 + max(0.0, m["old_drop_pp_vs_closed"] - 2.0)
        if best is None or (m["joint_score"], m["unknown_FAR"], m["old_drop_pp_vs_closed"]) < (best["joint_score"], best["unknown_FAR"], best["old_drop_pp_vs_closed"]):
            best = m
    return best or {}


def _run_one(run_dir: Path, adapter: str, source_tx_ids: list[str], kind: str, seed: int, epochs: int) -> list[dict]:
    raw_npz = run_dir / "ADV3B02_CORE90_SOFT_E200_PHASE1_SATUNKNOWN_SINGLEVIEW" / "features_satunknown_singleview.npz"
    repaired_npz = run_dir / adapter / "features_leo_repaired.npz"
    raw = _load_npz(raw_npz)
    repaired = _load_npz(repaired_npz)
    rows, x_raw = _make_groups(raw, repaired, source_tx_ids)
    roles = np.asarray([r["role"] for r in rows])
    source = roles == "source"
    proxy = roles == "proxy_unknown"
    known = np.asarray([bool(r["is_known_query"]) for r in rows], dtype=bool)
    unknown = np.asarray([bool(r["is_unknown_query"]) for r in rows], dtype=bool)
    closed = np.asarray([bool(r["closed_correct_known"]) for r in rows], dtype=bool)
    x = _standardize(x_raw, source)

    y = np.zeros(len(rows), dtype=np.float32)
    y[proxy] = 1.0
    train_mask = source | proxy
    scores = _train_scores(x, train_mask, y, kind, seed, epochs)

    out = []
    for policy in ["source_accept", "min_source_proxy", "mean_source_proxy"]:
        for source_q in [0.99, 0.995, 0.999, 0.9999]:
            pqs = [0.05] if policy == "source_accept" else [0.03, 0.05, 0.10, 0.15]
            for proxy_q in pqs:
                source_t = float(np.quantile(scores[source], source_q))
                proxy_t = float(np.quantile(scores[proxy], proxy_q))
                if policy == "source_accept":
                    threshold = source_t
                elif policy == "min_source_proxy":
                    threshold = min(source_t, proxy_t)
                else:
                    threshold = 0.5 * (source_t + proxy_t)
                m = _metrics(scores, float(threshold), known, unknown, closed)
                m.update({
                    "run_id": run_dir.name,
                    "adapter": adapter,
                    "model_kind": kind,
                    "mode": "source_proxy_train",
                    "threshold_policy": policy,
                    "source_accept_quantile": source_q,
                    "proxy_far_quantile": proxy_q,
                    "threshold": threshold,
                    "source_threshold": source_t,
                    "proxy_threshold": proxy_t,
                })
                out.append(m)

    y_oracle = np.ones(len(rows), dtype=np.float32)
    y_oracle[known & closed] = 0.0
    oracle_scores = _train_scores(x, known | unknown, y_oracle, kind, seed + 1009, epochs)
    m = _oracle(oracle_scores, known, unknown, closed)
    m.update({
        "run_id": run_dir.name,
        "adapter": adapter,
        "model_kind": kind,
        "mode": "target_label_oracle_repair_delta",
        "threshold_policy": "target_label_oracle",
        "source_accept_quantile": "",
        "proxy_far_quantile": "",
        "source_threshold": "",
        "proxy_threshold": "",
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
    p.add_argument("--epochs", type=int, default=220)
    p.add_argument("--seed", type=int, default=4070331)
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
