#!/usr/bin/env python
"""Evaluate source-only repair-ensemble manifold rejection for Phase1 features.

The script uses already trained source-only LEO repair adapters. For each target
run it intersects sample groups across adapter feature NPZs, fits old-class
manifolds from source rows only, and evaluates sat-only target-old/unknown rows.
No target clean features, target labels, or unknown query labels are used for
threshold calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


GROUP_FIELDS = ("dataset_role", "tx_ids", "rx_ids", "day_ids", "sig_ids")


def canonical_tx_id(value: object) -> str:
    text = str(value)
    if text.startswith("tx"):
        text = text[2:]
    return text.replace("_", "-")


def parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_tx_ids(text: str) -> list[str]:
    return [canonical_tx_id(x) for x in parse_csv_list(text)]


def _as_str(data: np.lib.npyio.NpzFile, key: str, n: int) -> np.ndarray:
    if key not in data.files:
        return np.asarray([""] * n, dtype=str)
    arr = np.asarray(data[key])
    if arr.shape == ():
        return np.asarray([str(arr.item())] * n, dtype=str)
    if arr.shape[0] != n:
        raise ValueError(f"{key} length mismatch: {arr.shape[0]} != {n}")
    return arr.astype(str)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        n = int(features.shape[0])
        return {
            "path": str(path),
            "features": features,
            "dataset_role": _as_str(data, "dataset_role", n),
            "tx_ids": _as_str(data, "tx_ids", n),
            "rx_ids": _as_str(data, "rx_ids", n),
            "day_ids": _as_str(data, "day_ids", n),
            "sig_ids": _as_str(data, "sig_ids", n),
        }


def group_key(payload: dict, index: int) -> tuple[str, str, str, str, str]:
    return tuple(str(payload[field][index]) for field in GROUP_FIELDS)


def grouped_rows(payload: dict) -> dict[tuple[str, str, str, str, str], dict]:
    buckets: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)
    for i in range(payload["features"].shape[0]):
        buckets[group_key(payload, i)].append(i)
    rows = {}
    for key, idx in buckets.items():
        role, tx, rx, day, sig = key
        feat = payload["features"][idx].mean(axis=0).astype(np.float32)
        rows[key] = {
            "role": role,
            "tx_id": canonical_tx_id(tx),
            "rx_id": rx,
            "day_id": day,
            "sig_id": sig,
            "feature": feat,
        }
    return rows


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1.0e-6)


def fit_adapter_model(rows: dict, source_tx_ids: list[str], eps: float) -> dict:
    source = [r for r in rows.values() if r["role"] == "source" and r["tx_id"] in source_tx_ids]
    if not source:
        raise ValueError("no source rows available for manifold fitting")
    dim = int(source[0]["feature"].shape[0])
    class_stats = {}
    all_source = []
    for tx in source_tx_ids:
        feats = np.asarray([r["feature"] for r in source if r["tx_id"] == tx], dtype=np.float32)
        if feats.size == 0:
            continue
        feats = l2_normalize(feats)
        mean = feats.mean(axis=0)
        mean = mean / max(float(np.linalg.norm(mean)), 1.0e-6)
        var = feats.var(axis=0) + float(eps)
        class_stats[tx] = {"mean": mean.astype(np.float32), "var": var.astype(np.float32)}
        all_source.append(feats)
    if not class_stats:
        raise ValueError("no source class stats built")
    all_source_arr = np.concatenate(all_source, axis=0) if all_source else np.zeros((0, dim), dtype=np.float32)
    return {"classes": class_stats, "source_count": int(all_source_arr.shape[0])}


def score_one(feature: np.ndarray, model: dict) -> dict:
    x = feature.astype(np.float32)
    x = x / max(float(np.linalg.norm(x)), 1.0e-6)
    sims = []
    mahs = []
    labels = []
    for tx, stat in model["classes"].items():
        mean = stat["mean"]
        var = stat["var"]
        sims.append(float(np.dot(x, mean)))
        delta = x - mean
        mahs.append(float(np.mean((delta * delta) / var)))
        labels.append(tx)
    sim_arr = np.asarray(sims, dtype=np.float64)
    mah_arr = np.asarray(mahs, dtype=np.float64)
    pred_sim = int(sim_arr.argmax())
    pred_mah = int(mah_arr.argmin())
    sim_sorted = np.sort(sim_arr)
    mah_sorted = np.sort(mah_arr)
    return {
        "pred_tx": labels[pred_sim],
        "pred_tx_mah": labels[pred_mah],
        "max_sim": float(sim_arr[pred_sim]),
        "sim_margin": float(sim_sorted[-1] - sim_sorted[-2]) if sim_sorted.size >= 2 else float(sim_sorted[-1]),
        "min_mah": float(mah_arr[pred_mah]),
        "mah_margin": float(mah_sorted[1] - mah_sorted[0]) if mah_sorted.size >= 2 else 0.0,
    }


def rank_scale(values: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    fit = np.sort(values[fit_mask].astype(np.float64))
    if fit.size == 0:
        raise ValueError("empty fit mask")
    return np.searchsorted(fit, values, side="right") / float(fit.size)


def evaluate_run(run_dir: Path, adapters: list[str], args: argparse.Namespace, source_tx_ids: list[str]) -> list[dict]:
    per_adapter = {}
    models = {}
    common: set[tuple[str, str, str, str, str]] | None = None
    for adapter in adapters:
        npz_path = run_dir / adapter / str(args.feature_relpath)
        if not npz_path.is_file():
            return []
        rows = grouped_rows(load_npz(npz_path))
        model = fit_adapter_model(rows, source_tx_ids, float(args.var_eps))
        per_adapter[adapter] = rows
        models[adapter] = model
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
        pred_counts = Counter(preds)
        pred_tx, pred_count = pred_counts.most_common(1)[0]
        pred_mah_counts = Counter(preds_mah)
        pred_mah_tx, pred_mah_count = pred_mah_counts.most_common(1)[0]
        adapter_count = len(adapters)
        records.append({
            "role": role,
            "tx_id": tx,
            "rx_id": base["rx_id"],
            "day_id": base["day_id"],
            "sig_id": base["sig_id"],
            "pred_tx": pred_tx,
            "pred_tx_mah": pred_mah_tx,
            "vote_agreement": pred_count / float(adapter_count),
            "mah_vote_agreement": pred_mah_count / float(adapter_count),
            "mean_max_sim": float(np.mean(max_sims)),
            "mean_sim_margin": float(np.mean(sim_margins)),
            "mean_min_mah": float(np.mean(min_mahs)),
            "mean_mah_margin": float(np.mean(mah_margins)),
            "std_max_sim": float(np.std(max_sims)),
            "std_min_mah": float(np.std(min_mahs)),
            "is_known_query": role == "target_old" and tx in source_tx_ids,
            "is_unknown_query": role == "target_unknown",
            "closed_correct_known": role == "target_old" and tx == pred_tx,
        })

    roles = np.asarray([r["role"] for r in records])
    source_mask = roles == "source"
    proxy_mask = roles == "proxy_unknown"
    known = np.asarray([bool(r["is_known_query"]) for r in records], dtype=bool)
    unknown = np.asarray([bool(r["is_unknown_query"]) for r in records], dtype=bool)
    closed = np.asarray([bool(r["closed_correct_known"]) for r in records], dtype=bool)
    if not source_mask.any() or not proxy_mask.any() or not known.any() or not unknown.any():
        return []

    raw_scores = {
        "mean_max_sim": np.asarray([r["mean_max_sim"] for r in records], dtype=np.float64),
        "mean_sim_margin": np.asarray([r["mean_sim_margin"] for r in records], dtype=np.float64),
        "neg_mean_min_mah": -np.asarray([r["mean_min_mah"] for r in records], dtype=np.float64),
        "vote_agreement": np.asarray([r["vote_agreement"] for r in records], dtype=np.float64),
        "mah_vote_agreement": np.asarray([r["mah_vote_agreement"] for r in records], dtype=np.float64),
    }
    scaled = {name: rank_scale(vals, source_mask | proxy_mask) for name, vals in raw_scores.items()}
    scaled["repair_ensemble_oldness"] = (
        0.30 * scaled["mean_max_sim"]
        + 0.20 * scaled["mean_sim_margin"]
        + 0.30 * scaled["neg_mean_min_mah"]
        + 0.10 * scaled["vote_agreement"]
        + 0.10 * scaled["mah_vote_agreement"]
    )
    scaled["sim_mah_vote"] = (
        0.25 * scaled["mean_max_sim"]
        + 0.35 * scaled["neg_mean_min_mah"]
        + 0.20 * scaled["vote_agreement"]
        + 0.20 * scaled["mah_vote_agreement"]
    )

    target_rxs = sorted({r["rx_id"] for r in records if r["role"] in {"target_old", "target_unknown"}})
    out = []
    for score_name, score in scaled.items():
        for policy in ["source_accept", "proxy_far", "max_source_proxy", "mean_source_proxy"]:
            for source_q in [0.001, 0.005, 0.010, 0.020, 0.050]:
                proxy_qs = [0.95] if policy == "source_accept" else [0.85, 0.90, 0.95, 0.97, 0.99]
                for proxy_q in proxy_qs:
                    source_t = float(np.quantile(score[source_mask], source_q))
                    proxy_t = float(np.quantile(score[proxy_mask], proxy_q))
                    if policy == "source_accept":
                        threshold = source_t
                    elif policy == "proxy_far":
                        threshold = proxy_t
                    elif policy == "max_source_proxy":
                        threshold = max(source_t, proxy_t)
                    elif policy == "mean_source_proxy":
                        threshold = 0.5 * (source_t + proxy_t)
                    else:
                        raise ValueError(policy)
                    accept = score >= threshold
                    m = metrics(accept, known, unknown, closed)
                    m.update({
                        "run_id": run_dir.name,
                        "target_rx": ",".join(target_rxs),
                        "mode": "repair_ensemble_manifold",
                        "adapter_set": ",".join(adapters),
                        "adapter_count": len(adapters),
                        "score_name": score_name,
                        "threshold_policy": policy,
                        "source_accept_quantile": source_q,
                        "proxy_far_quantile": proxy_q,
                        "threshold": threshold,
                        "source_threshold": source_t,
                        "proxy_threshold": proxy_t,
                        "common_group_count": len(records),
                    })
                    out.append(m)
    return out


def safe_rate(num: int, den: int) -> float:
    return float("nan") if den <= 0 else float(num) / float(den)


def metrics(accept: np.ndarray, known: np.ndarray, unknown: np.ndarray, closed: np.ndarray) -> dict:
    known_total = int(known.sum())
    unknown_total = int(unknown.sum())
    known_closed = int((known & closed).sum())
    known_correct_after = int((known & closed & accept).sum())
    known_accepted = int((known & accept).sum())
    unknown_accepted = int((unknown & accept).sum())
    closed_acc = safe_rate(known_closed, known_total)
    full_acc = safe_rate(known_correct_after, known_total)
    far = safe_rate(unknown_accepted, unknown_total)
    old_drop = 100.0 * (closed_acc - full_acc)
    return {
        "unknown_FAR": far,
        "known_closed_accuracy_no_reject": closed_acc,
        "known_full_accuracy_after_reject": full_acc,
        "old_drop_pp_vs_closed": old_drop,
        "known_coverage": safe_rate(known_accepted, known_total),
        "known_accepted_accuracy": safe_rate(known_correct_after, known_accepted),
        "known_query_count": known_total,
        "unknown_query_count": unknown_total,
        "known_closed_correct": known_closed,
        "known_correct_after_reject": known_correct_after,
        "unknown_accepted_count": unknown_accepted,
        "passes_unknown_far_target": bool(far <= 0.05),
        "passes_old_drop_target": bool(old_drop <= 2.0),
        "passes_dual_target": bool(far <= 0.05 and old_drop <= 2.0),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--metrics_json", type=Path, default=None)
    p.add_argument("--run_glob", default="phase1_adv3b02_multiview_keepold_*_20260702")
    p.add_argument("--adapters", default="LEOADAPT3_IDENTITY,LEOADAPT3_MEANSHIFT,LEOADAPT3_NORMSHIFT,LEOADAPT3_LINR_COS,LEOADAPT3_MLP_ID")
    p.add_argument("--feature_relpath", default="features_leo_repaired.npz")
    p.add_argument("--source_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    p.add_argument("--var_eps", type=float, default=1.0e-4)
    p.add_argument("--run_tag", default="V13_repair_ensemble_manifold")
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
    leading = ["run_id", "target_rx", "mode", "score_name", "threshold_policy"]
    fields = leading + [f for f in fields if f not in leading]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "phase": "phase1_repair_ensemble_manifold_reject_v13",
        "run_tag": str(args.run_tag),
        "rows": len(rows),
        "dual_pass": int(sum(1 for r in rows if bool(r.get("passes_dual_target")))),
        "out_csv": str(args.out_csv),
        "adapters": adapters,
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
