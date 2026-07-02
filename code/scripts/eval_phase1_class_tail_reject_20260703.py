#!/usr/bin/env python
"""Source-only class-tail rejection after classwise LEO repair.

V16 keeps the ADV3B02 Phase1 backbone frozen. It trains only source clean<-LEO
classwise repair statistics, then calibrates per-old-class acceptance tails from
source old and source proxy-unknown rows. Target old/unknown rows are used only
for final sat-only metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


KEY_FIELDS = ("dataset_role", "tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids")


def canonical_tx_id(value: object) -> str:
    text = str(value)
    if text.startswith("tx"):
        text = text[2:]
    return text.replace("_", "-")


def parse_tx_ids(text: str) -> list[str]:
    return [canonical_tx_id(x.strip()) for x in str(text).split(",") if x.strip()]


def _as_str(data: np.lib.npyio.NpzFile, key: str, n: int) -> np.ndarray:
    if key not in data.files:
        return np.asarray([""] * n, dtype=str)
    arr = np.asarray(data[key])
    if arr.shape == ():
        return np.asarray([str(arr.item())] * n, dtype=str)
    if arr.shape[0] != n:
        raise ValueError(f"{key} length mismatch")
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
            "eq_ids": _as_str(data, "eq_ids", n),
            "sig_ids": _as_str(data, "sig_ids", n),
        }


def row_key(payload: dict, index: int) -> tuple[str, ...]:
    return tuple(str(payload[field][index]) for field in KEY_FIELDS)


def source_pairs(clean: dict, sat: dict, source_roles: set[str]) -> list[tuple[int, int]]:
    clean_map = {}
    for i, role in enumerate(clean["dataset_role"]):
        if str(role) in source_roles:
            clean_map.setdefault(row_key(clean, i), int(i))
    pairs = []
    seen = set()
    for i, role in enumerate(sat["dataset_role"]):
        if str(role) not in source_roles:
            continue
        key = row_key(sat, i)
        if key in seen:
            continue
        seen.add(key)
        if key in clean_map:
            pairs.append((clean_map[key], int(i)))
    return pairs


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1.0e-6)


def fit_classwise(clean_npz: Path, train_sat_npzs: list[Path], source_tx_ids: list[str], source_roles: set[str], eps: float) -> dict:
    clean = load_npz(clean_npz)
    by_tx_clean = defaultdict(list)
    by_tx_sat = defaultdict(list)
    for sat_path in train_sat_npzs:
        sat = load_npz(sat_path)
        for ci, si in source_pairs(clean, sat, source_roles):
            tx = canonical_tx_id(sat["tx_ids"][si])
            if tx in source_tx_ids:
                by_tx_clean[tx].append(clean["features"][ci])
                by_tx_sat[tx].append(sat["features"][si])
    protos = []
    deltas = []
    var = []
    counts = {}
    for tx in source_tx_ids:
        c = np.asarray(by_tx_clean[tx], dtype=np.float32)
        s = np.asarray(by_tx_sat[tx], dtype=np.float32)
        if c.size == 0 or s.size == 0:
            raise ValueError(f"missing source clean/LEO pairs for tx={tx}")
        proto = c.mean(axis=0)
        proto = proto / max(float(np.linalg.norm(proto)), 1.0e-6)
        repaired = l2n(s + (c - s).mean(axis=0, keepdims=True))
        protos.append(proto.astype(np.float32))
        deltas.append((c - s).mean(axis=0).astype(np.float32))
        var.append(repaired.var(axis=0).astype(np.float32) + float(eps))
        counts[tx] = int(c.shape[0])
    return {
        "source_tx_ids": source_tx_ids,
        "prototypes": np.stack(protos, axis=0),
        "deltas": np.stack(deltas, axis=0),
        "var": np.stack(var, axis=0),
        "counts": counts,
        "source_pair_count": int(sum(counts.values())),
    }


def group_rows(payload: dict) -> list[dict]:
    buckets = defaultdict(list)
    for i in range(payload["features"].shape[0]):
        key = (
            str(payload["dataset_role"][i]),
            canonical_tx_id(payload["tx_ids"][i]),
            str(payload["rx_ids"][i]),
            str(payload["day_ids"][i]),
            str(payload["sig_ids"][i]),
        )
        buckets[key].append(i)
    rows = []
    for key, idx in sorted(buckets.items()):
        role, tx, rx, day, sig = key
        rows.append({
            "role": role,
            "tx_id": tx,
            "rx_id": rx,
            "day_id": day,
            "sig_id": sig,
            "feature": payload["features"][idx].mean(axis=0).astype(np.float32),
        })
    return rows


def repair_and_score(x: np.ndarray, model: dict, alpha: float, beta: float) -> dict:
    protos = model["prototypes"]
    deltas = model["deltas"]
    var = model["var"]
    x0 = x / max(float(np.linalg.norm(x)), 1.0e-6)
    raw_pred = int((protos @ x0).argmax())
    z = x + float(alpha) * deltas[raw_pred]
    if float(beta) > 0:
        z = (1.0 - float(beta)) * z + float(beta) * protos[raw_pred]
    z = z / max(float(np.linalg.norm(z)), 1.0e-6)
    sims = protos @ z
    pred = int(sims.argmax())
    delta = z.reshape(1, -1) - protos
    mah = np.mean((delta * delta) / var, axis=1)
    sim_sorted = np.sort(sims)
    mah_sorted = np.sort(mah)
    pred_mah = float(mah[pred])
    pred_cos_dist = float(1.0 - sims[pred])
    min_mah = float(mah.min())
    return {
        "pred_idx": pred,
        "pred_tx": model["source_tx_ids"][pred],
        "pred_mah": pred_mah,
        "min_mah": min_mah,
        "pred_cos_dist": pred_cos_dist,
        "max_sim": float(sims[pred]),
        "sim_margin": float(sim_sorted[-1] - sim_sorted[-2]) if sim_sorted.size >= 2 else float(sim_sorted[-1]),
        "mah_margin": float(mah_sorted[1] - mah_sorted[0]) if mah_sorted.size >= 2 else 0.0,
    }


def empirical_tail_score(value: float, fit_values: np.ndarray, higher_is_more_old: bool = False) -> float:
    vals = np.sort(fit_values.astype(np.float64))
    if vals.size == 0:
        return 0.0
    rank = np.searchsorted(vals, float(value), side="right") / float(vals.size)
    if higher_is_more_old:
        return rank
    return 1.0 - rank


def rank_scale(values: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    fit = np.sort(values[fit_mask].astype(np.float64))
    if fit.size == 0:
        raise ValueError("empty fit mask")
    return np.searchsorted(fit, values, side="right") / float(fit.size)


def safe_rate(num: int, den: int) -> float:
    return float("nan") if den <= 0 else float(num) / float(den)


def calc_metrics(accept: np.ndarray, known: np.ndarray, unknown: np.ndarray, closed: np.ndarray) -> dict:
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
        "passes_unknown_far_target": bool(far <= 0.05),
        "passes_old_drop_target": bool(old_drop <= 2.0),
        "passes_dual_target": bool(far <= 0.05 and old_drop <= 2.0),
    }


def class_thresholds(values: np.ndarray, pred_idx: np.ndarray, fit_mask: np.ndarray, quantile: float, n_classes: int) -> np.ndarray:
    thresholds = np.full((n_classes,), float(np.quantile(values[fit_mask], quantile)), dtype=np.float64)
    for cls in range(n_classes):
        cls_vals = values[fit_mask & (pred_idx == cls)]
        if cls_vals.size >= 3:
            thresholds[cls] = float(np.quantile(cls_vals, quantile))
    return thresholds


def evaluate_run(run_dir: Path, model: dict, args: argparse.Namespace) -> list[dict]:
    npz = run_dir / str(args.sat_npz_relpath)
    if not npz.is_file():
        return []
    rows = group_rows(load_npz(npz))
    source_tx_ids = set(model["source_tx_ids"])
    out = []
    n_classes = len(model["source_tx_ids"])
    for alpha in [float(x) for x in str(args.alphas).split(",") if x.strip()]:
        for beta in [float(x) for x in str(args.betas).split(",") if x.strip()]:
            scored = []
            for row in rows:
                s = repair_and_score(row["feature"], model, alpha, beta)
                tx = canonical_tx_id(row["tx_id"])
                scored.append({
                    **row,
                    **s,
                    "is_known_query": row["role"] == "target_old" and tx in source_tx_ids,
                    "is_unknown_query": row["role"] == "target_unknown",
                    "closed_correct_known": row["role"] == "target_old" and tx == s["pred_tx"],
                })
            roles = np.asarray([r["role"] for r in scored])
            source = roles == "source"
            proxy = roles == "proxy_unknown"
            known = np.asarray([bool(r["is_known_query"]) for r in scored], dtype=bool)
            unknown = np.asarray([bool(r["is_unknown_query"]) for r in scored], dtype=bool)
            closed = np.asarray([bool(r["closed_correct_known"]) for r in scored], dtype=bool)
            if not source.any() or not proxy.any() or not known.any() or not unknown.any():
                continue
            pred_idx = np.asarray([int(r["pred_idx"]) for r in scored], dtype=np.int64)
            dist_raw = {
                "pred_mah": np.asarray([r["pred_mah"] for r in scored], dtype=np.float64),
                "min_mah": np.asarray([r["min_mah"] for r in scored], dtype=np.float64),
                "pred_cos_dist": np.asarray([r["pred_cos_dist"] for r in scored], dtype=np.float64),
            }
            old_scores = {}
            old_scores["max_sim"] = rank_scale(np.asarray([r["max_sim"] for r in scored], dtype=np.float64), source | proxy)
            old_scores["sim_margin"] = rank_scale(np.asarray([r["sim_margin"] for r in scored], dtype=np.float64), source | proxy)
            old_scores["mah_margin"] = rank_scale(np.asarray([r["mah_margin"] for r in scored], dtype=np.float64), source | proxy)
            for name, values in dist_raw.items():
                src_tail = np.zeros(values.shape[0], dtype=np.float64)
                fit_mask = source
                for i in range(values.shape[0]):
                    cls = pred_idx[i]
                    cls_fit = values[fit_mask & (pred_idx == cls)]
                    if cls_fit.size < 3:
                        cls_fit = values[fit_mask]
                    src_tail[i] = empirical_tail_score(float(values[i]), cls_fit, higher_is_more_old=False)
                old_scores[f"{name}_source_tail"] = src_tail
                old_scores[f"neg_{name}_rank"] = 1.0 - rank_scale(values, source | proxy)
            old_scores["class_tail_combo"] = (
                0.35 * old_scores["pred_mah_source_tail"]
                + 0.25 * old_scores["pred_cos_dist_source_tail"]
                + 0.20 * old_scores["max_sim"]
                + 0.10 * old_scores["sim_margin"]
                + 0.10 * old_scores["mah_margin"]
            )
            old_scores["tail_margin_combo"] = (
                0.45 * old_scores["pred_mah_source_tail"]
                + 0.20 * old_scores["min_mah_source_tail"]
                + 0.20 * old_scores["sim_margin"]
                + 0.15 * old_scores["mah_margin"]
            )
            target_rxs = sorted({r["rx_id"] for r in scored if r["role"] in {"target_old", "target_unknown"}})
            for score_name, score in old_scores.items():
                for policy in ["global_source_accept", "global_proxy_far", "global_mean_source_proxy", "class_source_accept", "class_proxy_far", "class_mean_source_proxy"]:
                    for source_q in [0.001, 0.005, 0.010, 0.020, 0.050]:
                        proxy_qs = [0.95] if "source_accept" in policy else [0.85, 0.90, 0.95, 0.97, 0.99]
                        for proxy_q in proxy_qs:
                            source_t = float(np.quantile(score[source], source_q))
                            proxy_t = float(np.quantile(score[proxy], proxy_q))
                            if policy == "global_source_accept":
                                threshold = np.full(score.shape, source_t, dtype=np.float64)
                            elif policy == "global_proxy_far":
                                threshold = np.full(score.shape, proxy_t, dtype=np.float64)
                            elif policy == "global_mean_source_proxy":
                                threshold = np.full(score.shape, 0.5 * (source_t + proxy_t), dtype=np.float64)
                            elif policy == "class_source_accept":
                                t = class_thresholds(score, pred_idx, source, source_q, n_classes)
                                threshold = t[pred_idx]
                            elif policy == "class_proxy_far":
                                t = class_thresholds(score, pred_idx, proxy, proxy_q, n_classes)
                                threshold = t[pred_idx]
                            elif policy == "class_mean_source_proxy":
                                ts = class_thresholds(score, pred_idx, source, source_q, n_classes)
                                tp = class_thresholds(score, pred_idx, proxy, proxy_q, n_classes)
                                threshold = 0.5 * (ts[pred_idx] + tp[pred_idx])
                            else:
                                raise ValueError(policy)
                            accept = score >= threshold
                            m = calc_metrics(accept, known, unknown, closed)
                            m.update({
                                "run_id": run_dir.name,
                                "target_rx": ",".join(target_rxs),
                                "mode": "class_tail_repair_reject",
                                "score_name": score_name,
                                "threshold_policy": policy,
                                "source_accept_quantile": source_q,
                                "proxy_far_quantile": proxy_q,
                                "threshold": float(np.mean(threshold)),
                                "source_threshold": source_t,
                                "proxy_threshold": proxy_t,
                                "alpha": alpha,
                                "beta": beta,
                            })
                            out.append(m)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", type=Path, required=True)
    p.add_argument("--clean_npz", type=Path, required=True)
    p.add_argument("--train_sat_npz", type=Path, action="append", required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--metrics_json", type=Path, default=None)
    p.add_argument("--run_glob", default="phase1_adv3b02_multiview_keepold_*_20260702")
    p.add_argument("--sat_npz_relpath", default="ADV3B02_CORE90_SOFT_E200_PHASE1_SATUNKNOWN_SINGLEVIEW/features_satunknown_singleview.npz")
    p.add_argument("--source_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    p.add_argument("--source_roles", default="source")
    p.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
    p.add_argument("--betas", default="0.0,0.05,0.10")
    p.add_argument("--var_eps", type=float, default=1.0e-4)
    p.add_argument("--run_tag", default="V16_class_tail_repair_reject")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source_tx_ids = parse_tx_ids(args.source_tx_ids)
    source_roles = {x.strip() for x in str(args.source_roles).split(",") if x.strip()}
    model = fit_classwise(args.clean_npz, list(args.train_sat_npz), source_tx_ids, source_roles, float(args.var_eps))
    rows = []
    for run_dir in sorted(args.runs_root.glob(str(args.run_glob))):
        rows.extend(evaluate_run(run_dir, model, args))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ["run_id"]
    leading = ["run_id", "target_rx", "mode", "score_name", "threshold_policy", "alpha", "beta"]
    fields = leading + [f for f in fields if f not in leading]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "phase": "phase1_class_tail_repair_reject_v16",
        "run_tag": str(args.run_tag),
        "rows": len(rows),
        "dual_pass": int(sum(1 for r in rows if bool(r.get("passes_dual_target")))),
        "out_csv": str(args.out_csv),
        "source_pair_count": int(model["source_pair_count"]),
        "source_pair_count_by_tx": model["counts"],
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
