#!/usr/bin/env python
"""Evaluate sat-only Phase2-B old-class K-shot calibration and unknown rejection.

The Phase1 backbone/features are frozen. The script uses target_old K-shot
support from the same target receiver domain to build old-class prototypes.
It never uses target clean features and never uses target_unknown query rows for
threshold calibration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np


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
    if arr.shape == ():
        return np.asarray([str(arr.item())] * n, dtype=str)
    if arr.shape[0] != n:
        raise ValueError(f"{key} length mismatch")
    return arr.astype(str)


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        logits = np.asarray(data["tx_logits"], dtype=np.float32) if "tx_logits" in data.files else None
        n = int(features.shape[0])
        return {
            "features": features,
            "tx_logits": logits,
            "dataset_role": _as_str(data, "dataset_role", n),
            "tx_ids": _as_str(data, "tx_ids", n),
            "rx_ids": _as_str(data, "rx_ids", n),
            "day_ids": _as_str(data, "day_ids", n),
            "sig_ids": _as_str(data, "sig_ids", n),
            "sat_scenarios": _as_str(data, "sat_scenarios", n),
        }


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)


def _stable_score(parts: tuple[object, ...], seed: int) -> float:
    h = hashlib.sha256(("|".join([str(seed), *(str(p) for p in parts)])).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") / float(2**64 - 1)


def _group_rows(payload: dict, source_tx_ids: list[str]) -> list[dict]:
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
    old_set = set(source_tx_ids)
    rows = []
    class_to_tx = {i: tx for i, tx in enumerate(source_tx_ids)}
    for key, idx in sorted(grouped.items()):
        role, tx, rx, day, sig = key
        z = payload["features"][idx].mean(axis=0).astype(np.float32)
        if payload["tx_logits"] is not None:
            lo = payload["tx_logits"][idx].mean(axis=0).astype(np.float32)
            probs = _softmax(lo.reshape(1, -1))[0]
            pred = int(lo.argmax())
            pred_tx = class_to_tx.get(pred, str(pred))
            top2 = np.partition(probs, -2)[-2:] if probs.size >= 2 else np.asarray([0.0, probs.max()])
            logit_conf = float(probs.max())
            logit_margin = float(top2[-1] - top2[-2])
        else:
            pred_tx = ""
            logit_conf = float("nan")
            logit_margin = float("nan")
        rows.append({
            "role": role,
            "tx_id": tx,
            "rx_id": rx,
            "day_id": day,
            "sig_id": sig,
            "feature": z,
            "logit_pred_tx": pred_tx,
            "logit_conf": logit_conf,
            "logit_margin": logit_margin,
            "is_target_old": role == "target_old" and tx in old_set,
            "is_unknown": role == "target_unknown",
            "is_source_old": role == "source" and tx in old_set,
            "is_proxy_unknown": role == "proxy_unknown",
        })
    return rows


def _select_support(rows: list[dict], source_tx_ids: list[str], k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    support = []
    query = []
    for tx in source_tx_ids:
        tx_idx = [i for i, r in enumerate(rows) if r["role"] == "target_old" and r["tx_id"] == tx]
        ordered = sorted(tx_idx, key=lambda i: _stable_score((rows[i]["rx_id"], rows[i]["tx_id"], rows[i]["day_id"], rows[i]["sig_id"]), seed))
        support.extend(ordered[: min(int(k), len(ordered))])
        query.extend(ordered[min(int(k), len(ordered)) :])
    return np.asarray(support, dtype=np.int64), np.asarray(query, dtype=np.int64)


def _make_prototypes(rows: list[dict], idx: np.ndarray, source_tx_ids: list[str], shrink: float, source_idx: np.ndarray | None = None) -> np.ndarray:
    feats = np.stack([r["feature"] for r in rows], axis=0)
    protos = []
    for tx in source_tx_ids:
        target_members = [i for i in idx.tolist() if rows[i]["tx_id"] == tx]
        if not target_members:
            raise ValueError(f"no support for {tx}")
        target_proto = feats[target_members].mean(axis=0)
        if source_idx is not None and float(shrink) > 0:
            source_members = [i for i in source_idx.tolist() if rows[i]["tx_id"] == tx]
            if source_members:
                source_proto = feats[source_members].mean(axis=0)
                target_proto = (1.0 - float(shrink)) * target_proto + float(shrink) * source_proto
        protos.append(target_proto.astype(np.float32))
    return np.stack(protos, axis=0)


def _score_rows(rows: list[dict], query_idx: np.ndarray, protos: np.ndarray, source_tx_ids: list[str], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats = np.stack([rows[i]["feature"] for i in query_idx.tolist()], axis=0)
    if metric == "cosine":
        sims = _l2_normalize(feats) @ _l2_normalize(protos).T
    elif metric == "neg_l2":
        sims = -((feats[:, None, :] - protos[None, :, :]) ** 2).sum(axis=2)
    else:
        raise ValueError(metric)
    pred = sims.argmax(axis=1)
    sorted_s = np.sort(sims, axis=1)
    max_score = sorted_s[:, -1]
    margin = sorted_s[:, -1] - sorted_s[:, -2] if sims.shape[1] >= 2 else sorted_s[:, -1]
    pred_tx = np.asarray([source_tx_ids[int(i)] for i in pred], dtype=object)
    return pred_tx, max_score.astype(np.float64), margin.astype(np.float64)


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
    far = _safe_rate(unknown_accepted, unknown_total)
    old_drop = 100.0 * (closed_acc - full_acc)
    return {
        "unknown_FAR": far,
        "known_closed_accuracy_no_reject": closed_acc,
        "known_full_accuracy_after_reject": full_acc,
        "old_drop_pp_vs_closed": old_drop,
        "known_coverage": _safe_rate(known_accepted, known_total),
        "known_accepted_accuracy": _safe_rate(known_correct_after, known_accepted),
        "known_query_count": known_total,
        "unknown_query_count": unknown_total,
        "passes_unknown_far_target": bool(far <= 0.05),
        "passes_old_drop_target": bool(old_drop <= 2.0),
        "passes_dual_target": bool(far <= 0.05 and old_drop <= 2.0),
    }


def _joint_excess(row: dict) -> float:
    far = float(row.get("unknown_FAR", np.nan))
    drop = float(row.get("old_drop_pp_vs_closed", np.nan))
    return max(0.0, far - 0.05) + max(0.0, drop - 2.0) / 100.0


def _target_label_oracle(score: np.ndarray, known: np.ndarray, unknown: np.ndarray, closed: np.ndarray) -> tuple[dict, float]:
    valid = np.isfinite(score)
    if not valid.any():
        return _metrics(np.zeros_like(score, dtype=bool), known, unknown, closed), float("nan")
    s = score[valid]
    k = known[valid].astype(np.int64)
    u = unknown[valid].astype(np.int64)
    c = (known[valid] & closed[valid]).astype(np.int64)
    order = np.argsort(-s)
    s = s[order]
    k = k[order]
    u = u[order]
    c = c[order]
    # A threshold accepts all rows with score >= threshold. Evaluate only at
    # the last index of each tied score so all ties are handled consistently.
    last = np.r_[np.where(s[:-1] != s[1:])[0], len(s) - 1]
    ck = np.cumsum(k)[last].astype(np.float64)
    cu = np.cumsum(u)[last].astype(np.float64)
    cc = np.cumsum(c)[last].astype(np.float64)
    known_total = float(known.sum())
    unknown_total = float(unknown.sum())
    known_closed = float((known & closed).sum())
    closed_acc = np.nan if known_total <= 0 else known_closed / known_total
    far = np.full_like(cu, np.nan, dtype=np.float64) if unknown_total <= 0 else cu / unknown_total
    full = np.full_like(cc, np.nan, dtype=np.float64) if known_total <= 0 else cc / known_total
    drop = 100.0 * (closed_acc - full)
    coverage = np.full_like(ck, np.nan, dtype=np.float64) if known_total <= 0 else ck / known_total
    accepted_acc = np.divide(cc, ck, out=np.full_like(cc, np.nan, dtype=np.float64), where=ck > 0)
    far_excess = np.maximum(0.0, far - 0.05)
    drop_excess = np.maximum(0.0, drop - 2.0) / 100.0
    # Lexicographic choice: nearest dual objective, then lower FAR excess,
    # lower old-drop excess, higher full old accuracy.
    keys = np.lexsort((-full, drop_excess, far_excess, far_excess + drop_excess))
    best_i = int(keys[0])
    row = {
        "unknown_FAR": float(far[best_i]),
        "known_closed_accuracy_no_reject": float(closed_acc),
        "known_full_accuracy_after_reject": float(full[best_i]),
        "old_drop_pp_vs_closed": float(drop[best_i]),
        "known_coverage": float(coverage[best_i]),
        "known_accepted_accuracy": float(accepted_acc[best_i]),
        "known_query_count": int(known.sum()),
        "unknown_query_count": int(unknown.sum()),
        "passes_unknown_far_target": bool(float(far[best_i]) <= 0.05),
        "passes_old_drop_target": bool(float(drop[best_i]) <= 2.0),
        "passes_dual_target": bool(float(far[best_i]) <= 0.05 and float(drop[best_i]) <= 2.0),
    }
    return row, float(s[last[best_i]])


def _run_one(npz_path: Path, feature_tag: str, source_tx_ids: list[str], args: argparse.Namespace) -> list[dict]:
    payload = _load_npz(npz_path)
    rows = _group_rows(payload, source_tx_ids)
    source_idx = np.asarray([i for i, r in enumerate(rows) if r["is_source_old"]], dtype=np.int64)
    proxy_idx = np.asarray([i for i, r in enumerate(rows) if r["is_proxy_unknown"]], dtype=np.int64)
    unknown_idx = np.asarray([i for i, r in enumerate(rows) if r["is_unknown"]], dtype=np.int64)
    target_rxs = sorted({r["rx_id"] for r in rows if r["role"] in {"target_old", "target_unknown"}})
    out = []
    for k in [int(x) for x in str(args.k_values).split(",") if str(x).strip()]:
        support_idx, target_query_idx = _select_support(rows, source_tx_ids, k, int(args.seed))
        if support_idx.size == 0 or target_query_idx.size == 0 or unknown_idx.size == 0:
            continue
        eval_idx = np.concatenate([target_query_idx, unknown_idx])
        known = np.asarray([rows[i]["is_target_old"] for i in eval_idx.tolist()], dtype=bool)
        unknown = np.asarray([rows[i]["is_unknown"] for i in eval_idx.tolist()], dtype=bool)
        for shrink in [float(x) for x in str(args.shrink_values).split(",") if str(x).strip()]:
            protos = _make_prototypes(rows, support_idx, source_tx_ids, shrink=shrink, source_idx=source_idx)
            cal_idx = np.concatenate([support_idx, proxy_idx])
            cal_is_support = np.asarray([i in set(support_idx.tolist()) for i in cal_idx.tolist()], dtype=bool)
            cal_is_proxy = np.asarray([rows[i]["is_proxy_unknown"] for i in cal_idx.tolist()], dtype=bool)
            for metric in [x.strip() for x in str(args.metrics).split(",") if x.strip()]:
                pred_tx, max_score, margin = _score_rows(rows, eval_idx, protos, source_tx_ids, metric)
                true_tx = np.asarray([rows[i]["tx_id"] for i in eval_idx.tolist()], dtype=object)
                closed = known & (pred_tx == true_tx)
                cal_pred, cal_max, cal_margin = _score_rows(rows, cal_idx, protos, source_tx_ids, metric)
                for score_name, score, cal_score in [
                    ("max_score", max_score, cal_max),
                    ("margin", margin, cal_margin),
                ]:
                    oracle, oracle_threshold = _target_label_oracle(score, known, unknown, closed)
                    oracle.update({
                        "run_id": npz_path.parent.parent.name if npz_path.name.endswith(".npz") else npz_path.stem,
                        "feature_tag": feature_tag,
                        "target_rx": ",".join(target_rxs),
                        "mode": "phase2b_target_old_kshot_calib_oracle",
                        "k_shot": k,
                        "support_count": int(support_idx.size),
                        "target_old_query_count": int(target_query_idx.size),
                        "source_proxy_count": int(proxy_idx.size),
                        "shrink_to_source": shrink,
                        "metric": metric,
                        "score_name": score_name,
                        "threshold_policy": "target_label_oracle",
                        "support_accept_quantile": float("nan"),
                        "proxy_far_quantile": float("nan"),
                        "threshold": oracle_threshold,
                        "support_threshold": float("nan"),
                        "proxy_threshold": float("nan"),
                        "uses_target_clean": False,
                        "uses_target_old_support": True,
                        "uses_unknown_query_for_threshold": True,
                    })
                    out.append(oracle)
                    support_scores = cal_score[cal_is_support]
                    proxy_scores = cal_score[cal_is_proxy]
                    for policy in ["support_accept", "proxy_far", "max_support_proxy", "mean_support_proxy"]:
                        for support_q in [0.001, 0.005, 0.010, 0.020, 0.050]:
                            proxy_qs = [0.95] if policy == "support_accept" else [0.90, 0.95, 0.97, 0.99]
                            for proxy_q in proxy_qs:
                                support_t = float(np.quantile(support_scores, support_q))
                                proxy_t = float(np.quantile(proxy_scores, proxy_q)) if proxy_scores.size else support_t
                                if policy == "support_accept":
                                    threshold = support_t
                                elif policy == "proxy_far":
                                    threshold = proxy_t
                                elif policy == "max_support_proxy":
                                    threshold = max(support_t, proxy_t)
                                else:
                                    threshold = 0.5 * (support_t + proxy_t)
                                accept = score >= threshold
                                m = _metrics(accept, known, unknown, closed)
                                m.update({
                                    "run_id": npz_path.parent.parent.name if npz_path.name.endswith(".npz") else npz_path.stem,
                                    "feature_tag": feature_tag,
                                    "target_rx": ",".join(target_rxs),
                                    "mode": "phase2b_target_old_kshot_calib",
                                    "k_shot": k,
                                    "support_count": int(support_idx.size),
                                    "target_old_query_count": int(target_query_idx.size),
                                    "source_proxy_count": int(proxy_idx.size),
                                    "shrink_to_source": shrink,
                                    "metric": metric,
                                    "score_name": score_name,
                                    "threshold_policy": policy,
                                    "support_accept_quantile": support_q,
                                    "proxy_far_quantile": proxy_q,
                                    "threshold": threshold,
                                    "support_threshold": support_t,
                                    "proxy_threshold": proxy_t,
                                    "uses_target_clean": False,
                                    "uses_target_old_support": True,
                                    "uses_unknown_query_for_threshold": False,
                                })
                                out.append(m)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--run_glob", default="phase1_adv3b02_multiview_keepold_*_20260702")
    p.add_argument("--feature_relpaths", default="ADV3B02_CORE90_SOFT_E200_PHASE1_SATUNKNOWN_SINGLEVIEW/features_satunknown_singleview.npz,LEOADAPT3_LINR_COS/features_leo_repaired.npz,LEOADAPT3_MLP_ID/features_leo_repaired.npz")
    p.add_argument("--source_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    p.add_argument("--k_values", default="1,2,5,10,20,50")
    p.add_argument("--shrink_values", default="0.0,0.1,0.25,0.5")
    p.add_argument("--metrics", default="cosine,neg_l2")
    p.add_argument("--seed", type=int, default=4070361)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source_tx_ids = parse_tx_ids(args.source_tx_ids)
    relpaths = [x.strip() for x in str(args.feature_relpaths).split(",") if x.strip()]
    rows = []
    for run_dir in sorted(args.runs_root.glob(args.run_glob)):
        for rel in relpaths:
            npz_path = run_dir / rel
            if npz_path.is_file():
                tag = rel.split("/")[0]
                rows.extend(_run_one(npz_path, tag, source_tx_ids, args))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ["run_id"]
    leading = ["run_id", "feature_tag", "target_rx", "mode", "k_shot", "metric", "score_name", "threshold_policy"]
    fields = leading + [f for f in fields if f not in leading]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print({"rows": len(rows), "dual_pass": sum(1 for r in rows if bool(r.get("passes_dual_target"))), "out_csv": str(args.out_csv)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
