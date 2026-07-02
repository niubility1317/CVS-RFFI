#!/usr/bin/env python
"""Diagnose SATPHY11 binary reject + source prototype old-class rescue."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SOURCE_TX_IDS = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]


def _canon(value: object) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _softmax(x: np.ndarray) -> np.ndarray:
    y = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(y)
    return e / np.sum(e, axis=-1, keepdims=True)


def _quantile(values: list[float], q: float, fallback: float = 0.0) -> float:
    if not values:
        return float(fallback)
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(q)))


def _read_score_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _eval(rows: list[dict[str, str]], accepted: np.ndarray) -> tuple[float, float, float]:
    known_idx = [i for i, row in enumerate(rows) if int(float(row.get("is_known_query", "0"))) == 1]
    unknown_idx = [i for i, row in enumerate(rows) if int(float(row.get("is_unknown_query", "0"))) == 1]
    closed_correct = sum(int(float(rows[i].get("closed_correct_known", "0"))) for i in known_idx)
    known_correct_accepted = sum(
        int(float(rows[i].get("closed_correct_known", "0"))) for i in known_idx if bool(accepted[i])
    )
    unknown_accepted = sum(1 for i in unknown_idx if bool(accepted[i]))
    known_closed_acc = closed_correct / max(1, len(known_idx))
    known_full_acc = known_correct_accepted / max(1, len(known_idx))
    far = unknown_accepted / max(1, len(unknown_idx))
    cov = sum(1 for i in known_idx if bool(accepted[i])) / max(1, len(known_idx))
    return far, 100.0 * (known_closed_acc - known_full_acc), cov


def _load_grouped(feature_npz: Path) -> tuple[np.ndarray, list[dict[str, object]], list[str]]:
    with np.load(feature_npz, allow_pickle=True) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        logits = np.asarray(data["tx_logits"], dtype=np.float32)
        n = int(features.shape[0])
        fields = {}
        for key in ["dataset_role", "tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids"]:
            fields[key] = [_canon(v) for v in np.asarray(data[key]).reshape(-1).tolist()]
        manifest = {}
        if "manifest_json" in data.files:
            try:
                manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
            except Exception:
                manifest = {}
    groups: dict[tuple[str, str, str, str, str, str], list[int]] = defaultdict(list)
    for i in range(n):
        key = (
            fields["dataset_role"][i],
            fields["tx_ids"][i],
            fields["rx_ids"][i],
            fields["day_ids"][i],
            fields["eq_ids"][i],
            fields["sig_ids"][i],
        )
        groups[key].append(i)
    out_features: list[np.ndarray] = []
    meta: list[dict[str, object]] = []
    class_to_tx = {i: tx for i, tx in enumerate(SOURCE_TX_IDS)}
    for key, idx in sorted(groups.items()):
        z = features[np.asarray(idx, dtype=np.int64)].mean(axis=0)
        mean_logits = logits[np.asarray(idx, dtype=np.int64)].mean(axis=0)
        pred_class = int(np.argmax(mean_logits))
        pred_tx = class_to_tx.get(pred_class, str(pred_class))
        out_features.append(z)
        meta.append({"role": key[0], "tx_id": key[1], "pred_tx": pred_tx})
    return np.stack(out_features, axis=0), meta, list(manifest.keys())


def _standardize(source_x: np.ndarray, x: np.ndarray) -> np.ndarray:
    mean = source_x.mean(axis=0, keepdims=True)
    std = source_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (x - mean) / std


def _prototype_distances(x: np.ndarray, meta: list[dict[str, object]], *, metric: str) -> tuple[np.ndarray, list[str]]:
    source_known = {str(tx) for tx in SOURCE_TX_IDS}
    source_mask = np.asarray([m["role"] == "source" and str(m["tx_id"]) in source_known for m in meta], dtype=bool)
    xz = _standardize(x[source_mask], x)
    protos: dict[str, np.ndarray] = {}
    vars_: dict[str, np.ndarray] = {}
    for tx in SOURCE_TX_IDS:
        idx = [i for i, m in enumerate(meta) if source_mask[i] and str(m["tx_id"]) == tx]
        if not idx:
            continue
        cls = xz[np.asarray(idx, dtype=np.int64)]
        protos[tx] = cls.mean(axis=0)
        vars_[tx] = np.maximum(cls.var(axis=0), 1e-4)
    fallback = xz[source_mask].mean(axis=0)
    fallback_var = np.maximum(xz[source_mask].var(axis=0), 1e-4)
    scores: list[float] = []
    pred_txs: list[str] = []
    for i, m in enumerate(meta):
        pred_tx = str(m["pred_tx"])
        proto = protos.get(pred_tx, fallback)
        var = vars_.get(pred_tx, fallback_var)
        z = xz[i]
        if metric == "cosine":
            denom = max(float(np.linalg.norm(z) * np.linalg.norm(proto)), 1e-8)
            score = 1.0 - float(np.dot(z, proto) / denom)
        elif metric == "diag_mahalanobis":
            score = float(np.mean(((z - proto) ** 2) / var))
        else:
            score = float(np.linalg.norm(z - proto))
        scores.append(score)
        pred_txs.append(pred_tx)
    return np.asarray(scores, dtype=np.float64), pred_txs


def _format(rows: list[dict[str, object]], cols: list[str], limit: int = 30) -> str:
    rows = rows[:limit]
    widths = {col: len(col) for col in cols}
    rendered = []
    for row in rows:
        item = {}
        for col in cols:
            value = row.get(col, "")
            text = f"{value:.6g}" if isinstance(value, float) else str(value)
            item[col] = text
            widths[col] = max(widths[col], len(text))
        rendered.append(item)
    lines = [" ".join(col.ljust(widths[col]) for col in cols), " ".join("-" * widths[col] for col in cols)]
    for row in rendered:
        lines.append(" ".join(row[col].ljust(widths[col]) for col in cols))
    return "\n".join(lines)


def main() -> int:
    base = Path("/home/szu2070436088/2510044040/CV-SincNet/runs")
    score_paths = sorted(base.glob("phase1_adv3b02_satphysmv11_*_20260702/SATPHY11_*/score_table.csv"))
    q_grid = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
    results: list[dict[str, object]] = []
    feature_cache: dict[str, tuple[np.ndarray, list[dict[str, object]], dict[str, np.ndarray]]] = {}
    for score_path in score_paths:
        run_dir = score_path.parent.parent
        feature_npz = run_dir / "ADV3B02_CORE90_SOFT_E200_PHASE1_SATPHYSMV11" / "features_satphysmv11.npz"
        if str(feature_npz) not in feature_cache:
            x, meta, _ = _load_grouped(feature_npz)
            metric_scores = {
                "cosine": _prototype_distances(x, meta, metric="cosine")[0],
                "diag_mahalanobis": _prototype_distances(x, meta, metric="diag_mahalanobis")[0],
            }
            feature_cache[str(feature_npz)] = (x, meta, metric_scores)
        _, meta, metric_scores = feature_cache[str(feature_npz)]
        rows = _read_score_rows(score_path)
        if len(rows) != len(meta):
            raise RuntimeError(f"group mismatch for {score_path}: score_rows={len(rows)} grouped={len(meta)}")
        original = np.asarray([int(float(row.get("accepted", "0"))) == 1 for row in rows], dtype=bool)
        orig_far, orig_drop, orig_cov = _eval(rows, original)
        if orig_far > 0.20:
            continue
        source_mask = np.asarray([m["role"] == "source" for m in meta], dtype=bool)
        best: tuple[float, float, float, float, str, float] | None = None
        for metric, dist in metric_scores.items():
            for q in q_grid:
                threshold = _quantile(dist[source_mask].reshape(-1).tolist(), q, float(np.max(dist[source_mask])))
                proto_accept = dist <= threshold
                accepted = original | proto_accept
                far, drop, cov = _eval(rows, accepted)
                score = max(0.0, far - 0.05) + max(0.0, drop - 2.0) / 100.0
                cand = (score, far, drop, cov, metric, q)
                if best is None or cand < best:
                    best = cand
        if best is None:
            continue
        results.append(
            {
                "run_id": run_dir.name,
                "policy": score_path.parent.name,
                "orig_far": orig_far,
                "orig_drop": orig_drop,
                "orig_cov": orig_cov,
                "best_far": best[1],
                "best_drop": best[2],
                "best_cov": best[3],
                "metric": best[4],
                "source_q": best[5],
                "score": best[0],
            }
        )
    print({"score_tables": len(score_paths), "candidate_rows": len(results)})
    print(
        {
            "dual_pass": sum(1 for r in results if float(r["best_far"]) <= 0.05 and float(r["best_drop"]) <= 2.0),
            "far_pass": sum(1 for r in results if float(r["best_far"]) <= 0.05),
            "drop_pass": sum(1 for r in results if float(r["best_drop"]) <= 2.0),
        }
    )
    cols = ["run_id", "policy", "orig_far", "orig_drop", "best_far", "best_drop", "best_cov", "metric", "source_q", "score"]
    print(_format(sorted(results, key=lambda r: (float(r["score"]), float(r["best_far"]), float(r["best_drop"]))), cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
