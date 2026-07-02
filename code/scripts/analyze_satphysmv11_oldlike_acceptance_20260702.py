#!/usr/bin/env python
"""Evaluate source-calibrated old-like acceptance rules on SATPHY11 features."""

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
        return text[:-2]
    return text


def _softmax(x: np.ndarray) -> np.ndarray:
    y = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(y)
    return e / np.sum(e, axis=-1, keepdims=True)


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values.astype(np.float64), float(q)))


def _load_grouped(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=True) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        logits = np.asarray(data["tx_logits"], dtype=np.float32)
        n = int(features.shape[0])
        fields = {
            key: [_canon(v) for v in np.asarray(data[key]).reshape(-1).tolist()]
            for key in ["dataset_role", "tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids"]
        }
        manifest = {}
        if "manifest_json" in data.files:
            try:
                manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
            except Exception:
                manifest = {}
    grouped: dict[tuple[str, str, str, str, str, str], list[int]] = defaultdict(list)
    for i in range(n):
        grouped[
            (
                fields["dataset_role"][i],
                fields["tx_ids"][i],
                fields["rx_ids"][i],
                fields["day_ids"][i],
                fields["eq_ids"][i],
                fields["sig_ids"][i],
            )
        ].append(i)
    rows: list[dict[str, object]] = []
    z_mean: list[np.ndarray] = []
    class_to_tx = {i: tx for i, tx in enumerate(SOURCE_TX_IDS)}
    for key, idx in sorted(grouped.items()):
        idx_a = np.asarray(idx, dtype=np.int64)
        z = features[idx_a]
        lo = logits[idx_a]
        probs = _softmax(lo)
        mean_logits = lo.mean(axis=0)
        mean_probs = _softmax(mean_logits.reshape(1, -1))[0]
        pred_class = int(np.argmax(mean_logits))
        pred_tx = class_to_tx.get(pred_class, str(pred_class))
        pred_each = np.argmax(probs, axis=1)
        conf_each = np.max(probs, axis=1)
        sorted_probs = np.sort(mean_probs)
        margin = float(sorted_probs[-1] - sorted_probs[-2]) if sorted_probs.size >= 2 else 0.0
        entropy = float(-np.sum(mean_probs * np.log(np.clip(mean_probs, 1e-12, 1.0))))
        energy = float(-np.log(np.sum(np.exp(mean_logits - np.max(mean_logits)))) - np.max(mean_logits))
        z_norm = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
        center = z.mean(axis=0, keepdims=True)
        center_norm = center / max(float(np.linalg.norm(center)), 1e-8)
        cos = (z_norm @ center_norm.T).reshape(-1)
        rows.append(
            {
                "role": key[0],
                "tx_id": key[1],
                "pred_tx": pred_tx,
                "mean_conf": float(np.max(mean_probs)),
                "min_conf": float(np.min(conf_each)),
                "vote_frac": float(np.mean(pred_each == pred_class)),
                "margin": margin,
                "entropy": entropy,
                "energy": energy,
                "mean_cosine": float(np.mean(cos)),
                "min_cosine": float(np.min(cos)),
            }
        )
        z_mean.append(z.mean(axis=0))
    x = np.stack(z_mean, axis=0)
    meta = {key: np.asarray([row[key] for row in rows]) for key in rows[0].keys()}
    return x, meta, manifest


def _prototype_scores(x: np.ndarray, meta: dict[str, np.ndarray], metric: str) -> np.ndarray:
    source_mask = (meta["role"] == "source") & np.isin(meta["tx_id"], np.asarray(SOURCE_TX_IDS))
    source_x = x[source_mask]
    mu = source_x.mean(axis=0, keepdims=True)
    sigma = source_x.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    z = (x - mu) / sigma
    protos: dict[str, np.ndarray] = {}
    vars_: dict[str, np.ndarray] = {}
    for tx in SOURCE_TX_IDS:
        cls = z[source_mask & (meta["tx_id"] == tx)]
        if cls.size == 0:
            continue
        protos[tx] = cls.mean(axis=0)
        vars_[tx] = np.maximum(cls.var(axis=0), 1e-4)
    fallback = z[source_mask].mean(axis=0)
    fallback_var = np.maximum(z[source_mask].var(axis=0), 1e-4)
    scores = []
    for i, pred_tx in enumerate(meta["pred_tx"]):
        proto = protos.get(str(pred_tx), fallback)
        var = vars_.get(str(pred_tx), fallback_var)
        if metric == "cosine":
            denom = max(float(np.linalg.norm(z[i]) * np.linalg.norm(proto)), 1e-8)
            score = 1.0 - float(np.dot(z[i], proto) / denom)
        elif metric == "diag_mahalanobis":
            score = float(np.mean(((z[i] - proto) ** 2) / var))
        else:
            score = float(np.linalg.norm(z[i] - proto))
        scores.append(score)
    return np.asarray(scores, dtype=np.float64)


def _eval(meta: dict[str, np.ndarray], accepted: np.ndarray) -> tuple[float, float, float, float, float]:
    source_known = np.asarray(SOURCE_TX_IDS)
    known = (meta["role"] == "target_old") & np.isin(meta["tx_id"], source_known)
    unknown = meta["role"] == "target_unknown"
    closed_correct = known & (meta["pred_tx"] == meta["tx_id"])
    accepted_correct = accepted & closed_correct
    closed_acc = float(np.mean(closed_correct[known])) if np.any(known) else 0.0
    full_acc = float(np.mean(accepted_correct[known])) if np.any(known) else 0.0
    far = float(np.mean(accepted[unknown])) if np.any(unknown) else 0.0
    coverage = float(np.mean(accepted[known])) if np.any(known) else 0.0
    accepted_acc = float(np.mean(closed_correct[known][accepted[known]])) if np.any(known & accepted) else 0.0
    return far, 100.0 * (closed_acc - full_acc), coverage, closed_acc, accepted_acc


def _format(rows: list[dict[str, object]], cols: list[str], limit: int = 30) -> str:
    view = rows[:limit]
    widths = {c: len(c) for c in cols}
    rendered = []
    for row in view:
        item = {}
        for col in cols:
            val = row.get(col, "")
            text = f"{val:.6g}" if isinstance(val, float) else str(val)
            item[col] = text
            widths[col] = max(widths[col], len(text))
        rendered.append(item)
    lines = [" ".join(c.ljust(widths[c]) for c in cols), " ".join("-" * widths[c] for c in cols)]
    for row in rendered:
        lines.append(" ".join(row[c].ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def main() -> int:
    base = Path("/home/szu2070436088/2510044040/CV-SincNet/runs")
    feature_paths = sorted(base.glob("phase1_adv3b02_satphysmv11_*_20260702/ADV3B02_CORE90_SOFT_E200_PHASE1_SATPHYSMV11/features_satphysmv11.npz"))
    # q means source acceptance quantile. Low-side scores use 1-q thresholds.
    q_grid = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
    results: list[dict[str, object]] = []
    for path in feature_paths:
        x, meta, manifest = _load_grouped(path)
        proto_cos = _prototype_scores(x, meta, "cosine")
        proto_mah = _prototype_scores(x, meta, "diag_mahalanobis")
        source = (meta["role"] == "source") & np.isin(meta["tx_id"], np.asarray(SOURCE_TX_IDS))
        closed_ok = meta["pred_tx"] == meta["tx_id"]
        source_ok = source & closed_ok
        if not np.any(source_ok):
            source_ok = source
        score_map = {
            "mean_conf": np.asarray(meta["mean_conf"], dtype=np.float64),
            "min_conf": np.asarray(meta["min_conf"], dtype=np.float64),
            "vote_frac": np.asarray(meta["vote_frac"], dtype=np.float64),
            "margin": np.asarray(meta["margin"], dtype=np.float64),
            "mean_cosine": np.asarray(meta["mean_cosine"], dtype=np.float64),
            "min_cosine": np.asarray(meta["min_cosine"], dtype=np.float64),
            "neg_entropy": -np.asarray(meta["entropy"], dtype=np.float64),
            "neg_proto_cos": -proto_cos,
            "neg_proto_mah": -proto_mah,
        }
        for q in q_grid:
            thresholds = {name: _quantile(values[source_ok], 1.0 - q) for name, values in score_map.items()}
            families = {
                "logit_only": ["mean_conf", "vote_frac", "margin"],
                "stability_only": ["mean_conf", "vote_frac", "mean_cosine", "min_cosine"],
                "proto_cos": ["mean_conf", "vote_frac", "neg_proto_cos"],
                "proto_mah": ["mean_conf", "vote_frac", "neg_proto_mah"],
                "all_cos": ["mean_conf", "vote_frac", "margin", "mean_cosine", "neg_proto_cos"],
                "all_mah": ["mean_conf", "vote_frac", "margin", "mean_cosine", "neg_proto_mah"],
                "strict_all": list(score_map.keys()),
            }
            for family, names in families.items():
                accepted = np.ones(len(x), dtype=bool)
                for name in names:
                    accepted &= score_map[name] >= thresholds[name]
                far, drop, coverage, closed_acc, accepted_acc = _eval(meta, accepted)
                score = max(0.0, far - 0.05) + max(0.0, drop - 2.0) / 100.0
                results.append(
                    {
                        "run_id": path.parent.parent.name,
                        "family": family,
                        "q": q,
                        "far": far,
                        "drop": drop,
                        "coverage": coverage,
                        "closed_acc": closed_acc,
                        "accepted_acc": accepted_acc,
                        "dual": far <= 0.05 and drop <= 2.0,
                        "score": score,
                        "view_count": manifest.get("satellite_tta_view_count", ""),
                    }
                )
    print({"feature_files": len(feature_paths), "rows": len(results)})
    print(
        {
            "dual_pass": sum(1 for r in results if bool(r["dual"])),
            "far_pass": sum(1 for r in results if float(r["far"]) <= 0.05),
            "drop_pass": sum(1 for r in results if float(r["drop"]) <= 2.0),
        }
    )
    cols = ["run_id", "family", "q", "far", "drop", "coverage", "closed_acc", "accepted_acc", "dual", "score"]
    print(_format(sorted(results, key=lambda r: (float(r["score"]), float(r["far"]), float(r["drop"]))), cols))
    out = Path("/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_satphysmv11_constrained_matrix_20260702/satphysmv11_oldlike_acceptance_audit.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["run_id"])
        writer.writeheader()
        writer.writerows(results)
    print({"audit_csv": str(out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
