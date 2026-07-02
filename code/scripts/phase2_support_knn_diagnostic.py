#!/usr/bin/env python3
"""Summarize target-support prototype/kNN diagnostics for Phase2 feature runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(denom, 1e-12)


def _knn_predict(query: np.ndarray, support: np.ndarray, support_labels: np.ndarray, k: int) -> np.ndarray:
    similarities = query @ support.T
    predictions: list[str] = []
    k_eff = min(max(1, int(k)), int(support.shape[0]))
    for row in similarities:
        top_idx = np.argsort(row)[-k_eff:][::-1]
        counts: dict[str, int] = {}
        scores: dict[str, float] = {}
        for idx in top_idx:
            label = str(support_labels[int(idx)])
            counts[label] = counts.get(label, 0) + 1
            scores[label] = scores.get(label, 0.0) + float(row[int(idx)])
        predictions.append(max(scores, key=lambda label: (counts[label], scores[label])))
    return np.asarray(predictions, dtype=object)


def _prototype_predict(query: np.ndarray, support: np.ndarray, support_labels: np.ndarray) -> np.ndarray:
    labels = sorted({str(value) for value in support_labels.tolist()})
    prototypes = []
    for label in labels:
        prototypes.append(support[support_labels == label].mean(axis=0))
    prototype_matrix = _normalize_rows(np.vstack(prototypes))
    similarities = query @ prototype_matrix.T
    return np.asarray([labels[int(idx)] for idx in np.argmax(similarities, axis=1)], dtype=object)


def _accuracy(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float | None:
    if not bool(mask.any()):
        return None
    return float(np.mean(pred[mask] == truth[mask]))


def _harmonic(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if a + b <= 0:
        return 0.0
    return float(2 * a * b / (a + b))


def _receiver_from_name(name: str) -> str:
    markers = {
        "RX3_19": "3-19",
        "RX7_14": "7-14",
        "RX7_7": "7-7",
        "RX8_8": "8-8",
    }
    for marker, receiver in markers.items():
        if marker in name:
            return receiver
    return ""


def summarize_candidate(metrics_path: Path, methods: list[str], old_target: float, seen_new_target: float) -> list[dict[str, Any]]:
    candidate_dir = metrics_path.parent
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    manifest = payload["manifest"]
    split = manifest["split_indices_by_role"]
    feature_npz = candidate_dir / "features.npz"
    data = np.load(feature_npz, allow_pickle=True)
    features = _normalize_rows(data["features"])
    tx_ids = np.asarray(data["tx_ids"], dtype=object)

    support_idx = np.asarray(split["target_old_support"] + split["new_support"], dtype=int)
    old_query_idx = np.asarray(split["target_old_query"], dtype=int)
    new_query_idx = np.asarray(split["new_query"], dtype=int)
    query_idx = np.concatenate([old_query_idx, new_query_idx])

    support = features[support_idx]
    support_labels = tx_ids[support_idx]
    query = features[query_idx]
    truth = tx_ids[query_idx]
    old_mask = np.isin(query_idx, old_query_idx)
    new_mask = np.isin(query_idx, new_query_idx)

    rows: list[dict[str, Any]] = []
    for method in methods:
        if method == "proto":
            pred = _prototype_predict(query, support, support_labels)
        elif method.startswith("knn"):
            pred = _knn_predict(query, support, support_labels, int(method[3:]))
        else:
            raise ValueError(f"Unsupported method: {method}")

        old_acc = _accuracy(pred, truth, old_mask)
        seen_new_acc = _accuracy(pred, truth, new_mask)
        per_new: dict[str, float | None] = {}
        for tx_id in manifest.get("new_tx_ids", []):
            tx_mask = (truth == tx_id) & new_mask
            per_new[str(tx_id)] = _accuracy(pred, truth, tx_mask)

        rows.append(
            {
                "candidate": candidate_dir.name,
                "receiver": _receiver_from_name(candidate_dir.name),
                "strategy": "OLDRESCUE" if "OLDRESCUE" in candidate_dir.name else "BALANCED",
                "method": method,
                "old_acc": old_acc,
                "seen_new_acc": seen_new_acc,
                "H_old_new": _harmonic(old_acc, seen_new_acc),
                "passes_old_target": old_acc is not None and old_acc >= old_target,
                "passes_seen_new_target": seen_new_acc is not None and seen_new_acc >= seen_new_target,
                "passes_joint_target": (
                    old_acc is not None
                    and seen_new_acc is not None
                    and old_acc >= old_target
                    and seen_new_acc >= seen_new_target
                ),
                "per_new_acc": per_new,
                "support_count": int(support_idx.size),
                "old_query_count": int(old_query_idx.size),
                "new_query_count": int(new_query_idx.size),
                "feature_npz": str(feature_npz),
                "metrics_json": str(metrics_path),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--methods", default="proto,knn1,knn3,knn5")
    parser.add_argument("--old_target", type=float, default=0.80)
    parser.add_argument("--seen_new_target", type=float, default=0.65)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    methods = [part.strip() for part in str(args.methods).split(",") if part.strip()]
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(run_root.glob("*/metrics.json")):
        rows.extend(summarize_candidate(metrics_path, methods, args.old_target, args.seen_new_target))

    rows.sort(
        key=lambda row: (
            bool(row["passes_joint_target"]),
            float(row["H_old_new"] or 0.0),
            float(row["seen_new_acc"] or 0.0),
            float(row["old_acc"] or 0.0),
        ),
        reverse=True,
    )

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_root": str(run_root),
        "methods": methods,
        "old_target": float(args.old_target),
        "seen_new_target": float(args.seen_new_target),
        "candidate_method_count": len(rows),
        "joint_pass_count": int(sum(1 for row in rows if row["passes_joint_target"])),
        "rows": rows,
    }
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    fieldnames = [
        "candidate",
        "receiver",
        "strategy",
        "method",
        "old_acc",
        "seen_new_acc",
        "H_old_new",
        "passes_joint_target",
        "per_new_acc",
        "support_count",
        "old_query_count",
        "new_query_count",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key) for key in fieldnames}
            csv_row["per_new_acc"] = json.dumps(row.get("per_new_acc", {}), ensure_ascii=False, sort_keys=True)
            writer.writerow(csv_row)

    print(json.dumps({"joint_pass_count": summary["joint_pass_count"], "best": rows[:3]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
