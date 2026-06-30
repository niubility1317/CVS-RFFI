#!/usr/bin/env python
"""Evaluate target-old K-shot domain adaptation on saved z_id features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1.0e-8


def _parse_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _load_manifest(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "manifest_json" not in data:
        return {}
    raw = data["manifest_json"]
    try:
        item = raw.item() if getattr(raw, "shape", ()) == () else raw
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        if isinstance(item, str):
            return json.loads(item)
        if isinstance(item, dict):
            return item
    except Exception:
        return {}
    return {}


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def _domain_from_path(path: Path) -> str:
    match = re.search(r"stage2_short195s3_otherdomains_finaltest_20260630_1525_([^/\\]+)", str(path))
    if match:
        return match.group(1).replace("_", "-")
    match = re.search(r"_([0-9]+_[0-9]+)/[^/\\]+/features\.npz", str(path))
    if match:
        return match.group(1).replace("_", "-")
    return "unknown"


def _candidate_from_path(path: Path) -> str:
    return path.parent.name


def _take_role_indices(tx_ids: np.ndarray, roles: np.ndarray, tx: str, role: str) -> np.ndarray:
    return np.flatnonzero((tx_ids == str(tx)) & (roles == str(role))).astype(np.int64)


def _build_splits(path: Path, args: argparse.Namespace, k: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        manifest = _load_manifest(data)
        features = np.asarray(data[str(args.features_key)], dtype=np.float64)
        tx_ids = np.asarray(data[str(args.tx_ids_key)]).reshape(-1).astype(str)
        roles = np.asarray(data[str(args.role_key)]).reshape(-1).astype(str)
        source_txs = _parse_list(args.source_tx_ids) or [str(v) for v in manifest.get("source_tx_ids", [])]
        target_old_txs = _parse_list(args.target_old_tx_ids) or [str(v) for v in manifest.get("target_old_tx_ids", source_txs)]
        if not source_txs or not target_old_txs:
            raise ValueError(f"{path}: source_tx_ids and target_old_tx_ids are required")
        label_map = {tx: idx for idx, tx in enumerate(source_txs)}
        source_parts: list[np.ndarray] = []
        source_labels: list[int] = []
        support_parts: list[np.ndarray] = []
        support_labels: list[int] = []
        query_parts: list[np.ndarray] = []
        query_labels: list[int] = []
        per_tx_counts: dict[str, dict[str, int]] = {}
        for tx in target_old_txs:
            if tx not in label_map:
                raise ValueError(f"{path}: target-old tx {tx!r} not in source label map")
            label = label_map[tx]
            src_idx_all = _take_role_indices(tx_ids, roles, tx, str(args.source_role))
            tgt_idx_all = _take_role_indices(tx_ids, roles, tx, str(args.target_old_role))
            if src_idx_all.size < int(args.source_proto_per_tx):
                raise ValueError(f"{path}: tx {tx!r} has only {src_idx_all.size} source samples")
            if tgt_idx_all.size <= int(args.max_support_per_tx):
                raise ValueError(f"{path}: tx {tx!r} has only {tgt_idx_all.size} target-old samples")
            src_idx = src_idx_all[: int(args.source_proto_per_tx)]
            support_idx = tgt_idx_all[: int(k)]
            query_available = tgt_idx_all[int(args.max_support_per_tx) :]
            query_idx = query_available if int(args.query_per_tx) <= 0 else query_available[: int(args.query_per_tx)]
            if support_idx.size != int(k) or query_idx.size == 0:
                raise ValueError(f"{path}: tx {tx!r} cannot satisfy K={k} and query_per_tx={args.query_per_tx}")
            source_parts.append(src_idx)
            source_labels.extend([label] * int(src_idx.size))
            support_parts.append(support_idx)
            support_labels.extend([label] * int(support_idx.size))
            query_parts.append(query_idx)
            query_labels.extend([label] * int(query_idx.size))
            per_tx_counts[tx] = {
                "source": int(src_idx.size),
                "support": int(support_idx.size),
                "query": int(query_idx.size),
            }
        arrays = {
            "source_features": _normalize(features[np.concatenate(source_parts)]),
            "source_labels": np.asarray(source_labels, dtype=np.int64),
            "support_features": _normalize(features[np.concatenate(support_parts)]),
            "support_labels": np.asarray(support_labels, dtype=np.int64),
            "query_features": _normalize(features[np.concatenate(query_parts)]),
            "query_labels": np.asarray(query_labels, dtype=np.int64),
        }
        meta = {
            "manifest": manifest,
            "source_txs": source_txs,
            "target_old_txs": target_old_txs,
            "per_tx_counts": per_tx_counts,
        }
        return arrays, meta


def _centroids(features: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = []
    for label in range(num_classes):
        mask = labels == label
        if not np.any(mask):
            raise ValueError(f"missing class {label}")
        vec = features[mask].mean(axis=0)
        vec = vec / max(float(np.linalg.norm(vec)), EPS)
        out.append(vec)
    return np.stack(out, axis=0)


def _predict_centroid(query: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.argmax(np.matmul(query, centers.T), axis=1).astype(np.int64)


def _predict_knn(query: np.ndarray, train_x: np.ndarray, train_y: np.ndarray, k: int) -> np.ndarray:
    sims = np.matmul(query, train_x.T)
    kk = max(1, min(int(k), train_x.shape[0]))
    top = np.argpartition(sims, kth=sims.shape[1] - kk, axis=1)[:, -kk:]
    preds = []
    for i, idx in enumerate(top):
        labels = train_y[idx]
        scores = sims[i, idx]
        best_label = None
        best_tuple = None
        for label in sorted(set(int(v) for v in labels.tolist())):
            mask = labels == label
            vote = int(mask.sum())
            score = float(scores[mask].mean())
            item = (vote, score)
            if best_tuple is None or item > best_tuple:
                best_tuple = item
                best_label = int(label)
        preds.append(int(best_label))
    return np.asarray(preds, dtype=np.int64)


def _metrics(pred: np.ndarray, labels: np.ndarray, tx_labels: list[str]) -> dict[str, Any]:
    correct = pred == labels
    per_class = {}
    for idx, tx in enumerate(tx_labels):
        mask = labels == idx
        per_class[tx] = {
            "query_n": int(mask.sum()),
            "acc": float(correct[mask].mean()) if np.any(mask) else math.nan,
        }
    return {
        "old_acc": float(correct.mean()) if labels.size else math.nan,
        "query_n": int(labels.size),
        "correct_n": int(correct.sum()),
        "per_class": per_class,
    }


def evaluate_one(path: Path, args: argparse.Namespace, k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arrays, meta = _build_splits(path, args, k)
    num_classes = len(meta["source_txs"])
    source_centers = _centroids(arrays["source_features"], arrays["source_labels"], num_classes)
    support_centers = _centroids(arrays["support_features"], arrays["support_labels"], num_classes)
    query_x = arrays["query_features"]
    query_y = arrays["query_labels"]
    methods: list[tuple[str, np.ndarray | tuple[np.ndarray, np.ndarray, int]]] = []
    methods.append(("source_centroid", source_centers))
    methods.append(("support_centroid", support_centers))
    for rho in _parse_list(args.fusion_rhos):
        value = float(rho)
        centers = (1.0 - value) * source_centers + value * support_centers
        centers = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), EPS)
        methods.append((f"fused_centroid_rho{value:.2f}", centers))
    methods.append(("support_knn1", (arrays["support_features"], arrays["support_labels"], 1)))
    methods.append(("support_knn3", (arrays["support_features"], arrays["support_labels"], 3)))

    rows: list[dict[str, Any]] = []
    for method, model in methods:
        if isinstance(model, tuple):
            pred = _predict_knn(query_x, model[0], model[1], model[2])
        else:
            pred = _predict_centroid(query_x, model)
        metrics = _metrics(pred, query_y, meta["target_old_txs"])
        rows.append(
            {
                "feature_npz": str(path),
                "domain": _domain_from_path(path),
                "candidate": _candidate_from_path(path),
                "k_shot": int(k),
                "method": method,
                "old_acc": metrics["old_acc"],
                "query_n": metrics["query_n"],
                "correct_n": metrics["correct_n"],
                "per_class": metrics["per_class"],
                "source_proto_per_tx": int(args.source_proto_per_tx),
                "max_support_per_tx": int(args.max_support_per_tx),
                "query_per_tx": int(args.query_per_tx),
            }
        )
    payload = {
        "feature_npz": str(path),
        "domain": _domain_from_path(path),
        "candidate": _candidate_from_path(path),
        "k_shot": int(k),
        "source_txs": meta["source_txs"],
        "target_old_txs": meta["target_old_txs"],
        "per_tx_counts": meta["per_tx_counts"],
    }
    return rows, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-npz", action="append", default=[], help="Feature NPZ path or glob. Can be repeated.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--features-key", default="features")
    parser.add_argument("--tx-ids-key", default="tx_ids")
    parser.add_argument("--role-key", default="dataset_role")
    parser.add_argument("--source-role", default="source")
    parser.add_argument("--target-old-role", default="target_old")
    parser.add_argument("--source-tx-ids", default=None)
    parser.add_argument("--target-old-tx-ids", default=None)
    parser.add_argument("--k-shots", default="2,3,5,10")
    parser.add_argument("--source-proto-per-tx", type=int, default=48)
    parser.add_argument("--max-support-per-tx", type=int, default=10)
    parser.add_argument("--query-per-tx", type=int, default=0, help="0 means all target-old samples after max-support-per-tx.")
    parser.add_argument("--fusion-rhos", default="0.25,0.50,0.75,1.00")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_paths: list[Path] = []
    for item in args.feature_npz:
        if any(ch in item for ch in "*?["):
            feature_paths.extend(sorted(Path().glob(item)))
        else:
            feature_paths.append(Path(item))
    feature_paths = sorted({p.resolve() for p in feature_paths if p.exists()})
    if not feature_paths:
        raise FileNotFoundError("no feature NPZ files found")
    k_values = [int(v) for v in _parse_list(args.k_shots)]
    all_rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for path in feature_paths:
        for k in k_values:
            rows, payload = evaluate_one(path, args, k)
            all_rows.extend(rows)
            payloads.append(payload)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "schema": "target_old_kshot_adaptation_v1",
                "protocol_boundary": {
                    "stage": "Stage2-B target-old-only old-class adaptation diagnostic",
                    "unknown_query": "excluded",
                    "target_new": "excluded",
                    "deployment_claim": False,
                },
                "config": _jsonable(vars(args)),
                "rows": _jsonable(all_rows),
                "payloads": _jsonable(payloads),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    fieldnames = [
        "feature_npz",
        "domain",
        "candidate",
        "k_shot",
        "method",
        "old_acc",
        "query_n",
        "correct_n",
        "source_proto_per_tx",
        "max_support_per_tx",
        "query_per_tx",
    ]
    with args.summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"[OLD-KSHOT] files={len(feature_paths)} k={k_values} rows={len(all_rows)} output_json={args.output_json} summary_csv={args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
