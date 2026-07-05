#!/usr/bin/env python
"""Target-old-only ridge linear-probe upper-bound diagnostic.

This is a non-deployment diagnostic. It uses only `target_old` feature rows,
splits them into support/query per old TX, fits a closed-form ridge linear head
on support rows, and evaluates old target query accuracy. Rows from
target_new, target_unknown, and proxy_unknown are ignored for fitting,
calibration, thresholding, early stopping, and model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _canonical(value: object) -> str:
    text = str(value)
    if text.endswith(".0") and text[:-2].replace("-", "").isdigit():
        text = text[:-2]
    return text


def _as_str_array(value: np.ndarray, n: int, default: str = "") -> list[str]:
    arr = np.asarray(value)
    if arr.shape == ():
        return [_canonical(arr.item() or default)] * int(n)
    rows = arr.reshape(-1).tolist()
    if len(rows) < n:
        rows = rows + [default] * (n - len(rows))
    return [_canonical(v if v is not None else default) for v in rows[:n]]


def _parse_csv(text: str) -> list[str]:
    return [_canonical(x.strip()) for x in str(text or "").split(",") if x.strip()]


def _parse_float_csv(text: str) -> list[float]:
    return [float(x) for x in _parse_csv(text)]


def _load_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        if "features" not in data.files:
            raise ValueError(f"{path} does not contain features")
        features = np.asarray(data["features"], dtype=np.float32)
        n = int(features.shape[0])

        def pick(key: str, default: str = "") -> np.ndarray:
            return np.asarray(data[key]) if key in data.files else np.asarray([default] * n)

        manifest: dict[str, Any] = {}
        if "manifest_json" in data.files:
            try:
                manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
            except Exception:
                manifest = {}
        return {
            "features": features,
            "dataset_role": _as_str_array(pick("dataset_role"), n),
            "tx_ids": _as_str_array(pick("tx_ids"), n),
            "rx_ids": _as_str_array(pick("rx_ids"), n),
            "day_ids": _as_str_array(pick("day_ids"), n),
            "sig_ids": _as_str_array(pick("sig_ids"), n),
            "manifest": manifest,
        }


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1.0e-8)


def _target_old_indices(payload: dict[str, Any], target_old_tx_ids: Sequence[str]) -> dict[str, list[int]]:
    target_set = {_canonical(tx) for tx in target_old_tx_ids}
    by_tx: dict[str, list[tuple[tuple[str, str, str, int], int]]] = {tx: [] for tx in target_set}
    for i, role in enumerate(payload["dataset_role"]):
        tx = _canonical(payload["tx_ids"][i])
        if role != "target_old" or tx not in target_set:
            continue
        key = (str(payload["rx_ids"][i]), str(payload["day_ids"][i]), str(payload["sig_ids"][i]), int(i))
        by_tx[tx].append((key, i))
    return {tx: [i for _, i in sorted(rows, key=lambda item: item[0])] for tx, rows in by_tx.items()}


def _safe_rate(num: int, den: int) -> float | None:
    return None if int(den) <= 0 else float(num) / float(den)


def _index_hash(indices: Sequence[int]) -> str:
    text = ",".join(str(int(i)) for i in sorted(indices))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prepare_design(
    z: np.ndarray,
    support_idx: Sequence[int],
    query_idx: Sequence[int],
    *,
    standardize: bool,
    l2_normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(z, dtype=np.float32)
    if l2_normalize:
        x = _normalize_rows(x)
    x_support = x[np.asarray(support_idx, dtype=int)]
    x_query = x[np.asarray(query_idx, dtype=int)]
    if standardize:
        mu = x_support.mean(axis=0, keepdims=True)
        sigma = x_support.std(axis=0, keepdims=True)
        x_support = (x_support - mu) / np.maximum(sigma, 1.0e-6)
        x_query = (x_query - mu) / np.maximum(sigma, 1.0e-6)
    x_support = np.concatenate([x_support, np.ones((x_support.shape[0], 1), dtype=np.float32)], axis=1)
    x_query = np.concatenate([x_query, np.ones((x_query.shape[0], 1), dtype=np.float32)], axis=1)
    return x_support, x_query


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    xtx = x.T @ x
    reg = float(ridge_lambda) * np.eye(xtx.shape[0], dtype=np.float32)
    reg[-1, -1] = 0.0
    rhs = x.T @ y
    try:
        return np.linalg.solve(xtx + reg, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx + reg) @ rhs


def _evaluate_one(
    z: np.ndarray,
    by_tx: dict[str, list[int]],
    *,
    k: int,
    ridge_lambda: float,
    standardize: bool,
    l2_normalize: bool,
) -> dict[str, Any]:
    labels = sorted(tx for tx, rows in by_tx.items() if rows)
    support_idx: list[int] = []
    query_idx: list[int] = []
    query_labels: list[str] = []
    for tx in labels:
        rows = list(by_tx[tx])
        support = rows[: max(0, int(k))]
        query = rows[max(0, int(k)) :]
        support_idx.extend(support)
        query_idx.extend(query)
        query_labels.extend([tx] * len(query))
    if not support_idx:
        raise ValueError(f"k={k} leaves no support rows")
    if not query_idx:
        raise ValueError(f"k={k} leaves no query rows")
    x_support, x_query = _prepare_design(
        z,
        support_idx,
        query_idx,
        standardize=standardize,
        l2_normalize=l2_normalize,
    )
    label_to_col = {label: j for j, label in enumerate(labels)}
    y = np.full((len(support_idx), len(labels)), -1.0, dtype=np.float32)
    for row_idx, sample_idx in enumerate(support_idx):
        tx = next(label for label, rows in by_tx.items() if sample_idx in set(rows))
        y[row_idx, label_to_col[tx]] = 1.0
    w = _fit_ridge(x_support, y, float(ridge_lambda))
    logits = x_query @ w
    pred_pos = np.argmax(logits, axis=1)
    pred_labels = [labels[int(i)] for i in pred_pos]
    correct = [int(p == y_true) for p, y_true in zip(pred_labels, query_labels)]
    per_tx_total = {tx: 0 for tx in labels}
    per_tx_correct = {tx: 0 for tx in labels}
    confusion = {tx: {pred: 0 for pred in labels} for tx in labels}
    for y_true, ok in zip(query_labels, correct):
        per_tx_total[y_true] += 1
        per_tx_correct[y_true] += int(ok)
    for y_true, pred in zip(query_labels, pred_labels):
        confusion[y_true][pred] += 1
    per_tx_acc = {
        tx: _safe_rate(per_tx_correct.get(tx, 0), per_tx_total.get(tx, 0))
        for tx in labels
    }
    valid_accs = [v for v in per_tx_acc.values() if v is not None]
    invalid_classes = [tx for tx in labels if per_tx_total.get(tx, 0) <= 0]
    return {
        "k": int(k),
        "ridge_lambda": float(ridge_lambda),
        "support_count": int(len(support_idx)),
        "query_count": int(len(query_idx)),
        "support_query_overlap_count": int(len(set(support_idx) & set(query_idx))),
        "support_index_sha256": _index_hash(support_idx),
        "query_index_sha256": _index_hash(query_idx),
        "old_acc": _safe_rate(sum(correct), len(correct)),
        "macro_old_acc": None if not valid_accs else float(sum(valid_accs) / len(valid_accs)),
        "min_old_class_acc": None if not valid_accs else float(min(valid_accs)),
        "invalid_classes": invalid_classes,
        "per_tx_query_count": per_tx_total,
        "per_tx_accuracy": per_tx_acc,
        "confusion": confusion,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_npz(args.feature_npz)
    target_old_tx_ids = _parse_csv(args.target_old_tx_ids)
    if not target_old_tx_ids:
        target_old_tx_ids = [_canonical(x) for x in payload["manifest"].get("target_old_tx_ids", [])]
    if not target_old_tx_ids:
        raise ValueError("--target_old_tx_ids or manifest target_old_tx_ids is required")
    by_tx = _target_old_indices(payload, target_old_tx_ids)
    if not any(by_tx.values()):
        raise ValueError("no target_old rows are available for target-old-only linear probe")
    k_values = [int(k) for k in _parse_csv(args.k_values)]
    ridge_lambdas = _parse_float_csv(args.ridge_lambdas)
    if not k_values or not ridge_lambdas:
        raise ValueError("--k_values and --ridge_lambdas must be non-empty")
    ignored = sum(1 for role in payload["dataset_role"] if role != "target_old")
    z = np.asarray(payload["features"], dtype=np.float32)
    results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for k in k_values:
        for lam in ridge_lambdas:
            item = _evaluate_one(
                z,
                by_tx,
                k=int(k),
                ridge_lambda=float(lam),
                standardize=not bool(args.no_standardize),
                l2_normalize=not bool(args.no_l2_normalize),
            )
            item["ignored_non_target_old_rows"] = int(ignored)
            results.append(item)
            summary_rows.append(
                {
                    "k": item["k"],
                    "ridge_lambda": f"{item['ridge_lambda']:.8g}",
                    "support_count": item["support_count"],
                    "query_count": item["query_count"],
                    "old_acc": "" if item["old_acc"] is None else f"{float(item['old_acc']):.8f}",
                    "macro_old_acc": "" if item["macro_old_acc"] is None else f"{float(item['macro_old_acc']):.8f}",
                    "min_old_class_acc": "" if item["min_old_class_acc"] is None else f"{float(item['min_old_class_acc']):.8f}",
                    "support_query_overlap_count": item["support_query_overlap_count"],
                    "invalid_classes": ",".join(item["invalid_classes"]),
                }
            )
    metrics = {
        "phase": "target_old_only_linear_probe_upper_bound_diagnostic",
        "verdict_scope": "non_deployment_target_old_only_diagnostic",
        "feature_npz": str(args.feature_npz),
        "target_old_tx_ids": target_old_tx_ids,
        "k_values": k_values,
        "ridge_lambdas": ridge_lambdas,
        "standardize_support_only": not bool(args.no_standardize),
        "l2_normalize": not bool(args.no_l2_normalize),
        "lambda_selection_scope": "reported_grid_diagnostic_not_deployment_selection",
        "target_unknown_training_count": 0,
        "target_unknown_calibration_count": 0,
        "uses_unknown_query_for_threshold": False,
        "uses_unknown_for_model_selection": False,
        "support_query_source": "target_old_rows_only_deterministic_split",
        "results": results,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.summary_csv:
        Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", required=True)
    p.add_argument("--target_old_tx_ids", default="")
    p.add_argument("--k_values", default="1,2,5,10,20,50")
    p.add_argument("--ridge_lambdas", default="0.001,0.01,0.1,1.0,10.0")
    p.add_argument("--no_standardize", action="store_true")
    p.add_argument("--no_l2_normalize", action="store_true")
    p.add_argument("--output_json", default="")
    p.add_argument("--summary_csv", default="")
    return p.parse_args(argv)


def evaluate(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return run(parse_args(argv))


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(evaluate(argv), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
