#!/usr/bin/env python
"""Target-old-only upper-bound diagnostic for Stage2 feature packages.

This diagnostic uses only target receiver old-class rows. It splits
`target_old` rows into support/query per TX and reports old-class accuracy
under a frozen feature-space prototype classifier. It does not use unknown
rows for training, calibration, early stopping, or model selection.
"""

from __future__ import annotations

import argparse
import csv
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


def _ordered_target_old_indices(payload: dict[str, Any], target_old_tx_ids: Sequence[str]) -> dict[str, list[int]]:
    target_set = {_canonical(tx) for tx in target_old_tx_ids}
    by_tx: dict[str, list[tuple[tuple[str, str, str, int], int]]] = {tx: [] for tx in target_set}
    for i, role in enumerate(payload["dataset_role"]):
        tx = _canonical(payload["tx_ids"][i])
        if role != "target_old" or tx not in target_set:
            continue
        key = (
            str(payload["rx_ids"][i]),
            str(payload["day_ids"][i]),
            str(payload["sig_ids"][i]),
            int(i),
        )
        by_tx[tx].append((key, i))
    out: dict[str, list[int]] = {}
    for tx, rows in by_tx.items():
        out[tx] = [i for _, i in sorted(rows, key=lambda item: item[0])]
    return out


def _safe_rate(num: int, den: int) -> float | None:
    return None if int(den) <= 0 else float(num) / float(den)


def _evaluate_k(z: np.ndarray, by_tx: dict[str, list[int]], k: int) -> dict[str, Any]:
    support_indices: list[int] = []
    query_indices: list[int] = []
    prototypes: dict[str, np.ndarray] = {}
    for tx in sorted(by_tx):
        rows = list(by_tx[tx])
        support = rows[: max(0, int(k))]
        query = rows[max(0, int(k)) :]
        if support:
            prototypes[tx] = _normalize_rows(z[np.asarray(support, dtype=int)].mean(axis=0, keepdims=True))[0]
        support_indices.extend(support)
        query_indices.extend(query)
    if not prototypes:
        raise ValueError(f"k={k} leaves no support rows")

    labels = sorted(prototypes)
    proto_matrix = np.stack([prototypes[label] for label in labels], axis=0)
    per_tx_total = {tx: 0 for tx in labels}
    per_tx_correct = {tx: 0 for tx in labels}
    correct = 0
    rows: list[dict[str, Any]] = []
    for idx in query_indices:
        tx = next(label for label, indices in by_tx.items() if idx in set(indices))
        sims = z[idx].reshape(1, -1) @ proto_matrix.T
        pred = labels[int(np.argmax(sims))]
        is_correct = pred == tx
        correct += int(is_correct)
        per_tx_total[tx] += 1
        per_tx_correct[tx] += int(is_correct)
        rows.append({"row": int(idx), "tx_id": tx, "pred_tx_id": pred, "correct": int(is_correct)})

    per_tx_acc = {
        tx: _safe_rate(per_tx_correct.get(tx, 0), per_tx_total.get(tx, 0))
        for tx in labels
    }
    valid_accs = [v for v in per_tx_acc.values() if v is not None]
    overlap = len(set(support_indices) & set(query_indices))
    return {
        "k": int(k),
        "support_count": int(len(support_indices)),
        "query_count": int(len(query_indices)),
        "support_query_overlap_count": int(overlap),
        "old_acc": _safe_rate(correct, len(query_indices)),
        "min_old_class_acc": None if not valid_accs else float(min(valid_accs)),
        "per_tx_accuracy": per_tx_acc,
        "query_rows": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_npz(args.feature_npz)
    target_old_tx_ids = _parse_csv(args.target_old_tx_ids)
    if not target_old_tx_ids:
        manifest_ids = payload["manifest"].get("target_old_tx_ids", [])
        target_old_tx_ids = [_canonical(x) for x in manifest_ids]
    if not target_old_tx_ids:
        raise ValueError("--target_old_tx_ids or manifest target_old_tx_ids is required")
    z = _normalize_rows(np.asarray(payload["features"], dtype=np.float32))
    by_tx = _ordered_target_old_indices(payload, target_old_tx_ids)
    if not any(by_tx.values()):
        raise ValueError("no target_old rows are available for target-old-only diagnostic")
    k_values = [int(k) for k in _parse_csv(args.k_values)]
    if not k_values:
        raise ValueError("--k_values must contain at least one integer")

    ignored = sum(1 for role in payload["dataset_role"] if role != "target_old")
    results = []
    summary_rows = []
    detail_rows = []
    for k in k_values:
        item = _evaluate_k(z, by_tx, k)
        result = {key: value for key, value in item.items() if key != "query_rows"}
        result["ignored_non_target_old_rows"] = int(ignored)
        results.append(result)
        summary_rows.append(
            {
                "k": item["k"],
                "support_count": item["support_count"],
                "query_count": item["query_count"],
                "old_acc": "" if item["old_acc"] is None else f"{float(item['old_acc']):.8f}",
                "min_old_class_acc": "" if item["min_old_class_acc"] is None else f"{float(item['min_old_class_acc']):.8f}",
                "support_query_overlap_count": item["support_query_overlap_count"],
            }
        )
        for row in item["query_rows"]:
            detail_rows.append({"k": item["k"], **row})

    metrics = {
        "phase": "target_old_only_upper_bound_diagnostic",
        "verdict_scope": "non_deployment_target_old_only_diagnostic",
        "feature_npz": str(args.feature_npz),
        "target_old_tx_ids": target_old_tx_ids,
        "k_values": k_values,
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
    if args.detail_csv:
        Path(args.detail_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.detail_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(detail_rows[0].keys()) if detail_rows else ["k", "row"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detail_rows)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", required=True)
    p.add_argument("--target_old_tx_ids", default="")
    p.add_argument("--k_values", default="1,2,5,10,20,50")
    p.add_argument("--output_json", default="")
    p.add_argument("--summary_csv", default="")
    p.add_argument("--detail_csv", default="")
    return p.parse_args(argv)


def evaluate(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return run(parse_args(argv))


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(evaluate(argv), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
