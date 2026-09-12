#!/usr/bin/env python
"""Target-old-only feature-level fine-tuning diagnostic.

This script is diagnostic-only: it ignores target-new and unknown samples and
measures how much labeled target-domain old-class support can improve
target-old query accuracy on saved feature NPZ files.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def parse_str_list(text: str | None) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1.0e-12)


def class_order(labels: np.ndarray) -> list[str]:
    return sorted({str(v) for v in labels.tolist()})


def label_indices(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    return np.asarray([idx[str(v)] for v in labels.tolist()], dtype=np.int64)


def split_target_old(labels: np.ndarray, train_per_tx: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    train: list[int] = []
    query: list[int] = []
    for label in class_order(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 2:
            continue
        perm = idx[rng.permutation(len(idx))]
        k = min(int(train_per_tx), len(perm) - 1)
        train.extend(perm[:k].tolist())
        query.extend(perm[k:].tolist())
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(query), dtype=np.int64)


def nearest_centroid_predict(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, classes: list[str]) -> np.ndarray:
    train_x = l2_normalize(train_x)
    query_x = l2_normalize(query_x)
    centroids: list[np.ndarray] = []
    for label in classes:
        mask = train_y == label
        if not np.any(mask):
            centroids.append(np.zeros(train_x.shape[1], dtype=np.float64))
            continue
        c = train_x[mask].mean(axis=0)
        c = c / max(float(np.linalg.norm(c)), 1.0e-12)
        centroids.append(c)
    proto = np.vstack(centroids)
    return np.argmax(query_x @ proto.T, axis=1)


def ridge_predict(train_x: np.ndarray, train_y_idx: np.ndarray, query_x: np.ndarray, num_classes: int, ridge: float) -> np.ndarray:
    train_x = l2_normalize(train_x)
    query_x = l2_normalize(query_x)
    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = train_x.std(axis=0, keepdims=True)
    x_std = np.maximum(x_std, 1.0e-6)
    train_x = (train_x - x_mean) / x_std
    query_x = (query_x - x_mean) / x_std
    train_aug = np.concatenate([train_x, np.ones((train_x.shape[0], 1), dtype=np.float64)], axis=1)
    query_aug = np.concatenate([query_x, np.ones((query_x.shape[0], 1), dtype=np.float64)], axis=1)
    y = np.eye(num_classes, dtype=np.float64)[train_y_idx]
    reg = float(ridge) * np.eye(train_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(train_aug.T @ train_aug + reg, train_aug.T @ y)
    return np.argmax(query_aug @ weights, axis=1)


def accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    if len(truth) == 0:
        return float("nan")
    return float(np.mean(pred == truth))


def macro_accuracy(pred: np.ndarray, truth: np.ndarray, num_classes: int) -> float:
    vals: list[float] = []
    for i in range(num_classes):
        mask = truth == i
        if np.any(mask):
            vals.append(float(np.mean(pred[mask] == truth[mask])))
    return float(np.mean(vals)) if vals else float("nan")


@dataclass
class FeaturePayload:
    candidate_id: str
    features: np.ndarray
    tx_ids: np.ndarray
    roles: np.ndarray
    rx_ids: np.ndarray | None = None
    channel_views: np.ndarray | None = None
    sat_scenarios: np.ndarray | None = None


def optional_str_array(npz: np.lib.npyio.NpzFile, key: str, expected_len: int) -> np.ndarray | None:
    if key not in npz:
        return None
    arr = np.asarray(npz[key]).astype(str).reshape(-1)
    if arr.shape[0] != expected_len:
        raise ValueError(f"{key} length {arr.shape[0]} does not match features length {expected_len}")
    return arr


def load_payload(path: Path) -> FeaturePayload:
    z = np.load(path, allow_pickle=True)
    features = np.asarray(z["features"], dtype=np.float64)
    return FeaturePayload(
        candidate_id=path.parent.name,
        features=features,
        tx_ids=np.asarray(z["tx_ids"]).astype(str),
        roles=np.asarray(z["dataset_role"]).astype(str),
        rx_ids=optional_str_array(z, "rx_ids", features.shape[0]),
        channel_views=optional_str_array(z, "channel_views", features.shape[0]),
        sat_scenarios=optional_str_array(z, "sat_scenarios", features.shape[0]),
    )


def iter_feature_paths(input_run_root: Path) -> Iterable[Path]:
    yield from sorted(Path(input_run_root).glob("*/features.npz"))


def evaluate_payload(
    payload: FeaturePayload,
    *,
    train_grid: list[int],
    seeds: list[int],
    ridge_values: list[float],
    expected_target_receivers: set[str] | None = None,
    require_target_channel_view: str | None = None,
    allowed_target_sat_scenarios: set[str] | None = None,
) -> list[dict[str, object]]:
    source_mask = payload.roles == "source"
    target_mask = payload.roles == "target_old"
    if not np.any(target_mask):
        return []
    if expected_target_receivers:
        if payload.rx_ids is None:
            raise ValueError(f"{payload.candidate_id}: rx_ids missing; cannot verify target receiver")
        observed = {str(v) for v in payload.rx_ids[target_mask].tolist()}
        unexpected = observed - set(expected_target_receivers)
        if unexpected:
            raise ValueError(
                f"{payload.candidate_id}: target_old receivers {sorted(observed)} do not match "
                f"expected {sorted(expected_target_receivers)}"
            )
    if require_target_channel_view:
        if payload.channel_views is None:
            raise ValueError(f"{payload.candidate_id}: channel_views missing; cannot verify target view")
        observed_views = {str(v) for v in payload.channel_views[target_mask].tolist()}
        if observed_views != {require_target_channel_view}:
            raise ValueError(
                f"{payload.candidate_id}: target_old channel views {sorted(observed_views)} do not match "
                f"required {require_target_channel_view!r}"
            )
    if allowed_target_sat_scenarios:
        if payload.sat_scenarios is None:
            raise ValueError(f"{payload.candidate_id}: sat_scenarios missing; cannot verify target scenarios")
        observed_scenarios = {str(v) for v in payload.sat_scenarios[target_mask].tolist()}
        unexpected_scenarios = observed_scenarios - set(allowed_target_sat_scenarios)
        if unexpected_scenarios:
            raise ValueError(
                f"{payload.candidate_id}: target_old sat scenarios {sorted(observed_scenarios)} include "
                f"unexpected {sorted(unexpected_scenarios)}"
            )
    target_x_all = payload.features[target_mask]
    target_y_all = payload.tx_ids[target_mask]
    classes = class_order(target_y_all)
    rows: list[dict[str, object]] = []
    source_x = payload.features[source_mask]
    source_y = payload.tx_ids[source_mask]
    for train_per_tx in train_grid:
        for seed in seeds:
            support_idx, query_idx = split_target_old(target_y_all, int(train_per_tx), int(seed))
            if len(support_idx) == 0 or len(query_idx) == 0:
                continue
            support_x = target_x_all[support_idx]
            support_y = target_y_all[support_idx]
            query_x = target_x_all[query_idx]
            query_y = target_y_all[query_idx]
            truth = label_indices(query_y, classes)
            base = {
                "candidate_id": payload.candidate_id,
                "train_per_tx": int(train_per_tx),
                "seed": int(seed),
                "class_count": len(classes),
                "support_count": int(len(support_idx)),
                "query_count": int(len(query_idx)),
            }
            if np.any(source_mask):
                # Source prototypes are a frozen comparison baseline only; target-only fit uses support_x/support_y.
                pred = nearest_centroid_predict(source_x, source_y, query_x, classes)
                rows.append(
                    {
                        **base,
                        "method": "source_proto",
                        "target_old_accuracy": accuracy(pred, truth),
                        "target_old_macro_accuracy": macro_accuracy(pred, truth, len(classes)),
                    }
                )
            pred = nearest_centroid_predict(support_x, support_y, query_x, classes)
            rows.append(
                {
                    **base,
                    "method": "target_proto",
                    "target_old_accuracy": accuracy(pred, truth),
                    "target_old_macro_accuracy": macro_accuracy(pred, truth, len(classes)),
                }
            )
            support_y_idx = label_indices(support_y, classes)
            for ridge in ridge_values:
                pred = ridge_predict(support_x, support_y_idx, query_x, len(classes), float(ridge))
                rows.append(
                    {
                        **base,
                        "method": f"target_ridge_{ridge:g}",
                        "target_old_accuracy": accuracy(pred, truth),
                        "target_old_macro_accuracy": macro_accuracy(pred, truth, len(classes)),
                    }
                )
    baseline: dict[tuple[str, int, int], float] = {}
    for row in rows:
        if row["method"] == "source_proto":
            baseline[(str(row["candidate_id"]), int(row["train_per_tx"]), int(row["seed"]))] = float(
                row["target_old_accuracy"]
            )
    for row in rows:
        key = (str(row["candidate_id"]), int(row["train_per_tx"]), int(row["seed"]))
        base_acc = baseline.get(key)
        row["source_proto_delta"] = "" if base_acc is None else float(row["target_old_accuracy"]) - base_acc
    return rows


def summarize(rows: list[dict[str, object]], metadata: dict[str, object] | None = None) -> dict[str, object]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), int(row["train_per_tx"])), []).append(row)
    by_method_k = []
    for (method, train_per_tx), vals in sorted(groups.items()):
        acc = np.asarray([float(v["target_old_accuracy"]) for v in vals], dtype=np.float64)
        deltas = [
            float(v["source_proto_delta"])
            for v in vals
            if v.get("source_proto_delta") not in ("", None)
        ]
        by_method_k.append(
            {
                "method": method,
                "train_per_tx": train_per_tx,
                "row_count": len(vals),
                "target_old_accuracy_mean": float(np.mean(acc)),
                "target_old_accuracy_max": float(np.max(acc)),
                "target_old_accuracy_min": float(np.min(acc)),
                "source_proto_delta_mean": float(np.mean(deltas)) if deltas else None,
                "source_proto_delta_max": float(np.max(deltas)) if deltas else None,
            }
        )
    best = sorted(rows, key=lambda r: float(r["target_old_accuracy"]), reverse=True)[:20]
    best_group = sorted(by_method_k, key=lambda r: float(r["target_old_accuracy_mean"]), reverse=True)[:10]
    summary: dict[str, object] = {
        "schema": "target_old_only_finetune_diag_summary_v1",
        "diagnostic_type": "TARGET_OLD_ONLY_UPPER_BOUND_DIAGNOSTIC",
        "status": "COMPLETED_DIAGNOSTIC_TARGET_OLD_ONLY",
        "row_count": len(rows),
        "candidate_count": len({str(r["candidate_id"]) for r in rows}),
        "by_method_k": by_method_k,
        "best_rows": best,
        "best_method_k": best_group,
        "claim_boundary": (
            "TARGET_OLD_ONLY_UPPER_BOUND_DIAGNOSTIC; target_new and unknown samples ignored; "
            "no FAR, seen-new, or deployment-success claim"
        ),
    }
    if metadata:
        summary.update(metadata)
    return summary


def write_outputs(rows: list[dict[str, object]], output_dir: Path, metadata: dict[str, object] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "method",
        "train_per_tx",
        "seed",
        "class_count",
        "support_count",
        "query_count",
        "target_old_accuracy",
        "target_old_macro_accuracy",
        "source_proto_delta",
    ]
    with (output_dir / "target_old_only_result_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    (output_dir / "target_old_only_summary.json").write_text(
        json.dumps(summarize(rows, metadata=metadata), indent=2), encoding="utf-8"
    )


def run_synthetic_smoke(output_dir: Path) -> None:
    rng = np.random.default_rng(7)
    classes = ["a", "b", "c"]
    features = []
    labels = []
    roles = []
    for role, shift in [("source", 0.0), ("target_old", 0.35)]:
        for i, label in enumerate(classes):
            center = np.zeros(12)
            center[i] = 1.0 + shift
            x = center + rng.normal(0.0, 0.08, size=(12, 12))
            features.append(x)
            labels.extend([label] * len(x))
            roles.extend([role] * len(x))
    payload = FeaturePayload(
        candidate_id="synthetic",
        features=np.vstack(features),
        tx_ids=np.asarray(labels),
        roles=np.asarray(roles),
        rx_ids=np.asarray(["source-rx" if role == "source" else "20-1" for role in roles]),
        channel_views=np.asarray(["clean" if role == "source" else "satellite" for role in roles]),
        sat_scenarios=np.asarray(["" if role == "source" else "leo_clear_weak" for role in roles]),
    )
    rows = evaluate_payload(
        payload,
        train_grid=[2, 4],
        seeds=[11],
        ridge_values=[0.01, 1.0],
        expected_target_receivers={"20-1"},
        require_target_channel_view="satellite",
    )
    write_outputs(
        rows,
        output_dir,
        metadata={
            "run_id": output_dir.name,
            "input_run_root": "synthetic_smoke",
            "train_per_tx_grid": [2, 4],
            "seeds": [11],
            "ridge_values": [0.01, 1.0],
            "expected_target_receivers": ["20-1"],
            "require_target_channel_view": "satellite",
            "allowed_target_sat_scenarios": [],
            "source_baseline_role": "source_proto is a frozen comparison baseline only; target-only methods fit only target_old support.",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-tx-grid", type=parse_int_list, default=[5, 10, 20, 40, 50])
    parser.add_argument("--seeds", type=parse_int_list, default=[213920])
    parser.add_argument("--ridge-values", type=lambda s: [float(x) for x in s.split(",") if x.strip()], default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--expected-target-receivers", type=parse_str_list, default=[])
    parser.add_argument("--require-target-channel-view", default="")
    parser.add_argument("--allowed-target-sat-scenarios", type=parse_str_list, default=[])
    parser.add_argument("--smoke-synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_synthetic:
        run_synthetic_smoke(args.output_dir)
        return
    if args.input_run_root is None:
        raise SystemExit("--input-run-root is required unless --smoke-synthetic is used")
    rows: list[dict[str, object]] = []
    for path in iter_feature_paths(args.input_run_root):
        rows.extend(
            evaluate_payload(
                load_payload(path),
                train_grid=list(args.train_per_tx_grid),
                seeds=list(args.seeds),
                ridge_values=list(args.ridge_values),
                expected_target_receivers=set(args.expected_target_receivers) or None,
                require_target_channel_view=str(args.require_target_channel_view or "").strip() or None,
                allowed_target_sat_scenarios=set(args.allowed_target_sat_scenarios) or None,
            )
        )
    if not rows:
        raise SystemExit(f"no target_old diagnostic rows produced from {args.input_run_root}")
    metadata = {
        "run_id": args.output_dir.name,
        "input_run_root": str(args.input_run_root),
        "train_per_tx_grid": list(args.train_per_tx_grid),
        "seeds": list(args.seeds),
        "ridge_values": list(args.ridge_values),
        "expected_target_receivers": list(args.expected_target_receivers),
        "require_target_channel_view": str(args.require_target_channel_view or "").strip(),
        "allowed_target_sat_scenarios": list(args.allowed_target_sat_scenarios),
        "source_baseline_role": "source_proto is a frozen comparison baseline only; target-only methods fit only target_old support.",
    }
    write_outputs(rows, args.output_dir, metadata=metadata)
    summary = summarize(rows, metadata=metadata)
    print(json.dumps({"rows": summary["row_count"], "candidate_count": summary["candidate_count"], "best_method_k": summary["best_method_k"][:3]}, indent=2))


if __name__ == "__main__":
    main()
