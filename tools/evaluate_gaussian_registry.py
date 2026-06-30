#!/usr/bin/env python
"""Evaluate a Gaussian prototype registry for Stage2-B old/unknown diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cvsrffi.wisig_fewshot_payload import UNKNOWN_LABEL, build_sfe_payload_from_feature_arrays, parse_tx_id_list


EPS = 1.0e-6


@dataclass(frozen=True)
class GaussianEntry:
    label: int
    mode: str
    mean: np.ndarray
    inv_var: np.ndarray
    threshold: float
    density_threshold: float
    sample_count: int


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (Path, os.PathLike)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    return value


def _load_embedded_manifest(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
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


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, EPS)


def _diag_var(x: np.ndarray, global_var: np.ndarray, shrinkage: float) -> np.ndarray:
    if x.shape[0] <= 1:
        local = global_var.copy()
    else:
        local = np.var(x, axis=0, ddof=1)
    var = (1.0 - shrinkage) * local + shrinkage * global_var
    return np.maximum(var, EPS)


def _mahalanobis_diag(x: np.ndarray, mean: np.ndarray, inv_var: np.ndarray) -> np.ndarray:
    diff = x - mean.reshape(1, -1)
    return np.sum(diff * diff * inv_var.reshape(1, -1), axis=1)


def _cosine_density(samples: np.ndarray, anchors: np.ndarray, topk: int) -> np.ndarray:
    if samples.size == 0 or anchors.size == 0:
        return np.full((samples.shape[0],), -1.0, dtype=np.float64)
    sims = np.matmul(samples, anchors.T)
    k = max(1, min(int(topk), sims.shape[1]))
    part = np.partition(sims, kth=sims.shape[1] - k, axis=1)[:, -k:]
    return np.mean(part, axis=1)


def _quantile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(np.quantile(values, min(max(float(q), 0.0), 1.0)))


def _labels() -> dict[str, str]:
    return {
        "src_single": "source-only single Gaussian",
        "target_single": "source+target-old support single Gaussian",
        "target_modes": "source mode + target support mode per old class",
        "target_modes_density": "target modes + local density gate",
        "target_modes_density_margin": "target modes + local density + margin gate",
    }


def _build_entries(
    *,
    source_features: np.ndarray,
    source_labels: np.ndarray,
    support_features: np.ndarray,
    support_labels: np.ndarray,
    variant: str,
    radius_quantile: float,
    radius_slack: float,
    density_quantile: float,
    density_slack: float,
    shrinkage: float,
    support_weight: float,
    density_topk: int,
) -> tuple[list[GaussianEntry], dict[str, Any]]:
    all_train = np.concatenate([source_features, support_features], axis=0) if support_features.size else source_features
    global_var = np.var(all_train, axis=0, ddof=1) if all_train.shape[0] > 1 else np.ones((all_train.shape[1],), dtype=np.float64)
    global_var = np.maximum(global_var, EPS)
    labels = sorted(int(v) for v in np.unique(source_labels))
    entries: list[GaussianEntry] = []
    registry_summary: dict[str, Any] = {}

    for label in labels:
        src = source_features[source_labels == label]
        sup = support_features[support_labels == label] if support_features.size else np.empty((0, source_features.shape[1]))
        modes: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        if variant == "src_single" or sup.size == 0:
            modes.append(("source", src, src, src))
        elif variant == "target_single":
            src_repeat = max(1, int(round(float(support_weight))))
            combined = np.concatenate([src, np.repeat(sup, repeats=src_repeat, axis=0)], axis=0)
            calib = np.concatenate([src, sup], axis=0)
            modes.append(("target_registered", combined, calib, calib))
        else:
            modes.append(("source", src, src, src))
            support_var_samples = np.concatenate([src, sup], axis=0) if src.size else sup
            modes.append(("target_support", sup, sup, support_var_samples))

        registry_summary[str(label)] = []
        for mode_name, mean_samples, calib_samples, var_samples in modes:
            mean = np.mean(mean_samples, axis=0)
            inv_var = 1.0 / _diag_var(var_samples, global_var, shrinkage)
            dist = _mahalanobis_diag(calib_samples, mean, inv_var)
            threshold = _quantile(dist, radius_quantile, default=float(np.max(dist) if dist.size else 0.0)) + float(radius_slack)
            density_values = _cosine_density(calib_samples, calib_samples, density_topk)
            density_threshold = _quantile(density_values, density_quantile, default=-1.0) - float(density_slack)
            entries.append(
                GaussianEntry(
                    label=label,
                    mode=mode_name,
                    mean=mean,
                    inv_var=inv_var,
                    threshold=threshold,
                    density_threshold=density_threshold,
                    sample_count=int(calib_samples.shape[0]),
                )
            )
            registry_summary[str(label)].append(
                {
                    "mode": mode_name,
                    "samples": int(calib_samples.shape[0]),
                    "threshold": float(threshold),
                    "density_threshold": float(density_threshold),
                }
            )
    return entries, registry_summary


def _predict(
    *,
    entries: list[GaussianEntry],
    query_features: np.ndarray,
    support_features: np.ndarray,
    support_labels: np.ndarray,
    source_features: np.ndarray,
    source_labels: np.ndarray,
    variant: str,
    margin_quantile: float,
    margin_slack: float,
    density_topk: int,
) -> dict[str, np.ndarray]:
    dists = np.stack([_mahalanobis_diag(query_features, e.mean, e.inv_var) for e in entries], axis=1)
    scores = -dists
    order = np.argsort(dists, axis=1)
    best_idx = order[:, 0]
    second_idx = order[:, 1] if len(entries) > 1 else order[:, 0]
    best_entries = [entries[int(i)] for i in best_idx]
    pred_labels = np.asarray([e.label for e in best_entries], dtype=np.int64)
    best_dist = dists[np.arange(query_features.shape[0]), best_idx]
    second_score = scores[np.arange(query_features.shape[0]), second_idx]
    best_score = scores[np.arange(query_features.shape[0]), best_idx]
    score_margin = best_score - second_score
    radius_pass = np.asarray([best_dist[i] <= best_entries[i].threshold for i in range(query_features.shape[0])], dtype=bool)

    density_pass = np.ones((query_features.shape[0],), dtype=bool)
    density_values = np.full((query_features.shape[0],), np.nan, dtype=np.float64)
    if "density" in variant:
        for label in sorted(set(int(e.label) for e in entries)):
            pred_mask = pred_labels == label
            anchors = []
            if source_features.size:
                anchors.append(source_features[source_labels == label])
            if support_features.size:
                anchors.append(support_features[support_labels == label])
            anchor = np.concatenate([a for a in anchors if a.size], axis=0)
            vals = _cosine_density(query_features[pred_mask], anchor, density_topk)
            density_values[pred_mask] = vals
            thresholds = [e.density_threshold for e in entries if int(e.label) == int(label)]
            threshold = min(thresholds) if thresholds else -1.0
            density_pass[pred_mask] = vals >= threshold

    margin_pass = np.ones((query_features.shape[0],), dtype=bool)
    margin_threshold = None
    if "margin" in variant:
        train_features = np.concatenate([source_features, support_features], axis=0) if support_features.size else source_features
        train_labels = np.concatenate([source_labels, support_labels], axis=0) if support_features.size else source_labels
        train_dists = np.stack([_mahalanobis_diag(train_features, e.mean, e.inv_var) for e in entries], axis=1)
        train_scores = -train_dists
        train_order = np.argsort(train_dists, axis=1)
        train_best = train_order[:, 0]
        train_second = train_order[:, 1] if len(entries) > 1 else train_order[:, 0]
        train_pred = np.asarray([entries[int(i)].label for i in train_best], dtype=np.int64)
        correct_margin = train_scores[np.arange(train_features.shape[0]), train_best] - train_scores[
            np.arange(train_features.shape[0]), train_second
        ]
        correct_margin = correct_margin[train_pred == train_labels]
        margin_threshold = _quantile(correct_margin, margin_quantile, default=-1.0e9) - float(margin_slack)
        margin_pass = score_margin >= margin_threshold

    accepted = radius_pass & density_pass & margin_pass
    return {
        "pred_labels": pred_labels,
        "accepted": accepted,
        "best_dist": best_dist,
        "score_margin": score_margin,
        "radius_pass": radius_pass,
        "density_pass": density_pass,
        "margin_pass": margin_pass,
        "density_values": density_values,
        "margin_threshold": np.asarray([np.nan if margin_threshold is None else float(margin_threshold)]),
    }


def _metrics(query_labels: np.ndarray, pred_labels: np.ndarray, accepted: np.ndarray, closed_pred: np.ndarray) -> dict[str, float | int]:
    old_mask = query_labels != UNKNOWN_LABEL
    unknown_mask = query_labels == UNKNOWN_LABEL
    old_n = int(old_mask.sum())
    unknown_n = int(unknown_mask.sum())
    accepted_old = accepted & old_mask
    accepted_unknown = accepted & unknown_mask
    correct_accepted_old = accepted_old & (pred_labels == query_labels)
    closed_old_correct = (closed_pred[old_mask] == query_labels[old_mask]) if old_n else np.asarray([], dtype=bool)
    total_n = int(query_labels.shape[0])
    accepted_n = int(accepted.sum())
    unknown_rejected_n = int(unknown_n - accepted_unknown.sum())
    full_correct = int(correct_accepted_old.sum()) + unknown_rejected_n
    return {
        "old_acc": float(correct_accepted_old.sum() / old_n) if old_n else math.nan,
        "old_coverage": float(accepted_old.sum() / old_n) if old_n else math.nan,
        "unknown_far": float(accepted_unknown.sum() / unknown_n) if unknown_n else math.nan,
        "unknown_reject": float(1.0 - accepted_unknown.sum() / unknown_n) if unknown_n else math.nan,
        "coverage": float(accepted_n / total_n) if total_n else math.nan,
        "full_acc": float(full_correct / total_n) if total_n else math.nan,
        "accepted_old_acc": float(correct_accepted_old.sum() / max(int(accepted_old.sum()), 1)) if old_n else math.nan,
        "no_reject_old_acc": float(closed_old_correct.sum() / old_n) if old_n else math.nan,
        "old_n": old_n,
        "unknown_n": unknown_n,
        "accepted_n": accepted_n,
        "accepted_unknown_n": int(accepted_unknown.sum()),
        "rejected_unknown_n": unknown_rejected_n,
        "accepted_old_correct_n": int(correct_accepted_old.sum()),
    }


def build_payload(path: Path, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        manifest = _load_embedded_manifest(data)
        direct_required = ["source_features", "source_labels", "support_features", "support_labels", "query_features", "query_labels"]
        if all(name in data for name in direct_required):
            arrays = {name: np.asarray(data[name]) for name in direct_required}
            query_roles = np.asarray(data["query_roles"]).astype(str) if "query_roles" in data else np.asarray(["query"] * arrays["query_labels"].shape[0])
            arrays["query_roles"] = query_roles
            return arrays, manifest
        features_key = str(args.features_key)
        tx_ids_key = str(args.tx_ids_key)
        if features_key not in data or tx_ids_key not in data:
            raise KeyError(f"feature npz missing direct arrays and {features_key!r}/{tx_ids_key!r}; available={list(data.keys())}")
        source_ids = parse_tx_id_list(args.source_tx_ids or manifest.get("source_tx_ids", []))
        target_old_ids = parse_tx_id_list(args.target_old_tx_ids or manifest.get("target_old_tx_ids", source_ids))
        unknown_ids = parse_tx_id_list(args.unknown_tx_ids or manifest.get("unknown_tx_ids", []))
        if not source_ids or not target_old_ids or not unknown_ids:
            raise ValueError("source_tx_ids, target_old_tx_ids and unknown_tx_ids are required for full-feature NPZ mode")
        payload = build_sfe_payload_from_feature_arrays(
            features=data[features_key],
            tx_ids=data[tx_ids_key],
            dataset_roles=data["dataset_role"] if "dataset_role" in data else None,
            sample_metadata={key: data[key] for key in ["receiver", "rx", "day", "channel_view", "scenario"] if key in data},
            source_tx_ids=source_ids,
            target_old_tx_ids=target_old_ids,
            new_tx_ids=[],
            unknown_tx_ids=unknown_ids,
            shots=0,
            source_proto_per_tx=int(args.source_proto_per_tx),
            source_query_per_tx=int(args.source_query_per_tx),
            target_old_support_per_tx=int(args.target_old_support_per_tx),
            target_old_query_per_tx=int(args.target_old_query_per_tx),
            query_per_tx=int(args.unknown_query_per_tx),
            seed=int(args.seed),
            extra_metadata={"payload_source": str(path), "analysis_tool": "evaluate_gaussian_registry"},
        )
        arrays = {key: np.asarray(value) for key, value in payload.arrays.items()}
        payload_manifest = dict(manifest)
        payload_manifest.update(payload.manifest)
        return arrays, payload_manifest


def evaluate_file(path: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arrays, manifest = build_payload(path, args)
    source_features = _l2_normalize(np.asarray(arrays["source_features"], dtype=np.float64))
    support_features = _l2_normalize(np.asarray(arrays["support_features"], dtype=np.float64))
    query_features = _l2_normalize(np.asarray(arrays["query_features"], dtype=np.float64))
    source_labels = np.asarray(arrays["source_labels"], dtype=np.int64).reshape(-1)
    support_labels = np.asarray(arrays["support_labels"], dtype=np.int64).reshape(-1)
    query_labels = np.asarray(arrays["query_labels"], dtype=np.int64).reshape(-1)
    query_roles = np.asarray(arrays.get("query_roles", ["query"] * query_labels.shape[0])).astype(str)
    rows: list[dict[str, Any]] = []
    registry: dict[str, Any] = {}
    for variant in str(args.variants).split(","):
        variant = variant.strip()
        if not variant:
            continue
        entries, registry_summary = _build_entries(
            source_features=source_features,
            source_labels=source_labels,
            support_features=support_features,
            support_labels=support_labels,
            variant=variant,
            radius_quantile=float(args.radius_quantile),
            radius_slack=float(args.radius_slack),
            density_quantile=float(args.density_quantile),
            density_slack=float(args.density_slack),
            shrinkage=float(args.shrinkage),
            support_weight=float(args.support_weight),
            density_topk=int(args.density_topk),
        )
        pred = _predict(
            entries=entries,
            query_features=query_features,
            support_features=support_features,
            support_labels=support_labels,
            source_features=source_features,
            source_labels=source_labels,
            variant=variant,
            margin_quantile=float(args.margin_quantile),
            margin_slack=float(args.margin_slack),
            density_topk=int(args.density_topk),
        )
        row = _metrics(query_labels, pred["pred_labels"], pred["accepted"], pred["pred_labels"])
        row.update(
            {
                "feature_npz": str(path),
                "candidate": path.parent.name,
                "variant": variant,
                "mechanism": _labels().get(variant, variant),
                "radius_quantile": float(args.radius_quantile),
                "density_quantile": float(args.density_quantile),
                "margin_quantile": float(args.margin_quantile),
                "registry_entries": int(len(entries)),
                "target_old_support_per_tx": int(args.target_old_support_per_tx),
                "target_old_query_per_tx": int(args.target_old_query_per_tx),
                "unknown_query_per_tx": int(args.unknown_query_per_tx),
                "query_role_counts": {role: int((query_roles == role).sum()) for role in sorted(set(query_roles.tolist()))},
            }
        )
        rows.append(row)
        registry[variant] = registry_summary
    payload_summary = {
        "feature_npz": str(path),
        "manifest": manifest,
        "counts": {
            "source": int(source_features.shape[0]),
            "support": int(support_features.shape[0]),
            "query": int(query_features.shape[0]),
            "target_old_query": int((query_labels != UNKNOWN_LABEL).sum()),
            "unknown_query": int((query_labels == UNKNOWN_LABEL).sum()),
        },
        "registry": registry,
    }
    return rows, payload_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-npz", action="append", default=[], help="Feature NPZ path or glob. Can be repeated.")
    parser.add_argument("--run-root", default=None, help="Run root containing candidate subdirs with features.npz.")
    parser.add_argument("--candidate", action="append", default=[], help="Candidate subdir name under each run-root/domain root.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--features-key", default="features")
    parser.add_argument("--tx-ids-key", default="tx_ids")
    parser.add_argument("--source-tx-ids", default=None)
    parser.add_argument("--target-old-tx-ids", default=None)
    parser.add_argument("--unknown-tx-ids", default=None)
    parser.add_argument("--source-proto-per-tx", type=int, default=48)
    parser.add_argument("--source-query-per-tx", type=int, default=40)
    parser.add_argument("--target-old-support-per-tx", type=int, default=10)
    parser.add_argument("--target-old-query-per-tx", type=int, default=30)
    parser.add_argument("--unknown-query-per-tx", type=int, default=30)
    parser.add_argument("--seed", type=int, default=362017)
    parser.add_argument("--variants", default="src_single,target_single,target_modes,target_modes_density,target_modes_density_margin")
    parser.add_argument("--radius-quantile", type=float, default=0.99)
    parser.add_argument("--radius-slack", type=float, default=0.0)
    parser.add_argument("--density-quantile", type=float, default=0.05)
    parser.add_argument("--density-slack", type=float, default=0.02)
    parser.add_argument("--margin-quantile", type=float, default=0.03)
    parser.add_argument("--margin-slack", type=float, default=0.0)
    parser.add_argument("--density-topk", type=int, default=3)
    parser.add_argument("--shrinkage", type=float, default=0.30)
    parser.add_argument("--support-weight", type=float, default=4.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_paths: list[Path] = []
    for pattern in args.feature_npz:
        matches = sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[") else [Path(pattern)]
        feature_paths.extend(matches)
    if args.run_root:
        root = Path(args.run_root)
        candidates = args.candidate or ["SHORT195S3_STAGE2B_FINALTEST_BASELINE_K10", "SHORT195S3_STAGE2B_FINALTEST_MULTIPROTO_K10"]
        for cand in candidates:
            feature_paths.extend(sorted(root.glob(f"*/{cand}/features.npz")))
            direct = root / cand / "features.npz"
            if direct.exists():
                feature_paths.append(direct)
    feature_paths = sorted({p.resolve() for p in feature_paths if p.exists()})
    if not feature_paths:
        raise FileNotFoundError("no feature NPZ files found")

    all_rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for path in feature_paths:
        rows, payload = evaluate_file(path, args)
        all_rows.extend(rows)
        payloads.append(payload)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "schema": "stage2_gaussian_registry_eval_v1",
                "protocol_boundary": {
                    "stage": "Stage2-B old-class Gaussian registry diagnostic",
                    "unknown_query_role": "eval_only",
                    "target_new": "excluded",
                    "unknown_query_threshold_calibration": False,
                },
                "config": _as_jsonable(vars(args)),
                "rows": all_rows,
                "payloads": _as_jsonable(payloads),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    fieldnames = [
        "feature_npz",
        "candidate",
        "variant",
        "mechanism",
        "old_acc",
        "old_coverage",
        "coverage",
        "unknown_far",
        "unknown_reject",
        "full_acc",
        "accepted_old_acc",
        "no_reject_old_acc",
        "old_n",
        "unknown_n",
        "accepted_n",
        "accepted_unknown_n",
        "rejected_unknown_n",
        "accepted_old_correct_n",
        "registry_entries",
        "target_old_support_per_tx",
        "target_old_query_per_tx",
        "unknown_query_per_tx",
    ]
    with args.summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"[GPR] files={len(feature_paths)} rows={len(all_rows)} output_json={args.output_json} summary_csv={args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
