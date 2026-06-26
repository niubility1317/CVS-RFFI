"""Analyze Stage2 old/unknown feature separability from exported feature NPZs.

The tool is diagnostic-only. It never fits thresholds from unknown query labels
for launchable deployment decisions; any threshold that uses unknown labels is
reported as an oracle diagnostic to test whether the feature geometry is
separable in principle.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.wisig_fewshot_payload import build_sfe_payload_from_feature_arrays, parse_tx_id_list  # noqa: E402


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def stats(values: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return {"n": 0}
    vals_sorted = sorted(vals)

    def quantile(q: float) -> float:
        if len(vals_sorted) == 1:
            return vals_sorted[0]
        pos = (len(vals_sorted) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals_sorted[lo]
        return vals_sorted[lo] * (hi - pos) + vals_sorted[hi] * (pos - lo)

    return {
        "n": len(vals_sorted),
        "mean": statistics.fmean(vals_sorted),
        "min": vals_sorted[0],
        "q05": quantile(0.05),
        "q10": quantile(0.10),
        "median": statistics.median(vals_sorted),
        "q90": quantile(0.90),
        "q95": quantile(0.95),
        "max": vals_sorted[-1],
    }


def normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return arr / denom


def centroid_set(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    feats = normalize_rows(features)
    label_values = sorted(int(v) for v in np.unique(labels).tolist() if int(v) >= 0)
    vectors: list[np.ndarray] = []
    counts: dict[int, int] = {}
    for label in label_values:
        mask = labels == int(label)
        counts[int(label)] = int(mask.sum())
        vectors.append(normalize_rows(feats[mask].mean(axis=0, keepdims=True))[0])
    if not vectors:
        return np.empty((0, feats.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64), counts
    return np.asarray(vectors, dtype=np.float32), np.asarray(label_values, dtype=np.int64), counts


def fuse_prototypes(
    source_proto: np.ndarray,
    source_labels: np.ndarray,
    support_proto: np.ndarray,
    support_labels: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    support_by_label = {int(label): support_proto[i] for i, label in enumerate(support_labels.tolist())}
    out: list[np.ndarray] = []
    labels: list[int] = []
    for i, label in enumerate(source_labels.tolist()):
        labels.append(int(label))
        if int(label) in support_by_label:
            vec = (1.0 - float(rho)) * source_proto[i] + float(rho) * support_by_label[int(label)]
        else:
            vec = source_proto[i]
        out.append(normalize_rows(np.asarray(vec, dtype=np.float32).reshape(1, -1))[0])
    return np.asarray(out, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def score_against(features: np.ndarray, proto: np.ndarray, proto_labels: np.ndarray) -> dict[str, np.ndarray]:
    feats = normalize_rows(features)
    scores = feats @ proto.T
    order = np.argsort(scores, axis=1)
    best_idx = order[:, -1]
    second_idx = order[:, -2] if scores.shape[1] > 1 else best_idx
    best = scores[np.arange(scores.shape[0]), best_idx]
    second = scores[np.arange(scores.shape[0]), second_idx]
    return {
        "scores": scores,
        "best_score": best.astype(np.float64),
        "second_score": second.astype(np.float64),
        "margin": (best - second).astype(np.float64),
        "pred": proto_labels[best_idx].astype(np.int64),
    }


def pairwise_auroc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float | None:
    pos = np.asarray(positive_scores, dtype=np.float64)
    neg = np.asarray(negative_scores, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return None
    wins = 0.0
    for value in pos:
        wins += float((value > neg).sum())
        wins += 0.5 * float((value == neg).sum())
    return wins / float(pos.size * neg.size)


def threshold_metrics(
    *,
    best_scores: np.ndarray,
    pred: np.ndarray,
    true_labels: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, float]]:
    labels = np.asarray(true_labels, dtype=np.int64)
    old_mask = labels >= 0
    unknown_mask = labels == -1
    old_n = max(1, int(old_mask.sum()))
    unknown_n = max(1, int(unknown_mask.sum()))
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        accepted = np.asarray(best_scores, dtype=np.float64) >= float(threshold)
        old_correct = accepted & old_mask & (np.asarray(pred, dtype=np.int64) == labels)
        rows.append(
            {
                "threshold": float(threshold),
                "old_acc_full_denominator": float(old_correct.sum()) / float(old_n),
                "old_retention": float((accepted & old_mask).sum()) / float(old_n),
                "unknown_far": float((accepted & unknown_mask).sum()) / float(unknown_n),
                "coverage": float(accepted.sum()) / float(max(1, accepted.size)),
            }
        )
    return rows


def best_oracle_at_far(rows: list[dict[str, float]], far_limit: float) -> dict[str, float] | None:
    eligible = [row for row in rows if row["unknown_far"] <= float(far_limit) + 1e-12]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            row["old_acc_full_denominator"],
            row["old_retention"],
            -row["unknown_far"],
            -row["threshold"],
        ),
    )


def class_envelope_metrics(
    *,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    query_features: np.ndarray,
    query_labels: np.ndarray,
    proto: np.ndarray,
    proto_labels: np.ndarray,
    floor_quantile: float,
) -> dict[str, Any]:
    calib = score_against(calibration_features, proto, proto_labels)
    label_to_floor: dict[int, float] = {}
    for label in proto_labels.tolist():
        mask = np.asarray(calibration_labels, dtype=np.int64) == int(label)
        if not bool(mask.any()):
            continue
        label_idx = int(np.flatnonzero(proto_labels == int(label))[0])
        true_scores = calib["scores"][mask, label_idx]
        label_to_floor[int(label)] = stats([float(v) for v in true_scores])["q05" if floor_quantile == 0.05 else "q10"]
    q = score_against(query_features, proto, proto_labels)
    pass_mask = np.zeros_like(q["best_score"], dtype=bool)
    for i, pred_label in enumerate(q["pred"].tolist()):
        floor = label_to_floor.get(int(pred_label))
        if floor is not None and float(q["best_score"][i]) >= float(floor):
            pass_mask[i] = True
    labels = np.asarray(query_labels, dtype=np.int64)
    old_mask = labels >= 0
    unknown_mask = labels == -1
    old_n = max(1, int(old_mask.sum()))
    unknown_n = max(1, int(unknown_mask.sum()))
    return {
        "floor_quantile": float(floor_quantile),
        "floors": {str(k): v for k, v in sorted(label_to_floor.items())},
        "old_pass_rate": float((pass_mask & old_mask).sum()) / float(old_n),
        "old_correct_pass_rate": float((pass_mask & old_mask & (q["pred"] == labels)).sum()) / float(old_n),
        "unknown_pass_rate": float((pass_mask & unknown_mask).sum()) / float(unknown_n),
    }


def compactness(features: np.ndarray) -> float | None:
    arr = normalize_rows(features)
    if arr.shape[0] < 2:
        return None
    sim = arr @ arr.T
    tri = sim[np.triu_indices(arr.shape[0], k=1)]
    return float(np.mean(tri)) if tri.size else None


def load_manifest(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "manifest_json" not in data:
        return {}
    try:
        return json.loads(str(data["manifest_json"].item()))
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        return {"manifest_decode_error": str(exc)}


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        return {"json_decode_error": str(exc), "path": str(path)}


def first_nested(manifest: dict[str, Any], key: str) -> Any:
    if key in manifest:
        return manifest.get(key)
    for parent in ("protocol_payload", "embedded_manifest", "extra_metadata"):
        value = manifest.get(parent)
        if isinstance(value, dict):
            found = first_nested(value, key)
            if found is not None:
                return found
    return None


def manifest_list(manifest: dict[str, Any], key: str) -> list[str]:
    value = first_nested(manifest, key)
    return parse_tx_id_list(value)


def add_metadata_arrays(
    arrays: dict[str, np.ndarray],
    *,
    data: np.lib.npyio.NpzFile,
    source_idx: np.ndarray,
    support_idx: np.ndarray,
    query_idx: np.ndarray,
) -> None:
    for name in ("rx_ids", "day_ids", "eq_ids", "sig_ids", "channel_views", "sat_scenarios", "dataset_role"):
        if name not in data:
            continue
        arr = np.asarray(data[name]).reshape(-1)
        short = "dataset_roles" if name == "dataset_role" else name
        arrays[f"source_{short}"] = arr[source_idx].astype(str)
        arrays[f"support_{short}"] = arr[support_idx].astype(str)
        arrays[f"query_{short}"] = arr[query_idx].astype(str)


def build_payload_from_existing_manifest_splits(
    *,
    path: Path,
    data: np.lib.npyio.NpzFile,
    manifest: dict[str, Any],
    features_key: str,
    tx_ids_key: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    splits = first_nested(manifest, "split_indices_by_role")
    source_label_map_raw = first_nested(manifest, "source_label_map")
    if not isinstance(splits, dict) or not isinstance(source_label_map_raw, dict):
        return None
    required = ("source_prototype", "target_old_support", "target_old_query", "unknown_query")
    if any(name not in splits for name in required):
        return None
    features = np.asarray(data[features_key], dtype=np.float32)
    tx_ids = np.asarray(data[tx_ids_key]).reshape(-1).astype(str)
    source_label_map = {str(k): int(v) for k, v in source_label_map_raw.items()}

    def idx_for(name: str) -> np.ndarray:
        return np.asarray(splits.get(name, []), dtype=np.int64).reshape(-1)

    def labels_for(indices: np.ndarray, *, unknown: bool = False) -> np.ndarray:
        if unknown:
            return np.full((int(indices.size),), -1, dtype=np.int64)
        labels: list[int] = []
        missing: list[str] = []
        for tx in tx_ids[indices].astype(str).tolist():
            if tx not in source_label_map:
                missing.append(tx)
            else:
                labels.append(int(source_label_map[tx]))
        if missing:
            raise ValueError(
                f"{path} manifest split labels cannot resolve tx ids in source_label_map: "
                f"{sorted(set(missing))[:8]}"
            )
        return np.asarray(labels, dtype=np.int64)

    source_idx = idx_for("source_prototype")
    support_idx = idx_for("target_old_support")
    target_old_query_idx = idx_for("target_old_query")
    unknown_query_idx = idx_for("unknown_query")
    query_idx = np.concatenate([target_old_query_idx, unknown_query_idx])
    arrays = {
        "source_features": features[source_idx],
        "source_labels": labels_for(source_idx),
        "support_features": features[support_idx],
        "support_labels": labels_for(support_idx),
        "query_features": features[query_idx],
        "query_labels": np.concatenate([labels_for(target_old_query_idx), labels_for(unknown_query_idx, unknown=True)]),
        "source_tx_ids": tx_ids[source_idx].astype(str),
        "support_tx_ids": tx_ids[support_idx].astype(str),
        "query_tx_ids": tx_ids[query_idx].astype(str),
        "query_roles": np.asarray(
            ["target_old_query"] * int(target_old_query_idx.size)
            + ["unknown_query"] * int(unknown_query_idx.size),
            dtype=str,
        ),
        "source_sample_indices": source_idx.astype(np.int64),
        "support_sample_indices": support_idx.astype(np.int64),
        "query_sample_indices": query_idx.astype(np.int64),
    }
    add_metadata_arrays(arrays, data=data, source_idx=source_idx, support_idx=support_idx, query_idx=query_idx)
    merged_manifest = dict(manifest)
    merged_manifest["split_source"] = "existing_manifest_json"
    return arrays, merged_manifest


def build_payload_from_npz(path: Path, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        embedded_manifest = load_manifest(data)
        sibling_manifest = load_json_file(path.with_name("manifest.json"))
        manifest = dict(embedded_manifest)
        if sibling_manifest:
            manifest.update(sibling_manifest)
        features_key = str(args.features_key)
        tx_ids_key = str(args.tx_ids_key)
        if features_key not in data or tx_ids_key not in data:
            raise KeyError(f"{path} missing {features_key!r} or {tx_ids_key!r}; keys={list(data.files)}")
        if bool(args.prefer_existing_splits):
            existing = build_payload_from_existing_manifest_splits(
                path=path,
                data=data,
                manifest=manifest,
                features_key=features_key,
                tx_ids_key=tx_ids_key,
            )
            if existing is not None:
                return existing
        source_tx_ids = parse_tx_id_list(args.source_tx_ids) or manifest_list(manifest, "source_tx_ids")
        target_old_tx_ids = parse_tx_id_list(args.target_old_tx_ids) or manifest_list(manifest, "target_old_tx_ids")
        unknown_tx_ids = parse_tx_id_list(args.unknown_tx_ids) or manifest_list(manifest, "unknown_tx_ids")
        if not source_tx_ids or not target_old_tx_ids or not unknown_tx_ids:
            raise ValueError(
                f"{path} cannot infer source/target-old/unknown TX ids: "
                f"source={source_tx_ids} target_old={target_old_tx_ids} unknown={unknown_tx_ids}"
            )
        sample_metadata = {
            key: data[key]
            for key in ("rx_ids", "day_ids", "eq_ids", "sig_ids", "channel_views", "sat_scenarios")
            if key in data
        }
        payload = build_sfe_payload_from_feature_arrays(
            features=data[features_key],
            tx_ids=data[tx_ids_key],
            dataset_roles=data["dataset_role"] if "dataset_role" in data else None,
            sample_metadata=sample_metadata,
            source_tx_ids=source_tx_ids,
            target_old_tx_ids=target_old_tx_ids,
            new_tx_ids=[],
            unknown_tx_ids=unknown_tx_ids,
            shots=0,
            source_proto_per_tx=int(args.source_proto_per_tx),
            source_query_per_tx=int(args.source_query_per_tx),
            target_old_support_per_tx=int(args.target_old_support_per_tx),
            target_old_query_per_tx=int(args.target_old_query_per_tx),
            query_per_tx=int(args.unknown_query_per_tx),
            seed=int(args.seed),
            extra_metadata={"payload_source": str(path), "analysis_tool": "analyze_h06_feature_separability"},
        )
        promoted_manifest = dict(manifest)
        promoted_manifest["protocol_payload"] = payload.manifest
        promoted_manifest["split_source"] = "rebuilt_from_feature_arrays"
        return payload.arrays, promoted_manifest


def summarize_variant(
    *,
    name: str,
    proto: np.ndarray,
    proto_labels: np.ndarray,
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    query_labels: np.ndarray,
    query_roles: np.ndarray,
    threshold_source_features: np.ndarray,
    threshold_source_labels: np.ndarray,
    far_limit: float,
) -> dict[str, Any]:
    support_score = score_against(support_features, proto, proto_labels)
    query_score = score_against(query_features, proto, proto_labels)
    old_mask = np.asarray(query_labels, dtype=np.int64) >= 0
    unknown_mask = np.asarray(query_labels, dtype=np.int64) == -1
    support_n = max(1, int(support_labels.size))
    old_n = max(1, int(old_mask.sum()))
    unknown_n = max(1, int(unknown_mask.sum()))
    old_correct = old_mask & (query_score["pred"] == query_labels)
    support_correct = support_score["pred"] == support_labels
    candidate_thresholds = sorted(
        {
            float(v)
            for v in np.concatenate([query_score["best_score"], support_score["best_score"]], axis=0).tolist()
            if math.isfinite(float(v))
        },
        reverse=True,
    )
    if candidate_thresholds:
        candidate_thresholds.append(min(candidate_thresholds) - 1e-6)
    curve = threshold_metrics(
        best_scores=query_score["best_score"],
        pred=query_score["pred"],
        true_labels=query_labels,
        thresholds=candidate_thresholds,
    )
    oracle = best_oracle_at_far(curve, far_limit)
    support_floor = stats([float(v) for v in support_score["best_score"]]).get("q05")
    source_score = score_against(threshold_source_features, proto, proto_labels)
    source_floor = stats([float(v) for v in source_score["best_score"]]).get("q05")
    selected_thresholds: dict[str, Any] = {}
    for label, threshold in (("support_q05", support_floor), ("source_q05", source_floor)):
        if threshold is None:
            continue
        selected_thresholds[label] = threshold_metrics(
            best_scores=query_score["best_score"],
            pred=query_score["pred"],
            true_labels=query_labels,
            thresholds=[float(threshold)],
        )[0]
    return {
        "name": name,
        "support_acc": float(support_correct.sum()) / float(support_n),
        "old_query_acc_no_reject": float(old_correct.sum()) / float(old_n),
        "unknown_best_score_stats": stats([float(v) for v in query_score["best_score"][unknown_mask]]),
        "old_best_score_stats": stats([float(v) for v in query_score["best_score"][old_mask]]),
        "old_margin_stats": stats([float(v) for v in query_score["margin"][old_mask]]),
        "unknown_margin_stats": stats([float(v) for v in query_score["margin"][unknown_mask]]),
        "old_vs_unknown_best_score_auroc": pairwise_auroc(
            query_score["best_score"][old_mask],
            query_score["best_score"][unknown_mask],
        ),
        "oracle_threshold_uses_unknown_labels_diagnostic_only": oracle,
        "unsupervised_thresholds": selected_thresholds,
        "class_envelope_from_support": class_envelope_metrics(
            calibration_features=support_features,
            calibration_labels=support_labels,
            query_features=query_features,
            query_labels=query_labels,
            proto=proto,
            proto_labels=proto_labels,
            floor_quantile=0.05,
        ),
        "class_envelope_from_source": class_envelope_metrics(
            calibration_features=threshold_source_features,
            calibration_labels=threshold_source_labels,
            query_features=query_features,
            query_labels=query_labels,
            proto=proto,
            proto_labels=proto_labels,
            floor_quantile=0.05,
        ),
        "old_per_query_role": {
            str(role): {
                "n": int(((query_roles == role) & old_mask).sum()),
                "acc_no_reject": (
                    float((((query_roles == role) & old_correct).sum())) / float(max(1, int(((query_roles == role) & old_mask).sum())))
                ),
            }
            for role in sorted(set(str(v) for v in query_roles.tolist()))
            if int(((query_roles == role) & old_mask).sum()) > 0
        },
    }


def analyze_feature_npz(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    arrays, manifest = build_payload_from_npz(path, args)
    source_features = arrays["source_features"]
    source_labels = arrays["source_labels"]
    support_features = arrays["support_features"]
    support_labels = arrays["support_labels"]
    query_features = arrays["query_features"]
    query_labels = arrays["query_labels"]
    query_roles = np.asarray(arrays.get("query_roles", np.asarray([""] * query_labels.shape[0])), dtype=str)
    source_proto, source_proto_labels, source_counts = centroid_set(source_features, source_labels)
    support_proto, support_proto_labels, support_counts = centroid_set(support_features, support_labels)
    rhos = [float(v) for v in str(args.fusion_rhos).split(",") if str(v).strip()]
    variants: dict[str, Any] = {}
    for rho in rhos:
        proto, proto_labels = fuse_prototypes(source_proto, source_proto_labels, support_proto, support_proto_labels, rho)
        name = f"rho_{rho:g}"
        variants[name] = summarize_variant(
            name=name,
            proto=proto,
            proto_labels=proto_labels,
            support_features=support_features,
            support_labels=support_labels,
            query_features=query_features,
            query_labels=query_labels,
            query_roles=query_roles,
            threshold_source_features=source_features,
            threshold_source_labels=source_labels,
            far_limit=float(args.far_limit),
        )
    per_label: dict[str, Any] = {}
    source_by_label = {int(label): source_proto[i] for i, label in enumerate(source_proto_labels.tolist())}
    support_by_label = {int(label): support_proto[i] for i, label in enumerate(support_proto_labels.tolist())}
    for label in source_proto_labels.tolist():
        label = int(label)
        support_mask = support_labels == label
        old_query_mask = query_labels == label
        info: dict[str, Any] = {
            "source_count": int(source_counts.get(label, 0)),
            "support_count": int(support_counts.get(label, 0)),
            "old_query_count": int(old_query_mask.sum()),
            "support_compactness": compactness(support_features[support_mask]),
            "old_query_compactness": compactness(query_features[old_query_mask]),
        }
        if label in source_by_label and label in support_by_label:
            info["source_support_proto_cos"] = float(np.dot(source_by_label[label], support_by_label[label]))
        if bool(old_query_mask.any()) and label in support_by_label:
            q_centroid = normalize_rows(normalize_rows(query_features[old_query_mask]).mean(axis=0, keepdims=True))[0]
            info["support_query_centroid_cos"] = float(np.dot(support_by_label[label], q_centroid))
            info["source_query_centroid_cos"] = float(np.dot(source_by_label[label], q_centroid))
        per_label[str(label)] = info
    best_variant_name = max(
        variants,
        key=lambda key: (
            variants[key]["oracle_threshold_uses_unknown_labels_diagnostic_only"] or {"old_acc_full_denominator": -1}
        )["old_acc_full_denominator"],
    )
    counts = {
        "source": int(source_features.shape[0]),
        "support": int(support_features.shape[0]),
        "query": int(query_features.shape[0]),
        "target_old_query": int((query_labels >= 0).sum()),
        "unknown_query": int((query_labels == -1).sum()),
    }
    return {
        "feature_npz": str(path),
        "counts": counts,
        "manifest": {
            "checkpoint": manifest.get("checkpoint"),
            "target_channel_view": manifest.get("target_channel_view"),
            "star_ground_channel_impl": manifest.get("star_ground_channel_impl"),
            "target_channel_scenarios": manifest.get("target_channel_scenarios"),
            "source_tx_ids": manifest.get("source_tx_ids"),
            "target_old_tx_ids": manifest.get("target_old_tx_ids"),
            "unknown_tx_ids": manifest.get("unknown_tx_ids"),
            "split_source": manifest.get("split_source"),
            "protocol_payload": manifest.get("protocol_payload", {}),
        },
        "variants": variants,
        "best_oracle_variant": best_variant_name,
        "per_label_geometry": per_label,
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {"feature_npz": result["feature_npz"]}
        for name, variant in result["variants"].items():
            prefix = name + "."
            row[prefix + "support_acc"] = variant["support_acc"]
            row[prefix + "old_query_acc_no_reject"] = variant["old_query_acc_no_reject"]
            row[prefix + "auroc"] = variant["old_vs_unknown_best_score_auroc"]
            oracle = variant.get("oracle_threshold_uses_unknown_labels_diagnostic_only") or {}
            row[prefix + "oracle_far05_old_acc"] = oracle.get("old_acc_full_denominator")
            row[prefix + "oracle_far05_old_retention"] = oracle.get("old_retention")
            row[prefix + "oracle_far05_unknown_far"] = oracle.get("unknown_far")
            support_q05 = variant.get("unsupervised_thresholds", {}).get("support_q05", {})
            row[prefix + "support_q05_old_acc"] = support_q05.get("old_acc_full_denominator")
            row[prefix + "support_q05_unknown_far"] = support_q05.get("unknown_far")
            row[prefix + "support_envelope_old_correct"] = variant.get("class_envelope_from_support", {}).get("old_correct_pass_rate")
            row[prefix + "support_envelope_unknown_pass"] = variant.get("class_envelope_from_support", {}).get("unknown_pass_rate")
        rows.append(row)
    aggregate: dict[str, Any] = {"candidate_count": len(results), "metrics": {}}
    numeric_keys = sorted({key for row in rows for key, value in row.items() if finite_float(value) is not None})
    for key in numeric_keys:
        aggregate["metrics"][key] = stats([float(row[key]) for row in rows if finite_float(row.get(key)) is not None])
    return {"aggregate": aggregate, "rows": rows}


def find_feature_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for value in args.feature_npz:
        if any(ch in str(value) for ch in "*?[]"):
            paths.extend(Path(p) for p in sorted(glob.glob(value)))
        else:
            paths.append(Path(value))
    if args.run_root:
        paths.extend(sorted(Path(args.run_root).glob("*/features.npz")))
    unique = []
    seen = set()
    for path in paths:
        resolved = str(path)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if int(args.max_files) > 0:
        unique = unique[: int(args.max_files)]
    if not unique:
        raise FileNotFoundError("no feature NPZ files found")
    return unique


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-npz", action="append", default=[], help="Feature NPZ path or glob. Can be repeated.")
    parser.add_argument("--run-root", default=None, help="Run root containing candidate subdirs with features.npz.")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--features-key", default="features")
    parser.add_argument("--tx-ids-key", default="tx_ids")
    parser.add_argument("--prefer-existing-splits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source-tx-ids", default="")
    parser.add_argument("--target-old-tx-ids", default="")
    parser.add_argument("--unknown-tx-ids", default="")
    parser.add_argument("--source-proto-per-tx", type=int, default=112)
    parser.add_argument("--source-query-per-tx", type=int, default=56)
    parser.add_argument("--target-old-support-per-tx", type=int, default=5)
    parser.add_argument("--target-old-query-per-tx", type=int, default=30)
    parser.add_argument("--unknown-query-per-tx", type=int, default=30)
    parser.add_argument("--seed", type=int, default=191150)
    parser.add_argument("--fusion-rhos", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--far-limit", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = find_feature_paths(args)
    results = [analyze_feature_npz(path, args) for path in paths]
    aggregate = aggregate_results(results)
    out = {
        "schema": "h06_feature_separability_diagnostic_v1",
        "protocol_boundary": {
            "stage": "Stage2-B old-class calibration diagnostic",
            "unknown_query_role": "eval_only",
            "oracle_thresholds_use_unknown_labels": "diagnostic_only_not_launchable",
            "target_new": "excluded",
        },
        "config": {
            "source_proto_per_tx": int(args.source_proto_per_tx),
            "target_old_support_per_tx": int(args.target_old_support_per_tx),
            "target_old_query_per_tx": int(args.target_old_query_per_tx),
            "unknown_query_per_tx": int(args.unknown_query_per_tx),
            "fusion_rhos": [float(v) for v in str(args.fusion_rhos).split(",") if str(v).strip()],
            "far_limit": float(args.far_limit),
            "seed": int(args.seed),
        },
        "summary": aggregate["aggregate"],
        "rows": aggregate["rows"],
        "details": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.summary_csv is not None:
        write_summary_csv(args.summary_csv, aggregate["rows"])
    print(json.dumps({"output_json": str(args.output_json), "candidate_count": len(results), "summary_keys": len(out["summary"]["metrics"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
