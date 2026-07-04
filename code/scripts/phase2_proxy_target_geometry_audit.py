#!/usr/bin/env python
"""Audit proxy_unknown and target_unknown geometry against Stage2-C support prototypes.

This is a post-hoc diagnostic. target_unknown labels are used only for
reporting geometry and false-accept rates, not for fitting thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from phase2_collaborative_open_set_qknn_eval import (  # noqa: E402
    UNKNOWN_ROLE,
    _normalize_rows,
    _split_support_query_selected,
    load_feature_npz,
    validate_required_roles,
)


def _stats(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _safe_auroc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    pos = np.asarray(list(positive), dtype=np.float64)
    neg = np.asarray(list(negative), dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return None
    wins = 0.0
    for value in pos:
        wins += float(np.sum(value > neg)) + 0.5 * float(np.sum(value == neg))
    return float(wins / (pos.size * neg.size))


def _fpr_at_tpr(positive: Sequence[float], negative: Sequence[float], target_tpr: float) -> float | None:
    pos = np.asarray(list(positive), dtype=np.float64)
    neg = np.asarray(list(negative), dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return None
    threshold = float(np.quantile(pos, max(0.0, min(1.0, 1.0 - float(target_tpr)))))
    return float(np.mean(neg >= threshold))


def _role_labels(payload: Mapping[str, Any], role: str) -> list[str]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    return sorted({str(tx_ids[i]) for i in np.where(roles == role)[0].tolist()})


def _target_receivers(payload: Mapping[str, Any]) -> list[str]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    mask = np.isin(roles, ["target_old", "target_new", UNKNOWN_ROLE])
    return sorted({str(rx_ids[i]) for i in np.where(mask)[0].tolist()})


def _build_support_query_indices(
    payload: Mapping[str, Any],
    *,
    k_shot: int,
    query_per_class: int,
    seed: int,
    support_selection_policy: str,
) -> tuple[list[int], list[str], dict[str, list[int]]]:
    features = np.asarray(payload["features"], dtype=np.float32)
    old_labels = _role_labels(payload, "target_old")
    new_labels = _role_labels(payload, "target_new")
    unknown_labels = _role_labels(payload, UNKNOWN_ROLE)
    receivers = _target_receivers(payload)
    support_indices: list[int] = []
    support_labels: list[str] = []
    query_by_group: dict[str, list[int]] = {"target_old_query": [], "target_new_query": [], "target_unknown_query": []}
    for rx in receivers:
        for role, labels, group in [
            ("target_old", old_labels, "target_old_query"),
            ("target_new", new_labels, "target_new_query"),
        ]:
            for label in labels:
                support, query = _split_support_query_selected(
                    payload,
                    features=features,
                    role=role,
                    tx_id=label,
                    rx_id=rx,
                    k_shot=int(k_shot),
                    query_per_class=int(query_per_class),
                    seed=int(seed),
                    support_selection_policy=support_selection_policy,
                )
                if len(support) < int(k_shot) or len(query) < int(query_per_class):
                    raise RuntimeError(
                        "LOCAL_DATASET_EXTENSION_REQUIRED: incomplete support/query for "
                        f"rx={rx}, role={role}, tx_id={label}, support={len(support)}, query={len(query)}"
                    )
                support_indices.extend(int(i) for i in support)
                support_labels.extend([label] * len(support))
                query_by_group[group].extend(int(i) for i in query)
        for label in unknown_labels:
            support, query = _split_support_query_selected(
                payload,
                features=features,
                role=UNKNOWN_ROLE,
                tx_id=label,
                rx_id=rx,
                k_shot=0,
                query_per_class=int(query_per_class),
                seed=int(seed),
                support_selection_policy=support_selection_policy,
            )
            if support:
                raise RuntimeError("LOCAL_PROTOCOL_REPAIR_REQUIRED: target_unknown entered support")
            if len(query) < int(query_per_class):
                raise RuntimeError(
                    "LOCAL_DATASET_EXTENSION_REQUIRED: incomplete target_unknown query for "
                    f"rx={rx}, tx_id={label}, query={len(query)}"
                )
            query_by_group["target_unknown_query"].extend(int(i) for i in query)
    roles = np.asarray(payload["dataset_role"]).astype(str)
    query_by_group["proxy_unknown"] = [int(i) for i in np.where(roles == "proxy_unknown")[0].tolist()]
    return support_indices, support_labels, query_by_group


def _build_prototypes(features: np.ndarray, support_indices: Sequence[int], support_labels: Sequence[str]) -> tuple[list[str], np.ndarray]:
    labels = sorted(set(str(v) for v in support_labels))
    support_arr = np.asarray(support_indices, dtype=int)
    label_arr = np.asarray([str(v) for v in support_labels])
    protos = []
    for label in labels:
        rows = support_arr[label_arr == label]
        protos.append(_normalize_rows(features[rows].mean(axis=0, keepdims=True))[0])
    return labels, _normalize_rows(np.vstack(protos).astype(np.float32))


def _score_rows(features: np.ndarray, rows: Sequence[int], labels: Sequence[str], prototypes: np.ndarray, true_tx: np.ndarray) -> list[dict[str, Any]]:
    if not rows:
        return []
    idx = np.asarray(rows, dtype=int)
    x = _normalize_rows(features[idx])
    scores = x @ prototypes.T
    order = np.argsort(-scores, axis=1)
    out: list[dict[str, Any]] = []
    for local_i, global_i in enumerate(idx.tolist()):
        top = int(order[local_i, 0])
        second = int(order[local_i, 1]) if len(labels) > 1 else top
        out.append(
            {
                "row": int(global_i),
                "true_tx": str(true_tx[global_i]),
                "nearest_label": str(labels[top]),
                "max_cosine": float(scores[local_i, top]),
                "second_cosine": float(scores[local_i, second]),
                "margin": float(scores[local_i, top] - scores[local_i, second]),
                "correct_known_label": bool(str(true_tx[global_i]) == str(labels[top])),
            }
        )
    return out


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_feature_npz(args.feature_npz)
    validate_required_roles(payload)
    features = _normalize_rows(np.asarray(payload["features"], dtype=np.float32))
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    support_indices, support_labels, query_by_group = _build_support_query_indices(
        payload,
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
        support_selection_policy=str(args.support_selection_policy),
    )
    proto_labels, prototypes = _build_prototypes(features, support_indices, support_labels)
    support_scores = _score_rows(features, support_indices, proto_labels, prototypes, tx_ids)
    support_known_scores = [row["max_cosine"] for row in support_scores]
    threshold = float(np.quantile(np.asarray(support_known_scores, dtype=np.float64), float(args.known_accept_quantile)))

    groups: dict[str, Any] = {}
    detail_rows: list[dict[str, Any]] = []
    for group, rows in query_by_group.items():
        scored = _score_rows(features, rows, proto_labels, prototypes, tx_ids)
        for row in scored:
            accepted = bool(float(row["max_cosine"]) >= threshold)
            row["group"] = group
            row["accepted_by_support_threshold"] = accepted
            detail_rows.append(row)
        accept_rate = float(np.mean([float(row["accepted_by_support_threshold"]) for row in scored])) if scored else 0.0
        groups[group] = {
            "n": len(scored),
            "accept_rate_at_support_threshold": accept_rate,
            "reject_rate_at_support_threshold": 1.0 - accept_rate,
            "max_cosine": _stats([row["max_cosine"] for row in scored]),
            "margin": _stats([row["margin"] for row in scored]),
            "nearest_label_counts": dict(Counter(row["nearest_label"] for row in scored)),
        }
        if group in {"target_old_query", "target_new_query"}:
            groups[group]["top1_acc"] = float(np.mean([float(row["correct_known_label"]) for row in scored])) if scored else 0.0

    known_query_scores = [
        row["max_cosine"]
        for row in detail_rows
        if row["group"] in {"target_old_query", "target_new_query"}
    ]
    target_unknown_scores = [row["max_cosine"] for row in detail_rows if row["group"] == "target_unknown_query"]
    proxy_scores = [row["max_cosine"] for row in detail_rows if row["group"] == "proxy_unknown"]
    result = {
        "algorithm": "stage2_proxy_target_geometry_audit_v1",
        "feature_npz": str(args.feature_npz),
        "k_shot": int(args.k_shot),
        "query_per_class": int(args.query_per_class),
        "seed": int(args.seed),
        "support_selection_policy": str(args.support_selection_policy),
        "known_accept_quantile": float(args.known_accept_quantile),
        "support_threshold_source": "target_old_and_target_new_support_only",
        "support_threshold_uses_target_unknown": False,
        "support_known_accept_threshold": threshold,
        "prototype_labels": proto_labels,
        "support_count": len(support_indices),
        "groups": groups,
        "separation": {
            "known_vs_target_unknown_auroc": _safe_auroc(known_query_scores, target_unknown_scores),
            "known_vs_proxy_unknown_auroc": _safe_auroc(known_query_scores, proxy_scores),
            "target_unknown_fpr95_known": _fpr_at_tpr(known_query_scores, target_unknown_scores, 0.95),
            "proxy_unknown_fpr95_known": _fpr_at_tpr(known_query_scores, proxy_scores, 0.95),
            "target_unknown_minus_proxy_mean_max_cosine": (
                float(np.mean(target_unknown_scores) - np.mean(proxy_scores)) if target_unknown_scores and proxy_scores else None
            ),
        },
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            fields = ["group", "row", "true_tx", "nearest_label", "max_cosine", "second_cosine", "margin", "correct_known_label", "accepted_by_support_threshold"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(detail_rows)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", type=Path, required=True)
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--output_csv", type=Path)
    parser.add_argument("--k_shot", type=int, default=8)
    parser.add_argument("--query_per_class", type=int, default=12)
    parser.add_argument("--seed", type=int, default=4070801)
    parser.add_argument("--support_selection_policy", default="stable_first", choices=["stable_first", "centroid", "scenario_diverse"])
    parser.add_argument("--known_accept_quantile", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_audit(args)
    print(json.dumps({
        "feature_npz": result["feature_npz"],
        "support_threshold": result["support_known_accept_threshold"],
        "separation": result["separation"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
