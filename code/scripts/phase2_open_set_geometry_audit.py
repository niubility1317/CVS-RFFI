#!/usr/bin/env python
"""Open-set geometry audit for Stage2-C ADV3B02 feature packages.

The audit quantifies whether target_unknown rows are separable from old and
seen-new query rows under support/source-calibrated risks. It is diagnostic-only:
target_unknown labels are used only for post-hoc metrics, never for threshold
selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from phase2_aware_ci_eval import _fit_aware_model, _open_score, _query_indices, _raw_components  # noqa: E402
from phase2_collaborative_open_set_qknn_eval import load_feature_npz  # noqa: E402
from phase2_proxy_adapter_ci_eval import build_training_plan  # noqa: E402


def _auc_unknown_higher(known_scores: Sequence[float], unknown_scores: Sequence[float]) -> float | None:
    known = np.asarray([float(v) for v in known_scores], dtype=np.float64)
    unknown = np.asarray([float(v) for v in unknown_scores], dtype=np.float64)
    if known.size == 0 or unknown.size == 0:
        return None
    comp = unknown.reshape(-1, 1) > known.reshape(1, -1)
    ties = unknown.reshape(-1, 1) == known.reshape(1, -1)
    return float((comp.astype(np.float64) + 0.5 * ties.astype(np.float64)).mean())


def _fpr_at_tpr(known_scores: Sequence[float], unknown_scores: Sequence[float], target_tpr: float) -> float | None:
    known = np.asarray([float(v) for v in known_scores], dtype=np.float64)
    unknown = np.asarray([float(v) for v in unknown_scores], dtype=np.float64)
    if known.size == 0 or unknown.size == 0:
        return None
    thresholds = np.unique(np.concatenate([known, unknown]))
    best: float | None = None
    for threshold in thresholds:
        tpr = float((unknown >= threshold).sum()) / float(max(1, unknown.size))
        if tpr >= float(target_tpr):
            fpr = float((known >= threshold).sum()) / float(max(1, known.size))
            best = fpr if best is None else min(best, fpr)
    return best


def _stats(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "q05": float(np.quantile(arr, 0.05)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def _threshold_eval(
    *,
    threshold: float,
    known_scores: np.ndarray,
    old_scores: np.ndarray,
    seen_scores: np.ndarray,
    unknown_scores: np.ndarray,
) -> dict[str, Any]:
    return {
        "threshold": float(threshold),
        "known_reject_rate": float((known_scores >= threshold).sum()) / float(max(1, known_scores.size)),
        "old_reject_rate": float((old_scores >= threshold).sum()) / float(max(1, old_scores.size)),
        "seen_new_reject_rate": float((seen_scores >= threshold).sum()) / float(max(1, seen_scores.size)),
        "unknown_reject_rate": float((unknown_scores >= threshold).sum()) / float(max(1, unknown_scores.size)),
        "unknown_FAR": 1.0 - float((unknown_scores >= threshold).sum()) / float(max(1, unknown_scores.size)),
    }


def _oracle_at_far(
    *,
    known_scores: np.ndarray,
    old_scores: np.ndarray,
    seen_scores: np.ndarray,
    unknown_scores: np.ndarray,
    far_limit: float,
) -> dict[str, Any] | None:
    if unknown_scores.size == 0:
        return None
    thresholds = np.unique(np.concatenate([known_scores, old_scores, seen_scores, unknown_scores]))
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        row = _threshold_eval(
            threshold=float(threshold),
            known_scores=known_scores,
            old_scores=old_scores,
            seen_scores=seen_scores,
            unknown_scores=unknown_scores,
        )
        if float(row["unknown_FAR"]) <= float(far_limit):
            old_keep = 1.0 - float(row["old_reject_rate"])
            seen_keep = 1.0 - float(row["seen_new_reject_rate"])
            key = (old_keep + seen_keep, old_keep, seen_keep, float(row["unknown_reject_rate"]))
            if best is None or key > best["_key"]:
                best = {**row, "old_keep_rate": old_keep, "seen_new_keep_rate": seen_keep, "_key": key}
    if best is None:
        return None
    best.pop("_key", None)
    best["uses_target_unknown_labels"] = True
    best["diagnostic_only"] = True
    return best


def _flatten_component_rows(
    rows: Sequence[Mapping[str, Any]],
    component_names: Sequence[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        for component in component_names:
            summary = dict(row.get(component, {}))
            out.append(
                {
                    "component": component,
                    "role": row.get("role"),
                    "tx_id": row.get("tx_id"),
                    **summary,
                }
            )
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _component_scores(
    payload: Mapping[str, Any],
    model: Mapping[str, Any],
    indices: Sequence[int],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    comps = _raw_components(
        np.asarray(model["features"], dtype=np.float32),
        indices,
        prototypes=np.asarray(model["prototypes"], dtype=np.float32),
        label_values=list(model["label_values"]),
        diag_vars=model["diag_vars"],
        memory_features=np.asarray(model["memory_features"], dtype=np.float32),
        proxy_centroid=np.asarray(model["proxy_centroid"], dtype=np.float32),
        known_centroid=np.asarray(model["known_centroid"], dtype=np.float32),
        qknn_k=int(args.qknn_k),
    )
    aware_score = _open_score(comps, model["scales"], args)
    return {
        "aware_score": aware_score,
        "proto_dist": np.asarray(comps["proto_dist"], dtype=np.float64),
        "knn_dist": np.asarray(comps["knn_dist"], dtype=np.float64),
        "maha": np.asarray(comps["maha"], dtype=np.float64),
        "entropy": np.asarray(comps["entropy"], dtype=np.float64),
        "proxy_gap": np.asarray(comps["proxy_gap"], dtype=np.float64),
        "negative_margin": -np.asarray(comps["margin"], dtype=np.float64),
        "pred_pos": np.asarray(comps["pred_pos"], dtype=int),
        "pred_score": np.asarray(comps["pred_score"], dtype=np.float64),
    }


def run_geometry_audit(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_feature_npz(args.feature_npz)
    plan = build_training_plan(
        payload,
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
        support_selection_policy=str(args.support_selection_policy),
    )
    model = _fit_aware_model(payload, plan, args)
    query_idx = _query_indices(payload, plan, int(args.query_per_class), int(args.seed))
    train_known_idx = [*plan.source_old_indices, *plan.support_indices]
    proxy_idx = list(plan.proxy_unknown_indices)
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    label_values = list(model["label_values"])

    train_known_scores = _component_scores(payload, model, train_known_idx, args)
    proxy_scores = _component_scores(payload, model, proxy_idx, args)
    query_scores = _component_scores(payload, model, query_idx, args)
    query_roles = np.asarray([str(roles[int(i)]) for i in query_idx], dtype=object)
    query_tx = np.asarray([str(tx_ids[int(i)]) for i in query_idx], dtype=object)
    query_rx = np.asarray([str(rx_ids[int(i)]) for i in query_idx], dtype=object)
    role_masks = {
        "old": query_roles == "target_old",
        "seen_new": query_roles == "target_new",
        "unknown": query_roles == "target_unknown",
    }
    known_mask = role_masks["old"] | role_masks["seen_new"]
    component_names = ["aware_score", "proto_dist", "knn_dist", "maha", "entropy", "proxy_gap", "negative_margin"]
    component_summary: dict[str, Any] = {}
    for name in component_names:
        known_values = np.asarray(query_scores[name][known_mask], dtype=np.float64)
        old_values = np.asarray(query_scores[name][role_masks["old"]], dtype=np.float64)
        seen_values = np.asarray(query_scores[name][role_masks["seen_new"]], dtype=np.float64)
        unknown_values = np.asarray(query_scores[name][role_masks["unknown"]], dtype=np.float64)
        train_known_values = np.asarray(train_known_scores[name], dtype=np.float64)
        proxy_values = np.asarray(proxy_scores[name], dtype=np.float64)
        thresholds = {
            "train_known_q95": float(np.quantile(train_known_values, 0.95)),
            "train_known_q99": float(np.quantile(train_known_values, 0.99)),
            "support_source_q995": float(np.quantile(train_known_values, 0.995)),
            "proxy_median": float(np.quantile(proxy_values, 0.50)),
        }
        component_summary[name] = {
            "known_stats": _stats(known_values),
            "old_stats": _stats(old_values),
            "seen_new_stats": _stats(seen_values),
            "unknown_stats": _stats(unknown_values),
            "proxy_unknown_stats": _stats(proxy_values),
            "auroc_unknown_vs_known": _auc_unknown_higher(known_values, unknown_values),
            "fpr95_known_as_unknown": _fpr_at_tpr(known_values, unknown_values, 0.95),
            "thresholds_from_train_known_or_proxy": {
                key: _threshold_eval(
                    threshold=value,
                    known_scores=known_values,
                    old_scores=old_values,
                    seen_scores=seen_values,
                    unknown_scores=unknown_values,
                )
                for key, value in thresholds.items()
            },
            "oracle_far_05_diagnostic": _oracle_at_far(
                known_scores=known_values,
                old_scores=old_values,
                seen_scores=seen_values,
                unknown_scores=unknown_values,
                far_limit=0.05,
            ),
            "oracle_far_01_diagnostic": _oracle_at_far(
                known_scores=known_values,
                old_scores=old_values,
                seen_scores=seen_values,
                unknown_scores=unknown_values,
                far_limit=0.01,
            ),
        }

    pred_labels = np.asarray([str(label_values[int(pos)]) for pos in query_scores["pred_pos"].tolist()], dtype=object)
    per_unknown_tx: dict[str, Any] = {}
    for tx in sorted({str(v) for v in query_tx[role_masks["unknown"]].tolist()}):
        mask = role_masks["unknown"] & (query_tx == tx)
        pred_counts = Counter(str(v) for v in pred_labels[mask].tolist())
        per_unknown_tx[tx] = {
            "n": int(mask.sum()),
            "nearest_known_label_counts": dict(pred_counts),
            "top_absorbing_known_label": pred_counts.most_common(1)[0][0] if pred_counts else None,
            "component_stats": {name: _stats(query_scores[name][mask]) for name in component_names},
        }

    per_receiver: dict[str, Any] = {}
    for rx in sorted({str(v) for v in query_rx.tolist()}):
        mask = query_rx == rx
        per_receiver[rx] = {
            "n": int(mask.sum()),
            "old_n": int((mask & role_masks["old"]).sum()),
            "seen_new_n": int((mask & role_masks["seen_new"]).sum()),
            "unknown_n": int((mask & role_masks["unknown"]).sum()),
            "aware_auroc_unknown_vs_known": _auc_unknown_higher(
                query_scores["aware_score"][mask & known_mask],
                query_scores["aware_score"][mask & role_masks["unknown"]],
            ),
            "aware_fpr95_known_as_unknown": _fpr_at_tpr(
                query_scores["aware_score"][mask & known_mask],
                query_scores["aware_score"][mask & role_masks["unknown"]],
                0.95,
            ),
        }

    by_role_tx_rows: list[dict[str, Any]] = []
    for role_name, role_mask in role_masks.items():
        for tx in sorted({str(v) for v in query_tx[role_mask].tolist()}):
            mask = role_mask & (query_tx == tx)
            row: dict[str, Any] = {"role": role_name, "tx_id": tx, "n": int(mask.sum())}
            for name in component_names:
                stats = _stats(query_scores[name][mask])
                for stat_key in ["mean", "q50", "q90", "q95"]:
                    row[f"{name}_{stat_key}"] = stats.get(stat_key)
            by_role_tx_rows.append(row)
    if args.output_by_role_tx_csv:
        _write_csv(args.output_by_role_tx_csv, by_role_tx_rows)

    result = {
        "schema": "phase2_open_set_geometry_audit_v1",
        "feature_npz": str(args.feature_npz),
        "target_unknown_eval_only": True,
        "threshold_scope": "source_old_plus_target_old_seen_support_and_proxy_unknown_only",
        "target_unknown_used_for_threshold": False,
        "k_shot": int(args.k_shot),
        "query_per_class": int(args.query_per_class),
        "qknn_k": int(args.qknn_k),
        "counts": {
            "source_old_train": len(plan.source_old_indices),
            "target_support": len(plan.support_indices),
            "proxy_unknown_train": len(plan.proxy_unknown_indices),
            "target_old_query": int(role_masks["old"].sum()),
            "target_seen_new_query": int(role_masks["seen_new"].sum()),
            "target_unknown_query": int(role_masks["unknown"].sum()),
        },
        "target_receivers": plan.target_receivers,
        "old_labels": plan.old_labels,
        "seen_new_labels": plan.seen_new_labels,
        "unknown_labels": plan.unknown_labels,
        "component_summary": component_summary,
        "per_unknown_tx": per_unknown_tx,
        "per_receiver": per_receiver,
        "by_role_tx_rows": by_role_tx_rows,
        "run_command_argv": [str(v) for v in sys.argv],
        "run_cwd": str(Path.cwd()),
    }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", type=Path, required=True)
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--output_by_role_tx_csv", type=Path)
    p.add_argument("--k_shot", type=int, default=8)
    p.add_argument("--query_per_class", type=int, default=12)
    p.add_argument("--qknn_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=4070705)
    p.add_argument("--support_selection_policy", default="stable_first", choices=["stable_first", "centroid", "scenario_diverse", "strict_event_query_preserve"])
    p.add_argument("--component_known_quantile", type=float, default=0.95)
    p.add_argument("--proto_weight", type=float, default=0.35)
    p.add_argument("--knn_weight", type=float, default=0.30)
    p.add_argument("--maha_weight", type=float, default=0.20)
    p.add_argument("--entropy_weight", type=float, default=0.10)
    p.add_argument("--proxy_weight", type=float, default=0.05)
    p.add_argument("--maha_var_floor", type=float, default=1e-3)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    result = run_geometry_audit(args)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    best = {
        name: {
            "auroc_unknown_vs_known": result["component_summary"][name]["auroc_unknown_vs_known"],
            "fpr95_known_as_unknown": result["component_summary"][name]["fpr95_known_as_unknown"],
            "oracle_far_05": result["component_summary"][name]["oracle_far_05_diagnostic"],
        }
        for name in ["aware_score", "knn_dist", "maha", "proxy_gap"]
    }
    print(json.dumps({"counts": result["counts"], "selected_components": best}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
