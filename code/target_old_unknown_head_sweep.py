#!/usr/bin/env python
"""Stage2-B target-old head sweep with unknown eval-only rejection.

This is an old-first diagnostic. It fits lightweight old-class heads using only
target_old support samples, evaluates target_old query accuracy, and evaluates
target_unknown false accept rate without using unknown query samples for fitting
or threshold selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from target_old_only_finetune_diag import (
    FeaturePayload,
    class_order,
    iter_feature_paths,
    l2_normalize,
    label_indices,
    load_payload,
    parse_int_list,
    parse_str_list,
    split_target_old,
)


def parse_float_list(text: str) -> list[float]:
    vals = [float(item.strip()) for item in str(text or "").split(",") if item.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one float")
    return vals


def stable_method_name(prefix: str, value: float | int) -> str:
    return f"{prefix}_{value:g}"


def top_margin(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.argmax(scores, axis=1)
    if scores.shape[1] == 1:
        conf = scores[:, 0]
        margin = np.full(scores.shape[0], np.inf, dtype=np.float64)
        return pred, conf, margin
    part = np.partition(scores, -2, axis=1)
    top1 = part[:, -1]
    top2 = part[:, -2]
    return pred, top1, top1 - top2


def accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(pred == truth)) if truth.size else float("nan")


def safe_mean(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if mask.size else float("nan")


def harmonic(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0 or b <= 0.0:
        return 0.0
    return float(2.0 * a * b / (a + b))


def prototype_matrix(train_x: np.ndarray, train_y: np.ndarray, classes: list[str]) -> np.ndarray:
    x = l2_normalize(train_x)
    centroids: list[np.ndarray] = []
    for label in classes:
        mask = train_y == label
        if not np.any(mask):
            centroids.append(np.zeros(x.shape[1], dtype=np.float64))
            continue
        c = np.mean(x[mask], axis=0)
        c = c / max(float(np.linalg.norm(c)), 1.0e-12)
        centroids.append(c)
    return np.vstack(centroids)


def prototype_scores(query_x: np.ndarray, proto: np.ndarray) -> np.ndarray:
    return l2_normalize(query_x) @ proto.T


def shrink_prototypes(
    *,
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    classes: list[str],
    target_weight: float,
) -> np.ndarray:
    target_proto = prototype_matrix(target_x, target_y, classes)
    if source_x.size == 0:
        return target_proto
    source_proto = prototype_matrix(source_x, source_y, classes)
    proto = float(target_weight) * target_proto + (1.0 - float(target_weight)) * source_proto
    norm = np.linalg.norm(proto, axis=1, keepdims=True)
    return proto / np.maximum(norm, 1.0e-12)


def knn_class_scores(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, classes: list[str], k: int) -> np.ndarray:
    train_x = l2_normalize(train_x)
    query_x = l2_normalize(query_x)
    sim = query_x @ train_x.T
    out = np.empty((query_x.shape[0], len(classes)), dtype=np.float64)
    for i, label in enumerate(classes):
        idx = np.flatnonzero(train_y == label)
        if idx.size == 0:
            out[:, i] = -1.0
            continue
        kk = min(int(k), int(idx.size))
        cls_sim = sim[:, idx]
        part = np.partition(cls_sim, -kk, axis=1)[:, -kk:]
        out[:, i] = np.mean(part, axis=1)
    return out


@dataclass
class RidgeModel:
    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def fit_ridge(train_x: np.ndarray, train_y_idx: np.ndarray, num_classes: int, ridge: float) -> RidgeModel:
    x = l2_normalize(train_x)
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.maximum(np.std(x, axis=0, keepdims=True), 1.0e-6)
    x = (x - mean) / std
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    y = np.eye(num_classes, dtype=np.float64)[train_y_idx]
    reg = float(ridge) * np.eye(x_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
    return RidgeModel(weights=weights, mean=mean, std=std)


def ridge_scores(model: RidgeModel, query_x: np.ndarray) -> np.ndarray:
    x = l2_normalize(query_x)
    x = (x - model.mean) / model.std
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    return x_aug @ model.weights


def diag_lda_scores(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, classes: list[str], shrink: float) -> np.ndarray:
    x = l2_normalize(train_x)
    q = l2_normalize(query_x)
    means = []
    variances = []
    global_var = np.var(x, axis=0) + 1.0e-5
    for label in classes:
        cls = x[train_y == label]
        mu = np.mean(cls, axis=0)
        var = np.var(cls, axis=0) + 1.0e-5
        var = float(shrink) * global_var + (1.0 - float(shrink)) * var
        means.append(mu)
        variances.append(var)
    scores = []
    for mu, var in zip(means, variances):
        scores.append(-0.5 * np.sum(((q - mu) ** 2) / var + np.log(var), axis=1))
    return np.vstack(scores).T


def validate_role_metadata(
    payload: FeaturePayload,
    role: str,
    mask: np.ndarray,
    *,
    expected_receivers: set[str] | None,
    channel_view: str | None,
    allowed_scenarios: set[str] | None,
) -> None:
    if not np.any(mask):
        return
    if expected_receivers:
        if payload.rx_ids is None:
            raise ValueError(f"{payload.candidate_id}: rx_ids missing for {role}")
        observed = {str(v) for v in payload.rx_ids[mask].tolist()}
        if observed - expected_receivers:
            raise ValueError(f"{payload.candidate_id}: {role} receivers {sorted(observed)} not in {sorted(expected_receivers)}")
    if channel_view:
        if payload.channel_views is None:
            raise ValueError(f"{payload.candidate_id}: channel_views missing for {role}")
        observed = {str(v) for v in payload.channel_views[mask].tolist()}
        if observed != {channel_view}:
            raise ValueError(f"{payload.candidate_id}: {role} channel_views {sorted(observed)} != {channel_view}")
    if allowed_scenarios:
        if payload.sat_scenarios is None:
            raise ValueError(f"{payload.candidate_id}: sat_scenarios missing for {role}")
        observed = {str(v) for v in payload.sat_scenarios[mask].tolist()}
        if observed - allowed_scenarios:
            raise ValueError(f"{payload.candidate_id}: {role} sat_scenarios {sorted(observed)} not in {sorted(allowed_scenarios)}")


def threshold_configs(support_conf: np.ndarray, support_margin: np.ndarray, quantiles: list[float]) -> list[tuple[str, str, float]]:
    configs: list[tuple[str, str, float]] = [("none", "none", -math.inf)]
    for q in quantiles:
        configs.append((f"conf_q{q:g}", "confidence", float(np.quantile(support_conf, q))))
        finite_margin = support_margin[np.isfinite(support_margin)]
        margin_source = finite_margin if finite_margin.size else support_margin
        configs.append((f"margin_q{q:g}", "margin", float(np.quantile(margin_source, q))))
    return configs


def evaluate_scores(
    *,
    base: dict[str, object],
    method: str,
    old_scores: np.ndarray,
    unknown_scores: np.ndarray,
    support_scores: np.ndarray,
    old_truth: np.ndarray,
    quantiles: list[float],
) -> list[dict[str, object]]:
    old_pred, old_conf, old_margin = top_margin(old_scores)
    unknown_pred, unknown_conf, unknown_margin = top_margin(unknown_scores)
    _support_pred, support_conf, support_margin = top_margin(support_scores)
    out: list[dict[str, object]] = []
    old_correct = old_pred == old_truth
    for threshold_name, score_kind, threshold in threshold_configs(support_conf, support_margin, quantiles):
        if score_kind == "none":
            old_accept = np.ones(old_pred.shape[0], dtype=bool)
            unknown_accept = np.ones(unknown_pred.shape[0], dtype=bool)
        elif score_kind == "confidence":
            old_accept = old_conf >= threshold
            unknown_accept = unknown_conf >= threshold
        else:
            old_accept = old_margin >= threshold
            unknown_accept = unknown_margin >= threshold
        old_accept_rate = safe_mean(old_accept)
        old_correct_accept = safe_mean(old_accept & old_correct)
        unknown_far = safe_mean(unknown_accept)
        unknown_rejection = 1.0 - unknown_far
        accepted_acc = float(np.mean(old_correct[old_accept])) if np.any(old_accept) else float("nan")
        out.append(
            {
                **base,
                "method": method,
                "threshold_name": threshold_name,
                "threshold_score_kind": score_kind,
                "threshold_value": threshold,
                "target_old_full_accuracy": accuracy(old_pred, old_truth),
                "target_old_accept_rate": old_accept_rate,
                "target_old_correct_accept_rate": old_correct_accept,
                "target_old_accepted_accuracy": accepted_acc,
                "unknown_false_accept_rate": unknown_far,
                "unknown_rejection_rate": unknown_rejection,
                "old_unknown_hmean": harmonic(old_correct_accept, unknown_rejection),
            }
        )
    return out


def evaluate_payload(
    payload: FeaturePayload,
    *,
    train_grid: list[int],
    seeds: list[int],
    ridge_values: list[float],
    shrink_weights: list[float],
    knn_values: list[int],
    lda_shrink_values: list[float],
    threshold_quantiles: list[float],
    expected_target_receivers: set[str] | None,
    require_target_channel_view: str | None,
    allowed_target_sat_scenarios: set[str] | None,
) -> list[dict[str, object]]:
    source_mask = payload.roles == "source"
    target_mask = payload.roles == "target_old"
    unknown_mask = payload.roles == "target_unknown"
    if not np.any(target_mask) or not np.any(unknown_mask):
        return []
    validate_role_metadata(
        payload,
        "target_old",
        target_mask,
        expected_receivers=expected_target_receivers,
        channel_view=require_target_channel_view,
        allowed_scenarios=allowed_target_sat_scenarios,
    )
    validate_role_metadata(
        payload,
        "target_unknown",
        unknown_mask,
        expected_receivers=expected_target_receivers,
        channel_view=require_target_channel_view,
        allowed_scenarios=allowed_target_sat_scenarios,
    )
    target_x_all = payload.features[target_mask]
    target_y_all = payload.tx_ids[target_mask]
    unknown_x = payload.features[unknown_mask]
    classes = class_order(target_y_all)
    source_x = payload.features[source_mask]
    source_y = payload.tx_ids[source_mask]
    rows: list[dict[str, object]] = []
    for train_per_tx in train_grid:
        for seed in seeds:
            support_idx, query_idx = split_target_old(target_y_all, int(train_per_tx), int(seed))
            if support_idx.size == 0 or query_idx.size == 0:
                continue
            support_x = target_x_all[support_idx]
            support_y = target_y_all[support_idx]
            query_x = target_x_all[query_idx]
            query_y = target_y_all[query_idx]
            truth = label_indices(query_y, classes)
            support_truth = label_indices(support_y, classes)
            base = {
                "candidate_id": payload.candidate_id,
                "train_per_tx": int(train_per_tx),
                "seed": int(seed),
                "class_count": len(classes),
                "support_count": int(support_idx.size),
                "target_old_query_count": int(query_idx.size),
                "unknown_query_count": int(np.sum(unknown_mask)),
                "threshold_calibration_scope": "target_old_support_only_no_unknown_query_fit",
            }
            method_scores: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
            if np.any(source_mask):
                proto = prototype_matrix(source_x, source_y, classes)
                method_scores.append(
                    (
                        "source_proto_frozen",
                        prototype_scores(query_x, proto),
                        prototype_scores(unknown_x, proto),
                        prototype_scores(support_x, proto),
                    )
                )
            target_proto = prototype_matrix(support_x, support_y, classes)
            method_scores.append(
                (
                    "target_proto",
                    prototype_scores(query_x, target_proto),
                    prototype_scores(unknown_x, target_proto),
                    prototype_scores(support_x, target_proto),
                )
            )
            for weight in shrink_weights:
                proto = shrink_prototypes(
                    source_x=source_x,
                    source_y=source_y,
                    target_x=support_x,
                    target_y=support_y,
                    classes=classes,
                    target_weight=float(weight),
                )
                method_scores.append(
                    (
                        stable_method_name("proto_shrink_targetw", float(weight)),
                        prototype_scores(query_x, proto),
                        prototype_scores(unknown_x, proto),
                        prototype_scores(support_x, proto),
                    )
                )
            for k in knn_values:
                method_scores.append(
                    (
                        stable_method_name("target_knn", int(k)),
                        knn_class_scores(support_x, support_y, query_x, classes, int(k)),
                        knn_class_scores(support_x, support_y, unknown_x, classes, int(k)),
                        knn_class_scores(support_x, support_y, support_x, classes, int(k)),
                    )
                )
            for ridge in ridge_values:
                model = fit_ridge(support_x, support_truth, len(classes), float(ridge))
                method_scores.append(
                    (
                        stable_method_name("target_ridge", float(ridge)),
                        ridge_scores(model, query_x),
                        ridge_scores(model, unknown_x),
                        ridge_scores(model, support_x),
                    )
                )
            for shrink in lda_shrink_values:
                method_scores.append(
                    (
                        stable_method_name("target_diaglda_shrink", float(shrink)),
                        diag_lda_scores(support_x, support_y, query_x, classes, float(shrink)),
                        diag_lda_scores(support_x, support_y, unknown_x, classes, float(shrink)),
                        diag_lda_scores(support_x, support_y, support_x, classes, float(shrink)),
                    )
                )
            for method, old_scores, unknown_scores, support_scores in method_scores:
                rows.extend(
                    evaluate_scores(
                        base=base,
                        method=method,
                        old_scores=old_scores,
                        unknown_scores=unknown_scores,
                        support_scores=support_scores,
                        old_truth=truth,
                        quantiles=threshold_quantiles,
                    )
                )
    return rows


def summarize(rows: list[dict[str, object]], metadata: dict[str, object]) -> dict[str, object]:
    groups: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["method"]), int(row["train_per_tx"]), str(row["threshold_name"])),
            [],
        ).append(row)
    by_method_k_threshold = []
    for (method, train_per_tx, threshold_name), vals in sorted(groups.items()):
        old_full = np.asarray([float(v["target_old_full_accuracy"]) for v in vals], dtype=np.float64)
        old_accept = np.asarray([float(v["target_old_correct_accept_rate"]) for v in vals], dtype=np.float64)
        far = np.asarray([float(v["unknown_false_accept_rate"]) for v in vals], dtype=np.float64)
        hmean = np.asarray([float(v["old_unknown_hmean"]) for v in vals], dtype=np.float64)
        by_method_k_threshold.append(
            {
                "method": method,
                "train_per_tx": train_per_tx,
                "threshold_name": threshold_name,
                "row_count": len(vals),
                "target_old_full_accuracy_mean": float(np.mean(old_full)),
                "target_old_full_accuracy_max": float(np.max(old_full)),
                "target_old_correct_accept_rate_mean": float(np.mean(old_accept)),
                "target_old_correct_accept_rate_max": float(np.max(old_accept)),
                "unknown_false_accept_rate_mean": float(np.mean(far)),
                "unknown_false_accept_rate_min": float(np.min(far)),
                "old_unknown_hmean_mean": float(np.mean(hmean)),
                "old_unknown_hmean_max": float(np.max(hmean)),
            }
        )
    best_by_old = sorted(rows, key=lambda r: float(r["target_old_full_accuracy"]), reverse=True)[:30]
    best_by_hmean = sorted(rows, key=lambda r: float(r["old_unknown_hmean"]), reverse=True)[:30]
    old80 = [r for r in rows if float(r["target_old_full_accuracy"]) >= 0.80]
    return {
        "schema": "stage2b_target_old_unknown_head_sweep_summary_v1",
        "diagnostic_type": "STAGE2B_OLD_FIRST_HEAD_SWEEP_UNKNOWN_EVAL_ONLY",
        "status": "COMPLETED_DIAGNOSTIC" if rows else "NO_ROWS",
        "row_count": len(rows),
        "candidate_count": len({str(r["candidate_id"]) for r in rows}),
        "old80_row_count": len(old80),
        "old80_candidate_count": len({str(r["candidate_id"]) for r in old80}),
        "by_method_k_threshold": by_method_k_threshold,
        "best_method_k_threshold_by_old_mean": sorted(
            by_method_k_threshold,
            key=lambda r: float(r["target_old_full_accuracy_mean"]),
            reverse=True,
        )[:15],
        "best_method_k_threshold_by_hmean": sorted(
            by_method_k_threshold,
            key=lambda r: float(r["old_unknown_hmean_mean"]),
            reverse=True,
        )[:15],
        "best_rows_by_old": best_by_old,
        "best_rows_by_hmean": best_by_hmean,
        "claim_boundary": (
            "Stage2-B diagnostic: target_old support fits old heads; target_unknown is eval-only; "
            "no target_new support/query and no Stage2-C or deployment-success claim"
        ),
        **metadata,
    }


def write_outputs(rows: list[dict[str, object]], output_dir: Path, metadata: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "method",
        "threshold_name",
        "threshold_score_kind",
        "threshold_value",
        "train_per_tx",
        "seed",
        "class_count",
        "support_count",
        "target_old_query_count",
        "unknown_query_count",
        "target_old_full_accuracy",
        "target_old_accept_rate",
        "target_old_correct_accept_rate",
        "target_old_accepted_accuracy",
        "unknown_false_accept_rate",
        "unknown_rejection_rate",
        "old_unknown_hmean",
        "threshold_calibration_scope",
    ]
    with (output_dir / "old_unknown_head_sweep_result_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    (output_dir / "old_unknown_head_sweep_summary.json").write_text(
        json.dumps(summarize(rows, metadata), indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-tx-grid", type=parse_int_list, default=[5, 10, 20, 40, 50])
    parser.add_argument("--seeds", type=parse_int_list, default=[213920])
    parser.add_argument("--ridge-values", type=parse_float_list, default=[0.1, 1.0, 10.0])
    parser.add_argument("--shrink-target-weights", type=parse_float_list, default=[0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--knn-values", type=parse_int_list, default=[1, 3, 5])
    parser.add_argument("--diaglda-shrink-values", type=parse_float_list, default=[0.25, 0.5, 0.75])
    parser.add_argument("--threshold-quantiles", type=parse_float_list, default=[0.01, 0.05, 0.1, 0.2])
    parser.add_argument("--expected-target-receivers", type=parse_str_list, default=[])
    parser.add_argument("--require-target-channel-view", default="")
    parser.add_argument("--allowed-target-sat-scenarios", type=parse_str_list, default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for path in iter_feature_paths(args.input_run_root):
        rows.extend(
            evaluate_payload(
                load_payload(path),
                train_grid=list(args.train_per_tx_grid),
                seeds=list(args.seeds),
                ridge_values=list(args.ridge_values),
                shrink_weights=list(args.shrink_target_weights),
                knn_values=list(args.knn_values),
                lda_shrink_values=list(args.diaglda_shrink_values),
                threshold_quantiles=list(args.threshold_quantiles),
                expected_target_receivers=set(args.expected_target_receivers) or None,
                require_target_channel_view=str(args.require_target_channel_view or "").strip() or None,
                allowed_target_sat_scenarios=set(args.allowed_target_sat_scenarios) or None,
            )
        )
    if not rows:
        raise SystemExit(f"no Stage2-B old/unknown rows produced from {args.input_run_root}")
    metadata = {
        "run_id": args.output_dir.name,
        "input_run_root": str(args.input_run_root),
        "train_per_tx_grid": list(args.train_per_tx_grid),
        "seeds": list(args.seeds),
        "ridge_values": list(args.ridge_values),
        "shrink_target_weights": list(args.shrink_target_weights),
        "knn_values": list(args.knn_values),
        "diaglda_shrink_values": list(args.diaglda_shrink_values),
        "threshold_quantiles": list(args.threshold_quantiles),
        "expected_target_receivers": list(args.expected_target_receivers),
        "require_target_channel_view": str(args.require_target_channel_view or "").strip(),
        "allowed_target_sat_scenarios": list(args.allowed_target_sat_scenarios),
        "threshold_selection": "target_old_support_only_no_unknown_query_fit",
    }
    write_outputs(rows, args.output_dir, metadata)
    summary = summarize(rows, metadata)
    print(
        json.dumps(
            {
                "rows": summary["row_count"],
                "candidate_count": summary["candidate_count"],
                "old80_row_count": summary["old80_row_count"],
                "old80_candidate_count": summary["old80_candidate_count"],
                "best_old_mean": summary["best_method_k_threshold_by_old_mean"][:5],
                "best_hmean": summary["best_method_k_threshold_by_hmean"][:5],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
