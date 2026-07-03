#!/usr/bin/env python3
"""Fast Stage2-C sweep for source-guarded int8 qKNN heads.

The evaluated head stores quantized support embeddings and optional class
radius statistics, but not raw support IQ samples or full-precision support
embeddings. Source logits are used only as a lightweight old-class guard.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**32)


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_csv(value: str | None) -> list[float]:
    return [float(part) for part in _parse_csv(value)]


def _parse_int_csv(value: str | None) -> list[int]:
    return [int(part) for part in _parse_csv(value)]


def _accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    return float(np.mean(pred == truth))


def _split_indices(
    tx_ids: np.ndarray,
    roles: np.ndarray,
    tx: str,
    role: str,
    support_count: int,
    query_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    available = np.where((tx_ids == tx) & (roles == role))[0]
    required = int(support_count) + int(query_count)
    if available.size < required:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    rng = np.random.default_rng(_stable_seed(seed, f"{role}:{tx}"))
    shuffled = available[rng.permutation(available.size)]
    return shuffled[:support_count].astype(int), shuffled[support_count : support_count + query_count].astype(int)


def _classwise_topk_predict(scores: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), int(scores.shape[1])))
    top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    pred: list[str] = []
    for row_idx, candidates in enumerate(top_idx):
        class_scores: dict[str, float] = {}
        for candidate in candidates.tolist():
            label = str(labels[candidate])
            class_scores[label] = class_scores.get(label, 0.0) + float(scores[row_idx, candidate])
        pred.append(max(class_scores.items(), key=lambda item: (item[1], item[0]))[0])
    return np.asarray(pred, dtype=object)


def _softmax_confidence(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)
    top2 = np.partition(logits, kth=-2, axis=1)[:, -2:]
    margins = top2[:, 1] - top2[:, 0]
    return np.max(probs, axis=1), margins


def _prepare_class_splits(
    tx_ids: np.ndarray,
    roles: np.ndarray,
    labels: list[str],
    role: str,
    support_count: int,
    query_count: int,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label in labels:
        support, query = _split_indices(tx_ids, roles, label, role, support_count, query_count, seed)
        if support.size and query.size:
            splits[str(label)] = (support, query)
    return splits


def _build_support_bank(
    features: np.ndarray,
    support_indices: list[int],
    support_labels: list[str],
    old_labels: set[str],
    support_scenarios: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    support_features = _normalize_rows(features[np.asarray(support_indices, dtype=int)])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    scale = 127.0
    quantized = np.clip(np.rint(support_features * scale), -scale, scale).astype(np.int8)
    dequant = _normalize_rows(quantized.astype(np.float64) / scale)
    class_labels = sorted({str(label) for label in labels.tolist()})
    class_radii = []
    for label in class_labels:
        class_features = support_features[labels == label]
        prototype = _normalize_rows(class_features.mean(axis=0, keepdims=True))[0]
        class_radii.append(float(np.mean(1.0 - (class_features @ prototype))))
    radii_by_support = np.zeros(labels.shape[0], dtype=np.float64)
    for idx, label in enumerate(class_labels):
        radii_by_support[labels == label] = class_radii[idx]
    return {
        "features": dequant,
        "labels": labels,
        "is_old": np.asarray([str(label) in old_labels for label in labels], dtype=bool),
        "radii_by_support": radii_by_support,
        "class_labels": np.asarray(class_labels, dtype=object),
        "scenarios": None if support_scenarios is None else np.asarray(support_scenarios, dtype=object).astype(str),
    }


def _predict_from_bank(
    bank: dict[str, np.ndarray],
    query_features: np.ndarray,
    *,
    topk: int,
    old_bias: float,
    radius_norm: float,
    query_scenarios: np.ndarray | None = None,
    scenario_aware: bool = False,
) -> np.ndarray:
    query = _normalize_rows(query_features)
    if not scenario_aware:
        scores = query @ bank["features"].T
        if float(radius_norm) != 0.0:
            denom = np.power(np.maximum(bank["radii_by_support"], 1e-4), float(radius_norm))[None, :]
            scores = 1.0 - ((1.0 - scores) / denom)
        if float(old_bias) != 0.0:
            scores = scores + bank["is_old"].astype(np.float64)[None, :] * float(old_bias)
        return _classwise_topk_predict(scores, bank["labels"], topk)

    support_scenarios = bank.get("scenarios")
    if support_scenarios is None or query_scenarios is None:
        raise ValueError("scenario_aware requires support and query scenario labels")

    query_scenarios = np.asarray(query_scenarios, dtype=object).astype(str)
    pred = np.empty(query.shape[0], dtype=object)
    for scenario in sorted({str(value) for value in query_scenarios.tolist()}):
        query_mask = query_scenarios == scenario
        support_mask = support_scenarios == scenario
        # Fallback keeps the head usable when a K-shot support draw misses a scenario.
        if int(np.sum(support_mask)) < max(1, int(topk)) or len(set(bank["labels"][support_mask].tolist())) < 2:
            support_mask = np.ones_like(support_mask, dtype=bool)
        sub_bank = {
            "features": bank["features"][support_mask],
            "labels": bank["labels"][support_mask],
            "is_old": bank["is_old"][support_mask],
            "radii_by_support": bank["radii_by_support"][support_mask],
        }
        pred[query_mask] = _predict_from_bank(
            sub_bank,
            query[query_mask],
            topk=topk,
            old_bias=old_bias,
            radius_norm=radius_norm,
            scenario_aware=False,
        )
    return pred


def _evaluate_row(
    combo: tuple[str, str],
    *,
    features: np.ndarray,
    tx_ids: np.ndarray,
    source_label: np.ndarray,
    source_conf: np.ndarray,
    source_margin: np.ndarray,
    scenarios: np.ndarray,
    old_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    new_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    old_labels: list[str],
    topk: int,
    old_bias: float,
    radius_norm: float,
    source_guard_mode: str,
    source_conf_min: float,
    source_margin_min: float,
    scenario_aware: bool,
    old_target: float,
    old_floor: float,
    new_target: float,
    new_floor: float,
) -> dict[str, Any]:
    support_indices: list[int] = []
    support_labels: list[str] = []
    old_query_indices: list[int] = []
    new_query_indices: list[int] = []
    for label in old_labels:
        support, query = old_splits[label]
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
        old_query_indices.extend(query.tolist())
    for label in combo:
        support, query = new_splits[label]
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
        new_query_indices.extend(query.tolist())

    bank = _build_support_bank(
        features,
        support_indices,
        support_labels,
        set(old_labels),
        support_scenarios=scenarios[np.asarray(support_indices, dtype=int)] if scenario_aware else None,
    )
    query_idx = np.asarray(old_query_indices + new_query_indices, dtype=int)
    pred = _predict_from_bank(
        bank,
        features[query_idx],
        topk=topk,
        old_bias=old_bias,
        radius_norm=radius_norm,
        query_scenarios=scenarios[query_idx] if scenario_aware else None,
        scenario_aware=scenario_aware,
    )

    if source_guard_mode != "none":
        old_label_array = np.asarray(old_labels, dtype=object)
        pred_is_old = np.isin(pred, old_label_array)
        guard = (
            (source_conf[query_idx] >= float(source_conf_min))
            & (source_margin[query_idx] >= float(source_margin_min))
        )
        if source_guard_mode == "old_pred":
            guard = guard & pred_is_old
        elif source_guard_mode == "all":
            guard = guard
        else:
            raise ValueError(f"Unsupported source_guard_mode: {source_guard_mode}")
        pred[guard] = source_label[query_idx][guard]

    truth = tx_ids[query_idx]
    old_count = len(old_query_indices)
    old_pred = pred[:old_count]
    old_truth = truth[:old_count]
    new_pred = pred[old_count:]
    new_truth = truth[old_count:]
    per_old = {label: _accuracy(old_pred[old_truth == label], old_truth[old_truth == label]) for label in old_labels}
    per_new = {label: _accuracy(new_pred[new_truth == label], new_truth[new_truth == label]) for label in combo}
    old_acc = _accuracy(old_pred, old_truth)
    seen_new_acc = _accuracy(new_pred, new_truth)
    min_old_acc = min(per_old.values()) if per_old else 0.0
    min_new_acc = min(per_new.values()) if per_new else 0.0
    return {
        "new_tx_ids": list(combo),
        "method": (
            f"source_guarded_qknn8_k{topk}_oldbias{old_bias:g}_rnorm{radius_norm:g}"
            f"_sg{source_guard_mode}_c{source_conf_min:g}_m{source_margin_min:g}"
            f"{'_scenario' if scenario_aware else ''}"
        ),
        "old_acc": old_acc,
        "min_old_class_acc": min_old_acc,
        "seen_new_acc": seen_new_acc,
        "min_seen_new_class_acc": min_new_acc,
        "per_old_acc": per_old,
        "per_new_acc": per_new,
        "passes_joint_target": (
            old_acc >= old_target
            and min_old_acc >= old_floor
            and seen_new_acc >= new_target
            and min_new_acc >= new_floor
        ),
        "support_count": int(len(support_indices)),
        "old_query_count": int(len(old_query_indices)),
        "new_query_count": int(len(new_query_indices)),
        "stored_quantized_count": int(len(support_indices)),
        "stored_support_count": 0,
        "scenario_aware": bool(scenario_aware),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--candidate_new_tx_ids", default="")
    parser.add_argument("--combo_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=422001)
    parser.add_argument("--k_old", type=int, default=10)
    parser.add_argument("--k_new", type=int, default=10)
    parser.add_argument("--query_per_old", type=int, default=70)
    parser.add_argument("--query_per_new", type=int, default=70)
    parser.add_argument("--topk_grid", default="9")
    parser.add_argument("--old_bias_grid", default="0,0.005,0.01,0.02,0.04,0.08")
    parser.add_argument("--radius_norm_grid", default="0.3,0.4")
    parser.add_argument("--source_guard_modes", default="none")
    parser.add_argument("--source_conf_min_grid", default="0")
    parser.add_argument("--source_margin_min_grid", default="0")
    parser.add_argument("--scenario_aware_grid", default="false")
    parser.add_argument("--old_target", type=float, default=0.88)
    parser.add_argument("--old_floor", type=float, default=0.80)
    parser.add_argument("--seen_new_target", type=float, default=0.85)
    parser.add_argument("--seen_new_floor", type=float, default=0.80)
    args = parser.parse_args()

    data = np.load(Path(args.feature_npz), allow_pickle=True)
    features = _normalize_rows(data["features"])
    tx_ids = np.asarray(data["tx_ids"], dtype=object).astype(str)
    roles = np.asarray(data["dataset_role"], dtype=object).astype(str)
    logits = np.asarray(data["tx_logits"], dtype=np.float64)
    scenarios = np.asarray(data["sat_scenarios"], dtype=object).astype(str)
    old_labels = _parse_csv(args.old_tx_ids)
    old_label_array = np.asarray(old_labels, dtype=object)
    source_idx = np.argmax(logits, axis=1)
    source_label = old_label_array[source_idx]
    source_conf, source_margin = _softmax_confidence(logits)

    old_splits = _prepare_class_splits(
        tx_ids, roles, old_labels, "target_old", args.k_old, args.query_per_old, args.seed
    )
    if len(old_splits) != len(old_labels):
        missing = sorted(set(old_labels) - set(old_splits))
        raise RuntimeError(f"Missing old splits: {missing}")

    explicit_new = _parse_csv(args.candidate_new_tx_ids)
    if explicit_new:
        new_candidates = explicit_new
    else:
        new_candidates = sorted({str(label) for label in tx_ids[roles == "target_new"].tolist()})
    new_splits = _prepare_class_splits(
        tx_ids, roles, new_candidates, "target_new", args.k_new, args.query_per_new, args.seed
    )
    combos = list(itertools.combinations(sorted(new_splits), int(args.combo_size)))

    rows: list[dict[str, Any]] = []
    for combo in combos:
        for topk in _parse_int_csv(args.topk_grid):
            for old_bias in _parse_float_csv(args.old_bias_grid):
                for radius_norm in _parse_float_csv(args.radius_norm_grid):
                    for source_guard_mode in _parse_csv(args.source_guard_modes):
                        for source_conf_min in _parse_float_csv(args.source_conf_min_grid):
                            for source_margin_min in _parse_float_csv(args.source_margin_min_grid):
                                for scenario_aware_raw in _parse_csv(args.scenario_aware_grid):
                                    scenario_aware = str(scenario_aware_raw).lower() in {"1", "true", "yes", "y"}
                                    rows.append(
                                        _evaluate_row(
                                            tuple(combo),
                                            features=features,
                                            tx_ids=tx_ids,
                                            source_label=source_label,
                                            source_conf=source_conf,
                                            source_margin=source_margin,
                                            scenarios=scenarios,
                                            old_splits=old_splits,
                                            new_splits=new_splits,
                                            old_labels=old_labels,
                                            topk=topk,
                                            old_bias=old_bias,
                                            radius_norm=radius_norm,
                                            source_guard_mode=source_guard_mode,
                                            source_conf_min=source_conf_min,
                                            source_margin_min=source_margin_min,
                                            scenario_aware=scenario_aware,
                                            old_target=args.old_target,
                                            old_floor=args.old_floor,
                                            new_target=args.seen_new_target,
                                            new_floor=args.seen_new_floor,
                                        )
                                    )

    rows.sort(
        key=lambda row: (
            bool(row["passes_joint_target"]),
            min(
                row["old_acc"] / args.old_target,
                row["min_old_class_acc"] / args.old_floor,
                row["seen_new_acc"] / args.seen_new_target,
                row["min_seen_new_class_acc"] / args.seen_new_floor,
            ),
            row["old_acc"],
            row["seen_new_acc"],
        ),
        reverse=True,
    )

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "feature_npz": str(args.feature_npz),
        "old_tx_ids": old_labels,
        "eligible_new_tx_count": len(new_splits),
        "combo_size": int(args.combo_size),
        "k_old": int(args.k_old),
        "k_new": int(args.k_new),
        "query_per_old": int(args.query_per_old),
        "query_per_new": int(args.query_per_new),
        "old_target": float(args.old_target),
        "old_floor": float(args.old_floor),
        "seen_new_target": float(args.seen_new_target),
        "seen_new_floor": float(args.seen_new_floor),
        "scenario_aware_grid": _parse_csv(args.scenario_aware_grid),
        "joint_pass_count": int(sum(1 for row in rows if row["passes_joint_target"])),
        "combo_rows": rows,
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "method",
        "new_tx_ids",
        "old_acc",
        "min_old_class_acc",
        "seen_new_acc",
        "min_seen_new_class_acc",
        "passes_joint_target",
        "per_old_acc",
        "per_new_acc",
        "support_count",
        "stored_quantized_count",
        "stored_support_count",
        "scenario_aware",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key) for key in fieldnames}
            csv_row["new_tx_ids"] = ",".join(row["new_tx_ids"])
            csv_row["per_old_acc"] = json.dumps(row["per_old_acc"], ensure_ascii=False, sort_keys=True)
            csv_row["per_new_acc"] = json.dumps(row["per_new_acc"], ensure_ascii=False, sort_keys=True)
            writer.writerow(csv_row)

    print(
        json.dumps(
            {
                "eligible_new_tx_count": len(new_splits),
                "joint_pass_count": summary["joint_pass_count"],
                "best": rows[:10],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
