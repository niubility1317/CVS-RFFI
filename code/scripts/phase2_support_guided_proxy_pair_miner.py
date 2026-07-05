#!/usr/bin/env python3
"""Mine support-guided proxy hard pairs for Phase2 many-new qKNN repair.

This is a read-only miner. It uses target K-shot support labels and source /
proxy features, never target query labels. The output can feed
train_apply_phase1_iq_preadapter_20260703.py via --proxy_unknown_hard_pair_ids.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import phase2_qknn_active_support_select as active
import phase2_source_guarded_qknn_sweep as qknn


def _mean_proto(features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return qknn._normalize_rows(features[indices].mean(axis=0, keepdims=True))[0]


def _build_support_splits(
    *,
    features: np.ndarray,
    tx_ids: np.ndarray,
    roles: np.ndarray,
    scenarios: np.ndarray,
    logits: np.ndarray,
    labels: list[str],
    role: str,
    k: int,
    pool_per_class: int,
    policy: str,
    seed: int,
    exclude_pool_from_query: bool,
    old_labels: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    source_probs = active._softmax(logits)
    source_label_to_idx = {label: idx for idx, label in enumerate(old_labels)}
    source_prototypes: dict[str, np.ndarray] = {}
    for label in old_labels:
        source_idx = np.where((tx_ids == label) & (roles == "source"))[0].astype(int)
        if source_idx.size:
            source_prototypes[label] = _mean_proto(features, source_idx)
    raw = active._build_active_splits(
        tx_ids=tx_ids,
        roles=roles,
        features=features,
        scenarios=scenarios,
        source_probs=source_probs,
        source_label_to_idx=source_label_to_idx,
        source_prototypes=source_prototypes,
        labels=labels,
        role=role,
        k=int(k),
        query_per_class=1,
        pool_per_class=int(pool_per_class),
        policy=policy,
        seed=int(seed),
        exclude_pool_from_query=bool(exclude_pool_from_query),
    )
    return active._as_eval_splits(raw)


def _proxy_prototypes(features: np.ndarray, tx_ids: np.ndarray, roles: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for label in sorted({str(v) for v in tx_ids[roles == "proxy_unknown"].tolist()}):
        idx = np.where((tx_ids == label) & (roles == "proxy_unknown"))[0].astype(int)
        if idx.size:
            out[label] = _mean_proto(features, idx)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--new_tx_ids", required=True)
    parser.add_argument("--target_new_focus", default="10-10,2-13")
    parser.add_argument("--hard_old_focus", default="20-19,14-7")
    parser.add_argument("--hard_focus", default="")
    parser.add_argument("--new_role", default="target_unknown")
    parser.add_argument("--old_role", default="target_old")
    parser.add_argument("--seed", type=int, default=421029)
    parser.add_argument("--k_old", type=int, default=10)
    parser.add_argument("--k_new", type=int, default=10)
    parser.add_argument("--pool_per_old", type=int, default=10)
    parser.add_argument("--pool_per_new", type=int, default=10)
    parser.add_argument("--policy", default="stable_first")
    parser.add_argument("--exclude_pool_from_query", action="store_true")
    parser.add_argument("--top_proxy_per_focus", type=int, default=6)
    parser.add_argument("--top_pairs", type=int, default=12)
    args = parser.parse_args()

    data = np.load(Path(args.feature_npz), allow_pickle=True)
    features = qknn._normalize_rows(np.asarray(data["features"], dtype=np.float64))
    tx_ids = np.asarray(data["tx_ids"], dtype=object).astype(str)
    roles = np.asarray(data["dataset_role"], dtype=object).astype(str)
    scenarios = np.asarray(data["sat_scenarios"], dtype=object).astype(str)
    logits = np.asarray(data["tx_logits"], dtype=np.float64)
    old_labels = qknn._parse_csv(args.old_tx_ids)
    new_labels = qknn._parse_csv(args.new_tx_ids)
    focus_new = [label for label in qknn._parse_csv(args.target_new_focus) if label in set(new_labels)]
    all_class_labels = old_labels + new_labels
    hard_focus_raw = str(args.hard_focus).strip() or str(args.hard_old_focus)
    focus_hard = [label for label in qknn._parse_csv(hard_focus_raw) if label in set(all_class_labels)]

    old_splits = _build_support_splits(
        features=features,
        tx_ids=tx_ids,
        roles=roles,
        scenarios=scenarios,
        logits=logits,
        labels=old_labels,
        role=str(args.old_role),
        k=int(args.k_old),
        pool_per_class=int(args.pool_per_old),
        policy=str(args.policy),
        seed=int(args.seed),
        exclude_pool_from_query=bool(args.exclude_pool_from_query),
        old_labels=old_labels,
    )
    new_splits = _build_support_splits(
        features=features,
        tx_ids=tx_ids,
        roles=roles,
        scenarios=scenarios,
        logits=logits,
        labels=new_labels,
        role=str(args.new_role),
        k=int(args.k_new),
        pool_per_class=int(args.pool_per_new),
        policy=str(args.policy),
        seed=int(args.seed),
        exclude_pool_from_query=bool(args.exclude_pool_from_query),
        old_labels=old_labels,
    )

    support_proto: dict[str, np.ndarray] = {}
    for label, splits in {**old_splits, **new_splits}.items():
        support_proto[label] = _mean_proto(features, splits[0])
    proxy_proto = _proxy_prototypes(features, tx_ids, roles)
    proxy_labels = sorted(proxy_proto)
    proxy_matrix = np.stack([proxy_proto[label] for label in proxy_labels], axis=0)

    candidate_rows: list[dict[str, Any]] = []
    selected_pair_items: list[str] = []
    for new_label in focus_new:
        new_proto = support_proto[new_label]
        proxy_new_sim = proxy_matrix @ new_proto
        near_new_order = np.argsort(-proxy_new_sim)[: int(args.top_proxy_per_focus)]
        for hard_label in focus_hard:
            if hard_label == new_label or hard_label not in support_proto:
                continue
            hard_proto = support_proto[hard_label]
            old_index = old_labels.index(hard_label) if hard_label in old_labels else -1
            proxy_hard_sim = proxy_matrix @ hard_proto
            near_hard_order = np.argsort(-proxy_hard_sim)[: int(args.top_proxy_per_focus)]
            pair_rows: list[dict[str, Any]] = []
            for left_i in near_new_order.tolist():
                for right_i in near_hard_order.tolist():
                    if left_i == right_i:
                        continue
                    left_label = proxy_labels[left_i]
                    right_label = proxy_labels[right_i]
                    proxy_pair_sim = float(proxy_proto[left_label] @ proxy_proto[right_label])
                    score = (
                        float(proxy_new_sim[left_i])
                        + float(proxy_hard_sim[right_i])
                        + 0.5 * proxy_pair_sim
                        - 0.25 * abs(float(proxy_new_sim[right_i]) - float(proxy_hard_sim[left_i]))
                    )
                    pair_rows.append(
                        {
                            "target_new": new_label,
                            "hard_old": hard_label,
                            "hard_label": hard_label,
                            "old_index": int(old_index),
                            "left_proxy": left_label,
                            "right_proxy": right_label,
                            "left_to_target_new": float(proxy_new_sim[left_i]),
                            "right_to_hard_old": float(proxy_hard_sim[right_i]),
                            "right_to_hard_label": float(proxy_hard_sim[right_i]),
                            "proxy_pair_sim": proxy_pair_sim,
                            "analogy_score": float(score),
                        }
                    )
            pair_rows.sort(key=lambda row: row["analogy_score"], reverse=True)
            for row in pair_rows[: int(args.top_pairs)]:
                candidate_rows.append(row)
                selected_pair_items.append(f"{row['left_proxy']}:{row['right_proxy']}")
                if int(row["old_index"]) >= 0:
                    selected_pair_items.append(f"{row['left_proxy']}:{row['old_index']}")

    deduped: list[str] = []
    for item in selected_pair_items:
        if item not in deduped:
            deduped.append(item)

    manifest = {
        "method": "support_guided_proxy_pair_miner_v1",
        "feature_npz": str(args.feature_npz),
        "uses_target_query_labels": False,
        "support_selection": {
            "seed": int(args.seed),
            "policy": str(args.policy),
            "k_old": int(args.k_old),
            "k_new": int(args.k_new),
            "pool_per_old": int(args.pool_per_old),
            "pool_per_new": int(args.pool_per_new),
            "exclude_pool_from_query": bool(args.exclude_pool_from_query),
        },
        "old_tx_ids": old_labels,
        "new_tx_ids": new_labels,
        "target_new_focus": focus_new,
        "hard_old_focus": [label for label in focus_hard if label in set(old_labels)],
        "hard_focus": focus_hard,
        "selected_proxy_hard_pair_ids": deduped,
        "selected_proxy_hard_pair_ids_csv": ",".join(deduped),
        "candidate_rows": candidate_rows,
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "target_new",
        "hard_old",
        "hard_label",
        "old_index",
        "left_proxy",
        "right_proxy",
        "left_to_target_new",
        "right_to_hard_old",
        "right_to_hard_label",
        "proxy_pair_sim",
        "analogy_score",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow({key: row.get(key) for key in fields})
    print(json.dumps({"selected_proxy_hard_pair_ids_csv": manifest["selected_proxy_hard_pair_ids_csv"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
