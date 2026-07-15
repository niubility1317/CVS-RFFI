#!/usr/bin/env python
"""Audit a locked Stage2-C matrix and compute cross-K forgetting gates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from paper_reproduction.scripts.build_cvs_stage2c_candidate_lock import (
    FORMAL_K,
    FORMAL_NEW_COUNTS,
    FORMAL_RECEIVERS,
)


FORMAL_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
NEW_ACCURACY_TARGET = {5: 0.92, 10: 0.90, 20: 0.86}
K10_OLD_TARGET = 0.95
K10_MIN_OLD_CLASS_TARGET = 0.88
K5_MAX_DROP = 0.03


def _float(row: dict[str, Any], key: str) -> float:
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"non-finite metric {key}")
    return value


def _ids(row: dict[str, Any], key: str) -> tuple[str, ...]:
    payload = row[key]
    values = json.loads(payload) if isinstance(payload, str) else payload
    result = tuple(str(value) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate IDs in {key}")
    return result


def validate_nested_protocol(
    rows: Sequence[dict[str, Any]],
    *,
    expected_receivers: Sequence[str] = FORMAL_RECEIVERS,
    expected_scenarios: Sequence[str] = FORMAL_SCENARIOS,
    expected_new_counts: Sequence[int] = FORMAL_NEW_COUNTS,
    expected_k: Sequence[int] = FORMAL_K,
    minimum_seeds: int = 5,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("locked matrix is empty")
    candidate_ids = {str(row["candidate_id"]) for row in rows}
    lock_hashes = {str(row["candidate_lock_sha256"]) for row in rows}
    if len(candidate_ids) != 1 or len(lock_hashes) != 1:
        raise ValueError("matrix mixes candidates or candidate locks")
    receivers = sorted({str(row["receiver"]) for row in rows})
    scenarios = sorted({str(row["scenario"]) for row in rows})
    new_counts = sorted({int(row["new_class_count"]) for row in rows})
    k_values = sorted({int(row["k_shot"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    if set(receivers) != set(str(value) for value in expected_receivers):
        raise ValueError("target receiver coverage mismatch")
    if set(scenarios) != set(str(value) for value in expected_scenarios):
        raise ValueError("leo_weak scenario coverage mismatch")
    if new_counts != sorted(int(value) for value in expected_new_counts):
        raise ValueError("new-class-count coverage mismatch")
    if k_values != sorted(int(value) for value in expected_k):
        raise ValueError("K coverage mismatch")
    if len(seeds) < int(minimum_seeds):
        raise ValueError("insufficient independent confirmation seeds")
    index: dict[tuple[str, int, str, int, int], dict[str, Any]] = {}
    reference_old_labels: tuple[str, ...] | None = None
    reference_new_labels: dict[int, tuple[str, ...]] = {}
    for row in rows:
        key = (
            str(row["receiver"]),
            int(row["seed"]),
            str(row["scenario"]),
            int(row["new_class_count"]),
            int(row["k_shot"]),
        )
        if key in index:
            raise ValueError(f"duplicate formal matrix row: {key}")
        index[key] = row
        support = _ids(row, "support_ids_json")
        query = _ids(row, "query_ids_json")
        class_count = int(row["registered_class_count"])
        query_per_tx = int(row["query_per_tx"])
        old_labels = _ids(row, "old_tx_labels_json")
        new_labels = _ids(row, "new_tx_labels_json")
        if reference_old_labels is None:
            reference_old_labels = old_labels
        elif old_labels != reference_old_labels:
            raise ValueError("target-old TX labels drift across matrix rows")
        new_count = int(row["new_class_count"])
        if len(new_labels) != new_count or set(old_labels) & set(new_labels):
            raise ValueError("invalid old/new TX class split")
        if new_count not in reference_new_labels:
            reference_new_labels[new_count] = new_labels
        elif new_labels != reference_new_labels[new_count]:
            raise ValueError("target-new TX labels drift at fixed class count")
        if class_count != len(old_labels) + len(new_labels):
            raise ValueError("registered class count differs from locked TX labels")
        if str(row.get("support_query_view", "")) != "leo_weak_only" or int(
            row.get("clean_support_query_rows", -1)
        ) != 0:
            raise ValueError("formal matrix contains non-leo_weak or clean support/query")
        if len(support) != class_count * int(row["k_shot"]):
            raise ValueError(f"support cardinality drift: {key}")
        if len(query) != class_count * query_per_tx:
            raise ValueError(f"query cardinality drift: {key}")
        if set(support) & set(query):
            raise ValueError(f"support/query overlap: {key}")
    expected_count = (
        len(expected_receivers)
        * len(seeds)
        * len(expected_scenarios)
        * len(expected_new_counts)
        * len(expected_k)
    )
    if len(index) != expected_count:
        raise ValueError(f"matrix is incomplete: {len(index)}!={expected_count}")
    ordered_counts = sorted(int(value) for value in expected_new_counts)
    for lower, upper in zip(ordered_counts, ordered_counts[1:]):
        if reference_new_labels[upper][:lower] != reference_new_labels[lower]:
            raise ValueError(f"new-{lower} TX labels are not a prefix of new-{upper}")

    # The same physical sample IDs are reused across K and the three registered
    # leo_weak transforms.  K support sets must be exact prefixes/subsets and
    # query identities must remain identical.
    for receiver in expected_receivers:
        for seed in seeds:
            for new_count in expected_new_counts:
                reference_query: tuple[str, ...] | None = None
                support_by_k: dict[int, set[str]] = {}
                for scenario in expected_scenarios:
                    for k_shot in expected_k:
                        row = index[
                            (
                                str(receiver),
                                int(seed),
                                str(scenario),
                                int(new_count),
                                int(k_shot),
                            )
                        ]
                        query = _ids(row, "query_ids_json")
                        if reference_query is None:
                            reference_query = query
                        elif query != reference_query:
                            raise ValueError("query IDs drift across K/scenario")
                        support = set(_ids(row, "support_ids_json"))
                        if scenario == expected_scenarios[0]:
                            support_by_k[int(k_shot)] = support
                        elif support != support_by_k[int(k_shot)]:
                            raise ValueError("support IDs drift across scenarios")
                ordered_k = sorted(int(value) for value in expected_k)
                for lower, upper in zip(ordered_k, ordered_k[1:]):
                    if not support_by_k[lower] < support_by_k[upper]:
                        raise ValueError(f"support K{lower} is not nested in K{upper}")
    return {
        "candidate_id": next(iter(candidate_ids)),
        "candidate_lock_sha256": next(iter(lock_hashes)),
        "row_count": int(len(rows)),
        "receivers": receivers,
        "seeds": seeds,
        "scenarios": scenarios,
        "new_class_counts": new_counts,
        "k_values": k_values,
        "target_old_tx_labels": list(reference_old_labels or ()),
        "nested_target_new_tx_labels": {
            str(count): list(reference_new_labels[count]) for count in ordered_counts
        },
        "clean_support_query_rows": 0,
        "nested_support_pass": True,
        "query_identity_lock_pass": True,
    }


def clustered_paired_bootstrap(
    prediction_rows: Sequence[dict[str, Any]],
    *,
    repetitions: int = 10_000,
    seed: int = 20260715,
) -> dict[str, float]:
    clusters: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in prediction_rows:
        if int(row["k_shot"]) != 1 or str(row["evaluation_role"]) != "target_old":
            continue
        clusters[(str(row["receiver"]), int(row["seed"]))].append(
            float(row["candidate_correct"]) - float(row["direct_correct"])
        )
    if len(clusters) < 2 or any(not values for values in clusters.values()):
        raise ValueError("K1 paired bootstrap requires at least two receiver-seed clusters")
    keys = sorted(clusters)
    values = [np.asarray(clusters[key], dtype=np.float64) for key in keys]
    observed = float(np.concatenate(values).mean())
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(repetitions), dtype=np.float64)
    for index in range(int(repetitions)):
        selected = rng.integers(0, len(values), size=len(values))
        samples[index] = float(
            np.concatenate([values[int(position)] for position in selected]).mean()
        )
    return {
        "delta": observed,
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "cluster_count": int(len(values)),
        "repetitions": int(repetitions),
    }


def matched_k5_drop_summary(
    rows: Sequence[dict[str, Any]],
    *,
    new_class_count: int,
    metric: str,
) -> dict[str, float | int]:
    """Compare K5 with the exact receiver/seed/scenario-matched K10 row."""

    matched: dict[tuple[str, int, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if int(row["new_class_count"]) != int(new_class_count):
            continue
        k_shot = int(row["k_shot"])
        if k_shot not in (5, 10):
            continue
        key = (str(row["receiver"]), int(row["seed"]), str(row["scenario"]))
        if k_shot in matched[key]:
            raise ValueError(f"duplicate matched K{k_shot} row: {key}")
        matched[key][k_shot] = row
    if not matched or any(set(pair) != {5, 10} for pair in matched.values()):
        raise ValueError(
            f"incomplete matched K5/K10 rows for new-{new_class_count} {metric}"
        )
    drops = np.asarray(
        [
            _float(pair[10], metric) - _float(pair[5], metric)
            for pair in matched.values()
        ],
        dtype=np.float64,
    )
    return {
        "pair_count": int(drops.size),
        "mean_drop": float(drops.mean()),
        "max_drop": float(drops.max()),
    }


def evaluate_gates(
    rows: Sequence[dict[str, Any]], prediction_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_receiver_k: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_new_k: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        k = int(row["k_shot"])
        receiver = str(row["receiver"])
        new_count = int(row["new_class_count"])
        by_k[k].append(row)
        by_receiver_k[(receiver, k)].append(row)
        by_new_k[(new_count, k)].append(row)
    summary_by_k: dict[str, Any] = {}
    for k in FORMAL_K:
        values = by_k[int(k)]
        forgetting = float(np.mean([_float(row, "average_forgetting") for row in values]))
        identity_forgetting = float(
            np.mean([_float(row, "identity_average_forgetting") for row in values])
        )
        summary_by_k[str(k)] = {
            "old_acc_before_increment": float(
                np.mean([_float(row, "old_acc_before_increment") for row in values])
            ),
            "old_acc_after_increment": float(
                np.mean([_float(row, "old_acc_after_increment") for row in values])
            ),
            "average_forgetting": forgetting,
            "old_adaptation_gain": -forgetting,
            "identity_average_forgetting": identity_forgetting,
            "forgetting_delta_vs_identity": forgetting - identity_forgetting,
        }
    k_forgetting = [summary_by_k[str(k)]["average_forgetting"] for k in FORMAL_K]
    paired = clustered_paired_bootstrap(prediction_rows)
    gates: dict[str, bool] = {
        "k1_forgetting_overall_nonpositive": summary_by_k["1"][
            "average_forgetting"
        ]
        <= 0.0,
        "k1_forgetting_each_receiver_nonpositive": all(
            np.mean(
                [
                    _float(row, "average_forgetting")
                    for row in by_receiver_k[(receiver, 1)]
                ]
            )
            <= 0.0
            for receiver in FORMAL_RECEIVERS
        ),
        "k5_k10_k20_forgetting_no_worse_than_identity": all(
            summary_by_k[str(k)]["forgetting_delta_vs_identity"] <= 0.0
            for k in (5, 10, 20)
        ),
        "k1_direct_delta_at_least_2pp": paired["delta"] >= 0.02,
        "k1_direct_delta_ci_lower_positive": paired["ci95_lower"] > 0.0,
        "k1_direct_delta_each_receiver_nonnegative": all(
            np.mean(
                [
                    _float(row, "old_acc_after_increment")
                    - _float(row, "direct_adv3b02_old_acc")
                    for row in by_receiver_k[(receiver, 1)]
                ]
            )
            >= 0.0
            for receiver in FORMAL_RECEIVERS
        ),
    }
    matched_k5_drops: dict[str, dict[str, dict[str, float | int]]] = {}
    for new_count, target in NEW_ACCURACY_TARGET.items():
        k10 = by_new_k[(int(new_count), 10)]
        gates[f"k10_old_acc_new{new_count}"] = float(
            np.mean([_float(row, "old_acc_after_increment") for row in k10])
        ) >= K10_OLD_TARGET
        gates[f"k10_min_old_class_new{new_count}"] = float(
            np.min([_float(row, "min_old_class_acc") for row in k10])
        ) >= K10_MIN_OLD_CLASS_TARGET
        gates[f"k10_seen_new_acc_new{new_count}"] = float(
            np.mean([_float(row, "seen_new_acc") for row in k10])
        ) >= float(target)
        matched_k5_drops[str(new_count)] = {}
        for metric in (
            "old_acc_after_increment",
            "min_old_class_acc",
            "seen_new_acc",
            "h_old_new",
        ):
            drop_summary = matched_k5_drop_summary(
                rows,
                new_class_count=int(new_count),
                metric=metric,
            )
            matched_k5_drops[str(new_count)][metric] = drop_summary
            gates[f"k5_drop_{metric}_new{new_count}"] = (
                float(drop_summary["max_drop"]) <= K5_MAX_DROP
            )
    return {
        "summary_by_k": summary_by_k,
        "worst_K_forgetting": float(max(k_forgetting)),
        "mean_positive_forgetting": float(
            np.mean([max(value, 0.0) for value in k_forgetting])
        ),
        "k1_paired_vs_direct": paired,
        "k5_matched_drop_summary": matched_k5_drops,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "promotion_pass": all(gates.values()),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row_csv", type=Path, required=True)
    parser.add_argument("--prediction_csv", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _read_csv(args.row_csv)
    predictions = _read_csv(args.prediction_csv)
    protocol = validate_nested_protocol(rows)
    result = {
        "schema": "cvs_stage2c_locked_cross_k_summary_v1",
        "protocol": protocol,
        **evaluate_gates(rows, predictions),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["promotion_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
