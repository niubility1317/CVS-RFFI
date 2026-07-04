#!/usr/bin/env python
"""Audit whether Phase2 features support strict same-event receiver collaboration."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phase2_collaborative_open_set_qknn_eval import _split_support_query, _split_support_query_selected


STRICT_KEY_FIELDS = ("role", "tx_id", "day_id", "eq_id", "sig_id", "channel_view", "sat_scenario")
ROLE_MAP = {
    "target_old": "old",
    "target_new": "seen_new",
    "seen_new": "seen_new",
    "target_unknown": "unknown",
    "unknown": "unknown",
}


def _norm(value: Any) -> str:
    return str(value).strip()


def _strict_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(_norm(row.get(field, "")) for field in STRICT_KEY_FIELDS)


def audit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_receivers: Sequence[str],
    receiver_count: int,
    split_filter: str | None = None,
) -> dict[str, Any]:
    groups: dict[tuple[str, ...], set[str]] = defaultdict(set)
    role_totals: Counter[str] = Counter()
    for row in rows:
        if split_filter and _norm(row.get("split", "")) != str(split_filter):
            continue
        rx = _norm(row.get("rx_id", ""))
        if rx not in set(target_receivers):
            continue
        role = ROLE_MAP.get(_norm(row.get("role", "")), _norm(row.get("role", "")))
        if role not in {"old", "seen_new", "unknown"}:
            continue
        normalized = dict(row)
        normalized["role"] = role
        groups[_strict_key(normalized)].add(rx)
        role_totals[role] += 1

    receiver_histogram: Counter[int] = Counter(len(receivers) for receivers in groups.values())
    strict_groups = {key: receivers for key, receivers in groups.items() if len(receivers) >= 2}
    full_groups = {key: receivers for key, receivers in groups.items() if len(receivers) >= int(receiver_count)}
    max_receivers = max((len(receivers) for receivers in groups.values()), default=0)
    coverage_by_k = {
        str(k): sum(1 for receivers in groups.values() if len(receivers) >= k)
        for k in range(1, int(receiver_count) + 1)
    }
    return {
        "row_count": int(len(rows)),
        "target_receivers": list(target_receivers),
        "receiver_count": int(receiver_count),
        "role_totals": dict(sorted(role_totals.items())),
        "strict_key_fields": list(STRICT_KEY_FIELDS),
        "group_count": int(len(groups)),
        "strict_candidate_group_count": int(len(strict_groups)),
        "full_receiver_group_count": int(len(full_groups)),
        "max_receivers_per_strict_key": int(max_receivers),
        "receiver_count_histogram": {str(k): int(v) for k, v in sorted(receiver_histogram.items())},
        "coverage_by_min_receivers": coverage_by_k,
        "strict_event_candidate_possible": bool(strict_groups),
        "strict_full_receiver_candidate_possible": bool(full_groups),
        "interpretation": (
            "strict_same_event_candidate_available"
            if strict_groups
            else "no_shared_sig_key_receiver_domain_ranked_only"
        ),
    }


def _base_row_from_npz(z: Any, i: int, *, split: str = "all") -> dict[str, Any]:
    return {
        "role": _norm(z["dataset_role"][i]),
        "tx_id": _norm(z["tx_ids"][i]),
        "rx_id": _norm(z["rx_ids"][i]),
        "day_id": _norm(z["day_ids"][i]),
        "eq_id": _norm(z["eq_ids"][i]),
        "sig_id": _norm(z["sig_ids"][i]),
        "channel_view": _norm(z["channel_views"][i]),
        "sat_scenario": _norm(z["sat_scenarios"][i]),
        "split": split,
    }


def _load_npz_rows(feature_npz: Path, *, k_shot: int, query_per_class: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    z = np.load(feature_npz, allow_pickle=True)
    manifest = json.loads(str(z["manifest_json"].item())) if "manifest_json" in z.files else {}
    rows: list[dict[str, Any]] = []
    target_receivers = _target_receivers_from_manifest(manifest)
    old_labels = list(manifest.get("target_old_tx_ids", []))
    new_labels = list(manifest.get("new_tx_ids", []))
    unknown_labels = list(manifest.get("unknown_tx_ids", []))
    payload = {key: z[key] for key in z.files if key != "manifest_json"}
    features = z["features"]
    for rx in target_receivers:
        for label in old_labels:
            support, query = _split_support_query_selected(
                payload,
                features=features,
                role="target_old",
                tx_id=label,
                rx_id=rx,
                k_shot=k_shot,
                query_per_class=query_per_class,
                seed=seed,
                support_selection_policy="stable_first",
            )
            rows.extend(_base_row_from_npz(z, i, split="support") for i in support)
            rows.extend(_base_row_from_npz(z, i, split="query") for i in query)
        for label in new_labels:
            support, query = _split_support_query_selected(
                payload,
                features=features,
                role="target_new",
                tx_id=label,
                rx_id=rx,
                k_shot=k_shot,
                query_per_class=query_per_class,
                seed=seed,
                support_selection_policy="stable_first",
            )
            rows.extend(_base_row_from_npz(z, i, split="support") for i in support)
            rows.extend(_base_row_from_npz(z, i, split="query") for i in query)
        for label in unknown_labels:
            _, query = _split_support_query(
                payload,
                role="target_unknown",
                tx_id=label,
                rx_id=rx,
                k_shot=0,
                query_per_class=query_per_class,
                seed=seed,
            )
            rows.extend(_base_row_from_npz(z, i, split="query") for i in query)
    return rows, manifest


def _target_receivers_from_manifest(manifest: Mapping[str, Any]) -> list[str]:
    value = manifest.get("target_old", {}).get("rxs", "")
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _write_group_csv(path: Path, rows: Sequence[Mapping[str, Any]], target_receivers: Sequence[str]) -> None:
    groups: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in rows:
        rx = _norm(row.get("rx_id", ""))
        if rx not in set(target_receivers):
            continue
        role = ROLE_MAP.get(_norm(row.get("role", "")), _norm(row.get("role", "")))
        if role not in {"old", "seen_new", "unknown"}:
            continue
        normalized = dict(row)
        normalized["role"] = role
        groups[_strict_key(normalized)].add(rx)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [*STRICT_KEY_FIELDS, "receiver_count", "receiver_ids"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, receivers in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
            writer.writerow(
                {
                    **{field: key[i] for i, field in enumerate(STRICT_KEY_FIELDS)},
                    "receiver_count": len(receivers),
                    "receiver_ids": ",".join(sorted(receivers)),
                }
            )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_groups_csv", type=Path, required=True)
    parser.add_argument("--k_shot", type=int, default=8)
    parser.add_argument("--query_per_class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4070303)
    args = parser.parse_args(argv)

    rows, manifest = _load_npz_rows(
        args.feature_npz,
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
    )
    target_receivers = _target_receivers_from_manifest(manifest)
    result = audit_rows(rows, target_receivers=target_receivers, receiver_count=len(target_receivers), split_filter="query")
    result["feature_npz"] = str(args.feature_npz)
    result["manifest_checkpoint"] = manifest.get("checkpoint", "")
    result["target_channel_view"] = manifest.get("target_channel_view", "")
    result["target_channel_scenarios"] = manifest.get("target_channel_scenarios", [])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _write_group_csv(args.output_groups_csv, rows, target_receivers)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
