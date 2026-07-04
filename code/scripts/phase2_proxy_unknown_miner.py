#!/usr/bin/env python
"""Mine source-heldout proxy-unknown TX candidates for Stage2-C.

The miner is deliberately metadata-first: it scores candidate transmitters by
source-receiver coverage and sample balance, while excluding all protocol TX
sets and target receivers. It does not inspect target_unknown samples, features,
or thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for path in (str(CODE_ROOT), str(REPO_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
for path in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, path)

from cvsrffi.wisig_fewshot_payload import canonical_tx_id, parse_tx_id_list
from dataset_wisig import load_wisig_compact_pkl


def parse_id_list(value: str | Sequence[Any] | None) -> list[str]:
    return parse_tx_id_list(value)


def _labels(values: Sequence[Any]) -> list[str]:
    return [canonical_tx_id(value) for value in values]


def resolve_indices(labels: Sequence[str], spec: str | Sequence[Any] | None, *, field: str, allow_empty: bool = False) -> list[int]:
    requested = parse_id_list(spec)
    if not requested:
        if allow_empty:
            return []
        raise ValueError(f"{field} must not be empty")

    out: list[int] = []
    for item in requested:
        found: int | None = None
        try:
            raw_i = int(str(item))
        except Exception:
            raw_i = None
        if raw_i is not None and 0 <= raw_i < len(labels):
            found = raw_i
        else:
            for i, label in enumerate(labels):
                if label == item:
                    found = i
                    break
        if found is None:
            raise ValueError(f"cannot resolve {field} item {item!r} from labels={list(labels)}")
        out.append(int(found))
    return sorted(dict.fromkeys(out))


def _equalized_indices(eq_list: Sequence[Any], equalized: str) -> list[int]:
    text = str(equalized).strip().lower()
    if text == "both":
        return list(range(len(eq_list)))
    target = int(text)
    eq_values = [int(v) for v in eq_list]
    if target not in eq_values:
        raise ValueError(f"equalized={target} not in equalized_list={eq_values}")
    return [eq_values.index(target)]


def _sample_count(arr: Any) -> int:
    if arr is None:
        return 0
    try:
        return int(arr.shape[0])
    except Exception:
        try:
            return int(len(arr))
        except Exception:
            return 0


def _safe_nested_count(data: Any, tx_i: int, rx_i: int, day_i: int, eq_i: int) -> int:
    try:
        return _sample_count(data[tx_i][rx_i][day_i][eq_i])
    except Exception:
        return 0


def _family(tx_id: str) -> str:
    return str(tx_id).split("-", 1)[0]


def build_candidate_table(
    ds: Mapping[str, Any],
    *,
    source_tx_ids: Sequence[str],
    target_new_tx_ids: Sequence[str],
    target_unknown_tx_ids: Sequence[str],
    proxy_source_rxs: Sequence[str],
    target_rxs: Sequence[str],
    candidate_tx_ids: Sequence[str] | None = None,
    exclude_tx_ids: Sequence[str] = (),
    equalized: str = "1",
    min_source_rx_coverage: int = 3,
    min_samples_per_tx: int = 100,
    score_sample_cap: int = 5000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tx_labels = _labels(list(ds.get("tx_list", [])))
    rx_labels = _labels(list(ds.get("rx_list", [])))
    day_labels = _labels(list(ds.get("capture_date_list", [None])))
    eq_list = list(ds.get("equalized_list", [0]))
    data = ds["data"]

    reserved = set(source_tx_ids) | set(target_new_tx_ids) | set(target_unknown_tx_ids) | set(exclude_tx_ids)
    source_rx_idx = resolve_indices(rx_labels, proxy_source_rxs, field="proxy_source_rxs")
    target_rx_idx = resolve_indices(rx_labels, target_rxs, field="target_rxs", allow_empty=True)
    target_rx_set = {rx_labels[i] for i in target_rx_idx}
    allowed_rx_idx = [i for i in source_rx_idx if rx_labels[i] not in target_rx_set]
    if not allowed_rx_idx:
        raise ValueError("proxy_source_rxs minus target_rxs is empty; proxy_unknown would leak target receivers")

    if candidate_tx_ids:
        candidate_idx = resolve_indices(tx_labels, candidate_tx_ids, field="candidate_tx_ids")
    else:
        candidate_idx = list(range(len(tx_labels)))

    eq_idx = _equalized_indices(eq_list, equalized)
    rows: list[dict[str, Any]] = []
    for tx_i in candidate_idx:
        tx_id = tx_labels[tx_i]
        reason = ""
        if tx_id in reserved:
            reason = "reserved_protocol_tx"

        rx_counts: dict[str, int] = {}
        day_nonzero: set[str] = set()
        total = 0
        for rx_i in allowed_rx_idx:
            rx_total = 0
            for day_i, day_label in enumerate(day_labels):
                day_total = 0
                for eq_i in eq_idx:
                    day_total += _safe_nested_count(data, tx_i, rx_i, day_i, eq_i)
                if day_total > 0:
                    day_nonzero.add(day_label)
                rx_total += day_total
            rx_counts[rx_labels[rx_i]] = int(rx_total)
            total += int(rx_total)

        nonzero = [count for count in rx_counts.values() if count > 0]
        rx_coverage = len(nonzero)
        min_rx_samples = min(nonzero) if nonzero else 0
        mean_rx_samples = float(np.mean(nonzero)) if nonzero else 0.0
        rx_balance = float(min_rx_samples / mean_rx_samples) if mean_rx_samples > 0 else 0.0
        if not reason and rx_coverage < int(min_source_rx_coverage):
            reason = "insufficient_source_rx_coverage"
        if not reason and total < int(min_samples_per_tx):
            reason = "insufficient_source_samples"

        base_score = (
            float(rx_coverage) * 1_000_000.0
            + float(len(day_nonzero)) * 10_000.0
            + float(min(total, int(score_sample_cap)))
            + float(min_rx_samples) * 50.0
            + float(rx_balance) * 100.0
        )
        rows.append(
            {
                "tx_id": tx_id,
                "tx_index": int(tx_i),
                "family": _family(tx_id),
                "total_samples": int(total),
                "rx_coverage": int(rx_coverage),
                "source_rx_count": int(len(allowed_rx_idx)),
                "day_coverage": int(len(day_nonzero)),
                "rx_min_samples": int(min_rx_samples),
                "rx_mean_samples": round(mean_rx_samples, 6),
                "rx_balance": round(rx_balance, 6),
                "base_score": round(base_score, 6),
                "eligible": reason == "",
                "excluded_reason": reason,
                "rx_counts": rx_counts,
            }
        )

    audit = {
        "tx_count": len(tx_labels),
        "rx_count": len(rx_labels),
        "day_count": len(day_labels),
        "equalized_indices": eq_idx,
        "reserved_tx_ids": sorted(reserved),
        "source_tx_ids": list(source_tx_ids),
        "target_new_tx_ids": list(target_new_tx_ids),
        "target_unknown_tx_ids": list(target_unknown_tx_ids),
        "proxy_source_rxs_requested": list(proxy_source_rxs),
        "target_rxs": list(target_rxs),
        "proxy_source_rxs_used": [rx_labels[i] for i in allowed_rx_idx],
        "proxy_target_rx_overlap": sorted(set(proxy_source_rxs) & set(target_rxs)),
        "target_unknown_used_for_scoring": False,
        "selection_basis": "source_rx_metadata_counts_only",
    }
    return rows, audit


def select_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    family_repeat_penalty: float = 0.35,
) -> list[dict[str, Any]]:
    pool = [dict(row) for row in rows if bool(row.get("eligible"))]
    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    while pool and len(selected) < int(top_k):
        best_i = -1
        best_key: tuple[float, float, str] | None = None
        for i, row in enumerate(pool):
            family = str(row["family"])
            penalty = float(family_counts.get(family, 0)) * float(family_repeat_penalty) * 1_000_000.0
            adjusted = float(row["base_score"]) - penalty
            key = (adjusted, float(row["base_score"]), str(row["tx_id"]))
            if best_key is None or key > best_key:
                best_key = key
                best_i = i
        picked = pool.pop(best_i)
        picked["selected_rank"] = len(selected) + 1
        picked["adjusted_score"] = round(float(best_key[0]) if best_key is not None else float(picked["base_score"]), 6)
        family_counts[str(picked["family"])] = family_counts.get(str(picked["family"]), 0) + 1
        selected.append(picked)
    return selected


def _csv_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        flat = dict(row)
        flat["rx_counts"] = json.dumps(flat.get("rx_counts", {}), sort_keys=True, ensure_ascii=False)
        out.append(flat)
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = _csv_rows(rows)
    fields = [
        "selected_rank",
        "tx_id",
        "tx_index",
        "family",
        "eligible",
        "excluded_reason",
        "total_samples",
        "rx_coverage",
        "source_rx_count",
        "day_coverage",
        "rx_min_samples",
        "rx_mean_samples",
        "rx_balance",
        "base_score",
        "adjusted_score",
        "rx_counts",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in flat:
            writer.writerow(row)


def run_miner(args: argparse.Namespace) -> dict[str, Any]:
    ds = load_wisig_compact_pkl(args.wisig_pkl)
    source_tx_ids = parse_id_list(args.source_tx_ids)
    target_new_tx_ids = parse_id_list(args.target_new_tx_ids)
    target_unknown_tx_ids = parse_id_list(args.target_unknown_tx_ids)
    proxy_source_rxs = parse_id_list(args.proxy_source_rxs)
    target_rxs = parse_id_list(args.target_rxs)
    candidate_tx_ids = parse_id_list(args.candidate_tx_ids) if args.candidate_tx_ids else None
    exclude_tx_ids = parse_id_list(args.exclude_tx_ids)

    rows, audit = build_candidate_table(
        ds,
        source_tx_ids=source_tx_ids,
        target_new_tx_ids=target_new_tx_ids,
        target_unknown_tx_ids=target_unknown_tx_ids,
        proxy_source_rxs=proxy_source_rxs,
        target_rxs=target_rxs,
        candidate_tx_ids=candidate_tx_ids,
        exclude_tx_ids=exclude_tx_ids,
        equalized=args.equalized,
        min_source_rx_coverage=args.min_source_rx_coverage,
        min_samples_per_tx=args.min_samples_per_tx,
        score_sample_cap=args.score_sample_cap,
    )
    selected = select_candidates(rows, top_k=args.top_k, family_repeat_penalty=args.family_repeat_penalty)
    selected_ids = [str(row["tx_id"]) for row in selected]
    selected_set = set(selected_ids)
    full_rows = []
    rank_by_tx = {str(row["tx_id"]): row.get("selected_rank") for row in selected}
    adjusted_by_tx = {str(row["tx_id"]): row.get("adjusted_score") for row in selected}
    for row in rows:
        out = dict(row)
        tx_id = str(out["tx_id"])
        out["selected_rank"] = rank_by_tx.get(tx_id, "")
        out["adjusted_score"] = adjusted_by_tx.get(tx_id, "")
        out["selected"] = tx_id in selected_set
        full_rows.append(out)
    full_rows.sort(key=lambda row: (0 if row["selected"] else 1, row.get("selected_rank") or 999999, -float(row["base_score"]), str(row["tx_id"])))

    manifest = {
        "method": "source_heldout_proxy_unknown_metadata_miner_v1",
        "wisig_pkl": args.wisig_pkl,
        "top_k": int(args.top_k),
        "min_source_rx_coverage": int(args.min_source_rx_coverage),
        "min_samples_per_tx": int(args.min_samples_per_tx),
        "family_repeat_penalty": float(args.family_repeat_penalty),
        "selected_proxy_unknown_tx_ids": selected_ids,
        "selected_proxy_unknown_tx_ids_csv": ",".join(selected_ids),
        "eligible_count": int(sum(1 for row in rows if row.get("eligible"))),
        "candidate_count": int(len(rows)),
        "audit": audit,
        "selected_rows": selected,
    }

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_csv:
        write_csv(Path(args.output_csv), full_rows)
    print(",".join(selected_ids))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wisig_pkl", required=True)
    parser.add_argument("--source_tx_ids", required=True)
    parser.add_argument("--target_new_tx_ids", required=True)
    parser.add_argument("--target_unknown_tx_ids", required=True)
    parser.add_argument("--proxy_source_rxs", required=True)
    parser.add_argument("--target_rxs", required=True)
    parser.add_argument("--candidate_tx_ids", default=None)
    parser.add_argument("--exclude_tx_ids", default="")
    parser.add_argument("--equalized", default="1")
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--min_source_rx_coverage", type=int, default=3)
    parser.add_argument("--min_samples_per_tx", type=int, default=100)
    parser.add_argument("--score_sample_cap", type=int, default=5000)
    parser.add_argument("--family_repeat_penalty", type=float, default=0.35)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_csv", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_miner(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
