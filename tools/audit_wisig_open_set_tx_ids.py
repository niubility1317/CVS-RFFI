#!/usr/bin/env python
"""Audit WiSig source/new/unknown TX identity splits for open-set SFE runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
for path in (str(ROOT), str(CODE_ROOT)):
    if path in sys.path:
        sys.path.remove(path)
for path in (str(ROOT), str(CODE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cvsrffi.wisig_fewshot_payload import assert_disjoint_tx_sets, canonical_tx_id, parse_tx_id_list
from dataset_wisig import WiSigCompactDataset, load_wisig_compact_pkl
from export_spaceborne_features import _cap_dataset_per_tx, _resolve_tx_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pkl", required=True)
    parser.add_argument("--new-pkl", required=True)
    parser.add_argument("--source-tx-ids", required=True)
    parser.add_argument("--new-tx-ids", required=True)
    parser.add_argument("--unknown-tx-ids", required=True)
    parser.add_argument("--wisig-equalized", default="1")
    parser.add_argument("--wisig-domain", default="rx_day")
    parser.add_argument("--wisig-out-len", type=int, default=256)
    parser.add_argument("--max-samples-per-combo", type=int, default=0)
    parser.add_argument("--max-samples-per-tx", type=int, default=200)
    parser.add_argument("--source-proto-per-tx", type=int, default=20)
    parser.add_argument("--source-query-per-tx", type=int, default=20)
    parser.add_argument("--shots", type=int, default=20)
    parser.add_argument("--query-per-tx", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1457)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _resolve(ds: dict[str, Any], spec: str, field: str) -> dict[str, Any]:
    indices, labels = _resolve_tx_indices(ds.get("tx_list", []), spec, field=field)
    return {
        "requested": parse_tx_id_list(spec),
        "indices": [int(v) for v in indices],
        "labels": [canonical_tx_id(v) for v in labels],
    }


def _count_samples(
    ds: dict[str, Any],
    tx_indices: Sequence[int],
    *,
    equalized: str,
    domain: str,
    out_len: int,
    max_samples_per_combo: int,
    max_samples_per_tx: int,
    seed: int,
) -> dict[str, int]:
    base = WiSigCompactDataset(
        ds,
        out_len=int(out_len),
        equalized=("both" if str(equalized).lower() == "both" else int(equalized)),
        tx_keep=[int(v) for v in tx_indices],
        domain=str(domain),
        max_samples_per_combo=(None if int(max_samples_per_combo) <= 0 else int(max_samples_per_combo)),
        sample_strategy="random",
        seed=int(seed),
        build_index=True,
    )
    capped = _cap_dataset_per_tx(base, int(max_samples_per_tx), int(seed), split_source="open_set_audit")
    tx_list = list(ds.get("tx_list", []))
    counts: dict[str, int] = {}
    for item in capped.index:
        label = canonical_tx_id(tx_list[int(item.tx_i)] if int(item.tx_i) < len(tx_list) else int(item.tx_i))
        counts[label] = counts.get(label, 0) + 1
    return counts


def _requirements(args: argparse.Namespace, *, source: Sequence[str], new: Sequence[str], unknown: Sequence[str]) -> dict[str, dict[str, int]]:
    required: dict[str, dict[str, int]] = {}
    for tx in source:
        required[str(tx)] = {"role": "source", "min_samples": int(args.source_proto_per_tx) + int(args.source_query_per_tx)}
    for tx in new:
        required[str(tx)] = {"role": "new", "min_samples": int(args.shots) + int(args.query_per_tx)}
    for tx in unknown:
        required[str(tx)] = {"role": "unknown", "min_samples": int(args.query_per_tx)}
    return required


def main() -> int:
    args = parse_args()
    if not parse_tx_id_list(args.unknown_tx_ids):
        raise ValueError("--unknown-tx-ids must not be empty for a true open-set audit")

    source_ds = load_wisig_compact_pkl(str(args.source_pkl))
    new_ds = load_wisig_compact_pkl(str(args.new_pkl))
    source = _resolve(source_ds, str(args.source_tx_ids), "source_tx_ids")
    new = _resolve(new_ds, str(args.new_tx_ids), "new_tx_ids")
    unknown = _resolve(new_ds, str(args.unknown_tx_ids), "unknown_tx_ids")

    overlap_audit = assert_disjoint_tx_sets(
        source_tx_ids=source["labels"],
        new_tx_ids=new["labels"],
        unknown_tx_ids=unknown["labels"],
    )
    source_counts = _count_samples(
        source_ds,
        source["indices"],
        equalized=str(args.wisig_equalized),
        domain=str(args.wisig_domain),
        out_len=int(args.wisig_out_len),
        max_samples_per_combo=int(args.max_samples_per_combo),
        max_samples_per_tx=int(args.max_samples_per_tx),
        seed=int(args.seed),
    )
    target_counts = _count_samples(
        new_ds,
        list(new["indices"]) + list(unknown["indices"]),
        equalized=str(args.wisig_equalized),
        domain=str(args.wisig_domain),
        out_len=int(args.wisig_out_len),
        max_samples_per_combo=int(args.max_samples_per_combo),
        max_samples_per_tx=int(args.max_samples_per_tx),
        seed=int(args.seed) + 17,
    )
    observed_counts = dict(source_counts)
    observed_counts.update(target_counts)
    requirements = _requirements(args, source=source["labels"], new=new["labels"], unknown=unknown["labels"])
    sample_audit = {}
    insufficient = {}
    for tx, req in requirements.items():
        available = int(observed_counts.get(tx, 0))
        min_samples = int(req["min_samples"])
        item = {
            "role": req["role"],
            "available_after_caps": available,
            "min_required": min_samples,
            "ok": available >= min_samples,
        }
        sample_audit[tx] = item
        if not item["ok"]:
            insufficient[tx] = item

    result = {
        "source_pkl": str(args.source_pkl),
        "new_pkl": str(args.new_pkl),
        "source": source,
        "new": new,
        "unknown": unknown,
        "overlap_audit": overlap_audit,
        "disjoint_ok": all(not values for values in overlap_audit.values()),
        "sample_audit": sample_audit,
        "counts_ok": not bool(insufficient),
        "insufficient": insufficient,
        "settings": {
            "wisig_equalized": str(args.wisig_equalized),
            "wisig_domain": str(args.wisig_domain),
            "wisig_out_len": int(args.wisig_out_len),
            "max_samples_per_combo": int(args.max_samples_per_combo),
            "max_samples_per_tx": int(args.max_samples_per_tx),
            "source_proto_per_tx": int(args.source_proto_per_tx),
            "source_query_per_tx": int(args.source_query_per_tx),
            "shots": int(args.shots),
            "query_per_tx": int(args.query_per_tx),
            "seed": int(args.seed),
        },
    }
    text = json.dumps(result, indent=2, ensure_ascii=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text)
    if insufficient:
        raise ValueError(f"insufficient samples after caps: {insufficient}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
