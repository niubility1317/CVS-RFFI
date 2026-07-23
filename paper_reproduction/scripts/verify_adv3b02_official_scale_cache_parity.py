#!/usr/bin/env python3
"""Fail closed unless expanded caches preserve first20 new-class LEO rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from cvsrffi.leo_weak_cache import (
    FORMAL_LEO_WEAK_SCENARIOS,
    ids_sha256,
    sha256_file,
)
from paper_reproduction.scripts.build_adv3b02_paper_full_ci_bundle import (
    load_comparison_leo_cache_set,
)


def _write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify(args: argparse.Namespace) -> dict:
    reference_path = Path(args.reference_cache_set).resolve(strict=True)
    expanded_path = Path(args.expanded_cache_set).resolve(strict=True)
    preserved_labels = tuple(
        value.strip()
        for value in str(args.preserved_class_labels).split(",")
        if value.strip()
    )
    if len(preserved_labels) != 20 or len(set(preserved_labels)) != 20:
        raise ValueError("parity gate requires exactly 20 unique new-class labels")
    allowed_roles = {"target_old", "target_new"}
    reference, _reference_manifest, _reference_audit = (
        load_comparison_leo_cache_set(
            reference_path,
            expected_scope=str(args.reference_scope),
            allowed_roles=allowed_roles,
        )
    )
    expanded, _expanded_manifest, _expanded_audit = (
        load_comparison_leo_cache_set(
            expanded_path,
            expected_scope=str(args.expanded_scope),
            allowed_roles=allowed_roles,
        )
    )
    scenario_receipts = {}
    preserved_set = set(preserved_labels)
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        left = reference[scenario]
        right = expanded[scenario]
        left_tx = np.asarray(left["tx_ids"]).astype(str)
        right_tx = np.asarray(right["tx_ids"]).astype(str)
        left_mask = np.asarray(
            [value in preserved_set for value in left_tx], dtype=bool
        )
        right_mask = np.asarray(
            [value in preserved_set for value in right_tx], dtype=bool
        )
        if set(left_tx[left_mask].tolist()) != preserved_set:
            raise ValueError(f"reference preserved class coverage drift: {scenario}")
        if set(right_tx[right_mask].tolist()) != preserved_set:
            raise ValueError(f"expanded preserved class coverage drift: {scenario}")
        comparisons = {
            "tx_ids": (
                left_tx[left_mask].tolist(),
                right_tx[right_mask].tolist(),
            ),
            "sample_ids": (
                np.asarray(left["sample_ids"]).astype(str)[left_mask].tolist(),
                np.asarray(right["sample_ids"]).astype(str)[right_mask].tolist(),
            ),
            "post_channel_iq_sha256": (
                np.asarray(left["post_channel_iq_sha256"])
                .astype(str)[left_mask]
                .tolist(),
                np.asarray(right["post_channel_iq_sha256"])
                .astype(str)[right_mask]
                .tolist(),
            ),
        }
        for field, (observed_reference, observed_expanded) in comparisons.items():
            if observed_reference != observed_expanded:
                raise ValueError(
                    f"expanded cache parity mismatch: {scenario} {field}"
                )
        scenario_receipts[scenario] = {
            "row_count": int(np.sum(left_mask)),
            "sample_ids_sha256": ids_sha256(comparisons["sample_ids"][0]),
            "post_channel_iq_sha256_root": ids_sha256(
                comparisons["post_channel_iq_sha256"][0]
            ),
        }
    receipt = {
        "schema": "cvs.adv3b02.official_scale_cache_parity_receipt.v1",
        "status": "PASS",
        "reference_cache_set": str(reference_path),
        "reference_cache_set_sha256": sha256_file(reference_path),
        "expanded_cache_set": str(expanded_path),
        "expanded_cache_set_sha256": sha256_file(expanded_path),
        "preserved_class_labels": list(preserved_labels),
        "verified_fields": [
            "tx_ids",
            "sample_ids",
            "post_channel_iq_sha256",
        ],
        "scenario_receipts": scenario_receipts,
    }
    _write_new(Path(args.output), receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache-set", type=Path, required=True)
    parser.add_argument("--expanded-cache-set", type=Path, required=True)
    parser.add_argument(
        "--reference-scope",
        choices=("stage2_registered", "external_comparison_registered"),
        default="stage2_registered",
    )
    parser.add_argument(
        "--expanded-scope",
        choices=("stage2_registered", "external_comparison_registered"),
        default="external_comparison_registered",
    )
    parser.add_argument("--preserved-class-labels", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(verify(parse_args()), ensure_ascii=False, sort_keys=True))
