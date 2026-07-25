#!/usr/bin/env python3
"""Validate the real D104 metadata split without opening held performance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvsrffi.phase1_rb_metabias4_bundle import (  # noqa: E402
    merge_verified_phase1_tap_and_dual_archives,
)
from cvsrffi.stage2_d104_source_split import (  # noqa: E402
    load_d104_exclusion_manifest,
    partition_d104_source_pool,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tap-archive", type=Path, required=True)
    parser.add_argument("--dual-archive", type=Path, required=True)
    parser.add_argument("--exclusion-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tap_path = args.tap_archive.resolve(strict=True)
    dual_path = args.dual_archive.resolve(strict=True)
    exclusion_path = args.exclusion_manifest.resolve(strict=True)
    output = args.output_json.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable D104 split validation exists: {output}")
    exclusion = load_d104_exclusion_manifest(exclusion_path)
    tap = _load(tap_path)
    dual = _load(dual_path)
    merged = merge_verified_phase1_tap_and_dual_archives(tap, dual)
    source_pool = {
        "z_id": np.asarray(tap["z_id"], dtype=np.float32),
        "z_dom": np.asarray(merged["z_dom"], dtype=np.float32),
        "pre_relu": np.asarray(merged["pre_relu"], dtype=np.float32),
        "labels": merged["labels"].astype(str),
        "receiver_ids": merged["receiver_ids"].astype(str),
        "day_ids": merged["day_ids"].astype(str),
        "physical_ids": merged["physical_ids"].astype(str),
        "scenario_names": tap["scenario_names"].astype(str),
        "observation_ids": tap["observation_ids"].astype(str),
        "class_ids": merged["class_ids"].astype(str),
    }
    labeled, unlabeled, scorer, receipt = partition_d104_source_pool(source_pool)
    historical = set(exclusion["query_physical_ids_sorted"])
    held = set(scorer["physical_ids"].astype(str).tolist())
    cells = receipt["cells"]
    result = {
        "schema": "cvs.d104_r1.source_split.real_metadata_validation.v2",
        "status": "D104_SOURCE_SPLIT_METADATA_VALIDATED",
        "split_id": receipt["split_id"],
        "tap_archive_sha256": _sha256(tap_path),
        "dual_archive_sha256": _sha256(dual_path),
        "exclusion_manifest_sha256": _sha256(exclusion_path),
        "exclusion_query_count": len(historical),
        "historical_query_in_new_source_val_count": len(historical & held),
        "counts": receipt["counts"],
        "physical_id_roots": receipt["physical_id_roots"],
        "split_receipt_sha256": receipt["receipt_sha256"],
        "cell_count": receipt["cell_count"],
        "receiver_tx_group_count": receipt["receiver_tx_group_count"],
        "held_per_cell_min": min(row["source_val"] for row in cells),
        "held_per_cell_max": max(row["source_val"] for row in cells),
        "eligible_after_exclusion_min": min(
            row["eligible_after_historical_exclusion"] for row in cells
        ),
        "eligible_after_exclusion_max": max(
            row["eligible_after_historical_exclusion"] for row in cells
        ),
        "four_day_labeled_range": receipt["four_day_labeled_range"],
        "leave_day_labeled_range": receipt["leave_day_labeled_range"],
        "overlap_count": receipt["overlap_count"],
        "union_complete": receipt["union_complete"],
        "labeled_rows": len(labeled["physical_ids"]),
        "unlabeled_rows": len(unlabeled["physical_ids"]),
        "source_val_rows": len(scorer["physical_ids"]),
        "source_val_labels_used_for_split_grouping": True,
        "source_val_predictions_computed": False,
        "source_val_performance_computed": False,
        "target_access": False,
        "n607_run": False,
    }
    if (
        result["historical_query_in_new_source_val_count"] != 0
        or result["counts"] != {"L_s": 588, "U_s": 5292, "source_val": 2520}
        or result["held_per_cell_min"] != 15
        or result["held_per_cell_max"] != 15
        or result["overlap_count"] != 0
        or result["union_complete"] is not True
    ):
        raise ValueError("D104 real source split validation failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            result,
            stream,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
