#!/usr/bin/env python3
"""Build the immutable D104 v2 source split from sealed Phase1 archives."""

from __future__ import annotations

import argparse
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
    publish_d104_source_split_archives,
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tap-archive", type=Path, required=True)
    parser.add_argument("--dual-archive", type=Path, required=True)
    parser.add_argument("--exclusion-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--cache-set-sha256", required=True)
    parser.add_argument("--selection-salt-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tap = _load(args.tap_archive.resolve(strict=True))
    dual = _load(args.dual_archive.resolve(strict=True))
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
    result = publish_d104_source_split_archives(
        source_pool,
        exclusion_manifest_path=args.exclusion_manifest,
        output_dir=args.output_dir,
        checkpoint_sha256=args.checkpoint_sha256,
        runtime_sha256=args.runtime_sha256,
        cache_set_sha256=args.cache_set_sha256,
        selection_salt_receipt_sha256=(
            args.selection_salt_receipt_sha256
        ),
        upstream_audit={
            "formal_export_script": True,
            "source_val_labels_used_for_stratified_split": True,
            "performance_computed": False,
            "target_access": False,
        },
    )
    print(result["manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
