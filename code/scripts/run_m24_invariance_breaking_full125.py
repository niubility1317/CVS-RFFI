#!/usr/bin/env python3
"""Run G0-G4 invariance-breaking methods on the complete paired 125 grid."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cvsrffi.stage2_m24_invariance_breaking import G1, G2, G3, G4
from cvsrffi.stage2_m24_row_executor import run_m24_invariance_row_from_base_cache
from cvsrffi.stage2_m24_safe_residual import D0
from scripts.run_m24_d1_refit_matrix import (
    DEFAULT_CONDITIONS,
    DEFAULT_RECEIVERS,
    DEFAULT_SEEDS,
    _cache_root,
    _canonical_manifest_sha,
    _write_exclusive,
)


MATRIX_SCHEMA = "cvs.erbt_idr.m24.invariance_breaking_prediction_matrix.v1"
EVIDENCE_ARMS = (D0, G1, G2, G3, G4)
EXPECTED_INPUT_IDENTITIES = 125
EXPECTED_METHOD_ROWS = EXPECTED_INPUT_IDENTITIES * len(EVIDENCE_ARMS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--supplemental-feature-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-workers", type=int, default=2)
    return parser


def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(task["manifest_path"])
    payload_path = Path(task["payload_path"])
    output_root = Path(task["output_root"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    receipt = run_m24_invariance_row_from_base_cache(
        arm=str(task["arm"]),
        row_id=str(task["row_id"]),
        receiver=str(task["receiver"]),
        base_feature_cache_payload=payload_path,
        base_feature_cache_manifest=manifest_path,
        base_feature_cache_payload_sha256=str(manifest["payload_sha256"]),
        base_feature_cache_manifest_sha256=_canonical_manifest_sha(manifest_path),
        output_root=output_root / str(task["row_id"]),
        seed=int(task["method_seed"]),
        device=str(task["device"]),
    )
    return {
        "row_id": task["row_id"],
        "arm": task["arm"],
        "receiver": task["receiver"],
        "method_seed": task["method_seed"],
        "k_shot": task["k_shot"],
        "new_class_count": task["new_class_count"],
        "support_seed": int(manifest["support_seed"]),
        "query_seed": int(manifest["query_seed"]),
        "new_class_draw_seed": int(manifest["new_class_draw_seed"]),
        "capsule_id": str(manifest["capsule_id"]),
        "split_id": str(manifest["split_id"]),
        "feature_cache_root": str(manifest_path.parent),
        "receipt_path": str(output_root / str(task["row_id"]) / "row_execution_receipt.json"),
        "prediction": receipt["prediction"],
    }


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.max_workers <= 8:
        raise ValueError("max-workers must be between 1 and 8")
    roots = [Path(args.feature_root).absolute()]
    if args.supplemental_feature_root:
        roots.append(Path(args.supplemental_feature_root).absolute())
    feature_roots = tuple(roots)
    output_root = Path(args.output_root).absolute()
    output_root.mkdir(parents=True, exist_ok=False)
    tasks: list[dict[str, Any]] = []
    for receiver in DEFAULT_RECEIVERS:
        for method_seed in DEFAULT_SEEDS:
            for k_shot, new_count in DEFAULT_CONDITIONS:
                cache_root = _cache_root(feature_roots, receiver, method_seed, k_shot, new_count)
                manifest_path = cache_root / "features.manifest.json"
                payload_path = cache_root / "features.npz"
                for arm in EVIDENCE_ARMS:
                    row_id = f"rx{receiver}_m{method_seed}_k{k_shot}_new{new_count}__{arm}"
                    tasks.append({
                        "row_id": row_id,
                        "arm": arm,
                        "receiver": receiver,
                        "method_seed": method_seed,
                        "k_shot": k_shot,
                        "new_class_count": new_count,
                        "manifest_path": str(manifest_path),
                        "payload_path": str(payload_path),
                        "output_root": str(output_root),
                        "device": args.device,
                    })
    if len(tasks) != EXPECTED_METHOD_ROWS:
        raise ValueError(f"expected {EXPECTED_METHOD_ROWS} method rows, got {len(tasks)}")
    completed: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(_run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            completed.append(row)
            print(json.dumps({"completed": row["row_id"], "count": len(completed)}, sort_keys=True), flush=True)
    completed.sort(key=lambda row: str(row["row_id"]))
    matrix = {
        "schema": MATRIX_SCHEMA,
        "run_id": str(args.run_id),
        "status": "PREDICTIONS_COMPLETE_TRUTH_UNOPENED",
        "row_count": len(completed),
        "paired_input_identity_count": EXPECTED_INPUT_IDENTITIES,
        "method_rows_per_arm": EXPECTED_INPUT_IDENTITIES,
        "scenario_unit_count": EXPECTED_METHOD_ROWS * 3,
        "primary_d92_e0_baseline": "P2-A1_NO_RF32",
        "reference_arm": D0,
        "receivers": list(DEFAULT_RECEIVERS),
        "method_seeds": list(DEFAULT_SEEDS),
        "conditions": [{"k_shot": k, "new_class_count": n} for k, n in DEFAULT_CONDITIONS],
        "arms": list(EVIDENCE_ARMS),
        "feature_roots": [str(root) for root in feature_roots],
        "entries": completed,
        "query_truth_opened": False,
    }
    _write_exclusive(output_root / "matrix_index.json", matrix)
    print(json.dumps({"status": matrix["status"], "row_count": len(completed)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

