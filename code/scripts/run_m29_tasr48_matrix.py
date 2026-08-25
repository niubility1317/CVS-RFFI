#!/usr/bin/env python3
"""Run the preregistered 30-row M2.9 FFT96/TASR48 screen."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

from cvsrffi.stage2_m29_d92 import M29_ARMS
from cvsrffi.stage2_m29_row_executor import run_m29_row_from_base_cache
from cvsrffi.stage2_m29_tasr import load_phase1_tasr_bundle
from scripts.run_m24_d1_refit_matrix import _cache_root, _canonical_manifest_sha, _write_exclusive


MATRIX_SCHEMA = "cvs.erbt_idr.m29.tasr48_prediction_matrix.v1"
SCREEN_RECEIVERS = ("3-19", "8-8")
SCREEN_SEEDS = (7282101,)
SCREEN_CONDITIONS = ((1, 20), (5, 20), (10, 5))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--supplemental-feature-root")
    parser.add_argument("--tasr-bundle", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-workers", type=int, default=2)
    return parser


def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(task["manifest_path"])
    payload_path = Path(task["payload_path"])
    output_root = Path(task["output_root"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    receipt = run_m29_row_from_base_cache(
        arm=task["arm"],
        row_id=task["row_id"],
        receiver=task["receiver"],
        base_feature_cache_payload=payload_path,
        base_feature_cache_manifest=manifest_path,
        base_feature_cache_payload_sha256=manifest["payload_sha256"],
        base_feature_cache_manifest_sha256=_canonical_manifest_sha(manifest_path),
        tasr_bundle_path=task["tasr_bundle"],
        expected_checkpoint_sha256=task["checkpoint_sha256"],
        output_root=output_root / task["row_id"],
        seed=task["method_seed"],
        device=task["device"],
    )
    if receipt["status"] != "PREDICTIONS_COMPLETE_TRUTH_UNOPENED" or receipt["arm"] != task["arm"]:
        raise ValueError("M2.9 row receipt drift")
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
        "receipt_path": str(output_root / task["row_id"] / "row_execution_receipt.json"),
        "prediction": receipt["prediction"],
    }


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.max_workers <= 2:
        raise ValueError("M2.9 max-workers must be 1 or 2")
    bundle = load_phase1_tasr_bundle(args.tasr_bundle, expected_checkpoint_sha256=args.checkpoint_sha256)
    roots = [Path(args.feature_root).absolute()]
    if args.supplemental_feature_root:
        roots.append(Path(args.supplemental_feature_root).absolute())
    output_root = Path(args.output_root).absolute()
    output_root.mkdir(parents=True, exist_ok=False)
    tasks = []
    for receiver in SCREEN_RECEIVERS:
        for method_seed in SCREEN_SEEDS:
            for k_shot, new_count in SCREEN_CONDITIONS:
                cache_root = _cache_root(tuple(roots), receiver, method_seed, k_shot, new_count)
                for arm in M29_ARMS:
                    row_id = f"rx{receiver}_m{method_seed}_k{k_shot}_new{new_count}__{arm}"
                    tasks.append({
                        "row_id": row_id,
                        "arm": arm,
                        "receiver": receiver,
                        "method_seed": method_seed,
                        "k_shot": k_shot,
                        "new_class_count": new_count,
                        "manifest_path": str(cache_root / "features.manifest.json"),
                        "payload_path": str(cache_root / "features.npz"),
                        "tasr_bundle": str(Path(args.tasr_bundle).absolute()),
                        "checkpoint_sha256": str(args.checkpoint_sha256).lower(),
                        "output_root": str(output_root),
                        "device": args.device,
                    })
    if len(tasks) != 30:
        raise ValueError("M2.9 screen must contain exactly 30 rows")
    completed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(_run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            completed.append(row)
            print(json.dumps({"completed": row["row_id"], "count": len(completed)}, sort_keys=True), flush=True)
    completed.sort(key=lambda row: row["row_id"])
    matrix = {
        "schema": MATRIX_SCHEMA,
        "run_id": args.run_id,
        "matrix_kind": "screen",
        "status": "PREDICTIONS_COMPLETE_TRUTH_UNOPENED",
        "row_count": len(completed),
        "paired_input_identity_count": 6,
        "method_rows_per_arm": 6,
        "scenario_unit_count": 90,
        "reference_family": "best_frozen_fft96_weight_selected_after_same_row_scoring",
        "receivers": list(SCREEN_RECEIVERS),
        "method_seeds": list(SCREEN_SEEDS),
        "conditions": [{"k_shot": k, "new_class_count": n} for k, n in SCREEN_CONDITIONS],
        "arms": list(M29_ARMS),
        "feature_roots": [str(root) for root in roots],
        "tasr_bundle": {"path": str(Path(args.tasr_bundle).absolute()), "checkpoint_sha256": bundle.checkpoint_sha256, "component_id": bundle.component_id, "state_bytes": bundle.state_bytes},
        "entries": completed,
        "query_truth_opened": False,
    }
    _write_exclusive(output_root / "matrix_index.json", matrix)
    print(json.dumps({"status": matrix["status"], "row_count": len(completed)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
