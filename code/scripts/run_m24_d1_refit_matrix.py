#!/usr/bin/env python3
"""Run the paired D92-E0-noRF32/R1-compile/R2-refit falsification screen."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cvsrffi.stage2_m24_row_executor import run_m24_d1_evidence_row_from_base_cache
from cvsrffi.stage2_m24_safe_residual import D0, D1, D1_REFIT


MATRIX_SCHEMA = "cvs.erbt_idr.m24.d1_refit_prediction_matrix.v1"
EVIDENCE_ARMS = (D0, D1, D1_REFIT)
DEFAULT_RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
DEFAULT_SEEDS = (7282101, 7282102, 7282103, 7282104, 7282105)
DEFAULT_CONDITIONS = ((1, 20), (2, 20), (5, 20), (10, 20), (10, 5))
EXPECTED_INPUT_IDENTITIES = 125
EXPECTED_METHOD_ROWS = EXPECTED_INPUT_IDENTITIES * len(EVIDENCE_ARMS)


def _canonical_manifest_sha(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--supplemental-feature-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-workers", type=int, default=2)
    return parser


def _cache_root(
    roots: tuple[Path, ...], receiver: str, method_seed: int, k_shot: int, new_count: int
) -> Path:
    relative = (
        Path(f"rx_{receiver.replace('-', '_')}")
        / f"method_{method_seed}"
        / f"new{new_count}"
        / f"k_{k_shot}"
        / "stage2c"
    )
    matches = [root / relative for root in roots if (root / relative / "features.manifest.json").is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one feature cache for rx={receiver}, seed={method_seed}, "
            f"K={k_shot}, new={new_count}; found={len(matches)}"
        )
    return matches[0]


def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(task["manifest_path"])
    payload_path = Path(task["payload_path"])
    output_root = Path(task["output_root"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    receipt = run_m24_d1_evidence_row_from_base_cache(
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
    parity = receipt["d1_historical_parity"]
    if task["arm"] == D1 and (
        parity["prediction_disagreements"] != 0
        or parity["before_prediction_disagreements"] != 0
    ):
        raise RuntimeError(f"D1 compile parity failed for {task['row_id']}")
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
        "d1_historical_parity": parity,
    }


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.max_workers <= 8:
        raise ValueError("max-workers must be between 1 and 8")
    feature_roots = [Path(args.feature_root).absolute()]
    if args.supplemental_feature_root:
        feature_roots.append(Path(args.supplemental_feature_root).absolute())
    roots = tuple(feature_roots)
    output_root = Path(args.output_root).absolute()
    output_root.mkdir(parents=True, exist_ok=False)
    tasks: list[dict[str, Any]] = []
    for receiver in DEFAULT_RECEIVERS:
        for method_seed in DEFAULT_SEEDS:
            for k_shot, new_count in DEFAULT_CONDITIONS:
                cache_root = _cache_root(roots, receiver, method_seed, k_shot, new_count)
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
        "historical_full_rf32_role": "HISTORICAL_COMPARATOR_ONLY",
        "receivers": list(DEFAULT_RECEIVERS),
        "method_seeds": list(DEFAULT_SEEDS),
        "conditions": [{"k_shot": k, "new_class_count": n} for k, n in DEFAULT_CONDITIONS],
        "arms": list(EVIDENCE_ARMS),
        "feature_roots": [str(root) for root in roots],
        "entries": completed,
        "query_truth_opened": False,
    }
    _write_exclusive(output_root / "matrix_index.json", matrix)
    print(json.dumps({"status": matrix["status"], "row_count": len(completed)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
