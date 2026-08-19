#!/usr/bin/env python3
"""Run the base-cache-only M2.4 D1 expansion without opening query truth."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cvsrffi.stage2_m24_row_executor import run_m24_d1_row_from_base_cache


MATRIX_SCHEMA = "cvs.erbt_idr.m24.d1_expanded_prediction_matrix.v1"
DEFAULT_RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
DEFAULT_SEEDS = (7282101, 7282102, 7282103)
DEFAULT_CONDITIONS = ((1, 20), (5, 20), (10, 20), (10, 5))


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


def _run_entry(entry: dict[str, Any]) -> dict[str, Any]:
    receipt = run_m24_d1_row_from_base_cache(
        row_id=entry["row_id"],
        receiver=entry["receiver"],
        base_feature_cache_payload=entry["payload_path"],
        base_feature_cache_manifest=entry["manifest_path"],
        base_feature_cache_payload_sha256=entry["payload_sha256"],
        base_feature_cache_manifest_sha256=entry["manifest_sha256"],
        output_root=entry["output_root"],
        seed=entry["method_seed"],
        device="cpu",
    )
    parity = receipt["d1_historical_parity"]
    if parity["prediction_disagreements"] != 0:
        raise RuntimeError(f"D1 parity failed for {entry['row_id']}")
    return {
        **{key: entry[key] for key in ("row_id", "receiver", "method_seed", "k_shot", "new_class_count")},
        "support_seed": entry["support_seed"],
        "query_seed": entry["query_seed"],
        "new_class_draw_seed": entry["new_class_draw_seed"],
        "split_id": entry["split_id"],
        "receipt_path": str(Path(entry["output_root"]) / "row_execution_receipt.json"),
        "prediction_path": receipt["prediction"]["path"],
        "prediction_artifact_sha256": receipt["prediction"]["artifact_sha256"],
        "prediction_seal_sha256": receipt["prediction"]["seal_sha256"],
        "d1_historical_parity": parity,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    return parser


def main() -> int:
    args = _parser().parse_args()
    feature_root = Path(args.feature_root).absolute()
    output_root = Path(args.output_root).absolute()
    output_root.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, Any]] = []
    for receiver in DEFAULT_RECEIVERS:
        for method_seed in DEFAULT_SEEDS:
            for k_shot, new_count in DEFAULT_CONDITIONS:
                cache_root = (
                    feature_root
                    / f"rx_{receiver.replace('-', '_')}"
                    / f"method_{method_seed}"
                    / f"new{new_count}"
                    / f"k_{k_shot}"
                    / "stage2c"
                )
                manifest_path = cache_root / "features.manifest.json"
                payload_path = cache_root / "features.npz"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                row_id = f"rx{receiver}_m{method_seed}_k{k_shot}_new{new_count}__M24-D1-PHYSICAL256-F1"
                entries.append({
                    "row_id": row_id,
                    "receiver": receiver,
                    "method_seed": method_seed,
                    "k_shot": k_shot,
                    "new_class_count": new_count,
                    "support_seed": int(manifest["support_seed"]),
                    "query_seed": int(manifest["query_seed"]),
                    "new_class_draw_seed": int(manifest["new_class_draw_seed"]),
                    "split_id": str(manifest["split_id"]),
                    "payload_path": str(payload_path),
                    "manifest_path": str(manifest_path),
                    "payload_sha256": str(manifest["payload_sha256"]),
                    "manifest_sha256": _canonical_manifest_sha(manifest_path),
                    "output_root": str(output_root / row_id),
                })
    completed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        futures = {executor.submit(_run_entry, entry): entry["row_id"] for entry in entries}
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(json.dumps({"completed": result["row_id"], "count": len(completed)}, sort_keys=True), flush=True)
    completed.sort(key=lambda item: item["row_id"])
    matrix = {
        "schema": MATRIX_SCHEMA,
        "status": "PREDICTIONS_COMPLETE_TRUTH_UNOPENED",
        "row_count": len(completed),
        "receivers": list(DEFAULT_RECEIVERS),
        "method_seeds": list(DEFAULT_SEEDS),
        "conditions": [{"k_shot": k, "new_class_count": n} for k, n in DEFAULT_CONDITIONS],
        "entries": completed,
        "query_truth_opened": False,
    }
    _write_exclusive(output_root / "matrix_index.json", matrix)
    print(json.dumps({"status": matrix["status"], "row_count": len(completed)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
