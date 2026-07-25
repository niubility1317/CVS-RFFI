#!/usr/bin/env python3
"""Finalize exact runner resource counters after all held artifacts exist."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


POST_RESOURCE_ANALYSIS_RESERVE_BYTES = 16 * 1024**2
GPU_HOUR_LIMIT = 30.0
PEAK_MEMORY_LIMIT = 4 * 1024**3
DISK_LIMIT = 20 * 1024**3


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-status", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix_status_path = args.matrix_status.resolve(strict=True)
    status = json.loads(matrix_status_path.read_text(encoding="utf-8"))
    root = args.run_root.resolve(strict=True)
    output = args.output_json.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable resource receipt exists: {output}")
    if (
        status.get("status") != "ARTIFACTS_COMPLETE"
        or status.get("completed_fit_count") != 246
        or status.get("completed_meta_steps") != 98_400
    ):
        raise ValueError("resource receipt requires a complete fit matrix")
    measured_run_root_bytes = sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    # The scorer, gate, and archive receipts are written after this receipt.
    # Charge a conservative reserve so the formal disk gate cannot under-report
    # the final immutable run tree.
    run_root_bytes = (
        measured_run_root_bytes + POST_RESOURCE_ANALYSIS_RESERVE_BYTES
    )
    receipt = {
        "schema": "cvs.d103_r2.rxid_crossreceiver.runner_resources.v1",
        "status": "RUNNER_RESOURCES_COMPLETE",
        "matrix_status_sha256": _sha256_file(matrix_status_path),
        "total_gpu_hours": float(status["total_gpu_hours"]),
        "peak_memory_bytes": int(status["peak_memory_bytes"]),
        "run_root_bytes": int(run_root_bytes),
        "completed_fit_count": 246,
        "completed_meta_steps": 98_400,
        "limits": {
            "total_gpu_hours": GPU_HOUR_LIMIT,
            "peak_memory_bytes": PEAK_MEMORY_LIMIT,
            "run_root_bytes": DISK_LIMIT,
        },
    }
    receipt["passes_runner_resource_gate"] = bool(
        receipt["total_gpu_hours"] <= GPU_HOUR_LIMIT
        and receipt["peak_memory_bytes"] <= PEAK_MEMORY_LIMIT
        and receipt["run_root_bytes"] <= DISK_LIMIT
    )
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
