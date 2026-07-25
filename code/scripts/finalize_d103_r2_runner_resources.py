#!/usr/bin/env python3
"""Finalize exact runner resource counters after all held artifacts exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


POST_RESOURCE_ANALYSIS_RESERVE_BYTES = 16 * 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-status", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status = json.loads(
        args.matrix_status.resolve(strict=True).read_text(encoding="utf-8")
    )
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
        "total_gpu_hours": float(status["total_gpu_hours"]),
        "peak_memory_bytes": int(status["peak_memory_bytes"]),
        "run_root_bytes": int(run_root_bytes),
        "completed_fit_count": 246,
        "completed_meta_steps": 98_400,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
