#!/usr/bin/env python3
"""Run the deterministic source-only D102 held falsifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi.phase1_rb_metabias4_bundle import (  # noqa: E402
    RBMetaBias4Config,
    merge_verified_phase1_tap_and_dual_archives,
    sha256_file,
)
from cvsrffi.rb_metabias4_phase1_held_falsifier import (  # noqa: E402
    run_rb_metabias4_phase1_held_falsifier,
)


def _sha(value: str, name: str) -> str:
    result = value.strip().lower()
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise ValueError(f"{name} must be SHA256")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tap-archive", type=Path, required=True)
    parser.add_argument("--tap-archive-sha256", required=True)
    parser.add_argument("--dual-archive", type=Path, required=True)
    parser.add_argument("--dual-archive-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--method-lock-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    tap_expected = _sha(args.tap_archive_sha256, "tap archive")
    dual_expected = _sha(args.dual_archive_sha256, "dual archive")
    checkpoint_sha256 = _sha(args.checkpoint_sha256, "checkpoint")
    runtime_sha256 = _sha(args.runtime_sha256, "runtime")
    method_lock_sha256 = _sha(args.method_lock_sha256, "method lock")
    if (
        not args.tap_archive.is_file()
        or args.tap_archive.is_symlink()
        or sha256_file(args.tap_archive) != tap_expected
    ):
        raise ValueError("tap archive byte binding drift")
    receipt = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or receipt.exists():
        raise ValueError(f"output already exists: {args.output}")
    with np.load(args.tap_archive, allow_pickle=False) as archive:
        tap_arrays = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }
    if (
        not args.dual_archive.is_file()
        or args.dual_archive.is_symlink()
        or sha256_file(args.dual_archive) != dual_expected
    ):
        raise ValueError("dual archive byte binding drift")
    with np.load(args.dual_archive, allow_pickle=False) as archive:
        dual_arrays = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }
    arrays = merge_verified_phase1_tap_and_dual_archives(
        tap_arrays, dual_arrays
    )
    report = run_rb_metabias4_phase1_held_falsifier(
        arrays,
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=runtime_sha256,
        method_lock_sha256=method_lock_sha256,
        config=RBMetaBias4Config(),
    )
    body = json.dumps(
        report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(body)
    receipt.write_text(
        f"{hashlib.sha256(body).hexdigest()}  {args.output.name}\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "receipt_sha256": report["receipt_sha256"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
