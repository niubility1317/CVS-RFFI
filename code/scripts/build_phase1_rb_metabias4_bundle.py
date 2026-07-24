#!/usr/bin/env python3
"""Build one immutable D102 Phase1 aggregate component from a tap archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi.phase1_rb_metabias4_bundle import (  # noqa: E402
    RBMetaBias4Config,
    build_phase1_rb_metabias4_bundle,
    merge_verified_phase1_tap_and_dual_archives,
    save_phase1_rb_metabias4_bundle,
    sha256_file,
)


def _sha(value: str, name: str) -> str:
    result = value.strip().lower()
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise ValueError(f"{name} must be lowercase SHA256")
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--trust-radius", type=float, default=0.35)
    args = parser.parse_args(argv)
    expected = _sha(args.tap_archive_sha256, "tap archive")
    checkpoint_sha256 = _sha(args.checkpoint_sha256, "checkpoint")
    runtime_sha256 = _sha(args.runtime_sha256, "runtime")
    method_lock_sha256 = _sha(args.method_lock_sha256, "method lock")
    if (
        not args.tap_archive.is_file()
        or args.tap_archive.is_symlink()
        or sha256_file(args.tap_archive) != expected
    ):
        raise ValueError("tap archive byte binding drift")
    with np.load(args.tap_archive, allow_pickle=False) as archive:
        tap_arrays = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }
    dual_expected = _sha(args.dual_archive_sha256, "dual archive")
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
    config = RBMetaBias4Config(
        temperature=args.temperature, trust_radius=args.trust_radius
    )
    bundle = build_phase1_rb_metabias4_bundle(
        arrays,
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=runtime_sha256,
        method_lock_sha256=method_lock_sha256,
        config=config,
    )
    result = save_phase1_rb_metabias4_bundle(args.output_dir, bundle)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
