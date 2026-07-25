#!/usr/bin/env python3
"""Verify D104 c=1 ABI parity on the immutable 8400-row development tap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvsrffi.stage2_d104_angq_qknn import _decode_candidate  # noqa: E402
from cvsrffi.stage2_zid_student_t_qknn import (  # noqa: E402
    _quantize_rows,
    normalize_zid_rows,
)


EXPECTED_TAP_SHA256 = (
    "c6807d9156ab3ac8f7005707a3bd7eec342d2e4f0a43d4b96d5ea8a9574ec4c1"
)
EXPECTED_TAP_ROWS = 8400
HISTORICAL_DIAGNOSTIC_EXPOSED_ROWS = 2478


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tap-archive", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tap_path = args.tap_archive.resolve(strict=True)
    output = args.output_json.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable verification output exists: {output}")
    tap_sha256 = _sha256(tap_path)
    if tap_sha256 != EXPECTED_TAP_SHA256:
        raise ValueError("D104 c=1 verification tap SHA drift")
    with np.load(tap_path, allow_pickle=False) as archive:
        raw = np.asarray(archive["z_id"])
    if (
        raw.dtype != np.float32
        or raw.ndim != 2
        or raw.shape != (EXPECTED_TAP_ROWS, 160)
        or not np.isfinite(raw).all()
    ):
        raise ValueError("D104 c=1 verification tap array drift")

    started = time.monotonic()
    normalized = normalize_zid_rows(raw)
    legacy_codes, legacy_scales, legacy_decoded = _quantize_rows(normalized)
    c1_codes = np.empty_like(legacy_codes)
    c1_scales = np.empty_like(legacy_scales)
    c1_decoded = np.empty_like(legacy_decoded)
    for index, row in enumerate(normalized):
        scale, code, decoded, _ = _decode_candidate(row, 1.0)
        c1_scales[index] = scale
        c1_codes[index] = code
        c1_decoded[index] = decoded

    scale_changed = np.any(
        c1_scales.view(np.uint16) != legacy_scales.view(np.uint16),
        axis=0,
    )
    code_changed_rows = np.any(c1_codes != legacy_codes, axis=1)
    decoded_changed_rows = np.any(
        c1_decoded.view(np.uint32) != legacy_decoded.view(np.uint32),
        axis=1,
    )
    result = {
        "schema": "cvs.d104_r1.angq.c1_full_tap_verification.v1",
        "status": "VERIFIED" if not (
            bool(scale_changed)
            or bool(np.any(code_changed_rows))
            or bool(np.any(decoded_changed_rows))
        ) else "FAILED",
        "candidate": "D104-R1-ANGQ-RXID-MB4",
        "tap_archive_sha256": tap_sha256,
        "row_count": int(len(normalized)),
        "input_normalization_count": 1,
        "candidate_input_renormalization_count": 0,
        "c1_factor": 1.0,
        "scale_changed_rows": int(np.sum(c1_scales != legacy_scales)),
        "code_changed_rows": int(np.sum(code_changed_rows)),
        "decoded_changed_rows": int(np.sum(decoded_changed_rows)),
        "scale_bitwise_equal": bool(
            np.array_equal(
                c1_scales.view(np.uint16),
                legacy_scales.view(np.uint16),
            )
        ),
        "code_bitwise_equal": bool(np.array_equal(c1_codes, legacy_codes)),
        "decoded_bitwise_equal": bool(
            np.array_equal(
                c1_decoded.view(np.uint32),
                legacy_decoded.view(np.uint32),
            )
        ),
        "historical_diagnostic_exposed_rows_in_tap": (
            HISTORICAL_DIAGNOSTIC_EXPOSED_ROWS
        ),
        "new_formal_held_rows_used": 0,
        "truth_read": False,
        "performance_computed": False,
        "target_access": False,
        "formal_held_evidence": False,
        "n607_run": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    if result["status"] != "VERIFIED":
        raise ValueError("D104 c=1 full-tap bitwise parity failed")
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
