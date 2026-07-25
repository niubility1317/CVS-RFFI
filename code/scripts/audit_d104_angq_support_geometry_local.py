#!/usr/bin/env python3
"""Development-only fixed-tap geometry audit for D104 ANGQ.

The bound tap contains historically exposed diagnostic rows. The algorithm
does not receive their role, labels, new formal held rows, or truth.
"""

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

from cvsrffi.stage2_zid_student_t_qknn import (  # noqa: E402
    _quantize_rows,
    normalize_zid_rows,
)
from probe_d103_r2_outer_geometry_local import (  # noqa: E402
    _angular_grid_decode,
)

EXPECTED_TAP_SHA256 = (
    "c6807d9156ab3ac8f7005707a3bd7eec342d2e4f0a43d4b96d5ea8a9574ec4c1"
)
EXPECTED_TAP_ROWS = 8400
HISTORICAL_DIAGNOSTIC_QUERY_COUNT = 2478
HISTORICAL_DIAGNOSTIC_QUERY_ID_ROOT_SHA256 = (
    "036456779eea6594f2330f2e9a96cceda580088b0d451982198e3056f762854d"
)


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
        raise FileExistsError(f"immutable audit output exists: {output}")
    tap_sha256 = _sha256(tap_path)
    if tap_sha256 != EXPECTED_TAP_SHA256:
        raise ValueError("D104 geometry audit tap SHA drift")
    with np.load(tap_path, allow_pickle=False) as archive:
        support = np.asarray(archive["z_id"], dtype=np.float32)
    if len(support) != EXPECTED_TAP_ROWS:
        raise ValueError("D104 geometry audit tap row-count drift")
    normalized = normalize_zid_rows(support)
    started = time.monotonic()
    _, _, legacy = _quantize_rows(normalized)
    angular, factors, angular_cosine = _angular_grid_decode(normalized)
    legacy_cosine = np.sum(
        normalized.astype(np.float64) * legacy.astype(np.float64),
        axis=1,
    )
    result = {
        "schema": "cvs.d104_r1.angq.support_geometry_local_audit.v2",
        "status": "DEVELOPMENT_ONLY_SUPPORT_GEOMETRY_COMPLETE",
        "candidate": "D104-R1-ANGQ-RXID-MB4",
        "tap_archive_sha256": tap_sha256,
        "row_count": int(len(normalized)),
        "legacy_cosine_min": float(np.min(legacy_cosine)),
        "legacy_cosine_mean": float(np.mean(legacy_cosine)),
        "angular_cosine_min": float(np.min(angular_cosine)),
        "angular_cosine_mean": float(np.mean(angular_cosine)),
        "strictly_improved_rows": int(
            np.sum(angular_cosine > legacy_cosine + 1.0e-12)
        ),
        "equal_rows": int(
            np.sum(np.abs(angular_cosine - legacy_cosine) <= 1.0e-12)
        ),
        "regressed_rows": int(
            np.sum(angular_cosine < legacy_cosine - 1.0e-12)
        ),
        "factor_min": float(np.min(factors)),
        "factor_max": float(np.max(factors)),
        "factor_mean": float(np.mean(factors)),
        "decoded_norm_abs_error_max": float(
            np.max(
                np.abs(
                    np.linalg.norm(angular.astype(np.float64), axis=1)
                    - 1.0
                )
            )
        ),
        "elapsed_seconds": time.monotonic() - started,
        "query_argument_used": False,
        "new_formal_held_query_features_used": 0,
        "historical_diagnostic_query_features_in_input": (
            HISTORICAL_DIAGNOSTIC_QUERY_COUNT
        ),
        "historical_diagnostic_query_id_root_sha256": (
            HISTORICAL_DIAGNOSTIC_QUERY_ID_ROOT_SHA256
        ),
        "query_role_used_by_algorithm": False,
        "query_truth_read": False,
        "query_state_updates": 0,
        "performance_computed": False,
        "target_access": False,
        "formal_held_evidence": False,
        "n607_run": False,
    }
    if result["regressed_rows"] != 0:
        raise ValueError("deploy-isomorphic ANGQ regressed support geometry")
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
