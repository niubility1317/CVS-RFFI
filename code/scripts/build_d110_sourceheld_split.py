#!/usr/bin/env python3
"""Build the immutable D110 source-held scorer archive without scoring it."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvsrffi.rxid_metabias4_source_archive import _sha256_file  # noqa: E402
from cvsrffi.stage2_d110_sourceheld_split import (  # noqa: E402
    D110SourceHeldSplitError,
    load_d104_held_ids,
    load_historical_query_ids,
    load_physical_ids_only,
    publish_d110_sourceheld_split,
    validate_source_feature_pool,
)


def _sha(value: str, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise argparse.ArgumentTypeError(f"{name} must be a SHA256")
    return normalized


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(strict=True), allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _bind_file(path: Path, expected: str, name: str) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    actual = _sha256_file(resolved)
    if actual != expected:
        raise D110SourceHeldSplitError(f"{name} file SHA drift")
    return {"path": str(resolved), "sha256": actual}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-archive", type=Path, required=True)
    parser.add_argument("--dual-archive-sha256", required=True)
    parser.add_argument("--strict-tap-archive", type=Path, required=True)
    parser.add_argument("--strict-tap-archive-sha256", required=True)
    parser.add_argument("--historical-query-manifest", type=Path, required=True)
    parser.add_argument("--historical-query-manifest-sha256", required=True)
    parser.add_argument("--d104-held-package-manifest", type=Path, required=True)
    parser.add_argument("--d104-held-package-manifest-sha256", required=True)
    parser.add_argument("--d104-held-package-root", type=Path)
    parser.add_argument("--d104-ls-id-source", type=Path, required=True)
    parser.add_argument("--d104-ls-id-source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = {
        "dual_archive": _sha(args.dual_archive_sha256, "dual archive"),
        "strict_tap_archive": _sha(args.strict_tap_archive_sha256, "strict tap"),
        "historical_query_manifest": _sha(
            args.historical_query_manifest_sha256, "historical-query manifest"
        ),
        "d104_held_package_manifest": _sha(
            args.d104_held_package_manifest_sha256, "D104 package manifest"
        ),
        "d104_ls_id_source": _sha(args.d104_ls_id_source_sha256, "D104 L_s ID source"),
    }
    paths = {
        "dual_archive": args.dual_archive,
        "strict_tap_archive": args.strict_tap_archive,
        "historical_query_manifest": args.historical_query_manifest,
        "d104_held_package_manifest": args.d104_held_package_manifest,
        "d104_ls_id_source": args.d104_ls_id_source,
    }
    input_files = {
        name: _bind_file(paths[name], expected[name], name) for name in paths
    }
    dual = _load_npz(args.dual_archive)
    strict = _load_npz(args.strict_tap_archive)
    source_pool, validation = validate_source_feature_pool(dual, strict)
    historical = load_historical_query_ids(args.historical_query_manifest)
    d104_held = load_d104_held_ids(
        args.d104_held_package_manifest,
        package_root=args.d104_held_package_root,
    )
    d110_ls = load_physical_ids_only(args.d104_ls_id_source)
    result = publish_d110_sourceheld_split(
        source_pool,
        historical_query_ids=historical,
        d104_held_ids=d104_held,
        d110_ls_ids=d110_ls,
        validation_receipt=validation,
        input_files=input_files,
        output_dir=args.output_dir,
    )
    print(result["selection_receipt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

