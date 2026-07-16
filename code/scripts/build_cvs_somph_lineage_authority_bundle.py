#!/usr/bin/env python
"""Publish one signed SOMP-H authority bundle with its build-authority evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.somph_lineage_authority import (  # noqa: E402
    write_somph_lineage_authority_bundle,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-lock", type=Path, required=True)
    parser.add_argument(
        "--signed-authority-envelope",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-signed-authority-envelope-sha256",
        required=True,
    )
    parser.add_argument(
        "--authority-lock-build-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--cache-spec-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = write_somph_lineage_authority_bundle(
        args.authority_lock,
        signed_authority_envelope_path=args.signed_authority_envelope,
        expected_signed_authority_envelope_sha256=(
            args.expected_signed_authority_envelope_sha256
        ),
        authority_lock_build_receipt_path=(
            args.authority_lock_build_receipt
        ),
        cache_spec_manifest_path=args.cache_spec_manifest,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
