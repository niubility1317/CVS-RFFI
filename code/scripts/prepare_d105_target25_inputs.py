#!/usr/bin/env python3
"""Prepare one immutable D105 Target25 plan/context from a SHA-bound D92 index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi.stage2_d105_target25_inputs import (  # noqa: E402
    D105Target25InputError,
    prepare_d105_target25_inputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d92-matrix-index", type=Path, required=True)
    parser.add_argument("--d92-matrix-index-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = prepare_d105_target25_inputs(
            matrix_index_path=args.d92_matrix_index,
            expected_matrix_index_sha256=args.d92_matrix_index_sha256,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (D105Target25InputError, FileExistsError, OSError) as error:
        print(f"prepare_d105_target25_inputs: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
