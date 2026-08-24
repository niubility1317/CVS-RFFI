"""CLI for truth-last scoring of one tri-R4 DA0/DA1 row."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_meta_adapter_scorer import (  # noqa: E402
    MetaAdapterScoringError,
    score_meta_adapter_pair,
    write_score_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--da0", required=True, type=Path)
    parser.add_argument("--da1", required=True, type=Path)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="Task10 receipt.json; defaults to the prediction directory",
    )
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"scoring output already exists: {args.output}")
        score = score_meta_adapter_pair(
            args.da0,
            args.da1,
            args.truth,
            receipt_path=args.receipt,
        )
        write_score_json(score, args.output)
    except (MetaAdapterScoringError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
