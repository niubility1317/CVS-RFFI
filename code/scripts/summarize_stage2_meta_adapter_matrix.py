"""CLI for Target5/Target25 aggregation of closed meta-adapter scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_meta_adapter_scorer import (  # noqa: E402
    MetaAdapterScoringError,
    summarize_meta_adapter_matrix,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", nargs="+", required=True, type=Path)
    parser.add_argument("--target", required=True, choices=("Target5", "Target25"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"summary output already exists: {args.output}")
        payloads = [
            json.loads(path.read_text(encoding="utf-8-sig"))
            for path in args.scores
        ]
        decision = summarize_meta_adapter_matrix(
            payloads,
            expected_target=args.target,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(decision.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except (MetaAdapterScoringError, FileExistsError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
