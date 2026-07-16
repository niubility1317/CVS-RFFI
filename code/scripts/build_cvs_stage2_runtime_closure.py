#!/usr/bin/env python
"""Build the exact, read-only Phase2 predictor Python runtime closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase2_runtime_closure import (  # noqa: E402
    RUNTIME_MEMBER_ALLOWLIST_BY_PROFILE,
    build_phase2_runtime_closure,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-code-root",
        type=Path,
        default=CODE_ROOT,
        help="Repository code root containing cvsrffi/ and scripts/.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=sorted(RUNTIME_MEMBER_ALLOWLIST_BY_PROFILE),
        default="stage2_predictor",
    )
    args = parser.parse_args()
    result = build_phase2_runtime_closure(
        args.source_code_root,
        args.output_root,
        profile=args.profile,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
