#!/usr/bin/env python
"""Validate exact40 coverage of one built SOMP-H registered cache cell."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.somph_cache_build_matrix import (  # noqa: E402
    FIXED_N607_CACHE_OUTPUT_ROOT,
    SEEDS,
    validate_registered_cache_coverage,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if isinstance(args.seed, bool) or args.seed not in SEEDS:
        raise ValueError("seed is outside the fixed 713101-713106 grid")
    cache_set = (
        Path(FIXED_N607_CACHE_OUTPUT_ROOT)
        / f"rx_{args.receiver.replace('-', '_')}"
        / f"seed_{args.seed}"
        / "cache_set.json"
    )
    audit = validate_registered_cache_coverage(
        cache_set,
        expected_receiver=args.receiver,
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
