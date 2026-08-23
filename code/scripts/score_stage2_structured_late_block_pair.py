#!/usr/bin/env python3
"""Score an immutable DA0_REG0/DA1_REG0 late-block prediction pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_structured_late_block_pair_scorer import score_prediction_pair


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--da0", required=True)
    parser.add_argument("--da1", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = score_prediction_pair(
        args.da0,
        args.da1,
        args.truth,
        scenario=args.scenario,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

