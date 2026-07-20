#!/usr/bin/env python3
"""Audit and summarize the locked D92 125-job stability screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts import summarize_cvs_somph_diag_125_stability as base
from cvsrffi.stage2_d92_query_evaluation import CANDIDATE_D92


def main() -> int:
    base.CANDIDATE = CANDIDATE_D92
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = base.summarize(args.matrix_manifest, args.output_root)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
