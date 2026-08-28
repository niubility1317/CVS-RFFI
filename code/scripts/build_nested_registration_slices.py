#!/usr/bin/env python3
"""Build nested new1/2/3/5/10/15/20 Phase2-C slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_nested_registration_builder import (  # noqa: E402
    build_nested_registration_slices,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-support", type=Path, required=True)
    parser.add_argument("--max-query", type=Path, required=True)
    parser.add_argument("--truth-sidecar", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--capsule-id", required=True)
    parser.add_argument("--da-split-id", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    args = parser.parse_args()
    result = build_nested_registration_slices(
        max_support_path=args.max_support,
        max_query_path=args.max_query,
        truth_sidecar_path=args.truth_sidecar,
        scenario=args.scenario,
        output_root=args.output_root,
        capsule_id=args.capsule_id,
        da_split_id=args.da_split_id,
        base_checkpoint_path=args.base_checkpoint,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
