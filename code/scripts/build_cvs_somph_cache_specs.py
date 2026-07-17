#!/usr/bin/env python
"""Write the fixed 30-cell SOMP-H registered LEO_weak cache build specs."""

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
    MANIFEST_NAME,
    write_cache_build_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manysig-pkl", required=True)
    parser.add_argument("--manytx-pkl", required=True)
    parser.add_argument(
        "--cache-output-root",
        default=FIXED_N607_CACHE_OUTPUT_ROOT,
        help="Offline cache destination embedded into all 30 build specs.",
    )
    args = parser.parse_args()
    manifest = write_cache_build_matrix(
        output_root=args.output_root,
        manysig_pkl=args.manysig_pkl,
        manytx_pkl=args.manytx_pkl,
        cache_output_root=args.cache_output_root,
    )
    print(
        json.dumps(
            {
                "manifest": str((args.output_root.resolve() / MANIFEST_NAME)),
                "cell_count": manifest["cell_count"],
                "control_status": manifest["control_status"],
                "formal_launch_authority": manifest["formal_launch_authority"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
