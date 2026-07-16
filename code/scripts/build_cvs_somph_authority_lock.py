#!/usr/bin/env python
"""Build one unsigned SOMP-H authority lock from fixed offline code roots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.somph_authority_lock_builder import (  # noqa: E402
    write_somph_authority_lock_package,
)
from cvsrffi.somph_lineage_authority import (  # noqa: E402
    CHANNEL_CODE_LOGICAL_MEMBERS,
)


FIXED_EXPORTER_PATH = CODE_ROOT / "scripts" / "build_cvs_leo_weak_iq_cache.py"
FIXED_CHANNEL_CODE_MEMBERS = {
    "cvsrffi_eval.py": CODE_ROOT / "cvsrffi" / "eval.py",
    "cvsrffi_tensors.py": CODE_ROOT / "cvsrffi" / "tensors.py",
    "sat_channel.py": CODE_ROOT / "sat_channel.py",
    "training_controls.py": CODE_ROOT / "training_controls.py",
}
if tuple(FIXED_CHANNEL_CODE_MEMBERS) != CHANNEL_CODE_LOGICAL_MEMBERS:
    raise RuntimeError("fixed SOMP-H channel closure order drift")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-set", type=Path, required=True)
    parser.add_argument("--cache-spec-manifest", type=Path, required=True)
    parser.add_argument("--cache-spec-cell-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = write_somph_authority_lock_package(
        args.cache_set,
        cache_spec_manifest_path=args.cache_spec_manifest,
        cache_spec_cell_id=args.cache_spec_cell_id,
        exporter_path=FIXED_EXPORTER_PATH,
        channel_code_members=FIXED_CHANNEL_CODE_MEMBERS,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
