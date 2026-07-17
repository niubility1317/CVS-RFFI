#!/usr/bin/env python
"""Build one sealed formal SOMP-H before/after row pair from verified cache IQ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.somph_offline_target_package import (  # noqa: E402
    build_somph_offline_row_pair,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--authority-bundle", required=True)
    parser.add_argument("--authority-commit-sha256", required=True)
    parser.add_argument("--phase1-checkpoint", required=True)
    parser.add_argument("--sealed-runtime", required=True)
    parser.add_argument("--method-lock", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k-shot", type=int, required=True)
    parser.add_argument("--new-count", type=int, required=True)
    parser.add_argument("--query-per-tx", type=int, default=20)
    args = parser.parse_args()

    result = build_somph_offline_row_pair(
        cache_set_manifest_path=args.cache_manifest,
        authority_bundle_root=args.authority_bundle,
        expected_authority_commit_sha256=args.authority_commit_sha256,
        phase1_checkpoint_path=args.phase1_checkpoint,
        sealed_feature_runtime_path=args.sealed_runtime,
        method_lock_path=args.method_lock,
        output_root=args.output_root,
        receiver=args.receiver,
        seed=args.seed,
        k_shot=args.k_shot,
        new_class_count=args.new_count,
        query_per_tx=args.query_per_tx,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
