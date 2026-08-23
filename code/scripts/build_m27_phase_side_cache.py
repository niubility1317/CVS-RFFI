#!/usr/bin/env python3
"""Build one truth-free M2.7 Phase32 cache from sealed Stage2-C IQ."""

from __future__ import annotations

import argparse
import json

from cvsrffi.stage2_m27_phase_builder import (
    build_phase_side_cache_from_sealed_stage2c,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-feature-cache-payload", required=True)
    parser.add_argument("--base-feature-cache-manifest", required=True)
    parser.add_argument("--base-feature-cache-payload-sha256", required=True)
    parser.add_argument("--base-feature-cache-manifest-sha256", required=True)
    parser.add_argument("--after-package-root", required=True)
    parser.add_argument("--after-seal-path", required=True)
    parser.add_argument("--after-seal-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = build_phase_side_cache_from_sealed_stage2c(
        base_feature_cache_payload=args.base_feature_cache_payload,
        base_feature_cache_manifest=args.base_feature_cache_manifest,
        base_feature_cache_payload_sha256=args.base_feature_cache_payload_sha256,
        base_feature_cache_manifest_sha256=args.base_feature_cache_manifest_sha256,
        after_package_root=args.after_package_root,
        after_seal_path=args.after_seal_path,
        after_seal_sha256=args.after_seal_sha256,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
