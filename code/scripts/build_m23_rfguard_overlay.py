#!/usr/bin/env python3
"""Build one immutable ERBT-IDR M2.3 RFGuard overlay."""

from __future__ import annotations

import argparse
import json

from cvsrffi.stage2_m23_overlay_builder import build_m23_overlay_from_sealed_inputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-feature-cache-payload", required=True)
    parser.add_argument("--base-feature-cache-manifest", required=True)
    parser.add_argument("--base-feature-cache-payload-sha256", required=True)
    parser.add_argument("--base-feature-cache-manifest-sha256", required=True)
    parser.add_argument("--predictor-package-root", required=True)
    parser.add_argument("--predictor-seal-path", required=True)
    parser.add_argument("--predictor-seal-sha256", required=True)
    parser.add_argument("--phase1-component-dir", required=True)
    parser.add_argument("--expected-phase1-component-manifest-sha256", required=True)
    parser.add_argument("--overlay-payload-path", required=True)
    parser.add_argument("--overlay-manifest-path", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = build_m23_overlay_from_sealed_inputs(
        base_feature_cache_payload=args.base_feature_cache_payload,
        base_feature_cache_manifest=args.base_feature_cache_manifest,
        base_feature_cache_payload_sha256=args.base_feature_cache_payload_sha256,
        base_feature_cache_manifest_sha256=args.base_feature_cache_manifest_sha256,
        predictor_package_root=args.predictor_package_root,
        predictor_seal_path=args.predictor_seal_path,
        predictor_seal_sha256=args.predictor_seal_sha256,
        phase1_component_dir=args.phase1_component_dir,
        expected_phase1_component_manifest_sha256=(
            args.expected_phase1_component_manifest_sha256
        ),
        overlay_payload_path=args.overlay_payload_path,
        overlay_manifest_path=args.overlay_manifest_path,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
