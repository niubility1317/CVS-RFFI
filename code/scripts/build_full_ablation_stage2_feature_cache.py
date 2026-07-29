#!/usr/bin/env python3
"""Build three minimal stage-scoped caches from one sealed feature extraction."""

from __future__ import annotations

import argparse
import json

from cvsrffi.stage2_ablation_feature_builder import (
    build_feature_cache_from_sealed_row_pair,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for state in ("before", "after"):
        prefix = f"--{state}"
        parser.add_argument(prefix + "-package-root", required=True)
        parser.add_argument(prefix + "-seal-path", required=True)
        parser.add_argument(prefix + "-seal-sha256", required=True)
    parser.add_argument(
        "--phase1-deployment-binding-path",
        required=True,
    )
    parser.add_argument("--ground-component-dir", required=True)
    parser.add_argument("--ground-manifest-sha256", required=True)
    parser.add_argument("--phase1-prototype-path", required=True)
    parser.add_argument(
        "--phase1-prototype-manifest-path",
        required=True,
    )
    parser.add_argument(
        "--expected-phase1-prototype-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-phase1-prototype-manifest-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-phase1-bundle-sha256",
        required=True,
    )
    parser.add_argument("--cache-output-root", required=True)
    parser.add_argument(
        "--phase2-data-status",
        required=True,
        choices=("VALIDATED_ONCE",),
    )
    parser.add_argument("--capsule-id", required=True)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--k-shot", type=int, required=True)
    parser.add_argument("--method-seed", type=int, required=True)
    parser.add_argument("--support-seed", type=int, required=True)
    parser.add_argument("--query-seed", type=int, required=True)
    parser.add_argument(
        "--new-class-draw-seed",
        type=int,
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    args = _parser().parse_args()
    receipt = build_feature_cache_from_sealed_row_pair(
        before_package_root=args.before_package_root,
        before_seal_path=args.before_seal_path,
        before_seal_sha256=args.before_seal_sha256,
        after_package_root=args.after_package_root,
        after_seal_path=args.after_seal_path,
        after_seal_sha256=args.after_seal_sha256,
        phase1_deployment_binding_path=(
            args.phase1_deployment_binding_path
        ),
        ground_component_dir=args.ground_component_dir,
        ground_manifest_sha256=args.ground_manifest_sha256,
        phase1_prototype_path=args.phase1_prototype_path,
        phase1_prototype_manifest_path=(
            args.phase1_prototype_manifest_path
        ),
        expected_phase1_prototype_sha256=(
            args.expected_phase1_prototype_sha256
        ),
        expected_phase1_prototype_manifest_sha256=(
            args.expected_phase1_prototype_manifest_sha256
        ),
        expected_phase1_bundle_sha256=(
            args.expected_phase1_bundle_sha256
        ),
        cache_output_root=args.cache_output_root,
        phase2_data_status=args.phase2_data_status,
        capsule_id=args.capsule_id,
        split_id=args.split_id,
        k_shot=args.k_shot,
        method_seed=args.method_seed,
        support_seed=args.support_seed,
        query_seed=args.query_seed,
        new_class_draw_seed=args.new_class_draw_seed,
        device=args.device,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
