#!/usr/bin/env python3
"""Seal a current-schema Stage2-B sidecar from validated Stage2-B truth."""

from __future__ import annotations

import argparse
import json

from cvsrffi.stage2_ablation_scoring_sidecars import (
    publish_stage2b_scoring_sidecar,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stage2b-truth", required=True)
    parser.add_argument("--expected-source-truth-sha256", required=True)
    parser.add_argument("--predictor-package-root-sha256", required=True)
    parser.add_argument("--predictor-package-seal-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    receipt = publish_stage2b_scoring_sidecar(
        source_stage2b_truth_path=args.source_stage2b_truth,
        expected_source_truth_sha256=args.expected_source_truth_sha256,
        predictor_package_root_sha256=(
            args.predictor_package_root_sha256
        ),
        predictor_package_seal_sha256=(
            args.predictor_package_seal_sha256
        ),
        output_root=args.output_root,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
