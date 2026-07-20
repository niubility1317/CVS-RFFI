#!/usr/bin/env python
"""Publish one metadata-only, unsigned Phase2 data authority artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for value in (str(CODE_ROOT), str(REPO_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

from cvsrffi.phase2_data_authority import write_phase2_data_authority  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind an existing predictor manifest/seal, offline audit, and "
            "validation COMMIT metadata without opening support/query payloads"
        )
    )
    parser.add_argument("--predictor-manifest", type=Path, required=True)
    parser.add_argument("--predictor-seal", type=Path, required=True)
    parser.add_argument("--expected-predictor-seal-sha256", required=True)
    parser.add_argument("--offline-build-audit", type=Path, required=True)
    parser.add_argument("--expected-offline-build-audit-sha256", required=True)
    parser.add_argument("--data-validation-commit", type=Path, required=True)
    parser.add_argument("--expected-data-validation-commit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-output-sha256",
        default=None,
        help="optional integrity pin; never grants formal authority",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = write_phase2_data_authority(
        predictor_manifest_path=args.predictor_manifest,
        predictor_seal_path=args.predictor_seal,
        expected_predictor_seal_sha256=args.expected_predictor_seal_sha256,
        offline_build_audit_path=args.offline_build_audit,
        expected_offline_build_audit_sha256=(
            args.expected_offline_build_audit_sha256
        ),
        data_validation_commit_path=args.data_validation_commit,
        expected_data_validation_commit_sha256=(
            args.expected_data_validation_commit_sha256
        ),
        output_path=args.output,
        expected_output_sha256=args.expected_output_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
