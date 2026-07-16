#!/usr/bin/env python3
"""Run strict support-only diag-cosine on one sealed Stage2-B/C package pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_diag_cosine_exploration import (  # noqa: E402
    CANDIDATES,
    run_diag_cosine_exploration,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--enrollment-package-root", required=True)
    result.add_argument("--enrollment-seal-path", required=True)
    result.add_argument("--enrollment-seal-sha256", required=True)
    result.add_argument("--apply-package-root", required=True)
    result.add_argument("--apply-seal-path", required=True)
    result.add_argument("--apply-seal-sha256", required=True)
    result.add_argument("--output-root", required=True)
    result.add_argument("--device", choices=("cpu", "cuda:0", "cuda:1"), default="cuda:0")
    result.add_argument("--candidate", choices=CANDIDATES, default=CANDIDATES[0])
    result.add_argument("--parent-diag-root")
    result.add_argument("--expected-parent-commit-sha256")
    return result


def main() -> int:
    args = parser().parse_args()
    result = run_diag_cosine_exploration(
        enrollment_package_root=args.enrollment_package_root,
        enrollment_seal_path=args.enrollment_seal_path,
        enrollment_seal_sha256=args.enrollment_seal_sha256,
        apply_package_root=args.apply_package_root,
        apply_seal_path=args.apply_seal_path,
        apply_seal_sha256=args.apply_seal_sha256,
        output_root=args.output_root,
        device=args.device,
        candidate=args.candidate,
        parent_diag_root=args.parent_diag_root,
        expected_parent_commit_sha256=args.expected_parent_commit_sha256,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
