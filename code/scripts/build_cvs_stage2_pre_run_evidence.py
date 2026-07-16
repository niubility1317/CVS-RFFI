#!/usr/bin/env python
"""Build strict Linux pre-run evidence for one sealed Stage2 predictor package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase2_pre_run_evidence import (  # noqa: E402
    ISOLATION_PROFILES,
    build_phase2_pre_run_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-closure-root", type=Path, required=True)
    parser.add_argument("--predictor-package-root", type=Path, required=True)
    parser.add_argument("--detached-seal-path", type=Path, required=True)
    parser.add_argument("--expected-package-seal-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bwrap-executable", type=Path, required=True)
    parser.add_argument("--strace-executable", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--system-read-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--isolation-profile",
        choices=sorted(ISOLATION_PROFILES),
        default="stage2_predictor",
    )
    parser.add_argument(
        "--forbidden-scorer-truth-root", type=Path, action="append", required=True
    )
    args = parser.parse_args()
    result = build_phase2_pre_run_evidence(
        runtime_closure_root=args.runtime_closure_root,
        package_root=args.predictor_package_root,
        detached_seal=args.detached_seal_path,
        expected_package_seal_sha256=args.expected_package_seal_sha256,
        output_root=args.output_root,
        bwrap_executable=args.bwrap_executable,
        strace_executable=args.strace_executable,
        python_executable=args.python_executable,
        system_read_roots=args.system_read_root,
        forbidden_scorer_truth_roots=args.forbidden_scorer_truth_root,
        isolation_profile=args.isolation_profile,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
