#!/usr/bin/env python
"""Run one local diagnostic Stage2 cell; formal launch authority remains false."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase2_isolated_runner import execute_phase2_isolated  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bwrap", type=Path, required=True)
    parser.add_argument("--strace", type=Path, required=True)
    parser.add_argument("--runtime-closure-root", type=Path, required=True)
    parser.add_argument("--pre-run-evidence-root", type=Path, required=True)
    parser.add_argument("--predictor-package-root", type=Path, required=True)
    parser.add_argument("--detached-seal-path", type=Path, required=True)
    parser.add_argument("--expected-package-seal-sha256", required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--system-read-root", type=Path, action="append", required=True)
    parser.add_argument("--gpu-device", type=Path, action="append", default=[])
    parser.add_argument(
        "--forbidden-root",
        type=Path,
        action="append",
        required=True,
        help="Host scorer/truth root that must not overlap any predictor mount.",
    )
    parser.add_argument(
        "--forbidden-project-root",
        action="append",
        default=[],
        help="Sandbox-visible host project prefix that an actual open must never use.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=int)
    args = parser.parse_args()
    result = execute_phase2_isolated(
        bwrap=args.bwrap,
        strace_executable=args.strace,
        runtime_closure_root=args.runtime_closure_root,
        pre_run_evidence_root=args.pre_run_evidence_root,
        package_root=args.predictor_package_root,
        detached_seal=args.detached_seal_path,
        expected_package_seal_sha256=args.expected_package_seal_sha256,
        request_json=args.request_json,
        output_root=args.output_root,
        python_executable=args.python_executable,
        system_read_roots=args.system_read_root,
        gpu_devices=args.gpu_device,
        forbidden_roots=args.forbidden_root,
        forbidden_project_roots=args.forbidden_project_root,
        device=args.device,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
