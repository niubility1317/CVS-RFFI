#!/usr/bin/env python3
"""Run one fixed SOMP-H enrollment-only process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.somph_predictor_entry import run_somph_enrollment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--predictor-package-root", required=True)
    parser.add_argument("--detached-seal-path", required=True)
    parser.add_argument("--expected-seal-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = run_somph_enrollment(
        request_json=args.request_json,
        package_root=args.predictor_package_root,
        detached_seal_path=args.detached_seal_path,
        expected_seal_sha256=args.expected_seal_sha256,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
