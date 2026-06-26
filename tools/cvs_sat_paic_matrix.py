#!/usr/bin/env python
"""Generate local CVS-SAT-PAIC matrix and Markdown report artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from cvsrffi.paic_star_ground import build_paic_matrix, write_paic_payloads
from optimizer_validate_matrix import validate


def write_paic_artifacts(output_root: Path | str) -> dict[str, Path]:
    payload = build_paic_matrix()
    validation = validate(payload["candidates"], expected_count=int(payload["expected_count"]))
    return write_paic_payloads(output_root, payload, validation_payload=validation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "automation_reports" / "CV-SincNet" / "cvs_sat_paic_20260622")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = write_paic_artifacts(args.output_root)
    print(f"[CVS-SAT-PAIC] json={output['json_path']}")
    print(f"[CVS-SAT-PAIC] report={output['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
