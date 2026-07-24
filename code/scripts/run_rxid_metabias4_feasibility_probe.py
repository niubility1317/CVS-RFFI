#!/usr/bin/env python3
"""Run the bounded D103 source-held resource/constructability probe."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvsrffi.rxid_metabias4_feasibility_probe import (  # noqa: E402
    run_probe,
    validate_result_shape,
    write_probe_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tap-archive", type=Path, required=True)
    parser.add_argument("--dual-archive", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_probe(
        args.tap_archive.resolve(),
        args.dual_archive.resolve(),
        args.device,
    )
    validate_result_shape(result)
    write_probe_json(result, args.output_json.resolve())
    print(args.output_json.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
