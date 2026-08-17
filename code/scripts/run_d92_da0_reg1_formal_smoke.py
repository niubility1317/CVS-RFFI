#!/usr/bin/env python3
"""Run one signed-authority formal ERTB-IDR DA0_REG1 smoke artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_d92_da0_reg1_formal import parser, run_from_args  # noqa: E402


def main() -> int:
    receipt = run_from_args(parser().parse_args())
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
