"""CLI for the CVS-FSFA-V2 real-checkpoint no-query smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.factored_slow_fast_no_query_smoke import run_factored_slow_fast_no_query_smoke  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    receipt = run_factored_slow_fast_no_query_smoke(config, args.output, device=args.device)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
