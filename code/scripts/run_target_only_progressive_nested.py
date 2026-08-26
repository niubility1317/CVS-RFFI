"""Run grouped target-train OOF selection for report-parity SF-TAPFT V1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.target_only_progressive_runner import run_sf_tapft_grouped_selection  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", default=4, type=int)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    receipt = run_sf_tapft_grouped_selection(
        config,
        args.output_dir,
        device=args.device,
        folds=args.folds,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
