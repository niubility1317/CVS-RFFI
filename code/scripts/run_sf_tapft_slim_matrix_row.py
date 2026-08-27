"""Run one immutable support-only row from an SF-TAPFT slimming matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.sf_tapft_slim_matrix import build_row_config  # noqa: E402
from cvsrffi.target_only_progressive_runner import (  # noqa: E402
    run_sf_tapft_deploy_no_query,
    run_sf_tapft_grouped_selection,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", default=4, type=int)
    parser.add_argument("--mode", choices=("grouped", "deploy"), default="grouped")
    parser.add_argument("--deployment-inplace", action="store_true")
    parser.add_argument("--delta-only", action="store_true")
    args = parser.parse_args(argv)
    matrix = json.loads(args.matrix.read_text(encoding="utf-8-sig"))
    config, gpu = build_row_config(matrix, args.row_id)
    if args.mode == "deploy":
        receipt = run_sf_tapft_deploy_no_query(
            config,
            args.output_dir,
            device=args.device,
            deployment_inplace=bool(args.deployment_inplace),
            emit_clean_single_bundle=not bool(args.delta_only),
        )
    else:
        receipt = run_sf_tapft_grouped_selection(
            config,
            args.output_dir,
            device=args.device,
            folds=args.folds,
        )
    receipt = {**receipt, "matrix_row_id": args.row_id, "assigned_gpu": gpu}
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
