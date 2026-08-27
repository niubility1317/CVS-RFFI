from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.sf_tapft_deployment_benchmark import benchmark_deployment_runs
from cvsrffi.sf_tapft_slim_matrix import build_row_config
from cvsrffi.target_only_progressive_runner import run_sf_tapft_deploy_no_query


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark SF-TAPFT deployment adaptation")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--matrix")
    parser.add_argument("--row-id")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--measured-runs", type=int, default=10)
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--emit-clean-single-bundle", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    raw = json.loads(Path(args.config or args.matrix).read_text(encoding="utf-8-sig"))
    if args.matrix:
        if not args.row_id:
            raise ValueError("--row-id is required with --matrix")
        config, _gpu = build_row_config(raw, args.row_id)
    else:
        if args.row_id:
            raise ValueError("--row-id is only valid with --matrix")
        config = raw
    root = Path(args.output_root)

    def run_once(_kind: str, _index: int, destination: Path):
        return run_sf_tapft_deploy_no_query(
            config,
            destination,
            device=args.device,
            deployment_inplace=bool(args.inplace),
            emit_clean_single_bundle=bool(args.emit_clean_single_bundle),
        )

    result = benchmark_deployment_runs(
        run_once,
        root,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
    )
    report_path = root / "benchmark.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "BENCHMARK_COMPLETE", "report_path": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
