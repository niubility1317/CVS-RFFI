"""Export, predict, and truth-last score the SF plus D92-E0-noRF32 diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_sf_erbt_oldonly import (
    export_old_only_holdout,
    run_old_only_prediction,
    score_old_only_predictions,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--output-root", type=Path, required=True)
    export.add_argument("--k-shot", type=int, required=True)
    export.add_argument("--expected-support-iq-sha256", required=True)
    export.add_argument("--capsule-id", required=True)
    export.add_argument("--split-id", required=True)
    export.add_argument("--adaptation-capsule-id", required=True)
    export.add_argument("--adaptation-split-id", required=True)

    predict = commands.add_parser("predict")
    predict.add_argument("--bundle", type=Path, required=True)
    predict.add_argument("--support", type=Path, required=True)
    predict.add_argument("--query", type=Path, required=True)
    predict.add_argument("--data-handle", type=Path, required=True)
    predict.add_argument("--output-root", type=Path, required=True)
    predict.add_argument("--seed", type=int, required=True)
    predict.add_argument("--device", required=True)

    score = commands.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--truth", type=Path, required=True)
    score.add_argument("--prediction-receipt", type=Path, required=True)
    score.add_argument("--data-handle", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "export":
        result = export_old_only_holdout(
            args.source,
            args.output_root,
            k_shot=args.k_shot,
            expected_support_iq_sha256=args.expected_support_iq_sha256,
            capsule_id=args.capsule_id,
            split_id=args.split_id,
            adaptation_capsule_id=args.adaptation_capsule_id,
            adaptation_split_id=args.adaptation_split_id,
        )
    elif args.command == "predict":
        result = run_old_only_prediction(
            bundle_path=args.bundle,
            support_path=args.support,
            query_path=args.query,
            data_handle_path=args.data_handle,
            output_root=args.output_root,
            seed=args.seed,
            device=args.device,
        )
    else:
        result = score_old_only_predictions(
            args.predictions,
            args.truth,
            args.prediction_receipt,
            args.data_handle,
            args.output,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
