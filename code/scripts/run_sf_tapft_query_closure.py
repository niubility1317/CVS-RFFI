"""Predict or truth-last score one existing SF-TAPFT clean-single bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_sf_tapft_query_closure import (  # noqa: E402
    export_independent_query_inputs,
    materialize_truth_after_predictions,
    run_clean_query_prediction,
    score_clean_query_prediction,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-independent")
    export.add_argument("--support-source", type=Path, required=True)
    export.add_argument("--query-source", type=Path, required=True)
    export.add_argument("--package-manifest", type=Path, required=True)
    export.add_argument("--output-root", type=Path, required=True)
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
    predict.add_argument("--device", required=True)
    score = commands.add_parser("score")
    score.add_argument("--prediction-root", type=Path, required=True)
    score.add_argument("--truth", type=Path, required=True)
    score.add_argument("--data-handle", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    truth = commands.add_parser("materialize-truth")
    truth.add_argument("--prediction-root", type=Path, required=True)
    truth.add_argument("--truth-sidecar", type=Path, required=True)
    truth.add_argument("--data-handle", type=Path, required=True)
    truth.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export-independent":
        result = export_independent_query_inputs(
            support_source_path=args.support_source,
            query_source_path=args.query_source,
            package_manifest_path=args.package_manifest,
            output_root=args.output_root,
            expected_support_iq_sha256=args.expected_support_iq_sha256,
            capsule_id=args.capsule_id,
            split_id=args.split_id,
            adaptation_capsule_id=args.adaptation_capsule_id,
            adaptation_split_id=args.adaptation_split_id,
        )
    elif args.command == "predict":
        result = run_clean_query_prediction(
            bundle_path=args.bundle,
            support_path=args.support,
            query_path=args.query,
            data_handle_path=args.data_handle,
            output_root=args.output_root,
            device=args.device,
        )
    elif args.command == "score":
        result = score_clean_query_prediction(
            prediction_root=args.prediction_root,
            truth_path=args.truth,
            data_handle_path=args.data_handle,
            output_path=args.output,
        )
    else:
        result = materialize_truth_after_predictions(
            prediction_root=args.prediction_root,
            truth_sidecar_path=args.truth_sidecar,
            data_handle_path=args.data_handle,
            output_path=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
