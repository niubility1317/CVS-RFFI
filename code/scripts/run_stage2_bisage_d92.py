#!/usr/bin/env python3
"""Run one BiSAGE-D92 scene or independently score sealed predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_bisage_runner import (  # noqa: E402
    adapt_stage_a,
    adapt_stage_b_and_predict,
    frozen_checkpoint,
    score_truth_last,
)


def _job(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("job JSON must be an object")
    return payload


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    adapt = commands.add_parser("adapt-a")
    adapt.add_argument("--job", type=Path, required=True)
    adapt.add_argument("--scenario", required=True)
    adapt.add_argument("--checkpoint", required=True)
    adapt.add_argument("--output-root", type=Path, required=True)
    adapt.add_argument("--device", required=True)
    adapt.add_argument("--steps", type=int, default=3000)
    predict = commands.add_parser("adapt-b-predict")
    predict.add_argument("--job", type=Path, required=True)
    predict.add_argument("--scenario", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--stage-a-root", type=Path, required=True)
    predict.add_argument("--output-root", type=Path, required=True)
    predict.add_argument("--device", required=True)
    predict.add_argument("--steps", type=int, default=2000)
    predict.add_argument("--stage-a-group-result", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--receipt", type=Path, required=True)
    score.add_argument("--truth", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "score":
        result = score_truth_last(args.predictions, args.receipt, args.truth, args.output)
    else:
        job = _job(args.job)
        model = frozen_checkpoint(args.checkpoint, args.device)
        if args.command == "adapt-a":
            result = adapt_stage_a(job, args.scenario, model, args.output_root, args.device, steps=args.steps)
        else:
            group = json.loads(args.stage_a_group_result.read_text(encoding="utf-8"))
            if (
                group.get("schema") != "cvs.phase2.bisage_d92.stage_a_group.v1"
                or group.get("outer_key") != job.get("outer_key")
                or tuple(group.get("scenarios", ())) != (
                    "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"
                )
            ):
                raise ValueError("joint Stage A group binding drift")
            result = adapt_stage_b_and_predict(
                job, args.scenario, model, args.stage_a_root, args.output_root,
                args.device, steps=args.steps,
                enable_stage_b=bool(group.get("stage_a_all_scenarios_passed")),
            )
    print(json.dumps(dict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
