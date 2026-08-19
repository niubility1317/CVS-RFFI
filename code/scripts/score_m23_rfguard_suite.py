#!/usr/bin/env python3
"""Truth-last score an M2.3 F0-F5 suite and publish paired diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from cvsrffi.stage2_ablation_truth_scorer import (
    score_full_ablation_row,
    write_row_record_exclusive,
)
from cvsrffi.stage2_m23_row_executor import M23_F1_IF
from cvsrffi.stage2_m23_truth_diagnostics import (
    score_m23_four_state_artifact,
    score_m23_paired_artifacts,
)


SCORED_SUITE_SCHEMA = "cvs.erbt_idr.m23.scored_suite.v1"


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON root must be an object")
    return dict(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-index", required=True)
    parser.add_argument("--scoring-manifest", required=True)
    parser.add_argument("--scoring-manifest-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2301)
    return parser


def main() -> int:
    args = _parser().parse_args()
    suite = _load(args.suite_index)
    entries = suite.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("suite index has no entries")
    by_arm = {str(entry["arm"]): dict(entry) for entry in entries}
    if M23_F1_IF not in by_arm:
        raise ValueError("paired scoring requires the F1 reference arm")
    output = Path(args.output_root).absolute()
    output.mkdir(parents=True, exist_ok=False)
    scored_entries = []
    for entry in entries:
        arm = str(entry["arm"])
        receipt = _load(entry["receipt_path"])
        prediction = receipt["prediction"]
        row_identity = {
            "logical_row_key": f"{receipt['row_id']}::{arm}",
            "ablation_id": arm,
            "physical_execution_id": str(receipt["row_id"]),
            "effective_config_hash": str(receipt["candidate_lock_sha256"]),
            "alias_of": None,
        }
        score = score_full_ablation_row(
            prediction["path"],
            args.scoring_manifest,
            expected_prediction_artifact_sha256=prediction["artifact_sha256"],
            expected_prediction_seal_sha256=prediction["seal_sha256"],
            expected_scoring_manifest_sha256=args.scoring_manifest_sha256,
            row_identity=row_identity,
            behavior_receipt=receipt["behavior"],
            quantization_receipt=receipt["quantization"],
            resource_receipt=receipt["resource"],
        )
        arm_root = output / arm
        arm_root.mkdir(parents=False, exist_ok=False)
        score_path = arm_root / "same_row_score.json"
        write_row_record_exclusive(score_path, score)
        four_state = score_m23_four_state_artifact(
            prediction_path=prediction["path"],
            prediction_artifact_sha256=prediction["artifact_sha256"],
            prediction_seal_sha256=prediction["seal_sha256"],
            scoring_manifest_path=args.scoring_manifest,
            scoring_manifest_sha256=args.scoring_manifest_sha256,
        )
        four_state_path = arm_root / "four_state_score.json"
        write_row_record_exclusive(four_state_path, four_state)
        paired_path = None
        if arm != M23_F1_IF:
            reference_receipt = _load(by_arm[M23_F1_IF]["receipt_path"])
            reference_prediction = reference_receipt["prediction"]
            paired = score_m23_paired_artifacts(
                reference_prediction_path=reference_prediction["path"],
                reference_artifact_sha256=reference_prediction["artifact_sha256"],
                reference_seal_sha256=reference_prediction["seal_sha256"],
                candidate_prediction_path=prediction["path"],
                candidate_artifact_sha256=prediction["artifact_sha256"],
                candidate_seal_sha256=prediction["seal_sha256"],
                scoring_manifest_path=args.scoring_manifest,
                scoring_manifest_sha256=args.scoring_manifest_sha256,
                bootstrap_repeats=args.bootstrap_repeats,
                bootstrap_seed=args.bootstrap_seed,
            )
            paired_path = arm_root / "paired_vs_f1.json"
            write_row_record_exclusive(paired_path, paired)
        scored_entries.append(
            {
                "arm": arm,
                "same_row_score_path": str(score_path),
                "four_state_score_path": str(four_state_path),
                "paired_vs_f1_path": str(paired_path) if paired_path else None,
            }
        )
    result = {
        "schema": SCORED_SUITE_SCHEMA,
        "status": "PASS",
        "reference_arm": M23_F1_IF,
        "entries": scored_entries,
        "truth_opened_after_prediction_commit": True,
        "scorer_output_must_not_feed_predictor": True,
    }
    write_row_record_exclusive(output / "scored_suite_index.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
