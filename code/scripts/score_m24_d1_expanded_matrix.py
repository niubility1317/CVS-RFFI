#!/usr/bin/env python3
"""Truth-last score a completed M2.4 D1 expanded prediction matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cvsrffi.stage2_ablation_truth_scorer import score_full_ablation_row, write_row_record_exclusive
from cvsrffi.stage2_m23_truth_diagnostics import score_m23_four_state_artifact
from scripts.score_m24_safe_residual_suite import _legacy_scorer_receipts


SCORED_MATRIX_SCHEMA = "cvs.erbt_idr.m24.d1_expanded_scored_matrix.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-index", required=True)
    parser.add_argument("--scoring-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    matrix = json.loads(Path(args.matrix_index).read_text(encoding="utf-8-sig"))
    entries = matrix.get("entries", [])
    if matrix.get("status") != "PREDICTIONS_COMPLETE_TRUTH_UNOPENED" or len(entries) != 60:
        raise ValueError("expanded D1 prediction matrix is incomplete")
    output_root = Path(args.output_root).absolute()
    output_root.mkdir(parents=True, exist_ok=False)
    scored = []
    for entry in entries:
        receipt = json.loads(Path(entry["receipt_path"]).read_text(encoding="utf-8-sig"))
        if receipt["d1_historical_parity"]["prediction_disagreements"] != 0:
            raise ValueError("D1 parity failure forbids scoring")
        scoring_manifest = (
            Path(args.scoring_root)
            / "stage2c"
            / f"rx_{entry['receiver'].replace('-', '_')}"
            / f"method_{entry['method_seed']}"
            / f"new{entry['new_class_count']}"
            / "scoring_manifest.json"
        )
        quantization, resource = _legacy_scorer_receipts(
            receipt["quantization"], receipt["resource"]
        )
        row_identity = {
            "logical_row_key": entry["row_id"],
            "ablation_id": receipt["arm"],
            "physical_execution_id": entry["row_id"],
            "effective_config_hash": receipt["candidate_lock_sha256"],
            "alias_of": None,
        }
        prediction = receipt["prediction"]
        score = score_full_ablation_row(
            prediction["path"],
            scoring_manifest,
            expected_prediction_artifact_sha256=prediction["artifact_sha256"],
            expected_prediction_seal_sha256=prediction["seal_sha256"],
            expected_scoring_manifest_sha256=_sha256(scoring_manifest),
            row_identity=row_identity,
            behavior_receipt=receipt["behavior"],
            quantization_receipt=quantization,
            resource_receipt=resource,
        )
        row_root = output_root / entry["row_id"]
        row_root.mkdir(parents=False, exist_ok=False)
        score_path = row_root / "same_row_score.json"
        write_row_record_exclusive(score_path, score)
        four_state = score_m23_four_state_artifact(
            prediction_path=prediction["path"],
            prediction_artifact_sha256=prediction["artifact_sha256"],
            prediction_seal_sha256=prediction["seal_sha256"],
            scoring_manifest_path=scoring_manifest,
            scoring_manifest_sha256=_sha256(scoring_manifest),
        )
        four_state_path = row_root / "four_state_score.json"
        write_row_record_exclusive(four_state_path, four_state)
        scored.append({**entry, "same_row_score_path": str(score_path), "four_state_score_path": str(four_state_path)})
        print(json.dumps({"scored": entry["row_id"], "count": len(scored)}, sort_keys=True), flush=True)
    result = {
        "schema": SCORED_MATRIX_SCHEMA,
        "status": "PASS",
        "row_count": len(scored),
        "entries": scored,
        "truth_opened_after_all_predictions_complete": True,
        "scorer_output_must_not_feed_predictor": True,
    }
    write_row_record_exclusive(output_root / "scored_matrix_index.json", result)
    print(json.dumps({"status": "PASS", "row_count": len(scored)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
