#!/usr/bin/env python3
"""Truth-last scorer for an M2.6 B0/T1-T5 paired matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cvsrffi.stage2_ablation_truth_scorer import score_full_ablation_row, write_row_record_exclusive
from cvsrffi.stage2_m23_truth_diagnostics import score_m23_four_state_artifact, score_m23_paired_artifacts
from cvsrffi.stage2_m24_safe_residual import D1
from cvsrffi.stage2_m26_td_src256 import M26_ARMS
from scripts import score_m24_d1_refit_matrix as shared
from scripts.run_m26_td_src256_matrix import EVIDENCE_ARMS
from scripts.score_m24_safe_residual_suite import _legacy_scorer_receipts


SCORED_MATRIX_SCHEMA = "cvs.erbt_idr.m26.td_src256_scored_matrix.v1"
PAIRED_SCORE_FILENAME = "paired_vs_r0.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-index", required=True)
    parser.add_argument("--scoring-root", required=True)
    parser.add_argument("--supplemental-scoring-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    return parser


def _validate_matrix(matrix: dict[str, Any]) -> None:
    entries = list(matrix.get("entries", []))
    arms = tuple(matrix.get("arms", []))
    conditions = tuple((int(row["k_shot"]), int(row["new_class_count"])) for row in matrix.get("conditions", []))
    expected_identities = {
        (str(receiver), int(seed), int(k), int(n))
        for receiver in matrix.get("receivers", [])
        for seed in matrix.get("method_seeds", [])
        for k, n in conditions
    }
    seen: dict[tuple[str, int, int, int], set[str]] = {}
    row_ids: set[str] = set()
    for entry in entries:
        identity = (
            str(entry["receiver"]), int(entry["method_seed"]),
            int(entry["k_shot"]), int(entry["new_class_count"]),
        )
        arm = str(entry["arm"])
        row_id = str(entry["row_id"])
        if identity not in expected_identities or arm not in arms or row_id in row_ids:
            raise ValueError("M2.6 prediction matrix identity drift")
        row_ids.add(row_id)
        seen.setdefault(identity, set()).add(arm)
    if (
        matrix.get("status") != "PREDICTIONS_COMPLETE_TRUTH_UNOPENED"
        or arms != EVIDENCE_ARMS
        or int(matrix.get("paired_input_identity_count", -1)) != len(expected_identities)
        or len(entries) != len(expected_identities) * len(arms)
        or set(seen) != expected_identities
        or any(value != set(arms) for value in seen.values())
    ):
        raise ValueError("M2.6 prediction matrix is incomplete")


def main() -> int:
    args = _parser().parse_args()
    matrix = json.loads(Path(args.matrix_index).read_text(encoding="utf-8-sig"))
    _validate_matrix(matrix)
    roots = [Path(args.scoring_root).absolute()]
    if args.supplemental_scoring_root:
        roots.append(Path(args.supplemental_scoring_root).absolute())
    scoring_roots = tuple(roots)
    output_root = Path(args.output_root).absolute()
    output_root.mkdir(parents=True, exist_ok=False)
    groups: dict[tuple[str, int, int, int], dict[str, dict[str, Any]]] = {}
    scored_entries: list[dict[str, Any]] = []
    for entry in matrix["entries"]:
        receipt = json.loads(Path(entry["receipt_path"]).read_text(encoding="utf-8-sig"))
        scoring_manifest = shared._scoring_manifest(scoring_roots, entry)
        prediction = receipt["prediction"]
        quantization, resource = _legacy_scorer_receipts(receipt["quantization"], receipt["resource"])
        row_identity = {
            "logical_row_key": entry["row_id"],
            "ablation_id": entry["arm"],
            "physical_execution_id": entry["row_id"],
            "effective_config_hash": receipt["candidate_lock_sha256"],
            "alias_of": None,
        }
        score = score_full_ablation_row(
            prediction["path"], scoring_manifest,
            expected_prediction_artifact_sha256=prediction["artifact_sha256"],
            expected_prediction_seal_sha256=prediction["seal_sha256"],
            expected_scoring_manifest_sha256=shared._sha256(scoring_manifest),
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
            scoring_manifest_path=str(scoring_manifest),
            scoring_manifest_sha256=shared._sha256(scoring_manifest),
        )
        four_state_path = row_root / "four_state_score.json"
        write_row_record_exclusive(four_state_path, four_state)
        key = (entry["receiver"], entry["method_seed"], entry["k_shot"], entry["new_class_count"])
        groups.setdefault(key, {})[entry["arm"]] = {
            "entry": entry, "receipt": receipt, "four_state": four_state,
            "row_root": row_root, "scoring_manifest": scoring_manifest,
        }
        scored_entries.append({**entry, "same_row_score_path": str(score_path), "four_state_score_path": str(four_state_path)})
    paired_outputs = []
    for key, group in groups.items():
        if set(group) != set(EVIDENCE_ARMS):
            raise ValueError(f"M2.6 paired group incomplete: {key}")
        reference = group[D1]
        reference_prediction = reference["receipt"]["prediction"]
        for arm in M26_ARMS:
            candidate = group[arm]
            candidate_prediction = candidate["receipt"]["prediction"]
            paired = score_m23_paired_artifacts(
                reference_prediction_path=reference_prediction["path"],
                reference_artifact_sha256=reference_prediction["artifact_sha256"],
                reference_seal_sha256=reference_prediction["seal_sha256"],
                candidate_prediction_path=candidate_prediction["path"],
                candidate_artifact_sha256=candidate_prediction["artifact_sha256"],
                candidate_seal_sha256=candidate_prediction["seal_sha256"],
                scoring_manifest_path=str(candidate["scoring_manifest"]),
                scoring_manifest_sha256=shared._sha256(candidate["scoring_manifest"]),
                bootstrap_repeats=args.bootstrap_repeats,
                bootstrap_seed=2606,
            )
            paired_path = candidate["row_root"] / PAIRED_SCORE_FILENAME
            write_row_record_exclusive(paired_path, paired)
            forgetting_path = candidate["row_root"] / "standardized_forgetting.json"
            write_row_record_exclusive(forgetting_path, {
                "schema": "cvs.erbt_idr.m26.standardized_forgetting.v1",
                "reference_arm": D1,
                "candidate_arm": arm,
                "scenario_rows": shared.standardized_forgetting(reference["four_state"], candidate["four_state"]),
            })
            paired_outputs.append({
                "group": list(key), "candidate_arm": arm,
                "paired_path": str(paired_path),
                "standardized_forgetting_path": str(forgetting_path),
            })
    identity_count = int(matrix["paired_input_identity_count"])
    result = {
        "schema": SCORED_MATRIX_SCHEMA,
        "status": "PASS",
        "row_count": len(scored_entries),
        "paired_input_identity_count": identity_count,
        "method_rows_per_arm": identity_count,
        "scenario_unit_count": len(scored_entries) * 3,
        "primary_d92_e0_baseline": "P2-A1_NO_RF32",
        "reference_arm": D1,
        "entries": scored_entries,
        "paired_outputs": paired_outputs,
        "truth_opened_after_all_predictions_complete": True,
        "scorer_output_must_not_feed_predictor": True,
    }
    write_row_record_exclusive(output_root / "scored_matrix_index.json", result)
    print(json.dumps({"status": "PASS", "row_count": len(scored_entries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
