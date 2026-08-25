#!/usr/bin/env python3
"""Truth-last same-row scorer for the M2.9 FFT96/TASR48 screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cvsrffi.stage2_ablation_truth_scorer import score_full_ablation_row, write_row_record_exclusive
from cvsrffi.stage2_m23_truth_diagnostics import score_m23_four_state_artifact, score_m23_paired_artifacts
from cvsrffi.stage2_m29_d92 import M29_ARMS, TASR_ALPHA1
from scripts import score_m24_d1_refit_matrix as shared
from scripts.score_m24_safe_residual_suite import _legacy_scorer_receipts


SCORED_MATRIX_SCHEMA = "cvs.erbt_idr.m29.tasr48_scored_matrix.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-index", required=True)
    parser.add_argument("--scoring-root", required=True)
    parser.add_argument("--supplemental-scoring-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    return parser


def _validate(matrix: dict[str, Any]) -> None:
    conditions = tuple((int(row["k_shot"]), int(row["new_class_count"])) for row in matrix["conditions"])
    identities = {(str(r), int(s), k, n) for r in matrix["receivers"] for s in matrix["method_seeds"] for k, n in conditions}
    seen: dict[tuple[str, int, int, int], set[str]] = {}
    for entry in matrix["entries"]:
        key = (str(entry["receiver"]), int(entry["method_seed"]), int(entry["k_shot"]), int(entry["new_class_count"]))
        seen.setdefault(key, set()).add(str(entry["arm"]))
    if (
        matrix.get("status") != "PREDICTIONS_COMPLETE_TRUTH_UNOPENED"
        or tuple(matrix.get("arms", ())) != M29_ARMS
        or set(seen) != identities
        or any(value != set(M29_ARMS) for value in seen.values())
        or len(matrix["entries"]) != len(identities) * len(M29_ARMS)
    ):
        raise ValueError("M2.9 prediction matrix is incomplete")


def _m29_scorer_behavior(receipt: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(receipt)
    weights = dict(normalized.get("full_block_weights", {}))
    if weights == {"full": 1.0, "block3": 1.0}:
        normalized["full_block_weights"] = {"full": 1.0, "block3": 0.0}
    elif weights != {"full": 1.0, "block3": 0.0}:
        raise ValueError("M2.9 full/block weight receipt drift")
    return normalized


def _m29_scorer_receipts(
    quantization: dict[str, Any], resource: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_scope = resource.get("auxiliary_state_cost_in_candidate_resource")
    latency_scope = resource.get("auxiliary_prediction_cost_in_candidate_latency")
    if (
        not isinstance(state_scope, bool)
        or not isinstance(latency_scope, bool)
        or state_scope != latency_scope
    ):
        raise ValueError("M2.9 auxiliary resource scope drift")
    legacy_quantization, legacy_resource = _legacy_scorer_receipts(
        quantization, resource
    )
    legacy_resource["auxiliary_state_cost_in_candidate_resource"] = False
    legacy_resource["auxiliary_prediction_cost_in_candidate_latency"] = False
    return legacy_quantization, legacy_resource


def main() -> int:
    args = _parser().parse_args()
    matrix = json.loads(Path(args.matrix_index).read_text(encoding="utf-8-sig"))
    _validate(matrix)
    roots = [Path(args.scoring_root).absolute()]
    if args.supplemental_scoring_root:
        roots.append(Path(args.supplemental_scoring_root).absolute())
    output = Path(args.output_root).absolute()
    output.mkdir(parents=True, exist_ok=False)
    comparisons = output / "paired_comparisons"
    comparisons.mkdir()
    groups: dict[tuple[str, int, int, int], dict[str, dict[str, Any]]] = {}
    scored_entries = []
    for entry in matrix["entries"]:
        receipt = json.loads(Path(entry["receipt_path"]).read_text(encoding="utf-8-sig"))
        scoring_manifest = shared._scoring_manifest(tuple(roots), entry)
        prediction = receipt["prediction"]
        quantization, resource = _m29_scorer_receipts(
            receipt["quantization"], receipt["resource"]
        )
        identity = {
            "logical_row_key": entry["row_id"],
            "ablation_id": entry["arm"],
            "physical_execution_id": entry["row_id"],
            "effective_config_hash": receipt["candidate_lock_sha256"],
            "alias_of": None,
        }
        same_row = score_full_ablation_row(
            prediction["path"],
            scoring_manifest,
            expected_prediction_artifact_sha256=prediction["artifact_sha256"],
            expected_prediction_seal_sha256=prediction["seal_sha256"],
            expected_scoring_manifest_sha256=shared._sha256(scoring_manifest),
            row_identity=identity,
            behavior_receipt=_m29_scorer_behavior(receipt["behavior"]),
            quantization_receipt=quantization,
            resource_receipt=resource,
        )
        row_root = output / entry["row_id"]
        row_root.mkdir()
        same_path = row_root / "same_row_score.json"
        write_row_record_exclusive(same_path, same_row)
        four_state = score_m23_four_state_artifact(
            prediction_path=prediction["path"],
            prediction_artifact_sha256=prediction["artifact_sha256"],
            prediction_seal_sha256=prediction["seal_sha256"],
            scoring_manifest_path=str(scoring_manifest),
            scoring_manifest_sha256=shared._sha256(scoring_manifest),
        )
        four_path = row_root / "four_state_score.json"
        write_row_record_exclusive(four_path, four_state)
        key = (entry["receiver"], entry["method_seed"], entry["k_shot"], entry["new_class_count"])
        groups.setdefault(key, {})[entry["arm"]] = {"entry": entry, "receipt": receipt, "manifest": scoring_manifest, "four_state": four_state}
        scored_entries.append({**entry, "same_row_score_path": str(same_path), "four_state_score_path": str(four_path)})
    paired_outputs = []
    for key, group in groups.items():
        candidate = group[TASR_ALPHA1]
        candidate_prediction = candidate["receipt"]["prediction"]
        for reference_arm in M29_ARMS:
            if reference_arm == TASR_ALPHA1:
                continue
            reference = group[reference_arm]
            reference_prediction = reference["receipt"]["prediction"]
            paired = score_m23_paired_artifacts(
                reference_prediction_path=reference_prediction["path"],
                reference_artifact_sha256=reference_prediction["artifact_sha256"],
                reference_seal_sha256=reference_prediction["seal_sha256"],
                candidate_prediction_path=candidate_prediction["path"],
                candidate_artifact_sha256=candidate_prediction["artifact_sha256"],
                candidate_seal_sha256=candidate_prediction["seal_sha256"],
                scoring_manifest_path=str(candidate["manifest"]),
                scoring_manifest_sha256=shared._sha256(candidate["manifest"]),
                bootstrap_repeats=args.bootstrap_repeats,
                bootstrap_seed=2906,
            )
            slug = f"rx{key[0]}_m{key[1]}_k{key[2]}_new{key[3]}__tasr_vs_{reference_arm}"
            paired_path = comparisons / f"{slug}.json"
            write_row_record_exclusive(paired_path, paired)
            forgetting_path = comparisons / f"{slug}__forgetting.json"
            write_row_record_exclusive(forgetting_path, {
                "schema": "cvs.erbt_idr.m29.standardized_forgetting.v1",
                "reference_arm": reference_arm,
                "candidate_arm": TASR_ALPHA1,
                "scenario_rows": shared.standardized_forgetting(reference["four_state"], candidate["four_state"]),
            })
            paired_outputs.append({"group": list(key), "reference_arm": reference_arm, "candidate_arm": TASR_ALPHA1, "paired_path": str(paired_path), "standardized_forgetting_path": str(forgetting_path)})
    result = {
        "schema": SCORED_MATRIX_SCHEMA,
        "status": "PASS",
        "row_count": len(scored_entries),
        "paired_input_identity_count": int(matrix["paired_input_identity_count"]),
        "method_rows_per_arm": int(matrix["method_rows_per_arm"]),
        "scenario_unit_count": len(scored_entries) * 3,
        "reference_selection": "compare TASR48 against every frozen FFT96 weight and identity-only on identical rows",
        "entries": scored_entries,
        "paired_outputs": paired_outputs,
        "truth_opened_after_all_predictions_complete": True,
        "scorer_output_must_not_feed_predictor": True,
    }
    write_row_record_exclusive(output / "scored_matrix_index.json", result)
    print(json.dumps({"status": "PASS", "row_count": len(scored_entries), "paired_count": len(paired_outputs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
