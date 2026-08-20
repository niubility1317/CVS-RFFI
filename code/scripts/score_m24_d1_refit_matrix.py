#!/usr/bin/env python3
"""Truth-last score and pair the R0/R1/R2 M2.4 D1 refit screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from cvsrffi.stage2_ablation_truth_scorer import score_full_ablation_row, write_row_record_exclusive
from cvsrffi.stage2_m23_truth_diagnostics import score_m23_four_state_artifact, score_m23_paired_artifacts
from cvsrffi.stage2_m24_safe_residual import D0, D1, D1_REFIT
from scripts.score_m24_safe_residual_suite import _legacy_scorer_receipts
from scripts.run_m24_d1_refit_matrix import (
    DEFAULT_CONDITIONS,
    DEFAULT_RECEIVERS,
    DEFAULT_SEEDS,
    EVIDENCE_ARMS,
    EXPECTED_INPUT_IDENTITIES,
    EXPECTED_METHOD_ROWS,
)


SCORED_MATRIX_SCHEMA = "cvs.erbt_idr.m24.d1_refit_scored_matrix.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardized_forgetting(
    reference_four_state: dict[str, Any], candidate_four_state: dict[str, Any]
) -> list[dict[str, float | str]]:
    references = {row["scenario"]: row for row in reference_four_state["scenario_rows"]}
    output: list[dict[str, float | str]] = []
    for row in candidate_four_state["scenario_rows"]:
        scenario = str(row["scenario"])
        reference_pre = float(references[scenario]["states"]["DA1_REG0"]["old_accuracy"])
        within_pre = float(row["states"]["DA1_REG0"]["old_accuracy"])
        post = float(row["states"]["DA1_REG1"]["old_accuracy"])
        output.append({
            "scenario": scenario,
            "A_o_pre_within": within_pre,
            "A_o_pre_reference_r0": reference_pre,
            "A_o_post": post,
            "F_within": within_pre - post,
            "F_std": reference_pre - post,
        })
    return output


def _validate_complete_matrix(entries: list[dict[str, Any]]) -> None:
    expected_identities = {
        (receiver, seed, k_shot, new_count)
        for receiver in DEFAULT_RECEIVERS
        for seed in DEFAULT_SEEDS
        for k_shot, new_count in DEFAULT_CONDITIONS
    }
    observed_identities: set[tuple[str, int, int, int]] = set()
    observed_pairs: set[tuple[tuple[str, int, int, int], str]] = set()
    row_ids: set[str] = set()
    arms_by_identity: dict[tuple[str, int, int, int], set[str]] = {}
    for entry in entries:
        identity = (
            str(entry.get("receiver")),
            int(entry.get("method_seed", -1)),
            int(entry.get("k_shot", -1)),
            int(entry.get("new_class_count", -1)),
        )
        arm = str(entry.get("arm"))
        row_id = str(entry.get("row_id"))
        pair = (identity, arm)
        if (
            identity not in expected_identities
            or arm not in EVIDENCE_ARMS
            or pair in observed_pairs
            or not row_id
            or row_id in row_ids
        ):
            raise ValueError("prediction entries do not form the complete 125 identity grid")
        observed_identities.add(identity)
        observed_pairs.add(pair)
        row_ids.add(row_id)
        arms_by_identity.setdefault(identity, set()).add(arm)
    if (
        len(entries) != EXPECTED_METHOD_ROWS
        or observed_identities != expected_identities
        or len(observed_pairs) != EXPECTED_METHOD_ROWS
        or any(arms != set(EVIDENCE_ARMS) for arms in arms_by_identity.values())
    ):
        raise ValueError("prediction entries do not form the complete 125 identity grid")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-index", required=True)
    parser.add_argument("--scoring-root", required=True)
    parser.add_argument("--supplemental-scoring-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    return parser


def _scoring_manifest(
    roots: tuple[Path, ...], entry: dict[str, Any]
) -> Path:
    receiver = str(entry["receiver"]).replace("-", "_")
    method_seed = int(entry["method_seed"])
    new_count = int(entry["new_class_count"])
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            (
                root
                / "stage2c"
                / f"rx_{receiver}"
                / f"method_{method_seed}"
                / f"new{new_count}"
                / "scoring_manifest.json",
                root
                / f"rx_{receiver}"
                / f"method_{method_seed}"
                / f"new{new_count}"
                / "scorer"
                / "scoring_manifest.json",
            )
        )
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one scoring manifest for rx={entry['receiver']}, "
            f"seed={method_seed}, new={new_count}; found={len(matches)}"
        )
    return matches[0]


def main() -> int:
    args = _parser().parse_args()
    matrix = json.loads(Path(args.matrix_index).read_text(encoding="utf-8-sig"))
    entries = matrix.get("entries", [])
    if (
        matrix.get("status") != "PREDICTIONS_COMPLETE_TRUTH_UNOPENED"
        or int(matrix.get("paired_input_identity_count", -1)) != EXPECTED_INPUT_IDENTITIES
        or len(entries) != EXPECTED_METHOD_ROWS
    ):
        raise ValueError("D1 refit prediction matrix is incomplete")
    _validate_complete_matrix(entries)
    scoring_roots = [Path(args.scoring_root).absolute()]
    if args.supplemental_scoring_root:
        scoring_roots.append(Path(args.supplemental_scoring_root).absolute())
    roots = tuple(scoring_roots)
    output_root = Path(args.output_root).absolute()
    output_root.mkdir(parents=True, exist_ok=False)
    groups: dict[tuple[str, int, int, int], dict[str, dict[str, Any]]] = {}
    scored_entries: list[dict[str, Any]] = []
    for entry in entries:
        receipt = json.loads(Path(entry["receipt_path"]).read_text(encoding="utf-8-sig"))
        scoring_manifest = _scoring_manifest(roots, entry)
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
            scoring_manifest_path=str(scoring_manifest),
            scoring_manifest_sha256=_sha256(scoring_manifest),
        )
        four_state_path = row_root / "four_state_score.json"
        write_row_record_exclusive(four_state_path, four_state)
        key = (entry["receiver"], entry["method_seed"], entry["k_shot"], entry["new_class_count"])
        groups.setdefault(key, {})[entry["arm"]] = {
            "entry": entry,
            "receipt": receipt,
            "four_state": four_state,
            "row_root": row_root,
            "scoring_manifest": scoring_manifest,
        }
        scored_entries.append({
            **entry,
            "same_row_score_path": str(score_path),
            "four_state_score_path": str(four_state_path),
        })
    paired_outputs = []
    for key, group in groups.items():
        if set(group) != {D0, D1, D1_REFIT}:
            raise ValueError(f"R0/R1/R2 group incomplete: {key}")
        reference = group[D0]
        reference_prediction = reference["receipt"]["prediction"]
        for arm in (D1, D1_REFIT):
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
                scoring_manifest_sha256=_sha256(candidate["scoring_manifest"]),
                bootstrap_repeats=args.bootstrap_repeats,
                bootstrap_seed=2412,
            )
            paired_path = candidate["row_root"] / "paired_vs_r0.json"
            write_row_record_exclusive(paired_path, paired)
            forgetting = standardized_forgetting(reference["four_state"], candidate["four_state"])
            forgetting_path = candidate["row_root"] / "standardized_forgetting.json"
            write_row_record_exclusive(forgetting_path, {
                "schema": "cvs.erbt_idr.m24.standardized_forgetting.v1",
                "reference_arm": D0,
                "candidate_arm": arm,
                "scenario_rows": forgetting,
            })
            paired_outputs.append({
                "group": list(key),
                "candidate_arm": arm,
                "paired_path": str(paired_path),
                "standardized_forgetting_path": str(forgetting_path),
            })
    result = {
        "schema": SCORED_MATRIX_SCHEMA,
        "status": "PASS",
        "row_count": len(scored_entries),
        "paired_input_identity_count": EXPECTED_INPUT_IDENTITIES,
        "method_rows_per_arm": EXPECTED_INPUT_IDENTITIES,
        "scenario_unit_count": EXPECTED_METHOD_ROWS * 3,
        "primary_d92_e0_baseline": "P2-A1_NO_RF32",
        "reference_arm": D0,
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
