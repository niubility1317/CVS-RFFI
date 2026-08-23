#!/usr/bin/env python3
"""Build the machine-readable M2.8 screen/full125 analysis and promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from cvsrffi.stage2_m24_safe_residual import D1
from cvsrffi.stage2_m25_anchored_residual import B3
from cvsrffi.stage2_m28_local_flip_risk import M28_LOCAL_RISK_ARMS
from scripts import summarize_m24_d1_refit_full125 as shared


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _metric(result: dict[str, Any], arm: str, name: str) -> float:
    row = next(item for item in result["arm_summary"] if item["arm"] == arm)
    return float(row["metrics"][name]["pooled_query_weighted_mean"])


def _help_harm(result: dict[str, Any], arm: str) -> tuple[int, int]:
    row = next(
        item
        for item in result["help_harm"]["overall"]
        if item["candidate_arm"] == arm
    )
    return int(row["N_help"]), int(row["N_harm"])


def _diagnostics(matrix: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for entry in matrix["entries"]:
        if entry["arm"] not in M28_LOCAL_RISK_ARMS:
            continue
        receipt = json.loads(
            Path(entry["receipt_path"]).read_text(encoding="utf-8-sig")
        )
        for scene, audit in sorted(receipt["scenario_audit"].items()):
            fit = audit["risk_fit"]
            application = audit["query_application"]
            query_count = int(application["query_count"])
            class_accuracy = np.asarray(fit["class_loo_accuracy"], dtype=np.float64)
            rows.append(
                {
                    "arm": entry["arm"],
                    "receiver": entry["receiver"],
                    "method_seed": int(entry["method_seed"]),
                    "condition": f"K{entry['k_shot']}_new{entry['new_class_count']}",
                    "scene": scene,
                    "query_count": query_count,
                    "representation": audit["representation"],
                    "b3_selected_strength": float(audit["b3"]["selected_strength"]),
                    "rank1_event_count": int(fit["rank_event_count"][0]),
                    "rank2_event_count": int(fit["rank_event_count"][1]),
                    "rank1_success_count": int(fit["rank_success_count"][0]),
                    "rank2_success_count": int(fit["rank_success_count"][1]),
                    "class_loo_accuracy_mean": float(np.mean(class_accuracy)),
                    "class_loo_accuracy_min": float(np.min(class_accuracy)),
                    "class_loo_zero_count": int(np.sum(class_accuracy <= 0.0)),
                    "nonzero_pair_count": int(fit["nonzero_pair_count"]),
                    "radial_location": float(fit["radial_location"]),
                    "b3_flip_count": int(application["b3_flip_count"]),
                    "accepted_b3_flip_count": int(application["accepted_b3_flip_count"]),
                    "accepted_rank1_flip_count": int(application["accepted_rank1_flip_count"]),
                    "accepted_rank2_flip_count": int(application["accepted_rank2_flip_count"]),
                    "vetoed_b3_flip_count": int(application["vetoed_b3_flip_count"]),
                    "selected_b3_count": int(application["selected_b3_count"]),
                    "accepted_b3_flip_fraction": float(application["accepted_b3_flip_count"])
                    / max(1, int(application["b3_flip_count"])),
                    "selected_b3_fraction": float(application["selected_b3_count"])
                    / query_count,
                    "mean_accepted_posterior": application.get("mean_accepted_posterior"),
                    "mean_accepted_conformal_p": application.get("mean_accepted_conformal_p"),
                    "mean_accepted_radial_p": application.get("mean_accepted_radial_p"),
                    "mean_pair_direct_event_count": float(
                        application["mean_pair_direct_event_count"]
                    ),
                    "fallback_policy": fit.get("fallback_policy"),
                    "state_digest": fit["state_digest"],
                }
            )
    metrics = (
        "b3_selected_strength",
        "rank1_event_count",
        "rank2_event_count",
        "rank1_success_count",
        "rank2_success_count",
        "class_loo_accuracy_mean",
        "class_loo_accuracy_min",
        "class_loo_zero_count",
        "nonzero_pair_count",
        "radial_location",
        "b3_flip_count",
        "accepted_b3_flip_count",
        "accepted_rank1_flip_count",
        "accepted_rank2_flip_count",
        "vetoed_b3_flip_count",
        "selected_b3_count",
        "accepted_b3_flip_fraction",
        "selected_b3_fraction",
        "mean_accepted_posterior",
        "mean_accepted_conformal_p",
        "mean_accepted_radial_p",
        "mean_pair_direct_event_count",
    )
    group = lambda keys: shared._group_metric_rows(
        rows, keys, metrics, weight_key="query_count"
    )
    return {
        "metric_semantics": {
            "rank_event_count": "support LOO events where MGD rank candidate differs from B0 source class",
            "class_loo_accuracy": "destination-local support LOO stability, never query accuracy",
            "accepted_b3_flip_count": "complete B3 rows accepted for queries whose B3 argmax differs from B0",
            "mean_accepted_posterior": "hierarchically shrunken support-only pair posterior mean",
            "mean_accepted_conformal_p": "destination class support-calibrated conformal p-value",
            "mean_accepted_radial_p": "target-centre radial support-calibrated p-value",
        },
        "overall": group(("arm",)),
        "condition": group(("arm", "condition")),
        "receiver": group(("arm", "receiver")),
        "seed": group(("arm", "method_seed")),
        "scene": group(("arm", "scene")),
        "rows": rows,
    }


def _paired_vs_b3(score_root: Path, scored: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for entry in scored["entries"]:
        if entry["arm"] not in M28_LOCAL_RISK_ARMS:
            continue
        paired = json.loads(
            (score_root / entry["row_id"] / "paired_vs_b3.json").read_text(
                encoding="utf-8-sig"
            )
        )
        rows.append(
            {
                "candidate_arm": entry["arm"],
                "receiver": entry["receiver"],
                "method_seed": int(entry["method_seed"]),
                "condition": f"K{entry['k_shot']}_new{entry['new_class_count']}",
                "query_count": int(paired["query_count"]),
                "N_help": int(paired["N_help"]),
                "N_harm": int(paired["N_harm"]),
                "accuracy_delta": float(paired["accuracy_delta"]),
            }
        )
    output = []
    for arm in M28_LOCAL_RISK_ARMS:
        members = [row for row in rows if row["candidate_arm"] == arm]
        query_count = sum(row["query_count"] for row in members)
        output.append(
            {
                "candidate_arm": arm,
                "row_count": len(members),
                "query_count": query_count,
                "N_help": sum(row["N_help"] for row in members),
                "N_harm": sum(row["N_harm"] for row in members),
                "accuracy_delta": sum(
                    row["accuracy_delta"] * row["query_count"] for row in members
                )
                / query_count,
            }
        )
    return {"overall": output, "rows": rows}


def _gate(result: dict[str, Any]) -> dict[str, Any]:
    observed = {}
    passed = []
    b0_h = _metric(result, D1, "H")
    b3_h = _metric(result, B3, "H")
    b0_min_old = _metric(result, D1, "min_old")
    b0_min_new = _metric(result, D1, "min_new")
    for arm in M28_LOCAL_RISK_ARMS:
        help_count, harm_count = _help_harm(result, arm)
        row = {
            "delta_H_vs_B0": _metric(result, arm, "H") - b0_h,
            "delta_H_vs_B3": _metric(result, arm, "H") - b3_h,
            "N_help_vs_B0": help_count,
            "N_harm_vs_B0": harm_count,
            "delta_min_old_vs_B0": _metric(result, arm, "min_old") - b0_min_old,
            "delta_min_new_vs_B0": _metric(result, arm, "min_new") - b0_min_new,
        }
        row["pass"] = bool(
            row["delta_H_vs_B0"] >= 0.002
            and row["delta_H_vs_B3"] >= 0.0002
            and help_count > harm_count
            and row["delta_min_old_vs_B0"] >= -0.005
            and row["delta_min_new_vs_B0"] >= -0.005
        )
        observed[arm] = row
        if row["pass"]:
            passed.append(arm)
    return {
        "status": "PASS" if passed else "FAIL",
        "decision": "PROMOTE_TO_FULL125" if passed else "SCREEN_NEGATIVE_NO_FULL125",
        "eligible_arms": list(M28_LOCAL_RISK_ARMS),
        "passed_arms": passed,
        "thresholds": {
            "delta_H_vs_B0_min": 0.002,
            "delta_H_vs_B3_min": 0.0002,
            "help_must_exceed_harm_vs_B0": True,
            "max_min_old_drop_vs_B0": 0.005,
            "max_min_new_drop_vs_B0": 0.005,
        },
        "observed": observed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prediction_root = Path(args.prediction_root)
    score_root = Path(args.score_root)
    matrix = json.loads(
        (prediction_root / "matrix_index.json").read_text(encoding="utf-8-sig")
    )
    scored = json.loads(
        (score_root / "scored_matrix_index.json").read_text(encoding="utf-8-sig")
    )
    shared.ARMS = tuple(str(item) for item in matrix["arms"])
    shared.REFERENCE_ARM = D1
    shared.PARITY_ARM = None
    shared.EXPECTED_INPUT_IDENTITIES = int(matrix["paired_input_identity_count"])
    shared.SUMMARY_SCHEMA = "cvs.erbt_idr.m28.local_flip_risk.results_summary.v1"
    shared.SUMMARY_VERDICT = "M28_LOCAL_FLIP_RISK_MATRIX_MEASURED"
    result = shared.build_summary(prediction_root, score_root)
    matrix_kind = str(matrix.get("matrix_kind"))
    result["matrix"]["matrix_kind"] = matrix_kind
    result["evidence_boundary"] = (
        f"Same-row {matrix['paired_input_identity_count']}-identity screening evidence under p2_min_v1; not full-125 confirmation, Phase3, or deployment evidence."
        if matrix_kind == "screen"
        else "Same-row full-125 Stage2-C evidence under p2_min_v1; not Phase3 or deployment evidence."
    )
    result["m28_diagnostics"] = _diagnostics(matrix)
    result["paired_vs_b3"] = _paired_vs_b3(score_root, scored)
    result["screen_gate"] = _gate(result)
    result["verdict"] = result["screen_gate"]["decision"]
    _write_exclusive(Path(args.output), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "row_count": result["matrix"]["row_count"],
                "decision": result["screen_gate"]["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
