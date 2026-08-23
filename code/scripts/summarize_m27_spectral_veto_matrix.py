#!/usr/bin/env python3
"""Build the machine-readable M2.7 screen/full125 analysis and promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cvsrffi.stage2_m24_safe_residual import D1
from cvsrffi.stage2_m25_anchored_residual import B3
from cvsrffi.stage2_m27_spectral_veto import M27_SPECTRAL_VETO_ARMS
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
        if entry["arm"] not in M27_SPECTRAL_VETO_ARMS:
            continue
        receipt = json.loads(
            Path(entry["receipt_path"]).read_text(encoding="utf-8-sig")
        )
        for scene, audit in sorted(receipt["scenario_audit"].items()):
            fit = audit["representation_fit"]
            application = audit["query_application"]
            query_count = int(application["query_count"])
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
                    "reliability_accepted": float(fit["reliability_accepted"]),
                    "loo_accuracy": float(fit["loo_accuracy"]),
                    "old_loo_accuracy": float(fit["old_loo_accuracy"]),
                    "new_loo_accuracy": float(fit["new_loo_accuracy"]),
                    "margin_threshold": float(fit["margin_threshold"]),
                    "b3_flip_count": int(application["b3_flip_count"]),
                    "accepted_b3_flip_count": int(
                        application.get("accepted_b3_flip_count", 0)
                    ),
                    "vetoed_b3_flip_count": int(
                        application["vetoed_b3_flip_count"]
                    ),
                    "selected_b3_count": int(application["selected_b3_count"]),
                    "accepted_b3_flip_fraction": (
                        float(application.get("accepted_b3_flip_count", 0))
                        / max(1, int(application["b3_flip_count"]))
                    ),
                    "vetoed_b3_flip_fraction": (
                        float(application["vetoed_b3_flip_count"])
                        / max(1, int(application["b3_flip_count"]))
                    ),
                    "selected_b3_fraction": float(application["selected_b3_count"])
                    / query_count,
                    "state_digest": fit["state_digest"],
                }
            )
    metrics = (
        "b3_selected_strength",
        "reliability_accepted",
        "loo_accuracy",
        "old_loo_accuracy",
        "new_loo_accuracy",
        "margin_threshold",
        "b3_flip_count",
        "accepted_b3_flip_count",
        "vetoed_b3_flip_count",
        "selected_b3_count",
        "accepted_b3_flip_fraction",
        "vetoed_b3_flip_fraction",
        "selected_b3_fraction",
    )
    group = lambda keys: shared._group_metric_rows(
        rows, keys, metrics, weight_key="query_count"
    )
    return {
        "metric_semantics": {
            "loo_accuracy": "support-only class competition leave-one-out accuracy",
            "b3_flip_count": "queries where B3 changes the B0 argmax",
            "vetoed_b3_flip_count": "B3 flips rejected by representation consensus",
            "selected_b3_count": "complete score rows selected from B3 rather than B0",
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
        if entry["arm"] not in M27_SPECTRAL_VETO_ARMS:
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
    for arm in M27_SPECTRAL_VETO_ARMS:
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
    for arm in M27_SPECTRAL_VETO_ARMS:
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
        "decision": (
            "PROMOTE_TO_FULL125" if passed else "SCREEN_NEGATIVE_NO_FULL125"
        ),
        "eligible_arms": list(M27_SPECTRAL_VETO_ARMS),
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
    shared.SUMMARY_SCHEMA = "cvs.erbt_idr.m27.spectral_veto.results_summary.v1"
    shared.SUMMARY_VERDICT = "M27_SPECTRAL_VETO_MATRIX_MEASURED"
    result = shared.build_summary(prediction_root, score_root)
    matrix_kind = str(matrix.get("matrix_kind"))
    result["matrix"]["matrix_kind"] = matrix_kind
    result["evidence_boundary"] = (
        f"Same-row {matrix['paired_input_identity_count']}-identity screening evidence under p2_min_v1; not full-125 confirmation, Phase3, or deployment evidence."
        if matrix_kind == "screen"
        else "Same-row full-125 Stage2-C evidence under p2_min_v1; not Phase3 or deployment evidence."
    )
    result["m27_diagnostics"] = _diagnostics(matrix)
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
