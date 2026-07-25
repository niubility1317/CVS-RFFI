#!/usr/bin/env python3
"""Apply the frozen D104 source-held promotion gate without selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-json", type=Path, required=True)
    parser.add_argument("--tx-probe-json", type=Path, required=True)
    parser.add_argument("--matrix-status-json", type=Path, required=True)
    parser.add_argument("--runner-resource-json", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_json.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable D104 gate exists: {output}")
    scores = _read(args.scores_json.resolve(strict=True))
    tx = _read(args.tx_probe_json.resolve(strict=True))
    matrix = _read(args.matrix_status_json.resolve(strict=True))
    runner = _read(args.runner_resource_json.resolve(strict=True))
    split = _read(args.source_split_manifest.resolve(strict=True))
    rows = scores.get("performance_rows")
    if (
        scores.get("schema") != "cvs.d104_r1.rxid_angq.held_scores.v1"
        or not isinstance(rows, list)
        or len(rows) != 63
        or matrix.get("status") != "ARTIFACTS_COMPLETE"
        or matrix.get("planned_fit_count") != 246
        or matrix.get("completed_fit_count") != 246
        or matrix.get("failed_fit_count") != 0
        or split.get("split_id") != "d104_source_seed104713_v2"
        or split.get("partition", {}).get("counts")
        != {"L_s": 588, "U_s": 5292, "source_val": 2520}
    ):
        raise ValueError("D104 gate input closure drift")
    keys = {
        (row["held_receiver"], row["held_class"], int(row["K"]))
        for row in rows
    }
    receivers = sorted({key[0] for key in keys})
    classes = sorted({key[1] for key in keys if key[1] is not None})
    expected = {
        (receiver, None, k_shot)
        for receiver in receivers
        for k_shot in (1, 5, 10)
    } | {
        (receiver, class_id, 1)
        for receiver in receivers
        for class_id in classes
    }
    if len(receivers) != 7 or len(classes) != 6 or keys != expected:
        raise ValueError("D104 gate row identity matrix drift")

    quant = scores["quantization_receipt"]
    gates: dict[str, bool] = {
        "complete_63_row_matrix": True,
        "complete_252_arm_row_predictions": True,
        "matrix_246_fit_complete": True,
        "source_split_v2_closed": True,
        "tx_probe_le_25pct": float(tx["max_fold_score"]) <= 0.25,
        "M_HEAD_int8_each_row": float(
            quant["M_HEAD_min_top1_agreement"]
        )
        >= 0.995
        and int(quant["M_HEAD_margin_flip_count"]) == 0,
        "M_JOINT_int8_each_row": (
            float(quant["M_JOINT_min_top1_agreement"]) >= 0.995
            and int(quant["M_JOINT_margin_flip_count"]) == 0
        ),
        "head_simple_effect_row_nonnegative": all(
            float(row["simple_effects"]["H0_HEAD_at_base"]["balanced_accuracy"])
            >= -1.0e-12
            and float(row["simple_effects"]["H0_HEAD_at_base"]["per_class_floor"])
            >= -1.0e-12
            and int(row["simple_effects"]["H0_HEAD_at_base"]["correct_count"]) >= 0
            for row in rows
        ),
        "joint_simple_effect_row_nonnegative": all(
            float(row["simple_effects"]["D1_DA_at_ANGQ"]["balanced_accuracy"])
            >= -1.0e-12
            and float(row["simple_effects"]["D1_DA_at_ANGQ"]["per_class_floor"])
            >= -1.0e-12
            and int(row["simple_effects"]["D1_DA_at_ANGQ"]["correct_count"]) >= 0
            for row in rows
        ),
        "all_factorial_effects_reported": all(
            set(row["simple_effects"])
            == {
                "H0_HEAD_at_base",
                "H1_HEAD_at_DA",
                "D0_DA_at_legacy",
                "D1_DA_at_ANGQ",
            }
            and set(row["head_main_effect"])
            == {
                "balanced_accuracy",
                "per_class_floor",
                "joint_score",
                "correct_count",
            }
            and set(row["da_main_effect"])
            == {
                "balanced_accuracy",
                "per_class_floor",
                "joint_score",
                "correct_count",
            }
            and set(row["interaction"])
            == {
                "balanced_accuracy",
                "per_class_floor",
                "joint_score",
                "correct_count",
            }
            for row in rows
        ),
        "resource_query_mac_delta_zero": (
            int(scores["resource_component"]["query_mac_delta"]) == 0
        ),
        "runner_resource_receipt_present": bool(runner),
    }
    mean_joint = {
        arm: math.fsum(
            float(row["arm_metrics"][arm]["joint_score"]) for row in rows
        )
        / 63.0
        for arm in ("M0", "M_DA", "M_HEAD", "M_JOINT")
    }
    gates["M_HEAD_mean_joint_strictly_above_M0"] = (
        mean_joint["M_HEAD"] > mean_joint["M0"]
    )
    gates["M_JOINT_mean_joint_strictly_above_M_HEAD"] = (
        mean_joint["M_JOINT"] > mean_joint["M_HEAD"]
    )
    k1_rows = [row for row in rows if int(row["K"]) == 1]
    gates["K1_49_row_evidence_present"] = (
        len(k1_rows) == 49 and len(scores.get("day_stability_rows", [])) == 49
    )
    gates["K1_identifiability_each_row"] = all(
        isinstance(row.get("k1_evidence"), dict)
        and row["k1_evidence"]["active"] is True
        and int(row["k1_evidence"]["information_rank"]) == 4
        and float(row["k1_evidence"]["minimum_singular_value"]) >= 0.05
        and float(row["k1_evidence"]["condition_number"]) <= 10.0
        and float(row["k1_evidence"]["prior_fraction"]) <= 0.80
        and float(row["k1_evidence"]["coefficient_norm"]) >= 1.0e-4
        and float(row["k1_evidence"]["view_top1_agreement"]) >= 0.995
        and int(row["k1_evidence"]["view_margin_flip_count"]) == 0
        and float(row["k1_evidence"]["direction_cosine_median"]) >= 0.80
        for row in k1_rows
    )
    passed = all(gates.values())
    result = {
        "schema": "cvs.d104_r1.rxid_angq.held_gate.v1",
        "status": (
            "TARGET25_GATE_ELIGIBLE"
            if passed
            else "D104_REJECTED_SOURCE_HELD_GATE"
        ),
        "gates": gates,
        "mean_joint_score": mean_joint,
        "target25_gate_eligible": passed,
        "target25_authorized": False,
        "performance_selection_used": False,
        "row_subset_selected": False,
        "target_access": False,
    }
    _write_new(output, result)
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
