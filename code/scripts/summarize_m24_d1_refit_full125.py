#!/usr/bin/env python3
"""Build the complete machine-readable analysis for the M2.4 D1 full-125 matrix."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ARMS = (
    "M24-D0-HISTORICAL-F1",
    "M24-D1-COMPILE-PARITY",
    "M24-D1-REFIT",
)
REFERENCE_ARM = ARMS[0]
EXPECTED_INPUT_IDENTITIES = 125
SUMMARY_SCHEMA = "cvs.erbt_idr.m24.d1_refit_full125.results_summary.v1"
SUMMARY_VERDICT = "D1_COMPILE_PARITY_PASS_AND_R2_FULL125_MEASURED"
METRICS = ("A_o_pre", "A_o_post", "A_n", "H", "F", "min_old", "min_new")
RESOURCE_METRICS = (
    "state_bytes",
    "deployment_state_bytes",
    "registration_time_ms",
    "candidate_head_batch_query_latency_ms_per_row",
    "query_head_mac",
    "mac_equivalent_upper_bound",
    "closed_form_fit_count",
)


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _describe(values: Iterable[float]) -> dict[str, float | int]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"count": 0}
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "population_std": statistics.pstdev(clean),
        "min": min(clean),
        "p05": _quantile(clean, 0.05),
        "p25": _quantile(clean, 0.25),
        "median": _quantile(clean, 0.50),
        "p75": _quantile(clean, 0.75),
        "p95": _quantile(clean, 0.95),
        "max": max(clean),
    }


def _weighted_mean(rows: Iterable[Mapping[str, Any]], key: str, weight_key: str) -> float:
    members = [row for row in rows if row.get(key) is not None]
    total = sum(float(row[weight_key]) for row in members)
    if total <= 0:
        raise ValueError(f"nonpositive weight for metric {key}")
    return sum(float(row[key]) * float(row[weight_key]) for row in members) / total


def _group_metric_rows(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
    metrics: tuple[str, ...],
    *,
    weight_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output: list[dict[str, Any]] = []
    for identity, members in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        record = {key: value for key, value in zip(group_keys, identity)}
        record["record_count"] = len(members)
        record["query_count"] = int(sum(int(row[weight_key]) for row in members))
        record["metrics"] = {}
        for metric in metrics:
            values = [float(row[metric]) for row in members if row.get(metric) is not None]
            record["metrics"][metric] = {
                "pooled_query_weighted_mean": (
                    _weighted_mean(members, metric, weight_key) if values else None
                ),
                "record_distribution": _describe(values),
            }
        output.append(record)
    return output


def _distribution_groups(
    groups: Mapping[tuple[Any, ...], list[float]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    output = []
    for identity, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        record = {key: value for key, value in zip(keys, identity)}
        record["distribution"] = _describe(values)
        output.append(record)
    return output


def _append_distribution(
    targets: dict[str, dict[tuple[Any, ...], list[float]]],
    *,
    arm: str,
    condition: str,
    receiver: str,
    method_seed: int,
    labels: np.ndarray,
    values: np.ndarray,
) -> None:
    finite = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(finite)
    labels = np.asarray(labels).astype(str)[mask]
    finite = finite[mask]
    targets["overall"][(arm,)].extend(finite.tolist())
    targets["condition"][(arm, condition)].extend(finite.tolist())
    targets["receiver"][(arm, receiver)].extend(finite.tolist())
    targets["seed"][(arm, method_seed)].extend(finite.tolist())
    for scene in sorted(set(labels.tolist())):
        targets["scene"][(arm, scene)].extend(finite[labels == scene].tolist())


def _paired_group(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for identity, members in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        query_count = sum(int(row["query_count"]) for row in members)
        record = {key: value for key, value in zip(group_keys, identity)}
        record.update(
            {
                "record_count": len(members),
                "query_count": query_count,
                "N_help": sum(int(row["N_help"]) for row in members),
                "N_harm": sum(int(row["N_harm"]) for row in members),
                "reference_accuracy": sum(
                    float(row["reference_accuracy"]) * int(row["query_count"])
                    for row in members
                ) / query_count,
                "candidate_accuracy": sum(
                    float(row["candidate_accuracy"]) * int(row["query_count"])
                    for row in members
                ) / query_count,
                "accuracy_delta": sum(
                    float(row["accuracy_delta"]) * int(row["query_count"])
                    for row in members
                ) / query_count,
                "record_accuracy_delta_distribution": _describe(
                    float(row["accuracy_delta"]) for row in members
                ),
            }
        )
        pvalues = [float(row["mcnemar_exact_pvalue"]) for row in members if row.get("mcnemar_exact_pvalue") is not None]
        if pvalues:
            record["mcnemar_rows_p_lt_0_05"] = sum(value < 0.05 for value in pvalues)
            record["mcnemar_pvalue_distribution"] = _describe(pvalues)
        output.append(record)
    return output


def _delta_group(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for identity, members in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        record = {key: value for key, value in zip(group_keys, identity)}
        record["pair_count"] = len(members)
        record["metric_deltas"] = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in members]
            record["metric_deltas"][metric] = {
                "distribution": _describe(values),
                "positive_count": sum(value > 1e-12 for value in values),
                "negative_count": sum(value < -1e-12 for value in values),
                "zero_count": sum(abs(value) <= 1e-12 for value in values),
            }
        output.append(record)
    return output


def build_summary(prediction_root: Path, score_root: Path) -> dict[str, Any]:
    matrix = _load(prediction_root / "matrix_index.json")
    scored = _load(score_root / "scored_matrix_index.json")
    if matrix.get("status") != "PREDICTIONS_COMPLETE_TRUTH_UNOPENED":
        raise ValueError("prediction matrix is not truth-unopened complete")
    expected_rows = EXPECTED_INPUT_IDENTITIES * len(ARMS)
    if scored.get("status") != "PASS" or int(scored.get("row_count", -1)) != expected_rows:
        raise ValueError("scored matrix is incomplete")
    identities = {entry["row_id"]: entry for entry in scored["entries"]}
    if len(identities) != expected_rows:
        raise ValueError("scored matrix row identities are not unique")

    row_records: list[dict[str, Any]] = []
    scenario_records: list[dict[str, Any]] = []
    class_records: list[dict[str, Any]] = []
    four_state_records: list[dict[str, Any]] = []
    forgetting_records: list[dict[str, Any]] = []
    paired_records: list[dict[str, Any]] = []
    paired_role_records: list[dict[str, Any]] = []
    paired_scene_records: list[dict[str, Any]] = []
    paired_class_records: list[dict[str, Any]] = []
    margin_groups = {name: defaultdict(list) for name in ("overall", "condition", "receiver", "seed", "scene")}
    angle_groups = {name: defaultdict(list) for name in ("overall", "condition", "receiver", "seed", "scene")}

    for score_path in sorted(score_root.rglob("same_row_score.json")):
        score = _load(score_path)
        row_id = str(score["logical_row_key"])
        identity = identities[row_id]
        arm = str(identity["arm"])
        condition = f"K{identity['k_shot']}_new{identity['new_class_count']}"
        scenario_rows = []
        for scenario in score["scenario_rows"]:
            record = {
                "row_id": row_id,
                "arm": arm,
                "receiver": identity["receiver"],
                "method_seed": identity["method_seed"],
                "k_shot": identity["k_shot"],
                "new_class_count": identity["new_class_count"],
                "condition": condition,
                "scene": scenario["scenario"],
                "query_count": int(scenario["query_count"]),
                **{metric: float(scenario[metric]) for metric in METRICS},
            }
            scenario_records.append(record)
            scenario_rows.append(record)
            for class_label, accuracy in scenario["per_class_accuracy"].items():
                confusion = scenario["per_class_confusion"][class_label]
                class_records.append(
                    {
                        **{key: record[key] for key in ("arm", "receiver", "method_seed", "condition", "scene")},
                        "class_label": class_label,
                        "query_count": int(sum(int(value) for value in confusion.values())),
                        "accuracy": float(accuracy),
                    }
                )
        total_queries = sum(row["query_count"] for row in scenario_rows)
        aggregate = {metric: _weighted_mean(scenario_rows, metric, "query_count") for metric in METRICS}
        resource = score["resource"]
        receipt = _load(Path(identity["receipt_path"]))
        diagnostic_path = Path(receipt["truth_blind_diagnostics"]["path"])
        with np.load(diagnostic_path, allow_pickle=False) as diagnostics:
            _append_distribution(
                margin_groups,
                arm=arm,
                condition=condition,
                receiver=str(identity["receiver"]),
                method_seed=int(identity["method_seed"]),
                labels=diagnostics["scenarios"],
                values=diagnostics["top2_margin"],
            )
            _append_distribution(
                angle_groups,
                arm=arm,
                condition=condition,
                receiver=str(identity["receiver"]),
                method_seed=int(identity["method_seed"]),
                labels=diagnostics["center_scenarios"],
                values=diagnostics["center_angle_degrees"],
            )
        row_records.append(
            {
                "row_id": row_id,
                "arm": arm,
                "receiver": identity["receiver"],
                "method_seed": identity["method_seed"],
                "support_seed": identity["support_seed"],
                "query_seed": identity["query_seed"],
                "new_class_draw_seed": identity["new_class_draw_seed"],
                "split_id": identity["split_id"],
                "k_shot": identity["k_shot"],
                "new_class_count": identity["new_class_count"],
                "condition": condition,
                "query_count": total_queries,
                **aggregate,
                "status": score["status"],
                "truth_opened_after_prediction_commit": score["truth_opened_after_prediction_commit"],
                "parity": identity["d1_historical_parity"],
                **{key: resource.get(key) for key in RESOURCE_METRICS},
            }
        )

        four_state = _load(score_path.parent / "four_state_score.json")
        for scenario in four_state["scenario_rows"]:
            for state_name, state in scenario["states"].items():
                four_state_records.append(
                    {
                        "arm": arm,
                        "receiver": identity["receiver"],
                        "method_seed": identity["method_seed"],
                        "condition": condition,
                        "scene": scenario["scenario"],
                        "state": state_name,
                        "query_count": int(scenario["query_count"]),
                        "old_accuracy": state["old_accuracy"],
                        "new_accuracy": state["new_accuracy"],
                        "H_old_new": state["H_old_new"],
                    }
                )

        forgetting_path = score_path.parent / "standardized_forgetting.json"
        if forgetting_path.is_file():
            for scenario in _load(forgetting_path)["scenario_rows"]:
                query_count = next(row["query_count"] for row in scenario_rows if row["scene"] == scenario["scenario"])
                forgetting_records.append(
                    {
                        "candidate_arm": arm,
                        "receiver": identity["receiver"],
                        "method_seed": identity["method_seed"],
                        "condition": condition,
                        "scene": scenario["scenario"],
                        "query_count": query_count,
                        **{key: float(scenario[key]) for key in ("A_o_pre_within", "A_o_pre_reference_r0", "A_o_post", "F_within", "F_std")},
                    }
                )

        paired_path = score_path.parent / "paired_vs_r0.json"
        if paired_path.is_file():
            paired = _load(paired_path)
            base = {
                "candidate_arm": arm,
                "receiver": identity["receiver"],
                "method_seed": identity["method_seed"],
                "condition": condition,
            }
            paired_records.append(
                {
                    **base,
                    **{key: paired[key] for key in ("query_count", "N_help", "N_harm", "reference_accuracy", "candidate_accuracy", "accuracy_delta", "mcnemar_exact_pvalue")},
                }
            )
            for role, values in paired["by_role"].items():
                paired_role_records.append({**base, "role": role, **values})
            for scene, values in paired["by_scenario"].items():
                paired_scene_records.append({**base, "scene": scene, **values})
            for class_handle, values in paired["by_true_class"].items():
                paired_class_records.append({**base, "class_handle": class_handle, **values})

    arm_counts = defaultdict(int)
    for row in row_records:
        arm_counts[row["arm"]] += 1
    if dict(arm_counts) != {arm: EXPECTED_INPUT_IDENTITIES for arm in ARMS}:
        raise ValueError(f"unexpected arm counts: {dict(arm_counts)}")
    parity_rows = [row for row in row_records if row["arm"] == "M24-D1-COMPILE-PARITY"]
    parity_before = sum(int(row["parity"]["before_prediction_disagreements"]) for row in parity_rows)
    parity_after = sum(int(row["parity"]["prediction_disagreements"]) for row in parity_rows)
    if parity_before or parity_after:
        raise ValueError("R1 parity disagreements are nonzero")

    by_identity = {
        (row["receiver"], row["method_seed"], row["condition"], row["arm"]): row
        for row in row_records
    }
    delta_rows = []
    condition_names = [
        f"K{condition['k_shot']}_new{condition['new_class_count']}"
        for condition in matrix["conditions"]
    ]
    for candidate_arm in ARMS[1:]:
        for receiver in matrix["receivers"]:
            for seed in matrix["method_seeds"]:
                for condition in condition_names:
                    reference = by_identity[(receiver, seed, condition, REFERENCE_ARM)]
                    candidate = by_identity[(receiver, seed, condition, candidate_arm)]
                    delta_rows.append(
                        {
                            "candidate_arm": candidate_arm,
                            "receiver": receiver,
                            "method_seed": seed,
                            "condition": condition,
                            **{metric: candidate[metric] - reference[metric] for metric in METRICS},
                        }
                    )
    scenario_identity = {
        (row["receiver"], row["method_seed"], row["condition"], row["scene"], row["arm"]): row
        for row in scenario_records
    }
    scenario_delta_rows = []
    for candidate_arm in ARMS[1:]:
        for key, reference in scenario_identity.items():
            receiver, seed, condition, scene, arm = key
            if arm != REFERENCE_ARM:
                continue
            candidate = scenario_identity[(receiver, seed, condition, scene, candidate_arm)]
            scenario_delta_rows.append(
                {
                    "candidate_arm": candidate_arm,
                    "receiver": receiver,
                    "method_seed": seed,
                    "condition": condition,
                    "scene": scene,
                    **{metric: candidate[metric] - reference[metric] for metric in METRICS},
                }
            )

    margin_thresholds = {}
    for (arm,), values in margin_groups["overall"].items():
        margin_thresholds[arm] = {
            "count": len(values),
            "fraction_le_1e-3": sum(value <= 1e-3 for value in values) / len(values),
            "fraction_le_1e-2": sum(value <= 1e-2 for value in values) / len(values),
            "fraction_le_5e-2": sum(value <= 5e-2 for value in values) / len(values),
        }

    summary = {
        "schema": SUMMARY_SCHEMA,
        "run_id": matrix["run_id"],
        "status": "ANALYZED",
        "verdict": SUMMARY_VERDICT,
        "evidence_boundary": "Same-row full-125 Stage2-C evidence under p2_min_v1; not Phase3 or deployment evidence.",
        "protocol": {
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "prediction_matrix_status": matrix["status"],
            "prediction_truth_opened": matrix["query_truth_opened"],
            "truth_opened_after_all_predictions_complete": scored["truth_opened_after_all_predictions_complete"],
            "scorer_output_must_not_feed_predictor": scored["scorer_output_must_not_feed_predictor"],
        },
        "matrix": {
            "row_count": len(row_records),
            "paired_input_identity_count": matrix["paired_input_identity_count"],
            "scenario_unit_count": sum(3 for _ in row_records),
            "receivers": matrix["receivers"],
            "method_seeds": matrix["method_seeds"],
            "conditions": matrix["conditions"],
            "arm_counts": dict(sorted(arm_counts.items())),
            "pass_rows": sum(row["status"] == "PASS" for row in row_records),
            "R1_before_prediction_disagreements": parity_before,
            "R1_after_prediction_disagreements": parity_after,
            "primary_d92_e0_baseline": matrix["primary_d92_e0_baseline"],
        },
        "metric_semantics": {
            "A_o_pre": "registered-old accuracy before new-class registration",
            "A_o_post": "registered-old accuracy after new-class registration",
            "A_n": "registered-new accuracy after registration",
            "H": "old/new harmonic mean after registration",
            "F": "A_o_pre - A_o_post",
            "F_within": "candidate A_o_pre - candidate A_o_post",
            "F_std": "R0 A_o_pre - candidate A_o_post",
            "aggregate": "query-count-weighted across leo_clear_weak, leo_low_elev_weak, and leo_rain_weak",
        },
        "arm_summary": _group_metric_rows(row_records, ("arm",), METRICS, weight_key="query_count"),
        "condition_summary": _group_metric_rows(row_records, ("arm", "condition"), METRICS, weight_key="query_count"),
        "receiver_summary": _group_metric_rows(row_records, ("arm", "receiver"), METRICS, weight_key="query_count"),
        "seed_summary": _group_metric_rows(row_records, ("arm", "method_seed"), METRICS, weight_key="query_count"),
        "scene_summary": _group_metric_rows(scenario_records, ("arm", "scene"), METRICS, weight_key="query_count"),
        "old_new_summary": {
            "old": _group_metric_rows(row_records, ("arm",), ("A_o_pre", "A_o_post", "F", "min_old"), weight_key="query_count"),
            "new": _group_metric_rows(row_records, ("arm",), ("A_n", "H", "min_new"), weight_key="query_count"),
        },
        "four_state_summary": {
            "overall": _group_metric_rows(four_state_records, ("arm", "state"), ("old_accuracy", "new_accuracy", "H_old_new"), weight_key="query_count"),
            "scene": _group_metric_rows(four_state_records, ("arm", "state", "scene"), ("old_accuracy", "new_accuracy", "H_old_new"), weight_key="query_count"),
        },
        "same_row_delta_vs_r0": {
            "overall": _delta_group(delta_rows, ("candidate_arm",)),
            "condition": _delta_group(delta_rows, ("candidate_arm", "condition")),
            "receiver": _delta_group(delta_rows, ("candidate_arm", "receiver")),
            "seed": _delta_group(delta_rows, ("candidate_arm", "method_seed")),
            "scene": _delta_group(scenario_delta_rows, ("candidate_arm", "scene")),
        },
        "help_harm": {
            "overall": _paired_group(paired_records, ("candidate_arm",)),
            "condition": _paired_group(paired_records, ("candidate_arm", "condition")),
            "receiver": _paired_group(paired_records, ("candidate_arm", "receiver")),
            "seed": _paired_group(paired_records, ("candidate_arm", "method_seed")),
            "role": _paired_group(paired_role_records, ("candidate_arm", "role")),
            "scene": _paired_group(paired_scene_records, ("candidate_arm", "scene")),
            "class": _paired_group(paired_class_records, ("candidate_arm", "class_handle")),
        },
        "forgetting": {
            "overall": _group_metric_rows(forgetting_records, ("candidate_arm",), ("A_o_pre_within", "A_o_pre_reference_r0", "A_o_post", "F_within", "F_std"), weight_key="query_count"),
            "condition": _group_metric_rows(forgetting_records, ("candidate_arm", "condition"), ("F_within", "F_std"), weight_key="query_count"),
            "receiver": _group_metric_rows(forgetting_records, ("candidate_arm", "receiver"), ("F_within", "F_std"), weight_key="query_count"),
            "seed": _group_metric_rows(forgetting_records, ("candidate_arm", "method_seed"), ("F_within", "F_std"), weight_key="query_count"),
            "scene": _group_metric_rows(forgetting_records, ("candidate_arm", "scene"), ("F_within", "F_std"), weight_key="query_count"),
        },
        "class_summary": {
            "overall": _group_metric_rows(class_records, ("arm", "class_label"), ("accuracy",), weight_key="query_count"),
            "scene": _group_metric_rows(class_records, ("arm", "scene", "class_label"), ("accuracy",), weight_key="query_count"),
        },
        "margin": {
            "threshold_fractions": margin_thresholds,
            "overall": _distribution_groups(margin_groups["overall"], ("arm",)),
            "condition": _distribution_groups(margin_groups["condition"], ("arm", "condition")),
            "receiver": _distribution_groups(margin_groups["receiver"], ("arm", "receiver")),
            "seed": _distribution_groups(margin_groups["seed"], ("arm", "method_seed")),
            "scene": _distribution_groups(margin_groups["scene"], ("arm", "scene")),
        },
        "center_angle_degrees": {
            "overall": _distribution_groups(angle_groups["overall"], ("arm",)),
            "condition": _distribution_groups(angle_groups["condition"], ("arm", "condition")),
            "receiver": _distribution_groups(angle_groups["receiver"], ("arm", "receiver")),
            "seed": _distribution_groups(angle_groups["seed"], ("arm", "method_seed")),
            "scene": _distribution_groups(angle_groups["scene"], ("arm", "scene")),
        },
        "resource_summary": _group_metric_rows(row_records, ("arm",), RESOURCE_METRICS, weight_key="query_count"),
        "rows": row_records,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.prediction_root, args.score_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": summary["status"], **summary["matrix"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
