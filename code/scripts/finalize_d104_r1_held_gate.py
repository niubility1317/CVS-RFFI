#!/usr/bin/env python3
"""Apply the frozen D104 source-held gate from independently recomputed evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")
METRICS = (
    "balanced_accuracy",
    "per_class_floor",
    "joint_score",
    "correct_count",
)
RESOURCE_SCHEMA = "cvs.phase2.d104_r1.angq.resource_receipt.v1"
INT8_SCHEMA = "cvs.phase2.d104_r1.rxid_angq.int8_audit.v1.four_arm"
ABS_TOL = 1.0e-12
GPU_HOUR_LIMIT = 30.0
PEAK_MEMORY_LIMIT = 4 * 1024**3
DISK_LIMIT = 20 * 1024**3
PEAK_TEMPORARY_LIMIT = 16 * 1024


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    text = str(value)
    if (
        len(text) != 64
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return text


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


def _close(actual: object, expected: float, name: str) -> None:
    value = float(actual)
    if not math.isfinite(value) or not math.isclose(
        value,
        expected,
        rel_tol=0.0,
        abs_tol=ABS_TOL,
    ):
        raise ValueError(f"D104 serialized metric drift: {name}")


def _recompute_arm_metric(
    value: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, float | int]:
    if set(value) != {
        "balanced_accuracy",
        "per_class_floor",
        "joint_score",
        "correct_count",
        "query_count",
        "per_class_correct",
        "per_class_count",
    }:
        raise ValueError(f"D104 arm metric closure drift: {name}")
    correct = value["per_class_correct"]
    counts = value["per_class_count"]
    if (
        not isinstance(correct, list)
        or not isinstance(counts, list)
        or len(correct) != 6
        or len(counts) != 6
        or any(type(item) is not int for item in correct + counts)
        or any(denominator <= 0 for denominator in counts)
        or any(
            numerator < 0 or numerator > denominator
            for numerator, denominator in zip(correct, counts, strict=True)
        )
    ):
        raise ValueError(f"D104 integer class evidence drift: {name}")
    query_count = sum(counts)
    correct_count = sum(correct)
    if (
        type(value["query_count"]) is not int
        or value["query_count"] != query_count
        or type(value["correct_count"]) is not int
        or value["correct_count"] != correct_count
    ):
        raise ValueError(f"D104 integer total evidence drift: {name}")
    rates = [
        numerator / denominator
        for numerator, denominator in zip(correct, counts, strict=True)
    ]
    balanced = math.fsum(rates) / 6.0
    floor = min(rates)
    joint = (balanced + floor) / 2.0
    _close(value["balanced_accuracy"], balanced, f"{name}.balanced_accuracy")
    _close(value["per_class_floor"], floor, f"{name}.per_class_floor")
    _close(value["joint_score"], joint, f"{name}.joint_score")
    return {
        "balanced_accuracy": balanced,
        "per_class_floor": floor,
        "joint_score": joint,
        "correct_count": correct_count,
    }


def _delta(
    left: Mapping[str, float | int],
    right: Mapping[str, float | int],
) -> dict[str, float | int]:
    return {
        metric: (
            int(left[metric]) - int(right[metric])
            if metric == "correct_count"
            else float(left[metric]) - float(right[metric])
        )
        for metric in METRICS
    }


def _verify_effect(
    serialized: Mapping[str, Any],
    expected: Mapping[str, float | int],
    *,
    name: str,
    integer_correct: bool,
) -> None:
    if set(serialized) != set(METRICS):
        raise ValueError(f"D104 effect closure drift: {name}")
    for metric in METRICS:
        if metric == "correct_count" and integer_correct:
            if (
                type(serialized[metric]) is not int
                or serialized[metric] != expected[metric]
            ):
                raise ValueError(f"D104 integer effect drift: {name}.{metric}")
        else:
            _close(serialized[metric], float(expected[metric]), f"{name}.{metric}")


def _verify_resource_receipt(
    receipt: Mapping[str, Any],
    *,
    k_shot: int,
    name: str,
) -> None:
    body = dict(receipt)
    receipt_sha = body.pop("receipt_sha256", None)
    support_count = 6 * k_shot
    if (
        receipt.get("schema") != RESOURCE_SCHEMA
        or receipt_sha != _canonical_sha256(body)
        or receipt.get("registered_class_count") != 6
        or receipt.get("active_k") != k_shot
        or receipt.get("support_count") != support_count
        or receipt.get("factor_count") != 101
        or receipt.get("adaptation_mac_per_support") != 32_320
        or receipt.get("adaptation_mac_total") != 32_320 * support_count
        or receipt.get("adaptation_vector_elementwise_ops_per_support")
        != 64_640
        or receipt.get("adaptation_vector_elementwise_ops_total")
        != 64_640 * support_count
        or receipt.get("numeric_bank_array_bytes_delta") != 0
        or receipt.get("query_mac_before") != receipt.get("query_mac_after")
        or receipt.get("query_mac_delta") != 0
        or int(receipt.get("peak_temporary_bytes_upper_bound", -1))
        > PEAK_TEMPORARY_LIMIT
        or receipt.get("passes_peak_temporary_bytes_gate") is not True
        or int(receipt.get("actual_serialized_state_bytes_after", -1))
        > int(receipt.get("wire_bytes_gate", -2))
        or receipt.get("passes_wire_bytes_gate") is not True
        or receipt.get("passes_d104_resource_gate") is not True
        or receipt.get("query_features_used_for_scale") != 0
        or receipt.get("query_truth_read") is not False
        or receipt.get("query_state_updates") != 0
    ):
        raise ValueError(f"D104 resource receipt drift: {name}")


def _verify_row(
    row: Mapping[str, Any],
) -> tuple[dict[str, dict[str, float | int]], list[Mapping[str, Any]]]:
    receipt_body = dict(row)
    score_receipt = receipt_body.pop("score_receipt_sha256", None)
    if score_receipt != _canonical_sha256(receipt_body):
        raise ValueError("D104 row score receipt drift")
    arm_serialized = row.get("arm_metrics")
    if not isinstance(arm_serialized, Mapping) or tuple(arm_serialized) != ARMS:
        raise ValueError("D104 arm metric registry drift")
    arms = {
        arm: _recompute_arm_metric(
            arm_serialized[arm],
            name=f"{row['held_receiver']}/{row['held_class']}/{row['K']}/{arm}",
        )
        for arm in ARMS
    }
    pair_names = {
        "H0_HEAD_at_base": ("M_HEAD", "M0"),
        "H1_HEAD_at_DA": ("M_JOINT", "M_DA"),
        "D0_DA_at_legacy": ("M_DA", "M0"),
        "D1_DA_at_ANGQ": ("M_JOINT", "M_HEAD"),
    }
    expected_effects = {
        name: _delta(arms[left], arms[right])
        for name, (left, right) in pair_names.items()
    }
    serialized_effects = row.get("simple_effects")
    if (
        not isinstance(serialized_effects, Mapping)
        or set(serialized_effects) != set(pair_names)
    ):
        raise ValueError("D104 simple effect closure drift")
    for name, expected in expected_effects.items():
        _verify_effect(
            serialized_effects[name],
            expected,
            name=name,
            integer_correct=True,
        )
    head_main = {
        metric: (
            float(expected_effects["H0_HEAD_at_base"][metric])
            + float(expected_effects["H1_HEAD_at_DA"][metric])
        )
        / 2.0
        for metric in METRICS
    }
    da_main = {
        metric: (
            float(expected_effects["D0_DA_at_legacy"][metric])
            + float(expected_effects["D1_DA_at_ANGQ"][metric])
        )
        / 2.0
        for metric in METRICS
    }
    interaction = {
        metric: (
            float(arms["M_JOINT"][metric])
            - float(arms["M_HEAD"][metric])
            - float(arms["M_DA"][metric])
            + float(arms["M0"][metric])
        )
        for metric in METRICS
    }
    _verify_effect(
        row["head_main_effect"],
        head_main,
        name="head_main_effect",
        integer_correct=False,
    )
    _verify_effect(
        row["da_main_effect"],
        da_main,
        name="da_main_effect",
        integer_correct=False,
    )
    _verify_effect(
        row["interaction"],
        interaction,
        name="interaction",
        integer_correct=False,
    )
    resources = row.get("resource_receipts")
    if not isinstance(resources, Mapping) or set(resources) != {
        "head_effect",
        "joint_effect",
    }:
        raise ValueError("D104 row resource pair drift")
    for name, receipt in resources.items():
        if not isinstance(receipt, Mapping):
            raise ValueError("D104 row resource receipt must be an object")
        _verify_resource_receipt(receipt, k_shot=int(row["K"]), name=name)
    int8 = row.get("int8_audit")
    if (
        not isinstance(int8, Mapping)
        or int8.get("schema") != INT8_SCHEMA
        or int8.get("passes_d104_int8_gate") is not True
        or _canonical_sha256(
            {
                key: value
                for key, value in int8.items()
                if key != "receipt_sha256"
            }
        )
        != int8.get("receipt_sha256")
    ):
        raise ValueError("D104 row INT8 audit drift")
    return arms, list(resources.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-json", type=Path, required=True)
    parser.add_argument("--tx-probe-json", type=Path, required=True)
    parser.add_argument("--matrix-status-json", type=Path, required=True)
    parser.add_argument("--runner-resource-json", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument("--run-input-binding-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_json.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable D104 gate exists: {output}")
    scores_path = args.scores_json.resolve(strict=True)
    tx_path = args.tx_probe_json.resolve(strict=True)
    matrix_path = args.matrix_status_json.resolve(strict=True)
    runner_path = args.runner_resource_json.resolve(strict=True)
    split_path = args.source_split_manifest.resolve(strict=True)
    binding_path = args.run_input_binding_json.resolve(strict=True)
    scores = _read(scores_path)
    tx = _read(tx_path)
    matrix = _read(matrix_path)
    runner = _read(runner_path)
    split = _read(split_path)
    binding = _read(binding_path)
    _require_sha256(
        matrix.get("fit_access_receipt_sha256_root"),
        "matrix fit access receipt root",
    )
    _require_sha256(matrix.get("matrix_plan_sha256"), "matrix plan")
    rows = scores.get("performance_rows")
    if (
        scores.get("schema") != "cvs.d104_r1.rxid_angq.held_scores.v1"
        or not isinstance(rows, list)
        or len(rows) != 63
        or scores.get("score_set_receipt_sha256")
        != _canonical_sha256(
            {
                key: value
                for key, value in scores.items()
                if key != "score_set_receipt_sha256"
            }
        )
        or matrix.get("schema")
        != "cvs.d103_r2.rxid_crossreceiver.fit_matrix_status.v1"
        or matrix.get("status") != "ARTIFACTS_COMPLETE"
        or matrix.get("planned_fit_count") != 246
        or matrix.get("completed_fit_count") != 246
        or matrix.get("failed_fit_count") != 0
        or matrix.get("completed_meta_steps") != 98_400
        or matrix.get("fit_manifest_validation_pass") is not True
        or matrix.get("fit_access_receipt_count") != 246
        or matrix.get("result_fit_ids_unique") is not True
        or matrix.get("first_wave_complete") is not True
        or len(matrix.get("first_wave_planned_fit_ids", [])) != 16
        or len(matrix.get("first_wave_completed_results", [])) != 16
        or {
            row.get("fit_id")
            for row in matrix.get("first_wave_completed_results", [])
            if isinstance(row, Mapping)
        }
        != set(matrix.get("first_wave_planned_fit_ids", []))
        or any(
            not isinstance(row, Mapping) or row.get("returncode") != 0
            for row in matrix.get("first_wave_completed_results", [])
        )
        or split.get("schema") != "cvs.d104_r1.source_split.archive.v2"
        or split.get("split_id") != "d104_source_seed104713_v2"
        or split.get("partition", {}).get("counts")
        != {"L_s": 588, "U_s": 5292, "source_val": 2520}
        or split.get("partition", {}).get(
            "source_labels_used_for_stratified_split"
        )
        is not True
        or split.get("partition", {}).get(
            "query_truth_used_for_method_selection"
        )
        is not False
        or split.get("partition", {}).get(
            "query_truth_used_for_performance_selection"
        )
        is not False
        or split.get("partition", {}).get("source_val_performance_computed")
        is not False
    ):
        raise ValueError("D104 gate input closure drift")
    binding_body = dict(binding)
    binding_receipt = binding_body.pop("receipt_sha256", None)
    evidence = scores.get("evidence_binding")
    if (
        binding.get("schema")
        != "cvs.d104_r1.rxid_angq.run_input_binding.v1"
        or binding_receipt != _canonical_sha256(binding_body)
        or binding.get("source_split_manifest_sha256")
        != _sha256_file(split_path)
        or binding.get("split_id") != split.get("split_id")
        or binding.get("matrix_fit_input_sha256")
        != matrix.get("fit_input_sha256")
        or binding.get("matrix_fit_input_manifest_sha256")
        != matrix.get("fit_input_manifest_sha256")
        or binding.get("checkpoint_sha256")
        != split.get("inputs", {}).get("checkpoint_sha256")
        or binding.get("runtime_sha256")
        != split.get("inputs", {}).get("runtime_sha256")
        or not isinstance(evidence, Mapping)
        or evidence.get("method_lock_sha256")
        != binding.get("method_lock_sha256")
        or evidence.get("source_val_scorer_manifest_sha256")
        != binding.get("source_val_scorer_manifest_sha256")
        or evidence.get("source_val_scorer_archive_sha256")
        != binding.get("source_val_scorer_archive_sha256")
        or split.get("source_val", {}).get("scorer_manifest_sha256")
        != binding.get("source_val_scorer_manifest_sha256")
        or split.get("source_val", {}).get("scorer_archive", {}).get("sha256")
        != binding.get("source_val_scorer_archive_sha256")
        or binding.get("historical_exclusion_manifest")
        != split.get("historical_exclusion_manifest")
        or split.get("historical_exclusion_manifest")
        != {
            "sha256": (
                "3fd07b7afcb53b12a08df1643efae80c"
                "52917c893cc7453104e68932dc1f5b26"
            ),
            "content_root_sha256": (
                "89c91bc8bc11d74e6b12bd2df2c2eeac"
                "53ca75d8f2d3a983a8e823da52765b27"
            ),
            "query_count": 2478,
        }
        or binding.get("target_access") is not False
        or binding.get("formal_query_access") is not False
    ):
        raise ValueError("D104 split/matrix/scorer evidence binding drift")
    role_input = {
        "labeled_archive": split["roles"]["L_s"]["archive_sha256"],
        "unlabeled_archive": split["roles"]["U_s"]["archive_sha256"],
        "source_val_seal": split["source_val"]["seal_sha256"],
    }
    role_manifest_input = {
        "labeled_manifest": split["roles"]["L_s"]["manifest_sha256"],
        "unlabeled_manifest": split["roles"]["U_s"]["manifest_sha256"],
        "source_val_manifest": split["source_val"]["fit_manifest_sha256"],
    }
    if (
        role_input != binding["matrix_fit_input_sha256"]
        or role_manifest_input != binding["matrix_fit_input_manifest_sha256"]
    ):
        raise ValueError("D104 root role SHA differs from matrix binding")
    runner_body = dict(runner)
    runner_receipt = runner_body.pop("receipt_sha256", None)
    if (
        runner.get("schema")
        != "cvs.d103_r2.rxid_crossreceiver.runner_resources.v1"
        or runner.get("status") != "RUNNER_RESOURCES_COMPLETE"
        or runner_receipt != _canonical_sha256(runner_body)
        or runner.get("matrix_status_sha256") != _sha256_file(matrix_path)
        or runner.get("completed_fit_count") != 246
        or runner.get("completed_meta_steps") != 98_400
        or runner.get("limits")
        != {
            "total_gpu_hours": GPU_HOUR_LIMIT,
            "peak_memory_bytes": PEAK_MEMORY_LIMIT,
            "run_root_bytes": DISK_LIMIT,
        }
        or runner.get("passes_runner_resource_gate") is not True
        or float(runner.get("total_gpu_hours", math.inf)) > GPU_HOUR_LIMIT
        or int(runner.get("peak_memory_bytes", PEAK_MEMORY_LIMIT + 1))
        > PEAK_MEMORY_LIMIT
        or int(runner.get("run_root_bytes", DISK_LIMIT + 1)) > DISK_LIMIT
    ):
        raise ValueError("D104 runner resource receipt drift")
    tx_body = dict(tx)
    tx_receipt = tx_body.pop("receipt_sha256", None)
    if (
        tx_receipt != _canonical_sha256(tx_body)
        or tx.get("fold_count") != 7
        or tx.get("target_access") is not False
    ):
        raise ValueError("D104 TX probe receipt drift")

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

    verified_rows = []
    all_resources: list[Mapping[str, Any]] = []
    for row in rows:
        arms, resources = _verify_row(row)
        verified_rows.append((row, arms))
        all_resources.extend(resources)
    query_deltas = [int(row["query_mac_delta"]) for row in all_resources]
    resource_component = scores.get("resource_component")
    if (
        not isinstance(resource_component, Mapping)
        or resource_component.get("query_mac_delta_max") != max(query_deltas)
        or resource_component.get("query_mac_delta_min") != min(query_deltas)
        or resource_component.get("query_mac_delta_sum") != sum(query_deltas)
        or resource_component.get("phase2_state_bytes_max")
        != max(
            int(row["actual_serialized_state_bytes_after"])
            for row in all_resources
        )
        or resource_component.get("adaptation_mac_total_max")
        != max(int(row["adaptation_mac_total"]) for row in all_resources)
        or resource_component.get(
            "adaptation_vector_elementwise_ops_total_max"
        )
        != max(
            int(row["adaptation_vector_elementwise_ops_total"])
            for row in all_resources
        )
        or resource_component.get("peak_temporary_bytes_max")
        != max(
            int(row["peak_temporary_bytes_upper_bound"])
            for row in all_resources
        )
    ):
        raise ValueError("D104 aggregate resource receipt drift")
    head_agreement = [
        float(row["int8_audit"]["M_HEAD"]["top1_agreement"])
        for row in rows
    ]
    joint_agreement = [
        float(row["int8_audit"]["M_JOINT"]["top1_agreement"])
        for row in rows
    ]
    head_flips = [
        int(
            row["int8_audit"]["M_HEAD"][
                "teacher_winner_margin_flip_count"
            ]
        )
        for row in rows
    ]
    joint_flips = [
        int(
            row["int8_audit"]["M_JOINT"][
                "teacher_winner_margin_flip_count"
            ]
        )
        for row in rows
    ]
    quant = scores["quantization_receipt"]
    if (
        quant.get("M_HEAD_min_top1_agreement") != min(head_agreement)
        or quant.get("M_JOINT_min_top1_agreement") != min(joint_agreement)
        or quant.get("M_HEAD_margin_flip_count") != sum(head_flips)
        or quant.get("M_JOINT_margin_flip_count") != sum(joint_flips)
    ):
        raise ValueError("D104 aggregate INT8 receipt drift")

    gates: dict[str, bool] = {
        "complete_63_row_matrix": True,
        "complete_252_arm_row_predictions": True,
        "matrix_246_fit_complete": True,
        "source_split_v2_closed_and_bound": True,
        "integer_metric_recomputation_pass": True,
        "all_factorial_effects_recomputed": True,
        "all_row_resource_receipts_verified": True,
        "runner_resource_receipt_verified": True,
        "tx_probe_le_25pct": float(tx["max_fold_score"]) <= 0.25,
        "M_HEAD_int8_each_row": min(head_agreement) >= 0.995
        and sum(head_flips) == 0,
        "M_JOINT_int8_each_row": min(joint_agreement) >= 0.995
        and sum(joint_flips) == 0,
        "head_simple_effect_row_nonnegative": all(
            float(arms["M_HEAD"]["balanced_accuracy"])
            - float(arms["M0"]["balanced_accuracy"])
            >= -ABS_TOL
            and float(arms["M_HEAD"]["per_class_floor"])
            - float(arms["M0"]["per_class_floor"])
            >= -ABS_TOL
            and int(arms["M_HEAD"]["correct_count"])
            - int(arms["M0"]["correct_count"])
            >= 0
            for _row, arms in verified_rows
        ),
        "joint_simple_effect_row_nonnegative": all(
            float(arms["M_JOINT"]["balanced_accuracy"])
            - float(arms["M_HEAD"]["balanced_accuracy"])
            >= -ABS_TOL
            and float(arms["M_JOINT"]["per_class_floor"])
            - float(arms["M_HEAD"]["per_class_floor"])
            >= -ABS_TOL
            and int(arms["M_JOINT"]["correct_count"])
            - int(arms["M_HEAD"]["correct_count"])
            >= 0
            for _row, arms in verified_rows
        ),
        "resource_query_mac_delta_zero": all(
            value == 0 for value in query_deltas
        ),
    }
    mean_joint = {
        arm: math.fsum(
            float(arms[arm]["joint_score"]) for _row, arms in verified_rows
        )
        / 63.0
        for arm in ARMS
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
        "schema": "cvs.d104_r1.rxid_angq.held_gate.v2",
        "status": (
            "TARGET25_GATE_ELIGIBLE"
            if passed
            else "D104_REJECTED_SOURCE_HELD_GATE"
        ),
        "gates": gates,
        "mean_joint_score": mean_joint,
        "input_sha256": {
            "scores": _sha256_file(scores_path),
            "tx_probe": _sha256_file(tx_path),
            "matrix_status": _sha256_file(matrix_path),
            "runner_resource": _sha256_file(runner_path),
            "source_split_manifest": _sha256_file(split_path),
            "run_input_binding": _sha256_file(binding_path),
        },
        "target25_gate_eligible": passed,
        "target25_authorized": False,
        "performance_selection_used": False,
        "row_subset_selected": False,
        "target_access": False,
    }
    result["gate_receipt_sha256"] = _canonical_sha256(result)
    _write_new(output, result)
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
