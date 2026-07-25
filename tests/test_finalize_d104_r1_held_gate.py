from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "code" / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))
try:
    import finalize_d104_r1_held_gate as gate
finally:
    sys.path.remove(str(SCRIPT_ROOT))


def _resource(k: int) -> dict:
    support_count = 6 * k
    value = {
        "schema": gate.RESOURCE_SCHEMA,
        "candidate_id": "D104-R1-ANGQ-RXID-MB4",
        "registered_class_count": 6,
        "active_k": k,
        "support_count": support_count,
        "factor_count": 101,
        "adaptation_mac_per_support": 32_320,
        "adaptation_mac_total": 32_320 * support_count,
        "adaptation_mac_formula": "32320*registered_class_count*active_k",
        "adaptation_vector_elementwise_ops_per_support": 64_640,
        "adaptation_vector_elementwise_ops_total": 64_640 * support_count,
        "adaptation_vector_elementwise_ops_formula": (
            "64640*registered_class_count*active_k"
        ),
        "scalar_reduction_ops": {},
        "shared_input_normalization_excluded_from_angq_delta": True,
        "peak_temporary_bytes_upper_bound": 4096,
        "peak_temporary_bytes_gate": 16_384,
        "passes_peak_temporary_bytes_gate": True,
        "numeric_bank_array_bytes_before": 1000,
        "numeric_bank_array_bytes_after": 1000,
        "numeric_bank_array_bytes_delta": 0,
        "actual_serialized_state_bytes_before": 2000,
        "actual_serialized_state_bytes_after": 2100,
        "actual_serialized_state_bytes_delta": 100,
        "metadata_framing_bytes_before": 1000,
        "metadata_framing_bytes_after": 1100,
        "metadata_framing_bytes_delta": 100,
        "wire_bytes_gate": 16 * 1024 * 1024,
        "passes_wire_bytes_gate": True,
        "query_mac_before": 100,
        "query_mac_after": 100,
        "query_mac_delta": 0,
        "query_features_used_for_scale": 0,
        "query_truth_read": False,
        "query_state_updates": 0,
        "passes_d104_resource_gate": True,
    }
    value["receipt_sha256"] = gate._canonical_sha256(value)
    return value


def _int8() -> dict:
    value = {
        "schema": gate.INT8_SCHEMA,
        "M_HEAD": {
            "top1_agreement": 1.0,
            "teacher_winner_margin_flip_count": 0,
        },
        "M_JOINT": {
            "top1_agreement": 1.0,
            "teacher_winner_margin_flip_count": 0,
        },
        "query_truth_read": False,
        "query_state_updates": 0,
        "target25_authorized": False,
        "passes_d104_int8_gate": True,
    }
    value["receipt_sha256"] = gate._canonical_sha256(value)
    return value


def _metric(correct_per_class: int) -> dict:
    rate = correct_per_class / 100
    return {
        "balanced_accuracy": rate,
        "per_class_floor": rate,
        "joint_score": rate,
        "correct_count": correct_per_class * 6,
        "query_count": 600,
        "per_class_correct": [correct_per_class] * 6,
        "per_class_count": [100] * 6,
    }


def _row(receiver, held_class, k, *, head_correct: int = 60):
    arm = {
        "M0": _metric(50),
        "M_DA": _metric(55),
        "M_HEAD": _metric(head_correct),
        "M_JOINT": _metric(70),
    }

    pairs = {
        "H0_HEAD_at_base": ("M_HEAD", "M0"),
        "H1_HEAD_at_DA": ("M_JOINT", "M_DA"),
        "D0_DA_at_legacy": ("M_DA", "M0"),
        "D1_DA_at_ANGQ": ("M_JOINT", "M_HEAD"),
    }
    effects = {
        name: {
            metric: arm[left][metric] - arm[right][metric]
            for metric in gate.METRICS
        }
        for name, (left, right) in pairs.items()
    }
    head_main = {
        metric: (
            effects["H0_HEAD_at_base"][metric]
            + effects["H1_HEAD_at_DA"][metric]
        )
        / 2
        for metric in gate.METRICS
    }
    da_main = {
        metric: (
            effects["D0_DA_at_legacy"][metric]
            + effects["D1_DA_at_ANGQ"][metric]
        )
        / 2
        for metric in gate.METRICS
    }
    interaction = {
        metric: (
            arm["M_JOINT"][metric]
            - arm["M_HEAD"][metric]
            - arm["M_DA"][metric]
            + arm["M0"][metric]
        )
        for metric in gate.METRICS
    }
    value = {
        "held_receiver": receiver,
        "held_class": held_class,
        "K": k,
        "registered_classes": [f"c{i}" for i in range(6)],
        "arm_metrics": arm,
        "simple_effects": effects,
        "head_main_effect": head_main,
        "da_main_effect": da_main,
        "interaction": interaction,
        "prediction_receipt_sha256": "a" * 64,
        "truth_row_count": 600,
        "target_access": False,
        "k1_evidence": (
            {
                "active": True,
                "information_rank": 4,
                "minimum_singular_value": 0.1,
                "condition_number": 2.0,
                "prior_fraction": 0.5,
                "coefficient_norm": 0.1,
                "view_top1_agreement": 1.0,
                "view_margin_flip_count": 0,
                "direction_cosine_median": 0.9,
            }
            if k == 1
            else None
        ),
        "prediction_artifact_committed_before_truth": True,
        "truth_open_event_sha256": "b" * 64,
        "scorer_input_seal_sha256": "c" * 64,
        "resource_receipts": {
            "head_effect": _resource(k),
            "joint_effect": _resource(k),
        },
        "int8_audit": _int8(),
    }
    value["score_receipt_sha256"] = gate._canonical_sha256(value)
    return value


def _inputs():
    receivers = [f"r{i}" for i in range(7)]
    classes = [f"c{i}" for i in range(6)]
    rows = [
        _row(receiver, None, k)
        for receiver in receivers
        for k in (1, 5, 10)
    ] + [
        _row(receiver, class_id, 1)
        for receiver in receivers
        for class_id in classes
    ]
    resources = [
        receipt
        for row in rows
        for receipt in row["resource_receipts"].values()
    ]
    evidence = {
        "package_manifest_sha256": "1" * 64,
        "truth_input_seal_sha256": "2" * 64,
        "truth_package_root_sha256": "3" * 64,
        "source_val_scorer_manifest_sha256": "4" * 64,
        "source_val_scorer_archive_sha256": "5" * 64,
        "method_lock_sha256": "6" * 64,
        "registered_class_root_sha256": "7" * 64,
        "scorer_input_root_sha256": "8" * 64,
    }
    scores = {
        "schema": "cvs.d104_r1.rxid_angq.held_scores.v1",
        "split_id": "d104_source_seed104713_v2",
        "performance_rows": rows,
        "day_stability_rows": [{} for _ in range(49)],
        "quantization_receipt": {
            "M_HEAD_min_top1_agreement": 1.0,
            "M_HEAD_margin_flip_count": 0,
            "M_JOINT_min_top1_agreement": 1.0,
            "M_JOINT_margin_flip_count": 0,
        },
        "resource_component": {
            "phase2_state_bytes_max": 2100,
            "adaptation_mac_total_max": max(
                row["adaptation_mac_total"] for row in resources
            ),
            "adaptation_vector_elementwise_ops_total_max": max(
                row["adaptation_vector_elementwise_ops_total"]
                for row in resources
            ),
            "peak_temporary_bytes_max": 4096,
            "query_mac_before_max": 100,
            "query_mac_after_max": 100,
            "query_mac_delta_max": 0,
            "query_mac_delta_min": 0,
            "query_mac_delta_sum": 0,
            "row_count": 63,
        },
        "evidence_binding": evidence,
        "prediction_manifest_sha256": "9" * 64,
        "truth_sha256": "a" * 64,
        "truth_open_event_sha256": "b" * 64,
        "prediction_artifact_committed_before_truth": True,
        "target_access": False,
    }
    scores["score_set_receipt_sha256"] = gate._canonical_sha256(scores)
    tx = {
        "fold_count": 7,
        "max_fold_score": 0.2,
        "target_access": False,
    }
    tx["receipt_sha256"] = gate._canonical_sha256(tx)
    matrix = {
        "schema": "cvs.d103_r2.rxid_crossreceiver.fit_matrix_status.v1",
        "status": "ARTIFACTS_COMPLETE",
        "planned_fit_count": 246,
        "completed_fit_count": 246,
        "failed_fit_count": 0,
        "completed_meta_steps": 98_400,
        "fit_manifest_validation_pass": True,
        "fit_access_receipt_count": 246,
        "fit_access_receipt_sha256_root": "d" * 64,
        "matrix_plan_sha256": "c" * 64,
        "result_fit_ids_unique": True,
        "first_wave_planned_fit_ids": [
            f"fit_{index:03d}" for index in range(16)
        ],
        "first_wave_completed_results": [
            {"fit_id": f"fit_{index:03d}", "returncode": 0}
            for index in range(16)
        ],
        "first_wave_complete": True,
        "fit_input_sha256": {
            "labeled_archive": "e" * 64,
            "unlabeled_archive": "f" * 64,
            "source_val_seal": "0" * 64,
        },
        "fit_input_manifest_sha256": {
            "labeled_manifest": "1" * 64,
            "unlabeled_manifest": "2" * 64,
            "source_val_manifest": "3" * 64,
        },
    }
    split = {
        "schema": "cvs.d104_r1.source_split.archive.v2",
        "split_id": "d104_source_seed104713_v2",
        "inputs": {
            "checkpoint_sha256": "a" * 64,
            "runtime_sha256": "b" * 64,
        },
        "partition": {
            "counts": {"L_s": 588, "U_s": 5292, "source_val": 2520},
            "source_labels_used_for_stratified_split": True,
            "query_truth_used_for_method_selection": False,
            "query_truth_used_for_performance_selection": False,
            "source_val_performance_computed": False,
        },
        "roles": {
            "L_s": {
                "archive_sha256": "e" * 64,
                "manifest_sha256": "1" * 64,
            },
            "U_s": {
                "archive_sha256": "f" * 64,
                "manifest_sha256": "2" * 64,
            },
        },
        "source_val": {
            "seal_sha256": "0" * 64,
            "fit_manifest_sha256": "3" * 64,
            "scorer_manifest_sha256": "4" * 64,
            "scorer_archive": {"sha256": "5" * 64},
        },
        "historical_exclusion_manifest": {
            "sha256": (
                "3fd07b7afcb53b12a08df1643efae80c"
                "52917c893cc7453104e68932dc1f5b26"
            ),
            "content_root_sha256": (
                "89c91bc8bc11d74e6b12bd2df2c2eeac"
                "53ca75d8f2d3a983a8e823da52765b27"
            ),
            "query_count": 2478,
        },
    }
    binding = {
        "schema": "cvs.d104_r1.rxid_angq.run_input_binding.v1",
        "split_id": "d104_source_seed104713_v2",
        "source_split_manifest_sha256": "",
        "historical_exclusion_manifest": split[
            "historical_exclusion_manifest"
        ],
        "checkpoint_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
        "method_lock_sha256": "6" * 64,
        "matrix_fit_input_sha256": matrix["fit_input_sha256"],
        "matrix_fit_input_manifest_sha256": matrix[
            "fit_input_manifest_sha256"
        ],
        "source_val_scorer_manifest_sha256": "4" * 64,
        "source_val_scorer_archive_sha256": "5" * 64,
        "target_access": False,
        "formal_query_access": False,
    }
    runner = {
        "schema": "cvs.d103_r2.rxid_crossreceiver.runner_resources.v1",
        "status": "RUNNER_RESOURCES_COMPLETE",
        "matrix_status_sha256": "",
        "total_gpu_hours": 1.0,
        "peak_memory_bytes": 1024,
        "run_root_bytes": 2048,
        "completed_fit_count": 246,
        "completed_meta_steps": 98_400,
        "limits": {
            "total_gpu_hours": 30.0,
            "peak_memory_bytes": 4 * 1024**3,
            "run_root_bytes": 20 * 1024**3,
        },
        "passes_runner_resource_gate": True,
    }
    return {
        "scores": scores,
        "tx": tx,
        "matrix": matrix,
        "runner": runner,
        "split": split,
        "binding": binding,
    }


def _refresh_receipts(inputs):
    for row in inputs["scores"]["performance_rows"]:
        row["score_receipt_sha256"] = gate._canonical_sha256(
            {
                key: value
                for key, value in row.items()
                if key != "score_receipt_sha256"
            }
        )
    inputs["scores"]["score_set_receipt_sha256"] = gate._canonical_sha256(
        {
            key: value
            for key, value in inputs["scores"].items()
            if key != "score_set_receipt_sha256"
        }
    )


def _run(monkeypatch, tmp_path, inputs):
    paths = {}
    for name in ("scores", "tx", "matrix", "split"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(inputs[name]), encoding="utf-8")
        paths[name] = path
    inputs["runner"]["matrix_status_sha256"] = gate._sha256_file(
        paths["matrix"]
    )
    inputs["runner"]["receipt_sha256"] = gate._canonical_sha256(
        inputs["runner"]
    )
    paths["runner"] = tmp_path / "runner.json"
    paths["runner"].write_text(
        json.dumps(inputs["runner"]),
        encoding="utf-8",
    )
    inputs["binding"]["source_split_manifest_sha256"] = gate._sha256_file(
        paths["split"]
    )
    inputs["binding"]["receipt_sha256"] = gate._canonical_sha256(
        inputs["binding"]
    )
    paths["binding"] = tmp_path / "binding.json"
    paths["binding"].write_text(
        json.dumps(inputs["binding"]),
        encoding="utf-8",
    )
    output = tmp_path / "gate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gate",
            "--scores-json",
            str(paths["scores"]),
            "--tx-probe-json",
            str(paths["tx"]),
            "--matrix-status-json",
            str(paths["matrix"]),
            "--runner-resource-json",
            str(paths["runner"]),
            "--source-split-manifest",
            str(paths["split"]),
            "--run-input-binding-json",
            str(paths["binding"]),
            "--output-json",
            str(output),
        ],
    )
    assert gate.main() == 0
    return json.loads(output.read_text(encoding="utf-8"))


def test_d104_gate_positive_complete_matrix(monkeypatch, tmp_path) -> None:
    result = _run(monkeypatch, tmp_path, _inputs())
    assert result["status"] == "TARGET25_GATE_ELIGIBLE"
    assert result["target25_gate_eligible"] is True
    assert result["target25_authorized"] is False


def test_d104_gate_rejects_one_real_bad_row_without_subsetting(
    monkeypatch,
    tmp_path,
) -> None:
    inputs = _inputs()
    inputs["scores"]["performance_rows"][0] = _row(
        "r0",
        None,
        1,
        head_correct=40,
    )
    _refresh_receipts(inputs)
    result = _run(monkeypatch, tmp_path, inputs)
    assert result["status"] == "D104_REJECTED_SOURCE_HELD_GATE"
    assert result["target25_gate_eligible"] is False
    assert result["row_subset_selected"] is False


@pytest.mark.parametrize(
    "tamper",
    ("balanced_accuracy", "class_integer", "resource_query_mac"),
)
def test_d104_gate_fails_closed_on_serialized_or_resource_tamper(
    monkeypatch,
    tmp_path,
    tamper,
) -> None:
    inputs = _inputs()
    row = inputs["scores"]["performance_rows"][0]
    if tamper == "balanced_accuracy":
        row["arm_metrics"]["M0"]["balanced_accuracy"] += 0.01
    elif tamper == "class_integer":
        row["arm_metrics"]["M0"]["per_class_correct"][0] += 1
    else:
        receipt = row["resource_receipts"]["head_effect"]
        receipt["query_mac_delta"] = 1
        receipt["receipt_sha256"] = gate._canonical_sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "receipt_sha256"
            }
        )
    _refresh_receipts(inputs)
    with pytest.raises(ValueError):
        _run(monkeypatch, tmp_path, inputs)


@pytest.mark.parametrize(
    "binding_tamper",
    ("matrix_input", "root_role", "runner_limit"),
)
def test_d104_gate_fails_closed_on_root_matrix_or_runner_binding_drift(
    monkeypatch,
    tmp_path,
    binding_tamper,
) -> None:
    inputs = _inputs()
    if binding_tamper == "matrix_input":
        inputs["matrix"]["fit_input_sha256"]["labeled_archive"] = "9" * 64
    elif binding_tamper == "root_role":
        inputs["split"]["roles"]["L_s"]["archive_sha256"] = "9" * 64
    else:
        inputs["runner"]["total_gpu_hours"] = 31.0
    with pytest.raises(ValueError):
        _run(monkeypatch, tmp_path, inputs)
