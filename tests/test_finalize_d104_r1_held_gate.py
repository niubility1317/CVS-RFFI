from __future__ import annotations

import copy
import json
from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "code" / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))
try:
    import finalize_d104_r1_held_gate as gate
finally:
    sys.path.remove(str(SCRIPT_ROOT))


METRICS = {
    "balanced_accuracy",
    "per_class_floor",
    "joint_score",
    "correct_count",
}


def _row(receiver, held_class, k):
    arm = {
        name: {
            "balanced_accuracy": value,
            "per_class_floor": value,
            "joint_score": value,
            "correct_count": int(value * 100),
            "query_count": 100,
            "per_class_correct": [10] * 6,
            "per_class_count": [12] * 6,
        }
        for name, value in (
            ("M0", 0.50),
            ("M_DA", 0.55),
            ("M_HEAD", 0.60),
            ("M_JOINT", 0.70),
        )
    }
    effects = {
        "H0_HEAD_at_base": {
            "balanced_accuracy": 0.10,
            "per_class_floor": 0.10,
            "joint_score": 0.10,
            "correct_count": 10,
        },
        "H1_HEAD_at_DA": {
            "balanced_accuracy": 0.15,
            "per_class_floor": 0.15,
            "joint_score": 0.15,
            "correct_count": 15,
        },
        "D0_DA_at_legacy": {
            "balanced_accuracy": 0.05,
            "per_class_floor": 0.05,
            "joint_score": 0.05,
            "correct_count": 5,
        },
        "D1_DA_at_ANGQ": {
            "balanced_accuracy": 0.10,
            "per_class_floor": 0.10,
            "joint_score": 0.10,
            "correct_count": 10,
        },
    }
    return {
        "held_receiver": receiver,
        "held_class": held_class,
        "K": k,
        "arm_metrics": arm,
        "simple_effects": effects,
        "head_main_effect": {name: 0.1 for name in METRICS},
        "da_main_effect": {name: 0.1 for name in METRICS},
        "interaction": {name: 0.0 for name in METRICS},
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
    }


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
    return {
        "scores": {
            "schema": "cvs.d104_r1.rxid_angq.held_scores.v1",
            "performance_rows": rows,
            "day_stability_rows": [{} for _ in range(49)],
            "quantization_receipt": {
                "M_HEAD_min_top1_agreement": 1.0,
                "M_HEAD_margin_flip_count": 0,
                "M_JOINT_min_top1_agreement": 1.0,
                "M_JOINT_margin_flip_count": 0,
            },
            "resource_component": {"query_mac_delta": 0},
        },
        "tx": {"max_fold_score": 0.2},
        "matrix": {
            "status": "ARTIFACTS_COMPLETE",
            "planned_fit_count": 246,
            "completed_fit_count": 246,
            "failed_fit_count": 0,
        },
        "runner": {"resource": True},
        "split": {
            "split_id": "d104_source_seed104713_v2",
            "partition": {
                "counts": {"L_s": 588, "U_s": 5292, "source_val": 2520}
            },
        },
    }


def _run(monkeypatch, tmp_path, inputs):
    paths = {}
    for name, value in inputs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
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


def test_d104_gate_rejects_one_bad_row_without_subsetting(monkeypatch, tmp_path) -> None:
    inputs = _inputs()
    inputs["scores"]["performance_rows"][0]["simple_effects"][
        "H0_HEAD_at_base"
    ]["balanced_accuracy"] = -0.01
    result = _run(monkeypatch, tmp_path, inputs)
    assert result["status"] == "D104_REJECTED_SOURCE_HELD_GATE"
    assert result["target25_gate_eligible"] is False
    assert result["row_subset_selected"] is False
