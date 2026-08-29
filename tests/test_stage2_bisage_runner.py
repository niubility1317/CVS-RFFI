from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.stage2_bisage_runner import (
    BiSAGERunnerError,
    FOUR_STATES,
    SCENARIOS,
    build_prediction_receipt,
    joint_stage_a_gate,
    score_truth_last,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    query_tokens = np.asarray([f"qid_{index}" for index in range(8)])
    truth = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    handles = {index: f"cls_{index}" for index in range(8)}
    payload: dict[str, np.ndarray] = {"query_tokens": query_tokens}
    for state in FOUR_STATES:
        prefix = state.lower()
        class_ids = np.arange(6 if state.endswith("REG0") else 8)
        prediction = np.minimum(truth, len(class_ids) - 1)
        logits = np.zeros((len(truth), len(class_ids)), dtype=np.float32)
        logits[np.arange(len(truth)), prediction] = 1.0
        payload[f"{prefix}_class_ids"] = class_ids
        payload[f"{prefix}_class_handles"] = np.asarray([handles[index] for index in class_ids])
        payload[f"{prefix}_logits"] = logits
        payload[f"{prefix}_predictions"] = prediction
        payload[f"{prefix}_prediction_handles"] = np.asarray([handles[int(index)] for index in prediction])
    predictions = tmp_path / "predictions.npz"
    np.savez(predictions, **payload)
    receipt = tmp_path / "receipt.json"
    _write_json(receipt, {
        "status": "PREDICTIONS_COMPLETE",
        "query_truth_opened": False,
        "outer_key": "outer",
        "receiver": "3-19",
        "scenario": "leo_clear_weak",
        "capsule_id": "cap",
        "split_id": "split",
        "new_class_count": 2,
    })
    truth_path = tmp_path / "truth.json"
    _write_json(truth_path, {"receiver": "3-19", "rows": [
        {
            "query_token": str(token),
            "true_class_index": int(label),
            "true_class_handle": handles[int(label)],
        }
        for token, label in zip(query_tokens, truth)
    ]})
    return predictions, receipt, truth_path, tmp_path / "score.json"


def test_truth_last_scorer_reports_four_states_and_na_for_reg0(tmp_path: Path) -> None:
    predictions, receipt, truth, output = _fixture(tmp_path)
    result = score_truth_last(predictions, receipt, truth, output)
    assert result["status"] == "ANALYZED"
    assert tuple(result["states"]) == FOUR_STATES
    assert result["states"]["DA0_REG0"]["new_accuracy"] == "N/A"
    assert result["states"]["DA1_REG1"]["new_accuracy"] == 1.0
    assert result["truth_join_after_prediction_only"] is True
    assert output.is_file()


def test_truth_last_scorer_rejects_token_or_handle_drift(tmp_path: Path) -> None:
    predictions, receipt, truth, output = _fixture(tmp_path)
    payload = json.loads(truth.read_text(encoding="utf-8"))
    payload["rows"][0]["query_token"] = "wrong"
    _write_json(truth, payload)
    with pytest.raises(BiSAGERunnerError, match="token join"):
        score_truth_last(predictions, receipt, truth, output)


def test_truth_last_scorer_accepts_three_scene_sidecar_subset(tmp_path: Path) -> None:
    predictions, receipt, truth, output = _fixture(tmp_path)
    payload = json.loads(truth.read_text(encoding="utf-8"))
    payload["rows"] = payload["rows"] * 3
    _write_json(truth, payload)
    assert score_truth_last(predictions, receipt, truth, output)["status"] == "ANALYZED"


def test_joint_stage_a_gate_requires_all_three_scenarios() -> None:
    receipts = {
        scenario: {"scenario": scenario, "gate": {"stage_a_gate_passed": True}}
        for scenario in SCENARIOS
    }
    assert joint_stage_a_gate(receipts) is True
    receipts["leo_rain_weak"]["gate"]["stage_a_gate_passed"] = False
    assert joint_stage_a_gate(receipts) is False
    with pytest.raises(BiSAGERunnerError, match="all three"):
        joint_stage_a_gate({"leo_clear_weak": receipts["leo_clear_weak"]})


def test_real_prediction_receipt_builder_binds_receiver() -> None:
    receipt = build_prediction_receipt(
        {
            "outer_key": "rx_3_19__seed_713102__k_10__new_5",
            "receiver": "3-19",
            "capsule_id": "cap",
            "split_id": "split",
            "k_shot": 10,
            "new_class_count": 5,
        },
        "leo_clear_weak",
        query_rows=220,
        selected_mode="S2",
        stage_b_ran=True,
        mode_metrics={"selected_mode": "S2"},
    )
    assert receipt["receiver"] == "3-19"
    assert receipt["query_truth_opened"] is False
