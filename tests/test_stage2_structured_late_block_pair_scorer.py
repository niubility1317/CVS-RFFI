from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvsrffi.stage2_structured_late_block_pair_scorer import score_prediction_pair


def _prediction(path: Path, state: str, values: list[str]) -> None:
    rows = [
        {"sample_index": index, "query_token": f"q{index}", "predicted_class_id": value, "scores": [1.0, 0.0]}
        for index, value in enumerate(values)
    ]
    path.write_text(json.dumps({
        "status": "PREDICTIONS_COMPLETE", "state": state,
        "source_input_count": 0, "query_input_count": len(rows),
        "query_truth_loaded": False, "query_role_loaded": False,
        "query_batch_state_updated": False,
        "audit": {"protocol_schema": "p2_min_v1", "phase2_data_status": "VALIDATED_ONCE", "capsule_id": "c", "split_id": "s"},
        "predictions": rows,
    }), encoding="utf-8")


def test_scores_exact_opaque_pair_and_ignores_reg0_new_class(tmp_path: Path) -> None:
    da0, da1, truth, output = [tmp_path / name for name in ("da0.json", "da1.json", "truth.json", "score.json")]
    _prediction(da0, "DA0_REG0", ["a", "b", "a", "b", "a"])
    _prediction(da1, "DA1_REG0", ["a", "a", "b", "b", "b"])
    truth.write_text(json.dumps({"rows": [
        {"query_token": "q0", "scenario": "leo_clear_weak", "evaluation_role": "target_old", "true_class_handle": "a"},
        {"query_token": "q1", "scenario": "leo_clear_weak", "evaluation_role": "target_old", "true_class_handle": "a"},
        {"query_token": "q2", "scenario": "leo_clear_weak", "evaluation_role": "target_old", "true_class_handle": "b"},
        {"query_token": "q3", "scenario": "leo_clear_weak", "evaluation_role": "target_old", "true_class_handle": "b"},
        {"query_token": "q4", "scenario": "leo_clear_weak", "evaluation_role": "target_new", "true_class_handle": None},
    ]}), encoding="utf-8")
    result = score_prediction_pair(da0, da1, truth, scenario="leo_clear_weak", output_path=output)
    assert result["DA0_REG0"]["old_class_mean_accuracy"] == pytest.approx(0.5)
    assert result["DA1_REG0"]["old_class_mean_accuracy"] == pytest.approx(1.0)
    assert result["DA1_REG0_minus_DA0_REG0"]["old_class_mean_delta_pp"] == pytest.approx(50.0)
    assert result["REG0_new_class_metrics"] == "N/A"
