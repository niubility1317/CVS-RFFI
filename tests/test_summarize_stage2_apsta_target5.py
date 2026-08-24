from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_stage2_apsta_target5 import summarize


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_summary_uses_row_mean_and_global_floor_and_applies_joint_gate(tmp_path: Path) -> None:
    scores = tmp_path / "scores"
    predictions = tmp_path / "predictions"
    for index, (da0, da1, floor0, floor1, step) in enumerate(
        [(0.70, 0.72, 0.20, 0.21, 30), (0.80, 0.81, 0.30, 0.31, 0)]
    ):
        _write(scores / f"row{index}_pair.json", {
            "status": "ANALYZED", "scenario": "leo_clear_weak",
            "protocol_schema": "p2_min_v1", "phase2_data_status": "VALIDATED_ONCE",
            "DA0_REG0": {"old_class_mean_accuracy": da0, "old_class_floor_accuracy": floor0},
            "DA1_REG0": {"old_class_mean_accuracy": da1, "old_class_floor_accuracy": floor1},
            "DA1_REG0_minus_DA0_REG0": {},
            "scorer_output_must_not_feed_predictor": True,
        })
        _write(predictions / f"row{index}_da1_reg0.json", {
            "status": "PREDICTIONS_COMPLETE", "state": "DA1_REG0",
            "query_truth_loaded": False, "query_role_loaded": False,
            "query_batch_state_updated": False,
            "audit": {"selected_step": step, "fallback_to_teacher": step == 0, "backward_count": 300},
        })
    result = summarize(tmp_path, expected_rows=2)
    assert result["DA1_REG0_minus_DA0_REG0"]["old_class_mean_delta_pp"] == 1.5
    assert result["DA1_REG0_minus_DA0_REG0"]["old_class_floor_delta_pp"] == 1.0
    assert result["verdict"] == "PROMOTE_TARGET25"
    assert result["support_selected_step_histogram"] == {"0": 1, "30": 1}

