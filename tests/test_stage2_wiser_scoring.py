from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.stage2_wiser_scoring import score_wiser_predictions


def test_truth_last_scorer_aligns_tokens_and_reports_three_probes(tmp_path: Path) -> None:
    tokens = np.asarray([f"q{index}" for index in range(12)])
    truth_values = np.repeat(np.arange(6), 2)
    exact = truth_values.copy()
    p2 = exact.copy()
    p2[0] = 1
    features = np.eye(6, dtype=np.float32)[truth_values]
    features[1::2] += 0.01
    predictions = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions,
        query_tokens=tokens,
        p1_predictions=exact,
        p2_predictions=p2,
        p3_predictions=exact,
        query_z_id=features,
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "PREDICTIONS_COMPLETE",
                "arm": "A",
                "receiver": "rx",
                "scenario": "leo_clear_weak",
                "query_rows": 12,
                "expected_query_tokens": tokens.tolist(),
                "query_truth_opened": False,
                "query_role_opened": False,
                "support_state_frozen_before_query": True,
            }
        ),
        encoding="utf-8",
    )
    truth = tmp_path / "truth.json"
    truth.write_text(
        json.dumps(
            {
                "receiver": "rx",
                "rows": [
                    {"query_token": token, "true_class_index": int(class_id)}
                    for token, class_id in zip(tokens.tolist(), truth_values.tolist())
                ],
            }
        ),
        encoding="utf-8",
    )

    result = score_wiser_predictions(predictions, receipt, truth)

    assert result["status"] == "ANALYZED"
    assert result["truth_join_after_prediction_only"] is True
    assert result["probes"]["P1_SOURCE_HEAD"]["balanced_accuracy"] == 1.0
    assert result["probes"]["P2_SOURCE_PROTOTYPE"]["balanced_accuracy"] == pytest.approx(11 / 12)
    assert result["probes"]["P3_OLD_D92"]["floor"] == 1.0
    assert result["geometry"]["within_trace"] > 0.0
    assert result["geometry"]["between_within_ratio"] > 0.0


def test_truth_last_scorer_rejects_truncated_prediction_registry(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions,
        query_tokens=np.asarray(["q0"]),
        p1_predictions=np.asarray([0]),
        p2_predictions=np.asarray([0]),
        p3_predictions=np.asarray([0]),
        query_z_id=np.zeros((1, 160), dtype=np.float32),
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "PREDICTIONS_COMPLETE",
                "arm": "A",
                "receiver": "rx",
                "scenario": "leo_clear_weak",
                "query_rows": 2,
                "expected_query_tokens": ["q0", "q1"],
                "query_truth_opened": False,
                "query_role_opened": False,
                "support_state_frozen_before_query": True,
            }
        ),
        encoding="utf-8",
    )
    truth = tmp_path / "truth.json"
    truth.write_text(
        json.dumps(
            {
                "receiver": "rx",
                "rows": [
                    {"query_token": "q0", "true_class_index": 0},
                    {"query_token": "q1", "true_class_index": 1},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="token join"):
        score_wiser_predictions(predictions, receipt, truth)
