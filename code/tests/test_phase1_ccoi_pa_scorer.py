import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from score_phase1_ccoi_pa import score_streams  # noqa: E402


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_independent_scorer_joins_truth_after_prediction(tmp_path):
    prediction = tmp_path / "prediction.jsonl"
    truth = tmp_path / "truth.jsonl"
    pred_rows = []
    truth_rows = []
    for scenario in ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"):
        for index, pair in enumerate(((0, 0), (1, 0))):
            sample_id = f"{scenario}:{index}"
            pred_rows.append(
                {
                    "sample_id": sample_id,
                    "scenario": scenario,
                    "loader": "test_unseen_day_seen_rx",
                    "receiver": "rx0",
                    "predicted_class": pair[0],
                }
            )
            truth_rows.append({"sample_id": sample_id, "true_class": pair[1]})
    _write(prediction, pred_rows)
    _write(truth, truth_rows)

    metrics = score_streams(prediction, truth)

    assert metrics["status"] == "ANALYZED"
    assert metrics["scenario"]["clean"]["aggregate"]["accuracy"] == 50.0
    assert metrics["truth_joined_after_prediction"] is True


def test_scorer_rejects_truth_leakage_in_prediction_stream(tmp_path):
    prediction = tmp_path / "prediction.jsonl"
    truth = tmp_path / "truth.jsonl"
    _write(prediction, [{"sample_id": "x", "predicted_class": 0, "true_class": 0}])
    _write(truth, [{"sample_id": "x", "true_class": 0}])

    with pytest.raises(ValueError, match="truth-blind"):
        score_streams(prediction, truth)


def test_scorer_rejects_missing_receiver_identity_for_main_loader(tmp_path):
    prediction = tmp_path / "prediction.jsonl"
    truth = tmp_path / "truth.jsonl"
    pred_rows = []
    truth_rows = []
    for scenario in ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"):
        sample_id = f"{scenario}:0"
        pred_rows.append(
            {
                "sample_id": sample_id,
                "scenario": scenario,
                "loader": "test_unseen_day_seen_rx",
                "receiver": -1,
                "predicted_class": 0,
            }
        )
        truth_rows.append({"sample_id": sample_id, "true_class": 0})
    _write(prediction, pred_rows)
    _write(truth, truth_rows)

    with pytest.raises(ValueError, match="receiver identity"):
        score_streams(prediction, truth)
