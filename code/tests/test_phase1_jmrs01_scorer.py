from __future__ import annotations

import json
from pathlib import Path

import pytest

from score_phase1_jmrs01 import REQUIRED_ARTIFACTS, _receiver_probes, score_prediction_streams


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_scorer_rejects_incomplete_scenario_matrix(tmp_path: Path) -> None:
    prediction = tmp_path / "predictions.jsonl"
    truth = tmp_path / "truth.jsonl"
    _write_jsonl(
        prediction,
        [{"sample_id": "M0:clean:0", "row": "M0", "scenario": "clean", "predicted_class": 0}],
    )
    _write_jsonl(truth, [{"sample_id": "M0:clean:0", "true_class": 0}])

    with pytest.raises(ValueError, match="scenario matrix is incomplete"):
        score_prediction_streams(prediction, truth, tmp_path / "score")


def test_scorer_writes_declared_artifact_closure_from_literal_fixture(tmp_path: Path) -> None:
    predictions: list[dict] = []
    truths: list[dict] = []
    scenarios = ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    for row in ("M0", "S1", "D1"):
        for receiver in (0, 1):
            for index, scenario in enumerate(scenarios):
                sample_id = f"{row}:{receiver}:{scenario}"
                true_class = receiver
                predicted = true_class if row != "S1" else 1 - true_class
                predictions.append(
                    {
                        "sample_id": sample_id,
                        "row": row,
                        "scenario": scenario,
                        "scope": "held_audit",
                        "held_receiver": receiver,
                        "receiver": receiver,
                        "day": index % 2,
                        "base_index": receiver,
                        "predicted_class": predicted,
                        "base_predicted_class": true_class,
                        "embedding": [float(receiver), float(index), 0.0],
                        "reliability": 0.8,
                        "parameter_count": 1024,
                        "runtime_ms_per_sample": 0.2,
                    }
                )
                truths.append({"sample_id": sample_id, "true_class": true_class})
    prediction_path = tmp_path / "predictions.jsonl"
    truth_path = tmp_path / "truth.jsonl"
    output = tmp_path / "score"
    _write_jsonl(prediction_path, predictions)
    _write_jsonl(truth_path, truths)

    result = score_prediction_streams(prediction_path, truth_path, output, bootstrap_resamples=50)

    assert result["status"] == "ANALYZED"
    assert result["loro"]["D1"]["clean"]["macro_accuracy"] == 1.0
    assert result["loro"]["S1"]["clean"]["macro_accuracy"] == 0.0
    assert all((output / name).is_file() for name in REQUIRED_ARTIFACTS)
    identity = json.loads((output / "mechanism_identity_stability.json").read_text(encoding="utf-8"))
    assert set(identity["D1"]["folds"]) == {"0", "1"}
    assert identity["D1"]["aggregation"] == "mean_of_fold_local_geometry"
    decision = json.loads((output / "mechanism_decision.json").read_text(encoding="utf-8"))
    assert decision["D1"]["next_stage"] == "DO_NOT_PROMOTE"
    assert decision["D1"]["all_gates_pass"] is False


def test_receiver_probes_are_fit_inside_each_loro_coordinate_system() -> None:
    rows = []
    for held_receiver, offset in ((6, 0.0), (5, 100.0)):
        for receiver in (0, 1):
            for sample in range(4):
                rows.append(
                    {
                        "scope": "probe_fit",
                        "scenario": "clean",
                        "held_receiver": held_receiver,
                        "receiver": receiver,
                        "embedding": [offset + 10.0 * receiver + sample * 0.01, 0.0],
                    }
                )
                rows.append(
                    {
                        "scope": "probe_eval",
                        "scenario": "clean",
                        "held_receiver": held_receiver,
                        "receiver": receiver,
                        "embedding": [offset + 10.0 * receiver + sample * 0.01, 0.0],
                    }
                )

    result = _receiver_probes({"D1": rows}, seed=3)["D1"]

    assert result["status"] == "COMPLETE"
    assert set(result["folds"]) == {"5", "6"}
    assert result["best_balanced_accuracy"] == 1.0
