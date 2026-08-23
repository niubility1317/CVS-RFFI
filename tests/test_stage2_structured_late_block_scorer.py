from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import cvsrffi.stage2_structured_late_block_scorer as scorer


QUERY_IDS = np.asarray(["q-1", "q-2", "q-3", "q-4", "q-5", "q-6"])
PREDICTIONS = np.asarray([10, 20, 20, 20, 10, 30], dtype=np.int64)
SCORES = np.asarray(
    [
        [0.9, 0.1, 0.0],
        [0.2, 0.7, 0.1],
        [0.1, 0.8, 0.1],
        [0.0, 0.9, 0.1],
        [0.8, 0.1, 0.1],
        [0.1, 0.1, 0.8],
    ],
    dtype=np.float32,
)
TRUTH_BY_ID = {"q-1": 10, "q-2": 10, "q-3": 20, "q-4": 20, "q-5": 30, "q-6": 30}


def _write_prediction(
    path: Path,
    *,
    query_ids: np.ndarray = QUERY_IDS,
    predicted_class_ids: np.ndarray = PREDICTIONS,
    scores: np.ndarray = SCORES,
    extra: dict[str, np.ndarray] | None = None,
) -> None:
    np.savez(
        path,
        query_ids=query_ids,
        predicted_class_ids=predicted_class_ids,
        scores=scores,
        **(extra or {}),
    )


def _write_truth(path: Path, *, rows: list[dict[str, object]] | None = None) -> None:
    if rows is None:
        rows = [
            {
                "query_token": query_id,
                "true_class_index": TRUTH_BY_ID[query_id],
                "ignored_existing_sidecar_metadata": "not-read-by-scorer",
            }
            for query_id in reversed(QUERY_IDS.tolist())
        ]
    path.write_text(
        json.dumps(
            {
                "schema": "existing.truth.sidecar",
                "stage": "stage2b",
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )


def test_scores_da1_reg0_old_metrics_and_closes_schema(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predictions.npz"
    truth_path = tmp_path / "truth.json"
    output_path = tmp_path / "score.json"
    _write_prediction(prediction_path)
    _write_truth(truth_path)

    result = scorer.score_stage2b_predictions(
        prediction_path,
        truth_path,
        output_path=output_path,
    )

    assert result == json.loads(output_path.read_text(encoding="utf-8"))
    assert set(result) == {
        "schema",
        "status",
        "state",
        "registration_state",
        "join_policy",
        "prediction_rows_verified_before_truth_open",
        "truth_rows_joined",
        "old_class_metrics",
        "new_class_metrics",
        "mrior_comparison",
        "scorer_output_must_not_feed_predictor",
    }
    assert result["state"] == "DA1_REG0"
    assert result["registration_state"] == "REG0"
    assert result["join_policy"] == "exact_opaque_query_id"
    assert result["prediction_rows_verified_before_truth_open"] == 6
    assert result["truth_rows_joined"] == 6
    old = result["old_class_metrics"]
    assert old["micro_accuracy"] == pytest.approx(4 / 6)
    assert old["macro_accuracy"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert old["per_class_accuracy"] == {"10": 0.5, "20": 1.0, "30": 0.5}
    assert old["per_class_correct"] == {"10": 1, "20": 2, "30": 1}
    assert old["per_class_total"] == {"10": 2, "20": 2, "30": 2}
    assert old["floor_accuracy"] == 0.5
    assert result["new_class_metrics"] == {
        "new_class_accuracy": "N/A",
        "seen_new_accuracy": "N/A",
        "old_new_harmonic_mean": "N/A",
        "reason": "REG0 has no registered new classes",
    }
    assert result["mrior_comparison"]["status"] == "UNKNOWN"
    assert result["mrior_comparison"]["promotion_verdict"] == "UNKNOWN"
    assert result["scorer_output_must_not_feed_predictor"] is True


@pytest.mark.parametrize(
    "invalid_case",
    ["extra_schema", "length_mismatch", "duplicate_id", "nonfinite_score"],
)
def test_invalid_prediction_is_fully_rejected_before_truth_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_case: str,
) -> None:
    prediction_path = tmp_path / "predictions.npz"
    truth_path = tmp_path / "truth.json"
    output_path = tmp_path / "score.json"
    _write_truth(truth_path)
    if invalid_case == "extra_schema":
        _write_prediction(
            prediction_path,
            extra={"query_truth": np.asarray([10, 10, 20, 20, 30, 30])},
        )
    elif invalid_case == "length_mismatch":
        _write_prediction(
            prediction_path,
            predicted_class_ids=PREDICTIONS[:-1],
        )
    elif invalid_case == "duplicate_id":
        duplicate = QUERY_IDS.copy()
        duplicate[-1] = duplicate[0]
        _write_prediction(prediction_path, query_ids=duplicate)
    else:
        nonfinite = SCORES.copy()
        nonfinite[0, 0] = np.nan
        _write_prediction(prediction_path, scores=nonfinite)

    truth_opened = False

    def forbidden_truth_open(_path):
        nonlocal truth_opened
        truth_opened = True
        raise AssertionError("truth must remain unopened")

    monkeypatch.setattr(scorer, "_load_truth_json", forbidden_truth_open)
    with pytest.raises(ValueError, match="prediction"):
        scorer.score_stage2b_predictions(
            prediction_path,
            truth_path,
            output_path=output_path,
        )
    assert truth_opened is False
    assert not output_path.exists()


def test_truth_join_requires_exact_opaque_id_set(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predictions.npz"
    truth_path = tmp_path / "truth.json"
    output_path = tmp_path / "score.json"
    _write_prediction(prediction_path)
    rows = [
        {"query_token": query_id, "true_class_index": TRUTH_BY_ID[query_id]}
        for query_id in QUERY_IDS.tolist()[:-1]
    ]
    rows.append({"query_token": "different-opaque-id", "true_class_index": 30})
    _write_truth(truth_path, rows=rows)

    with pytest.raises(ValueError, match="exact opaque-ID join"):
        scorer.score_stage2b_predictions(
            prediction_path,
            truth_path,
            output_path=output_path,
        )
    assert not output_path.exists()


def test_output_preflight_prevents_overwrite_and_truth_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction_path = tmp_path / "predictions.npz"
    truth_path = tmp_path / "truth.json"
    output_path = tmp_path / "score.json"
    _write_prediction(prediction_path)
    _write_truth(truth_path)
    output_path.write_text("keep-existing-score", encoding="utf-8")
    truth_opened = False

    def forbidden_truth_open(_path):
        nonlocal truth_opened
        truth_opened = True
        raise AssertionError("truth must remain unopened")

    monkeypatch.setattr(scorer, "_load_truth_json", forbidden_truth_open)
    with pytest.raises(ValueError, match="already exists"):
        scorer.score_stage2b_predictions(
            prediction_path,
            truth_path,
            output_path=output_path,
        )
    assert output_path.read_text(encoding="utf-8") == "keep-existing-score"
    assert truth_opened is False


def test_scorer_has_no_baseline_or_predictor_feedback_interface() -> None:
    parameters = set(inspect.signature(scorer.score_stage2b_predictions).parameters)
    assert parameters == {"prediction_path", "truth_path", "output_path"}
    assert not {
        "baseline",
        "mrior",
        "model",
        "predictor",
        "checkpoint",
        "query_truth",
        "query_role",
    } & parameters


def test_cli_writes_the_same_closed_score_schema(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predictions.npz"
    truth_path = tmp_path / "truth.json"
    output_path = tmp_path / "score.json"
    _write_prediction(prediction_path)
    _write_truth(truth_path)
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "score_adv3b02_structured_lateblock_stage2b.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--predictions",
            str(prediction_path),
            "--truth",
            str(truth_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["schema"] == "cvs.stage2.structured_lateblock.score.v1"
    assert result["state"] == "DA1_REG0"
    assert result["mrior_comparison"]["status"] == "UNKNOWN"
