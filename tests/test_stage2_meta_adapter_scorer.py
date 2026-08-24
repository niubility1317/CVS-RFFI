from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import cvsrffi.stage2_meta_adapter_scorer as scorer  # noqa: E402


QUERY_IDS = np.asarray(["q-1", "q-2", "q-3", "q-4"])
DA0 = np.asarray([10, 10, 20, 20], dtype=np.int64)
DA1 = np.asarray([10, 20, 20, 20], dtype=np.int64)
SCORES = np.asarray(
    [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]],
    dtype=np.float32,
)
TRUTH = {
    "schema": "cvs.truth.v1",
    "rows": [
        {"query_token": "q-3", "true_class_index": 20},
        {"query_token": "q-1", "true_class_index": 10},
        {"query_token": "q-4", "true_class_index": 20},
        {"query_token": "q-2", "true_class_index": 10},
    ],
}


def _write_prediction(
    path: Path,
    *,
    query_ids: np.ndarray = QUERY_IDS,
    predicted: np.ndarray = DA0,
    scores: np.ndarray = SCORES,
    extra: dict[str, np.ndarray] | None = None,
) -> None:
    np.savez(
        path,
        query_ids=query_ids,
        predicted_class_ids=predicted,
        scores=scores,
        **(extra or {}),
    )


def _write_truth(path: Path, payload: dict[str, object] | None = None) -> None:
    path.write_text(
        json.dumps(payload or TRUTH),
        encoding="utf-8",
    )


def test_scorer_requires_identical_query_ids_and_reports_reg0_na(tmp_path: Path) -> None:
    da0_path = tmp_path / "predictions_DA0_REG0.npz"
    da1_path = tmp_path / "predictions_DA1_REG0.npz"
    truth_path = tmp_path / "truth.json"
    _write_prediction(da0_path)
    _write_prediction(da1_path, predicted=DA1)
    _write_truth(truth_path)

    score = scorer.score_meta_adapter_pair(da0_path, da1_path, truth_path)

    assert score.da0.state == "DA0_REG0"
    assert score.da1.state == "DA1_REG0"
    assert score.da0.seen_new_acc is None
    assert score.da1.h_old_new is None
    assert score.da0.mean_old_acc == pytest.approx(1.0)
    assert score.da1.mean_old_acc == pytest.approx(0.75)
    assert score.da0.old_class_floor == pytest.approx(1.0)
    assert score.da1.old_class_floor == pytest.approx(0.5)
    assert score.mean_delta_pp == pytest.approx(-25.0)
    assert score.floor_delta_pp == pytest.approx(-50.0)


def test_scorer_rejects_itemwise_row_id_drift_before_truth_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    da0_path = tmp_path / "da0.npz"
    da1_path = tmp_path / "da1.npz"
    truth_path = tmp_path / "truth.json"
    _write_prediction(da0_path)
    _write_prediction(da1_path, query_ids=QUERY_IDS[[1, 0, 2, 3]], predicted=DA1)
    _write_truth(truth_path)

    truth_opened = False

    def forbidden_truth_open(_path: str | Path):
        nonlocal truth_opened
        truth_opened = True
        raise AssertionError("truth must remain unopened")

    monkeypatch.setattr(scorer, "_load_truth", forbidden_truth_open)
    with pytest.raises(ValueError, match="same ordered query IDs"):
        scorer.score_meta_adapter_pair(da0_path, da1_path, truth_path)
    assert truth_opened is False


def test_scorer_rejects_invalid_prediction_before_truth_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    da0_path = tmp_path / "da0.npz"
    da1_path = tmp_path / "da1.npz"
    truth_path = tmp_path / "truth.json"
    _write_prediction(da0_path, extra={"query_truth": np.asarray([10, 10, 20, 20])})
    _write_prediction(da1_path, predicted=DA1)
    _write_truth(truth_path)

    truth_opened = False

    def forbidden_truth_open(_path: str | Path):
        nonlocal truth_opened
        truth_opened = True
        raise AssertionError("truth must remain unopened")

    monkeypatch.setattr(scorer, "_load_truth", forbidden_truth_open)
    with pytest.raises(ValueError, match="prediction artifact"):
        scorer.score_meta_adapter_pair(da0_path, da1_path, truth_path)
    assert truth_opened is False


def test_promotion_requires_both_mean_and_floor_thresholds() -> None:
    assert scorer.summarize_rows(mean_delta_pp=1.1, floor_delta_pp=0.4).promote is False
    assert scorer.summarize_rows(mean_delta_pp=0.9, floor_delta_pp=1.0).promote is False
    assert scorer.summarize_rows(mean_delta_pp=1.0, floor_delta_pp=0.5).promote is True


def test_score_json_writer_refuses_overwrite(tmp_path: Path) -> None:
    da0_path = tmp_path / "da0.npz"
    da1_path = tmp_path / "da1.npz"
    truth_path = tmp_path / "truth.json"
    output_path = tmp_path / "score.json"
    _write_prediction(da0_path)
    _write_prediction(da1_path, predicted=DA1)
    _write_truth(truth_path)
    output_path.write_text("keep", encoding="utf-8")

    score = scorer.score_meta_adapter_pair(da0_path, da1_path, truth_path)
    with pytest.raises(FileExistsError, match="already exists"):
        scorer.write_score_json(score, output_path)
    assert output_path.read_text(encoding="utf-8") == "keep"
