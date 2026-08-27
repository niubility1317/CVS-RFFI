from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from cvsrffi import stage2_metric_scorer as scorer
from cvsrffi.phase2_canonical_summary import (
    CanonicalSummaryError,
    summarize_scored_rows,
)
from cvsrffi.stage2_prediction_artifact import NPZ_FIELD_ALLOWLIST
from scripts.summarize_phase2_canonical_union import main as summary_main


SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
HANDLES = (f"cls_{index:064x}" for index in range(1, 3))
HANDLE_0, HANDLE_1 = tuple(HANDLES)


def _imbalanced_rows() -> list[dict[str, object]]:
    definitions = [
        (0, 0, "rx-a", "day-a", SCENARIOS[0]),
        (0, 0, "rx-a", "day-a", SCENARIOS[0]),
        (0, 0, "rx-a", "day-a", SCENARIOS[0]),
        (0, 0, "rx-a", "day-a", SCENARIOS[1]),
        (0, 0, "rx-a", "day-a", SCENARIOS[1]),
        (1, 1, "rx-a", "day-a", SCENARIOS[1]),
        (1, 1, "rx-a", "day-a", SCENARIOS[2]),
        (1, 1, "rx-a", "day-a", SCENARIOS[2]),
        (1, 0, "rx-b", "day-b", SCENARIOS[0]),
        (2, 0, "rx-b", "day-b", SCENARIOS[2]),
    ]
    return [
        {
            "true_class_index": true_class,
            "predicted_class_index": predicted_class,
            "receiver_label": receiver,
            "day_label": day,
            "scenario": scenario,
            "query_token": f"qid-{index:02d}",
        }
        for index, (
            true_class,
            predicted_class,
            receiver,
            day,
            scenario,
        ) in enumerate(definitions)
    ]


def test_imbalanced_rows_compute_exact_micro_and_unweighted_macros() -> None:
    summary = summarize_scored_rows(_imbalanced_rows())

    assert summary["schema"] == "cvs.phase2.canonical_summary.v1"
    assert summary["sample_count"] == 10
    assert summary["correct_count"] == 8
    assert summary["sample_micro_accuracy"] == pytest.approx(0.8)
    assert summary["receiver_macro_accuracy"] == pytest.approx(0.5)
    assert summary["day_macro_accuracy"] == pytest.approx(0.5)
    assert summary["class_macro_accuracy"] == pytest.approx(7.0 / 12.0)
    assert summary["scene_macro_accuracy"] == pytest.approx(29.0 / 36.0)


def test_group_and_observed_cell_counts_are_complete_and_stably_ordered() -> None:
    summary = summarize_scored_rows(reversed(_imbalanced_rows()))

    assert summary["class_group_count"] == 3
    assert summary["receiver_group_count"] == 2
    assert summary["day_group_count"] == 2
    assert summary["scene_group_count"] == 3
    assert summary["observed_cell_count"] == 6
    assert [row["true_class_index"] for row in summary["class_metrics"]] == [0, 1, 2]
    assert [row["receiver_label"] for row in summary["receiver_metrics"]] == [
        "rx-a",
        "rx-b",
    ]
    assert [row["day_label"] for row in summary["day_metrics"]] == [
        "day-a",
        "day-b",
    ]
    assert [row["scenario"] for row in summary["scene_metrics"]] == list(SCENARIOS)
    assert [
        (
            row["true_class_index"],
            row["receiver_label"],
            row["day_label"],
            row["scenario"],
        )
        for row in summary["cell_metrics"]
    ] == [
        (0, "rx-a", "day-a", SCENARIOS[0]),
        (0, "rx-a", "day-a", SCENARIOS[1]),
        (1, "rx-a", "day-a", SCENARIOS[1]),
        (1, "rx-a", "day-a", SCENARIOS[2]),
        (1, "rx-b", "day-b", SCENARIOS[0]),
        (2, "rx-b", "day-b", SCENARIOS[2]),
    ]
    assert summary["observed_cell_count"] == len(summary["cell_metrics"])


def test_duplicate_scenario_query_token_is_rejected() -> None:
    rows = _imbalanced_rows()
    rows[-1]["scenario"] = rows[0]["scenario"]
    rows[-1]["query_token"] = rows[0]["query_token"]
    with pytest.raises(CanonicalSummaryError, match="duplicate"):
        summarize_scored_rows(rows)


@pytest.mark.parametrize(
    "rows,match",
    [
        ([], "nonempty"),
        ([None], "mapping"),
        ([{}], "missing"),
        ([{**_imbalanced_rows()[0], "true_class_index": None}], "true_class_index"),
        ([{**_imbalanced_rows()[0], "true_class_index": True}], "true_class_index"),
        ([{**_imbalanced_rows()[0], "true_class_index": "0"}], "true_class_index"),
        ([{**_imbalanced_rows()[0], "true_class_index": 0.0}], "true_class_index"),
        ([{**_imbalanced_rows()[0], "true_class_index": -1}], "true_class_index"),
        ([{**_imbalanced_rows()[0], "true_class_index": math.nan}], "true_class_index"),
        ([{**_imbalanced_rows()[0], "true_class_index": math.inf}], "true_class_index"),
        ([{**_imbalanced_rows()[0], "predicted_class_index": None}], "predicted_class_index"),
        ([{**_imbalanced_rows()[0], "predicted_class_index": False}], "predicted_class_index"),
        ([{**_imbalanced_rows()[0], "receiver_label": ""}], "receiver_label"),
        ([{**_imbalanced_rows()[0], "day_label": " "}], "day_label"),
        ([{**_imbalanced_rows()[0], "scenario": "mixed_orbit"}], "scenario"),
        ([{**_imbalanced_rows()[0], "query_token": 7}], "query_token"),
    ],
)
def test_empty_missing_malformed_and_unscored_rows_are_rejected(rows, match) -> None:
    with pytest.raises(CanonicalSummaryError, match=match):
        summarize_scored_rows(rows)


def _scorer_case() -> tuple[dict[str, object], dict[str, object], dict[str, np.ndarray]]:
    truth_rows = [
        {
            "scenario": scenario,
            "query_token": f"qid_{index + 1:064x}",
            "true_class_index": 0 if index != 1 else 1,
            "true_class_handle": HANDLE_0 if index != 1 else HANDLE_1,
            "transmitter_label": "tx-a" if index != 1 else "tx-b",
            "evaluation_role": "target_old",
            "receiver_label": "rx-a",
            "day_label": f"day-{index}",
        }
        for index, scenario in enumerate(SCENARIOS)
    ]
    truth = {
        "stage": "stage2b",
        "receiver": "rx-a",
        "seed": 701,
        "rows": truth_rows,
    }
    binding = {
        "stage": "stage2b",
        "receiver": "rx-a",
        "row_id": "row-001",
        "k_shot": 1,
        "candidate_lock_sha256": "a" * 64,
        "predictor_package_root_sha256": "b" * 64,
        "scenarios": list(SCENARIOS),
    }
    candidate_after = np.asarray([HANDLE_1, HANDLE_1, HANDLE_0])
    arrays = {
        "query_tokens": np.asarray([row["query_token"] for row in truth_rows]),
        "scenarios": np.asarray(SCENARIOS),
        "candidate_after": candidate_after,
        "candidate_before": np.asarray([HANDLE_0, HANDLE_1, HANDLE_0]),
        "identity_after": candidate_after.copy(),
        "identity_before": np.asarray([HANDLE_0, HANDLE_1, HANDLE_0]),
        "direct": np.asarray([HANDLE_0, HANDLE_1, HANDLE_0]),
        "shared_view_counts": np.asarray([1, 3, 5], dtype=np.int64),
    }
    return truth, binding, arrays


def test_scorer_adds_truth_dimensions_and_predicted_index_without_mutating_predictor_arrays() -> None:
    truth, binding, arrays = _scorer_case()
    before = {key: value.copy() for key, value in arrays.items()}
    scorer._validate_truth_rows(truth)

    _formal_rows, prediction_rows = scorer.score_prediction_arrays(
        binding=binding,
        arrays=arrays,
        truth=truth,
    )

    assert [row["predicted_class_index"] for row in prediction_rows] == [1, 1, 0]
    assert [row["day_label"] for row in prediction_rows] == [
        "day-0",
        "day-1",
        "day-2",
    ]
    required = {
        "true_class_index",
        "predicted_class_index",
        "receiver_label",
        "day_label",
        "scenario",
        "query_token",
    }
    assert all(required <= set(row) for row in prediction_rows)
    assert set(arrays) == set(before)
    assert all(np.array_equal(arrays[key], before[key]) for key in arrays)
    assert not {
        "true_class_index",
        "predicted_class_index",
        "receiver_label",
        "day_label",
    }.intersection(NPZ_FIELD_ALLOWLIST)


def test_scorer_rejects_missing_day_and_inconsistent_handle_index_mapping() -> None:
    truth, _binding, _arrays = _scorer_case()
    del truth["rows"][0]["day_label"]
    with pytest.raises(scorer.Stage2ScoringError, match="day_label"):
        scorer._validate_truth_rows(truth)

    truth, _binding, _arrays = _scorer_case()
    truth["rows"][1]["true_class_handle"] = HANDLE_0
    with pytest.raises(scorer.Stage2ScoringError, match="handle.*indices"):
        scorer._validate_truth_rows(truth)


def test_cli_aggregates_formal_json_list_and_jsonl_with_exact_csv_schema(
    tmp_path: Path,
) -> None:
    rows = _imbalanced_rows()
    formal_json = tmp_path / "formal.json"
    formal_json.write_text(
        json.dumps(
            {
                "schema": scorer.FORMAL_PREDICTIONS_SCHEMA,
                "predictions": rows[:3],
            }
        ),
        encoding="utf-8",
    )
    list_json = tmp_path / "rows.json"
    list_json.write_text(json.dumps(rows[3:6]), encoding="utf-8")
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(row) for row in rows[6:]) + "\n",
        encoding="utf-8",
    )
    out_root = tmp_path / "summary"

    assert summary_main(
        [
            "--input",
            str(formal_json),
            "--input",
            str(list_json),
            "--input",
            str(jsonl),
            "--out-root",
            str(out_root),
        ]
    ) == 0

    assert {path.name for path in out_root.iterdir()} == {
        "summary.json",
        "cell_metrics.csv",
    }
    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["sample_count"] == 10
    assert "predictions" not in summary
    with (out_root / "cell_metrics.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
        assert handle.tell() > 0
    assert list(csv_rows[0]) == [
        "true_class_index",
        "receiver_label",
        "day_label",
        "scenario",
        "sample_count",
        "correct_count",
        "accuracy",
    ]
    assert len(csv_rows) == summary["observed_cell_count"]


def test_cli_rejects_existing_root_and_invalid_input_without_creating_root(
    tmp_path: Path,
) -> None:
    valid_input = tmp_path / "valid.json"
    valid_input.write_text(json.dumps(_imbalanced_rows()), encoding="utf-8")
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    assert summary_main(
        ["--input", str(valid_input), "--out-root", str(existing)]
    ) == 2
    assert marker.read_text(encoding="utf-8") == "keep"

    invalid_input = tmp_path / "invalid.json"
    invalid_input.write_text(json.dumps([{"scenario": SCENARIOS[0]}]), encoding="utf-8")
    absent = tmp_path / "absent"
    assert summary_main(
        ["--input", str(invalid_input), "--out-root", str(absent)]
    ) == 2
    assert not absent.exists()
