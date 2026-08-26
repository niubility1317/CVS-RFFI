from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.jmrs02_j0 import DEFAULT_COMBINATIONS, analyze_j0_rows, fisher_ratio
from score_phase1_jmrs02_j0 import REQUIRED_ARTIFACTS, score_j0_prediction_streams


SCENARIOS = ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
ROWS = ("M0", "S1", "R1", "R2", "D1", "P2")


def _joined_fixture() -> list[dict]:
    rescue = {
        "M0": set(),
        "S1": {0},
        "R1": {0, 1},
        "R2": {2},
        "D1": {2, 3},
        "P2": {4},
    }
    rows: list[dict] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for index in range(12):
            true_class = index % 2
            receiver = index % 2
            day = (index // 2) % 2
            base_correct = index >= 6
            for row in ROWS:
                correct = base_correct if row == "M0" else index in rescue[row]
                predicted = true_class if correct else 1 - true_class
                alpha = 0.10 if row == "D1" else 0.0
                rows.append(
                    {
                        "sample_id": f"{row}:{scenario}:{index}",
                        "row": row,
                        "scope": "held_audit",
                        "scenario": scenario,
                        "held_receiver": receiver,
                        "receiver": receiver,
                        "day": day,
                        "base_index": index,
                        "true_class": true_class,
                        "predicted_class": predicted,
                        "safe_predicted_class": predicted,
                        "safe_alpha": alpha,
                        "embedding": [
                            float(4 * true_class + 0.1 * index),
                            float(receiver + 0.01 * scenario_index),
                        ],
                        "runtime_ms_per_sample": 0.2 + 0.01 * ROWS.index(row),
                        "parameter_count": 100 * ROWS.index(row),
                    }
                )
    return rows


def _write_streams(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path]:
    predictions = tmp_path / "predictions.jsonl"
    truths = tmp_path / "truth.jsonl"
    predictions.write_text(
        "".join(
            json.dumps({key: value for key, value in row.items() if key != "true_class"}) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    truths.write_text(
        "".join(
            json.dumps({"sample_id": row["sample_id"], "true_class": row["true_class"]}) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return predictions, truths


def test_fisher_ratio_is_scale_and_dimension_comparable() -> None:
    labels = np.asarray([0, 0, 1, 1])
    base = np.asarray([[0.0], [0.2], [3.0], [3.2]])
    expanded = np.concatenate((10.0 * base, np.zeros((4, 2))), axis=1)

    assert fisher_ratio(base, labels) == pytest.approx(fisher_ratio(expanded, labels))


def test_j0_separates_family_effect_system_safety_and_zero_alpha_gate() -> None:
    result = analyze_j0_rows(_joined_fixture(), bootstrap_resamples=100, seed=7)

    semantic = result["semantic_audit"]
    assert semantic["R1"]["nondegraded_receiver_count_vs_family_sham"] == 2
    assert semantic["R1"]["family_sham_row"] == "S1"
    assert semantic["R1"]["family_sham_scope"] == "legacy_shared_capacity_control_not_family_specific"
    assert semantic["R1"]["nondegraded_receiver_count_vs_M0"] == 0
    assert semantic["R1"]["safe_gate"]["evaluable_nonzero_alpha"] is False
    assert semantic["R1"]["safe_gate"]["passes"] is False
    assert semantic["D1"]["safe_gate"]["evaluable_nonzero_alpha"] is True
    assert semantic["D1"]["breadth_vs_core90_rescue"]["receiver_count"] == 2


def test_j0_pair_and_triple_synergy_use_core90_rescue_sets() -> None:
    result = analyze_j0_rows(_joined_fixture(), bootstrap_resamples=100, seed=11)

    joint = result["joint_rescue"]
    assert tuple(DEFAULT_COMBINATIONS[0]) == ("R1", "D1")
    assert joint["R1+D1"]["synergy"] == pytest.approx(2.0 / 12.0)
    assert joint["R1+D1"]["synergy_pp"] == pytest.approx(100.0 * 2.0 / 12.0)
    assert (
        joint["R1+D1"]["synergy_group_bootstrap"]["aggregation"]
        == "preaggregated_receiver_day_scenario_group_counts"
    )
    assert joint["R2+D1"]["synergy"] == pytest.approx(0.0)
    assert joint["R1+D1+P2"]["synergy"] == pytest.approx(3.0 / 12.0)
    assert joint["R1+D1"]["unique_rescue_vs_S1_count"] == 12
    assert joint["R1+D1"]["rescue_jaccard"] == pytest.approx(0.0)
    assert set(joint["R1+D1"]["member_unique_rescue_count"]) == {"R1", "D1"}


def test_cost_is_explicitly_incremental_not_full_system() -> None:
    result = analyze_j0_rows(_joined_fixture(), bootstrap_resamples=20, seed=3)

    cost = result["cost_scope"]["D1"]
    assert cost["incremental_runtime_ms_per_sample"] > 0.0
    assert cost["runtime_scope"] == "post_cached_core90_branch_only"
    assert cost["full_system_runtime_ms_per_sample"] is None
    assert result["cost_scope"]["M0"]["runtime_scope"] == "cached_core90_access_only"
    assert result["decision"]["direct_joint_training_authorized"] is False


def test_j0_stream_scorer_writes_fresh_five_artifact_closure(tmp_path: Path) -> None:
    prediction_path, truth_path = _write_streams(tmp_path, _joined_fixture())
    output = tmp_path / "j0"

    result = score_j0_prediction_streams(
        prediction_path,
        truth_path,
        output,
        bootstrap_resamples=20,
        seed=5,
    )

    assert result["status"] == "ANALYZED"
    assert result["artifact_count"] == 5
    assert all((output / name).is_file() for name in REQUIRED_ARTIFACTS)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        score_j0_prediction_streams(
            prediction_path,
            truth_path,
            output,
            bootstrap_resamples=20,
            seed=5,
        )


def test_j0_rejects_missing_preregistered_row() -> None:
    rows = [row for row in _joined_fixture() if row["row"] != "P2"]

    with pytest.raises(ValueError, match="missing preregistered rows"):
        analyze_j0_rows(rows, bootstrap_resamples=20, seed=5)
