from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from audit_phase1_jmrs01 import (
    REQUIRED_SCENARIOS,
    build_loro_partition,
    validate_output_root,
    validate_source_only_args,
    write_closed_prediction_truth_streams,
)


def _args(**overrides) -> argparse.Namespace:
    values = {
        "rows": "M0,R1,R2,D1,P1,P2,S1",
        "train_role": "L_s",
        "select_role": "V_select",
        "cal_role": "V_cal",
        "audit_role": "V_select",
        "target_or_query_access": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_source_only_contract_rejects_target_access_and_removed_d2() -> None:
    with pytest.raises(ValueError, match="target/query access"):
        validate_source_only_args(_args(target_or_query_access=True))
    with pytest.raises(ValueError, match="known transmitted symbols"):
        validate_source_only_args(_args(rows="M0,D2"))


def test_loro_partition_never_exposes_held_receiver_to_train_select_or_cal() -> None:
    train_rx = torch.tensor([0, 1, 2, 3, 0, 3])
    select_rx = torch.tensor([0, 1, 2, 3])
    cal_rx = torch.tensor([0, 1, 2, 3])

    fold = build_loro_partition(train_rx, select_rx, cal_rx, held_receiver=2)

    assert train_rx[fold.train].tolist() == [0, 1, 3, 0, 3]
    assert select_rx[fold.select].tolist() == [0, 1, 3]
    assert cal_rx[fold.cal].tolist() == [0, 1, 3]
    assert select_rx[fold.audit].tolist() == [2]
    assert not fold.select.logical_and(fold.audit).any()


def test_output_root_is_non_overwriting(tmp_path: Path) -> None:
    candidate = tmp_path / "fresh"
    assert validate_output_root(candidate) == candidate.resolve()
    candidate.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validate_output_root(candidate)


def test_truth_is_not_written_until_prediction_ids_close(tmp_path: Path) -> None:
    predictions = [{"sample_id": "a", "row": "M0", "scenario": "clean", "predicted_class": 1}]
    mismatched_truth = [{"sample_id": "b", "true_class": 1}]

    with pytest.raises(RuntimeError, match="closure mismatch"):
        write_closed_prediction_truth_streams(tmp_path, predictions, mismatched_truth)

    assert (tmp_path / "predictions.jsonl").is_file()
    assert not (tmp_path / "truth.jsonl").exists()


def test_closed_stream_contains_all_four_scenarios(tmp_path: Path) -> None:
    predictions = []
    truth = []
    for index, scenario in enumerate(REQUIRED_SCENARIOS):
        sample_id = f"D1:{scenario}:0"
        predictions.append(
            {"sample_id": sample_id, "row": "D1", "scenario": scenario, "predicted_class": index}
        )
        truth.append({"sample_id": sample_id, "true_class": index})

    result = write_closed_prediction_truth_streams(tmp_path, predictions, truth)

    assert result["truth_written_after_prediction_close"] is True
    assert result["prediction_count"] == 4
    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert {row["scenario"] for row in rows} == set(REQUIRED_SCENARIOS)
