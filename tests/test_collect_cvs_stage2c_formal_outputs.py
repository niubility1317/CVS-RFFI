from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from paper_reproduction.scripts.collect_cvs_stage2c_formal_outputs import (
    collect_outputs,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> Path:
    commands: list[list[str]] = []
    for index in range(2):
        out_dir = tmp_path / f"cell_{index}"
        commands.append(["python", "benchmark.py", "--out_dir", str(out_dir)])
        rows = [
            {
                "receiver": f"r{index}",
                "seed": 1,
                "scenario": scenario,
                "new_class_count": 5,
                "k_shot": 1,
            }
            for scenario in ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
        ]
        predictions = [
            {**row, "query_id": f"q:{index}:{position}"}
            for position, row in enumerate(rows)
        ]
        _write_csv(out_dir / "formal_rows.csv", rows)
        _write_csv(out_dir / "formal_predictions.csv", predictions)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "cvs_stage2c_effective8_generated_execution_plan_v1",
                "expected_counts": {
                    "benchmark_invocations": 2,
                    "formal_scenario_rows": 6,
                },
                "commands": {"benchmark": commands},
            }
        ),
        encoding="utf-8",
    )
    return plan


def test_collect_outputs_binds_all_inputs_and_rows(tmp_path: Path) -> None:
    plan = _fixture(tmp_path)
    result = collect_outputs(
        plan,
        out_dir=tmp_path / "combined",
        expected_invocations=2,
        expected_formal_rows=6,
    )
    assert result["benchmark_invocations"] == 2
    assert result["formal_row_count"] == 6
    assert result["formal_prediction_count"] == 6
    assert len(result["input_artifacts"]) == 2
    assert (tmp_path / "combined" / "formal_rows.csv").is_file()


def test_collect_outputs_rejects_missing_cell(tmp_path: Path) -> None:
    plan = _fixture(tmp_path)
    (tmp_path / "cell_1" / "formal_rows.csv").unlink()
    with pytest.raises(FileNotFoundError, match="evidence is missing"):
        collect_outputs(
            plan,
            out_dir=tmp_path / "combined",
            expected_invocations=2,
            expected_formal_rows=6,
        )


def test_collect_outputs_rejects_schema_drift(tmp_path: Path) -> None:
    plan = _fixture(tmp_path)
    _write_csv(
        tmp_path / "cell_1" / "formal_rows.csv",
        [
            {
                "receiver": "r1",
                "seed": 1,
                "scenario": scenario,
                "new_class_count": 5,
                "k_shot": 1,
                "extra": "drift",
            }
            for scenario in ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
        ],
    )
    with pytest.raises(ValueError, match="CSV schema drift"):
        collect_outputs(
            plan,
            out_dir=tmp_path / "combined",
            expected_invocations=2,
            expected_formal_rows=6,
        )
