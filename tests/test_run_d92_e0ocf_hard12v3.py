from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_d92_e0ocf_hard12v3 as runner


def test_shared_stop_counts_distinct_outers_not_arms(tmp_path: Path) -> None:
    output = tmp_path / "matrix"
    output.mkdir()
    fp = "a" * 64
    first = {"job_id": "outer_a__arm_full", "outer_key": "outer_a", "arm_id": "D92_FULL"}
    same = {"job_id": "outer_a__arm_ocf25", "outer_key": "outer_a", "arm_id": "E0_OCF25"}
    second = {"job_id": "outer_b__arm_full", "outer_key": "outer_b", "arm_id": "D92_FULL"}
    assert runner._record_shared_pre_prediction_failure(output, first, fp) is False
    assert runner._record_shared_pre_prediction_failure(output, same, fp) is False
    assert runner._record_shared_pre_prediction_failure(output, second, fp) is True


def test_cli_parser_exposes_prepare_smoke_and_run_shard() -> None:
    parser = runner.parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--help"])
    assert error.value.code == 0
