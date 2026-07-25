from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "script_name,forbidden",
    (
        ("run_d104_r1_held_predictor.py", ("truth", "role")),
        ("prepare_d104_r1_held_packages.py", ("target",)),
    ),
)
def test_d104_prediction_side_cli_boundaries(script_name, forbidden) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / script_name
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    lowered = result.stdout.lower()
    assert all(value not in lowered for value in forbidden)


@pytest.mark.parametrize(
    "flag,value",
    (
        ("--gpus", "0,1,2,3"),
        ("--workers-per-gpu", "1"),
    ),
)
def test_d104_formal_pipeline_rejects_nonfrozen_lanes_before_io(
    tmp_path, flag, value
) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "run_d104_r1_held_pipeline.py"
    )
    run_root = tmp_path / "must_not_exist"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source-split-root",
            str(tmp_path / "missing_source"),
            "--checkpoint-sha256",
            "a" * 64,
            "--runtime-sha256",
            "b" * 64,
            "--method-lock-sha256",
            "c" * 64,
            "--run-root",
            str(run_root),
            flag,
            value,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert not run_root.exists()
