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
