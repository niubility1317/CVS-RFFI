from pathlib import Path
import subprocess
import sys


def test_d104_predictor_cli_has_no_truth_or_role_argument() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "run_d104_r1_held_predictor.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    lowered = result.stdout.lower()
    assert "truth" not in lowered
    assert "role" not in lowered
    assert "--package-root" in result.stdout
    assert "--output-dir" in result.stdout
