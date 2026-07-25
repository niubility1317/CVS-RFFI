from pathlib import Path
import subprocess
import sys


def test_predictor_cli_has_no_truth_argument() -> None:
    script = Path(__file__).resolve().parents[1] / "code" / "scripts" / "run_d103_r2_held_predictor.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--package-root" in result.stdout
    assert "truth" not in result.stdout.lower()
