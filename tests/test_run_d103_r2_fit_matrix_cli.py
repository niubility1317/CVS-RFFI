from pathlib import Path
import subprocess
import sys


def test_fit_matrix_cli_help_and_two_worker_default() -> None:
    script = Path(__file__).resolve().parents[1] / "code" / "scripts" / "run_d103_r2_fit_matrix.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--workers-per-gpu" in result.stdout
    assert "--output-root" in result.stdout
