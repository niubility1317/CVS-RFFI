from pathlib import Path
import subprocess
import sys


def test_truth_side_scorer_cli_help() -> None:
    script = Path(__file__).resolve().parents[1] / "code" / "scripts" / "score_d103_r2_held_predictions.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--truth-json" in result.stdout
    assert "--prediction-root" in result.stdout
