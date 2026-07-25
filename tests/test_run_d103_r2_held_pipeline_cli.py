from pathlib import Path
import subprocess
import sys


def test_complete_pipeline_cli_help() -> None:
    script = Path(__file__).resolve().parents[1] / "code" / "scripts" / "run_d103_r2_held_pipeline.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--source-split-root" in result.stdout
    assert "--run-root" in result.stdout
