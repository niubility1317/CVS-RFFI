from pathlib import Path
import subprocess
import sys


def test_real_feature_smoke_cli_help() -> None:
    script = Path(__file__).resolve().parents[1] / "code" / "scripts" / "run_d103_r2_real_feature_smoke.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--tap-archive" in result.stdout
    assert "--output-json" in result.stdout
