from pathlib import Path
import subprocess
import sys


def test_prepare_held_packages_cli_help() -> None:
    script = Path(__file__).resolve().parents[1] / "code" / "scripts" / "prepare_d103_r2_held_packages.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--source-val-archive" in result.stdout
    assert "--fits-root" in result.stdout
