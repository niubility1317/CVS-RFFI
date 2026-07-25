from pathlib import Path
import subprocess
import sys


def test_assemble_held_receipts_cli_help() -> None:
    script = Path(__file__).resolve().parents[1] / "code" / "scripts" / "assemble_d103_r2_held_receipts.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--runner-resource-json" in result.stdout
    assert "--scores-json" in result.stdout
