from pathlib import Path
import subprocess
import sys


def test_export_d104_source_split_cli_declares_frozen_inputs() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "export_d104_source_split.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for flag in (
        "--tap-archive",
        "--dual-archive",
        "--exclusion-manifest",
        "--checkpoint-sha256",
        "--runtime-sha256",
        "--cache-set-sha256",
        "--selection-salt-receipt-sha256",
        "--output-dir",
    ):
        assert flag in result.stdout
