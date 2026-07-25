from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_comparator_builder_cli_help() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "build_d103_r2_d102_comparators.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--labeled-archive" in result.stdout
    assert "--code-sha256" in result.stdout
