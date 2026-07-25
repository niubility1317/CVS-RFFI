from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_gate_cli_help() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "run_d103_r1_held_gate.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--receipts-json" in result.stdout
