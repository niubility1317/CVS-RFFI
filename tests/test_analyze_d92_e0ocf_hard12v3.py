from __future__ import annotations

import subprocess
import sys


def test_analyzer_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "code/scripts/analyze_d92_e0ocf_hard12v3.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "matrix-manifest" in result.stdout
