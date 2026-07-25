from __future__ import annotations

import subprocess
import sys


def test_help_exposes_all_lineage_and_output_arguments() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "code/scripts/export_d103_r1_source_splits.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for token in (
        "--cache-set",
        "--selection-salt-receipt",
        "--runtime",
        "--export-receipt",
        "--parity-receipt",
        "--checkpoint",
        "--class-ids",
        "--output-dir",
    ):
        assert token in result.stdout
