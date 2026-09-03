from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_rx_v2_real_checkpoint_smoke_entrypoint_imports_from_repository_root() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repository_root / "code" / "scripts" / "smoke_adv3b02_daot_stn_rx_v2.py"),
            "--help",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "no-query smoke" in result.stdout
