from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "build_cvs_stage2c_effective8_strict_plan.py",
    "run_cvs_stage2c_effective8_strict_package.py",
    "run_cvs_stage2c_effective8_strict_plan.py",
    "authorize_cvs_stage2c_effective8_strict_plan.py",
)


def test_strict_cli_entrypoints_resolve_repository_imports() -> None:
    for name in SCRIPTS:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "paper_reproduction/scripts" / name), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (name, completed.stdout, completed.stderr)
