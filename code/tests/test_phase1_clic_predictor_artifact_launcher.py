from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
ENTRY = CODE_ROOT / "phase1_clic_target_leo_cli.py"
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_clic_predictor_artifacts12_v2_20260812.sh"


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{relative}"


def test_top_level_c_predictor_entry_help_avoids_package_module_shadowing() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(CODE_ROOT)
    result = subprocess.run(
        [sys.executable, "-u", str(ENTRY), "--help"],
        cwd=CODE_ROOT.parent,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--seal-c-predictor-state" in result.stdout


def test_predictor_artifact_v2_launcher_uses_top_level_entry_and_exact_matrix() -> None:
    result = subprocess.run(
        ["bash", _wsl_path(LAUNCHER), "--dry-run"],
        cwd=CODE_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 12
    assert sum("stage=CLIC_C_DESCRIPTOR" in line for line in lines) == 6
    assert sum("stage=CLIC_G_BUNDLE" in line for line in lines) == 6
    c_lines = [line for line in lines if "stage=CLIC_C_DESCRIPTOR" in line]
    assert all("phase1_clic_target_leo_cli.py" in line for line in c_lines)
    assert all("cvsrffi/phase1_clic_target_leo.py" not in line for line in c_lines)
    assert all(" -m " not in f" {line} " for line in c_lines)
