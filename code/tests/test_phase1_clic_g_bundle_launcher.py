from __future__ import annotations

import subprocess
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_clic_g_bundles6_v1_20260812.sh"


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{relative}"


def test_g_bundle_launcher_is_exact_six_fold_source_only_matrix() -> None:
    result = subprocess.run(
        ["bash", _wsl_path(LAUNCHER), "--dry-run"],
        cwd=CODE_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 6
    assert all("stage=CLIC_G_BUNDLE" in line and "arm=G" in line for line in lines)
    for fold in range(1, 7):
        matches = [line for line in lines if f"fold={fold}" in line]
        assert len(matches) == 1
        assert f"F{fold}G_CLIC12/g_deployment_bundle.zip" in matches[0]
    joined = "\n".join(lines)
    assert joined.count("phase1_clic12_20260812_v5") == 12
    assert joined.count("phase1_clic_postfreeze_20260812_v4") == 6
    assert joined.count("phase1_clic_source_leo_20260812_v4") == 12
    for forbidden in (
        "F1C_CLIC12",
        "--target",
        "--truth",
        "--score",
        "--role",
        "--query",
        "--package",
    ):
        assert forbidden not in joined
