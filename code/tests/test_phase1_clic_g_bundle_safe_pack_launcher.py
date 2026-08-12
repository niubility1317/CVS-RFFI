from __future__ import annotations

import subprocess
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_clic_g_bundles6_v3_safe_pack_20260812.sh"
PREVIOUS = CODE_ROOT / "scripts" / "launch_phase1_clic_g_bundles6_v2_serial_20260812.sh"


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{relative}"


def _dry_run(path: Path) -> list[str]:
    result = subprocess.run(
        ["bash", _wsl_path(path), "--dry-run"],
        cwd=CODE_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_g_bundle_safe_pack_launcher_preserves_exact_six_fold_matrix() -> None:
    lines = _dry_run(LAUNCHER)
    assert len(lines) == 6
    for fold, line in enumerate(lines, start=1):
        assert f"stage=CLIC_G_BUNDLE_SAFE_PACK_SERIAL fold={fold} arm=G" in line
        assert f"F{fold}G_CLIC12/g_deployment_bundle.zip" in line
    joined = "\n".join(lines)
    assert "phase1_clic_g_bundles_20260812_v3_safe_pack" in joined
    for forbidden in ("F1C_CLIC12", "--target", "--truth", "--score", "--role", "--query", "--package"):
        assert forbidden not in joined


def test_g_bundle_safe_pack_launcher_only_changes_execution_identity() -> None:
    current = "\n".join(_dry_run(LAUNCHER))
    previous = "\n".join(_dry_run(PREVIOUS))
    normalized = current.replace(
        "phase1_clic_g_bundles_20260812_v3_safe_pack",
        "phase1_clic_g_bundles_20260812_v2_serial",
    ).replace("CLIC_G_BUNDLE_SAFE_PACK_SERIAL", "CLIC_G_BUNDLE_SERIAL")
    assert normalized == previous
    text = LAUNCHER.read_text(encoding="utf-8")
    loop = text.split('for fold in 1 2 3 4 5 6; do')[-1]
    assert loop.index('wait "${worker_pid}"') < loop.index('>>"${LOG_ROOT}/pids_g_bundles6_safe_pack.tsv"')
