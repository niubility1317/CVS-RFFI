from __future__ import annotations

import subprocess
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_clic_g_bundles6_v2_serial_20260812.sh"


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{relative}"


def test_g_bundle_serial_launcher_preserves_exact_six_fold_matrix() -> None:
    result = subprocess.run(
        ["bash", _wsl_path(LAUNCHER), "--dry-run"],
        cwd=CODE_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 6
    for fold, line in enumerate(lines, start=1):
        assert f"stage=CLIC_G_BUNDLE_SERIAL fold={fold} arm=G" in line
        assert f"F{fold}G_CLIC12/g_deployment_bundle.zip" in line
    joined = "\n".join(lines)
    assert "phase1_clic_g_bundles_20260812_v2_serial" in joined
    for forbidden in ("F1C_CLIC12", "--target", "--truth", "--score", "--role", "--query", "--package"):
        assert forbidden not in joined


def test_g_bundle_serial_launcher_waits_each_worker_before_next_fold() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    loop = text.split('for fold in 1 2 3 4 5 6; do')[-1]
    wait_index = loop.index('wait "${worker_pid}"')
    record_index = loop.index('>>"${LOG_ROOT}/pids_g_bundles6_serial.tsv"')
    assert wait_index < record_index
    assert 'declare -a pids' not in loop
    assert 'for index in "${!pids[@]}"' not in loop
