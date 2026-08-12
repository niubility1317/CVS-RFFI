from __future__ import annotations

import subprocess
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_clic_target_metrics12_v1_20260812.sh"


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{relative}"


def test_target_metrics_launcher_is_exact_baseline_independent_matrix() -> None:
    result = subprocess.run(
        ["bash", _wsl_path(LAUNCHER), "--dry-run"],
        cwd=CODE_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 12
    for fold in range(1, 7):
        fold_lines = [line for line in lines if f"fold={fold}" in line]
        assert len(fold_lines) == 2
        for arm in ("C", "G"):
            line = next(line for line in fold_lines if f"arm={arm}" in line)
            candidate = f"F{fold}{arm}_CLIC12"
            assert "--seal-target-metrics" in line
            assert f"{candidate}.prediction.json" in line
            assert f"{candidate}.metrics.json" in line
    joined = "\n".join(lines)
    assert joined.count("--truth-sidecar") == 12
    assert len(
        {
            token
            for line in lines
            for token in line.split()
            if token.endswith("/sealed_target/truth_sidecar.json")
        }
    ) == 1
    for forbidden in (
        "--adv3b02-reference",
        "--score-target-prediction",
        "--publish-target-prediction",
        "--package",
        "--target-fit",
        "--target-update",
        "--threshold",
        "--class-order",
        "--retry",
    ):
        assert forbidden not in joined


def test_target_metrics_launcher_has_fresh_roots_and_waits_workers() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    guard = '[[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]]'
    assert text.index(guard) < text.index('mkdir -p "${METRICS_ROOT}" "${LOG_ROOT}"')
    assert 'for fold in 1 2 3 4 5 6; do' in text
    assert 'for arm in C G; do' in text
    assert 'wait "${pids[index]}"' in text
