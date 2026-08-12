from __future__ import annotations

import subprocess
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_clic_target_prediction12_v1_20260812.sh"


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{relative}"


def test_target_prediction_launcher_dry_run_is_exact_iq_only_matrix() -> None:
    result = subprocess.run(
        ["bash", _wsl_path(LAUNCHER), "--dry-run"],
        cwd=CODE_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 14
    assert sum("stage=TARGET_VALIDATION" in line for line in lines) == 1
    assert sum("stage=TARGET_PACKAGE" in line for line in lines) == 1
    prediction_lines = [line for line in lines if "stage=TARGET_PREDICTION" in line]
    assert len(prediction_lines) == 12
    assert sum("arm=C" in line for line in prediction_lines) == 6
    assert sum("arm=G" in line for line in prediction_lines) == 6
    for fold in range(1, 7):
        fold_lines = [line for line in prediction_lines if f"fold={fold}" in line]
        assert len(fold_lines) == 2
        assert any(
            f"F{fold}C_CLIC12/c_predictor_state.json" in line for line in fold_lines
        )
        assert any(
            f"F{fold}G_CLIC12/g_deployment_bundle.zip" in line for line in fold_lines
        )
    joined = "\n".join(lines)
    assert sum(
        "phase1_clic_predictor_artifacts_20260812_v2" in line
        for line in prediction_lines
    ) == 6
    assert sum(
        "phase1_clic_g_bundles_20260812_v3_safe_pack" in line
        for line in prediction_lines
    ) == 6
    assert joined.count("--package") == 12
    for forbidden in (
        "--truth-sidecar",
        "--adv3b02-reference",
        "--score-target-prediction",
        "--target-fit",
        "--target-update",
        "--target-role",
        "--query-truth",
        "--selection",
        "--retry",
    ):
        assert forbidden not in joined
