from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh"


def _dry_run(tmp_path: Path, only: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "ROOT": ROOT.as_posix(),
            "RUNS_ROOT": (tmp_path / "runs" / "muse_task7").as_posix(),
            "GPU": "3",
            "PYTHON": "/opt/conda/envs/cvs/bin/python",
        }
    )
    return subprocess.run(
        ["bash", LAUNCHER.relative_to(ROOT).as_posix(), "--dry-run", f"--only={only}"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_launcher_freezes_protocol_and_all_required_evaluations():
    text = LAUNCHER.read_text(encoding="utf-8")
    for token in (
        "--labeled_ratio 0.07",
        "--unlabeled_ratio 0.63",
        "--source_cal_ratio 0.15",
        "--source_select_ratio 0.15",
        "--checkpoint_selection final_only",
        "--epochs 200",
    ):
        assert token in text
    for scenario in ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"):
        assert scenario in text
    assert "ARTIFACTS_COMPLETE" in text
    assert "EVAL_FAILED_${scenario^^}" in text


def test_m3_dry_run_prints_one_training_and_four_real_evaluation_commands_without_outputs(tmp_path):
    result = _dry_run(tmp_path, "M3")

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[MUSE-TRAIN-CMD]") == 1
    assert result.stdout.count("[MUSE-EVAL-CMD]") == 4
    assert "--muse_level M3" in result.stdout
    assert "--base_candidate ADV3B02_CORE90_SOFT_E200" in result.stdout
    assert "eval_ssdg_sat_per_rx.py" in result.stdout
    for scenario in ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"):
        assert f"scenario={scenario}" in result.stdout
        assert f"metrics_{scenario}.json" in result.stdout
        assert f"eval_{scenario}.log" in result.stdout
    assert not (tmp_path / "runs").exists()


def test_dry_run_maps_m0_to_control_and_m1_m2_m3_to_progressive_muse_levels(tmp_path):
    result = _dry_run(tmp_path, "M0,M1,M2,M3")

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[MUSE-TRAIN-CMD]") == 4
    assert result.stdout.count("[MUSE-EVAL-CMD]") == 16
    for level in ("M0", "M1", "M2", "M3"):
        assert f"candidate={level}" in result.stdout
        assert f"--muse_level {level}" in result.stdout
    assert "candidate=M0 capabilities=ADV3B02_CONTROL" in result.stdout
    assert "candidate=M1 capabilities=BASE" in result.stdout
    assert "candidate=M2 capabilities=BASE_FUSION_HML" in result.stdout
    assert "candidate=M3 capabilities=BASE_FUSION_HML_SATELLITE_CROSSRX_PROTO" in result.stdout


def test_only_rejects_unknown_candidate_without_creating_outputs(tmp_path):
    result = _dry_run(tmp_path, "M4")

    assert result.returncode == 2
    assert "unknown candidate" in result.stderr.lower()
    assert not (tmp_path / "runs").exists()
