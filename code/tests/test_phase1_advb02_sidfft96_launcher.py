import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "code/scripts/launch_phase1_advb02_sidfft96_leo_weak_20260821.sh"
BASH = r"C:\Program Files\Git\bin\bash.exe"


def _run(*arguments: str, **environment: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [BASH, SCRIPT, *arguments],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, **{key: str(value) for key, value in environment.items()}},
    )


def test_launcher_dry_run_freezes_protocol_and_matrix():
    result = _run("--dry-run")
    assert result.returncode == 0, result.stderr
    output = result.stdout.replace("\\", "")

    assert "matrix=P0,S0,S1,S2,S3" in output
    assert "--labeled_ratio 0.07" in output
    assert "--unlabeled_ratio 0.63" in output
    assert "--source_cal_ratio 0.15" in output
    assert "--source_select_ratio 0.15" in output
    assert "--phase1_source_role_protocol l_s_u_s_v_cal_v_select" in output
    assert "--lambda_sat_cls 0.68" in output
    assert "--lambda_sat_cons 0.0" in output
    assert "--sat_cons_start_epoch 80" in output
    assert "leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in output
    assert "--sid_fft96_mode center" in output
    assert "--sid_fft96_mode phase" in output
    assert "--sid_fft96_mode sid" in output
    assert "mixed_orbit" not in output
    assert "--use_ntrs" not in output
    assert "--use_crra" not in output


def test_launcher_rejects_existing_output(tmp_path):
    (tmp_path / "S1_CENTER").mkdir()

    result = _run("--only=S1", RUNS_ROOT=tmp_path, DRY_RUN="0")

    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr


def test_launcher_rejects_non_source_dataset_path():
    result = _run("--dry-run", WISIG_PKL="/tmp/ManyTx.pkl")

    assert result.returncode == 4
    assert "refusing non-source" in result.stderr
