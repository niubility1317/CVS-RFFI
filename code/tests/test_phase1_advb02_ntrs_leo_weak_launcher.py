import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "code/scripts/launch_phase1_advb02_ntrs_leo_weak_20260820.sh"
BASH = r"C:\Program Files\Git\bin\bash.exe"


def _dry_run() -> str:
    result = subprocess.run(
        [BASH, SCRIPT, "--dry-run"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.replace("\\", "")


def test_ntrs_launcher_inherits_core90_and_the_confirmed_phase1_roles():
    out = _dry_run()
    assert "ADVB02_NTRS_LEO_WEAK_E200" in out
    assert "seed=392034" in out
    assert "ratios=0.07/0.63/0.15/0.15" in out
    assert "roles=L_s/U_s/V_cal/V_select" in out
    assert "mixed_orbit" not in out
    assert "--phase1_source_role_protocol l_s_u_s_v_cal_v_select" in out
    assert "--labeled_ratio 0.07" in out
    assert "--unlabeled_ratio 0.63" in out
    assert "--source_cal_ratio 0.15" in out
    assert "--source_select_ratio 0.15" in out
    assert "--model_variant lite_d" in out
    assert "--branch_ablation no_dac" in out
    assert "--domain_enhancer rcn_stats" in out
    assert "--epochs 200" in out
    assert "--label_epochs 130" in out
    assert "--sat_training_mode concat_masked" in out
    assert "--sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in out
    assert "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in out


def test_ntrs_launcher_freezes_the_full_first_version_and_independent_test():
    out = _dry_run()
    assert "--use_ntrs" in out
    assert "--use_crra" not in out
    assert "--ntrs_rank 8" in out
    assert "--ntrs_alpha_max 0.20" in out
    assert "--ntrs_slow_ema_decay 0.95" in out
    assert "--ntrs_support_tau 1.0" in out
    assert "--ntrs_energy_threshold 0.10" in out
    assert "--ntrs_unknown_rescue false" in out
    assert "--ntrs_target_adapter false" in out
    assert "--lambda_sat_cons 0.0" in out
    assert "--lambda_ntrs_sat_kl 0.01" in out
    assert "--lambda_ntrs_margin 0.03" in out
    assert "--lambda_ntrs_relation 0.02" in out
    assert "--lambda_ntrs_cond_decorr 0.01" in out
    assert "--lambda_ntrs_min_correction 0.001" in out
    assert "--lambda_ntrs_subspace 0.02" in out
    assert "--lambda_ntrs_correctability 0.02" in out
    assert "--eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in out
    assert "--eval_ntrs_telemetry" in out
    assert "final_ssdg.pth" in out


def test_ntrs_launcher_rejects_non_source_dataset_paths():
    result = subprocess.run(
        [BASH, SCRIPT, "--dry-run"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "WISIG_PKL": "/tmp/ManyTx.pkl"},
    )
    assert result.returncode == 4
    assert "refusing non-source" in result.stderr
