import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "code/scripts/launch_phase1_advb02_crra_leo_weak_20260819.sh"


def _dry_run() -> str:
    result = subprocess.run(
        ["bash", SCRIPT, "--dry-run"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    # The launcher deliberately prints shell-escaped commands; test the
    # argument contract rather than its display escaping.
    return result.stdout.replace("\\", "")


def test_crra_leo_launcher_freezes_confirmed_protocol_and_core90_schedule():
    out = _dry_run()

    assert "ADVB02_CRRA_S_LEO_WEAK_E200" in out
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
    assert "--lr 0.0002" in out
    assert "--weight_decay 0.0001" in out
    assert "--sat_training_mode concat_masked" in out
    assert "--sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in out
    assert "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in out


def test_crra_leo_launcher_has_one_kl_weight_crra_schedule_and_final_independent_test():
    out = _dry_run()

    assert "--use_crra" in out
    assert "--crra_scenario leo_weak" in out
    assert "--crra_rank 8" in out
    assert "--crra_alpha_max 0.25" in out
    assert "--crra_start_epoch 17" in out
    assert "--crra_ramp_epochs 30" in out
    assert "--crra_s3_lr_scale 0.25" in out
    assert "--lambda_sat_cons 0.05" in out
    assert "--lambda_crra_sat_kl 0.0" in out
    assert "--lambda_crra_pair 0.05" in out
    assert "--lambda_crra_energy 0.001" in out
    assert "--lambda_crra_gate_l1 0.001" in out
    assert "--lambda_crra_nuisance 0.02" in out
    assert "--lambda_crra_condition_tx_adv 0.02" in out
    assert "--lambda_crra_sat_shell 0.0" in out
    assert "--eval_sat_channel" in out
    assert "--eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in out
    assert "--eval_crra_telemetry" in out
    assert "final_ssdg.pth" in out
