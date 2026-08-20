import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "code/scripts/launch_phase1_advb02_ntrs_leo_weak_20260820.sh"
BASH = r"C:\Program Files\Git\bin\bash.exe"


def _dry_run(profile: str = "full") -> str:
    result = subprocess.run(
        [BASH, SCRIPT, "--dry-run"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "NTRS_PROFILE": profile},
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


def test_ntrs_matrix_control_disables_only_ntrs():
    out = _dry_run("control")
    assert "candidate=ADVB02_CORE90_LEO_WEAK_CONTROL_E200" in out
    assert "profile=control" in out
    assert "--no_use_ntrs" in out
    assert "--use_ntrs" not in out
    assert "--sat_training_mode concat_masked" in out
    assert "--sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in out
    assert "--seed 392034" in out
    assert "mixed_orbit" not in out
    assert "--eval_ntrs_telemetry" not in out


def test_ntrs_matrix_identity_structure_ablation_is_group_local():
    out = _dry_run("no_identity_structure")
    assert "candidate=ADVB02_NTRS_NO_IDSTRUCT_LEO_WEAK_E200" in out
    assert "profile=no_identity_structure" in out
    assert "--use_ntrs" in out
    assert "--lambda_ntrs_sat_kl 0" in out
    assert "--lambda_ntrs_margin 0" in out
    assert "--lambda_ntrs_relation 0" in out
    assert "--lambda_ntrs_class_conditional 0" in out
    assert "--lambda_ntrs_receiver 0.02" in out
    assert "--lambda_ntrs_correctability 0.02" in out


def test_ntrs_matrix_nuisance_factorization_ablation_is_group_local():
    out = _dry_run("no_nuisance_factorization")
    assert "candidate=ADVB02_NTRS_NO_NUISANCE_LEO_WEAK_E200" in out
    assert "profile=no_nuisance_factorization" in out
    assert "--lambda_ntrs_receiver 0" in out
    assert "--lambda_ntrs_day 0" in out
    assert "--lambda_ntrs_channel 0" in out
    assert "--lambda_ntrs_context_tx_adv 0" in out
    assert "--lambda_ntrs_cond_decorr 0" in out
    assert "--lambda_ntrs_shared_rx 0" in out
    assert "--lambda_ntrs_margin 0.03" in out
    assert "--lambda_ntrs_correctability 0.02" in out


def test_ntrs_matrix_embed_residual_ablation_is_group_local():
    out = _dry_run("no_embed_residual")
    assert "candidate=ADVB02_NTRS_NO_EMBEDRES_LEO_WEAK_E200" in out
    assert "profile=no_embed_residual" in out
    assert "--ntrs_alpha_max 0" in out
    assert "--lambda_ntrs_min_correction 0" in out
    assert "--lambda_ntrs_alpha 0" in out
    assert "--lambda_ntrs_subspace 0" in out
    assert "--lambda_ntrs_margin 0.03" in out
    assert "--lambda_ntrs_receiver 0.02" in out


def test_ntrs_matrix_safety_loss_ablation_is_group_local():
    out = _dry_run("no_safety_losses")
    assert "candidate=ADVB02_NTRS_NO_SAFETY_LEO_WEAK_E200" in out
    assert "profile=no_safety_losses" in out
    assert "--lambda_ntrs_correctability 0" in out
    assert "--lambda_ntrs_score_stability 0" in out
    assert "--lambda_ntrs_class_attraction 0" in out
    assert "--lambda_ntrs_margin 0.03" in out
    assert "--lambda_ntrs_receiver 0.02" in out
    assert "--ntrs_alpha_max 0.20" in out


def test_ntrs_launcher_rejects_unknown_matrix_profile():
    result = subprocess.run(
        [BASH, SCRIPT, "--dry-run"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "NTRS_PROFILE": "not_a_profile"},
    )
    assert result.returncode == 2
    assert "unknown NTRS_PROFILE" in result.stderr


def test_recovery_launcher_rejects_seed_override():
    result = subprocess.run(
        [BASH, SCRIPT, "--dry-run"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "NTRS_PROFILE": "v2_min_shared_head", "SEED": "999"},
    )
    assert result.returncode == 4
    assert "seed must be 392034" in result.stderr


def test_recovery_matrix_profiles_freeze_bypass_lr_and_shared_head_controls():
    d1 = _dry_run("v2_identity_bypass")
    assert "candidate=ADVB02_NTRS_V2_D1_BYPASS_E200" in d1
    assert "--ntrs_variant v2_min" in d1
    assert "--ntrs_identity_bypass true" in d1
    assert "--ntrs_core_lr_mode baseline" in d1

    d2 = _dry_run("v2_identity_bypass_v1_lr")
    assert "candidate=ADVB02_NTRS_V2_D2_BYPASS_V1LR_E200" in d2
    assert "--ntrs_identity_bypass true" in d2
    assert "--ntrs_core_lr_mode v1" in d2

    d3 = _dry_run("v1_fair_core_lr")
    assert "candidate=ADVB02_NTRS_V1_D3_FAIRLR_E200" in d3
    assert "--ntrs_variant v1" in d3
    assert "--ntrs_identity_bypass false" in d3
    assert "--ntrs_core_lr_mode baseline" in d3

    v2 = _dry_run("v2_min_shared_head")
    assert "candidate=ADVB02_NTRS_V2_MIN_SHARED_E200" in v2
    assert "--ntrs_variant v2_min" in v2
    assert "--ntrs_identity_bypass false" in v2
    assert "--ntrs_core_lr_mode baseline" in v2
    assert "--lambda_ntrs_sat_kl 0.01" in v2
    assert "--lambda_ntrs_margin 0.03" in v2
    assert "--lambda_ntrs_min_correction 0.001" in v2
    assert "--lambda_ntrs_receiver 0" in v2
    assert "--lambda_ntrs_correctability 0" in v2


def test_adapter_only_matrix_profiles_freeze_checkpoint_q_losses_and_leo_weak_protocol():
    a0 = _dry_run("a0_control")
    assert "candidate=ADVB02_NTRS_A0_CONTROL_r1_E200" in a0
    assert "--from_scratch true" in a0
    assert "--no_use_ntrs" in a0

    bypass = _dry_run("a0_bypass")
    assert "candidate=ADVB02_NTRS_A0_BYPASS_r1_E200" in bypass
    assert "--ntrs_variant v2_min" in bypass
    assert "--ntrs_identity_bypass true" in bypass
    assert "--from_scratch true" in bypass

    random_q = _dry_run("a1_random_q")
    assert "candidate=ADVB02_NTRS_A1_RANDOM_Q_r1_E200" in random_q
    assert "--ntrs_variant v3_adapter" in random_q
    assert "--ntrs_q_trainable false" in random_q
    assert "--ntrs_adapter_only true" in random_q
    assert "--from_scratch false" in random_q
    assert "--ntrs_alpha_max 0.02" in random_q
    assert "--lambda_ntrs_robust_ce 1.0" in random_q
    assert "--lambda_ntrs_clean_zero 1.0" in random_q
    assert "--lambda_ntrs_sat_relative 0.10" in random_q

    trainable_q = _dry_run("a1_trainable_q")
    assert "candidate=ADVB02_NTRS_A1_TRAINABLE_Q_r1_E200" in trainable_q
    assert "--ntrs_q_trainable true" in trainable_q
    assert "--ntrs_use_support_gate false" in trainable_q
    assert "--use_ema_teacher false" in trainable_q

    a2 = _dry_run("a2_teacher_margin")
    assert "candidate=ADVB02_NTRS_A2_TEACHER_MARGIN_r1_E200" in a2
    assert "--ntrs_alpha_max 0.05" in a2
    assert "--lambda_ntrs_sat_kl 0.01" in a2
    assert "--lambda_ntrs_margin 0.03" in a2

    a3 = _dry_run("a3_support_gate")
    assert "--ntrs_use_support_gate true" in a3
    a4 = _dry_run("a4_joint_core")
    assert "--ntrs_adapter_only false" in a4
    assert "--ntrs_core_lr_mode adapter_joint" in a4
    assert "--ntrs_core_lr_ratio 0.02" in a4
    assert "--lambda_teacher_clean_kl 1.0" in a4
    assert "--lambda_teacher_zid_mse 1.0" in a4

    for output in (a0, bypass, random_q, trainable_q, a2, a3, a4):
        assert "--seed 392034" in output
        assert "--labeled_ratio 0.07" in output
        assert "--unlabeled_ratio 0.63" in output
        assert "--source_cal_ratio 0.15" in output
        assert "--source_select_ratio 0.15" in output
        assert "mixed_orbit" not in output
        assert "--eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in output


def test_v4_counterfactual_operator_matrix_profiles_are_orthogonal_and_leo_weak_only():
    expected = {
        "b0_constant": ("B0_CONSTANT", "--ntrs_context_mode constant", "--ntrs_operator_mode operator"),
        "b0_shuffled": ("B0_SHUFFLED", "--ntrs_context_mode shuffled", "--ntrs_operator_mode operator"),
        "b0_random_feature": ("B0_RANDOM_FEATURE", "--ntrs_context_mode normalized", "--ntrs_q_trainable false"),
        "b1_metadata": ("B1_METADATA", "--ntrs_context_mode metadata_teacher", "--lambda_ntrs_q_distill 1.0"),
        "b1_normalized": ("B1_NORMALIZED", "--ntrs_context_mode normalized", "--lambda_ntrs_q_distill 0"),
        "b2_additive": ("B2_ADDITIVE", "--ntrs_operator_mode pca_additive", "--ntrs_q_trainable false"),
        "b2_operator": ("B2_OPERATOR", "--ntrs_operator_mode operator", "--lambda_ntrs_pair_shift 1.0"),
        "b3_risk": ("B3_RISK", "--lambda_ntrs_harm 2.0", "--lambda_ntrs_rescue 1.0"),
    }
    for profile, needles in expected.items():
        out = _dry_run(profile)
        assert "--ntrs_variant v4_operator" in out
        assert "--ntrs_adapter_only true" in out
        assert "--from_scratch false" in out
        assert "--seed 392034" in out
        assert "mixed_orbit" not in out
        assert "--sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in out
        assert "--eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in out
        for needle in needles:
            assert needle in out
    assert "--ntrs_pca_artifact" in _dry_run("b2_additive")
