import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_epoc_r7_launcher_declares_source_only_floor_protected_route():
    result = subprocess.run(
        [
            "bash",
            "code/scripts/launch_phase1_epoc_r7_floor_protected_shell_20260706.sh",
            "--dry-run",
            "--only=EPOC_R7_FLOOR_LOCKED_SHELL",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "EPOC_R7_FLOOR_LOCKED_SHELL" in out
    assert "route=source_only_floor_protected_feature_shell" in out
    assert "old_floor_first=1" in out
    assert "real_unknown_classes_in_training=0" in out
    assert "target_receiver_samples_in_training=0" in out
    assert "target_unknown_training_count=0" in out
    assert "manytx_in_training=0" in out
    assert "source_heldout_proxy_unknown=1" in out
    assert "virtual_unknown_shell=1" in out
    assert "ManySig.pkl" in out
    assert "ManyTx.pkl" not in out
    assert "--new_wisig_pkl" not in out
    assert "target_unknown_training_count=0" in out
    assert "--target_unknown" not in out
    assert "--teacher_ckpt" in out
    assert "--lambda_teacher_clean_kl" in out
    assert "--lambda_teacher_sat_kl" in out
    assert "--lambda_teacher_zid_mse" in out
    assert "--lambda_zid_compact" in out
    assert "--lambda_source_episode" in out
    assert "--source_episode_radius_mode min_three_sigma_core" in out
    assert "--lambda_proxy_unknown" in out
    assert "--proxy_unknown_virtual_count 32" in out
    assert "--lambda_soft_unknown_mixup 0.00000" in out
    assert "--sat_train_scenarios" in out
    assert "leo_clear_weak" in out
    assert "leo_low_elev_weak" in out
    assert "leo_rain_weak" in out
    assert "--phase2_export_prototypes true" in out
    assert "--proxy_unknown_start_epoch 55" in out
    assert "--phase2_fuse_radius_cap_deg 14" in out
    assert "--best_metric joint_safe" in out
    assert "--enable_joint_safe_guard true" in out


def test_epoc_r7_launcher_exposes_two_floor_protected_candidates():
    result = subprocess.run(
        ["bash", "code/scripts/launch_phase1_epoc_r7_floor_protected_shell_20260706.sh", "--dry-run"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "candidates=2" in out
    assert "EPOC_R7_FLOOR_LOCKED_SHELL" in out
    assert "EPOC_R7_BALANCED_LOW_DENSITY" in out
    assert out.count("[EPOC-R7-CANDIDATE]") == 2
