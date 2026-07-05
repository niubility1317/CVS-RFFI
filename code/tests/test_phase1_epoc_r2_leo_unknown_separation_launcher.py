import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_epoc_r2_launcher_declares_source_only_leo_unknown_separation_route():
    result = subprocess.run(
        [
            "bash",
            "code/scripts/launch_phase1_epoc_r2_leo_unknown_separation_20260705.sh",
            "--dry-run",
            "--only=EPOC_R2_BALANCED_SEP",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "EPOC_R2_BALANCED_SEP" in out
    assert "route=source_only_leo_unknown_separation" in out
    assert "real_unknown_classes_in_training=0" in out
    assert "target_receiver_samples_in_training=0" in out
    assert "ManySig.pkl" in out
    assert "ManyTx.pkl" not in out
    assert "--new_wisig_pkl" not in out
    assert "target_unknown" not in out
    assert "--teacher_ckpt" in out
    assert "--lambda_teacher_clean_kl" in out
    assert "--lambda_teacher_sat_kl" in out
    assert "--lambda_teacher_zid_mse" in out
    assert "--lambda_source_episode" in out
    assert "--source_episode_radius_mode min_three_sigma_core" in out
    assert "--lambda_proxy_unknown" in out
    assert "--proxy_unknown_virtual_mode mixed" in out
    assert "--lambda_soft_unknown_mixup" in out
    assert "--soft_unknown_mixup_order 4" in out
    assert "--sat_train_scenarios" in out
    assert "leo_clear_weak" in out
    assert "leo_low_elev_weak" in out
    assert "leo_rain_weak" in out
    assert "--phase2_export_prototypes true" in out
    assert "--phase2_fuse_accept_policy local_component" in out
    assert "--best_metric joint_safe" in out
    assert "--enable_joint_safe_guard true" in out


def test_epoc_r2_launcher_exposes_old_floor_and_unknown_separation_candidates():
    result = subprocess.run(
        [
            "bash",
            "code/scripts/launch_phase1_epoc_r2_leo_unknown_separation_20260705.sh",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "candidates=4" in out
    assert "EPOC_R2_OLD_FLOOR" in out
    assert "EPOC_R2_BALANCED_SEP" in out
    assert "EPOC_R2_LEO_HARD_NEG" in out
    assert "EPOC_R2_SOFT_RING" in out
    assert out.count("[EPOC-R2-CANDIDATE]") == 4
