import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dry_run(*extra: str) -> str:
    result = subprocess.run(
        [
            "bash",
            "code/scripts/launch_phase1_epoc_r4_teacher_tailq_20260706.sh",
            "--dry-run",
            *extra,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_epoc_r4_launcher_declares_source_only_teacher_tailq_route():
    out = _dry_run("--only=EPOC_R4_TEACHER_LOCK_TAILQ")

    assert "EPOC_R4_TEACHER_LOCK_TAILQ" in out
    assert "route=source_only_teacher_locked_tail_quarantine" in out
    assert "base=ADV3B02_CORE90_SOFT_E200" in out
    assert "real_unknown_classes_in_training=0" in out
    assert "target_receiver_samples_in_training=0" in out
    assert "manytx_in_training=0" in out
    assert "ManySig.pkl" in out
    assert "ManyTx.pkl" not in out
    assert "--new_wisig_pkl" not in out
    assert "target_unknown" not in out
    assert "--teacher_ckpt" in out
    assert "--lambda_teacher_clean_kl 1.10" in out
    assert "--lambda_teacher_sat_kl 0.42" in out
    assert "--lambda_teacher_zid_mse 0.160" in out
    assert "--proxy_unknown_tail_quarantine_weight 0.14" in out
    assert "--proxy_unknown_source_safe_weight 0.08" in out
    assert "--proxy_unknown_energy_margin_quantile_weight 0.18" in out
    assert "--proxy_unknown_radius_inter_ratio_target 0.14" in out
    assert "--soft_unknown_mixup_order 6" in out
    assert "--ow_feat_radius_deg 14" in out
    assert "--phase2_fuse_radius_cap_deg 16" in out
    assert "--phase2_fuse_tail_auto_accept false" in out
    assert "--best_metric joint_safe" in out
    assert "--enable_joint_safe_guard true" in out
    assert "leo_clear_weak" in out
    assert "leo_low_elev_weak" in out
    assert "leo_rain_weak" in out


def test_epoc_r4_launcher_exposes_two_distinct_candidates_on_low_memory_gpus():
    out = _dry_run()

    assert "candidates=2" in out
    assert "EPOC_R4_TEACHER_LOCK_TAILQ" in out
    assert "EPOC_R4_SOURCE_OUTWARD_SHELL" in out
    assert out.count("[EPOC-R4-CANDIDATE]") == 2
    assert "CUDA_VISIBLE_DEVICES=4" in out
    assert "CUDA_VISIBLE_DEVICES=5" in out
