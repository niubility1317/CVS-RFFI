import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dry_run(*extra: str) -> str:
    result = subprocess.run(
        [
            "bash",
            "code/scripts/launch_phase1_epoc_r6_reciprocal_shell_distill_20260706.sh",
            "--dry-run",
            *extra,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_epoc_r6_launcher_declares_source_only_reciprocal_shell_distill_route():
    out = _dry_run("--only=EPOC_R6_RECIPROCAL_SHELL_KD")

    assert "EPOC_R6_RECIPROCAL_SHELL_KD" in out
    assert "route=source_only_adv3b02_reciprocal_shell_distill" in out
    assert "base=ADV3B02_CORE90_SOFT_E200" in out
    assert "real_unknown_classes_in_training=0" in out
    assert "target_receiver_samples_in_training=0" in out
    assert "target_unknown_training_count=0" in out
    assert "manytx_in_training=0" in out
    assert "ManySig.pkl" in out
    assert "ManyTx.pkl" not in out
    assert "--new_wisig_pkl" not in out
    assert "--target_unknown" not in out
    assert "--teacher_ckpt" in out
    assert "--proxy_unknown_virtual_mode legacy_hard" in out
    assert "--proxy_unknown_shell_outward_accept_weight" in out
    assert "--proxy_unknown_radius_inter_ratio_weight" in out
    assert "--proxy_unknown_bridge_accept_target 0.00" in out
    assert "--phase2_fuse_tail_auto_accept false" in out
    assert "--phase2_fuse_global_ball_accept false" in out
    assert "--best_metric joint_safe" in out
    assert "leo_clear_weak" in out
    assert "leo_low_elev_weak" in out
    assert "leo_rain_weak" in out


def test_epoc_r6_launcher_uses_low_vram_gpu0_and_gpu1_candidates():
    out = _dry_run()

    assert "candidates=2" in out
    assert "EPOC_R6_RECIPROCAL_SHELL_KD" in out
    assert "EPOC_R6_KNOWN_FLOOR_SHELL_KD" in out
    assert out.count("[EPOC-R6-CANDIDATE]") == 2
    assert "CUDA_VISIBLE_DEVICES=0" in out
    assert "CUDA_VISIBLE_DEVICES=1" in out
