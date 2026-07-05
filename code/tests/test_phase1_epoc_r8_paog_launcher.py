import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dry_run(*extra: str) -> str:
    result = subprocess.run(
        [
            "bash",
            "code/scripts/launch_phase1_epoc_r8_paog_20260706.sh",
            "--dry-run",
            *extra,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_epoc_r8_launcher_declares_source_only_paog_route():
    out = _dry_run("--only=EPOC_R8_PAOG_RADIUS_ENERGY")

    assert "EPOC_R8_PAOG_RADIUS_ENERGY" in out
    assert "route=source_only_adv3b02_paog" in out
    assert "algorithm=ADV3B02_PAOG" in out
    assert "base=ADV3B02_CORE90_SOFT_E200" in out
    assert "real_unknown_classes_in_training=0" in out
    assert "target_receiver_samples_in_training=0" in out
    assert "target_unknown_training_count=0" in out
    assert "stage2_unknown_query_eval_only=1" in out
    assert "stage2_success_claim=0" in out
    assert "deployment_success_claim=0" in out
    assert "qknn8_same_row_eval_required=1" in out
    assert "manytx_allowed_only_in_stage2_eval=1" in out
    assert "manytx_in_training=0" in out
    assert "ManySig.pkl" in out
    assert "ManyTx.pkl" not in out
    assert "--new_wisig_pkl" not in out
    assert "--target_unknown" not in out
    assert "--teacher_ckpt" in out
    assert "--lambda_teacher_clean_kl 2.00" in out
    assert "--lambda_teacher_sat_kl 0.75" in out
    assert "--lambda_teacher_zid_mse 0.380" in out
    assert "--lambda_zid_compact 0.052" in out
    assert "--lambda_proxy_unknown 0.0080" in out
    assert "--proxy_unknown_virtual_count 64" in out
    assert "--proxy_unknown_energy_margin 1.65" in out
    assert "--proxy_unknown_bridge_accept_target 0.00" in out
    assert "--proxy_unknown_tail_accept_target 0.02" in out
    assert "--proxy_unknown_radius_inter_ratio_target 0.06" in out
    assert "--lambda_source_episode 0.0090" in out
    assert "--source_episode_radius_mode min_three_sigma_core" in out
    assert "--phase2_fuse_radius_cap_deg 14" in out
    assert "--phase2_export_prototypes true" in out
    assert "--phase2_fuse_tail_auto_accept false" in out
    assert "--phase2_fuse_global_ball_accept false" in out
    assert "--best_metric joint_safe" in out
    assert "--enable_joint_safe_guard true" in out
    assert "leo_clear_weak" in out
    assert "leo_low_elev_weak" in out
    assert "leo_rain_weak" in out


def test_epoc_r8_launcher_exposes_two_paog_candidates_on_low_memory_gpus():
    out = _dry_run()

    assert "candidates=2" in out
    assert "EPOC_R8_PAOG_RADIUS_ENERGY" in out
    assert "EPOC_R8_PAOG_SHELL_BALANCED" in out
    assert out.count("[EPOC-R8-CANDIDATE]") == 2
    assert "CUDA_VISIBLE_DEVICES=0" in out
    assert "CUDA_VISIBLE_DEVICES=1" in out


def test_epoc_r8_launcher_rejects_manytx_as_phase1_training_input():
    result = subprocess.run(
        [
            "bash",
            "-lc",
            (
                "WISIG_PKL=/tmp/ManyTx.pkl "
                "bash code/scripts/launch_phase1_epoc_r8_paog_20260706.sh "
                "--dry-run --only=EPOC_R8_PAOG_RADIUS_ENERGY"
            ),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refusing non-source Phase1 WISIG_PKL" in result.stderr
