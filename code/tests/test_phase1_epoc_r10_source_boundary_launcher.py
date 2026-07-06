import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dry_run(*extra: str) -> str:
    result = subprocess.run(
        [
            "bash",
            "code/scripts/launch_phase1_epoc_r10_source_boundary_20260706.sh",
            "--dry-run",
            *extra,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_r10_launcher_declares_source_only_boundary_route_without_real_unknowns():
    out = _dry_run("--only=EPOC_R10_BOUNDARY_NOPROXY")

    assert "EPOC_R10_BOUNDARY_NOPROXY" in out
    assert "algorithm=ADV3B02_SOURCE_BOUNDARY" in out
    assert "route=source_only_teacher_boundary_compact_vos" in out
    assert "base=ADV3B02_CORE90_SOFT_E200" in out
    assert "teacher_ckpt=ADV3B02_CORE90_SOFT_E200" in out
    assert "phase1_dataset=ManySig_only" in out
    assert "real_unknown_classes_in_training=0" in out
    assert "target_receiver_samples_in_training=0" in out
    assert "target_unknown_training_count=0" in out
    assert "manytx_in_training=0" in out
    assert "proxy_unknown_real_tx_calibration=0" in out
    assert "virtual_unknown_only=1" in out
    assert "stage2_unknown_query_eval_only=1" in out
    assert "stage2_success_claim=0" in out
    assert "deployment_success_claim=0" in out
    assert "qknn8_same_row_eval_required=1" in out
    assert "ManySig.pkl" in out
    assert "ManyTx.pkl" not in out
    assert "--new_wisig_pkl" not in out
    assert "--target_unknown" not in out
    assert "--proxy_unknown_tx_ids" not in out
    assert "--lambda_teacher_clean_kl 2.30" in out
    assert "--lambda_teacher_sat_kl 0.92" in out
    assert "--lambda_teacher_zid_mse 0.460" in out
    assert "--lambda_zid_compact 0.064" in out
    assert "--lambda_source_episode 0.0120" in out
    assert "--lambda_proxy_unknown 0.0000" in out
    assert "--lambda_soft_unknown_mixup 0.00000" in out
    assert "--source_episode_radius_cap_deg 18" in out
    assert "--phase2_fuse_radius_cap_deg 11" in out


def test_r10_launcher_exposes_two_candidates_on_idle_low_memory_gpus():
    out = _dry_run()

    assert "candidates=2" in out
    assert "EPOC_R10_BOUNDARY_NOPROXY" in out
    assert "EPOC_R10_GENTLE_VOS_LATE" in out
    assert out.count("[EPOC-R10-CANDIDATE]") == 2
    assert "CUDA_VISIBLE_DEVICES=4" in out
    assert "CUDA_VISIBLE_DEVICES=5" in out
    assert "--lambda_proxy_unknown 0.0010" in out
    assert "--lambda_soft_unknown_mixup 0.00001" in out
    assert "--proxy_unknown_start_epoch 120" in out
    assert "--proxy_unknown_virtual_count 16" in out
    assert "--phase2_fuse_radius_cap_deg 12" in out


def test_r10_launcher_rejects_non_source_phase1_training_inputs():
    forbidden = [
        "/tmp/ManyTx.pkl",
        "/tmp/ManyRx.pkl",
        "/tmp/SingleDay.pkl",
        "/tmp/new_wisig.pkl",
        "/tmp/target.pkl",
        "/tmp/unknown.pkl",
    ]

    for pkl_path in forbidden:
        result = subprocess.run(
            [
                "bash",
                "-lc",
                (
                    f"WISIG_PKL={pkl_path} "
                    "bash code/scripts/launch_phase1_epoc_r10_source_boundary_20260706.sh "
                    "--dry-run --only=EPOC_R10_BOUNDARY_NOPROXY"
                ),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode != 0
        assert "refusing non-source Phase1 WISIG_PKL" in result.stderr
