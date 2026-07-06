import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dry_run(*extra: str) -> str:
    result = subprocess.run(
        [
            "bash",
            "code/scripts/launch_phase1_epoc_r9_source_anchor_20260706.sh",
            "--dry-run",
            *extra,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_r9_launcher_declares_source_only_anchor_route_without_real_unknowns():
    out = _dry_run("--only=EPOC_R9_ANCHOR_NOPROXY")

    assert "EPOC_R9_ANCHOR_NOPROXY" in out
    assert "algorithm=ADV3B02_SOURCE_ANCHOR" in out
    assert "route=source_only_teacher_anchor_stable_feature_repair" in out
    assert "base=ADV3B02_CORE90_SOFT_E200" in out
    assert "teacher_ckpt=ADV3B02_CORE90_SOFT_E200" in out
    assert "phase1_dataset=ManySig_only" in out
    assert "real_unknown_classes_in_training=0" in out
    assert "target_receiver_samples_in_training=0" in out
    assert "target_unknown_training_count=0" in out
    assert "manytx_in_training=0" in out
    assert "proxy_unknown_real_tx_calibration=0" in out
    assert "virtual_unknown_only=1" in out
    assert "threshold_selection_label_scope=support_or_source_old_only" in out
    assert "stage2_unknown_query_eval_only=1" in out
    assert "stage2_success_claim=0" in out
    assert "deployment_success_claim=0" in out
    assert "qknn8_same_row_eval_required=1" in out
    assert "ManySig.pkl" in out
    assert "ManyTx.pkl" not in out
    assert "--new_wisig_pkl" not in out
    assert "--target_unknown" not in out
    assert "--proxy_unknown_tx_ids" not in out
    assert "--teacher_ckpt" in out
    assert "--lambda_teacher_clean_kl 2.20" in out
    assert "--lambda_teacher_sat_kl 0.90" in out
    assert "--lambda_teacher_zid_mse 0.420" in out
    assert "--lambda_proxy_unknown 0.0000" in out
    assert "--lambda_soft_unknown_mixup 0.00000" in out
    assert "--proxy_unknown_start_epoch 120" in out
    assert "--lambda_source_episode 0.0100" in out
    assert "--phase2_export_prototypes true" in out
    assert "--phase2_fuse_radius_cap_deg 13" in out
    assert "leo_clear_weak" in out
    assert "leo_low_elev_weak" in out
    assert "leo_rain_weak" in out


def test_r9_launcher_exposes_two_candidates_on_idle_low_memory_gpus():
    out = _dry_run()

    assert "candidates=2" in out
    assert "EPOC_R9_ANCHOR_NOPROXY" in out
    assert "EPOC_R9_GENTLE_VIRTUAL_LATE" in out
    assert out.count("[EPOC-R9-CANDIDATE]") == 2
    assert "CUDA_VISIBLE_DEVICES=2" in out
    assert "CUDA_VISIBLE_DEVICES=3" in out
    assert "--lambda_proxy_unknown 0.0020" in out
    assert "--lambda_soft_unknown_mixup 0.00003" in out
    assert "--proxy_unknown_start_epoch 90" in out
    assert "--proxy_unknown_virtual_count 24" in out
    assert "--phase2_fuse_radius_cap_deg 14" in out


def test_r9_launcher_rejects_non_source_phase1_training_inputs():
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
                    "bash code/scripts/launch_phase1_epoc_r9_source_anchor_20260706.sh "
                    "--dry-run --only=EPOC_R9_ANCHOR_NOPROXY"
                ),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode != 0
        assert "refusing non-source Phase1 WISIG_PKL" in result.stderr
