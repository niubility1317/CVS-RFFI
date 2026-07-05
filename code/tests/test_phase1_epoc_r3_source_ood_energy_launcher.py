import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dry_run(*extra: str) -> str:
    result = subprocess.run(
        [
            "bash",
            "code/scripts/launch_phase1_epoc_r3_source_ood_energy_20260706.sh",
            "--dry-run",
            *extra,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_epoc_r3_launcher_declares_source_only_energy_vos_route():
    out = _dry_run("--only=EPOC_R3_ENERGY_VOS_GUARD")

    assert "EPOC_R3_ENERGY_VOS_GUARD" in out
    assert "route=source_only_energy_vos_geometry_repair" in out
    assert "base=ADV3B02_CORE90_SOFT_E200" in out
    assert "real_unknown_classes_in_training=0" in out
    assert "target_receiver_samples_in_training=0" in out
    assert "manytx_in_training=0" in out
    assert "ManySig.pkl" in out
    assert "ManyTx.pkl" not in out
    assert "--new_wisig_pkl" not in out
    assert "target_unknown" not in out
    assert "--teacher_ckpt" in out
    assert "--lambda_teacher_clean_kl" in out
    assert "--lambda_teacher_sat_kl" in out
    assert "--lambda_teacher_zid_mse" in out
    assert "--lambda_energy_in 0.002" in out
    assert "--lambda_energy_out 0.004" in out
    assert "--proxy_unknown_virtual_mode legacy_hard" in out
    assert "--proxy_unknown_energy_margin_quantile_weight 0.12" in out
    assert "--proxy_unknown_low_density_accept_weight 0.12" in out
    assert "--proxy_unknown_radius_inter_ratio_weight 0.14" in out
    assert "--soft_unknown_mixup_order 5" in out
    assert "--ow_feat_tail_mode robust_3sigma" in out
    assert "--phase2_export_prototypes true" in out
    assert "--phase2_fuse_accept_policy local_component" in out
    assert "--best_metric joint_safe" in out
    assert "--enable_joint_safe_guard true" in out
    assert "leo_clear_weak" in out
    assert "leo_low_elev_weak" in out
    assert "leo_rain_weak" in out


def test_epoc_r3_launcher_exposes_two_distinct_candidates_on_free_gpus():
    out = _dry_run()

    assert "candidates=2" in out
    assert "EPOC_R3_ENERGY_VOS_GUARD" in out
    assert "EPOC_R3_TIGHT_CORE_MARGIN" in out
    assert out.count("[EPOC-R3-CANDIDATE]") == 2
    assert "CUDA_VISIBLE_DEVICES=2" in out
    assert "CUDA_VISIBLE_DEVICES=3" in out
