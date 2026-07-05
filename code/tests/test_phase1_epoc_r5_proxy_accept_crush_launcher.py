import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dry_run(*extra: str) -> str:
    result = subprocess.run(
        [
            "bash",
            "code/scripts/launch_phase1_epoc_r5_proxy_accept_crush_20260706.sh",
            "--dry-run",
            *extra,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_epoc_r5_launcher_declares_source_only_proxy_accept_crush_route():
    out = _dry_run("--only=EPOC_R5_BRIDGE_CRUSH")

    assert "EPOC_R5_BRIDGE_CRUSH" in out
    assert "route=source_only_teacher_locked_proxy_accept_crush" in out
    assert "base=ADV3B02_CORE90_SOFT_E200" in out
    assert "real_unknown_classes_in_training=0" in out
    assert "target_receiver_samples_in_training=0" in out
    assert "manytx_in_training=0" in out
    assert "ManySig.pkl" in out
    assert "ManyTx.pkl" not in out
    assert "--new_wisig_pkl" not in out
    assert "target_unknown" not in out
    assert "--teacher_ckpt" in out
    assert "--proxy_unknown_tail_quarantine_weight 0.40" in out
    assert "--proxy_unknown_bridge_accept_weight 0.28" in out
    assert "--proxy_unknown_shell_outward_accept_weight 0.30" in out
    assert "--proxy_unknown_energy_margin_quantile_weight 0.45" in out
    assert "--proxy_unknown_radius_inter_ratio_weight 0.36" in out
    assert "--proxy_unknown_bridge_accept_target 0.01" in out
    assert "--proxy_unknown_tail_accept_target 0.05" in out
    assert "--proxy_unknown_radius_inter_ratio_target 0.08" in out
    assert "--lambda_energy_in" not in out
    assert "--lambda_energy_out" not in out
    assert "--best_metric joint_safe" in out
    assert "--enable_joint_safe_guard true" in out
    assert "leo_clear_weak" in out
    assert "leo_low_elev_weak" in out
    assert "leo_rain_weak" in out


def test_epoc_r5_launcher_uses_idle_gpu6_and_gpu7_candidates():
    out = _dry_run()

    assert "candidates=2" in out
    assert "EPOC_R5_BRIDGE_CRUSH" in out
    assert "EPOC_R5_CORE_SHELL_REJECT" in out
    assert out.count("[EPOC-R5-CANDIDATE]") == 2
    assert "CUDA_VISIBLE_DEVICES=6" in out
    assert "CUDA_VISIBLE_DEVICES=7" in out
