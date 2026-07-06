import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "R8_RADIUS": {
        "gpu": "0",
        "seed": "706801",
        "ckpt": "EPOC_R8_PAOG_RADIUS_ENERGY/best_joint_safe_ssdg.pth",
    },
    "R8_SHELL": {
        "gpu": "1",
        "seed": "706811",
        "ckpt": "EPOC_R8_PAOG_SHELL_BALANCED/best_joint_safe_ssdg.pth",
    },
    "R9_ANCHOR": {
        "gpu": "2",
        "seed": "706901",
        "ckpt": "EPOC_R9_ANCHOR_NOPROXY/best_joint_safe_ssdg.pth",
    },
    "R9_GENTLE": {
        "gpu": "3",
        "seed": "706911",
        "ckpt": "EPOC_R9_GENTLE_VIRTUAL_LATE/best_joint_safe_ssdg.pth",
    },
    "R10_BOUNDARY": {
        "gpu": "4",
        "seed": "7061001",
        "ckpt": "EPOC_R10_BOUNDARY_NOPROXY/best_joint_safe_ssdg.pth",
    },
    "R10_GENTLE": {
        "gpu": "5",
        "seed": "7061011",
        "ckpt": "EPOC_R10_GENTLE_VOS_LATE/best_joint_safe_ssdg.pth",
    },
}


def _dry_run(*extra: str) -> str:
    result = subprocess.run(
        [
            "bash",
            "code/scripts/launch_phase2_r8_r9_r10_qknn8_collab_20260706.sh",
            "--dry-run",
            *extra,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def _flat_cmd(out: str) -> str:
    return out.replace("\\,", ",")


def test_launcher_declares_stage2c_qknn8_all_receivers_unknown_eval_only():
    out = _dry_run("--only=R10_GENTLE")
    flat = _flat_cmd(out)

    assert "protocol=Stage2-C" in out
    assert "unknown_query_eval_only=true" in out
    assert "proxy_unknown_real_tx_calibration=0" in out
    assert "stage2_success_claim=0" in out
    assert "deployment_success_claim=0" in out
    assert "collab_counts=all" in out
    assert "qknn_k=8" in out
    assert "k_shot=8" in out
    assert "target_receivers=20-1,3-19,7-14,7-7,8-8" in out
    assert "leo_clear_weak" in out
    assert "leo_low_elev_weak" in out
    assert "leo_rain_weak" in out
    assert "--proxy_unknown_tx_ids" not in out
    assert "ManyTx.pkl" in out
    assert "--unknown_tx_ids" in out
    assert "--collab_counts all" in out
    assert "--collab_group_policy exact_k" in out
    assert "--partial_collab_min_receivers 1" in out
    assert "--query_per_class 20" in out
    assert "--qknn_k 8" in out
    assert "--support_selection_policy stable_first" in out
    assert "--event_alignment_policy receiver_domain_ranked" in out
    assert "--max_event_bytes 1152" in out
    assert "--max_event_latency_ms 20" in out
    assert "--evidence_packet_bytes 40" in out
    assert "--source_tx_ids 0,1,2,3,4,5" in flat
    assert "--source_rxs 0,1,2,3,4,5,6" in flat
    assert "--target_old_tx_ids 0,1,2,3,4,5" in flat
    assert "--target_old_rxs 20-1,3-19,7-14,7-7,8-8" in flat
    assert "--new_tx_ids 1-10,1-12,1-14,1-16,1-18,1-8,10-11,10-4" in flat
    assert "--new_rxs 20-1,3-19,7-14,7-7,8-8" in flat
    assert "--unknown_tx_ids 10-7,11-1,11-10,11-19,11-4,11-7,12-19,12-20" in flat
    assert "CUDA_VISIBLE_DEVICES=5" in out


def test_launcher_exposes_all_r8_r9_r10_cases_on_gpus_0_to_5():
    out = _dry_run()

    for case, expected in CASES.items():
        assert f"case={case}" in out
        assert f"CUDA_VISIBLE_DEVICES={expected['gpu']}" in out
        assert f"--seed {expected['seed']}" in out
        assert expected["ckpt"] in out
        assert f"runs/phase2_r8_r9_r10_qknn8_collab_20260706/{case}/features_stage2c_leo_multirx.npz" in out
        assert f"runs/phase2_r8_r9_r10_qknn8_collab_20260706/{case}/qknn8_collab_base.json" in out
        assert f"runs/phase2_r8_r9_r10_qknn8_collab_20260706/{case}/qknn8_collab_base_evidence.csv" in out
        assert f"logs/phase2_r8_r9_r10_qknn8_collab_20260706/{case}.out" in out
    assert out.count("[R8R9R10-QKNN8-COLLAB-CASE]") == 6


def test_launcher_rejects_unknown_only_case():
    result = subprocess.run(
        [
            "bash",
            "code/scripts/launch_phase2_r8_r9_r10_qknn8_collab_20260706.sh",
            "--dry-run",
            "--only=BAD_CASE",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--only must be all or one known case" in result.stderr
