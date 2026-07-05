import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_epoc_b_ospr_qknn_collab_launcher_declares_full_stage2c_route():
    result = subprocess.run(
        ["bash", "code/scripts/launch_phase2_epoc_b_ospr_qknn_collab_20260705.sh", "--dry-run"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "EPOC_DISTILL_B_KDHI/best_joint_safe_ssdg.pth" in out
    assert "protocol=Stage2-C" in out
    assert "unknown_query_eval_only=true" in out
    assert "ground_training_unknown_seen=false" in out
    assert "event_alignment_policy=receiver_domain_ranked" in out
    assert "verdict_scope=NON_DEPLOYMENT_DIAGNOSTIC" in out
    assert "--target_old_tx_ids" in out
    assert "--new_tx_ids" in out
    assert "--unknown_tx_ids" in out
    assert "--proxy_unknown_tx_ids" in out
    assert "--target_old_channel_view" in out
    assert "--target_new_channel_view" in out
    assert "--proxy_unknown_channel_view" in out
    assert "leo_clear_weak" in out
    assert "leo_low_elev_weak" in out
    assert "leo_rain_weak" in out
    assert "--star_ground_channel_impl" in out
    assert "simplified_leo_residual" in out
    assert "--collab_counts all" in out
    assert "--qknn_k 8" in out
    assert "--k_shot 8" in out
    assert "--target_old_acc 0.99" in out
    assert "--target_seen_new_acc 0.97" in out
    assert "--target_unknown_reject 0.99" in out
    assert "--max_event_bytes 1152" in out
    assert "--max_event_latency_ms 20" in out
    assert "--event_alignment_policy receiver_domain_ranked" in out
