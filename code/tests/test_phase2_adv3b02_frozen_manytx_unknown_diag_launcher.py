import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_adv3b02_frozen_manytx_unknown_diag_launcher_declares_protocol_route():
    result = subprocess.run(
        [
            "bash",
            "code/scripts/launch_phase2_adv3b02_frozen_manytx_unknown_diag_20260706.sh",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "base_model=ADV3B02_CORE90_SOFT_E200" in out
    assert "phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200" in out
    assert "source_tx_ids=14-10,14-7,20-15,20-19,6-15,8-20" in out
    assert "source_receivers=1-1,1-19,14-7,18-2,19-2,2-1,2-19" in out
    assert "target_receivers=20-1,3-19,7-14,7-7,8-8" in out
    assert "target_new_tx_ids=1-10,1-12,1-14,1-16,1-18,1-8,10-11,10-4" in out
    assert "unknown_tx_ids=10-7,11-1,11-10,11-19,11-4,11-7,12-19,12-20" in out
    assert "proxy_unknown_receivers=1-1,1-19,14-7,18-2,19-2,2-1" in out
    assert "protocol=Stage2-C" in out
    assert "unknown_query_eval_only=true" in out
    assert "ground_training_unknown_seen=false" in out
    assert "event_alignment_policy=receiver_domain_ranked" in out
    assert "verdict_scope=NON_DEPLOYMENT_DIAGNOSTIC" in out
    assert "resource_proxy=max_event_bytes=1152 max_event_latency_ms=20" in out
    assert "target_goals old_acc=0.99 min_old=0.95 seen_new_acc=0.97 min_seen=0.93 unknown_reject=0.99" in out
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
    assert "phase2_frozen_manytx_unknown_diagnostic.py" in out
    assert "--collab_counts all" in out
    assert "--collab_group_policy available_up_to_k" in out
    assert "--partial_collab_min_receivers 1" in out
    assert "--qknn_k 8" in out
    assert "--k_shot 8" in out
    assert "--max_event_bytes 1152" in out
    assert "--max_event_latency_ms 20" in out
