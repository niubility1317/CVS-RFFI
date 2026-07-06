import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dry_run(*extra: str) -> str:
    result = subprocess.run(
        [
            "bash",
            "code/scripts/launch_phase2_r8_qknn8_riskgate_20260706.sh",
            "--dry-run",
            *extra,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_r8_qknn8_riskgate_defaults_to_unknown_eval_only_without_real_proxy_unknown():
    out = _dry_run("--only=RADIUS")

    assert "qknn_k=8" in out
    assert "k_shot=8" in out
    assert "unknown_query_eval_only=true" in out
    assert "ground_training_unknown_seen=false" in out
    assert "proxy_unknown_real_tx_calibration=0" in out
    assert "stage2_success_claim=0" in out
    assert "deployment_success_claim=0" in out
    assert "--collab_counts all" in out
    assert "--unknown_gate_mode support_envelope_full" in out
    assert "--fusion_policy old_protected_unknown_confirm_cvs" in out
    assert "--virtual_unknown_calibration_enabled" in out
    assert "--class_negative_risk_enabled" in out
    assert "--class_shell_unknown_risk_enabled" in out
    assert "--proxy_unknown_tx_ids" not in out

