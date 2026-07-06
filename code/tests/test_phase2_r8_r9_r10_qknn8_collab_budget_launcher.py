import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dry_run(*extra: str) -> str:
    result = subprocess.run(
        [
            "bash",
            "code/scripts/launch_phase2_r8_r9_r10_qknn8_collab_budget_20260706.sh",
            "--dry-run",
            *extra,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_budget_launcher_uses_existing_features_and_budget_policy():
    out = _dry_run("--only=R10_GENTLE")

    assert "source_run_id=phase2_r8_r9_r10_qknn8_collab_20260706" in out
    assert "collab_counts=all" in out
    assert "collab_group_policy=available_up_to_k" in out
    assert "partial_collab_min_receivers=1" in out
    assert "actual_receiver_count_histogram" in out
    assert "protocol=Stage2-C" in out
    assert "unknown_query_eval_only=true" in out
    assert "proxy_unknown_real_tx_calibration=0" in out
    assert "stage2_success_claim=0" in out
    assert "deployment_success_claim=0" in out
    assert "--collab_counts all" in out
    assert "--collab_group_policy available_up_to_k" in out
    assert "--event_alignment_policy receiver_domain_ranked" in out
    assert "features_stage2c_leo_multirx.npz" in out
    assert "qknn8_collab_budget.json" in out
    assert "qknn8_collab_budget_evidence.csv" in out
    assert "CUDA_VISIBLE_DEVICES=5" in out
    assert "export_spaceborne_features.py" not in out


def test_budget_launcher_exposes_all_six_cases():
    out = _dry_run()

    for case, gpu in {
        "R8_RADIUS": 0,
        "R8_SHELL": 1,
        "R9_ANCHOR": 2,
        "R9_GENTLE": 3,
        "R10_BOUNDARY": 4,
        "R10_GENTLE": 5,
    }.items():
        assert f"case={case}" in out
        assert f"CUDA_VISIBLE_DEVICES={gpu}" in out
        assert f"runs/phase2_r8_r9_r10_qknn8_collab_20260706/{case}/features_stage2c_leo_multirx.npz" in out
        assert f"runs/phase2_r8_r9_r10_qknn8_collab_budget_20260706/{case}/qknn8_collab_budget.json" in out
    assert out.count("[R8R9R10-QKNN8-COLLAB-BUDGET-CASE]") == 6


def test_budget_launcher_rejects_unknown_only_case():
    result = subprocess.run(
        [
            "bash",
            "code/scripts/launch_phase2_r8_r9_r10_qknn8_collab_budget_20260706.sh",
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
