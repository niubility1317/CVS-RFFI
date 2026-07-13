from paper_reproduction.scripts.summarize_cvs_publication_stage2 import summarize_groups


def test_summary_keeps_joint_run_rows_and_reports_uncertainty() -> None:
    rows = [
        {"phase": "stage2b", "method": "x", "k_shot": 1, "accuracy": 0.4},
        {"phase": "stage2b", "method": "x", "k_shot": 1, "accuracy": 0.6},
    ]
    summary = summarize_groups(
        rows,
        group_fields=("phase", "method", "k_shot"),
        metric_keys=("accuracy",),
    )
    assert len(summary) == 1
    assert summary[0]["accuracy_n"] == 2
    assert summary[0]["accuracy_mean"] == 0.5
    assert summary[0]["accuracy_std"] > 0
    assert summary[0]["accuracy_ci95_low"] < 0.5 < summary[0]["accuracy_ci95_high"]
