import pytest

from scripts.summarize_adv3b02_partial_matrix import _arm_row_metrics


def test_arm_row_metrics_keeps_partial_diagnostic_metrics_same_row() -> None:
    score = {
        "before": {"old_acc": 0.8},
        "after": {
            "old_acc": 0.7,
            "seen_new_acc": 0.5,
            "h_old_new": 7.0 / 12.0,
            "by_tx": {
                "old_a": {"accuracy": 0.8, "role": "target_old"},
                "old_b": {"accuracy": 0.6, "role": "target_old"},
                "new_a": {"accuracy": 0.5, "role": "target_new"},
            },
            "by_scenario": {
                "clear": {"old_to_new_rate": 0.1, "new_to_old_rate": 0.2},
                "rain": {"old_to_new_rate": 0.3, "new_to_old_rate": 0.4},
            },
        },
    }

    metrics = _arm_row_metrics(score)

    assert metrics["old_before"] == 0.8
    assert metrics["old_after"] == 0.7
    assert metrics["old_gain"] == pytest.approx(-0.1)
    assert metrics["seen_new"] == 0.5
    assert metrics["floor"] == 0.5
    assert metrics["min_old"] == 0.6
    assert metrics["min_new"] == 0.5
    assert metrics["forgetting"] == pytest.approx(0.1)
    assert metrics["old_to_new"] == pytest.approx(0.2)
    assert metrics["new_to_old"] == pytest.approx(0.3)
