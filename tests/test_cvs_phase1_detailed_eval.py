from paper_reproduction.scripts.evaluate_cvs_phase1_detailed import aggregate_score_rows


def test_aggregate_score_rows_keeps_receiver_transmitter_day_context() -> None:
    rows = [
        {
            "scenario": "leo_clear_weak",
            "split": "test_unseen_day_unseen_rx",
            "receiver_label": "20-1",
            "transmitter_label": "14-10",
            "predicted_transmitter_label": "14-10",
            "day_label": "2021_03_15",
            "correct": 1,
        },
        {
            "scenario": "leo_clear_weak",
            "split": "test_unseen_day_unseen_rx",
            "receiver_label": "20-1",
            "transmitter_label": "14-10",
            "predicted_transmitter_label": "14-7",
            "day_label": "2021_03_15",
            "correct": 0,
        },
    ]
    detailed = aggregate_score_rows(rows)
    joint = [row for row in detailed if row["group_type"] == "per_receiver_transmitter_day"]
    assert joint == [
        {
            "scenario": "leo_clear_weak",
            "group_type": "per_receiver_transmitter_day",
            "split": "ALL",
            "receiver_label": "20-1",
            "transmitter_label": "14-10",
            "day_label": "2021_03_15",
            "sample_count": 2,
            "correct_count": 1,
            "accuracy": 0.5,
            "confusion_json": '{"14-10->14-10": 1, "14-10->14-7": 1}',
        }
    ]
