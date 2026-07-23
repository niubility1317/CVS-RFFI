from __future__ import annotations

import numpy as np

from paper_reproduction.scripts.run_adv3b02_paper_full_newclass_no_leo_diagnostic import (
    _raw_key,
    replace_new_class_iq,
    score_predictions,
)


def test_replace_new_class_iq_preserves_old_rows() -> None:
    original = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    arrays = {
        "leo_weak_iq": original,
        "dataset_role": np.asarray(["target_old", "target_new", "target_new"]),
        "tx_ids": np.asarray(["old", "new1", "new2"]),
        "rx_ids": np.asarray(["rx", "rx", "rx"]),
        "day_ids": np.asarray(["day", "day", "day"]),
        "eq_ids": np.asarray(["1", "1", "1"]),
        "sig_ids": np.asarray(["0", "1", "2"]),
    }
    lookup = {
        _raw_key("new1", "rx", "day", "1", "1"): np.ones((2, 4), np.float32),
        _raw_key("new2", "rx", "day", "1", "2"): np.full(
            (2, 4), 2.0, np.float32
        ),
    }
    mixed, audit = replace_new_class_iq(arrays, lookup)
    np.testing.assert_array_equal(mixed[0], original[0])
    np.testing.assert_array_equal(mixed[1], lookup[("new1", "rx", "day", "1", "1")])
    np.testing.assert_array_equal(mixed[2], lookup[("new2", "rx", "day", "1", "2")])
    assert audit["old_class_rows_unchanged"] == 1
    assert audit["new_class_rows_replaced_with_raw"] == 2


def test_score_predictions_keeps_joint_row_metrics() -> None:
    truth = np.asarray([0, 0, 1, 1, 2, 2])
    before = np.asarray([0, 0, 1, 1, 0, 0])
    after = np.asarray([0, 1, 1, 1, 2, 0])
    result = score_predictions(
        before=before,
        after=after,
        truth=truth,
        old_count=2,
    )
    assert result["old_acc_before_increment"] == 1.0
    assert result["old_acc_after_increment"] == 0.75
    assert result["seen_new_acc"] == 0.5
    assert result["candidate_average_forgetting"] == 0.25
    assert result["min_old_class_acc"] == 0.5
