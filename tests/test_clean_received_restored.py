import numpy as np
import pytest

from experiments.pamf_phase3_restoration.clean_received_restored import joint_counts


def test_all_eight_paired_correctness_cases():
    truth = np.zeros(8, dtype=np.int64)
    clean = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    received = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    restored = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    result = joint_counts(truth, clean, received, restored)
    assert result['count'] == 8
    assert set(result['cells']) == {f'{a}{b}{c}' for a in (0, 1) for b in (0, 1) for c in (0, 1)}
    assert all(value == 1 for value in result['cells'].values())
    assert result['clean_wrong_received_wrong_restored_correct'] == 1
    assert result['clean_wrong_received_correct_restored_correct'] == 1
    assert result['clean_correct_received_correct_restored_wrong'] == 1


def test_mismatched_or_out_of_range_predictions_are_rejected():
    y = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError):
        joint_counts(y, y[:1], y, y)
    with pytest.raises(ValueError):
        joint_counts(y, y, np.array([0, 7]), y)
