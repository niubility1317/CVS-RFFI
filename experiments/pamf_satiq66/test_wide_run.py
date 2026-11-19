"""Checks for beam-grained scoring and matched supervised sampling."""

import numpy as np

from experiments.pamf_satiq66.train import Batches
from experiments.pamf_satiq66.wide_run import describe


def test_beam_scores_keep_same_satellite_wrong_beam_as_wrong():
    logits = np.zeros((4, 66), dtype=np.float32)
    labels = np.array([0, 1, 3, 4])
    logits[np.arange(4), [0, 0, 3, 5]] = 10
    satellite = np.repeat(np.arange(22), 3)
    scored = describe(logits, labels, satellite)
    assert scored["accuracy"] == .5
    assert scored["satellite_accuracy"] == 1.
    assert scored["same_satellite_wrong_beam"] == 2
    assert scored["beam_accuracy_given_correct_satellite"] == .5


def test_beam_batch_has_two_physical_packets_on_distinct_days():
    y = np.repeat(np.arange(66), 8 * 15)
    day = np.tile(np.repeat(np.arange(8), 15), 66)
    chosen = Batches({"y": y, "day": day}, seed=2026111902, records_per_class_day=15).next()
    assert np.all(np.bincount(y[chosen]) == 2)
    assert len(set(chosen)) == 132
    assert all(len(set(day[chosen][y[chosen] == label])) == 2 for label in range(66))
