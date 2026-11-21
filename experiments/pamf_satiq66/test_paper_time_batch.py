import numpy as np

from experiments.pamf_satiq66.paper_time_batch import nearest_temporal_negatives


def test_cross_satellite_negatives_preserve_the_default_same_satellite_lookup():
    times = np.array([4, 0, 2, 8, 20, 3, 40, 13], dtype=np.int64) * 1_000_000_000
    labels = np.array([0, 1, 2, 1, 3, -1, 0, 2])
    satellites = np.array([100, 100, 200, 300])

    np.testing.assert_array_equal(
        nearest_temporal_negatives(times, labels, satellites),
        [1, 0, -1, 0, -1, -1, -1, -1],
    )
    np.testing.assert_array_equal(
        nearest_temporal_negatives(times, labels, satellites, cross_satellite=True),
        [2, 2, 1, 7, -1, -1, -1, 3],
    )
