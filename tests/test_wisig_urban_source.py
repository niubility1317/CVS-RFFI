import numpy as np

from experiments.pamf_lora_main.dura_real_gain_wisig_urban_source import urban_training_indices


def test_only_legal_full_low_urban_unlabeled_rows_are_selected():
    unlabeled = np.array([0, 2, 3, 5])
    quantile = np.array([.2, .8, .1, .7, .9, .5])
    scenes = np.array([5, 5, 5, 4, 5, 5])
    np.testing.assert_array_equal(urban_training_indices(unlabeled, quantile, scenes), [0, 5])
