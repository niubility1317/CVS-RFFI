import unittest
import numpy as np

from experiments.phase1_dura_orthogonal.spectral_crossview_probe import summarize_ranks, jointly_damaged


class CrossViewProbeTest(unittest.TestCase):
    def test_rank_metrics(self):
        result = summarize_ranks(np.array([1, 2, 4, 4]), 4)
        self.assertEqual(result['count'], 4)
        self.assertEqual(result['top1'], .25)
        self.assertEqual(result['mean_reciprocal_rank'], .5)
        self.assertEqual(result['median_rank'], 3.)

    def test_only_pairs_with_two_damaged_views_are_selected(self):
        mask = jointly_damaged(np.array(['clean', 'practical_mid', 'practical_high']),
                               np.array(['practical_mid', 'clean', 'practical_low_urban']))
        np.testing.assert_array_equal(mask, [False, False, True])
        with self.assertRaises(ValueError):
            jointly_damaged(np.array(['clean']), np.array(['clean', 'clean']))

    def test_invalid_ranks_rejected(self):
        for ranks in (np.array([0, 1]), np.array([1, 5]), np.array([1.5, 2]), np.array([])):
            with self.assertRaises(ValueError):
                summarize_ranks(ranks, 4)


if __name__ == '__main__':
    unittest.main()
