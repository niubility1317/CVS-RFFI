import unittest
import numpy as np
from experiments.pamf_phase3_restoration.restore_complementarity import correctness_cells, pair_marginal


class ComplementarityTest(unittest.TestCase):
    def test_all_32_patterns_and_pair_marginals(self):
        truth = np.zeros(32, dtype=np.int64)
        predictions = [np.array([0 if i & (1 << (4-j)) else 1 for i in range(32)], dtype=np.int64) for j in range(5)]
        cells = correctness_cells(truth, predictions)
        self.assertEqual(cells, {format(i, '05b'): 1 for i in range(32)})
        self.assertEqual(pair_marginal(cells, 0, 4), {'00': 8, '01': 8, '10': 8, '11': 8})

    def test_reject_misaligned_or_out_of_range(self):
        with self.assertRaises(ValueError):
            correctness_cells(np.array([0]), [np.array([0])]*4)
        with self.assertRaises(ValueError):
            correctness_cells(np.array([0]), [np.array([0])]*4+[np.array([6])])
        with self.assertRaises(ValueError):
            correctness_cells(np.array([0, 1]), [np.array([0])]*5)

if __name__ == '__main__':
    unittest.main()
