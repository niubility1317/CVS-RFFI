import unittest
import numpy as np
from experiments.phase1_source_continuous_fusion.diagnostic import comparison


class ComparisonTest(unittest.TestCase):
    def test_interior_only_is_separate_from_endpoint_union(self):
        classes = np.array([[0, 1, 1, 1, 0], [0, 0, 0, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
        result = comparison(classes, np.ones(4, dtype=int), np.array([.99, .99, .99, .2]))
        self.assertEqual(result['trusted'], 3)
        self.assertEqual(result['endpoint_union'], 2)
        self.assertEqual(result['interior_only'], 1)
        self.assertEqual(result['all_alpha_union'], 3)


if __name__ == '__main__': unittest.main()
