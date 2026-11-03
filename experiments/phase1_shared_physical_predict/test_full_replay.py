import unittest
import numpy as np
from experiments.phase1_shared_physical_predict.full_replay import check_decisions


class DecisionCheckTest(unittest.TestCase):
    def test_accepts_exact_decisions_and_rejects_either_mismatch(self):
        raw = np.array([0, 2, 4], dtype=np.int64)
        restored = np.array([1, 2, 5], dtype=np.int64)
        check_decisions((raw, restored), raw.copy(), restored.copy())
        with self.assertRaises(ValueError):
            check_decisions((raw, restored), np.array([0, 1, 4]), restored)
        with self.assertRaises(ValueError):
            check_decisions((raw, restored), raw, np.array([1, 2, 4]))


if __name__ == '__main__':
    unittest.main()
