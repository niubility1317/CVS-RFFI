import unittest

import numpy as np
import torch

from cvsrffi.ecrs_sampling import MatchedLabeledSampler, RXDayBalancedQueue


class SamplingTests(unittest.TestCase):
    def test_u_balanced_full_coverage_and_exact_resume(self):
        rx = torch.tensor([0] * 8 + [1] * 8)
        day = np.tile(np.repeat([0, 1], 4), 2)
        queue = RXDayBalancedQueue(rx, day, seed=42)
        first = queue.take(8)
        self.assertEqual(len(set(first)), 8)
        for r in [0, 1]:
            for d in [0, 1]:
                self.assertEqual(sum(int(rx[i]) == r and day[i] == d for i in first), 2)
        state = queue.state_dict()
        next_indices = queue.take(24)
        self.assertEqual(len(set(first + next_indices[:8])), 16)
        resumed = RXDayBalancedQueue(rx, day, seed=999)
        resumed.load_state_dict(state)
        self.assertEqual(next_indices, resumed.take(24))
        self.assertEqual(resumed.coverage, queue.coverage)
        self.assertEqual(queue.coverage['unique_fraction'], 1.)

    def test_u_unequal_cells_no_repeat_before_complete_cycle(self):
        queue = RXDayBalancedQueue([0, 0, 0, 1], [0, 0, 0, 0])
        self.assertEqual(len(set(queue.take(4))), 4)
        self.assertEqual(queue.coverage['completed_cycles'], 1)
        with self.assertRaises(ValueError):
            RXDayBalancedQueue([1, 0, 0, 0], [0, 0, 0, 0]).load_state_dict(queue.state_dict())

    def test_l_grid_day_matching_and_exact_resume(self):
        rows = [(tx, rx, day) for tx in range(8) for rx in range(5) for day in range(3) for _ in range(4)]
        labels, rx, day = np.array(rows).T
        sampler = MatchedLabeledSampler(labels, rx, day, seed=2)
        indices, counts = sampler.take_batch()
        self.assertEqual(len(indices), 120)
        self.assertEqual(len(set(indices)), 120)
        self.assertEqual(counts['missing_cells'], 0)
        self.assertEqual(len(set(labels[indices])), 6)
        for receiver in range(5):
            self.assertEqual(set(day[np.array(indices)[rx[indices] == receiver]]), {0})
        state = sampler.state_dict()
        expected = [sampler.take_batch() for _ in range(5)]
        restored = MatchedLabeledSampler(labels, rx, day, seed=99)
        restored.load_state_dict(state)
        self.assertEqual([restored.take_batch() for _ in range(5)], expected)

    def test_l_missing_cells_are_short_not_synthetic(self):
        labels = [0, 0, 1, 1, 999]
        rx = [0, 1, 0, 0, 1]
        day = [0] * 5
        sampler = MatchedLabeledSampler(labels, rx, day, tx_per_batch=2, packets_per_cell=4,
                                         label_mask=[True, True, True, True, False])
        indices, counts = sampler.take_batch()
        self.assertEqual(set(indices), {0, 1, 2, 3})
        self.assertEqual(counts['actual'], 4)
        self.assertEqual(counts['requested'], 16)
        self.assertEqual(counts['missing_cells'], 1)
        self.assertEqual(counts['short_cells'], 3)
        self.assertNotIn(999, counts['selected_tx'])


if __name__ == '__main__':
    unittest.main()
