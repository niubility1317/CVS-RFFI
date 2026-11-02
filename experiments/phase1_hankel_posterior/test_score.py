import unittest
import numpy as np

from experiments.phase1_hankel_posterior.score import count_cell, validate_fixed


class ScoreTest(unittest.TestCase):
    def test_repair_and_damage_are_paired(self):
        target = np.array([0, 1, 2, 3])
        baseline = np.array([0, 0, 2, 0])
        candidate = np.array([0, 1, 0, 0])
        self.assertEqual(count_cell(target, baseline, candidate, np.ones(4, dtype=bool)),
                         dict(count=4, baseline_correct=2, candidate_correct=2, repairs=1, damages=1))

    def test_prediction_must_be_fixed_before_truth(self):
        with self.assertRaises(ValueError):
            validate_fixed({'status': 'PARTIAL'}, 'fixed')
        m = dict(status='SOURCE_PREDICTIONS_FIXED', commit='fixed', count=672, source_only=True,
                 DURA_loaded=False, F_frozen=True, TX_truth_reads=0, U_s_truth_reads=0,
                 V_reads=0, test_reads=0, model_updates=0)
        validate_fixed(m, 'fixed')
        for k in ('TX_truth_reads', 'U_s_truth_reads', 'V_reads', 'test_reads', 'model_updates'):
            bad = dict(m, **{k: 1})
            with self.assertRaises(ValueError): validate_fixed(bad, 'fixed')

if __name__ == '__main__': unittest.main()
