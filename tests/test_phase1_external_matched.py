import unittest

from experiments.phase1_external_matched.runner import PROFILES, SEEDS, TASKS, VIEWS, is_score_job


class Phase1ExternalMatchedTests(unittest.TestCase):
    def test_fixed_matrix(self):
        self.assertEqual(len(PROFILES), 10)
        self.assertEqual(len(set(PROFILES)), 10)
        self.assertEqual(SEEDS, tuple(range(2026092701, 2026092705)))
        self.assertEqual(TASKS, ('cross_rx', 'cross_both'))
        self.assertEqual(len(VIEWS), 7)
        self.assertEqual(VIEWS[0], 'clean')

    def test_detect_existing_phase2_scorer(self):
        self.assertTrue(is_score_job(['python', '-m', 'experiments.pamf_phase2_practical.group1_dispatch',
                                      '--job', 'score-predict-radionet-s2026092701-c0001']))
        self.assertFalse(is_score_job(['python', '--job', 'predict-radionet-s2026092701-c0001']))
        self.assertFalse(is_score_job(['python', '--job']))


if __name__ == '__main__':
    unittest.main()
