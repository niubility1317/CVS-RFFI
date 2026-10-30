import unittest

from experiments.phase1_external_matched.runner import PROFILES, SEEDS, TASKS, VIEWS, is_score_job, gpu_eligible, phase2_capacity


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

    def test_capacity_includes_other_users(self):
        device = {'pids': {1, 2, 3}, 'free': 12000, 'blocked': True}
        self.assertTrue(gpu_eligible(device, False))
        self.assertFalse(gpu_eligible(device, True))
        self.assertFalse(gpu_eligible(dict(device, pids={1, 2, 3, 4}), False))
        self.assertFalse(gpu_eligible(dict(device, free=11999), False))

    def test_phase2_refill_window(self):
        state = {'status':'RUNNING', 'controller_pid':10, 'active':{
            'a': {'kind':'predict','gpu':0,'pid':11},
            'b': {'kind':'predict','gpu':0,'pid':12}}}
        self.assertTrue(phase2_capacity(state, 0, lambda p: p in {10, 11, 12}, {11, 12, 20}))
        self.assertFalse(phase2_capacity(state, 0, lambda p: p in {10, 11, 12}, {11, 20}))
        self.assertFalse(phase2_capacity(state, 0, lambda p: p in {10, 11}, {11, 12, 20}))
        self.assertFalse(phase2_capacity(state, 0, lambda p: p in {11, 12}, {11, 12, 20}))
        self.assertTrue(phase2_capacity(dict(state, status='ANALYZED'), 0, lambda p: False, set()))


if __name__ == '__main__':
    unittest.main()
