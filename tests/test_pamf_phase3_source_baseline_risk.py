import importlib
from pathlib import Path
import unittest

import numpy as np


class BaselineRiskTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/source_baseline_risk.py'
        self.assertTrue(path.is_file(), 'Original-baseline risk calibration is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.source_baseline_risk')

    def fixture(self):
        y = np.repeat(np.tile(np.arange(6), 5), 100).astype(np.int64)
        rx = np.repeat([1, 3, 4, 6, 8], 600).astype(np.int64)
        level = np.tile(np.arange(100), 30).astype(np.float64)
        return np.asarray(['known-' + str(i) for i in range(len(y))]), y, rx, np.repeat(level[:, None], 7, axis=1)

    def test_actual_encoder_gain_can_use_frr_budget_without_splitting_ties(self):
        module = self.module(); ids, y, rx, scores = self.fixture()
        original = y.copy(); original[scores[:, 0] < 10] = (original[scores[:, 0] < 10] + 1) % 6
        result = module.calibrate_known(ids, y, y.copy(), original, scores, rx)
        np.testing.assert_array_equal(result['thresholds'], np.repeat(5., 7))
        self.assertTrue(result['constraints_passed'])
        for value in result['methods'].values():
            self.assertAlmostEqual(value['overall']['macro_frr'], .05)
            self.assertAlmostEqual(value['overall']['original_closed_minus_new_open'], -.05)
            self.assertFalse(value['next_candidate_passed'])
        self.assertFalse(result['unknown_inputs_used'])

    def test_baseline_loss_exact_boundary_limits_threshold_and_encoder_is_separate_gate(self):
        module = self.module(); ids, y, rx, scores = self.fixture()
        closed = y.copy(); closed[scores[:, 0] == 99] = (closed[scores[:, 0] == 99] + 1) % 6
        result = module.calibrate_known(ids, y, closed, y.copy(), scores, rx)
        np.testing.assert_array_equal(result['thresholds'], np.ones(7))
        self.assertTrue(result['constraints_passed'])
        self.assertFalse(result['encoder_protection_passed'])
        self.assertAlmostEqual(result['methods']['msp']['overall']['original_closed_minus_new_open'], .02)
        # Wrong predictions rejected incur FRR, but no new correctness loss.
        scores[closed != y] = -1.
        second = module.calibrate_known(ids, y, closed, y.copy(), scores, rx)
        self.assertGreater(second['methods']['msp']['overall']['rejected_count'], result['methods']['msp']['overall']['rejected_count'])
        self.assertEqual(second['methods']['msp']['overall']['rejected_correct_count'], result['methods']['msp']['overall']['rejected_correct_count'])

    def test_single_rx_constraint_cannot_be_hidden_by_overall_average(self):
        module = self.module(); ids, y, rx, scores = self.fixture()
        scores[rx == 1] -= 1000.
        result = module.calibrate_known(ids, y, y.copy(), y.copy(), scores, rx)
        np.testing.assert_array_equal(result['thresholds'], np.repeat(-998., 7))
        self.assertAlmostEqual(result['methods']['msp']['by_receiver']['1']['macro_frr'], .02)

    def test_inputs_are_only_known_and_infeasible_encoder_does_not_claim_pass(self):
        module = self.module(); ids, y, rx, scores = self.fixture()
        for bad in (np.full(len(y), 6, dtype=np.int64), np.ones((len(y), 1), dtype=np.int64)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                module.calibrate_known(ids, y, y, bad, scores, rx)
        duplicate = ids.copy(); duplicate[0] = duplicate[1]
        with self.assertRaises(ValueError): module.calibrate_known(duplicate, y, y, y, scores, rx)
        with self.assertRaises(TypeError): module.calibrate_known(ids, y, y, y, scores, rx, novel_scores=scores)
        result = module.calibrate_known(ids, y, (y + 1) % 6, y.copy(), scores, rx)
        self.assertFalse(result['constraints_passed'])
        self.assertFalse(result['encoder_protection_passed'])
        self.assertEqual(result['methods']['msp']['overall']['rejected_count'], 0)

    def test_sorted_solver_matches_brute_force_global_candidates_and_row_permutation(self):
        module = self.module(); ids, y, rx, _ = self.fixture()
        rng = np.random.default_rng(7); scores = rng.integers(0, 40, size=(len(y), 7)).astype(np.float64)
        positions = np.tile(np.arange(100), 30)
        original, new = y.copy(), y.copy()
        original[np.isin(positions, [95, 96])] = (original[np.isin(positions, [95, 96])] + 1) % 6
        new[positions == 99] = (new[positions == 99] + 1) % 6
        result = module.calibrate_known(ids, y, new, original, scores, rx)
        expected = []
        for j in range(7):
            valid = []
            for threshold in np.r_[np.unique(scores[:, j]), np.nextafter(scores[:, j].max(), np.inf)]:
                rejected = scores[:, j] < threshold
                passed = True
                for mask in [np.ones(len(y), bool)] + [rx == value for value in (1, 3, 4, 6, 8)]:
                    frr = np.mean([rejected[mask & (y == c)].mean() for c in range(6)])
                    loss = np.mean([(original[mask & (y == c)] == c).mean()
                        - ((new[mask & (y == c)] == c) & ~rejected[mask & (y == c)]).mean() for c in range(6)])
                    passed &= frr <= .05 + 1e-12 and loss <= .02 + 1e-12
                if passed: valid.append(threshold)
            expected.append(max(valid))
        np.testing.assert_array_equal(result['thresholds'], expected)
        order = rng.permutation(len(y))
        reordered = module.calibrate_known(ids[order], y[order], new[order], original[order], scores[order], rx[order])
        np.testing.assert_array_equal(reordered['thresholds'], result['thresholds'])


if __name__ == '__main__': unittest.main()
