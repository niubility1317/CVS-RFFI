from fractions import Fraction
import importlib
import unittest

import numpy as np


RX = [1, 3, 4, 6, 8]


class SourceRiskCalibrationTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.source_risk_calibration')

    def rows(self, per_class=10):
        y = np.tile(np.repeat(np.arange(6), per_class), 5).astype(np.int64)
        rx = np.repeat(RX, 6 * per_class).astype(np.int64)
        ids = np.asarray(['known-%05d' % i for i in range(len(y))])
        return ids, y, y.copy(), rx

    def feasible(self, threshold, y, closed, values, rx):
        rejected = values < threshold
        for mask in [np.ones(len(y), dtype=bool)] + [rx == r for r in RX]:
            frr = sum((Fraction(int(np.count_nonzero(mask & (y == label) & rejected)), int(np.count_nonzero(mask & (y == label)))) for label in range(6)), Fraction()) / 6
            loss = sum((Fraction(int(np.count_nonzero(mask & (y == label) & rejected & (closed == y))), int(np.count_nonzero(mask & (y == label)))) for label in range(6)), Fraction()) / 6
            if frr > Fraction(1, 25) or loss > Fraction(3, 200): return False
        return True

    def test_sorted_cumulative_result_matches_small_brute_force_with_ties(self):
        module = self.module(); ids, y, closed, rx = self.rows(8)
        rng = np.random.default_rng(92)
        # Use an independent fixed error mask; tied score groups must stay whole.
        errors = np.arange(len(y)) % 3 == 0; closed[errors] = (y[errors] + 1) % 6
        values = rng.integers(-5, 8, (len(y), 7)).astype(np.float64)
        result = module.calibrate_known(ids, y, closed, values, rx)
        expected = []
        for j in range(7):
            levels = np.unique(values[:, j]); candidates = np.r_[levels, np.nextafter(levels[-1], np.inf)]
            expected.append(max(t for t in candidates if self.feasible(t, y, closed, values[:, j], rx)))
        np.testing.assert_array_equal(result['thresholds'], expected)
        self.assertTrue(result['constraints_passed'])
        self.assertEqual(result['known_count'], len(y))
        for method in result['methods'].values():
            self.assertEqual(set(method['by_receiver']), {'1', '3', '4', '6', '8'})
            self.assertTrue(method['constraints_passed'])

    def test_exact_accuracy_loss_boundary_allows_nine_correct_per_rx(self):
        module = self.module(); ids, y, closed, rx = self.rows(100)
        scores = np.zeros(len(y))
        for receiver in RX:
            rows = np.flatnonzero((rx == receiver) & (y == 0))
            scores[rows[:9]] = -2.; scores[rows[9]] = -1.
        result = module.calibrate_known(ids, y, closed, np.repeat(scores[:, None], 7, axis=1), rx)
        np.testing.assert_array_equal(result['thresholds'], np.repeat(-1., 7))
        first = next(iter(result['methods'].values()))
        self.assertEqual(first['overall']['macro_correct_rejection_loss'], .015)
        self.assertTrue(first['overall']['checks']['macro_correct_rejection_loss_le_015'])
        self.assertTrue(all(group['rejected_correct_count'] == 9 for group in first['by_receiver'].values()))

    def test_original_errors_cost_no_new_accuracy_but_frr_boundary_still_binds(self):
        module = self.module(); ids, y, closed, rx = self.rows(100)
        scores = np.zeros(len(y))
        for receiver in RX:
            rows = np.flatnonzero((rx == receiver) & (y == 0))
            scores[rows[:24]] = -2.; scores[rows[24]] = -1.; closed[rows[:25]] = 1
        result = module.calibrate_known(ids, y, closed, np.repeat(scores[:, None], 7, axis=1), rx)
        np.testing.assert_array_equal(result['thresholds'], np.repeat(-1., 7))
        first = next(iter(result['methods'].values()))
        self.assertEqual(first['overall']['macro_frr'], .04)
        self.assertEqual(first['overall']['macro_correct_rejection_loss'], 0.)
        self.assertTrue(all(group['rejected_correct_count'] == 0 for group in first['by_receiver'].values()))

    def test_each_rx_constraint_and_rare_class_macro_not_pooled(self):
        module = self.module(); ids, y, closed, rx = self.rows(10)
        scores = np.zeros(len(y)); scores[0] = -2.
        result = module.calibrate_known(ids, y, closed, np.repeat(scores[:, None], 7, axis=1), rx)
        np.testing.assert_array_equal(result['thresholds'], np.repeat(-2., 7))
        # A rare class cannot be hidden by many observations from other classes.
        keep = ~((rx == 1) & (y == 0)); keep[0] = True
        result = module.calibrate_known(ids[keep], y[keep], closed[keep], np.repeat(scores[keep, None], 7, axis=1), rx[keep])
        np.testing.assert_array_equal(result['thresholds'], np.repeat(-2., 7))

    def test_row_and_class_permutations_leave_all_global_thresholds_unchanged(self):
        module = self.module(); ids, y, closed, rx = self.rows(20)
        values = np.random.default_rng(6).normal(size=(len(y), 7))
        a = module.calibrate_known(ids, y, closed, values, rx)
        order = np.random.default_rng(8).permutation(len(y))
        b = module.calibrate_known(ids[order], 5-y[order], 5-closed[order], values[order], rx[order])
        np.testing.assert_array_equal(a['thresholds'], b['thresholds'])

    def test_all_equal_and_largest_finite_scores_return_finite_feasible_threshold(self):
        module = self.module(); ids, y, closed, rx = self.rows(1)
        for value in [1., np.finfo(np.float64).max]:
            with self.subTest(value=value):
                result = module.calibrate_known(ids, y, closed, np.full((len(y), 7), value), rx)
                np.testing.assert_array_equal(result['thresholds'], np.repeat(value, 7))
                self.assertTrue(np.isfinite(result['thresholds']).all())
                self.assertTrue(all(method['overall']['rejected_count'] == 0 for method in result['methods'].values()))

    def test_invalid_shapes_hidden_labels_missing_cells_and_ids_fail(self):
        module = self.module(); ids, y, closed, rx = self.rows(2)
        scores = np.ones((len(y), 7))
        bad = []
        duplicate = ids.copy(); duplicate[1] = duplicate[0]; bad.append((duplicate, y, closed, scores, rx))
        unknown = y.copy(); unknown[0] = 6; bad.append((ids, unknown, closed, scores, rx))
        broken = closed.copy(); broken[0] = -1; bad.append((ids, y, broken, scores, rx))
        foreign = rx.copy(); foreign[0] = 30; bad.append((ids, y, closed, scores, foreign))
        bad.append((ids, y, closed, scores[:, :6], rx)); bad.append((ids, y, closed, scores * np.nan, rx))
        keep = ~((rx == 1) & (y == 0)); bad.append((ids[keep], y[keep], closed[keep], scores[keep], rx[keep]))
        for values in bad:
            with self.subTest(), self.assertRaises(ValueError): module.calibrate_known(*values)
        with self.assertRaises(TypeError):
            module.calibrate_known(ids, y, closed, scores, rx, unknown_scores=np.ones(3))

    def test_explicit_static_five_matches_original_columns_and_tie_constraints(self):
        module = self.module(); ids, y, closed, rx = self.rows(100)
        values = np.random.default_rng(19).integers(-7, 9, (len(y), 7)).astype(np.float64)
        closed[::7] = (closed[::7] + 1) % 6
        original = module.calibrate_known(ids, y, closed, values, rx)
        explicit = module.calibrate_known(ids, y, closed, values[:, :5], rx,
                                          method_names=module.STATIC_METHODS)
        self.assertEqual(explicit['method_names'], list(module.STATIC_METHODS))
        np.testing.assert_array_equal(explicit['thresholds'], original['thresholds'][:5])
        for name in module.STATIC_METHODS:
            self.assertEqual(explicit['methods'][name], original['methods'][name])
        self.assertEqual(explicit['limits'], original['limits'])
        self.assertTrue(explicit['constraints_passed'])

    def test_explicit_method_names_cannot_reorder_subset_or_mislabel_columns(self):
        module = self.module(); ids, y, closed, rx = self.rows(2)
        scores = np.ones((len(y), 5))
        for names in [[], ['msp'], list(module.STATIC_METHODS)[::-1],
                      ['msp', 'maxlogit', 'margin', 'energy', 'linear_oe'], 'msp']:
            with self.subTest(names=names), self.assertRaises(ValueError):
                module.calibrate_known(ids, y, closed, scores, rx, method_names=names)
        with self.assertRaises(ValueError):
            module.calibrate_known(ids, y, closed, np.ones((len(y), 7)), rx,
                                    method_names=module.STATIC_METHODS)


if __name__ == '__main__':
    unittest.main()
