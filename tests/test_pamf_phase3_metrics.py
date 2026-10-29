import unittest
import numpy as np

from experiments.pamf_phase3_open_world import metrics


class MetricDefinitions(unittest.TestCase):
    def evaluate(self, y=(0, 0, 1, 1), p=(0, 1, 1, 0), s=(2., 2., 2., 2.),
                 uy=('a', 'a', 'b', 'b'), us=(0., 0., 0., 0.), threshold=1.):
        return metrics.open_set_metrics(np.asarray(y), np.asarray(p), np.asarray(s),
            np.asarray(uy), np.asarray(us), threshold, (0, 1), ('a', 'b'), np.asarray(y))

    def test_acceptance_does_not_count_as_identity_accuracy(self):
        r = self.evaluate()
        self.assertEqual(r['known_macro_acceptance'], 1.)
        self.assertEqual(r['known_macro_open_accuracy'], .5)
        self.assertEqual(r['unknown_macro_rejection'], 1.)
        self.assertAlmostEqual(r['H_OS'], 2 / 3)
        self.assertEqual(r['original_closed_minus_new_open'], .5)

    def test_macro_invariant_to_repeating_one_class(self):
        small = self.evaluate(y=(0, 1), p=(0, 1), s=(2., 0.), uy=('a', 'b'), us=(1., 1.))
        large = self.evaluate(y=tuple([0] * 100 + [1]), p=tuple([0] * 100 + [1]),
            s=tuple([2.] * 100 + [0.]), uy=tuple(['a'] * 77 + ['b']), us=tuple([1.] * 78))
        for key in ('known_macro_acceptance', 'known_macro_open_accuracy', 'unknown_macro_rejection',
                    'H_OS', 'auroc_known_macro', 'oscr_auc_macro'):
            self.assertAlmostEqual(small[key], large[key])
        self.assertEqual(small['auroc_known_macro'], .5)
        self.assertEqual(large['known_macro_acceptance'], .5)

    def test_ties_are_whole_at_operating_threshold_and_symmetric_in_curve(self):
        r = self.evaluate(s=(1.,) * 4, us=(1.,) * 4)
        self.assertEqual(r['known_macro_acceptance'], 1.)
        self.assertEqual(r['unknown_macro_rejection'], 0.)
        self.assertEqual(r['H_OS'], 0.)
        self.assertEqual(r['auroc_known_macro'], .5)
        self.assertEqual(r['oscr_auc_macro'], .25)

    def test_missing_required_class_is_not_a_measured_zero(self):
        r = self.evaluate(uy=('a', 'a'), us=(0., 0.))
        self.assertFalse(r['coverage']['unknown_complete'])
        self.assertIsNone(r['unknown_macro_rejection'])
        self.assertIsNone(r['H_OS'])
        self.assertIsNone(r['oscr_auc_macro'])

    def test_full_world_zero_new_is_zero_and_missing_is_none(self):
        self.assertEqual(metrics.open_world_h(.8, 0., .9), 0.)
        self.assertIsNone(metrics.open_world_h(.8, None, .9))
        self.assertAlmostEqual(metrics.open_world_h(.8, .8, .8), .8)

    def test_negative_inputs_rejected(self):
        for kwargs in ({'s': (float('nan'),) * 4}, {'us': (float('inf'),) * 4},
                       {'threshold': float('nan')}, {'p': (0, 1)}, {'uy': ('x',) * 4}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.evaluate(**kwargs)


if __name__ == '__main__':
    unittest.main()
