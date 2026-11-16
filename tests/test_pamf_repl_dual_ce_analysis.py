import unittest
from experiments.pamf_repl_dual_ce.analyze_segment import guard_counts


class DualCEMetricsTests(unittest.TestCase):
    def test_counts_before_and_undefined_after_separately(self):
        value = dict(actual_parameter_check=True, norm_nonexpansion=True, quantization_fallback=True,
                     anchor_cosines_before={'L_full': -.2, 'L_clean': .3},
                     anchor_cosines_after={'L_full': None, 'L_clean': None})
        result = guard_counts(value)
        self.assertEqual(result['full_before_conflict'], 1)
        self.assertEqual(result['ce_before_conflict'], 0)
        self.assertEqual(result['full_after_undefined'], 1)
        self.assertEqual(result['ce_after_undefined'], 1)
        self.assertEqual(result['fallback'], 1)
        value['anchor_cosines_after']['L_clean'] = -.01
        with self.assertRaises(ValueError):
            guard_counts(value)


if __name__ == '__main__':
    unittest.main()
