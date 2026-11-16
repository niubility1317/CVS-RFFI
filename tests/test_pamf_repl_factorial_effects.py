import unittest
from experiments.pamf_repl_factorial_effects import contrast_plan, summarize_effects


class FactorialEffectTests(unittest.TestCase):
    def test_simple_effects_and_interaction_are_not_conflated(self):
        cells = {(w,a,b): 80+2*(w==6)+3*a+5*b+7*a*b
                 for w in (3.,6.) for a in (0,1) for b in (0,1)}
        result = summarize_effects(cells)
        self.assertEqual(result['by_weight']['3.0']['A_at_B0'],3)
        self.assertEqual(result['by_weight']['3.0']['A_at_B1'],10)
        self.assertEqual(result['by_weight']['3.0']['B_at_A0'],5)
        self.assertEqual(result['by_weight']['3.0']['B_at_A1'],12)
        self.assertEqual(result['by_weight']['3.0']['AB_interaction'],7)
        self.assertEqual(result['weight_effects']['A1_B1'],2)
        self.assertEqual(len(contrast_plan()),12)

    def test_missing_cell_is_an_error(self):
        cells = {(w,a,b):1. for w in (3.,6.) for a in (0,1) for b in (0,1)}
        del cells[(6.,1,1)]
        with self.assertRaises(ValueError):
            summarize_effects(cells)


if __name__ == '__main__':
    unittest.main()
