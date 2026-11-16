import unittest
from experiments.pamf_repl_attribution_factorial.analyze_segment import component_stat


class FactorialSegmentAnalysisTests(unittest.TestCase):
    def test_zero_component_has_undefined_direction(self):
        empty = {'L_norm':1.,'U_norm':0.,'dot':0.,'cosine':0.}
        self.assertEqual(component_stat(empty), {'defined':0,'conflict':0,'cosine_sum':0.})
        conflict = {'L_norm':2.,'U_norm':3.,'dot':-3.,'cosine':-.5}
        self.assertEqual(component_stat(conflict), {'defined':1,'conflict':1,'cosine_sum':-.5})


if __name__ == '__main__':
    unittest.main()
