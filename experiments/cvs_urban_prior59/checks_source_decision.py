"""Regression for NumPy source-comparison scalars; synthetic metadata only."""
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def load_report(design):
    """Load the real report with an in-memory design, without training imports."""
    name = '_urban_prior59_source_decision_check'
    package = types.ModuleType(name)
    package.__path__ = [str(Path(__file__).parent)]
    package.design = design
    sys.modules[name] = package
    sys.modules[name + '.design'] = design
    spec = importlib.util.spec_from_file_location(
        name + '.report', Path(__file__).with_name('report.py'))
    report = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = report
    spec.loader.exec_module(report)
    return report


class SourceDecisionChecks(unittest.TestCase):
    def decide(self, clean=.98, current=(.90, .76, .62), old=(.90, .74, .60)):
        d = types.ModuleType('design')
        d.BASE = Path('/synthetic/current')
        d.PROJECT = '/synthetic/project'
        d.REFERENCE_RUN = 'reference'
        row = {'row_id': 'URBAN-s2026092701'}
        d.rows = lambda: [row]
        d.config = lambda _: {'synthetic': 'current'}
        d.old_config = lambda: {'synthetic': 'reference'}
        done = dict(target_access=False, final_source_metrics=dict(
            source_val_count=27000, source_val_accuracy=clean))
        control = dict(target_access=False, final_source_metrics=dict(
            source_val_count=27000, source_val_accuracy=.98))
        common = dict(target_access=False, formal_scores_read=False)
        q = dict(common, config=d.config(row), NN=12600, training_updates=0,
                 source_accuracies=[list(current), list(current)])
        reference = dict(common, config=d.old_config(),
                         source_accuracies=[list(old), list(old)])
        inputs = {
            d.BASE / row['row_id'] / 'source/completion.json': done,
            d.BASE / row['row_id'] / 'source_qualification/complete.json': q,
            Path(d.PROJECT) / 'runs' / d.REFERENCE_RUN /
            'B01-s2026092701/source_qualification/complete.json': reference,
            Path(d.PROJECT) / 'runs' / d.REFERENCE_RUN /
            'B00-s2026092701/source/completion.json': control,
        }
        reads = []
        writes = {}

        def read(path):
            reads.append(path)
            return inputs[path]

        def write(path, value):
            # Exercise the actual JSON boundary that failed, not a mocked dump.
            writes[path] = json.dumps(value, allow_nan=False)

        d.read, d.write = read, write
        result = load_report(d).source_decision()
        self.assertEqual(set(reads), set(inputs))
        self.assertEqual(len(reads), 4)
        self.assertEqual(set(writes), {d.BASE / 'source_matrix_decision.json'})
        self.assertEqual(json.loads(next(iter(writes.values()))), result)
        self.assertTrue(all(type(v) is bool for v in result['checks'].values()))
        self.assertIs(type(result['source_qualified']), bool)
        self.assertFalse(result['formal_scores_read'])
        self.assertTrue(result['formal_evaluation_always_required'])
        self.assertEqual(result['NN'], 0)
        return result

    def test_numpy_comparison_is_not_native_json_boolean(self):
        comparison = np.asarray([.76, .76]).mean() >= .74
        self.assertIs(type(comparison), np.bool_)
        with self.assertRaises(TypeError):
            json.dumps({'comparison': comparison})

    def test_pass_is_json_serializable(self):
        result = self.decide(clean=np.float64(.98))
        self.assertTrue(result['source_qualified'])
        self.assertTrue(all(result['checks'].values()))

    def test_each_original_condition_can_fail_and_still_requires_formal(self):
        cases = [
            (dict(clean=np.float64(.978)), 'fullV_clean_within_0p1pp'),
            (dict(current=(.90, .742, .602)),
             'mid_low_mean_gain_at_least_0p5pp'),
            (dict(current=(.90, .78, .599)), 'low_not_below_originalAs30'),
        ]
        for arguments, failed in cases:
            with self.subTest(failed=failed):
                result = self.decide(**arguments)
                self.assertFalse(result['source_qualified'])
                self.assertEqual([k for k, v in result['checks'].items() if not v],
                                 [failed])

    def test_original_thresholds_and_tolerance_are_unchanged(self):
        result = self.decide(clean=np.float64(.979),
                             current=(.90, .75, .60))
        self.assertTrue(result['source_qualified'])
        self.assertAlmostEqual(result['fullV_clean_vs_NoAs30_pp'], -.1)
        self.assertAlmostEqual(result['mid_low_mean_vs_As30_pp'], .5)
        self.assertEqual(result['low_vs_As30_pp'], 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
