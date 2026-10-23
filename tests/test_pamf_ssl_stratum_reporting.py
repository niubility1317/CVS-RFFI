import json
from pathlib import Path
import tempfile
import unittest

from experiments.pamf_ssl_research.report import report
from tests.test_pamf_ssl_goal_counts import fixture, save, SEEDS, VIEWS


def add_receiver(rt):
    paths = [(arm, seed, Path(folder) / 'scores.json')
             for seed in SEEDS for arm, folder in rt['baseline_scores'][str(seed)].items()]
    for row in rt['rows']:
        config = json.loads(Path(row['score_config']).read_text())
        paths.append((row['method'], row['model_seed'], Path(config['output_root']) / 'scores.json'))
    for method, seed, path in paths:
        value = json.loads(path.read_text())
        correct = 60 if method in ('F20', 'F21') else {SEEDS[0]: 65, SEEDS[1]: 55, SEEDS[2]: 60}[seed]
        for view in VIEWS:
            cm = [[0] * 6 for _ in range(6)]
            cm[0][0], cm[0][1] = correct, 100 - correct
            value['results'].append(dict(test='main', readout='raw', view=view,
                dimension='receiver', stratum='synthetic_receiver', count=100,
                accuracy=correct / 100, confusion=cm))
        save(path, value)


class TargetStratumReportingTest(unittest.TestCase):
    def test_paired_seed_changes_are_preserved_even_when_mean_is_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = fixture(root, clean_added=(4, 1680, 3356), six_extra=1)
            add_receiver(rt)
            report(rt, root / 'report')
            value = json.loads((root / 'report/strata.json').read_text())
            row = next(r for r in value['rows'] if r['method'] == 'PSM'
                       and r['view'] == 'clean' and r['dimension'] == 'receiver')
            self.assertEqual(row['delta_F20_pp'], 0.)
            self.assertEqual(row['delta_F20_sample_sd_pp'], 5.)
            self.assertEqual(row['positive_seeds_F20'], 1)
            self.assertEqual(row['correct_s' + str(SEEDS[1])], 55)
            overall = next(r for r in value['rows'] if r['method'] == 'PSM'
                           and r['view'] == 'clean' and r['dimension'] == 'overall')
            self.assertEqual(overall['delta_F21_pp'], 1.)
            self.assertTrue(json.loads((root / 'report/analysis.json').read_text())['goal_met'])

    def test_missing_one_seed_stratum_is_not_silently_averaged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = fixture(root, clean_added=(4, 1680, 3356), six_extra=1)
            add_receiver(rt)
            config = json.loads(Path(rt['rows'][0]['score_config']).read_text())
            path = Path(config['output_root']) / 'scores.json'
            value = json.loads(path.read_text())
            value['results'] = [r for r in value['results'] if r['dimension'] != 'receiver']
            save(path, value)
            with self.assertRaisesRegex(ValueError, 'stratum coverage'):
                report(rt, root / 'report')

    def test_early_controller_state_does_not_open_any_score_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save(root / 'state.json', dict(status='RUNNING'))
            with self.assertRaisesRegex(ValueError, 'All six target-test'):
                report(dict(run_root=str(root)), root / 'report')
            self.assertFalse((root / 'report').exists())
