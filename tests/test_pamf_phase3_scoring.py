import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class Phase3ScoringTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/scoring.py'
        self.assertTrue(path.is_file(), 'Phase3 independent truth-side scorer is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.scoring')

    def fixture(self, root, *, order=None, accepted=None, scores=None, y=None, closed=None, scenario=None):
        pred = root / 'prediction'
        pred.mkdir()
        ids = np.asarray(['opaque-a', 'opaque-b', 'opaque-c', 'opaque-d'])
        y = np.asarray([0, 1, 6, 7] if y is None else y, dtype=np.int64)
        closed = np.asarray([0, 1, 0, 1] if closed is None else closed, dtype=np.int64)
        accepted = np.asarray([True, True, False, False] if accepted is None else accepted, dtype=bool)[:, None]
        knownness = np.asarray([.9, .8, .2, .1] if scores is None else scores, dtype=np.float64)[:, None]
        np.savez(pred / 'predictions.npz', ids=ids, method_names=np.asarray(['distance']),
                 closed_prediction=closed, knownness=knownness, accepted=accepted,
                 prediction=np.where(accepted, closed[:, None], -1))
        (pred / 'complete.json').write_text(json.dumps(dict(status='ALL_PREDICTIONS_COMPLETE', truth_read=False, query_fit=False)), encoding='utf-8')
        order = np.arange(4) if order is None else np.asarray(order)
        scen = np.asarray(['low', 'urban', 'low', 'urban'] if scenario is None else scenario)
        truth = root / 'truth.npz'
        np.savez(truth, ids=ids[order], y=y[order], receiver=np.asarray([2, 3, 2, 3], dtype=np.int64)[order], scenario=scen[order])
        return pred / 'predictions.npz', truth

    def replace_npz(self, path, **changes):
        with np.load(path, allow_pickle=False) as data:
            values = {k: data[k].copy() for k in data.files}
        values.update(changes)
        np.savez(path, **values)

    def test_perfect_reordered_truth_scores_and_passes(self):
        scorer = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, truth = self.fixture(root, order=[3, 1, 0, 2])
            result = scorer.score(pred, truth, root / 'scores')
            row = result['methods']['distance']
            self.assertEqual(row['overall']['known_closed_accuracy'], 1.)
            self.assertEqual(row['overall']['known_open_accuracy'], 1.)
            self.assertEqual(row['overall']['known_macro_frr'], 0.)
            self.assertEqual(row['overall']['unknown_macro_rejection'], 1.)
            self.assertEqual(row['overall']['auroc_known'], 1.)
            self.assertEqual(row['overall']['oscr'], 1.)
            self.assertTrue(row['goals']['pass'])
            self.assertEqual(set(row['by_receiver']), {'2', '3'})
            self.assertEqual(set(row['by_scenario']), {'low', 'urban'})
            self.assertEqual(set(row['by_unknown_transmitter']), {'6', '7'})
            self.assertTrue((root / 'scores/scores.json').is_file())
            self.assertEqual(json.loads((root / 'scores/complete.json').read_text())['status'], 'SCORED_COMPLETE')
            self.assertLess(row['overall']['confidence_intervals95']['unknown_rejection'][0], 1.)

    def test_missing_completion_rejects_before_truth_is_opened(self):
        scorer = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, truth = self.fixture(root)
            (pred.parent / 'complete.json').unlink()
            real_load = np.load
            def guarded_load(path, *args, **kwargs):
                if Path(path) == truth:
                    self.fail('Truth was opened before prediction completion')
                return real_load(path, *args, **kwargs)
            with patch.object(scorer.np, 'load', side_effect=guarded_load), self.assertRaises(ValueError):
                scorer.score(pred, truth, root / 'scores')

    def test_incomplete_or_truth_exposed_marker_keeps_truth_closed(self):
        scorer = self.module()
        for change in ({'status': 'RUNNING'}, {'truth_read': True}, {'query_fit': True}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                pred, truth = self.fixture(root)
                marker = dict(status='ALL_PREDICTIONS_COMPLETE', truth_read=False, query_fit=False)
                marker.update(change)
                (pred.parent / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')
                with patch.object(scorer.np, 'load', side_effect=AssertionError('NPZ must remain closed')), self.assertRaises(ValueError):
                    scorer.score(pred, truth, root / 'scores')

    def test_duplicate_or_missing_ids_fail(self):
        scorer = self.module()
        for side, ids in (('pred', ['a', 'a', 'c', 'd']), ('truth', ['a', 'a', 'c', 'd']), ('truth', ['a', 'b', 'c', 'missing'])):
            with self.subTest(side=side, ids=ids), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                pred, truth = self.fixture(root)
                self.replace_npz(pred if side == 'pred' else truth, ids=np.asarray(ids))
                with self.assertRaises(ValueError):
                    scorer.score(pred, truth, root / 'scores')

    def test_invalid_predictions_and_dimensions_fail_before_truth(self):
        scorer = self.module()
        changes = [dict(prediction=np.zeros((4, 1), dtype=np.int64)),
                   dict(accepted=np.ones((4, 1), dtype=np.int64)),
                   dict(knownness=np.ones((4, 2))),
                   dict(knownness=np.asarray([[.9], [.8], [float('nan')], [.1]])),
                   dict(closed_prediction=np.asarray([0, 1, 6, 1])),
                   dict(method_names=np.asarray(['distance', 'distance']))]
        for change in changes:
            with self.subTest(keys=list(change)), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                pred, truth = self.fixture(root)
                self.replace_npz(pred, **change)
                real_load = np.load
                def guarded_load(path, *args, **kwargs):
                    if Path(path) == truth:
                        self.fail('Invalid predictions must be rejected before truth IO')
                    return real_load(path, *args, **kwargs)
                with patch.object(scorer.np, 'load', side_effect=guarded_load), self.assertRaises(ValueError):
                    scorer.score(pred, truth, root / 'scores')

    def test_all_rejected_fails_despite_perfect_unknown_rejection(self):
        scorer = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, truth = self.fixture(root, accepted=[False] * 4)
            row = scorer.score(pred, truth, root / 'scores')['methods']['distance']
            self.assertEqual(row['overall']['unknown_macro_rejection'], 1.)
            self.assertEqual(row['overall']['known_open_accuracy'], 0.)
            self.assertEqual(row['overall']['known_macro_frr'], 1.)
            self.assertFalse(row['goals']['pass'])

    def test_empty_known_or_unknown_is_null_and_cannot_pass(self):
        scorer = self.module()
        for y in ([0, 1, 2, 3], [6, 7, 8, 9]):
            with self.subTest(y=y), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                pred, truth = self.fixture(root, y=y)
                row = scorer.score(pred, truth, root / 'scores')['methods']['distance']
                self.assertFalse(row['goals']['pass'])
                self.assertIsNone(row['overall']['auroc_known'])
                self.assertIsNone(row['overall']['oscr'])
                if y[0] == 0:
                    self.assertIsNone(row['overall']['unknown_macro_rejection'])
                else:
                    self.assertIsNone(row['overall']['known_macro_frr'])

    def test_ties_and_misclassification_have_correct_oscr(self):
        scorer = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, truth = self.fixture(root, scores=[.5] * 4, closed=[0, 0, 0, 0])
            row = scorer.score(pred, truth, root / 'scores')['methods']['distance']['overall']
            self.assertAlmostEqual(row['auroc_known'], .5)
            self.assertAlmostEqual(row['oscr'], .25)
            self.assertEqual(row['known_closed_accuracy'], .5)

    def test_macro_rejection_is_not_frequency_weighted(self):
        scorer = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, truth = self.fixture(root, y=[0, 6, 6, 7], accepted=[True, False, False, True])
            row = scorer.score(pred, truth, root / 'scores')['methods']['distance']['overall']
            self.assertAlmostEqual(row['unknown_macro_rejection'], .5)
            self.assertAlmostEqual(row['unknown_far'], 1. / 3.)

    def test_each_scenario_unknown_goal_is_required(self):
        scorer = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, truth = self.fixture(root, scenario=['known_only', 'known_only', 'unknown_only', 'unknown_only'])
            row = scorer.score(pred, truth, root / 'scores')['methods']['distance']
            self.assertEqual(row['overall']['unknown_macro_rejection'], 1.)
            self.assertFalse(row['goals']['each_scenario_unknown_rejection_at_least_95'])
            self.assertFalse(row['goals']['pass'])

    def test_output_directory_cannot_be_overwritten(self):
        scorer = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, truth = self.fixture(root)
            output = root / 'scores'
            output.mkdir()
            with self.assertRaises(FileExistsError):
                scorer.score(pred, truth, output)

    def test_large_rank_metrics_do_not_allocate_pairwise_matrix(self):
        scorer = self.module()
        count = 100000
        known = np.arange(count) < count // 2
        scores = np.where(known, 1., 0.)
        auroc, oscr = scorer.rank_metrics(known, known, scores)
        self.assertEqual(auroc, 1.)
        self.assertEqual(oscr, 1.)

    def test_cli_produces_independent_score_artifacts(self):
        self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, truth = self.fixture(root, order=[2, 0, 3, 1])
            command = [sys.executable, '-B', '-m', 'experiments.pamf_phase3_open_world.scoring',
                       '--predictions', str(pred), '--truth', str(truth), '--output', str(root / 'cli_scores')]
            completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                       text=True, encoding='utf-8', capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary['status'], 'SCORED_COMPLETE')
            self.assertTrue(summary['goals']['distance'])
            self.assertTrue((root / 'cli_scores/scores.json').is_file())


if __name__ == '__main__':
    unittest.main()
