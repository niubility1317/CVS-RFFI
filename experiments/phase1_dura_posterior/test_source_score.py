import unittest

import numpy as np

from experiments.phase1_dura_posterior.score_source import score_cell, validate_manifest


class ScoreTest(unittest.TestCase):
    def test_scoring_rejects_incomplete_predictions_and_truth_access(self):
        fixed = dict(status='SOURCE_PREDICTIONS_FIXED', commit='fixed', count=672, fit_ids=8192,
                     F_frozen=True, DURA_frozen=True, gate_frozen=True, posterior_frozen=True,
                     TX_labels_read=0, U_s_truth_reads=0, V_reads=0, test_reads=0,
                     model_updates_after_prediction=0, posterior_kind='mean')
        self.assertEqual(validate_manifest(fixed, 'fixed'), 'mean')
        for change in ({'count': 671}, {'U_s_truth_reads': 1}, {'test_reads': 1},
                       {'model_updates_after_prediction': 1}, {'posterior_kind': 'unregistered'}):
            with self.assertRaises(ValueError):
                validate_manifest(dict(fixed, **change), 'fixed')

    def test_paired_repair_and_damage_use_original_dura(self):
        # Two repairs and one damage relative to DURA; another row is wrong in both.
        truth = np.array([0, 1, 2, 0, 1])
        dura = np.array([1, 0, 2, 0, 0])
        candidate = np.array([0, 1, 0, 0, 0])
        mask = np.ones(5, dtype=bool)
        result = score_cell(truth, dura, candidate, mask)
        self.assertEqual(result, dict(count=5, dura_correct=2, candidate_correct=3,
                                      repairs=2, damages=1))

    def test_unselected_truth_cannot_affect_score(self):
        truth = np.array([0, 1, -1])
        base = np.array([1, 1, 0])
        new = np.array([0, 1, 2])
        selected = np.array([True, True, False])
        a = score_cell(truth, base, new, selected)
        truth[2] = 2
        self.assertEqual(a, score_cell(truth, base, new, selected))
        self.assertEqual(a['repairs'], 1)
        self.assertEqual(a['damages'], 0)


if __name__ == '__main__':
    unittest.main()
