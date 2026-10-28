import unittest
import numpy as np

from experiments.pamf_phase3_open_world import scores


class Phase3ScoresTests(unittest.TestCase):
    def test_scores_are_finite_and_class_permutation_invariant(self):
        z = np.array([[1., 0.], [0., 1.], [-1., -1.]])
        logits = np.array([[1000., 997.], [3., 8.], [-9., -10.]])
        prototypes = np.eye(2)
        result = scores.knownness_scores(z, logits, prototypes)
        self.assertEqual(result.shape, (3, 5))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result, scores.knownness_scores(z, logits[:, ::-1], prototypes[::-1]))
        self.assertGreater(result[0, 4], result[2, 4])

    def test_scores_do_not_depend_on_other_query_records(self):
        z = np.array([[1., 0.], [0., 1.]])
        logits = np.array([[4., 1.], [2., 5.]])
        batch = scores.knownness_scores(z, logits, np.eye(2))
        single = scores.knownness_scores(z[:1], logits[:1], np.eye(2))
        np.testing.assert_array_equal(batch[:1], single)

    def test_final_score_threshold_retains_source_budget_with_ties(self):
        values = np.repeat(np.arange(100, dtype=float)[:, None], 5, axis=1)
        thresholds = scores.source_thresholds(values, retention=.99)
        self.assertTrue(np.all((values >= thresholds).mean(0) >= .99))
        np.testing.assert_array_equal(thresholds, np.zeros(5))
        tied = scores.source_thresholds(np.ones((100, 5)), retention=.99)
        np.testing.assert_array_equal(tied, np.ones(5))

    def test_rejection_preserves_default_head_and_marks_reject_minus_one(self):
        logits = np.array([[1., 8.], [4., 2.]])
        values = np.array([[.9] * 5, [.1] * 5])
        result = scores.apply_rejection(logits, values, np.full(5, .5))
        np.testing.assert_array_equal(result['closed_prediction'], [1, 0])
        np.testing.assert_array_equal(result['prediction'][0], [1] * 5)
        np.testing.assert_array_equal(result['prediction'][1], [-1] * 5)

    def test_prototypes_require_visible_labeled_source_classes(self):
        z = np.array([[1., 0.], [.8, .2], [0., 1.]])
        prototypes = scores.fit_prototypes(z, np.array([0, 0, 1]), 2)
        self.assertEqual(prototypes.shape, (2, 2))
        np.testing.assert_allclose(np.linalg.norm(prototypes, axis=1), 1.)
        for labels in (np.array([0, -1, 1]), np.array([0, 0, 0])):
            with self.assertRaises(ValueError):
                scores.fit_prototypes(z, labels, 2)

    def test_invalid_score_inputs_fail_before_calibration(self):
        for value in (np.empty((0, 5)), np.full((2, 5), np.nan), np.ones((2, 4))):
            with self.assertRaises(ValueError):
                scores.source_thresholds(value, .99)
        with self.assertRaises(ValueError):
            scores.knownness_scores(np.zeros((2, 3)), np.ones((2, 2)), np.eye(2))


if __name__ == '__main__':
    unittest.main()
