import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2ScenarioResidualCompletionTest(unittest.TestCase):
    def test_boosts_new_class_missing_query_scenario_from_support_residual(self):
        from phase2_support_metric_qknn_probe import _scenario_residual_completion_scores

        features = np.asarray(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )
        support_indices = np.asarray([0, 1, 2, 3, 4], dtype=int)
        query_indices = np.asarray([5, 6], dtype=int)
        support_labels = np.asarray(["new-a", "new-a", "new-b", "new-b", "old-a"], dtype=object)
        scenarios = np.asarray(
            ["rain", "rain", "clear", "rain", "clear", "rain", "clear"],
            dtype=object,
        )
        scores = np.zeros((2, 3), dtype=np.float64)

        adjusted, count, stored = _scenario_residual_completion_scores(
            scores,
            features=features,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            scenarios=scenarios,
            class_labels=["new-a", "new-b", "old-a"],
            old_labels=["old-a"],
            new_labels=["new-a", "new-b"],
            weight=0.5,
            min_classes=2,
            clip=1.0,
            scope="new",
        )

        self.assertGreater(count, 0)
        self.assertGreater(stored, 0)
        self.assertGreater(adjusted[1, 0], scores[1, 0])
        self.assertEqual(adjusted[1, 2], scores[1, 2])

    def test_zero_weight_preserves_scores(self):
        from phase2_support_metric_qknn_probe import _scenario_residual_completion_scores

        scores = np.asarray([[0.2, 0.1]], dtype=np.float64)
        adjusted, count, stored = _scenario_residual_completion_scores(
            scores,
            features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            support_indices=np.asarray([0], dtype=int),
            support_labels=np.asarray(["new-a"], dtype=object),
            query_indices=np.asarray([1], dtype=int),
            scenarios=np.asarray(["clear", "clear"], dtype=object),
            class_labels=["new-a", "new-b"],
            old_labels=[],
            new_labels=["new-a", "new-b"],
            weight=0.0,
            min_classes=2,
            clip=1.0,
            scope="new",
        )

        np.testing.assert_allclose(adjusted, scores)
        self.assertEqual(count, 0)
        self.assertEqual(stored, 0)

    def test_replaces_hard_masked_missing_scenario_score(self):
        from phase2_support_metric_qknn_probe import _scenario_residual_completion_scores

        features = np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )
        adjusted, count, stored = _scenario_residual_completion_scores(
            np.asarray([[-1.0e9, 0.2]], dtype=np.float64),
            features=features,
            support_indices=np.asarray([0, 1, 2], dtype=int),
            support_labels=np.asarray(["new-a", "new-b", "new-b"], dtype=object),
            query_indices=np.asarray([3], dtype=int),
            scenarios=np.asarray(["rain", "clear", "rain", "clear"], dtype=object),
            class_labels=["new-a", "new-b"],
            old_labels=[],
            new_labels=["new-a", "new-b"],
            weight=1.0,
            min_classes=1,
            clip=1.0,
            scope="new",
        )

        self.assertGreater(count, 0)
        self.assertGreater(stored, 0)
        self.assertGreater(adjusted[0, 0], -1.0e6)

    def test_scenario_proto_refine_sharpens_same_scenario_pair(self):
        from phase2_support_metric_qknn_probe import _scenario_proto_refine_scores

        features = np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )
        scores = np.zeros((1, 2), dtype=np.float64)

        adjusted, count, stored = _scenario_proto_refine_scores(
            scores,
            features=features,
            support_indices=np.asarray([0, 1, 2, 3], dtype=int),
            support_labels=np.asarray(["new-a", "new-a", "new-b", "new-b"], dtype=object),
            query_indices=np.asarray([4], dtype=int),
            scenarios=np.asarray(["clear", "clear", "clear", "clear", "clear"], dtype=object),
            class_labels=["new-a", "new-b"],
            old_labels=[],
            new_labels=["new-a", "new-b"],
            weight=0.2,
            min_support=1,
            clip=1.0,
            scope="new",
        )

        self.assertGreater(count, 0)
        self.assertGreater(stored, 0)
        self.assertGreater(adjusted[0, 0], adjusted[0, 1])

    def test_scenario_pair_refine_learns_same_scenario_boundary(self):
        from phase2_support_metric_qknn_probe import _scenario_pair_refine_scores

        features = np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )
        scores = np.zeros((1, 2), dtype=np.float64)

        adjusted, count, stored, pairs = _scenario_pair_refine_scores(
            scores,
            features=features,
            support_indices=np.asarray([0, 1, 2, 3], dtype=int),
            support_labels=np.asarray(["new-a", "new-a", "new-b", "new-b"], dtype=object),
            query_indices=np.asarray([4], dtype=int),
            scenarios=np.asarray(["clear", "clear", "clear", "clear", "clear"], dtype=object),
            class_labels=["new-a", "new-b"],
            old_labels=[],
            new_labels=["new-a", "new-b"],
            weight=0.5,
            top_pairs=1,
            min_support=2,
            min_similarity=-1.0,
            alpha=0.01,
            clip=1.0,
            query_topm=0,
            scope="new",
            center="none",
        )

        self.assertGreater(count, 0)
        self.assertGreater(stored, 0)
        self.assertIn("new-a<->new-b", pairs)
        self.assertGreater(adjusted[0, 0], adjusted[0, 1])


if __name__ == "__main__":
    unittest.main()
