import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2QknnOldOnlyScenarioFallbackTest(unittest.TestCase):
    def test_old_only_fallback_keeps_missing_new_class_masked(self):
        from phase2_confusion_aware_qknn_probe import _class_scores

        features = np.asarray(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.05, 0.95],
                [-1.0, 0.0],
                [0.0, -1.0],
                [-0.95, 0.05],
            ],
            dtype=np.float64,
        )
        support_indices = np.asarray([0, 1, 2, 3, 4, 5], dtype=int)
        support_labels = np.asarray(["old-a", "old-a", "old-b", "old-b", "new-c", "new-d"], dtype=object)
        query_indices = np.asarray([6], dtype=int)
        scenarios = np.asarray(["s0", "s0", "s1", "s1", "s0", "s0", "s1"], dtype=object)
        labels = ["old-a", "old-b", "new-c", "new-d"]

        scores, _radii, _proto_sim = _class_scores(
            features=features,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            scenarios=scenarios,
            class_labels=labels,
            old_labels={"old-a", "old-b"},
            topm=1,
            proto_mix=0.0,
            radius_norm=0.0,
            old_bias=0.0,
            neg_lambda=0.0,
            neg_threshold=1.0,
            neg_margin=0.0,
            mutual_only=False,
            scenario_aware=True,
            scenario_class_fallback=True,
            scenario_class_fallback_labels={"old-a", "old-b"},
        )

        self.assertGreater(scores[0, labels.index("old-a")], -1e8)
        self.assertGreater(scores[0, labels.index("old-b")], -1e8)
        self.assertLess(scores[0, labels.index("new-c")], -1e8)
        self.assertLess(scores[0, labels.index("new-d")], -1e8)


if __name__ == "__main__":
    unittest.main()
