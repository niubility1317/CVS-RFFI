import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2QknnScenarioClassFallbackTest(unittest.TestCase):
    def test_scenario_aware_keeps_missing_class_mask_without_explicit_fallback(self):
        from phase2_confusion_aware_qknn_probe import _class_scores

        scores, _radii, _proto_sim = _class_scores(
            features=self._features(),
            support_indices=np.asarray([0, 1, 2], dtype=int),
            support_labels=np.asarray(["a", "c", "b"], dtype=object),
            query_indices=np.asarray([3], dtype=int),
            scenarios=self._scenarios(),
            class_labels=["a", "b", "c"],
            old_labels=set(),
            topm=1,
            proto_mix=0.0,
            radius_norm=0.0,
            old_bias=0.0,
            neg_lambda=0.0,
            neg_threshold=2.0,
            neg_margin=0.0,
            mutual_only=False,
            scenario_aware=True,
        )

        self.assertLess(scores[0, 1], -1e8)

    def test_scenario_aware_scores_fall_back_per_missing_class_when_enabled(self):
        from phase2_confusion_aware_qknn_probe import _class_scores

        scores, _radii, _proto_sim = _class_scores(
            features=self._features(),
            support_indices=np.asarray([0, 1, 2], dtype=int),
            support_labels=np.asarray(["a", "c", "b"], dtype=object),
            query_indices=np.asarray([3], dtype=int),
            scenarios=self._scenarios(),
            class_labels=["a", "b", "c"],
            old_labels=set(),
            topm=1,
            proto_mix=0.0,
            radius_norm=0.0,
            old_bias=0.0,
            neg_lambda=0.0,
            neg_threshold=2.0,
            neg_margin=0.0,
            mutual_only=False,
            scenario_aware=True,
            scenario_class_fallback=True,
        )

        self.assertGreater(scores[0, 1], -1e8)
        self.assertGreater(scores[0, 1], scores[0, 0])
        self.assertGreater(scores[0, 1], scores[0, 2])

    @staticmethod
    def _features():
        features = np.asarray(
            [
                [1.0, 0.0, 0.0],  # class a support in clear
                [0.0, 0.0, 1.0],  # class c support in clear
                [0.0, 1.0, 0.0],  # class b support in rain only
                [0.0, 1.0, 0.0],  # class b query in clear
            ],
            dtype=np.float64,
        )
        return features

    @staticmethod
    def _scenarios():
        return np.asarray(
            ["leo_clear_weak", "leo_clear_weak", "leo_rain_weak", "leo_clear_weak"],
            dtype=object,
        )


if __name__ == "__main__":
    unittest.main()
