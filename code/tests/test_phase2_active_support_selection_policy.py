import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2ActiveSupportSelectionPolicyTest(unittest.TestCase):
    def test_old_stable_new_scenario_centroid_splits_roles(self):
        from phase2_qknn_active_support_select import _select_support, _stable_order

        features = np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.7, 0.7],
            ],
            dtype=np.float64,
        )
        features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        candidates = np.arange(features.shape[0], dtype=int)
        scenarios = np.asarray(["clear", "clear", "rain", "rain", "low"], dtype=object)

        old_support = _select_support(
            policy="old_stable_new_scenario_centroid",
            label="old-a",
            role="target_old",
            candidates=candidates,
            features=features,
            scenarios=scenarios,
            source_probs=np.zeros((features.shape[0], 1), dtype=np.float64),
            source_label_to_idx={},
            source_prototypes={},
            k=3,
            seed=123,
        )
        expected_old = _stable_order(candidates, label="target_old:old-a", seed=123)[:3].astype(int).tolist()
        self.assertEqual(old_support, expected_old)

        new_support = _select_support(
            policy="old_stable_new_scenario_centroid",
            label="new-a",
            role="target_unknown",
            candidates=candidates,
            features=features,
            scenarios=scenarios,
            source_probs=np.zeros((features.shape[0], 1), dtype=np.float64),
            source_label_to_idx={},
            source_prototypes={},
            k=3,
            seed=123,
        )
        self.assertEqual({str(scenarios[idx]) for idx in new_support}, {"clear", "rain", "low"})


if __name__ == "__main__":
    unittest.main()
