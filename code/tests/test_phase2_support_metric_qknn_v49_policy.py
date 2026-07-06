import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2SupportMetricQknnV49PolicyTest(unittest.TestCase):
    def test_v49_uses_role_split_old_only_fallback_for_reliable_many_new_support(self):
        from phase2_support_metric_qknn_probe import _adaptive_qknn_overrides

        reliable_many_new = {
            "adaptive_support_min_k": 10.0,
            "adaptive_new_class_count": 20.0,
            "adaptive_support_max_offdiag_proto_sim": 0.975,
            "adaptive_support_p90_offdiag_proto_sim": 0.823,
            "adaptive_support_mean_radius": 0.125,
        }
        low_k_many_new = dict(reliable_many_new)
        low_k_many_new["adaptive_support_min_k"] = 5.0

        reliable = _adaptive_qknn_overrides(
            policy="stable_dualview_v49",
            geometry=reliable_many_new,
            aux_available=True,
        )
        low_k = _adaptive_qknn_overrides(
            policy="stable_dualview_v49",
            geometry=low_k_many_new,
            aux_available=True,
        )

        self.assertEqual(reliable["adaptive_qknn_policy"], "stable_dualview_v49")
        self.assertEqual(reliable["topm"], 1)
        self.assertEqual(reliable["scenario_class_fallback"], "old_role_only")

        self.assertEqual(low_k["topm"], 4)
        self.assertFalse(low_k["scenario_class_fallback"])
        self.assertGreater(low_k["labelprop_weight"], 0.0)

    def test_role_split_fallback_keeps_strict_new_scores(self):
        from phase2_support_metric_qknn_probe import _role_split_old_only_fallback_scores

        fallback = np.array(
            [
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
                [30.0, 31.0, 32.0, 33.0],
                [40.0, 41.0, 42.0, 43.0],
            ],
            dtype=np.float64,
        )
        strict = fallback + 1000.0

        mixed = _role_split_old_only_fallback_scores(
            fallback,
            strict_scores=strict,
            old_query_count=2,
            old_label_count=2,
        )

        np.testing.assert_array_equal(mixed[:2, :2], fallback[:2, :2])
        np.testing.assert_array_equal(mixed[2:, 2:], strict[2:, 2:])
        np.testing.assert_array_equal(mixed[:2, 2:], fallback[:2, 2:])
        np.testing.assert_array_equal(mixed[2:, :2], fallback[2:, :2])


if __name__ == "__main__":
    unittest.main()
