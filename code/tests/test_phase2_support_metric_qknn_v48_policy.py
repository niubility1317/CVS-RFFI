import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2SupportMetricQknnV48PolicyTest(unittest.TestCase):
    def test_v48_strengthens_support_only_pair_linear_for_reliable_many_new_support(self):
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
            policy="stable_dualview_v48",
            geometry=reliable_many_new,
            aux_available=True,
        )
        low_k = _adaptive_qknn_overrides(
            policy="stable_dualview_v48",
            geometry=low_k_many_new,
            aux_available=True,
        )

        self.assertEqual(reliable["adaptive_qknn_policy"], "stable_dualview_v48")
        self.assertEqual(reliable["topm"], 1)
        self.assertEqual(reliable["scenario_class_fallback"], "old_only")
        self.assertAlmostEqual(reliable["support_loo_pair_linear_weight"], 0.02)
        self.assertAlmostEqual(reliable["support_loo_pair_linear_alpha"], 0.2)
        self.assertAlmostEqual(reliable["support_loo_pair_linear_clip"], 1.5)
        self.assertEqual(reliable["support_loo_pair_linear_scope"], "new")

        self.assertEqual(low_k["topm"], 4)
        self.assertFalse(low_k["scenario_class_fallback"])
        self.assertGreater(low_k["labelprop_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
