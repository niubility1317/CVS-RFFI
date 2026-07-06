import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2SupportMetricQknnV54PolicyTest(unittest.TestCase):
    def test_v54_keeps_low_k_policy_and_reuses_high_k_v49_guard(self):
        from phase2_support_metric_qknn_probe import _adaptive_qknn_overrides

        many_new_low_k = {
            "adaptive_support_min_k": 5.0,
            "adaptive_new_class_count": 20.0,
            "adaptive_support_max_offdiag_proto_sim": 0.982,
            "adaptive_support_p90_offdiag_proto_sim": 0.822,
            "adaptive_support_mean_radius": 0.104,
        }
        many_new_high_k = dict(many_new_low_k)
        many_new_high_k["adaptive_support_min_k"] = 10.0

        low_k = _adaptive_qknn_overrides(
            policy="stable_dualview_v54",
            geometry=many_new_low_k,
            aux_available=True,
        )
        high_k = _adaptive_qknn_overrides(
            policy="stable_dualview_v54",
            geometry=many_new_high_k,
            aux_available=True,
        )

        self.assertEqual(low_k["adaptive_qknn_requested_policy"], "stable_dualview_v54")
        self.assertEqual(low_k["adaptive_qknn_policy"], "stable_dualview_v54")
        self.assertEqual(low_k["adaptive_qknn_effective_policy"], "stable_dualview_v54")
        self.assertNotIn("scenario_class_fallback", low_k)
        self.assertTrue(low_k["role_balanced_assignment"])
        self.assertGreater(low_k["ridge_head_weight"], 0.0)
        self.assertGreater(low_k["support_quality_weight"], 0.0)
        self.assertEqual(low_k["topm"], 2)
        self.assertEqual(low_k["proto_mix"], 0.45)
        self.assertEqual(low_k["aux_score_weight"], 0.26)

        self.assertEqual(high_k["adaptive_qknn_requested_policy"], "stable_dualview_v54")
        self.assertEqual(high_k["adaptive_qknn_policy"], "stable_dualview_v49")
        self.assertEqual(high_k["adaptive_qknn_effective_policy"], "stable_dualview_v49")
        self.assertEqual(high_k["topm"], 1)
        self.assertEqual(high_k["scenario_class_fallback"], "old_role_only")
        self.assertGreater(high_k["source_target_transport_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
