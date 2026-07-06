import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2SupportMetricQknnV50PolicyTest(unittest.TestCase):
    def test_v50_extends_v49_with_conservative_top2_gate_only_for_reliable_many_new_support(self):
        from phase2_support_metric_qknn_probe import _adaptive_qknn_overrides, _top2_pair_gate_policy_settings

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
            policy="stable_dualview_v50",
            geometry=reliable_many_new,
            aux_available=True,
        )
        low_k = _adaptive_qknn_overrides(
            policy="stable_dualview_v50",
            geometry=low_k_many_new,
            aux_available=True,
        )

        self.assertEqual(reliable["adaptive_qknn_policy"], "stable_dualview_v50")
        self.assertEqual(reliable["topm"], 1)
        self.assertEqual(reliable["scenario_class_fallback"], "old_role_only")
        self.assertFalse(low_k["scenario_class_fallback"])
        self.assertEqual(low_k["topm"], 4)

        enabled = _top2_pair_gate_policy_settings(
            policy="stable_dualview_v50",
            new_class_count=20,
            min_support=10,
        )
        disabled = _top2_pair_gate_policy_settings(
            policy="stable_dualview_v50",
            new_class_count=20,
            min_support=5,
        )

        self.assertGreater(enabled["weight"], 0.0)
        self.assertGreater(enabled["top_pairs"], 0)
        self.assertEqual(enabled["query_pair_weight"], 0.0)
        self.assertLessEqual(enabled["weight"], 0.025)
        self.assertEqual(disabled["weight"], 0.0)
        self.assertEqual(disabled["top_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
