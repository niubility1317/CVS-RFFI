import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2SupportMetricQknnV47PolicyTest(unittest.TestCase):
    def test_v47_uses_old_only_scenario_fallback_for_reliable_many_new_support(self):
        from phase2_support_metric_qknn_probe import _adaptive_qknn_overrides

        reliable_many_new = {
            "adaptive_support_min_k": 10.0,
            "adaptive_new_class_count": 20.0,
            "adaptive_support_max_offdiag_proto_sim": 0.982,
            "adaptive_support_p90_offdiag_proto_sim": 0.822,
            "adaptive_support_mean_radius": 0.104,
        }
        low_k_many_new = dict(reliable_many_new)
        low_k_many_new["adaptive_support_min_k"] = 5.0

        reliable = _adaptive_qknn_overrides(
            policy="stable_dualview_v47",
            geometry=reliable_many_new,
            aux_available=True,
        )
        low_k = _adaptive_qknn_overrides(
            policy="stable_dualview_v47",
            geometry=low_k_many_new,
            aux_available=True,
        )

        self.assertEqual(reliable["adaptive_qknn_policy"], "stable_dualview_v47")
        self.assertEqual(reliable["topm"], 1)
        self.assertEqual(reliable["scenario_class_fallback"], "old_only")
        self.assertEqual(reliable["support_loo_pair_rescue_proto_neighbors"], 0)

        self.assertEqual(low_k["topm"], 4)
        self.assertFalse(low_k["scenario_class_fallback"])
        self.assertGreater(low_k["labelprop_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
