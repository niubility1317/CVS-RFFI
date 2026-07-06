import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2SupportMetricQknnV44PolicyTest(unittest.TestCase):
    def test_v44_uses_single_neighbor_when_support_is_reliable_under_many_new_load(self):
        from phase2_support_metric_qknn_probe import _adaptive_qknn_overrides

        geometry = {
            "adaptive_support_min_k": 10.0,
            "adaptive_new_class_count": 20.0,
            "adaptive_support_max_offdiag_proto_sim": 0.982,
            "adaptive_support_p90_offdiag_proto_sim": 0.822,
            "adaptive_support_mean_radius": 0.104,
        }

        overrides = _adaptive_qknn_overrides(
            policy="stable_dualview_v44",
            geometry=geometry,
            aux_available=True,
        )

        self.assertEqual(overrides["adaptive_qknn_policy"], "stable_dualview_v44")
        self.assertEqual(overrides["topm"], 1)
        self.assertGreater(overrides["source_target_transport_weight"], 0.0)
        self.assertEqual(overrides["query_cluster_weight"], 0.0)
        self.assertEqual(overrides["transductive_proto_weight"], 0.0)
        self.assertEqual(overrides["dense_cluster_weight"], 0.0)

    def test_v44_keeps_v42_topm_for_low_k_many_new_support(self):
        from phase2_support_metric_qknn_probe import _adaptive_qknn_overrides

        geometry = {
            "adaptive_support_min_k": 5.0,
            "adaptive_new_class_count": 20.0,
            "adaptive_support_max_offdiag_proto_sim": 0.982,
            "adaptive_support_p90_offdiag_proto_sim": 0.822,
            "adaptive_support_mean_radius": 0.104,
        }

        overrides = _adaptive_qknn_overrides(
            policy="stable_dualview_v44",
            geometry=geometry,
            aux_available=True,
        )

        self.assertEqual(overrides["adaptive_qknn_policy"], "stable_dualview_v44")
        self.assertEqual(overrides["topm"], 4)
        self.assertGreater(overrides["support_loo_pair_rescue_weight"], 0.0)
        self.assertGreater(overrides["source_target_transport_weight"], 0.0)

    def test_v44_uses_single_neighbor_for_k10_n14_pressure_case(self):
        from phase2_support_metric_qknn_probe import _adaptive_qknn_overrides

        geometry = {
            "adaptive_support_min_k": 10.0,
            "adaptive_new_class_count": 14.0,
            "adaptive_support_max_offdiag_proto_sim": 0.982,
            "adaptive_support_p90_offdiag_proto_sim": 0.822,
            "adaptive_support_mean_radius": 0.104,
        }

        overrides = _adaptive_qknn_overrides(
            policy="stable_dualview_v44",
            geometry=geometry,
            aux_available=True,
        )

        self.assertEqual(overrides["topm"], 1)


if __name__ == "__main__":
    unittest.main()
