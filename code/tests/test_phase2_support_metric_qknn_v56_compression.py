import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2SupportMetricQknnV56CompressionTest(unittest.TestCase):
    def test_centroid_budget_keeps_representative_codes_per_class(self):
        from phase2_support_metric_qknn_probe import _compress_support_codes

        features = np.asarray(
            [
                [1.00, 0.00],
                [0.95, 0.05],
                [0.40, 0.90],
                [-1.00, 0.00],
                [-0.95, 0.05],
                [-0.40, 0.90],
            ],
            dtype=float,
        )
        support_indices = np.asarray([0, 1, 2, 3, 4, 5], dtype=int)
        support_labels = np.asarray(["old-a", "old-a", "old-a", "new-b", "new-b", "new-b"], dtype=object)
        scenarios = np.asarray(["r1", "r1", "r2", "r1", "r1", "r2"], dtype=object)

        kept_indices, kept_labels = _compress_support_codes(
            features=features,
            support_indices=support_indices,
            support_labels=support_labels,
            scenarios=scenarios,
            per_class=1,
            mode="centroid",
        )

        self.assertEqual(kept_indices.tolist(), [1, 4])
        self.assertEqual(kept_labels.tolist(), ["old-a", "new-b"])

    def test_scenario_centroid_budget_spreads_across_scenarios_when_available(self):
        from phase2_support_metric_qknn_probe import _compress_support_codes

        features = np.asarray(
            [
                [1.00, 0.00],
                [0.98, 0.02],
                [0.20, 0.98],
                [-1.00, 0.00],
                [-0.98, 0.02],
                [-0.20, 0.98],
            ],
            dtype=float,
        )
        support_indices = np.asarray([0, 1, 2, 3, 4, 5], dtype=int)
        support_labels = np.asarray(["old-a", "old-a", "old-a", "new-b", "new-b", "new-b"], dtype=object)
        scenarios = np.asarray(["r1", "r1", "r2", "r1", "r1", "r2"], dtype=object)

        kept_indices, kept_labels = _compress_support_codes(
            features=features,
            support_indices=support_indices,
            support_labels=support_labels,
            scenarios=scenarios,
            per_class=2,
            mode="scenario_centroid",
        )

        self.assertEqual(kept_indices.tolist(), [1, 2, 4, 5])
        self.assertEqual(kept_labels.tolist(), ["old-a", "old-a", "new-b", "new-b"])

    def test_centroid_hard_neighbor_budget_keeps_center_and_boundary_codes(self):
        from phase2_support_metric_qknn_probe import _compress_support_codes

        features = np.asarray(
            [
                [1.00, 0.00],
                [0.96, 0.04],
                [0.20, 0.98],
                [-1.00, 0.00],
                [-0.96, 0.04],
                [0.30, 0.95],
            ],
            dtype=float,
        )
        support_indices = np.asarray([0, 1, 2, 3, 4, 5], dtype=int)
        support_labels = np.asarray(["old-a", "old-a", "old-a", "new-b", "new-b", "new-b"], dtype=object)
        scenarios = np.asarray(["r1", "r1", "r2", "r1", "r1", "r2"], dtype=object)

        kept_indices, kept_labels = _compress_support_codes(
            features=features,
            support_indices=support_indices,
            support_labels=support_labels,
            scenarios=scenarios,
            per_class=2,
            mode="centroid_hard_neighbor",
        )

        self.assertEqual(kept_indices.tolist(), [1, 2, 4, 5])
        self.assertEqual(kept_labels.tolist(), ["old-a", "old-a", "new-b", "new-b"])

    def test_centroid_hard_diverse_budget_keeps_center_boundary_and_spread_codes(self):
        from phase2_support_metric_qknn_probe import _compress_support_codes

        features = np.asarray(
            [
                [1.00, 0.00],
                [0.90, 0.10],
                [0.00, 1.00],
                [0.00, -1.00],
                [-1.00, 0.00],
                [-0.90, 0.10],
                [0.00, 1.00],
                [0.00, -1.00],
            ],
            dtype=float,
        )
        support_indices = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)
        support_labels = np.asarray(
            ["old-a", "old-a", "old-a", "old-a", "new-b", "new-b", "new-b", "new-b"],
            dtype=object,
        )
        scenarios = np.asarray(["r1", "r1", "r2", "r3", "r1", "r1", "r2", "r3"], dtype=object)

        kept_indices, kept_labels = _compress_support_codes(
            features=features,
            support_indices=support_indices,
            support_labels=support_labels,
            scenarios=scenarios,
            per_class=3,
            mode="centroid_hard_diverse",
        )

        self.assertEqual(kept_indices.tolist(), [0, 2, 3, 4, 6, 7])
        self.assertEqual(kept_labels.tolist(), ["old-a", "old-a", "old-a", "new-b", "new-b", "new-b"])

    def test_v56_keeps_support_code_budget_off_and_high_k_uses_v49_guard(self):
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
            policy="stable_dualview_v56",
            geometry=many_new_low_k,
            aux_available=True,
        )
        high_k = _adaptive_qknn_overrides(
            policy="stable_dualview_v56",
            geometry=many_new_high_k,
            aux_available=True,
        )

        self.assertEqual(low_k["adaptive_qknn_requested_policy"], "stable_dualview_v56")
        self.assertEqual(low_k["adaptive_qknn_policy"], "stable_dualview_v56")
        self.assertNotIn("support_code_budget_per_class", low_k)
        self.assertNotIn("support_code_budget_mode", low_k)

        self.assertEqual(high_k["adaptive_qknn_requested_policy"], "stable_dualview_v56")
        self.assertEqual(high_k["adaptive_qknn_policy"], "stable_dualview_v49")
        self.assertEqual(high_k["topm"], 1)
        self.assertEqual(high_k["scenario_class_fallback"], "old_role_only")

    def test_support_proto_anchor_recovers_class_score_without_raw_support_codes(self):
        from phase2_support_metric_qknn_probe import _support_proto_anchor_scores

        features = np.asarray(
            [
                [1.00, 0.00],
                [0.80, 0.20],
                [0.00, 1.00],
                [0.20, 0.80],
                [0.95, 0.05],
            ],
            dtype=float,
        )
        support_indices = np.asarray([0, 1, 2, 3], dtype=int)
        support_labels = np.asarray(["old-a", "old-a", "new-b", "new-b"], dtype=object)
        query_indices = np.asarray([4], dtype=int)
        compressed_scores = np.asarray([[0.10, 0.20]], dtype=float)

        adjusted, stored_scalars = _support_proto_anchor_scores(
            compressed_scores,
            features=features,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            class_labels=["old-a", "new-b"],
            old_labels={"old-a"},
            weight=0.50,
            radius_norm=0.0,
            old_bias=0.0,
            clip=2.0,
        )

        self.assertEqual(stored_scalars, 4)
        self.assertGreater(adjusted[0, 0], adjusted[0, 1])


if __name__ == "__main__":
    unittest.main()
