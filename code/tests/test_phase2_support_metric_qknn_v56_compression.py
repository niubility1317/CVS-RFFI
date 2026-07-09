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
    def test_target_new_role_reports_current_semantics_when_legacy_role_is_used_for_sampling(self):
        from phase2_support_metric_qknn_probe import _resolve_new_split_role

        roles = np.asarray(["target_old", "target_unknown", "target_unknown"], dtype=object)

        resolved = _resolve_new_split_role(roles=roles, requested_role="target_new")

        self.assertEqual(resolved.requested_role, "target_new")
        self.assertEqual(resolved.selection_role, "target_unknown")
        self.assertTrue(resolved.used_legacy_target_new_role)

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

    def test_role_specific_budget_can_compress_old_while_preserving_new_codes(self):
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
            per_class=0,
            mode="centroid_hard_diverse",
            old_labels={"old-a"},
            old_per_class=1,
            new_per_class=3,
        )

        self.assertEqual(kept_indices.tolist(), [0, 4, 6, 7])
        self.assertEqual(kept_labels.tolist(), ["old-a", "new-b", "new-b", "new-b"])

    def test_new_role_protection_keeps_high_radius_new_class_uncompressed(self):
        from phase2_support_metric_qknn_probe import _compress_support_codes

        features = np.asarray(
            [
                [1.00, 0.00],
                [0.95, 0.05],
                [0.90, 0.10],
                [1.00, 0.00],
                [0.00, 1.00],
                [0.00, -1.00],
                [1.00, 0.00],
                [-1.00, 0.00],
                [-1.00, 0.00],
                [-0.95, 0.05],
                [-0.90, 0.10],
            ],
            dtype=float,
        )
        support_indices = np.arange(features.shape[0], dtype=int)
        support_labels = np.asarray(
            [
                "old-a",
                "old-a",
                "old-a",
                "new-risk",
                "new-risk",
                "new-risk",
                "new-risk",
                "new-stable",
                "new-stable",
                "new-stable",
                "new-stable",
            ],
            dtype=object,
        )
        scenarios = np.asarray(["r1"] * features.shape[0], dtype=object)

        kept_indices, kept_labels = _compress_support_codes(
            features=features,
            support_indices=support_indices,
            support_labels=support_labels,
            scenarios=scenarios,
            per_class=0,
            mode="centroid_hard_diverse",
            old_labels={"old-a"},
            old_per_class=2,
            new_per_class=2,
            new_protect_top_classes=1,
            new_protect_metric="radius",
        )

        self.assertEqual(kept_labels.tolist().count("old-a"), 2)
        self.assertEqual(kept_labels.tolist().count("new-risk"), 4)
        self.assertEqual(kept_labels.tolist().count("new-stable"), 2)
        self.assertTrue({3, 4, 5, 6}.issubset(set(kept_indices.tolist())))

    def test_new_role_extra_budget_gives_medium_risk_class_more_codes(self):
        from phase2_support_metric_qknn_probe import _compress_support_codes

        features = np.asarray(
            [
                [1.00, 0.00],
                [0.95, 0.05],
                [0.90, 0.10],
                [1.00, 0.00],
                [0.00, 1.00],
                [0.00, -1.00],
                [-1.00, 0.00],
                [0.00, 1.00],
                [0.50, 0.86],
                [-0.50, 0.86],
                [0.00, 0.95],
                [-1.00, 0.00],
                [-0.98, 0.02],
                [-0.96, 0.04],
                [-0.94, 0.06],
            ],
            dtype=float,
        )
        support_indices = np.arange(features.shape[0], dtype=int)
        support_labels = np.asarray(
            [
                "old-a",
                "old-a",
                "old-a",
                "new-risk",
                "new-risk",
                "new-risk",
                "new-risk",
                "new-medium",
                "new-medium",
                "new-medium",
                "new-medium",
                "new-stable",
                "new-stable",
                "new-stable",
                "new-stable",
            ],
            dtype=object,
        )
        scenarios = np.asarray(["r1"] * features.shape[0], dtype=object)

        kept_indices, kept_labels = _compress_support_codes(
            features=features,
            support_indices=support_indices,
            support_labels=support_labels,
            scenarios=scenarios,
            per_class=0,
            mode="centroid_hard_diverse",
            old_labels={"old-a"},
            old_per_class=2,
            new_per_class=2,
            new_protect_top_classes=1,
            new_protect_metric="radius",
            new_extra_budget_top_classes=2,
            new_extra_budget_per_class=3,
        )

        self.assertEqual(kept_labels.tolist().count("old-a"), 2)
        self.assertEqual(kept_labels.tolist().count("new-risk"), 4)
        self.assertEqual(kept_labels.tolist().count("new-medium"), 3)
        self.assertEqual(kept_labels.tolist().count("new-stable"), 2)
        self.assertTrue({3, 4, 5, 6}.issubset(set(kept_indices.tolist())))

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

    def test_v59_applies_hard_diverse_compression_overlay_only_for_k10(self):
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
            policy="stable_dualview_v59",
            geometry=many_new_low_k,
            aux_available=True,
        )
        high_k = _adaptive_qknn_overrides(
            policy="stable_dualview_v59",
            geometry=many_new_high_k,
            aux_available=True,
        )

        self.assertEqual(low_k["adaptive_qknn_requested_policy"], "stable_dualview_v59")
        self.assertEqual(low_k["adaptive_qknn_policy"], "stable_dualview_v59")
        self.assertNotIn("support_code_budget_per_class", low_k)
        self.assertNotIn("support_code_budget_mode", low_k)
        self.assertNotIn("transform_mode", low_k)
        self.assertNotIn("topm", low_k)
        self.assertNotIn("proto_mix", low_k)

        self.assertEqual(high_k["adaptive_qknn_requested_policy"], "stable_dualview_v59")
        self.assertEqual(high_k["adaptive_qknn_policy"], "stable_dualview_v59")
        self.assertEqual(high_k["support_code_budget_per_class"], 8)
        self.assertEqual(high_k["support_code_budget_mode"], "centroid_hard_diverse")
        self.assertNotIn("transform_mode", high_k)
        self.assertNotIn("topm", high_k)
        self.assertNotIn("proto_mix", high_k)

    def test_v63_applies_old_role_compression_and_local_competition_only_for_k10(self):
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
            policy="stable_dualview_v63",
            geometry=many_new_low_k,
            aux_available=True,
        )
        high_k = _adaptive_qknn_overrides(
            policy="stable_dualview_v63",
            geometry=many_new_high_k,
            aux_available=True,
        )

        self.assertEqual(low_k["adaptive_qknn_requested_policy"], "stable_dualview_v63")
        self.assertEqual(low_k["adaptive_qknn_policy"], "stable_dualview_v63")
        self.assertNotIn("support_code_old_budget_per_class", low_k)
        self.assertNotIn("local_competition_weight", low_k)
        self.assertNotIn("transform_mode", low_k)

        self.assertEqual(high_k["adaptive_qknn_requested_policy"], "stable_dualview_v63")
        self.assertEqual(high_k["adaptive_qknn_policy"], "stable_dualview_v63")
        self.assertEqual(high_k["support_code_budget_per_class"], 0)
        self.assertEqual(high_k["support_code_budget_mode"], "centroid_hard_diverse")
        self.assertEqual(high_k["support_code_old_budget_per_class"], 5)
        self.assertEqual(high_k["support_code_new_budget_per_class"], 0)
        self.assertEqual(high_k["local_competition_weight"], 0.02)
        self.assertEqual(high_k["local_competition_k"], 5)
        self.assertEqual(high_k["local_competition_clip"], 2.0)
        self.assertEqual(high_k["local_competition_scope"], "role")
        self.assertNotIn("transform_mode", high_k)
        self.assertNotIn("topm", high_k)
        self.assertNotIn("proto_mix", high_k)

    def test_v66_applies_weak_new_class_protected_compression_only_for_k10(self):
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
            policy="stable_dualview_v66",
            geometry=many_new_low_k,
            aux_available=True,
        )
        high_k = _adaptive_qknn_overrides(
            policy="stable_dualview_v66",
            geometry=many_new_high_k,
            aux_available=True,
        )

        self.assertEqual(low_k["adaptive_qknn_requested_policy"], "stable_dualview_v66")
        self.assertEqual(low_k["adaptive_qknn_policy"], "stable_dualview_v66")
        self.assertNotIn("support_code_old_budget_per_class", low_k)
        self.assertNotIn("support_code_new_protect_top_classes", low_k)
        self.assertNotIn("local_competition_weight", low_k)
        self.assertNotIn("transform_mode", low_k)

        self.assertEqual(high_k["adaptive_qknn_requested_policy"], "stable_dualview_v66")
        self.assertEqual(high_k["adaptive_qknn_policy"], "stable_dualview_v66")
        self.assertEqual(high_k["support_code_budget_per_class"], 0)
        self.assertEqual(high_k["support_code_budget_mode"], "centroid_hard_diverse")
        self.assertEqual(high_k["support_code_old_budget_per_class"], 5)
        self.assertEqual(high_k["support_code_new_budget_per_class"], 9)
        self.assertEqual(high_k["support_code_new_protect_top_classes"], 8)
        self.assertEqual(high_k["support_code_new_protect_metric"], "radius")
        self.assertEqual(high_k["local_competition_weight"], 0.02)
        self.assertEqual(high_k["local_competition_k"], 5)
        self.assertEqual(high_k["local_competition_clip"], 2.0)
        self.assertEqual(high_k["local_competition_scope"], "role")
        self.assertNotIn("transform_mode", high_k)
        self.assertNotIn("topm", high_k)
        self.assertNotIn("proto_mix", high_k)

    def test_v67_applies_deeper_weak_new_class_protected_compression_only_for_k10(self):
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
            policy="stable_dualview_v67",
            geometry=many_new_low_k,
            aux_available=True,
        )
        high_k = _adaptive_qknn_overrides(
            policy="stable_dualview_v67",
            geometry=many_new_high_k,
            aux_available=True,
        )

        self.assertEqual(low_k["adaptive_qknn_requested_policy"], "stable_dualview_v67")
        self.assertEqual(low_k["adaptive_qknn_policy"], "stable_dualview_v67")
        self.assertNotIn("support_code_old_budget_per_class", low_k)
        self.assertNotIn("support_code_new_protect_top_classes", low_k)
        self.assertNotIn("local_competition_weight", low_k)
        self.assertNotIn("transform_mode", low_k)

        self.assertEqual(high_k["adaptive_qknn_requested_policy"], "stable_dualview_v67")
        self.assertEqual(high_k["adaptive_qknn_policy"], "stable_dualview_v67")
        self.assertEqual(high_k["support_code_budget_per_class"], 0)
        self.assertEqual(high_k["support_code_budget_mode"], "centroid_hard_diverse")
        self.assertEqual(high_k["support_code_old_budget_per_class"], 5)
        self.assertEqual(high_k["support_code_new_budget_per_class"], 8)
        self.assertEqual(high_k["support_code_new_protect_top_classes"], 8)
        self.assertEqual(high_k["support_code_new_protect_metric"], "radius")
        self.assertEqual(high_k["local_competition_weight"], 0.02)
        self.assertEqual(high_k["local_competition_k"], 5)
        self.assertEqual(high_k["local_competition_clip"], 2.0)
        self.assertEqual(high_k["local_competition_scope"], "role")
        self.assertNotIn("transform_mode", high_k)
        self.assertNotIn("topm", high_k)
        self.assertNotIn("proto_mix", high_k)

    def test_v69_applies_lighter_labelprop_v66_compression_only_for_k10(self):
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
            policy="stable_dualview_v69",
            geometry=many_new_low_k,
            aux_available=True,
        )
        high_k = _adaptive_qknn_overrides(
            policy="stable_dualview_v69",
            geometry=many_new_high_k,
            aux_available=True,
        )

        self.assertEqual(low_k["adaptive_qknn_requested_policy"], "stable_dualview_v69")
        self.assertEqual(low_k["adaptive_qknn_policy"], "stable_dualview_v69")
        self.assertNotIn("support_code_old_budget_per_class", low_k)
        self.assertNotIn("support_code_new_protect_top_classes", low_k)
        self.assertNotIn("labelprop_weight", low_k)
        self.assertNotIn("scenario_residual_weight", low_k)

        self.assertEqual(high_k["adaptive_qknn_requested_policy"], "stable_dualview_v69")
        self.assertEqual(high_k["adaptive_qknn_policy"], "stable_dualview_v69")
        self.assertEqual(high_k["support_code_budget_per_class"], 0)
        self.assertEqual(high_k["support_code_budget_mode"], "centroid_hard_diverse")
        self.assertEqual(high_k["support_code_old_budget_per_class"], 5)
        self.assertEqual(high_k["support_code_new_budget_per_class"], 9)
        self.assertEqual(high_k["support_code_new_protect_top_classes"], 8)
        self.assertEqual(high_k["support_code_new_protect_metric"], "radius")
        self.assertEqual(high_k["local_competition_weight"], 0.02)
        self.assertEqual(high_k["local_competition_k"], 5)
        self.assertEqual(high_k["local_competition_clip"], 2.0)
        self.assertEqual(high_k["local_competition_scope"], "role")
        self.assertEqual(high_k["labelprop_weight"], 0.015)
        self.assertEqual(high_k["labelprop_k"], 10)
        self.assertEqual(high_k["labelprop_alpha"], 0.72)
        self.assertEqual(high_k["labelprop_temperature"], 0.05)
        self.assertEqual(high_k["labelprop_rounds"], 8)
        self.assertEqual(high_k["labelprop_clip"], 2.0)
        self.assertEqual(high_k["labelprop_scope"], "all")
        self.assertEqual(high_k["scenario_residual_weight"], 0.5)
        self.assertEqual(high_k["scenario_residual_min_classes"], 2)
        self.assertEqual(high_k["scenario_residual_clip"], 0.5)
        self.assertEqual(high_k["scenario_residual_scope"], "new")
        self.assertNotIn("transform_mode", high_k)
        self.assertNotIn("topm", high_k)
        self.assertNotIn("proto_mix", high_k)

    def test_adaptive_override_merges_scenario_residual_params(self):
        from phase2_support_metric_qknn_probe import _adaptive_qknn_scenario_residual_overrides

        params = {
            "scenario_residual_weight": 0.0,
            "scenario_residual_min_classes": 1,
            "scenario_residual_clip": 0.3,
            "scenario_residual_scope": "all",
        }
        adaptive_overrides = {
            "scenario_residual_weight": 0.5,
            "scenario_residual_min_classes": 2,
            "scenario_residual_clip": 0.5,
            "scenario_residual_scope": "new",
        }

        merged = _adaptive_qknn_scenario_residual_overrides(params, adaptive_overrides)

        self.assertEqual(merged["scenario_residual_weight"], 0.5)
        self.assertEqual(merged["scenario_residual_min_classes"], 2)
        self.assertEqual(merged["scenario_residual_clip"], 0.5)
        self.assertEqual(merged["scenario_residual_scope"], "new")

    def test_v70_applies_more_efficient_v69_compression_only_for_k10(self):
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
            policy="stable_dualview_v70",
            geometry=many_new_low_k,
            aux_available=True,
        )
        high_k = _adaptive_qknn_overrides(
            policy="stable_dualview_v70",
            geometry=many_new_high_k,
            aux_available=True,
        )

        self.assertEqual(low_k["adaptive_qknn_requested_policy"], "stable_dualview_v70")
        self.assertEqual(low_k["adaptive_qknn_policy"], "stable_dualview_v70")
        self.assertNotIn("support_code_old_budget_per_class", low_k)
        self.assertNotIn("support_code_new_protect_top_classes", low_k)
        self.assertNotIn("labelprop_weight", low_k)
        self.assertNotIn("scenario_residual_weight", low_k)

        self.assertEqual(high_k["adaptive_qknn_requested_policy"], "stable_dualview_v70")
        self.assertEqual(high_k["adaptive_qknn_policy"], "stable_dualview_v70")
        self.assertEqual(high_k["support_code_budget_per_class"], 0)
        self.assertEqual(high_k["support_code_budget_mode"], "centroid_hard_diverse")
        self.assertEqual(high_k["support_code_old_budget_per_class"], 5)
        self.assertEqual(high_k["support_code_new_budget_per_class"], 8)
        self.assertEqual(high_k["support_code_new_protect_top_classes"], 10)
        self.assertEqual(high_k["support_code_new_protect_metric"], "radius")
        self.assertEqual(high_k["local_competition_weight"], 0.02)
        self.assertEqual(high_k["local_competition_k"], 5)
        self.assertEqual(high_k["local_competition_clip"], 2.0)
        self.assertEqual(high_k["local_competition_scope"], "role")
        self.assertEqual(high_k["labelprop_weight"], 0.015)
        self.assertEqual(high_k["labelprop_k"], 10)
        self.assertEqual(high_k["labelprop_alpha"], 0.72)
        self.assertEqual(high_k["labelprop_temperature"], 0.05)
        self.assertEqual(high_k["labelprop_rounds"], 8)
        self.assertEqual(high_k["labelprop_clip"], 2.0)
        self.assertEqual(high_k["labelprop_scope"], "all")
        self.assertEqual(high_k["scenario_residual_weight"], 0.5)
        self.assertEqual(high_k["scenario_residual_min_classes"], 2)
        self.assertEqual(high_k["scenario_residual_clip"], 0.5)
        self.assertEqual(high_k["scenario_residual_scope"], "new")
        self.assertNotIn("transform_mode", high_k)
        self.assertNotIn("topm", high_k)
        self.assertNotIn("proto_mix", high_k)

    def test_v73_applies_compact_radius_protosim_floor75_branch_only_for_k10(self):
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
            policy="stable_dualview_v73",
            geometry=many_new_low_k,
            aux_available=True,
        )
        high_k = _adaptive_qknn_overrides(
            policy="stable_dualview_v73",
            geometry=many_new_high_k,
            aux_available=True,
        )

        self.assertEqual(low_k["adaptive_qknn_requested_policy"], "stable_dualview_v73")
        self.assertEqual(low_k["adaptive_qknn_policy"], "stable_dualview_v73")
        self.assertNotIn("support_code_old_budget_per_class", low_k)
        self.assertNotIn("support_code_new_protect_top_classes", low_k)
        self.assertNotIn("labelprop_weight", low_k)
        self.assertNotIn("scenario_residual_weight", low_k)

        self.assertEqual(high_k["adaptive_qknn_requested_policy"], "stable_dualview_v73")
        self.assertEqual(high_k["adaptive_qknn_policy"], "stable_dualview_v73")
        self.assertEqual(high_k["support_code_budget_per_class"], 0)
        self.assertEqual(high_k["support_code_budget_mode"], "centroid_hard_diverse")
        self.assertEqual(high_k["support_code_old_budget_per_class"], 5)
        self.assertEqual(high_k["support_code_new_budget_per_class"], 8)
        self.assertEqual(high_k["support_code_new_protect_top_classes"], 8)
        self.assertEqual(high_k["support_code_new_protect_metric"], "radius_proto_sim")
        self.assertEqual(high_k["local_competition_weight"], 0.02)
        self.assertEqual(high_k["labelprop_weight"], 0.015)
        self.assertEqual(high_k["labelprop_k"], 10)
        self.assertEqual(high_k["labelprop_scope"], "all")
        self.assertEqual(high_k["scenario_residual_weight"], 0.5)
        self.assertEqual(high_k["scenario_residual_scope"], "new")
        self.assertNotIn("transform_mode", high_k)
        self.assertNotIn("topm", high_k)
        self.assertNotIn("proto_mix", high_k)

    def test_v76_applies_budget7_radius_protosim_protect12_branch_only_for_k10(self):
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
            policy="stable_dualview_v76",
            geometry=many_new_low_k,
            aux_available=True,
        )
        high_k = _adaptive_qknn_overrides(
            policy="stable_dualview_v76",
            geometry=many_new_high_k,
            aux_available=True,
        )

        self.assertEqual(low_k["adaptive_qknn_requested_policy"], "stable_dualview_v76")
        self.assertEqual(low_k["adaptive_qknn_policy"], "stable_dualview_v76")
        self.assertNotIn("support_code_old_budget_per_class", low_k)
        self.assertNotIn("support_code_new_protect_top_classes", low_k)
        self.assertNotIn("labelprop_weight", low_k)
        self.assertNotIn("scenario_residual_weight", low_k)

        self.assertEqual(high_k["adaptive_qknn_requested_policy"], "stable_dualview_v76")
        self.assertEqual(high_k["adaptive_qknn_policy"], "stable_dualview_v76")
        self.assertEqual(high_k["support_code_budget_per_class"], 0)
        self.assertEqual(high_k["support_code_budget_mode"], "centroid_hard_diverse")
        self.assertEqual(high_k["support_code_old_budget_per_class"], 5)
        self.assertEqual(high_k["support_code_new_budget_per_class"], 7)
        self.assertEqual(high_k["support_code_new_protect_top_classes"], 12)
        self.assertEqual(high_k["support_code_new_protect_metric"], "radius_proto_sim")
        self.assertEqual(high_k["local_competition_weight"], 0.02)
        self.assertEqual(high_k["labelprop_weight"], 0.015)
        self.assertEqual(high_k["labelprop_k"], 10)
        self.assertEqual(high_k["labelprop_scope"], "all")
        self.assertEqual(high_k["scenario_residual_weight"], 0.5)
        self.assertEqual(high_k["scenario_residual_scope"], "new")
        self.assertNotIn("transform_mode", high_k)
        self.assertNotIn("topm", high_k)
        self.assertNotIn("proto_mix", high_k)

    def test_v78_applies_budget6_radius_protosim_protect12_high_compression_branch_only_for_k10(self):
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
            policy="stable_dualview_v78",
            geometry=many_new_low_k,
            aux_available=True,
        )
        high_k = _adaptive_qknn_overrides(
            policy="stable_dualview_v78",
            geometry=many_new_high_k,
            aux_available=True,
        )

        self.assertEqual(low_k["adaptive_qknn_requested_policy"], "stable_dualview_v78")
        self.assertEqual(low_k["adaptive_qknn_policy"], "stable_dualview_v78")
        self.assertNotIn("support_code_old_budget_per_class", low_k)
        self.assertNotIn("support_code_new_protect_top_classes", low_k)
        self.assertNotIn("labelprop_weight", low_k)
        self.assertNotIn("scenario_residual_weight", low_k)

        self.assertEqual(high_k["adaptive_qknn_requested_policy"], "stable_dualview_v78")
        self.assertEqual(high_k["adaptive_qknn_policy"], "stable_dualview_v78")
        self.assertEqual(high_k["support_code_budget_per_class"], 0)
        self.assertEqual(high_k["support_code_budget_mode"], "centroid_hard_diverse")
        self.assertEqual(high_k["support_code_old_budget_per_class"], 5)
        self.assertEqual(high_k["support_code_new_budget_per_class"], 6)
        self.assertEqual(high_k["support_code_new_protect_top_classes"], 12)
        self.assertEqual(high_k["support_code_new_protect_metric"], "radius_proto_sim")
        self.assertEqual(high_k["local_competition_weight"], 0.02)
        self.assertEqual(high_k["labelprop_weight"], 0.015)
        self.assertEqual(high_k["labelprop_k"], 10)
        self.assertEqual(high_k["labelprop_scope"], "all")
        self.assertEqual(high_k["scenario_residual_weight"], 0.5)
        self.assertEqual(high_k["scenario_residual_scope"], "new")
        self.assertNotIn("transform_mode", high_k)
        self.assertNotIn("topm", high_k)
        self.assertNotIn("proto_mix", high_k)

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
