import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _write_npz(
    path: Path,
    *,
    include_unknown: bool = True,
    aligned: bool = True,
    include_source: bool = True,
    include_bad_proxy: bool = False,
    include_proxy: bool = False,
) -> None:
    rows = []

    def add(role, tx, rx, day, sig, scenario, feature):
        rows.append((role, tx, rx, day, sig, scenario, np.asarray(feature, dtype=np.float32)))

    for rx in ["rx-a", "rx-b"]:
        if include_source:
            add("source", "old-a", "src-a", "d0", f"src-{rx}", "", [1.0, 0.0, 0.0])
        add("target_old", "old-a", rx, "d1", f"old-support-{rx}", "leo_clear_weak", [1.0, 0.0, 0.0])
        add("target_new", "new-a", rx, "d1", f"new-support-{rx}", "leo_clear_weak", [0.0, 1.0, 0.0])
        old_sig = "old-query" if aligned else f"old-query-{rx}"
        new_sig = "new-query" if aligned else f"new-query-{rx}"
        unk_sig = "unk-query" if aligned else f"unk-query-{rx}"
        add("target_old", "old-a", rx, "d2", old_sig, "leo_clear_weak", [0.98, 0.02, 0.0])
        add("target_old", "old-a", rx, "d2", f"{old_sig}-2", "leo_clear_weak", [0.99, 0.01, 0.0])
        add("target_new", "new-a", rx, "d2", new_sig, "leo_clear_weak", [0.02, 0.98, 0.0])
        add("target_new", "new-a", rx, "d2", f"{new_sig}-2", "leo_clear_weak", [0.01, 0.99, 0.0])
        if include_unknown:
            add("target_unknown", "unk-a", rx, "d2", unk_sig, "leo_clear_weak", [0.0, 0.0, 1.0])
            add("target_unknown", "unk-a", rx, "d2", f"{unk_sig}-2", "leo_clear_weak", [0.0, 0.01, 0.99])
    if include_bad_proxy:
        add("proxy_unknown", "unk-a", "rx-a", "d3", "bad-proxy", "leo_clear_weak", [0.0, 0.0, 1.0])
    if include_proxy:
        add("proxy_unknown", "proxy-a", "src-proxy", "d3", "proxy-1", "leo_clear_weak", [0.0, 0.0, 1.0])

    manifest = {
        "source_tx_ids": ["old-a"],
        "target_old_tx_ids": ["old-a"],
        "new_tx_ids": ["new-a"],
        "unknown_tx_ids": ["unk-a"],
        "target_channel_view": "satellite/LEO",
    }
    np.savez(
        path,
        features=np.stack([r[6] for r in rows]).astype(np.float32),
        dataset_role=np.asarray([r[0] for r in rows], dtype=object),
        tx_ids=np.asarray([r[1] for r in rows], dtype=object),
        rx_ids=np.asarray([r[2] for r in rows], dtype=object),
        day_ids=np.asarray([r[3] for r in rows], dtype=object),
        sig_ids=np.asarray([r[4] for r in rows], dtype=object),
        sat_scenarios=np.asarray([r[5] for r in rows], dtype=object),
        channel_views=np.asarray(["satellite" if r[5] else "clean" for r in rows], dtype=object),
        manifest_json=np.asarray(json.dumps(manifest)),
    )


class Phase2CollaborativeOpenSetQknnEvalTest(unittest.TestCase):
    def test_builds_qknn8_evidence_and_reports_one_to_all_receivers(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
            )
            result = evaluate_collaborative_open_set_evidence(
                evidence,
                collab_counts="all",
                threshold_selection_label_scope=metadata["threshold_scope"],
                protocol_metadata=metadata,
                strict_protocol_metadata=True,
            )

        self.assertEqual(result["receiver_count"], 2)
        self.assertEqual(set(result["counts"]), {"1", "2"})
        self.assertEqual(result["counts"]["2"]["old_acc"], 1.0)
        self.assertEqual(result["counts"]["2"]["seen_new_acc"], 1.0)
        self.assertGreaterEqual(result["counts"]["2"]["unknown_reject_rate"], 0.0)
        self.assertEqual(metadata["event_alignment"], "role_tx_day_sig_scenario")
        self.assertGreater(metadata["prototype_storage_bytes"], 0)

    def test_records_scorer_cvs_packet_resource_bytes(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
                unknown_gate_mode="support_envelope",
                evidence_packet_bytes=96,
            )
            result = evaluate_collaborative_open_set_evidence(
                evidence,
                collab_counts="2",
                threshold_selection_label_scope=metadata["threshold_scope"],
                protocol_metadata=metadata,
                strict_protocol_metadata=True,
                fusion_policy="scorer_cvs",
                consensus_gap_threshold=0.5,
                consensus_score_threshold=0.0,
                latency_budget_ms=10.0,
            )

        self.assertEqual(metadata["evidence_bytes_per_receiver_event"], 96)
        self.assertEqual(result["counts"]["2"]["bytes_per_event"], 192)
        self.assertEqual(result["fusion_policy"], "scorer_cvs")

    def test_requires_target_unknown_rows_for_open_set_eval(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz, include_unknown=False)
            with self.assertRaisesRegex(RuntimeError, "LOCAL_DATASET_EXTENSION_REQUIRED"):
                build_collaborative_evidence(load_feature_npz(npz), k_shot=1, query_per_class=1)

    def test_requires_source_receivers_to_verify_disjoint_protocol(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz, include_source=False)
            with self.assertRaisesRegex(RuntimeError, "LOCAL_PROTOCOL_REPAIR_REQUIRED"):
                build_collaborative_evidence(load_feature_npz(npz), k_shot=1, query_per_class=1)

    def test_rejects_proxy_unknown_that_overlaps_target_unknown(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz, include_bad_proxy=True)
            with self.assertRaisesRegex(RuntimeError, "proxy_unknown calibration rows must be source-only"):
                build_collaborative_evidence(load_feature_npz(npz), k_shot=1, query_per_class=1)

    def test_requires_per_receiver_support_and_query_coverage(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            with self.assertRaisesRegex(RuntimeError, "incomplete Stage2-C coverage"):
                build_collaborative_evidence(load_feature_npz(npz), k_shot=2, query_per_class=2)

    def test_strict_event_key_keeps_partial_receiver_groups_for_count_one(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz, aligned=False)
            evidence, metadata = build_collaborative_evidence(load_feature_npz(npz), k_shot=1, query_per_class=1)

        self.assertGreater(len(evidence), 0)
        self.assertTrue(metadata["strict_same_event_collaboration"])
        self.assertEqual(metadata["event_alignment"], "role_tx_day_sig_scenario")
        self.assertEqual(max(len({row["receiver_id"] for row in evidence if row["event_id"] == event_id}) for event_id in {row["event_id"] for row in evidence}), 1)

    def test_receiver_domain_ranked_policy_is_explicitly_marked(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz, aligned=False)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                event_alignment_policy="receiver_domain_ranked",
            )

        self.assertGreater(len(evidence), 0)
        self.assertEqual(metadata["event_alignment_policy"], "receiver_domain_ranked")
        self.assertFalse(metadata["strict_same_event_collaboration"])
        self.assertEqual(metadata["event_alignment"], "receiver_domain_ranked_by_role_tx_scenario")

    def test_support_envelope_gate_records_radius_and_margin_risk(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
                support_selection_policy="centroid",
                unknown_gate_mode="support_envelope",
            )

        self.assertEqual(metadata["support_selection_policy"], "centroid")
        self.assertEqual(metadata["unknown_gate_mode"], "support_envelope")
        self.assertIn("score_risk", evidence[0])
        self.assertIn("radius_risk", evidence[0])
        self.assertIn("margin_risk", evidence[0])
        self.assertIn("class_radius", evidence[0])

    def test_mahalanobis_gate_records_class_conditional_risk(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
                unknown_gate_mode="support_envelope_mahalanobis",
                mahalanobis_temperature=0.2,
            )

        self.assertEqual(metadata["unknown_gate_mode"], "support_envelope_mahalanobis")
        self.assertIn("mahalanobis_risk", evidence[0])
        self.assertIn("mahalanobis_temperature", metadata)

    def test_evt_gate_records_tail_risk(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
                unknown_gate_mode="support_envelope_evt",
                evt_tail_quantile=0.8,
                evt_temperature=0.05,
            )

        self.assertEqual(metadata["unknown_gate_mode"], "support_envelope_evt")
        self.assertIn("evt_risk", evidence[0])
        self.assertIn("evt_tail_quantile", metadata)

    def test_oldness_gate_records_candidate_class_risk(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
                unknown_gate_mode="support_envelope_oldness",
                oldness_quantile=0.05,
                oldness_temperature=0.05,
            )

        self.assertEqual(metadata["unknown_gate_mode"], "support_envelope_oldness")
        self.assertEqual(metadata["active_risk_components"], ["score", "radius", "margin", "oldness"])
        self.assertIn("oldness_risk", evidence[0])
        self.assertIn("oldness_quantile", metadata)

    def test_scenario_aware_qknn_prefers_matching_support_scenario(self):
        from phase2_collaborative_open_set_qknn_eval import build_qknn_memory, qknn_scores

        memory = build_qknn_memory(
            np.asarray(
                [
                    [0.7, 0.7, 0.0],
                    [0.69, 0.71, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            ["old-a", "old-a", "new-b", "new-a"],
            old_labels={"old-a"},
            support_scenarios=["leo_clear_weak", "leo_clear_weak", "leo_clear_weak", "leo_rain_weak"],
        )
        pred_global, _, _, _, _, _, _, _ = qknn_scores(
            memory,
            np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
            top_k=1,
            query_scenarios=["leo_clear_weak"],
            scenario_aware=False,
        )
        pred_scenario, _, _, _, _, _, _, _ = qknn_scores(
            memory,
            np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
            top_k=1,
            query_scenarios=["leo_clear_weak"],
            scenario_aware=True,
        )

        self.assertEqual(str(pred_global[0]), "new-a")
        self.assertEqual(str(pred_scenario[0]), "old-a")

    def test_leave_one_out_support_calibration_excludes_self_neighbor(self):
        from phase2_collaborative_open_set_qknn_eval import build_qknn_memory, qknn_scores

        features = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.2, 0.8, 0.0],
            ],
            dtype=np.float32,
        )
        memory = build_qknn_memory(
            features,
            ["old-a", "old-a", "new-a", "new-a"],
            old_labels={"old-a"},
            support_scenarios=["leo_clear_weak"] * 4,
        )
        _, self_scores, _, _, _, _, _, _ = qknn_scores(memory, features, top_k=1)
        _, loo_scores, _, _, _, _, _, _ = qknn_scores(memory, features, top_k=1, exclude_support_indices=range(features.shape[0]))

        self.assertLess(float(np.mean(loo_scores)), float(np.mean(self_scores)))

    def test_candidate_class_top_m_limits_qknn_support_classes(self):
        from phase2_collaborative_open_set_qknn_eval import build_qknn_memory, qknn_scores

        features = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 0.9],
            ],
            dtype=np.float32,
        )
        memory = build_qknn_memory(
            features,
            ["old-a", "old-a", "new-a", "new-a", "new-b", "new-b"],
            old_labels={"old-a"},
        )
        pred, _, _, candidate_counts, support_neighbor_counts, support_densities, second_labels, second_scores = qknn_scores(
            memory,
            np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
            top_k=2,
            candidate_class_top_m=1,
        )

        self.assertEqual(str(pred[0]), "new-a")
        self.assertEqual(int(candidate_counts[0]), 1)
        self.assertEqual(int(support_neighbor_counts[0]), 2)
        self.assertAlmostEqual(float(support_densities[0]), 1.0)
        self.assertEqual(str(second_labels[0]), "")
        self.assertEqual(float(second_scores[0]), 0.0)

    def test_prototype_score_blend_can_correct_neighbor_only_collision(self):
        from phase2_collaborative_open_set_qknn_eval import build_qknn_memory, qknn_scores

        features = np.asarray(
            [
                [0.8, 0.6, 0.0],
                [0.8, -0.6, 0.0],
                [0.9, 0.435, 0.0],
                [0.9, 0.435, 0.0],
            ],
            dtype=np.float32,
        )
        memory = build_qknn_memory(
            features,
            ["old-a", "old-a", "new-a", "new-a"],
            old_labels={"old-a"},
        )

        pred_neighbor_only, _, _, _, _, _, _, _ = qknn_scores(
            memory,
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            top_k=1,
        )
        pred_blended, score, margin, _, support_neighbor_counts, _, second_labels, _ = qknn_scores(
            memory,
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            top_k=1,
            prototype_score_blend=10.0,
        )

        self.assertEqual(str(pred_neighbor_only[0]), "new-a")
        self.assertEqual(str(pred_blended[0]), "old-a")
        self.assertGreater(float(score[0]), 0.0)
        self.assertGreater(float(margin[0]), 0.0)
        self.assertEqual(int(support_neighbor_counts[0]), 0)
        self.assertEqual(str(second_labels[0]), "new-a")

    def test_zero_prototype_score_blend_matches_default_qknn_scores(self):
        from phase2_collaborative_open_set_qknn_eval import build_qknn_memory, qknn_scores

        features = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.2, 0.8, 0.0],
            ],
            dtype=np.float32,
        )
        memory = build_qknn_memory(features, ["old-a", "old-a", "new-a", "new-a"], old_labels={"old-a"})
        default = qknn_scores(memory, features, top_k=2)
        explicit_zero = qknn_scores(memory, features, top_k=2, prototype_score_blend=0.0)

        for left, right in zip(default, explicit_zero):
            np.testing.assert_array_equal(left, right)

    def test_prototype_score_blend_uses_same_calibration_score_scale(self):
        from phase2_collaborative_open_set_qknn_eval import build_qknn_memory, _threshold_from_calibration

        features = np.asarray(
            [
                [0.8, 0.6, 0.0],
                [0.8, -0.6, 0.0],
                [0.9, 0.435, 0.0],
                [0.9, 0.435, 0.0],
            ],
            dtype=np.float32,
        )
        memory = build_qknn_memory(features, ["old-a", "old-a", "new-a", "new-a"], old_labels={"old-a"})

        threshold_default, _ = _threshold_from_calibration(
            memory,
            features,
            None,
            top_k=1,
            support_quantile=0.5,
            proxy_quantile=0.95,
            support_calibration_mode="leave_one_out",
        )
        threshold_blended, _ = _threshold_from_calibration(
            memory,
            features,
            None,
            top_k=1,
            support_quantile=0.5,
            proxy_quantile=0.95,
            support_calibration_mode="leave_one_out",
            prototype_score_blend=2.0,
        )

        self.assertNotAlmostEqual(threshold_default, threshold_blended, places=8)

    def test_negative_prototype_score_blend_is_rejected(self):
        from phase2_collaborative_open_set_qknn_eval import build_qknn_memory, qknn_scores

        features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        memory = build_qknn_memory(features, ["old-a", "new-a"], old_labels={"old-a"})

        with self.assertRaisesRegex(ValueError, "prototype_score_blend"):
            qknn_scores(memory, features, top_k=1, prototype_score_blend=-1.0)

    def test_mahalanobis_score_blend_can_correct_local_neighbor_collision(self):
        from phase2_collaborative_open_set_qknn_eval import build_qknn_memory, qknn_scores

        features = np.asarray(
            [
                [0.8, 0.6, 0.0],
                [0.8, -0.6, 0.0],
                [0.96, 0.28, 0.0],
                [0.96, 0.28, 0.0],
            ],
            dtype=np.float32,
        )
        memory = build_qknn_memory(
            features,
            ["old-a", "old-a", "new-a", "new-a"],
            old_labels={"old-a"},
        )

        pred_neighbor_only, _, _, _, _, _, _, _ = qknn_scores(
            memory,
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            top_k=1,
        )
        pred_blended, score, margin, _, support_neighbor_counts, support_densities, second_labels, _ = qknn_scores(
            memory,
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            top_k=1,
            mahalanobis_score_blend=2.0,
        )

        self.assertEqual(str(pred_neighbor_only[0]), "new-a")
        self.assertEqual(str(pred_blended[0]), "old-a")
        self.assertGreater(float(score[0]), 0.0)
        self.assertGreater(float(margin[0]), 0.0)
        self.assertEqual(int(support_neighbor_counts[0]), 0)
        self.assertEqual(float(support_densities[0]), 0.0)
        self.assertEqual(str(second_labels[0]), "new-a")

    def test_mahalanobis_score_blend_uses_same_calibration_score_scale(self):
        from phase2_collaborative_open_set_qknn_eval import build_qknn_memory, _threshold_from_calibration

        features = np.asarray(
            [
                [0.8, 0.6, 0.0],
                [0.8, -0.6, 0.0],
                [0.96, 0.28, 0.0],
                [0.96, 0.28, 0.0],
            ],
            dtype=np.float32,
        )
        memory = build_qknn_memory(features, ["old-a", "old-a", "new-a", "new-a"], old_labels={"old-a"})

        threshold_default, _ = _threshold_from_calibration(
            memory,
            features,
            None,
            top_k=1,
            support_quantile=0.5,
            proxy_quantile=0.95,
            support_calibration_mode="leave_one_out",
        )
        threshold_blended, _ = _threshold_from_calibration(
            memory,
            features,
            None,
            top_k=1,
            support_quantile=0.5,
            proxy_quantile=0.95,
            support_calibration_mode="leave_one_out",
            mahalanobis_score_blend=2.0,
        )

        self.assertNotAlmostEqual(threshold_default, threshold_blended, places=8)

    def test_negative_mahalanobis_score_blend_is_rejected(self):
        from phase2_collaborative_open_set_qknn_eval import build_qknn_memory, qknn_scores

        features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        memory = build_qknn_memory(features, ["old-a", "new-a"], old_labels={"old-a"})

        with self.assertRaisesRegex(ValueError, "mahalanobis_score_blend"):
            qknn_scores(memory, features, top_k=1, mahalanobis_score_blend=-1.0)

    def test_zero_mahalanobis_score_blend_matches_default_qknn_scores(self):
        from phase2_collaborative_open_set_qknn_eval import build_qknn_memory, qknn_scores

        features = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.2, 0.8, 0.0],
            ],
            dtype=np.float32,
        )
        memory = build_qknn_memory(features, ["old-a", "old-a", "new-a", "new-a"], old_labels={"old-a"})
        default = qknn_scores(memory, features, top_k=2)
        explicit_zero = qknn_scores(memory, features, top_k=2, mahalanobis_score_blend=0.0)

        for left, right in zip(default, explicit_zero):
            np.testing.assert_array_equal(left, right)

    def test_support_density_reliability_is_recorded_in_evidence(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
                receiver_reliability_policy="support_density",
            )

        self.assertEqual(metadata["receiver_reliability_policy"], "support_density")
        self.assertIn("support_neighbor_count", evidence[0])
        self.assertIn("support_density", evidence[0])
        self.assertIn("second_label", evidence[0])
        self.assertIn("second_score", evidence[0])
        self.assertIn("label_score_gap", evidence[0])
        self.assertIn("audit_full_top1_label", evidence[0])
        self.assertIn("audit_full_second_label", evidence[0])
        self.assertIn("audit_full_label_score_gap", evidence[0])
        self.assertIn("candidate_audit_disagreement", evidence[0])
        self.assertIn("candidate_audit_risk", evidence[0])
        self.assertIn("class_radius_z", evidence[0])
        self.assertEqual(float(evidence[0]["reliability"]), float(evidence[0]["support_density"]))

    def test_prototype_assisted_qknn_is_marked_in_metadata_and_evidence(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
                prototype_score_blend=0.5,
            )

        self.assertTrue(metadata["prototype_assisted_qknn"])
        self.assertAlmostEqual(metadata["prototype_score_blend"], 0.5)
        self.assertIn("prototype_score_blend", evidence[0])
        self.assertIn("prototype_assisted", evidence[0])
        self.assertIn("prototype_only_top1", evidence[0])

    def test_mahalanobis_score_assisted_qknn_is_marked_in_metadata_and_evidence(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
                mahalanobis_score_blend=0.5,
                mahalanobis_score_temperature=0.2,
            )

        self.assertTrue(metadata["mahalanobis_score_assisted_qknn"])
        self.assertAlmostEqual(metadata["mahalanobis_score_blend"], 0.5)
        self.assertAlmostEqual(metadata["mahalanobis_score_temperature"], 0.2)
        self.assertIn("mahalanobis_score_blend", evidence[0])
        self.assertIn("mahalanobis_score_temperature", evidence[0])
        self.assertIn("mahalanobis_score_assisted", evidence[0])

    def test_source_only_proxy_unknown_sets_threshold_scope(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz, include_proxy=True)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
            )

        self.assertEqual(metadata["threshold_scope"], "source_only")
        self.assertEqual({row["role"] for row in evidence}, {"old", "seen_new", "unknown"})
        self.assertIn("unk-a", metadata["unknown_tx_ids"])
        self.assertNotIn("proxy-a", metadata["unknown_tx_ids"])

    def test_class_score_thresholds_are_recorded_when_enabled(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
                support_calibration_mode="leave_one_out",
                score_threshold_combine="qknn_only",
                class_score_threshold_enabled=True,
                class_score_threshold_quantile=0.5,
            )

        self.assertTrue(metadata["class_score_threshold_enabled"])
        self.assertAlmostEqual(metadata["class_score_threshold_quantile"], 0.5)
        self.assertTrue(metadata["receiver_class_thresholds"])
        first_rx_thresholds = next(iter(metadata["receiver_class_thresholds"].values()))
        self.assertIn("old-a", first_rx_thresholds)
        self.assertIn("new-a", first_rx_thresholds)
        self.assertIn("effective_score_threshold", evidence[0])
        self.assertIn("class_score_threshold", evidence[0])
        self.assertEqual(int(evidence[0]["class_score_threshold_enabled"]), 1)
        self.assertIn("score_threshold_source", evidence[0])

    def test_class_score_thresholds_use_true_label_score_not_top1_score(self):
        from phase2_collaborative_open_set_qknn_eval import (
            _label_thresholds_from_calibration,
            build_qknn_memory,
            qknn_scores,
        )

        features = np.asarray(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.98, 0.20],
            ],
            dtype=np.float32,
        )
        labels = ["old-a", "old-a", "new-a"]
        memory = build_qknn_memory(features, labels, old_labels={"old-a"})
        pred, scores, *_ = qknn_scores(
            memory,
            features,
            top_k=1,
            exclude_support_indices=range(features.shape[0]),
        )
        thresholds = _label_thresholds_from_calibration(
            memory,
            features,
            labels,
            None,
            top_k=1,
            support_quantile=0.5,
            proxy_quantile=0.5,
            support_calibration_mode="leave_one_out",
            min_support=2,
        )

        self.assertEqual(str(pred[0]), "new-a")
        self.assertGreater(float(scores[0]), 0.5)
        self.assertAlmostEqual(float(thresholds["old-a"]), 0.0)

    def test_class_score_thresholds_are_disabled_by_default(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
            )

        self.assertFalse(metadata["class_score_threshold_enabled"])
        self.assertEqual(metadata["receiver_class_thresholds"], {rx: {} for rx in metadata["target_receiver_ids"]})
        self.assertEqual(int(evidence[0]["class_score_threshold_enabled"]), 0)

    def test_candidate_audit_gap_can_raise_unknown_risk(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
                candidate_class_top_m=1,
                candidate_audit_unknown_risk_enabled=True,
                candidate_audit_min_gap=2.0,
                candidate_audit_gap_risk=0.99,
            )

        self.assertTrue(metadata["candidate_audit_unknown_risk_enabled"])
        self.assertGreaterEqual(float(evidence[0]["candidate_audit_risk"]), 0.99)
        self.assertGreaterEqual(float(evidence[0]["unknown_risk"]), 0.99)
        self.assertGreaterEqual(float(evidence[0]["score_risk"]), 0.99)

    def test_scenario_aware_and_radius_norm_are_recorded_in_metadata(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            _, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
                scenario_aware=True,
                radius_norm=0.3,
                old_bias=0.1,
                candidate_class_top_m=2,
                prototype_score_blend=0.2,
                mahalanobis_score_blend=0.3,
                support_calibration_mode="leave_one_out",
                score_threshold_combine="qknn_only",
            )

        self.assertTrue(metadata["scenario_aware"])
        self.assertAlmostEqual(metadata["radius_norm"], 0.3)
        self.assertAlmostEqual(metadata["old_bias"], 0.1)
        self.assertEqual(metadata["candidate_class_top_m"], 2)
        self.assertAlmostEqual(metadata["prototype_score_blend"], 0.2)
        self.assertAlmostEqual(metadata["mahalanobis_score_blend"], 0.3)
        self.assertEqual(metadata["support_calibration_mode"], "leave_one_out")
        self.assertEqual(metadata["score_threshold_combine"], "qknn_only")


if __name__ == "__main__":
    unittest.main()
