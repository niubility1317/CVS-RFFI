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
    source_tx_id: str = "old-a",
    extra_source_tx_ids: tuple[str, ...] = (),
) -> None:
    rows = []

    def add(role, tx, rx, day, sig, scenario, feature):
        rows.append((role, tx, rx, day, sig, scenario, np.asarray(feature, dtype=np.float32)))

    for rx in ["rx-a", "rx-b"]:
        if include_source:
            add("source", source_tx_id, "src-a", "d0", f"src-{rx}", "", [1.0, 0.0, 0.0])
            for extra_i, extra_tx in enumerate(extra_source_tx_ids):
                add("source", extra_tx, "src-a", "d0", f"src-{rx}-extra-{extra_i}", "", [0.0, 1.0, 0.0])
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
        "source_tx_ids": [source_tx_id, *extra_source_tx_ids],
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

    def test_requires_target_old_to_belong_to_source_tx_set(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz, source_tx_id="other-old")
            with self.assertRaisesRegex(RuntimeError, "old_not_in_source"):
                build_collaborative_evidence(load_feature_npz(npz), k_shot=1, query_per_class=1)

    def test_rejects_seen_new_or_unknown_that_belongs_to_source_tx_set(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz, extra_source_tx_ids=("new-a",))
            with self.assertRaisesRegex(RuntimeError, "non_old_in_source"):
                build_collaborative_evidence(load_feature_npz(npz), k_shot=1, query_per_class=1)

    def test_requires_per_receiver_support_and_query_coverage(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            with self.assertRaisesRegex(RuntimeError, "incomplete Stage2-C coverage"):
                build_collaborative_evidence(load_feature_npz(npz), k_shot=2, query_per_class=2)

    def test_strict_event_key_requires_shared_events_across_target_receivers(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz, aligned=False)
            with self.assertRaisesRegex(RuntimeError, "NO_ALIGNED_COLLABORATIVE_EVENTS"):
                build_collaborative_evidence(load_feature_npz(npz), k_shot=1, query_per_class=1)

    def test_strict_event_key_allows_partial_receiver_groups_when_configured(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            rows = []

            def add(role, tx, rx, day, sig, scenario, feature):
                rows.append((role, tx, rx, day, sig, scenario, np.asarray(feature, dtype=np.float32)))

            for rx in ["rx-a", "rx-b", "rx-c"]:
                add("source", "old-a", "src-a", "d0", f"src-{rx}", "", [1.0, 0.0, 0.0])
                add("target_old", "old-a", rx, "d1", f"old-support-{rx}", "leo_clear_weak", [1.0, 0.0, 0.0])
                add("target_new", "new-a", rx, "d1", f"new-support-{rx}", "leo_clear_weak", [0.0, 1.0, 0.0])
                shared = "shared-query" if rx in {"rx-a", "rx-b"} else "rx-c-query"
                add("target_old", "old-a", rx, "d2", shared, "leo_clear_weak", [0.98, 0.02, 0.0])
                add("target_new", "new-a", rx, "d2", shared, "leo_clear_weak", [0.02, 0.98, 0.0])
                add("target_unknown", "unk-a", rx, "d2", shared, "leo_clear_weak", [0.0, 0.0, 1.0])
            manifest = {
                "source_tx_ids": ["old-a"],
                "target_old_tx_ids": ["old-a"],
                "new_tx_ids": ["new-a"],
                "unknown_tx_ids": ["unk-a"],
                "target_channel_view": "satellite/LEO",
            }
            np.savez(
                npz,
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
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=1,
                strict_event_min_receivers=2,
            )

        self.assertGreater(len(evidence), 0)
        self.assertEqual(metadata["event_alignment_policy"], "strict_event_key")
        self.assertTrue(metadata["strict_same_event_collaboration"])
        self.assertEqual(metadata["strict_event_min_receivers"], 2)
        self.assertIn(2, {int(row["strict_event_receiver_count"]) for row in evidence})

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

    def test_strict_event_query_preserve_keeps_shared_receiver_event_for_query(self):
        from phase2_collaborative_open_set_qknn_eval import _select_support_indices

        rows = [
            ("target_old", "old-a", "rx-a", "d1", "shared", "leo_clear_weak"),
            ("target_old", "old-a", "rx-a", "d1", "unique-a", "leo_clear_weak"),
            ("target_old", "old-a", "rx-b", "d1", "shared", "leo_clear_weak"),
            ("target_old", "old-a", "rx-c", "d1", "shared", "leo_clear_weak"),
        ]
        payload = {
            "dataset_role": np.asarray([r[0] for r in rows], dtype=object),
            "tx_ids": np.asarray([r[1] for r in rows], dtype=object),
            "rx_ids": np.asarray([r[2] for r in rows], dtype=object),
            "day_ids": np.asarray([r[3] for r in rows], dtype=object),
            "eq_ids": np.asarray(["eq-1" for _ in rows], dtype=object),
            "sig_ids": np.asarray([r[4] for r in rows], dtype=object),
            "channel_views": np.asarray(["satellite" for _ in rows], dtype=object),
            "sat_scenarios": np.asarray([r[5] for r in rows], dtype=object),
        }
        features = np.eye(len(rows), dtype=np.float32)

        support = _select_support_indices(
            payload,
            features,
            [0, 1],
            k_shot=1,
            policy="strict_event_query_preserve",
        )

        self.assertEqual(support, [1])

    def test_source_old_prototype_shrinkage_blends_old_centroid_only(self):
        from phase2_collaborative_open_set_qknn_eval import build_qknn_memory

        memory = build_qknn_memory(
            np.asarray(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            ["old-a", "new-a"],
            old_labels={"old-a"},
            source_old_features=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            source_old_labels=["old-a"],
            source_old_prototype_shrinkage_alpha=0.50,
        )

        label_to_pos = {str(label): i for i, label in enumerate(memory.centroid_labels.tolist())}
        expected_old = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
        expected_old = expected_old / np.linalg.norm(expected_old)
        np.testing.assert_allclose(memory.centroids[label_to_pos["old-a"]], expected_old, atol=1e-6)
        np.testing.assert_allclose(
            memory.centroids[label_to_pos["new-a"]],
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            atol=1e-6,
        )
        self.assertEqual(memory.source_old_prototype_shrinkage_applied, {"old-a": 0.5})

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

    def test_full_support_envelope_records_all_calibrated_risks(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
                unknown_gate_mode="support_envelope_full",
                mahalanobis_temperature=0.2,
                evt_temperature=0.05,
                oldness_temperature=0.05,
            )

        self.assertEqual(metadata["unknown_gate_mode"], "support_envelope_full")
        self.assertEqual(
            metadata["active_risk_components"],
            ["score", "radius", "margin", "mahalanobis", "evt", "oldness"],
        )
        for key in (
            "score_risk",
            "radius_risk",
            "margin_risk",
            "mahalanobis_risk",
            "evt_risk",
            "oldness_risk",
        ):
            self.assertIn(key, evidence[0])

    def test_consensus_support_envelope_records_all_calibrated_risks(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
                unknown_gate_mode="support_envelope_consensus",
            )

        self.assertEqual(metadata["unknown_gate_mode"], "support_envelope_consensus")
        self.assertEqual(
            metadata["active_risk_components"],
            ["score", "radius", "margin", "mahalanobis", "evt", "oldness"],
        )
        self.assertIn("unknown_risk", evidence[0])
        self.assertIn("evt_risk", evidence[0])
        self.assertIn("oldness_risk", evidence[0])

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

    def test_seen_new_prototype_calibration_moves_only_seen_new_centroid(self):
        from phase2_collaborative_open_set_qknn_eval import build_qknn_memory

        features = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
            ],
            dtype=np.float32,
        )
        labels = ["old-a", "old-a", "new-a", "new-a"]
        base = build_qknn_memory(features, labels, old_labels={"old-a"})
        calibrated = build_qknn_memory(
            features,
            labels,
            old_labels={"old-a"},
            prototype_calibration_policy="teen_blend",
            prototype_calibration_alpha=0.5,
            prototype_calibration_top_m=1,
        )
        positions = {str(label): int(i) for i, label in enumerate(base.centroid_labels.tolist())}
        old_pos = positions["old-a"]
        new_pos = positions["new-a"]

        np.testing.assert_allclose(base.centroids[old_pos], calibrated.centroids[old_pos])
        base_similarity = float(base.centroids[new_pos] @ base.centroids[old_pos])
        calibrated_similarity = float(calibrated.centroids[new_pos] @ calibrated.centroids[old_pos])
        self.assertGreater(calibrated_similarity, base_similarity)

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

    def test_prototype_calibration_is_marked_in_metadata_and_evidence(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
                prototype_calibration_policy="teen_blend",
                prototype_calibration_alpha=0.25,
                prototype_calibration_top_m=1,
            )

        self.assertEqual(metadata["prototype_calibration_policy"], "teen_blend")
        self.assertAlmostEqual(metadata["prototype_calibration_alpha"], 0.25)
        self.assertEqual(metadata["prototype_calibration_top_m"], 1)
        self.assertIn("prototype_calibration_policy", evidence[0])
        self.assertIn("prototype_calibration_alpha", evidence[0])
        self.assertIn("prototype_calibration_top_m", evidence[0])

    def test_support_center_feature_adapter_transforms_normalized_support(self):
        from phase2_collaborative_open_set_qknn_eval import (
            _apply_feature_adapter,
            _fit_feature_adapter,
            _normalize_rows,
        )

        support = np.asarray(
            [
                [1.0, 0.80, 0.0],
                [1.0, 1.20, 0.0],
                [0.90, 1.00, 0.0],
            ],
            dtype=np.float32,
        )
        none_adapter = _fit_feature_adapter(
            support,
            policy="none",
            strength=1.0,
            variance_floor=1e-4,
        )
        centered_adapter = _fit_feature_adapter(
            support,
            policy="support_center",
            strength=1.0,
            variance_floor=1e-4,
        )

        normalized = _normalize_rows(support)
        self.assertTrue(np.allclose(_apply_feature_adapter(support, none_adapter), normalized))
        centered = _apply_feature_adapter(support, centered_adapter)
        self.assertFalse(np.allclose(centered, normalized))
        self.assertLess(np.linalg.norm(centered.mean(axis=0)), np.linalg.norm(normalized.mean(axis=0)))

    def test_support_bn_affine_adapter_clamps_variance_floor_for_zero_variance_support(self):
        from phase2_collaborative_open_set_qknn_eval import _apply_feature_adapter, _fit_feature_adapter

        support = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        adapter = _fit_feature_adapter(
            support,
            policy="support_bn_affine",
            strength=1.0,
            variance_floor=0.0,
        )
        adapted = _apply_feature_adapter(support, adapter)

        self.assertTrue(np.all(np.isfinite(adapter.scale)))
        self.assertTrue(np.all(adapter.scale >= 1e-8))
        self.assertTrue(np.all(np.isfinite(adapted)))

    def test_feature_adapter_is_marked_in_metadata_and_evidence(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
                feature_adapter_policy="support_center",
                feature_adapter_strength=0.5,
                feature_adapter_variance_floor=1e-5,
            )

        self.assertEqual(metadata["feature_adapter_policy"], "support_center")
        self.assertAlmostEqual(metadata["feature_adapter_strength"], 0.5)
        self.assertAlmostEqual(metadata["feature_adapter_variance_floor"], 1e-5)
        self.assertIn("feature_adapter_policy", evidence[0])
        self.assertIn("feature_adapter_strength", evidence[0])
        self.assertIn("feature_adapter_variance_floor", evidence[0])
        self.assertEqual(evidence[0]["feature_adapter_policy"], "support_center")
        self.assertAlmostEqual(float(evidence[0]["feature_adapter_strength"]), 0.5)
        self.assertAlmostEqual(float(evidence[0]["feature_adapter_variance_floor"]), 1e-5)

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

    def test_support_quality_class_verifier_can_rerank_ambiguous_candidate(self):
        from phase2_collaborative_open_set_qknn_eval import (
            _support_quality_class_verifier,
            build_qknn_memory,
            qknn_scores,
        )

        features = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.99, 0.01, 0.0],
                [0.80, 0.60, 0.0],
                [0.79, 0.61, 0.0],
            ],
            dtype=np.float32,
        )
        memory = build_qknn_memory(
            features,
            ["old-a", "old-a", "new-a", "new-a"],
            old_labels={"old-a"},
            support_scenarios=["leo_clear_weak"] * 4,
        )
        query = np.asarray([[0.96, 0.28, 0.0]], dtype=np.float32)
        pred, _, _, _, _, _, _, _ = qknn_scores(memory, query, top_k=3)

        verifier = _support_quality_class_verifier(
            memory,
            query,
            rx="rx-a",
            scenario="leo_clear_weak",
            qknn_k=3,
            scenario_aware=False,
            radius_norm=0.0,
            old_bias=0.0,
            candidate_class_top_m=0,
            class_verifier_top_m=2,
            prototype_blend=0.0,
            mahalanobis_blend=0.0,
            mahalanobis_score_temp=0.2,
            receiver_threshold=0.0,
            receiver_class_thresholds={},
            receiver_class_conformal_scores={
                "rx-a": {
                    "old-a": [0.99, 0.99, 0.99],
                    "new-a": [0.10, 0.10, 0.10],
                }
            },
            receiver_class_reliabilities={"rx-a": {"old-a": 0.10, "new-a": 1.0}},
            score_threshold_combine="qknn_only",
            class_score_threshold_enabled=False,
            class_conformal_enabled=True,
            unknown_gate_mode="score",
            risk_temperature=0.035,
            radius_temperature=0.02,
            margin_temperature=0.02,
            mahalanobis_temperature=0.2,
            evt_temperature=0.05,
            oldness_temperature=0.05,
            class_shell_unknown_risk_enabled=False,
            class_shell_radius_scale=1.25,
            class_shell_risk_temperature=0.05,
            class_shell_risk_margin=0.0,
            pvalue_weight=4.0,
            reliability_weight=4.0,
            risk_weight=0.0,
        )

        self.assertEqual(str(pred[0]), "old-a")
        self.assertEqual(verifier["top1_label"], "new-a")
        self.assertEqual(verifier["second_label"], "old-a")
        self.assertGreater(verifier["top1_verified_score"], verifier["second_verified_score"])

    def test_support_quality_class_verifier_is_marked_in_metadata_and_evidence(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
                class_conformal_enabled=True,
                class_conformal_min_support=1,
                receiver_class_reliability_policy="support_calibrated",
                class_verifier_policy="support_quality",
                class_verifier_top_m=2,
                class_verifier_pvalue_weight=2.0,
                class_verifier_reliability_weight=3.0,
                class_verifier_risk_weight=0.5,
            )

        self.assertEqual(metadata["class_verifier_policy"], "support_quality")
        self.assertEqual(metadata["class_verifier_top_m"], 2)
        self.assertAlmostEqual(metadata["class_verifier_pvalue_weight"], 2.0)
        self.assertAlmostEqual(metadata["class_verifier_reliability_weight"], 3.0)
        self.assertAlmostEqual(metadata["class_verifier_risk_weight"], 0.5)
        self.assertIn("class_verifier_changed", evidence[0])
        self.assertIn("class_verifier_top1_verified_score", evidence[0])

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

    def test_virtual_unknown_calibration_is_support_derived_and_recorded(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
                virtual_unknown_calibration_enabled=True,
                virtual_unknown_samples_per_class=2,
                virtual_unknown_noise_scale=0.0,
            )

        self.assertEqual(metadata["threshold_scope"], "support_virtual_unknown")
        self.assertTrue(metadata["virtual_unknown_calibration_enabled"])
        self.assertEqual(metadata["virtual_unknown_samples_per_class"], 2)
        self.assertEqual({row["role"] for row in evidence}, {"old", "seen_new", "unknown"})
        self.assertEqual(int(evidence[0]["virtual_unknown_calibration_enabled"]), 1)
        self.assertGreater(int(evidence[0]["virtual_unknown_count"]), 0)

    def test_virtual_unknown_risk_is_independent_from_threshold_calibration(self):
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
                virtual_unknown_risk_enabled=True,
                virtual_unknown_risk_samples_per_class=2,
                virtual_unknown_noise_scale=0.0,
            )

        self.assertEqual(metadata["threshold_scope"], "support_known_only")
        self.assertFalse(metadata["virtual_unknown_calibration_enabled"])
        self.assertTrue(metadata["virtual_unknown_risk_enabled"])
        self.assertEqual(metadata["virtual_unknown_risk_samples_per_class"], 2)
        self.assertIn("virtual_unknown", metadata["active_risk_components"])
        self.assertEqual({row["role"] for row in evidence}, {"old", "seen_new", "unknown"})
        self.assertEqual(int(evidence[0]["virtual_unknown_calibration_enabled"]), 0)
        self.assertEqual(int(evidence[0]["virtual_unknown_risk_enabled"]), 1)
        self.assertGreater(int(evidence[0]["virtual_unknown_count"]), 0)
        self.assertIn("virtual_unknown_risk", evidence[0])
        self.assertIn("virtual_unknown_score", evidence[0])

    def test_class_negative_risk_is_support_derived_and_recorded(self):
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
                class_evidence_top_m=1,
                class_negative_risk_enabled=True,
                class_negative_samples_per_class=2,
                class_negative_mix_alpha=0.50,
                class_negative_neighbor_count=1,
                class_negative_risk_temperature=0.05,
            )

        self.assertTrue(metadata["class_negative_risk_enabled"])
        self.assertEqual(metadata["class_negative_samples_per_class"], 2)
        self.assertIn("class_negative", metadata["active_risk_components"])
        self.assertEqual({row["role"] for row in evidence}, {"old", "seen_new", "unknown"})
        self.assertEqual(int(evidence[0]["class_negative_risk_enabled"]), 1)
        self.assertIn("class_negative_risk", evidence[0])
        self.assertIn("class_negative_score", evidence[0])
        self.assertIn("class_evidence_top1_class_negative_risk", evidence[0])
        self.assertIn("class_evidence_top1_class_negative_score", evidence[0])

    def test_class_negative_weak_evidence_mode_records_raw_and_effective_risk(self):
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
                support_calibration_mode="leave_one_out",
                class_conformal_enabled=True,
                class_evidence_top_m=1,
                class_negative_risk_enabled=True,
                class_negative_samples_per_class=2,
                class_negative_mix_alpha=0.50,
                class_negative_neighbor_count=1,
                class_negative_risk_temperature=0.05,
                class_negative_combine_mode="weak_evidence",
                class_negative_weak_margin=0.10,
                class_negative_weak_pvalue=0.50,
                class_negative_weak_reliability=0.50,
            )

        self.assertEqual(metadata["class_negative_combine_mode"], "weak_evidence")
        self.assertAlmostEqual(metadata["class_negative_weak_margin"], 0.10)
        self.assertAlmostEqual(metadata["class_negative_weak_pvalue"], 0.50)
        self.assertAlmostEqual(metadata["class_negative_weak_reliability"], 0.50)
        row = evidence[0]
        self.assertEqual(row["class_negative_combine_mode"], "weak_evidence")
        self.assertIn("class_negative_raw_risk", row)
        self.assertIn("class_negative_weakness", row)
        self.assertIn("class_evidence_top1_class_negative_raw_risk", row)
        self.assertIn("class_evidence_top1_class_negative_weakness", row)
        self.assertGreaterEqual(float(row["class_negative_risk"]), 0.0)
        self.assertLessEqual(float(row["class_negative_risk"]), float(row["class_negative_raw_risk"]) + 1e-12)
        self.assertGreaterEqual(float(row["class_negative_weakness"]), 0.0)
        self.assertLessEqual(float(row["class_negative_weakness"]), 1.0)

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

    def test_class_conformal_pvalues_are_support_derived_and_recorded(self):
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
                class_conformal_enabled=True,
                class_conformal_min_support=1,
            )

        self.assertTrue(metadata["class_conformal_enabled"])
        self.assertTrue(metadata["receiver_class_conformal_counts"])
        first_rx_counts = next(iter(metadata["receiver_class_conformal_counts"].values()))
        self.assertIn("old-a", first_rx_counts)
        self.assertIn("new-a", first_rx_counts)
        self.assertEqual({row["role"] for row in evidence}, {"old", "seen_new", "unknown"})
        self.assertEqual(int(evidence[0]["class_conformal_enabled"]), 1)
        self.assertIn("class_conformal_pvalue", evidence[0])
        self.assertIn("class_conformal_support_count", evidence[0])

    def test_class_conformal_defaults_fail_closed_with_single_support(self):
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
                class_conformal_enabled=True,
            )

        self.assertEqual(metadata["class_conformal_min_support"], 2)
        self.assertTrue(metadata["receiver_class_conformal_counts"])
        first_rx_counts = next(iter(metadata["receiver_class_conformal_counts"].values()))
        self.assertEqual(first_rx_counts, {})
        self.assertEqual(int(evidence[0]["class_conformal_enabled"]), 1)
        self.assertEqual(float(evidence[0]["class_conformal_pvalue"]), 0.0)
        self.assertEqual(int(evidence[0]["class_conformal_support_count"]), 0)

    def test_class_evidence_top_m_records_per_label_scores_and_pvalues(self):
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
                class_conformal_enabled=True,
                class_conformal_min_support=1,
                class_evidence_top_m=2,
            )

        self.assertEqual(metadata["class_evidence_top_m"], 2)
        first = evidence[0]
        self.assertEqual(int(first["class_evidence_top_m"]), 2)
        self.assertIn("class_evidence_top1_label", first)
        self.assertIn("class_evidence_top1_score", first)
        self.assertIn("class_evidence_top1_margin", first)
        self.assertIn("class_evidence_top1_conformal_pvalue", first)
        self.assertIn("class_evidence_top1_support_count", first)
        self.assertIn("class_evidence_top1_unknown_risk", first)
        self.assertIn("class_evidence_top1_class_radius_z", first)
        self.assertIn("class_evidence_top2_label", first)
        self.assertIn("class_evidence_top2_score", first)
        self.assertIn("class_evidence_top2_margin", first)
        self.assertIn("class_evidence_top2_conformal_pvalue", first)
        self.assertIn("class_evidence_top2_support_count", first)
        self.assertIn("class_evidence_top2_unknown_risk", first)
        self.assertIn("class_evidence_top2_class_radius_z", first)

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
