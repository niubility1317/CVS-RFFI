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
        self.assertIn("radius_risk", evidence[0])
        self.assertIn("margin_risk", evidence[0])
        self.assertIn("class_radius", evidence[0])

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
        pred_global, _, _ = qknn_scores(
            memory,
            np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
            top_k=1,
            query_scenarios=["leo_clear_weak"],
            scenario_aware=False,
        )
        pred_scenario, _, _ = qknn_scores(
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
        _, self_scores, _ = qknn_scores(memory, features, top_k=1)
        _, loo_scores, _ = qknn_scores(memory, features, top_k=1, exclude_support_indices=range(features.shape[0]))

        self.assertLess(float(np.mean(loo_scores)), float(np.mean(self_scores)))

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
                support_calibration_mode="leave_one_out",
                score_threshold_combine="qknn_only",
            )

        self.assertTrue(metadata["scenario_aware"])
        self.assertAlmostEqual(metadata["radius_norm"], 0.3)
        self.assertAlmostEqual(metadata["old_bias"], 0.1)
        self.assertEqual(metadata["support_calibration_mode"], "leave_one_out")
        self.assertEqual(metadata["score_threshold_combine"], "qknn_only")


if __name__ == "__main__":
    unittest.main()
