import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class CollaborativeOpenSetQknnEvalTest(unittest.TestCase):
    def test_reports_counts_metrics_and_resource_telemetry_for_one_to_all_receivers(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "old-1",
                "receiver_id": "rx-a",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-b",
                "known_score": 0.60,
                "known_margin": 0.05,
                "unknown_risk": 0.20,
                "reliability": 0.70,
                "latency_ms": 4.0,
                "bytes": 96,
            },
            {
                "event_id": "old-1",
                "receiver_id": "rx-b",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": 0.90,
                "known_margin": 0.40,
                "unknown_risk": 0.10,
                "reliability": 0.95,
                "latency_ms": 6.0,
                "bytes": 96,
            },
            {
                "event_id": "new-1",
                "receiver_id": "rx-a",
                "role": "seen_new",
                "true_label": "new-a",
                "predicted_label": "new-a",
                "known_score": 0.82,
                "known_margin": 0.24,
                "unknown_risk": 0.18,
                "reliability": 0.90,
                "latency_ms": 5.0,
                "bytes": 96,
            },
            {
                "event_id": "new-1",
                "receiver_id": "rx-b",
                "role": "seen_new",
                "true_label": "new-a",
                "predicted_label": "new-a",
                "known_score": 0.70,
                "known_margin": 0.18,
                "unknown_risk": 0.25,
                "reliability": 0.80,
                "latency_ms": 7.0,
                "bytes": 96,
            },
            {
                "event_id": "unk-1",
                "receiver_id": "rx-a",
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "old-a",
                "known_score": 0.55,
                "known_margin": 0.05,
                "unknown_risk": 0.92,
                "reliability": 0.90,
                "latency_ms": 8.0,
                "bytes": 96,
            },
            {
                "event_id": "unk-1",
                "receiver_id": "rx-b",
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "new-a",
                "known_score": 0.52,
                "known_margin": 0.02,
                "unknown_risk": 0.88,
                "reliability": 0.85,
                "latency_ms": 9.0,
                "bytes": 96,
            },
        ]

        result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="all",
            unknown_risk_threshold=0.80,
            accept_margin_threshold=0.10,
        )

        self.assertEqual(result["receiver_count"], 2)
        self.assertEqual(set(result["counts"]), {"1", "2"})
        self.assertEqual(result["counts"]["1"]["participating_receivers"], 1)
        self.assertEqual(result["counts"]["2"]["participating_receivers"], 2)
        self.assertGreaterEqual(result["counts"]["2"]["old_acc"], result["counts"]["1"]["old_acc"])
        self.assertEqual(result["counts"]["2"]["seen_new_acc"], 1.0)
        self.assertEqual(result["counts"]["2"]["unknown_reject_rate"], 1.0)
        self.assertEqual(result["counts"]["2"]["unknown_FAR"], 0.0)
        self.assertEqual(result["counts"]["2"]["min_old_class_acc"], 1.0)
        self.assertEqual(result["counts"]["2"]["bytes_per_event"], 192.0)
        self.assertGreaterEqual(result["counts"]["2"]["latency_ms_p95"], result["counts"]["2"]["latency_ms_p50"])

    def test_rejects_threshold_fitting_from_unknown_query_rows(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        with self.assertRaisesRegex(ValueError, "unknown query"):
            evaluate_collaborative_open_set_evidence(
                [
                    {
                        "event_id": "unk-1",
                        "receiver_id": "rx-a",
                        "role": "unknown",
                        "true_label": "__unknown__",
                        "predicted_label": "old-a",
                        "unknown_risk": 0.9,
                        "calibration_role": "threshold_fit",
                    }
                ]
            )

    def test_fails_closed_on_unknown_role_and_bad_event_join(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        with self.assertRaisesRegex(ValueError, "unknown evidence role"):
            evaluate_collaborative_open_set_evidence(
                [
                    {
                        "event_id": "bad-role",
                        "receiver_id": "rx-a",
                        "role": "target-new-typo",
                        "true_label": "12-20",
                    }
                ]
            )

        with self.assertRaisesRegex(ValueError, "duplicate receiver_id"):
            evaluate_collaborative_open_set_evidence(
                [
                    {
                        "event_id": "dup",
                        "receiver_id": "rx-a",
                        "role": "old",
                        "true_label": "old-a",
                    },
                    {
                        "event_id": "dup",
                        "receiver_id": "rx-a",
                        "role": "old",
                        "true_label": "old-a",
                    },
                ]
            )

        with self.assertRaisesRegex(ValueError, "inconsistent true_label"):
            evaluate_collaborative_open_set_evidence(
                [
                    {
                        "event_id": "mixed",
                        "receiver_id": "rx-a",
                        "role": "old",
                        "true_label": "old-a",
                    },
                    {
                        "event_id": "mixed",
                        "receiver_id": "rx-b",
                        "role": "old",
                        "true_label": "old-b",
                    },
                ]
            )

    def test_records_threshold_scope_and_requires_prior_reliability_for_dynamic_selection(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "old-1",
                "receiver_id": "rx-a",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": 0.8,
                "known_margin": 0.4,
                "unknown_risk": 0.1,
                "reliability": 0.9,
            }
        ]

        result = evaluate_collaborative_open_set_evidence(
            rows,
            threshold_selection_label_scope="source_only",
            unknown_query_eval_only=True,
        )
        self.assertEqual(result["threshold_selection_label_scope"], "source_only")
        self.assertEqual(result["denominator_policy"], "per_k_available_receivers")
        self.assertEqual(result["evidence_scope"], "offline_evidence_metrics_only")

        with self.assertRaisesRegex(ValueError, "threshold_selection_label_scope"):
            evaluate_collaborative_open_set_evidence(rows, threshold_selection_label_scope="unknown_query")

        with self.assertRaisesRegex(ValueError, "reliability_source"):
            evaluate_collaborative_open_set_evidence(
                rows,
                receiver_selection_policy="reliability_prior",
            )

    def test_high_unknown_risk_vetoes_overconfident_acceptance(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        result = evaluate_collaborative_open_set_evidence(
            [
                {
                    "event_id": "overconfident-unknown",
                    "receiver_id": "rx-a",
                    "role": "unknown",
                    "true_label": "__unknown__",
                    "predicted_label": "old-a",
                    "known_score": 0.99,
                    "known_margin": 0.95,
                    "unknown_risk": 0.99,
                    "latency_ms": 3.0,
                    "bytes": 32,
                }
            ],
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
        )

        k1 = result["counts"]["1"]
        self.assertEqual(k1["unknown_FAR"], 0.0)
        self.assertEqual(k1["defer_rate"], 1.0)
        self.assertEqual(k1["open_set_confusion"], {"unknown->defer": 1})

    def test_consensus_veto_rejects_high_risk_low_consensus_unknown(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "split-unknown",
                "receiver_id": "rx-a",
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "old-a",
                "known_score": 0.65,
                "known_margin": 0.50,
                "unknown_risk": 0.99,
            },
            {
                "event_id": "split-unknown",
                "receiver_id": "rx-b",
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "new-a",
                "known_score": 0.61,
                "known_margin": 0.45,
                "unknown_risk": 0.97,
            },
        ]

        default_result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
        )
        veto_result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            fusion_policy="consensus_veto",
            consensus_gap_threshold=0.5,
        )

        self.assertEqual(default_result["counts"]["2"]["open_set_confusion"], {"unknown->defer": 1})
        self.assertEqual(veto_result["counts"]["2"]["unknown_reject_rate"], 1.0)
        self.assertEqual(veto_result["counts"]["2"]["open_set_confusion"], {"unknown->unknown_reject": 1})
        self.assertEqual(veto_result["fusion_policy"], "consensus_veto")

    def test_scorer_cvs_requests_more_receivers_under_latency_budget(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "uncertain-old",
                "receiver_id": "rx-a",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": 0.45,
                "known_margin": 0.02,
                "unknown_risk": 0.30,
                "radius_risk": 0.20,
                "margin_risk": 0.30,
                "latency_ms": 2.0,
            },
            {
                "event_id": "uncertain-old",
                "receiver_id": "rx-b",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": 0.48,
                "known_margin": 0.03,
                "unknown_risk": 0.35,
                "radius_risk": 0.25,
                "margin_risk": 0.35,
                "latency_ms": 2.5,
            },
        ]

        result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="1",
            fusion_policy="scorer_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_gap_threshold=0.5,
            consensus_score_threshold=0.6,
            latency_budget_ms=5.0,
        )

        k1 = result["counts"]["1"]
        self.assertEqual(k1["request_more_rate"], 1.0)
        self.assertEqual(k1["unresolved_rate"], 1.0)
        self.assertEqual(k1["open_set_confusion"], {"old->request_more": 1})

    def test_scorer_cvs_rejects_high_risk_without_known_rescue(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "unknown-risk",
                "receiver_id": "rx-a",
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "old-a",
                "known_score": 0.40,
                "known_margin": 0.02,
                "unknown_risk": 0.95,
                "radius_risk": 0.90,
                "margin_risk": 0.92,
            },
            {
                "event_id": "unknown-risk",
                "receiver_id": "rx-b",
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "old-b",
                "known_score": 0.35,
                "known_margin": 0.01,
                "unknown_risk": 0.97,
                "radius_risk": 0.94,
                "margin_risk": 0.93,
            },
        ]

        result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            fusion_policy="scorer_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_gap_threshold=0.5,
            consensus_score_threshold=0.6,
        )

        k2 = result["counts"]["2"]
        self.assertEqual(k2["unknown_reject_rate"], 1.0)
        self.assertEqual(k2["open_set_confusion"], {"unknown->unknown_reject": 1})
        self.assertEqual(result["fusion_policy"], "scorer_cvs")

    def test_scorer_cvs_component_vote_uses_mahalanobis_as_extra_channel(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        row = {
            "event_id": "one-hot-risk",
            "receiver_id": "rx-a",
            "role": "unknown",
            "true_label": "__unknown__",
            "predicted_label": "old-a",
            "known_score": 0.20,
            "known_margin": 0.01,
            "unknown_risk": 0.95,
            "score_risk": 0.10,
            "radius_risk": 0.10,
            "margin_risk": 0.10,
            "mahalanobis_risk": 0.95,
        }

        result = evaluate_collaborative_open_set_evidence(
            [row],
            collab_counts="1",
            fusion_policy="scorer_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.6,
            scorer_component_vote_threshold=0.5,
        )

        k1 = result["counts"]["1"]
        self.assertEqual(k1["unknown_reject_rate"], 0.0)
        self.assertEqual(k1["open_set_confusion"], {"unknown->defer": 1})

    def test_scorer_cvs_component_vote_uses_evt_as_extra_channel(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        row = {
            "event_id": "evt-risk",
            "receiver_id": "rx-a",
            "role": "unknown",
            "true_label": "__unknown__",
            "predicted_label": "old-a",
            "known_score": 0.20,
            "known_margin": 0.01,
            "unknown_risk": 0.95,
            "score_risk": 0.10,
            "radius_risk": 0.10,
            "margin_risk": 0.10,
            "evt_risk": 0.95,
        }

        result = evaluate_collaborative_open_set_evidence(
            [row],
            collab_counts="1",
            fusion_policy="scorer_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.6,
            scorer_component_vote_threshold=0.5,
        )

        k1 = result["counts"]["1"]
        self.assertEqual(k1["unknown_reject_rate"], 0.0)
        self.assertEqual(k1["open_set_confusion"], {"unknown->defer": 1})

    def test_scorer_cvs_component_vote_uses_oldness_as_extra_channel(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        row = {
            "event_id": "oldness-risk",
            "receiver_id": "rx-a",
            "role": "unknown",
            "true_label": "__unknown__",
            "predicted_label": "old-a",
            "known_score": 0.20,
            "known_margin": 0.01,
            "unknown_risk": 0.95,
            "score_risk": 0.10,
            "radius_risk": 0.10,
            "margin_risk": 0.10,
            "oldness_risk": 0.95,
        }

        result = evaluate_collaborative_open_set_evidence(
            [row],
            collab_counts="1",
            fusion_policy="scorer_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.6,
            scorer_component_vote_threshold=0.5,
        )

        k1 = result["counts"]["1"]
        self.assertEqual(k1["unknown_reject_rate"], 0.0)
        self.assertEqual(k1["open_set_confusion"], {"unknown->defer": 1})

    def test_strict_protocol_metadata_validates_stage2_boundaries(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        row = {
            "event_id": "new-numeric-label",
            "receiver_id": "20-1",
            "role": "seen_new",
            "true_label": "12-20",
            "predicted_label": "12-20",
            "known_score": 0.9,
            "known_margin": 0.3,
            "unknown_risk": 0.1,
        }

        with self.assertRaisesRegex(ValueError, "disjoint"):
            evaluate_collaborative_open_set_evidence(
                [row],
                strict_protocol_metadata=True,
                protocol_metadata={
                    "source_receiver_ids": ["20-1"],
                    "target_receiver_ids": ["20-1"],
                    "old_tx_ids": ["14-10"],
                    "seen_new_tx_ids": ["12-20"],
                    "unknown_tx_ids": ["13-20"],
                    "target_channel_view": "leo_clear_weak",
                },
            )

        result = evaluate_collaborative_open_set_evidence(
            [row],
            strict_protocol_metadata=True,
            protocol_metadata={
                "source_receiver_ids": ["1-1"],
                "target_receiver_ids": ["20-1"],
                "old_tx_ids": ["14-10"],
                "seen_new_tx_ids": ["12-20"],
                "unknown_tx_ids": ["13-20"],
                "target_channel_view": "leo_clear_weak",
            },
        )
        self.assertTrue(result["stage2_protocol"]["validated"])

    def test_seen_new_per_class_floor_does_not_depend_on_label_prefix(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        result = evaluate_collaborative_open_set_evidence(
            [
                {
                    "event_id": "new-numeric-label",
                    "receiver_id": "rx-a",
                    "role": "seen_new",
                    "true_label": "12-20",
                    "predicted_label": "12-20",
                    "known_score": 0.9,
                    "known_margin": 0.3,
                    "unknown_risk": 0.1,
                    "reliability": 1.0,
                    "latency_ms": 3.0,
                    "bytes": 72,
                }
            ],
            collab_counts="all",
        )

        k1 = result["counts"]["1"]
        self.assertEqual(k1["per_seen_new_class_acc"], {"12-20": 1.0})
        self.assertEqual(k1["min_seen_new_class_acc"], 1.0)

    def test_protocol_expected_classes_are_counted_when_absent_from_matched_subset(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        result = evaluate_collaborative_open_set_evidence(
            [
                {
                    "event_id": "old-present",
                    "receiver_id": "rx-a",
                    "role": "old",
                    "true_label": "old-a",
                    "predicted_label": "old-a",
                    "known_score": 0.9,
                    "known_margin": 0.3,
                    "unknown_risk": 0.1,
                }
            ],
            strict_protocol_metadata=True,
            protocol_metadata={
                "source_receiver_ids": ["src-a"],
                "target_receiver_ids": ["rx-a"],
                "old_tx_ids": ["old-a", "old-missing"],
                "seen_new_tx_ids": ["new-missing"],
                "unknown_tx_ids": ["unk-a"],
                "target_channel_view": "leo_clear_weak",
            },
        )

        k1 = result["counts"]["1"]
        self.assertEqual(k1["missing_old_classes"], ["old-missing"])
        self.assertEqual(k1["missing_seen_new_classes"], ["new-missing"])
        self.assertEqual(k1["per_old_class_acc"]["old-missing"], 0.0)
        self.assertEqual(k1["min_old_class_acc"], 0.0)


if __name__ == "__main__":
    unittest.main()
