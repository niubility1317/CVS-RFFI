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

        result = evaluate_collaborative_open_set_evidence(
            rows,
            threshold_selection_label_scope="support_virtual_unknown",
            unknown_query_eval_only=True,
        )
        self.assertEqual(result["threshold_selection_label_scope"], "support_virtual_unknown")

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

    def test_resource_budget_forces_defer_when_selected_receivers_exceed_bytes(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "budget-old",
                "receiver_id": "rx-a",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": 0.95,
                "known_margin": 0.40,
                "unknown_risk": 0.05,
                "latency_ms": 2.0,
                "bytes": 96,
            },
            {
                "event_id": "budget-old",
                "receiver_id": "rx-b",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": 0.94,
                "known_margin": 0.38,
                "unknown_risk": 0.05,
                "latency_ms": 2.5,
                "bytes": 96,
            },
        ]

        result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            fusion_policy="scorer_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.6,
            max_event_bytes=120.0,
        )

        k2 = result["counts"]["2"]
        self.assertEqual(k2["old_acc"], 0.0)
        self.assertEqual(k2["open_set_confusion"], {"old->defer": 1})
        self.assertEqual(k2["resource_budget_violation_count"], 1)
        self.assertEqual(k2["resource_budget_violation_rate"], 1.0)

    def test_progressive_budget_requests_receivers_until_confident(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "progressive-old",
                "receiver_id": "rx-a",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": 0.45,
                "known_margin": 0.02,
                "unknown_risk": 0.30,
                "score_risk": 0.10,
                "radius_risk": 0.20,
                "margin_risk": 0.30,
                "latency_ms": 2.0,
                "bytes": 96,
            },
            {
                "event_id": "progressive-old",
                "receiver_id": "rx-b",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": 0.90,
                "known_margin": 0.30,
                "unknown_risk": 0.05,
                "score_risk": 0.05,
                "radius_risk": 0.05,
                "margin_risk": 0.05,
                "latency_ms": 2.5,
                "bytes": 96,
            },
        ]

        result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="1,2",
            fusion_policy="scorer_cvs",
            collaboration_policy="progressive_budget",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_gap_threshold=0.5,
            consensus_score_threshold=0.6,
            latency_budget_ms=5.0,
        )

        self.assertEqual(result["collaboration_policy"], "progressive_budget")
        self.assertEqual(result["counts"]["1"]["open_set_confusion"], {"old->defer": 1})
        self.assertEqual(result["counts"]["1"]["participating_receivers_avg"], 1.0)
        self.assertEqual(result["counts"]["2"]["open_set_confusion"], {"old->old": 1})
        self.assertEqual(result["counts"]["2"]["old_acc"], 1.0)
        self.assertEqual(result["counts"]["2"]["participating_receivers_avg"], 2.0)
        self.assertEqual(result["counts"]["2"]["bytes_per_event"], 192.0)

    def test_adaptive_gain_requests_high_risk_unknown_boundary_receivers(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "adaptive-unknown",
                "receiver_id": "rx-a",
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "old-a",
                "known_score": 0.55,
                "known_margin": 0.04,
                "unknown_risk": 0.70,
                "score_risk": 0.70,
                "radius_risk": 0.65,
                "margin_risk": 0.60,
                "reliability": 0.90,
                "latency_ms": 2.0,
                "bytes": 72,
            },
            {
                "event_id": "adaptive-unknown",
                "receiver_id": "rx-b",
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "old-b",
                "known_score": 0.20,
                "known_margin": 0.01,
                "unknown_risk": 0.95,
                "score_risk": 0.93,
                "radius_risk": 0.94,
                "margin_risk": 0.92,
                "reliability": 0.95,
                "latency_ms": 3.0,
                "bytes": 72,
            },
            {
                "event_id": "adaptive-unknown",
                "receiver_id": "rx-c",
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "old-a",
                "known_score": 0.90,
                "known_margin": 0.35,
                "unknown_risk": 0.10,
                "score_risk": 0.10,
                "radius_risk": 0.10,
                "margin_risk": 0.10,
                "reliability": 0.80,
                "latency_ms": 1.0,
                "bytes": 72,
            },
        ]

        result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="1,2",
            fusion_policy="scorer_cvs",
            collaboration_policy="adaptive_gain",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_gap_threshold=0.5,
            consensus_score_threshold=0.6,
            scorer_component_vote_threshold=0.5,
            adaptive_gain_min_risk=0.6,
            latency_budget_ms=8.0,
        )

        self.assertEqual(result["collaboration_policy"], "adaptive_gain")
        self.assertEqual(result["counts"]["1"]["open_set_confusion"], {"unknown->defer": 1})
        self.assertEqual(result["counts"]["2"]["unknown_reject_rate"], 1.0)
        self.assertEqual(result["counts"]["2"]["participating_receivers_avg"], 2.0)
        self.assertEqual(result["counts"]["2"]["collaboration_stop_reasons"], {"budget_exhausted_unknown_reject": 1})

    def test_adaptive_gain_can_pick_candidate_outside_fixed_prefix(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "adaptive-old",
                "receiver_id": "rx-a",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-b",
                "known_score": 0.50,
                "known_margin": 0.02,
                "unknown_risk": 0.40,
                "reliability": 0.90,
                "latency_ms": 2.0,
                "bytes": 72,
            },
            {
                "event_id": "adaptive-old",
                "receiver_id": "rx-b",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-b",
                "known_score": 0.52,
                "known_margin": 0.02,
                "unknown_risk": 0.45,
                "reliability": 0.10,
                "latency_ms": 20.0,
                "bytes": 720,
            },
            {
                "event_id": "adaptive-old",
                "receiver_id": "rx-c",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": 0.95,
                "known_margin": 0.50,
                "unknown_risk": 0.05,
                "reliability": 0.99,
                "latency_ms": 1.0,
                "bytes": 72,
            },
        ]

        result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            fusion_policy="scorer_cvs",
            collaboration_policy="adaptive_gain",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_gap_threshold=0.3,
            consensus_score_threshold=0.6,
            adaptive_gain_min_risk=0.3,
            adaptive_gain_latency_weight=0.1,
            adaptive_gain_bytes_weight=0.01,
            latency_budget_ms=8.0,
        )

        self.assertEqual(result["counts"]["2"]["old_acc"], 1.0)
        self.assertEqual(result["counts"]["2"]["bytes_per_event"], 144.0)
        self.assertEqual(result["counts"]["2"]["participating_receivers_max"], 2)

    def test_support_utility_prefers_supported_low_cost_receiver(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "support-utility-old",
                "receiver_id": "rx-a",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-b",
                "known_score": 0.45,
                "known_margin": 0.02,
                "unknown_risk": 0.55,
                "reliability": 0.80,
                "support_density": 0.30,
                "class_conformal_pvalue": 0.10,
                "class_conformal_support_count": 1,
                "latency_ms": 2.0,
                "bytes": 72,
            },
            {
                "event_id": "support-utility-old",
                "receiver_id": "rx-b",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-b",
                "known_score": 0.65,
                "known_margin": 0.08,
                "unknown_risk": 0.50,
                "reliability": 0.95,
                "support_density": 0.10,
                "class_conformal_pvalue": 0.05,
                "class_conformal_support_count": 1,
                "latency_ms": 20.0,
                "bytes": 300,
            },
            {
                "event_id": "support-utility-old",
                "receiver_id": "rx-c",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": 0.95,
                "known_margin": 0.45,
                "unknown_risk": 0.05,
                "reliability": 0.90,
                "support_density": 0.95,
                "class_conformal_pvalue": 0.90,
                "class_conformal_support_count": 3,
                "latency_ms": 1.0,
                "bytes": 72,
            },
        ]

        result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            fusion_policy="scorer_cvs",
            collaboration_policy="support_utility",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_gap_threshold=0.3,
            consensus_score_threshold=0.6,
            adaptive_gain_min_risk=0.3,
            adaptive_gain_latency_weight=0.1,
            adaptive_gain_bytes_weight=0.01,
            max_event_bytes=200,
            latency_budget_ms=8.0,
        )

        k2 = result["counts"]["2"]
        self.assertEqual(result["collaboration_policy"], "support_utility")
        self.assertEqual(k2["old_acc"], 1.0)
        self.assertEqual(k2["bytes_per_event"], 144.0)
        self.assertEqual(k2["participating_receivers_avg"], 2.0)
        self.assertEqual(k2["collaboration_stop_reasons"], {"budget_exhausted_accept": 1})

    def test_rb_capr_utility_prefers_role_balanced_supported_receiver(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "rb-capr-old",
                "receiver_id": "rx-a",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-b",
                "known_score": 0.44,
                "known_margin": 0.02,
                "unknown_risk": 0.58,
                "risk_component_agreement": 0.35,
                "reliability": 0.80,
                "support_density": 0.35,
                "class_conformal_pvalue": 0.10,
                "class_conformal_support_count": 1,
                "latency_ms": 2.0,
                "bytes": 72,
            },
            {
                "event_id": "rb-capr-old",
                "receiver_id": "rx-b",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "new-a",
                "known_score": 0.88,
                "known_margin": 0.35,
                "unknown_risk": 0.15,
                "risk_component_agreement": 0.10,
                "reliability": 0.95,
                "support_density": 0.90,
                "class_conformal_pvalue": 0.95,
                "class_conformal_support_count": 3,
                "latency_ms": 1.0,
                "bytes": 72,
            },
            {
                "event_id": "rb-capr-old",
                "receiver_id": "rx-c",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": 0.90,
                "known_margin": 0.40,
                "unknown_risk": 0.08,
                "risk_component_agreement": 0.05,
                "reliability": 0.90,
                "support_density": 0.92,
                "class_conformal_pvalue": 0.90,
                "class_conformal_support_count": 3,
                "latency_ms": 1.0,
                "bytes": 72,
            },
        ]

        metadata = {
            "source_receiver_ids": ["src-a", "src-b"],
            "target_receiver_ids": ["rx-a", "rx-b", "rx-c"],
            "old_tx_ids": ["old-a", "old-b"],
            "seen_new_tx_ids": ["new-a"],
            "unknown_tx_ids": ["unk-a"],
            "target_channel_view": "leo_clear_weak",
        }
        result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            fusion_policy="scorer_cvs",
            collaboration_policy="rb_capr_utility",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_gap_threshold=0.3,
            consensus_score_threshold=0.6,
            adaptive_gain_min_risk=0.3,
            adaptive_gain_latency_weight=0.1,
            adaptive_gain_bytes_weight=0.01,
            rb_capr_utility_min_delta=0.01,
            rb_capr_seen_new_balance_weight=0.10,
            rb_capr_old_floor_weight=0.80,
            rb_capr_unknown_confirm_weight=0.10,
            rb_capr_max_avg_rx_target=2.0,
            protocol_metadata=metadata,
            strict_protocol_metadata=True,
        )

        k2 = result["counts"]["2"]
        self.assertEqual(result["collaboration_policy"], "rb_capr_utility")
        self.assertEqual(k2["old_acc"], 1.0)
        self.assertEqual(k2["participating_receivers_avg"], 2.0)
        self.assertEqual(k2["collaboration_stop_reasons"], {"budget_exhausted_accept": 1})

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

    def test_seen_new_rescue_accepts_high_confidence_enrolled_class(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        row = {
            "event_id": "seen-new-rescue",
            "receiver_id": "rx-a",
            "role": "seen_new",
            "true_label": "new-a",
            "predicted_label": "new-a",
            "known_score": 0.95,
            "known_margin": 0.45,
            "unknown_risk": 0.95,
            "score_risk": 0.95,
            "radius_risk": 0.94,
            "margin_risk": 0.93,
        }
        metadata = {
            "source_receiver_ids": ["src-a"],
            "target_receiver_ids": ["rx-a"],
            "old_tx_ids": ["old-a"],
            "seen_new_tx_ids": ["new-a"],
            "unknown_tx_ids": ["unk-a"],
            "target_channel_view": "leo_clear_weak",
        }

        no_rescue = evaluate_collaborative_open_set_evidence(
            [row],
            collab_counts="1",
            fusion_policy="scorer_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.6,
            strict_protocol_metadata=True,
            protocol_metadata=metadata,
        )
        rescued = evaluate_collaborative_open_set_evidence(
            [row],
            collab_counts="1",
            fusion_policy="scorer_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.6,
            strict_protocol_metadata=True,
            protocol_metadata=metadata,
            seen_new_rescue_enabled=True,
            seen_new_rescue_risk_scale=0.5,
            seen_new_rescue_min_score=0.8,
            seen_new_rescue_min_margin=0.2,
        )

        self.assertEqual(no_rescue["counts"]["1"]["open_set_confusion"], {"seen_new->defer": 1})
        self.assertEqual(rescued["counts"]["1"]["seen_new_acc"], 1.0)
        self.assertEqual(rescued["counts"]["1"]["seen_new_rescue_count"], 1)
        self.assertEqual(rescued["seen_new_rescue_enabled"], True)

    def test_seen_new_rescue_guard_does_not_apply_to_unknown_role(self):
        from evaluation.collaborative_open_set_qknn_eval import _fuse_event

        base_row = {
            "event_id": "role-invariance",
            "receiver_id": "rx-a",
            "predicted_label": "new-a",
            "known_score": 0.95,
            "known_margin": 0.45,
            "unknown_risk": 0.95,
            "score_risk": 0.95,
            "radius_risk": 0.94,
            "margin_risk": 0.93,
        }
        common = dict(
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            unknown_quantile=0.75,
            fusion_policy="scorer_cvs",
            consensus_gap_threshold=0.0,
            consensus_score_threshold=0.6,
            scorer_component_vote_threshold=0.5,
            seen_new_rescue_labels={"new-a"},
            seen_new_rescue_enabled=True,
            seen_new_rescue_risk_scale=0.5,
            seen_new_rescue_min_score=0.8,
            seen_new_rescue_min_margin=0.2,
        )

        as_seen_new = _fuse_event([{**base_row, "role": "seen_new", "true_label": "new-a"}], **common)
        as_unknown = _fuse_event([{**base_row, "role": "unknown", "true_label": "__unknown__"}], **common)

        self.assertTrue(as_seen_new["seen_new_rescue_applied"])
        self.assertEqual(as_seen_new["decision"], "accept")
        self.assertEqual(as_seen_new["output_label"], "new-a")
        self.assertFalse(as_unknown["seen_new_rescue_applied"])
        self.assertNotEqual(as_unknown["decision"], "accept")
        self.assertNotEqual(as_unknown["output_label"], "new-a")

    def test_class_set_gate_guards_seen_new_rescue_without_true_role(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        row = {
            "event_id": "unknown-looks-new",
            "receiver_id": "rx-a",
            "role": "unknown",
            "true_label": "__unknown__",
            "predicted_label": "new-a",
            "known_score": 0.95,
            "known_margin": 0.45,
            "unknown_risk": 0.95,
            "score_risk": 0.95,
            "radius_risk": 0.94,
            "margin_risk": 0.93,
        }
        metadata = {
            "source_receiver_ids": ["src-a"],
            "target_receiver_ids": ["rx-a"],
            "old_tx_ids": ["old-a"],
            "seen_new_tx_ids": ["new-a"],
            "unknown_tx_ids": ["unk-a"],
            "target_channel_view": "leo_clear_weak",
        }

        result = evaluate_collaborative_open_set_evidence(
            [row],
            collab_counts="1",
            fusion_policy="scorer_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.6,
            strict_protocol_metadata=True,
            protocol_metadata=metadata,
            seen_new_rescue_enabled=True,
            seen_new_rescue_risk_scale=0.5,
            seen_new_rescue_min_score=0.8,
            seen_new_rescue_min_margin=0.2,
            class_set_gate_enabled=True,
            seen_new_gate_max_effective_unknown_risk=0.4,
        )

        self.assertEqual(result["counts"]["1"]["unknown_FAR"], 0.0)
        self.assertEqual(result["counts"]["1"]["unknown_defer"], 1)

    def test_vote_margin_label_fusion_can_select_receiver_consensus(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "vote-margin",
                "receiver_id": "rx-a",
                "role": "old",
                "true_label": "old-b",
                "predicted_label": "old-a",
                "known_score": 0.95,
                "known_margin": 0.20,
                "unknown_risk": 0.10,
            },
            {
                "event_id": "vote-margin",
                "receiver_id": "rx-b",
                "role": "old",
                "true_label": "old-b",
                "predicted_label": "old-b",
                "known_score": 0.45,
                "known_margin": 0.30,
                "unknown_risk": 0.10,
            },
            {
                "event_id": "vote-margin",
                "receiver_id": "rx-c",
                "role": "old",
                "true_label": "old-b",
                "predicted_label": "old-b",
                "known_score": 0.44,
                "known_margin": 0.31,
                "unknown_risk": 0.10,
            },
        ]
        metadata = {
            "source_receiver_ids": ["src-a"],
            "target_receiver_ids": ["rx-a", "rx-b", "rx-c"],
            "old_tx_ids": ["old-a", "old-b"],
            "seen_new_tx_ids": ["new-a"],
            "unknown_tx_ids": ["unk-a"],
            "target_channel_view": "leo_clear_weak",
        }

        score_sum = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="3",
            fusion_policy="scorer_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.1,
            protocol_metadata=metadata,
            strict_protocol_metadata=True,
        )
        vote_margin = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="3",
            fusion_policy="scorer_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.1,
            label_fusion_policy="vote_margin",
            protocol_metadata=metadata,
            strict_protocol_metadata=True,
        )

        self.assertEqual(score_sum["counts"]["3"]["old_acc"], 0.0)
        self.assertEqual(vote_margin["counts"]["3"]["old_acc"], 1.0)
        self.assertEqual(vote_margin["label_fusion_policy"], "vote_margin")

    def test_class_reliability_can_select_lower_score_reliable_label(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        base_row = {
            "event_id": "class-rel",
            "receiver_id": "rx-a",
            "role": "old",
            "true_label": "old-a",
            "predicted_label": "old-b",
            "known_score": 1.00,
            "known_margin": 0.01,
            "unknown_risk": 0.95,
            "score_risk": 0.95,
            "radius_risk": 0.95,
            "margin_risk": 0.95,
            "class_conformal_pvalue": 0.01,
            "class_conformal_support_count": 2,
            "class_evidence_top_m": 2,
            "class_evidence_top1_label": "old-b",
            "class_evidence_top1_score": 1.00,
            "class_evidence_top1_margin": 0.01,
            "class_evidence_top1_conformal_pvalue": 0.01,
            "class_evidence_top1_support_count": 2,
            "class_evidence_top1_unknown_risk": 0.95,
            "class_evidence_top1_score_risk": 0.95,
            "class_evidence_top1_radius_risk": 0.95,
            "class_evidence_top1_margin_risk": 0.95,
            "class_evidence_top2_label": "old-a",
            "class_evidence_top2_score": 0.45,
            "class_evidence_top2_margin": 0.40,
            "class_evidence_top2_conformal_pvalue": 0.95,
            "class_evidence_top2_support_count": 2,
            "class_evidence_top2_unknown_risk": 0.10,
            "class_evidence_top2_score_risk": 0.10,
            "class_evidence_top2_radius_risk": 0.10,
            "class_evidence_top2_margin_risk": 0.10,
        }
        second_row = dict(base_row)
        second_row["receiver_id"] = "rx-b"
        rows = [base_row, second_row]
        metadata = {
            "source_receiver_ids": ["src-a"],
            "target_receiver_ids": ["rx-a", "rx-b"],
            "old_tx_ids": ["old-a", "old-b"],
            "seen_new_tx_ids": ["new-a"],
            "unknown_tx_ids": ["unk-a"],
            "target_channel_view": "leo_clear_weak",
        }

        default_result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            fusion_policy="cp_set_cvs",
            unknown_risk_threshold=0.98,
            accept_margin_threshold=0.03,
            consensus_score_threshold=0.1,
            conformal_rescue_min_pvalue=0.15,
            protocol_metadata=metadata,
            strict_protocol_metadata=True,
        )
        reliable_result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            fusion_policy="cp_set_cvs",
            unknown_risk_threshold=0.98,
            accept_margin_threshold=0.03,
            consensus_score_threshold=0.1,
            conformal_rescue_min_pvalue=0.15,
            class_reliability_policy="conformal_margin_risk",
            protocol_metadata=metadata,
            strict_protocol_metadata=True,
        )

        self.assertEqual(default_result["counts"]["2"]["old_acc"], 0.0)
        self.assertEqual(reliable_result["counts"]["2"]["old_acc"], 1.0)
        self.assertEqual(reliable_result["class_reliability_policy"], "conformal_margin_risk")
        self.assertLess(reliable_result["counts"]["2"]["mean_label_class_reliability"], 1.0)

    def test_weighted_vote_margin_uses_class_reliability_for_tied_receiver_votes(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        base_row = {
            "event_id": "weighted-vote",
            "receiver_id": "rx-a",
            "role": "old",
            "true_label": "old-a",
            "predicted_label": "old-b",
            "known_score": 1.00,
            "known_margin": 0.60,
            "unknown_risk": 0.95,
            "score_risk": 0.95,
            "radius_risk": 0.95,
            "margin_risk": 0.95,
            "class_conformal_pvalue": 0.01,
            "class_conformal_support_count": 2,
            "class_evidence_top_m": 2,
            "class_evidence_top1_label": "old-b",
            "class_evidence_top1_score": 1.00,
            "class_evidence_top1_margin": 0.60,
            "class_evidence_top1_conformal_pvalue": 0.01,
            "class_evidence_top1_support_count": 2,
            "class_evidence_top1_unknown_risk": 0.95,
            "class_evidence_top1_score_risk": 0.95,
            "class_evidence_top1_radius_risk": 0.95,
            "class_evidence_top1_margin_risk": 0.95,
            "class_evidence_top2_label": "old-a",
            "class_evidence_top2_score": 0.70,
            "class_evidence_top2_margin": 0.30,
            "class_evidence_top2_conformal_pvalue": 0.95,
            "class_evidence_top2_support_count": 2,
            "class_evidence_top2_unknown_risk": 0.10,
            "class_evidence_top2_score_risk": 0.10,
            "class_evidence_top2_radius_risk": 0.10,
            "class_evidence_top2_margin_risk": 0.10,
        }
        second_row = dict(base_row)
        second_row["receiver_id"] = "rx-b"
        rows = [base_row, second_row]
        metadata = {
            "source_receiver_ids": ["src-a"],
            "target_receiver_ids": ["rx-a", "rx-b"],
            "old_tx_ids": ["old-a", "old-b"],
            "seen_new_tx_ids": ["new-a"],
            "unknown_tx_ids": ["unk-a"],
            "target_channel_view": "leo_clear_weak",
        }

        unweighted = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            fusion_policy="cp_set_cvs",
            unknown_risk_threshold=0.98,
            accept_margin_threshold=0.03,
            consensus_score_threshold=0.1,
            conformal_rescue_min_pvalue=0.15,
            class_reliability_policy="conformal_margin_risk",
            label_fusion_policy="vote_margin",
            protocol_metadata=metadata,
            strict_protocol_metadata=True,
        )
        weighted = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            fusion_policy="cp_set_cvs",
            unknown_risk_threshold=0.98,
            accept_margin_threshold=0.03,
            consensus_score_threshold=0.1,
            conformal_rescue_min_pvalue=0.15,
            class_reliability_policy="conformal_margin_risk",
            label_fusion_policy="weighted_vote_margin",
            protocol_metadata=metadata,
            strict_protocol_metadata=True,
        )

        self.assertEqual(unweighted["counts"]["2"]["old_acc"], 0.0)
        self.assertEqual(weighted["counts"]["2"]["old_acc"], 1.0)
        self.assertEqual(weighted["label_fusion_policy"], "weighted_vote_margin")

    def test_weighted_vote_margin_agreement_tracks_selected_label_weight(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "low-weight-high-margin",
                "receiver_id": "rx-a",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": 0.20,
                "known_margin": 0.90,
                "unknown_risk": 0.90,
                "score_risk": 0.90,
                "radius_risk": 0.90,
                "margin_risk": 0.90,
                "class_conformal_pvalue": 0.01,
                "class_conformal_support_count": 2,
            },
            {
                "event_id": "low-weight-high-margin",
                "receiver_id": "rx-b",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-b",
                "known_score": 0.80,
                "known_margin": 0.10,
                "unknown_risk": 0.10,
                "score_risk": 0.10,
                "radius_risk": 0.10,
                "margin_risk": 0.10,
                "class_conformal_pvalue": 0.95,
                "class_conformal_support_count": 2,
            },
        ]
        metadata = {
            "source_receiver_ids": ["src-a"],
            "target_receiver_ids": ["rx-a", "rx-b"],
            "old_tx_ids": ["old-a", "old-b"],
            "seen_new_tx_ids": ["new-a"],
            "unknown_tx_ids": ["unk-a"],
            "target_channel_view": "leo_clear_weak",
        }

        result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            fusion_policy="cp_set_cvs",
            unknown_risk_threshold=0.98,
            accept_margin_threshold=0.03,
            consensus_score_threshold=0.1,
            conformal_rescue_min_pvalue=0.15,
            class_reliability_policy="conformal_margin_risk",
            label_fusion_policy="weighted_vote_margin",
            protocol_metadata=metadata,
            strict_protocol_metadata=True,
        )

        self.assertEqual(result["counts"]["2"]["known_coverage"], 0.0)
        self.assertEqual(result["counts"]["2"]["defer_rate"], 1.0)

    def test_class_reliability_is_monotonic_around_pvalue_floor(self):
        from evaluation.collaborative_open_set_qknn_eval import _class_reliability

        below = _class_reliability(
            policy="conformal_margin_risk",
            pvalue=0.149,
            margin_value=0.10,
            support_count=2,
            unknown_risk_value=0.10,
            accept_margin_threshold=0.10,
            conformal_rescue_min_pvalue=0.15,
        )
        above = _class_reliability(
            policy="conformal_margin_risk",
            pvalue=0.151,
            margin_value=0.10,
            support_count=2,
            unknown_risk_value=0.10,
            accept_margin_threshold=0.10,
            conformal_rescue_min_pvalue=0.15,
        )
        higher_risk = _class_reliability(
            policy="conformal_margin_risk",
            pvalue=0.151,
            margin_value=0.10,
            support_count=2,
            unknown_risk_value=0.90,
            accept_margin_threshold=0.10,
            conformal_rescue_min_pvalue=0.15,
        )

        self.assertLessEqual(below, above)
        self.assertLess(higher_risk, above)

    def test_strict_protocol_rejects_evidence_receiver_outside_target_scope(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "scope",
                "receiver_id": "src-a",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": 0.90,
                "known_margin": 0.20,
                "unknown_risk": 0.10,
            }
        ]

        with self.assertRaisesRegex(ValueError, "target_receiver_ids"):
            evaluate_collaborative_open_set_evidence(
                rows,
                collab_counts="1",
                protocol_metadata={
                    "source_receiver_ids": ["src-a"],
                    "target_receiver_ids": ["rx-a"],
                    "old_tx_ids": ["old-a"],
                    "seen_new_tx_ids": ["new-a"],
                    "unknown_tx_ids": ["unk-a"],
                    "target_channel_view": "leo_clear_weak",
                },
                strict_protocol_metadata=True,
            )

    def test_class_set_gate_defers_unknown_that_looks_old(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "unknown-looks-old",
                "receiver_id": "rx-a",
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "old-a",
                "known_score": 0.95,
                "known_margin": 0.40,
                "unknown_risk": 0.20,
                "score_risk": 0.20,
                "radius_risk": 0.10,
                "margin_risk": 0.10,
            },
            {
                "event_id": "unknown-looks-old",
                "receiver_id": "rx-b",
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "old-a",
                "known_score": 0.96,
                "known_margin": 0.42,
                "unknown_risk": 0.25,
                "score_risk": 0.25,
                "radius_risk": 0.15,
                "margin_risk": 0.15,
            },
        ]
        metadata = {
            "source_receiver_ids": ["src-a"],
            "target_receiver_ids": ["rx-a", "rx-b"],
            "old_tx_ids": ["old-a"],
            "seen_new_tx_ids": ["new-a"],
            "unknown_tx_ids": ["unk-a"],
            "target_channel_view": "leo_clear_weak",
        }

        result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            fusion_policy="scorer_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.6,
            strict_protocol_metadata=True,
            protocol_metadata=metadata,
            class_set_gate_enabled=True,
            old_gate_min_receivers=3,
        )

        k2 = result["counts"]["2"]
        self.assertEqual(k2["unknown_FAR"], 0.0)
        self.assertEqual(k2["unknown_defer"], 1)
        self.assertEqual(result["class_set_gate_enabled"], True)

    def test_class_set_gate_fails_closed_when_density_field_missing(self):
        from evaluation.collaborative_open_set_qknn_eval import _fuse_event

        row = {
            "event_id": "missing-density",
            "receiver_id": "rx-a",
            "predicted_label": "old-a",
            "known_score": 0.95,
            "known_margin": 0.40,
            "unknown_risk": 0.10,
        }

        fused = _fuse_event(
            [row],
            fusion_policy="scorer_cvs",
            old_labels={"old-a"},
            unknown_risk_threshold=0.8,
            unknown_quantile=0.75,
            accept_margin_threshold=0.1,
            consensus_gap_threshold=0.0,
            consensus_score_threshold=0.6,
            scorer_component_vote_threshold=0.5,
            class_set_gate_enabled=True,
            old_gate_min_support_density=0.5,
        )

        self.assertEqual(fused["decision"], "defer")
        self.assertIn("support_density:missing", fused["class_set_gate_reason"])

    def test_class_set_gate_fails_closed_when_radius_z_field_missing(self):
        from evaluation.collaborative_open_set_qknn_eval import _fuse_event

        row = {
            "event_id": "missing-radius-z",
            "receiver_id": "rx-a",
            "predicted_label": "old-a",
            "known_score": 0.95,
            "known_margin": 0.40,
            "unknown_risk": 0.10,
            "support_density": 0.75,
        }

        fused = _fuse_event(
            [row],
            fusion_policy="scorer_cvs",
            old_labels={"old-a"},
            unknown_risk_threshold=0.8,
            unknown_quantile=0.75,
            accept_margin_threshold=0.1,
            consensus_gap_threshold=0.0,
            consensus_score_threshold=0.6,
            scorer_component_vote_threshold=0.5,
            class_set_gate_enabled=True,
            old_gate_max_radius_z=2.0,
        )

        self.assertEqual(fused["decision"], "defer")
        self.assertIn("radius_z:missing", fused["class_set_gate_reason"])

    def test_class_set_gate_allows_seen_new_when_gate_passes(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "seen-new-safe",
                "receiver_id": "rx-a",
                "role": "seen_new",
                "true_label": "new-a",
                "predicted_label": "new-a",
                "known_score": 0.90,
                "known_margin": 0.30,
                "unknown_risk": 0.92,
                "score_risk": 0.92,
                "radius_risk": 0.05,
                "margin_risk": 0.01,
            },
            {
                "event_id": "seen-new-safe",
                "receiver_id": "rx-b",
                "role": "seen_new",
                "true_label": "new-a",
                "predicted_label": "new-a",
                "known_score": 0.88,
                "known_margin": 0.32,
                "unknown_risk": 0.91,
                "score_risk": 0.91,
                "radius_risk": 0.04,
                "margin_risk": 0.01,
            },
        ]
        metadata = {
            "source_receiver_ids": ["src-a"],
            "target_receiver_ids": ["rx-a", "rx-b"],
            "old_tx_ids": ["old-a"],
            "seen_new_tx_ids": ["new-a"],
            "unknown_tx_ids": ["unk-a"],
            "target_channel_view": "leo_clear_weak",
        }

        result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            fusion_policy="scorer_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.6,
            strict_protocol_metadata=True,
            protocol_metadata=metadata,
            seen_new_rescue_enabled=True,
            seen_new_rescue_risk_scale=0.5,
            class_set_gate_enabled=True,
            seen_new_gate_min_receivers=2,
            seen_new_gate_max_effective_unknown_risk=0.6,
            seen_new_gate_max_component_agreement=0.4,
        )

        k2 = result["counts"]["2"]
        self.assertEqual(k2["seen_new_acc"], 1.0)
        self.assertEqual(k2["seen_new_rescue_count"], 1)

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

    def test_scorer_cvs_component_vote_respects_explicit_active_components(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        row = {
            "event_id": "component-isolation",
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
            "oldness_risk": 0.95,
        }

        result = evaluate_collaborative_open_set_evidence(
            [row],
            collab_counts="1",
            fusion_policy="scorer_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.6,
            scorer_component_vote_threshold=0.4,
            scorer_risk_components=["score", "radius", "margin"],
        )

        k1 = result["counts"]["1"]
        self.assertEqual(result["active_risk_components"], ["score", "radius", "margin"])
        self.assertEqual(k1["unknown_reject_rate"], 0.0)
        self.assertEqual(k1["open_set_confusion"], {"unknown->defer": 1})

    def test_scorer_cvs_can_use_virtual_unknown_component(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        row = {
            "event_id": "virtual-risk",
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
            "virtual_unknown_risk": 0.95,
        }

        result = evaluate_collaborative_open_set_evidence(
            [row],
            collab_counts="1",
            fusion_policy="scorer_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.6,
            scorer_component_vote_threshold=0.5,
            scorer_risk_components=["score", "virtual_unknown"],
        )

        k1 = result["counts"]["1"]
        self.assertEqual(result["active_risk_components"], ["score", "virtual_unknown"])
        self.assertEqual(k1["unknown_reject_rate"], 1.0)
        self.assertEqual(k1["open_set_confusion"], {"unknown->unknown_reject": 1})

    def test_scorer_cvs_conformal_rescue_accepts_strong_known(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        row = {
            "event_id": "known-conformal",
            "receiver_id": "rx-a",
            "role": "old",
            "true_label": "old-a",
            "predicted_label": "old-a",
            "known_score": 0.95,
            "known_margin": 0.20,
            "unknown_risk": 0.95,
            "score_risk": 0.95,
            "radius_risk": 0.10,
            "margin_risk": 0.10,
            "class_conformal_pvalue": 0.90,
        }

        result = evaluate_collaborative_open_set_evidence(
            [row],
            collab_counts="1",
            fusion_policy="scorer_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.6,
            scorer_component_vote_threshold=0.5,
            scorer_risk_components=["score", "radius", "margin"],
            conformal_rescue_enabled=True,
            conformal_rescue_min_pvalue=0.5,
            conformal_rescue_risk_scale=0.1,
            protocol_metadata={
                "source_receiver_ids": ["src-a"],
                "target_receiver_ids": ["rx-a"],
                "old_tx_ids": ["old-a"],
                "seen_new_tx_ids": ["new-a"],
                "unknown_tx_ids": ["unk-a"],
                "target_channel_view": "leo_clear_weak",
            },
            strict_protocol_metadata=True,
        )

        k1 = result["counts"]["1"]
        self.assertEqual(k1["old_acc"], 1.0)
        self.assertEqual(k1["open_set_confusion"], {"old->old": 1})
        self.assertTrue(result["conformal_rescue_enabled"])

    def test_scorer_cvs_conformal_rescue_does_not_accept_multichannel_unknown(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        row = {
            "event_id": "unknown-conformal-risk",
            "receiver_id": "rx-a",
            "role": "unknown",
            "true_label": "__unknown__",
            "predicted_label": "old-a",
            "known_score": 0.95,
            "known_margin": 0.20,
            "unknown_risk": 0.95,
            "score_risk": 0.95,
            "radius_risk": 0.95,
            "margin_risk": 0.95,
            "class_conformal_pvalue": 0.90,
        }

        result = evaluate_collaborative_open_set_evidence(
            [row],
            collab_counts="1",
            fusion_policy="scorer_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.6,
            scorer_component_vote_threshold=0.5,
            scorer_risk_components=["score", "radius", "margin"],
            conformal_rescue_enabled=True,
            conformal_rescue_min_pvalue=0.5,
            conformal_rescue_risk_scale=0.1,
            protocol_metadata={
                "source_receiver_ids": ["src-a"],
                "target_receiver_ids": ["rx-a"],
                "old_tx_ids": ["old-a"],
                "seen_new_tx_ids": ["new-a"],
                "unknown_tx_ids": ["unk-a"],
                "target_channel_view": "leo_clear_weak",
            },
            strict_protocol_metadata=True,
        )

        k1 = result["counts"]["1"]
        self.assertEqual(k1["unknown_FAR"], 0.0)
        self.assertEqual(k1["open_set_confusion"], {"unknown->defer": 1})

    def test_cp_set_cvs_requires_class_conformal_gate(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        base = {
            "event_id": "known-cp-set",
            "receiver_id": "rx-a",
            "role": "old",
            "true_label": "old-a",
            "predicted_label": "old-a",
            "known_score": 0.95,
            "known_margin": 0.20,
            "unknown_risk": 0.10,
            "score_risk": 0.10,
            "radius_risk": 0.10,
            "margin_risk": 0.10,
            "class_conformal_support_count": 2,
        }
        protocol_metadata = {
            "source_receiver_ids": ["src-a"],
            "target_receiver_ids": ["rx-a"],
            "old_tx_ids": ["old-a"],
            "seen_new_tx_ids": ["new-a"],
            "unknown_tx_ids": ["unk-a"],
            "target_channel_view": "leo_clear_weak",
        }
        low = dict(base, class_conformal_pvalue=0.10)
        high = dict(base, class_conformal_pvalue=0.90)

        low_result = evaluate_collaborative_open_set_evidence(
            [low],
            collab_counts="1",
            fusion_policy="cp_set_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.6,
            scorer_component_vote_threshold=0.5,
            scorer_risk_components=["score", "radius", "margin"],
            conformal_rescue_min_pvalue=0.5,
            protocol_metadata=protocol_metadata,
            strict_protocol_metadata=True,
        )
        high_result = evaluate_collaborative_open_set_evidence(
            [high],
            collab_counts="1",
            fusion_policy="cp_set_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.6,
            scorer_component_vote_threshold=0.5,
            scorer_risk_components=["score", "radius", "margin"],
            conformal_rescue_min_pvalue=0.5,
            protocol_metadata=protocol_metadata,
            strict_protocol_metadata=True,
        )

        self.assertEqual(low_result["counts"]["1"]["open_set_confusion"], {"old->defer": 1})
        self.assertEqual(high_result["counts"]["1"]["open_set_confusion"], {"old->old": 1})

    def test_cp_set_cvs_uses_top_m_class_evidence(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = []
        for receiver_id in ("rx-a", "rx-b"):
            rows.append({
                "event_id": "topm-cp-set-old",
                "receiver_id": receiver_id,
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-b",
                "known_score": 0.90,
                "known_margin": 0.20,
                "unknown_risk": 0.10,
                "score_risk": 0.10,
                "radius_risk": 0.10,
                "margin_risk": 0.10,
                "class_conformal_pvalue": 0.10,
                "class_conformal_support_count": 2,
                "class_evidence_top_m": 2,
                "class_evidence_top1_label": "old-b",
                "class_evidence_top1_score": 0.90,
                "class_evidence_top1_conformal_pvalue": 0.10,
                "class_evidence_top1_support_count": 2,
                "class_evidence_top2_label": "old-a",
                "class_evidence_top2_score": 0.85,
                "class_evidence_top2_margin": 0.30,
                "class_evidence_top2_conformal_pvalue": 0.90,
                "class_evidence_top2_support_count": 2,
                "class_evidence_top2_unknown_risk": 0.10,
                "class_evidence_top2_score_risk": 0.10,
                "class_evidence_top2_radius_risk": 0.10,
                "class_evidence_top2_margin_risk": 0.10,
                "class_evidence_top2_mahalanobis_risk": 0.10,
                "class_evidence_top2_evt_risk": 0.10,
                "class_evidence_top2_oldness_risk": 0.10,
                "class_evidence_top2_class_radius_z": 0.0,
            })

        result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            fusion_policy="cp_set_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.6,
            scorer_component_vote_threshold=0.5,
            scorer_risk_components=["score", "radius", "margin"],
            conformal_rescue_min_pvalue=0.5,
            label_fusion_policy="vote_margin",
            protocol_metadata={
                "source_receiver_ids": ["src-a"],
                "target_receiver_ids": ["rx-a", "rx-b"],
                "old_tx_ids": ["old-a", "old-b"],
                "seen_new_tx_ids": ["new-a"],
                "unknown_tx_ids": ["unk-a"],
                "target_channel_view": "leo_clear_weak",
            },
            strict_protocol_metadata=True,
        )

        k2 = result["counts"]["2"]
        self.assertEqual(k2["open_set_confusion"], {"old->old": 1})
        self.assertEqual(k2["old_acc"], 1.0)
        self.assertLess(k2["unknown_FAR"], 1.0)

    def test_cp_set_cvs_uses_selected_top_m_label_risk(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = []
        for receiver_id in ("rx-a", "rx-b"):
            rows.append({
                "event_id": "topm-risk-old",
                "receiver_id": receiver_id,
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-b",
                "known_score": 0.90,
                "known_margin": 0.02,
                "unknown_risk": 0.95,
                "score_risk": 0.95,
                "radius_risk": 0.95,
                "margin_risk": 0.95,
                "class_conformal_pvalue": 0.10,
                "class_conformal_support_count": 2,
                "class_evidence_top_m": 2,
                "class_evidence_top1_label": "old-b",
                "class_evidence_top1_score": 0.90,
                "class_evidence_top1_margin": 0.02,
                "class_evidence_top1_conformal_pvalue": 0.10,
                "class_evidence_top1_support_count": 2,
                "class_evidence_top1_unknown_risk": 0.95,
                "class_evidence_top1_score_risk": 0.95,
                "class_evidence_top1_radius_risk": 0.95,
                "class_evidence_top1_margin_risk": 0.95,
                "class_evidence_top2_label": "old-a",
                "class_evidence_top2_score": 0.85,
                "class_evidence_top2_margin": 0.30,
                "class_evidence_top2_conformal_pvalue": 0.90,
                "class_evidence_top2_support_count": 2,
                "class_evidence_top2_unknown_risk": 0.10,
                "class_evidence_top2_score_risk": 0.10,
                "class_evidence_top2_radius_risk": 0.10,
                "class_evidence_top2_margin_risk": 0.10,
                "class_evidence_top2_mahalanobis_risk": 0.10,
                "class_evidence_top2_evt_risk": 0.10,
                "class_evidence_top2_oldness_risk": 0.10,
                "class_evidence_top2_class_radius_z": 0.0,
            })

        result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            fusion_policy="cp_set_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.6,
            scorer_component_vote_threshold=0.5,
            scorer_risk_components=["score", "radius", "margin"],
            conformal_rescue_min_pvalue=0.5,
            label_fusion_policy="vote_margin",
            protocol_metadata={
                "source_receiver_ids": ["src-a"],
                "target_receiver_ids": ["rx-a", "rx-b"],
                "old_tx_ids": ["old-a", "old-b"],
                "seen_new_tx_ids": ["new-a"],
                "unknown_tx_ids": ["unk-a"],
                "target_channel_view": "leo_clear_weak",
            },
            strict_protocol_metadata=True,
        )

        k2 = result["counts"]["2"]
        self.assertEqual(k2["open_set_confusion"], {"old->old": 1})
        self.assertEqual(k2["old_acc"], 1.0)

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
