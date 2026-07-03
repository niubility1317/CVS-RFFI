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
            include_event_results=True,
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
        self.assertIn("mean_receiver_pair_label_disagreement", result["counts"]["2"])
        self.assertIn("mean_receiver_pair_unknown_risk_range", result["counts"]["2"])
        self.assertEqual(len(result["counts"]["2"]["event_results"]), 3)
        self.assertIn("receiver_pair_label_disagreement", result["counts"]["2"]["event_results"][0])
        self.assertEqual(result["counts"]["2"]["event_results"][0]["event_id"], "old-1")
        self.assertIn("rx-a", result["counts"]["2"]["event_results"][0]["selected_receiver_ids"])

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

    def test_support_quality_prior_selects_support_calibrated_receiver(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "quality-old",
                "receiver_id": "rx-a",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-b",
                "known_score": 0.95,
                "known_margin": 0.20,
                "unknown_risk": 0.10,
                "support_density": 0.05,
                "class_conformal_pvalue": 0.05,
                "receiver_class_reliability": 0.05,
            },
            {
                "event_id": "quality-old",
                "receiver_id": "rx-b",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": 0.70,
                "known_margin": 0.20,
                "unknown_risk": 0.10,
                "support_density": 0.95,
                "class_conformal_pvalue": 0.95,
                "receiver_class_reliability": 0.95,
            },
        ]

        fixed = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="1",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            receiver_selection_policy="fixed_receiver_order",
        )
        support_quality = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="1",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            receiver_selection_policy="support_quality_prior",
        )

        self.assertEqual(fixed["counts"]["1"]["old_acc"], 0.0)
        self.assertEqual(support_quality["counts"]["1"]["old_acc"], 1.0)
        self.assertEqual(support_quality["receiver_selection_policy"], "support_quality_prior")

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
            "radius_risk": 0.10,
            "margin_risk": 0.10,
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

    def test_seen_new_rescue_guard_does_not_use_true_role(self):
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
        role_free = _fuse_event([{k: v for k, v in base_row.items() if k not in {"role", "true_label"}}], **common)

        self.assertFalse(as_seen_new["seen_new_rescue_applied"])
        self.assertFalse(as_unknown["seen_new_rescue_applied"])
        self.assertFalse(role_free["seen_new_rescue_applied"])
        self.assertEqual(as_seen_new["decision"], as_unknown["decision"])
        self.assertEqual(as_seen_new["decision"], role_free["decision"])
        self.assertEqual(as_seen_new["output_label"], as_unknown["output_label"])
        self.assertEqual(as_seen_new["output_label"], role_free["output_label"])

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

    def test_receiver_class_reliability_weights_receiver_label_pairs(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        base_row = {
            "event_id": "receiver-class-rel",
            "receiver_id": "rx-a",
            "role": "old",
            "true_label": "old-a",
            "predicted_label": "old-b",
            "known_score": 1.00,
            "known_margin": 0.30,
            "unknown_risk": 0.10,
            "score_risk": 0.10,
            "radius_risk": 0.10,
            "margin_risk": 0.10,
            "class_conformal_pvalue": 0.80,
            "class_conformal_support_count": 2,
            "receiver_class_reliability": 0.05,
            "class_evidence_top_m": 2,
            "class_evidence_top1_label": "old-b",
            "class_evidence_top1_score": 1.00,
            "class_evidence_top1_margin": 0.30,
            "class_evidence_top1_conformal_pvalue": 0.80,
            "class_evidence_top1_support_count": 2,
            "class_evidence_top1_unknown_risk": 0.10,
            "class_evidence_top1_score_risk": 0.10,
            "class_evidence_top1_radius_risk": 0.10,
            "class_evidence_top1_margin_risk": 0.10,
            "class_evidence_top1_receiver_class_reliability": 0.05,
            "class_evidence_top2_label": "old-a",
            "class_evidence_top2_score": 0.80,
            "class_evidence_top2_margin": 0.50,
            "class_evidence_top2_conformal_pvalue": 0.80,
            "class_evidence_top2_support_count": 2,
            "class_evidence_top2_unknown_risk": 0.10,
            "class_evidence_top2_score_risk": 0.10,
            "class_evidence_top2_radius_risk": 0.10,
            "class_evidence_top2_margin_risk": 0.10,
            "class_evidence_top2_receiver_class_reliability": 1.00,
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
            unknown_risk_threshold=0.80,
            accept_margin_threshold=0.05,
            consensus_score_threshold=0.1,
            conformal_rescue_min_pvalue=0.20,
            protocol_metadata=metadata,
            strict_protocol_metadata=True,
        )
        reliable_result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            fusion_policy="cp_set_cvs",
            unknown_risk_threshold=0.80,
            accept_margin_threshold=0.05,
            consensus_score_threshold=0.1,
            conformal_rescue_min_pvalue=0.20,
            receiver_class_reliability_policy="support_calibrated",
            protocol_metadata=metadata,
            strict_protocol_metadata=True,
        )

        self.assertEqual(default_result["counts"]["2"]["old_acc"], 0.0)
        self.assertEqual(reliable_result["counts"]["2"]["old_acc"], 1.0)
        self.assertEqual(reliable_result["receiver_class_reliability_policy"], "support_calibrated")
        self.assertGreater(reliable_result["counts"]["2"]["mean_label_receiver_class_reliability"], 0.9)

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

    def test_candidate_set_cvs_recovers_top_m_label_with_unknown_safety_valve(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = []
        for receiver_id in ("rx-a", "rx-b", "rx-c"):
            rows.append({
                "event_id": "candidate-set-old",
                "receiver_id": receiver_id,
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-b",
                "known_score": 0.90,
                "known_margin": 0.10,
                "unknown_risk": 0.30,
                "score_risk": 0.30,
                "radius_risk": 0.30,
                "margin_risk": 0.30,
                "class_conformal_pvalue": 0.10,
                "class_conformal_support_count": 2,
                "class_evidence_top_m": 2,
                "class_evidence_top1_label": "old-b",
                "class_evidence_top1_score": 0.90,
                "class_evidence_top1_margin": 0.10,
                "class_evidence_top1_conformal_pvalue": 0.10,
                "class_evidence_top1_support_count": 2,
                "class_evidence_top1_unknown_risk": 0.30,
                "class_evidence_top2_label": "old-a",
                "class_evidence_top2_score": 0.88,
                "class_evidence_top2_margin": 0.25,
                "class_evidence_top2_conformal_pvalue": 0.90,
                "class_evidence_top2_support_count": 2,
                "class_evidence_top2_unknown_risk": 0.20,
                "class_evidence_top2_score_risk": 0.20,
                "class_evidence_top2_radius_risk": 0.20,
                "class_evidence_top2_margin_risk": 0.20,
                "class_evidence_top2_mahalanobis_risk": 0.20,
                "class_evidence_top2_evt_risk": 0.20,
                "class_evidence_top2_oldness_risk": 0.20,
                "class_evidence_top2_class_radius_z": 0.0,
            })
        for receiver_id in ("rx-a", "rx-b", "rx-c"):
            rows.append({
                "event_id": "candidate-set-unknown",
                "receiver_id": receiver_id,
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "old-a",
                "known_score": 0.86,
                "known_margin": 0.20,
                "unknown_risk": 0.92,
                "score_risk": 0.92,
                "radius_risk": 0.92,
                "margin_risk": 0.92,
                "class_conformal_pvalue": 0.90,
                "class_conformal_support_count": 2,
                "class_evidence_top_m": 1,
                "class_evidence_top1_label": "old-a",
                "class_evidence_top1_score": 0.86,
                "class_evidence_top1_margin": 0.20,
                "class_evidence_top1_conformal_pvalue": 0.90,
                "class_evidence_top1_support_count": 2,
                "class_evidence_top1_unknown_risk": 0.92,
                "class_evidence_top1_score_risk": 0.92,
                "class_evidence_top1_radius_risk": 0.92,
                "class_evidence_top1_margin_risk": 0.92,
            })

        result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="3",
            fusion_policy="candidate_set_cvs",
            unknown_risk_threshold=0.8,
            accept_margin_threshold=0.1,
            consensus_score_threshold=0.0,
            scorer_component_vote_threshold=0.5,
            scorer_risk_components=["score", "radius", "margin"],
            candidate_set_min_receivers=2,
            candidate_set_min_top1_receivers=0,
            candidate_set_min_conformal_pvalue=0.5,
            candidate_set_max_label_unknown_risk=0.8,
            candidate_set_max_event_unknown_risk=0.8,
            candidate_set_unknown_reject_risk=0.8,
            protocol_metadata={
                "source_receiver_ids": ["src-a"],
                "target_receiver_ids": ["rx-a", "rx-b", "rx-c"],
                "old_tx_ids": ["old-a", "old-b"],
                "seen_new_tx_ids": ["new-a"],
                "unknown_tx_ids": ["unk-a"],
                "target_channel_view": "leo_clear_weak",
            },
            strict_protocol_metadata=True,
        )

        k3 = result["counts"]["3"]
        self.assertEqual(result["fusion_policy"], "candidate_set_cvs")
        self.assertEqual(k3["old_acc"], 1.0)
        self.assertEqual(k3["unknown_FAR"], 0.0)
        self.assertEqual(k3["unknown_reject_rate"], 1.0)
        self.assertEqual(k3["open_set_confusion"], {"old->old": 1, "unknown->unknown_reject": 1})

    def test_candidate_set_cvs_pairguard_blocks_unreliable_accepts(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        def make_rows(*, omit_shell=False, reliability=1.0):
            rows = []
            for receiver_id, predicted_label, risk in (
                ("rx-a", "old-a", 0.10),
                ("rx-b", "old-b", 0.80),
            ):
                row = {
                    "event_id": "pairguard-old",
                    "receiver_id": receiver_id,
                    "role": "old",
                    "true_label": "old-a",
                    "predicted_label": predicted_label,
                    "known_score": 0.90,
                    "known_margin": 0.20,
                    "unknown_risk": risk,
                    "score_risk": risk,
                    "radius_risk": risk,
                    "margin_risk": risk,
                    "class_conformal_pvalue": 0.90,
                    "class_conformal_support_count": 2,
                    "receiver_class_reliability": reliability,
                    "class_evidence_top_m": 1,
                    "class_evidence_top1_label": "old-a",
                    "class_evidence_top1_score": 0.90,
                    "class_evidence_top1_margin": 0.20,
                    "class_evidence_top1_conformal_pvalue": 0.90,
                    "class_evidence_top1_support_count": 2,
                    "class_evidence_top1_unknown_risk": 0.10,
                    "class_evidence_top1_score_risk": 0.10,
                    "class_evidence_top1_radius_risk": 0.10,
                    "class_evidence_top1_margin_risk": 0.10,
                    "class_evidence_top1_receiver_class_reliability": reliability,
                }
                if not omit_shell:
                    row["class_shell_risk"] = 0.05
                    row["class_evidence_top1_class_shell_risk"] = 0.05
                rows.append(row)
            return rows

        def evaluate(rows, **kwargs):
            pairguard_mode = kwargs.pop("candidate_set_pairguard_mode", "boundary_veto")
            min_event_unknown_risk = kwargs.pop("candidate_set_pairguard_min_event_unknown_risk", 0.90)
            return evaluate_collaborative_open_set_evidence(
                rows,
                collab_counts="2",
                fusion_policy="candidate_set_cvs",
                unknown_risk_threshold=0.8,
                accept_margin_threshold=0.1,
                consensus_score_threshold=0.0,
                scorer_component_vote_threshold=1.0,
                scorer_risk_components=["score", "radius", "margin"],
                candidate_set_min_receivers=2,
                candidate_set_min_conformal_pvalue=0.5,
                candidate_set_max_label_unknown_risk=0.8,
                candidate_set_max_event_unknown_risk=1.0,
                candidate_set_unknown_reject_risk=1.1,
                protocol_metadata={
                    "source_receiver_ids": ["src-a"],
                    "target_receiver_ids": ["rx-a", "rx-b"],
                    "old_tx_ids": ["old-a", "old-b"],
                    "seen_new_tx_ids": ["new-a"],
                    "unknown_tx_ids": ["unk-a"],
                    "target_channel_view": "leo_clear_weak",
                },
                strict_protocol_metadata=True,
                include_event_results=True,
                **kwargs,
            )

        base_event = evaluate(make_rows())["counts"]["2"]["event_results"][0]
        self.assertTrue(base_event["candidate_set_accept"])
        self.assertEqual(base_event["decision"], "accept")

        disagreement_event = evaluate(
            make_rows(),
            candidate_set_max_receiver_pair_label_disagreement=0.25,
        )["counts"]["2"]["event_results"][0]
        self.assertFalse(disagreement_event["candidate_set_accept"])
        self.assertEqual(disagreement_event["decision"], "defer")
        self.assertEqual(disagreement_event["candidate_set_max_receiver_pair_label_disagreement"], 0.25)

        risk_range_event = evaluate(
            make_rows(),
            candidate_set_max_receiver_pair_unknown_risk_range=0.20,
        )["counts"]["2"]["event_results"][0]
        self.assertFalse(risk_range_event["candidate_set_accept"])
        self.assertEqual(risk_range_event["decision"], "defer")
        self.assertEqual(risk_range_event["candidate_set_max_receiver_pair_unknown_risk_range"], 0.20)

        reliability_event = evaluate(
            make_rows(reliability=0.40),
            receiver_class_reliability_policy="support_calibrated",
            candidate_set_min_label_receiver_class_reliability=0.80,
        )["counts"]["2"]["event_results"][0]
        self.assertFalse(reliability_event["candidate_set_accept"])
        self.assertEqual(reliability_event["decision"], "defer")
        self.assertEqual(reliability_event["candidate_set_min_label_receiver_class_reliability"], 0.80)

        shell_event = evaluate(
            make_rows(omit_shell=True),
            candidate_set_require_label_shell_observed=True,
        )["counts"]["2"]["event_results"][0]
        self.assertFalse(shell_event["candidate_set_accept"])
        self.assertEqual(shell_event["decision"], "defer")
        self.assertTrue(shell_event["candidate_set_require_label_shell_observed"])

        with self.assertRaisesRegex(ValueError, "candidate_set_max_receiver_pair_label_disagreement"):
            evaluate(make_rows(), candidate_set_max_receiver_pair_label_disagreement=1.1)
        with self.assertRaisesRegex(ValueError, "candidate_set_max_receiver_pair_unknown_risk_range"):
            evaluate(make_rows(), candidate_set_max_receiver_pair_unknown_risk_range=-0.1)
        with self.assertRaisesRegex(ValueError, "candidate_set_min_label_receiver_class_reliability"):
            evaluate(make_rows(), candidate_set_min_label_receiver_class_reliability=1.1)

    def test_candidate_set_cvs_boundary_pairguard_only_vetoes_risky_events(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        def make_rows(event_risks):
            rows = []
            for receiver_id, predicted_label, risk in (
                ("rx-a", "old-a", event_risks[0]),
                ("rx-b", "old-b", event_risks[1]),
            ):
                rows.append({
                    "event_id": "boundary-pairguard-old",
                    "receiver_id": receiver_id,
                    "role": "old",
                    "true_label": "old-a",
                    "predicted_label": predicted_label,
                    "known_score": 0.90,
                    "known_margin": 0.20,
                    "unknown_risk": risk,
                    "score_risk": risk,
                    "radius_risk": risk,
                    "margin_risk": risk,
                    "class_shell_risk": 0.05,
                    "class_conformal_pvalue": 0.90,
                    "class_conformal_support_count": 2,
                    "receiver_class_reliability": 0.95,
                    "class_evidence_top_m": 1,
                    "class_evidence_top1_label": "old-a",
                    "class_evidence_top1_score": 0.90,
                    "class_evidence_top1_margin": 0.20,
                    "class_evidence_top1_conformal_pvalue": 0.90,
                    "class_evidence_top1_support_count": 2,
                    "class_evidence_top1_unknown_risk": 0.10,
                    "class_evidence_top1_score_risk": 0.10,
                    "class_evidence_top1_radius_risk": 0.10,
                    "class_evidence_top1_margin_risk": 0.10,
                    "class_evidence_top1_class_shell_risk": 0.05,
                    "class_evidence_top1_receiver_class_reliability": 0.95,
                })
            return rows

        def evaluate(rows, **kwargs):
            pairguard_mode = kwargs.pop("candidate_set_pairguard_mode", "boundary_veto")
            min_event_unknown_risk = kwargs.pop("candidate_set_pairguard_min_event_unknown_risk", 0.90)
            return evaluate_collaborative_open_set_evidence(
                rows,
                collab_counts="2",
                fusion_policy="candidate_set_cvs",
                unknown_risk_threshold=0.8,
                accept_margin_threshold=0.1,
                consensus_score_threshold=0.0,
                scorer_component_vote_threshold=1.0,
                scorer_risk_components=["score", "radius", "margin"],
                candidate_set_min_receivers=2,
                candidate_set_min_conformal_pvalue=0.5,
                candidate_set_max_label_unknown_risk=0.8,
                candidate_set_max_event_unknown_risk=1.0,
                candidate_set_unknown_reject_risk=1.1,
                candidate_set_max_receiver_pair_label_disagreement=0.25,
                candidate_set_max_receiver_pair_unknown_risk_range=0.20,
                candidate_set_pairguard_mode=pairguard_mode,
                candidate_set_pairguard_min_event_unknown_risk=min_event_unknown_risk,
                candidate_set_pairguard_min_label_unknown_risk=0.90,
                candidate_set_pairguard_min_shell_risk=0.90,
                protocol_metadata={
                    "source_receiver_ids": ["src-a"],
                    "target_receiver_ids": ["rx-a", "rx-b"],
                    "old_tx_ids": ["old-a", "old-b"],
                    "seen_new_tx_ids": ["new-a"],
                    "unknown_tx_ids": ["unk-a"],
                    "target_channel_view": "leo_clear_weak",
                },
                strict_protocol_metadata=True,
                include_event_results=True,
                **kwargs,
            )

        low_risk_event = evaluate(make_rows((0.10, 0.80)))["counts"]["2"]["event_results"][0]
        self.assertTrue(low_risk_event["candidate_set_pairguard_disagreement_failed"])
        self.assertTrue(low_risk_event["candidate_set_pairguard_risk_range_failed"])
        self.assertFalse(low_risk_event["candidate_set_pairguard_boundary_trigger"])
        self.assertFalse(low_risk_event["candidate_set_pairguard_veto"])
        self.assertTrue(low_risk_event["candidate_set_accept"])
        self.assertEqual(low_risk_event["decision"], "accept")

        high_risk_event = evaluate(make_rows((0.10, 0.95)))["counts"]["2"]["event_results"][0]
        self.assertTrue(high_risk_event["candidate_set_pairguard_boundary_trigger"])
        self.assertTrue(high_risk_event["candidate_set_pairguard_veto"])
        self.assertFalse(high_risk_event["candidate_set_accept"])
        self.assertEqual(high_risk_event["decision"], "defer")

        with self.assertRaisesRegex(ValueError, "candidate_set_pairguard_mode"):
            evaluate(make_rows((0.10, 0.80)), candidate_set_pairguard_mode="bad_mode")
        with self.assertRaisesRegex(ValueError, "candidate_set_pairguard_min_event_unknown_risk"):
            evaluate(make_rows((0.10, 0.80)), candidate_set_pairguard_min_event_unknown_risk=1.1)

    def test_dual_route_cvs_uses_support_quality_rescue_when_safe(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        def row(receiver_id, label, unknown_risk, reliability, pvalue, prior, shell_risk=0.05):
            return {
                "event_id": "dual-route-old",
                "receiver_id": receiver_id,
                "role": "old",
                "true_label": "old-a",
                "predicted_label": label,
                "known_score": 0.92 if label == "old-a" else 0.50,
                "known_margin": 0.35 if label == "old-a" else 0.04,
                "unknown_risk": unknown_risk,
                "score_risk": unknown_risk,
                "radius_risk": unknown_risk,
                "margin_risk": unknown_risk,
                "class_shell_risk": shell_risk,
                "class_conformal_pvalue": pvalue,
                "class_conformal_support_count": 2,
                "receiver_class_reliability": reliability,
                "support_density": reliability,
                "receiver_deployment_prior": prior,
                "bytes": 40.0,
                "latency_ms": 0.1,
            }

        rows = [
            row("rx-a", "old-b", 0.50, 0.20, 0.10, 0.10),
            row("rx-b", "old-b", 0.55, 0.25, 0.10, 0.20),
            row("rx-c", "old-a", 0.10, 0.95, 0.95, 0.95),
            row("rx-d", "old-a", 0.12, 0.90, 0.90, 0.90),
        ]
        result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            fusion_policy="candidate_set_cvs",
            collaboration_policy="dual_route_cvs",
            scorer_risk_components=["score", "radius", "margin"],
            candidate_set_min_receivers=2,
            candidate_set_min_conformal_pvalue=0.75,
            candidate_set_max_label_unknown_risk=0.80,
            candidate_set_max_event_unknown_risk=1.0,
            candidate_set_unknown_reject_risk=0.80,
            dual_route_rescue_min_pvalue=0.75,
            dual_route_rescue_min_receiver_class_reliability=0.75,
            receiver_class_reliability_policy="support_calibrated",
            include_event_results=True,
            protocol_metadata={
                "source_receiver_ids": ["src-a"],
                "target_receiver_ids": ["rx-a", "rx-b", "rx-c", "rx-d"],
                "old_tx_ids": ["old-a", "old-b"],
                "seen_new_tx_ids": ["new-a"],
                "unknown_tx_ids": ["unk-a"],
                "target_channel_view": "leo_clear_weak",
            },
            strict_protocol_metadata=True,
        )

        event = result["counts"]["2"]["event_results"][0]
        self.assertEqual(result["collaboration_policy"], "dual_route_cvs")
        self.assertEqual(result["dual_route_rescue_min_pvalue"], 0.75)
        self.assertEqual(result["counts"]["2"]["old_acc"], 1.0)
        self.assertEqual(result["counts"]["2"]["dual_route_rescue_count"], 1)
        self.assertEqual(event["dual_route_selected_route"], "rescue")
        self.assertEqual(event["selected_receiver_ids"], "rx-c,rx-d")
        self.assertEqual(event["dual_route_safety_receiver_order"], "rx-a,rx-b")
        self.assertEqual(event["dual_route_rescue_receiver_order"], "rx-c,rx-d")
        self.assertEqual(event["dual_route_rescue_selection_policy"], "deployment_prior_quality")

    def test_dual_route_cvs_blocks_high_unknown_risk_rescue(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        def row(receiver_id, label, unknown_risk, reliability, pvalue, prior):
            return {
                "event_id": "dual-route-unknown",
                "receiver_id": receiver_id,
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": label,
                "known_score": 0.92,
                "known_margin": 0.35,
                "unknown_risk": unknown_risk,
                "score_risk": unknown_risk,
                "radius_risk": unknown_risk,
                "margin_risk": unknown_risk,
                "class_shell_risk": 0.05,
                "class_conformal_pvalue": pvalue,
                "class_conformal_support_count": 2,
                "receiver_class_reliability": reliability,
                "support_density": reliability,
                "receiver_deployment_prior": prior,
                "bytes": 40.0,
                "latency_ms": 0.1,
            }

        rows = [
            row("rx-a", "old-a", 0.92, 0.20, 0.10, 0.10),
            row("rx-b", "old-a", 0.90, 0.25, 0.10, 0.20),
            row("rx-c", "old-a", 0.70, 0.95, 0.95, 0.95),
            row("rx-d", "old-a", 0.70, 0.90, 0.90, 0.90),
        ]
        result = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="2",
            fusion_policy="candidate_set_cvs",
            collaboration_policy="dual_route_cvs",
            scorer_risk_components=["score", "radius", "margin"],
            candidate_set_min_receivers=2,
            candidate_set_min_conformal_pvalue=0.75,
            candidate_set_max_label_unknown_risk=1.0,
            candidate_set_max_event_unknown_risk=1.0,
            candidate_set_unknown_reject_risk=0.80,
            dual_route_rescue_min_pvalue=0.75,
            dual_route_rescue_min_receiver_class_reliability=0.75,
            dual_route_rescue_max_label_unknown_risk=0.60,
            receiver_class_reliability_policy="support_calibrated",
            include_event_results=True,
            protocol_metadata={
                "source_receiver_ids": ["src-a"],
                "target_receiver_ids": ["rx-a", "rx-b", "rx-c", "rx-d"],
                "old_tx_ids": ["old-a", "old-b"],
                "seen_new_tx_ids": ["new-a"],
                "unknown_tx_ids": ["unk-a"],
                "target_channel_view": "leo_clear_weak",
            },
            strict_protocol_metadata=True,
        )

        event = result["counts"]["2"]["event_results"][0]
        self.assertEqual(result["counts"]["2"]["unknown_FAR"], 0.0)
        self.assertEqual(result["counts"]["2"]["unknown_reject_rate"], 1.0)
        self.assertEqual(result["counts"]["2"]["dual_route_rescue_count"], 0)
        self.assertEqual(event["dual_route_selected_route"], "safety")
        self.assertFalse(event["dual_route_rescue_ok"])
        self.assertEqual(event["selected_receiver_ids"], "rx-a,rx-b")

    def test_available_up_to_k_keeps_partial_class_groups(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        def row(event_id, receiver_id, label):
            return {
                "event_id": event_id,
                "receiver_id": receiver_id,
                "role": "old",
                "true_label": label,
                "predicted_label": label,
                "known_score": 0.90,
                "known_margin": 0.50,
                "unknown_risk": 0.05,
                "score_risk": 0.05,
                "radius_risk": 0.05,
                "margin_risk": 0.05,
                "bytes": 40.0,
                "latency_ms": 0.1,
            }

        rows = [row("old-a-event", rx, "old-a") for rx in ("rx-a", "rx-b", "rx-c")]
        rows.extend(row("old-b-event", rx, "old-b") for rx in ("rx-a", "rx-b", "rx-c", "rx-d"))
        protocol = {
            "source_receiver_ids": ["src-a"],
            "target_receiver_ids": ["rx-a", "rx-b", "rx-c", "rx-d"],
            "old_tx_ids": ["old-a", "old-b"],
            "seen_new_tx_ids": ["new-a"],
            "unknown_tx_ids": ["unk-a"],
            "target_channel_view": "leo_clear_weak",
        }

        exact = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="4",
            fusion_policy="risk_margin",
            protocol_metadata=protocol,
            strict_protocol_metadata=True,
        )
        budgeted = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="4",
            fusion_policy="risk_margin",
            collab_group_policy="available_up_to_k",
            partial_collab_min_receivers=2,
            protocol_metadata=protocol,
            strict_protocol_metadata=True,
        )

        self.assertEqual(exact["counts"]["4"]["missing_old_classes"], ["old-a"])
        self.assertEqual(budgeted["counts"]["4"]["missing_old_classes"], [])
        self.assertEqual(budgeted["counts"]["4"]["old_acc"], 1.0)
        self.assertEqual(budgeted["counts"]["4"]["participating_receivers_avg"], 3.5)
        self.assertEqual(budgeted["counts"]["4"]["receiver_budget"], 4)
        self.assertEqual(budgeted["counts"]["4"]["min_required_receivers"], 2)
        self.assertEqual(budgeted["counts"]["4"]["actual_receiver_count_histogram"], {"3": 1, "4": 1})
        self.assertEqual(budgeted["counts"]["4"]["partial_group_count"], 1)
        self.assertEqual(budgeted["exact_max_requested_group_count"], 1)
        self.assertEqual(budgeted["policy_eligible_group_count_at_max_budget"], 2)
        self.assertEqual(budgeted["collab_group_policy"], "available_up_to_k")

    def test_available_up_to_k_rejects_invalid_partial_min_receivers(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = [
            {
                "event_id": "old-a-event",
                "receiver_id": "rx-a",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": 0.90,
                "known_margin": 0.50,
                "unknown_risk": 0.05,
            }
        ]
        protocol = {
            "source_receiver_ids": ["src-a"],
            "target_receiver_ids": ["rx-a"],
            "old_tx_ids": ["old-a"],
            "seen_new_tx_ids": ["new-a"],
            "unknown_tx_ids": ["unk-a"],
            "target_channel_view": "leo_clear_weak",
        }

        with self.assertRaisesRegex(ValueError, "partial_collab_min_receivers must be >= 1"):
            evaluate_collaborative_open_set_evidence(
                rows,
                collab_counts="1",
                fusion_policy="risk_margin",
                collab_group_policy="available_up_to_k",
                partial_collab_min_receivers=0,
                protocol_metadata=protocol,
                strict_protocol_metadata=True,
            )

    def test_candidate_set_cvs_vetoes_high_label_component_agreement(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = []
        for receiver_id in ("rx-a", "rx-b", "rx-c"):
            rows.append({
                "event_id": "component-veto-old",
                "receiver_id": receiver_id,
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": 0.90,
                "known_margin": 0.20,
                "unknown_risk": 0.30,
                "score_risk": 0.10,
                "radius_risk": 0.10,
                "margin_risk": 0.10,
                "class_conformal_pvalue": 0.90,
                "class_conformal_support_count": 2,
                "class_evidence_top_m": 1,
                "class_evidence_top1_label": "old-a",
                "class_evidence_top1_score": 0.90,
                "class_evidence_top1_margin": 0.20,
                "class_evidence_top1_conformal_pvalue": 0.90,
                "class_evidence_top1_support_count": 2,
                "class_evidence_top1_unknown_risk": 0.30,
                "class_evidence_top1_score_risk": 0.10,
                "class_evidence_top1_radius_risk": 0.10,
                "class_evidence_top1_margin_risk": 0.10,
            })
            rows.append({
                "event_id": "component-veto-unknown",
                "receiver_id": receiver_id,
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "old-a",
                "known_score": 0.88,
                "known_margin": 0.20,
                "unknown_risk": 0.60,
                "score_risk": 0.90,
                "radius_risk": 0.90,
                "margin_risk": 0.10,
                "class_conformal_pvalue": 0.90,
                "class_conformal_support_count": 2,
                "class_evidence_top_m": 1,
                "class_evidence_top1_label": "old-a",
                "class_evidence_top1_score": 0.88,
                "class_evidence_top1_margin": 0.20,
                "class_evidence_top1_conformal_pvalue": 0.90,
                "class_evidence_top1_support_count": 2,
                "class_evidence_top1_unknown_risk": 0.60,
                "class_evidence_top1_score_risk": 0.90,
                "class_evidence_top1_radius_risk": 0.90,
                "class_evidence_top1_margin_risk": 0.10,
            })

        metadata = {
            "source_receiver_ids": ["src-a"],
            "target_receiver_ids": ["rx-a", "rx-b", "rx-c"],
            "old_tx_ids": ["old-a"],
            "seen_new_tx_ids": ["new-a"],
            "unknown_tx_ids": ["unk-a"],
            "target_channel_view": "leo_clear_weak",
        }
        unsafe = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="3",
            fusion_policy="candidate_set_cvs",
            unknown_risk_threshold=0.8,
            scorer_component_vote_threshold=0.5,
            scorer_risk_components=["score", "radius", "margin"],
            candidate_set_min_receivers=2,
            candidate_set_max_event_unknown_risk=0.9,
            protocol_metadata=metadata,
            strict_protocol_metadata=True,
        )
        guarded = evaluate_collaborative_open_set_evidence(
            rows,
            collab_counts="3",
            fusion_policy="candidate_set_cvs",
            unknown_risk_threshold=0.8,
            scorer_component_vote_threshold=0.5,
            scorer_risk_components=["score", "radius", "margin"],
            candidate_set_min_receivers=2,
            candidate_set_max_event_unknown_risk=0.9,
            candidate_set_max_label_risk_component_agreement=0.5,
            protocol_metadata=metadata,
            strict_protocol_metadata=True,
        )

        self.assertEqual(unsafe["counts"]["3"]["unknown_FAR"], 1.0)
        self.assertEqual(guarded["counts"]["3"]["unknown_FAR"], 0.0)
        self.assertEqual(guarded["counts"]["3"]["old_acc"], 1.0)
        self.assertEqual(guarded["candidate_set_max_label_risk_component_agreement"], 0.5)

    def test_candidate_set_cvs_vetoes_event_high_unknown_risk_fraction(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = []
        for receiver_id, risk in (("rx-a", 0.999), ("rx-b", 0.999), ("rx-c", 0.10)):
            rows.append({
                "event_id": "unknown-looks-old",
                "receiver_id": receiver_id,
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "old-a",
                "known_score": 0.90,
                "known_margin": 0.30,
                "unknown_risk": risk,
                "score_risk": 0.10,
                "radius_risk": 0.10,
                "margin_risk": 0.10,
                "class_conformal_pvalue": 0.90,
                "class_conformal_support_count": 2,
                "class_evidence_top_m": 1,
                "class_evidence_top1_label": "old-a",
                "class_evidence_top1_score": 0.90,
                "class_evidence_top1_margin": 0.30,
                "class_evidence_top1_conformal_pvalue": 0.90,
                "class_evidence_top1_support_count": 2,
                "class_evidence_top1_unknown_risk": risk,
                "class_evidence_top1_score_risk": 0.10,
                "class_evidence_top1_radius_risk": 0.10,
                "class_evidence_top1_margin_risk": 0.10,
                "class_evidence_top1_mahalanobis_risk": 0.10,
                "class_evidence_top1_evt_risk": 0.10,
                "class_evidence_top1_oldness_risk": 0.10,
                "bytes": 40.0,
                "latency_ms": 0.1,
            })

        metadata = {
            "source_receiver_ids": ["src-a"],
            "target_receiver_ids": ["rx-a", "rx-b", "rx-c"],
            "old_tx_ids": ["old-a"],
            "seen_new_tx_ids": ["new-a"],
            "unknown_tx_ids": ["unk-a"],
            "target_channel_view": "leo_clear_weak",
        }
        common = dict(
            collab_counts="3",
            fusion_policy="candidate_set_cvs",
            scorer_risk_components=["score", "radius", "margin"],
            candidate_set_min_receivers=2,
            candidate_set_min_conformal_pvalue=0.5,
            candidate_set_max_label_unknown_risk=1.0,
            candidate_set_max_event_unknown_risk=1.0,
            candidate_set_unknown_reject_risk=1.1,
            protocol_metadata=metadata,
            strict_protocol_metadata=True,
        )
        unguarded = evaluate_collaborative_open_set_evidence(rows, **common)
        guarded = evaluate_collaborative_open_set_evidence(
            rows,
            candidate_set_event_high_unknown_risk_veto=0.99,
            candidate_set_max_label_high_unknown_risk_fraction=0.5,
            candidate_set_high_unknown_risk_threshold=0.8,
            **common,
        )

        self.assertEqual(unguarded["counts"]["3"]["unknown_FAR"], 1.0)
        self.assertEqual(guarded["counts"]["3"]["unknown_FAR"], 0.0)
        self.assertEqual(guarded["counts"]["3"]["unknown_reject_rate"], 1.0)
        self.assertEqual(guarded["counts"]["3"]["candidate_set_high_unknown_veto_count"], 1)
        self.assertEqual(guarded["counts"]["3"]["candidate_set_high_unknown_veto_by_role"], {"unknown": 1})
        self.assertEqual(guarded["candidate_set_event_high_unknown_risk_veto"], 0.99)

        with self.assertRaisesRegex(ValueError, "candidate_set_max_label_high_unknown_risk_fraction"):
            evaluate_collaborative_open_set_evidence(
                rows,
                candidate_set_max_label_high_unknown_risk_fraction=1.5,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "candidate_set_high_unknown_risk_threshold"):
            evaluate_collaborative_open_set_evidence(
                rows,
                candidate_set_high_unknown_risk_threshold=-0.1,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "candidate_set_event_high_unknown_risk_veto"):
            evaluate_collaborative_open_set_evidence(
                rows,
                candidate_set_event_high_unknown_risk_veto=-1.0,
                **common,
            )

    def test_candidate_set_cvs_can_reject_high_class_shell_risk(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        rows = []
        for receiver_id in ("rx-a", "rx-b", "rx-c"):
            rows.append({
                "event_id": "shell-outlier-unknown",
                "receiver_id": receiver_id,
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "old-a",
                "known_score": 0.95,
                "known_margin": 0.40,
                "unknown_risk": 0.05,
                "score_risk": 0.05,
                "radius_risk": 0.05,
                "margin_risk": 0.05,
                "class_shell_risk": 0.92,
                "class_conformal_pvalue": 0.95,
                "class_conformal_support_count": 2,
                "class_evidence_top_m": 1,
                "class_evidence_top1_label": "old-a",
                "class_evidence_top1_score": 0.95,
                "class_evidence_top1_margin": 0.40,
                "class_evidence_top1_conformal_pvalue": 0.95,
                "class_evidence_top1_support_count": 2,
                "class_evidence_top1_unknown_risk": 0.05,
                "class_evidence_top1_score_risk": 0.05,
                "class_evidence_top1_radius_risk": 0.05,
                "class_evidence_top1_margin_risk": 0.05,
                "class_evidence_top1_class_shell_risk": 0.92,
                "bytes": 40.0,
                "latency_ms": 0.1,
            })

        metadata = {
            "source_receiver_ids": ["src-a"],
            "target_receiver_ids": ["rx-a", "rx-b", "rx-c"],
            "old_tx_ids": ["old-a"],
            "seen_new_tx_ids": ["new-a"],
            "unknown_tx_ids": ["unk-a"],
            "target_channel_view": "leo_clear_weak",
        }
        common = dict(
            collab_counts="3",
            fusion_policy="candidate_set_cvs",
            scorer_risk_components=["score", "radius", "margin", "class_shell"],
            candidate_set_min_receivers=2,
            candidate_set_min_conformal_pvalue=0.5,
            candidate_set_max_label_unknown_risk=1.0,
            candidate_set_max_event_unknown_risk=1.0,
            candidate_set_max_label_risk_component_agreement=1.0,
            candidate_set_unknown_reject_risk=1.1,
            protocol_metadata=metadata,
            strict_protocol_metadata=True,
        )

        unguarded = evaluate_collaborative_open_set_evidence(rows, **common)
        guarded = evaluate_collaborative_open_set_evidence(
            rows,
            candidate_set_max_label_shell_risk=0.80,
            candidate_set_shell_reject_risk=0.90,
            **common,
        )

        self.assertEqual(unguarded["counts"]["3"]["unknown_FAR"], 1.0)
        self.assertEqual(guarded["counts"]["3"]["unknown_FAR"], 0.0)
        self.assertEqual(guarded["counts"]["3"]["unknown_reject_rate"], 1.0)
        self.assertEqual(guarded["counts"]["3"]["candidate_set_shell_veto_count"], 1)
        self.assertEqual(guarded["counts"]["3"]["candidate_set_shell_veto_by_role"], {"unknown": 1})
        self.assertEqual(guarded["candidate_set_max_label_shell_risk"], 0.80)
        self.assertEqual(guarded["candidate_set_shell_reject_risk"], 0.90)

        with self.assertRaisesRegex(ValueError, "candidate_set_max_label_shell_risk"):
            evaluate_collaborative_open_set_evidence(
                rows,
                candidate_set_max_label_shell_risk=1.5,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "candidate_set_shell_reject_risk"):
            evaluate_collaborative_open_set_evidence(
                rows,
                candidate_set_shell_reject_risk=-0.1,
                **common,
            )

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
