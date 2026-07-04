import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class Phase2OrbitPcetCiEvalTest(unittest.TestCase):
    def test_pcet_raises_unknown_risk_for_unstable_tail_evidence(self):
        from phase2_orbit_pcet_ci_eval import augment_pcet_evidence

        rows = [
            {
                "event_id": "unknown-like-known",
                "receiver_id": "rx-a",
                "predicted_label": "old-a",
                "known_score": 0.90,
                "known_margin": 0.01,
                "unknown_risk": 0.10,
                "class_evidence_top1_label": "old-a",
                "class_evidence_top1_score": 0.90,
                "class_evidence_top2_score": 0.89,
                "class_evidence_top1_conformal_pvalue": 0.05,
                "class_evidence_top1_receiver_class_reliability": 0.10,
                "class_evidence_top1_support_count": 1,
                "class_evidence_top1_evt_risk": 0.92,
                "class_evidence_top1_class_shell_risk": 0.90,
            }
        ]

        out = augment_pcet_evidence(rows)
        row = out[0]

        self.assertGreater(row["pcet_proto_consistency_risk"], 0.80)
        self.assertGreater(row["pcet_tail_risk"], 0.90)
        self.assertGreaterEqual(row["pcet_unknown_risk"], 0.59)
        self.assertEqual(row["unknown_risk"], row["pcet_unknown_risk"])
        self.assertFalse(row["pcet_safe_known_cap_applied"])

    def test_pcet_caps_safe_support_confirmed_known_risk(self):
        from phase2_orbit_pcet_ci_eval import augment_pcet_evidence

        rows = [
            {
                "event_id": "known-safe",
                "receiver_id": "rx-a",
                "predicted_label": "old-a",
                "known_score": 0.92,
                "known_margin": 0.20,
                "unknown_risk": 0.70,
                "class_evidence_top1_label": "old-a",
                "class_evidence_top1_score": 0.92,
                "class_evidence_top2_score": 0.40,
                "class_evidence_top1_conformal_pvalue": 0.95,
                "class_evidence_top1_receiver_class_reliability": 0.90,
                "class_evidence_top1_support_count": 3,
                "class_evidence_top1_evt_risk": 0.10,
                "class_evidence_top1_class_shell_risk": 0.10,
            }
        ]

        out = augment_pcet_evidence(rows, safe_known_risk_cap=0.55)
        row = out[0]

        self.assertTrue(row["pcet_safe_known_cap_applied"])
        self.assertLessEqual(row["pcet_unknown_risk"], 0.55)

    def test_pcet_augmented_evidence_works_with_stage2_protocol_eval(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence
        from phase2_orbit_pcet_ci_eval import augment_pcet_evidence

        rows = []
        for receiver_id in ("rx-a", "rx-b"):
            rows.append(
                {
                    "event_id": "old-ok",
                    "receiver_id": receiver_id,
                    "role": "old",
                    "true_label": "old-a",
                    "predicted_label": "old-a",
                    "known_score": 0.92,
                    "known_margin": 0.20,
                    "unknown_risk": 0.20,
                    "class_conformal_pvalue": 0.95,
                    "receiver_class_reliability": 0.90,
                    "support_density": 0.90,
                    "class_evidence_top1_label": "old-a",
                    "class_evidence_top1_score": 0.92,
                    "class_evidence_top2_score": 0.40,
                    "class_evidence_top1_conformal_pvalue": 0.95,
                    "class_evidence_top1_receiver_class_reliability": 0.90,
                    "class_evidence_top1_support_count": 3,
                    "class_evidence_top1_class_shell_risk": 0.10,
                    "bytes": 128.0,
                    "latency_ms": 0.1,
                }
            )
            rows.append(
                {
                    "event_id": "unk-tail",
                    "receiver_id": receiver_id,
                    "role": "unknown",
                    "true_label": "__unknown__",
                    "predicted_label": "old-a",
                    "known_score": 0.90,
                    "known_margin": 0.01,
                    "unknown_risk": 0.20,
                    "class_conformal_pvalue": 0.05,
                    "receiver_class_reliability": 0.10,
                    "support_density": 0.10,
                    "class_evidence_top1_label": "old-a",
                    "class_evidence_top1_score": 0.90,
                    "class_evidence_top2_score": 0.89,
                    "class_evidence_top1_conformal_pvalue": 0.05,
                    "class_evidence_top1_receiver_class_reliability": 0.10,
                    "class_evidence_top1_support_count": 1,
                    "class_evidence_top1_class_shell_risk": 0.95,
                    "bytes": 128.0,
                    "latency_ms": 0.1,
                }
            )

        result = evaluate_collaborative_open_set_evidence(
            augment_pcet_evidence(rows),
            collab_counts=[2],
            fusion_policy="scg_qknn_cvs",
            scorer_risk_components=["score", "radius", "margin", "evt", "class_shell"],
            candidate_set_min_receivers=2,
            candidate_set_min_top1_receivers=1,
            candidate_set_unknown_reject_risk=0.62,
            candidate_set_max_label_unknown_risk=0.60,
            candidate_set_max_event_unknown_risk=0.66,
            candidate_set_max_label_shell_risk=0.70,
            candidate_set_shell_reject_risk=0.70,
            unknown_risk_threshold=0.62,
            accept_margin_threshold=0.05,
            label_fusion_policy="weighted_vote_margin",
            receiver_class_reliability_policy="support_calibrated",
            strict_protocol_metadata=True,
            protocol_metadata={
                "source_receiver_ids": ["src-a"],
                "target_receiver_ids": ["rx-a", "rx-b"],
                "old_tx_ids": ["old-a"],
                "seen_new_tx_ids": ["new-a"],
                "unknown_tx_ids": ["unk-a"],
                "target_channel_view": "leo_clear_weak",
            },
        )

        k2 = result["counts"]["2"]
        self.assertEqual(k2["old_acc"], 1.0)
        self.assertEqual(k2["unknown_reject_rate"], 1.0)
        self.assertEqual(k2["unknown_FAR"], 0.0)


if __name__ == "__main__":
    unittest.main()
