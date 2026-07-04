import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class Phase2OrbitSovcCiEvalTest(unittest.TestCase):
    def test_sovc_raises_unknown_risk_for_weak_verifier(self):
        from phase2_orbit_sovc_ci_eval import augment_sovc_evidence

        rows = [
            {
                "event_id": "unknown-like-known",
                "receiver_id": "rx-a",
                "predicted_label": "old-a",
                "unknown_risk": 0.20,
                "pcet_unknown_risk": 0.30,
                "class_verifier_changed": 1,
                "class_verifier_top1_raw_score": 0.90,
                "class_verifier_top1_verified_score": 0.10,
                "class_verifier_second_verified_score": 0.09,
                "class_verifier_top1_pvalue": 0.05,
                "class_verifier_top1_receiver_class_reliability": 0.10,
                "class_verifier_top1_unknown_risk": 0.92,
                "class_verifier_top1_class_negative_risk": 0.90,
                "class_verifier_top1_class_shell_risk": 0.95,
            }
        ]

        row = augment_sovc_evidence(rows)[0]

        self.assertGreater(row["sovc_verifier_risk"], 0.70)
        self.assertGreater(row["sovc_unknown_risk"], 0.40)
        self.assertEqual(row["unknown_risk"], row["sovc_unknown_risk"])
        self.assertFalse(row["sovc_safe_known_cap_applied"])

    def test_sovc_caps_safe_verified_known_risk(self):
        from phase2_orbit_sovc_ci_eval import augment_sovc_evidence

        rows = [
            {
                "event_id": "known-safe",
                "receiver_id": "rx-a",
                "predicted_label": "old-a",
                "unknown_risk": 0.70,
                "pcet_unknown_risk": 0.70,
                "class_verifier_changed": 0,
                "class_verifier_top1_raw_score": 0.60,
                "class_verifier_top1_verified_score": 0.50,
                "class_verifier_second_verified_score": 0.10,
                "class_verifier_top1_pvalue": 0.95,
                "class_verifier_top1_receiver_class_reliability": 0.90,
                "class_verifier_top1_unknown_risk": 0.05,
                "class_verifier_top1_class_negative_risk": 0.05,
                "class_verifier_top1_class_shell_risk": 0.05,
            }
        ]

        row = augment_sovc_evidence(rows, safe_known_risk_cap=0.45)[0]

        self.assertTrue(row["sovc_safe_known_cap_applied"])
        self.assertLessEqual(row["sovc_unknown_risk"], 0.45)

    def test_sovc_augmented_evidence_works_with_stage2_protocol_eval(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence
        from phase2_orbit_sovc_ci_eval import augment_sovc_evidence

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
                    "pcet_unknown_risk": 0.20,
                    "class_conformal_pvalue": 0.95,
                    "receiver_class_reliability": 0.90,
                    "support_density": 0.90,
                    "class_verifier_changed": 0,
                    "class_verifier_top1_raw_score": 0.80,
                    "class_verifier_top1_verified_score": 0.70,
                    "class_verifier_second_verified_score": 0.10,
                    "class_verifier_top1_pvalue": 0.95,
                    "class_verifier_top1_receiver_class_reliability": 0.90,
                    "class_verifier_top1_unknown_risk": 0.05,
                    "class_verifier_top1_class_negative_risk": 0.05,
                    "class_verifier_top1_class_shell_risk": 0.05,
                    "bytes": 128.0,
                    "latency_ms": 0.1,
                }
            )
            rows.append(
                {
                    "event_id": "unk-verifier",
                    "receiver_id": receiver_id,
                    "role": "unknown",
                    "true_label": "__unknown__",
                    "predicted_label": "old-a",
                    "known_score": 0.90,
                    "known_margin": 0.01,
                    "unknown_risk": 0.20,
                    "pcet_unknown_risk": 0.50,
                    "class_conformal_pvalue": 0.05,
                    "receiver_class_reliability": 0.10,
                    "support_density": 0.10,
                    "class_verifier_changed": 1,
                    "class_verifier_top1_raw_score": 0.90,
                    "class_verifier_top1_verified_score": 0.10,
                    "class_verifier_second_verified_score": 0.09,
                    "class_verifier_top1_pvalue": 0.05,
                    "class_verifier_top1_receiver_class_reliability": 0.10,
                    "class_verifier_top1_unknown_risk": 0.95,
                    "class_verifier_top1_class_negative_risk": 0.95,
                    "class_verifier_top1_class_shell_risk": 0.95,
                    "bytes": 128.0,
                    "latency_ms": 0.1,
                }
            )

        result = evaluate_collaborative_open_set_evidence(
            augment_sovc_evidence(rows, base_weight=0.40, verifier_weight=0.60),
            collab_counts=[2],
            fusion_policy="scg_qknn_cvs",
            scorer_risk_components=["score", "radius", "margin", "evt", "class_shell"],
            candidate_set_min_receivers=2,
            candidate_set_min_top1_receivers=1,
            candidate_set_unknown_reject_risk=0.58,
            candidate_set_max_label_unknown_risk=0.58,
            candidate_set_max_event_unknown_risk=0.62,
            candidate_set_max_label_shell_risk=0.70,
            candidate_set_shell_reject_risk=0.70,
            unknown_risk_threshold=0.58,
            accept_margin_threshold=0.04,
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
