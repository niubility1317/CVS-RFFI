import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class Phase2OrbitEnpcCiEvalTest(unittest.TestCase):
    def test_enpc_pressure_uses_support_verifier_fields(self):
        from phase2_orbit_enpc_ci_eval import augment_enpc_evidence

        weak = {
            "event_id": "weak",
            "receiver_id": "rx-a",
            "role": "unknown",
            "predicted_label": "old-a",
            "known_score": 0.40,
            "known_margin": 0.01,
            "unknown_risk": 0.80,
            "class_verifier_top1_verified_score": 0.05,
            "class_verifier_second_verified_score": 0.04,
            "class_verifier_top1_pvalue": 0.05,
            "class_verifier_top1_receiver_class_reliability": 0.10,
            "class_verifier_top1_unknown_risk": 0.90,
        }
        strong = {
            **weak,
            "event_id": "strong",
            "role": "old",
            "known_score": 0.92,
            "known_margin": 0.30,
            "unknown_risk": 0.05,
            "class_verifier_top1_verified_score": 0.80,
            "class_verifier_second_verified_score": 0.10,
            "class_verifier_top1_pvalue": 0.95,
            "class_verifier_top1_receiver_class_reliability": 0.90,
            "class_verifier_top1_unknown_risk": 0.05,
        }

        rows = augment_enpc_evidence([weak, strong])

        self.assertGreater(rows[0]["enpc_episode_negative_pressure"], rows[1]["enpc_episode_negative_pressure"])
        self.assertLess(rows[0]["enpc_support_confidence"], rows[1]["enpc_support_confidence"])

    def test_enpc_collaborative_eval_accepts_known_and_rejects_unknown(self):
        from phase2_orbit_enpc_ci_eval import (
            EnpcProfile,
            augment_enpc_evidence,
            evaluate_enpc_collaborative_evidence,
        )

        rows = []
        for rx in ("rx-a", "rx-b", "rx-c"):
            rows.append(
                {
                    "event_id": "old-ok",
                    "receiver_id": rx,
                    "role": "old",
                    "true_label": "old-a",
                    "predicted_label": "old-a",
                    "known_score": 0.92,
                    "known_margin": 0.30,
                    "unknown_risk": 0.05,
                    "class_verifier_top1_verified_score": 0.80,
                    "class_verifier_second_verified_score": 0.10,
                    "class_verifier_top1_pvalue": 0.95,
                    "class_verifier_top1_receiver_class_reliability": 0.90,
                    "class_verifier_top1_unknown_risk": 0.05,
                    "bytes": 128.0,
                    "latency_ms": 0.1,
                }
            )
            rows.append(
                {
                    "event_id": "unk-bad",
                    "receiver_id": rx,
                    "role": "unknown",
                    "true_label": "__unknown__",
                    "predicted_label": "old-a" if rx != "rx-c" else "old-b",
                    "known_score": 0.40,
                    "known_margin": 0.01,
                    "unknown_risk": 0.95,
                    "class_verifier_top1_verified_score": 0.05,
                    "class_verifier_second_verified_score": 0.04,
                    "class_verifier_top1_pvalue": 0.05,
                    "class_verifier_top1_receiver_class_reliability": 0.10,
                    "class_verifier_top1_unknown_risk": 0.95,
                    "bytes": 128.0,
                    "latency_ms": 0.1,
                }
            )

        profile = EnpcProfile(
            name="test",
            description="test",
            accept_confidence=0.30,
            accept_margin=0.02,
            accept_max_pressure=0.70,
            support_accept_confidence=0.60,
            reject_pressure=0.65,
            reject_min_high_fraction=0.50,
            reject_min_disagreement=0.50,
            min_accept_receivers=1,
        )
        result = evaluate_enpc_collaborative_evidence(
            augment_enpc_evidence(rows),
            profile=profile,
            collab_counts=[1, 2, 3],
            collab_group_policy="available_up_to_k",
            partial_collab_min_receivers=1,
            max_event_bytes=384.0,
            max_event_latency_ms=20.0,
            metadata={
                "source_receiver_ids": ["src-a"],
                "target_receiver_ids": ["rx-a", "rx-b", "rx-c"],
                "old_tx_ids": ["old-a"],
                "seen_new_tx_ids": ["new-a"],
                "unknown_tx_ids": ["unk-a"],
                "target_channel_view": "leo_clear_weak",
            },
        )

        k3 = result["counts"]["3"]
        self.assertEqual(k3["old_acc"], 1.0)
        self.assertEqual(k3["unknown_reject_rate"], 1.0)
        self.assertEqual(k3["unknown_FAR"], 0.0)
        self.assertEqual(k3["per_old_class_total"]["old-a"], 1)
        self.assertIn("old->correct", k3["open_set_confusion"])

    def test_enpc_support_only_augmentation_does_not_require_unknown_rows(self):
        from phase2_orbit_enpc_ci_eval import augment_enpc_evidence

        rows = [
            {
                "event_id": "old-only",
                "receiver_id": "rx-a",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": 0.90,
                "known_margin": 0.20,
                "unknown_risk": 0.10,
            }
        ]

        out = augment_enpc_evidence(rows)

        self.assertEqual(len(out), 1)
        self.assertIn("enpc_episode_negative_pressure", out[0])
        self.assertIn("enpc_support_confidence", out[0])


if __name__ == "__main__":
    unittest.main()
