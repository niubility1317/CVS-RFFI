import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class Phase2CoteCiEvalTest(unittest.TestCase):
    def test_known_shield_prevents_high_risk_old_from_rejection(self):
        from phase2_cote_ci_eval import PROFILES, _fuse_cote_event

        row = {
            "event_id": "e1",
            "role": "old",
            "true_label": "old-a",
            "receiver_id": "rx-a",
            "class_evidence_top1_label": "old-a",
            "class_evidence_top1_score": 0.95,
            "class_evidence_top1_margin": 0.40,
            "class_evidence_top1_conformal_pvalue": 1.0,
            "class_evidence_top1_receiver_class_reliability": 1.0,
            "class_evidence_top1_unknown_risk": 0.90,
            "pcet_unknown_risk": 0.90,
            "support_density": 1.0,
            "reliability": 1.0,
            "bytes": 128,
            "latency_ms": 1.0,
        }
        out = _fuse_cote_event(
            [row, {**row, "receiver_id": "rx-b"}],
            profile=PROFILES[1],
            top_m=1,
            old_labels={"old-a"},
            seen_labels={"new-a"},
            max_event_bytes=1152,
            max_event_latency_ms=20,
        )
        self.assertEqual(out["output_label"], "old-a")
        self.assertEqual(out["decision"], "accept_known")

    def test_unknown_requires_no_known_shield_and_cross_receiver_risk(self):
        from phase2_cote_ci_eval import PROFILES, UNKNOWN_LABEL, _fuse_cote_event

        base = {
            "event_id": "u1",
            "role": "unknown",
            "true_label": "unk-a",
            "class_evidence_top1_label": "old-a",
            "class_evidence_top1_score": 0.15,
            "class_evidence_top1_margin": 0.01,
            "class_evidence_top1_conformal_pvalue": 0.0,
            "class_evidence_top1_receiver_class_reliability": 0.1,
            "class_evidence_top1_unknown_risk": 0.99,
            "pcet_unknown_risk": 0.99,
            "class_negative_risk": 0.99,
            "support_density": 0.1,
            "reliability": 0.1,
            "bytes": 128,
            "latency_ms": 1.0,
        }
        rows = [{**base, "receiver_id": f"rx-{i}"} for i in range(3)]
        out = _fuse_cote_event(
            rows,
            profile=PROFILES[1],
            top_m=1,
            old_labels={"old-a"},
            seen_labels={"new-a"},
            max_event_bytes=1152,
            max_event_latency_ms=20,
        )
        self.assertEqual(out["output_label"], UNKNOWN_LABEL)
        self.assertEqual(out["decision"], "reject_unknown")


if __name__ == "__main__":
    unittest.main()
