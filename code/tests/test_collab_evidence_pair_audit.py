import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class CollabEvidencePairAuditTest(unittest.TestCase):
    def test_builds_pair_matrix_and_error_rows(self):
        from scripts.collab_evidence_pair_audit import build_pair_audit

        rows = [
            {
                "event_id": "old-1",
                "receiver_id": "rx-a",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": "0.90",
                "known_margin": "0.40",
                "unknown_risk": "0.10",
                "latency_ms": "2.0",
                "bytes": "40",
            },
            {
                "event_id": "old-1",
                "receiver_id": "rx-b",
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "known_score": "0.85",
                "known_margin": "0.30",
                "unknown_risk": "0.15",
                "latency_ms": "3.0",
                "bytes": "40",
            },
            {
                "event_id": "unknown-1",
                "receiver_id": "rx-a",
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "old-a",
                "known_score": "0.80",
                "known_margin": "0.20",
                "unknown_risk": "0.95",
                "latency_ms": "2.0",
                "bytes": "40",
            },
            {
                "event_id": "unknown-1",
                "receiver_id": "rx-b",
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "old-b",
                "known_score": "0.82",
                "known_margin": "0.25",
                "unknown_risk": "0.96",
                "latency_ms": "3.0",
                "bytes": "40",
            },
        ]

        pair_rows, error_rows = build_pair_audit(
            rows,
            {
                "unknown_risk_threshold": 0.8,
                "accept_margin_threshold": 0.1,
                "fusion_policy": "consensus_veto",
                "consensus_gap_threshold": 0.5,
            },
            max_error_rows=10,
        )

        self.assertEqual(len(pair_rows), 1)
        self.assertEqual(pair_rows[0]["receiver_pair"], "rx-a+rx-b")
        self.assertEqual(pair_rows[0]["old_acc"], 1.0)
        self.assertIn("mean_receiver_pair_label_disagreement", pair_rows[0])
        self.assertEqual(error_rows, [])


if __name__ == "__main__":
    unittest.main()
