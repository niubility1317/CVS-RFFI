import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2EventAlignmentAuditTest(unittest.TestCase):
    def test_audit_marks_strict_candidate_when_sig_key_spans_receivers(self):
        from phase2_event_alignment_audit import audit_rows

        rows = [
            {
                "role": "target_old",
                "tx_id": "a",
                "rx_id": "r1",
                "day_id": "d",
                "eq_id": "1",
                "sig_id": "42",
                "channel_view": "satellite",
                "sat_scenario": "leo_clear_weak",
            },
            {
                "role": "target_old",
                "tx_id": "a",
                "rx_id": "r2",
                "day_id": "d",
                "eq_id": "1",
                "sig_id": "42",
                "channel_view": "satellite",
                "sat_scenario": "leo_clear_weak",
            },
        ]
        result = audit_rows(rows, target_receivers=["r1", "r2"], receiver_count=2)
        self.assertEqual(result["strict_candidate_group_count"], 1)
        self.assertEqual(result["full_receiver_group_count"], 1)
        self.assertTrue(result["strict_event_candidate_possible"])

    def test_audit_rejects_rank_only_when_sig_key_is_not_shared(self):
        from phase2_event_alignment_audit import audit_rows

        rows = [
            {
                "role": "target_old",
                "tx_id": "a",
                "rx_id": "r1",
                "day_id": "d",
                "eq_id": "1",
                "sig_id": "42",
                "channel_view": "satellite",
                "sat_scenario": "leo_clear_weak",
            },
            {
                "role": "target_old",
                "tx_id": "a",
                "rx_id": "r2",
                "day_id": "d",
                "eq_id": "1",
                "sig_id": "43",
                "channel_view": "satellite",
                "sat_scenario": "leo_clear_weak",
            },
        ]
        result = audit_rows(rows, target_receivers=["r1", "r2"], receiver_count=2)
        self.assertEqual(result["strict_candidate_group_count"], 0)
        self.assertFalse(result["strict_event_candidate_possible"])

    def test_query_only_audit_does_not_count_support_rows(self):
        from phase2_event_alignment_audit import audit_rows

        rows = [
            {
                "role": "target_old",
                "tx_id": "a",
                "rx_id": "r1",
                "day_id": "d",
                "eq_id": "1",
                "sig_id": "1",
                "channel_view": "satellite",
                "sat_scenario": "leo_clear_weak",
                "split": "support",
            },
            {
                "role": "target_old",
                "tx_id": "a",
                "rx_id": "r2",
                "day_id": "d",
                "eq_id": "1",
                "sig_id": "1",
                "channel_view": "satellite",
                "sat_scenario": "leo_clear_weak",
                "split": "support",
            },
        ]
        result = audit_rows(rows, target_receivers=["r1", "r2"], receiver_count=2, split_filter="query")
        self.assertEqual(result["strict_candidate_group_count"], 0)
        self.assertFalse(result["strict_event_candidate_possible"])


if __name__ == "__main__":
    unittest.main()
