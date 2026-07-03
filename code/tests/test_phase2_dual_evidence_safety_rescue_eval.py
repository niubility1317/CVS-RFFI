import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2DualEvidenceSafetyRescueEvalTest(unittest.TestCase):
    def test_replaces_known_label_only_after_safety_accept(self):
        from phase2_dual_evidence_safety_rescue_eval import fuse_dual_evidence_event_results

        safety_count = {
            "bytes_per_event": 80.0,
            "latency_ms_p95": 0.2,
            "event_results": [
                {
                    "event_id": "old-1",
                    "role": "old",
                    "true_label": "old-a",
                    "decision": "accept",
                    "output_label": "old-b",
                },
                {
                    "event_id": "unk-1",
                    "role": "unknown",
                    "true_label": "__unknown__",
                    "decision": "unknown_reject",
                    "output_label": "__unknown__",
                },
                {
                    "event_id": "old-2",
                    "role": "old",
                    "true_label": "old-a",
                    "decision": "defer",
                    "output_label": "",
                },
            ],
        }
        known_count = {
            "event_results": [
                {
                    "event_id": "old-1",
                    "role": "old",
                    "true_label": "old-a",
                    "decision": "accept",
                    "output_label": "old-a",
                },
                {
                    "event_id": "unk-1",
                    "role": "unknown",
                    "true_label": "__unknown__",
                    "decision": "accept",
                    "output_label": "old-a",
                },
                {
                    "event_id": "old-2",
                    "role": "old",
                    "true_label": "old-a",
                    "decision": "accept",
                    "output_label": "old-a",
                },
            ],
        }

        fused = fuse_dual_evidence_event_results(safety_count, known_count)

        self.assertEqual(fused["old_acc"], 0.5)
        self.assertEqual(fused["unknown_reject_rate"], 1.0)
        self.assertEqual(fused["unknown_FAR"], 0.0)
        self.assertEqual(fused["defer_rate"], 1 / 3)
        by_event = {event["event_id"]: event for event in fused["event_results"]}
        self.assertEqual(by_event["old-1"]["output_label"], "old-a")
        self.assertEqual(by_event["old-1"]["label_source"], "known_route_safe_accept")
        self.assertEqual(by_event["unk-1"]["decision"], "unknown_reject")
        self.assertEqual(by_event["old-2"]["decision"], "defer")


if __name__ == "__main__":
    unittest.main()
