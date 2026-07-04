import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2SocaprQknn8ParetoEvalTest(unittest.TestCase):
    def test_joint_score_rewards_unknown_reject_and_penalizes_far_defer(self):
        from phase2_socapr_qknn8_pareto_eval import _joint_score

        better = {
            "old_acc": 0.8,
            "seen_new_acc": 0.7,
            "unknown_reject_rate": 0.9,
            "unknown_FAR": 0.05,
            "defer_rate": 0.1,
        }
        worse = {
            "old_acc": 0.8,
            "seen_new_acc": 0.7,
            "unknown_reject_rate": 0.1,
            "unknown_FAR": 0.9,
            "defer_rate": 0.1,
        }
        self.assertGreater(_joint_score(better), _joint_score(worse))

    def test_route_configs_pin_qknn8_and_resource_packet_size(self):
        from phase2_socapr_qknn8_pareto_eval import ROUTE_CONFIGS

        known = ROUTE_CONFIGS["known_route"]
        safety = ROUTE_CONFIGS["safety_route"]
        self.assertIn("--qknn_k", known)
        self.assertEqual(known[known.index("--qknn_k") + 1], "8")
        self.assertEqual(safety[safety.index("--qknn_k") + 1], "8")
        self.assertEqual(known[known.index("--evidence_packet_bytes") + 1], "40")
        self.assertEqual(safety[safety.index("--evidence_packet_bytes") + 1], "128")
        self.assertIn("--class_shell_unknown_risk_enabled", safety)


if __name__ == "__main__":
    unittest.main()
