import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2SocaprDualRouteVetoEvalTest(unittest.TestCase):
    def test_discounted_safety_risk_keeps_strong_known_candidates_lower_risk(self):
        from phase2_socapr_dual_route_veto_eval import _discounted_safety_risk

        safety = {"unknown_risk": "1.0"}
        strong = {"known_score": "0.9", "known_margin": "0.8", "unknown_risk": "0.01"}
        weak = {"known_score": "0.1", "known_margin": "0.02", "unknown_risk": "0.01"}
        strong_risk = _discounted_safety_risk(
            strong,
            safety,
            score_anchor=0.7,
            margin_anchor=0.4,
            safety_weight=0.2,
            discount_mode="prod",
        )
        weak_risk = _discounted_safety_risk(
            weak,
            safety,
            score_anchor=0.7,
            margin_anchor=0.4,
            safety_weight=0.2,
            discount_mode="prod",
        )
        self.assertLess(strong_risk, weak_risk)
        self.assertGreaterEqual(strong_risk, 0.01)

    def test_build_dual_route_requires_matching_safety_rows(self):
        from phase2_socapr_dual_route_veto_eval import build_dual_route_evidence

        known = [
            {
                "event_id": "e1",
                "receiver_id": "r1",
                "role": "old",
                "true_label": "a",
                "known_score": "0.8",
                "known_margin": "0.6",
                "unknown_risk": "0.0",
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "missing safety row"):
            build_dual_route_evidence(known, [])

    def test_build_dual_route_combines_resource_costs(self):
        from phase2_socapr_dual_route_veto_eval import build_dual_route_evidence

        known = [
            {
                "event_id": "e1",
                "receiver_id": "r1",
                "role": "old",
                "true_label": "a",
                "known_score": "0.8",
                "known_margin": "0.6",
                "unknown_risk": "0.0",
                "bytes": "40",
                "latency_ms": "0.1",
            }
        ]
        safety = [
            {
                "event_id": "e1",
                "receiver_id": "r1",
                "role": "old",
                "true_label": "a",
                "unknown_risk": "1.0",
                "bytes": "128",
                "latency_ms": "0.2",
            }
        ]
        combined = build_dual_route_evidence(known, safety)
        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0]["bytes"], 168.0)
        self.assertEqual(combined[0]["latency_ms"], 0.2)
        self.assertEqual(combined[0]["reliability_source"], "socapr_dual_route_known_candidate_safety_veto")


if __name__ == "__main__":
    unittest.main()
