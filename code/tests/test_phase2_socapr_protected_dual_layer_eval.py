import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2SocaprProtectedDualLayerEvalTest(unittest.TestCase):
    def test_strong_known_candidate_is_protected_from_safety_veto(self):
        from phase2_socapr_protected_dual_layer_eval import build_protected_dual_layer_evidence

        known = [
            {
                "event_id": "e1",
                "receiver_id": "r1",
                "role": "old",
                "true_label": "old_a",
                "known_score": "0.92",
                "known_margin": "0.45",
                "support_density": "0.75",
                "class_conformal_pvalue": "0.50",
                "unknown_risk": "0.02",
            }
        ]
        safety = [
            {
                "event_id": "e1",
                "receiver_id": "r1",
                "role": "old",
                "true_label": "old_a",
                "unknown_risk": "1.0",
                "class_negative_risk": "1.0",
                "class_shell_risk": "1.0",
                "evt_risk": "1.0",
            }
        ]
        protected = build_protected_dual_layer_evidence(known, safety)
        self.assertEqual(protected[0]["protected_decision"], "known_protected_accept")
        self.assertEqual(protected[0]["old_guard_pass"], 1)
        self.assertLessEqual(protected[0]["unknown_risk"], 0.02)

    def test_weak_candidate_with_two_safety_signals_is_vetoed(self):
        from phase2_socapr_protected_dual_layer_eval import build_protected_dual_layer_evidence

        known = [
            {
                "event_id": "e1",
                "receiver_id": "r1",
                "role": "unknown",
                "true_label": "u1",
                "known_score": "0.35",
                "known_margin": "0.03",
                "support_density": "0.10",
                "class_conformal_pvalue": "0.01",
                "unknown_risk": "0.05",
            }
        ]
        safety = [
            {
                "event_id": "e1",
                "receiver_id": "r1",
                "role": "unknown",
                "true_label": "u1",
                "unknown_risk": "0.95",
                "class_negative_risk": "0.91",
                "class_shell_risk": "0.88",
                "evt_risk": "0.20",
            }
        ]
        protected = build_protected_dual_layer_evidence(known, safety)
        self.assertEqual(protected[0]["protected_decision"], "unknown_veto")
        self.assertEqual(protected[0]["unknown_veto_pass"], 1)
        self.assertGreaterEqual(protected[0]["unknown_risk"], 0.95)

    def test_safety_veto_requires_weak_known_candidate(self):
        from phase2_socapr_protected_dual_layer_eval import build_protected_dual_layer_evidence

        known = [
            {
                "event_id": "e1",
                "receiver_id": "r1",
                "role": "unknown",
                "true_label": "u1",
                "known_score": "0.82",
                "known_margin": "0.40",
                "support_density": "0.10",
                "unknown_risk": "0.05",
            }
        ]
        safety = [
            {
                "event_id": "e1",
                "receiver_id": "r1",
                "role": "unknown",
                "true_label": "u1",
                "unknown_risk": "0.99",
                "class_negative_risk": "0.99",
                "class_shell_risk": "0.99",
                "evt_risk": "0.99",
            }
        ]
        protected = build_protected_dual_layer_evidence(known, safety)
        self.assertNotEqual(protected[0]["protected_decision"], "unknown_veto")
        self.assertEqual(protected[0]["unknown_veto_pass"], 0)
        self.assertAlmostEqual(protected[0]["unknown_risk"], 0.05)

    def test_summary_row_exports_baseline_deltas_and_constraint(self):
        from phase2_socapr_protected_dual_layer_eval import _summary_row

        row = _summary_row(
            threshold=0.7,
            fusion_policy="risk_margin",
            count="5",
            metrics={
                "old_acc": 0.79,
                "known_coverage": 0.90,
                "known_accepted_accuracy": 0.88,
                "unknown_FAR": 0.03,
            },
            baseline_metrics={
                "old_acc": 0.80,
                "known_coverage": 0.91,
                "known_accepted_accuracy": 0.87,
            },
            old_tolerance=0.0,
            coverage_tolerance=0.0,
        )
        self.assertAlmostEqual(row["old_acc_delta_vs_known_route"], -0.01)
        self.assertAlmostEqual(row["known_coverage_delta_vs_known_route"], -0.01)
        self.assertAlmostEqual(row["known_accepted_acc_delta_vs_known_route"], 0.01)
        self.assertEqual(row["baseline_constraint_pass"], 0)


if __name__ == "__main__":
    unittest.main()
