import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


class AdaptationSafetyTest(unittest.TestCase):
    def test_rollback_gate_triggers_on_old_class_drop_and_unknown_rise(self):
        from cvsrffi.adaptation_safety import SafetyRule, evaluate_rollback_gate

        decision = evaluate_rollback_gate(
            before_metrics={"old_class_accuracy": 0.92, "unknown_false_accept_rate": 0.02, "coverage": 0.80},
            after_metrics={"old_class_accuracy": 0.84, "unknown_false_accept_rate": 0.11, "coverage": 0.70},
            rules=[
                SafetyRule("old_class_accuracy", "max_drop", 0.05),
                SafetyRule("unknown_false_accept_rate", "max_rise", 0.05),
                SafetyRule("coverage", "min", 0.50),
            ],
        )

        self.assertFalse(decision.accepted)
        self.assertTrue(decision.rollback_triggered)
        self.assertEqual([item["metric"] for item in decision.triggered_rules], ["old_class_accuracy", "unknown_false_accept_rate"])

    def test_rollback_gate_skips_missing_metrics_without_false_trigger(self):
        from cvsrffi.adaptation_safety import SafetyRule, evaluate_rollback_gate

        decision = evaluate_rollback_gate(
            before_metrics={"coverage": 0.9},
            after_metrics={"coverage": 0.8},
            rules=[SafetyRule("unknown_false_accept_rate", "max_rise", 0.05)],
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(len(decision.skipped_rules), 1)

    def test_nested_metric_paths_support_target_adaptation_safety(self):
        from cvsrffi.adaptation_safety import SafetyRule, evaluate_rollback_gate

        decision = evaluate_rollback_gate(
            before_metrics={"test": {"tx_acc": 80.0}, "target": {"tx_acc": 50.0}, "sat_score": 60.0},
            after_metrics={"test": {"tx_acc": 79.5}, "target": {"tx_acc": 55.0}, "sat_score": 59.6},
            rules=[
                SafetyRule("test.tx_acc", "max_drop", 1.0),
                SafetyRule("sat_score", "max_drop", 1.0),
                SafetyRule("target.tx_acc", "min_gain", 0.0),
            ],
        )

        self.assertTrue(decision.accepted)


if __name__ == "__main__":
    unittest.main()
