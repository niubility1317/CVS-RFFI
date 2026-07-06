import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2SupportMetricQknnScenarioFallbackCliTest(unittest.TestCase):
    def test_new_only_explicit_mode_targets_new_labels(self):
        from phase2_support_metric_qknn_probe import _resolve_scenario_class_fallback_mode

        enabled, labels, role_old_only, normalized = _resolve_scenario_class_fallback_mode(
            "new",
            old_labels=["old-a", "old-b"],
            new_labels=["new-a", "new-b"],
            enable_scenario_class_fallback=False,
            scenario_class_fallback_labels=None,
            use_role_split_old_only_fallback=False,
        )

        self.assertTrue(enabled)
        self.assertEqual(labels, {"new-a", "new-b"})
        self.assertFalse(role_old_only)
        self.assertEqual(normalized, "new_only")

    def test_none_preserves_adaptive_fallback_state(self):
        from phase2_support_metric_qknn_probe import _resolve_scenario_class_fallback_mode

        enabled, labels, role_old_only, normalized = _resolve_scenario_class_fallback_mode(
            "none",
            old_labels=["old-a"],
            new_labels=["new-a"],
            enable_scenario_class_fallback=True,
            scenario_class_fallback_labels={"old-a"},
            use_role_split_old_only_fallback=True,
        )

        self.assertTrue(enabled)
        self.assertEqual(labels, {"old-a"})
        self.assertTrue(role_old_only)
        self.assertEqual(normalized, "none")

    def test_rejects_unknown_mode(self):
        from phase2_support_metric_qknn_probe import _resolve_scenario_class_fallback_mode

        with self.assertRaises(ValueError):
            _resolve_scenario_class_fallback_mode(
                "query_labels",
                old_labels=["old-a"],
                new_labels=["new-a"],
                enable_scenario_class_fallback=False,
                scenario_class_fallback_labels=None,
                use_role_split_old_only_fallback=False,
            )


if __name__ == "__main__":
    unittest.main()
