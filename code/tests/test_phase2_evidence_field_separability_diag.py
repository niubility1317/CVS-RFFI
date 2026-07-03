import unittest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2EvidenceFieldSeparabilityDiagTest(unittest.TestCase):
    def test_scans_risk_fields_under_far_constraint(self):
        from phase2_evidence_field_separability_diag import scan_fields

        rows = [
            {
                "role": "old",
                "true_label": "old-a",
                "predicted_label": "old-a",
                "unknown_risk": 0.1,
                "radius_risk": 0.2,
            },
            {
                "role": "seen_new",
                "true_label": "new-a",
                "predicted_label": "new-a",
                "unknown_risk": 0.2,
                "radius_risk": 0.3,
            },
            {
                "role": "unknown",
                "true_label": "__unknown__",
                "predicted_label": "old-a",
                "unknown_risk": 0.9,
                "radius_risk": 0.8,
            },
        ]

        result = scan_fields(
            rows,
            risk_fields=["unknown_risk", "radius_risk"],
            max_combo_size=1,
            modes=["max"],
            far_targets=[0.0],
            max_thresholds=64,
        )

        best = result["best_by_far_target"]["0.0"]
        self.assertTrue(result["diagnostic_only"])
        self.assertTrue(result["uses_query_labels_for_oracle_sweep"])
        self.assertEqual(best["unknown_FAR"], 0.0)
        self.assertEqual(best["unknown_reject_rate"], 1.0)
        self.assertEqual(best["old_acc"], 1.0)
        self.assertEqual(best["seen_new_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
