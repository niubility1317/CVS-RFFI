import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2OldProtectedArcCiEvalTest(unittest.TestCase):
    def test_strong_old_candidate_caps_unknown_risk(self):
        from phase2_old_protected_arc_ci_eval import _profile_by_name, augment_arc_evidence

        row = {
            "predicted_label": "14-10",
            "class_evidence_top1_label": "14-10",
            "known_score": 0.80,
            "known_margin": 0.20,
            "support_density": 0.70,
            "receiver_class_reliability": 0.90,
            "unknown_risk": 0.70,
            "socapr_safety_route_unknown_risk": 0.95,
            "socapr_safety_class_negative_risk": 0.10,
        }

        out = augment_arc_evidence([row], _profile_by_name("arc_old_floor"), old_labels={"14-10"})[0]

        self.assertTrue(out["arc_strong_old_candidate"])
        self.assertLessEqual(out["unknown_risk"], 0.30)

    def test_weak_candidate_raises_empty_set_risk(self):
        from phase2_old_protected_arc_ci_eval import _profile_by_name, augment_arc_evidence

        row = {
            "predicted_label": "19-3",
            "class_evidence_top1_label": "19-3",
            "known_score": 0.10,
            "known_margin": 0.005,
            "support_density": 0.05,
            "receiver_class_reliability": 0.10,
            "unknown_risk": 0.05,
            "socapr_safety_route_unknown_risk": 0.95,
            "socapr_safety_class_negative_risk": 0.85,
        }

        out = augment_arc_evidence([row], _profile_by_name("arc_unknown_safe"), old_labels={"14-10"})[0]

        self.assertFalse(out["arc_strong_old_candidate"])
        self.assertGreater(out["arc_empty_candidate_risk"], 0.70)
        self.assertGreater(out["unknown_risk"], 0.70)

    def test_parse_default_profiles(self):
        from phase2_old_protected_arc_ci_eval import _profile_names

        self.assertEqual(
            _profile_names("all"),
            ["arc_old_floor", "arc_balanced", "arc_unknown_safe"],
        )


if __name__ == "__main__":
    unittest.main()
