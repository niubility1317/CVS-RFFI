import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2OldProtectedPcetVetoCiEvalTest(unittest.TestCase):
    def test_strong_known_caps_added_unknown_risk(self):
        from phase2_old_protected_pcet_veto_ci_eval import _profile_by_name, augment_opv_evidence

        row = {
            "unknown_risk": 0.10,
            "socapr_safety_route_unknown_risk": 1.0,
            "socapr_safety_class_negative_risk": 1.0,
            "known_score": 0.90,
            "known_margin": 0.30,
            "support_density": 0.80,
            "receiver_class_reliability": 0.95,
        }

        out = augment_opv_evidence([row], _profile_by_name("opv_balanced"))[0]

        self.assertTrue(out["opv_strong_known_cap_applied"])
        self.assertLessEqual(out["unknown_risk"], 0.52)

    def test_weak_known_high_tail_raises_unknown_risk(self):
        from phase2_old_protected_pcet_veto_ci_eval import _profile_by_name, augment_opv_evidence

        row = {
            "unknown_risk": 0.10,
            "socapr_safety_route_unknown_risk": 0.90,
            "socapr_safety_class_negative_risk": 0.80,
            "socapr_safety_class_shell_risk": 0.70,
            "known_score": 0.20,
            "known_margin": 0.01,
            "support_density": 0.10,
            "receiver_class_reliability": 0.20,
        }

        out = augment_opv_evidence([row], _profile_by_name("opv_unknown_push"))[0]

        self.assertFalse(out["opv_strong_known_cap_applied"])
        self.assertGreater(out["opv_weak_known_gate"], 0.50)
        self.assertGreater(out["unknown_risk"], 0.50)

    def test_parse_default_profiles(self):
        from phase2_old_protected_pcet_veto_ci_eval import _profile_names

        self.assertEqual(
            _profile_names("all"),
            ["opv_ultra_preserve", "opv_preserve", "opv_balanced", "opv_unknown_push"],
        )

    def test_summary_order_is_not_unknown_metric_ranking(self):
        from phase2_old_protected_pcet_veto_ci_eval import _pre_registered_summary_order

        profile_names = ["opv_ultra_preserve", "opv_unknown_push"]
        policy_names = ["opu_old_preserve"]
        ultra = {
            "profile": "opv_ultra_preserve",
            "policy": "opu_old_preserve",
            "collab_count": 1,
            "unknown_reject_rate": 0.10,
        }
        unknown_push = {
            "profile": "opv_unknown_push",
            "policy": "opu_old_preserve",
            "collab_count": 1,
            "unknown_reject_rate": 0.90,
        }

        self.assertLess(
            _pre_registered_summary_order(ultra, profile_names, policy_names),
            _pre_registered_summary_order(unknown_push, profile_names, policy_names),
        )


if __name__ == "__main__":
    unittest.main()
