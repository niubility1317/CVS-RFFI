import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2OldProtectedUnknownConfirmCiEvalTest(unittest.TestCase):
    def test_policy_names_parse_all(self):
        from phase2_old_protected_unknown_confirm_ci_eval import POLICIES, _parse_policy_names

        self.assertEqual(_parse_policy_names("all"), [policy.name for policy in POLICIES])

    def test_unknown_policy_lookup_rejects_bad_name(self):
        from phase2_old_protected_unknown_confirm_ci_eval import _policy_by_name

        with self.assertRaisesRegex(ValueError, "unknown OPU policy"):
            _policy_by_name("missing")

    def test_joint_score_penalizes_far_and_half_defer(self):
        from phase2_old_protected_unknown_confirm_ci_eval import _joint_score

        low_far = {
            "old_acc": 0.8,
            "seen_new_acc": 0.7,
            "unknown_reject_rate": 0.9,
            "unknown_FAR": 0.05,
            "defer_rate": 0.2,
        }
        high_far = dict(low_far)
        high_far["unknown_FAR"] = 0.8
        self.assertGreater(_joint_score(low_far), _joint_score(high_far))


if __name__ == "__main__":
    unittest.main()
