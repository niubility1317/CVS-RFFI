import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2OprOpuAdapterCiEvalTest(unittest.TestCase):
    def test_default_policy_subset_is_old_preserving(self):
        from phase2_opr_opu_adapter_ci_eval import parse_args

        args = parse_args(["--feature_npz", "features.npz", "--output_dir", "out"])
        self.assertIn("opu_old_preserve", args.policies)
        self.assertIn("opu_old_guarded", args.policies)

    def test_adapter_args_preserve_unknown_exclusion_inputs(self):
        from phase2_opr_opu_adapter_ci_eval import _adapter_args, parse_args

        args = parse_args(
            [
                "--feature_npz",
                "features.npz",
                "--output_dir",
                "out",
                "--adapter_epochs",
                "2",
                "--old_preserve_weight",
                "5",
            ]
        )
        adapted = _adapter_args(args)
        self.assertEqual(adapted.adapter_epochs, 2)
        self.assertEqual(adapted.old_preserve_weight, 5.0)
        self.assertFalse(hasattr(adapted, "target_unknown"))


if __name__ == "__main__":
    unittest.main()
