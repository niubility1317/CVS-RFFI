import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2ProfileGuardedOprOpuCiEvalTest(unittest.TestCase):
    def test_default_profiles_include_base_and_conservative_adapters(self):
        from phase2_profile_guarded_opr_opu_ci_eval import parse_args

        args = parse_args(["--feature_npz", "features.npz", "--output_dir", "out"])
        self.assertIn("base", args.profiles)
        self.assertIn("conservative", args.profiles)
        self.assertIn("known_tight", args.profiles)

    def test_guard_rejects_profile_when_proxy_unknown_surrogate_worsens(self):
        from phase2_profile_guarded_opr_opu_ci_eval import _guard_report, parse_args

        args = parse_args(["--feature_npz", "features.npz", "--output_dir", "out"])
        metrics = {
            "source_proto_acc_before": 1.0,
            "source_proto_acc_after": 1.0,
            "support_proto_acc_before": 1.0,
            "support_proto_acc_after": 1.0,
            "proxy_max_logit_before_mean": 0.4,
            "proxy_max_logit_after_mean": 0.5,
            "mean_source_residual_norm": 0.01,
            "mean_support_residual_norm": 0.01,
            "training_counts": {"target_unknown_training_count": 0},
        }

        report = _guard_report(metrics, args)

        self.assertFalse(report["guard_pass"])
        self.assertFalse(report["proxy_unknown_surrogate_pass"])
        self.assertLess(report["proxy_max_logit_reduction"], 0.0)

    def test_guard_rejects_any_target_unknown_training_count(self):
        from phase2_profile_guarded_opr_opu_ci_eval import _guard_report, parse_args

        args = parse_args(["--feature_npz", "features.npz", "--output_dir", "out"])
        metrics = {
            "source_proto_acc_before": 1.0,
            "source_proto_acc_after": 1.0,
            "support_proto_acc_before": 1.0,
            "support_proto_acc_after": 1.0,
            "proxy_max_logit_before_mean": 0.5,
            "proxy_max_logit_after_mean": 0.4,
            "mean_source_residual_norm": 0.01,
            "mean_support_residual_norm": 0.01,
            "training_counts": {"target_unknown_training_count": 1},
        }

        report = _guard_report(metrics, args)

        self.assertFalse(report["guard_pass"])
        self.assertEqual(report["target_unknown_training_count"], 1)

    def test_select_profile_falls_back_to_base_if_only_base_passes(self):
        from phase2_profile_guarded_opr_opu_ci_eval import _select_profile

        selected = _select_profile(
            [
                {"profile_name": "base", "guard_pass": True, "guard_score": 2.0, "total_fp16_state_bytes": 0},
                {
                    "profile_name": "open_light",
                    "guard_pass": False,
                    "guard_score": 2.2,
                    "total_fp16_state_bytes": 1000,
                },
            ]
        )

        self.assertEqual(selected["selected_profile"], "base")

    def test_select_profile_uses_guard_score_not_profile_kind_first(self):
        from phase2_profile_guarded_opr_opu_ci_eval import _select_profile

        selected = _select_profile(
            [
                {"profile_name": "base", "guard_pass": True, "guard_score": 2.0, "total_fp16_state_bytes": 0},
                {
                    "profile_name": "conservative",
                    "guard_pass": True,
                    "guard_score": 1.9,
                    "total_fp16_state_bytes": 1000,
                },
            ]
        )

        self.assertEqual(selected["selected_profile"], "base")
        self.assertEqual(selected["selection_reason"], "known_and_proxy_surrogate_guard_score")


if __name__ == "__main__":
    unittest.main()
