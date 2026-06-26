import unittest
from pathlib import Path
import sys

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


class Cen31DistillLossTest(unittest.TestCase):
    def test_feature_and_relation_kd_are_zero_when_no_teacher_sample_is_reliable(self):
        from train_cen31_distill import FeatureProjector, feature_kd_loss, relation_kd_loss

        projector = FeatureProjector(4, 4)
        student = torch.randn(3, 4)
        teacher = torch.randn(3, 4)
        mask = torch.zeros(3, dtype=torch.bool)

        self.assertEqual(float(feature_kd_loss(student, teacher, projector, mask).item()), 0.0)
        self.assertEqual(float(relation_kd_loss(student, teacher, projector, mask).item()), 0.0)

    def test_teacher_cli_args_do_not_override_checkpoint_architecture(self):
        from types import SimpleNamespace

        from train_cen31_distill import build_teacher_cli_args

        args = SimpleNamespace(
            dataset="wisig",
            num_classes=16,
            batch_size=8,
            eval_batch_size=16,
            device="cuda:0",
            seed=1337,
            sample_rate_hz=25e6,
            model_variant="lite_h",
            branch_ablation="time_only",
            domain_branch_ablation="time_only",
            arch_family="cvcnn",
        )

        teacher_args = build_teacher_cli_args(args)

        self.assertEqual(teacher_args.dataset, "wisig")
        self.assertEqual(teacher_args.num_classes, 16)
        self.assertEqual(teacher_args.sample_rate_hz, 25e6)
        for student_only_key in ("model_variant", "branch_ablation", "domain_branch_ablation", "arch_family"):
            self.assertFalse(hasattr(teacher_args, student_only_key), student_only_key)

    def test_sat_view_distill_loss_combines_clean_teacher_targets(self):
        from types import SimpleNamespace

        from train_cen31_distill import FeatureProjector, compute_sat_view_distill_losses

        projector = FeatureProjector(4, 4)
        ce = torch.nn.CrossEntropyLoss()
        labels = torch.tensor([0, 1])
        teacher_out = {
            "tx_logits": torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
            "z_id": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        }
        student_out = {
            "tx_logits": torch.tensor([[3.0, 0.2], [0.3, 2.8]]),
            "z_id": torch.tensor([[0.9, 0.1, 0.0, 0.0], [0.1, 0.8, 0.0, 0.0]]),
        }
        args = SimpleNamespace(
            kd_temperature=3.0,
            lambda_sat_view_ce=1.0,
            lambda_sat_view_kd=0.5,
            lambda_sat_view_feature_kd=0.25,
            lambda_sat_view_relation_kd=0.125,
        )

        losses = compute_sat_view_distill_losses(
            student_out,
            teacher_out,
            labels,
            projector,
            torch.ones(2, dtype=torch.bool),
            ce,
            args,
        )

        expected = (
            args.lambda_sat_view_ce * losses["ce"]
            + args.lambda_sat_view_kd * losses["kd"]
            + args.lambda_sat_view_feature_kd * losses["feat"]
            + args.lambda_sat_view_relation_kd * losses["rel"]
        )
        self.assertGreater(float(losses["loss"].item()), 0.0)
        self.assertTrue(torch.allclose(losses["loss"], expected))

    def test_sat_view_distill_loss_can_include_group_ce(self):
        from types import SimpleNamespace

        from cvsrffi.losses import SmoothGroupDROState
        from train_cen31_distill import FeatureProjector, compute_sat_view_distill_losses

        projector = FeatureProjector(4, 4)
        ce = torch.nn.CrossEntropyLoss()
        labels = torch.tensor([0, 1, 0, 1])
        teacher_out = {
            "tx_logits": torch.tensor([[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]]),
            "z_id": torch.eye(4),
        }
        student_out = {
            "tx_logits": torch.tensor([[2.5, 0.1], [0.2, 2.5], [0.3, 1.8], [1.5, 0.4]]),
            "z_id": torch.eye(4) * 0.8,
        }
        args = SimpleNamespace(
            kd_temperature=3.0,
            lambda_sat_view_ce=0.0,
            lambda_sat_view_kd=0.0,
            lambda_sat_view_feature_kd=0.0,
            lambda_sat_view_relation_kd=0.0,
            lambda_sat_view_group_ce=1.0,
            group_ce_mode="hard",
            label_smoothing=0.0,
            group_ce_top_frac=0.5,
            group_ce_min_domains=2,
            groupdro_tau=0.5,
            groupdro_cap=0.65,
        )

        losses = compute_sat_view_distill_losses(
            student_out,
            teacher_out,
            labels,
            projector,
            torch.ones(4, dtype=torch.bool),
            ce,
            args,
            domain_labels=torch.tensor([0, 0, 1, 1]),
            groupdro_state=SmoothGroupDROState(),
        )

        self.assertGreater(float(losses["group_ce"].item()), 0.0)
        self.assertTrue(torch.allclose(losses["loss"], losses["group_ce"]))

    def test_balanced_selection_respects_clean_guard_drop(self):
        from train_cen31_distill import clean_guard_allows_balanced_update, compute_balanced_selection_score
        from types import SimpleNamespace

        args = SimpleNamespace(
            best_clean_weight=0.55,
            best_receiver_floor_weight=0.10,
            best_sat_mean_weight=0.25,
            best_sat_floor_weight=0.10,
        )

        score = compute_balanced_selection_score(82.0, 65.0, 49.0, 47.0, args)

        self.assertAlmostEqual(score, 68.55)
        self.assertTrue(clean_guard_allows_balanced_update(82.0, 82.5, 1.0))
        self.assertFalse(clean_guard_allows_balanced_update(80.9, 82.5, 1.0))

    def test_sat_loss_ramp_scales_sat_terms_only(self):
        from types import SimpleNamespace

        from train_cen31_distill import sat_view_loss_scale

        args = SimpleNamespace(sat_view_loss_start_epoch=10, sat_view_loss_ramp_epochs=5)

        self.assertEqual(sat_view_loss_scale(9, args), 0.0)
        self.assertAlmostEqual(sat_view_loss_scale(10, args), 0.2)
        self.assertAlmostEqual(sat_view_loss_scale(12, args), 0.6)
        self.assertEqual(sat_view_loss_scale(14, args), 1.0)


if __name__ == "__main__":
    unittest.main()
