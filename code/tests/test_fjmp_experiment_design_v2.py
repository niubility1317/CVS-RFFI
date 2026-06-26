import unittest

import torch
import torch.nn.functional as F

from FJMP.frozen_joint_prototype_head import (
    CalibratedFusion,
    compute_relative_harm_metrics,
    do_no_harm_loss,
    margin_preservation_loss,
)
from FJMP.experiment_manifest import build_experiment_manifest
from post_stage_eval import summarize_epoch_records


class FJMPExperimentDesignV2Test(unittest.TestCase):
    def test_relative_harm_metrics_report_changed_rescue_harm_net_and_confidence(self):
        base_logits = torch.tensor(
            [
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
                [0.0, 4.0, 0.0],
            ]
        )
        fused_logits = torch.tensor(
            [
                [0.0, 6.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 6.0],
                [7.0, 0.0, 0.0],
            ]
        )
        y = torch.tensor([0, 1, 1, 0])
        metrics = compute_relative_harm_metrics(
            base_logits,
            fused_logits,
            y,
            accept_proto=torch.tensor([True, False, True, False]),
            ood_reject=torch.tensor([False, True, False, True]),
        )

        self.assertAlmostEqual(float(metrics["changed_pred_rate"]), 0.50)
        self.assertAlmostEqual(float(metrics["rescue_rate"]), 0.25)
        self.assertAlmostEqual(float(metrics["harm_rate"]), 0.25)
        self.assertAlmostEqual(float(metrics["net_gain_rate"]), 0.0)
        self.assertAlmostEqual(float(metrics["proto_accept_rate"]), 0.50)
        self.assertAlmostEqual(float(metrics["ood_reject_rate"]), 0.50)
        self.assertGreater(float(metrics["harm_conf_mean"]), 0.0)

    def test_residual_fusion_centers_proto_logits_and_respects_gate(self):
        fusion = CalibratedFusion(alpha=0.3, mode="residual", learnable=False, eta=0.1, eta_max=0.2)
        base_logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        proto_logits = torch.tensor([[1.0, 2.0, 6.0], [9.0, 0.0, 0.0]])
        fused = fusion(
            base_logits=base_logits,
            proto_logits=proto_logits,
            accept_proto=torch.tensor([True, False]),
        )

        centered = proto_logits[0] - proto_logits[0].mean()
        self.assertTrue(torch.allclose(fused["logits"][0], base_logits[0] + 0.1 * centered, atol=1e-6))
        self.assertTrue(torch.equal(fused["logits"][1], base_logits[1]))
        self.assertAlmostEqual(float(fused["eta"]), 0.1)

    def test_dnh_and_margin_losses_are_zero_when_fused_does_not_hurt_base(self):
        base_logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
        fused_same = base_logits.clone()
        fused_worse = torch.tensor([[0.0, 5.0], [0.0, 5.0]])
        y = torch.tensor([0, 1])

        self.assertAlmostEqual(float(do_no_harm_loss(fused_same, base_logits, y)), 0.0)
        self.assertAlmostEqual(float(margin_preservation_loss(fused_same, base_logits, y)), 0.0)
        self.assertGreater(float(do_no_harm_loss(fused_worse, base_logits, y)), 0.0)
        self.assertGreater(float(margin_preservation_loss(fused_worse, base_logits, y)), 0.0)

    def test_train_fjmp_parser_accepts_v2_documented_arguments(self):
        from FJMP import train_fjmp

        args = train_fjmp.build_arg_parser().parse_args(
            [
                "--baseline_ckpt",
                "runs/base/latest_model.pth",
                "--output_dir",
                "runs/fjmp",
                "--init_zdom_mode",
                "zero",
                "--ce_on",
                "both",
                "--kd_on",
                "proto",
                "--fusion_mode",
                "residual",
                "--eta",
                "0.05",
                "--eta_max",
                "0.10",
                "--lambda_dnh",
                "0.3",
                "--lambda_margin_preserve",
                "0.1",
                "--lambda_proxy_cons",
                "0.2",
                "--proxy_aug",
                "sat07",
                "--enable_conf_gate",
                "true",
                "--gate_type",
                "ood",
                "--save_epoch_metrics_csv",
                "true",
            ]
        )

        self.assertEqual(args.init_zdom_mode, "zero")
        self.assertEqual(args.ce_on, "both")
        self.assertEqual(args.kd_on, "proto")
        self.assertEqual(args.fusion_mode, "residual")
        self.assertAlmostEqual(args.eta, 0.05)
        self.assertAlmostEqual(args.eta_max, 0.10)
        self.assertAlmostEqual(args.lambda_dnh, 0.3)
        self.assertTrue(args.enable_conf_gate)
        self.assertEqual(args.gate_type, "ood")

    def test_manifest_contains_all_documented_layers_and_priority_batches(self):
        manifest = build_experiment_manifest()
        ids = {row["id"] for row in manifest}

        for required_id in [
            "L0-00",
            "L0-05",
            "L1-16",
            "L2-17",
            "L3-24",
            "L4-31",
            "L5-19",
            "L6-15",
            "A11",
            "B17",
            "C19",
            "D23",
            "E17",
            "F19",
        ]:
            self.assertIn(required_id, ids)

        self.assertGreaterEqual(len(manifest), 124)
        self.assertTrue(all("hypothesis" in row and "purpose" in row and "args" in row for row in manifest))

    def test_epoch_summary_keeps_final_best_source_best_proxy_and_best_test_separate(self):
        records = [
            {
                "epoch": 1,
                "exp_id": "X",
                "val_source": 90.0,
                "proxy_val_rx_day": 80.0,
                "unseen_day_unseen_rx": 79.0,
                "harm_rate": 0.010,
                "rescue_rate": 0.020,
            },
            {
                "epoch": 2,
                "exp_id": "X",
                "val_source": 91.0,
                "proxy_val_rx_day": 81.0,
                "unseen_day_unseen_rx": 82.0,
                "harm_rate": 0.015,
                "rescue_rate": 0.020,
            },
            {
                "epoch": 3,
                "exp_id": "X",
                "val_source": 92.0,
                "proxy_val_rx_day": 78.0,
                "unseen_day_unseen_rx": 80.0,
                "harm_rate": 0.030,
                "rescue_rate": 0.010,
            },
        ]

        summary = summarize_epoch_records(
            records,
            proxy_key="proxy_val_rx_day",
            test_key="unseen_day_unseen_rx",
            source_key="val_source",
        )

        self.assertEqual(summary["final_epoch"], 3)
        self.assertAlmostEqual(summary["final_test"], 80.0)
        self.assertEqual(summary["best_source_epoch"], 3)
        self.assertEqual(summary["best_proxy_epoch"], 2)
        self.assertEqual(summary["best_test_epoch"], 2)
        self.assertAlmostEqual(summary["final_minus_best_test"], -2.0)
        self.assertAlmostEqual(summary["harm_rate"], 0.030)
        self.assertAlmostEqual(summary["rescue_rate"], 0.010)


if __name__ == "__main__":
    unittest.main()
