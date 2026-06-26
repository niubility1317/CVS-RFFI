import math
import unittest
from pathlib import Path

import torch

from FJMP.experiment_manifest import build_experiment_manifest
from FJMP.fjmp_v2_losses import (
    angular_diversity_loss,
    assignment_entropy_loss,
    boundary_trimmed_ce_loss,
    delta_ratio_loss,
    get_fjmp_v2_stage_weights,
    logit_residual_loss,
    prototype_usage_loss,
    selective_kd_loss,
)
from FJMP.fjmp_v2_proto_head import SafeResidualProtoHead
from FJMP.prototype_metrics import (
    compute_dead_proto_rate,
    compute_delta_stats,
    compute_harm_rescue,
    compute_proto_pairwise_cos,
    compute_rho_stats,
    compute_usage_entropy,
)


ROOT = Path(__file__).resolve().parents[1]


class FJMPV2SafeResidualTest(unittest.TestCase):
    def test_safe_residual_head_anchors_base_logits_and_blocks_base_gradients(self):
        torch.manual_seed(11)
        head = SafeResidualProtoHead(
            in_dim=5,
            proto_dim=7,
            num_classes=4,
            K=3,
            rho_max=0.15,
            delta_clip=0.25,
            proto_dropout=0.0,
        )
        z_id = torch.randn(6, 5, requires_grad=True)
        base_logits = torch.randn(6, 4, requires_grad=True)

        fused, aux = head(z_id, base_logits)
        loss = fused.pow(2).mean()
        loss.backward()

        self.assertEqual(fused.shape, (6, 4))
        self.assertEqual(aux["proto_logits"].shape, (6, 4))
        self.assertEqual(aux["sim"].shape, (6, 4, 3))
        self.assertEqual(aux["prototypes"].shape, (4, 3, 7))
        self.assertTrue(torch.all(aux["rho"] >= 0.0))
        self.assertLessEqual(float(aux["rho"].max()), 0.15 + 1e-6)
        self.assertLessEqual(float(aux["delta_logits"].abs().max()), 0.25 + 1e-6)
        expected = base_logits.detach() + aux["rho"] * aux["delta_logits"]
        self.assertTrue(torch.allclose(fused, expected, atol=1e-6))
        self.assertIsNone(base_logits.grad)
        self.assertIsNotNone(z_id.grad)

    def test_boundary_ce_and_selective_kd_weight_only_base_safe_samples(self):
        base_logits = torch.tensor(
            [
                [8.0, 0.0, -1.0],
                [2.0, 1.8, 0.0],
                [0.0, 4.0, 0.0],
            ]
        )
        fused_logits = torch.tensor(
            [
                [7.5, 0.4, -1.0],
                [1.5, 2.6, 0.0],
                [3.0, 0.0, 0.0],
            ],
            requires_grad=True,
        )
        y = torch.tensor([0, 0, 0])

        ce, w_ce = boundary_trimmed_ce_loss(fused_logits, base_logits, y, return_weight=True)
        kd, w_kd = selective_kd_loss(fused_logits, base_logits, y, return_weight=True)

        self.assertTrue(torch.isfinite(ce))
        self.assertLess(float(w_ce[0]), float(w_ce[1]))
        self.assertGreater(float(w_ce[2]), 0.0)
        self.assertGreater(float(w_kd[0]), 0.99)
        self.assertGreater(float(w_kd[1]), 0.0)
        self.assertEqual(float(w_kd[2]), 0.0)
        kd.backward()
        self.assertIsNotNone(fused_logits.grad)

    def test_v2_proto_delta_and_metric_functions_are_finite_and_named(self):
        torch.manual_seed(5)
        sim = torch.randn(8, 3, 3)
        y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        prototypes = torch.randn(3, 3, 6)
        base_logits = torch.randn(8, 3)
        delta_logits = torch.randn(8, 3) * 0.2
        rho = torch.rand(8, 1) * 0.15

        loss_usage, usage, q = prototype_usage_loss(sim, y, K=3)
        losses = [
            angular_diversity_loss(prototypes),
            loss_usage,
            assignment_entropy_loss(q, K=3),
            delta_ratio_loss(rho * delta_logits, base_logits)[0],
            logit_residual_loss(delta_logits),
        ]
        for loss in losses:
            self.assertTrue(torch.isfinite(loss))

        pairwise = compute_proto_pairwise_cos(prototypes)
        usage_entropy = compute_usage_entropy(usage)
        dead_rate = compute_dead_proto_rate(usage, K=3)
        delta_stats = compute_delta_stats(rho * delta_logits, base_logits)
        rho_stats = compute_rho_stats(rho)
        harm = compute_harm_rescue(base_logits, base_logits + rho * delta_logits, y)

        self.assertIn("proto_pairwise_cos_mean", pairwise)
        self.assertIn("usage_entropy_mean", usage_entropy)
        self.assertGreaterEqual(float(dead_rate["dead_proto_rate"]), 0.0)
        self.assertIn("delta_ratio_p95", delta_stats)
        self.assertIn("rho_p95", rho_stats)
        self.assertIn("net_gain", harm)

    def test_stage_weights_match_documented_short_train_schedule(self):
        stage1 = get_fjmp_v2_stage_weights(1)
        stage2 = get_fjmp_v2_stage_weights(6)
        stage3 = get_fjmp_v2_stage_weights(10)

        self.assertEqual(stage1.stage, "stage1")
        self.assertAlmostEqual(stage1.rho_max, 0.03)
        self.assertEqual(stage2.stage, "stage2")
        self.assertGreater(stage2.rho_max, stage1.rho_max)
        self.assertLessEqual(stage2.rho_max, 0.15)
        self.assertEqual(stage3.stage, "stage3")
        self.assertTrue(stage3.freeze_prototypes)
        self.assertAlmostEqual(stage3.ce_trim, 0.2)

    def test_manifest_config_and_parser_expose_fjmp_v2_main_and_ablations(self):
        from FJMP import train_fjmp

        args = train_fjmp.build_arg_parser().parse_args(
            [
                "--baseline_ckpt",
                "runs/base/latest_model.pth",
                "--output_dir",
                "runs/fjmp_v2/main",
                "--fjmp_version",
                "v2",
            ]
        )

        self.assertEqual(args.fjmp_version, "v2")
        self.assertEqual(args.model_name, "FJMP_V2_K3_SAFE_RESIDUAL")
        self.assertEqual(args.num_prototypes, 3)
        self.assertEqual(args.proto_dim, 256)
        self.assertEqual(args.zdom_mode, "zero")
        self.assertEqual(args.epochs, 12)
        self.assertAlmostEqual(args.rho_max, 0.15)
        self.assertAlmostEqual(args.delta_clip, 3.0)
        self.assertAlmostEqual(args.lambda_ce_trim, 1.0)
        self.assertAlmostEqual(args.lambda_kd_selective, 0.3)

        manifest = build_experiment_manifest(["FJMP-V2"])
        ids = {row["id"] for row in manifest}
        for required in ["V2-01", "V2-02", "V2-03", "V2-04", "V2-05", "V2-06"]:
            self.assertIn(required, ids)

        main = next(row for row in manifest if row["id"] == "V2-04")
        self.assertEqual(main["args"]["fjmp_version"], "v2")
        self.assertEqual(main["args"]["model_name"], "FJMP_V2_K3_SAFE_RESIDUAL")
        for removed in [
            "lambda_margin_preserve",
            "lambda_gate_view_gap",
            "lambda_worst_domain_view",
            "lambda_proto_sgv",
            "lambda_sgv_safe",
            "lambda_sgv_margin",
            "lambda_pres_clean",
            "lambda_pres_sat",
            "lambda_harm",
        ]:
            self.assertEqual(main["args"].get(removed, 0.0), 0.0)

        config_text = (ROOT / "configs" / "fjmp_v2.yaml").read_text(encoding="utf-8")
        self.assertIn("FJMP_V2_K3_SAFE_RESIDUAL", config_text)
        self.assertIn("boundary_trimmed_ce", config_text)
        self.assertIn("selective_kd", config_text)
        self.assertIn("gate_view_gap", config_text)


if __name__ == "__main__":
    unittest.main()
