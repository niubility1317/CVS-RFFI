import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


class TinyFrozenBase(nn.Module):
    def __init__(self, num_classes=4, feature_dim=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(2, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_feature(self, x):
        return self.proj(self.pool(x).squeeze(-1))

    def classify(self, x, feat=None):
        if feat is None:
            feat = self.extract_feature(x)
        return self.classifier(feat)


def sample_meta(batch_size=3):
    return {
        "orbit": torch.tensor([0, 1, 2][:batch_size]),
        "state": torch.tensor([0, 1, 2][:batch_size]),
        "weather": torch.tensor([0, 2, 3][:batch_size]),
        "theta_deg": torch.tensor([20.0, 45.0, 80.0][:batch_size]),
        "h_km": torch.tensor([700.0, 12000.0, 35786.0][:batch_size]),
        "d_km": torch.tensor([1600.0, 18000.0, 38000.0][:batch_size]),
        "pl_db": torch.tensor([165.0, 180.0, 190.0][:batch_size]),
        "fD_hz": torch.tensor([12000.0, -8000.0, 200.0][:batch_size]),
        "cfo_hz": torch.tensor([120.0, -40.0, 0.0][:batch_size]),
        "snr_db": torch.tensor([12.0, 20.0, 28.0][:batch_size]),
        "K_db": torch.tensor([3.0, 8.0, 15.0][:batch_size]),
    }


class PhyConReconTest(unittest.TestCase):
    def test_unet_default_shape_and_parameter_budget(self):
        from SGC.recon.cx_unet_1d import CxResUNet1D, count_parameters

        torch.manual_seed(1)
        model = CxResUNet1D()
        x_t = torch.randn(2, 2, 256)
        y = torch.randn(2, 2, 256)
        c = torch.randn(2, 24)
        t = torch.tensor([3, 17])

        out = model(x_t=x_t, y=y, t=t, c=c)

        self.assertEqual(out.shape, (2, 2, 256))
        self.assertTrue(torch.logical_and(out >= -1.0, out <= 1.0).all())
        self.assertGreaterEqual(count_parameters(model), 180_000)
        self.assertLessEqual(count_parameters(model), 250_000)

    def test_condition_encoder_uses_meta_or_proxy_without_tx_label(self):
        from SGC.recon.condition_encoder import PhyConditionEncoder, estimate_phy_proxy, normalize_sat_meta

        y = torch.randn(3, 2, 256)
        encoder = PhyConditionEncoder(out_dim=24)
        meta = sample_meta()
        meta["tx"] = torch.tensor([1, 2, 3])

        cond_from_meta = encoder(normalize_sat_meta(meta, device=y.device))
        cond_from_proxy = encoder.from_iq_proxy(y)
        proxy = estimate_phy_proxy(y)

        self.assertEqual(cond_from_meta.shape, (3, 24))
        self.assertEqual(cond_from_proxy.shape, (3, 24))
        self.assertEqual(proxy.shape[0], 3)
        self.assertTrue(torch.isfinite(cond_from_meta).all())
        self.assertTrue(torch.isfinite(cond_from_proxy).all())

    def test_residual_gate_and_bounded_residual_are_safe(self):
        from SGC.recon.complex_ops import apply_bounded_residual, residual_ratio
        from SGC.recon.residual_gate import ResidualSafetyGate

        y = torch.randn(4, 2, 128)
        delta = torch.randn_like(y) * 20.0
        c = torch.randn(4, 24)
        gate = ResidualSafetyGate(cond_dim=24)(y, c)
        x_hat = apply_bounded_residual(y, delta, gate, rho=0.05)
        ratio = residual_ratio(x_hat, y)

        self.assertEqual(gate.shape, (4, 1, 1))
        self.assertTrue(torch.logical_and(gate >= 0, gate <= 1).all())
        self.assertLessEqual(float((x_hat - y).detach().abs().max()), 0.0501)
        self.assertTrue(torch.isfinite(ratio).all())

    def test_losses_are_finite_and_keep_base_frozen(self):
        from SGC.recon.identity_losses import identity_preservation_loss
        from SGC.recon.stft_losses import stft_mag_phase_loss

        torch.manual_seed(5)
        base = TinyFrozenBase()
        for param in base.parameters():
            param.requires_grad = False
        x_clean = torch.randn(5, 2, 128)
        x_hat = (x_clean + 0.03 * torch.randn_like(x_clean)).requires_grad_()
        labels = torch.tensor([0, 1, 2, 3, 1])

        loss_id, logs = identity_preservation_loss(base, x_hat, x_clean, labels)
        loss_tf = stft_mag_phase_loss(x_hat, x_clean, n_fft=32, hop_length=8, win_length=32)
        (loss_id + loss_tf).backward()

        self.assertTrue(torch.isfinite(loss_id))
        self.assertTrue(torch.isfinite(loss_tf))
        self.assertIn("id_feature_cos", logs)
        self.assertFalse(any(p.grad is not None for p in base.parameters()))

    def test_differentiable_sat_channel_backprops_to_input(self):
        from SGC.recon.channel_losses import channel_consistency_loss
        from SGC.recon.diff_sat_channel import DifferentiableSatChannel

        torch.manual_seed(7)
        x_hat = torch.randn(3, 2, 64, requires_grad=True)
        y_sat = torch.randn(3, 2, 64)
        phi = sample_meta()
        channel = DifferentiableSatChannel(fs_hz=25e6, fc_hz=2.462e9)

        y_reproj = channel(x_hat, phi)
        loss, logs = channel_consistency_loss(channel, x_hat, y_sat, phi, n_fft=32, hop_length=8, win_length=32)
        loss.backward()

        self.assertEqual(y_reproj.shape, x_hat.shape)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_chan_time", logs)
        self.assertIsNotNone(x_hat.grad)
        self.assertGreater(float(x_hat.grad.abs().sum()), 0.0)

    def test_resdiff_and_consistency_expose_correction_contract(self):
        from SGC.recon.cx_consistency import CxConsistency
        from SGC.recon.cx_resdiff import CxResDiff

        torch.manual_seed(9)
        y = torch.randn(2, 2, 256)
        meta = sample_meta(batch_size=2)
        diff = CxResDiff()
        cm = CxConsistency()

        out_diff = diff.correct(y, meta=meta, rho=0.05)
        out_cm = cm.correct(y, meta=meta, steps=2, rho=0.05)

        for out in (out_diff, out_cm):
            self.assertIn("x_hat", out)
            self.assertIn("delta", out)
            self.assertIn("gate", out)
            self.assertIn("residual_ratio", out)
            self.assertEqual(out["x_hat"].shape, y.shape)
            self.assertLessEqual(float((out["x_hat"] - y).detach().abs().max()), 0.0501)

    def test_configs_and_scripts_expose_dry_run_contracts(self):
        from post_stage_common import load_yaml_or_json
        from SGC.distill_recon_consistency import build_arg_parser as build_distill_parser
        from SGC.eval_recon_frontend import build_arg_parser as build_eval_parser
        from SGC.eval_recon_sgc_joint import build_arg_parser as build_joint_eval_parser
        from SGC.train_recon_diffusion import build_arg_parser as build_diff_parser
        from SGC.train_recon_sgc_joint import build_arg_parser as build_joint_parser

        config_dir = CODE / "SGC" / "configs"
        diff_cfg = load_yaml_or_json(str(config_dir / "recon_cxresdiff_020m.yaml"))
        cm_cfg = load_yaml_or_json(str(config_dir / "recon_cxconsistency_020m.yaml"))
        joint_cfg = load_yaml_or_json(str(config_dir / "recon_sgc_joint.yaml"))

        self.assertEqual(diff_cfg["model"]["type"], "cx_residual_consistency_unet_1d")
        self.assertEqual(cm_cfg["consistency"]["steps_eval"], [1, 2, 4])
        self.assertAlmostEqual(float(joint_cfg["joint_finetune"]["rho_max"]), 0.10)

        for parser_fn in (build_diff_parser, build_distill_parser, build_eval_parser, build_joint_parser):
            parser = parser_fn()
            args = parser.parse_args(["--teacher_ckpt", "dummy.pth", "--output_dir", "out", "--dry_run"])
            self.assertTrue(args.dry_run)

        joint_eval_args = build_joint_eval_parser().parse_args(
            [
                "--teacher_ckpt",
                "teacher.pth",
                "--joint_ckpt",
                "latest_recon_sgc_joint.pth",
                "--output_dir",
                "out",
                "--dry_run",
            ]
        )
        self.assertEqual(joint_eval_args.joint_ckpt, "latest_recon_sgc_joint.pth")
        self.assertTrue(joint_eval_args.dry_run)

    def test_joint_epoch_report_can_include_shared_test_summary(self):
        from SGC.train_recon_sgc_joint import format_joint_epoch_report
        from training_test_eval import TrainingTestEvalResult

        report = format_joint_epoch_report(
            epoch=3,
            epochs=20,
            lr_recon=2e-5,
            lr_sgc=5e-5,
            epoch_time_s=12.3,
            rho=0.1,
            r_max=0.1,
            logs={"train/loss_total": 1.0},
            latest_path="latest.pth",
            best_path="best.pth",
            eval_result=TrainingTestEvalResult(
                val_stats={"tx_acc": 91.25, "dom_acc": float("nan"), "tx_correct": 73, "tx_total": 80},
                named_test_stats={},
                test_stats={"tx_acc": 90.08, "tx_correct": 183762, "tx_total": 204000},
                lines=[
                    "[TEST]  overall_tx=90.08% (183762/204000)",
                    "[TEST-SPLIT]",
                    "          unseen_day_unseen_rx(days=['2021_03_15'], rxs=[7]): tx=86.00% (10320/12000)",
                ],
            ),
        )

        self.assertIn("[VAL]   tx=91.25%", report)
        self.assertIn("[TEST]  overall_tx=90.08% (183762/204000)", report)
        self.assertIn("[TEST-SPLIT]", report)
        self.assertIn("unseen_day_unseen_rx", report)


if __name__ == "__main__":
    unittest.main()
