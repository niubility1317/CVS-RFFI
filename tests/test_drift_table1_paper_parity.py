import argparse
import subprocess
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class DriftTable1PaperParityTest(unittest.TestCase):
    def test_compact_receiver_targets_use_only_training_receiver_domains(self):
        from baselines.common.paper_protocol import compact_receiver_targets, train_receiver_count

        split_info = {"train_rxs_idx": [0, 2, 10]}
        raw = torch.tensor([0, 2, 10, 2, 0], dtype=torch.long)

        self.assertEqual(train_receiver_count(split_info, fallback=12), 3)
        self.assertTrue(torch.equal(compact_receiver_targets(raw, split_info), torch.tensor([0, 1, 2, 1, 0])))

        with self.assertRaises(ValueError):
            compact_receiver_targets(torch.tensor([1], dtype=torch.long), split_info)

    def test_drift_default_mse_is_raw_negative_mse(self):
        from baselines.drift import train_cvs
        from baselines.drift.losses import negative_mse_separation

        parser = argparse.ArgumentParser()
        train_cvs.add_drift_method_args(parser)
        args = parser.parse_args([])

        self.assertFalse(args.normalize_features_for_mse)
        self.assertEqual(args.grl_schedule, "constant")
        self.assertEqual(args.domain_discriminator_layers, 2)
        self.assertAlmostEqual(args.grl_coeff, 1.0)
        self.assertEqual(args.center_mode, "ema")
        self.assertAlmostEqual(args.center_momentum, 0.95)
        z_tx = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
        z_rx = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        raw_loss = negative_mse_separation(z_tx, z_rx)
        normalized_loss = negative_mse_separation(z_tx, z_rx, normalize=True)
        self.assertAlmostEqual(float(raw_loss), -6.5, places=6)
        self.assertNotAlmostEqual(float(raw_loss), float(normalized_loss), places=6)

    def test_drift_ema_center_memory_uses_cross_batch_centers(self):
        from baselines.drift.losses import ReceiverCenterEMA, receiver_style_transfer_center_loss

        memory = ReceiverCenterEMA(num_receivers=2, feature_dim=2, momentum=0.5)
        first_z = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
        second_z = torch.tensor([[3.0, 0.0], [5.0, 0.0]])
        labels = torch.tensor([0, 0], dtype=torch.long)

        first_loss = memory(first_z, labels)
        second_loss = memory(second_z, labels)
        batch_local_second = receiver_style_transfer_center_loss(second_z, labels)

        self.assertAlmostEqual(float(first_loss), 1.0, places=6)
        self.assertAlmostEqual(float(batch_local_second), 1.0, places=6)
        self.assertAlmostEqual(float(second_loss), 10.0, places=6)
        self.assertAlmostEqual(float(memory.centers[0, 0]), 2.5, places=6)
        self.assertEqual(int(memory.initialized[0]), 1)

    def test_drift_domain_discriminator_defaults_to_two_layer_fc(self):
        from torch import nn

        from baselines.drift.model import DRIFTModel

        paper_model = DRIFTModel(num_tx=6, num_rx=3, domain_discriminator_layers=2)
        legacy_model = DRIFTModel(num_tx=6, num_rx=3, domain_discriminator_layers=3)

        paper_linear_layers = [m for m in paper_model.domain_discriminator.modules() if isinstance(m, nn.Linear)]
        legacy_linear_layers = [m for m in legacy_model.domain_discriminator.modules() if isinstance(m, nn.Linear)]
        self.assertEqual(len(paper_linear_layers), 2)
        self.assertEqual(len(legacy_linear_layers), 3)

    def test_grl_coeff_and_lambda_grl_do_not_square_scale_feature_gradient(self):
        from baselines.common.grl import gradient_reverse

        feature_a = torch.tensor([[1.0, -2.0]], requires_grad=True)
        feature_b = feature_a.detach().clone().requires_grad_(True)
        feature_c = feature_a.detach().clone().requires_grad_(True)
        weight = torch.tensor([[0.25], [-0.5]])

        lambda_grl_loss_weight = 2.0
        grl_coeff = 3.0

        (lambda_grl_loss_weight * gradient_reverse(feature_a, grl_coeff).matmul(weight).sum()).backward()
        (lambda_grl_loss_weight * gradient_reverse(feature_b, 1.0).matmul(weight).sum()).backward()
        (1.0 * gradient_reverse(feature_c, grl_coeff).matmul(weight).sum()).backward()

        base_grad = -weight.flatten()
        self.assertTrue(torch.allclose(feature_a.grad.flatten(), lambda_grl_loss_weight * grl_coeff * base_grad))
        self.assertTrue(torch.allclose(feature_b.grad.flatten(), lambda_grl_loss_weight * base_grad))
        self.assertTrue(torch.allclose(feature_c.grad.flatten(), grl_coeff * base_grad))

    def test_riei_mi_loss_uses_absolute_normalized_inner_product(self):
        from baselines.riei_fd.losses import mutual_independence_loss

        z_e = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        z_r = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])

        self.assertAlmostEqual(float(mutual_independence_loss(z_e, z_r)), 0.5, places=6)

    def test_riei_sum_reductions_match_paper_formula_scale(self):
        from baselines.riei_fd.losses import entropy_from_logits, mutual_independence_loss, riei_total_loss

        z_e = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        z_r = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])
        logits = torch.zeros(2, 2)
        outputs = {
            "z_e": z_e,
            "z_r": z_r,
            "emitter_logits": torch.tensor([[4.0, -4.0], [-4.0, 4.0]]),
            "receiver_logits": torch.tensor([[4.0, -4.0], [-4.0, 4.0]]),
            "cross_emitter_logits": logits,
            "cross_receiver_logits": logits,
        }
        labels = torch.tensor([0, 1], dtype=torch.long)

        self.assertAlmostEqual(float(mutual_independence_loss(z_e, z_r, reduction="mean")), 0.5, places=6)
        self.assertAlmostEqual(float(mutual_independence_loss(z_e, z_r, reduction="sum")), 1.0, places=6)
        self.assertAlmostEqual(float(entropy_from_logits(logits, reduction="sum")), 2.0 * torch.log(torch.tensor(2.0)).item(), places=6)
        mean_loss = riei_total_loss(outputs, labels, labels, ce_reduction="mean", mi_reduction="mean", ie_reduction="mean")
        sum_loss = riei_total_loss(outputs, labels, labels, ce_reduction="sum", mi_reduction="sum", ie_reduction="sum")

        self.assertAlmostEqual(float(sum_loss["loss_mi"]), 2.0 * float(mean_loss["loss_mi"]), places=6)
        self.assertAlmostEqual(float(sum_loss["loss_ie"]), 2.0 * float(mean_loss["loss_ie"]), places=6)
        self.assertAlmostEqual(float(sum_loss["loss_ce"]), 2.0 * float(mean_loss["loss_ce"]), places=6)

    def test_queue_exposes_drift_table1_paper_methods(self):
        proc = subprocess.run(
            [
                "bash",
                "-lc",
                "WISIG_PROTOCOL=drift_day1 RUN_ROOT=./tmp_runs LOG_ROOT=./tmp_logs "
                "bash run_wisig_paper_scope_queue.sh --dry-run --methods riei_fd,drift,cvsrffi_cen_a31 --gpu-ids 0,1",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        out = proc.stdout + proc.stderr
        self.assertNotIn("baselines.paper_resnet.train", out)
        self.assertNotIn("--paper_method erm", out)
        self.assertNotIn("--paper_method dann", out)
        self.assertNotIn("--paper_method mtl", out)
        self.assertIn("baselines.riei_fd.train", out)
        self.assertIn("baselines.drift.train", out)
        self.assertIn("CEN_A31_a22_satboost_ce1p28_stack", out)
        self.assertIn("--paper_eval_last_n 5", out)
        self.assertIn("--paper_eval_name drift_last5", out)
        self.assertIn("--paper_eval_name riei_last5", out)
        self.assertNotIn("--paper_eval_name riei_last10", out)
        self.assertIn("--no-normalize_features_for_mse", out)
        self.assertIn("--grl_coeff 1.0", out)
        self.assertIn("--domain_discriminator_layers 2", out)
        self.assertIn("--center_mode ema", out)
        self.assertIn("--center_momentum 0.95", out)
        self.assertIn("--ce_reduction sum", out)
        self.assertIn("--mi_reduction sum", out)
        self.assertIn("--ie_reduction sum", out)
        self.assertIn("--grl_schedule constant", out)
        self.assertIn("--group_ce_min_domains 3", out)
        self.assertIn("--fishr_min_domains 3", out)

    def test_cvs_ratio_paper_methods_default_to_last_epoch_eval(self):
        proc = subprocess.run(
            [
                "bash",
                "-lc",
                "RUN_ID=tmp_cvs_paper_last1 DRY_RUN=1 TRAIN_RATIOS=0.1 "
                "METHODS=riei_paper_nosat,drift_paper_nosat,riei_paper_sat,drift_paper_sat "
                "GPU_IDS=0,1,2,3 SAT_TRAIN_AUG=1 PYTHON_BIN=python3 "
                "bash run_cvs_fixed_riei_drift_ratio_sweep.sh",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        out = proc.stdout + proc.stderr
        self.assertEqual(out.count("--paper_eval_last_n 1"), 4)
        self.assertIn("--paper_eval_name cvs_riei_paper_nosat_last1", out)
        self.assertIn("--paper_eval_name cvs_drift_paper_nosat_last1", out)
        self.assertIn("--paper_eval_name cvs_riei_paper_sat_last1", out)
        self.assertIn("--paper_eval_name cvs_drift_paper_sat_last1", out)
        self.assertIn("--wisig_split_strategy random", out)
        self.assertIn("--wisig_cap_strategy random", out)
        self.assertNotIn("--paper_eval_last_n 0", out)

    def test_cvs_ratio_launcher_explains_original_vs_fixed_labels(self):
        help_proc = subprocess.run(
            ["bash", "-lc", "bash run_cvs_fixed_riei_drift_ratio_sweep.sh --help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        self.assertEqual(help_proc.returncode, 0, help_proc.stdout + help_proc.stderr)
        help_out = help_proc.stdout + help_proc.stderr
        self.assertIn("riei_paper_* / drift_paper_*  = original paper method tags.", help_out)
        self.assertIn("riei_fixed_* / drift_fixed_*  = optimized implementation tags.", help_out)
        self.assertIn("RIEI fixed keeps CE+MI-IE and adds lambda_feature_norm=0.0001.", help_out)
        self.assertIn("DRIFT fixed keeps raw negative-MSE separation and adds mse_cap=4000.", help_out)

        dry_proc = subprocess.run(
            [
                "bash",
                "-lc",
                "RUN_ID=tmp_cvs_method_label_notes DRY_RUN=1 TRAIN_RATIOS=0.1 "
                "METHODS=riei_paper_nosat,riei_fixed_sat,drift_paper_nosat,drift_fixed_sat "
                "GPU_IDS=0,1,2,3 SAT_TRAIN_AUG=1 PYTHON_BIN=python3 "
                "bash run_cvs_fixed_riei_drift_ratio_sweep.sh",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        self.assertEqual(dry_proc.returncode, 0, dry_proc.stdout + dry_proc.stderr)
        out = dry_proc.stdout + dry_proc.stderr
        self.assertIn("method_label_convention paper=original_paper fixed=fix_optimized", out)
        self.assertIn("fixed_delta riei=lambda_feature_norm=0.0001 drift=mse_cap=4000", out)
        self.assertIn("version=original_paper method_tag=riei_paper_*", out)
        self.assertIn("version=original_paper method_tag=drift_paper_*", out)
        self.assertIn("version=fix_optimized method_tag=riei_fixed_*", out)
        self.assertIn("version=fix_optimized method_tag=drift_fixed_*", out)

    def test_queue_cli_run_roots_are_not_reset_to_protocol_defaults(self):
        proc = subprocess.run(
            [
                "bash",
                "-lc",
                "WISIG_PROTOCOL=riei_original "
                "bash run_wisig_paper_scope_queue.sh --dry-run --methods riei_fd --gpu-ids 0 "
                "--wisig-pkl /tmp/fake.pkl --run-root ./tmp_cli_runs --log-root ./tmp_cli_logs",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        out = proc.stdout + proc.stderr
        self.assertIn("run_root=./tmp_cli_runs", out)
        self.assertIn("log_root=./tmp_cli_logs", out)
        self.assertNotIn("runs/wisig_paper_scope_riei_original", out)
        self.assertIn("--paper_eval_last_n 10", out)
        self.assertIn("--paper_eval_name riei_last10", out)
        self.assertNotIn("--paper_eval_name riei_last5", out)

    def test_top_level_optional_riei_drift_day1_does_not_force_riei_last10(self):
        proc = subprocess.run(
            [
                "bash",
                "-lc",
                "PYTHON_BIN=/mnt/c/Users/lh594/.conda/envs/ssr-gpu/python.exe "
                "bash code/scripts/launch_paper_repro_20260605_145347_riei_drift.sh "
                "--dry-run --suites drift_day1 --with-riei-drift-day1 --gpu-ids 0",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        out = proc.stdout + proc.stderr
        self.assertIn("METHODS=drift,riei_fd", out.replace("\\,", ","))
        self.assertIn("WISIG_PROTOCOL=drift_day1", out)
        self.assertIn("DRIFT_PAPER_EVAL_LAST_N=5", out)
        self.assertNotIn("RIEI_PAPER_EVAL_LAST_N=10", out)


if __name__ == "__main__":
    unittest.main()
