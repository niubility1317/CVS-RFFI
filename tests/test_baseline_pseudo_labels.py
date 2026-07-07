import argparse
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]


class BaselinePseudoLabelModuleTest(unittest.TestCase):
    def test_parser_and_config_expose_default_off_pseudo_label_controls(self):
        from baselines.common.pseudo_labels import add_pseudo_label_args, build_pseudo_label_config

        parser = argparse.ArgumentParser()
        add_pseudo_label_args(parser)

        default_args = parser.parse_args([])
        default_cfg = build_pseudo_label_config(default_args)
        self.assertFalse(default_cfg.enabled)
        self.assertEqual(default_cfg.start_epoch, 150)
        self.assertAlmostEqual(default_cfg.threshold, 0.90)

        enabled_args = parser.parse_args(
            [
                "--use_pseudo_labels",
                "--pseudo_start_epoch",
                "2",
                "--pseudo_threshold",
                "0.95",
                "--pseudo_margin",
                "0.20",
                "--lambda_pseudo",
                "0.5",
            ]
        )
        cfg = build_pseudo_label_config(enabled_args)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.start_epoch, 2)
        self.assertAlmostEqual(cfg.threshold, 0.95)
        self.assertAlmostEqual(cfg.margin, 0.20)
        self.assertAlmostEqual(cfg.weight, 0.5)

    def test_compute_pseudo_label_loss_respects_epoch_gate_and_confidence_filter(self):
        from baselines.common.pseudo_labels import PseudoLabelConfig, compute_pseudo_label_loss

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(
                    torch.tensor(
                        [
                            [4.0, -4.0],
                            [-4.0, 4.0],
                        ]
                    )
                )

            def forward(self, x):
                return x @ self.w

        model = TinyModel()
        batch = {"iq": torch.eye(2), "label": torch.tensor([1, 0], dtype=torch.long)}
        cfg = PseudoLabelConfig(enabled=True, start_epoch=3, threshold=0.90, margin=0.10, weight=0.5)

        early = compute_pseudo_label_loss(model, batch, torch.device("cpu"), cfg, epoch=2)
        self.assertFalse(early.active)
        self.assertEqual(early.total, 2)
        self.assertEqual(early.selected, 2)
        self.assertEqual(float(early.loss), 0.0)
        self.assertAlmostEqual(early.metrics["pseudo/active"], 0.0)
        self.assertAlmostEqual(early.metrics["pseudo/coverage"], 1.0)
        self.assertGreater(early.metrics["pseudo/confidence_all"], 0.0)
        self.assertIn("pseudo/confidence_max", early.metrics)

        active = compute_pseudo_label_loss(model, batch, torch.device("cpu"), cfg, epoch=3)
        self.assertTrue(active.active)
        self.assertEqual(active.total, 2)
        self.assertEqual(active.selected, 2)
        self.assertGreater(float(active.loss.detach()), 0.0)
        self.assertAlmostEqual(active.metrics["pseudo/coverage"], 1.0)

    def test_pseudo_label_precision_uses_true_label_for_masked_source_samples(self):
        from baselines.common.pseudo_labels import PseudoLabelConfig, compute_pseudo_label_loss

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.tensor([[4.0, -4.0], [-4.0, 4.0]]))

            def forward(self, x):
                return x @ self.w

        batch = {
            "iq": torch.eye(2),
            "label": torch.tensor([-1, -1], dtype=torch.long),
            "true_label": torch.tensor([0, 1], dtype=torch.long),
        }
        cfg = PseudoLabelConfig(enabled=True, start_epoch=1, threshold=0.90, margin=0.10, weight=1.0)

        active = compute_pseudo_label_loss(TinyModel(), batch, torch.device("cpu"), cfg, epoch=1)

        self.assertTrue(active.active)
        self.assertEqual(active.selected, 2)
        self.assertAlmostEqual(active.metrics["pseudo/precision"], 1.0)


class BaselinePseudoLabelTrainerHookTest(unittest.TestCase):
    def test_training_loop_calls_pseudo_step_after_start_epoch_and_merges_loss(self):
        from baselines.common.cvs_trainer import run_validation_gated_training

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))

            def forward(self, x):
                return x @ self.w

        def batch(x, y):
            return {"iq": torch.tensor(x, dtype=torch.float32), "label": torch.tensor(y, dtype=torch.long)}

        pseudo_epochs = []

        def train_step(model, batch, device, epoch, step):
            return {"loss": 1.0}

        def pseudo_step(model, device, epoch, step):
            pseudo_epochs.append(epoch)
            if epoch < 2:
                return {"loss": 0.0, "pseudo/active": 0.0}
            return {"loss": 0.5, "pseudo/active": 1.0, "pseudo/selected": 2.0}

        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                history = run_validation_gated_training(
                    model=TinyModel(),
                    train_loader=[batch([[1.0, 0.0], [0.0, 1.0]], [0, 1])],
                    val_loader=[batch([[1.0, 0.0], [0.0, 1.0]], [0, 1])],
                    named_test_loaders={"test_unseen_day_unseen_rx": [batch([[1.0, 0.0]], [0])]},
                    device=torch.device("cpu"),
                    epochs=2,
                    optimizer=torch.optim.SGD(TinyModel().parameters(), lr=0.1),
                    train_step_fn=train_step,
                    pseudo_step_fn=pseudo_step,
                    output_dir=tmp,
                )

        self.assertEqual(pseudo_epochs, [1, 2])
        self.assertAlmostEqual(history.epochs[0]["train_loss"], 1.0)
        self.assertAlmostEqual(history.epochs[1]["train_loss"], 1.5)
        self.assertEqual(history.epochs[1]["train_pseudo"]["pseudo/selected"], 2.0)
        self.assertIn("[PSEUDO-METRICS]", stdout.getvalue())


class BaselinePseudoLabelEntrypointTest(unittest.TestCase):
    def test_all_cvs_baseline_entrypoints_accept_pseudo_label_switches(self):
        scripts = [
            ROOT / "baselines" / "cvcnn_ce" / "train_cvs.py",
            ROOT / "baselines" / "drift" / "train_cvs.py",
            ROOT / "baselines" / "riei_fd" / "train_cvs.py",
            ROOT / "baselines" / "ra_collab" / "train_cvs.py",
        ]
        for script in scripts:
            with self.subTest(script=script):
                proc = subprocess.run(
                    [sys.executable, str(script), "--help"],
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertIn("--use_pseudo_labels", proc.stdout)
                self.assertIn("--use_source_ssl_split", proc.stdout)
                self.assertIn("--wisig_unlabeled_ratio", proc.stdout)
                self.assertIn("--pseudo_start_epoch", proc.stdout)
                self.assertIn("--pseudo_threshold", proc.stdout)
                self.assertIn("--lambda_pseudo", proc.stdout)


if __name__ == "__main__":
    unittest.main()
