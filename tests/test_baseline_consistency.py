import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]


class BaselineAugmentationConsistencyTest(unittest.TestCase):
    def test_soft_consistency_is_finite_and_backpropagates(self):
        from baselines.common.consistency import (
            AugmentationConsistencyConfig,
            compute_augmentation_consistency_loss,
        )

        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2.0, -1.0], [-1.0, 2.0]]))
        batch = {"iq": torch.eye(2), "label": torch.tensor([-1, -1])}
        cfg = AugmentationConsistencyConfig(enabled=True, start_epoch=1, temperature=1.0, weight=1.0)

        result = compute_augmentation_consistency_loss(
            model,
            batch,
            torch.device("cpu"),
            cfg,
            epoch=1,
            sat_augment=lambda x: torch.flip(x, dims=[-1]),
        )
        result.loss.backward()

        self.assertTrue(result.active)
        self.assertTrue(torch.isfinite(result.loss))
        self.assertGreater(float(result.loss.detach()), 0.0)
        self.assertIsNotNone(model.weight.grad)
        self.assertIn("consistency/agreement", result.metrics)

    def test_parser_defaults_to_separate_disabled_route(self):
        from baselines.common.consistency import (
            add_augmentation_consistency_args,
            build_augmentation_consistency_config,
        )

        parser = argparse.ArgumentParser()
        add_augmentation_consistency_args(parser)
        cfg = build_augmentation_consistency_config(parser.parse_args([]))
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.start_epoch, 1)
        self.assertAlmostEqual(cfg.temperature, 1.0)
        self.assertAlmostEqual(cfg.weight, 1.0)

    def test_three_phase1_entrypoints_accept_consistency_switches(self):
        for method in ("cvcnn_ce", "riei_fd", "drift"):
            script = ROOT / "baselines" / method / "train_cvs.py"
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
            self.assertIn("--use_augmentation_consistency", proc.stdout)
            self.assertIn("--consistency_temperature", proc.stdout)
            self.assertIn("--lambda_consistency", proc.stdout)


def _find_bash():
    git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git" / "usr" / "bin" / "bash.exe"
    return str(git_bash) if git_bash.exists() else shutil.which("bash")


def _bash_path(path: str, bash: str) -> str:
    path = path.replace("\\", "/")
    if len(path) >= 3 and path[1:3] == ":/":
        if "/Git/usr/bin/bash.exe" in bash.replace("\\", "/"):
            return f"/{path[0].lower()}{path[2:]}"
        return f"/mnt/{path[0].lower()}{path[2:]}"
    return path


class Phase1SslMatrixLauncherTest(unittest.TestCase):
    def test_dry_run_emits_two_separate_three_method_routes(self):
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required")
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tmp_posix = _bash_path(tmp, bash)
            python_bin = _bash_path(sys.executable, bash)
            command = (
                f'RUN_ROOT_BASE="{tmp_posix}/runs" LOG_ROOT_BASE="{tmp_posix}/logs" '
                f'PYTHON_BIN="{python_bin}" DRY_RUN=1 '
                "bash scripts/launchers/run_phase1_ssl_baseline_matrix.sh"
            )
            proc = subprocess.run(
                [bash, "-lc", command],
                cwd=ROOT,
                env=os.environ.copy(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, output)
        self.assertEqual(output.count("baselines.cvcnn_ce.train"), 2)
        self.assertEqual(output.count("baselines.riei_fd.train"), 2)
        self.assertEqual(output.count("baselines.drift.train"), 2)
        self.assertIn("--wisig_labeled_ratio 0.1", output)
        self.assertIn("--wisig_unlabeled_ratio 0.6", output)
        self.assertIn("--wisig_source_val_ratio 0.3", output)
        self.assertIn("--use_pseudo_labels", output)
        self.assertIn("--use_augmentation_consistency", output)
        self.assertIn("--pseudo_start_epoch 150", output)
        self.assertIn("--pseudo_threshold 0.95", output)
        self.assertIn("--consistency_temperature 1.0", output)
        self.assertEqual(output.count("--use_sat_channel_view_aug"), 6)
        self.assertNotIn("--use_pseudo_labels --use_augmentation_consistency", output)


if __name__ == "__main__":
    unittest.main()
