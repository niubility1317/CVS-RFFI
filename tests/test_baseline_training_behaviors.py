import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class BaselineTrainerFinalEvalTest(unittest.TestCase):
    def test_final_eval_runs_when_last_epoch_did_not_improve(self):
        from baselines.common.cvs_trainer import run_validation_gated_training

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))

            def forward(self, x):
                return x @ self.w

        def batch(x, y):
            return {"iq": torch.tensor(x, dtype=torch.float32), "label": torch.tensor(y, dtype=torch.long)}

        train_loader = [batch([[1.0, 0.0], [0.0, 1.0]], [0, 1])]
        val_loader = [batch([[1.0, 0.0], [0.0, 1.0]], [0, 1])]
        named_tests = {
            "test_unseen_day_seen_rx": [batch([[1.0, 0.0]], [0])],
            "test_seen_day_unseen_rx": [batch([[0.0, 1.0]], [1])],
            "test_unseen_day_unseen_rx": [batch([[1.0, 0.0], [0.0, 1.0]], [0, 1])],
        }
        extra_calls = []

        def train_step(model, batch, device, epoch, step):
            # Make epoch 1 best, then degrade validation accuracy so epoch 2 is not tested by best-val gate.
            if epoch == 2:
                with torch.no_grad():
                    model.w.mul_(-1.0)
            return {"loss": 0.0}

        def extra_test(model, device):
            extra_calls.append(True)
            return {"sat_channel": {"clear_leo": {"aggregate": {"tx_acc": 12.5, "tx_correct": 1, "tx_total": 8}}}}

        with tempfile.TemporaryDirectory() as tmp:
            run_validation_gated_training(
                model=TinyModel(),
                train_loader=train_loader,
                val_loader=val_loader,
                named_test_loaders=named_tests,
                device=torch.device("cpu"),
                epochs=2,
                optimizer=torch.optim.SGD(TinyModel().parameters(), lr=0.1),
                train_step_fn=train_step,
                extra_test_fn=extra_test,
                output_dir=tmp,
            )
            metrics = json.loads((Path(tmp) / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(metrics["epochs"][-1]["epoch"], 2)
        self.assertFalse(metrics["epochs"][-1]["tested"])
        self.assertEqual(metrics["final"]["epoch"], 2)
        self.assertEqual(metrics["final"]["reason"], "post_training")
        self.assertIn("test_unseen_day_unseen_rx", metrics["final"]["test_named"])
        self.assertIn("sat_channel", metrics["final"]["extra_tests"])
        self.assertGreaterEqual(len(extra_calls), 2)

    def test_final_eval_retests_even_when_last_epoch_was_best(self):
        from baselines.common.cvs_trainer import run_validation_gated_training

        class ImprovingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(0.0))

            def forward(self, x):
                return self.scale * x

        def batch(x, y):
            return {"iq": torch.tensor(x, dtype=torch.float32), "label": torch.tensor(y, dtype=torch.long)}

        train_loader = [batch([[1.0, 0.0], [0.0, 1.0]], [0, 1])]
        val_loader = [batch([[1.0, 0.0], [0.0, 1.0]], [0, 1])]
        named_tests = {"test_unseen_day_unseen_rx": [batch([[1.0, 0.0], [0.0, 1.0]], [0, 1])]}
        extra_calls = []
        model = ImprovingModel()

        def train_step(model, batch, device, epoch, step):
            del batch, device, step
            with torch.no_grad():
                model.scale.fill_(float(epoch))
            return {"loss": 0.0}

        def val_loss(logits, labels, batch, output):
            del labels, batch, output
            return -logits.max()

        def extra_test(model, device):
            del model, device
            extra_calls.append(True)
            return {"sat_channel": {"clear_leo": {"aggregate": {"tx_acc": 50.0, "tx_correct": 1, "tx_total": 2}}}}

        with tempfile.TemporaryDirectory() as tmp:
            run_validation_gated_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                named_test_loaders=named_tests,
                device=torch.device("cpu"),
                epochs=2,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                train_step_fn=train_step,
                val_loss_fn=val_loss,
                best_metric="loss",
                extra_test_fn=extra_test,
                output_dir=tmp,
            )
            metrics = json.loads((Path(tmp) / "metrics.json").read_text(encoding="utf-8"))

        self.assertTrue(metrics["epochs"][-1]["tested"])
        self.assertEqual(metrics["final"]["reason"], "post_training")
        self.assertTrue(metrics["final"]["last_epoch_tested"])
        self.assertEqual(len(extra_calls), 3)

    def test_collaborative_baseline_also_records_observation_level_eval(self):
        from baselines.common.cvs_trainer import run_validation_gated_training

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))

            def forward(self, x):
                return x @ self.w

        def batch(x, y):
            return {"iq": torch.tensor(x, dtype=torch.float32), "label": torch.tensor(y, dtype=torch.long)}

        model = TinyModel()
        train_loader = [batch([[1.0, 0.0], [0.0, 1.0]], [0, 1])]
        val_loader = [batch([[1.0, 0.0], [0.0, 1.0]], [0, 1])]
        named_tests = {"test_unseen_day_unseen_rx": [batch([[1.0, 0.0], [0.0, 1.0]], [0, 1])]}

        def train_step(model, batch, device, epoch, step):
            del model, batch, device, epoch, step
            return {"loss": 0.0}

        def fake_collab_eval(model, loader, device):
            del model, loader, device
            return {"tx_acc": 100.0, "tx_correct": 1, "tx_total": 1, "num_groups": 1, "receiver_observations": 2}

        with tempfile.TemporaryDirectory() as tmp:
            run_validation_gated_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                named_test_loaders=named_tests,
                device=torch.device("cpu"),
                epochs=1,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                train_step_fn=train_step,
                test_evaluate_fn=fake_collab_eval,
                output_dir=tmp,
            )
            metrics = json.loads((Path(tmp) / "metrics.json").read_text(encoding="utf-8"))

        final = metrics["final"]
        self.assertEqual(final["test_protocols"]["primary"], "collaborative_group")
        self.assertEqual(final["test_named"]["test_unseen_day_unseen_rx"]["tx_total"], 1)
        self.assertEqual(final["test_named_collab"]["test_unseen_day_unseen_rx"]["tx_total"], 1)
        self.assertEqual(final["test_named_obs"]["test_unseen_day_unseen_rx"]["tx_total"], 2)
        self.assertEqual(final["test_overall_collab"]["tx_total"], 1)
        self.assertEqual(final["test_overall_obs"]["tx_total"], 2)


class BaselineSatViewAugmentTest(unittest.TestCase):
    def test_baseline_sat_eval_default_matches_main_ood_splits(self):
        import argparse
        from baselines.common.cvs_sat_eval import MAIN_SAT_EVAL_ON, MAIN_SAT_EVAL_ON_NAMES, add_cvs_sat_eval_args, resolve_sat_eval_loader_names

        parser = argparse.ArgumentParser()
        add_cvs_sat_eval_args(parser)
        args = parser.parse_args([])
        named_loaders = {name: [] for name in MAIN_SAT_EVAL_ON_NAMES}

        self.assertEqual(args.eval_sat_on, MAIN_SAT_EVAL_ON)
        self.assertEqual(resolve_sat_eval_loader_names(named_loaders, "main"), MAIN_SAT_EVAL_ON_NAMES)

    def test_satellite_view_augment_duplicates_supervised_batch(self):
        from baselines.common.augmentation import (
            SatGroundChannelViewAugment,
            supervised_sat_view_batch,
        )

        augment = SatGroundChannelViewAugment(scenarios=["clear_leo"], p=1.0, seed=123)
        batch = {
            "iq": torch.randn(2, 2, 16),
            "label": torch.tensor([0, 1], dtype=torch.long),
            "receiver": torch.tensor([2, 3], dtype=torch.long),
        }

        out = supervised_sat_view_batch(batch, torch.device("cpu"), augment)

        self.assertEqual(tuple(out["iq"].shape), (4, 2, 16))
        self.assertTrue(torch.equal(out["label"], torch.tensor([0, 1, 0, 1])))
        self.assertTrue(torch.equal(out["receiver"], torch.tensor([2, 3, 2, 3])))
        self.assertFalse(torch.allclose(out["iq"][:2], out["iq"][2:]))


class BaselineWiSigPaperProtocolTest(unittest.TestCase):
    def test_drift_day1_split_uses_paper_receiver_groups_and_sample_counts(self):
        from dataset_wisig import make_wisig_drift_day1_split

        rx_labels = [
            "1-1",
            "1-19",
            "14-7",
            "18-2",
            "19-2",
            "2-1",
            "2-19",
            "20-1",
            "3-19",
            "7-14",
            "7-7",
            "8-8",
        ]
        data = []
        for tx in range(2):
            tx_rows = []
            for rx in range(len(rx_labels)):
                day_rows = []
                samples = torch.zeros(10, 16, 2, dtype=torch.float32).numpy()
                samples[:, :, 0] = float(tx)
                samples[:, :, 1] = float(rx)
                day_rows.append([samples])
                tx_rows.append(day_rows)
            data.append(tx_rows)
        ds = {
            "data": data,
            "tx_list": ["tx0", "tx1"],
            "rx_list": rx_labels,
            "capture_date_list": ["2021_03_01"],
            "equalized_list": [1],
        }

        train, val, test, named, meta, info = make_wisig_drift_day1_split(
            ds,
            out_len=16,
            train_samples_per_combo=8,
            val_samples_per_combo=2,
            test_samples_per_combo=2,
        )

        self.assertEqual(info["mode"], "wisig_drift_day1_receiver_disjoint_800_200")
        self.assertEqual(info["train_rxs_label"], ["1-1", "14-7", "7-7"])
        self.assertEqual(info["test_rxs_label"], ["1-19", "19-2", "2-1", "2-19", "20-1", "7-14", "8-8"])
        self.assertEqual(len(train), 2 * 3 * 8)
        self.assertEqual(len(val), 2 * 3 * 2)
        self.assertEqual(len(test), 2 * 7 * 2)
        self.assertIn("test_seen_day_unseen_rx", named)
        self.assertEqual(meta["test_seen_day_unseen_rx"]["paper_protocol_alias"], "drift_day1_unseen_receiver")

    def test_riei_original_split_uses_two_source_receiver_holdout_counts(self):
        from dataset_wisig import make_wisig_riei_receiver_holdout_split

        rx_labels = ["1-1", "1-19", "7-7", "8-8"]
        data = []
        for tx in range(2):
            tx_rows = []
            for rx in range(len(rx_labels)):
                day_rows = []
                for day in range(2):
                    samples = torch.zeros(10, 16, 2, dtype=torch.float32).numpy()
                    samples[:, :, 0] = float(tx)
                    samples[:, :, 1] = float(rx * 10 + day)
                    day_rows.append([samples])
                tx_rows.append(day_rows)
            data.append(tx_rows)
        ds = {
            "data": data,
            "tx_list": ["tx0", "tx1"],
            "rx_list": rx_labels,
            "capture_date_list": ["2021_03_01", "2021_03_08"],
            "equalized_list": [1],
        }

        train, val, test, named, meta, info = make_wisig_riei_receiver_holdout_split(
            ds,
            out_len=16,
            train_rxs=["1-1", "7-7"],
            test_rxs=["1-19"],
            train_samples_per_combo=6,
            val_samples_per_combo=3,
            test_samples_per_combo=4,
            seed=7,
        )

        self.assertEqual(info["mode"], "wisig_riei_two_source_receiver_holdout_14400_4800")
        self.assertEqual(info["train_rxs_label"], ["1-1", "7-7"])
        self.assertEqual(info["test_rxs_label"], ["1-19"])
        self.assertEqual(len(train), 2 * 2 * 6)
        self.assertEqual(len(val), 2 * 2 * 3)
        self.assertEqual(len(test), 2 * 1 * 4)
        self.assertEqual(info["aggregate_test_keys"], ["test_seen_day_unseen_rx"])
        self.assertIn("test_seen_day_unseen_rx", named)
        self.assertEqual(meta["test_seen_day_unseen_rx"]["paper_protocol_alias"], "riei_two_source_receiver_holdout")

    def test_trainer_records_paper_eval_window_summary(self):
        from baselines.common.cvs_trainer import run_validation_gated_training

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.eye(2))

            def forward(self, x):
                return x @ self.w

        def batch(x, y):
            return {"iq": torch.tensor(x, dtype=torch.float32), "label": torch.tensor(y, dtype=torch.long)}

        model = TinyModel()
        train_loader = [batch([[1.0, 0.0], [0.0, 1.0]], [0, 1])]
        val_loader = [batch([[1.0, 0.0], [0.0, 1.0]], [0, 1])]
        named_tests = {"test_seen_day_unseen_rx": [batch([[1.0, 0.0], [0.0, 1.0]], [0, 1])]}

        def train_step(model, batch, device, epoch, step):
            del model, batch, device, epoch, step
            return {"loss": 0.0}

        with tempfile.TemporaryDirectory() as tmp:
            run_validation_gated_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                named_test_loaders=named_tests,
                device=torch.device("cpu"),
                epochs=3,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                train_step_fn=train_step,
                paper_eval_last_n=2,
                paper_eval_name="drift_last2",
                output_dir=tmp,
            )
            metrics = json.loads((Path(tmp) / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(metrics["final"]["paper_eval_window"]["name"], "drift_last2")
        self.assertEqual(metrics["final"]["paper_eval_window"]["n"], 2)
        self.assertEqual(metrics["final"]["paper_eval_window"]["epochs"], [2, 3])

    def test_trainer_can_skip_best_val_tests_until_final(self):
        from baselines.common.cvs_trainer import run_validation_gated_training

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.eye(2))

            def forward(self, x):
                return x @ self.w

        def batch(x, y):
            return {"iq": torch.tensor(x, dtype=torch.float32), "label": torch.tensor(y, dtype=torch.long)}

        model = TinyModel()
        train_loader = [batch([[1.0, 0.0], [0.0, 1.0]], [0, 1])]
        val_loader = [batch([[1.0, 0.0], [0.0, 1.0]], [0, 1])]
        named_tests = {"test_seen_day_unseen_rx": [batch([[1.0, 0.0], [0.0, 1.0]], [0, 1])]}

        def train_step(model, batch, device, epoch, step):
            del model, batch, device, epoch, step
            return {"loss": 0.0}

        with tempfile.TemporaryDirectory() as tmp:
            run_validation_gated_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                named_test_loaders=named_tests,
                device=torch.device("cpu"),
                epochs=2,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                train_step_fn=train_step,
                test_on_val_improve=False,
                output_dir=tmp,
            )
            metrics_path = Path(tmp) / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            checkpoint_exists = (Path(tmp) / "best_by_val.pt").exists()

        self.assertTrue(checkpoint_exists)
        self.assertFalse(any(epoch["tested"] for epoch in metrics["epochs"]))
        self.assertEqual(metrics["best"]["test_on_val_improve"], False)
        self.assertEqual(metrics["final"]["reason"], "post_training")
        self.assertIn("test_overall", metrics["final"])


class BaselineServerLaunchTest(unittest.TestCase):
    def test_method_file_entrypoints_help_work_from_server_style_root(self):
        cases = [
            ("cvcnn_ce", "CVCNN-CE CVS-RFFI baseline"),
            ("drift", "DRIFT CVS-RFFI training"),
            ("riei_fd", "RIEI-FD CVS-RFFI training"),
            ("ra_collab", "RA-Collab RFFI CVS training"),
        ]
        for method_dir, expected in cases:
            with self.subTest(method_dir=method_dir):
                proc = subprocess.run(
                    [sys.executable, str(ROOT / "baselines" / method_dir / "train.py"), "--help"],
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )

                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertIn(expected, proc.stdout)

    def test_ra_collab_cvs_uses_single_raw_loader_set_when_sat_eval_enabled(self):
        from baselines.ra_collab import train_cvs

        class TinyModel(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.p = nn.Parameter(torch.tensor(0.0))

            def to(self, device):
                return self

            def parameters(self):
                return [self.p]

        fake_split = SimpleNamespace(num_classes=2, num_receivers=3)
        fake_loaders = SimpleNamespace(
            train=[],
            val=[],
            named_tests={"test_unseen_day_unseen_rx": []},
            split=fake_split,
        )
        build_calls = []

        def fake_build(args, device, **kwargs):
            build_calls.append(kwargs)
            return fake_loaders

        argv = [
            "train.py",
            "--wisig_pkl",
            str(ROOT / "Dataset_WigSig" / "ManySig.pkl"),
            "--eval_sat_channel",
            "--eval_sat_scenarios",
            "clear_leo",
            "--epochs",
            "1",
        ]
        with mock.patch.object(sys, "argv", argv), \
            mock.patch.object(train_cvs, "build_cvs_loaders", side_effect=fake_build), \
            mock.patch.object(train_cvs, "RACollabRFFI", TinyModel), \
            mock.patch.object(train_cvs, "run_validation_gated_training") as run_training:
            train_cvs.main()

        self.assertEqual(len(build_calls), 1)
        self.assertEqual(build_calls[0], {})
        self.assertEqual(run_training.call_count, 1)

    def test_ra_collab_train_cvs_imports_local_spectrogram(self):
        code = f"""
import sys
from pathlib import Path

root = Path({str(ROOT)!r})
for path in [root, root / "code"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baselines.ra_collab import train_cvs
print(train_cvs.SpectrogramTransform.__module__)
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("baselines.ra_collab.spectrogram", proc.stdout)

    def test_cvs_baseline_entrypoints_accept_satellite_view_aug_switches(self):
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
                self.assertIn("--use_sat_channel_view_aug", proc.stdout)
                self.assertIn("--sat_train_scenario", proc.stdout)
                self.assertIn("--sat_view_prob", proc.stdout)

if __name__ == "__main__":
    unittest.main()

