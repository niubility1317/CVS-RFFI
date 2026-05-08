import unittest
import importlib

import torch
from torch.utils.data import DataLoader, Dataset


class _TinyDictDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[int(idx)]


class TestPaperBaselineParity(unittest.TestCase):
    def test_riei_mi_uses_signed_cosine_from_paper(self):
        from baselines.riei.losses import mutual_independence_loss

        z_e = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        z_r = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

        self.assertTrue(torch.allclose(mutual_independence_loss(z_e, z_r), torch.tensor(0.0)))

    def test_drift_style_transfer_center_module_clusters_same_receiver(self):
        from baselines.drift.losses import receiver_style_transfer_center_loss

        z_rx = torch.tensor([
            [1.0, 1.0],
            [3.0, 3.0],
            [10.0, 10.0],
            [14.0, 14.0],
        ])
        rx = torch.tensor([0, 0, 1, 1])

        loss = receiver_style_transfer_center_loss(z_rx, rx)

        self.assertTrue(torch.allclose(loss, torch.tensor(5.0)))

    def test_drift_negative_mse_is_bounded_when_normalized(self):
        from baselines.drift.losses import negative_mse_separation

        z_tx = torch.tensor([[1000.0, 0.0], [0.0, 1000.0]])
        z_rx = torch.tensor([[-1000.0, 0.0], [0.0, -1000.0]])

        loss = negative_mse_separation(z_tx, z_rx, normalize=True)

        self.assertGreaterEqual(float(loss), -4.0)
        self.assertLessEqual(float(loss), 0.0)

    def test_drift_total_loss_uses_bounded_separation_by_default(self):
        from baselines.drift.losses import compute_drift_loss

        outputs = {
            "tx_logits": torch.tensor([[5.0, -5.0], [-5.0, 5.0]]),
            "rx_logits": torch.tensor([[5.0, -5.0], [-5.0, 5.0]]),
            "domain_logits": torch.tensor([[5.0, -5.0], [-5.0, 5.0]]),
            "z_tx": torch.tensor([[1000.0, 0.0], [0.0, 1000.0]]),
            "z_rx": torch.tensor([[-1000.0, 0.0], [0.0, -1000.0]]),
        }
        labels = torch.tensor([0, 1])

        losses = compute_drift_loss(outputs, labels, labels, lambda_mse=0.02)

        self.assertGreater(float(losses["loss"]), -1.0)

    def test_best_val_gate_runs_tests_only_on_improvement(self):
        from baselines.common.cvs_trainer import BestValTestGate

        gate = BestValTestGate()
        self.assertTrue(gate.should_test(10.0))
        self.assertFalse(gate.should_test(9.0))
        self.assertFalse(gate.should_test(10.0))
        self.assertTrue(gate.should_test(10.1))

    def test_cvcnn_baseline_forward_outputs_logits(self):
        from baselines.cvcnn.model import BasicCVCNN, SincCVCNN

        model = BasicCVCNN(num_classes=5, input_len=128, base_channels=8, embedding_dim=16)
        logits = model(torch.randn(4, 2, 128))

        self.assertEqual(tuple(logits.shape), (4, 5))

        sinc_model = SincCVCNN(
            num_classes=5,
            input_len=128,
            base_channels=8,
            embedding_dim=16,
            sinc_kernel_size=31,
            sample_rate_hz=25e6,
        )
        sinc_logits = sinc_model(torch.randn(4, 2, 128))
        self.assertEqual(tuple(sinc_logits.shape), (4, 5))
        self.assertEqual(len(model.tail), len(sinc_model.tail))

    def test_each_cvs_baseline_has_own_train_module(self):
        modules = [
            "baselines.cvcnn.train",
            "baselines.riei.train",
            "baselines.drift.train",
            "baselines.receiver_agnostic_rffi.train",
            "baselines.tifs2025_channel_receiver_rffi.train",
        ]

        for name in modules:
            with self.subTest(name=name):
                module = importlib.import_module(name)
                self.assertTrue(callable(getattr(module, "main", None)))

    def test_validation_loss_plateau_controller_matches_paper_schedule(self):
        from baselines.common.cvs_trainer import ValidationLossPlateauController

        param = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([param], lr=1.0)
        controller = ValidationLossPlateauController(
            optimizer,
            lr_factor=0.5,
            lr_patience=2,
            early_stop_patience=4,
            min_delta=0.0,
        )

        self.assertTrue(controller.step(1.00, epoch=1).improved)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0)
        self.assertFalse(controller.step(1.10, epoch=2).stop_training)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0)
        self.assertTrue(controller.step(1.20, epoch=3).lr_reduced)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.5)
        self.assertFalse(controller.step(1.30, epoch=4).stop_training)
        self.assertTrue(controller.step(1.40, epoch=5).stop_training)

    def test_sat_eval_main_alias_selects_three_cvs_ood_dimensions(self):
        from baselines.common.cvs_sat_eval import resolve_sat_eval_loader_names

        named = {
            "test_unseen_day_seen_rx": object(),
            "test_seen_day_unseen_rx": object(),
            "test_unseen_day_unseen_rx": object(),
            "test_rx_7": object(),
        }

        self.assertEqual(
            resolve_sat_eval_loader_names(named, "main"),
            ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"],
        )

    def test_receiver_agnostic_collaborative_eval_groups_receiver_predictions(self):
        from baselines.common.cvs_data import collate_cvs_dict
        from baselines.receiver_agnostic_rffi.cvs_collaborative import evaluate_collaborative_tx

        samples = [
            {"iq": torch.tensor([4.0, 0.0]), "label": 0, "receiver": 0, "day": 0, "sig_i": 7, "meta": {}},
            {"iq": torch.tensor([0.0, 4.0]), "label": 0, "receiver": 1, "day": 0, "sig_i": 7, "meta": {}},
            {"iq": torch.tensor([0.0, 5.0]), "label": 1, "receiver": 0, "day": 0, "sig_i": 8, "meta": {}},
        ]
        loader = DataLoader(_TinyDictDataset(samples), batch_size=3, collate_fn=collate_cvs_dict)

        stats = evaluate_collaborative_tx(
            model=torch.nn.Identity(),
            loader=loader,
            device=torch.device("cpu"),
            forward_fn=lambda model, batch, device: batch["iq"],
            fusion="soft",
        )

        self.assertEqual(stats["tx_total"], 2)
        self.assertEqual(stats["tx_correct"], 2)
        self.assertEqual(stats["num_groups"], 2)

    def test_tifs_siamese_dataset_prefers_aligned_receiver_pairs(self):
        from baselines.tifs2025_channel_receiver_rffi.data import SiamesePairDataset

        samples = [
            {"iq": torch.tensor([1.0]), "label": 0, "receiver": 0, "day": 0, "sig_i": 10, "meta": {}},
            {"iq": torch.tensor([2.0]), "label": 0, "receiver": 1, "day": 0, "sig_i": 10, "meta": {}},
            {"iq": torch.tensor([3.0]), "label": 0, "receiver": 0, "day": 0, "sig_i": 11, "meta": {}},
            {"iq": torch.tensor([4.0]), "label": 0, "receiver": 1, "day": 0, "sig_i": 12, "meta": {}},
        ]

        ds = SiamesePairDataset(
            _TinyDictDataset(samples),
            augment=lambda x: x,
            spec_transform=lambda x: x,
            seed=0,
        )

        self.assertEqual(ds.pair_mode, "aligned")
        self.assertEqual(ds.aligned_pair_count, 1)
        x1, x2, label = ds[0]
        self.assertEqual(label, 0)
        self.assertEqual({float(x1.item()), float(x2.item())}, {1.0, 2.0})


if __name__ == "__main__":
    unittest.main()
