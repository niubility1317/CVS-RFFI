import argparse
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


class TinyTargetDataset(Dataset):
    def __init__(self):
        self.labels = [0, 0, 0, 1, 1, 2, 2, 2, 2]
        self.index = [
            type("WiSigIndexLike", (), {"rx_i": 7})(),
            type("WiSigIndexLike", (), {"rx_i": 7})(),
            type("WiSigIndexLike", (), {"rx_i": 7})(),
            type("WiSigIndexLike", (), {"rx_i": 8})(),
            type("WiSigIndexLike", (), {"rx_i": 8})(),
            type("WiSigIndexLike", (), {"rx_i": 8})(),
            type("WiSigIndexLike", (), {"rx_i": 9})(),
            type("WiSigIndexLike", (), {"rx_i": 9})(),
            type("WiSigIndexLike", (), {"rx_i": 9})(),
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        y = self.labels[idx]
        x = torch.full((2, 8), float(idx))
        return x, y, 0, {"base_index": idx}


class TinyRxTxDataset(Dataset):
    def __init__(self):
        self.index = []
        self.samples = []
        for rx in [7, 8]:
            for tx in [0, 1, 2]:
                for sample_idx in range(4):
                    self.index.append(type("WiSigIndexLike", (), {"rx_i": rx, "tx_i": tx})())
                    self.samples.append((rx, tx, sample_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rx, tx, _ = self.samples[idx]
        x = torch.full((2, 8), float(idx))
        return x, tx, 0, {"rx_i": rx, "tx_i": tx, "base_index": idx}


class UnlabeledTargetDataset(Dataset):
    def __init__(self, n=9):
        self.n = n
        self.label_reads = 0

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self.label_reads += 1
        x = torch.full((2, 8), float(idx))
        return x, -1, 0, {"base_index": idx}


class TinyLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 3)

    def forward(self, x, y_tx=None, grl_lambda=1.0, return_aux=True, domain_labels=None):
        z = x.mean(dim=-1)
        z = torch.cat([z, z], dim=1)
        logits = self.head(self.norm(z))
        return {"tx_logits": logits, "z_id": z, "dom_logits": logits.new_zeros(logits.size(0), 1)}


class TargetDomainAdaptationTest(unittest.TestCase):
    def test_fewshot_selector_balances_classes_when_requested(self):
        from target_domain_adaptation import select_fewshot_indices

        selected = select_fewshot_indices(TinyTargetDataset(), samples_per_class=2, max_samples=0, seed=7)

        labels = [TinyTargetDataset().labels[i] for i in selected]
        self.assertEqual(len(selected), 6)
        self.assertEqual(labels.count(0), 2)
        self.assertEqual(labels.count(1), 2)
        self.assertEqual(labels.count(2), 2)

    def test_unlabeled_target_selector_uses_total_budget_without_reading_labels(self):
        from target_domain_adaptation import select_unlabeled_target_indices

        dataset = UnlabeledTargetDataset(n=20)
        selected = select_unlabeled_target_indices(dataset, num_samples=5, seed=11)

        self.assertEqual(len(selected), 5)
        self.assertEqual(len(set(selected)), 5)
        self.assertTrue(all(0 <= idx < len(dataset) for idx in selected))
        self.assertEqual(dataset.label_reads, 0)

    def test_unlabeled_target_selector_can_budget_each_receiver_without_reading_labels(self):
        from target_domain_adaptation import select_unlabeled_target_indices_per_rx

        dataset = TinyTargetDataset()
        selected = select_unlabeled_target_indices_per_rx(dataset, samples_per_rx=2, seed=11)

        rx_counts = {}
        for idx in selected:
            rx = dataset.index[idx].rx_i
            rx_counts[rx] = rx_counts.get(rx, 0) + 1
        self.assertEqual(rx_counts, {7: 2, 8: 2, 9: 2})

    def test_target_selector_can_budget_each_transmitter_inside_each_receiver(self):
        from target_domain_adaptation import select_target_indices_per_rx_tx

        dataset = TinyRxTxDataset()
        selected = select_target_indices_per_rx_tx(dataset, samples_per_rx_tx=2, seed=17)

        counts = {}
        for idx in selected:
            item = dataset.index[idx]
            key = (item.rx_i, item.tx_i)
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(selected), 12)
        self.assertEqual(counts, {(rx, tx): 2 for rx in [7, 8] for tx in [0, 1, 2]})

    def test_entropy_adapter_loss_uses_clean_and_satellite_views(self):
        from target_domain_adaptation import (
            EntropyMinimizationLogitAdapter,
            TargetAdaptLossConfig,
            compute_target_adaptation_loss,
            configure_target_adaptation_parameters,
        )

        torch.manual_seed(3)
        base = TinyLogitModel()
        adapter = EntropyMinimizationLogitAdapter(base, num_classes=3)
        params = configure_target_adaptation_parameters(adapter, update_norm=True, update_classifier=False)
        x_clean = torch.randn(5, 2, 8)
        x_sat = x_clean + 0.05 * torch.randn_like(x_clean)

        loss, logs = compute_target_adaptation_loss(
            adapter,
            x_clean,
            x_sat,
            TargetAdaptLossConfig(conf_threshold=0.0, margin_threshold=-1.0),
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("target_adapt/loss_entropy", logs)
        self.assertIn("target_adapt/loss_consistency", logs)
        self.assertIn("target_adapt/pseudo_coverage", logs)
        self.assertTrue(any(p.grad is not None for p in params))
        self.assertTrue(torch.isfinite(adapter.logit_bias.grad).all())

    def test_single_view_target_loss_skips_view_consistency(self):
        from target_domain_adaptation import (
            EntropyMinimizationLogitAdapter,
            TargetAdaptLossConfig,
            compute_target_adaptation_loss,
        )

        torch.manual_seed(5)
        adapter = EntropyMinimizationLogitAdapter(TinyLogitModel(), num_classes=3)
        x_target = torch.randn(4, 2, 8)

        loss, logs = compute_target_adaptation_loss(
            adapter,
            x_target,
            None,
            TargetAdaptLossConfig(conf_threshold=0.0, margin_threshold=-1.0),
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(logs["target_adapt/loss_consistency"]), 0.0)
        self.assertIn("target_adapt/loss_entropy", logs)

    def test_logit_lora_adapter_starts_as_base_prediction_and_freezes_base(self):
        from target_domain_adaptation import build_target_adapter, configure_target_adaptation_parameters

        torch.manual_seed(19)
        base = TinyLogitModel()
        x = torch.randn(4, 2, 8)
        base_logits = base(x)["tx_logits"].detach()
        adapter = build_target_adapter(base, num_classes=3, adapter_type="logit_lora", adapter_rank=2)
        params = configure_target_adaptation_parameters(adapter, update_norm=True, update_classifier=True)

        out = adapter(x)
        loss = out["tx_logits"].sum()
        loss.backward()

        self.assertTrue(torch.allclose(out["tx_logits"], base_logits, atol=1e-6))
        self.assertTrue(any(p.grad is not None for p in params))
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        self.assertTrue(all(name.startswith(("logit_", "log_temperature", "lora_")) for name, p in adapter.named_parameters() if p.requires_grad))

    def test_feature_residual_adapter_starts_as_base_prediction_and_uses_z_id(self):
        from target_domain_adaptation import build_target_adapter, configure_target_adaptation_parameters

        torch.manual_seed(23)
        base = TinyLogitModel()
        x = torch.randn(3, 2, 8)
        base_logits = base(x)["tx_logits"].detach()
        adapter = build_target_adapter(base, num_classes=3, adapter_type="feature_residual", adapter_bottleneck=2)
        params = configure_target_adaptation_parameters(adapter, update_norm=True, update_classifier=True)
        out = adapter(x)
        loss = out["tx_logits"].sum()
        loss.backward()

        self.assertTrue(torch.allclose(out["tx_logits"], base_logits, atol=1e-6))
        self.assertIn("adapter_delta_logits", out)
        self.assertTrue(any(p.grad is not None for p in params))
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        self.assertTrue(all(name.startswith(("logit_", "log_temperature", "adapter_")) for name, p in adapter.named_parameters() if p.requires_grad))

    def test_train_target_adapt_parser_accepts_experiment_shape(self):
        from train_target_adapt import build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--teacher_ckpt",
                "runs/best_base_explore/BEX02_fishr002_mixed_e170/best_primary_ood_model.pth",
                "--output_dir",
                "runs/target_adapt/bex02_t1_p01_k2",
                "--target_loader",
                "test_unseen_day_unseen_rx",
                "--target_num_samples",
                "64",
                "--target_channel_view",
                "provided_satellite",
                "--target_label_mode",
                "labeled",
                "--sat_train_scenario",
                "clear_leo",
                "--eval_sat_channel",
                "true",
            ]
        )

        self.assertEqual(args.target_loader, "test_unseen_day_unseen_rx")
        self.assertEqual(args.target_samples_per_class, 0)
        self.assertEqual(args.target_num_samples, 64)
        self.assertEqual(args.target_samples_per_rx_tx, 0)
        self.assertEqual(args.target_channel_view, "provided_satellite")
        self.assertEqual(args.target_label_mode, "labeled")
        self.assertTrue(args.eval_sat_channel)
        self.assertEqual(args.target_adapter_type, "logit_calibration")

    def test_train_target_adapt_parser_accepts_rx_tx_balanced_budget(self):
        from train_target_adapt import build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--teacher_ckpt",
                "teacher.pth",
                "--output_dir",
                "runs/tmp",
                "--target_samples_per_rx_tx",
                "3",
            ]
        )

        self.assertEqual(args.target_samples_per_rx_tx, 3)

    def test_train_target_adapt_parser_accepts_adapter_and_rollback_options(self):
        from train_target_adapt import build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--teacher_ckpt",
                "teacher.pth",
                "--output_dir",
                "runs/tmp",
                "--target_adapter_type",
                "logit_lora",
                "--adapter_rank",
                "8",
                "--freeze_base_stats",
                "true",
                "--rollback_enabled",
                "true",
            ]
        )

        self.assertEqual(args.target_adapter_type, "logit_lora")
        self.assertEqual(args.adapter_rank, 8)
        self.assertTrue(args.freeze_base_stats)
        self.assertTrue(args.rollback_enabled)

    def test_eval_loader_excludes_unlabeled_adaptation_indices(self):
        from torch.utils.data import DataLoader

        from train_target_adapt import build_eval_data_ctx_excluding_target_indices

        target_ds = TinyTargetDataset()
        data_ctx = {
            "named_test_loaders": {
                "test_unseen_day_unseen_rx": DataLoader(target_ds, batch_size=2, shuffle=False),
                "test_other": DataLoader(target_ds, batch_size=2, shuffle=False),
            },
            "named_test_meta": {
                "test_unseen_day_unseen_rx": {"size": len(target_ds), "note": "target"},
                "test_other": {"size": len(target_ds)},
            },
        }

        eval_ctx = build_eval_data_ctx_excluding_target_indices(
            data_ctx,
            target_loader_name="test_unseen_day_unseen_rx",
            adaptation_indices=[1, 3, 5],
            batch_size=2,
            num_workers=0,
            device=torch.device("cpu"),
            prefetch_factor=2,
        )

        kept = eval_ctx["named_test_loaders"]["test_unseen_day_unseen_rx"].dataset.indices
        self.assertEqual(list(kept), [0, 2, 4, 6, 7, 8])
        self.assertEqual(eval_ctx["named_test_meta"]["test_unseen_day_unseen_rx"]["size"], 6)
        self.assertIs(eval_ctx["named_test_loaders"]["test_other"], data_ctx["named_test_loaders"]["test_other"])

    def test_target_finetune_loader_keeps_small_fewshot_batches(self):
        from torch.utils.data import Subset

        from train_target_adapt import build_arg_parser, build_target_finetune_loader

        args = build_arg_parser().parse_args(
            [
                "--teacher_ckpt",
                "teacher.pth",
                "--output_dir",
                "runs/tmp",
                "--batch_size",
                "64",
                "--num_workers",
                "0",
            ]
        )
        loader = build_target_finetune_loader(
            Subset(TinyTargetDataset(), [0, 1, 2]),
            args,
            device=torch.device("cpu"),
        )

        self.assertEqual(len(loader), 1)
        x, y, *_ = next(iter(loader))
        self.assertEqual(x.shape[0], 3)
        self.assertEqual(y.shape[0], 3)

    def test_launcher_dry_run_mentions_bex02_target_adapt_and_queue(self):
        script = ROOT / "code" / "scripts" / "run_target_adapt_bex02_6gpu.sh"
        text = script.read_text(encoding="utf-8")

        self.assertIn("train_target_adapt.py", text)
        self.assertIn("BEX02_fishr002_mixed_e170", text)
        self.assertIn("latest_model.pth", text)
        self.assertIn("/home/szu2070436088/2510044040/CV-SincNet/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth", text)
        self.assertIn("--target_channel_view", text)
        self.assertIn("provided_satellite", text)
        self.assertIn("--target_label_mode", text)
        self.assertIn("labeled", text)
        self.assertIn("unlabeled", text)
        self.assertIn("GPU_IDS:-6,7", text)
        self.assertNotIn("--target_train_scenarios", text)
        self.assertNotIn("--sat_train_scenario mixed_orbit", text)
        self.assertIn("--target_num_samples", text)
        self.assertNotIn("--target_samples_per_class", text)
        self.assertIn("--target_loader", text)
        self.assertIn("--target_samples_per_rx", text)
        self.assertIn("short_tag", text)
        self.assertNotIn("s/^/[${exp_id}|GPU${gpu_id}] /", text)

    def test_8gpu_sweep_launcher_covers_epoch_weight_label_and_sat_eval_grid(self):
        script = ROOT / "code" / "scripts" / "run_target_adapt_bex02_sweep_8gpu.sh"
        doc = ROOT / "docs" / "target_domain_adaptation_sweep_8gpu.md"

        script_text = script.read_text(encoding="utf-8")
        doc_text = doc.read_text(encoding="utf-8")

        self.assertIn("GPU_IDS:-0,1,2,3,4,5,6,7", script_text)
        self.assertIn("EPOCHS_CSV", script_text)
        self.assertIn("20,50,100", script_text)
        self.assertIn("ADAPT_WEIGHTS_CSV", script_text)
        self.assertIn("labeled", script_text)
        self.assertIn("unlabeled", script_text)
        self.assertIn("--eval_sat_channel", script_text)
        self.assertNotIn("--eval_sat_channel false", script_text)
        self.assertIn("--eval_sat_on", script_text)
        self.assertIn("--eval_sat_scenarios", script_text)
        self.assertIn("--epochs", script_text)
        self.assertIn("--lr_adapt", script_text)
        self.assertIn("--anchor_weight", script_text)
        self.assertIn("train_target_adapt.py", script_text)
        self.assertIn("best_target_adapt.pth", script_text)

        self.assertIn("run_target_adapt_bex02_sweep_8gpu.sh", doc_text)
        self.assertIn("20, 50, 100", doc_text)
        self.assertIn("labeled", doc_text)
        self.assertIn("unlabeled", doc_text)
        self.assertIn("星地信道", doc_text)
        self.assertIn("--dry-run", doc_text)

    def test_clean_target_control_launcher_keeps_grid_but_adapts_on_clean_target_samples(self):
        script = ROOT / "code" / "scripts" / "run_target_adapt_bex02_clean_target_8gpu.sh"
        sweep_script = ROOT / "code" / "scripts" / "run_target_adapt_bex02_sweep_8gpu.sh"
        doc = ROOT / "docs" / "target_domain_adaptation_clean_target_control.md"

        script_text = script.read_text(encoding="utf-8")
        sweep_text = sweep_script.read_text(encoding="utf-8")
        doc_text = doc.read_text(encoding="utf-8")

        self.assertIn("TARGET_CHANNEL_VIEW=\"${TARGET_CHANNEL_VIEW:-clean}\"", script_text)
        self.assertIn("target_adapt_bex02_clean_target_8gpu", script_text)
        self.assertIn("BEX02_tadapt_clean", script_text)
        self.assertIn("run_target_adapt_bex02_sweep_8gpu.sh", script_text)

        self.assertIn("GPU_IDS:-0,1,2,3,4,5,6,7", sweep_text)
        self.assertIn("TARGET_CHANNEL_VIEW=\"${TARGET_CHANNEL_VIEW:-provided_satellite}\"", sweep_text)
        self.assertIn("--target_channel_view", sweep_text)
        self.assertIn('"${TARGET_CHANNEL_VIEW}"', sweep_text)
        self.assertIn("20,50,100", sweep_text)
        self.assertIn("safe,base,strong", sweep_text)
        self.assertIn("labeled,unlabeled", sweep_text)
        self.assertIn("--eval_sat_channel", sweep_text)
        self.assertIn("--eval_detail_every", sweep_text)

        self.assertIn("干净的目标域样本", doc_text)
        self.assertIn("--target_channel_view clean", doc_text)
        self.assertIn("run_target_adapt_bex02_clean_target_8gpu.sh", doc_text)
        self.assertIn("run_target_adapt_bex02_sweep_8gpu.sh", doc_text)
        self.assertIn("--dry-run", doc_text)

    def test_rx_tx_balanced_launcher_sets_per_transmitter_budgets_for_both_label_routes(self):
        script = ROOT / "code" / "scripts" / "run_target_adapt_bex02_rx_tx_balanced_8gpu.sh"
        sweep_script = ROOT / "code" / "scripts" / "run_target_adapt_bex02_sweep_8gpu.sh"
        doc = ROOT / "docs" / "target_domain_adaptation_rx_tx_balanced.md"

        script_text = script.read_text(encoding="utf-8")
        sweep_text = sweep_script.read_text(encoding="utf-8")
        doc_text = doc.read_text(encoding="utf-8")

        self.assertIn("TARGET_SAMPLES_PER_RX_TX=\"${TARGET_SAMPLES_PER_RX_TX:-2,3}\"", script_text)
        self.assertIn("BEX02_tadapt_rxtx", script_text)
        self.assertIn("target_adapt_bex02_rx_tx_balanced_8gpu", script_text)
        self.assertIn("--target_samples_per_rx_tx", sweep_text)
        self.assertIn("TARGET_SAMPLES_PER_RX_TX_CSV", sweep_text)
        self.assertIn("labeled,unlabeled", sweep_text)
        self.assertIn("20,50,100", sweep_text)
        self.assertIn("--eval_sat_channel", sweep_text)

        self.assertIn("每个目标接收机", doc_text)
        self.assertIn("每个发射机", doc_text)
        self.assertIn("2, 3", doc_text)
        self.assertIn("run_target_adapt_bex02_rx_tx_balanced_8gpu.sh", doc_text)

    def test_single_rx_tx5_tx10_launcher_runs_each_target_receiver_for_30_epochs(self):
        script = ROOT / "code" / "scripts" / "run_target_adapt_bex02_single_rx_rxtx_8gpu.sh"
        sweep_script = ROOT / "code" / "scripts" / "run_target_adapt_bex02_sweep_8gpu.sh"
        doc = ROOT / "docs" / "target_domain_adaptation_single_rx_rxtx.md"

        script_text = script.read_text(encoding="utf-8")
        sweep_text = sweep_script.read_text(encoding="utf-8")
        doc_text = doc.read_text(encoding="utf-8")

        self.assertIn("TARGET_LOADERS", script_text)
        self.assertIn("test_unseen_day_rx_7,test_unseen_day_rx_8,test_unseen_day_rx_9,test_unseen_day_rx_10,test_unseen_day_rx_11", script_text)
        self.assertIn("TARGET_SAMPLES_PER_RX_TX=\"${TARGET_SAMPLES_PER_RX_TX:-5,10}\"", script_text)
        self.assertIn("EPOCHS=\"${EPOCHS:-30}\"", script_text)
        self.assertIn("ADAPT_WEIGHTS=\"${ADAPT_WEIGHTS:-base}\"", script_text)
        self.assertIn("TARGET_LABEL_MODES=\"${TARGET_LABEL_MODES:-labeled,unlabeled}\"", script_text)
        self.assertIn("BEX02_tadapt_single_rx_rxtx", script_text)

        self.assertIn("TARGET_LOADERS_CSV", sweep_text)
        self.assertIn("--target-loaders", sweep_text)
        self.assertIn("target_loader", sweep_text)

        self.assertIn("RX 7, 8, 9, 10, 11", doc_text)
        self.assertIn("每个发射机 5, 10", doc_text)
        self.assertIn("30", doc_text)
        self.assertIn("20", doc_text)
        self.assertIn("run_target_adapt_bex02_single_rx_rxtx_8gpu.sh", doc_text)


if __name__ == "__main__":
    unittest.main()
