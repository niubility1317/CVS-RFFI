import sys
import tempfile
import unittest
import json
from collections import OrderedDict
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated.fed_trainer import FederatedTrainer, _client_seen_weighted_avg


class TinyClientDataset(Dataset):
    def __init__(self):
        self.index = []
        self.samples = []
        for rx in [0, 1]:
            for sig in range(4):
                tx = sig % 2
                self.index.append(SimpleNamespace(tx_i=tx, rx_i=rx, day_i=0, sig_i=sig))
                x = torch.zeros(2, 4)
                x[tx, :] = 1.0 + rx
                self.samples.append((x, tx, rx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, tx, rx = self.samples[idx]
        meta = {"tx_i": tx, "rx_i": rx, "day_i": 0}
        return x.clone(), tx, rx, meta


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 2)

    def forward(self, x, *args, **kwargs):
        return {"tx_logits": self.fc(x.flatten(1))}


class TinyMixStyleClassifier(TinyClassifier):
    def __init__(self):
        super().__init__()
        self.id_backbone = nn.Module()
        self.id_backbone.mixstyle = SimpleNamespace(p=0.5, strength=0.8)
        self.id_backbone.mixstyle_on = True


class TinyDGClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 2)
        self.dom = nn.Linear(8, 2)

    def forward(self, x, *args, **kwargs):
        feat = x.flatten(1)
        return {
            "tx_logits": self.fc(feat),
            "dom_logits": self.dom(feat),
            "adv_dom_logits": self.dom(feat),
            "z_id": feat,
            "z_dom": torch.flip(feat, dims=[1]),
        }


class TinyVMBClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 2)
        self.dom = nn.Linear(8, 2)
        self.rx = nn.Linear(8, 2)
        self.tx_adv = nn.Linear(8, 2)

    def forward(self, x, *args, **kwargs):
        feat = x.flatten(1)
        z_id = feat + 0.01
        z_dom = torch.flip(feat, dims=[1])
        return {
            "tx_logits": self.fc(z_id),
            "dom_logits": self.dom(z_dom),
            "adv_dom_logits": self.rx(z_id),
            "tx_adv_logits": self.tx_adv(z_dom),
            "z_id": z_id,
            "z_dom": z_dom,
        }


class TinyReceiverAgnosticClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 2)
        self.rx = nn.Linear(8, 2)

    def forward(self, x, *args, **kwargs):
        feat = x.flatten(1)
        return {
            "tx_logits": self.fc(feat),
            "rx_logits": self.rx(feat),
            "z_id": feat,
            "z_dom": torch.flip(feat, dims=[1]),
        }


def eval_loader(model, loader, device, domain_label_map=None, max_batches=0):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)["tx_logits"]
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
    return {"tx_acc": 100.0 * correct / max(1, total), "tx_correct": correct, "tx_total": total, "dom_acc": float("nan"), "probe_dom_acc": float("nan")}


class FederatedTrainerSmokeTest(unittest.TestCase):
    def _base_cfg(self, tmp):
        return SimpleNamespace(
            train_mode="fedprox",
            fl_local_objective="ce",
            fl_client_key="receiver",
            fl_rounds=1,
            fl_local_epochs=1,
            fl_clients_per_round=1.0,
            fl_agg_weight="uniform",
            fl_min_samples_per_client=1,
            fl_drop_small_clients=False,
            fl_verbose_clients=False,
            batch_size=2,
            num_workers=0,
            lr=0.01,
            wd=0.0,
            grad_clip=1.0,
            seed=7,
            eval_max_batches=0,
            output_dir=tmp,
            fedprox_mu=0.01,
        )

    def test_one_round_saves_fedprox_checkpoint(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            trainer = FederatedTrainer(
                TinyClassifier(),
                dataset,
                val_loader,
                {
                    "test_unseen_day_seen_rx": val_loader,
                    "test_seen_day_unseen_rx": val_loader,
                    "test_unseen_day_unseen_rx": val_loader,
                },
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=lambda model, loaders, device, domain_label_map=None, max_batches=0: {
                    name: eval_loader(model, loader, device) for name, loader in loaders.items()
                },
                named_test_meta={
                    "test_unseen_day_seen_rx": {"days_label": ["d2"], "rxs_idx": [0, 1]},
                    "test_seen_day_unseen_rx": {"days_label": ["d0"], "rxs_idx": [2]},
                    "test_unseen_day_unseen_rx": {"days_label": ["d2"], "rxs_idx": [2]},
                },
            )

            captured = StringIO()
            with redirect_stdout(captured):
                summary = trainer.train()

            self.assertEqual(summary["best_round"], 1)
            self.assertEqual(summary["train_mode"], "fedprox")
            self.assertTrue((Path(tmp) / "last_checkpoint.pt").is_file())
            metrics_text = (Path(tmp) / "metrics.csv").read_text(encoding="utf-8")
            self.assertIn("train_loss_fedprox", metrics_text)
            self.assertIn("named_test_tx_acc_json", metrics_text)
            self.assertIn("round_time_s", metrics_text)
            self.assertIn("round_train_time_s", metrics_text)
            self.assertIn("round_eval_time_s", metrics_text)
            self.assertIn("round_test_time_s", metrics_text)
            self.assertIn("round_extra_eval_time_s", metrics_text)
            log_text = (Path(tmp) / "logs.jsonl").read_text(encoding="utf-8")
            self.assertIn("client_loss_fedprox_avg", log_text)
            self.assertIn("client_pred_hist_sum", log_text)
            self.assertIn("global_named_test", log_text)
            self.assertIn("global_test_overall", log_text)
            self.assertIn("round_time_s", log_text)
            self.assertIn("global_eval_timing", log_text)
            self.assertIn('"event": "fed_config"', log_text)
            summary_text = (Path(tmp) / "summary.json").read_text(encoding="utf-8")
            self.assertIn("best_eval", summary_text)
            self.assertIn("last_eval", summary_text)
            config_path = Path(tmp) / "federated_config.json"
            self.assertTrue(config_path.is_file())
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for section in ["run", "runtime", "data", "federated", "stylebank", "protobank", "grl", "losses", "satellite", "evaluation"]:
                self.assertIn(section, config)
            self.assertIn("torch_num_threads", config["runtime"])
            self.assertIn("omp_num_threads", config["runtime"])
            self.assertEqual(config["federated"]["train_mode"], "fedprox")
            self.assertEqual(config["stylebank"]["style_domain_semantics"], "constructed_style_view_id")
            self.assertEqual(config["stylebank"]["d_raw_semantics"], "mapped_target_receiver_domain_for_each_view")
            self.assertIn("client_num_samples", config["client_splits"])
            stdout = captured.getvalue()
            self.assertIn("[FED-CONFIG-BEGIN]", stdout)
            self.assertIn("[FED-CONFIG-RUNTIME]", stdout)
            self.assertIn("torch_threads=", stdout)
            self.assertIn("[FED-CONFIG-STYLEBANK]", stdout)
            self.assertIn("style_domain=constructed_style_view_id", stdout)
            self.assertIn("d_raw=mapped_target_receiver_domain_for_each_view", stdout)
            self.assertIn("[FED-CONFIG-GRL]", stdout)
            self.assertIn("[FED-CONFIG-PROTOBANK]", stdout)
            self.assertIn("[FED-CONFIG-SAT-SPLITS] count=3", stdout)
            self.assertIn(
                "splits=test_unseen_day_seen_rx,test_seen_day_unseen_rx,test_unseen_day_unseen_rx",
                stdout,
            )
            self.assertIn("=" * 120, stdout)
            self.assertIn("[FED-ROUND-BEGIN][R001/001]", stdout)
            self.assertIn("[FED-TEST][R001]", stdout)
            self.assertIn("[FED-TEST-SPLIT][R001]", stdout)
            self.assertIn("[FED-ROUND-END][R001/001]", stdout)
            self.assertIn("[FED-TIME][R001]", stdout)
            self.assertIn("train=", stdout)
            self.assertIn("eval=", stdout)
            self.assertIn("-" * 120, stdout)
            self.assertIn("test=", stdout)
            self.assertIn("unseen_day_seen_rx", stdout)
            self.assertIn("seen_day_unseen_rx", stdout)

    def test_vmb_stage1_domain_unsup_pretrain_logs_bridge_metrics(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            cfg.train_mode = "fedcvs_vmb"
            cfg.fl_local_objective = "receiver_agnostic_bex02"
            cfg.fl_vmb_stage = "auto"
            cfg.fl_vmb_pretrain_rounds = 1
            cfg.fl_vmb_stage1_local_steps = 1
            cfg.fl_vmb_stage1_objective = "domain_unsup_pretrain"
            cfg.fl_vmb_stage1_use_aux_losses = True
            cfg.lambda_domain_unsup_pretrain = 0.2
            cfg.lambda_domain_unsup_metadata_ce = 0.5
            cfg.lambda_domain_unsup_var = 0.05
            cfg.domain_unsup_pretrain_method = "metadata_consistency"
            cfg.domain_unsup_noise_std = 0.01
            cfg.domain_unsup_amp_jitter = 0.03
            cfg.domain_unsup_logit_cons_weight = 0.1
            cfg.domain_unsup_client_compact_weight = 0.5
            cfg.fl_domain_pretrain_train_scope = "all"
            cfg.lambda_rx_adv = 0.0
            cfg.grl_lambda = 1.0
            cfg.num_classes = 2
            trainer = FederatedTrainer(
                TinyVMBClassifier(),
                dataset,
                val_loader,
                {"test_unseen_day_unseen_rx": val_loader},
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=lambda model, loaders, device, domain_label_map=None, max_batches=0: {
                    name: eval_loader(model, loader, device) for name, loader in loaders.items()
                },
                named_test_meta={"test_unseen_day_unseen_rx": {"days_label": ["d2"], "rxs_idx": [2]}},
            )

            captured = StringIO()
            with redirect_stdout(captured):
                summary = trainer.train()

            self.assertEqual(summary["best_round"], 1)
            metrics_text = (Path(tmp) / "metrics.csv").read_text(encoding="utf-8")
            self.assertIn("loss_domain_unsup_pretrain", metrics_text)
            self.assertIn("domain_unsup_active", metrics_text)
            self.assertIn("domain_unsup_client_compact", metrics_text)
            log_text = (Path(tmp) / "logs.jsonl").read_text(encoding="utf-8")
            self.assertIn("client_loss_domain_unsup_pretrain_avg", log_text)
            self.assertIn("client_domain_unsup_client_compact_avg", log_text)
            self.assertIn("client_stage1_domain_pretrain_active_rate", log_text)
            config = json.loads((Path(tmp) / "federated_config.json").read_text(encoding="utf-8"))
            self.assertTrue(config["domain_pretrain"]["enabled"])
            self.assertEqual(config["domain_pretrain"]["method"], "metadata_consistency")
            self.assertEqual(config["domain_pretrain"]["train_scope"], "all")
            self.assertEqual(config["domain_pretrain"]["domain_unsup_client_compact_weight"], 0.5)
            stdout = captured.getvalue()
            self.assertIn("[FED-DOMAIN-PRETRAIN][R001]", stdout)
            self.assertIn("client_compact=", stdout)
            self.assertIn("unseen_day_unseen_rx", stdout)

    def test_training_uses_partial_client_batches_and_seen_weighted_metrics(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            cfg.batch_size = 3
            trainer = FederatedTrainer(
                TinyClassifier(),
                dataset,
                val_loader,
                {"test_unseen_day_unseen_rx": val_loader},
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=lambda model, loaders, device, domain_label_map=None, max_batches=0: {
                    name: eval_loader(model, loader, device) for name, loader in loaders.items()
                },
            )

            result = trainer.train_one_client("rx0", 1)

            self.assertEqual(result["seen"], 4)
            self.assertAlmostEqual(
                _client_seen_weighted_avg(
                    {
                        "small": {"loss": 10.0, "seen": 1},
                        "large": {"loss": 2.0, "seen": 9},
                    },
                    "loss",
                ),
                2.8,
            )

    def test_federated_round_controls_lr_and_mixstyle_schedule(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            cfg.fl_rounds = 3
            cfg.lr = 0.01
            cfg.lr_min = 0.001
            cfg.use_mixstyle = True
            cfg.mixstyle_p = 0.5
            cfg.mixstyle_strength = 0.8
            cfg.mixstyle_late_start = 1
            cfg.mixstyle_late_ramp_epochs = 2
            cfg.mixstyle_late_min_p = 0.1
            cfg.mixstyle_late_min_strength = 0.2
            cfg.mixstyle_stop_epoch = 0
            trainer = FederatedTrainer(
                TinyMixStyleClassifier(),
                dataset,
                val_loader,
                {"test_unseen_day_unseen_rx": val_loader},
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=lambda model, loaders, device, domain_label_map=None, max_batches=0: {
                    name: eval_loader(model, loader, device) for name, loader in loaders.items()
                },
            )

            self.assertAlmostEqual(trainer._round_lr(1), 0.01)
            self.assertAlmostEqual(trainer._round_lr(3), 0.001)
            state1 = trainer._configure_mixstyle_for_round(1)
            state3 = trainer._configure_mixstyle_for_round(3)

            self.assertEqual(state1["phase"], "base")
            self.assertEqual(state3["phase"], "late_anneal")
            self.assertAlmostEqual(state3["p"], 0.1)
            self.assertAlmostEqual(state3["strength"], 0.2)

    def test_fedprox_local_training_reports_positive_proximal_term(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            cfg.fl_local_epochs = 2
            cfg.fedprox_mu = 10.0
            trainer = FederatedTrainer(
                TinyClassifier(),
                dataset,
                val_loader,
                {"test_unseen_day_unseen_rx": val_loader},
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=lambda model, loaders, device, domain_label_map=None, max_batches=0: {
                    name: eval_loader(model, loader, device) for name, loader in loaders.items()
                },
            )

            result = trainer.train_one_client("rx0", 1)

            self.assertIn("loss_fedprox", result)
            self.assertGreater(result["loss_fedprox"], 0.0)
            self.assertEqual(sum(result["label_hist"]), result["seen"])
            self.assertEqual(sum(result["pred_hist"]), result["seen"])

    def test_bex02_dg_objective_logs_fishr_and_satellite_terms(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            cfg.train_mode = "fedavg"
            cfg.fl_local_objective = "bex02_dg"
            cfg.fedprox_mu = 0.0
            cfg.label_smoothing = 0.0
            cfg.grl_lambda = 1.0
            cfg.lambda_dom = 0.1
            cfg.lambda_adv = 0.1
            cfg.lambda_orth = 0.1
            cfg.lambda_cons = 0.1
            cfg.lambda_group_ce = 0.1
            cfg.lambda_fishr = 0.1
            cfg.fishr_min_domains = 1
            cfg.group_ce_min_domains = 1
            cfg.group_ce_top_frac = 1.0
            cfg.group_ce_mode = "hard"
            cfg.use_sat_consistency = True
            cfg.sat_cons_start_epoch = 1
            cfg.sat_train_scenario = "mixed_orbit"
            cfg.lambda_sat_cls = 0.1
            cfg.lambda_sat_cons = 0.1
            cfg.use_aug = False
            cfg.use_mixstyle = True
            trainer = FederatedTrainer(
                TinyDGClassifier(),
                dataset,
                val_loader,
                {"test_unseen_day_unseen_rx": val_loader},
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=lambda model, loaders, device, domain_label_map=None, max_batches=0: {
                    name: eval_loader(model, loader, device) for name, loader in loaders.items()
                },
                domain_label_map={0: 0, 1: 1},
                sat_transform_fn=lambda x, scenario, round_idx, batch_idx: x + 0.05,
            )

            result = trainer.train_one_client("rx0", 1)

            self.assertIn("loss_fishr", result)
            self.assertIn("loss_sat_cls", result)
            self.assertIn("loss_sat_cons", result)
            self.assertGreaterEqual(result["loss_fishr"], 0.0)

    def test_bex02_dg_training_logs_activation_diagnostics(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            cfg.train_mode = "fedavg"
            cfg.fl_local_objective = "bex02_dg"
            cfg.fl_rounds = 1
            cfg.fedprox_mu = 0.0
            cfg.label_smoothing = 0.0
            cfg.grl_lambda = 1.0
            cfg.lambda_dom = 0.1
            cfg.lambda_adv = 0.1
            cfg.lambda_orth = 0.1
            cfg.lambda_cons = 0.1
            cfg.lambda_group_ce = 0.1
            cfg.lambda_fishr = 0.1
            cfg.fishr_min_domains = 1
            cfg.group_ce_min_domains = 1
            cfg.group_ce_top_frac = 1.0
            cfg.group_ce_mode = "hard"
            cfg.use_sat_consistency = True
            cfg.sat_cons_start_epoch = 1
            cfg.sat_train_scenario = "mixed_orbit"
            cfg.lambda_sat_cls = 0.1
            cfg.lambda_sat_cons = 0.1
            cfg.use_aug = False
            cfg.use_mixstyle = False
            trainer = FederatedTrainer(
                TinyDGClassifier(),
                dataset,
                val_loader,
                {"test_unseen_day_unseen_rx": val_loader},
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=lambda model, loaders, device, domain_label_map=None, max_batches=0: {
                    name: eval_loader(model, loader, device) for name, loader in loaders.items()
                },
                domain_label_map={0: 0, 1: 1},
                sat_transform_fn=lambda x, scenario, round_idx, batch_idx: x + 0.05,
            )

            captured = StringIO()
            with redirect_stdout(captured):
                trainer.train()

            log_text = (Path(tmp) / "logs.jsonl").read_text(encoding="utf-8")
            metrics_text = (Path(tmp) / "metrics.csv").read_text(encoding="utf-8")
            stdout = captured.getvalue()
            self.assertIn("client_diag_domain_count_avg", log_text)
            self.assertIn("client_diag_fishr_active_rate", log_text)
            self.assertIn("client_diag_sat_aug_active_rate", log_text)
            self.assertIn("diag_domain_count", metrics_text)
            self.assertIn("diag_fishr_active", metrics_text)
            self.assertIn("zdom_target_acc", metrics_text)
            self.assertIn("[FED-DG-DIAG][R001]", stdout)
            self.assertIn("grl_rx_adv_active", stdout)
            self.assertIn("zdom_acc", stdout)

    def test_fed_fishr_training_logs_global_summary_for_single_domain_clients(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            cfg.train_mode = "fedavg"
            cfg.fl_local_objective = "ce"
            cfg.fl_rounds = 1
            cfg.fl_clients_per_round = 1.0
            cfg.batch_size = 4
            cfg.num_classes = 2
            cfg.fedprox_mu = 0.0
            cfg.lambda_fishr = 0.0
            cfg.use_fed_fishr = True
            cfg.lambda_fed_fishr = 0.6
            cfg.fed_fishr_mode = "reweight"
            cfg.fed_fishr_gradient_scope = "classifier_head"
            cfg.fed_fishr_start_round = 1
            cfg.fed_fishr_min_clients = 2
            cfg.fed_fishr_min_count = 2
            cfg.fed_fishr_max_samples_per_class = 2
            cfg.fed_fishr_sketch_dim = 0
            cfg.fed_fishr_momentum = 0.0
            cfg.fed_fishr_reweight_floor = 0.05
            cfg.fed_fishr_reweight_cap = 0.95
            trainer = FederatedTrainer(
                TinyDGClassifier(),
                dataset,
                val_loader,
                {"test_unseen_day_unseen_rx": val_loader},
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=lambda model, loaders, device, domain_label_map=None, max_batches=0: {
                    name: eval_loader(model, loader, device) for name, loader in loaders.items()
                },
                domain_label_map={0: 0, 1: 1},
            )

            captured = StringIO()
            with redirect_stdout(captured):
                summary = trainer.train()

            self.assertTrue(summary["global_fed_fishr_summary"]["enabled"])
            self.assertTrue(summary["global_fed_fishr_summary"]["active"])
            self.assertEqual(summary["global_fed_fishr_summary"]["active_classes"], 2)
            self.assertEqual(summary["global_fed_fishr_summary"]["client_count"], 2)
            log_row = json.loads((Path(tmp) / "logs.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            self.assertTrue(log_row["global_fed_fishr_summary"]["active"])
            self.assertTrue(log_row["global_fed_fishr_summary"]["reweight_active"])
            self.assertGreaterEqual(log_row["global_fed_fishr_summary"]["weight_max_delta"], 0.0)
            self.assertEqual(log_row["client_diag_fishr_active_rate"], 0.0)
            metrics_text = (Path(tmp) / "metrics.csv").read_text(encoding="utf-8")
            self.assertIn("global_fed_fishr_active", metrics_text)
            self.assertIn("global_fed_fishr_active_classes", metrics_text)
            self.assertIn("fed_fishr_payload_bytes", metrics_text)
            self.assertIn("[FED-FISHR][R001]", captured.getvalue())

    def test_fedcvs_vmb_training_uses_server_gradient_path_and_logs_diagnostics(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            cfg.train_mode = "fedcvs_vmb"
            cfg.fl_local_objective = "receiver_agnostic_bex02"
            cfg.num_classes = 2
            cfg.fl_rounds = 1
            cfg.fl_clients_per_round = 1.0
            cfg.fl_vmb_stage = "stage2"
            cfg.fl_vmb_pretrain_rounds = 0
            cfg.fl_vmb_batches_per_client = 1
            cfg.fl_vmb_server_lr = 0.05
            cfg.fl_vmb_server_momentum = 0.0
            cfg.fl_vmb_weight_decay = 0.0
            cfg.fl_vmb_domain_balanced_sampling = True
            cfg.fl_vmb_domain_balanced_aggregation = True
            cfg.fl_vmb_transmitter_balanced_batch = True
            cfg.fl_vmb_freeze_rx_stage2 = True
            cfg.fl_vmb_prototype_ema = 0.5
            cfg.fl_vmb_prototype_clip_norm = 1.0
            cfg.tau_vmb_tx = 0.1
            cfg.tau_vmb_rx = 0.1
            cfg.lambda_vmb_tx_proto = 0.0
            cfg.lambda_vmb_rx_proto = 0.0
            cfg.lambda_tx_adv_r = 0.1
            cfg.fl_vmb_adv_warmup_rounds = 0
            cfg.fedprox_mu = 0.0
            cfg.label_smoothing = 0.0
            cfg.grl_lambda = 1.0
            cfg.lambda_rx_adv = 0.1
            cfg.lambda_dom = 0.0
            cfg.lambda_adv = 0.0
            cfg.lambda_orth = 0.0
            cfg.lambda_cons = 0.0
            cfg.lambda_group_ce = 0.0
            cfg.lambda_fishr = 0.0
            cfg.fishr_min_domains = 1
            cfg.group_ce_min_domains = 1
            cfg.group_ce_top_frac = 1.0
            cfg.group_ce_mode = "hard"
            cfg.use_sat_consistency = False
            cfg.use_aug = False
            cfg.use_mixstyle = False
            cfg.use_fed_style_bank = True
            cfg.fl_style_code_dim = 4
            trainer = FederatedTrainer(
                TinyVMBClassifier(),
                dataset,
                val_loader,
                {"test_unseen_day_unseen_rx": val_loader},
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=lambda model, loaders, device, domain_label_map=None, max_batches=0: {
                    name: eval_loader(model, loader, device) for name, loader in loaders.items()
                },
                domain_label_map={0: 0, 1: 1},
            )

            captured = StringIO()
            with redirect_stdout(captured):
                summary = trainer.train()

            self.assertEqual(summary["train_mode"], "fedcvs_vmb")
            self.assertTrue((Path(tmp) / "last_checkpoint.pt").is_file())
            self.assertTrue(summary["global_vmb_proto_summary"]["enabled"])
            metrics_text = (Path(tmp) / "metrics.csv").read_text(encoding="utf-8")
            self.assertIn("vmb_grad_norm", metrics_text)
            self.assertIn("train_loss_tx_adv_r", metrics_text)
            log_text = (Path(tmp) / "logs.jsonl").read_text(encoding="utf-8")
            self.assertIn("vmb_server_update", log_text)
            self.assertIn("global_vmb_proto_summary", log_text)
            config = json.loads((Path(tmp) / "federated_config.json").read_text(encoding="utf-8"))
            self.assertTrue(config["fedcvs_vmb"]["enabled"])
            self.assertEqual(config["fedcvs_vmb"]["method"], "FedCVS-RFFI-VMB")
            stdout = captured.getvalue()
            self.assertIn("[FED-CONFIG-VMB]", stdout)
            self.assertIn("[FED-VMB][R001]", stdout)

    def test_fedcvs_vmb_auto_stage1_uses_local_pretrain_state_average(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            cfg.train_mode = "fedcvs_vmb"
            cfg.fl_local_objective = "receiver_agnostic_bex02"
            cfg.num_classes = 2
            cfg.fl_rounds = 1
            cfg.fl_clients_per_round = 1.0
            cfg.fl_vmb_stage = "auto"
            cfg.fl_vmb_pretrain_rounds = 1
            cfg.fl_vmb_stage1_local_steps = 1
            cfg.fl_vmb_batches_per_client = 1
            cfg.fl_vmb_server_lr = 0.05
            cfg.fl_vmb_server_momentum = 0.0
            cfg.fl_vmb_weight_decay = 0.0
            cfg.fl_vmb_domain_balanced_sampling = True
            cfg.fl_vmb_domain_balanced_aggregation = True
            cfg.fl_vmb_transmitter_balanced_batch = True
            cfg.fl_vmb_freeze_rx_stage2 = True
            cfg.fl_vmb_prototype_ema = 0.5
            cfg.fl_vmb_prototype_clip_norm = 1.0
            cfg.tau_vmb_tx = 0.1
            cfg.tau_vmb_rx = 0.1
            cfg.lambda_vmb_tx_proto = 0.0
            cfg.lambda_vmb_rx_proto = 0.0
            cfg.lambda_tx_adv_r = 0.1
            cfg.fl_vmb_adv_warmup_rounds = 0
            cfg.fedprox_mu = 0.0
            cfg.label_smoothing = 0.0
            cfg.grl_lambda = 1.0
            cfg.lambda_rx_adv = 0.1
            cfg.lambda_dom = 0.0
            cfg.lambda_adv = 0.0
            cfg.lambda_orth = 0.0
            cfg.lambda_cons = 0.0
            cfg.lambda_group_ce = 0.0
            cfg.lambda_fishr = 0.0
            cfg.fishr_min_domains = 1
            cfg.group_ce_min_domains = 1
            cfg.group_ce_top_frac = 1.0
            cfg.group_ce_mode = "hard"
            cfg.use_sat_consistency = False
            cfg.use_aug = False
            cfg.use_mixstyle = False
            cfg.use_fed_style_bank = True
            cfg.fl_style_code_dim = 4
            trainer = FederatedTrainer(
                TinyVMBClassifier(),
                dataset,
                val_loader,
                {"test_unseen_day_unseen_rx": val_loader},
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=lambda model, loaders, device, domain_label_map=None, max_batches=0: {
                    name: eval_loader(model, loader, device) for name, loader in loaders.items()
                },
                domain_label_map={0: 0, 1: 1},
            )

            trainer.train()

            log_text = (Path(tmp) / "logs.jsonl").read_text(encoding="utf-8")
            metrics_text = (Path(tmp) / "metrics.csv").read_text(encoding="utf-8")
            self.assertIn('"vmb_stage": "stage1"', log_text)
            self.assertIn('"mode": "state_average"', log_text)
            self.assertIn('"num_packets_seen": 2', log_text)
            self.assertIn("vmb_client_drift_norm", metrics_text)
            self.assertIn("vmb_comm_payload_bytes", metrics_text)

    def test_split_bex02_route_logs_kd_tokens_and_conflict_aggregation(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            cfg.train_mode = "split_bex02"
            cfg.fl_local_objective = "local_virtual_bex02"
            cfg.num_classes = 2
            cfg.fl_rounds = 1
            cfg.fl_clients_per_round = 1.0
            cfg.fl_vmb_stage = "stage2"
            cfg.fl_vmb_pretrain_rounds = 0
            cfg.fl_vmb_batches_per_client = 1
            cfg.fl_vmb_server_lr = 0.05
            cfg.fl_vmb_server_momentum = 0.0
            cfg.fl_vmb_weight_decay = 0.0
            cfg.fl_vmb_domain_balanced_sampling = True
            cfg.fl_vmb_domain_balanced_aggregation = True
            cfg.fl_vmb_transmitter_balanced_batch = True
            cfg.fl_vmb_freeze_rx_stage2 = True
            cfg.fl_vmb_prototype_ema = 0.5
            cfg.fl_vmb_prototype_clip_norm = 1.0
            cfg.tau_vmb_tx = 0.1
            cfg.tau_vmb_rx = 0.1
            cfg.lambda_vmb_tx_proto = 0.0
            cfg.lambda_vmb_rx_proto = 0.0
            cfg.lambda_tx_adv_r = 0.0
            cfg.fl_vmb_adv_warmup_rounds = 0
            cfg.fl_conflict_agg = "cosine_clip"
            cfg.fedprox_mu = 0.0
            cfg.label_smoothing = 0.0
            cfg.grl_lambda = 1.0
            cfg.lambda_rx_adv = 0.0
            cfg.lambda_dom = 0.0
            cfg.lambda_adv = 0.0
            cfg.lambda_orth = 0.0
            cfg.lambda_cons = 0.0
            cfg.lambda_group_ce = 0.0
            cfg.lambda_fishr = 0.0
            cfg.fishr_min_domains = 1
            cfg.group_ce_min_domains = 1
            cfg.group_ce_top_frac = 1.0
            cfg.group_ce_mode = "hard"
            cfg.use_sat_consistency = False
            cfg.use_aug = False
            cfg.use_mixstyle = False
            cfg.use_fed_style_bank = True
            cfg.use_logit_anchors = True
            cfg.lambda_logit_kd = 0.25
            cfg.kd_temperature = 2.0
            cfg.kd_reliability_gate = 0.0
            cfg.kd_margin_min = 0.0
            cfg.kd_anchor_ema = 0.0
            cfg.kd_min_count = 1
            cfg.activation_token_route = "quantized"
            cfg.token_quant_bits = 4
            cfg.token_sketch_dim = 4
            cfg.token_rank = 1
            cfg.split_layer = "z_id"
            cfg.fl_style_code_dim = 4
            cfg.fl_probe_every = 1
            cfg.feature_probe_export = "features_probe.pt"
            cfg.probe_max_samples = 8
            trainer = FederatedTrainer(
                TinyVMBClassifier(),
                dataset,
                val_loader,
                {"test_unseen_day_unseen_rx": val_loader},
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=lambda model, loaders, device, domain_label_map=None, max_batches=0: {
                    name: eval_loader(model, loader, device) for name, loader in loaders.items()
                },
                domain_label_map={0: 0, 1: 1},
            )

            captured = StringIO()
            with redirect_stdout(captured):
                summary = trainer.train()

            self.assertEqual(summary["train_mode"], "split_bex02")
            config = json.loads((Path(tmp) / "federated_config.json").read_text(encoding="utf-8"))
            self.assertTrue(config["distillation"]["enabled"])
            self.assertEqual(config["compression"]["activation_token_route"], "quantized")
            self.assertEqual(config["conflict_aggregation"]["mode"], "cosine_clip")
            self.assertEqual(config["feature_probe"]["probe_every"], 1)
            log_text = (Path(tmp) / "logs.jsonl").read_text(encoding="utf-8")
            metrics_text = (Path(tmp) / "metrics.csv").read_text(encoding="utf-8")
            stdout = captured.getvalue()
            self.assertIn("global_logit_anchor_summary", log_text)
            self.assertIn("activation_token_payload_bytes", log_text)
            self.assertIn("vmb_conflict_summary", log_text)
            self.assertIn("global_feature_probe_summary", log_text)
            self.assertIn('"num_packets_seen": 2', log_text)
            self.assertIn("train_loss_logit_kd", metrics_text)
            self.assertIn("activation_token_payload_bytes", metrics_text)
            self.assertIn("vmb_conflicts_resolved", metrics_text)
            self.assertIn("feature_probe_samples", metrics_text)
            probe_files = sorted(Path(tmp).glob("features_probe*.pt"))
            self.assertTrue(probe_files, "expected online feature-probe export")
            probe_payload = torch.load(probe_files[0], map_location="cpu", weights_only=False)
            self.assertIn("z_t", probe_payload)
            self.assertIn("z_r", probe_payload)
            self.assertIn("tx", probe_payload)
            self.assertIn("rx", probe_payload)
            self.assertGreater(int(probe_payload["tx"].numel()), 0)
            self.assertIn("[FED-CONFIG-DISTILL]", stdout)
            self.assertIn("[FED-CONFIG-SPLIT]", stdout)
            self.assertIn("[FED-FEATURE-PROBE]", stdout)

    def test_extra_satellite_eval_runs_and_prints_every_round(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        eval_rounds = []
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            cfg.fl_rounds = 2
            cfg.fl_test_eval_interval = 1
            cfg.eval_sat_channel = True
            cfg.eval_sat_on = "main"
            cfg.eval_sat_scenarios = "clear_leo"
            trainer = FederatedTrainer(
                TinyClassifier(),
                dataset,
                val_loader,
                {
                    "test_unseen_day_seen_rx": val_loader,
                    "test_seen_day_unseen_rx": val_loader,
                    "test_unseen_day_unseen_rx": val_loader,
                },
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=lambda model, loaders, device, domain_label_map=None, max_batches=0: {
                    name: eval_loader(model, loader, device) for name, loader in loaders.items()
                },
                extra_eval_fn=lambda model, device, round_idx: eval_rounds.append(int(round_idx)) or {
                    "sat_channel": {
                        "clear_leo": {
                            "aggregate": {"tx_acc": 12.5, "tx_correct": 1, "tx_total": 8},
                            "strict_udu": 12.5,
                            "selected_names": [
                                "test_unseen_day_seen_rx",
                                "test_seen_day_unseen_rx",
                                "test_unseen_day_unseen_rx",
                            ],
                            "named": {
                                "test_unseen_day_seen_rx": {"tx_acc": 12.5, "tx_correct": 1, "tx_total": 8},
                                "test_seen_day_unseen_rx": {"tx_acc": 12.5, "tx_correct": 1, "tx_total": 8},
                                "test_unseen_day_unseen_rx": {"tx_acc": 12.5, "tx_correct": 1, "tx_total": 8},
                            },
                        }
                    }
                },
            )

            captured = StringIO()
            with redirect_stdout(captured):
                trainer.train()

            self.assertEqual(eval_rounds, [1, 2])
            stdout = captured.getvalue()
            self.assertIn("[FED-SAT-TEST][R001]", stdout)
            self.assertIn("[FED-SAT-TEST][R002]", stdout)
            self.assertIn("[SAT-TEST-SPLIT] scenario=clear_leo test_unseen_day_seen_rx", stdout)
            log_text = (Path(tmp) / "logs.jsonl").read_text(encoding="utf-8")
            self.assertIn("global_extra_tests", log_text)
            self.assertIn("clear_leo", log_text)

    def test_heavy_federated_eval_can_run_every_interval_then_last_rounds(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        named_rounds = []
        extra_rounds = []
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            cfg.fl_rounds = 12
            cfg.fl_test_eval_interval = 10
            cfg.fl_test_eval_last_n = 2
            cfg.fl_test_eval_final_offsets = ""

            def scheduled_named_eval(model, loaders, device, domain_label_map=None, max_batches=0):
                named_rounds.append(int(trainer._current_eval_round))
                return {name: eval_loader(model, loader, device) for name, loader in loaders.items()}

            trainer = FederatedTrainer(
                TinyClassifier(),
                dataset,
                val_loader,
                {"test_unseen_day_unseen_rx": val_loader},
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=scheduled_named_eval,
                extra_eval_fn=lambda model, device, round_idx: extra_rounds.append(int(round_idx)) or {
                    "sat_channel": {"clear_leo": {"aggregate": {"tx_acc": 10.0, "tx_correct": 1, "tx_total": 10}}}
                },
            )

            captured = StringIO()
            with redirect_stdout(captured):
                trainer.train()

            self.assertEqual(named_rounds, [10, 11, 12])
            self.assertEqual(extra_rounds, [10, 11, 12])
            stdout = captured.getvalue()
            self.assertIn("[FED-TEST-SKIP][R001] next=R010", stdout)
            self.assertIn("[FED-SAT-TEST][R010]", stdout)
            log_rows = [
                json.loads(line)
                for line in (Path(tmp) / "logs.jsonl").read_text(encoding="utf-8").splitlines()
                if '"round"' in line
            ]
            self.assertFalse(log_rows[0]["global_test_eval_ran"])
            self.assertTrue(log_rows[9]["global_test_eval_ran"])

    def test_default_heavy_federated_eval_runs_on_final_fifth_third_and_last_rounds(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        named_rounds = []
        extra_rounds = []
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            cfg.fl_rounds = 12

            def scheduled_named_eval(model, loaders, device, domain_label_map=None, max_batches=0):
                named_rounds.append(int(trainer._current_eval_round))
                return {name: eval_loader(model, loader, device) for name, loader in loaders.items()}

            trainer = FederatedTrainer(
                TinyClassifier(),
                dataset,
                val_loader,
                {"test_unseen_day_unseen_rx": val_loader},
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=scheduled_named_eval,
                extra_eval_fn=lambda model, device, round_idx: extra_rounds.append(int(round_idx)) or {
                    "sat_channel": {"clear_leo": {"aggregate": {"tx_acc": 10.0, "tx_correct": 1, "tx_total": 10}}}
                },
            )

            captured = StringIO()
            with redirect_stdout(captured):
                trainer.train()

            self.assertEqual(named_rounds, [8, 10, 12])
            self.assertEqual(extra_rounds, [8, 10, 12])
            stdout = captured.getvalue()
            self.assertIn("[FED-TEST-SKIP][R001] next=R008", stdout)
            self.assertIn("[FED-SAT-TEST][R008]", stdout)
            self.assertIn("[FED-SAT-TEST][R010]", stdout)
            self.assertIn("[FED-SAT-TEST][R012]", stdout)

    def test_receiver_agnostic_bex02_logs_grl_receiver_and_baseline_sat_view_terms(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        sat_calls = []
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            cfg.train_mode = "fedprox"
            cfg.fl_local_objective = "receiver_agnostic_bex02"
            cfg.fl_sat_aug_mode = "baseline_view"
            cfg.fedprox_mu = 0.01
            cfg.label_smoothing = 0.0
            cfg.grl_lambda = 1.0
            cfg.lambda_dom = 0.0
            cfg.lambda_adv = 0.0
            cfg.lambda_orth = 0.0
            cfg.lambda_cons = 0.0
            cfg.lambda_group_ce = 0.0
            cfg.lambda_fishr = 0.0
            cfg.fishr_min_domains = 1
            cfg.group_ce_min_domains = 1
            cfg.group_ce_top_frac = 1.0
            cfg.group_ce_mode = "hard"
            cfg.use_sat_consistency = True
            cfg.sat_cons_start_epoch = 1
            cfg.sat_train_scenario = "mixed_orbit"
            cfg.sat_train_scenarios = "clear_leo"
            cfg.lambda_sat_cls = 0.0
            cfg.lambda_sat_cons = 0.0
            cfg.lambda_rx_adv = 0.7
            cfg.rx_weight = 0.7
            cfg.use_aug = False
            cfg.use_mixstyle = False

            def sat_transform(x, scenario, round_idx, batch_idx):
                sat_calls.append((scenario, int(round_idx), int(batch_idx), int(x.size(0))))
                return x + 0.05

            trainer = FederatedTrainer(
                TinyReceiverAgnosticClassifier(),
                dataset,
                val_loader,
                {"test_unseen_day_unseen_rx": val_loader},
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=lambda model, loaders, device, domain_label_map=None, max_batches=0: {
                    name: eval_loader(model, loader, device) for name, loader in loaders.items()
                },
                domain_label_map={0: 0, 1: 1},
                sat_transform_fn=sat_transform,
            )

            result = trainer.train_one_client("rx0", 1)

            self.assertIn("loss_rx_adv", result)
            self.assertIn("loss_baseline_sat_view", result)
            self.assertGreater(result["loss_rx_adv"], 0.0)
            self.assertGreater(result["loss_baseline_sat_view"], 0.0)
            self.assertTrue(sat_calls)
            self.assertTrue(all(call[0] == "clear_leo" for call in sat_calls))

    def test_fedcgrl_training_logs_dynamic_receiver_adv_lambda(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            cfg.train_mode = "fedprox"
            cfg.fl_local_objective = "receiver_agnostic_bex02"
            cfg.num_classes = 2
            cfg.fl_rounds = 1
            cfg.fl_clients_per_round = 1.0
            cfg.fedprox_mu = 0.0
            cfg.label_smoothing = 0.0
            cfg.grl_lambda = 1.0
            cfg.lambda_dom = 0.0
            cfg.lambda_adv = 0.0
            cfg.lambda_orth = 0.0
            cfg.lambda_cons = 0.0
            cfg.lambda_group_ce = 0.0
            cfg.lambda_fishr = 0.0
            cfg.fishr_min_domains = 1
            cfg.group_ce_min_domains = 1
            cfg.group_ce_top_frac = 1.0
            cfg.group_ce_mode = "hard"
            cfg.use_sat_consistency = False
            cfg.use_aug = False
            cfg.use_mixstyle = False
            cfg.lambda_rx_adv = 1.0
            cfg.use_fed_cgrl = True
            cfg.fed_cgrl_base_lambda = 0.5
            cfg.fed_cgrl_min_lambda = 0.05
            cfg.fed_cgrl_max_lambda = 1.5
            cfg.fed_cgrl_warmup_rounds = 1
            cfg.fed_cgrl_leak_target_acc = 20.0
            cfg.fed_cgrl_leak_gain = 0.5
            cfg.fed_cgrl_leak_stat = "p90"
            cfg.fed_cgrl_tx_loss_guard = 0.0
            cfg.fed_cgrl_tx_guard_release_rounds = 0
            cfg.fed_cgrl_conflict_threshold = -0.10
            cfg.fed_cgrl_conflict_source = "auto"
            cfg.fed_cgrl_ema = 1.0
            trainer = FederatedTrainer(
                TinyReceiverAgnosticClassifier(),
                dataset,
                val_loader,
                {"test_unseen_day_unseen_rx": val_loader},
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=lambda model, loaders, device, domain_label_map=None, max_batches=0: {
                    name: eval_loader(model, loader, device) for name, loader in loaders.items()
                },
                domain_label_map={0: 0, 1: 1},
            )

            captured = StringIO()
            with redirect_stdout(captured):
                summary = trainer.train()

            self.assertEqual(summary["train_mode"], "fedprox")
            stdout = captured.getvalue()
            self.assertIn("[FED-CONFIG-CGRL] enabled=1", stdout)
            self.assertIn("[FED-CGRL][R001]", stdout)
            self.assertIn("lambda=0.5000", stdout)

            config = json.loads((Path(tmp) / "federated_config.json").read_text(encoding="utf-8"))
            self.assertIn("fed_cgrl", config)
            self.assertTrue(config["fed_cgrl"]["enabled"])
            self.assertEqual(config["fed_cgrl"]["base_lambda"], 0.5)
            self.assertEqual(config["fed_cgrl"]["leak_stat"], "p90")
            self.assertEqual(config["fed_cgrl"]["conflict_source"], "auto")

            metrics_text = (Path(tmp) / "metrics.csv").read_text(encoding="utf-8")
            self.assertIn("fed_cgrl_lambda_rx_adv", metrics_text)
            self.assertIn("fed_cgrl_leak_gate", metrics_text)
            self.assertIn("fed_cgrl_global_lambda_rx_adv_p90", metrics_text)
            self.assertIn("fed_cgrl_global_grl_target_acc_p90", metrics_text)
            self.assertIn("fed_cgrl_global_worst_client_id", metrics_text)
            self.assertIn("fed_cgrl_global_conflict_source", metrics_text)
            log_text = (Path(tmp) / "logs.jsonl").read_text(encoding="utf-8")
            self.assertIn("client_fed_cgrl_lambda_rx_adv_avg", log_text)
            self.assertIn("global_fed_cgrl_summary", log_text)
            self.assertIn('"fed_cgrl_lambda_rx_adv": 0.5', log_text)
            round_rows = [json.loads(line) for line in log_text.splitlines() if '"event":' not in line]
            cgrl = round_rows[-1]["global_fed_cgrl_summary"]
            self.assertIn("lambda_rx_adv_p90", cgrl)
            self.assertIn("grl_target_acc_p90", cgrl)
            self.assertIn("grl_target_acc_worst_client", cgrl)
            self.assertIn("conflict_source", cgrl)

    def test_fixed_lambda_receiver_adv_control_is_not_overridden_by_fedcgrl(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            cfg.train_mode = "fedprox"
            cfg.fl_local_objective = "receiver_agnostic_bex02"
            cfg.num_classes = 2
            cfg.fedprox_mu = 0.0
            cfg.label_smoothing = 0.0
            cfg.grl_lambda = 1.0
            cfg.lambda_dom = 0.0
            cfg.lambda_adv = 0.0
            cfg.lambda_orth = 0.0
            cfg.lambda_cons = 0.0
            cfg.lambda_group_ce = 0.0
            cfg.lambda_fishr = 0.0
            cfg.fishr_min_domains = 1
            cfg.group_ce_min_domains = 1
            cfg.group_ce_top_frac = 1.0
            cfg.group_ce_mode = "hard"
            cfg.use_sat_consistency = False
            cfg.use_aug = False
            cfg.use_mixstyle = False
            cfg.lambda_rx_adv = 0.25
            cfg.use_fed_cgrl = False
            trainer = FederatedTrainer(
                TinyReceiverAgnosticClassifier(),
                dataset,
                val_loader,
                {"test_unseen_day_unseen_rx": val_loader},
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=lambda model, loaders, device, domain_label_map=None, max_batches=0: {
                    name: eval_loader(model, loader, device) for name, loader in loaders.items()
                },
                domain_label_map={0: 0, 1: 1},
            )
            fed_cgrl_lambda, fed_cgrl_metrics = trainer._fed_cgrl_for_client("rx0", 1)
            batch = next(iter(DataLoader(dataset, batch_size=2)))
            x, y, d_raw = batch[0], batch[1], batch[2]
            objective = trainer._compute_local_objective(
                "rx0",
                x,
                y,
                d_raw,
                1,
                0,
                {k: v.to(trainer.device) for k, v in trainer.global_state.items()},
                0.0,
                set(),
                fed_cgrl_lambda_rx_adv=fed_cgrl_lambda,
                fed_cgrl_metrics=fed_cgrl_metrics,
            )

            expected = float(objective["loss_cls"]) + cfg.lambda_rx_adv * float(objective["loss_rx_adv"])
            self.assertIsNone(fed_cgrl_lambda)
            self.assertEqual(fed_cgrl_metrics, {})
            self.assertGreater(float(objective["loss_rx_adv"]), 0.0)
            self.assertAlmostEqual(float(objective["loss"].detach().item()), expected, places=5)

    def test_fedcgrl_conflict_gate_uses_non_vmb_update_cosine_summary(self):
        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(tmp)
            cfg.train_mode = "fedprox"
            cfg.fl_local_objective = "receiver_agnostic_bex02"
            cfg.use_fed_cgrl = True
            cfg.lambda_rx_adv = 1.0
            cfg.fed_cgrl_base_lambda = 1.0
            cfg.fed_cgrl_warmup_rounds = 0
            cfg.fed_cgrl_conflict_threshold = -0.10
            cfg.fed_cgrl_conflict_gate_min = 0.35
            cfg.fed_cgrl_conflict_source = "client_delta"
            trainer = FederatedTrainer(
                TinyReceiverAgnosticClassifier(),
                dataset,
                val_loader,
                {"test_unseen_day_unseen_rx": val_loader},
                cfg,
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=lambda model, loaders, device, domain_label_map=None, max_batches=0: {
                    name: eval_loader(model, loader, device) for name, loader in loaders.items()
                },
                domain_label_map={0: 0, 1: 1},
            )
            trainer.global_state = OrderedDict({"w": torch.tensor([0.0, 0.0])})
            client_states = {
                "rx0": OrderedDict({"w": torch.tensor([1.0, 0.0])}),
                "rx1": OrderedDict({"w": torch.tensor([-1.0, 0.0])}),
            }

            summary = trainer._fed_cgrl_conflict_summary_from_client_states(client_states, ["rx0", "rx1"])
            trainer.fed_cgrl.update_after_round(
                {"rx0": {"seen": 1}, "rx1": {"seen": 1}},
                round_idx=1,
                conflict_summary=summary,
            )
            decision = trainer.fed_cgrl.lambda_for_client("rx0", round_idx=2)

            self.assertEqual(summary["source"], "client_delta")
            self.assertEqual(summary["grad_cos_pairs"], 1)
            self.assertAlmostEqual(summary["grad_cos_min_before"], -1.0)
            self.assertEqual(summary["conflict_signal_available"], 1.0)
            self.assertLess(decision.conflict_gate, 1.0)


if __name__ == "__main__":
    unittest.main()
