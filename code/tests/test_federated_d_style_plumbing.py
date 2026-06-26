import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_federated_trainer_smoke import TinyClientDataset, eval_loader


class DomainRecordingClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 2)
        self.dom = nn.Linear(8, 6)
        self.seen_domain_labels = []

    def forward(self, x, *args, domain_labels=None, **kwargs):
        if domain_labels is not None:
            self.seen_domain_labels.append(domain_labels.detach().cpu().clone())
        feat = x.flatten(1)
        return {
            "tx_logits": self.fc(feat),
            "dom_logits": self.dom(feat),
            "adv_dom_logits": self.dom(feat),
            "z_id": feat,
            "z_dom": torch.flip(feat, dims=[1]),
        }


class TinyOneDomainHeadClassifier(DomainRecordingClassifier):
    def __init__(self):
        super().__init__()
        self.dom = nn.Linear(8, 1)


class FederatedDStylePlumbingTest(unittest.TestCase):
    def _cfg(self, tmp):
        return SimpleNamespace(
            train_mode="fedavg",
            fl_local_objective="bex02_dg",
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
            fedprox_mu=0.0,
            lambda_dom=0.0,
            lambda_adv=0.0,
            lambda_cons=0.0,
            lambda_group_ce=0.0,
            lambda_fishr=0.0,
            lambda_orth=0.0,
            lambda_rx_adv=0.0,
            min_batch_domains_for_domain_loss=2,
            min_batch_domain_frac=0.15,
            use_fed_style_bank=False,
            use_fl_style_bank_stats=False,
            fl_style_domain_label_mode="constructed",
            fl_style_zdom_probe_every=0,
            fl_style_zdom_probe_real_samples=0,
            fl_style_zdom_probe_max_examples=4,
            fl_style_sampling_policy="diverse",
            fl_style_transform_mix_alpha=1.0,
            fl_style_real_mix_samples=0,
            fl_style_real_mix_start_round=0,
            use_proto_evidence_bank=False,
        )

    def test_style_batch_fn_domains_are_used_for_model_and_losses(self):
        from federated.fed_trainer import FederatedTrainer
        from federated.virtual_domain_sampler import VirtualDomainSampler, VirtualStyleView

        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        model = DomainRecordingClassifier()

        def style_batch_fn(x, y, d_raw, round_idx, batch_idx, trainer):
            del round_idx, batch_idx, trainer
            sampler = VirtualDomainSampler(clean_style_id=0)
            return sampler.build_batch(x, y, d_raw, [VirtualStyleView(x=x + 0.25, source="remote", style_id=3)])

        with tempfile.TemporaryDirectory() as tmp:
            trainer = FederatedTrainer(
                model,
                dataset,
                val_loader,
                {"test_unseen_day_unseen_rx": val_loader},
                self._cfg(tmp),
                device=torch.device("cpu"),
                criterion=nn.CrossEntropyLoss(),
                evaluate_loader_fn=eval_loader,
                evaluate_named_loaders_fn=lambda model, loaders, device, domain_label_map=None, max_batches=0: {
                    name: eval_loader(model, loader, device) for name, loader in loaders.items()
                },
                style_batch_fn=style_batch_fn,
            )

            result = trainer.train_one_client("rx0", 1)

        self.assertGreater(result["seen"], 0)
        self.assertTrue(model.seen_domain_labels)
        first = model.seen_domain_labels[0]
        self.assertTrue(torch.equal(first, torch.tensor([0, 0, 1, 1])))
        self.assertIn("style_num_domains", result)
        self.assertEqual(result["style_num_domains"], 2.0)

    def test_domain_losses_skip_when_style_domains_exceed_head_dimension(self):
        from federated.fed_trainer import FederatedTrainer
        from federated.virtual_domain_sampler import VirtualDomainSampler, VirtualStyleView

        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        model = TinyOneDomainHeadClassifier()
        cfg = self._cfg(tempfile.mkdtemp())
        cfg.fl_local_objective = "receiver_agnostic_bex02"
        cfg.lambda_dom = 1.0
        cfg.lambda_adv = 1.0
        cfg.lambda_rx_adv = 1.0
        cfg.min_batch_domain_frac = 0.0

        def style_batch_fn(x, y, d_raw, round_idx, batch_idx, trainer):
            del round_idx, batch_idx, trainer
            sampler = VirtualDomainSampler(clean_style_id=0)
            return sampler.build_batch(x, y, d_raw, [VirtualStyleView(x=x + 0.25, source="remote", style_id=99)])

        trainer = FederatedTrainer(
            model,
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
            style_batch_fn=style_batch_fn,
        )

        result = trainer.train_one_client("rx0", 1)

        self.assertEqual(result["loss_dom"], 0.0)
        self.assertEqual(result["loss_adv"], 0.0)
        self.assertEqual(result["loss_rx_adv"], 0.0)

    def test_style_bank_is_opt_in_and_legacy_alias_enables_it(self):
        from federated.fed_trainer import _style_bank_enabled

        self.assertFalse(_style_bank_enabled(SimpleNamespace()))
        self.assertFalse(_style_bank_enabled(SimpleNamespace(use_fed_style_bank=False, use_fl_style_bank_stats=False)))
        self.assertTrue(_style_bank_enabled(SimpleNamespace(use_fed_style_bank=True, use_fl_style_bank_stats=False)))
        self.assertTrue(_style_bank_enabled(SimpleNamespace(use_fed_style_bank=False, use_fl_style_bank_stats=True)))

    def test_default_style_bank_batch_uses_constructed_style_ids(self):
        from federated.fed_trainer import FederatedTrainer
        from federated.style_packet import StylePacket

        class FakeStyleBank:
            def __init__(self):
                self.styles = [
                    StylePacket(
                        client_id="rx6",
                        round_idx=1,
                        count=2,
                        stats={"gain": 0.1},
                        style_id=11,
                        metadata={"target_domain_label": 6},
                    ),
                    StylePacket(
                        client_id="rx7",
                        round_idx=1,
                        count=2,
                        stats={"gain": 0.2},
                        style_id=12,
                        metadata={"target_domain_label": 7},
                    ),
                ]

            def diagnostics(self):
                return {"num_centroids": len(self.styles)}

            def sample_remote_style(self, exclude_client_id):
                return self.styles[0]

            def sample_remote_styles(self, exclude_client_id, k):
                return self.styles[:k]

        class FakeStyleTransform:
            def transform(self, x, style):
                return x + (0.01 * int(style.metadata["target_domain_label"]))

        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        cfg = self._cfg(tempfile.mkdtemp())
        cfg.fl_style_replay_start_round = 1
        cfg.fl_style_phys_start_round = 1
        cfg.fl_style_replay_prob = 1.0
        cfg.fl_style_max_views = 2
        trainer = FederatedTrainer(
            DomainRecordingClassifier(),
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
        trainer.style_bank = FakeStyleBank()
        trainer.style_transform = FakeStyleTransform()

        x = torch.zeros(2, 1, 2, 4)
        y = torch.tensor([0, 1])
        d_raw = torch.tensor([4, 4])
        batch = trainer._build_default_style_batch("rx4", x, y, d_raw, round_idx=1, batch_idx=0)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.d_style.tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(batch.d_raw.tolist(), [4, 4, 6, 6, 7, 7])
        self.assertEqual(batch.metadata["style_domain_semantics"], "constructed_style_view_id")
        self.assertEqual(batch.metadata["d_raw_semantics"], "mapped_target_receiver_domain_for_each_view")

    def test_receiver_agnostic_fallback_adv_head_is_not_double_counted(self):
        from federated.fed_trainer import FederatedTrainer
        from federated.virtual_domain_sampler import VirtualDomainSampler, VirtualStyleView

        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        model = DomainRecordingClassifier()
        cfg = self._cfg(tempfile.mkdtemp())
        cfg.fl_local_objective = "receiver_agnostic_bex02"
        cfg.lambda_rx_adv = 1.0
        cfg.lambda_adv = 1.0
        cfg.min_batch_domain_frac = 0.0
        cfg.fl_style_dg_start_round = 1
        cfg.fl_style_dg_min_domains = 2

        def style_batch_fn(x, y, d_raw, round_idx, batch_idx, trainer):
            del round_idx, batch_idx, trainer
            sampler = VirtualDomainSampler(clean_style_id=0)
            return sampler.build_batch(x, y, d_raw, [VirtualStyleView(x=x + 0.25, source="remote", style_id=3)])

        trainer = FederatedTrainer(
            model,
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
            style_batch_fn=style_batch_fn,
        )

        result = trainer.train_one_client("rx0", 1)

        self.assertGreater(result["loss_rx_adv"], 0.0)
        self.assertEqual(result["loss_adv"], 0.0)
        self.assertEqual(result["diag_rx_adv_active"], 1.0)
        self.assertEqual(result["diag_adv_active"], 0.0)

    def test_cvs_satellite_losses_stay_active_with_style_batch(self):
        from federated.fed_trainer import FederatedTrainer
        from federated.virtual_domain_sampler import VirtualDomainSampler, VirtualStyleView

        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        model = DomainRecordingClassifier()
        cfg = self._cfg(tempfile.mkdtemp())
        cfg.fl_local_objective = "receiver_agnostic_bex02"
        cfg.fl_sat_aug_mode = "cvs_consistency"
        cfg.use_sat_consistency = True
        cfg.sat_cons_start_epoch = 1
        cfg.sat_train_scenario = "mixed_orbit"
        cfg.lambda_sat_cls = 0.1
        cfg.lambda_sat_cons = 0.1
        cfg.lambda_rx_adv = 0.0
        cfg.lambda_adv = 0.0
        sat_calls = []

        def style_batch_fn(x, y, d_raw, round_idx, batch_idx, trainer):
            del round_idx, batch_idx, trainer
            sampler = VirtualDomainSampler(clean_style_id=0)
            return sampler.build_batch(x, y, d_raw, [VirtualStyleView(x=x + 0.25, source="remote", style_id=3)])

        def sat_transform(x, scenario, round_idx, batch_idx):
            sat_calls.append((scenario, int(round_idx), int(batch_idx), int(x.size(0))))
            return x + 0.05

        trainer = FederatedTrainer(
            model,
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
            style_batch_fn=style_batch_fn,
            sat_transform_fn=sat_transform,
        )

        result = trainer.train_one_client("rx0", 1)

        self.assertTrue(sat_calls)
        self.assertGreater(result["loss_sat_cls"], 0.0)
        self.assertGreaterEqual(result["loss_sat_cons"], 0.0)
        self.assertEqual(result["diag_sat_aug_active"], 1.0)
        self.assertEqual(result["diag_sat_cls_active"], 1.0)
        self.assertEqual(result["diag_sat_cons_active"], 1.0)

    def test_default_federated_style_bank_builds_remote_style_batch(self):
        from federated.fed_trainer import FederatedTrainer
        from federated.style_packet import StylePacket

        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        model = DomainRecordingClassifier()
        cfg = self._cfg(tempfile.mkdtemp())
        cfg.train_mode = "fedavg"
        cfg.fl_local_objective = "receiver_agnostic_bex02"
        cfg.use_fed_style_bank = True
        cfg.use_fl_style_bank_stats = True
        cfg.fl_style_replay_start_round = 1
        cfg.fl_style_phys_start_round = 1
        cfg.fl_style_dg_start_round = 1
        cfg.fl_style_min_remote_centroids = 1
        cfg.fl_style_max_views = 2
        cfg.fl_style_replay_prob = 1.0
        cfg.fl_style_dg_min_domains = 3
        cfg.lambda_rx_adv = 1.0

        trainer = FederatedTrainer(
            model,
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
        self.assertIsNotNone(trainer.style_bank)
        trainer.style_bank.update(
            [
                StylePacket(client_id="rx1", round_idx=0, count=8, stats={"iq_rms": 1.10, "amp_std": 0.2}),
                StylePacket(client_id="rx2", round_idx=0, count=8, stats={"iq_rms": 0.92, "amp_std": 0.1}),
            ]
        )

        result = trainer.train_one_client("rx0", 1)

        self.assertGreaterEqual(result["style_num_domains"], 3.0)
        self.assertGreaterEqual(result["style_batch_views"], 3.0)
        self.assertEqual(result["diag_style_batch_active"], 1.0)
        self.assertEqual(result["diag_rx_adv_active"], 1.0)
        self.assertTrue(model.seen_domain_labels)
        first = model.seen_domain_labels[0]
        self.assertTrue(torch.equal(first, torch.tensor([0, 0, 1, 1, 2, 2])))

    def test_default_stylebank_uses_metadata_target_domain_when_client_id_is_not_parseable(self):
        from federated.fed_trainer import FederatedTrainer
        from federated.style_packet import StylePacket

        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        model = DomainRecordingClassifier()
        cfg = self._cfg(tempfile.mkdtemp())
        cfg.train_mode = "fedavg"
        cfg.fl_local_objective = "receiver_agnostic_bex02"
        cfg.use_fed_style_bank = True
        cfg.use_fl_style_bank_stats = True
        cfg.fl_style_replay_start_round = 1
        cfg.fl_style_phys_start_round = 1
        cfg.fl_style_dg_start_round = 1
        cfg.fl_style_min_remote_centroids = 1
        cfg.fl_style_max_views = 1
        cfg.fl_style_replay_prob = 1.0
        cfg.fl_style_dg_min_domains = 2
        cfg.lambda_rx_adv = 1.0

        trainer = FederatedTrainer(
            model,
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
        trainer.style_bank.update(
            [
                StylePacket(
                    client_id="station_A_day7",
                    round_idx=0,
                    count=8,
                    stats={"iq_rms": 1.08, "amp_std": 0.1},
                    metadata={"target_domain_label": 4},
                )
            ]
        )

        result = trainer.train_one_client("rx0", 1)

        self.assertEqual(result["diag_rx_adv_active"], 1.0)
        self.assertTrue(model.seen_domain_labels)
        self.assertTrue(torch.equal(model.seen_domain_labels[0], torch.tensor([0, 0, 1, 1])))

    def test_default_stylebank_keeps_raw_target_receiver_labels_outside_constructed_d_style(self):
        from federated.fed_trainer import FederatedTrainer
        from federated.style_packet import StylePacket

        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        cfg = self._cfg(tempfile.mkdtemp())
        cfg.train_mode = "fedavg"
        cfg.fl_local_objective = "receiver_agnostic_bex02"
        cfg.use_fed_style_bank = True
        cfg.use_fl_style_bank_stats = True
        cfg.fl_style_replay_start_round = 1
        cfg.fl_style_phys_start_round = 1
        cfg.fl_style_dg_start_round = 1
        cfg.fl_style_min_remote_centroids = 1
        cfg.fl_style_max_views = 1
        cfg.fl_style_replay_prob = 1.0
        cfg.fl_style_dg_min_domains = 2

        trainer = FederatedTrainer(
            DomainRecordingClassifier(),
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
        trainer.style_bank.update(
            [
                StylePacket(
                    client_id="station_A_day7",
                    round_idx=0,
                    count=8,
                    stats={"iq_rms": 1.08, "amp_std": 0.1},
                    metadata={"target_domain_label": 4},
                )
            ]
        )
        x = torch.stack([dataset[0][0], dataset[2][0]], dim=0)
        y = torch.tensor([dataset[0][1], dataset[2][1]])
        d_raw = torch.tensor([dataset[0][2], dataset[2][2]])

        batch = trainer._build_default_style_batch("rx0", x, y, d_raw, 1, 0)

        self.assertIsNotNone(batch)
        self.assertTrue(torch.equal(batch.d_style.cpu(), torch.tensor([0, 0, 1, 1])))
        self.assertTrue(torch.equal(batch.d_raw.cpu(), torch.tensor([0, 0, 4, 4])))
        self.assertEqual(batch.metadata["target_domain_labels"], (0, 4))
        self.assertEqual(batch.metadata["source_domain_labels"], (0, 4))

    def test_stylebank_target_receiver_mode_passes_target_domain_labels_to_model(self):
        from federated.fed_trainer import FederatedTrainer
        from federated.style_packet import StylePacket

        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        model = DomainRecordingClassifier()
        cfg = self._cfg(tempfile.mkdtemp())
        cfg.train_mode = "fedavg"
        cfg.fl_local_objective = "receiver_agnostic_bex02"
        cfg.use_fed_style_bank = True
        cfg.use_fl_style_bank_stats = True
        cfg.fl_style_domain_label_mode = "target_receiver"
        cfg.fl_style_replay_start_round = 1
        cfg.fl_style_phys_start_round = 1
        cfg.fl_style_dg_start_round = 1
        cfg.fl_style_min_remote_centroids = 1
        cfg.fl_style_max_views = 1
        cfg.fl_style_replay_prob = 1.0
        cfg.fl_style_dg_min_domains = 2
        cfg.lambda_rx_adv = 1.0

        trainer = FederatedTrainer(
            model,
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
        trainer.style_bank.update(
            [
                StylePacket(
                    client_id="station_A_day7",
                    round_idx=0,
                    count=8,
                    stats={"iq_rms": 1.08, "amp_std": 0.1},
                    metadata={"source_domain_label": 4},
                )
            ]
        )

        result = trainer.train_one_client("rx0", 1)

        self.assertEqual(result["style_domain_label_mode"], "target_receiver")
        self.assertEqual(result["diag_rx_adv_active"], 1.0)
        self.assertTrue(model.seen_domain_labels)
        self.assertTrue(torch.equal(model.seen_domain_labels[0], torch.tensor([0, 0, 4, 4])))

    def test_stylebank_zdom_probe_reports_virtual_and_real_domain_accuracy(self):
        from federated.fed_trainer import FederatedTrainer
        from federated.style_packet import StylePacket

        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        cfg = self._cfg(tempfile.mkdtemp())
        cfg.train_mode = "fedavg"
        cfg.fl_local_objective = "receiver_agnostic_bex02"
        cfg.use_fed_style_bank = True
        cfg.use_fl_style_bank_stats = True
        cfg.fl_style_domain_label_mode = "target_receiver"
        cfg.fl_style_replay_start_round = 1
        cfg.fl_style_phys_start_round = 1
        cfg.fl_style_dg_start_round = 1
        cfg.fl_style_min_remote_centroids = 1
        cfg.fl_style_max_views = 1
        cfg.fl_style_replay_prob = 1.0
        cfg.fl_style_dg_min_domains = 2
        cfg.lambda_rx_adv = 1.0
        cfg.fl_style_zdom_probe_every = 1
        cfg.fl_style_zdom_probe_real_samples = 2
        cfg.fl_style_zdom_probe_max_examples = 3

        trainer = FederatedTrainer(
            DomainRecordingClassifier(),
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
        trainer.style_bank.update(
            [
                StylePacket(
                    client_id="station_A_day7",
                    round_idx=0,
                    count=8,
                    stats={"iq_rms": 1.08, "amp_std": 0.1},
                    metadata={"target_domain_label": 4},
                )
            ]
        )

        result = trainer.train_one_client("rx0", 1)

        probe = result["style_zdom_probe"]
        self.assertEqual(probe["mode"], "target_receiver")
        self.assertEqual(probe["virtual"]["total"], 2)
        self.assertEqual(probe["virtual"]["target_hist"], {"4": 2})
        self.assertEqual(probe["real"]["total"], 2)
        self.assertEqual(probe["real"]["target_hist"], {"1": 2})
        self.assertIn("style_zdom_virtual_acc", result)
        self.assertIn("style_zdom_real_acc", result)

    def test_stylebank_real_mix_adds_true_other_domain_samples_to_target_receiver_batch(self):
        from federated.fed_trainer import FederatedTrainer
        from federated.style_packet import StylePacket

        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        cfg = self._cfg(tempfile.mkdtemp())
        cfg.use_fed_style_bank = True
        cfg.use_fl_style_bank_stats = True
        cfg.fl_style_domain_label_mode = "target_receiver"
        cfg.fl_style_replay_start_round = 1
        cfg.fl_style_phys_start_round = 1
        cfg.fl_style_dg_start_round = 1
        cfg.fl_style_min_remote_centroids = 1
        cfg.fl_style_max_views = 1
        cfg.fl_style_replay_prob = 1.0
        cfg.fl_style_dg_min_domains = 2
        cfg.fl_style_real_mix_samples = 2
        cfg.fl_style_real_mix_start_round = 1

        trainer = FederatedTrainer(
            DomainRecordingClassifier(),
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
        trainer.style_bank.update(
            [
                StylePacket(
                    client_id="station_A_day7",
                    round_idx=0,
                    count=8,
                    stats={"iq_rms": 1.08, "amp_std": 0.1},
                    metadata={"target_domain_label": 4},
                )
            ]
        )
        x = torch.stack([dataset[0][0], dataset[2][0]], dim=0)
        y = torch.tensor([dataset[0][1], dataset[2][1]])
        d_raw = torch.tensor([dataset[0][2], dataset[2][2]])

        batch = trainer._build_default_style_batch("rx0", x, y, d_raw, 1, 0)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.sources, ("clean", "remote_style:station_A_day7", "real_other_domain"))
        self.assertTrue(torch.equal(batch.d_style.cpu(), torch.tensor([0, 0, 1, 1, 2, 2])))
        self.assertEqual(batch.metadata["raw_style_ids"], (0, 0, -2000))
        self.assertEqual(batch.metadata["target_domain_labels"], (0, 4, 1))
        self.assertTrue(torch.equal(batch.d_raw.cpu()[-2:], torch.tensor([1, 1])))

        result = trainer.train_one_client("rx0", 1)

        self.assertGreater(result["seen"], len(dataset))
        self.assertEqual(result["style_domain_label_mode"], "target_receiver")


if __name__ == "__main__":
    unittest.main()
