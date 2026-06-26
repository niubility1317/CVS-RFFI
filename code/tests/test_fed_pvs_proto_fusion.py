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


class FedPvsProtoFusionTest(unittest.TestCase):
    def test_proto_evidence_bank_retains_multiple_reliable_prototypes_per_class(self):
        from federated.proto_evidence_bank import ProtoEvidence, ProtoEvidenceBank

        bank = ProtoEvidenceBank(max_per_class=3)
        bank.add(
            ProtoEvidence(
                class_id=1,
                prototype=torch.tensor([1.0, 0.0]),
                count=8,
                margin=0.8,
                entropy=0.1,
                intra_var=0.05,
                client_drift=0.1,
                clean_sat_kl=0.0,
                client_id="rx0",
                style_id=0,
                mode="clean",
            )
        )
        bank.add(
            ProtoEvidence(
                class_id=1,
                prototype=torch.tensor([0.0, 1.0]),
                count=6,
                margin=0.7,
                entropy=0.2,
                intra_var=0.1,
                client_drift=0.2,
                clean_sat_kl=0.1,
                client_id="rx1",
                style_id=2,
                mode="sat",
            )
        )

        entries = bank.get_class(1)

        self.assertEqual(len(entries), 2)
        self.assertNotEqual(entries[0].client_id, entries[1].client_id)
        self.assertTrue(all(0.0 <= item.reliability <= 1.0 for item in entries))
        self.assertEqual(bank.summary()["num_classes"], 1)

    def test_conservative_probability_fusion_caps_rho_and_reports_harm_rescue(self):
        from federated.reliability_fusion import conservative_probability_fusion, harm_rescue_report

        p_base = torch.tensor([[0.49, 0.51], [0.55, 0.45], [0.90, 0.10]])
        p_proto = torch.tensor([[0.90, 0.10], [0.20, 0.80], [0.20, 0.80]])
        y = torch.tensor([0, 1, 0])

        fused = conservative_probability_fusion(p_base, p_proto, rho=0.50, max_rho=0.05)
        report = harm_rescue_report(p_base, fused, y)

        expected = 0.95 * p_base + 0.05 * p_proto
        self.assertTrue(torch.allclose(fused, expected / expected.sum(dim=1, keepdim=True)))
        self.assertEqual(report["base_correct"], 1)
        self.assertEqual(report["fused_correct"], 2)
        self.assertEqual(report["rescue"], 1)
        self.assertEqual(report["harm"], 0)
        self.assertEqual(report["net_gain"], 1)

    def test_collaborative_probability_fusion_soft_mean_includes_clean_and_style_views(self):
        from federated.reliability_fusion import collaborative_probability_fusion

        p_base = torch.tensor([[0.60, 0.40], [0.20, 0.80]])
        p_style_a = torch.tensor([[0.30, 0.70], [0.40, 0.60]])
        p_style_b = torch.tensor([[0.90, 0.10], [0.10, 0.90]])

        fused = collaborative_probability_fusion(
            p_base,
            torch.stack([p_style_a, p_style_b], dim=0),
            mode="soft",
        )

        expected = (p_base + p_style_a + p_style_b) / 3.0
        expected = expected / expected.sum(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(fused, expected))

    def test_collaborative_probability_fusion_adaptive_ignores_uncertain_style_view(self):
        from federated.reliability_fusion import collaborative_probability_fusion

        p_base = torch.tensor([[0.80, 0.20], [0.20, 0.80]])
        p_style = torch.tensor([[0.10, 0.90], [0.50, 0.50]])

        fused = collaborative_probability_fusion(
            p_base,
            torch.stack([p_style], dim=0),
            mode="adaptive",
            aux_reliabilities=torch.tensor([1.0]),
            base_weight=1.0,
            max_aux_weight=1.0,
        )

        self.assertGreater(float(fused[0, 1]), float(p_base[0, 1]))
        self.assertTrue(torch.allclose(fused[1], p_base[1], atol=1e-6))

    def test_federated_trainer_updates_proto_evidence_bank_and_reports_fusion(self):
        from federated.fed_trainer import FederatedTrainer
        from tests.test_federated_trainer_smoke import TinyClientDataset, TinyDGClassifier, eval_loader

        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                train_mode="fedavg",
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
                fedprox_mu=0.0,
                use_fed_style_bank=False,
                use_fl_style_bank_stats=False,
                use_proto_evidence_bank=True,
                proto_max_per_class=3,
                proto_top_m=2,
                proto_temperature=0.1,
                proto_rho_max=0.05,
                proto_fusion_eval=True,
            )
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
            )

            summary = trainer.train()

        self.assertTrue(summary["global_proto_evidence_summary"]["enabled"])
        self.assertGreater(summary["global_proto_evidence_summary"]["num_prototypes"], 0)
        self.assertIn("proto_fusion", summary["last_eval"])
        fusion = summary["last_eval"]["proto_fusion"]["aggregate"]
        self.assertIn("rescue", fusion)
        self.assertIn("harm", fusion)

    def test_federated_trainer_reports_stylebank_virtual_collaborative_fusion(self):
        from federated.fed_trainer import FederatedTrainer
        from federated.style_packet import StylePacket
        from tests.test_federated_trainer_smoke import TinyClientDataset, TinyDGClassifier, eval_loader

        dataset = TinyClientDataset()
        val_loader = DataLoader(dataset, batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                train_mode="fedavg",
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
                fedprox_mu=0.0,
                use_fed_style_bank=True,
                use_fl_style_bank_stats=True,
                fl_style_bank_momentum=0.5,
                fl_style_bank_max_centroids=8,
                fl_style_bank_merge_radius=0.0,
                use_proto_evidence_bank=False,
                use_style_collab_eval=True,
                style_collab_views=1,
                style_collab_fusion="adaptive",
                style_collab_base_weight=1.0,
                style_collab_max_aux_weight=1.0,
            )
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
            )
            trainer.style_bank.update(
                [StylePacket(client_id="rx9", round_idx=0, count=8, stats={"iq_rms": 1.08, "amp_std": 0.1})]
            )

            summary = trainer.train()
            metrics_text = (Path(tmp) / "metrics.csv").read_text(encoding="utf-8")

        self.assertIn("style_collab_fusion", summary["last_eval"])
        collab = summary["last_eval"]["style_collab_fusion"]
        self.assertTrue(collab["enabled"])
        self.assertEqual(collab["fusion"], "adaptive")
        for key in ["total", "base_correct", "fused_correct", "rescue", "harm", "net_gain"]:
            self.assertIn(key, collab["aggregate"])
        self.assertIn("style_collab_rescue", metrics_text)
        self.assertIn("style_collab_harm", metrics_text)


if __name__ == "__main__":
    unittest.main()
