import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn


class BackboneStabilityOptionsTest(unittest.TestCase):
    def test_single_backbone_off_modes_preserve_parameter_count(self):
        from model import build_model

        base = build_model(
            num_classes=6,
            dataset="wisig",
            input_len=128,
            sample_rate_hz=25e6,
            model_variant="lite_d",
            branch_ablation="no_dac",
        )
        explicit = build_model(
            num_classes=6,
            dataset="wisig",
            input_len=128,
            sample_rate_hz=25e6,
            model_variant="lite_d",
            branch_ablation="no_dac",
            time_stability_mode="off",
            freq_stability_mode="off",
        )

        self.assertEqual(sum(p.numel() for p in base.parameters()), sum(p.numel() for p in explicit.parameters()))
        self.assertEqual(explicit.time_stability_mode, "off")
        self.assertEqual(explicit.freq_stability_mode, "off")

    def test_single_backbone_runs_phase_delta_and_dsq_modes(self):
        from model import build_model

        torch.manual_seed(7)
        model = build_model(
            num_classes=6,
            dataset="wisig",
            input_len=128,
            sample_rate_hz=25e6,
            model_variant="lite_d",
            branch_ablation="no_dac",
            time_stability_mode="phase_delta",
            freq_stability_mode="dsq",
            time_stability_channels=4,
            freq_stability_channels=4,
        )
        model.eval()
        x = torch.randn(2, 2, 128)
        with torch.no_grad():
            out = model(x, return_aux=True)

        self.assertEqual(out["logits"].shape, (2, 6))
        self.assertEqual(out["t_emb"].shape[0], 2)
        self.assertEqual(out["f_emb"].shape[0], 2)
        self.assertEqual(model.time_stability_mode, "phase_delta")
        self.assertEqual(model.freq_stability_mode, "dsq")

    def test_dual_backbone_resolves_same_domain_stability_modes(self):
        from model_dual_cvsincnet import build_dual_model

        torch.manual_seed(11)
        model = build_dual_model(
            num_classes=6,
            num_domains=3,
            dataset="wisig",
            input_len=128,
            sample_rate_hz=25e6,
            model_variant="lite_d",
            branch_ablation="no_dac",
            domain_branch_ablation="no_stats",
            id_time_stability_mode="phase_delta",
            id_freq_stability_mode="dsq",
            domain_time_stability_mode="same",
            domain_freq_stability_mode="same",
            time_stability_channels=4,
            freq_stability_channels=4,
            fast_infer_when_no_aux=False,
        )
        model.eval()
        x = torch.randn(2, 2, 128)
        y = torch.tensor([0, 1])
        d = torch.tensor([0, 1])
        with torch.no_grad():
            out = model(x, y_tx=y, domain_labels=d, return_aux=True)

        self.assertEqual(out["tx_logits"].shape, (2, 6))
        self.assertEqual(model.id_backbone.time_stability_mode, "phase_delta")
        self.assertEqual(model.dom_backbone.time_stability_mode, "phase_delta")
        self.assertEqual(model.id_backbone.freq_stability_mode, "dsq")
        self.assertEqual(model.dom_backbone.freq_stability_mode, "dsq")

    def test_invalid_backbone_stability_modes_raise_clear_errors(self):
        from model import build_model

        with self.assertRaisesRegex(ValueError, "time_stability_mode"):
            build_model(model_variant="lite_d", time_stability_mode="bad")
        with self.assertRaisesRegex(ValueError, "freq_stability_mode"):
            build_model(model_variant="lite_d", freq_stability_mode="bad")


class FederatedBaselineViewCeOnlyTest(unittest.TestCase):
    def test_federated_baseline_view_ce_only_keeps_satellite_out_of_full_dg_batch(self):
        from federated.fed_trainer import FederatedTrainer

        class RecordingFedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, x, y_tx=None, grl_lambda=1.0, return_aux=True, domain_labels=None):
                del y_tx, grl_lambda, return_aux, domain_labels
                self.calls.append(int(x.size(0)))
                score = x[:, 0, 0].float()
                logits = torch.stack([score, -score], dim=1)
                feat = torch.stack([score, score * 0.0, score * 0.0, score * 0.0], dim=1)
                dom_logits = torch.zeros(x.size(0), 2, device=x.device)
                return {
                    "tx_logits": logits,
                    "z_id": feat,
                    "z_dom": feat,
                    "dom_logits": dom_logits,
                    "adv_dom_logits": dom_logits,
                }

        model = RecordingFedModel()
        trainer = object.__new__(FederatedTrainer)
        trainer.model = model
        trainer.vmb_enabled = False
        trainer.cfg = SimpleNamespace(
            train_mode="fedprox",
            fl_local_objective="receiver_agnostic_bex02",
            fl_sat_aug_mode="baseline_view",
            use_sat_consistency=True,
            sat_cons_start_epoch=1,
            fl_baseline_view_ce_only=True,
            fl_baseline_view_ce_weight=1.0,
            use_aug=False,
            grl_lambda=1.0,
            lambda_rx_adv=0.0,
            lambda_dom=0.0,
            lambda_adv=0.0,
            lambda_orth=0.0,
            lambda_cons=0.0,
            lambda_group_ce=0.0,
            lambda_fishr=0.0,
            lambda_sat_cls=0.0,
            lambda_sat_cons=0.0,
            lambda_fed_proto=0.0,
            fishr_min_domains=99,
            min_batch_domains_for_domain_loss=99,
            min_batch_domain_frac=1.0,
            group_ce_min_domains=99,
            use_fed_proto_stats=False,
            use_proto_evidence_bank=False,
            fl_style_zdom_probe_every=0,
            seed=1337,
        )
        trainer.device = torch.device("cpu")
        trainer.criterion = nn.CrossEntropyLoss()
        trainer.augment_fn = None
        trainer.sat_transform_fn = lambda x, scenario, round_idx, batch_idx: -x
        trainer.style_batch_fn = None
        trainer.style_bank = None
        trainer.style_transform = None
        trainer.virtual_domain_sampler = SimpleNamespace(clean_style_id=0)
        trainer.domain_label_map = {}
        trainer.global_proto_stats = None
        trainer.proto_evidence_bank = None
        trainer.logit_anchor_bank = None
        trainer.activation_token_codec = None
        trainer.client_splits = {}
        trainer.train_dataset = []

        x = torch.tensor([[[2.0, 0.0], [0.0, 0.0]], [[-2.0, 0.0], [0.0, 0.0]]])
        y = torch.tensor([0, 1])
        d = torch.tensor([0, 0])
        metrics = FederatedTrainer._compute_local_objective(
            trainer,
            "rx0",
            x,
            y,
            d,
            round_idx=1,
            batch_idx=0,
            global_params={},
            mu=0.0,
            exclude_for_prox=(),
        )
        calls_during_objective = list(model.calls)

        clean_ce = nn.CrossEntropyLoss()(model(x)["tx_logits"], y).item()
        sat_ce = nn.CrossEntropyLoss()(model(-x)["tx_logits"], y).item()
        self.assertEqual(calls_during_objective, [2, 2])
        self.assertAlmostEqual(float(metrics["loss_cls"]), clean_ce, places=5)
        self.assertAlmostEqual(float(metrics["loss_baseline_sat_view"]), sat_ce, places=5)
        self.assertAlmostEqual(float(metrics["loss"].detach().item()), clean_ce + sat_ce, places=5)
        self.assertEqual(metrics["diag_baseline_sat_view_active"], 1.0)
        self.assertEqual(metrics["diag_sat_cls_active"], 1.0)

    def test_global_zid_coral_loss_is_added_when_enabled(self):
        from federated.fed_trainer import FederatedTrainer
        from federated.fedcvs_vmb import FedCVSCoralStatsBank, build_class_conditional_coral_stats

        class RecordingFedModel(nn.Module):
            def forward(self, x, y_tx=None, grl_lambda=1.0, return_aux=True, domain_labels=None):
                del y_tx, grl_lambda, return_aux, domain_labels
                score = x[:, 0, 0].float()
                logits = torch.stack([score, -score], dim=1)
                feat = torch.stack([score, score * 0.0 + 1.0], dim=1)
                dom_logits = torch.zeros(x.size(0), 2, device=x.device)
                return {
                    "tx_logits": logits,
                    "z_id": feat,
                    "z_dom": feat,
                    "dom_logits": dom_logits,
                    "adv_dom_logits": dom_logits,
                }

        trainer = object.__new__(FederatedTrainer)
        trainer.model = RecordingFedModel()
        trainer.vmb_enabled = True
        trainer.cfg = SimpleNamespace(
            train_mode="fedcvs_vmb",
            fl_local_objective="receiver_agnostic_bex02",
            fl_vmb_stage="auto",
            fl_vmb_pretrain_rounds=10,
            fl_vmb_stage1_objective="ce",
            fl_sat_aug_mode="baseline_view",
            use_sat_consistency=False,
            sat_cons_start_epoch=99,
            fl_baseline_view_ce_only=True,
            fl_baseline_view_ce_weight=1.0,
            use_aug=False,
            grl_lambda=1.0,
            lambda_rx_adv=0.0,
            lambda_dom=0.0,
            lambda_adv=0.0,
            lambda_orth=0.0,
            lambda_cons=0.0,
            lambda_group_ce=0.0,
            lambda_fishr=0.0,
            lambda_sat_cls=0.0,
            lambda_sat_cons=0.0,
            lambda_fed_proto=0.0,
            lambda_vmb_tx_proto=0.0,
            lambda_vmb_rx_proto=0.0,
            lambda_tx_adv_r=0.0,
            lambda_logit_kd=0.0,
            lambda_fed_coral=0.5,
            lambda_fed_coral_virtual=0.0,
            use_fed_coral=True,
            fed_coral_feature="z_id",
            fed_coral_mode="diag",
            fed_coral_start_round=1,
            fed_coral_min_count=1,
            fed_coral_scope="zid_global",
            fishr_min_domains=99,
            min_batch_domains_for_domain_loss=99,
            min_batch_domain_frac=1.0,
            group_ce_min_domains=99,
            use_fed_proto_stats=False,
            use_proto_evidence_bank=False,
            fl_style_zdom_probe_every=0,
            seed=1337,
        )
        trainer.device = torch.device("cpu")
        trainer.criterion = nn.CrossEntropyLoss()
        trainer.augment_fn = None
        trainer.sat_transform_fn = None
        trainer.style_batch_fn = None
        trainer.style_bank = None
        trainer.style_transform = None
        trainer.virtual_domain_sampler = SimpleNamespace(clean_style_id=0)
        trainer.domain_label_map = {}
        trainer.global_proto_stats = None
        trainer.proto_evidence_bank = None
        trainer.logit_anchor_bank = None
        trainer.activation_token_codec = None
        trainer.client_splits = {}
        trainer.train_dataset = []
        trainer.global_coral_bank = FedCVSCoralStatsBank(num_classes=2, momentum=0.0, mode="diag")
        reference_stats = build_class_conditional_coral_stats(
            torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            torch.tensor([0, 1]),
            num_classes=2,
            mode="diag",
        )
        trainer.global_coral_bank.update(reference_stats)

        x = torch.tensor([[[2.0, 0.0], [0.0, 0.0]], [[-2.0, 0.0], [0.0, 0.0]]])
        y = torch.tensor([0, 1])
        d = torch.tensor([0, 0])
        metrics = FederatedTrainer._compute_local_objective(
            trainer,
            "rx0",
            x,
            y,
            d,
            round_idx=1,
            batch_idx=0,
            global_params={},
            mu=0.0,
            exclude_for_prox=(),
        )

        self.assertGreater(float(metrics["loss_coral_zid_global"]), 0.0)
        self.assertEqual(metrics["diag_coral_global_active"], 1.0)
        expected_total = float(metrics["loss_cls"]) + 0.5 * float(metrics["loss_coral_zid_global"])
        self.assertAlmostEqual(float(metrics["loss"].detach().item()), expected_total, places=5)

    def test_satellite_view_zid_coral_virtual_loss_is_added_when_enabled(self):
        from federated.fed_trainer import FederatedTrainer
        from federated.fedcvs_vmb import FedCVSCoralStatsBank

        class RecordingFedModel(nn.Module):
            def forward(self, x, y_tx=None, grl_lambda=1.0, return_aux=True, domain_labels=None):
                del y_tx, grl_lambda, return_aux, domain_labels
                score = x[:, 0, 0].float()
                logits = torch.stack([score, -score], dim=1)
                feat = torch.stack([score, score * 0.0 + 1.0], dim=1)
                dom_logits = torch.zeros(x.size(0), 2, device=x.device)
                return {
                    "tx_logits": logits,
                    "z_id": feat,
                    "z_dom": feat,
                    "dom_logits": dom_logits,
                    "adv_dom_logits": dom_logits,
                }

        trainer = object.__new__(FederatedTrainer)
        trainer.model = RecordingFedModel()
        trainer.vmb_enabled = True
        trainer.cfg = SimpleNamespace(
            train_mode="fedcvs_vmb",
            fl_local_objective="receiver_agnostic_bex02",
            fl_vmb_stage="auto",
            fl_vmb_pretrain_rounds=10,
            fl_vmb_stage1_objective="ce",
            fl_sat_aug_mode="baseline_view",
            use_sat_consistency=True,
            sat_cons_start_epoch=1,
            fl_baseline_view_ce_only=True,
            fl_baseline_view_ce_weight=1.0,
            use_aug=False,
            grl_lambda=1.0,
            lambda_rx_adv=0.0,
            lambda_dom=0.0,
            lambda_adv=0.0,
            lambda_orth=0.0,
            lambda_cons=0.0,
            lambda_group_ce=0.0,
            lambda_fishr=0.0,
            lambda_sat_cls=0.0,
            lambda_sat_cons=0.0,
            lambda_fed_proto=0.0,
            lambda_vmb_tx_proto=0.0,
            lambda_vmb_rx_proto=0.0,
            lambda_tx_adv_r=0.0,
            lambda_logit_kd=0.0,
            lambda_fed_coral=0.0,
            lambda_fed_coral_virtual=0.5,
            use_fed_coral=True,
            fed_coral_feature="z_id",
            fed_coral_mode="diag",
            fed_coral_start_round=1,
            fed_coral_min_count=1,
            fed_coral_scope="zid_virtual",
            fishr_min_domains=99,
            min_batch_domains_for_domain_loss=99,
            min_batch_domain_frac=1.0,
            group_ce_min_domains=99,
            use_fed_proto_stats=False,
            use_proto_evidence_bank=False,
            fl_style_zdom_probe_every=0,
            seed=1337,
        )
        trainer.device = torch.device("cpu")
        trainer.criterion = nn.CrossEntropyLoss()
        trainer.augment_fn = None
        trainer.sat_transform_fn = lambda x, scenario, round_idx, batch_idx: -x
        trainer.style_batch_fn = None
        trainer.style_bank = None
        trainer.style_transform = None
        trainer.virtual_domain_sampler = SimpleNamespace(clean_style_id=0)
        trainer.domain_label_map = {}
        trainer.global_proto_stats = None
        trainer.proto_evidence_bank = None
        trainer.logit_anchor_bank = None
        trainer.activation_token_codec = None
        trainer.client_splits = {}
        trainer.train_dataset = []
        trainer.global_coral_bank = FedCVSCoralStatsBank(num_classes=2, momentum=0.0, mode="diag")

        x = torch.tensor([[[2.0, 0.0], [0.0, 0.0]], [[-2.0, 0.0], [0.0, 0.0]]])
        y = torch.tensor([0, 1])
        d = torch.tensor([0, 0])
        metrics = FederatedTrainer._compute_local_objective(
            trainer,
            "rx0",
            x,
            y,
            d,
            round_idx=1,
            batch_idx=0,
            global_params={},
            mu=0.0,
            exclude_for_prox=(),
        )

        self.assertGreater(float(metrics["loss_coral_zid_virtual"]), 0.0)
        self.assertEqual(metrics["diag_coral_virtual_active"], 1.0)
        expected_total = (
            float(metrics["loss_cls"])
            + float(metrics["loss_baseline_sat_view"])
            + 0.5 * float(metrics["loss_coral_zid_virtual"])
        )
        self.assertAlmostEqual(float(metrics["loss"].detach().item()), expected_total, places=5)

    def test_zdom_negative_control_requires_zdom_feature_bank(self):
        from federated.fed_trainer import FederatedTrainer
        from federated.fedcvs_vmb import FedCVSCoralStatsBank, build_class_conditional_coral_stats

        class DifferentDimZdomModel(nn.Module):
            def forward(self, x, y_tx=None, grl_lambda=1.0, return_aux=True, domain_labels=None):
                del y_tx, grl_lambda, return_aux, domain_labels
                score = x[:, 0, 0].float()
                logits = torch.stack([score, -score], dim=1)
                z_id = torch.stack([score, score * 0.0 + 1.0], dim=1)
                z_dom = torch.stack([score, score * 0.0, score * 0.0 + 1.0], dim=1)
                dom_logits = torch.zeros(x.size(0), 2, device=x.device)
                return {
                    "tx_logits": logits,
                    "z_id": z_id,
                    "z_dom": z_dom,
                    "dom_logits": dom_logits,
                    "adv_dom_logits": dom_logits,
                }

        trainer = object.__new__(FederatedTrainer)
        trainer.model = DifferentDimZdomModel()
        trainer.vmb_enabled = True
        trainer.cfg = SimpleNamespace(
            train_mode="fedcvs_vmb",
            fl_local_objective="receiver_agnostic_bex02",
            fl_vmb_stage="auto",
            fl_vmb_pretrain_rounds=10,
            fl_vmb_stage1_objective="ce",
            fl_sat_aug_mode="baseline_view",
            use_sat_consistency=False,
            sat_cons_start_epoch=99,
            fl_baseline_view_ce_only=True,
            fl_baseline_view_ce_weight=1.0,
            use_aug=False,
            grl_lambda=1.0,
            lambda_rx_adv=0.0,
            lambda_dom=0.0,
            lambda_adv=0.0,
            lambda_orth=0.0,
            lambda_cons=0.0,
            lambda_group_ce=0.0,
            lambda_fishr=0.0,
            lambda_sat_cls=0.0,
            lambda_sat_cons=0.0,
            lambda_fed_proto=0.0,
            lambda_vmb_tx_proto=0.0,
            lambda_vmb_rx_proto=0.0,
            lambda_tx_adv_r=0.0,
            lambda_logit_kd=0.0,
            lambda_fl_coral_zdom_global=0.5,
            use_fed_coral=True,
            fed_coral_feature="z_id",
            fed_coral_mode="diag",
            fed_coral_start_round=1,
            fed_coral_min_count=1,
            fishr_min_domains=99,
            min_batch_domains_for_domain_loss=99,
            min_batch_domain_frac=1.0,
            group_ce_min_domains=99,
            use_fed_proto_stats=False,
            use_proto_evidence_bank=False,
            fl_style_zdom_probe_every=0,
            seed=1337,
        )
        trainer.device = torch.device("cpu")
        trainer.criterion = nn.CrossEntropyLoss()
        trainer.augment_fn = None
        trainer.sat_transform_fn = None
        trainer.style_batch_fn = None
        trainer.style_bank = None
        trainer.style_transform = None
        trainer.virtual_domain_sampler = SimpleNamespace(clean_style_id=0)
        trainer.domain_label_map = {}
        trainer.global_proto_stats = None
        trainer.proto_evidence_bank = None
        trainer.logit_anchor_bank = None
        trainer.activation_token_codec = None
        trainer.client_splits = {}
        trainer.train_dataset = []
        trainer.global_coral_bank = FedCVSCoralStatsBank(num_classes=2, momentum=0.0, mode="diag")
        trainer.global_coral_bank.update(
            build_class_conditional_coral_stats(
                torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
                torch.tensor([0, 1]),
                num_classes=2,
                mode="diag",
            )
        )

        x = torch.tensor([[[2.0, 0.0], [0.0, 0.0]], [[-2.0, 0.0], [0.0, 0.0]]])
        y = torch.tensor([0, 1])
        d = torch.tensor([0, 0])
        metrics = FederatedTrainer._compute_local_objective(
            trainer,
            "rx0",
            x,
            y,
            d,
            round_idx=1,
            batch_idx=0,
            global_params={},
            mu=0.0,
            exclude_for_prox=(),
        )

        self.assertEqual(float(metrics["loss_coral_zdom_global"]), 0.0)
        self.assertEqual(metrics["diag_coral_zdom_active"], 0.0)

    def test_vmb_stage1_ce_pretraining_uses_baseline_satellite_ce_only(self):
        from federated.fed_trainer import FederatedTrainer

        class RecordingFedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, x, y_tx=None, grl_lambda=1.0, return_aux=True, domain_labels=None):
                del y_tx, grl_lambda, return_aux, domain_labels
                self.calls.append(int(x.size(0)))
                score = x[:, 0, 0].float()
                logits = torch.stack([score, -score], dim=1)
                feat = torch.stack([score, score * 0.0, score * 0.0, score * 0.0], dim=1)
                dom_logits = torch.zeros(x.size(0), 2, device=x.device)
                return {
                    "tx_logits": logits,
                    "z_id": feat,
                    "z_dom": feat,
                    "dom_logits": dom_logits,
                    "adv_dom_logits": dom_logits,
                }

        model = RecordingFedModel()
        trainer = object.__new__(FederatedTrainer)
        trainer.model = model
        trainer.vmb_enabled = True
        trainer.cfg = SimpleNamespace(
            train_mode="fedcvs_vmb",
            fl_local_objective="receiver_agnostic_bex02",
            fl_vmb_stage="auto",
            fl_vmb_pretrain_rounds=10,
            fl_vmb_stage1_objective="ce",
            fl_sat_aug_mode="baseline_view",
            use_sat_consistency=True,
            sat_cons_start_epoch=1,
            fl_baseline_view_ce_only=True,
            fl_baseline_view_ce_weight=1.0,
            use_aug=False,
            grl_lambda=1.0,
            lambda_rx_adv=0.0,
            lambda_dom=0.0,
            lambda_adv=0.0,
            lambda_orth=0.0,
            lambda_cons=0.0,
            lambda_group_ce=0.0,
            lambda_fishr=0.0,
            lambda_sat_cls=0.0,
            lambda_sat_cons=0.0,
            lambda_fed_proto=0.0,
            lambda_vmb_tx_proto=0.0,
            lambda_vmb_rx_proto=0.0,
            lambda_tx_adv_r=0.0,
            lambda_logit_kd=0.0,
            fishr_min_domains=99,
            min_batch_domains_for_domain_loss=99,
            min_batch_domain_frac=1.0,
            group_ce_min_domains=99,
            use_fed_proto_stats=False,
            use_proto_evidence_bank=False,
            fl_style_zdom_probe_every=0,
            seed=1337,
        )
        trainer.device = torch.device("cpu")
        trainer.criterion = nn.CrossEntropyLoss()
        trainer.augment_fn = None
        trainer.sat_transform_fn = lambda x, scenario, round_idx, batch_idx: -x
        trainer.style_batch_fn = None
        trainer.style_bank = None
        trainer.style_transform = None
        trainer.virtual_domain_sampler = SimpleNamespace(clean_style_id=0)
        trainer.domain_label_map = {}
        trainer.global_proto_stats = None
        trainer.proto_evidence_bank = None
        trainer.logit_anchor_bank = None
        trainer.activation_token_codec = None
        trainer.client_splits = {}
        trainer.train_dataset = []

        x = torch.tensor([[[2.0, 0.0], [0.0, 0.0]], [[-2.0, 0.0], [0.0, 0.0]]])
        y = torch.tensor([0, 1])
        d = torch.tensor([0, 0])
        metrics = FederatedTrainer._compute_local_objective(
            trainer,
            "rx0",
            x,
            y,
            d,
            round_idx=1,
            batch_idx=0,
            global_params={},
            mu=0.0,
            exclude_for_prox=(),
        )
        calls_during_objective = list(model.calls)

        clean_ce = nn.CrossEntropyLoss()(model(x)["tx_logits"], y).item()
        sat_ce = nn.CrossEntropyLoss()(model(-x)["tx_logits"], y).item()
        self.assertEqual(calls_during_objective, [2, 2])
        self.assertAlmostEqual(float(metrics["loss_cls"]), clean_ce, places=5)
        self.assertAlmostEqual(float(metrics["loss_baseline_sat_view"]), sat_ce, places=5)
        self.assertAlmostEqual(float(metrics["loss"].detach().item()), clean_ce + sat_ce, places=5)
        self.assertEqual(metrics["vmb_stage"], "stage1")
        self.assertEqual(metrics["diag_baseline_sat_view_active"], 1.0)
        self.assertEqual(metrics["diag_sat_cls_active"], 1.0)
        self.assertEqual(metrics["diag_rx_adv_active"], 0.0)


if __name__ == "__main__":
    unittest.main()
