import sys
import tempfile
import unittest
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class TinyWisigLikeDataset(Dataset):
    def __init__(self, n_per_receiver=8, num_receivers=2, num_tx=4, length=128):
        self.samples = []
        gen = torch.Generator().manual_seed(11)
        for rx in range(num_receivers):
            for i in range(n_per_receiver):
                tx = i % num_tx
                x = torch.randn(2, length, generator=gen) + float(tx) * 0.01 + float(rx) * 0.001
                meta = {"tx_i": tx, "rx_i": rx, "day_i": 0, "sig_i": i}
                self.samples.append((x, tx, rx, meta))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_cfg(mode, output_dir, log_dir):
    return SimpleNamespace(
        train_mode=mode,
        fl_client_key="receiver",
        fl_min_samples_per_client=1,
        fl_drop_small_clients=False,
        fl_verbose_clients=False,
        batch_size=4,
        fl_num_workers=0,
        output_dir=str(output_dir),
        log_dir=str(log_dir),
        fedbase_paper_method={"fedriei": "FedRIEI", "fedfa": "FedFA", "fucl": "FUCL", "rafl": "RAFL"}[mode],
        wisig_train_ratio=0.1,
        epochs=1,
        fl_rounds=1,
        fl_clients_per_round=1.0,
        seed=5,
        eval_max_batches=1,
        sat_eval_max_batches=-1,
        eval_sat_channel=False,
        eval_sat_scenario_list=[],
        eval_sat_on="main",
        fl_test_eval_interval=1,
        fl_test_eval_last_n=0,
        fl_test_eval_final_offsets="",
        fl_local_epochs=1,
        lr=0.001 if mode != "fedfa" else 0.01,
        wd=0.0,
        grad_clip=1.0,
        fedriei_lambda_mi=1.2,
        fedriei_lambda_ie=1.2,
        fedriei_gradient_compression="none",
        fedriei_compression_noise_std=0.01,
        fedriei_server_lr=0.0,
        fedfa_align_lambda=0.03,
        fucl_temperature=0.05,
        fucl_pretrain_lr=0.0003,
        fucl_finetune_lr=0.001,
        fucl_finetune_epochs=1,
        fucl_local_validation_ratio=0.1,
        fucl_local_lr_patience=10,
        fucl_local_lr_decay=0.1,
        fucl_local_early_stop_patience=20,
        fucl_local_max_epochs=0,
        fucl_validation_max_batches=1,
        fucl_channel_noise_std=0.01,
        fucl_sample_rate_hz=500000.0,
        fucl_tdl_rms_delay_min_ns=5.0,
        fucl_tdl_rms_delay_max_ns=300.0,
        fucl_tdl_doppler_min_hz=0.0,
        fucl_tdl_doppler_max_hz=5.0,
        fucl_tdl_snr_min_db=0.0,
        fucl_tdl_snr_max_db=80.0,
        fucl_tdl_num_taps=3,
        fucl_cis_n_fft=64,
        fucl_cis_hop_length=32,
        fucl_cis_win_length=64,
        fucl_cis_crop_fraction=0.30,
        fucl_cis_freq_bins=26,
        fucl_cis_time_bins=126,
        fucl_cis_normalize="none",
        rafl_lambda_rx=1.0,
        rafl_momentum=0.0,
        rafl_client_selection="label_loss_driven",
        rafl_selected_clients=0,
        rafl_selection_max_batches=1,
        rafl_candidate_clients=0,
        rafl_candidate_fraction=1.0,
        rafl_selection_eval_ratio=0.1,
        rafl_selection_dataset="internal_train_split",
        rafl_input_version="wisig_native",
        rafl_spec_n_fft=64,
        rafl_spec_hop_length=32,
        rafl_spec_win_length=64,
        rafl_spec_freq_bins=52,
        rafl_spec_time_bins=126,
        rafl_spec_normalize="zscore",
    )


class FedbasePaperTrainerSmokeTest(unittest.TestCase):
    def test_four_paper_modes_complete_one_round_on_wisig_like_batches(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        dataset = TinyWisigLikeDataset()
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        for mode in ["fedriei", "fedfa", "fucl", "rafl"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                cfg = make_cfg(mode, tmp_path / "runs", tmp_path / "logs")
                model = build_fedbase_paper_model(mode, num_classes=4, num_receivers=2, feature_dim=32)
                trainer = FedbasePaperTrainer(
                    model,
                    dataset,
                    loader,
                    {"test_unseen_day_unseen_rx": loader},
                    cfg,
                    device=torch.device("cpu"),
                    split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
                    named_test_meta={"test_unseen_day_unseen_rx": {"size": len(dataset)}},
                )

                summary = trainer.train()

                self.assertEqual(summary["train_mode"], mode)
                self.assertTrue((tmp_path / "runs" / "summary.json").is_file())
                self.assertTrue((tmp_path / "logs" / "logs.jsonl").is_file())
                self.assertTrue((tmp_path / "logs" / "metrics.csv").is_file())

    def test_named_eval_allows_unseen_receiver_labels(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        train_ds = TinyWisigLikeDataset(n_per_receiver=4, num_receivers=2, num_tx=4)
        test_ds = TinyWisigLikeDataset(n_per_receiver=4, num_receivers=1, num_tx=4)
        for idx, (x, y, _rx, meta) in enumerate(test_ds.samples):
            test_ds.samples[idx] = (x, y, 2, {**meta, "rx_i": 2})
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

        for mode in ["fedriei", "fedfa", "fucl", "rafl"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                cfg = make_cfg(mode, tmp_path / "runs", tmp_path / "logs")
                model = build_fedbase_paper_model(mode, num_classes=4, num_receivers=2, feature_dim=32)
                trainer = FedbasePaperTrainer(
                    model,
                    train_ds,
                    test_loader,
                    {"test_unseen_day_unseen_rx": test_loader},
                    cfg,
                    device=torch.device("cpu"),
                    split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
                )

                stats = trainer._evaluate_named()

                self.assertIn("test_unseen_day_unseen_rx", stats)
                self.assertEqual(stats["test_unseen_day_unseen_rx"]["tx_total"], len(test_ds))

    def test_rafl_all_selection_ignores_fraction(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        dataset = TinyWisigLikeDataset(n_per_receiver=4, num_receivers=3, num_tx=4)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("rafl", tmp_path / "runs", tmp_path / "logs")
            cfg.fl_clients_per_round = 0.5
            cfg.rafl_client_selection = "all"
            model = build_fedbase_paper_model("rafl", num_classes=4, num_receivers=3, feature_dim=32)
            trainer = FedbasePaperTrainer(
                model,
                dataset,
                loader,
                {"test_unseen_day_unseen_rx": loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
            )

            self.assertEqual(set(trainer._selected_clients(1)), set(trainer.client_splits.keys()))

    def test_rafl_adaptive_selection_counts_follow_current_client_count(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        dataset = TinyWisigLikeDataset(n_per_receiver=2, num_receivers=12, num_tx=4)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("rafl", tmp_path / "runs", tmp_path / "logs")
            cfg.fl_clients_per_round = 0.5
            cfg.rafl_candidate_fraction = 1.0
            model = build_fedbase_paper_model("rafl", num_classes=4, num_receivers=12, feature_dim=32)
            trainer = FedbasePaperTrainer(
                model,
                dataset,
                loader,
                {"test_unseen_day_unseen_rx": loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
            )

            self.assertEqual(trainer._resolve_rafl_selected_count(12), 6)
            self.assertEqual(trainer._resolve_rafl_candidate_count(12, 6), 12)

            cfg.rafl_candidate_fraction = 0.25
            self.assertEqual(trainer._resolve_rafl_candidate_count(12, 6), 6)

            cfg.rafl_selected_clients = 20
            cfg.rafl_candidate_clients = 3
            self.assertEqual(trainer._resolve_rafl_selected_count(12), 12)
            self.assertEqual(trainer._resolve_rafl_candidate_count(12, 12), 12)

    def test_rafl_config_records_spectrogram_and_external_E_rxj(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        train_ds = TinyWisigLikeDataset(n_per_receiver=4, num_receivers=2, num_tx=4)
        selection_ds = TinyWisigLikeDataset(n_per_receiver=2, num_receivers=2, num_tx=4)
        loader = DataLoader(train_ds, batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("rafl", tmp_path / "runs", tmp_path / "logs")
            cfg.rafl_input_version = "paper_52x126"
            model = build_fedbase_paper_model("rafl", num_classes=4, num_receivers=2, feature_dim=32)
            FedbasePaperTrainer(
                model,
                train_ds,
                loader,
                {"test_unseen_day_unseen_rx": loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
                rafl_selection_dataset=selection_ds,
            )

            with (tmp_path / "runs" / "fedbase_config.json").open("r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["rafl_spectrogram"]["input_version"], "paper_52x126")
            self.assertEqual(payload["rafl_spectrogram"]["representation"], "log_magnitude_stft")
            self.assertEqual(payload["rafl_spectrogram"]["input_channels"], 1)
            self.assertEqual(payload["rafl_spectrogram"]["paper_target_shape_b_c_f_t"], ["B", 1, 52, 126])
            self.assertTrue(payload["rafl_spectrogram"]["resize_to_paper_shape"])
            self.assertEqual(payload["rafl_selection_source"], "external_heldout_E_rxj")
            self.assertTrue(all("Raw-IQ ResNet1D" not in item for item in payload["method_level_adaptations"]))

    def test_rafl_config_records_strict_lld_profile_and_resolved_counts(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        train_ds = TinyWisigLikeDataset(n_per_receiver=4, num_receivers=6, num_tx=4)
        loader = DataLoader(train_ds, batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("rafl", tmp_path / "runs", tmp_path / "logs")
            cfg.fedbase_paper_profile = "strict_paper"
            cfg.fl_rounds = 300
            cfg.rafl_selected_clients = 5
            cfg.rafl_candidate_clients = 10
            cfg.rafl_input_version = "paper_52x126"
            model = build_fedbase_paper_model("rafl", num_classes=4, num_receivers=6, feature_dim=32)
            trainer = FedbasePaperTrainer(
                model,
                train_ds,
                loader,
                {"test_unseen_day_unseen_rx": loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
            )

            with (tmp_path / "runs" / "fedbase_config.json").open("r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertEqual(payload["fedbase_paper_profile"], "strict_paper")
            self.assertEqual(payload["rafl_client_selection_profile"], "paper_strict_lld")
            self.assertEqual(payload["rafl_candidate_clients"], 10)
            self.assertEqual(payload["rafl_selected_clients"], 5)
            self.assertEqual(payload["rafl_resolved_candidate_clients"], 6)
            self.assertEqual(payload["rafl_resolved_selected_clients"], 5)
            self.assertEqual(payload["rafl_spectrogram"]["input_version"], "paper_52x126")

    def test_rafl_input_versions_produce_paper_wisig_native_and_complex_shapes(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        dataset = TinyWisigLikeDataset(n_per_receiver=2, num_receivers=1, num_tx=2, length=128)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("rafl", tmp_path / "runs", tmp_path / "logs")
            model = build_fedbase_paper_model("rafl", num_classes=4, num_receivers=1, feature_dim=32)
            trainer = FedbasePaperTrainer(
                model,
                dataset,
                loader,
                {"test_unseen_day_unseen_rx": loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
            )
            x = torch.randn(2, 2, 128)

            cfg.rafl_input_version = "wisig_native"
            native = trainer._rafl_spectrogram(x)
            cfg.rafl_input_version = "paper_52x126"
            paper = trainer._rafl_spectrogram(x)
            cfg.rafl_input_version = "wisig_complex"
            complex_spec = trainer._rafl_spectrogram(x)

            self.assertEqual(tuple(paper.shape), (2, 1, 52, 126))
            self.assertNotEqual(tuple(native.shape[-2:]), (52, 126))
            self.assertEqual(complex_spec.size(1), 2)
            self.assertEqual(tuple(complex_spec.shape[-2:]), tuple(native.shape[-2:]))

    def test_rafl_label_losses_use_heldout_E_rxj_not_training_loader(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        train_ds = TinyWisigLikeDataset(n_per_receiver=4, num_receivers=1, num_tx=1)
        selection_ds = TinyWisigLikeDataset(n_per_receiver=4, num_receivers=1, num_tx=1)
        for idx, (x, _y, _rx, meta) in enumerate(selection_ds.samples):
            selection_ds.samples[idx] = (x, 3, 0, {**meta, "tx_i": 3, "rx_i": 0})
        loader = DataLoader(train_ds, batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("rafl", tmp_path / "runs", tmp_path / "logs")
            model = build_fedbase_paper_model("rafl", num_classes=4, num_receivers=1, feature_dim=32)
            trainer = FedbasePaperTrainer(
                model,
                train_ds,
                loader,
                {"test_unseen_day_unseen_rx": loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
                rafl_selection_dataset=selection_ds,
            )

            losses = trainer._rafl_label_losses(["rx0"])

            self.assertEqual(set(losses["rx0"].keys()), {3})

    def test_fucl_internal_validation_split_holds_out_ten_percent_per_client(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        dataset = TinyWisigLikeDataset(n_per_receiver=10, num_receivers=2, num_tx=4)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("fucl", tmp_path / "runs", tmp_path / "logs")
            model = build_fedbase_paper_model("fucl", num_classes=4, num_receivers=2, feature_dim=32)
            trainer = FedbasePaperTrainer(
                model,
                dataset,
                loader,
                {"test_unseen_day_unseen_rx": loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
            )

            self.assertEqual({cid: len(ids) for cid, ids in trainer.client_splits.items()}, {"rx0": 9, "rx1": 9})
            self.assertEqual({cid: len(ids) for cid, ids in trainer.fucl_validation_client_splits.items()}, {"rx0": 1, "rx1": 1})

    def test_fucl_local_pretrain_records_plateau_lr_decay_and_early_stop(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        dataset = TinyWisigLikeDataset(n_per_receiver=6, num_receivers=1, num_tx=2, length=128)
        loader = DataLoader(dataset, batch_size=3, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("fucl", tmp_path / "runs", tmp_path / "logs")
            cfg.batch_size = 3
            cfg.fucl_local_max_epochs = 25
            cfg.fucl_local_lr_patience = 2
            cfg.fucl_local_early_stop_patience = 4
            model = build_fedbase_paper_model("fucl", num_classes=4, num_receivers=1, feature_dim=32)
            trainer = FedbasePaperTrainer(
                model,
                dataset,
                loader,
                {"test_unseen_day_unseen_rx": loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
            )
            gen = torch.Generator().manual_seed(23)
            trainer._make_fucl_views = lambda x, *_args: (  # type: ignore[method-assign]
                torch.randn(int(x.size(0)), 1, 26, 126, generator=gen),
                torch.randn(int(x.size(0)), 1, 26, 126, generator=gen),
            )
            validation_values = iter([1.0, 1.1, 1.2, 1.3, 1.4])
            trainer._fucl_client_validation_loss = lambda *_args, **_kwargs: next(validation_values)  # type: ignore[attr-defined]

            result = trainer._train_fucl_pretrain_round(["rx0"], round_idx=1)

            metrics = result["components"]["fucl_client_pretrain"]["rx0"]
            self.assertEqual(metrics["epochs_run"], 5)
            self.assertTrue(metrics["early_stopped"])
            self.assertEqual(metrics["lr_reductions"], 2)
            self.assertAlmostEqual(metrics["best_val_loss"], 1.0)

    def test_fucl_finetune_keeps_client_specific_classifiers_for_paper_eval(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        dataset = TinyWisigLikeDataset(n_per_receiver=4, num_receivers=2, num_tx=4, length=128)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("fucl", tmp_path / "runs", tmp_path / "logs")
            cfg.batch_size = 2
            cfg.fucl_finetune_epochs = 1
            model = build_fedbase_paper_model("fucl", num_classes=4, num_receivers=2, feature_dim=32)
            trainer = FedbasePaperTrainer(
                model,
                dataset,
                loader,
                {"test_unseen_day_unseen_rx": loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
            )
            gen = torch.Generator().manual_seed(37)
            trainer._make_fucl_views = lambda x, *_args: (  # type: ignore[method-assign]
                torch.randn(int(x.size(0)), 1, 26, 126, generator=gen),
                torch.randn(int(x.size(0)), 1, 26, 126, generator=gen),
            )

            result = trainer._train_fucl_finetune()

            self.assertEqual(set(trainer.fucl_client_states.keys()), {"rx0", "rx1"})
            self.assertEqual(result["fucl_finetune_paper_eval_mode"], "client_specific_classification_nns")
            self.assertIn("client_specific_classification_nns", result["fucl_finetune_adapter"])

    def test_fucl_client_specific_eval_uses_seen_client_and_ensembles_unseen_receiver(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        train_dataset = TinyWisigLikeDataset(n_per_receiver=2, num_receivers=2, num_tx=4, length=128)
        test_dataset = TinyWisigLikeDataset(n_per_receiver=1, num_receivers=3, num_tx=4, length=128)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("fucl", tmp_path / "runs", tmp_path / "logs")
            cfg.batch_size = 2
            model = build_fedbase_paper_model("fucl", num_classes=4, num_receivers=2, feature_dim=16)
            trainer = FedbasePaperTrainer(
                model,
                train_dataset,
                train_loader,
                {"test_unseen_day_unseen_rx": test_loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
            )

            def state_predicting(target_class: int):
                client_model = build_fedbase_paper_model("fucl", num_classes=4, num_receivers=2, feature_dim=16)
                with torch.no_grad():
                    for param in client_model.parameters():
                        param.zero_()
                    client_model.classifier[-1].bias[int(target_class)] = 5.0
                return client_model.state_dict()

            trainer.fucl_client_states = {"rx0": state_predicting(0), "rx1": state_predicting(1)}
            trainer._fucl_signal_representation = lambda x: torch.zeros(int(x.size(0)), 1, 26, 126)  # type: ignore[method-assign]

            stats = trainer._evaluate_fucl_client_specific_loader(test_loader)

            self.assertEqual(stats["tx_total"], 3)
            self.assertEqual(stats["fucl_seen_receiver_matched"], 2)
            self.assertEqual(stats["fucl_unseen_receiver_ensembled"], 1)
            self.assertEqual(stats["fucl_eval_mode"], "client_specific_seen_receiver_else_source_client_ensemble")

    def test_fucl_finetune_records_plateau_lr_decay_and_early_stop(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        dataset = TinyWisigLikeDataset(n_per_receiver=6, num_receivers=1, num_tx=2, length=128)
        loader = DataLoader(dataset, batch_size=3, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("fucl", tmp_path / "runs", tmp_path / "logs")
            cfg.batch_size = 3
            cfg.fucl_local_max_epochs = 25
            cfg.fucl_local_lr_patience = 2
            cfg.fucl_local_early_stop_patience = 4
            model = build_fedbase_paper_model("fucl", num_classes=4, num_receivers=1, feature_dim=32)
            trainer = FedbasePaperTrainer(
                model,
                dataset,
                loader,
                {"test_unseen_day_unseen_rx": loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
            )
            gen = torch.Generator().manual_seed(41)
            trainer._make_fucl_views = lambda x, *_args: (  # type: ignore[method-assign]
                torch.randn(int(x.size(0)), 1, 26, 126, generator=gen),
                torch.randn(int(x.size(0)), 1, 26, 126, generator=gen),
            )
            validation_values = iter([1.0, 1.1, 1.2, 1.3, 1.4])
            trainer._fucl_client_supervised_validation_loss = lambda *_args, **_kwargs: next(validation_values)  # type: ignore[attr-defined]

            result = trainer._train_fucl_finetune()

            metrics = result["fucl_client_finetune"]["rx0"]
            self.assertEqual(metrics["epochs_run"], 5)
            self.assertTrue(metrics["early_stopped"])
            self.assertEqual(metrics["lr_reductions"], 2)
            self.assertAlmostEqual(metrics["best_val_loss"], 1.0)

    def test_final_offsets_match_federated_eval_schedule(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        dataset = TinyWisigLikeDataset(n_per_receiver=4, num_receivers=2, num_tx=4)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("fedfa", tmp_path / "runs", tmp_path / "logs")
            cfg.fl_rounds = 10
            cfg.fl_test_eval_interval = 0
            cfg.fl_test_eval_final_offsets = "5,3,1"
            model = build_fedbase_paper_model("fedfa", num_classes=4, num_receivers=2, feature_dim=32)
            trainer = FedbasePaperTrainer(
                model,
                dataset,
                loader,
                {"test_unseen_day_unseen_rx": loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
            )

            rounds = [r for r in range(1, 11) if trainer._should_run_eval(r)]

            self.assertEqual(rounds, [6, 8, 10])

    def test_tests_run_on_validation_improvement_and_final_round(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        dataset = TinyWisigLikeDataset(n_per_receiver=4, num_receivers=2, num_tx=4)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("fedfa", tmp_path / "runs", tmp_path / "logs")
            cfg.fl_rounds = 3
            cfg.fl_test_eval_interval = 0
            cfg.fl_test_eval_last_n = 0
            cfg.fl_test_eval_final_offsets = ""
            model = build_fedbase_paper_model("fedfa", num_classes=4, num_receivers=2, feature_dim=32)
            trainer = FedbasePaperTrainer(
                model,
                dataset,
                loader,
                {"test_unseen_day_unseen_rx": loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
            )
            current_round = {"value": 0}
            test_rounds = []
            val_by_round = {1: 10.0, 2: 20.0, 3: 15.0}
            test_by_round = {1: 60.0, 2: 50.0, 3: 40.0}

            def fake_train_round(selected):
                del selected
                current_round["value"] += 1
                return {"train_loss": 0.0, "train_acc": 0.0}

            def fake_val_eval(loader_obj, max_batches=0):
                del loader_obj, max_batches
                return {"tx_acc": val_by_round[current_round["value"]], "tx_correct": 1, "tx_total": 1}

            def fake_named_eval():
                round_idx = current_round["value"]
                test_rounds.append(round_idx)
                return {"test_unseen_day_unseen_rx": {"tx_acc": test_by_round[round_idx], "tx_correct": 1, "tx_total": 1}}

            trainer._train_round_fedfa = fake_train_round
            trainer._evaluate_loader = fake_val_eval
            trainer._evaluate_named = fake_named_eval
            trainer._evaluate_sat_named = lambda: {}

            summary = trainer.train()

            self.assertEqual(test_rounds, [1, 2, 3])
            self.assertEqual(summary["best_round"], 2)
            self.assertEqual(summary["best_val_tx_acc"], 20.0)
            self.assertEqual(summary["best_primary_test_tx_acc"], 50.0)
            self.assertEqual(summary["final_primary_test_tx_acc"], 40.0)

    def test_fedriei_final_window_summary_and_loss_metrics_are_logged(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        dataset = TinyWisigLikeDataset(n_per_receiver=4, num_receivers=2, num_tx=4)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("fedriei", tmp_path / "runs", tmp_path / "logs")
            cfg.fl_rounds = 3
            cfg.fl_test_eval_interval = 0
            cfg.fl_test_eval_last_n = 2
            cfg.fl_test_eval_final_offsets = ""
            model = build_fedbase_paper_model("fedriei", num_classes=4, num_receivers=2, feature_dim=32)
            trainer = FedbasePaperTrainer(
                model,
                dataset,
                loader,
                {"test_unseen_day_unseen_rx": loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
            )
            current_round = {"value": 0}
            val_by_round = {1: 10.0, 2: 9.0, 3: 8.0}
            test_by_round = {1: 30.0, 2: 40.0, 3: 50.0}

            def fake_train_standard(selected, round_idx):
                del selected
                current_round["value"] = int(round_idx)
                return {
                    "train_loss": 1.0,
                    "train_acc": 2.0,
                    "components": {
                        "ce_phase_loss_ce": 1.1,
                        "disentangle_phase_loss_mi": 0.2,
                        "disentangle_phase_loss_ie": 0.3,
                        "disentangle_phase_loss": -0.12,
                    },
                }

            def fake_val_eval(loader_obj, max_batches=0):
                del loader_obj, max_batches
                return {"tx_acc": val_by_round[current_round["value"]], "tx_correct": 1, "tx_total": 1}

            def fake_named_eval():
                return {
                    "test_unseen_day_unseen_rx": {
                        "tx_acc": test_by_round[current_round["value"]],
                        "tx_correct": 1,
                        "tx_total": 1,
                    }
                }

            trainer._train_standard_round = fake_train_standard
            trainer._evaluate_loader = fake_val_eval
            trainer._evaluate_named = fake_named_eval
            trainer._evaluate_sat_named = lambda: {}

            summary = trainer.train()

            self.assertEqual(summary["paper_eval_window"]["name"], "final2")
            self.assertEqual(summary["paper_eval_window"]["rounds"], [2, 3])
            self.assertAlmostEqual(summary["paper_eval_window"]["primary_test_tx_acc"]["mean"], 45.0)
            self.assertAlmostEqual(summary["paper_eval_window"]["primary_test_tx_acc"]["std"], 5.0)
            with (tmp_path / "logs" / "metrics.csv").open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[-1]["fedriei_loss_mi"], "0.2")
            self.assertEqual(rows[-1]["fedriei_loss_ie"], "0.3")
            self.assertIn("named_test_tx_acc_json", rows[-1])
            self.assertEqual(json.loads(rows[-1]["named_test_tx_acc_json"])["test_unseen_day_unseen_rx"], 50.0)
            named_lines = trainer._format_named_test_stdout(
                {"test_unseen_day_unseen_rx": {"tx_acc": 55.5, "tx_total": 12}},
                round_label="003",
            )
            self.assertIn("[FEDBASE-TEST][R003]", named_lines[0])
            self.assertIn("split=test_unseen_day_unseen_rx", named_lines[0])
            sat_lines = trainer._format_sat_test_stdout(
                {
                    "clear_leo": {
                        "aggregate": {"tx_acc": 42.0, "tx_correct": 21, "tx_total": 50},
                        "strict_udu": 44.0,
                        "named": {"test_unseen_day_unseen_rx": {"tx_acc": 44.0, "tx_total": 12}},
                        "selected_names": ["test_unseen_day_unseen_rx"],
                    }
                },
                round_label="FINETUNE",
            )
            self.assertIn("[FEDBASE-SAT][FINETUNE]", sat_lines[0])
            self.assertIn("scenario=clear_leo", sat_lines[0])
            self.assertIn("split=aggregate", sat_lines[0])
            self.assertTrue(any("split=strict_udu" in line for line in sat_lines))

    def test_satellite_stdout_and_metrics_use_nested_eval_structure(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        dataset = TinyWisigLikeDataset(n_per_receiver=4, num_receivers=2, num_tx=4)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        named_loaders = {
            "test_unseen_day_seen_rx": loader,
            "test_seen_day_unseen_rx": loader,
            "test_unseen_day_unseen_rx": loader,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("fedfa", tmp_path / "runs", tmp_path / "logs")
            cfg.eval_sat_channel = True
            cfg.eval_sat_scenario_list = ["clear_leo"]
            cfg.eval_sat_on = "main"
            model = build_fedbase_paper_model("fedfa", num_classes=4, num_receivers=2, feature_dim=32)
            trainer = FedbasePaperTrainer(
                model,
                dataset,
                loader,
                named_loaders,
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
            )
            values = iter([30.0, 40.0, 50.0])

            def fake_sat_loader(loader_obj, scenario, *, seed, max_batches=0):
                del loader_obj, scenario, seed, max_batches
                acc = next(values)
                return {"tx_acc": acc, "tx_correct": int(acc), "tx_total": 100}

            trainer._evaluate_sat_loader = fake_sat_loader

            stats = trainer._evaluate_sat_named()
            payload = json.loads(trainer._sat_tx_acc_json(stats))
            lines = trainer._format_sat_test_stdout(stats, round_label="001")

            self.assertEqual(payload["clear_leo"]["strict_udu"], 50.0)
            self.assertEqual(payload["clear_leo"]["named"]["test_seen_day_unseen_rx"], 40.0)
            self.assertIn("aggregate", payload["clear_leo"])
            self.assertTrue(any("split=aggregate" in line for line in lines))
            self.assertTrue(any("split=strict_udu" in line for line in lines))
            self.assertTrue(any("split=test_unseen_day_unseen_rx" in line for line in lines))

    def test_fedriei_local_train_returns_algorithm1_compressed_gradient(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        dataset = TinyWisigLikeDataset(n_per_receiver=4, num_receivers=2, num_tx=4)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("fedriei", tmp_path / "runs", tmp_path / "logs")
            cfg.fedriei_gradient_compression = "signsgd"
            model = build_fedbase_paper_model("fedriei", num_classes=4, num_receivers=2, feature_dim=32)
            trainer = FedbasePaperTrainer(
                model,
                dataset,
                loader,
                {"test_unseen_day_unseen_rx": loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
            )

            result = trainer._local_train_fedriei(sorted(trainer.client_splits)[0], 1)

            self.assertIn("gradient", result)
            self.assertTrue(result["gradient"])
            flattened = torch.cat([value.reshape(-1).float() for value in result["gradient"].values()])
            self.assertTrue(torch.all((flattened == -1.0) | (flattened == 1.0)))

    def test_fucl_views_change_across_client_and_batch(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        dataset = TinyWisigLikeDataset(n_per_receiver=4, num_receivers=2, num_tx=4)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("fucl", tmp_path / "runs", tmp_path / "logs")
            model = build_fedbase_paper_model("fucl", num_classes=4, num_receivers=2, feature_dim=32)
            trainer = FedbasePaperTrainer(
                model,
                dataset,
                loader,
                {"test_unseen_day_unseen_rx": loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
            )
            x = torch.zeros(4, 2, 128)

            a1, _ = trainer._make_fucl_views(x, 1, "rx0", 0)
            a2, _ = trainer._make_fucl_views(x, 1, "rx0", 1)
            b1, _ = trainer._make_fucl_views(x, 1, "rx1", 0)

            self.assertEqual(tuple(a1.shape), (4, 1, 26, 126))
            self.assertFalse(torch.allclose(a1, a2))
            self.assertFalse(torch.allclose(a1, b1))

    def test_fucl_best_checkpoint_uses_finetune_not_pretrain_random_classifier_val(self):
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        dataset = TinyWisigLikeDataset(n_per_receiver=4, num_receivers=2, num_tx=4)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("fucl", tmp_path / "runs", tmp_path / "logs")
            cfg.fl_rounds = 2
            model = build_fedbase_paper_model("fucl", num_classes=4, num_receivers=2, feature_dim=32)
            trainer = FedbasePaperTrainer(
                model,
                dataset,
                loader,
                {"test_unseen_day_unseen_rx": loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
            )
            current_round = {"value": 0}
            named_calls = []

            def fake_pretrain(selected, round_idx):
                del selected
                current_round["value"] = int(round_idx)
                return {"train_loss": 0.1 * round_idx, "train_acc": float("nan")}

            def fake_contrastive_val(round_idx):
                return {1: 5.0, 2: 2.0}[int(round_idx)]

            def fake_finetune():
                current_round["value"] = 3
                trainer.fucl_client_states = {
                    "rx0": {key: value.detach().cpu().clone() for key, value in trainer.global_state.items()}
                }
                return {"train_loss": 0.25, "train_acc": 75.0}

            def fake_val_eval(loader_obj, max_batches=0):
                del loader_obj, max_batches
                return {"tx_acc": {1: 90.0, 2: 95.0, 3: 40.0}[current_round["value"]], "tx_correct": 1, "tx_total": 1}

            def fake_aggregate_named_eval():
                return {"test_unseen_day_unseen_rx": {"tx_acc": 11.0, "tx_correct": 1, "tx_total": 1}}

            def fake_client_named_eval():
                named_calls.append(current_round["value"])
                return {"test_unseen_day_unseen_rx": {"tx_acc": 33.0, "tx_correct": 1, "tx_total": 1}}

            trainer._train_fucl_pretrain_round = fake_pretrain
            trainer._fucl_contrastive_val_loss = fake_contrastive_val
            trainer._train_fucl_finetune = fake_finetune
            trainer._evaluate_loader = fake_val_eval
            trainer._evaluate_fucl_client_specific_loader = fake_val_eval
            trainer._evaluate_named = fake_aggregate_named_eval
            trainer._evaluate_fucl_client_specific_named = fake_client_named_eval
            trainer._evaluate_sat_named = lambda: {}
            trainer._evaluate_fucl_client_specific_sat_named = lambda: {}

            summary = trainer.train()

            self.assertEqual(named_calls, [3])
            self.assertEqual(summary["best_round"], "finetune")
            self.assertEqual(summary["best_val_tx_acc"], 40.0)
            self.assertEqual(summary["best_primary_test_tx_acc"], 33.0)
            self.assertEqual(summary["fucl_best_pretrain_round"], 2)
            self.assertEqual(summary["fucl_best_pretrain_val_loss"], 2.0)
            self.assertTrue(summary["fucl_client_state_checkpointed"])
            checkpoint = torch.load(tmp_path / "runs" / "best_model.pt", map_location="cpu")
            self.assertIn("fucl_client_states", checkpoint)
            self.assertEqual(sorted(checkpoint["fucl_client_states"]), ["rx0"])
            self.assertEqual(checkpoint["fucl_final_eval_mode"], "client_specific_seen_receiver_else_source_client_ensemble")
            with (tmp_path / "logs" / "metrics.csv").open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["val_tx_acc"], "nan")
            self.assertEqual(rows[-1]["fucl_common_aggregate_primary_test_tx_acc"], "11.0")
            with (tmp_path / "logs" / "logs.jsonl").open("r", encoding="utf-8") as f:
                first_event = json.loads(f.readline())
            self.assertEqual(first_event["components"]["fucl_pretrain_classifier_val_tx_acc_diagnostic"], 90.0)

    def test_rafl_trainer_scales_receiver_head_loss_by_lambda(self):
        import federated.fedbase_paper_trainer as trainer_mod
        from federated.fedbase_paper_trainer import FedbasePaperTrainer, build_fedbase_paper_model

        dataset = TinyWisigLikeDataset(n_per_receiver=4, num_receivers=2, num_tx=4)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = make_cfg("rafl", tmp_path / "runs", tmp_path / "logs")
            cfg.rafl_lambda_rx = 0.1
            model = build_fedbase_paper_model("rafl", num_classes=4, num_receivers=2, feature_dim=32)
            trainer = FedbasePaperTrainer(
                model,
                dataset,
                loader,
                {"test_unseen_day_unseen_rx": loader},
                cfg,
                device=torch.device("cpu"),
                split_info={"primary_named_test": "test_unseen_day_unseen_rx"},
            )

            captured_grl = []
            original_forward_outputs = trainer_mod._forward_outputs

            def capture_forward(model_obj, x, *, grl_lambda=1.0):
                captured_grl.append(float(grl_lambda))
                return original_forward_outputs(model_obj, x, grl_lambda=grl_lambda)

            try:
                trainer_mod._forward_outputs = capture_forward
                result = trainer._local_train_rafl(next(iter(trainer.client_splits)), 1)
            finally:
                trainer_mod._forward_outputs = original_forward_outputs

            self.assertAlmostEqual(result["metrics"]["rafl_lambda_rx"], 0.1, places=6)
            self.assertAlmostEqual(result["metrics"]["rafl_receiver_loss_weight"], 0.1, places=6)
            self.assertTrue(captured_grl)
            self.assertTrue(all(abs(value - 1.0) < 1e-8 for value in captured_grl))


if __name__ == "__main__":
    unittest.main()
