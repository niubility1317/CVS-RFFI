import argparse
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


class PostStageTrainerEntrypointsTest(unittest.TestCase):
    def test_mean_logs_ignores_nonfinite_optional_batch_metrics(self):
        from post_stage_common import mean_logs

        result = mean_logs(
            [
                {"active": 0.0, "optional": float("nan")},
                {"active": 1.0, "optional": 3.0},
                {"active": 1.0, "optional": float("inf")},
            ]
        )

        self.assertEqual(result["active"], 2.0 / 3.0)
        self.assertEqual(result["optional"], 3.0)

    def test_post_stage_sat_eval_defaults_to_three_main_ood_splits(self):
        from post_stage_cli import MAIN_SAT_EVAL_ON, add_sat_eval_args
        from cvsrffi.eval import MAIN_SAT_EVAL_ON_NAMES, resolve_sat_eval_loader_names

        parser = argparse.ArgumentParser()
        add_sat_eval_args(parser)
        args = parser.parse_args([])
        named_loaders = {name: [] for name in MAIN_SAT_EVAL_ON_NAMES}

        self.assertEqual(args.eval_sat_on, MAIN_SAT_EVAL_ON)
        self.assertEqual(resolve_sat_eval_loader_names(named_loaders, ""), MAIN_SAT_EVAL_ON_NAMES)

    def test_train_fjmp_parser_accepts_documented_command_shape(self):
        from FJMP import train_fjmp

        args = train_fjmp.build_arg_parser().parse_args(
            [
                "--baseline_ckpt",
                "runs/base/latest_model.pth",
                "--feature_input",
                "z_id",
                "--num_prototypes",
                "2",
                "--freeze_backbone",
                "true",
                "--strict_raw",
                "true",
                "--output_dir",
                "runs/proto",
            ]
        )

        self.assertEqual(args.feature_input, "z_id")
        self.assertEqual(args.num_prototypes, 2)
        self.assertTrue(args.freeze_backbone)
        self.assertTrue(args.strict_raw)

    def test_train_ssdg_parser_accepts_launcher_compatibility_args(self):
        from SSDG import train_ssdg

        args = train_ssdg.build_arg_parser().parse_args(
            [
                "--baseline_ckpt",
                "runs/base/latest_model.pth",
                "--split_mode",
                "tx_rx_day_1_7_2",
                "--labeled_ratio",
                "0.08",
                "--unlabeled_ratio",
                "0.72",
                "--source_val_ratio",
                "0.20",
                "--pseudo_threshold_mode",
                "global",
                "--tau_conf",
                "0.85",
                "--use_unlabeled",
                "false",
                "--use_sat_consistency",
                "--sat_train_scenario",
                "mixed_orbit",
                "--lambda_sat_cls",
                "0.08",
                "--lambda_sat_cons",
                "0.04",
                "--output_dir",
                "runs/ssdg",
            ]
        )

        self.assertFalse(args.use_unlabeled)
        self.assertTrue(args.use_sat_consistency)
        self.assertEqual(args.tau_conf, 0.85)
        self.assertEqual(args.metrics_csv, "")
        self.assertEqual(args.metrics_jsonl, "")

    def test_train_ssdg_epoch_telemetry_contains_complete_loss_and_config_fields(self):
        from SSDG import train_ssdg

        args = train_ssdg.build_arg_parser().parse_args(["--output_dir", "runs/ssdg"])
        row = train_ssdg._build_ssdg_epoch_telemetry_row(
            args=args,
            epoch=2,
            epochs=5,
            lr=2e-4,
            epoch_time_s=1.5,
            phase="label",
            train_logs={
                "train/loss": 2.0,
                "train/loss_tx_labeled": 1.2,
                "train/loss_sat_cons_labeled": 0.3,
                "train/w_loss_sat_cons_labeled": 0.012,
                "train/grad_total": 4.0,
                "train/skipped_nonfinite_loss": 0,
                "train/skipped_nonfinite_grad": 0,
            },
            val_stats={"tx_acc": 80.0},
            test_stats={"tx_acc": 70.0},
            named_test_stats={"target_rx": {"tx_acc": 60.0}},
            sat_test_stats={"mixed_orbit": {"aggregate": {"tx_acc": 40.0}}},
            stage_state={"phase": "S1_core", "dom_scale": 1.0},
            mixstyle_state={"enabled": True, "p": 0.18},
            aug_state={"scale": 0.2},
            loss_weights={"sat_cons": 0.04, "dom": 1.0},
            best_score=80.0,
            best_val=80.0,
            best_test=70.0,
            best_epoch=2,
            latest_path="latest.pth",
            best_path="best.pth",
            is_best=True,
        )

        self.assertEqual(row["schema"], "ssdg_epoch_telemetry_v1")
        self.assertEqual(row["optimizer"], "AdamW")
        self.assertIn("train_loss_tx_labeled", row)
        self.assertIn("train_w_loss_sat_cons_labeled", row)
        self.assertEqual(row["loss_weight_sat_cons"], 0.04)
        self.assertEqual(row["stage_phase"], "S1_core")
        self.assertEqual(row["mixstyle_enabled"], True)
        self.assertEqual(row["aug_scale"], 0.2)
        self.assertEqual(row["train_skipped_nonfinite_loss"], 0)

    def test_train_ssdg_epoch_telemetry_writer_outputs_csv_and_jsonl(self):
        from SSDG import train_ssdg

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = pathlib.Path(tmp) / "metrics_epoch.csv"
            jsonl_path = pathlib.Path(tmp) / "metrics_epoch.jsonl"
            rows = [
                {"schema": "ssdg_epoch_telemetry_v1", "epoch": 1, "train_loss": 2.0},
                {"schema": "ssdg_epoch_telemetry_v1", "epoch": 2, "train_loss": 1.0, "train_grad_total": 3.0},
            ]

            train_ssdg._write_ssdg_epoch_telemetry(csv_path, jsonl_path, rows)

            csv_text = csv_path.read_text(encoding="utf-8")
            jsonl_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]

        self.assertIn("train_grad_total", csv_text)
        self.assertEqual(jsonl_rows[-1]["epoch"], 2)
        self.assertEqual(jsonl_rows[-1]["train_loss"], 1.0)

    def test_train_ssdg_parser_accepts_train_py_sat_eval_args(self):
        from SSDG import train_ssdg

        args = train_ssdg.build_arg_parser().parse_args(
            [
                "--output_dir",
                "runs/ssdg",
                "--eval_sat_channel",
                "true",
                "--eval_sat_scenarios",
                "clear_leo,rain_leo",
                "--eval_sat_on",
                "main",
                "--sat_eval_max_batches",
                "2",
                "--sat_seed",
                "2028",
            ]
        )

        self.assertTrue(args.eval_sat_channel)
        self.assertEqual(args.eval_sat_scenarios, "clear_leo,rain_leo")
        self.assertEqual(args.eval_sat_on, "main")
        self.assertEqual(args.sat_eval_max_batches, 2)
        self.assertEqual(args.sat_seed, 2028)

    def test_train_ssdg_epochs_alias_means_total_epochs(self):
        from SSDG import train_ssdg

        args = train_ssdg.build_arg_parser().parse_args(
            [
                "--output_dir",
                "runs/ssdg",
                "--label_epochs",
                "150",
                "--pseudo_epochs",
                "50",
                "--epochs",
                "180",
            ]
        )

        total_epochs = train_ssdg._resolve_epoch_schedule(args)

        self.assertEqual(total_epochs, 180)
        self.assertEqual(args.label_epochs, 150)
        self.assertEqual(args.pseudo_epochs, 30)

    def test_train_ssdg_best_score_supports_cross_domain_and_satellite_selection(self):
        from SSDG import train_ssdg

        val = {"tx_acc": 91.0}
        test = {"tx_acc": 73.0}
        sat = {
            "clear_leo": {"aggregate": {"tx_acc": 64.0}},
            "rain_leo": {"aggregate": {"tx_acc": 58.0}},
        }

        self.assertEqual(train_ssdg._best_score(val, test, sat, "clean_val_tx"), 91.0)
        self.assertEqual(train_ssdg._best_score(val, test, sat, "test_overall_tx"), 73.0)
        self.assertEqual(train_ssdg._best_score(val, test, sat, "sat_mean_tx"), 61.0)
        self.assertEqual(train_ssdg._best_score(val, test, sat, "sat_worst_tx"), 58.0)
        expected_hmean = 2.0 * 91.0 * 58.0 / (91.0 + 58.0)
        self.assertAlmostEqual(train_ssdg._best_score(val, test, sat, "source_val_sat_hmean"), expected_hmean)

    def test_train_ssdg_dry_run_does_not_require_dataset_or_model_context(self):
        from SSDG import train_ssdg

        args = train_ssdg.build_arg_parser().parse_args(
            [
                "--output_dir",
                "runs/ssdg",
                "--dry_run",
            ]
        )

        with mock.patch.object(train_ssdg, "_build_ssdg_wisig_data", side_effect=AssertionError("data build called")):
            self.assertEqual(train_ssdg.train(args), 0)

    def test_train_ssdg_sat_consistency_weight_is_part_of_training_loss(self):
        text = (pathlib.Path(__file__).resolve().parents[1] / "SSDG" / "train_ssdg.py").read_text(encoding="utf-8")

        self.assertIn("loss_sat_cons_l", text)
        self.assertIn('cur_w["sat_cons"] * loss_sat_cons_l', text)
        self.assertIn('"train/loss_sat_cons_labeled": loss_sat_cons_l.detach()', text)

    def test_train_ssdg_defaults_to_two_stage_1_6_3_protocol(self):
        from SSDG import train_ssdg

        args = train_ssdg.build_arg_parser().parse_args(
            [
                "--output_dir",
                "runs/ssdg",
            ]
        )

        self.assertEqual(args.baseline_ckpt, "")
        self.assertTrue(args.from_scratch)
        self.assertEqual(args.split_mode, "tx_rx_day_1_6_3")
        self.assertEqual(args.labeled_ratio, 0.08)
        self.assertEqual(args.unlabeled_ratio, 0.72)
        self.assertEqual(args.source_val_ratio, 0.20)
        self.assertEqual(args.label_epochs, 170)
        self.assertEqual(args.pseudo_epochs, 100)
        self.assertAlmostEqual(args.lr, 2e-4)
        self.assertAlmostEqual(args.lambda_domain, 1.0)
        self.assertAlmostEqual(args.lambda_adv, 0.45)
        self.assertAlmostEqual(args.lambda_orth, 0.05)
        self.assertAlmostEqual(args.lambda_group_ce, 0.10)
        self.assertAlmostEqual(args.lambda_fishr, 0.02)
        self.assertTrue(args.use_sat_consistency)
        self.assertEqual(args.sat_train_scenario, "mixed_orbit")
        self.assertAlmostEqual(args.lambda_sat_cls, 0.10)
        self.assertAlmostEqual(args.lambda_sat_cons, 0.00)
        self.assertTrue(args.pseudo_domain_gate)
        self.assertTrue(args.pseudo_temporal_gate)
        self.assertTrue(args.pseudo_strong_agreement)
        self.assertEqual(args.pseudo_threshold_mode, "rx_day_quantile")

    def test_train_ssdg_epoch_block_uses_train_py_style_headers(self):
        from SSDG import train_ssdg

        block = train_ssdg.format_ssdg_epoch_block(
            epoch=171,
            epochs=270,
            lr=2e-4,
            epoch_time_s=12.5,
            phase="pseudo",
            train_logs={
                "train/loss": 3.0,
                "train/loss_tx_labeled": 2.0,
                "train/loss_domain_labeled": 0.5,
                "train/loss_adv_labeled": 0.4,
                "train/loss_orth_labeled": 0.1,
                "train/loss_group_ce_labeled": 0.7,
                "train/loss_sat_cls_labeled": 1.1,
                "train/loss_sat_cons_labeled": 0.2,
                "train/loss_fishr_labeled": 0.2,
                "train/loss_unlabeled": 0.9,
                "train/reliable_ratio": 0.3,
                "train/domain_pass": 0.8,
                "train/temporal_pass": 0.5,
                "train/strong_pass": 0.4,
                "train/pseudo_conf": 0.91,
            },
            val_stats={"tx_acc": 88.0, "dom_acc": 77.0},
            test_stats={"tx_acc": 66.0, "tx_correct": 66, "tx_total": 100},
            named_test_stats={
                "test_unseen_day_seen_rx": {"tx_acc": 70.0, "tx_correct": 7, "tx_total": 10},
                "test_seen_day_unseen_rx": {"tx_acc": 60.0, "tx_correct": 6, "tx_total": 10},
                "test_unseen_day_unseen_rx": {"tx_acc": 50.0, "tx_correct": 5, "tx_total": 10},
            },
            named_test_meta={
                "test_unseen_day_seen_rx": {"days_label": ["d3"], "rxs_idx": [0]},
                "test_seen_day_unseen_rx": {"days_label": ["d1"], "rxs_idx": [7]},
                "test_unseen_day_unseen_rx": {"days_label": ["d3"], "rxs_idx": [7]},
            },
            sat_test_stats={
                "clear_leo": {
                    "aggregate": {"tx_acc": 44.4, "tx_correct": 4, "tx_total": 9},
                    "strict_udu": 33.3,
                    "selected_names": ["test_unseen_day_unseen_rx"],
                }
            },
            best_val=89.0,
            best_test=67.0,
            best_epoch=150,
            latest_path="runs/latest_ssdg.pth",
            best_path="runs/best_val_ssdg.pth",
            is_best=False,
        )

        self.assertIn("[EPOCH-BEGIN] E171/270", block)
        self.assertIn("[STAGE] phase=pseudo", block)
        self.assertIn("[MIXSTYLE-EPOCH]", block)
        self.assertIn("[AUG] scale=", block)
        self.assertIn("[LOSS-CORE-RAW]", block)
        self.assertIn("[LOSS-CORE-W]", block)
        self.assertIn("[LOSS-SAT-RAW]", block)
        self.assertIn("[LOSS-WEIGHT]", block)
        self.assertIn("[LOSS-TOP]", block)
        self.assertIn("[LOSS-PSEUDO]", block)
        self.assertIn("selected=0/0 correct=0", block)
        self.assertIn("[VAL]   tx=88.00%", block)
        self.assertIn("[TEST]  overall_tx=66.00% (66/100)", block)
        self.assertIn("[TEST-SPLIT]", block)
        self.assertIn("[SAT-TEST] scenario=clear_leo", block)
        self.assertIn("[BEST-SOURCE-VAL] val_tx=89.00% @ E150 heldout_test=67.00%", block)
        self.assertIn("[EPOCH-END] E171/270", block)

    def test_train_ssdg_pseudo_stats_count_selected_and_correct_labels(self):
        from SSDG import train_ssdg

        block = train_ssdg.format_ssdg_epoch_block(
            epoch=152,
            epochs=200,
            lr=5e-5,
            epoch_time_s=10.0,
            phase="pseudo",
            train_logs={
                "train/pseudo_total": 12,
                "train/pseudo_selected": 5,
                "train/pseudo_correct": 4,
                "train/pseudo_conf": 0.92,
            },
            val_stats={"tx_acc": 80.0, "dom_acc": 70.0},
            test_stats={"tx_acc": 60.0, "tx_correct": 6, "tx_total": 10},
            named_test_stats={},
            named_test_meta={},
            sat_test_stats={},
            best_val=80.0,
            best_test=60.0,
            best_epoch=152,
            latest_path="runs/latest_ssdg.pth",
            best_path="runs/best_val_ssdg.pth",
            is_best=True,
        )

        self.assertIn("total=12 selected=5/12 correct=4", block)
        self.assertIn("precision=80.000%", block)

    def test_ssdg_split_is_stratified_by_tx_rx_day(self):
        from SSDG import train_ssdg

        index = []
        for tx in range(2):
            for rx in range(2):
                for day in range(2):
                    for sig in range(10):
                        index.append(SimpleNamespace(tx_i=tx, rx_i=rx, day_i=day, eq_i=0, sig_i=sig))
        dataset = SimpleNamespace(index=index)

        labeled, unlabeled, val = train_ssdg.split_tx_rx_day_1_7_2(
            dataset,
            labeled_ratio=0.08,
            unlabeled_ratio=0.72,
            source_val_ratio=0.20,
        )

        self.assertEqual(len(labeled), 6)
        self.assertEqual(len(unlabeled), 58)
        self.assertEqual(len(val), 16)
        self.assertEqual(set(labeled).intersection(unlabeled), set())
        self.assertEqual(set(labeled).intersection(val), set())
        self.assertEqual(set(unlabeled).intersection(val), set())
        self.assertEqual({dataset.index[idx].tx_i for idx in labeled}, {0, 1})

    def test_ssdg_split_rejects_rho_label_above_protocol_limit(self):
        from SSDG import train_ssdg

        dataset = SimpleNamespace(
            index=[SimpleNamespace(tx_i=0, rx_i=0, day_i=0, eq_i=0, sig_i=i) for i in range(20)]
        )
        with self.assertRaisesRegex(ValueError, "rho_label"):
            train_ssdg.split_tx_rx_day_1_7_2(
                dataset,
                labeled_ratio=0.10,
                unlabeled_ratio=0.70,
                source_val_ratio=0.20,
            )

    def test_ssdg_temporal_gate_requires_neighbor_agreement_in_same_stream(self):
        from SSDG import train_ssdg

        meta = {
            "rx_i": [0, 0, 0, 0, 0, 1],
            "day_i": [1, 1, 1, 1, 1, 1],
            "eq_i": [0, 0, 0, 0, 0, 0],
            "sig_i": [10, 11, 12, 13, 14, 10],
            "base_index": [10, 11, 12, 13, 14, 200],
        }
        pseudo = [2, 2, 3, 3, 4, 2]
        conf = [0.95, 0.94, 0.96, 0.93, 0.99, 0.99]

        mask = train_ssdg.temporal_neighbor_agreement_mask(
            pseudo,
            conf,
            meta,
            window=1,
            min_conf=0.90,
        )

        self.assertEqual(mask, [True, True, True, True, False, False])

        meta_far = dict(meta)
        meta_far["rx_i"] = [0, 0]
        meta_far["day_i"] = [1, 1]
        meta_far["eq_i"] = [0, 0]
        meta_far["sig_i"] = [10, 11]
        meta_far["base_index"] = [10, 200]
        far_mask = train_ssdg.temporal_neighbor_agreement_mask([2, 2], [0.95, 0.94], meta_far, window=1, min_conf=0.90)
        self.assertEqual(far_mask, [False, False])

    def test_post_stage_epoch_eval_aggregates_wisig_named_tests(self):
        from post_stage_eval import resolve_sat_eval_max_batches, summarize_post_stage_tests

        named = {
            "test_unseen_day_seen_rx": {"tx_correct": 8, "tx_total": 10, "tx_acc": 80.0, "dom_acc": 0.0},
            "test_seen_day_unseen_rx": {"tx_correct": 7, "tx_total": 10, "tx_acc": 70.0, "dom_acc": 0.0},
            "test_unseen_day_unseen_rx": {"tx_correct": 5, "tx_total": 10, "tx_acc": 50.0, "dom_acc": 0.0},
            "test_rx_7": {"tx_correct": 10, "tx_total": 10, "tx_acc": 100.0, "dom_acc": 0.0},
        }
        data_ctx = {
            "named_test_meta": {
                "test_unseen_day_seen_rx": {"size": 10},
                "test_seen_day_unseen_rx": {"size": 10},
                "test_unseen_day_unseen_rx": {"size": 10},
            }
        }

        test_stats, lines = summarize_post_stage_tests(
            named,
            data_ctx,
            dataset="wisig",
        )

        self.assertEqual(test_stats["tx_correct"], 20)
        self.assertEqual(test_stats["tx_total"], 30)
        self.assertAlmostEqual(test_stats["tx_acc"], 100.0 * 20 / 30)
        self.assertTrue(any(line.startswith("[TEST]") for line in lines))
        self.assertTrue(any(line.startswith("[TEST-SPLIT]") for line in lines))
        self.assertTrue(any("unseen_day_unseen_rx" in line for line in lines))
        self.assertEqual(resolve_sat_eval_max_batches(-1, 3), 3)
        self.assertEqual(resolve_sat_eval_max_batches(2, 3), 2)

    def test_post_stage_trainers_accept_train_py_sat_eval_args(self):
        from FJMP import train_fjmp

        fjmp_args = train_fjmp.build_arg_parser().parse_args(
            [
                "--baseline_ckpt",
                "runs/base/latest_model.pth",
                "--output_dir",
                "runs/fjmp",
                "--eval_sat_channel",
                "true",
                "--eval_sat_scenarios",
                "storm_mp",
                "--eval_sat_on",
                "test_unseen_day_unseen_rx",
                "--sat_eval_max_batches",
                "-1",
                "--sat_seed",
                "2029",
                "--sat_fs_hz",
                "25000000",
                "--sat_fc_hz",
                "2462000000",
            ]
        )

        self.assertTrue(fjmp_args.eval_sat_channel)
        self.assertEqual(fjmp_args.eval_sat_scenarios, "storm_mp")
        self.assertEqual(fjmp_args.sat_eval_max_batches, -1)


if __name__ == "__main__":
    unittest.main()
