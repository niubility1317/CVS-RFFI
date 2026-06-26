import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SGVBPFJMPDesignTest(unittest.TestCase):
    def test_documented_sgv_bp_modules_exist_with_expected_public_symbols(self):
        expected = {
            "FJMP/star_ground_view.py": [
                "class StarGroundViewGenerator",
                "def sgv_strength_config",
                "def estimate_sat_reliability",
            ],
            "FJMP/logit_calibration.py": [
                "class CenteredTemperatureCalibrator",
                "def center_logits",
                "def clip_delta_norm",
            ],
            "FJMP/base_protected_fusion.py": [
                "class BaseProtectedFusion",
                "def stage_rho_max",
                "safe_logits",
            ],
            "FJMP/sgv_bp_losses.py": [
                "def sgv_bp_stage_config",
                "def compute_sample_strata",
                "def compute_sgv_bp_losses",
                "def gate_view_gap_loss",
                "def worst_domain_view_loss",
            ],
            "FJMP/sgv_bp_metrics.py": [
                "def compute_sgv_bp_metrics",
                "def compute_proxy_safe_score",
                "def warning_flags",
            ],
            "FJMP/sgv_sampler.py": [
                "class PairedSGVBatchSampler",
                "def validate_paired_batch",
            ],
        }

        for relative_path, symbols in expected.items():
            path = ROOT / relative_path
            self.assertTrue(path.is_file(), f"Missing documented module {relative_path}")
            text = path.read_text(encoding="utf-8")
            for symbol in symbols:
                self.assertIn(symbol, text, f"{relative_path} does not expose {symbol}")

    def test_train_fjmp_parser_accepts_sgv_bp_ultimate_command_arguments(self):
        import train_fjmp

        args = train_fjmp.build_arg_parser().parse_args(
            [
                "--baseline_ckpt",
                "runs/cvs_rffi_staged/B3b_stable_sat07_cls_only/latest_model.pth",
                "--output_dir",
                "runs/fjmp_sgv_bp/EXP-04",
                "--model_name",
                "SGV-BP-FJMP",
                "--num_prototypes",
                "3",
                "--prototype_init",
                "class_mean",
                "--aggregation",
                "top2_mean",
                "--zdom_usage",
                "detached_gate_only",
                "--fusion_mode",
                "base_protected_residual",
                "--logit_calibration",
                "centered_temperature",
                "--rho_init",
                "0.03",
                "--rho_max_stage1",
                "0.10",
                "--rho_max_stage2",
                "0.25",
                "--rho_max_stage3",
                "0.30",
                "--max_delta_norm",
                "3.0",
                "--use_sgv",
                "--sgv_train_strength",
                "low,mid",
                "--sgv_eval_strength",
                "low,mid,high",
                "--use_sat_reliability",
                "--lambda_ce_head_clean",
                "0.30",
                "--lambda_ce_head_sat",
                "0.15",
                "--lambda_pres_clean",
                "3.0",
                "--lambda_pres_sat",
                "1.5",
                "--lambda_harm",
                "2.0",
                "--lambda_kd_easy",
                "1.5",
                "--lambda_kd_mid",
                "0.5",
                "--lambda_sgv_head",
                "0.5",
                "--lambda_sgv_safe",
                "1.0",
                "--lambda_proto_sgv",
                "0.2",
                "--lambda_worst_domain_view",
                "0.3",
                "--lambda_gate_easy",
                "0.08",
                "--lambda_gate_view_gap",
                "0.03",
                "--lambda_delta",
                "0.04",
                "--selection_metric",
                "best_proxy_safe_score",
                "--epochs",
                "30",
            ]
        )

        self.assertEqual(args.model_name, "SGV-BP-FJMP")
        self.assertEqual(args.aggregation, "top2_mean")
        self.assertEqual(args.prototype_aggregation, "top2_mean")
        self.assertEqual(args.zdom_usage, "detached_gate_only")
        self.assertEqual(args.fusion_mode, "base_protected_residual")
        self.assertTrue(args.use_sgv)
        self.assertTrue(args.use_sat_reliability)
        self.assertAlmostEqual(args.rho_max_stage3, 0.30)
        self.assertEqual(args.selection_metric, "best_proxy_safe_score")

    def test_manifest_contains_documented_sgv_bp_experiment_matrix(self):
        from FJMP.experiment_manifest import build_experiment_manifest

        manifest = build_experiment_manifest(["SGV-BP"])
        ids = {row["id"] for row in manifest}

        for required_id in [f"EXP-{i:02d}" for i in range(17)]:
            self.assertIn(required_id, ids)

        main = next(row for row in manifest if row["id"] == "EXP-04")
        self.assertEqual(main["args"]["model_name"], "SGV-BP-FJMP")
        self.assertEqual(main["args"]["fusion_mode"], "base_protected_residual")
        self.assertEqual(main["args"]["num_prototypes"], 3)
        self.assertEqual(main["args"]["aggregation"], "top2_mean")
        self.assertTrue(main["args"]["use_sgv"])

    def test_sgv_bp_launcher_exists_and_targets_exp04_plan(self):
        path = ROOT / "scripts/run_fjmp_sgv_bp_8gpu.sh"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        self.assertIn("--plan FULL", text)
        self.assertIn("--plan CORE", text)
        self.assertIn("--model_name SGV-BP-FJMP", text)
        self.assertIn("--selection_metric best_proxy_safe_score", text)

    def test_sgv_bp_launcher_defaults_to_full_core_then_rest_queue(self):
        path = ROOT / "scripts/run_fjmp_sgv_bp_8gpu.sh"
        text = path.read_text(encoding="utf-8")

        self.assertIn('PLAN="${PLAN:-FULL}"', text)
        self.assertIn('plan.upper() in {"FULL", "ALL"}', text)
        self.assertIn('core = [row for row in selected_rows if str(row.get("batch", "")).upper() == "CORE"]', text)
        self.assertIn('selected_rows = core + rest', text)

    def test_train_fjmp_loss_breakdown_formatters_expose_raw_weighted_and_top_terms(self):
        import train_fjmp

        logs = {
            "train/loss": 17.0,
            "train/fjmp_loss": 0.1,
            "train/sgv_loss": 16.9,
            "train/sgv_ce_head_clean": 1.2,
            "train/sgv_sgv_safe": 11.0,
            "train_weighted/loss_sep": 0.01,
            "train_weighted/sgv_ce_head_clean": 0.36,
            "train_weighted/sgv_sgv_safe": 11.0,
        }

        raw = train_fjmp._format_loss_line("[LOSS-SGV-RAW]", logs, train_fjmp.SGV_LOSS_KEYS, prefix="train/")
        weighted = train_fjmp._format_loss_line("[LOSS-SGV-W]", logs, train_fjmp.SGV_LOSS_KEYS, prefix="train_weighted/")
        top = train_fjmp._format_top_loss_contributors(logs, top_k=2)

        self.assertIn("sgv_loss=16.9000", raw)
        self.assertIn("sgv_ce_head_clean=1.2000", raw)
        self.assertIn("sgv_sgv_safe=11.0000", weighted)
        self.assertIn("[LOSS-TOP]", top)
        self.assertLess(top.index("sgv_sgv_safe=11.0000"), top.index("sgv_ce_head_clean=0.3600"))

    def test_train_fjmp_passes_domain_groups_to_worst_domain_view_loss(self):
        path = ROOT / "train_fjmp.py"
        text = path.read_text(encoding="utf-8")

        self.assertIn("x, y, extra = move_batch(batch, device)", text)
        self.assertIn("sgv_group = domain_from_extra(extra, data_ctx[\"domain_label_map\"], device)", text)
        self.assertIn("group=sgv_group", text)

    def test_manifest_contains_loss_design_batch_for_next_cross_domain_run(self):
        from FJMP.experiment_manifest import build_experiment_manifest

        manifest = build_experiment_manifest(["LOSS-DESIGN"])
        ids = {row["id"] for row in manifest}

        self.assertEqual(len(manifest), 16)
        for required_id in [f"LD-{i:02d}" for i in range(16)]:
            self.assertIn(required_id, ids)

        recommended = next(row for row in manifest if row["id"] == "LD-01")
        self.assertEqual(recommended["batch"], "LOSS-DESIGN")
        self.assertEqual(recommended["args"]["fusion_mode"], "base_protected_residual")
        self.assertTrue(recommended["args"]["use_sat_reliability"])
        self.assertLess(recommended["args"]["lambda_sgv_safe"], 1.0)
        self.assertGreater(recommended["args"]["lambda_worst_domain_view"], 0.0)

    def test_loss_design_launcher_uses_dynamic_8gpu_queue(self):
        path = ROOT / "scripts/run_fjmp_loss_design_8gpu.sh"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")

        self.assertIn("run_fjmp_sgv_bp_8gpu.sh", text)
        self.assertIn("--plan LOSS-DESIGN", text)
        self.assertIn("--gpu-ids", text)
        self.assertIn("logs/fjmp_loss_design", text)
        self.assertIn("runs/fjmp_loss_design", text)

    def test_train_fjmp_prints_audit_friendly_experiment_config(self):
        path = ROOT / "train_fjmp.py"
        text = path.read_text(encoding="utf-8")

        self.assertIn("def _print_experiment_config", text)
        self.assertIn("[CONFIG-BEGIN]", text)
        self.assertIn("[CONFIG-LOSS]", text)
        self.assertIn("[CONFIG-END]", text)
        self.assertIn("_print_experiment_config(args, legacy_ce_on, legacy_kd_on, out_dir)", text)
        self.assertIn("[LOSS-FJMP-RAW]", text)
        self.assertIn("[LOSS-FJMP-W]", text)
        self.assertIn("[LOSS-FJMP-WEIGHT]", text)
        self.assertIn("[LOSS-SGV-RAW]", text)
        self.assertIn("[LOSS-SGV-W]", text)
        self.assertIn("[LOSS-SGV-WEIGHT]", text)
        self.assertIn("[LOSS-TOP]", text)
        self.assertIn("--metrics_csv", text)
        self.assertIn("[METRICS]", text)

    def test_manifest_contains_a03_a06_reproduction_batch(self):
        from FJMP.experiment_manifest import build_experiment_manifest

        manifest = build_experiment_manifest(["A03-A06-REPRO"])
        ids = {row["id"] for row in manifest}

        self.assertEqual(len(manifest), 8)
        for required_id in [f"R83-{i:02d}" for i in range(8)]:
            self.assertIn(required_id, ids)

        a03_e7 = next(row for row in manifest if row["id"] == "R83-02")
        a06_e7 = next(row for row in manifest if row["id"] == "R83-03")
        self.assertEqual(a03_e7["args"]["num_prototypes"], 3)
        self.assertEqual(a03_e7["args"]["zdom_mode"], "zero")
        self.assertEqual(a03_e7["args"]["epochs"], 7)
        self.assertFalse(a03_e7["args"]["save_checkpoints"])
        self.assertEqual(a06_e7["args"]["init_zdom_mode"], "zero")
        self.assertEqual(a06_e7["args"]["epochs"], 7)
        self.assertFalse(a06_e7["args"]["save_checkpoints"])

    def test_a03_a06_reproduction_launcher_exists(self):
        path = ROOT / "scripts/run_fjmp_a03_a06_repro_8gpu.sh"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")

        self.assertIn("run_fjmp_sgv_bp_8gpu.sh", text)
        self.assertIn("--plan A03-A06-REPRO", text)
        self.assertIn("logs/fjmp_a03_a06_repro", text)
        self.assertIn("runs/fjmp_a03_a06_repro", text)

    def test_dynamic_launcher_places_metrics_next_to_train_log(self):
        path = ROOT / "scripts/run_fjmp_sgv_bp_8gpu.sh"
        text = path.read_text(encoding="utf-8")

        self.assertIn("metrics_csv=\"${log%.log}_metrics_epoch.csv\"", text)
        self.assertIn("--metrics_csv", text)
        self.assertIn("METRICS_CSV=${metrics_csv}", text)
        self.assertIn("${LOG_ROOT}/${exp_id}_*_metrics_epoch.csv", text)

    def test_train_fjmp_saves_and_prints_diagnostic_best_udu_checkpoint(self):
        path = ROOT / "train_fjmp.py"
        text = path.read_text(encoding="utf-8")

        self.assertIn("best_udu = float(\"-inf\")", text)
        self.assertIn("best_udu_fjmp.pth", text)
        self.assertIn("[BEST-UDU]", text)
        self.assertIn("diagnostic_test_selection", text)
        self.assertIn("save_checkpoints", text)
        self.assertIn("path={best_udu_path}", text)


if __name__ == "__main__":
    unittest.main()
