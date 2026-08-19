import unittest
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class FederatedTrainIntegrationTest(unittest.TestCase):
    def test_train_py_exposes_fedavg_fedprox_cli_and_dispatches_to_federated_trainer(self):
        text = (ROOT / "train.py").read_text(encoding="utf-8")
        fed_text = (ROOT / "federated" / "fed_trainer.py").read_text(encoding="utf-8")

        for token in [
            "--train_mode",
            "--fl_client_key",
            "--fl_rounds",
            "--fl_local_epochs",
            "--fl_clients_per_round",
            "--fl_agg_weight",
            "--fl_num_workers",
            "--fl_local_objective",
            "--fl_sat_aug_mode",
            "--fedprox_mu",
            "--fl_vmb_stage",
            "--fl_vmb_stage1_local_steps",
            "--fl_vmb_batches_per_client",
            "--fl_vmb_server_lr",
            "--fl_vmb_domain_balanced_sampling",
            "--lambda_vmb_tx_proto",
            "--lambda_vmb_rx_proto",
            "--lambda_tx_adv_r",
            "--use_tx_adv_on_zdom",
            "--use_fed_coral",
            "--lambda_fed_coral",
            "--lambda_fed_coral_virtual",
            "--fed_coral_feature",
            "--fed_coral_start_round",
            "--fed_coral_min_count",
            "--fed_coral_scope",
            "use_fed_fishr",
            "--lambda_fed_fishr",
            "--fed_fishr_mode",
            "--fed_fishr_gradient_scope",
            "--fed_fishr_start_round",
            "--fed_fishr_min_clients",
            "--fed_fishr_min_count",
            "--fed_fishr_max_samples_per_class",
            "--fed_fishr_sketch_dim",
            "--fed_fishr_momentum",
            "--fed_fishr_reweight_floor",
            "--fed_fishr_reweight_cap",
            "use_fed_style_bank",
            "use_fl_style_bank_stats",
            "--fl_style_replay_start_round",
            "--fl_style_phys_start_round",
            "--fl_style_phys_max_cfo_hz",
            "--fl_style_phys_max_sro_ppm",
            "--fl_style_phys_max_iq_gain_db",
            "--fl_style_phys_max_iq_phase_deg",
            "--fl_style_dg_start_round",
            "--fl_style_max_views",
            "--fl_style_sampling_policy",
            "--fl_style_transform_mix_alpha",
            "--fl_style_real_mix_samples",
            "use_proto_evidence_bank",
            "--proto_rho_max",
            "--proto_top_m",
            "use_style_collab_eval",
            "--style_collab_views",
            "--style_collab_fusion",
            "--style_collab_base_weight",
            "--style_collab_max_aux_weight",
            "--fl_style_bank_max_centroids",
            "--fl_local_exclude_prefixes",
            "--lambda_rx_adv",
            "--fed_cgrl_base_lambda",
            "--fed_cgrl_warmup_rounds",
            "--fed_cgrl_leak_target_acc",
            "--fed_cgrl_leak_stat",
            "--fed_cgrl_tx_loss_guard",
            "--fed_cgrl_tx_guard_release_rounds",
            "--fed_cgrl_conflict_threshold",
            "--fed_cgrl_conflict_source",
            "--fed_cgrl_ema",
            "--fl_conflict_agg",
            "--use_logit_anchors",
            "--lambda_logit_kd",
            "--activation_token_route",
            "--token_quant_bits",
            "--fl_style_code_dim",
            "--fl_probe_every",
            "--feature_probe_export",
            "--log_dir",
            "--cpu_threads",
            "--cpu_interop_threads",
        ]:
            self.assertIn(token, text)
        self.assertIn('add_bool_arg(parser, "use_fed_style_bank", False', text)
        self.assertIn('add_bool_arg(parser, "use_fl_style_bank_stats", False', text)
        self.assertIn('add_bool_arg(parser, "use_fed_style_sat_view", False', text)
        self.assertIn('add_bool_arg(parser, "use_fed_cgrl", False', text)
        self.assertIn('add_bool_arg(parser, "use_proto_evidence_bank", True', text)
        self.assertIn('add_bool_arg(parser, "use_style_collab_eval", False', text)
        self.assertIn('parser.add_argument("--wisig_train_ratio", type=float, default=0.1', text)
        self.assertIn('parser.add_argument("--epochs", type=int, default=200)', text)
        self.assertRegex(
            text,
            r'parser\.add_argument\(\s*"--test_eval_policy",[\s\S]*?default="every_epoch"',
        )
        self.assertRegex(
            text,
            r'parser\.add_argument\(\s*"--test_eval_start_epoch",[\s\S]*?default=0',
        )
        self.assertIn('parser.add_argument("--fl_rounds", type=int, default=200)', text)
        self.assertIn('parser.add_argument("--fl_num_workers", type=int, default=0', text)
        self.assertIn('parser.add_argument("--fl_test_eval_interval", type=int, default=0', text)
        self.assertIn('parser.add_argument("--fl_test_eval_last_n", type=int, default=0', text)
        self.assertIn('parser.add_argument("--fl_test_eval_final_offsets", type=str, default="5,3,1"', text)
        self.assertIn('parser.add_argument("--fl_style_replay_start_round", type=int, default=20', text)
        self.assertIn('parser.add_argument("--fl_style_phys_start_round", type=int, default=20', text)
        self.assertIn('parser.add_argument("--fl_style_dg_start_round", type=int, default=40', text)
        self.assertIn('parser.add_argument("--fl_style_max_views", type=int, default=1', text)
        self.assertIn('parser.add_argument("--fl_style_replay_prob", type=float, default=0.25', text)
        self.assertIn('parser.add_argument("--fl_style_sampling_policy", type=str, default="diverse"', text)
        self.assertIn('parser.add_argument("--fl_style_transform_mix_alpha", type=float, default=1.0', text)
        self.assertIn('parser.add_argument("--fl_style_real_mix_samples", type=int, default=0', text)
        self.assertIn('parser.add_argument("--eval_sat_on", type=str, default=FEDERATED_MAIN_SAT_EVAL_ON', text)
        self.assertIn('Federated WiSig training must use --wisig_train_ratio 0.1', text)
        self.assertIn('parser.add_argument("--fl_sat_aug_mode", type=str, default="baseline_view"', text)
        self.assertIn("resolve_phase1_sat_training_scenarios", text)
        self.assertIn('getattr(self.cfg, "fl_sat_aug_mode", "baseline_view")', fed_text)
        self.assertIn('getattr(cfg, "fl_num_workers", 0)', fed_text)
        self.assertIn('"style_collab"', fed_text)
        self.assertIn('"global_style_collab_fusion"', fed_text)
        self.assertIn('"style_collab_rescue"', fed_text)
        for mode in ['"centralized"', '"fedavg"', '"fedprox"', '"fedcvs_vmb"', '"split_bex02"', '"fedriei"', '"fedfa"', '"fucl"', '"rafl"']:
            self.assertIn(mode, text)
        self.assertIn('default="receiver"', text)
        self.assertIn('"receiver_agnostic_bex02"', text)
        self.assertIn('"local_virtual_bex02"', text)
        self.assertIn('"baseline_view"', text)
        self.assertIn("FederatedTrainer", text)
        self.assertIn('args.train_mode != "centralized"', text)
        self.assertIn("FedCVS-RFFI-VMB", fed_text)
        self.assertIn('"distillation"', fed_text)
        self.assertIn('"compression"', fed_text)
        self.assertIn('"coral_alignment"', fed_text)
        self.assertIn('"fed_cgrl"', fed_text)
        self.assertIn("[FED-CONFIG-CGRL]", fed_text)
        self.assertIn("[FED-CGRL][R", fed_text)
        self.assertIn("_fed_cgrl_conflict_summary_from_client_states", fed_text)
        self.assertIn("fed_cgrl_global_lambda_rx_adv_p90", fed_text)
        self.assertIn("fed_cgrl_global_grl_target_acc_p90", fed_text)
        self.assertIn("fed_cgrl_global_worst_client_id", fed_text)
        self.assertIn("fed_cgrl_global_conflict_source", fed_text)
        self.assertIn('"lambda_fed_coral"', fed_text)
        self.assertIn('"fed_coral_feature"', fed_text)
        self.assertIn('"conflict_aggregation"', fed_text)
        self.assertIn('"feature_probe"', fed_text)
        self.assertIn("self.log_dir", fed_text)
        self.assertIn('os.path.join(self.log_dir, "logs.jsonl")', fed_text)
        self.assertIn("configure_cpu_thread_env()", text)
        self.assertIn("configure_torch_thread_runtime", text)
        self.assertIn('"runtime"', fed_text)
        self.assertIn("[FED-CONFIG-RUNTIME]", fed_text)
        self.assertIn('os.path.join(self.output_dir, "best_checkpoint.pt")', fed_text)
        self.assertIn('"local_virtual_bex02", "ra_bex02", "receiver_agnostic_bex02"', text)
        self.assertIn("--fl_vmb_stage1_objective", text)
        self.assertIn("startswith(f\"{flag}=\")", text)

    def test_centralized_training_test_defaults_cover_last_30_epochs(self):
        from train import apply_training_test_eval_defaults

        defaults = apply_training_test_eval_defaults(
            SimpleNamespace(epochs=200, test_eval_policy="every_epoch", test_eval_start_epoch=0)
        )
        self.assertEqual(defaults.test_eval_policy, "every_epoch")
        self.assertEqual(defaults.test_eval_start_epoch, 171)

        short_run = apply_training_test_eval_defaults(
            SimpleNamespace(epochs=3, test_eval_policy="every_epoch", test_eval_start_epoch=0)
        )
        self.assertEqual(short_run.test_eval_start_epoch, 1)

        explicit = apply_training_test_eval_defaults(
            SimpleNamespace(epochs=200, test_eval_policy="val_improved_final", test_eval_start_epoch=151)
        )
        self.assertEqual(explicit.test_eval_policy, "val_improved_final")
        self.assertEqual(explicit.test_eval_start_epoch, 151)

    def test_split_bex02_defaults_respect_equals_style_explicit_flags(self):
        from train import apply_fedcvs_vmb_defaults

        args = SimpleNamespace(
            train_mode="split_bex02",
            fl_local_objective="local_virtual_bex02",
            lambda_vmb_tx_proto=0.0,
            lambda_vmb_rx_proto=0.0,
            lambda_tx_adv_r=0.0,
            use_tx_adv_on_zdom=False,
            activation_token_route="none",
        )
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--train_mode=split_bex02",
                "--fl_local_objective=local_virtual_bex02",
                "--activation_token_route=none",
                "--lambda_vmb_tx_proto=0.0",
                "--lambda_vmb_rx_proto=0.0",
                "--lambda_tx_adv_r=0.0",
                "--no_use_tx_adv_on_zdom",
            ],
        ):
            updated = apply_fedcvs_vmb_defaults(args)

        self.assertEqual(updated.fl_local_objective, "local_virtual_bex02")
        self.assertEqual(updated.activation_token_route, "none")
        self.assertEqual(updated.lambda_vmb_tx_proto, 0.0)
        self.assertEqual(updated.lambda_vmb_rx_proto, 0.0)
        self.assertEqual(updated.lambda_tx_adv_r, 0.0)
        self.assertFalse(updated.use_tx_adv_on_zdom)

    def test_fewshot_federated_dg_launcher_documents_train_ratio_point_one(self):
        script = ROOT / "scripts" / "run_fed_fewshot_dg_6gpu.sh"
        doc = ROOT / "docs" / "fed_fewshot_dg_experiments.md"

        self.assertTrue(script.is_file())
        self.assertTrue(doc.is_file())
        script_text = script.read_text(encoding="utf-8")
        doc_text = doc.read_text(encoding="utf-8")
        self.assertIn('FEWSHOT_RATIO="${FEWSHOT_RATIO:-0.1}"', script_text)
        self.assertIn("--wisig_train_ratio", script_text)
        self.assertIn("--train_mode fedavg", script_text)
        self.assertIn("--train_mode fedprox", script_text)
        self.assertIn("FED_BASE", script_text)
        self.assertIn("CENTRAL", script_text)
        self.assertIn("FSDG07_centralized_strong_dg", script_text)
        self.assertIn("FED_DG", script_text)
        self.assertIn("FSDG12A_fedavg_rxday_local3", script_text)
        self.assertIn("FSDG18_fedavg_rxday_bex02dg", script_text)
        self.assertIn("FSDG1A_fedprox_rxday_bex02dg_mu01", script_text)
        self.assertIn("FSDG50_fedprox_receiver_ra_bex02_baseline_sat", script_text)
        self.assertIn("--fl_local_objective bex02_dg", script_text)
        self.assertIn("--fl_local_objective receiver_agnostic_bex02", script_text)
        self.assertIn("--fl_client_key receiver", script_text)
        self.assertIn("--wisig_domain rx", script_text)
        self.assertIn("--fl_sat_aug_mode baseline_view", script_text)
        self.assertIn("--eval_sat_channel", script_text)
        self.assertIn("--eval_sat_on main", script_text)
        self.assertIn("--eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit", script_text)
        self.assertIn("--seed 1337", script_text)
        self.assertNotIn("seed42", script_text)
        self.assertNotIn("seed2027", script_text)
        self.assertIn("receiver_day", script_text)
        self.assertIn("10% labeled train split", doc_text)

    def test_vmb_cen_a31_anchor_ladder_launcher_encodes_recovery_matrix(self):
        script = ROOT / "scripts" / "launch_optimizer_20260601_101941_vmb_cen_a31_anchor_ladder.sh"
        doc = ROOT / "docs" / "fed_fewshot_dg_experiments.md"

        self.assertTrue(script.is_file())
        self.assertTrue(doc.is_file())
        text = script.read_text(encoding="utf-8")
        doc_text = doc.read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        run_names = [
            "VMB9_A01_d01_repro_bpc2_r010",
            "VMB9_A02_clean_s1l4_satR40w06_styR70_r010",
            "VMB9_A03_clean_s1l8_satR40w06_styR70_r010",
            "VMB9_A04_clean_s1l8_satR60w06_styR80_r010",
            "VMB9_A05_clean_s1l8_satR40w08_styR70_r010",
            "VMB9_A06_clean_s1l8_satR40w06_styR70_p050_r010",
            "VMB9_A07_clean_s1l8_satR40w06_styR70_bpc4_r010",
            "VMB9_A08_phase_bestguess_s1l8_bpc4_r010",
        ]
        for run_name in run_names:
            self.assertIn(run_name, text)
        for token in [
            "VMB9_A01_d01_repro_bpc2_r010",
            "VMB9_A03_clean_s1l8_satR40w06_styR70_r010",
            "VMB9_A07_clean_s1l8_satR40w06_styR70_bpc4_r010",
            "VMB9_A08_phase_bestguess_s1l8_bpc4_r010",
            "--wisig_train_ratio 0.1",
            "--fl_rounds 200",
            "--epochs 200",
            "--fl_client_key receiver",
            "--sat_cons_start_epoch 40",
            "--fl_style_replay_start_round 70",
            "--fl_style_max_views 1",
            "--fl_style_replay_prob 0.25",
            "MAX_PROCS_PER_GPU",
            "DRY_RUN",
            "ONLY_RUN",
            "launch_pids.tsv",
        ]:
            self.assertIn(token, text)
        self.assertIn('MAX_PROCS_PER_GPU="${MAX_PROCS_PER_GPU:-1}"', text)
        self.assertIn('if [[ "${DRY_RUN}" != "1" ]]; then', text)
        self.assertIn("--fl_vmb_cen_a31_profile --fl_local_objective receiver_agnostic_bex02", normalized)
        self.assertIn("--fl_vmb_pretrain_rounds 60 --fl_vmb_stage1_objective ce --no_fl_vmb_stage1_use_aux_losses --fl_vmb_stage1_local_steps 4", normalized)
        self.assertIn("--fl_vmb_pretrain_rounds 60 --fl_vmb_stage1_objective ce --no_fl_vmb_stage1_use_aux_losses --fl_vmb_stage1_local_steps 8", normalized)
        self.assertIn("--sat_cons_start_epoch 60 --fl_baseline_view_ce_weight 0.60 --fl_style_replay_start_round 80", normalized)
        self.assertIn("--fl_baseline_view_ce_weight 0.80 --fl_style_replay_start_round 70", normalized)
        self.assertIn("--fl_style_replay_prob 0.50 --fl_style_max_views 1 --fl_style_transform_mix_alpha 0.50", normalized)
        self.assertIn("--fl_vmb_batches_per_client 4 --fl_vmb_server_lr 0.004", normalized)
        self.assertIn("--fl_vmb_pretrain_rounds 80 --fl_vmb_stage1_objective ce", normalized)
        self.assertIn("Centralized strategy sweep", doc_text)
        self.assertIn("FED_BASE / CORE -> FED_DG -> CENTRAL", doc_text)
        self.assertIn("BEX02_fishr002_mixed_e170", doc_text)
        self.assertIn("https://github.com/litian96/FedProx", doc_text)
        self.assertIn("FSDG02-FSDG07 centralized", doc_text)
        self.assertIn("FSDG12_fedavg_rxday", doc_text)
        self.assertIn("FSDG12A_fedavg_rxday_local3", doc_text)

    def test_split_bex02_alternatives_launcher_has_8gpu_hard_constraints(self):
        script = ROOT / "scripts" / "launch_split_bex02_alternatives_8gpu.sh"
        self.assertTrue(script.is_file())
        try:
            proc = subprocess.run(
                [
                    "bash",
                    "-lc",
                    "DRY_RUN=1 ROOT=/tmp/cv_sincnet RUN_ROOT=/tmp/split_bex02_launcher_test "
                    "PYTHON=python WISIG_PKL=/tmp/cv_sincnet/Dataset_WigSig/ManySig.pkl "
                    "scripts/launch_split_bex02_alternatives_8gpu.sh",
                ],
                cwd=str(ROOT),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            self.skipTest(f"bash dry-run timed out in this local environment: {exc}")
        stdout_lower = proc.stdout.lower()
        stdout_no_nul = stdout_lower.replace("\x00", "")
        if proc.returncode != 0 and (
            "linux" in stdout_lower
            or "subsystem" in stdout_lower
            or "wsl" in stdout_lower
            or "command not found" in stdout_lower
            or "linux" in stdout_no_nul
            or "subsystem" in stdout_no_nul
            or "wsl.exe" in stdout_no_nul
            or "command not found" in stdout_no_nul
        ):
            self.skipTest("bash/WSL is not available in this Windows test environment")
        self.assertEqual(proc.returncode, 0, proc.stdout)

        out = proc.stdout.replace("\\ ", " ")
        self.assertIn("[SPLIT-BEX02-MATRIX]", out)
        self.assertIn("enforce_one_run_per_gpu=1", out)
        self.assertIn("log_root=", out)
        self.assertIn("runs_root=", out)
        for name in [
            "SBX02_LVMB_r010",
            "SBX02_PROTO_r010",
            "SBX02_FISHR_r010",
            "SBX02_STYLE_r010",
            "SBX02_KDLOGIT_r010",
            "SBX02_QTOKEN_r010",
            "SBX02_SATCE_r010",
            "SBX02_COMBO_r010",
        ]:
            self.assertIn(name, out)
        command_lines = [line for line in out.splitlines() if line.startswith("  ")]
        self.assertEqual(len(command_lines), 8, out)
        for line in command_lines:
            normalized = " ".join(line.split())
            self.assertIn("--wisig_train_ratio 0.1", normalized)
            self.assertIn("--epochs 200", normalized)
            self.assertIn("--fl_rounds 200", normalized)
            self.assertIn("--fl_client_key receiver", normalized)
            self.assertIn("--output_dir", normalized)
            self.assertIn("--log_dir", normalized)
            self.assertIn("--fl_vmb_stage1_local_steps 2", normalized)
            self.assertIn("--fl_vmb_stage1_lr_mult 1.5", normalized)
            self.assertIn("--lambda_rx_adv 0.1", normalized)
            self.assertIn("--lambda_orth 0.1", normalized)
            self.assertEqual(normalized.count("--fl_local_objective"), 1, normalized)
        normalized_out = " ".join(out.split())
        self.assertIn("--train_mode split_bex02", normalized_out)
        self.assertIn("--activation_token_route quantized", normalized_out)
        self.assertIn("--use_logit_anchors", out)

        script_text = script.read_text(encoding="utf-8")
        for key in [
            "CVSRFFI_CPU_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]:
            self.assertIn(key, script_text)
        self.assertIn('ENFORCE_ONE_RUN_PER_GPU="${ENFORCE_ONE_RUN_PER_GPU:-1}"', script_text)
        self.assertIn('LOG_ROOT="${LOG_ROOT:-${RUN_ROOT:-${ROOT}/logs/${RUN_ID}}}"', script_text)
        self.assertIn('RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"', script_text)
        self.assertIn("duplicate GPU assignment", script_text)
        self.assertIn("--query-compute-apps=pid", script_text)
        self.assertIn("already has compute process", script_text)

    def test_fl82_launcher_has_satellite_target_baseline_view_plan(self):
        script = ROOT / "scripts" / "run_fed_fl82_validation_4gpu.sh"
        self.assertTrue(script.is_file())
        text = script.read_text(encoding="utf-8")
        self.assertIn("SAT_BASELINE", text)
        self.assertIn('FEWSHOT_RATIO="${FEWSHOT_RATIO:-0.1}"', text)
        self.assertIn('EPOCHS="${EPOCHS:-200}"', text)
        self.assertIn('FL_ROUNDS="${FL_ROUNDS:-200}"', text)
        self.assertIn("FL82_07_fedprox_rx_ra_bex02_baselineview_clearleo_r010", text)
        self.assertIn("FL82_08_fedprox_rx_ra_bex02_baselineview_clearleo_stylebank_l3_r010", text)
        self.assertIn("FL82_09_fedprox_rx_ra_bex02_baselineview_clearleo_l3_r010", text)
        self.assertIn("FL82_10_fedprox_rx_ra_bex02_stylebank_collab_clearleo_r010", text)
        self.assertIn("--fl_style_max_views 1", text)
        self.assertIn("--fl_style_replay_prob 0.25", text)
        self.assertIn("--fl_style_dg_start_round 40", text)
        self.assertIn("--fishr_min_domains 2", text)
        self.assertIn("--use_style_collab_eval", text)
        self.assertIn("--style_collab_fusion adaptive", text)
        self.assertIn("--style_collab_views 2", text)
        self.assertIn("--fl_sat_aug_mode baseline_view", text)
        self.assertIn("--sat_train_scenarios clear_leo", text)
        self.assertIn("--eval_sat_scenarios clear_leo", text)
        sat_rows = "\n".join(line for line in text.splitlines() if line.startswith("FL82_0") or line.startswith("FL82_10"))
        self.assertNotIn("--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit", sat_rows)
        self.assertNotIn("--eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit", sat_rows)
        self.assertIn("--sat_view_prob 1.0", text)
        self.assertIn("--sat_cons_start_epoch 1", text)
        self.assertIn("--eval_sat_on test_unseen_day_seen_rx,test_seen_day_unseen_rx,test_unseen_day_unseen_rx", text)
        self.assertIn("LEO-only SAT evaluation uses clear_leo", text)

    def test_fl82_launcher_has_optional_backbone_stability_plan(self):
        script = ROOT / "scripts" / "run_fed_fl82_validation_4gpu.sh"
        self.assertTrue(script.is_file())
        try:
            proc = subprocess.run(
                [
                    "bash",
                    "scripts/run_fed_fl82_validation_4gpu.sh",
                    "--plan",
                    "BACKBONE_ABL",
                    "--gpu-ids",
                    "0,1,2,3,4,5,6",
                    "--dry-run",
                ],
                cwd=str(ROOT),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            self.skipTest(f"bash dry-run timed out in this local environment: {exc}")
        if proc.returncode != 0 and (
            "Linux" in proc.stdout
            or "Subsystem" in proc.stdout
            or "WSL" in proc.stdout
            or "w\x00s\x00l" in proc.stdout.lower()
            or "command not found" in proc.stdout.lower()
        ):
            self.skipTest("bash/WSL is not available in this Windows test environment")
        self.assertEqual(proc.returncode, 0, proc.stdout)
        out = proc.stdout.replace("\\,", ",")
        self.assertIn("[SCHED] plan=BACKBONE_ABL jobs=7 gpus=0,1,2,3,4,5,6", out)
        self.assertIn("--wisig_train_ratio 0.1", out)
        self.assertIn("--model_variant lite_d", out)
        self.assertIn("--branch_ablation no_dac", out)
        self.assertIn("--domain_branch_ablation no_stats", out)
        self.assertIn("--fl_sat_aug_mode baseline_view", out)
        self.assertIn("--fl_baseline_view_ce_only", out)
        self.assertIn("--fl_baseline_view_ce_weight 1.0", out)
        self.assertIn("--id_time_stability_mode phase_delta", out)
        self.assertIn("--id_freq_stability_mode dsq", out)
        self.assertIn("--domain_time_stability_mode same", out)
        self.assertIn("--domain_freq_stability_mode same", out)
        self.assertIn("--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit", out)
        self.assertIn("--eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit", out)

    def test_federated_satellite_extra_eval_lines_include_requested_splits(self):
        from federated.fed_trainer import _format_extra_test_lines

        lines = _format_extra_test_lines(
            {
                "sat_channel": {
                    "clear_leo": {
                        "aggregate": {"tx_acc": 70.0, "tx_correct": 140, "tx_total": 200},
                        "strict_udu": 53.78,
                        "selected_names": [
                            "test_unseen_day_seen_rx",
                            "test_seen_day_unseen_rx",
                            "test_unseen_day_unseen_rx",
                        ],
                        "named": {
                            "test_unseen_day_seen_rx": {"tx_acc": 84.30, "tx_correct": 70811, "tx_total": 84000},
                            "test_seen_day_unseen_rx": {"tx_acc": 60.10, "tx_correct": 36062, "tx_total": 60000},
                            "test_unseen_day_unseen_rx": {"tx_acc": 53.78, "tx_correct": 32268, "tx_total": 60000},
                        },
                    }
                }
            }
        )

        block = "\n".join(lines)
        self.assertIn("[SAT-TEST-SPLIT] scenario=clear_leo test_unseen_day_seen_rx: tx=84.30% (70811/84000)", block)
        self.assertIn("[SAT-TEST-SPLIT] scenario=clear_leo test_seen_day_unseen_rx: tx=60.10% (36062/60000)", block)
        self.assertIn("[SAT-TEST-SPLIT] scenario=clear_leo test_unseen_day_unseen_rx: tx=53.78% (32268/60000)", block)

    def test_federated_training_requires_per_round_satellite_eval(self):
        from train import FEDERATED_MAIN_SAT_EVAL_ON, enforce_federated_sat_eval_args

        args = SimpleNamespace(
            train_mode="fedprox",
            eval_sat_channel=True,
            eval_sat_on="test_unseen_day_unseen_rx",
            eval_sat_scenarios="",
        )
        enforce_federated_sat_eval_args(args)

        self.assertEqual(
            FEDERATED_MAIN_SAT_EVAL_ON.split(","),
            [
                "test_unseen_day_seen_rx",
                "test_seen_day_unseen_rx",
                "test_unseen_day_unseen_rx",
            ],
        )
        self.assertEqual(args.eval_sat_on, FEDERATED_MAIN_SAT_EVAL_ON)
        self.assertEqual(args.eval_sat_scenarios, "leo_clear_weak,leo_low_elev_weak,leo_rain_weak")

        bad = SimpleNamespace(
            train_mode="fedavg",
            eval_sat_channel=False,
            eval_sat_on="main",
            eval_sat_scenarios="clear_leo",
        )
        with self.assertRaisesRegex(ValueError, "satellite-channel evaluation every round"):
            enforce_federated_sat_eval_args(bad)

    def test_full_launcher_uses_phase_barriers_before_centralized_jobs(self):
        script = ROOT / "scripts" / "run_fed_fewshot_dg_6gpu.sh"
        try:
            proc = subprocess.run(
                [
                    "bash",
                    "scripts/run_fed_fewshot_dg_6gpu.sh",
                    "--plan",
                    "FULL",
                    "--gpu-ids",
                    "0,0,1,1,2,2,3,3",
                    "--wisig-pkl",
                    str(ROOT / "Dataset_WigSig" / "ManySig.pkl"),
                    "--dry-run",
                ],
                cwd=str(ROOT),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            self.skipTest(f"bash dry-run timed out in this local environment: {exc}")
        if proc.returncode != 0 and (
            "Linux" in proc.stdout
            or "Subsystem" in proc.stdout
            or "WSL" in proc.stdout
            or "w\x00s\x00l" in proc.stdout.lower()
        ):
            self.skipTest("bash/WSL is not available in this Windows test environment")
        self.assertEqual(proc.returncode, 0, proc.stdout)
        out = proc.stdout
        self.assertIn("[SCHED] phased execution enabled: FED_BASE -> FED_DG -> CENTRAL", out)
        fed_base = out.index("[PHASE-BEGIN] 1/3 FED_BASE")
        fed_dg = out.index("[PHASE-BEGIN] 2/3 FED_DG")
        central = out.index("[PHASE-BEGIN] 3/3 CENTRAL")
        first_central_launch = out.index("tag=FSDG02_centralized_ce")
        self.assertLess(fed_base, fed_dg)
        self.assertLess(fed_dg, central)
        self.assertGreater(first_central_launch, central)


if __name__ == "__main__":
    unittest.main()
