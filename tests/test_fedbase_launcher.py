import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


class FedbaseLauncherTest(unittest.TestCase):
    def test_direct_rafl_mode_applies_paper_defaults(self):
        sys.path.insert(0, str(ROOT / "code"))
        import train

        old_argv = list(sys.argv)
        try:
            sys.argv = ["train.py", "--train_mode", "rafl"]
            args = SimpleNamespace(
                train_mode="rafl",
                fedbase_paper_method="",
                fl_client_key="receiver_day",
                batch_size=128,
                lr=0.0002,
                fl_local_epochs=1,
                fl_clients_per_round=1.0,
                epochs=1,
                fl_rounds=1,
                rafl_candidate_clients=0,
                rafl_selected_clients=0,
                rafl_momentum=0.9,
            )

            out = train.apply_fedbase_paper_defaults(args)
        finally:
            sys.argv = old_argv

        self.assertEqual(out.fl_client_key, "receiver")
        self.assertEqual(out.batch_size, 64)
        self.assertEqual(out.lr, 0.001)
        self.assertEqual(out.fl_local_epochs, 5)
        self.assertEqual(out.fl_clients_per_round, 0.5)
        self.assertEqual(out.epochs, 200)
        self.assertEqual(out.fl_rounds, 200)
        self.assertEqual(out.rafl_candidate_clients, 0)
        self.assertEqual(out.rafl_selected_clients, 0)
        self.assertEqual(out.rafl_momentum, 0.0)

    def test_direct_fedbase_modes_apply_method_defaults(self):
        sys.path.insert(0, str(ROOT / "code"))
        import train

        expected = {
            "fedriei": {"batch_size": 16, "lr": 0.0001, "fl_local_epochs": 1, "fl_clients_per_round": 1.0},
            "fedfa": {"batch_size": 64, "lr": 0.01, "fl_local_epochs": 4, "fl_clients_per_round": 1.0, "fl_agg_weight": "uniform"},
            "fucl": {"batch_size": 128, "lr": 0.001, "fl_local_epochs": 1, "fl_clients_per_round": 1.0, "fucl_finetune_epochs": 20, "fedbase_feature_dim": 128},
            "rafl": {"batch_size": 64, "lr": 0.001, "fl_local_epochs": 5, "fl_clients_per_round": 0.5, "rafl_momentum": 0.0},
        }
        old_argv = list(sys.argv)
        try:
            for mode, checks in expected.items():
                with self.subTest(mode=mode):
                    sys.argv = ["train.py", "--train_mode", mode]
                    args = SimpleNamespace(
                        train_mode=mode,
                        fedbase_paper_method="",
                        fl_client_key="receiver_day",
                        batch_size=999,
                        lr=9.0,
                        wd=9.0,
                        fl_local_epochs=99,
                        fl_clients_per_round=0.25,
                        fl_agg_weight="num_samples",
                        epochs=1,
                        fl_rounds=1,
                        fucl_finetune_epochs=1,
                        fedbase_feature_dim=999,
                        rafl_momentum=0.9,
                    )
                    out = train.apply_fedbase_paper_defaults(args)

                    self.assertEqual(out.fl_client_key, "receiver")
                    self.assertEqual(out.epochs, 200)
                    self.assertEqual(out.fl_rounds, 200)
                    self.assertEqual(out.wd, 0.0)
                    for attr, value in checks.items():
                        self.assertEqual(getattr(out, attr), value)
        finally:
            sys.argv = old_argv

    def test_strict_paper_profile_overrides_only_method_specific_round_protocols(self):
        sys.path.insert(0, str(ROOT / "code"))
        import train

        strict_expectations = {
            "fucl": {"fl_rounds": 5, "fucl_local_validation_ratio": 0.1, "fucl_local_lr_patience": 10, "fucl_local_early_stop_patience": 20, "fucl_local_max_epochs": 200},
            "rafl": {"fl_rounds": 300, "rafl_candidate_clients": 10, "rafl_selected_clients": 5, "rafl_input_version": "paper_52x126"},
            "fedfa": {"fl_rounds": 40, "batch_size": 64, "fl_local_epochs": 4, "lr": 0.01, "fedfa_align_lambda": 0.03},
        }
        old_argv = list(sys.argv)
        try:
            for mode, checks in strict_expectations.items():
                with self.subTest(mode=mode):
                    sys.argv = ["train.py", "--train_mode", mode, "--fedbase_paper_profile", "strict_paper"]
                    args = SimpleNamespace(
                        train_mode=mode,
                        fedbase_paper_method="",
                        fedbase_paper_profile="strict_paper",
                        fl_client_key="receiver_day",
                        batch_size=999,
                        lr=9.0,
                        wd=9.0,
                        fl_local_epochs=99,
                        fl_clients_per_round=0.25,
                        fl_agg_weight="num_samples",
                        epochs=1,
                        fl_rounds=1,
                        fucl_finetune_epochs=1,
                        fucl_local_validation_ratio=0.5,
                        fucl_local_lr_patience=99,
                        fucl_local_lr_decay=0.9,
                        fucl_local_early_stop_patience=99,
                        fucl_local_max_epochs=0,
                        fedbase_feature_dim=999,
                        fedfa_align_lambda=9.0,
                        rafl_momentum=0.9,
                        rafl_candidate_clients=0,
                        rafl_selected_clients=0,
                        rafl_input_version="wisig_complex",
                    )
                    out = train.apply_fedbase_paper_defaults(args)

                    self.assertEqual(out.fl_client_key, "receiver")
                    self.assertEqual(out.epochs, 200)
                    self.assertEqual(out.fedbase_paper_profile, "strict_paper")
                    for attr, value in checks.items():
                        self.assertEqual(getattr(out, attr), value)
        finally:
            sys.argv = old_argv

    def test_train_help_exposes_four_fedbase_paper_modes(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "code" / "train.py"), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=60,
        )

        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        help_text = proc.stdout
        for mode in ["fedriei", "fedfa", "fucl", "rafl"]:
            self.assertIn(mode, help_text)
        for flag in [
            "--fedbase_paper_method",
            "--fedriei_lambda_mi",
            "--fedriei_lambda_ie",
            "--fedriei_gradient_compression",
            "--fedriei_compression_noise_std",
            "--fedriei_server_lr",
            "--fedfa_align_lambda",
            "--fedbase_paper_profile",
            "--fucl_temperature",
            "--fucl_pretrain_lr",
            "--fucl_finetune_lr",
            "--fucl_local_validation_ratio",
            "--fucl_local_lr_patience",
            "--fucl_local_lr_decay",
            "--fucl_local_early_stop_patience",
            "--fucl_local_max_epochs",
            "--fucl_sample_rate_hz",
            "--fucl_tdl_rms_delay_min_ns",
            "--fucl_tdl_rms_delay_max_ns",
            "--fucl_tdl_doppler_min_hz",
            "--fucl_tdl_doppler_max_hz",
            "--fucl_tdl_snr_min_db",
            "--fucl_tdl_snr_max_db",
            "--fucl_cis_n_fft",
            "--fucl_cis_hop_length",
            "--fucl_cis_freq_bins",
            "--fucl_cis_time_bins",
            "--rafl_lambda_rx",
            "--rafl_momentum",
            "--rafl_client_selection",
            "--rafl_input_version",
            "--rafl_selected_clients",
            "--rafl_candidate_clients",
            "--rafl_candidate_fraction",
            "--rafl_selection_eval_ratio",
            "--rafl_selection_dataset",
            "--rafl_spec_freq_bins",
            "--rafl_spec_time_bins",
        ]:
            self.assertIn(flag, help_text)

    def test_fedbase_queue_dry_run_emits_paper_named_cvs_commands(self):
        script = ROOT / "run_fedbase_paper_queue.sh"
        self.assertTrue(script.is_file())
        env = dict(os.environ)
        env.update(
            {
                "DRY_RUN": "1",
                "ROOT": "/tmp/cv_sincnet",
                "WISIG_PKL": "/tmp/cv_sincnet/Dataset_WigSig/ManySig.pkl",
                "RUN_ROOT": "/tmp/fedbase_runs",
                "LOG_ROOT": "/tmp/fedbase_logs",
                "GPU_IDS": "0,1",
                "TIMESTAMP": "20260603_122924",
            }
        )
        try:
            proc = subprocess.run(
                ["bash", script.name, "--dry-run"],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=60,
            )
        except FileNotFoundError:
            self.skipTest("bash is not available in this Windows environment")

        self.assertEqual(proc.returncode, 0, proc.stdout)
        out = proc.stdout
        for mode in ["fedriei", "fedfa", "fucl", "rafl"]:
            self.assertIn(f"--train_mode {mode}", out)
            pattern = rf'\d{{8}}_\d{{6}}_fedbase_{re.escape(mode)}_cvsrffi_r010'
            self.assertRegex(out, rf'--run_name "{pattern}"')
            self.assertRegex(out, rf'--output_dir "[^"]*/{pattern}"')
            self.assertRegex(out, rf'--log_dir "[^"]*/{pattern}"')
        for token in [
            "--wisig_train_ratio 0.1",
            "--epochs 200",
            "--fl_rounds 200",
            "--fl_client_key receiver",
            "--eval_sat_channel",
            "--fedriei_gradient_compression none",
            "--fedriei_compression_noise_std 0.01",
            "--fedbase_feature_dim 128",
            "--fucl_finetune_epochs 20",
            "--fucl_cis_n_fft 64",
            "--fucl_cis_hop_length 32",
            "--fucl_cis_freq_bins 26",
            "--fucl_cis_time_bins 126",
            "--rafl_lambda_rx 0.1",
            "--rafl_momentum 0.0",
            "--rafl_selection_dataset internal_train_split",
            "--rafl_input_version wisig_complex",
            "--rafl_spec_n_fft 52",
            "--rafl_spec_hop_length 2",
            "--fl_clients_per_round 0.5",
            "--rafl_selected_clients 0",
            "--rafl_candidate_clients 0",
            "--rafl_candidate_fraction 1.0",
            "--rafl_selection_eval_ratio 0.1",
        ]:
            self.assertIn(token, out)
        self.assertEqual(len([line for line in out.splitlines() if line.startswith("CMD ")]), 4)
        self.assertNotIn("PID ", out)

    def test_fedbase_queue_defaults_to_dry_run(self):
        script = ROOT / "run_fedbase_paper_queue.sh"
        try:
            proc = subprocess.run(
                ["bash", script.name],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=60,
            )
        except FileNotFoundError:
            self.skipTest("bash is not available in this Windows environment")

        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("dry_run=1", proc.stdout)
        self.assertIn("gpus=0,1", proc.stdout)
        self.assertIn("max_jobs_per_gpu=2", proc.stdout)
        self.assertEqual(len([line for line in proc.stdout.splitlines() if line.startswith("CMD ")]), 4)
        self.assertNotIn("PID ", proc.stdout)

    def test_fedbase_queue_strict_profile_emits_method_specific_paper_rounds(self):
        script = ROOT / "run_fedbase_paper_queue.sh"
        try:
            proc = subprocess.run(
                ["bash", script.name, "--dry-run", "--profile", "strict_paper", "--methods", "fedfa,fucl,rafl"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=60,
            )
        except FileNotFoundError:
            self.skipTest("bash is not available in this Windows environment")

        self.assertEqual(proc.returncode, 0, proc.stdout)
        out = proc.stdout
        self.assertIn("profile=strict_paper", out)
        self.assertRegex(out, r"--train_mode fedfa .* --fl_rounds 40 .* --fedbase_paper_method FedFA --fedbase_paper_profile strict_paper")
        self.assertRegex(out, r"--train_mode fucl .* --fl_rounds 5 .* --fedbase_paper_method FUCL --fedbase_paper_profile strict_paper")
        self.assertRegex(out, r"--train_mode rafl .* --fl_rounds 300 .* --fedbase_paper_method RAFL --fedbase_paper_profile strict_paper")
        self.assertIn("--fucl_local_max_epochs 200", out)
        self.assertIn("--rafl_selected_clients 5 --rafl_candidate_clients 10", out)
        self.assertIn("--rafl_input_version paper_52x126", out)
        self.assertEqual(len([line for line in out.splitlines() if line.startswith("CMD ")]), 3)

    def test_queue_prefers_code_train_when_root_train_also_exists(self):
        script = ROOT / "run_rafl_input_versions_queue.sh"
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp_root = Path(td)
                (tmp_root / "code").mkdir()
                (tmp_root / "code" / "train.py").write_text("# preferred train entry\n", encoding="utf-8")
                (tmp_root / "train.py").write_text("# legacy root train entry\n", encoding="utf-8")
                (tmp_root / script.name).write_bytes(script.read_bytes())

                env = dict(os.environ)
                env.update(
                    {
                        "DRY_RUN": "1",
                        "WISIG_PKL": str(tmp_root / "Dataset_WigSig" / "ManySig.pkl").replace("\\", "/"),
                        "RUN_ROOT": str(tmp_root / "runs").replace("\\", "/"),
                        "LOG_ROOT": str(tmp_root / "logs").replace("\\", "/"),
                        "GPU_IDS": "0",
                        "TIMESTAMP": "20260603_180001",
                    }
                )
                proc = subprocess.run(
                    ["bash", script.name, "--dry-run"],
                    cwd=tmp_root,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                    timeout=60,
                )
        except FileNotFoundError:
            self.skipTest("bash is not available in this Windows environment")

        self.assertEqual(proc.returncode, 0, proc.stdout)
        out = proc.stdout.replace("\\", "/")
        self.assertRegex(out, r"train_script=\S+/code/train\.py")

    def test_rafl_input_versions_queue_emits_two_adaptive_commands(self):
        script = ROOT / "run_rafl_input_versions_queue.sh"
        self.assertTrue(script.is_file())
        env = dict(os.environ)
        env.update(
            {
                "DRY_RUN": "1",
                "ROOT": "/tmp/cv_sincnet",
                "WISIG_PKL": "/tmp/cv_sincnet/Dataset_WigSig/ManySig.pkl",
                "RUN_ROOT": "/tmp/fedbase_runs",
                "LOG_ROOT": "/tmp/fedbase_logs",
                "GPU_IDS": "0,1",
                "TIMESTAMP": "20260603_180000",
            }
        )
        try:
            proc = subprocess.run(
                ["bash", script.name, "--dry-run"],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=60,
            )
        except FileNotFoundError:
            self.skipTest("bash is not available in this Windows environment")

        self.assertEqual(proc.returncode, 0, proc.stdout)
        out = proc.stdout
        self.assertIn("fedbase_rafl_paper_52x126_cvsrffi_r010", out)
        self.assertIn("fedbase_rafl_wisig_complex_cvsrffi_r010", out)
        self.assertIn(
            "--rafl_input_version paper_52x126 --rafl_spec_n_fft 52 --rafl_spec_hop_length 2 --rafl_spec_win_length 52 --rafl_spec_freq_bins 52 --rafl_spec_time_bins 126",
            out,
        )
        self.assertIn("--rafl_input_version wisig_complex --rafl_spec_n_fft 52 --rafl_spec_hop_length 2 --rafl_spec_win_length 52", out)
        self.assertIn("--rafl_selection_dataset internal_train_split", out)
        self.assertIn("--rafl_momentum 0.0", out)
        self.assertIn("--rafl_selected_clients 0 --rafl_candidate_clients 0 --rafl_candidate_fraction 1.0", out)
        self.assertEqual(len([line for line in out.splitlines() if line.startswith("CMD ")]), 2)
        self.assertNotIn("PID ", out)

    def test_rafl_input_versions_queue_strict_profile_binds_paper_input_to_lld(self):
        script = ROOT / "run_rafl_input_versions_queue.sh"
        self.assertTrue(script.is_file())
        env = dict(os.environ)
        env.update(
            {
                "DRY_RUN": "1",
                "ROOT": "/tmp/cv_sincnet",
                "WISIG_PKL": "/tmp/cv_sincnet/Dataset_WigSig/ManySig.pkl",
                "RUN_ROOT": "/tmp/fedbase_runs",
                "LOG_ROOT": "/tmp/fedbase_logs",
                "GPU_IDS": "0",
                "TIMESTAMP": "20260603_180000",
                "RAFL_PROFILE": "strict_paper",
                "VARIANTS": "paper_52x126",
            }
        )
        try:
            proc = subprocess.run(
                ["bash", script.name, "--dry-run", "--profile", "strict_paper", "--variants", "paper_52x126"],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=60,
            )
        except FileNotFoundError:
            self.skipTest("bash is not available in this Windows environment")

        self.assertEqual(proc.returncode, 0, proc.stdout)
        out = proc.stdout
        self.assertIn("--fedbase_paper_profile strict_paper", out)
        self.assertIn("--fl_rounds 300", out)
        self.assertIn("--rafl_selected_clients 5 --rafl_candidate_clients 10", out)
        self.assertIn("--rafl_input_version paper_52x126", out)
        self.assertNotIn("--rafl_input_version wisig_complex", out)


if __name__ == "__main__":
    unittest.main()
