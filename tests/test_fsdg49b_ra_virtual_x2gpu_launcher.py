import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "launch_fsdg49b_ra_virtual_x2gpu_20260601.sh"


def _find_bash() -> str | None:
    git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git" / "usr" / "bin" / "bash.exe"
    if git_bash.exists():
        return str(git_bash)
    return shutil.which("bash")


def _bash_path(path: str, bash: str) -> str:
    path = path.replace("\\", "/")
    if len(path) >= 3 and path[1] == ":" and path[2] == "/":
        if "/Git/usr/bin/bash.exe" in bash.replace("\\", "/"):
            return f"/{path[0].lower()}{path[2:]}"
        return f"/mnt/{path[0].lower()}{path[2:]}"
    return path


class Fsdg49bRaVirtualX2GpuLauncherTest(unittest.TestCase):
    def _dry_run_output(self) -> str:
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tmp_posix = _bash_path(tmp, bash)
            python_bin = _bash_path(sys.executable, bash)
            command = (
                f'ROOT="{tmp_posix}" '
                f'PYTHON="{python_bin}" '
                f'WISIG_PKL="{tmp_posix}/ManySig.pkl" '
                "DRY_RUN=1 "
                f'bash "{_bash_path(str(SCRIPT), bash)}"'
            )
            proc = subprocess.run(
                [bash, "-lc", command],
                cwd=ROOT,
                env=os.environ.copy(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, output)
        return output

    def test_dry_run_renders_sixteen_commands_two_per_gpu(self):
        output = self._dry_run_output()
        union_lines = [ln for ln in output.splitlines() if ln.startswith("[X2GPU-CMD]")]
        self.assertEqual(len(union_lines), 16, output)

        gpu_counts = Counter()
        for ln in output.splitlines():
            if ln.startswith("[X2GPU]"):
                parts = dict(item.split("=", 1) for item in ln.split() if "=" in item)
                gpu_counts[int(parts["gpu"])] += 1
        self.assertEqual(gpu_counts, Counter({gpu: 2 for gpu in range(8)}))

    def test_each_command_keeps_formal_fl_contract_and_singleton_flags(self):
        output = self._dry_run_output()
        commands = [ln.removeprefix("[X2GPU-CMD] ") for ln in output.splitlines() if ln.startswith("[X2GPU-CMD]")]
        self.assertEqual(len(commands), 16, output)
        seen_runs = set()

        for command in commands:
            tokens = shlex.split(command)

            def flag_value(flag: str) -> str:
                self.assertIn(flag, tokens, f"missing {flag} in {command}")
                self.assertEqual(tokens.count(flag), 1, f"duplicate {flag} in {command}")
                idx = tokens.index(flag)
                self.assertLess(idx + 1, len(tokens), f"missing value for {flag} in {command}")
                return tokens[idx + 1]

            run_name = flag_value("--run_name")
            self.assertNotIn(run_name, seen_runs)
            seen_runs.add(run_name)

            self.assertEqual(flag_value("--wisig_train_ratio"), "0.1")
            self.assertEqual(flag_value("--epochs"), "200")
            self.assertEqual(flag_value("--fl_rounds"), "200")
            self.assertEqual(flag_value("--fl_client_key"), "receiver")
            self.assertEqual(flag_value("--fl_local_objective"), "receiver_agnostic_bex02")
            self.assertEqual(tokens.count("--fl_sat_aug_mode"), 1, f"duplicate SAT mode in {command}")
            self.assertEqual(tokens.count("--lambda_sat_cls"), 1, f"duplicate SAT cls weight in {command}")
            self.assertEqual(tokens.count("--lambda_sat_cons"), 1, f"duplicate SAT cons weight in {command}")

            if run_name.startswith("VMB_"):
                self.assertEqual(flag_value("--train_mode"), "fedcvs_vmb")
                self.assertEqual(flag_value("--fl_vmb_stage"), "auto")
                self.assertIn("--fl_vmb_batches_per_client", tokens)
            else:
                self.assertEqual(flag_value("--train_mode"), "fedprox")
                self.assertIn("--fedprox_mu", tokens)
                self.assertIn("--lambda_rx_adv", tokens)
                self.assertIn("--grl_lambda", tokens)

        self.assertEqual(len(seen_runs), 16)

    def test_matrix_contains_fsdg49b_ra_virtual_and_vmb_lanes(self):
        output = self._dry_run_output()
        for expected in [
            "FSDG49B_R01_repro_zidcoral003_r010",
            "FSDG49B_R02_zidcoral000_control_r010",
            "FSDG49B_R04_zidcoral003_start60_r010",
            "RA_BEX02_R06_grl075_r010",
            "RA_BEX02_R08_localepoch3_r010",
            "SATCE_R09_late_w025_R100_r010",
            "SATCE_R10_late_w050_R120_r010",
            "VIRT_R11_style_stats_forced_probe_R80_r010",
            "VIRT_R14_zdom_coral0005_forced_probe_r010",
            "VMB_R15_bpc4_no_cen_r010",
            "VMB_R16_bpc8_no_cen_r010",
        ]:
            self.assertIn(expected, output)

        self.assertIn("--lambda_fl_coral_zid_global 0.003", output)
        self.assertIn("--lambda_fl_coral_zdom_global 0.0005", output)
        self.assertIn("--fl_baseline_view_ce_weight 0.25", output)
        self.assertIn("--fl_baseline_view_ce_weight 0.50", output)
        self.assertIn("--use_fl_style_bank_stats", output)
        self.assertIn("--use_fed_style_bank", output)
        self.assertIn("--fl_style_domain_label_mode target_receiver", output)
        self.assertIn("--fl_style_zdom_probe_every 10", output)
        self.assertIn("--fl_style_zdom_probe_force_batch", output)
        self.assertIn("--fl_style_replay_prob 0.00", output)
        self.assertIn("--fl_vmb_batches_per_client 8", output)

    def test_vmb_bpc8_is_clean_underfit_control_without_style_replay(self):
        output = self._dry_run_output()
        commands = [ln.removeprefix("[X2GPU-CMD] ") for ln in output.splitlines() if ln.startswith("[X2GPU-CMD]")]
        vmb_bpc8 = next(cmd for cmd in commands if "--run_name VMB_R16_bpc8_no_cen_r010" in cmd)
        tokens = shlex.split(vmb_bpc8)
        self.assertEqual(tokens[tokens.index("--fl_vmb_batches_per_client") + 1], "8")
        self.assertNotIn("--use_fed_style_bank", tokens)
        self.assertNotIn("--fl_style_replay_prob", tokens)

    def test_non_dry_run_counts_planned_processes_against_gpu_capacity(self):
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "train.py").write_text("import time\ntime.sleep(0.2)\n", encoding="utf-8")
            fake_bin = tmp_path / "fake-bin"
            fake_bin.mkdir()
            fake_smi = fake_bin / "nvidia-smi"
            fake_smi.write_text(
                "#!/usr/bin/env bash\n"
                "for arg in \"$@\"; do\n"
                "  if [[ \"$arg\" == \"--id=0\" ]]; then echo 99999; exit 0; fi\n"
                "done\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_smi.chmod(0o755)

            tmp_posix = _bash_path(tmp, bash)
            command = (
                f'ROOT="{tmp_posix}" '
                f'PYTHON="{_bash_path(sys.executable, bash)}" '
                f'NVIDIA_SMI_BIN="{_bash_path(str(fake_smi), bash)}" '
                f'WISIG_PKL="{tmp_posix}/ManySig.pkl" '
                "MAX_PROCS_PER_GPU=2 ONLY_GPU=0 "
                f'bash "{_bash_path(str(SCRIPT), bash)}"'
            )
            proc = subprocess.run(
                [bash, "-lc", command],
                cwd=ROOT,
                env=os.environ.copy(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

            output = proc.stdout + proc.stderr
            self.assertEqual(proc.returncode, 0, output)
            self.assertIn("[SKIP] gpu=0 active_compute=1 planned=1 max=2", output)
            pid_file = tmp_path / "logs" / "20260601_220035_fsdg49b_ra_virtual_x2gpu" / "launch_pids.tsv"
            rows = pid_file.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(len(rows), 2, output)
        self.assertIn("FSDG49B_R01_repro_zidcoral003_r010\t0\t", rows[1])
        self.assertNotIn("FSDG49B_R02_zidcoral000_control_r010", "\n".join(rows))

    def test_non_dry_run_refuses_duplicate_run_id_outputs(self):
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "train.py").write_text("print('should not launch')\n", encoding="utf-8")
            fake_bin = tmp_path / "fake-bin"
            fake_bin.mkdir()
            fake_smi = fake_bin / "nvidia-smi"
            fake_smi.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_smi.chmod(0o755)
            log_root = tmp_path / "logs" / "20260601_220035_fsdg49b_ra_virtual_x2gpu"
            log_root.mkdir(parents=True)
            (log_root / "launch_pids.tsv").write_text(
                "run_name\tgpu\tpid\tlog_path\toutput_dir\n"
                "FSDG49B_R01_repro_zidcoral003_r010\t0\t123\told.log\told.run\n",
                encoding="utf-8",
            )

            tmp_posix = _bash_path(tmp, bash)
            command = (
                f'ROOT="{tmp_posix}" '
                f'PYTHON="{_bash_path(sys.executable, bash)}" '
                f'NVIDIA_SMI_BIN="{_bash_path(str(fake_smi), bash)}" '
                f'WISIG_PKL="{tmp_posix}/ManySig.pkl" '
                "ONLY_INDEX=0 "
                f'bash "{_bash_path(str(SCRIPT), bash)}"'
            )
            proc = subprocess.run(
                [bash, "-lc", command],
                cwd=ROOT,
                env=os.environ.copy(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

        output = proc.stdout + proc.stderr
        self.assertNotEqual(proc.returncode, 0, output)
        self.assertIn("already recorded", output)
        self.assertIn("refusing duplicate launch", output)
        self.assertNotIn("should not launch", output)


if __name__ == "__main__":
    unittest.main()
