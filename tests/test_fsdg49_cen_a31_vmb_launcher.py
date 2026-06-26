import os
import shutil
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "launch_fsdg49_cen_a31_vmb_union_20260601.sh"


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


class Fsdg49CenA31VmbLauncherTest(unittest.TestCase):
    def test_launcher_contains_required_matrix_and_hard_constraints(self):
        text = SCRIPT.read_text(encoding="utf-8")

        for required in [
            "--wisig_train_ratio 0.1",
            "--epochs 200",
            "--fl_rounds 200",
            "--fl_client_key receiver",
            "--fl_clients_per_round 1.0",
            "FSDG49A_anchor_fedprox_ra_cvs_r010_r200",
            "FSDG49D_rxadv_cenA31_full_r010",
            "VMB_AUDIT_auto_stage20_receiver_r010",
            "VMB_FULL_rxadv_cenA31_audit_r010",
        ]:
            self.assertIn(required, text)

        self.assertIn("--train_mode fedprox", text)
        self.assertIn("--fl_local_objective receiver_agnostic_bex02", text)
        self.assertIn("--fl_sat_aug_mode cvs_consistency", text)
        self.assertIn("--fl_sat_aug_mode baseline_view", text)
        self.assertIn("--fl_baseline_view_ce_only", text)
        self.assertIn("--domain_freq_stability_mode dsq", text)
        self.assertIn("--group_ce_mode smooth_dro_capped", text)
        self.assertIn("--train_mode fedcvs_vmb", text)
        self.assertIn("--fl_vmb_stage auto", text)
        self.assertIn("--fl_style_zdom_probe_every 10", text)

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

    def test_launcher_dry_run_renders_eight_unique_gpu_commands(self):
        output = self._dry_run_output()
        self.assertEqual(output.count("[UNION-CMD]"), 8, output)
        for gpu in range(8):
            self.assertIn(f"gpu={gpu}", output)
        self.assertIn("FSDG49D_rxadv_cenA31_full_r010", output)
        self.assertIn("--lambda_rx_adv 1.0", output)
        self.assertIn("--lambda_sat_cls 0.10", output)

    def test_dry_run_commands_enforce_contract_per_run(self):
        output = self._dry_run_output()
        command_lines = [line.removeprefix("[UNION-CMD] ") for line in output.splitlines() if line.startswith("[UNION-CMD]")]
        self.assertEqual(len(command_lines), 8, output)

        seen_runs = set()
        for expected_gpu, command in enumerate(command_lines):
            tokens = shlex.split(command)
            self.assertIn(f"CUDA_VISIBLE_DEVICES={expected_gpu}", tokens)

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
            self.assertEqual(flag_value("--lambda_rx_adv"), "1.0")

            if run_name.startswith("VMB_"):
                self.assertEqual(flag_value("--train_mode"), "fedcvs_vmb")
                self.assertEqual(flag_value("--fl_vmb_stage"), "auto")
                self.assertIn("--fl_style_zdom_probe_every", tokens)
                self.assertIn("--use_tx_adv_on_zdom", tokens)
            else:
                self.assertEqual(flag_value("--train_mode"), "fedprox")
                self.assertEqual(flag_value("--fedprox_mu"), "0.01")

            if run_name in {
                "FSDG49A_anchor_fedprox_ra_cvs_r010_r200",
                "FSDG49B_rxadv_zidcoral_diag_r010",
            }:
                self.assertEqual(flag_value("--fl_sat_aug_mode"), "cvs_consistency")
                self.assertEqual(flag_value("--lambda_sat_cls"), "0.10")

            if run_name in {
                "FSDG49C_cenA31_sat_dsq_groupce_r010",
                "FSDG49D_rxadv_cenA31_full_r010",
                "FSDG49E_rxadv_satonly_no_groupce_r010",
            }:
                self.assertEqual(flag_value("--fl_sat_aug_mode"), "baseline_view")
                self.assertIn("--fl_baseline_view_ce_only", tokens)

            if run_name in {
                "FSDG49C_cenA31_sat_dsq_groupce_r010",
                "FSDG49D_rxadv_cenA31_full_r010",
                "VMB_FULL_rxadv_cenA31_audit_r010",
            }:
                self.assertEqual(flag_value("--domain_freq_stability_mode"), "dsq")
                self.assertEqual(flag_value("--group_ce_mode"), "smooth_dro_capped")

        self.assertEqual(len(seen_runs), 8)

    def test_non_dry_run_skips_gpu_at_capacity(self):
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "train.py").write_text("print('should not launch')\n", encoding="utf-8")
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            fake_nvidia = bin_dir / "nvidia-smi"
            fake_nvidia.write_text("#!/usr/bin/env bash\nprintf '111\\n222\\n'\n", encoding="utf-8")
            os.chmod(fake_nvidia, 0o755)

            tmp_posix = _bash_path(tmp, bash)
            command = (
                f'PATH="{tmp_posix}/bin:$PATH" '
                f'ROOT="{tmp_posix}" '
                f'PYTHON="{_bash_path(sys.executable, bash)}" '
                f'WISIG_PKL="{tmp_posix}/ManySig.pkl" '
                "ONLY_INDEX=0 MAX_PROCS_PER_GPU=2 "
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
        self.assertIn("[SKIP] gpu=0 active_compute=2 max=2", output)
        self.assertNotIn("should not launch", output)

    def test_non_dry_run_fails_closed_when_gpu_query_fails(self):
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "train.py").write_text("print('should not launch')\n", encoding="utf-8")
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            fake_nvidia = bin_dir / "nvidia-smi"
            fake_nvidia.write_text("#!/usr/bin/env bash\necho query failed >&2\nexit 42\n", encoding="utf-8")
            os.chmod(fake_nvidia, 0o755)

            tmp_posix = _bash_path(tmp, bash)
            command = (
                f'PATH="{tmp_posix}/bin:$PATH" '
                f'ROOT="{tmp_posix}" '
                f'PYTHON="{_bash_path(sys.executable, bash)}" '
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
        self.assertIn("nvidia-smi query failed", output)
        self.assertIn("refusing to launch", output)

    def test_non_dry_run_refuses_duplicate_run_id_outputs(self):
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "train.py").write_text("print('should not launch')\n", encoding="utf-8")
            log_root = tmp_path / "logs" / "20260601_163012_fsdg49_cen_a31_vmb_union"
            log_root.mkdir(parents=True)
            (log_root / "launch_pids.tsv").write_text(
                "run_name\tgpu\tpid\tlog_path\toutput_dir\n"
                "FSDG49A_anchor_fedprox_ra_cvs_r010_r200\t0\t123\told.log\told.run\n",
                encoding="utf-8",
            )

            tmp_posix = _bash_path(tmp, bash)
            command = (
                f'ROOT="{tmp_posix}" '
                f'PYTHON="{_bash_path(sys.executable, bash)}" '
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
