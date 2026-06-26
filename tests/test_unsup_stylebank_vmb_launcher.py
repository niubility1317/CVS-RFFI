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
SCRIPT = ROOT / "code" / "scripts" / "launch_unsup_stylebank_vmb_8gpu_20260603.sh"


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


class UnsupStylebankVmbLauncherTest(unittest.TestCase):
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

    def test_dry_run_renders_eight_commands_one_per_gpu(self):
        output = self._dry_run_output()
        command_lines = [ln for ln in output.splitlines() if ln.startswith("[USBV-CMD]")]
        self.assertEqual(len(command_lines), 8, output)
        gpu_counts = Counter()
        for ln in output.splitlines():
            if ln.startswith("[USBV]"):
                parts = dict(item.split("=", 1) for item in ln.split() if "=" in item)
                gpu_counts[int(parts["gpu"])] += 1
        self.assertEqual(gpu_counts, Counter({gpu: 1 for gpu in range(8)}))

    def test_each_command_keeps_formal_contract_and_singleton_flags(self):
        output = self._dry_run_output()
        commands = [ln.removeprefix("[USBV-CMD] ") for ln in output.splitlines() if ln.startswith("[USBV-CMD]")]
        self.assertEqual(len(commands), 8, output)
        seen_runs = set()
        singleton_flags = [
            "--run_name",
            "--wisig_train_ratio",
            "--epochs",
            "--fl_rounds",
            "--fl_client_key",
            "--train_mode",
            "--fl_local_objective",
            "--fl_vmb_stage1_objective",
            "--fl_sat_aug_mode",
            "--lambda_sat_cls",
            "--lambda_sat_cons",
        ]
        for command in commands:
            tokens = shlex.split(command)

            def flag_value(flag: str) -> str:
                self.assertIn(flag, tokens, f"missing {flag} in {command}")
                self.assertEqual(tokens.count(flag), 1, f"duplicate {flag} in {command}")
                idx = tokens.index(flag)
                self.assertLess(idx + 1, len(tokens), f"missing value for {flag} in {command}")
                return tokens[idx + 1]

            for flag in singleton_flags:
                flag_value(flag)
            run_name = flag_value("--run_name")
            self.assertNotIn(run_name, seen_runs)
            seen_runs.add(run_name)
            self.assertEqual(flag_value("--wisig_train_ratio"), "0.1")
            self.assertEqual(flag_value("--epochs"), "200")
            self.assertEqual(flag_value("--fl_rounds"), "200")
            self.assertEqual(flag_value("--fl_client_key"), "receiver")
            self.assertEqual(flag_value("--train_mode"), "fedcvs_vmb")
            self.assertEqual(flag_value("--fl_local_objective"), "receiver_agnostic_bex02")
        self.assertEqual(len(seen_runs), 8)

    def test_matrix_contains_domain_stylebank_fishr_mixstyle_and_diagnostic_lanes(self):
        output = self._dry_run_output()
        for expected in [
            "USBV_E0_vmb_ra_anchor_r010",
            "USBV_E1_domain_consistency_r010",
            "USBV_E2_domain_metadata_r010",
            "USBV_E3_stylebank_probe_r010",
            "USBV_E4_domain_stylebank_probe_r010",
            "USBV_E5_domain_stylebank_fishr_r010",
            "USBV_E6_domain_stylebank_fishr_mixstyle_r010",
            "USBV_E7_real_mix_upper_bound_diag_r010",
        ]:
            self.assertIn(expected, output)
        self.assertIn("--domain_unsup_pretrain_method consistency", output)
        self.assertIn("--domain_unsup_pretrain_method metadata_consistency", output)
        self.assertIn("--use_fed_style_bank", output)
        self.assertIn("--fl_style_domain_label_mode target_receiver", output)
        self.assertIn("--style_gate_min_accept_rate 0.50", output)
        self.assertIn("--lambda_fishr 0.01", output)
        self.assertIn("--use_mixstyle", output)
        self.assertIn("--fl_style_real_mix_samples 8", output)


if __name__ == "__main__":
    unittest.main()
