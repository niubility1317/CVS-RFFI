import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "launch_unsup_stylebank_vmb_fix_20260603.sh"


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


class UnsupStylebankVmbFixLauncherTest(unittest.TestCase):
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

    def test_fixed_matrix_keeps_contract_and_repaired_domain_scope(self):
        output = self._dry_run_output()
        commands = [ln.removeprefix("[USBV-CMD] ") for ln in output.splitlines() if ln.startswith("[USBV-CMD]")]
        self.assertEqual(len(commands), 8, output)
        domain_commands = []
        for command in commands:
            tokens = shlex.split(command)

            def flag_value(flag: str) -> str:
                self.assertIn(flag, tokens, f"missing {flag} in {command}")
                self.assertEqual(tokens.count(flag), 1, f"duplicate {flag} in {command}")
                idx = tokens.index(flag)
                self.assertLess(idx + 1, len(tokens), f"missing value for {flag} in {command}")
                return tokens[idx + 1]

            self.assertEqual(flag_value("--wisig_train_ratio"), "0.1")
            self.assertEqual(flag_value("--epochs"), "200")
            self.assertEqual(flag_value("--fl_rounds"), "200")
            self.assertEqual(flag_value("--fl_client_key"), "receiver")
            self.assertEqual(flag_value("--train_mode"), "fedcvs_vmb")
            self.assertEqual(flag_value("--fl_local_objective"), "receiver_agnostic_bex02")
            if flag_value("--fl_vmb_stage1_objective") == "domain_unsup_pretrain":
                domain_commands.append(tokens)

        self.assertEqual(len(domain_commands), 6, output)
        for tokens in domain_commands:
            self.assertIn("--fl_domain_pretrain_train_scope", tokens)
            self.assertEqual(tokens[tokens.index("--fl_domain_pretrain_train_scope") + 1], "all")
            self.assertIn("--domain_unsup_client_compact_weight", tokens)
            self.assertEqual(tokens[tokens.index("--domain_unsup_client_compact_weight") + 1], "0.50")

    def test_fixed_matrix_uses_f_series_names_and_delays_style_interventions(self):
        output = self._dry_run_output()
        for expected in [
            "USBV_F0_vmb_ra_anchor_r010",
            "USBV_F1_domain_consistency_fixed_r010",
            "USBV_F2_domain_metadata_fixed_r010",
            "USBV_F4_domain_stylebank_probe_fixed_r010",
            "USBV_F7_real_mix_upper_bound_diag_fixed_r010",
        ]:
            self.assertIn(expected, output)
        self.assertNotIn("USBV_E1_domain_consistency_r010", output)
        self.assertIn("--fl_style_replay_start_round 70", output)
        self.assertIn("--fl_style_dg_start_round 120", output)
        self.assertIn("--fl_style_real_mix_start_round 120", output)


if __name__ == "__main__":
    unittest.main()
