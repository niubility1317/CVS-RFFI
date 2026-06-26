import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
SCRIPT = CODE_ROOT / "scripts" / "launch_ce_grl_central_vs_fed_20260603.sh"


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


class CeGrlCentralFedLauncherTest(unittest.TestCase):
    def _dry_run_output(self) -> str:
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp:
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
                cwd=PROJECT_ROOT,
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

    def test_dry_run_renders_four_clean_commands(self):
        output = self._dry_run_output()
        self.assertEqual(output.count("[CE-GRL-CMD]"), 4, output)
        for run_name in [
            "CEGRL_CEN_CE_r010",
            "CEGRL_CEN_GRL_r010",
            "CEGRL_FEDAVG_CE_receiver_r010",
            "CEGRL_FEDAVG_GRL_receiver_r010",
        ]:
            self.assertIn(run_name, output)

    def test_each_command_has_only_ce_grl_contract(self):
        output = self._dry_run_output()
        commands = [line.removeprefix("[CE-GRL-CMD] ") for line in output.splitlines() if line.startswith("[CE-GRL-CMD]")]
        self.assertEqual(len(commands), 4, output)

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
            self.assertEqual(flag_value("--wisig_domain"), "rx")
            self.assertEqual(flag_value("--lambda_dom"), "0")
            self.assertEqual(flag_value("--lambda_orth"), "0")
            self.assertEqual(flag_value("--lambda_cons"), "0")
            self.assertEqual(flag_value("--lambda_group_ce"), "0")
            self.assertEqual(flag_value("--lambda_fishr"), "0")
            self.assertEqual(flag_value("--lambda_proto"), "0")
            self.assertEqual(flag_value("--lambda_supcon_id"), "0")
            self.assertEqual(flag_value("--lambda_sat_cls"), "0")
            self.assertEqual(flag_value("--lambda_sat_cons"), "0")
            self.assertEqual(flag_value("--label_smoothing"), "0.0")
            self.assertIn("--force_ce_grl_only", tokens)
            self.assertIn("--no_use_aug", tokens)
            self.assertIn("--no_use_mixstyle", tokens)
            self.assertIn("--no_use_sat_consistency", tokens)
            self.assertIn("--no_use_fed_style_bank", tokens)
            self.assertIn("--no_use_fl_style_bank_stats", tokens)

            self.assertNotIn("--use_mixstyle", tokens)
            self.assertNotIn("--use_sat_consistency", tokens)
            self.assertNotIn("--use_fed_style_bank", tokens)
            self.assertNotIn("--train_mode", tokens[tokens.index("--train_mode") + 2 :])

            if run_name == "CEGRL_CEN_CE_r010":
                self.assertEqual(flag_value("--train_mode"), "centralized")
                self.assertEqual(flag_value("--lambda_adv"), "0")
                self.assertEqual(flag_value("--lambda_rx_adv"), "0")
                self.assertNotIn("--fl_local_objective", tokens)
            elif run_name == "CEGRL_CEN_GRL_r010":
                self.assertEqual(flag_value("--train_mode"), "centralized")
                self.assertEqual(flag_value("--lambda_adv"), "1.0")
                self.assertEqual(flag_value("--lambda_rx_adv"), "0")
                self.assertNotIn("--fl_local_objective", tokens)
            elif run_name == "CEGRL_FEDAVG_CE_receiver_r010":
                self.assertEqual(flag_value("--train_mode"), "fedavg")
                self.assertEqual(flag_value("--fl_local_objective"), "ce")
                self.assertEqual(flag_value("--fl_local_epochs"), "1")
                self.assertEqual(flag_value("--fl_clients_per_round"), "1.0")
                self.assertEqual(flag_value("--lambda_adv"), "0")
                self.assertEqual(flag_value("--lambda_rx_adv"), "0")
            elif run_name == "CEGRL_FEDAVG_GRL_receiver_r010":
                self.assertEqual(flag_value("--train_mode"), "fedavg")
                self.assertEqual(flag_value("--fl_local_objective"), "receiver_agnostic_bex02")
                self.assertEqual(flag_value("--fl_local_epochs"), "1")
                self.assertEqual(flag_value("--fl_clients_per_round"), "1.0")
                self.assertEqual(flag_value("--lambda_adv"), "0")
                self.assertEqual(flag_value("--lambda_rx_adv"), "1.0")

        self.assertEqual(len(seen_runs), 4)

    def test_train_help_exposes_force_ce_grl_only(self):
        text = (CODE_ROOT / "train.py").read_text(encoding="utf-8")
        self.assertIn("force_ce_grl_only", text)
        self.assertIn("def apply_force_ce_grl_only", text)


if __name__ == "__main__":
    unittest.main()
