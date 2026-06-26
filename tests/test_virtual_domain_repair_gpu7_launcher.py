import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "launch_virtual_domain_repair_gpu7_20260602.sh"


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


class VirtualDomainRepairGpu7LauncherTest(unittest.TestCase):
    def _dry_run_output(self) -> str:
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tmp_posix = _bash_path(tmp, bash)
            command = (
                f'ROOT="{tmp_posix}" '
                f'PYTHON="{_bash_path(sys.executable, bash)}" '
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

    def test_renders_two_gpu7_virtual_domain_commands(self):
        output = self._dry_run_output()
        lines = [ln for ln in output.splitlines() if ln.startswith("[GPU7-VDOM-CMD]")]
        self.assertEqual(len(lines), 2, output)
        self.assertIn("FSDG49B_GPU7_R15_style_tbal_ultrasoft_p005_R80_r010", output)
        self.assertIn("FSDG49B_GPU7_R16_style_realmix16_upper_R80_r010", output)

    def test_formal_contract_and_fsdg49b_core_are_preserved(self):
        output = self._dry_run_output()
        commands = [ln.split(" ", 3)[3] for ln in output.splitlines() if ln.startswith("[GPU7-VDOM-CMD]")]
        for command in commands:
            tokens = shlex.split(command)

            def value(flag: str) -> str:
                self.assertIn(flag, tokens, command)
                self.assertEqual(tokens.count(flag), 1, command)
                return tokens[tokens.index(flag) + 1]

            self.assertEqual(value("--wisig_train_ratio"), "0.1")
            self.assertEqual(value("--epochs"), "200")
            self.assertEqual(value("--fl_rounds"), "200")
            self.assertEqual(value("--fl_client_key"), "receiver")
            self.assertEqual(value("--train_mode"), "fedprox")
            self.assertEqual(value("--fl_local_objective"), "receiver_agnostic_bex02")
            self.assertEqual(value("--fl_style_domain_label_mode"), "target_receiver")
            self.assertEqual(value("--fl_style_sampling_policy"), "target_balanced")
            self.assertIn("--fl_style_zdom_probe_force_batch", tokens)
            self.assertIn("--use_style_collab_eval", tokens)

    def test_ultrasoft_candidate_does_not_train_on_virtual_dg(self):
        output = self._dry_run_output()
        cmd = next(line for line in output.splitlines() if "FSDG49B_GPU7_R15_style_tbal_ultrasoft_p005_R80_r010" in line)
        tokens = shlex.split(cmd.split(" ", 3)[3])
        self.assertEqual(tokens[tokens.index("--fl_style_replay_prob") + 1], "0.05")
        self.assertEqual(tokens[tokens.index("--fl_style_transform_mix_alpha") + 1], "0.10")
        self.assertEqual(tokens[tokens.index("--fl_style_dg_start_round") + 1], "999")
        self.assertNotIn("--fl_style_real_mix_samples", tokens)

    def test_realmix_upper_bound_is_marked_by_real_mix_and_late_dg(self):
        output = self._dry_run_output()
        cmd = next(line for line in output.splitlines() if "FSDG49B_GPU7_R16_style_realmix16_upper_R80_r010" in line)
        tokens = shlex.split(cmd.split(" ", 3)[3])
        self.assertEqual(tokens[tokens.index("--fl_style_real_mix_samples") + 1], "16")
        self.assertEqual(tokens[tokens.index("--fl_style_real_mix_start_round") + 1], "80")
        self.assertEqual(tokens[tokens.index("--fl_style_dg_start_round") + 1], "120")
        self.assertEqual(tokens[tokens.index("--fl_style_replay_prob") + 1], "0.25")


if __name__ == "__main__":
    unittest.main()
