import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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


class CenA31ArchCompareLauncherTest(unittest.TestCase):
    def test_dry_run_keeps_cen_a31_strategy_and_varies_only_arch_family(self):
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo_root) as tmp:
            tmp_posix = _bash_path(tmp, bash)
            python_bin = _bash_path(sys.executable, bash)
            train_script = _bash_path(str(repo_root / "code" / "train.py"), bash)
            launcher = _bash_path(str(repo_root / "code" / "scripts" / "launch_cen_a31_arch_compare_20260603.sh"), bash)
            command = (
                f'ROOT="{_bash_path(str(repo_root), bash)}" '
                f'PYTHON="{python_bin}" '
                f'CVS_TRAIN_SCRIPT="{train_script}" '
                f'LOG_ROOT="{tmp_posix}/logs" '
                f'RUNS_ROOT="{tmp_posix}/runs" '
                f'bash "{launcher}" --dry-run'
            )

            proc = subprocess.run(
                [bash, "-lc", command],
                cwd=repo_root,
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
        self.assertIn("--wisig_train_ratio 0.1", output)
        self.assertIn("--epochs 170", output)
        self.assertIn("--use_concat_sat_channel_aug", output)
        self.assertIn("--concat_sat_ce_weight 1.28", output)
        self.assertIn("--lambda_proto 0.015", output)
        self.assertIn("--lambda_supcon_id 0.02", output)
        self.assertIn("--lambda_fishr 0.005", output)
        self.assertIn("--arch_family cvsincnet", output)
        self.assertIn("--arch_family resnet18_1d", output)
        self.assertIn("--arch_family cvcnn", output)
        self.assertIn("--arch_family sinc_cvcnn", output)


if __name__ == "__main__":
    unittest.main()
