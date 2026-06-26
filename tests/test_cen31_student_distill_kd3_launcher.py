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


class Cen31StudentDistillKd3LauncherTest(unittest.TestCase):
    def _run_dry_run(self, extra_env: dict[str, str] | None = None) -> tuple[int, str]:
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo_root) as tmp:
            tmp_posix = _bash_path(tmp, bash)
            python_bin = _bash_path(sys.executable, bash)
            distill_script = _bash_path(str(repo_root / "code" / "train_cen31_distill.py"), bash)
            launcher = _bash_path(str(repo_root / "code" / "scripts" / "launch_cen31_student_distill_kd3_balanced_20260603.sh"), bash)
            command = (
                f'ROOT="{_bash_path(str(repo_root), bash)}" '
                f'PYTHON="{python_bin}" '
                f'DISTILL_SCRIPT="{distill_script}" '
                f'TEACHER_CKPT="{tmp_posix}/teacher/cen31_best_primary.pth" '
                f'LOG_ROOT="{tmp_posix}/logs" '
                f'RUNS_ROOT="{tmp_posix}/runs" '
                f'bash "{launcher}" --dry-run'
            )

            env = os.environ.copy()
            if extra_env:
                env.update(extra_env)
            proc = subprocess.run(
                [bash, "-lc", command],
                cwd=repo_root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

        return proc.returncode, proc.stdout + proc.stderr

    def test_kd3_dry_run_enables_balanced_clean_sat_receiver_selection(self):
        returncode, output = self._run_dry_run()

        self.assertEqual(returncode, 0, output)
        self.assertEqual(output.count("[CEN31-STUDENT-KD3]"), 3, output)
        self.assertIn("--wisig_train_ratio 0.1", output)
        self.assertIn("--epochs 300", output)
        self.assertIn("--best_select_metric clean_sat_joint", output)
        self.assertIn("--best_balanced_save_path", output)
        self.assertIn("--lambda_group_ce 0.06", output)
        self.assertIn("--group_ce_mode smooth_dro_capped", output)
        self.assertIn("--lambda_sat_view_group_ce 0.03", output)
        self.assertIn("--sat_view_loss_ramp_epochs 60", output)
        self.assertIn("--sat_select_eval_interval 20", output)
        self.assertIn("--sat_select_max_batches 5", output)
        self.assertIn("--eval_sat_channel", output)
        self.assertIn("--eval_sat_on main", output)
        self.assertIn("CEN31KD3_lite_f_clean_anchor_satlight_r010", output)
        self.assertIn("CEN31KD3_lite_f_rx8_clean_strict_r010", output)
        self.assertIn("CEN31KD3_cvcnn_rx8_satlight_r010", output)
        self.assertNotIn("--lambda_sat_view_ce 1.00", output)
        self.assertNotIn("--lambda_sat_view_kd 0.45", output)

    def test_only_candidate_filters_kd3_matrix(self):
        returncode, output = self._run_dry_run({"ONLY_CANDIDATE": "KD3_F_RX8_CLEAN_STRICT"})

        self.assertEqual(returncode, 0, output)
        self.assertEqual(output.count("[CEN31-STUDENT-KD3]"), 1, output)
        self.assertIn("KD3_F_RX8_CLEAN_STRICT", output)
        self.assertIn("--no_use_sat_view_kd", output)
        self.assertNotIn("KD3_F_CLEAN_ANCHOR_SATLIGHT", output)


if __name__ == "__main__":
    unittest.main()
