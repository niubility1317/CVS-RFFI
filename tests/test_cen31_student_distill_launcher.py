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


class Cen31StudentDistillLauncherTest(unittest.TestCase):
    def _run_dry_run(self, extra_env: dict[str, str] | None = None) -> tuple[int, str]:
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo_root) as tmp:
            tmp_posix = _bash_path(tmp, bash)
            python_bin = _bash_path(sys.executable, bash)
            distill_script = _bash_path(str(repo_root / "code" / "train_cen31_distill.py"), bash)
            launcher = _bash_path(str(repo_root / "code" / "scripts" / "launch_cen31_student_distill_20260603.sh"), bash)
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

    def test_dry_run_defines_cen31_teacher_and_latency_student_matrix(self):
        returncode, output = self._run_dry_run()
        self.assertEqual(returncode, 0, output)
        self.assertIn("--teacher_ckpt", output)
        self.assertIn("cen31_best_primary.pth", output)
        self.assertIn("--epochs 300", output)
        self.assertNotIn("--epochs 90", output)
        self.assertIn("--model_variant lite_f", output)
        self.assertIn("--model_variant lite_g", output)
        self.assertIn("--model_variant lite_h", output)
        self.assertIn("--arch_family cvcnn", output)
        self.assertIn("KD_SINCCVCNN_HYBRID", output)
        self.assertIn("CEN31KD_sinc_cvcnn_hybrid_r010", output)
        self.assertIn("gpu=5", output)
        self.assertIn("--arch_family sinc_cvcnn", output)
        self.assertIn("--lambda_kd", output)
        self.assertIn("--lambda_feature_kd", output)
        self.assertIn("--latency_profile_json", output)

    def test_only_candidate_can_launch_sinc_cvcnn_without_restarting_existing_matrix(self):
        returncode, output = self._run_dry_run({"ONLY_CANDIDATE": "KD_SINCCVCNN_HYBRID"})

        self.assertEqual(returncode, 0, output)
        self.assertEqual(output.count("[CEN31-STUDENT-KD]"), 1, output)
        self.assertIn("KD_SINCCVCNN_HYBRID", output)
        self.assertIn("--arch_family sinc_cvcnn", output)
        self.assertNotIn("KD_F_BALANCED", output)


if __name__ == "__main__":
    unittest.main()
