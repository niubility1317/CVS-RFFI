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


class CvsRffiLauncherTest(unittest.TestCase):
    def test_wisig_baseline_queue_defaults_use_riei_drift_ratio010(self):
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo_root) as tmp:
            tmp_posix = _bash_path(tmp, bash)
            python_bin = _bash_path(sys.executable, bash)
            command = (
                f'LOG_ROOT="{tmp_posix}/logs" '
                f'RUN_ROOT="{tmp_posix}/runs" '
                f'bash run_cvs_baseline_queue.sh --python "{python_bin}" --dry-run'
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
        self.assertIn("methods=riei_fd,drift", output)
        self.assertIn("train_ratio=0.1", output)
        self.assertIn("sat_eval=1", output)
        self.assertIn("sat_view_aug=0", output)
        self.assertIn("baselines.riei_fd.train", output)
        self.assertIn("baselines.drift.train", output)
        self.assertIn("--epochs 200", output)
        self.assertNotIn("baselines.ra_collab.train", output)

    def test_wisig_baseline_queue_accepts_canonical_method_names(self):
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo_root) as tmp:
            tmp_posix = _bash_path(tmp, bash)
            python_bin = _bash_path(sys.executable, bash)
            command = (
                f'LOG_ROOT="{tmp_posix}/logs" '
                f'RUN_ROOT="{tmp_posix}/runs" '
                f'bash run_cvs_baseline_queue.sh --python "{python_bin}" '
                "--methods cvcnn_ce,riei_fd,ra_collab --dry-run"
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
        self.assertIn("cvcnn_ce_seed1337", output)
        self.assertIn("riei_fd_seed1337", output)
        self.assertIn("ra_collab_seed1337", output)

    def test_r010_cvsrffi_riei_drift_queue_exposes_split_matrix_and_ablations(self):
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo_root) as tmp:
            tmp_posix = _bash_path(tmp, bash)
            python_bin = _bash_path(sys.executable, bash)
            command = (
                f'LOG_ROOT="{tmp_posix}/logs" '
                f'RUN_ROOT="{tmp_posix}/runs" '
                f'bash run_cvsrffi_riei_drift_r010_queue.sh --python "{python_bin}" '
                "--dry-run --splits rx3_d0,rx7_d012 "
                "--methods riei_fd,drfit,cvs_full,cvs_no_satboost,cvs_no_mixstyle,"
                "cvs_no_fishr,cvs_no_group_ce,cvs_no_proto_supcon,cvs_no_dsq --gpu-ids 0,1"
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
        self.assertIn("protocol=cvs_day_rx train_ratio=0.1", output)
        self.assertIn("split_strategy=random cap_strategy=random", output)
        self.assertIn("--wisig_split_strategy random", output)
        self.assertIn("--wisig_cap_strategy random", output)
        self.assertIn("gpu_refill_queue=1 max_train_per_gpu=2 final_only_test=1", output)
        self.assertIn("[ALIAS] drfit -> drift", output)
        self.assertIn("train_days=0 test_days=2,3 train_rxs=0,3,6", output)
        self.assertIn("train_days=0,1,2 test_days=3 train_rxs=0,1,2,3,4,5,6", output)
        self.assertIn("baselines.riei_fd.train", output)
        self.assertIn("baselines.drift.train", output)
        self.assertIn("--no_test_on_val_improve", output)
        self.assertIn("--paper_eval_last_n 0", output)
        self.assertIn("--wisig_protocol cvs_day_rx", output)
        self.assertIn("--wisig_train_ratio 0.1", output)
        self.assertNotIn("--wisig_train_ratio 0.2", output)
        self.assertIn("--test_eval_policy val_improved_final", output)
        self.assertIn("--test_eval_start_epoch 999999", output)
        self.assertNotIn("--test_eval_policy every_epoch", output)
        self.assertIn("CEN_ABL_FULL_a31_stack_rx3d0_r010", output)
        self.assertIn("CEN_ABL_NO_SATBOOST_a31_stack_rx3d0_r010", output)
        self.assertIn("--no_use_mixstyle", output)
        self.assertIn("--lambda_fishr 0.00", output)
        self.assertIn("--lambda_group_ce 0.00", output)
        self.assertIn("--no_use_proto_memory --lambda_proto 0.00 --lambda_supcon_id 0.00", output)
        self.assertIn("--domain_enhancer off --domain_enhancer_strength 0.0 --domain_freq_stability_mode off", output)

    def test_r010_comparison_plus_ablation_limits_ablations_to_selected_splits(self):
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo_root) as tmp:
            tmp_posix = _bash_path(tmp, bash)
            python_bin = _bash_path(sys.executable, bash)
            command = (
                f'LOG_ROOT="{tmp_posix}/logs" '
                f'RUN_ROOT="{tmp_posix}/runs" '
                f'bash run_cvsrffi_riei_drift_r010_queue.sh --python "{python_bin}" '
                "--dry-run --plan comparison_plus_ablation --splits rx3_d0,rx7_d01 "
                "--ablation-splits rx7_d01 --gpu-ids 0,1"
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
        self.assertIn("plan=comparison_plus_ablation", output)
        self.assertIn("methods=riei_fd_sat,drift_sat,cvs_full", output)
        self.assertIn("ablation_splits=rx7_d01", output)
        self.assertIn("riei_fd_sat5_rx3d0_r010", output)
        self.assertIn("drift_sat5_rx3d0_r010", output)
        self.assertIn("CEN_ABL_FULL_a31_stack_rx3d0_r010", output)
        self.assertIn("skip ablation outside ABLATION_SPLITS=rx7_d01", output)
        self.assertNotIn("CEN_ABL_NO_SATBOOST_a31_stack_rx3d0_r010", output)
        self.assertIn("CEN_ABL_NO_SATBOOST_a31_stack_rx7d01_r010", output)
        self.assertIn("--use_sat_channel_view_aug", output)

    def test_r010_sat_comparison_enables_training_sat_view_for_baselines(self):
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo_root) as tmp:
            tmp_posix = _bash_path(tmp, bash)
            python_bin = _bash_path(sys.executable, bash)
            command = (
                f'LOG_ROOT="{tmp_posix}/logs" '
                f'RUN_ROOT="{tmp_posix}/runs" '
                f'bash run_cvsrffi_riei_drift_r010_queue.sh --python "{python_bin}" '
                "--dry-run --plan sat_comparison --splits rx3_d0,rx7_d01 --gpu-ids 0,1"
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
        self.assertIn("plan=sat_comparison", output)
        self.assertIn("methods=riei_fd_sat,drift_sat", output)
        self.assertIn("baseline_sat_view scenarios=clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit", output)
        self.assertIn("riei_fd_sat5_rx3d0_r010_seed1337", output)
        self.assertIn("drift_sat5_rx7d01_r010_seed1337", output)
        self.assertIn("--use_sat_channel_view_aug", output)
        self.assertIn("--sat_train_scenarios clear_leo\\,low_elev_leo\\,rain_leo\\,storm_mp\\,mixed_orbit", output)
        self.assertIn("--sat_view_prob 1.00", output)
        self.assertIn("--no_test_on_val_improve", output)
        self.assertIn("--paper_eval_last_n 0", output)
        self.assertNotIn("CEN_ABL_FULL", output)

    def test_r010_comparison_defaults_to_sat_and_dynamic_receiver_splits(self):
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo_root) as tmp:
            tmp_posix = _bash_path(tmp, bash)
            python_bin = _bash_path(sys.executable, bash)
            command = (
                f'LOG_ROOT="{tmp_posix}/logs" '
                f'RUN_ROOT="{tmp_posix}/runs" '
                f'bash run_cvsrffi_riei_drift_r010_queue.sh --python "{python_bin}" '
                "--dry-run --plan comparison "
                "--splits rx2lo_d0,rx2sp_d01,rx4hi_d012,rx6sp_d0 --gpu-ids 0,1,2"
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
        self.assertIn("methods=riei_fd_sat,drift_sat,cvs_full", output)
        self.assertIn("train_days=0 test_days=2,3 train_rxs=0,1", output)
        self.assertIn("train_days=0,1 test_days=2,3 train_rxs=0,6", output)
        self.assertIn("train_days=0,1,2 test_days=3 train_rxs=3,4,5,6", output)
        self.assertIn("train_days=0 test_days=2,3 train_rxs=0,1,2,4,5,6", output)
        self.assertIn("riei_fd_sat5_rx2lod0_r010_seed1337", output)
        self.assertIn("drift_sat5_rx2spd01_r010_seed1337", output)
        self.assertIn("CEN_ABL_FULL_a31_stack_rx4hid012_r010_seed1337", output)
        self.assertIn("--use_sat_channel_view_aug", output)
        self.assertIn("--sat_train_scenarios clear_leo\\,low_elev_leo\\,rain_leo\\,storm_mp\\,mixed_orbit", output)

    def test_r010_refill_queue_launches_gpu_queue_processes(self):
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo_root) as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "fake.pkl").write_bytes(b"fake")
            tmp_posix = _bash_path(tmp, bash)
            command = (
                f'LOG_ROOT="{tmp_posix}/logs" '
                f'RUN_ROOT="{tmp_posix}/runs" '
                f'WISIG_PKL="{tmp_posix}/fake.pkl" '
                "PYTHON_BIN=true "
                "CVS_TRAIN_SCRIPT=run_cvsrffi_riei_drift_r010_queue.sh "
                "bash run_cvsrffi_riei_drift_r010_queue.sh "
                "--plan comparison --splits rx3_d0 --methods riei_fd,drift,cvs_full "
                "--gpu-ids 0 --no-skip-done; sleep 1"
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
            launch_pids = tmp_path / "logs" / "launch_pids.tsv"
            launch_text = launch_pids.read_text(encoding="utf-8") if launch_pids.exists() else ""
            queue_logs = list((tmp_path / "logs").glob("gpu_0_queue_*.log"))
            queue_text = queue_logs[0].read_text(encoding="utf-8") if queue_logs else ""

        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, output)
        self.assertIn("refill_queue_pid=", output)
        self.assertIn("queue\tgpu_0\t", launch_text)
        self.assertIn("initial_external_count=", queue_text)

    def test_r010_jobs_argument_schedules_exact_split_method_pairs(self):
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo_root) as tmp:
            tmp_posix = _bash_path(tmp, bash)
            python_bin = _bash_path(sys.executable, bash)
            command = (
                f'LOG_ROOT="{tmp_posix}/logs" '
                f'RUN_ROOT="{tmp_posix}/runs" '
                f'bash run_cvsrffi_riei_drift_r010_queue.sh --python "{python_bin}" '
                "--dry-run --jobs rx7_d01:riei_fd,rx7_d012:cvs_full --gpu-ids 0,1"
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
        self.assertIn("jobs=rx7_d01:riei_fd,rx7_d012:cvs_full", output)
        self.assertIn("riei_fd_rx7d01_r010_seed1337", output)
        self.assertIn("CEN_ABL_FULL_a31_stack_rx7d012_r010_seed1337", output)
        self.assertNotIn("drift_rx7d01_r010_seed1337", output)

    def test_bex02_receiver_curriculum_dry_run_matches_baseline_smoke(self):
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo_root) as tmp:
            tmp_posix = _bash_path(tmp, bash)
            python_bin = _bash_path(sys.executable, bash)
            command = (
                f'LOG_ROOT="{tmp_posix}/logs" '
                f'RUN_ROOT="{tmp_posix}/runs" '
                "bash code/scripts/run_cvs_rffi_rx_curriculum_bex02_6gpu.sh "
                f'--plan SMOKE --gpu-ids 0 --python "{python_bin}" --dry-run'
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
        self.assertIn("CVS-RFFI BEX02 receiver-curriculum launcher", output)
        self.assertIn("PLAN=SMOKE TOTAL_JOBS=2", output)
        self.assertIn("BEX02_fishr002_mixed_e170_plain_T1_P01_K2_train-0-10_test-rest", output)
        self.assertIn("--wisig_train_days 0\\,1", output)
        self.assertIn("--wisig_test_days 2\\,3", output)
        self.assertIn("--wisig_train_rxs 0\\,10", output)
        self.assertIn("--wisig_test_rxs 1\\,2\\,3\\,4\\,5\\,6\\,7\\,8\\,9\\,11", output)
        self.assertIn("--epochs 170", output)
        self.assertIn("--test_eval_policy val_improved_final", output)
        self.assertIn("--sat_train_scenario mixed_orbit", output)
        self.assertIn("--lambda_fishr 0.02", output)
        self.assertIn("--fishr_min_domains 4", output)

    def test_post_mode_missing_trainers_exits_nonzero(self):
        bash = _find_bash()
        if bash is None:
            self.skipTest("bash is required for launcher tests")

        repo_root = Path(__file__).resolve().parents[1]
        staged_launcher = repo_root / "scripts" / "run_cvs_rffi_staged_8gpu.sh"
        if not staged_launcher.exists():
            self.skipTest(f"staged launcher is not present: {staged_launcher}")

        with tempfile.TemporaryDirectory(dir=repo_root) as tmp:
            tmp_posix = _bash_path(tmp, bash)
            command = (
                f'LOG_ROOT="{tmp_posix}/logs" '
                f'RUN_ROOT="{tmp_posix}/runs" '
                "PROTO_TRAINER=missing_train_fjmp.py "
                "SSDG_TRAINER=missing_train_ssdg.py "
                "bash scripts/run_cvs_rffi_staged_8gpu.sh "
                "--mode post --run-post --dry-run"
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

        self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("[MISSING] required trainer not found: missing_train_fjmp.py", proc.stdout)
        self.assertIn("Selected queues finished with one or more launcher-detected failures.", proc.stdout)


if __name__ == "__main__":
    unittest.main()
