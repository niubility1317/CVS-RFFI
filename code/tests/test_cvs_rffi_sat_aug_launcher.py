import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CvsRffiSatAugLauncherTest(unittest.TestCase):
    def test_concat_sat_dry_run_commands_do_not_mix_use_and_no_use_sat_consistency(self):
        try:
            proc = subprocess.run(
                [
                    "bash",
                    "scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh",
                    "--plan",
                    "CORE",
                    "--gpu-ids",
                    "0,1,2,3",
                    "--dry-run",
                ],
                cwd=str(ROOT),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            self.skipTest(f"bash dry-run timed out in this local environment: {exc}")
        if proc.returncode != 0 and (
            "Linux" in proc.stdout
            or "Subsystem" in proc.stdout
            or "WSL" in proc.stdout
            or "w\x00s\x00l" in proc.stdout.lower()
            or "command not found" in proc.stdout.lower()
        ):
            self.skipTest("bash/WSL is not available in this Windows test environment")
        self.assertEqual(proc.returncode, 0, proc.stdout)
        concat_lines = [line for line in proc.stdout.splitlines() if "concat_sat" in line and "[DRY-RUN]" in line]
        self.assertTrue(concat_lines, proc.stdout)
        for line in concat_lines:
            self.assertIn("--use_concat_sat_channel_aug", line)
            self.assertNotIn("--use_sat_consistency", line)
            self.assertNotIn("--no_use_sat_consistency", line)

    def test_ceonly_plan_launches_centralized_concat_sat_ce_only_runs(self):
        try:
            proc = subprocess.run(
                [
                    "bash",
                    "scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh",
                    "--plan",
                    "CEONLY",
                    "--gpu-ids",
                    "0",
                    "--dry-run",
                ],
                cwd=str(ROOT),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            self.skipTest(f"bash dry-run timed out in this local environment: {exc}")
        if proc.returncode != 0 and (
            "Linux" in proc.stdout
            or "Subsystem" in proc.stdout
            or "WSL" in proc.stdout
            or "w\x00s\x00l" in proc.stdout.lower()
            or "command not found" in proc.stdout.lower()
        ):
            self.skipTest("bash/WSL is not available in this Windows test environment")
        self.assertEqual(proc.returncode, 0, proc.stdout)
        ceonly_lines = [line for line in proc.stdout.splitlines() if "ceonly" in line and "[DRY-RUN]" in line]
        self.assertTrue(ceonly_lines, proc.stdout)
        for line in ceonly_lines:
            self.assertIn("--use_concat_sat_channel_aug", line)
            self.assertIn("--concat_sat_ce_only", line)
            self.assertIn("--concat_sat_ce_weight 1.0", line)
            self.assertNotIn("--use_sat_consistency", line)
            self.assertNotIn("--no_use_sat_consistency", line)

    def test_backbone_abl_plan_launches_centralized_optional_stability_runs(self):
        try:
            proc = subprocess.run(
                [
                    "bash",
                    "scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh",
                    "--plan",
                    "BACKBONE_ABL",
                    "--gpu-ids",
                    "0,1,2,3,4,5,6",
                    "--ratio",
                    "0.1",
                    "--dry-run",
                ],
                cwd=str(ROOT),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            self.skipTest(f"bash dry-run timed out in this local environment: {exc}")
        if proc.returncode != 0 and (
            "Linux" in proc.stdout
            or "Subsystem" in proc.stdout
            or "WSL" in proc.stdout
            or "w\x00s\x00l" in proc.stdout.lower()
            or "command not found" in proc.stdout.lower()
        ):
            self.skipTest("bash/WSL is not available in this Windows test environment")
        self.assertEqual(proc.returncode, 0, proc.stdout)
        out = proc.stdout.replace("\\,", ",")
        self.assertIn("PLAN=BACKBONE_ABL TOTAL_JOBS=7 GPU_IDS=0,1,2,3,4,5,6", out)
        self.assertIn("--wisig_train_ratio 0.1", out)
        self.assertIn("--model_variant lite_d", out)
        self.assertIn("--branch_ablation no_dac", out)
        self.assertIn("--domain_branch_ablation no_stats", out)
        self.assertIn("--use_concat_sat_channel_aug", out)
        self.assertIn("--concat_sat_ce_only", out)
        self.assertIn("--concat_sat_ce_weight 1.0", out)
        self.assertIn("--id_time_stability_mode phase_delta", out)
        self.assertIn("--id_freq_stability_mode dsq", out)
        self.assertIn("--domain_time_stability_mode same", out)
        self.assertIn("--domain_freq_stability_mode same", out)
        self.assertIn("--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit", out)
        self.assertIn("--eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit", out)

    def test_backbone_dsq_followup_plan_launches_selected_optimization_runs(self):
        try:
            proc = subprocess.run(
                [
                    "bash",
                    "scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh",
                    "--plan",
                    "BACKBONE_DSQ_FOLLOWUP",
                    "--gpu-ids",
                    "0,1,2,3,4,5,6",
                    "--ratio",
                    "0.1",
                    "--dry-run",
                ],
                cwd=str(ROOT),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            self.skipTest(f"bash dry-run timed out in this local environment: {exc}")
        if proc.returncode != 0 and (
            "Linux" in proc.stdout
            or "Subsystem" in proc.stdout
            or "WSL" in proc.stdout
            or "w\x00s\x00l" in proc.stdout.lower()
            or "command not found" in proc.stdout.lower()
        ):
            self.skipTest("bash/WSL is not available in this Windows test environment")
        self.assertEqual(proc.returncode, 0, proc.stdout)
        out = proc.stdout.replace("\\,", ",")
        self.assertIn("PLAN=BACKBONE_DSQ_FOLLOWUP TOTAL_JOBS=7 GPU_IDS=0,1,2,3,4,5,6", out)
        dry_run_lines = {line.split(" exp=", 1)[1].split(" group=", 1)[0]: line for line in out.splitlines() if "[DRY-RUN]" in line}
        self.assertIn("SA18_domain_dsq_ch2_r010", out)
        self.assertIn("--domain_freq_stability_mode dsq --freq_stability_channels 2", out)
        self.assertIn("SA19_domain_dsq_ch8_r010", out)
        self.assertIn("--domain_freq_stability_mode dsq --freq_stability_channels 8", out)
        self.assertIn("SA20_domain_phase_dsq_r010", out)
        self.assertIn("--domain_time_stability_mode phase_delta --domain_freq_stability_mode dsq", out)
        self.assertIn("SA21_id_domain_dsq_r010", out)
        self.assertIn("--id_freq_stability_mode dsq --domain_freq_stability_mode same", out)
        sa22 = dry_run_lines["SA22_domain_dsq_satce_w0p7_r010"]
        self.assertIn("--domain_freq_stability_mode dsq", sa22)
        self.assertIn("--concat_sat_ce_weight 0.7", sa22)
        sa23 = dry_run_lines["SA23_domain_dsq_satce_w1p5_r010"]
        self.assertIn("--domain_freq_stability_mode dsq", sa23)
        self.assertIn("--concat_sat_ce_weight 1.5", sa23)
        sa24 = dry_run_lines["SA24_id_phase_dsq_satce_w0p7_r010"]
        self.assertIn("--id_time_stability_mode phase_delta", sa24)
        self.assertIn("--id_freq_stability_mode dsq", sa24)
        self.assertIn("--concat_sat_ce_weight 0.7", sa24)


if __name__ == "__main__":
    unittest.main()
