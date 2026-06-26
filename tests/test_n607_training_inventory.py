import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import n607_training_inventory as inv  # noqa: E402


REMOTE_ROOT = "/home/szu2070436088/2510044040/CV-SincNet"


def proc(pid, ppid, cmdline, cwd=REMOTE_ROOT, exe="/usr/bin/python", environ=None):
    return {
        "pid": str(pid),
        "ppid": str(ppid),
        "cwd": cwd,
        "exe": exe,
        "cmdline": cmdline,
        "environ": environ or [],
    }


def snapshot(processes, gpu_pids):
    return {
        "collected_at": "2026-06-03T22:00:00+0800",
        "host": "dell-DSS8440",
        "gpu_compute": [
            {
                "pid": str(pid),
                "gpu_uuid": f"GPU-{idx}",
                "process_name": "python",
                "used_memory_mib": "1024",
            }
            for idx, pid in enumerate(gpu_pids)
        ],
        "processes": processes,
    }


class N607TrainingInventoryTests(unittest.TestCase):
    def test_centralized_train_py_is_active(self):
        data = inv.classify_snapshot(
            snapshot(
                [
                    proc(
                        101,
                        1,
                        ["python", "-u", "train.py", "--train_mode", "centralized", "--run_name", "CEN31"],
                    )
                ],
                [101],
            )
        )

        self.assertTrue(data["centralized_active"])
        self.assertFalse(data["federated_vmb_active"])
        self.assertEqual(data["active_training_processes"][0]["lane"], "centralized")

    def test_fedprox_train_mode_is_federated_even_when_entry_is_train_py(self):
        data = inv.classify_snapshot(
            snapshot(
                [
                    proc(
                        202,
                        1,
                        [
                            "python",
                            "-u",
                            "train.py",
                            "--train_mode",
                            "fedprox",
                            "--fl_local_objective",
                            "receiver_agnostic_bex02",
                        ],
                    )
                ],
                [202],
            )
        )

        self.assertFalse(data["centralized_active"])
        self.assertTrue(data["federated_vmb_active"])
        self.assertEqual(data["active_training_processes"][0]["lane"], "federated_vmb")

    def test_cen31_distill_script_counts_as_centralized_training(self):
        data = inv.classify_snapshot(
            snapshot(
                [
                    proc(
                        303,
                        1,
                        [
                            "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python",
                            "-u",
                            f"{REMOTE_ROOT}/code/train_cen31_distill.py",
                            "--run_name",
                            "CEN31KD_lite_f_physlite_r010",
                            "--output_dir",
                            f"{REMOTE_ROOT}/runs/cen31_student/CEN31KD_lite_f_physlite_r010",
                        ],
                    )
                ],
                [303],
            )
        )

        self.assertTrue(data["centralized_active"])
        self.assertEqual(data["active_training_processes"][0]["lane"], "centralized")

    def test_module_style_sgc_training_under_bash_lc_is_detected(self):
        data = inv.classify_snapshot(
            snapshot(
                [
                    proc(
                        400,
                        1,
                        ["bash", "-lc", "CUDA_VISIBLE_DEVICES=0 python -u -m SGC.train_recon_sgc_joint --run_name J1"],
                        exe="/bin/bash",
                    ),
                    proc(
                        401,
                        400,
                        ["python", "-u", "-m", "SGC.train_recon_sgc_joint", "--run_name", "J1", "--output_dir", f"{REMOTE_ROOT}/runs/J1"],
                    ),
                ],
                [401],
            )
        )

        active_pids = {row["pid"] for row in data["active_training_processes"]}
        self.assertIn("401", active_pids)
        self.assertTrue(data["centralized_active"])
        self.assertTrue(data["launcher_context"])

    def test_unknown_project_gpu_training_blocks_both_lanes(self):
        data = inv.classify_snapshot(
            snapshot(
                [
                    proc(
                        505,
                        1,
                        [
                            "python",
                            "-u",
                            f"{REMOTE_ROOT}/code/custom_experiment.py",
                            "--run_name",
                            "CUSTOM",
                            "--output_dir",
                            f"{REMOTE_ROOT}/runs/custom",
                        ],
                    )
                ],
                [505],
            )
        )

        self.assertTrue(data["unknown_training_active"])
        self.assertTrue(data["centralized_active"])
        self.assertTrue(data["federated_vmb_active"])
        self.assertEqual(data["monitor_state"], 0)

    def test_grep_noise_is_not_training(self):
        data = inv.classify_snapshot(
            snapshot(
                [
                    proc(
                        606,
                        1,
                        ["grep", "train.py"],
                        cwd="/tmp",
                        exe="/bin/grep",
                    )
                ],
                [],
            )
        )

        self.assertFalse(data["centralized_active"])
        self.assertFalse(data["federated_vmb_active"])
        self.assertEqual(data["active_training_processes"], [])


if __name__ == "__main__":
    unittest.main()
