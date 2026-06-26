import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "run_fed_fewshot_dg_6gpu.sh"
if not SCRIPT.exists():
    SCRIPT = ROOT / "scripts" / "run_fed_fewshot_dg_6gpu.sh"


class FederatedLauncherGpuAutoTest(unittest.TestCase):
    def test_launcher_auto_detects_idle_gpus_before_starting_jobs(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('AUTO_IDLE_GPUS="${AUTO_IDLE_GPUS:-1}"', text)
        self.assertIn("gpu_is_idle()", text)
        self.assertIn("refresh_free_gpus()", text)
        self.assertIn("acquire_gpu_for_job()", text)
        self.assertIn("nvidia-smi --id=", text)
        self.assertIn("--query-compute-apps=pid", text)
        self.assertIn("gpu_in_running_set", text)
        self.assertIn("ACQUIRED_GPU=", text)
        self.assertIn("acquire_gpu_for_job", text)
        self.assertIn('gpu="${ACQUIRED_GPU}"', text)

    def test_launcher_queue_contains_federated_proto_stats_experiment(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("FSDG51_fedprox_receiver_proto_stats", text)
        self.assertIn("--use_fed_proto_stats", text)
        self.assertIn("--lambda_fed_proto", text)


if __name__ == "__main__":
    unittest.main()
