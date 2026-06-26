import re
import sys
import unittest
from pathlib import Path

import torch


class CEN31SincSharedOptimizerTest(unittest.TestCase):
    def test_rcn_minimal_6stats_encoder_uses_six_inputs(self):
        from model_dual_cvsincnet import RCNStatEncoder

        encoder = RCNStatEncoder(out_dim=12, hidden=8, stat_mode="minimal6").eval()
        x = torch.randn(3, 2, 64)
        with torch.no_grad():
            stats = encoder._iq_stats(x)
            out = encoder(x)

        self.assertEqual(encoder.stat_dim, 6)
        self.assertEqual(stats.shape, (3, 6))
        self.assertEqual(out.shape, (3, 12))

    def test_launcher_declares_exactly_16_unique_centralized_candidates(self):
        code_dir = Path(__file__).resolve().parents[1]
        script = code_dir / "scripts" / "launch_cen31_sinc_shared_optimizer_20260605.sh"
        text = script.read_text(encoding="utf-8")
        entries = re.findall(r'^\s+"(\d+\|\d+\|([^|]+)\|[^"]*)"', text, flags=re.MULTILINE)
        names = [name for _, name in entries]

        self.assertEqual(len(names), 16)
        self.assertEqual(len(set(names)), 16)
        self.assertIn("sinc_shared_baseline_api", names)
        self.assertIn("rcn_minimal_6stats", names)
        self.assertIn("--train_mode centralized", text)
        self.assertIn("--test_eval_policy interval_final", text)
        self.assertIn("--test_eval_interval", text)
        self.assertIn("forbidden_every_epoch=true", text)
        self.assertIn("forbidden_val_improved_extra=true", text)
        self.assertNotIn("fedavg", text.lower())
        self.assertNotIn("fedcvs_vmb", text.lower())

    def test_profiler_accepts_sinc_shared_candidate_switches(self):
        root = Path(__file__).resolve().parents[2]
        tools_dir = root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        from profile_cen_a31_architectures import profile_architecture

        row = profile_architecture(
            "cvsincnet",
            batch_size=1,
            input_len=64,
            num_classes=8,
            num_domains=3,
            device=torch.device("cpu"),
            warmup=0,
            iters=1,
            freq_feature_source="sinc_energy",
            pa_feature_source="sinc_lowrank",
            pa_orders=(1, 5),
            use_aux_spectral_stats=False,
            channel_trim_scale=0.75,
        )

        self.assertEqual(row["freq_feature_source"], "sinc_energy")
        self.assertEqual(row["pa_feature_source"], "sinc_lowrank")
        self.assertEqual(row["pa_orders"], "1,5")
        self.assertEqual(row["use_aux_spectral_stats"], 0)
        self.assertGreater(row["module_forwards_deploy"], 0)

    def test_candidate_profiler_declares_same_16_candidate_matrix(self):
        root = Path(__file__).resolve().parents[2]
        tools_dir = root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        from profile_cen31_sinc_shared_candidates import CANDIDATES

        names = [str(item["name"]) for item in CANDIDATES]
        self.assertEqual(len(names), 16)
        self.assertEqual(len(set(names)), 16)
        self.assertEqual(names[0], "sinc_shared_baseline_api")
        self.assertEqual(names[-1], "rcn_minimal_6stats")


if __name__ == "__main__":
    unittest.main()
