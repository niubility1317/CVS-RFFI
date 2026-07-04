import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from test_phase2_proxy_adapter_ci_eval import _write_tiny_npz


class Phase2OldFloorHnfrAdapterCiEvalTest(unittest.TestCase):
    def test_old_floor_margin_loss_only_penalizes_low_margin(self):
        from phase2_old_floor_hnfr_adapter_ci_eval import old_floor_margin_loss

        labels = torch.tensor([0, 1], dtype=torch.long)
        safe_logits = torch.tensor([[4.0, 1.0], [0.5, 3.0]], dtype=torch.float32)
        risky_logits = torch.tensor([[1.5, 1.2], [1.0, 1.4]], dtype=torch.float32)

        self.assertEqual(float(old_floor_margin_loss(safe_logits, labels, margin=1.0).item()), 0.0)
        self.assertGreater(float(old_floor_margin_loss(risky_logits, labels, margin=1.0).item()), 0.0)

    def test_train_old_floor_adapter_keeps_unknown_eval_only_metadata(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz
        from phase2_old_floor_hnfr_adapter_ci_eval import parse_args, train_old_floor_adapter
        from phase2_proxy_adapter_ci_eval import build_training_plan

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npz"
            _write_tiny_npz(path)
            payload = load_feature_npz(path)
            plan = build_training_plan(
                payload,
                k_shot=2,
                query_per_class=2,
                seed=3,
                support_selection_policy="stable_first",
            )
            args = parse_args(
                [
                    "--feature_npz",
                    str(path),
                    "--output_dir",
                    str(Path(tmp) / "out"),
                    "--backend",
                    "enpc",
                    "--device",
                    "cpu",
                    "--adapter_epochs",
                    "1",
                    "--adapter_rank",
                    "2",
                    "--batch_size",
                    "4",
                    "--k_shot",
                    "2",
                    "--query_per_class",
                    "2",
                ]
            )
            _adapter, metrics = train_old_floor_adapter(payload, plan, args)

        self.assertEqual(metrics["training_counts"]["target_unknown_training_count"], 0)
        self.assertEqual(metrics["training_counts"]["target_unknown_eval_only"], 6)
        self.assertIn("target_old_support_proto_acc_after", metrics)
        self.assertIn("old_floor_weight", metrics["loss_weights"])


if __name__ == "__main__":
    unittest.main()
