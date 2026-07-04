import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _write_npz(path: Path) -> None:
    rng = np.random.default_rng(31)
    rows = []
    centers = {
        "old-a": np.array([1, 0, 0, 0], dtype=np.float32),
        "new-a": np.array([0, 1, 0, 0], dtype=np.float32),
        "proxy-a": np.array([0, 0, 1, 0], dtype=np.float32),
        "unk-a": np.array([0, 0, 0, 1], dtype=np.float32),
    }
    spec = [
        ("source", "old-a", ["src-rx"], 10),
        ("proxy_unknown", "proxy-a", ["src-rx"], 10),
        ("target_old", "old-a", ["rx-a", "rx-b"], 8),
        ("target_new", "new-a", ["rx-a", "rx-b"], 8),
        ("target_unknown", "unk-a", ["rx-a", "rx-b"], 8),
    ]
    for role, tx, receivers, count in spec:
        for rx in receivers:
            for i in range(count):
                vec = centers[tx] + rng.normal(0, 0.01, size=4).astype(np.float32)
                vec = vec / max(float(np.linalg.norm(vec)), 1e-8)
                rows.append((role, tx, rx, f"{role}-{rx}-{i}", vec))
    np.savez_compressed(
        path,
        features=np.asarray([row[4] for row in rows], dtype=np.float32),
        tx_ids=np.asarray([row[1] for row in rows]),
        rx_ids=np.asarray([row[2] for row in rows]),
        day_ids=np.asarray(["d0"] * len(rows)),
        sig_ids=np.asarray([row[3] for row in rows]),
        dataset_role=np.asarray([row[0] for row in rows]),
        channel_views=np.asarray(["leo"] * len(rows)),
        sat_scenarios=np.asarray(["leo_clear_weak"] * len(rows)),
        raw_labels=np.asarray([row[1] for row in rows]),
        domain_labels=np.asarray([row[2] for row in rows]),
    )


class Phase2VirtualRadiusAdapterCiEvalTest(unittest.TestCase):
    def test_vra_training_keeps_target_unknown_eval_only(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz
        from phase2_proxy_adapter_ci_eval import build_training_plan
        from phase2_virtual_radius_adapter_ci_eval import parse_args, train_vra_adapter

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npz"
            _write_npz(path)
            payload = load_feature_npz(path)
            plan = build_training_plan(payload, k_shot=2, query_per_class=2, seed=7, support_selection_policy="stable_first")
            args = parse_args(
                [
                    "--feature_npz",
                    str(path),
                    "--output_dir",
                    str(Path(tmp) / "out"),
                    "--device",
                    "cpu",
                    "--adapter_epochs",
                    "2",
                    "--adapter_rank",
                    "2",
                    "--batch_size",
                    "4",
                    "--virtual_count",
                    "4",
                    "--k_shot",
                    "2",
                    "--query_per_class",
                    "2",
                ]
            )
            _adapter, metrics = train_vra_adapter(payload, plan, args)

        self.assertEqual(metrics["training_counts"]["target_unknown_training_count"], 0)
        self.assertEqual(metrics["training_counts"]["target_unknown_eval_only"], 16)
        self.assertGreater(metrics["state_bytes"]["adapter_fp16_bytes"], 0)
        self.assertIn("support_proto_acc_after", metrics)


if __name__ == "__main__":
    unittest.main()
