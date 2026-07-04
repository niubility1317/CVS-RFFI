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
    rng = np.random.default_rng(41)
    rows = []
    centers = {
        "old-a": np.array([1, 0, 0, 0], dtype=np.float32),
        "new-a": np.array([0, 1, 0, 0], dtype=np.float32),
        "proxy-a": np.array([0, 0, 1, 0], dtype=np.float32),
        "unk-a": np.array([0, 0, 0, 1], dtype=np.float32),
    }
    for role, tx, receivers, count in [
        ("source", "old-a", ["src-rx"], 10),
        ("proxy_unknown", "proxy-a", ["src-rx"], 10),
        ("target_old", "old-a", ["rx-a", "rx-b"], 8),
        ("target_new", "new-a", ["rx-a", "rx-b"], 8),
        ("target_unknown", "unk-a", ["rx-a", "rx-b"], 8),
    ]:
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


class Phase2SourceOpenMetricCiEvalTest(unittest.TestCase):
    def test_source_metric_excludes_target_roles_from_training(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz
        from phase2_source_open_metric_ci_eval import parse_args, train_source_metric

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npz"
            _write_npz(path)
            payload = load_feature_npz(path)
            args = parse_args(
                [
                    "--feature_npz",
                    str(path),
                    "--output_dir",
                    str(Path(tmp) / "out"),
                    "--device",
                    "cpu",
                    "--metric_epochs",
                    "2",
                    "--virtual_count",
                    "4",
                ]
            )
            log_scale, metrics = train_source_metric(payload, args)

        self.assertEqual(log_scale.shape[0], payload["features"].shape[1])
        self.assertEqual(metrics["training_counts"]["target_support"], 0)
        self.assertEqual(metrics["training_counts"]["target_unknown_training_count"], 0)
        self.assertEqual(metrics["training_counts"]["target_unknown_eval_only"], 16)
        self.assertGreater(metrics["state_bytes"]["total_fp16_state_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
