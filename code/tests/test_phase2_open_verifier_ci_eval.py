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
    rng = np.random.default_rng(23)
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
                rows.append((role, tx, rx, f"sig-{role}-{rx}-{i}", vec))
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


class Phase2OpenVerifierCiEvalTest(unittest.TestCase):
    def test_open_verifier_keeps_unknown_eval_only_and_all_counts(self):
        from phase2_open_verifier_ci_eval import parse_args, run_ovc_ci

        with tempfile.TemporaryDirectory() as tmp:
            feature_npz = Path(tmp) / "features.npz"
            _write_npz(feature_npz)
            args = parse_args(
                [
                    "--feature_npz",
                    str(feature_npz),
                    "--output_json",
                    str(Path(tmp) / "out.json"),
                    "--device",
                    "cpu",
                    "--verifier_epochs",
                    "3",
                    "--k_shot",
                    "2",
                    "--query_per_class",
                    "2",
                    "--collab_counts",
                    "all",
                ]
            )
            result = run_ovc_ci(args)

        self.assertTrue(result["target_unknown_eval_only"])
        self.assertEqual(result["training_counts"]["target_unknown_training_count"], 0)
        self.assertEqual(result["training_counts"]["target_unknown_eval_only"], 16)
        self.assertEqual(sorted({row["collab_count"] for row in result["summary_rows"]}), [1, 2])
        self.assertGreater(result["state_bytes"]["total_fp16_state_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
