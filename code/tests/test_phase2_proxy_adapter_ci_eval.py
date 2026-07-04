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


def _write_tiny_npz(path: Path) -> None:
    rng = np.random.default_rng(7)
    roles = []
    tx_ids = []
    rx_ids = []
    day_ids = []
    sig_ids = []
    scenarios = []
    channel_views = []
    features = []
    centers = {
        "old-a": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "new-a": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "unk-a": np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "proxy-a": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }
    for role, tx, rx, count in [
        ("source", "old-a", "src-rx", 8),
        ("proxy_unknown", "proxy-a", "src-rx", 8),
        ("target_old", "old-a", "tgt-rx", 6),
        ("target_new", "new-a", "tgt-rx", 6),
        ("target_unknown", "unk-a", "tgt-rx", 6),
    ]:
        for i in range(count):
            roles.append(role)
            tx_ids.append(tx)
            rx_ids.append(rx)
            day_ids.append("d0")
            sig_ids.append(f"{role}-{i}")
            scenarios.append("leo_clear_weak")
            channel_views.append("leo")
            vec = centers[tx] + rng.normal(0.0, 0.01, size=4).astype(np.float32)
            features.append(vec / max(float(np.linalg.norm(vec)), 1e-8))
    np.savez_compressed(
        path,
        features=np.asarray(features, dtype=np.float32),
        tx_ids=np.asarray(tx_ids),
        rx_ids=np.asarray(rx_ids),
        day_ids=np.asarray(day_ids),
        sig_ids=np.asarray(sig_ids),
        dataset_role=np.asarray(roles),
        channel_views=np.asarray(channel_views),
        sat_scenarios=np.asarray(scenarios),
        raw_labels=np.asarray(tx_ids),
        domain_labels=np.asarray(rx_ids),
    )


class Phase2ProxyAdapterCiEvalTest(unittest.TestCase):
    def test_training_plan_excludes_target_unknown(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz
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

        self.assertEqual(len(plan.target_unknown_indices), 6)
        self.assertTrue(set(plan.target_unknown_indices).isdisjoint(plan.source_old_indices))
        self.assertTrue(set(plan.target_unknown_indices).isdisjoint(plan.proxy_unknown_indices))
        self.assertTrue(set(plan.target_unknown_indices).isdisjoint(plan.support_indices))
        self.assertEqual(plan.training_roles if hasattr(plan, "training_roles") else ["source", "proxy_unknown"], ["source", "proxy_unknown"])

    def test_adapter_npz_manifest_marks_unknown_eval_only(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz
        from phase2_proxy_adapter_ci_eval import apply_adapter, build_training_plan, parse_args, save_adapted_npz, train_adapter

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "features.npz"
            dst = Path(tmp) / "adapted.npz"
            _write_tiny_npz(src)
            payload = load_feature_npz(src)
            args = parse_args(
                [
                    "--feature_npz",
                    str(src),
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
            plan = build_training_plan(
                payload,
                k_shot=2,
                query_per_class=2,
                seed=3,
                support_selection_policy="stable_first",
            )
            adapter, metrics = train_adapter(payload, plan, args)
            adapted = apply_adapter(payload, adapter, "cpu")
            save_adapted_npz(src, dst, adapted, {"target_unknown_eval_only": True, "train_metrics": metrics})
            with np.load(dst, allow_pickle=True) as data:
                manifest = str(np.asarray(data["manifest_json"]).item())

        self.assertIn("target_unknown_eval_only", manifest)
        self.assertEqual(adapted.shape[0], payload["features"].shape[0])


if __name__ == "__main__":
    unittest.main()
