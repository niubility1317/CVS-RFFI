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


def _write_ospr_tiny_npz(path: Path) -> None:
    rng = np.random.default_rng(31)
    centers = {
        "old-a": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "old-b": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "new-a": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "unk-a": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "proxy-a": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    roles = []
    tx_ids = []
    rx_ids = []
    features = []
    sig_ids = []
    for role, tx, rx, count in [
        ("source", "old-a", "src-rx0", 10),
        ("source", "old-b", "src-rx0", 10),
        ("proxy_unknown", "proxy-a", "src-rx0", 8),
        ("target_old", "old-a", "tgt-rx0", 6),
        ("target_old", "old-b", "tgt-rx0", 6),
        ("target_new", "new-a", "tgt-rx0", 6),
        ("target_unknown", "unk-a", "tgt-rx0", 6),
    ]:
        for i in range(count):
            roles.append(role)
            tx_ids.append(tx)
            rx_ids.append(rx)
            sig_ids.append(f"{role}-{tx}-{i}")
            z = centers[tx] + rng.normal(0.0, 0.01, size=6).astype(np.float32)
            features.append(z / max(float(np.linalg.norm(z)), 1e-8))
    np.savez_compressed(
        path,
        features=np.asarray(features, dtype=np.float32),
        tx_ids=np.asarray(tx_ids),
        rx_ids=np.asarray(rx_ids),
        day_ids=np.asarray(["d0"] * len(roles)),
        sig_ids=np.asarray(sig_ids),
        dataset_role=np.asarray(roles),
        channel_views=np.asarray(["leo"] * len(roles)),
        sat_scenarios=np.asarray(["leo_clear_weak"] * len(roles)),
        raw_labels=np.asarray(tx_ids),
        domain_labels=np.asarray(rx_ids),
    )


class Phase2OsprCiEvalTest(unittest.TestCase):
    def test_training_plan_uses_source_heldout_without_target_unknown(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz
        from phase2_ospr_ci_eval import build_ospr_training_plan

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "features.npz"
            _write_ospr_tiny_npz(src)
            payload = load_feature_npz(src)
            plan = build_ospr_training_plan(
                payload,
                k_shot=2,
                query_per_class=2,
                seed=9,
                support_selection_policy="stable_first",
                source_holdout_per_class=2,
            )

        self.assertEqual(len(plan.source_holdout_indices), 4)
        self.assertEqual(len(plan.target_unknown_indices), 6)
        forbidden = set(plan.target_unknown_indices)
        self.assertTrue(forbidden.isdisjoint(plan.source_fit_indices))
        self.assertTrue(forbidden.isdisjoint(plan.source_holdout_indices))
        self.assertTrue(forbidden.isdisjoint(plan.proxy_unknown_indices))
        self.assertTrue(forbidden.isdisjoint(plan.support_indices))
        self.assertEqual(plan.target_unknown_training_count, 0)
        self.assertEqual(plan.source_holdout_calibration_count, 4)

    def test_training_plan_rejects_tx_overlap_before_training(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz
        from phase2_ospr_ci_eval import build_ospr_training_plan

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "features.npz"
            _write_ospr_tiny_npz(src)
            with np.load(src, allow_pickle=True) as data:
                arrays = {key: data[key] for key in data.files}
            arrays["tx_ids"] = arrays["tx_ids"].astype(object)
            unknown_mask = arrays["dataset_role"].astype(str) == "target_unknown"
            arrays["tx_ids"][unknown_mask] = "old-a"
            np.savez_compressed(src, **arrays)
            payload = load_feature_npz(src)

            with self.assertRaisesRegex(RuntimeError, "LOCAL_PROTOCOL_REPAIR_REQUIRED"):
                build_ospr_training_plan(
                    payload,
                    k_shot=2,
                    query_per_class=2,
                    seed=9,
                    support_selection_policy="stable_first",
                    source_holdout_per_class=2,
                )

    def test_training_plan_requires_source_holdout_per_class(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz
        from phase2_ospr_ci_eval import build_ospr_training_plan

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "features.npz"
            _write_ospr_tiny_npz(src)
            payload = load_feature_npz(src)

            with self.assertRaisesRegex(RuntimeError, "source-heldout calibration"):
                build_ospr_training_plan(
                    payload,
                    k_shot=2,
                    query_per_class=2,
                    seed=9,
                    support_selection_policy="stable_first",
                    source_holdout_per_class=10,
                )

    def test_ospr_adapter_manifest_records_resource_and_protocol_counts(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz
        from phase2_ospr_ci_eval import (
            apply_adapter,
            build_ospr_training_plan,
            parse_args,
            save_ospr_adapted_npz,
            train_ospr_adapter,
        )

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "features.npz"
            dst = Path(tmp) / "adapted.npz"
            _write_ospr_tiny_npz(src)
            payload = load_feature_npz(src)
            args = parse_args(
                [
                    "--feature_npz",
                    str(src),
                    "--output_dir",
                    str(Path(tmp) / "out"),
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
                    "--source_holdout_per_class",
                    "2",
                    "--backend",
                    "enpc",
                ]
            )
            plan = build_ospr_training_plan(
                payload,
                k_shot=2,
                query_per_class=2,
                seed=9,
                support_selection_policy="stable_first",
                source_holdout_per_class=2,
            )
            adapter, metrics = train_ospr_adapter(payload, plan, args)
            adapted = apply_adapter(payload, adapter, "cpu")
            save_ospr_adapted_npz(src, dst, adapted, {"algorithm": "OSPR-CI", "train_metrics": metrics})
            with np.load(dst, allow_pickle=True) as data:
                manifest = str(np.asarray(data["manifest_json"]).item())

        self.assertEqual(adapted.shape[0], payload["features"].shape[0])
        self.assertIn("OSPR-CI", manifest)
        self.assertIn("target_unknown_training_count", manifest)
        self.assertEqual(metrics["training_counts"]["target_unknown_training_count"], 0)
        self.assertEqual(metrics["training_counts"]["source_holdout_calibration"], 4)
        self.assertGreater(metrics["state_bytes"]["qknn8_support_int8_bytes"], 0)
        self.assertNotIn("resource_real_pass", manifest)


if __name__ == "__main__":
    unittest.main()
