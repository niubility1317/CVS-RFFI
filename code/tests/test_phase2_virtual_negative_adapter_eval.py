import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _write_npz(path: Path, *, include_unknown: bool = True) -> None:
    rows = []

    def add(role, tx, rx, day, sig, scenario, feature):
        rows.append((role, tx, rx, day, sig, scenario, np.asarray(feature, dtype=np.float32)))

    for rx in ["rx-a", "rx-b", "rx-c"]:
        add("source", "old-a", "src-a", "d0", f"src-old-{rx}", "", [1.0, 0.0, 0.0])
        add("target_old", "old-a", rx, "d1", f"old-support-{rx}", "leo_clear_weak", [1.0, 0.0, 0.0])
        add("target_new", "new-a", rx, "d1", f"new-support-{rx}", "leo_clear_weak", [0.0, 1.0, 0.0])
        add("target_old", "old-a", rx, "d2", "old-query", "leo_clear_weak", [0.98, 0.02, 0.0])
        add("target_old", "old-a", rx, "d2", "old-query-2", "leo_clear_weak", [0.99, 0.01, 0.0])
        add("target_new", "new-a", rx, "d2", "new-query", "leo_clear_weak", [0.02, 0.98, 0.0])
        add("target_new", "new-a", rx, "d2", "new-query-2", "leo_clear_weak", [0.01, 0.99, 0.0])
        if include_unknown:
            add("target_unknown", "unk-a", rx, "d2", "unk-query", "leo_clear_weak", [0.45, 0.45, 0.8])
            add("target_unknown", "unk-a", rx, "d2", "unk-query-2", "leo_clear_weak", [0.44, 0.45, 0.8])
    manifest = {
        "source_tx_ids": ["old-a"],
        "target_old_tx_ids": ["old-a"],
        "new_tx_ids": ["new-a"],
        "unknown_tx_ids": ["unk-a"],
        "target_channel_view": "satellite/LEO",
    }
    np.savez(
        path,
        features=np.stack([r[6] for r in rows]).astype(np.float32),
        dataset_role=np.asarray([r[0] for r in rows], dtype=object),
        tx_ids=np.asarray([r[1] for r in rows], dtype=object),
        rx_ids=np.asarray([r[2] for r in rows], dtype=object),
        day_ids=np.asarray([r[3] for r in rows], dtype=object),
        sig_ids=np.asarray([r[4] for r in rows], dtype=object),
        sat_scenarios=np.asarray([r[5] for r in rows], dtype=object),
        channel_views=np.asarray(["satellite" if r[5] else "clean" for r in rows], dtype=object),
        manifest_json=np.asarray(json.dumps(manifest)),
    )


class Phase2VirtualNegativeAdapterEvalTest(unittest.TestCase):
    def test_builds_virtual_negative_evidence_and_reports_one_to_all_receivers(self):
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz
        from phase2_virtual_negative_adapter_eval import build_virtual_negative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_virtual_negative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                virtual_negative_policy="shell_mix",
            )
            result = evaluate_collaborative_open_set_evidence(
                evidence,
                collab_counts="all",
                threshold_selection_label_scope=metadata["threshold_scope"],
                protocol_metadata=metadata,
                strict_protocol_metadata=True,
                collab_group_policy="available_up_to_k",
                partial_collab_min_receivers=1,
                unknown_risk_threshold=0.65,
                accept_margin_threshold=0.02,
            )

        self.assertEqual(result["receiver_count"], 3)
        self.assertEqual(set(result["counts"]), {"1", "2", "3"})
        self.assertEqual(metadata["adapter_type"], "support_virtual_negative_boundary")
        self.assertEqual(metadata["adapter_update_scope"], "support_old_seen_new_only")
        self.assertEqual(metadata["threshold_scope"], "support_virtual_unknown")
        self.assertTrue(metadata["unknown_query_eval_only"])
        self.assertTrue(metadata["non_deployment_diagnostic"])
        self.assertGreater(metadata["state_size_bytes"], 0)
        self.assertTrue(all(v > 0 for v in metadata["virtual_negative_count"].values()))
        self.assertTrue(all(row["calibration_role"] == "query" for row in evidence))
        self.assertTrue(all(row["threshold_selection_label_scope"] == "support_virtual_unknown" for row in evidence))

    def test_requires_unknown_query_rows_for_open_set_eval(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz
        from phase2_virtual_negative_adapter_eval import build_virtual_negative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz, include_unknown=False)
            with self.assertRaisesRegex(RuntimeError, "LOCAL_DATASET_EXTENSION_REQUIRED"):
                build_virtual_negative_evidence(load_feature_npz(npz), k_shot=1, query_per_class=1)

    def test_unknown_rows_are_not_marked_as_calibration(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz
        from phase2_virtual_negative_adapter_eval import build_virtual_negative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_virtual_negative_evidence(load_feature_npz(npz), k_shot=1, query_per_class=2)
        unknown_rows = [row for row in evidence if row["role"] == "unknown"]
        self.assertTrue(unknown_rows)
        self.assertEqual(metadata["threshold_scope_detail"], "support_old_seen_new_virtual_negatives_only")
        self.assertTrue(all(row["true_label"] == "__unknown__" for row in unknown_rows))
        self.assertTrue(all(row["calibration_role"] == "query" for row in unknown_rows))


if __name__ == "__main__":
    unittest.main()
