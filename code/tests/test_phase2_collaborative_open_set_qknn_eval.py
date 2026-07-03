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


def _write_npz(path: Path, *, include_unknown: bool = True, aligned: bool = True) -> None:
    rows = []

    def add(role, tx, rx, day, sig, scenario, feature):
        rows.append((role, tx, rx, day, sig, scenario, np.asarray(feature, dtype=np.float32)))

    for rx in ["rx-a", "rx-b"]:
        add("source", "old-a", "src-a", "d0", f"src-{rx}", "", [1.0, 0.0, 0.0])
        add("target_old", "old-a", rx, "d1", f"old-support-{rx}", "leo_clear_weak", [1.0, 0.0, 0.0])
        add("target_new", "new-a", rx, "d1", f"new-support-{rx}", "leo_clear_weak", [0.0, 1.0, 0.0])
        old_sig = "old-query" if aligned else f"old-query-{rx}"
        new_sig = "new-query" if aligned else f"new-query-{rx}"
        unk_sig = "unk-query" if aligned else f"unk-query-{rx}"
        add("target_old", "old-a", rx, "d2", old_sig, "leo_clear_weak", [0.98, 0.02, 0.0])
        add("target_old", "old-a", rx, "d2", f"{old_sig}-2", "leo_clear_weak", [0.99, 0.01, 0.0])
        add("target_new", "new-a", rx, "d2", new_sig, "leo_clear_weak", [0.02, 0.98, 0.0])
        add("target_new", "new-a", rx, "d2", f"{new_sig}-2", "leo_clear_weak", [0.01, 0.99, 0.0])
        if include_unknown:
            add("target_unknown", "unk-a", rx, "d2", unk_sig, "leo_clear_weak", [0.0, 0.0, 1.0])

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


class Phase2CollaborativeOpenSetQknnEvalTest(unittest.TestCase):
    def test_builds_qknn8_evidence_and_reports_one_to_all_receivers(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence
        from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                qknn_k=1,
            )
            result = evaluate_collaborative_open_set_evidence(
                evidence,
                collab_counts="all",
                threshold_selection_label_scope=metadata["threshold_scope"],
                protocol_metadata=metadata,
                strict_protocol_metadata=True,
            )

        self.assertEqual(result["receiver_count"], 2)
        self.assertEqual(set(result["counts"]), {"1", "2"})
        self.assertEqual(result["counts"]["2"]["old_acc"], 1.0)
        self.assertEqual(result["counts"]["2"]["seen_new_acc"], 1.0)
        self.assertGreaterEqual(result["counts"]["2"]["unknown_reject_rate"], 0.0)
        self.assertEqual(metadata["event_alignment"], "role_tx_day_sig_scenario")
        self.assertGreater(metadata["prototype_storage_bytes"], 0)

    def test_requires_target_unknown_rows_for_open_set_eval(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz, include_unknown=False)
            with self.assertRaisesRegex(RuntimeError, "LOCAL_DATASET_EXTENSION_REQUIRED"):
                build_collaborative_evidence(load_feature_npz(npz), k_shot=1, query_per_class=1)

    def test_refuses_rank_aligned_pseudo_collaboration(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz, aligned=False)
            with self.assertRaisesRegex(RuntimeError, "NO_ALIGNED_COLLABORATIVE_EVENTS"):
                build_collaborative_evidence(load_feature_npz(npz), k_shot=1, query_per_class=1)

    def test_receiver_domain_ranked_policy_is_explicitly_marked(self):
        from phase2_collaborative_open_set_qknn_eval import load_feature_npz, build_collaborative_evidence

        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "features.npz"
            _write_npz(npz, aligned=False)
            evidence, metadata = build_collaborative_evidence(
                load_feature_npz(npz),
                k_shot=1,
                query_per_class=2,
                event_alignment_policy="receiver_domain_ranked",
            )

        self.assertGreater(len(evidence), 0)
        self.assertEqual(metadata["event_alignment_policy"], "receiver_domain_ranked")
        self.assertFalse(metadata["strict_same_event_collaboration"])
        self.assertEqual(metadata["event_alignment"], "receiver_domain_ranked_by_role_tx_scenario")


if __name__ == "__main__":
    unittest.main()
