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


def _write_min_npz(path: Path) -> None:
    rows = []

    def add(role, tx, rx, day, sig, scenario, feature):
        rows.append((role, tx, rx, day, sig, scenario, np.asarray(feature, dtype=np.float32)))

    for rx in ["rx-a", "rx-b"]:
        add("source", "old-a", "src-a", "d0", f"src-{rx}-1", "", [1.0, 0.0, 0.0, 0.0])
        add("source", "old-a", "src-b", "d0", f"src-{rx}-2", "", [0.98, 0.02, 0.0, 0.0])
        add("target_old", "old-a", rx, "d1", f"old-support-{rx}", "leo_clear_weak", [1.0, 0.0, 0.0, 0.0])
        add("target_new", "new-a", rx, "d1", f"new-support-{rx}", "leo_clear_weak", [0.0, 1.0, 0.0, 0.0])
        add("target_old", "old-a", rx, "d2", "old-query", "leo_clear_weak", [0.99, 0.01, 0.0, 0.0])
        add("target_new", "new-a", rx, "d2", "new-query", "leo_clear_weak", [0.01, 0.99, 0.0, 0.0])
        add("target_unknown", "unk-a", rx, "d2", "unk-query", "leo_clear_weak", [0.0, 0.0, 1.0, 0.0])
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


class Phase2KeraCiEvalTest(unittest.TestCase):
    def test_seen_new_enrollment_is_not_overridden_by_old_anchor(self):
        from phase2_kera_ci_eval import _fuse_kera_event, _profile_by_name

        rows = [
            {
                "event_id": "new-1",
                "role": "seen_new",
                "true_label": "new-a",
                "receiver_id": "rx-a",
                "top_label": "new-a",
                "top_label_set": "seen_new",
                "known_score": 0.86,
                "prototype_score": 0.88,
                "old_anchor_score": 0.10,
                "margin": 0.18,
                "unknown_score": 0.20,
                "quality": 0.91,
                "bytes": 192,
                "latency_ms": 0.7,
            },
            {
                "event_id": "new-1",
                "role": "seen_new",
                "true_label": "new-a",
                "receiver_id": "rx-b",
                "top_label": "old-a",
                "top_label_set": "old",
                "known_score": 0.72,
                "prototype_score": 0.74,
                "old_anchor_score": 0.90,
                "margin": 0.04,
                "unknown_score": 0.24,
                "quality": 0.82,
                "bytes": 192,
                "latency_ms": 0.7,
            },
        ]

        out = _fuse_kera_event(
            rows,
            profile=_profile_by_name("kera_primary"),
            max_event_bytes=1152,
            max_event_latency_ms=20,
        )

        self.assertEqual(out["output_label"], "new-a")
        self.assertEqual(out["output_action"], "accept")
        self.assertEqual(out["decision"], "accept_seen_new_enrollment")

    def test_old_anchor_still_protects_high_confidence_old(self):
        from phase2_kera_ci_eval import _fuse_kera_event, _profile_by_name

        rows = [
            {
                "event_id": "old-1",
                "role": "old",
                "true_label": "old-a",
                "receiver_id": f"rx-{idx}",
                "top_label": "old-a",
                "top_label_set": "old",
                "known_score": 0.90,
                "prototype_score": 0.91,
                "old_anchor_score": 0.94,
                "margin": 0.22,
                "unknown_score": 0.62,
                "quality": 0.95,
                "bytes": 192,
                "latency_ms": 0.7,
            }
            for idx in range(2)
        ]

        out = _fuse_kera_event(
            rows,
            profile=_profile_by_name("kera_primary"),
            max_event_bytes=1152,
            max_event_latency_ms=20,
        )

        self.assertEqual(out["output_label"], "old-a")
        self.assertEqual(out["decision"], "accept_old_anchor_guard")

    def test_unknown_gate_rejects_only_after_known_enrollment_fails(self):
        from phase2_kera_ci_eval import UNKNOWN_LABEL, _fuse_kera_event, _profile_by_name

        rows = [
            {
                "event_id": "unk-1",
                "role": "unknown",
                "true_label": "unk-a",
                "receiver_id": "rx-a",
                "top_label": "old-a",
                "top_label_set": "old",
                "known_score": 0.24,
                "prototype_score": 0.25,
                "old_anchor_score": 0.18,
                "margin": 0.01,
                "unknown_score": 0.88,
                "quality": 0.60,
                "bytes": 192,
                "latency_ms": 0.7,
            },
            {
                "event_id": "unk-1",
                "role": "unknown",
                "true_label": "unk-a",
                "receiver_id": "rx-b",
                "top_label": "new-a",
                "top_label_set": "seen_new",
                "known_score": 0.22,
                "prototype_score": 0.24,
                "old_anchor_score": 0.00,
                "margin": 0.01,
                "unknown_score": 0.90,
                "quality": 0.58,
                "bytes": 192,
                "latency_ms": 0.7,
            },
        ]

        out = _fuse_kera_event(
            rows,
            profile=_profile_by_name("kera_primary"),
            max_event_bytes=1152,
            max_event_latency_ms=20,
        )

        self.assertEqual(out["output_label"], UNKNOWN_LABEL)
        self.assertEqual(out["decision"], "reject_unknown_after_enrollment_fail")

    def test_feature_pipeline_keeps_unknown_eval_only(self):
        from phase2_kera_ci_eval import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            feature_npz = root / "features.npz"
            output_json = root / "kera.json"
            output_csv = root / "kera.csv"
            _write_min_npz(feature_npz)

            rc = main(
                [
                    "--feature_npz",
                    str(feature_npz),
                    "--output_json",
                    str(output_json),
                    "--output_summary_csv",
                    str(output_csv),
                    "--profiles",
                    "kera_primary",
                    "--collab_counts",
                    "all",
                    "--collab_group_policy",
                    "same_max_budget",
                    "--k_shot",
                    "1",
                    "--query_per_class",
                    "1",
                    "--seed",
                    "11",
                    "--support_selection_policy",
                    "stable_first",
                    "--event_alignment_policy",
                    "receiver_domain_ranked",
                ]
            )
            result = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual(result["algorithm"], "KERA-CI")
        self.assertEqual(result["receiver_total"], 2)
        self.assertTrue(result["unknown_query_eval_only"])
        self.assertEqual(result["target_unknown_training_count"], 0)
        self.assertFalse(result["threshold_uses_target_unknown"])
        self.assertFalse(result["profile_selection_uses_target_unknown"])
        self.assertFalse(result["pseudo_unknown_uses_target_unknown"])
        self.assertEqual(result["summary_rows"][0]["old_acc"], 1.0)
        self.assertEqual(result["summary_rows"][0]["seen_new_acc"], 1.0)
        self.assertEqual(result["summary_rows"][0]["unknown_reject"], 1.0)


if __name__ == "__main__":
    unittest.main()
