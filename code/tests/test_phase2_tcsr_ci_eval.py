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
        add("source", "old-a", "src-a", "d0", f"src-{rx}", "", [1.0, 0.0, 0.0])
        add("target_old", "old-a", rx, "d1", f"old-support-{rx}", "leo_clear_weak", [1.0, 0.0, 0.0])
        add("target_new", "new-a", rx, "d1", f"new-support-{rx}", "leo_clear_weak", [0.0, 1.0, 0.0])
        add("target_old", "old-a", rx, "d2", "old-query", "leo_clear_weak", [0.99, 0.01, 0.0])
        add("target_new", "new-a", rx, "d2", "new-query", "leo_clear_weak", [0.01, 0.99, 0.0])
        add("target_unknown", "unk-a", rx, "d2", "unk-query", "leo_clear_weak", [0.0, 0.0, 1.0])
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


class Phase2TcsrCiEvalTest(unittest.TestCase):
    def test_fuse_accepts_known_and_rejects_low_similarity_unknown(self):
        from phase2_tcsr_ci_eval import UNKNOWN_LABEL, _fuse_tcsr_event, _profile_by_name

        profile = _profile_by_name("tcsr_support_tight")
        old_rows = [
            {
                "event_id": "old-1",
                "role": "old",
                "true_label": "old-a",
                "receiver_id": f"rx-{idx}",
                "top_label": "old-a",
                "top_label_set": "old",
                "support_score": 0.98,
                "prototype_score": 0.99,
                "margin": 0.60,
                "class_threshold": 0.90,
                "bytes": 128,
                "latency_ms": 0.2,
            }
            for idx in range(2)
        ]
        unknown_rows = [
            {
                **old_rows[0],
                "event_id": "unk-1",
                "role": "unknown",
                "true_label": "unk-a",
                "top_label": "old-a",
                "top_label_set": "old",
                "support_score": 0.20,
                "prototype_score": 0.15,
                "margin": 0.01,
                "class_threshold": 0.90,
            },
            {
                **old_rows[1],
                "event_id": "unk-1",
                "role": "unknown",
                "true_label": "unk-a",
                "top_label": "new-a",
                "top_label_set": "seen_new",
                "support_score": 0.18,
                "prototype_score": 0.12,
                "margin": 0.0,
                "class_threshold": 0.90,
            },
        ]

        old_out = _fuse_tcsr_event(
            old_rows,
            profile=profile,
            max_event_bytes=1152,
            max_event_latency_ms=20,
        )
        unknown_out = _fuse_tcsr_event(
            unknown_rows,
            profile=profile,
            max_event_bytes=1152,
            max_event_latency_ms=20,
        )

        self.assertEqual(old_out["output_label"], "old-a")
        self.assertEqual(old_out["output_action"], "accept")
        self.assertEqual(unknown_out["output_label"], UNKNOWN_LABEL)
        self.assertEqual(unknown_out["output_action"], "reject_unknown")

    def test_unknown_probe_rejects_high_score_without_stable_consensus(self):
        from phase2_tcsr_ci_eval import UNKNOWN_LABEL, _fuse_tcsr_event, _profile_by_name

        rows = [
            {
                "event_id": "unk-split",
                "role": "unknown",
                "true_label": "unk-a",
                "receiver_id": "rx-a",
                "top_label": "old-a",
                "top_label_set": "old",
                "support_score": 0.96,
                "prototype_score": 0.96,
                "margin": 0.01,
                "class_threshold": 0.90,
                "bytes": 128,
                "latency_ms": 0.2,
            },
            {
                "event_id": "unk-split",
                "role": "unknown",
                "true_label": "unk-a",
                "receiver_id": "rx-b",
                "top_label": "new-a",
                "top_label_set": "seen_new",
                "support_score": 0.95,
                "prototype_score": 0.95,
                "margin": 0.01,
                "class_threshold": 0.90,
                "bytes": 128,
                "latency_ms": 0.2,
            },
        ]

        out = _fuse_tcsr_event(
            rows,
            profile=_profile_by_name("tcsr_unknown_probe"),
            max_event_bytes=1152,
            max_event_latency_ms=20,
        )

        self.assertEqual(out["output_label"], UNKNOWN_LABEL)
        self.assertEqual(out["output_action"], "reject_unknown")
        self.assertEqual(out["decision"], "reject_unknown_no_consensus")

    def test_evaluate_reports_all_collab_counts_and_denominators(self):
        from phase2_tcsr_ci_eval import evaluate_tcsr, _profile_by_name

        rows = []
        for receiver in ["rx-a", "rx-b", "rx-c"]:
            for role, label, top_label, score in [
                ("old", "old-a", "old-a", 0.98),
                ("seen_new", "new-a", "new-a", 0.97),
                ("unknown", "unk-a", "old-a", 0.20),
            ]:
                rows.append(
                    {
                        "event_id": f"{role}-1",
                        "role": role,
                        "true_label": label,
                        "receiver_id": receiver,
                        "top_label": top_label,
                        "top_label_set": "old" if top_label == "old-a" else "seen_new",
                        "support_score": score,
                        "prototype_score": score,
                        "margin": 0.40 if role != "unknown" else 0.0,
                        "class_threshold": 0.90,
                        "bytes": 128,
                        "latency_ms": 0.2,
                    }
                )

        result = evaluate_tcsr(
            rows,
            profiles=[_profile_by_name("tcsr_support_tight")],
            collab_counts="all",
            collab_group_policy="same_max_budget",
            receiver_selection_policy="support_score_prior",
            max_event_bytes=1152,
            max_event_latency_ms=20,
            target_gates={
                "old_acc": 0.99,
                "min_old": 0.95,
                "seen_new_acc": 0.97,
                "min_seen": 0.93,
                "unknown_reject": 0.99,
            },
            include_event_results=False,
        )

        self.assertEqual([row["collab_count"] for row in result["summary_rows"]], [1, 2, 3])
        for row in result["summary_rows"]:
            self.assertEqual(row["event_count"], 3)
            self.assertEqual(row["old_total"], 1)
            self.assertEqual(row["seen_new_total"], 1)
            self.assertEqual(row["unknown_total"], 1)
            self.assertTrue(row["resource_pass"])
            self.assertTrue(row["unknown_query_eval_only"])
            self.assertEqual(row["target_unknown_training_count"], 0)

    def test_feature_pipeline_keeps_unknown_out_of_threshold_sources(self):
        from phase2_tcsr_ci_eval import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            feature_npz = root / "features.npz"
            output_json = root / "tcsr.json"
            output_csv = root / "tcsr.csv"
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
                    "tcsr_support_tight",
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
        self.assertTrue(result["unknown_query_eval_only"])
        self.assertEqual(result["target_unknown_training_count"], 0)
        self.assertFalse(result["profile_selection_uses_target_unknown"])
        self.assertEqual(result["threshold_source"], "target_old_and_target_new_support_only")
        self.assertFalse(result["threshold_uses_target_unknown"])
        self.assertEqual(result["target_receivers"], ["rx-a", "rx-b"])
        self.assertNotIn("unk-a", result["class_thresholds"])
        self.assertEqual(result["summary_rows"][1]["old_acc"], 1.0)
        self.assertEqual(result["summary_rows"][1]["seen_new_acc"], 1.0)
        self.assertEqual(result["summary_rows"][1]["unknown_reject"], 1.0)


if __name__ == "__main__":
    unittest.main()
