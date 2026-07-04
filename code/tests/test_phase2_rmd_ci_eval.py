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


class Phase2RmdCiEvalTest(unittest.TestCase):
    def test_old_anchor_and_density_accept_known(self):
        from phase2_rmd_ci_eval import _fuse_rmd_event, _profile_by_name

        rows = [
            {
                "event_id": "old-1",
                "role": "old",
                "true_label": "old-a",
                "receiver_id": f"rx-{idx}",
                "top_label": "old-a",
                "top_label_set": "old",
                "rmd_score": 0.94,
                "density_score": 0.93,
                "shell_risk": 0.05,
                "old_anchor_score": 0.96,
                "margin": 0.35,
                "quality": 0.95,
                "bytes": 176,
                "latency_ms": 0.6,
            }
            for idx in range(2)
        ]

        out = _fuse_rmd_event(
            rows,
            profile=_profile_by_name("rmd_primary"),
            max_event_bytes=1152,
            max_event_latency_ms=20,
        )

        self.assertEqual(out["output_label"], "old-a")
        self.assertEqual(out["output_action"], "accept")
        self.assertEqual(out["decision"], "accept_old_rmd_anchor")

    def test_shell_risk_rejects_unknown_without_old_guard(self):
        from phase2_rmd_ci_eval import UNKNOWN_LABEL, _fuse_rmd_event, _profile_by_name

        rows = [
            {
                "event_id": "unk-1",
                "role": "unknown",
                "true_label": "unk-a",
                "receiver_id": "rx-a",
                "top_label": "old-a",
                "top_label_set": "old",
                "rmd_score": 0.30,
                "density_score": 0.18,
                "shell_risk": 0.92,
                "old_anchor_score": 0.20,
                "margin": 0.03,
                "quality": 0.90,
                "bytes": 176,
                "latency_ms": 0.6,
            },
            {
                "event_id": "unk-1",
                "role": "unknown",
                "true_label": "unk-a",
                "receiver_id": "rx-b",
                "top_label": "new-a",
                "top_label_set": "seen_new",
                "rmd_score": 0.28,
                "density_score": 0.16,
                "shell_risk": 0.95,
                "old_anchor_score": 0.00,
                "margin": 0.02,
                "quality": 0.88,
                "bytes": 176,
                "latency_ms": 0.6,
            },
        ]

        out = _fuse_rmd_event(
            rows,
            profile=_profile_by_name("rmd_primary"),
            max_event_bytes=1152,
            max_event_latency_ms=20,
        )

        self.assertEqual(out["output_label"], UNKNOWN_LABEL)
        self.assertEqual(out["output_action"], "reject_unknown")
        self.assertEqual(out["decision"], "reject_unknown_rmd_shell")

    def test_evaluate_reports_m_counts_and_resource_gate(self):
        from phase2_rmd_ci_eval import evaluate_rmd, _profile_by_name

        rows = []
        for receiver in ["rx-a", "rx-b", "rx-c"]:
            for role, label, top_label, rmd, density, shell in [
                ("old", "old-a", "old-a", 0.95, 0.95, 0.05),
                ("seen_new", "new-a", "new-a", 0.92, 0.92, 0.08),
                ("unknown", "unk-a", "old-a", 0.20, 0.15, 0.95),
            ]:
                rows.append(
                    {
                        "event_id": f"{role}-1",
                        "role": role,
                        "true_label": label,
                        "receiver_id": receiver,
                        "top_label": top_label,
                        "top_label_set": "old" if top_label == "old-a" else "seen_new",
                        "rmd_score": rmd,
                        "density_score": density,
                        "shell_risk": shell,
                        "old_anchor_score": 0.96 if role == "old" else 0.10,
                        "margin": 0.35 if role != "unknown" else 0.02,
                        "quality": 0.90,
                        "bytes": 176,
                        "latency_ms": 0.6,
                    }
                )

        result = evaluate_rmd(
            rows,
            profiles=[_profile_by_name("rmd_primary")],
            collab_counts="all",
            collab_group_policy="same_max_budget",
            receiver_selection_policy="quality_prior",
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
            self.assertTrue(row["resource_proxy_pass"])
            self.assertTrue(row["unknown_query_eval_only"])
            self.assertEqual(row["target_unknown_training_count"], 0)
            self.assertFalse(row["shell_negatives_use_target_unknown"])

    def test_feature_pipeline_uses_support_shell_without_unknown(self):
        from phase2_rmd_ci_eval import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            feature_npz = root / "features.npz"
            output_json = root / "rmd.json"
            output_csv = root / "rmd.csv"
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
                    "rmd_primary",
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
        self.assertEqual(result["algorithm"], "RMD-CI")
        self.assertTrue(result["unknown_query_eval_only"])
        self.assertEqual(result["target_unknown_training_count"], 0)
        self.assertFalse(result["threshold_uses_target_unknown"])
        self.assertFalse(result["profile_selection_uses_target_unknown"])
        self.assertFalse(result["reliability_uses_target_unknown"])
        self.assertFalse(result["shell_negatives_use_target_unknown"])
        self.assertEqual(result["calibration_source"], "target_old_and_target_new_support_plus_support_shell_negatives")
        self.assertEqual(result["target_receivers"], ["rx-a", "rx-b"])
        self.assertNotIn("unk-a", result["class_calibrators"])
        self.assertNotIn("unk-a", result["shell_negative_counts"])
        self.assertEqual(result["summary_rows"][0]["old_acc"], 1.0)
        self.assertEqual(result["summary_rows"][0]["seen_new_acc"], 1.0)
        self.assertEqual(result["summary_rows"][0]["unknown_reject"], 1.0)


if __name__ == "__main__":
    unittest.main()
