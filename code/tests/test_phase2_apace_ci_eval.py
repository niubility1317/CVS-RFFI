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


class Phase2ApaceCiEvalTest(unittest.TestCase):
    def test_old_anchor_protection_precedes_unknown_rejection(self):
        from phase2_apace_ci_eval import _fuse_apace_event, _profile_by_name

        rows = [
            {
                "event_id": "old-1",
                "role": "old",
                "true_label": "old-a",
                "receiver_id": f"rx-{idx}",
                "top_label": "old-a",
                "top_label_set": "old",
                "proto_score": 0.96,
                "target_proto_score": 0.95,
                "old_anchor_score": 0.97,
                "density_score": 0.92,
                "conformal_p": 0.90,
                "open_energy": 0.88,
                "margin": 0.42,
                "quality": 0.95,
                "bytes": 160,
                "latency_ms": 0.5,
            }
            for idx in range(2)
        ]

        out = _fuse_apace_event(
            rows,
            profile=_profile_by_name("apace_primary"),
            max_event_bytes=1152,
            max_event_latency_ms=20,
        )

        self.assertEqual(out["output_label"], "old-a")
        self.assertEqual(out["output_action"], "accept")
        self.assertEqual(out["decision"], "accept_old_anchor_protected")

    def test_multi_evidence_unknown_rejects_when_no_old_protection(self):
        from phase2_apace_ci_eval import UNKNOWN_LABEL, _fuse_apace_event, _profile_by_name

        rows = [
            {
                "event_id": "unk-1",
                "role": "unknown",
                "true_label": "unk-a",
                "receiver_id": "rx-a",
                "top_label": "old-a",
                "top_label_set": "old",
                "proto_score": 0.58,
                "target_proto_score": 0.56,
                "old_anchor_score": 0.20,
                "density_score": 0.10,
                "conformal_p": 0.02,
                "open_energy": 0.96,
                "margin": 0.02,
                "quality": 0.92,
                "bytes": 160,
                "latency_ms": 0.5,
            },
            {
                "event_id": "unk-1",
                "role": "unknown",
                "true_label": "unk-a",
                "receiver_id": "rx-b",
                "top_label": "new-a",
                "top_label_set": "seen_new",
                "proto_score": 0.55,
                "target_proto_score": 0.54,
                "old_anchor_score": 0.18,
                "density_score": 0.08,
                "conformal_p": 0.01,
                "open_energy": 0.97,
                "margin": 0.01,
                "quality": 0.90,
                "bytes": 160,
                "latency_ms": 0.5,
            },
        ]

        out = _fuse_apace_event(
            rows,
            profile=_profile_by_name("apace_primary"),
            max_event_bytes=1152,
            max_event_latency_ms=20,
        )

        self.assertEqual(out["output_label"], UNKNOWN_LABEL)
        self.assertEqual(out["output_action"], "reject_unknown")
        self.assertEqual(out["decision"], "reject_unknown_multi_evidence")

    def test_evaluate_reports_all_counts_resource_proxy_and_scope_fields(self):
        from phase2_apace_ci_eval import evaluate_apace, _profile_by_name

        rows = []
        for receiver in ["rx-a", "rx-b", "rx-c"]:
            for role, label, top_label, pvalue, density, energy in [
                ("old", "old-a", "old-a", 0.95, 0.95, 0.10),
                ("seen_new", "new-a", "new-a", 0.92, 0.92, 0.12),
                ("unknown", "unk-a", "old-a", 0.01, 0.08, 0.98),
            ]:
                rows.append(
                    {
                        "event_id": f"{role}-1",
                        "role": role,
                        "true_label": label,
                        "receiver_id": receiver,
                        "top_label": top_label,
                        "top_label_set": "old" if top_label == "old-a" else "seen_new",
                        "proto_score": 0.96 if role != "unknown" else 0.55,
                        "target_proto_score": 0.96 if role != "unknown" else 0.55,
                        "old_anchor_score": 0.96 if role == "old" else 0.15,
                        "density_score": density,
                        "conformal_p": pvalue,
                        "open_energy": energy,
                        "margin": 0.40 if role != "unknown" else 0.01,
                        "quality": 0.90,
                        "bytes": 160,
                        "latency_ms": 0.5,
                    }
                )

        result = evaluate_apace(
            rows,
            profiles=[_profile_by_name("apace_primary")],
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
            self.assertEqual(row["old_total"], 1)
            self.assertEqual(row["seen_new_total"], 1)
            self.assertEqual(row["unknown_total"], 1)
            self.assertTrue(row["resource_proxy_pass"])
            self.assertTrue(row["unknown_query_eval_only"])
            self.assertEqual(row["target_unknown_training_count"], 0)
            self.assertFalse(row["profile_selection_uses_target_unknown"])
            self.assertFalse(row["reliability_uses_target_unknown"])

    def test_resource_proxy_failure_blocks_target_pass(self):
        from phase2_apace_ci_eval import evaluate_apace, _profile_by_name

        rows = []
        for role, label, top_label, pvalue, density, energy in [
            ("old", "old-a", "old-a", 0.95, 0.95, 0.10),
            ("seen_new", "new-a", "new-a", 0.95, 0.95, 0.10),
            ("unknown", "unk-a", "old-a", 0.01, 0.05, 0.98),
        ]:
            rows.append(
                {
                    "event_id": f"{role}-1",
                    "role": role,
                    "true_label": label,
                    "receiver_id": "rx-a",
                    "top_label": top_label,
                    "top_label_set": "old" if top_label == "old-a" else "seen_new",
                    "proto_score": 0.96 if role != "unknown" else 0.55,
                    "target_proto_score": 0.96 if role != "unknown" else 0.55,
                    "old_anchor_score": 0.96 if role == "old" else 0.15,
                    "density_score": density,
                    "conformal_p": pvalue,
                    "open_energy": energy,
                    "margin": 0.40 if role != "unknown" else 0.01,
                    "quality": 0.90,
                    "bytes": 2048,
                    "latency_ms": 0.5,
                }
            )

        result = evaluate_apace(
            rows,
            profiles=[_profile_by_name("apace_primary")],
            collab_counts="1",
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

        row = result["summary_rows"][0]
        self.assertEqual(row["old_acc"], 1.0)
        self.assertEqual(row["seen_new_acc"], 1.0)
        self.assertEqual(row["unknown_reject"], 1.0)
        self.assertFalse(row["resource_proxy_pass"])
        self.assertFalse(row["target_pass"])
        self.assertEqual(row["verdict"], "NON_DEPLOYMENT_DIAGNOSTIC")

    def test_feature_pipeline_keeps_unknown_out_of_calibration_sources(self):
        from phase2_apace_ci_eval import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            feature_npz = root / "features.npz"
            output_json = root / "apace.json"
            output_csv = root / "apace.csv"
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
                    "apace_primary",
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
        self.assertEqual(result["algorithm"], "APACE-CI")
        self.assertTrue(result["unknown_query_eval_only"])
        self.assertEqual(result["target_unknown_training_count"], 0)
        self.assertFalse(result["threshold_uses_target_unknown"])
        self.assertFalse(result["profile_selection_uses_target_unknown"])
        self.assertFalse(result["reliability_uses_target_unknown"])
        self.assertEqual(result["calibration_source"], "target_old_and_target_new_support_plus_source_old_anchor")
        self.assertEqual(result["target_receivers"], ["rx-a", "rx-b"])
        self.assertNotIn("unk-a", result["class_calibrators"])
        self.assertEqual(result["summary_rows"][0]["old_acc"], 1.0)
        self.assertEqual(result["summary_rows"][0]["seen_new_acc"], 1.0)
        self.assertEqual(result["summary_rows"][0]["unknown_reject"], 1.0)


if __name__ == "__main__":
    unittest.main()
