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

    for src_rx in ["src-a", "src-b"]:
        for n in range(3):
            add("source", "old-a", src_rx, "d0", f"src-a-{src_rx}-{n}", "", [1.0, 0.0, 0.0, 0.0])
            add("source", "old-b", src_rx, "d0", f"src-b-{src_rx}-{n}", "", [0.0, 1.0, 0.0, 0.0])
    for rx in ["rx-a", "rx-b", "rx-c"]:
        add("target_old", "old-a", rx, "d1", f"old-a-support-{rx}", "leo_clear_weak", [1.0, 0.0, 0.0, 0.0])
        add("target_old", "old-b", rx, "d1", f"old-b-support-{rx}", "leo_clear_weak", [0.0, 1.0, 0.0, 0.0])
        add("target_new", "new-a", rx, "d1", f"new-support-{rx}", "leo_clear_weak", [0.0, 0.0, 1.0, 0.0])
        for q in range(2):
            add("target_old", "old-a", rx, "d2", f"old-a-query-{q}", "leo_clear_weak", [0.98, 0.02, 0.0, 0.0])
            add("target_old", "old-b", rx, "d2", f"old-b-query-{q}", "leo_clear_weak", [0.02, 0.98, 0.0, 0.0])
            add("target_new", "new-a", rx, "d2", f"new-query-{q}", "leo_clear_weak", [0.0, 0.02, 0.98, 0.0])
            add("target_unknown", "unk-a", rx, "d2", f"unk-query-{q}", "leo_clear_weak", [0.0, 0.0, 0.0, 1.0])
    manifest = {
        "source_tx_ids": ["old-a", "old-b"],
        "target_old_tx_ids": ["old-a", "old-b"],
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


class Phase2CrispCCiEvalTest(unittest.TestCase):
    def test_fusion_uses_topk_seen_new_rescue_before_old_absorption(self):
        from phase2_crisp_c_ci_eval import _fuse_crisp_event, _profile_by_name

        rows = [
            {
                "event_id": "seen|new-a|rank00000",
                "role": "seen_new",
                "true_label": "new-a",
                "receiver_id": "rx-a",
                "top_label": "old-a",
                "top_label_set": "old",
                "best_old_label": "old-a",
                "best_seen_new_label": "new-a",
                "old_accept_score": 0.62,
                "seen_new_accept_score": 0.86,
                "reject_score": 0.12,
                "seen_new_residual": 0.03,
                "old_envelope_violation": 0.18,
                "conformal_p_old": 0.20,
                "conformal_p_seen_new": 0.90,
                "bytes": 128,
                "latency_ms": 0.45,
                "quality": 0.95,
            },
            {
                "event_id": "seen|new-a|rank00000",
                "role": "seen_new",
                "true_label": "new-a",
                "receiver_id": "rx-b",
                "top_label": "new-a",
                "top_label_set": "seen_new",
                "best_old_label": "old-a",
                "best_seen_new_label": "new-a",
                "old_accept_score": 0.50,
                "seen_new_accept_score": 0.84,
                "reject_score": 0.08,
                "seen_new_residual": 0.02,
                "old_envelope_violation": 0.20,
                "conformal_p_old": 0.18,
                "conformal_p_seen_new": 0.92,
                "bytes": 128,
                "latency_ms": 0.45,
                "quality": 0.95,
            },
        ]

        out = _fuse_crisp_event(
            rows,
            profile=_profile_by_name("crisp_primary"),
            max_event_bytes=512,
            max_event_latency_ms=20,
        )

        self.assertEqual(out["output_label"], "new-a")
        self.assertEqual(out["output_action"], "accept")
        self.assertEqual(out["decision"], "accept_seen_new_residual")

    def test_unknown_reject_requires_old_and_seen_new_envelope_violation(self):
        from phase2_crisp_c_ci_eval import UNKNOWN_LABEL, _fuse_crisp_event, _profile_by_name

        rows = [
            {
                "event_id": "unk|unk-a|rank00000",
                "role": "unknown",
                "true_label": "unk-a",
                "receiver_id": f"rx-{idx}",
                "top_label": "old-a",
                "top_label_set": "old",
                "best_old_label": "old-a",
                "best_seen_new_label": "new-a",
                "old_accept_score": 0.24,
                "seen_new_accept_score": 0.22,
                "reject_score": 0.92,
                "seen_new_residual": 0.80,
                "old_envelope_violation": 0.75,
                "conformal_p_old": 0.02,
                "conformal_p_seen_new": 0.01,
                "bytes": 128,
                "latency_ms": 0.45,
                "quality": 0.90,
            }
            for idx in range(3)
        ]

        out = _fuse_crisp_event(
            rows,
            profile=_profile_by_name("crisp_primary"),
            max_event_bytes=512,
            max_event_latency_ms=20,
        )

        self.assertEqual(out["output_label"], UNKNOWN_LABEL)
        self.assertEqual(out["output_action"], "reject_unknown")
        self.assertEqual(out["decision"], "reject_outside_old_new_envelopes")

    def test_pipeline_reports_all_m_counts_and_keeps_unknown_eval_only(self):
        from phase2_crisp_c_ci_eval import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            feature_npz = root / "features.npz"
            output_json = root / "crisp.json"
            output_csv = root / "crisp_summary.csv"
            evidence_csv = root / "crisp_evidence.csv"
            _write_min_npz(feature_npz)

            rc = main(
                [
                    "--feature_npz",
                    str(feature_npz),
                    "--output_json",
                    str(output_json),
                    "--output_summary_csv",
                    str(output_csv),
                    "--output_evidence_csv",
                    str(evidence_csv),
                    "--profiles",
                    "crisp_primary",
                    "--collab_counts",
                    "all",
                    "--collab_group_policy",
                    "same_max_budget",
                    "--k_shot",
                    "1",
                    "--query_per_class",
                    "2",
                    "--seed",
                    "19",
                    "--support_selection_policy",
                    "stable_first",
                    "--event_alignment_policy",
                    "receiver_domain_ranked",
                ]
            )
            data = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual(data["algorithm"], "CRISP-C")
        self.assertEqual(data["receiver_total"], 3)
        self.assertEqual([row["collab_count"] for row in data["summary_rows"]], [1, 2, 3])
        self.assertTrue(data["unknown_query_eval_only"])
        self.assertEqual(data["target_unknown_training_count"], 0)
        self.assertFalse(data["threshold_uses_target_unknown"])
        self.assertFalse(data["profile_selection_uses_target_unknown"])
        self.assertFalse(data["prototype_fit_uses_target_unknown"])
        self.assertEqual(data["crisp_metadata"]["in_orbit_method"], "qknn8_residual_interval_sketch")
        self.assertGreater(data["evidence_row_count"], 0)
        self.assertTrue(any("class_evidence_top1_label" in row for row in data["evidence_preview"]))
        self.assertEqual(data["summary_rows"][0]["old_acc"], 1.0)
        self.assertEqual(data["summary_rows"][0]["seen_new_acc"], 1.0)
        self.assertEqual(data["summary_rows"][0]["unknown_reject"], 1.0)

    def test_resource_gate_cannot_create_target_pass(self):
        from phase2_crisp_c_ci_eval import evaluate_crisp, _profile_by_name

        rows = []
        for rx in ["rx-a", "rx-b"]:
            rows.append(
                {
                    "event_id": "old-1",
                    "role": "old",
                    "true_label": "old-a",
                    "receiver_id": rx,
                    "top_label": "old-a",
                    "top_label_set": "old",
                    "best_old_label": "old-a",
                    "best_seen_new_label": "new-a",
                    "old_accept_score": 0.95,
                    "seen_new_accept_score": 0.20,
                    "reject_score": 0.05,
                    "seen_new_residual": 0.40,
                    "old_envelope_violation": 0.00,
                    "conformal_p_old": 0.95,
                    "conformal_p_seen_new": 0.05,
                    "bytes": 400,
                    "latency_ms": 0.45,
                    "quality": 1.0,
                }
            )
            rows.append(
                {
                    "event_id": "seen-1",
                    "role": "seen_new",
                    "true_label": "new-a",
                    "receiver_id": rx,
                    "top_label": "old-a",
                    "top_label_set": "old",
                    "best_old_label": "old-a",
                    "best_seen_new_label": "new-a",
                    "old_accept_score": 0.90,
                    "seen_new_accept_score": 0.10,
                    "reject_score": 0.05,
                    "seen_new_residual": 0.50,
                    "old_envelope_violation": 0.00,
                    "conformal_p_old": 0.90,
                    "conformal_p_seen_new": 0.05,
                    "bytes": 400,
                    "latency_ms": 0.45,
                    "quality": 1.0,
                }
            )

        result = evaluate_crisp(
            rows,
            profiles=[_profile_by_name("crisp_primary")],
            collab_counts="all",
            collab_group_policy="same_max_budget",
            receiver_selection_policy="quality_prior",
            max_event_bytes=256,
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
        self.assertFalse(row["resource_proxy_pass"])
        self.assertFalse(row["target_pass"])
        self.assertEqual(row["verdict"], "NON_DEPLOYMENT_DIAGNOSTIC")


if __name__ == "__main__":
    unittest.main()
