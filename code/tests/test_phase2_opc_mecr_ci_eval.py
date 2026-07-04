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


def _write_min_stage2c_npz(path: Path) -> None:
    rows = []

    def add(role, tx, rx, day, sig, scenario, feature):
        rows.append((role, tx, rx, day, sig, scenario, np.asarray(feature, dtype=np.float32)))

    for rx in ["rx-a", "rx-b"]:
        add("source", "old-a", "src-a", "d0", f"source-{rx}", "", [1.0, 0.0, 0.0])
        add("target_old", "old-a", rx, "d1", f"old-support-{rx}", "leo_clear_weak", [1.0, 0.0, 0.0])
        add("target_new", "new-a", rx, "d1", f"new-support-{rx}", "leo_clear_weak", [0.0, 1.0, 0.0])
        add("target_old", "old-a", rx, "d2", "old-query", "leo_clear_weak", [0.98, 0.02, 0.0])
        add("target_new", "new-a", rx, "d2", "new-query", "leo_clear_weak", [0.02, 0.98, 0.0])
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


class Phase2OpcMecrCiEvalTest(unittest.TestCase):
    def test_profile_names_parse_all(self):
        from phase2_opc_mecr_ci_eval import PROFILES, _profile_names

        self.assertEqual(_profile_names("all"), [profile.name for profile in PROFILES])

    def test_old_safe_gate_accepts_old_without_strong_unknown_override(self):
        from phase2_opc_mecr_ci_eval import _fuse_opc_mecr_event, _profile_by_name

        row = {
            "event_id": "old-1",
            "role": "old",
            "true_label": "old-a",
            "receiver_id": "rx-a",
            "class_evidence_top1_label": "old-a",
            "class_evidence_top1_score": 0.96,
            "class_evidence_top1_margin": 0.34,
            "class_evidence_top1_conformal_pvalue": 0.94,
            "class_evidence_top1_receiver_class_reliability": 0.92,
            "class_evidence_top1_support_count": 8,
            "class_evidence_top1_unknown_risk": 0.32,
            "pcet_unknown_risk": 0.32,
            "class_negative_risk": 0.20,
            "class_shell_risk": 0.20,
            "support_density": 0.92,
            "reliability": 0.95,
            "bytes": 128,
            "latency_ms": 0.5,
        }

        out = _fuse_opc_mecr_event(
            [row, {**row, "receiver_id": "rx-b"}],
            profile=_profile_by_name("opc_old_guard"),
            top_m=1,
            old_labels={"old-a"},
            seen_labels={"new-a"},
            max_event_bytes=1152,
            max_event_latency_ms=20,
        )

        self.assertEqual(out["output_label"], "old-a")
        self.assertEqual(out["decision"], "accept_old_safe")
        self.assertTrue(out["old_safe"])

    def test_unknown_reject_requires_strong_unknown_and_no_consensus(self):
        from phase2_opc_mecr_ci_eval import UNKNOWN_LABEL, _fuse_opc_mecr_event, _profile_by_name

        base = {
            "event_id": "unk-1",
            "role": "unknown",
            "true_label": "unk-a",
            "class_evidence_top1_label": "old-a",
            "class_evidence_top1_score": 0.08,
            "class_evidence_top1_margin": 0.0,
            "class_evidence_top1_conformal_pvalue": 0.0,
            "class_evidence_top1_receiver_class_reliability": 0.05,
            "class_evidence_top1_support_count": 1,
            "class_evidence_top1_unknown_risk": 0.96,
            "pcet_unknown_risk": 0.96,
            "class_negative_risk": 0.96,
            "class_shell_risk": 0.94,
            "mahalanobis_risk": 0.91,
            "evt_risk": 0.90,
            "support_density": 0.02,
            "reliability": 0.05,
            "bytes": 128,
            "latency_ms": 0.5,
        }
        rows = [{**base, "receiver_id": f"rx-{i}"} for i in range(3)]

        out = _fuse_opc_mecr_event(
            rows,
            profile=_profile_by_name("mecr_balanced"),
            top_m=1,
            old_labels={"old-a"},
            seen_labels={"new-a"},
            max_event_bytes=1152,
            max_event_latency_ms=20,
        )

        self.assertEqual(out["output_label"], UNKNOWN_LABEL)
        self.assertEqual(out["output_action"], "reject_unknown")
        self.assertEqual(out["decision"], "reject_unknown_no_consensus")
        self.assertTrue(out["strong_unknown"])
        self.assertTrue(out["no_known_consensus"])

    def test_defer_is_not_encoded_as_unknown_reject_label(self):
        from phase2_opc_mecr_ci_eval import DEFER_LABEL, _fuse_opc_mecr_event, _profile_by_name

        row = {
            "event_id": "ambiguous-1",
            "role": "unknown",
            "true_label": "unk-a",
            "receiver_id": "rx-a",
            "class_evidence_top1_label": "old-a",
            "class_evidence_top1_score": 0.10,
            "class_evidence_top1_margin": 0.0,
            "class_evidence_top1_conformal_pvalue": 0.05,
            "class_evidence_top1_receiver_class_reliability": 0.05,
            "class_evidence_top1_support_count": 1,
            "class_evidence_top1_unknown_risk": 0.55,
            "pcet_unknown_risk": 0.55,
            "class_negative_risk": 0.45,
            "class_shell_risk": 0.10,
            "support_density": 0.05,
            "reliability": 0.05,
            "bytes": 128,
            "latency_ms": 0.5,
        }

        out = _fuse_opc_mecr_event(
            [row],
            profile=_profile_by_name("mecr_balanced"),
            top_m=1,
            old_labels={"old-a"},
            seen_labels={"new-a"},
            max_event_bytes=1152,
            max_event_latency_ms=20,
        )

        self.assertEqual(out["output_label"], DEFER_LABEL)
        self.assertEqual(out["output_action"], "defer")
        self.assertEqual(out["decision"], "defer")

    def test_evaluate_opc_mecr_reports_all_collab_counts_and_protocol_flags(self):
        from phase2_opc_mecr_ci_eval import evaluate_opc_mecr, _profile_by_name

        rows = []
        for receiver in ["rx-a", "rx-b", "rx-c"]:
            for role, true_label in [("old", "old-a"), ("seen_new", "new-a"), ("unknown", "unk-a")]:
                is_unknown = role == "unknown"
                rows.append(
                    {
                        "event_id": f"{role}-1",
                        "role": role,
                        "true_label": true_label,
                        "receiver_id": receiver,
                        "class_evidence_top1_label": "new-a" if role == "seen_new" else "old-a",
                        "class_evidence_top1_score": 0.95 if not is_unknown else 0.08,
                        "class_evidence_top1_margin": 0.34 if not is_unknown else 0.0,
                        "class_evidence_top1_conformal_pvalue": 0.94 if not is_unknown else 0.0,
                        "class_evidence_top1_receiver_class_reliability": 0.92 if not is_unknown else 0.05,
                        "class_evidence_top1_support_count": 8,
                        "class_evidence_top1_unknown_risk": 0.30 if not is_unknown else 0.96,
                        "pcet_unknown_risk": 0.30 if not is_unknown else 0.96,
                        "class_negative_risk": 0.20 if not is_unknown else 0.96,
                        "class_shell_risk": 0.20 if not is_unknown else 0.94,
                        "mahalanobis_risk": 0.20 if not is_unknown else 0.91,
                        "evt_risk": 0.20 if not is_unknown else 0.90,
                        "support_density": 0.92 if not is_unknown else 0.02,
                        "reliability": 0.95 if not is_unknown else 0.05,
                        "bytes": 128,
                        "latency_ms": 0.5,
                    }
                )

        result = evaluate_opc_mecr(
            rows,
            profiles=[_profile_by_name("opc_old_guard")],
            collab_counts="all",
            collab_group_policy="same_max_budget",
            receiver_selection_policy="support_quality_prior",
            top_m=1,
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
        self.assertTrue(all(row["unknown_query_eval_only"] for row in result["summary_rows"]))
        self.assertTrue(all(row["target_unknown_training_count"] == 0 for row in result["summary_rows"]))
        self.assertFalse(result["profile_selection_uses_target_unknown"])
        self.assertIn("best_posthoc_eval_row", result)
        self.assertNotIn("best_eval_row", result)
        self.assertEqual(
            result["joint_score_scope"],
            "posthoc_evaluation_analysis_only_not_profile_or_threshold_selection",
        )
        for row in result["summary_rows"]:
            self.assertIn("event_count", row)
            self.assertIn("old_total", row)
            self.assertIn("seen_new_total", row)
            self.assertIn("unknown_total", row)

    def test_known_consensus_rate_requires_safe_gate(self):
        from phase2_opc_mecr_ci_eval import evaluate_opc_mecr, _profile_by_name

        rows = []
        for receiver in ["rx-a", "rx-b"]:
            rows.append(
                {
                    "event_id": "old-weak",
                    "role": "old",
                    "true_label": "old-a",
                    "receiver_id": receiver,
                    "class_evidence_top1_label": "old-a",
                    "class_evidence_top1_score": 0.05,
                    "class_evidence_top1_margin": 0.0,
                    "class_evidence_top1_conformal_pvalue": 0.0,
                    "class_evidence_top1_receiver_class_reliability": 0.0,
                    "class_evidence_top1_support_count": 1,
                    "class_evidence_top1_unknown_risk": 0.4,
                    "pcet_unknown_risk": 0.4,
                    "class_negative_risk": 0.2,
                    "class_shell_risk": 0.1,
                    "support_density": 0.0,
                    "reliability": 0.0,
                    "bytes": 128,
                    "latency_ms": 0.5,
                }
            )
        result = evaluate_opc_mecr(
            rows,
            profiles=[_profile_by_name("opc_old_guard")],
            collab_counts="all",
            collab_group_policy="same_max_budget",
            receiver_selection_policy="support_quality_prior",
            top_m=1,
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

        self.assertTrue(all(float(row["known_consensus_rate"]) == 0.0 for row in result["summary_rows"]))

    def test_feature_pipeline_keeps_target_unknown_out_of_calibration_sources(self):
        from phase2_opc_mecr_ci_eval import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            feature_npz = root / "features.npz"
            output_json = root / "opc.json"
            output_csv = root / "opc.csv"
            _write_min_stage2c_npz(feature_npz)

            rc = main(
                [
                    "--feature_npz",
                    str(feature_npz),
                    "--output_json",
                    str(output_json),
                    "--output_summary_csv",
                    str(output_csv),
                    "--profiles",
                    "opc_old_guard",
                    "--collab_counts",
                    "all",
                    "--collab_group_policy",
                    "same_max_budget",
                    "--top_m",
                    "1",
                    "--k_shot",
                    "1",
                    "--query_per_class",
                    "1",
                    "--qknn_k",
                    "1",
                    "--seed",
                    "7",
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
        self.assertEqual(result["joint_score_scope"], "posthoc_evaluation_analysis_only_not_profile_or_threshold_selection")
        metadata = result["base_pcet_known_preserving"]["qknn_metadata"]
        self.assertTrue(metadata["unknown_query_eval_only"])
        self.assertFalse(metadata["labeled_unknown_support_used_for_boundary_fit"])
        self.assertNotIn("unknown", metadata["threshold_scope"])
        self.assertIn(metadata["threshold_scope"], {"source_only", "support_known_only"})
        self.assertEqual(metadata["unknown_tx_ids"], ["unk-a"])
        for receiver_counts in metadata["receiver_class_conformal_counts"].values():
            self.assertNotIn("unk-a", receiver_counts)
        for receiver_thresholds in metadata["receiver_class_thresholds"].values():
            self.assertNotIn("unk-a", receiver_thresholds)
        for receiver_reliability in metadata["receiver_class_reliabilities"].values():
            self.assertNotIn("unk-a", receiver_reliability)


if __name__ == "__main__":
    unittest.main()
