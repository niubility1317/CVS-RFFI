import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class Phase2C3RStage2CCiEvalTest(unittest.TestCase):
    def test_profile_names_parse_all(self):
        from phase2_c3r_stage2c_ci_eval import PROFILES, _profile_names

        self.assertEqual(_profile_names("all"), [profile.name for profile in PROFILES])

    def test_old_shield_prevents_high_risk_old_rejection(self):
        from phase2_c3r_stage2c_ci_eval import _fuse_c3r_event, _profile_by_name

        row = {
            "event_id": "e-old",
            "role": "old",
            "true_label": "old-a",
            "receiver_id": "rx-a",
            "class_evidence_top1_label": "old-a",
            "class_evidence_top1_score": 0.95,
            "class_evidence_top1_margin": 0.35,
            "class_evidence_top1_conformal_pvalue": 0.95,
            "class_evidence_top1_receiver_class_reliability": 0.95,
            "class_evidence_top1_support_count": 8,
            "class_evidence_top1_unknown_risk": 0.50,
            "pcet_unknown_risk": 0.50,
            "class_negative_risk": 0.40,
            "class_shell_risk": 0.20,
            "support_density": 0.90,
            "reliability": 0.95,
            "bytes": 128,
            "latency_ms": 0.5,
        }

        out = _fuse_c3r_event(
            [row, {**row, "receiver_id": "rx-b"}],
            profile=_profile_by_name("c3r_old_anchor"),
            top_m=1,
            old_labels={"old-a"},
            seen_labels={"new-a"},
            max_event_bytes=1152,
            max_event_latency_ms=20,
        )

        self.assertEqual(out["output_label"], "old-a")
        self.assertEqual(out["decision"], "accept_old_shield")

    def test_unknown_reject_requires_no_known_shield_and_risk_agreement(self):
        from phase2_c3r_stage2c_ci_eval import UNKNOWN_LABEL, _fuse_c3r_event, _profile_by_name

        base = {
            "event_id": "e-unk",
            "role": "unknown",
            "true_label": "unk-a",
            "class_evidence_top1_label": "old-a",
            "class_evidence_top1_score": 0.12,
            "class_evidence_top1_margin": 0.01,
            "class_evidence_top1_conformal_pvalue": 0.0,
            "class_evidence_top1_receiver_class_reliability": 0.05,
            "class_evidence_top1_support_count": 1,
            "class_evidence_top1_unknown_risk": 0.96,
            "pcet_unknown_risk": 0.96,
            "class_negative_risk": 0.95,
            "class_shell_risk": 0.92,
            "mahalanobis_risk": 0.90,
            "evt_risk": 0.88,
            "support_density": 0.05,
            "reliability": 0.10,
            "bytes": 128,
            "latency_ms": 0.5,
        }
        rows = [{**base, "receiver_id": f"rx-{i}"} for i in range(3)]

        out = _fuse_c3r_event(
            rows,
            profile=_profile_by_name("c3r_balanced"),
            top_m=1,
            old_labels={"old-a"},
            seen_labels={"new-a"},
            max_event_bytes=1152,
            max_event_latency_ms=20,
        )

        self.assertEqual(out["output_label"], UNKNOWN_LABEL)
        self.assertEqual(out["decision"], "reject_unknown")

    def test_evaluate_c3r_reports_all_collab_counts(self):
        from phase2_c3r_stage2c_ci_eval import evaluate_c3r, _profile_by_name

        rows = []
        for receiver in ["rx-a", "rx-b", "rx-c"]:
            for role, true_label in [("old", "old-a"), ("seen_new", "new-a"), ("unknown", "unk-a")]:
                rows.append(
                    {
                        "event_id": f"{role}-1",
                        "role": role,
                        "true_label": true_label,
                        "receiver_id": receiver,
                        "class_evidence_top1_label": "old-a" if role != "seen_new" else "new-a",
                        "class_evidence_top1_score": 0.95 if role != "unknown" else 0.10,
                        "class_evidence_top1_margin": 0.35 if role != "unknown" else 0.01,
                        "class_evidence_top1_conformal_pvalue": 0.95 if role != "unknown" else 0.0,
                        "class_evidence_top1_receiver_class_reliability": 0.95 if role != "unknown" else 0.05,
                        "class_evidence_top1_support_count": 8,
                        "pcet_unknown_risk": 0.35 if role != "unknown" else 0.96,
                        "class_negative_risk": 0.20 if role != "unknown" else 0.95,
                        "class_shell_risk": 0.20 if role != "unknown" else 0.95,
                        "support_density": 0.90 if role != "unknown" else 0.05,
                        "reliability": 0.95 if role != "unknown" else 0.10,
                        "bytes": 128,
                        "latency_ms": 0.5,
                        "threshold_selection_label_scope": "support_only",
                    }
                )

        result = evaluate_c3r(
            rows,
            profiles=[_profile_by_name("c3r_old_anchor")],
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


if __name__ == "__main__":
    unittest.main()
