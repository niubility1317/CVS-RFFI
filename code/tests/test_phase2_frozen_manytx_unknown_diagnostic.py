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


def _write_npz(path: Path) -> None:
    rows = []

    def add(role, tx, rx, day, sig, scenario, feature):
        rows.append((role, tx, rx, day, sig, scenario, np.asarray(feature, dtype=np.float32)))

    for rx in ["20-1", "3-19", "7-14"]:
        add("source", "14-10", "1-1", "d0", f"src-{rx}", "", [1.0, 0.0, 0.0])
        add("target_old", "14-10", rx, "d1", f"old-support-{rx}", "leo_clear_weak", [1.0, 0.0, 0.0])
        add("target_new", "1-16", rx, "d1", f"new-support-{rx}", "leo_clear_weak", [0.0, 1.0, 0.0])
        add("target_old", "14-10", rx, "d2", "old-query", "leo_clear_weak", [0.99, 0.01, 0.0])
        add("target_new", "1-16", rx, "d2", "new-query", "leo_clear_weak", [0.01, 0.99, 0.0])
        add("target_unknown", "10-1", rx, "d2", "unk-query", "leo_clear_weak", [0.0, 0.0, 1.0])
    manifest = {
        "source_tx_ids": ["14-10"],
        "target_old_tx_ids": ["14-10"],
        "new_tx_ids": ["1-16"],
        "unknown_tx_ids": ["10-1"],
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


def _write_legacy_npz(path: Path) -> None:
    rows = []

    def add(role, tx, rx, day, sig, scenario, feature):
        rows.append((role, tx, rx, day, sig, scenario, np.asarray(feature, dtype=np.float32)))

    rx = "20-1"
    for old in ["14-10"]:
        add("source", old, "1-1", "d0", f"src-{old}", "", [1.0, 0.0, 0.0])
        add("target_old", old, rx, "d1", f"old-support-{old}", "leo_clear_weak", [1.0, 0.0, 0.0])
        add("target_old", old, rx, "d2", f"old-query-{old}", "leo_clear_weak", [0.99, 0.01, 0.0])
    add("target_new", "1-16", rx, "d1", "new-support", "leo_clear_weak", [0.0, 1.0, 0.0])
    add("target_new", "1-16", rx, "d2", "new-query", "leo_clear_weak", [0.01, 0.99, 0.0])
    add("target_new", "10-1", rx, "d2", "legacy-unknown-query", "leo_clear_weak", [0.0, 0.0, 1.0])
    manifest = {
        "source_tx_ids": ["14-10"],
        "target_old_tx_ids": ["14-10"],
        "new_tx_ids": ["1-16"],
        "unknown_tx_ids": ["10-1"],
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


class Phase2FrozenManytxUnknownDiagnosticTest(unittest.TestCase):
    def test_reports_eval_only_unknown_and_one_to_all_receiver_counts(self):
        from phase2_frozen_manytx_unknown_diagnostic import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            npz = root / "features.npz"
            output = root / "diag.json"
            summary = root / "summary.csv"
            _write_npz(npz)
            rc = main(
                [
                    "--feature_npz",
                    str(npz),
                    "--output_json",
                    str(output),
                    "--output_summary_csv",
                    str(summary),
                    "--k_shot",
                    "1",
                    "--query_per_class",
                    "1",
                    "--qknn_k",
                    "1",
                    "--receiver_class_reliability_policy",
                    "none",
                    "--class_reliability_policy",
                    "none",
                    "--class_verifier_policy",
                    "none",
                    "--max_event_bytes",
                    "1024",
                    "--max_event_latency_ms",
                    "25",
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            summary_exists = summary.exists()

        self.assertEqual(rc, 0)
        self.assertEqual(set(payload["counts"]), {"1", "2", "3"})
        safety = payload["protocol_safety"]
        self.assertTrue(safety["diagnostic_only"])
        self.assertFalse(safety["stage2_success_claim"])
        self.assertFalse(safety["ground_training_unknown_access"])
        self.assertFalse(safety["uses_unknown_query_for_threshold"])
        self.assertTrue(safety["unknown_query_eval_only"])
        self.assertEqual(safety["target_unknown_training_count"], 0)
        self.assertEqual(safety["target_unknown_calibration_count"], 0)
        self.assertEqual(safety["target_unknown_query_count"], 3)
        self.assertEqual(safety["receiver_count"], 3)
        self.assertEqual(safety["threshold_scope"], "support_known_only")
        self.assertEqual([row["collab_count"] for row in payload["summary_rows"]], [1, 2, 3])
        for row in payload["summary_rows"]:
            self.assertIn("candidate_set_shell_veto_count", row)
            self.assertIn("candidate_set_shell_veto_rate", row)
        self.assertTrue(summary_exists)

    def test_rejects_bad_non_positive_query_per_class(self):
        from phase2_frozen_manytx_unknown_diagnostic import parse_args

        with self.assertRaises(SystemExit):
            parse_args(["--feature_npz", "x.npz", "--output_json", "out.json", "--query_per_class", "0"])

    def test_accepts_core_support_selection_policy_names(self):
        from phase2_frozen_manytx_unknown_diagnostic import parse_args

        args = parse_args(
            [
                "--feature_npz",
                "x.npz",
                "--output_json",
                "out.json",
                "--support_selection_policy",
                "strict_event_query_preserve",
            ]
        )
        self.assertEqual(args.support_selection_policy, "strict_event_query_preserve")

    def test_accepts_seen_new_rescue_cli_knobs(self):
        from phase2_frozen_manytx_unknown_diagnostic import parse_args

        args = parse_args(
            [
                "--feature_npz",
                "x.npz",
                "--output_json",
                "out.json",
                "--seen_new_rescue_enabled",
                "--seen_new_rescue_risk_scale",
                "0.4",
                "--seen_new_rescue_min_score",
                "0.7",
                "--seen_new_rescue_min_margin",
                "0.1",
                "--seen_new_rescue_min_agreement",
                "0.6",
                "--conformal_rescue_enabled",
                "--conformal_rescue_min_pvalue",
                "0.2",
                "--conformal_rescue_risk_scale",
                "0.3",
                "--conformal_rescue_min_agreement",
                "0.7",
                "--rescue_unknown_veto_enabled",
                "--rescue_unknown_veto_event_risk",
                "0.9",
                "--rescue_unknown_veto_label_risk",
                "0.8",
                "--rescue_unknown_veto_shell_risk",
                "0.85",
                "--rescue_unknown_veto_component_agreement",
                "0.6",
                "--rescue_unknown_veto_min_sources",
                "2",
                "--rescue_unknown_veto_action",
                "defer",
            ]
        )

        self.assertTrue(args.seen_new_rescue_enabled)
        self.assertEqual(args.seen_new_rescue_risk_scale, 0.4)
        self.assertEqual(args.seen_new_rescue_min_score, 0.7)
        self.assertEqual(args.seen_new_rescue_min_margin, 0.1)
        self.assertEqual(args.seen_new_rescue_min_agreement, 0.6)
        self.assertTrue(args.conformal_rescue_enabled)
        self.assertEqual(args.conformal_rescue_min_pvalue, 0.2)
        self.assertEqual(args.conformal_rescue_risk_scale, 0.3)
        self.assertEqual(args.conformal_rescue_min_agreement, 0.7)
        self.assertTrue(args.rescue_unknown_veto_enabled)
        self.assertEqual(args.rescue_unknown_veto_event_risk, 0.9)
        self.assertEqual(args.rescue_unknown_veto_label_risk, 0.8)
        self.assertEqual(args.rescue_unknown_veto_shell_risk, 0.85)
        self.assertEqual(args.rescue_unknown_veto_component_agreement, 0.6)
        self.assertEqual(args.rescue_unknown_veto_min_sources, 2)
        self.assertEqual(args.rescue_unknown_veto_action, "defer")

    def test_accepts_seen_new_old_contrast_cli_knobs(self):
        from phase2_frozen_manytx_unknown_diagnostic import parse_args

        args = parse_args(
            [
                "--feature_npz",
                "x.npz",
                "--output_json",
                "out.json",
                "--seen_new_old_contrast_weight",
                "0.25",
                "--seen_new_old_contrast_margin",
                "0.02",
                "--seen_new_contrast_gate_enabled",
                "--seen_new_contrast_gate_min_delta",
                "0.08",
                "--seen_new_contrast_gate_min_receivers",
                "2",
                "--seen_new_contrast_risk_relief_enabled",
                "--seen_new_contrast_risk_relief_min_delta",
                "0.05",
                "--seen_new_contrast_risk_relief_min_receivers",
                "2",
                "--seen_new_contrast_risk_relief_min_support_count",
                "3",
                "--seen_new_contrast_risk_relief_min_pvalue",
                "0.7",
                "--seen_new_contrast_risk_relief_min_receiver_class_reliability",
                "0.75",
                "--seen_new_contrast_label_risk_scale",
                "0.5",
                "--seen_new_contrast_event_risk_scale",
                "0.6",
                "--seen_new_contrast_component_agreement_scale",
                "0.7",
                "--candidate_set_max_label_shell_risk",
                "0.65",
                "--candidate_set_shell_reject_risk",
                "0.72",
            ]
        )

        self.assertEqual(args.seen_new_old_contrast_weight, 0.25)
        self.assertEqual(args.seen_new_old_contrast_margin, 0.02)
        self.assertTrue(args.seen_new_contrast_gate_enabled)
        self.assertEqual(args.seen_new_contrast_gate_min_delta, 0.08)
        self.assertEqual(args.seen_new_contrast_gate_min_receivers, 2)
        self.assertTrue(args.seen_new_contrast_risk_relief_enabled)
        self.assertEqual(args.seen_new_contrast_risk_relief_min_delta, 0.05)
        self.assertEqual(args.seen_new_contrast_risk_relief_min_receivers, 2)
        self.assertEqual(args.seen_new_contrast_risk_relief_min_support_count, 3)
        self.assertEqual(args.seen_new_contrast_risk_relief_min_pvalue, 0.7)
        self.assertEqual(args.seen_new_contrast_risk_relief_min_receiver_class_reliability, 0.75)
        self.assertEqual(args.seen_new_contrast_label_risk_scale, 0.5)
        self.assertEqual(args.seen_new_contrast_event_risk_scale, 0.6)
        self.assertEqual(args.seen_new_contrast_component_agreement_scale, 0.7)
        self.assertEqual(args.candidate_set_max_label_shell_risk, 0.65)
        self.assertEqual(args.candidate_set_shell_reject_risk, 0.72)

    def test_can_repair_legacy_target_new_unknown_roles_from_manifest(self):
        from phase2_frozen_manytx_unknown_diagnostic import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            npz = root / "legacy.npz"
            output = root / "diag.json"
            _write_legacy_npz(npz)
            rc = main(
                [
                    "--feature_npz",
                    str(npz),
                    "--output_json",
                    str(output),
                    "--repair_legacy_roles_from_manifest",
                    "--k_shot",
                    "1",
                    "--query_per_class",
                    "1",
                    "--qknn_k",
                    "1",
                    "--receiver_class_reliability_policy",
                    "none",
                    "--class_reliability_policy",
                    "none",
                    "--class_verifier_policy",
                    "none",
                    "--candidate_set_min_receivers",
                    "1",
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        safety = payload["protocol_safety"]
        self.assertTrue(safety["legacy_role_repair_applied"])
        self.assertEqual(safety["legacy_target_unknown_rows_after"], 1)
        self.assertFalse(safety["uses_unknown_query_for_threshold"])
        self.assertEqual(set(payload["counts"]), {"1"})


if __name__ == "__main__":
    unittest.main()
