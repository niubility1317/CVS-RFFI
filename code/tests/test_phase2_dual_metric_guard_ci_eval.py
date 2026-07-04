import csv
import json
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _row(event_id, role, true_label, receiver, predicted, score, margin, risk):
    return {
        "event_id": event_id,
        "role": role,
        "true_label": true_label,
        "receiver_id": receiver,
        "predicted_label": predicted,
        "known_score": score,
        "label_score_gap": margin,
        "known_margin": margin,
        "unknown_risk": risk,
        "score_risk": risk,
        "radius_risk": risk,
        "margin_risk": max(0.0, 1.0 - margin),
        "reliability": 1.0 - risk,
        "receiver_class_reliability": 1.0 - risk,
        "support_density": 1.0,
        "latency_ms": 0.2,
        "bytes": 128.0,
        "threshold_selection_label_scope": "source_only",
    }


def _write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class Phase2DualMetricGuardCiEvalTest(unittest.TestCase):
    def test_merges_metric_risk_without_using_target_unknown_for_calibration(self):
        from phase2_dual_metric_guard_ci_eval import merge_dual_metric_rows

        base = [
            _row("old-1", "old", "old-a", "rx-a", "old-a", 0.94, 0.42, 0.10),
            _row("new-1", "seen_new", "new-a", "rx-a", "old-a", 0.62, 0.03, 0.35),
            _row("unk-1", "unknown", "__unknown__", "rx-a", "old-a", 0.67, 0.02, 0.38),
        ]
        metric = [
            _row("old-1", "old", "old-a", "rx-a", "old-a", 0.91, 0.38, 0.22),
            _row("new-1", "seen_new", "new-a", "rx-a", "new-a", 0.87, 0.30, 0.18),
            _row("unk-1", "unknown", "__unknown__", "rx-a", "old-a", 0.32, 0.01, 0.93),
        ]

        merged, metadata = merge_dual_metric_rows(
            base,
            metric,
            old_labels={"old-a"},
            seen_new_labels={"new-a"},
            metric_rescue_min_score=0.80,
            metric_rescue_min_margin=0.20,
            metric_reject_risk=0.85,
            base_old_core_min_score=0.90,
            base_old_core_min_margin=0.30,
        )

        by_event = {row["event_id"]: row for row in merged}
        self.assertEqual(by_event["old-1"]["predicted_label"], "old-a")
        self.assertLessEqual(by_event["old-1"]["unknown_risk"], 0.22)
        self.assertEqual(by_event["new-1"]["predicted_label"], "new-a")
        self.assertEqual(by_event["new-1"]["dual_metric_route"], "metric_seen_new_rescue")
        self.assertGreaterEqual(by_event["unk-1"]["unknown_risk"], 0.93)
        self.assertEqual(by_event["unk-1"]["dual_metric_route"], "metric_reject_guard")
        self.assertTrue(metadata["target_unknown_eval_only"])
        self.assertEqual(metadata["target_unknown_training_count"], 0)
        self.assertFalse(metadata["threshold_uses_target_unknown"])

    def test_cli_outputs_same_row_metrics_for_m_one_to_n(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_rows = []
            metric_rows = []
            for rx in ["rx-a", "rx-b"]:
                base_rows.extend(
                    [
                        _row("old-1", "old", "old-a", rx, "old-a", 0.94, 0.42, 0.10),
                        _row("new-1", "seen_new", "new-a", rx, "old-a", 0.62, 0.03, 0.35),
                        _row("unk-1", "unknown", "__unknown__", rx, "old-a", 0.67, 0.02, 0.38),
                    ]
                )
                metric_rows.extend(
                    [
                        _row("old-1", "old", "old-a", rx, "old-a", 0.91, 0.38, 0.22),
                        _row("new-1", "seen_new", "new-a", rx, "new-a", 0.87, 0.30, 0.18),
                        _row("unk-1", "unknown", "__unknown__", rx, "old-a", 0.32, 0.01, 0.93),
                    ]
                )
            base_csv = tmp_path / "base.csv"
            metric_csv = tmp_path / "metric.csv"
            output_json = tmp_path / "dual.json"
            output_csv = tmp_path / "dual_evidence.csv"
            _write_csv(base_csv, base_rows)
            _write_csv(metric_csv, metric_rows)

            from phase2_dual_metric_guard_ci_eval import main

            rc = main(
                [
                    "--base_evidence_csv",
                    str(base_csv),
                    "--metric_evidence_csv",
                    str(metric_csv),
                    "--output_json",
                    str(output_json),
                    "--output_evidence_csv",
                    str(output_csv),
                    "--old_labels",
                    "old-a",
                    "--seen_new_labels",
                    "new-a",
                    "--collab_counts",
                    "all",
                    "--unknown_risk_threshold",
                    "0.85",
                    "--accept_margin_threshold",
                    "0.05",
                ]
            )

            self.assertEqual(rc, 0)
            result = json.loads(output_json.read_text(encoding="utf-8"))
            counts = [row["participating_receivers"] for row in result["summary_rows"]]
            self.assertEqual(counts, [1, 2])
            best = result["best_joint_row"]
            self.assertEqual(best["old_acc"], 1.0)
            self.assertEqual(best["seen_new_acc"], 1.0)
            self.assertEqual(best["unknown_reject_rate"], 1.0)
            self.assertFalse(best["target_pass"])
            self.assertTrue(output_csv.exists())


if __name__ == "__main__":
    unittest.main()
